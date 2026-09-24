"""SpeculativeGenerationBatch bookkeeping: what the engine's loop adds around one mlx-vlm round."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_engine.model_kit.batched_vision import speculative
from mlx_engine.model_kit.batched_vision.batch_generator import GenerationBatch, _PrefixCacheSaveState
from mlx_engine.model_kit.batched_vision.speculative import Drafter, SpeculativeGenerationBatch

H = 4


def _argmax(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


class _Cache:
    keys = True

    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)
        self.filtered = []

    def extract(self, idx):
        return f"extracted:{idx}"

    def extend(self, other):
        pass

    def filter(self, keep):
        self.filtered.append(keep.tolist())


class _Model:
    """Decode forwards return logits (argmax 0) and, when asked, a hidden state per row."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, cache=None, **kwargs):
        self.calls.append({"input_ids": input_ids.tolist(), "return_hidden": kwargs.get("return_hidden")})
        b, n = input_ids.shape
        out = SimpleNamespace(logits=mx.zeros((b, n, 8)))
        if kwargs.get("return_hidden"):
            out.hidden_states = [mx.ones((b, n, H)) * 7]
        return out


class _DraftModel:
    def __init__(self):
        self.resets, self.filters = [], []
        self.speculative_total_rounds = 0
        self.speculative_total_accepted = 0.0
        self.speculative_total_drafted = 0

    def reset(self, model, left_padding=None):
        self.resets.append(left_padding)

    def filter_batch(self, keep):
        self.filters.append(list(keep))


class _Round:
    """Stands in for speculative_round: hands back scripted tokens, honours truncate."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, model, drafter, prompt_cache, bonus, hidden, samplers, budgets, block_size, truncate):
        tokens_by_row = self.script.pop(0)
        self.calls.append(dict(bonus=list(bonus), budgets=list(budgets), block_size=block_size, rows=hidden.shape[0]))
        new_tokens, finish = [], []
        for i, toks in enumerate(tokens_by_row):
            cut, reason = truncate(i, toks)
            new_tokens.append(cut)
            finish.append(reason)
        return speculative.RoundResult(new_tokens=new_tokens, finish=finish,
                                       hidden=mx.ones((len(bonus), 1, H)), accepted=[len(t) - 1 for t in new_tokens])


def _batch(model, rows, *, stop=frozenset(), max_tokens=100, top_logprobs=0, speculative_rows=None, pending=True):
    drafter = Drafter(model=_DraftModel(), kind="mtp", block_size=3)
    batch = SpeculativeGenerationBatch(
        model=model, uids=list(range(10, 10 + rows)), inputs=mx.array([5] * rows, dtype=mx.int32),
        prompt_cache=[_Cache()], samplers=[_argmax] * rows, stop_criteria=lambda t: t in stop,
        max_tokens=[max_tokens] * rows, top_logprobs=[top_logprobs] * rows, all_tokens=[[1]] * rows,
        logits_processors=[[]] * rows, prefix_cache_save_states=[_PrefixCacheSaveState([], 0, [], None)] * rows,
        drafter=drafter,
    )
    batch._hidden = mx.zeros((rows, 1, H))
    if speculative_rows is not None:
        for row, flag in zip(batch._rows, speculative_rows):
            row.speculative = flag
    if not pending:
        batch._pending = [False] * rows
        batch._bonus = [5] * rows
    return batch


def test_the_pending_token_goes_out_first_then_a_round_emits_several(monkeypatch):
    model = _Model()
    fake = _Round([[[7, 8, 9]]])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    batch = _batch(model, 1)

    responses = batch.next()

    assert [r.token for r in responses] == [5, 7, 8, 9]
    assert all(r.finish_reason is None for r in responses)
    assert batch._rows[0].tokens == [1, 5, 7, 8, 9] and batch._rows[0].num_tokens == 4
    assert fake.calls[0]["bonus"] == [5] and fake.calls[0]["block_size"] == 3
    assert batch._bonus == [9] and batch._pending == [False] and batch._next_tokens.tolist() == [9]
    assert batch.drafter.model.resets == [[0]]  # batch-shaped drafter state for one batch row
    assert model.calls == []  # no plain forward happened


def test_a_stop_token_inside_the_round_ends_the_row_with_its_cache(monkeypatch):
    fake = _Round([[[7, 2, 8]]])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    batch = _batch(_Model(), 1, stop={2})

    responses = batch.next()

    assert [(r.token, r.finish_reason) for r in responses] == [(5, None), (7, None), (2, "stop")]
    assert responses[-1].prompt_cache == ["extracted:0"] and responses[-1].all_tokens == [1, 5, 7, 2]
    assert len(batch) == 0


def test_max_tokens_cuts_the_round_and_says_length(monkeypatch):
    fake = _Round([[[7, 8, 9]]])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    batch = _batch(_Model(), 1, max_tokens=3)  # the pending token is the 1st, so two more fit

    responses = batch.next()

    assert [(r.token, r.finish_reason) for r in responses] == [(5, None), (7, None), (8, "length")]
    assert fake.calls[0]["budgets"] == [2]


def test_top_logprobs_or_an_opt_out_makes_the_step_plain(monkeypatch):
    fake = _Round([])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    for batch in (_batch(_Model(), 1, top_logprobs=1), _batch(_Model(), 1, speculative_rows=[False])):
        responses = batch.next()
        assert [r.token for r in responses] == [5] and fake.calls == []
        assert batch.model.calls[0]["return_hidden"] is True  # the plain step keeps the hidden state fresh
        assert batch._hidden.shape == (1, 1, H) and batch._pending == [True]


def test_after_a_round_a_plain_tick_feeds_the_bonus_without_emitting_it_again(monkeypatch):
    model = _Model()
    monkeypatch.setattr(speculative, "speculative_round", _Round([[[7, 8]]]))
    batch = _batch(model, 1)
    assert [r.token for r in batch.next()] == [5, 7, 8]
    batch._rope_deltas = mx.array([1], dtype=mx.int32)  # an image batch: rounds are off

    assert batch.next() == []  # the bonus 8 was already emitted: fed, not repeated
    assert model.calls[-1]["input_ids"] == [[8]] and batch._pending == [True]
    responses = batch.next()  # decode-ahead resumes
    assert [r.token for r in responses] == [0] and batch._rows[0].tokens == [1, 5, 7, 8, 0]


def test_a_joining_row_keeps_pending_flags_and_hidden_states_aligned(monkeypatch):
    fake = _Round([[[7, 8]], [[20, 21], [30]]])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    batch = _batch(_Model(), 1)
    batch.next()  # row 10 emitted 5, 7, 8; bonus 8 pending=False

    newcomer = GenerationBatch(
        model=batch.model, uids=[11], inputs=mx.array([6], dtype=mx.int32), prompt_cache=[_Cache()],
        samplers=[_argmax], stop_criteria=lambda t: False, max_tokens=[100], all_tokens=[[2]],
        logits_processors=[[]], prefix_cache_save_states=[_PrefixCacheSaveState([], 0, [], None)],
    )
    newcomer._next_hidden = mx.zeros((1, 1, H))
    batch.append_prefilled_sequence(newcomer)

    assert batch._pending == [False, True] and batch._hidden.shape == (2, 1, H) and batch._drafter_rows == 0
    responses = batch.next()
    assert [(r.uid, r.token) for r in responses] == [(11, 6), (10, 20), (10, 21), (11, 30)]
    assert fake.calls[1]["bonus"] == [8, 6] and fake.calls[1]["rows"] == 2
    assert batch.drafter.model.resets[-1] == [0, 0]


def test_a_finished_row_leaves_the_drafter_and_hidden_state_in_step(monkeypatch):
    fake = _Round([[[7, 2], [8, 9]], [[10]]])
    monkeypatch.setattr(speculative, "speculative_round", fake)
    batch = _batch(_Model(), 2, stop={2})

    batch.next()
    assert len(batch) == 1 and batch.uids == [11] and batch._bonus == [9] and batch._hidden.shape == (1, 1, H)
    assert batch.drafter.model.filters == [[1]]
    responses = batch.next()
    assert [(r.uid, r.token) for r in responses] == [(11, 10)] and fake.calls[1]["bonus"] == [9]


def test_drafter_problem_reads_only_configs(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_mtp", "text_config": {"hidden_size": 5120, "vocab_size": 248320}}')
    target = {"text_config": {"hidden_size": 5120, "vocab_size": 248320}}
    assert speculative.drafter_problem(tmp_path, target) is None
    assert "hidden_size" in speculative.drafter_problem(tmp_path, {"text_config": {"hidden_size": 4096}})
    (tmp_path / "config.json").write_text('{"model_type": "dflash"}')
    assert "not supported" in speculative.drafter_problem(tmp_path, target)
    assert "config.json" in speculative.drafter_problem(tmp_path / "missing", target)
