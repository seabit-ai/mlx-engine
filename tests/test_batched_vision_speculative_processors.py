"""Speculative rounds for rows with logits processors (lmk spec-with-tools, SPD-027/028).

A tiny world: a real byte-level BPE tokenizer and the real Qwen3.5 tool guard (llguidance grammar);
a fake target whose raw argmax follows a script except where it is "tempted" by a token the guard
forbids; a fake drafter that proposes the raw argmax (so tempted drafts must be rejected). A round
must emit exactly what token-by-token decoding with the same processor emits."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_engine.model_kit.batched_vision import speculative
from mlx_engine.model_kit.batched_vision.batch_generator import _PrefixCacheSaveState, _apply_logits_processors
from mlx_engine.model_kit.batched_vision.speculative import Drafter, SpeculativeGenerationBatch
from mlx_engine.tool_protocols import Qwen35ToolContext
from mlx_engine.tool_runtime import create_qwen35_reasoning_guard_logits_processor

SPECIALS = ["<unk>", "<eos>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
THINKING = "I should look up the weather in Paris first."
TEXT = "Let me check that for you."
BODY = "<function=lookup>\n<parameter=query>\nweather in Paris\n</parameter>\n</function>\n"


def _hf_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([THINKING, TEXT, BODY, "You are a helpful assistant."],
                            trainers.BpeTrainer(vocab_size=320, special_tokens=["<unk>", "<eos>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")
    hf.add_special_tokens({"additional_special_tokens": SPECIALS[2:]})
    return hf


class _Tokenizer:
    """The attributes of mlx-engine's TokenizerWrapper the Qwen guard factory reads."""

    def __init__(self, hf):
        self._tokenizer = hf
        ids = {s: hf.convert_tokens_to_ids(s) for s in SPECIALS}
        self.ids = ids
        self.eos_token_ids = {ids["<eos>"]}
        self.think_start_tokens = (ids["<think>"],)
        self.think_end_tokens = (ids["</think>"],)
        self.tool_call_start_tokens = (ids["<tool_call>"],)
        self.tool_call_end_tokens = (ids["</tool_call>"],)
        self.vocab_size = len(hf.get_vocab())

    def encode(self, text, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=False)

    def get_vocab(self):
        return self._tokenizer.get_vocab()


TOKENIZER = _Tokenizer(_hf_tokenizer())
IDS = TOKENIZER.ids
EOS = IDS["<eos>"]
V = TOKENIZER.vocab_size
NL = TOKENIZER.encode("\n")[0]
LET = TOKENIZER.encode("Let")[0]
PROMPT = TOKENIZER.encode("You are a helpful assistant.") + [IDS["<think>"], NL]

# thinking (prompt opened it) | plain text | tool-call body under the grammar | the tail after the call
SEGMENTS = [
    ("thinking", TOKENIZER.encode(THINKING) + [IDS["</think>"]]),
    ("text", TOKENIZER.encode("\n\n") + TOKENIZER.encode(TEXT) + [IDS["<tool_call>"]]),
    ("body", [NL] + TOKENIZER.encode(BODY) + [IDS["</tool_call>"]]),
    ("tail", [NL, EOS]),
]
SCRIPT = [t for _, seg in SEGMENTS for t in seg]
SEGMENT_OF = [name for name, seg in SEGMENTS for _ in seg]
_start = {name: sum(len(s) for _, s in SEGMENTS[:i]) for i, (name, _) in enumerate(SEGMENTS)}
# where the raw model prefers a token the guard forbids at that point
TEMPTATIONS = {
    _start["thinking"] + 2: IDS["<tool_call>"],   # a tool call while reasoning is open
    _start["thinking"] + 5: IDS["<tool_call>"],
    _start["body"] + 3: EOS,                      # ending inside the call
    _start["body"] + 9: IDS["</tool_call>"],      # closing the call mid-body
    _start["tail"]: LET,                          # prose after a call
}


def _raw_logits(generated: list[int]) -> list[float]:
    k = len(generated)
    row = [-4.0] * V  # sharp enough that a sampled run mostly follows the script (and its temptations)
    if generated == SCRIPT[:k] and k < len(SCRIPT):
        row[SCRIPT[k]] = 5.0
        if k in TEMPTATIONS:
            row[TEMPTATIONS[k]] = 10.0
    else:
        row[EOS] = 5.0
    return row


def _raw_argmax(generated: list[int]) -> int:
    row = _raw_logits(generated)
    return max(range(V), key=row.__getitem__)


class _Target:
    """Holds the fed tokens as its 'cache'; logits (and the MTP hidden state) are a function of them."""

    def __init__(self):
        self.fed = list(PROMPT)

    def generated(self, extra=()):
        return self.fed[len(PROMPT):] + list(extra)

    def __call__(self, input_ids, cache=None, **kwargs):
        rows = []
        for t in input_ids.reshape(-1).tolist():
            self.fed.append(int(t))
            rows.append(_raw_logits(self.generated()))
        logits = mx.array([rows])
        return SimpleNamespace(logits=logits, hidden_states=[logits])

    def rollback_speculative_cache(self, cache, gdn_states, accepted, block_size):
        accepted = accepted[0] if isinstance(accepted, list) else accepted   # mtp rounds pass one per row
        del self.fed[len(self.fed) - (block_size - 1 - accepted):]

    def speculative_logits_from_hidden(self, hidden):
        return hidden

    def speculative_argmax_from_hidden(self, hidden):
        return mx.argmax(hidden, axis=-1)


class _DFlashConfig:
    block_size = 4
    target_layer_ids = [0]
    sliding_window = 16


class _Draft:
    """Proposes the raw argmax continuation from what the target has been fed plus the bonus;
    `wrong` lists (round, position) pairs where it proposes a plainly wrong token instead."""
    config = _DFlashConfig()
    dflash_initial_block_size = None
    prefer_requested_block_size = True

    def __init__(self, target, wrong=()):
        self.target, self.wrong = target, set(wrong)
        self.rounds_at: list[int] = []   # generated length when each round drafted (bonus included)
        self.accept_lens, self.draft_lens = [], []
        self.speculative_total_rounds = 0
        self.speculative_total_accepted = 0.0
        self.speculative_total_drafted = 0

    def reset(self, model, left_padding=None):
        return ["draft-cache"]

    def filter_batch(self, keep):
        pass

    def set_shared_kv(self, *args, **kwargs):
        pass

    def accept_verified_tokens_batch(self, *args, **kwargs):
        pass

    def propose(self, bonus: int, n: int) -> list[int]:
        seq = self.target.generated([bonus])
        self.rounds_at.append(len(seq))
        out = []
        for i in range(n):
            t = _raw_argmax(seq)
            if (len(self.rounds_at) - 1, i) in self.wrong:
                t = LET if t != LET else NL
            out.append(t)
            seq.append(t)
        return out

    def draft_block(self, bonus, context, cache, block_size, sampler, dtype):
        return mx.array([self.propose(int(bonus), block_size - 1)], dtype=dtype)


class _Cache:
    keys = True

    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)

    def extract(self, idx):
        return "extracted"

    def filter(self, keep):
        pass


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.int32)


_greedy.greedy = True


def _categorical(logprobs):
    return mx.random.categorical(logprobs).astype(mx.int32)


def _guard():
    return create_qwen35_reasoning_guard_logits_processor(
        tokenizer=TOKENIZER, context=Qwen35ToolContext(tool_names=("lookup",), reasoning_open=True))


def _batch(kind, *, speculative_on, sampler=_greedy, processors=True, max_tokens=len(SCRIPT) + 20, wrong=()):
    target = _Target()
    draft = _Draft(target, wrong)
    procs = [_guard()] if processors else []
    # the prefill's last position, as _PromptPrefill.generate hands it over
    logits = mx.array([_raw_logits([])])
    if procs:
        logits = _apply_logits_processors(logits, [list(PROMPT)], [procs])
    first = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    drafter = Drafter(model=draft, kind=kind, block_size=4, target_layer_ids=(0,), window=16)
    batch = SpeculativeGenerationBatch(
        model=target, uids=[1], inputs=first, prompt_cache=[_Cache()], samplers=[sampler],
        stop_criteria=lambda t: t == EOS, max_tokens=[max_tokens], top_logprobs=[0], all_tokens=[list(PROMPT)],
        logits_processors=[procs], prefix_cache_save_states=[_PrefixCacheSaveState([], 0, [], None)], drafter=drafter,
    )
    batch._hidden = mx.zeros((1, 1, V))
    batch._rows[0].speculative = speculative_on
    batch._rows[0].draft_tokens = 3
    return batch, draft


def _mtp_hooks(monkeypatch, draft_of):
    """The MTP helpers speculative_round takes from mlx-vlm, bound to the fake drafter."""
    monkeypatch.setattr(speculative, "_cache_positions", lambda cache, rows: [len(draft_of()[0].target.fed)])
    monkeypatch.setattr(speculative._mtp, "_mtp_draft_block_active",
                        lambda model, bonus, hidden, block_size, *a, **k: mx.array(
                            [model.propose(int(bonus[0]), block_size - 1)], dtype=mx.int32))


def _run_responses(batch) -> list:
    out = []
    for _ in range(4 * len(SCRIPT)):
        if not len(batch):
            return out
        out.extend(batch.next())
    raise AssertionError("the row never finished")


def _run(batch) -> list[int]:
    return [r.token for r in _run_responses(batch)]


# ---- SPD-028: the round -> plain step handoff

class _Recorder:
    """A processor on the generic path: sees the row's tokens with the last one appended."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def __call__(self, tokens, logits):
        self.seen.append(tokens.tolist())
        return logits


class _ScriptedRound:
    """Stands in for dflash_round: emits a fixed block, feeds the target like a verify pass would."""

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, model, drafter, prompt_cache, bonus, context, draft_cache, sampler, budget, block_size,
                 truncate, emitted, rope_deltas=None, **kwargs):
        model.fed.extend([bonus] + self.tokens[:-1])
        cut, reason = truncate(0, list(self.tokens))
        return speculative.RoundResult(new_tokens=[cut], finish=[reason], hidden=mx.zeros((1, 1, V)),
                                       accepted=[len(cut) - 1])


def _round_then_plain(monkeypatch, batch, round_tokens):
    monkeypatch.setattr(speculative, "dflash_round", _ScriptedRound(round_tokens))
    decisions = iter([True, False])
    monkeypatch.setattr(SpeculativeGenerationBatch, "_can_round", lambda self: next(decisions, False))
    emitted = [r.token for r in batch.next()]   # the pending token, then the round's block
    emitted += [r.token for r in batch.next()]  # a plain step feeds the round's last token (its bonus)
    return emitted


def test_a_plain_step_after_a_round_shows_the_bonus_to_a_processor_once(monkeypatch):
    batch, _ = _batch("dflash", speculative_on=True, processors=False)
    recorder = _Recorder()
    batch._rows[0].logits_processors = [recorder]
    first = int(batch._next_tokens.item())
    _round_then_plain(monkeypatch, batch, [11, 12, 13])

    assert recorder.seen == [PROMPT + [first, 11, 12, 13]]
    assert batch._rows[0].tokens == PROMPT + [first, 11, 12, 13]


def test_a_plain_step_after_a_round_inside_a_tool_call_does_not_feed_the_grammar_twice(monkeypatch):
    # the round ends on a token inside the call body: the bonus must reach llguidance exactly once
    batch, _ = _batch("dflash", speculative_on=True)
    k = _start["body"]
    before = SCRIPT[:k - 1]                    # everything up to the <tool_call> the pending token will be
    batch._rows[0].tokens.extend(before)
    batch._next_tokens = mx.array([SCRIPT[k - 1]], dtype=mx.int32)
    guard = batch._rows[0].logits_processors[0]
    guard(mx.array(batch._rows[0].tokens), mx.zeros((1, V)))   # the guard as the prefill left it, at this point
    batch.model.fed.extend(before)

    emitted = _round_then_plain(monkeypatch, batch, SCRIPT[k:k + 4])

    assert emitted[:5] == SCRIPT[k - 1:k + 4]
    assert batch._rows[0].tokens == PROMPT + SCRIPT[:k + 4]


# ---- the walk: a round with processors emits what token-by-token decoding emits

@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_a_greedy_round_with_the_tool_guard_emits_the_plain_path_tokens_across_all_four_segments(monkeypatch, kind):
    plain_batch, plain_draft = _batch(kind, speculative_on=False)
    plain = _run(plain_batch)
    assert plain == SCRIPT and plain_draft.rounds_at == []      # the guard masked every temptation

    spec_batch, draft = _batch(kind, speculative_on=True, wrong={(1, 1), (6, 0)})
    if kind == "mtp":
        _mtp_hooks(monkeypatch, lambda: [draft])
    spec = _run(spec_batch)

    assert spec == plain
    assert {SEGMENT_OF[min(at, len(SCRIPT) - 1)] for at in draft.rounds_at} == {"thinking", "text", "body", "tail"}
    assert draft.speculative_total_drafted > 0 and 0 < draft.speculative_total_accepted < draft.speculative_total_drafted


def test_without_the_guard_the_same_world_goes_off_script():
    unguarded, _ = _batch("dflash", speculative_on=False, processors=False)
    assert _run(unguarded) != SCRIPT


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_a_sampled_round_with_the_tool_guard_draws_what_the_plain_path_draws(monkeypatch, kind):
    # one sampler call per emitted token on both paths, in the same order: same seed, same tokens
    mx.random.seed(7)
    plain_batch, _ = _batch(kind, speculative_on=False, sampler=_categorical, max_tokens=60)
    plain = _run(plain_batch)

    mx.random.seed(7)
    spec_batch, draft = _batch(kind, speculative_on=True, sampler=_categorical, max_tokens=60)
    if kind == "mtp":
        _mtp_hooks(monkeypatch, lambda: [draft])
    spec = _run(spec_batch)

    assert spec == plain
    assert draft.speculative_total_drafted > 0


def test_rows_with_processors_may_round_top_logprobs_still_may_not():
    batch, _ = _batch("dflash", speculative_on=True)
    assert batch._can_round()
    batch._rows[0].top_logprobs = 1
    assert not batch._can_round()


# ---- the cache a finished row hands off holds exactly the tokens it reports (SPD-032)

@pytest.mark.parametrize("kind", ["dflash", "mtp"])
@pytest.mark.parametrize("processors", [True, False])
def test_a_row_finished_by_a_round_hands_off_a_cache_that_holds_exactly_its_all_tokens(monkeypatch, kind, processors):
    batch, draft = _batch(kind, speculative_on=True, processors=processors)
    if kind == "mtp":
        _mtp_hooks(monkeypatch, lambda: [draft])
    responses = _run_responses(batch)
    last = responses[-1]
    assert last.finish_reason == "stop" and draft.rounds_at, "the row finished inside a round"
    assert last.all_tokens == batch.model.fed


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
@pytest.mark.parametrize("max_tokens", [1, 7, 10, 13])
def test_max_tokens_landing_mid_block_emits_the_plain_path_tokens_and_a_matching_cache(monkeypatch, kind, max_tokens):
    plain_batch, _ = _batch(kind, speculative_on=False, max_tokens=max_tokens)
    plain = _run_responses(plain_batch)
    spec_batch, draft = _batch(kind, speculative_on=True, max_tokens=max_tokens)
    if kind == "mtp":
        _mtp_hooks(monkeypatch, lambda: [draft])
    spec = _run_responses(spec_batch)

    assert [r.token for r in spec] == [r.token for r in plain] and len(spec) == max_tokens
    assert spec[-1].finish_reason == "length"
    assert spec[-1].all_tokens == spec_batch.model.fed
    assert plain[-1].all_tokens == plain_batch.model.fed


def test_the_walk_stops_at_a_stop_token_instead_of_walking_past_it():
    seen = []

    def recorder(tokens, logits):
        seen.append(tokens.tolist()[-1])
        return logits

    target = [4, EOS, 6, 7]   # the target agrees with every draft, EOS included
    logits = mx.array([[[5.0 if v == t else 0.0 for v in range(V)] for t in target]])
    accepted, new_tokens = speculative._processed_walk(
        lambda j: logits[:, j, :], 3, [4, EOS, 6], [1, 2], [recorder], _greedy, budget=10, base_position=0,
        stop=lambda t: t == EOS)
    assert new_tokens == [4, EOS]
    assert seen == [3, 4]              # the processor never saw EOS as a fed token
    assert accepted == 1               # EOS is emitted, not fed: the cache keeps bonus + 4
