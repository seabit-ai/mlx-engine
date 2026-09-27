"""Context check (lmk design 2026-09-26-context-check): the prefill scores how well the model predicts its own
prompt segment, teacher-forced, on the segment's tail; the score does not depend on how the segment was chunked."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_engine.model_kit.batched_vision.batch_generator as batcher
from mlx_engine.model_kit.batched_vision.context_check import ContextCheck, SurpriseMeter, take_final_hidden
from mlx_engine.model_kit.batched_vision.prompt_cache.coordinator import (
    VlmPromptCacheCoordinator,
    _HotPromptCacheEntry,
    _HotRestoreCandidate,
)

V = 16
TABLE = mx.random.normal((V, V), key=mx.random.key(5)) * 2   # next-token logits by current token


class _Cache:
    keys = True

    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)

    def extract(self, idx):
        return self


class _Model:
    """The 'hidden state' of a position is its logits row (projection = identity), a function of its token.
    Records which calls asked for hidden states and what they got back."""

    def __init__(self, capture=False):
        self.calls = []
        self.capture = capture
        self.config = SimpleNamespace(use_bidirectional_attention=None)
        self.model_type = None

    def speculative_logits_from_hidden(self, hidden):
        return hidden

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kwargs):
        ids = input_ids.reshape(-1)
        rows = TABLE[ids][None]
        self.calls.append({"n": ids.size, "return_hidden": kwargs.get("return_hidden", False)})
        out = SimpleNamespace(logits=rows, hidden_states=None)
        states = []
        if kwargs.get("capture_layer_ids") is not None:
            states.append(mx.zeros((1, ids.size, 3)))   # a captured layer, as a DFlash drafter wants
        if kwargs.get("return_hidden"):
            states.append(rows)
        out.hidden_states = states or None
        return out


def _expected(segment, tail):
    logprobs = TABLE - mx.logsumexp(TABLE, axis=-1, keepdims=True)
    end = len(segment) - 1
    return [logprobs[segment[i], segment[i + 1]].item() for i in range(max(0, end - tail), end)]


def _prefill(segment, step, tail=None, monkeypatch=None, model=None, drafter=None, prefix=(), targets=None):
    if tail is not None:
        monkeypatch.setattr(batcher, "SurpriseMeter",
                            lambda seg, check, project: SurpriseMeter(seg, check, project, tail=tail))
    model = model or _Model()
    check = ContextCheck(restore_source="disk", restored_tokens=0, targets=targets)
    prefill = batcher._PromptPrefill(
        model=model, uid=1, input_ids=list(segment), max_tokens=1, top_logprobs=0,
        sampler=lambda lp: mx.argmax(lp, axis=-1), logits_processors=[],
        inputs_embeds=mx.zeros((1, len(segment), 2)), prompt_kwargs={},
        prefix_cache_save_state=batcher._PrefixCacheSaveState([], 0, [], None),
        prefill_step_size=step, cache=[_Cache()], all_tokens=list(prefix), context_check=check, drafter=drafter,
    )
    while prefill.needs_processing():
        prefill.prompt_step()
    prefill.generate(lambda t: False)
    return check, model


SEGMENT = [3, 7, 1, 1, 9, 4, 12, 0, 5, 5, 8, 2, 15, 6]


@pytest.mark.parametrize("step", [2, 3, 5, 64])
def test_the_score_is_the_teacher_forced_logprob_of_the_segment_whatever_the_chunking(step):
    check, _ = _prefill(SEGMENT, step)
    assert check.logprobs == pytest.approx(_expected(SEGMENT, tail=10_000), abs=1e-5)
    assert check.segment_tokens == len(SEGMENT) and len(check.logprobs) == len(SEGMENT) - 1


def test_after_a_restored_prefix_positions_count_from_the_segments_start(monkeypatch):
    check, _ = _prefill(SEGMENT, 3, tail=4, monkeypatch=monkeypatch, prefix=range(300))
    assert check.logprobs == pytest.approx(_expected(SEGMENT, tail=4), abs=1e-5)


def test_only_the_tail_is_scored_and_only_chunks_that_reach_it_return_hidden_states(monkeypatch):
    check, model = _prefill(SEGMENT, 3, tail=4, monkeypatch=monkeypatch)
    assert check.logprobs == pytest.approx(_expected(SEGMENT, tail=4), abs=1e-5)
    assert [c["return_hidden"] for c in model.calls][:2] == [False, False]   # positions 0..5 are before the tail


def test_the_summary_reports_surprise_as_negative_logprob():
    check = ContextCheck(restore_source="hot", restored_tokens=512, hot_trimmed_tokens=3,
                         segment_tokens=5, logprobs=[-0.5, -1.0, -3.0, -0.1])
    s = check.summary()
    assert (s["restore_source"], s["restored_tokens"], s["hot_trimmed_tokens"], s["scored_tokens"]) == ("hot", 512, 3, 4)
    assert s["surprise_mean"] == pytest.approx(1.15) and s["surprise_max"] == pytest.approx(3.0)
    assert s["surprise_p90"] == pytest.approx(3.0)


def test_a_model_without_a_projection_is_reported_not_scored():
    class Bare(_Model):
        speculative_logits_from_hidden = None

    check, model = _prefill(SEGMENT, 4, model=Bare())
    assert check.logprobs == [] and "projection" in check.summary()["unsupported"]
    assert not any(c["return_hidden"] for c in model.calls)


def test_with_a_dflash_drafter_the_captured_layers_stay_what_the_drafter_asked_for():
    seen = []
    drafter = SimpleNamespace(is_dflash=True, window=4,
                              capture_kwargs=lambda: {"capture_layer_ids": [0]},
                              hidden_from=lambda out: seen.append(len(out.hidden_states)) or out.hidden_states[0])
    check, _ = _prefill(SEGMENT, 5, model=_Model(), drafter=drafter)
    assert seen and set(seen) == {1}     # the check's final-layer states were taken out before the drafter looked
    assert check.logprobs == pytest.approx(_expected(SEGMENT, tail=10_000), abs=1e-5)


def test_take_final_hidden_leaves_states_it_did_not_add():
    out = SimpleNamespace(hidden_states=["captured", "final"])
    assert take_final_hidden(out, added=False) == "final" and out.hidden_states == ["captured", "final"]
    assert take_final_hidden(out, added=True) == "final" and out.hidden_states == ["captured"]


def test_a_hot_restore_says_so_and_how_much_it_trimmed():
    class Trimmable:
        def is_trimmable(self):
            return True

        def trim(self, n):
            return n

    coordinator = object.__new__(VlmPromptCacheCoordinator)
    entry = _HotPromptCacheEntry(prompt_input_ids=list(range(10)), image_spans=[], prompt_cache=[Trimmable()],
                                 rope_deltas=None)
    restored = coordinator._load_hot_restore_plan(_HotRestoreCandidate(entry=entry, cached_prefix_len=7))
    assert (restored.source, restored.trimmed_tokens, restored.cached_prefix_len) == ("hot", 3, 7)


def test_named_targets_are_scored_wherever_they_are_and_only_they(monkeypatch):
    targets = [2, 3, 9, 13, 40]            # 40 is past the segment; 13 is its last token, predicted by position 12
    check, model = _prefill(SEGMENT, 4, targets=targets)
    lp = TABLE - mx.logsumexp(TABLE, axis=-1, keepdims=True)
    assert check.logprobs == pytest.approx([lp[SEGMENT[t - 1], SEGMENT[t]].item() for t in (2, 3, 9, 13)], abs=1e-5)
    assert [c["return_hidden"] for c in model.calls] == [True, False, True, True]   # chunks 0-3, 4-7, 8-11, 12-13
    assert check.summary()["scored"] == "targets"


def test_the_plain_lm_head_is_preferred_to_the_exact_verifier_projection():
    from mlx_engine.model_kit.batched_vision.context_check import projection_for

    head = lambda h: h  # noqa: E731
    model = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False), lm_head=head,
                            speculative_logits_from_hidden=lambda h: None)
    assert projection_for(model) is head
    own = lambda h: h * 0  # noqa: E731
    capped = SimpleNamespace(args=SimpleNamespace(final_logit_softcapping=30.0), lm_head=head,
                             speculative_logits_from_hidden=own)
    assert projection_for(capped) is own    # a softcap is part of the logits: the model's own method applies it
