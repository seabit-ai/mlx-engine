"""A text call that asks for captured layers (a DFlash drafter) goes to mlx-vlm's own call, which numbers positions
from 0 when no RoPE state is stored; after a restored prefix that is wrong. The patch hands it the cache's positions
(lmk research 2026-09-26-sampling-seed exp01)."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_engine.model_kit.patches import qwen3_5 as patch


class _KV:
    def __init__(self, offset):
        self.offset = offset


def _call(monkeypatch, cache, n, rows=1, **kwargs):
    seen = {}
    monkeypatch.setattr(patch, "OriginalVlmQwen3_5LanguageModelCall",
                        lambda self, inputs, **kw: seen.update(kw) or "original")
    lm = SimpleNamespace(model=SimpleNamespace(fa_idx=1), _position_ids=None, _rope_deltas=None)
    patch._patched_vlm_qwen3_5_language_model_call(lm, mx.zeros((rows, n), dtype=mx.int32), cache=cache, **kwargs)
    return seen


def test_a_captured_text_call_after_a_restored_prefix_gets_the_caches_positions(monkeypatch):
    seen = _call(monkeypatch, [None, _KV(256)], 4, capture_layer_ids=[5, 19])
    assert seen["position_ids"].tolist() == [[256, 257, 258, 259]]


def test_a_captured_decode_step_in_a_batch_gets_each_rows_position(monkeypatch):
    seen = _call(monkeypatch, [None, _KV(mx.array([300, 12]))], 1, rows=2, capture_layer_ids=[5])
    assert seen["position_ids"].tolist() == [[300], [12]]


@pytest.mark.parametrize("extra", [{"position_ids": mx.array([[7, 8, 9]])}, {"rope_deltas": mx.array([[0]])}])
def test_calls_that_carry_their_own_rope_state_are_left_alone(monkeypatch, extra):
    seen = _call(monkeypatch, [None, _KV(256)], 3, capture_layer_ids=[5], **extra)
    if "position_ids" in extra:
        assert seen["position_ids"].tolist() == [[7, 8, 9]]
    else:
        assert "position_ids" not in seen
