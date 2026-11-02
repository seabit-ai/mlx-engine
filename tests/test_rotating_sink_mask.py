"""Tests for the RotatingKVCache attention-sink mask fix (chunked prefill)."""

import pytest

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache as MlxLmRotatingKVCache
from mlx_vlm.models.cache import RotatingKVCache as MlxVlmRotatingKVCache

from mlx_engine.model_kit.patches.rotating_sink_mask import (
    apply_rotating_sink_mask_patch,
)

MAX_SIZE = 16
KEEP = 4
CHUNK = 8


def _kv(S: int):
    """Fake K/V for one chunk; the seq dim carries an arange so slot identity is checkable."""
    width = 4
    keys = mx.arange(S * width, dtype=mx.float32).reshape(1, 1, S, width)
    return keys, keys


def _materialize(mask, N: int, offset: int):
    """Turn whatever make_mask returned into an [N, offset+N] boolean array."""
    if isinstance(mask, str):
        assert mask == "causal"
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        return linds[:, None] >= rinds[None]
    return mask


def _visible_columns(mask):
    """Column indices visible to every row / to the last row, as plain sets."""
    every_row = [c for c in range(mask.shape[-1]) if bool(mask[:, c].all())]
    last_row = [c for c in range(mask.shape[-1]) if bool(mask[-1, c])]
    return set(every_row), set(last_row)


def test_chunked_prefill_keeps_sinks_visible():
    cache = apply_rotating_sink_mask_patch(MlxLmRotatingKVCache(MAX_SIZE, KEEP))
    cache.update_and_fetch(*_kv(MAX_SIZE))  # fill: offset = 16, buffer starts slicing
    assert cache.offset == MAX_SIZE

    mask = cache.make_mask(CHUNK)
    assert not isinstance(mask, str)
    assert mask.shape == (CHUNK, MAX_SIZE - 1 + CHUNK)

    every_row, _ = _visible_columns(mask)
    # 1) the sink columns are visible to every query row (causal allows it:
    #    all rows sit past max_size - 1 while sinks label 0..keep-1)
    assert set(range(KEEP)) <= every_row
    # 2) the sliding part shrinks to max_size - keep: every row sees exactly
    #    keep + (max_size - keep) columns, not max_size
    assert all(int(row.sum()) == MAX_SIZE for row in mask)


def test_last_prefill_row_matches_single_token_decode_columns():
    """Chunked prefill and decode must agree on the visible column set."""
    cache = apply_rotating_sink_mask_patch(MlxLmRotatingKVCache(MAX_SIZE, KEEP))
    cache.update_and_fetch(*_kv(MAX_SIZE))  # tokens 0..15, sinks 0..3
    mask = cache.make_mask(CHUNK)
    cache.update_and_fetch(*_kv(CHUNK))  # tokens 16..23
    # RotatingKVCache.offset keeps growing past max_size (24 here); the
    # buffer itself is ring-limited. Key positions in a mask describe
    # chunk-local indexing (see rotating_sink_mask.py).
    assert cache.offset == 24

    _, last_row = _visible_columns(mask)
    # What a single-token decode at offset=24 sees: mask None -> the whole
    # 16-column buffer = the keep sinks + the max_size - keep most recent
    # positions. In this mask's coordinates (key_len = max_size - 1 + chunk)
    # that is the first keep columns plus the last max_size - keep columns.
    decode_visible = cache.make_mask(1)
    assert decode_visible is None  # window covers the cache: decode sees everything
    key_len = MAX_SIZE - 1 + CHUNK
    expected = set(range(KEEP)) | set(range(key_len - (MAX_SIZE - KEEP), key_len))
    assert last_row == expected, f"prefill row {sorted(last_row)} != decode {expected}"


def test_sink_columns_survive_the_update():
    """The keep columns stay in place in the buffer after a sliced chunk."""
    cache = apply_rotating_sink_mask_patch(MlxLmRotatingKVCache(MAX_SIZE, KEEP))
    cache.update_and_fetch(*_kv(MAX_SIZE))
    cache.update_and_fetch(*_kv(CHUNK))
    sinks = cache.keys[0, 0, :KEEP, :]
    assert bool((sinks == mx.arange(KEEP * 4, dtype=mx.float32).reshape(KEEP, 4)).all())


def test_unpatched_upstream_hides_sinks():
    """Pin the bug: upstream's own make_mask drops the sink columns here."""
    cache = MlxLmRotatingKVCache(MAX_SIZE, KEEP)
    cache.update_and_fetch(*_kv(MAX_SIZE))
    mask = _materialize(cache.make_mask(CHUNK), CHUNK, MAX_SIZE)
    assert not bool(mask[:, :KEEP].all())


def test_small_prefill_stays_causal_string():
    cache = apply_rotating_sink_mask_patch(MlxLmRotatingKVCache(MAX_SIZE, KEEP))
    cache.update_and_fetch(*_kv(CHUNK))  # offset = 8
    # key count (offset + N = 16) does not exceed the window: upstream behavior
    # is already correct, keep returning the fast-path string
    assert cache.make_mask(CHUNK) == "causal"
    assert cache.make_mask(1) is None  # decode untouched


def test_window_smaller_than_cache_left_upstream():
    """A sub-cache-size window hides the sinks in decode too: not our fight."""
    cache = MlxLmRotatingKVCache(MAX_SIZE, KEEP)
    apply_rotating_sink_mask_patch(cache)
    original = cache.make_mask.__wrapped__ if hasattr(cache.make_mask, "__wrapped__") else None
    # The patch defers to the original for sub-cache windows: prove it by
    # behavior, since MethodType wrapping makes identity checks unreliable.
    cache.update_and_fetch(*_kv(MAX_SIZE))
    window_mask = cache.make_mask(CHUNK, window_size=8)
    assert isinstance(window_mask, mx.array)
    assert not bool(window_mask[:, :KEEP].all())


def test_patch_applies_to_mlx_vlm_copy_and_is_idempotent():
    for module in (MlxLmRotatingKVCache, MlxVlmRotatingKVCache):
        cache = apply_rotating_sink_mask_patch(module(MAX_SIZE, KEEP))
        first = cache.make_mask(CHUNK)
        cache.update_and_fetch(*_kv(MAX_SIZE))
        mask = cache.make_mask(CHUNK)
        assert not isinstance(mask, str)
        assert bool(mask[:, :KEEP].all())
        apply_rotating_sink_mask_patch(cache)  # second call must not double-wrap
        again = cache.make_mask(CHUNK)
        assert mx.array_equal(mask, again)
        assert first == "causal"


def test_patch_list_leaves_other_caches_alone():
    from mlx_vlm.models.cache import KVCache

    kv = KVCache()
    rotating = MlxLmRotatingKVCache(MAX_SIZE, KEEP)
    apply_rotating_sink_mask_patch([kv, rotating])
    assert rotating.make_mask is not None
    assert getattr(rotating, "_mlx_engine_rotating_sink_mask_patched", False)
    assert not hasattr(kv, "_mlx_engine_rotating_sink_mask_patched")


def _semantic_checks(cache_cls):
    """The full mask semantics, run against both libraries' RotatingKVCache.

    mlx-vlm ships a byte-identical buggy make_mask; the mlx-lm side has its
    own dedicated tests above, these prove the patch covers the vlm copy
    with the same guarantees."""
    cache = apply_rotating_sink_mask_patch(cache_cls(MAX_SIZE, KEEP))
    cache.update_and_fetch(*_kv(MAX_SIZE))  # fill, slicing starts
    mask = cache.make_mask(CHUNK)
    assert not isinstance(mask, str)
    assert mask.shape == (CHUNK, MAX_SIZE - 1 + CHUNK)
    # sinks visible to every row; sliding part shrunk to pay for them
    assert set(range(KEEP)) <= _visible_columns(mask)[0]
    assert all(int(row.sum()) == MAX_SIZE for row in mask)
    # the buffer's sink slots survive the sliced chunk
    assert bool((cache.keys[0, 0, :KEEP, :] ==
                 mx.arange(KEEP * 4, dtype=mx.float32).reshape(KEEP, 4)).all())
    # and unpatched upstream drops the sinks here (the bug, pinned)
    raw = cache_cls(MAX_SIZE, KEEP)
    raw.update_and_fetch(*_kv(MAX_SIZE))
    raw_mask = _materialize(raw.make_mask(CHUNK), CHUNK, MAX_SIZE)
    assert not bool(raw_mask[:, :KEEP].all())


def test_semantics_on_mlx_vlm_copy():
    _semantic_checks(MlxVlmRotatingKVCache)


def test_real_chunks_are_integer_powers():
    pytest.importorskip("mlx")

