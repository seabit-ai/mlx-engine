import mlx.core as mx
import pytest
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from mlx_engine.model_kit.batched_vision.kv_quant import (
    KVQuantParams,
    batch_from_scalar_quantized,
    kv_token_capacity,
    make_prompt_cache_for,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.disk_budget import (
    _estimate_layer_cache_bytes,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.records import (
    assemble_prompt_cache_chunks,
    make_prompt_cache_layout,
    prepare_prompt_cache_records_for_chunk,
    record_kind_for_prompt_cache,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
)

HEADS, DIM = 2, 64


class _HybridModel:
    """A model whose cache topology mixes full attention, linear state and a window."""

    def make_cache(self):
        return [KVCache(), ArraysCache(size=2), RotatingKVCache(max_size=8), KVCache()]


def _filled_quantized(tokens: int, bits: int = 8, seed: int = 0) -> tuple[QuantizedKVCache, mx.array, mx.array]:
    cache = QuantizedKVCache(group_size=64, bits=bits)
    keys = mx.random.normal((1, HEADS, tokens, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    values = mx.random.normal((1, HEADS, tokens, DIM), key=mx.random.key(seed + 1)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)
    return cache, keys, values


def _packed(cache) -> list:
    keys, values = cache.state
    mx.eval(*keys, *values)
    return [a.tolist() for a in (*keys, *values)]


def test_only_full_attention_layers_are_quantized():
    cache = make_prompt_cache_for(_HybridModel(), KVQuantParams(bits=8, group_size=64))
    assert [type(c).__name__ for c in cache] == [
        "QuantizedKVCache", "ArraysCache", "RotatingKVCache", "QuantizedKVCache",
    ]
    assert cache[0].bits == 8 and cache[0].group_size == 64


def test_no_params_means_the_plain_topology():
    cache = make_prompt_cache_for(_HybridModel(), None)
    assert [type(c).__name__ for c in cache] == ["KVCache", "ArraysCache", "RotatingKVCache", "KVCache"]


def test_a_scalar_quantized_cache_becomes_a_one_row_batch_and_back():
    cache, _, _ = _filled_quantized(300)
    batch = batch_from_scalar_quantized(cache)
    assert isinstance(batch, BatchQuantizedKVCache)
    assert batch._idx == 300 and batch.offset.tolist() == [300] and batch.left_padding.tolist() == [0]
    back = batch.extract(0)
    assert back.offset == 300 and back.bits == 8
    assert _packed(back) == _packed(cache)


def test_an_empty_scalar_quantized_cache_promotes_to_an_empty_batch():
    batch = batch_from_scalar_quantized(QuantizedKVCache(group_size=64, bits=4))
    assert batch.keys is None and batch.bits == 4 and batch.offset.tolist() == [0]


def test_the_batch_keeps_growing_after_promotion():
    cache, _, _ = _filled_quantized(300)
    batch = batch_from_scalar_quantized(cache)
    more = mx.random.normal((1, HEADS, 5, DIM)).astype(mx.bfloat16)
    keys, values = batch.update_and_fetch(more, more)
    assert batch._idx == 305 and keys[0].shape[2] == 305


def test_quantized_records_are_kv_deltas_that_round_trip_through_chunks():
    cache, _, _ = _filled_quantized(512)
    assert record_kind_for_prompt_cache(cache) == RECORD_KIND_KV_DELTA
    chunk_caches = []
    for start, end in ((0, 256), (256, 512)):
        record_caches, kinds = prepare_prompt_cache_records_for_chunk([cache], start, end)
        assert kinds == [RECORD_KIND_KV_DELTA]
        chunk = record_caches[0]
        assert type(chunk).__name__ == "QuantizedKVCache" and chunk.offset == 256
        chunk_caches.append(record_caches)
    layout = make_prompt_cache_layout(chunk_caches[-1], [RECORD_KIND_KV_DELTA])
    chunks = [PromptPrefixChunk(key="a", start=0, end=256), PromptPrefixChunk(key="b", start=256, end=512)]
    assembled = assemble_prompt_cache_chunks(chunk_caches, chunks, layout)[0]
    assert assembled.offset == 512 and assembled.bits == 8 and assembled.group_size == 64
    assert _packed(assembled) == _packed(cache)


def test_a_chunk_past_the_snapshot_is_refused():
    from mlx_engine.model_kit.batched_vision.prompt_cache.records import PromptCacheRecordCoverageError
    cache, _, _ = _filled_quantized(300)
    with pytest.raises(PromptCacheRecordCoverageError):
        prepare_prompt_cache_records_for_chunk([cache], 256, 512)


def test_mixing_quantized_and_full_precision_chunks_is_refused():
    q, _, _ = _filled_quantized(256)
    full = KVCache()
    full.state = (mx.zeros((1, HEADS, 256, DIM)), mx.zeros((1, HEADS, 256, DIM)))
    layout = make_prompt_cache_layout([q], [RECORD_KIND_KV_DELTA])
    chunks = [PromptPrefixChunk(key="a", start=0, end=256), PromptPrefixChunk(key="b", start=256, end=512)]
    with pytest.raises(ValueError, match="different KV cache settings"):
        assemble_prompt_cache_chunks([[full], [q]], chunks, layout)


def test_quantized_bytes_are_about_half_of_bf16_at_8_bits():
    q, keys, values = _filled_quantized(256)
    full = KVCache()
    full.state = (keys, values)
    q_bytes = _estimate_layer_cache_bytes(q, 4096)
    full_bytes = _estimate_layer_cache_bytes(full, 4096)
    # 8 bits + 2×16-bit scale/bias per group of 64 = 8.5 bits per element vs 16
    assert 0.50 < q_bytes / full_bytes < 0.56


def test_token_capacity_reads_the_allocated_slots():
    q, keys, _ = _filled_quantized(300)
    assert kv_token_capacity(q) == 512  # allocated in steps of 256
    full = KVCache()
    full.state = (keys, keys)
    assert kv_token_capacity(full) == 300


# --- speculative verify on a quantized cache -------------------------------------------------
# mlx-vlm verifies a drafted block position by position over dense keys; on a quantized cache the
# engine attends the whole block once with a causal mask. The two must agree.

from mlx_engine.model_kit.batched_vision.kv_quant import verify_block_attention


def _per_position_reference(queries, dense_keys, dense_values, scale, start=0):
    """mlx-vlm's loop (language.py, the `target_verify and L > 1` branch) on dequantized keys."""
    block = queries.shape[2]
    prefix = dense_keys.shape[-2] - block
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                queries[:, :, i : i + 1, :],
                dense_keys[:, :, start : prefix + i + 1, :],
                dense_values[:, :, start : prefix + i + 1, :],
                scale=scale,
            )
            for i in range(block)
        ],
        axis=2,
    )


def _dequantized(cache, keys, values):
    dk = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits)
    dv = mx.dequantize(*values, group_size=cache.group_size, bits=cache.bits)
    return dk, dv


def test_verify_block_attention_matches_the_per_position_loop_on_dequantized_keys():
    mx.random.seed(7)
    rows, q_heads, kv_heads, dim, prefix, block = 1, 4, 2, 64, 300, 4
    cache = QuantizedKVCache(group_size=64, bits=8)
    cache.update_and_fetch(mx.random.normal((rows, kv_heads, prefix, dim)), mx.random.normal((rows, kv_heads, prefix, dim)))
    keys, values = cache.update_and_fetch(mx.random.normal((rows, kv_heads, block, dim)), mx.random.normal((rows, kv_heads, block, dim)))
    queries = mx.random.normal((rows, q_heads, block, dim))
    scale = dim**-0.5

    # the reference first: the quantized attention scales its queries in place
    ref = _per_position_reference(queries, *_dequantized(cache, keys, values), scale)
    out = verify_block_attention(queries, keys, values, cache=cache, scale=scale, mask=None)
    assert out.shape == ref.shape == (rows, q_heads, block, dim)
    assert mx.allclose(out, ref, atol=1e-3, rtol=1e-3).item()


def test_verify_block_attention_hides_each_rows_left_padding_in_a_batch_cache():
    mx.random.seed(11)
    rows, q_heads, kv_heads, dim, prefix, block = 2, 4, 2, 64, 40, 3
    pads = [5, 0]
    cache = BatchQuantizedKVCache(pads, group_size=64, bits=8)
    cache.update_and_fetch(mx.random.normal((rows, kv_heads, prefix, dim)), mx.random.normal((rows, kv_heads, prefix, dim)))
    keys, values = cache.update_and_fetch(mx.random.normal((rows, kv_heads, block, dim)), mx.random.normal((rows, kv_heads, block, dim)))
    queries = mx.random.normal((rows, q_heads, block, dim))
    scale = dim**-0.5

    dk, dv = _dequantized(cache, keys, values)
    refs = [_per_position_reference(queries[row : row + 1], dk[row : row + 1], dv[row : row + 1], scale, start=pad)
            for row, pad in enumerate(pads)]  # before: the quantized attention scales its queries in place
    out = verify_block_attention(queries, keys, values, cache=cache, scale=scale, mask=None)
    for row, (pad, ref) in enumerate(zip(pads, refs)):
        assert mx.allclose(out[row : row + 1], ref, atol=1e-3, rtol=1e-3).item(), f"row {row} (pad {pad})"


def test_the_patched_left_padded_attention_only_takes_over_quantized_verify_blocks():
    """mlx-vlm 0.6.16's exact verifier asks the language module's helper first; dense caches
    and single tokens must still reach the original."""
    from mlx_engine.model_kit.patches import qwen3_5 as patches

    calls = []
    original = patches.OriginalVlmQwen3_5LeftPaddedAttention
    patches.OriginalVlmQwen3_5LeftPaddedAttention = lambda *a, **k: calls.append("original") or None
    try:
        q = mx.zeros((1, 2, 3, 8))
        assert patches._patched_vlm_qwen3_5_left_padded_attention(q, q, q, cache=KVCache(), scale=1.0, mask=None) is None
        one = mx.zeros((1, 2, 1, 8))
        quantized = QuantizedKVCache(group_size=64, bits=8)
        assert patches._patched_vlm_qwen3_5_left_padded_attention(one, one, one, cache=quantized, scale=1.0, mask=None) is None
        assert calls == ["original", "original"]
    finally:
        patches.OriginalVlmQwen3_5LeftPaddedAttention = original
