"""KV cache quantization on the batched vision path.

mlx-vlm already ships the cache classes (QuantizedKVCache, BatchQuantizedKVCache)
and attention that dispatches on ``cache.bits``. This module is the plumbing
that was missing on this path: building quantized caches for prompt prefill,
promoting one scalar cache into a batch cache, and the type checks the disk
record code needs. Only full-attention ``KVCache`` layers are quantized;
rotating-window and array-state layers keep their own cache types.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx_vlm.models.cache import (
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    make_prompt_cache,
)


@dataclass(frozen=True)
class KVQuantParams:
    bits: int
    group_size: int = 64


def make_prompt_cache_for(
    model: Any,
    kv_quant: Optional[KVQuantParams],
    make_prompt_cache: Callable[[Any], list[Any]] = make_prompt_cache,
) -> list[Any]:
    """``make_prompt_cache`` with full-attention layers swapped for quantized ones.

    ``make_prompt_cache`` is a parameter so a caller can hand in the one its own
    module imports (tests monkeypatch ``batch_generator.make_prompt_cache``)."""
    prompt_cache = make_prompt_cache(model)
    if kv_quant is None:
        return prompt_cache
    return [
        (
            QuantizedKVCache(group_size=kv_quant.group_size, bits=kv_quant.bits)
            if type(layer_cache) is KVCache
            else layer_cache
        )
        for layer_cache in prompt_cache
    ]


def is_quantized_kv_cache(cache: Any) -> bool:
    # Same module-independent identifier the record code uses for class names.
    return type(cache).__name__ == "QuantizedKVCache"


class TrimmableBatchQuantizedKVCache(BatchQuantizedKVCache):
    """mlx-vlm's batch quantized cache reports itself untrimmable, which makes
    speculative rollback mistake it for an SSM state. Trimming is only index
    bookkeeping, exactly as in BatchKVCache.trim."""

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, int(n))
        self._idx -= n
        self.offset = self.offset - n
        return n


def batch_from_scalar_quantized(cache: QuantizedKVCache) -> BatchQuantizedKVCache:
    """The BatchKVCache.merge([cache]) equivalent for one quantized cache."""
    batch_cache = TrimmableBatchQuantizedKVCache(
        [0], group_size=cache.group_size, bits=cache.bits
    )
    if cache.keys is None or cache.offset == 0:
        return batch_cache
    keys, values = cache.state
    batch_cache.keys = tuple(mx.contiguous(k) for k in keys)
    batch_cache.values = tuple(mx.contiguous(v) for v in values)
    batch_cache.offset = mx.array([cache.offset])
    batch_cache._idx = cache.offset
    return batch_cache


def kv_token_capacity(cache: Any) -> int:
    """Allocated token slots of a KVCache or QuantizedKVCache (probe bookkeeping)."""
    keys = cache.keys
    if isinstance(keys, (tuple, list)):
        keys = keys[0]
    return int(keys.shape[2])


def verify_block_attention(
    queries: mx.array,
    keys: Any,
    values: Any,
    *,
    cache: Any,
    scale: float,
    mask: Any,
) -> mx.array:
    """Target-verify attention over a block of L > 1 new tokens on a quantized cache.

    mlx-vlm verifies a speculative block position by position, slicing dense keys,
    which a quantized cache cannot serve: its keys are (packed, scales, biases)
    tuples. The same result comes from one quantized attention over the whole
    block with a causal mask — block token i sees the prefix and block tokens
    0..i — which quantized_scaled_dot_product_attention already supports. Rows
    of a batch cache that start with left padding do not see the padding.
    ``keys``/``values`` are what update_and_fetch returned: sliced to the valid
    length, the block at the end."""
    from mlx_vlm.models.base import scaled_dot_product_attention

    block = queries.shape[2]
    total = keys[0].shape[-2]
    q_idx = mx.arange(total - block, total)[:, None]
    k_idx = mx.arange(total)[None, :]
    allowed = q_idx >= k_idx  # [block, total]
    left_padding = getattr(cache, "left_padding", None)
    if isinstance(left_padding, mx.array) and left_padding.ndim > 0 and left_padding.size > 0:
        pads = mx.maximum(left_padding, 0).astype(mx.int32)  # [rows]
        allowed = allowed[None, None] & (k_idx[None, None] >= pads[:, None, None, None])  # [rows, 1, block, total]
    if isinstance(mask, mx.array) and mask.ndim >= 2:
        given = mask[..., -block:, :total]
        if given.dtype == mx.bool_:
            allowed = allowed & given
        else:  # additive
            allowed = mx.where(allowed, given, mx.finfo(given.dtype).min)
    return scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=allowed)
