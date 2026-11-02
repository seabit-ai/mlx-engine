"""RotatingKVCache attention-sink mask fix for chunked prefill.

Upstream mlx-lm and mlx-vlm ``RotatingKVCache.make_mask`` hide the
attention-sink (``keep``) columns from multi-token (chunked) prefill queries
once the cache starts slicing (``offset > max_size``): the sliding-window cut
is applied to a key count that includes the whole current chunk, so the sink
columns at the buffer head fall outside ``window_size``. A single-token decode
query at the same position sees the whole buffer (mask ``None`` when the
window covers the cache), so chunked prefill and decode disagree on the
visible column set and in-window recall breaks.
Ref: https://github.com/ml-explore/mlx-lm/issues/1978 — bug report and fix
direction credited to Robert Price (blackweb.ai).

The patched ``make_mask`` gives every prefill query row, under causal
constraints, exactly the columns decode would show: the ``keep`` sink columns
plus the ``max_size - keep`` most recent rotating columns — the sliding part
shrinks from ``max_size`` to ``max_size - keep`` to pay for the sinks.

Decode (N == 1) and windows smaller than the cache are left upstream: a
sub-cache-size window hides the sinks in decode too, so upstream semantics
are already self-consistent there.
"""

from types import MethodType
from typing import Any

import mlx.core as mx

_PATCHED_ATTR = "_mlx_engine_rotating_sink_mask_patched"


def _make_sink_visible_make_mask(original_make_mask):
    """Build the patched make_mask around the original bound method."""

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        if N <= 1 or (window_size is not None and window_size < self.max_size):
            return original_make_mask(N, window_size, return_array)

        # Same key-count convention as upstream: the chunk being prefilled is
        # part of the buffer this mask describes, and once the cache slices,
        # query rows are labelled from max_size - 1 so the causal band stays
        # exact for the columns the trim dropped.
        offset = min(self.max_size - 1, self.offset)
        key_len = offset + N
        if not return_array and key_len <= self.max_size:
            # Buffer fits under the window: the cut cannot bind, plain causal
            # already shows the sinks.
            return "causal"
        query_positions = mx.arange(offset, offset + N)[:, None]
        key_positions = mx.arange(key_len)[None]
        causal = query_positions >= key_positions
        recent = query_positions < key_positions + (self.max_size - self.keep)
        sink_columns = key_positions < self.keep
        return causal & (recent | sink_columns)

    return make_mask


def _is_rotating_kv_cache(cache: Any) -> bool:
    # mlx-lm and mlx-vlm ship the same class name, layout and buggy make_mask;
    # match by name + shape attrs so one check covers both without importing
    # either (keeps this module import-light for the engine).
    return (
        type(cache).__name__ == "RotatingKVCache"
        and hasattr(cache, "make_mask")
        and hasattr(cache, "max_size")
        and hasattr(cache, "offset")
        and hasattr(cache, "keep")
    )


def apply_rotating_sink_mask_patch(cache: Any) -> Any:
    """Bind a sink-visible make_mask onto RotatingKVCache instances.

    Accepts a single cache or a prompt-cache list; non-rotating caches are
    left untouched. Idempotent per instance.
    """
    caches = cache if isinstance(cache, list) else [cache]
    for entry in caches:
        if getattr(entry, _PATCHED_ATTR, False) or not _is_rotating_kv_cache(entry):
            continue
        original_make_mask = entry.make_mask
        entry.make_mask = MethodType(
            _make_sink_visible_make_mask(original_make_mask), entry
        )
        entry._mlx_engine_rotating_sink_mask_patched = True
    return cache

