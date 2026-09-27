"""
Sampling helpers shared by sequential and batched generation.
"""

import math
from typing import Optional

import mlx.core as mx


@mx.compile
def _apply_top_k(
    logprobs: mx.array,
    top_k: int,
) -> mx.array:
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(
        logprobs,
        mask_idx,
        mx.array(-float("inf"), logprobs.dtype),
        axis=-1,
    )


@mx.compile
def _apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            mx.array(False, tokens_to_remove.dtype),
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


@mx.compile
def _apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


def create_sampler(
    temp: Optional[float],
    top_p: Optional[float],
    min_p: Optional[float],
    min_tokens_to_keep: Optional[int],
    top_k: Optional[int],
    seed: Optional[int] = None,
):
    temp = 0.0 if temp is None else temp
    top_p = 0.0 if top_p is None else top_p
    min_p = 0.0 if min_p is None else min_p
    min_tokens_to_keep = 1 if min_tokens_to_keep is None else min_tokens_to_keep
    top_k = 0 if top_k is None else top_k

    if temp == 0:

        def greedy(logprobs):
            return mx.argmax(logprobs, axis=-1)

        greedy.greedy = True  # speculative decoding verifies greedy rows with the fused argmax
        return greedy

    def truncate(logprobs):
        if top_p > 0 and top_p < 1.0:
            logprobs = _apply_top_p(logprobs, top_p)
        if min_p != 0.0:
            logprobs = _apply_min_p(logprobs, min_p, min_tokens_to_keep)
        if top_k > 0:
            logprobs = _apply_top_k(logprobs, top_k)
        return logprobs

    if seed is not None:
        return SeededSampler(seed, truncate, temp)

    # Avoid mlx_lm.make_sampler's module-global compiled random helpers: they
    # capture thread-local PRNG state before batched worker threads call them.
    def sampler(logprobs):
        return mx.random.categorical(truncate(logprobs) * (1 / temp))

    return sampler


_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def position_key(seed: int, position: int) -> mx.array:
    """The PRNG key for one request's draw at one generated position: a pure function of
    (seed, position), so no other request and no earlier call can move it."""
    return mx.random.key(_splitmix64((seed & _MASK64) ^ _splitmix64(position)))


class SeededSampler:
    """One request's sampler with its own randomness: the token at generated position p is drawn
    with position_key(seed, p) — counter-based, not a stream. That makes a row's draws independent
    of the other rows in the batch and of how the decode loop reached p: a plain step, a
    speculative verify walk and a drafter proposal for position p all use the same key. The
    verify walks call `sample_target` with explicit positions (mlx-vlm's positioned-sampler
    protocol); a caller that knows no positions calls the sampler and gets 0, 1, 2, ... in call
    order (mlx-lm's batch generator draws once per emitted token, so that is the position)."""

    def __init__(self, seed: int, truncate, temp: float):
        self.seed = int(seed)
        self._truncate = truncate
        self._inv_temp = 1 / temp
        self._next_position = 0

    def _draw(self, logprobs: mx.array, position: int) -> mx.array:
        return mx.random.categorical(
            self._truncate(logprobs) * self._inv_temp,
            key=position_key(self.seed, position),
        )

    def sample_target(self, logprobs: mx.array, *, row_ids, positions) -> mx.array:
        """logprobs [N, V], one row per position; row_ids name batch rows in mlx-vlm's protocol and
        are ignored here — this sampler belongs to one request."""
        positions = [int(p) for p in positions]
        if logprobs.ndim == 1:
            logprobs = logprobs[None]
        if logprobs.shape[0] != len(positions):
            raise ValueError(
                f"{logprobs.shape[0]} rows of logprobs for {len(positions)} positions"
            )
        return mx.concatenate(
            [self._draw(logprobs[i : i + 1], p) for i, p in enumerate(positions)],
            axis=0,
        )

    def __call__(self, logprobs: mx.array) -> mx.array:
        """Any leading shape ([V], [N, V], [1, N, V] from mlx-lm's speculative loop): one position per row."""
        lead = logprobs.shape[:-1]
        rows = 1
        for n in lead:
            rows *= n
        positions = range(self._next_position, self._next_position + rows)
        self._next_position += rows
        drawn = self.sample_target(logprobs.reshape(rows, logprobs.shape[-1]), row_ids=[0] * rows,
                                   positions=positions)
        return drawn.reshape(lead)
