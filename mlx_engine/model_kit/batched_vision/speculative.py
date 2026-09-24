"""Speculative decoding with a native MTP drafter inside the continuous-batching loop.

mlx-vlm owns the maths (draft block, verify pass, accept walk, cache rollback).
This file owns what the engine's decode loop needs on top of that: one decode
step is one draft/verify round for every row; rows join and leave between
rounds; stop tokens and token budgets are applied *before* the target cache is
rolled back, so a finished row's cache holds exactly its emitted tokens; and a
per-step decision whether the current batch can be sped up at all — otherwise
the step is an ordinary one and nothing is lost but the speed-up.

Only Qwen3.5-family MTP drafters (`qwen3_5_mtp`) are supported: they carry no
shared-KV state of their own, which is what makes joining and leaving cheap.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx_vlm.speculative import mtp as _mtp
from mlx_vlm.speculative.common import _record_speculative_round, generation_stream
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

from mlx_engine.model_kit.batched_vision.batch_generator import (
    GenerationBatch,
    _is_scalar_prompt_cache,
    _materialize_step_outputs,
    _sync_scalar_rope_deltas,
)

logger = logging.getLogger(__name__)

SUPPORTED_DRAFTER_MODEL_TYPES = ("qwen3_5_mtp",)
TOKEN_DTYPE = mx.int32


@dataclass
class Drafter:
    model: Any
    kind: str
    block_size: int  # tokens verified per round, the bonus token included

    @property
    def rounds(self) -> int:
        return int(getattr(self.model, "speculative_total_rounds", 0))

    @property
    def accepted(self) -> int:
        return int(round(getattr(self.model, "speculative_total_accepted", 0.0)))

    @property
    def drafted(self) -> int:
        return int(getattr(self.model, "speculative_total_drafted", 0))


def drafter_config(path: str | Path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def drafter_problem(path: str | Path, target_config: dict) -> Optional[str]:
    """Why this drafter cannot serve this model, or None. Cheap: reads two configs."""
    try:
        config = drafter_config(path)
    except (OSError, ValueError) as e:
        return f"cannot read the drafter's config.json: {e}"
    model_type = config.get("model_type")
    if model_type not in SUPPORTED_DRAFTER_MODEL_TYPES:
        return (
            f"drafter model_type {model_type!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_DRAFTER_MODEL_TYPES)})"
        )
    draft_text = config.get("text_config") or {}
    target_text = target_config.get("text_config") or target_config
    for key in ("hidden_size", "vocab_size"):
        mine, theirs = draft_text.get(key), target_text.get(key)
        if mine is not None and theirs is not None and mine != theirs:
            return f"drafter {key} {mine} does not match the model's {theirs}"
    return None


def load_mtp_drafter(path: str | Path, model: Any, target_config: dict) -> Drafter:
    problem = drafter_problem(path, target_config)
    if problem:
        raise ValueError(f"draft model at {path} is not compatible: {problem}")
    draft_model, kind = load_drafter(str(path))
    validate_drafter_compatibility(model, draft_model, kind)
    block_size = int(getattr(draft_model.config, "block_size", 3))
    return Drafter(model=draft_model, kind=kind, block_size=block_size)


def _language_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _cache_positions(prompt_cache: list[Any], rows: int) -> list[int]:
    """Each row's valid target-KV length (the next position to write)."""
    _, positions = _mtp._mtp_cache_positions(prompt_cache, rows)
    return positions


@dataclass
class _Verify:
    hidden: mx.array
    gdn_states: Any
    target_tokens: Optional[mx.array]


def _verify_block(lm: Any, verify_input: mx.array, prompt_cache: list[Any], rope_deltas: Any,
                  greedy: bool) -> _Verify:
    """The target's forward over [bonus, drafts...], the way the plain decode step calls it
    (same RoPE deltas), returning the last-layer hidden and the linear-attention states the
    rollback needs. Greedy targets come from the fused argmax, like mlx-vlm's own loop."""
    # speculative_verify: mlx-vlm 0.6.16 routes the block through its exact verifier (the
    # same call as its speculative_verify_hidden), instead of a flag on the attention layers.
    kwargs = dict(cache=prompt_cache, capture_layer_ids=[], return_hidden=True,
                  return_shared_kv=True, skip_logits=True, speculative_verify=True)
    if rope_deltas is not None:
        kwargs["rope_deltas"] = rope_deltas
    out = lm(verify_input, **kwargs)
    hidden = out.hidden_states[-1]
    target = lm.speculative_argmax_from_hidden(hidden) if greedy else None
    return _Verify(hidden=hidden, gdn_states=getattr(out, "gdn_states", None), target_tokens=target)


@dataclass
class RoundResult:
    new_tokens: list[list[int]]      # per row, already cut at a stop token or the budget
    finish: list[Optional[str]]      # per row: None, "stop" or "length"
    hidden: mx.array                 # [rows, 1, H]: the state that predicted each row's last token
    accepted: list[int]              # drafted tokens the target agreed with, per row


def speculative_round(
    model: Any,
    drafter: Drafter,
    prompt_cache: list[Any],
    bonus: list[int],
    hidden: mx.array,
    samplers: list[Callable[[mx.array], mx.array]],
    budgets: list[int],
    block_size: int,
    truncate: Callable[[int, list[int]], tuple[list[int], Optional[str]]],
    rope_deltas: Any = None,
) -> RoundResult:
    """One draft/verify round for every row of the batch.

    ``bonus`` is each row's last emitted token, not yet in the target cache;
    ``hidden`` the target state that predicted it. Drafting is greedy (the
    drafter's own head); acceptance runs each row's own sampler against the
    target's logits, so sampling requests get their sampler's distribution.
    """
    lm = _language_model(model)
    rows = len(bonus)
    positions = _cache_positions(prompt_cache, rows)
    drafter.model.set_shared_kv(
        {},
        kv_offset=max(positions),
        position=_mtp._mtp_draft_position(mx.array(positions)),
        kv_valid_len=mx.array(positions),
        left_padding=None,
    )
    draft_tokens = _mtp._mtp_draft_block_active(
        drafter.model,
        bonus,
        hidden,
        block_size,
        samplers[0],
        TOKEN_DTYPE,
        positions,
        greedy_sampling=True,
    )
    verify_input = mx.concatenate(
        [mx.array(bonus, dtype=TOKEN_DTYPE)[:, None], draft_tokens.astype(TOKEN_DTYPE)],
        axis=1,
    )
    # Greedy rows are verified the way mlx-vlm verifies them (the fused argmax over
    # the block): the same kernels as its own loop, so the output matches plain
    # decoding token for token. Sampling rows draw the target token with their own
    # sampler from logits projected per position.
    all_greedy = all(getattr(sampler, "greedy", False) for sampler in samplers)
    with mx.stream(generation_stream):
        verify = _verify_block(lm, verify_input, prompt_cache, rope_deltas, all_greedy)
    hidden_full = verify.hidden  # [rows, block_size, H]

    if verify.target_tokens is not None:
        _, walked_rows = _mtp._speculative_walk_batch(draft_tokens, verify.target_tokens, budgets)
    else:
        walked_rows = []
        for i in range(rows):
            _, walked = _mtp._speculative_walk_batch_deferred_greedy(
                lm,
                hidden_full[i : i + 1],
                draft_tokens[i : i + 1],
                samplers[i],
                [budgets[i]],
            )
            walked_rows.append(walked[0])

    new_tokens: list[list[int]] = []
    finish: list[Optional[str]] = []
    for i in range(rows):
        tokens, reason = truncate(i, walked_rows[i])
        if not tokens:
            raise RuntimeError("a speculative round produced no token for a row")
        new_tokens.append(tokens)
        finish.append(reason)
    # The last new token is the next bonus: it stays out of the cache, like the
    # token decode-ahead keeps pending. Everything before it was verified.
    accepted = [len(tokens) - 1 for tokens in new_tokens]
    _record_speculative_round(drafter.model, sum(accepted) / rows, block_size - 1)
    drafter.model.accept_verified_tokens_batch(
        hidden_full, draft_tokens, accepted, new_tokens, samplers[0], TOKEN_DTYPE, greedy=True
    )
    if any(a < block_size - 1 for a in accepted):
        with mx.stream(generation_stream):
            lm.rollback_speculative_cache(prompt_cache, verify.gdn_states, accepted, block_size)
    if rows == 1:
        next_hidden = hidden_full[:, accepted[0] : accepted[0] + 1, :]
    else:
        next_hidden = hidden_full[mx.arange(rows), mx.array(accepted), :][:, None, :]
    next_hidden = _mtp._mtp_draft_hidden(lm, next_hidden)
    mx.eval(next_hidden)
    return RoundResult(new_tokens=new_tokens, finish=finish, hidden=next_hidden, accepted=accepted)


class SpeculativeGenerationBatch(GenerationBatch):
    """GenerationBatch whose step is a speculative round whenever the batch allows it.

    Two per-row facts travel with the batch: whether the row's last sampled
    token is still *pending* (sampled by a forward but not emitted, the
    decode-ahead convention of the plain batch) and the hidden state that
    predicted it. A speculative step emits pending tokens first, then runs a
    round. A plain step is used when any row asks for top_logprobs or carries
    logits processors, when the batch has image RoPE state, or when the hidden
    state is unknown; plain steps keep the hidden state fresh so speculation
    can resume.
    """

    def __init__(self, *args, drafter: Drafter, **kwargs):
        super().__init__(*args, **kwargs)
        self.drafter = drafter
        self._hidden: Optional[mx.array] = None
        self._pending: list[bool] = [True] * len(self._rows)
        self._bonus: list[int] = []
        self._drafter_rows = 0  # rows the drafter's state currently covers; 0 = reset needed

    @classmethod
    def empty(cls, model, stop_criteria, top_logprobs_k=0, *, drafter: Drafter):
        return cls(
            model=model,
            uids=[],
            inputs=None,
            prompt_cache=[],
            samplers=[],
            stop_criteria=stop_criteria,
            max_tokens=[],
            top_logprobs_k=top_logprobs_k,
            top_logprobs=[],
            logits_processors=[],
            prefix_cache_save_states=[],
            drafter=drafter,
        )

    # ---- what the plain batch does, plus hidden-state bookkeeping

    def _forward_logits(self, inputs: mx.array, fwd_kwargs: dict) -> mx.array:
        output = self.model(
            inputs[:, None], cache=self.prompt_cache, return_hidden=True, **fwd_kwargs
        )
        hidden_states = getattr(output, "hidden_states", None)
        self._hidden = hidden_states[-1][:, -1:, :] if hidden_states else None
        logits = output.logits if hasattr(output, "logits") else output
        return logits[:, -1, :]

    def append_prefilled_sequence(self, prefilled: GenerationBatch):
        super().append_prefilled_sequence(prefilled)
        joined = getattr(prefilled, "_next_hidden", None)
        if self._hidden is not None and joined is not None and len(self._pending) > 0:
            self._hidden = mx.concatenate([self._hidden, joined])
        elif len(self._pending) == 0:
            self._hidden = joined
        else:
            self._hidden = None  # a plain step refreshes it for every row
        self._pending = self._pending + [True] * len(prefilled)
        self._bonus = self._bonus + [0] * len(prefilled)
        self._drafter_rows = 0

    def filter(self, keep: list[int]):
        super().filter(keep)
        self._pending = [self._pending[i] for i in keep]
        self._bonus = [self._bonus[i] for i in keep] if self._bonus else []
        if self._hidden is not None:
            self._hidden = self._hidden[mx.array(keep, mx.int32)] if keep else None
        if self._drafter_rows and keep and len(keep) < self._drafter_rows:
            self.drafter.model.filter_batch(keep)
            self._drafter_rows = len(keep)
        elif not keep:
            self._drafter_rows = 0

    # ---- the step

    def _round_block_size(self) -> int:
        sizes = [row.draft_tokens for row in self._rows if row.draft_tokens]
        drafts = min(sizes) if sizes else self.drafter.block_size - 1
        return max(1, drafts) + 1

    def _has_image_rope(self) -> bool:
        """Qwen3.5 carries mRoPE deltas for every prompt; only image prompts have non-zero ones,
        and the verify pass has no way to hand them to the model."""
        if self._rope_deltas is None:
            return False
        return bool(mx.any(self._rope_deltas != 0).item())

    # Rounds run while one request is decoding. With two or more rows, mlx-vlm's
    # batched rollback did not reproduce plain decoding on mixed-length rows (exp03:
    # identical prompts fine, code + story diverged at token 53), so a batch of
    # several rows takes plain steps until it is alone again. The batch bookkeeping
    # below already handles several rows for the day that path is trusted.
    MAX_ROUND_ROWS = 1

    def _can_round(self) -> bool:
        if self._hidden is None or self._has_image_rope():
            return False
        if len(self._rows) > self.MAX_ROUND_ROWS:
            return False
        if any(not row.speculative or row.top_logprobs > 0 or row.logits_processors for row in self._rows):
            return False
        return self._round_block_size() >= 2

    def next(self) -> list[GenerationBatch.Response]:
        if not self._rows:
            return []
        if self._can_round():
            responses = self._emit_pending()
            if self._rows:
                responses.extend(self._round())
            return responses
        if all(self._pending):
            return super().next()
        return self._plain_tick()

    def _plain_tick(self) -> list[GenerationBatch.Response]:
        """A plain step for a batch where some rows' last token already went out with
        a round: those rows are fed without emitting again, the others emit as in
        decode-ahead. Afterwards every row is pending again."""
        pending = list(self._pending)
        inputs = self._next_tokens
        prev_logprobs, prev_top_idx, prev_top = (
            self._next_token_logprobs, self._next_top_idx, self._next_top_logprobs
        )
        old_lens = [len(row.tokens) for row in self._rows]
        self._advance(inputs)
        tokens, logprob_list, top_idx_list, top_logprob_list = _materialize_step_outputs(
            inputs, prev_logprobs, prev_top_idx, prev_top
        )
        responses, keep = [], []
        for idx, row in enumerate(self._rows):
            if not pending[idx]:
                keep.append(idx)
                continue
            token = tokens[idx]
            if len(row.tokens) == old_lens[idx]:
                row.tokens.append(token)
            row.num_tokens += 1
            reason = self._finish_reason(row, token)
            self._emit_cache_save_snapshot(idx)
            top = None
            if row.top_logprobs > 0 and top_idx_list is not None:
                top = list(zip(top_idx_list[idx][: row.top_logprobs], top_logprob_list[idx][: row.top_logprobs]))
            responses.append(
                self.Response(
                    uid=row.uid,
                    token=token,
                    token_logprob=logprob_list[idx] if logprob_list is not None else 0.0,
                    finish_reason=reason,
                    top_logprobs=top,
                    prompt_cache=self.extract_cache(idx) if reason else None,
                    all_tokens=list(row.tokens) if reason else None,
                    rope_deltas=self.extract_rope_deltas(idx) if reason else None,
                )
            )
            if reason is None:
                keep.append(idx)
        self._pending = [True] * len(self._rows)
        if len(keep) < len(self._rows):
            self.filter(keep)
        return responses

    def _finish_reason(self, row, token: int) -> Optional[str]:
        if self.stop_criteria(token):
            return "stop"
        if row.num_tokens >= row.max_tokens:
            return "length"
        return None

    def _response(self, idx: int, row, token: int, finish_reason: Optional[str]):
        return self.Response(
            uid=row.uid,
            token=token,
            token_logprob=0.0,
            finish_reason=finish_reason,
            top_logprobs=None,
            prompt_cache=self.extract_cache(idx) if finish_reason else None,
            all_tokens=list(row.tokens) if finish_reason else None,
            rope_deltas=self.extract_rope_deltas(idx) if finish_reason else None,
        )

    def _emit_pending(self) -> list[GenerationBatch.Response]:
        """Emit the tokens the last forward sampled (every row is pending here)."""
        if not any(self._pending):
            return []
        mx.eval(self._next_tokens)
        tokens = [int(t) for t in self._next_tokens.tolist()]
        responses, keep = [], []
        for idx, row in enumerate(self._rows):
            if not self._pending[idx]:
                keep.append(idx)
                continue
            token = tokens[idx]
            # The snapshot goes out before the token joins row.tokens: this token was sampled
            # but not fed, so the cache does not hold it yet. The plain path keeps row.tokens
            # = fed tokens (decode-ahead); a snapshot taken one token early made the store skip
            # the chunk ("kv cache snapshot covers [0, 255), not [0, 256)").
            self._emit_cache_save_snapshot(idx)
            row.tokens.append(token)
            row.num_tokens += 1
            reason = self._finish_reason(row, token)
            responses.append(self._response(idx, row, token, reason))
            if reason is None:
                keep.append(idx)
        self._bonus = tokens
        self._pending = [False] * len(self._rows)
        if len(keep) < len(self._rows):
            self.filter(keep)
        return responses

    def _round(self) -> list[GenerationBatch.Response]:
        rows = self._rows
        count = len(rows)
        if self._drafter_rows != count:
            scalar = count == 1 and _is_scalar_prompt_cache(self.prompt_cache)
            self.drafter.model.reset(self.model, left_padding=None if scalar else [0] * count)
            self._drafter_rows = count
        budgets = [max(1, row.max_tokens - row.num_tokens) for row in rows]
        block_size = min(self._round_block_size(), max(budgets) + 1)

        def truncate(idx: int, tokens: list[int]) -> tuple[list[int], Optional[str]]:
            row = rows[idx]
            out: list[int] = []
            for token in tokens:
                out.append(token)
                if self.stop_criteria(token):
                    return out, "stop"
                if row.num_tokens + len(out) >= row.max_tokens:
                    return out, "length"
            return out, None

        # Positions continue exactly as in the plain step: same RoPE deltas, same side state.
        if self._rope_deltas is not None:
            _sync_scalar_rope_deltas(self.model, self.prompt_cache, self._rope_deltas)
        result = speculative_round(
            self.model,
            self.drafter,
            self.prompt_cache,
            self._bonus,
            self._hidden,
            [row.sampler for row in rows],
            budgets,
            block_size,
            truncate,
            rope_deltas=self._rope_deltas,
        )
        responses, keep = [], []
        for idx, row in enumerate(rows):
            tokens, reason = result.new_tokens[idx], result.finish[idx]
            for n, token in enumerate(tokens):
                last = n == len(tokens) - 1
                if last:
                    # everything before the last token was fed to the target in the verify pass;
                    # the last one is the next bonus, not in the cache yet — snapshot before it
                    # joins row.tokens (see _emit_pending)
                    self._emit_cache_save_snapshot(idx)
                row.tokens.append(token)
                row.num_tokens += 1
                responses.append(self._response(idx, row, token, reason if last else None))
            if reason is None:
                keep.append(idx)
        self._bonus = [tokens[-1] for tokens in result.new_tokens]
        self._hidden = result.hidden
        self._next_tokens = mx.array(self._bonus, dtype=TOKEN_DTYPE)
        self._next_token_logprobs = None
        self._next_top_idx = None
        self._next_top_logprobs = None
        self._pending = [False] * count
        if len(keep) < count:
            self.filter(keep)
        return responses
