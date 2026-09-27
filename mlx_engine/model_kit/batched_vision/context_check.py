"""Context check (lmk design 2026-09-26-context-check): while a request's uncached prompt segment is prefilled,
score how well the model predicts that segment's own tokens (teacher-forced: the log-probability of token
t+1 given everything up to t, cached prefix included). A restored or prefilled context that is intact
predicts the conversation it belongs to; a lost or corrupt one is surprised by it. The numbers are recorded
per request with where the prefix came from, for the caller to log; nothing here decides anything."""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx

# The segment's last TAIL_TOKENS positions are scored: one lm_head projection row each, on top of the forward
# that computes the hidden states anyway. A cap, not a measured optimum — the cost per token is written down
# in lmk research/2026-09-26-context-check; the tail is where the new turn is, next to what the model writes.
TAIL_TOKENS = 512


@dataclass
class ContextCheck:
    restore_source: str               # "none" (cold) / "hot" (the in-memory cache of the last request) / "disk"
    restored_tokens: int              # prefix tokens that came from that source (the checkpoint the restore landed on)
    hot_trimmed_tokens: int = 0       # hot restores: tokens trimmed off the end of the held cache to fit this prompt
    segment_tokens: int = 0           # tokens prefilled for this request (the uncached segment)
    logprobs: list = field(default_factory=list)
    ms: float = 0.0                   # time spent scoring (projection + reduction), not the prefill itself
    unsupported: Optional[str] = None

    def summary(self) -> dict:
        out = {"restore_source": self.restore_source, "restored_tokens": self.restored_tokens,
               "hot_trimmed_tokens": self.hot_trimmed_tokens, "segment_tokens": self.segment_tokens,
               "scored_tokens": len(self.logprobs), "ms": round(self.ms, 1), "unsupported": self.unsupported}
        if self.logprobs:
            surprise = sorted(-lp for lp in self.logprobs)
            out.update(surprise_mean=round(sum(surprise) / len(surprise), 4),
                       surprise_p90=round(surprise[min(len(surprise) - 1, math.ceil(0.9 * len(surprise)) - 1)], 4),
                       surprise_max=round(surprise[-1], 4))
        return out


class SurpriseMeter:
    """Scores the tail of one request's segment, chunk by chunk as the prefill calls the model.
    Position i of the segment predicts token i+1; the segment's last position predicts the first generated
    token and is not scored, nor is the segment's first token (predicted by the restored prefix, whose last
    hidden state is not at hand)."""

    def __init__(self, segment: list[int], check: ContextCheck, project: Optional[Callable[[mx.array], mx.array]],
                 tail: int = TAIL_TOKENS):
        self._segment = list(segment)
        self.check = check
        self._project = project
        check.segment_tokens = len(segment)
        end = len(segment) - 1                      # positions [0, end) have their next token in the segment
        self._lo, self._hi = max(0, end - tail), max(0, end)
        if project is None and check.unsupported is None:
            check.unsupported = "the model exposes no projection from hidden states"

    def wants(self, start: int, n: int) -> bool:
        return self._project is not None and start < self._hi and start + n > self._lo

    def add(self, start: int, hidden: Optional[Any]) -> None:
        """hidden: the final-layer states for segment positions [start, start + n), shape [1, n, H]."""
        if hidden is None or self._project is None:
            return
        n = hidden.shape[1]
        first, last = max(start, self._lo), min(start + n, self._hi)
        if first >= last:
            return
        began = time.perf_counter()
        rows = hidden[:, first - start : last - start, :]
        logits = self._project(rows).reshape(last - first, -1).astype(mx.float32)
        targets = mx.array(self._segment[first + 1 : last + 1], dtype=mx.int32)
        picked = mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        logprobs = picked - mx.logsumexp(logits, axis=-1)
        self.check.logprobs.extend(logprobs.tolist())
        self.check.ms += (time.perf_counter() - began) * 1000


def projection_for(model: Any) -> Optional[Callable[[mx.array], mx.array]]:
    project = getattr(model, "speculative_logits_from_hidden", None)
    return project if callable(project) else None


def take_final_hidden(output: Any, added: bool) -> Optional[mx.array]:
    """The final-layer states from a forward. When the check asked for them (`added`), they are the last entry of
    hidden_states and are taken out, so a DFlash drafter still sees only the layers it captured."""
    states = getattr(output, "hidden_states", None)
    if not states:
        return None
    final = states[-1]
    if added:
        output.hidden_states = list(states[:-1]) or None
    return final
