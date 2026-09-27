"""Per-request seeds on the batched path (lmk sampling-seed): a request's draws are a pure function
of (seed, generated position), so a replay with the same seed draws the same tokens whatever else
is in the batch and whichever way (plain step, speculative walk) the loop reached a position."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_engine.model_kit.batched_vision import speculative
from mlx_engine.model_kit.batched_vision.batch_generator import GenerationBatch, _PrefixCacheSaveState
from mlx_engine.utils.sampling import SeededSampler, create_sampler, position_key

import tests.test_batched_vision_speculative_processors as world

V = 64


def _flat(rows=1, vocab=V):
    return mx.log(mx.ones((rows, vocab)) / vocab)


def _seeded(seed, temp=1.0, top_p=None, top_k=None):
    return create_sampler(temp, top_p, None, None, top_k, seed=seed)


# ---- the sampler

def test_no_seed_keeps_the_unseeded_sampler_and_temp_zero_stays_greedy():
    assert not hasattr(create_sampler(1.0, None, None, None, None), "sample_target")
    greedy = create_sampler(0.0, None, None, None, None, seed=5)
    assert getattr(greedy, "greedy", False) and not hasattr(greedy, "sample_target")


def test_the_same_seed_draws_the_same_tokens_and_another_seed_does_not():
    draws = lambda seed: [int(_seeded(seed).sample_target(_flat(), row_ids=[0], positions=[p]).item())
                          for p in range(32)]
    assert draws(7) == draws(7)
    assert draws(7) != draws(8)


def test_a_draw_depends_on_its_position_not_on_the_calls_before_it():
    a, b = _seeded(3), _seeded(3)
    for _ in range(5):
        a(_flat())                      # a has drawn positions 0..4 by call order
    at_5 = int(b.sample_target(_flat(), row_ids=[0], positions=[5]).item())
    assert int(a(_flat()).item()) == at_5
    mx.random.categorical(_flat())      # the global PRNG is not this sampler's
    assert int(b.sample_target(_flat(), row_ids=[0], positions=[5]).item()) == at_5


def test_a_block_of_positions_draws_what_one_call_per_position_draws():
    block = _seeded(11).sample_target(_flat(4), row_ids=[0] * 4, positions=[9, 10, 11, 12]).tolist()
    single = _seeded(11)
    assert block == [int(single.sample_target(_flat(), row_ids=[0], positions=[p]).item()) for p in (9, 10, 11, 12)]


def test_truncation_still_applies_to_a_seeded_draw():
    logprobs = mx.log(mx.softmax(mx.arange(V, dtype=mx.float32)[None] / 8, axis=-1))
    s = _seeded(1, top_k=1)
    assert {int(s.sample_target(logprobs, row_ids=[0], positions=[p]).item()) for p in range(20)} == {V - 1}


def test_position_keys_differ_across_seeds_and_positions():
    keys = {tuple(position_key(seed, p).tolist()) for seed in (0, 1, 2**63) for p in range(50)}
    assert len(keys) == 150


# ---- the plain batch: a row's tokens do not depend on who else is in it

class _Cache:
    keys = True

    def __init__(self):
        self.state = mx.array([0], dtype=mx.int32)

    def extract(self, idx):
        return "extracted"

    def extend(self, other):
        pass

    def filter(self, keep):
        pass


TABLE = mx.random.normal((V, V), key=mx.random.key(1234)) * 2  # next-token logits by fed token, fairly flat


class _MarkovModel:
    """Each row's logits are a function of the token it is fed, never of the other rows."""

    def __call__(self, input_ids, cache=None, **kwargs):
        return SimpleNamespace(logits=TABLE[input_ids.reshape(-1)][:, None, :])


def _logprobs_after(token: int) -> mx.array:
    logits = TABLE[mx.array([token])]
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _run_plain(samplers, first_tokens, steps=24):
    batch = GenerationBatch(
        model=_MarkovModel(), uids=list(range(len(samplers))), inputs=mx.array(first_tokens, dtype=mx.int32),
        prompt_cache=[_Cache()], samplers=samplers, stop_criteria=lambda t: False,
        max_tokens=[steps] * len(samplers), logits_processors=[[] for _ in samplers],
        prefix_cache_save_states=[_PrefixCacheSaveState([], 0, [], None) for _ in samplers],
        all_tokens=[[0] for _ in samplers],
    )
    out = {uid: [] for uid in range(len(samplers))}
    while len(batch):
        for r in batch.next():
            out[r.uid].append(r.token)
    return out


def _first(sampler, token=0):
    return int(sampler.sample_target(_logprobs_after(token), row_ids=[0], positions=[0]).item())


def test_a_seeded_row_draws_the_same_tokens_alone_and_next_to_other_rows():
    alone = _run_plain([_seeded(42)], [_first(_seeded(42))])[0]
    unseeded = create_sampler(1.0, None, None, None, None)
    crowded = _run_plain([_seeded(5), unseeded, _seeded(42)], [3, 4, _first(_seeded(42))])[2]
    assert crowded == alone and len(alone) == 24


def test_the_plain_step_draws_generated_position_k_with_the_key_for_k():
    seed = 9
    tokens = [_first(_seeded(seed))]
    ref = _seeded(seed)
    for k in range(1, 12):
        tokens.append(int(ref.sample_target(_logprobs_after(tokens[-1]), row_ids=[0], positions=[k]).item()))
    assert _run_plain([_seeded(seed)], [tokens[0]], steps=12)[0] == tokens


# ---- speculative rounds draw what the plain path draws, with no global seed at all

# The processors test's world follows its script almost surely at temp 1 (logits 5 against -4), so any key
# draws the same token there. Here the target is a flat Markov chain instead: a draw at the wrong position
# picks a different token most of the time, and sampled draws reject the drafter's argmax proposals often.
_WORLD_TABLE = (mx.random.normal((world.V, world.V), key=mx.random.key(99)) * 3).tolist()
for _row in _WORLD_TABLE:
    _row[world.EOS] -= 6.0   # runs reach max_tokens instead of ending early


def _flat_world_logits(generated: list[int]) -> list[float]:
    return list(_WORLD_TABLE[generated[-1] if generated else world.PROMPT[-1]])


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
@pytest.mark.parametrize("processors", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_seeded_round_draws_what_the_seeded_plain_path_draws(monkeypatch, kind, processors, seed):
    monkeypatch.setattr(world, "_raw_logits", _flat_world_logits)
    plain_batch, _ = world._batch(kind, speculative_on=False, sampler=_seeded(seed), processors=processors,
                                  max_tokens=40)
    plain = world._run(plain_batch)

    mx.random.seed(seed + 100)  # a different global PRNG state must not matter
    spec_batch, draft = world._batch(kind, speculative_on=True, sampler=_seeded(seed), processors=processors,
                                     max_tokens=40)
    if kind == "mtp":
        world._mtp_hooks(monkeypatch, lambda: [draft])
    spec = world._run(spec_batch)

    assert spec == plain and len(plain) == 40
    # the rounds both accepted and rejected drafts: the positions were exercised on both sides of a rejection
    assert 0 < draft.speculative_total_accepted < draft.speculative_total_drafted


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_plain_ticks_between_rounds_draw_what_the_plain_path_draws(monkeypatch, kind):
    # a plain tick after a round feeds a bonus that already went out: its draw is the next position, not one past
    monkeypatch.setattr(world, "_raw_logits", _flat_world_logits)
    plain = world._run(world._batch(kind, speculative_on=False, sampler=_seeded(5), processors=False,
                                    max_tokens=40)[0])
    can_round = speculative.SpeculativeGenerationBatch._can_round
    turns = iter([True, False] * 40)
    monkeypatch.setattr(speculative.SpeculativeGenerationBatch, "_can_round",
                        lambda self: can_round(self) and next(turns, False))
    spec_batch, draft = world._batch(kind, speculative_on=True, sampler=_seeded(5), processors=False, max_tokens=40)
    if kind == "mtp":
        world._mtp_hooks(monkeypatch, lambda: [draft])
    assert world._run(spec_batch) == plain
    assert draft.speculative_total_drafted > 0


def test_the_flat_world_is_sensitive_to_the_seed(monkeypatch):
    monkeypatch.setattr(world, "_raw_logits", _flat_world_logits)
    runs = {tuple(world._run(world._batch("dflash", speculative_on=False, sampler=_seeded(seed), processors=False,
                                          max_tokens=40)[0])) for seed in (1, 2, 3)}
    assert len(runs) == 3


def test_a_seeded_dflash_proposal_draws_with_the_target_key_for_its_position(monkeypatch):
    seen = []

    class Recording(SeededSampler):
        def sample_target(self, logprobs, *, row_ids, positions):
            seen.append(list(positions))
            return super().sample_target(logprobs, row_ids=row_ids, positions=positions)

    sampler = Recording(4, lambda lp: lp, 1.0)
    proposals = []

    def draft_block(self, bonus, context, cache, block_size, draft_sampler, dtype):
        proposals.append(draft_sampler(mx.zeros((1, block_size - 1, world.V))))
        return mx.array([self.propose(int(bonus), block_size - 1)], dtype=dtype)

    monkeypatch.setattr(world._Draft, "draft_block", draft_block)
    batch, _ = world._batch("dflash", speculative_on=True, sampler=sampler, processors=False, max_tokens=60)
    batch.next()  # the pending token, then the first round at emitted = 1
    assert proposals and seen[1] == [1, 2, 3]   # seen[0] is the prefill's first draw


# ---- the other two paths a seed can take, and the seed's range

import threading

from mlx_engine import generate as engine_generate
from mlx_engine.utils.set_seed import set_seed


class _Tok:
    eos_token_ids = {63}

    def decode(self, token):
        return "x" if isinstance(token, int) else "x" * len(token)


class _SequentialKit:
    """What _sequential_generation reads from a ModelKit; the model itself is stream_generate's (faked)."""
    draft_model = None
    model = None
    prefill_step_size = 512
    tokenizer = _Tok()

    def __init__(self):
        self.pending_requests = {}
        self.generation_lock = threading.Lock()

    def is_shutdown(self):
        return False

    def process_prompt(self, prompt_tokens, *args):
        return prompt_tokens, None

    def is_cross_prompt_cache_active(self):
        return False


def _fake_stream_generate(*, sampler, max_tokens, **kwargs):
    logprobs = _logprobs_after(0)
    for _ in range(max_tokens):
        yield SimpleNamespace(token=int(sampler(logprobs).item()) % 63, text="x", logprobs=logprobs[0],
                              from_draft=False)


def _sequential_tokens(monkeypatch, seed, disturb=False):
    monkeypatch.setattr(engine_generate, "stream_generate", _fake_stream_generate)
    if disturb:
        mx.random.seed(12345)
    results = engine_generate._sequential_generation(_SequentialKit(), [1, 2, 3], temp=1.0, seed=seed,
                                                     max_tokens=24, request_id="r")
    return [t.id for r in results for t in r.tokens]


@pytest.mark.parametrize("seed", [0, 7, 2**63, 2**64 - 1])
def test_the_sequential_path_takes_any_64_bit_seed_and_replays_it(monkeypatch, seed):
    first = _sequential_tokens(monkeypatch, seed)
    assert len(first) == 24 and _sequential_tokens(monkeypatch, seed, disturb=True) == first


def test_on_the_sequential_path_seeds_that_differ_above_bit_31_do_not_collide(monkeypatch):
    assert _sequential_tokens(monkeypatch, 5) != _sequential_tokens(monkeypatch, 5 + 2**32)


def test_set_seed_takes_the_64_bit_range_and_refuses_what_is_outside_it():
    import numpy as np

    set_seed(5)
    low = np.random.randint(0, 2**31, size=4).tolist()
    set_seed(5 + 2**32)
    assert np.random.randint(0, 2**31, size=4).tolist() != low
    set_seed(2**64 - 1)
    for bad in (-1, 2**64):
        with pytest.raises(ValueError):
            set_seed(bad)


def _tiny_llama():
    from mlx_lm.models import llama

    model = llama.Model(llama.ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                                        num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=V,
                                        num_key_value_heads=4))
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: mx.random.normal(p.shape, key=mx.random.key(3)) * 0.5,
                                   model.parameters()))
    mx.eval(model.parameters())
    return model


def _mlx_lm_batch(model, rows, steps=20):
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(model, max_tokens=steps, completion_batch_size=4, prefill_batch_size=1)
    uids = gen.insert([p for p, _ in rows], [steps] * len(rows), samplers=[s for _, s in rows])
    out = {uid: [] for uid in uids}
    while True:
        prompt_responses, responses = gen.next()
        if not prompt_responses and not responses:
            break
        for r in responses:
            out[r.uid].append(r.token)
    gen.close()
    return [out[uid] for uid in uids]


def test_on_the_mlx_lm_batched_path_a_seeded_row_draws_the_same_alone_and_in_company():
    # BatchedModelKit (text-only models) hands each request's sampler to mlx-lm's BatchGenerator,
    # which calls it once per step: positions in call order
    model = _tiny_llama()
    alone = _mlx_lm_batch(model, [([1, 2, 3], _seeded(42))])[0]
    crowded = _mlx_lm_batch(model, [([5, 6], _seeded(7)), ([1, 2, 3], _seeded(42)),
                                    ([9], create_sampler(1.0, None, None, None, None))])[1]
    other = _mlx_lm_batch(model, [([1, 2, 3], _seeded(43))])[0]
    assert crowded == alone and len(alone) == 20
    assert other != alone


# ---- a consumer that stops reading takes its row out of the batch

class _Response(SimpleNamespace):
    pass


def _batched_kit(tokens):
    from mlx_engine.model_kit.batched_model_kit import BatchedModelKit

    kit = object.__new__(BatchedModelKit)
    kit.tokenizer = _Tok()
    kit._shutdown = threading.Event()
    kit._shutdown.set()  # __del__ calls shutdown(); nothing to stop here
    kit.removed = []
    kit.samplers = []

    def generate(*, sampler, **kwargs):
        kit.samplers.append(sampler)
        for t in tokens:
            yield _Response(token=t, text="x", token_logprob=0.0, top_logprobs=None, finish_reason=None,
                            from_draft=False)

    kit.generate = generate
    kit.remove = kit.removed.append
    return kit


def test_closing_a_batched_generation_early_removes_its_row():
    kit = _batched_kit([1, 2, 3, 4, 5])
    results = engine_generate._batched_generation(kit, [1], request_id="req-1", temp=1.0, seed=9)
    next(results)
    results.close()
    assert kit.removed == ["req-1"]
    assert isinstance(kit.samplers[0], SeededSampler) and kit.samplers[0].seed == 9


def test_a_batched_generation_read_to_the_end_removes_nothing():
    kit = _batched_kit([1, 2, 63])
    assert [r.stop_condition.stop_reason for r in engine_generate._batched_generation(kit, [1], request_id="r")
            if r.stop_condition] == ["eos_token"]
    assert kit.removed == []
