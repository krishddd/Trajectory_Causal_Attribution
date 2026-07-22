"""Tests for the ablation engine (stochastic run-forward)."""

from _demo_agent import buggy_agent, verifier
from agent_replay.ablation import AblationEngine


def _engine(recording):
    return AblationEngine(buggy_agent, recording, verifier)


def test_factual_fail_is_failure(recording):
    engine = _engine(recording)
    fails = engine.factual_fail()
    assert all(fails)  # factual run failed


def test_ablate_from_fail_step_reduces_failure(recording, fail_step):
    engine = _engine(recording)
    abl = engine.ablate_from(fail_step, rollouts=80)
    rate = sum(abl) / len(abl)
    # Resampling the culprit rescues the run a meaningful fraction of the time.
    assert 0.0 < rate < 1.0


def test_ablate_after_fail_step_locked(recording, fail_step):
    engine = _engine(recording)
    abl = engine.ablate_from(fail_step + 1, rollouts=40)
    assert all(abl)  # failure locked in


def test_parallel_rollouts_match_serial(recording, fail_step):
    plan_seed = fail_step
    serial = AblationEngine(buggy_agent, recording, verifier, max_workers=1)
    parallel = AblationEngine(buggy_agent, recording, verifier, max_workers=4)
    # Each rollout is pure in (plan, seed_tag, k), so parallel execution must
    # return byte-identical results — only faster. Guards the determinism invariant.
    s = serial.ablate_from(plan_seed, rollouts=32)
    p = parallel.ablate_from(plan_seed, rollouts=32)
    assert s == p
    # Adaptive path (batched) must agree too.
    sa = serial.ablate_from(plan_seed, rollouts=64, adaptive=True)
    pa = parallel.ablate_from(plan_seed, rollouts=64, adaptive=True)
    assert sa == pa


def test_coalition_value_bounds(recording):
    engine = _engine(recording)
    v_empty = engine.coalition_value(set(), rollouts=40, seed_tag=1)
    v_full = engine.coalition_value(set(range(len(recording))), rollouts=40, seed_tag=2)
    assert 0.0 <= v_empty <= 1.0
    assert v_full == 1.0  # holding everything factual == guaranteed failure


def test_rollouts_are_independent_across_seed_tags(recording, fail_step):
    engine = _engine(recording)
    a = engine.ablate_from(fail_step, rollouts=20, seed_tag=10)
    b = engine.ablate_from(fail_step, rollouts=20, seed_tag=11)
    # Different seed tags should not produce identical failure sequences.
    assert a != b or len(set(a)) == 1
