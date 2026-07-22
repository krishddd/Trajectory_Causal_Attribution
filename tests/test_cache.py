"""Cross-analysis rollout cache: reuse prefix-hold rollouts, never coalition ones."""

from _demo_agent import buggy_agent, health_scorer, verifier
from agent_replay import RolloutCache, analyze
from agent_replay.ablation import AblationEngine
from agent_replay.attribution import attribute, shapley_attribution
from agent_replay.drift import drift


def test_cache_makes_ablate_from_reused(recording):
    cache = RolloutCache()
    engine = AblationEngine(buggy_agent, recording, verifier, cache=cache)
    first = engine.ablate_from(2, rollouts=30)
    assert cache.misses == 1 and cache.hits == 0
    second = engine.ablate_from(2, rollouts=30)
    assert cache.hits == 1
    assert first == second  # identical cached list


def test_cache_matches_uncached_result(recording, fail_step):
    # Same base_seed + rollouts -> cached attribution must equal the uncached one.
    uncached = attribute(recording, buggy_agent, verifier, rollouts=60, base_seed=500)
    cached = attribute(
        recording, buggy_agent, verifier, rollouts=60, base_seed=500, cache=RolloutCache()
    )
    assert cached.culprit_index == uncached.culprit_index
    assert [round(s.attribution, 9) for s in cached.steps] == [
        round(s.attribution, 9) for s in uncached.steps
    ]


def test_coalition_bypasses_cache(recording):
    # Shapley coalition values must NOT be cached (independent draws for variance).
    cache = RolloutCache()
    engine = AblationEngine(buggy_agent, recording, verifier, cache=cache)
    shapley_attribution(engine, rollouts=20, permutation_pairs=3)
    # Coalition plans never touch the cache: no misses recorded from them.
    assert cache.misses == 0
    assert cache.hits == 0


def test_analyze_shares_rollouts_between_attribute_and_drift(recording, fail_step):
    result = analyze(
        recording,
        buggy_agent,
        verifier,
        rollouts=40,
        base_seed=777,
        state_scorer=health_scorer,
    )
    cache = result["cache"]
    # drift's per-step ablate_from(i) reuses attribution's contrastive rollouts.
    assert cache.hits > 0
    # And the shared-cache results equal running each analysis standalone.
    standalone_attr = attribute(recording, buggy_agent, verifier, rollouts=40, base_seed=777)
    standalone_drift = drift(
        recording, buggy_agent, verifier, rollouts=40, base_seed=777, state_scorer=health_scorer
    )
    assert result["attribution"].culprit_index == standalone_attr.culprit_index
    assert result["drift"].commitment_index == standalone_drift.commitment_index
    assert [round(p.p_success, 9) for p in result["drift"].points] == [
        round(p.p_success, 9) for p in standalone_drift.points
    ]


def test_analyze_handles_passing_run():
    from agent_replay.recorder import record

    def passing(ctx, task="t"):
        return {"ok": True}

    traj = record(passing, {}, session_id="p", seed=0, verifier=lambda r: 1.0)
    out = analyze(traj, passing, lambda r: 1.0 if r["ok"] else 0.0, rollouts=10)
    assert out["attribution"] is None  # passing run: nothing to attribute
    assert out["drift"] is not None
