"""Adaptive (sequential-stopping) rollouts: same verdict, fewer rollouts."""

from agent_replay.ablation import AblationEngine
from agent_replay.attribution import attribute, contrastive_attribution
from agent_replay.mock_agent import buggy_agent, verifier
from agent_replay.replayer import ReplayPlan


def test_run_plan_adaptive_stops_early_on_decisive_plan(recording):
    engine = AblationEngine(buggy_agent, recording, verifier)
    # ablate_from(fail_step+1) is locked-in -> always fails -> CI tightens fast.
    plan = ReplayPlan.ablate_from(len(recording), len(recording))  # all held == factual fail
    fails = engine.run_plan_adaptive(
        plan, target_ci_width=0.3, min_rollouts=8, max_rollouts=200, batch=8
    )
    assert all(fails)  # deterministic failure
    assert len(fails) < 200  # stopped well before the cap
    assert len(fails) >= 8


def test_run_plan_adaptive_respects_min_and_max(recording):
    engine = AblationEngine(buggy_agent, recording, verifier)
    plan = ReplayPlan.ablate_from(0, len(recording))
    fails = engine.run_plan_adaptive(
        plan, target_ci_width=0.0, min_rollouts=16, max_rollouts=32, batch=8
    )
    # Impossible target width -> runs to the cap; never below min.
    assert 16 <= len(fails) <= 32


def test_adaptive_contrastive_localises_same_culprit(recording, fail_step):
    # Adaptive and fixed-N must agree on the point of commitment.
    adaptive = contrastive_attribution(recording_engine(recording), 200, adaptive=True)
    from agent_replay.attribution import point_of_commitment

    assert point_of_commitment(adaptive) == fail_step


def recording_engine(recording):
    return AblationEngine(buggy_agent, recording, verifier)


def test_adaptive_uses_fewer_rollouts_than_cap(recording, fail_step):
    # The locked-in steps after the culprit resolve almost immediately.
    engine = recording_engine(recording)
    cap = 200
    after = engine.ablate_from(fail_step + 1, cap, adaptive=True, target_ci_width=0.25)
    assert len(after) < cap  # decisive (always-fail) step stops early


def test_attribute_adaptive_end_to_end(recording, fail_step):
    result = attribute(
        recording, buggy_agent, verifier, rollouts=150, method="contrastive", adaptive=True
    )
    assert result.point_of_commitment == fail_step
    assert result.culprit_index == fail_step
