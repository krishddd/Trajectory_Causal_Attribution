"""Tests for deterministic replay and intervention plans."""

from agent_replay.mock_agent import buggy_agent, verifier
from agent_replay.replayer import ReplayPlan, replay


def test_factual_replay_reproduces_outcome(recording):
    plan = ReplayPlan.factual(len(recording))
    result = replay(buggy_agent, recording, plan, seed=recording.seed)
    # Held steps return recorded outputs -> identical result.
    assert result["trace"] == [s.output for s in recording.steps]
    assert result["ok"] is False
    assert verifier(result) == 0.0


def test_factual_replay_is_seed_independent(recording):
    plan = ReplayPlan.factual(len(recording))
    r1 = replay(buggy_agent, recording, plan, seed=111)
    r2 = replay(buggy_agent, recording, plan, seed=999)
    # All steps held -> policy never runs -> seed cannot matter.
    assert r1["trace"] == r2["trace"]


def test_ablate_from_before_fail_can_rescue(recording, fail_step):
    plan = ReplayPlan.ablate_from(fail_step, len(recording))
    # At least one seed should rescue the run (draw OK at the fail step).
    rescued = any(replay(buggy_agent, recording, plan, seed=s)["ok"] for s in range(30))
    assert rescued


def test_ablate_after_fail_stays_failed(recording, fail_step):
    plan = ReplayPlan.ablate_from(fail_step + 1, len(recording))
    # Fail step is held at its factual BAD value -> always fails.
    for s in range(30):
        assert replay(buggy_agent, recording, plan, seed=s)["ok"] is False


def test_forced_intervention(recording, fail_step):
    # Force the fail step to OK: run should succeed regardless of seed.
    plan = ReplayPlan(held=set(range(fail_step)), forced={fail_step: "OK"})
    for s in range(10):
        assert replay(buggy_agent, recording, plan, seed=s)["ok"] is True


def test_plan_decisions():
    plan = ReplayPlan(held={0, 1}, forced={2: "X"}, removed={3})
    assert plan.decision(0) == "hold"
    assert plan.decision(2) == "force"
    assert plan.decision(3) == "remove"
    assert plan.decision(4) == "resample"


def test_coalition_plan():
    plan = ReplayPlan.coalition({1, 3})
    assert plan.held == {1, 3}
    assert plan.decision(1) == "hold"
    assert plan.decision(2) == "resample"
