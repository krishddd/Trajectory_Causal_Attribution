"""Multi-trajectory aggregation: systematic step blame across many failures."""

from _demo_agent import buggy_agent, verifier
from agent_replay import aggregate, aggregate_runs
from agent_replay.attribution import attribute
from agent_replay.recorder import record


def _failing_trajectories(n, fail_step=3, n_steps=6):
    """Record ``n`` distinct failing runs of the fixture agent (same task)."""
    trajs = []
    seed = 1
    while len(trajs) < n:
        task = {"task": "t", "fail_step": fail_step, "n_steps": n_steps}
        traj = record(buggy_agent, task, session_id=f"r{seed}", seed=seed, verifier=verifier)
        if traj.outcome_score is not None and traj.outcome_score < 0.5:
            trajs.append(traj)
        seed += 1
    return trajs


def test_aggregate_localises_systematic_weak_step_by_name():
    trajs = _failing_trajectories(6, fail_step=3)
    agg = aggregate_runs(
        trajs,
        buggy_agent,
        verifier,
        rollouts=50,
        label="support-agent",
    )
    assert agg.n_runs == 6
    # The fatal tool step is always step 3 named "tool_step_3": it must be the
    # systematic culprit, blamed by name in every run.
    assert agg.systematic_culprit == "tool:tool_step_3"
    top = agg.steps[0]
    assert top.name == "tool_step_3"
    assert top.n_culprit == top.n_present == 6
    assert top.culprit_rate == 1.0
    # A benign reasoning step should not be the systematic culprit.
    benign = [s for s in agg.steps if s.name == "reason_step_0"][0]
    assert benign.n_culprit == 0


def test_aggregate_pools_ci_over_runs():
    trajs = _failing_trajectories(5, fail_step=3)
    results = [attribute(t, buggy_agent, verifier, rollouts=40) for t in trajs]
    agg = aggregate(results)
    top = agg.steps[0]
    # Pooled interval brackets the mean and is a real interval.
    assert top.ci.low <= top.mean_attribution <= top.ci.high
    assert len(top.points) == top.n_present


def test_aggregate_skips_passing_runs():
    trajs = _failing_trajectories(3, fail_step=3)

    # A trivially-passing run of a different agent shape, mixed in.
    def passing(ctx, task="t"):
        return {"ok": True}

    passing_traj = record(passing, {}, session_id="p", seed=0, verifier=lambda r: 1.0)
    agg = aggregate_runs([*trajs, passing_traj], buggy_agent, verifier, rollouts=40)
    assert agg.n_runs == 3
    assert agg.n_skipped == 1


def test_aggregate_text_and_dict():
    trajs = _failing_trajectories(4, fail_step=3)
    agg = aggregate_runs(trajs, buggy_agent, verifier, rollouts=40, label="svc")
    text = agg.to_text()
    assert "Aggregate attribution for 'svc'" in text
    assert "Systematic weak step" in text
    d = agg.to_dict()
    assert d["systematic_culprit"] == "tool:tool_step_3"
    assert d["n_runs"] == 4
    assert len(d["steps"]) == len(agg.steps)


def test_aggregate_empty_is_safe():
    agg = aggregate([])
    assert agg.n_runs == 0
    assert agg.systematic_culprit is None
    assert agg.steps == []
