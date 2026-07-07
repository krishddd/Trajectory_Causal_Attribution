"""Ground-truth synthetic SCMs, mirroring the Causal Agent Replay validation.

The paper validates its estimators on synthetic structural causal models with
*analytically known* attribution — pivotal single steps, and two-step AND/OR
interactions — and checks the Shapley efficiency axiom quantitatively. These
tests encode those checks so a regression in the estimator is caught immediately.
"""

from agent_replay.ablation import AblationEngine
from agent_replay.attribution import attribute, shapley_attribution
from agent_replay.recorder import record

P = 0.5  # per-step "bad" probability


def _ok(result):
    return 1.0 if result["ok"] else 0.0


# --- AND failure: fails iff BOTH steps go bad -------------------------------


def and_agent(ctx, task="and"):
    a = ctx.llm("step_a", produce=lambda: "bad" if ctx.rng.random() < P else "ok")
    b = ctx.llm("step_b", produce=lambda: "bad" if ctx.rng.random() < P else "ok")
    return {"a": a, "b": b, "ok": not (a == "bad" and b == "bad")}


def _record_failing(agent, sid):
    for seed in range(200):
        traj = record(agent, {}, session_id=sid, seed=seed, verifier=_ok)
        if traj.outcome_score == 0.0:
            return traj
    raise AssertionError("no failing seed found")


def test_and_failure_shapley_splits_credit():
    """AND-failure: the two culprits must share credit roughly equally."""
    traj = _record_failing(and_agent, "and")
    engine = AblationEngine(and_agent, traj, _ok)
    steps = shapley_attribution(engine, rollouts=120, permutation_pairs=16)
    phi = {s.index: s.shapley for s in steps}
    # Symmetric roles -> near-equal Shapley shares (not one step blamed 100%).
    assert abs(phi[0] - phi[1]) < 0.18
    assert phi[0] > 0.1 and phi[1] > 0.1


def test_and_failure_efficiency_axiom():
    """sum(phi) == v(full) - v(empty), estimated independently (CAR's 0.909 check)."""
    traj = _record_failing(and_agent, "and2")
    engine = AblationEngine(and_agent, traj, _ok)
    steps = shapley_attribution(engine, rollouts=150, permutation_pairs=16)
    total = sum(s.shapley for s in steps)
    v_full = engine.coalition_value({0, 1}, rollouts=150, seed_tag=777)  # == 1.0
    v_empty = engine.coalition_value(set(), rollouts=150, seed_tag=778)
    assert abs(total - (v_full - v_empty)) < 0.15


# --- OR failure: fails iff EITHER step goes bad -----------------------------


def or_agent(ctx, task="or"):
    a = ctx.llm("step_a", produce=lambda: "bad" if ctx.rng.random() < P else "ok")
    b = ctx.llm("step_b", produce=lambda: "bad" if ctx.rng.random() < P else "ok")
    return {"a": a, "b": b, "ok": not (a == "bad" or b == "bad")}


def test_or_failure_efficiency_axiom():
    traj = _record_failing(or_agent, "or")
    engine = AblationEngine(or_agent, traj, _ok)
    steps = shapley_attribution(engine, rollouts=150, permutation_pairs=16)
    total = sum(s.shapley for s in steps)
    v_full = engine.coalition_value({0, 1}, rollouts=150, seed_tag=881)
    v_empty = engine.coalition_value(set(), rollouts=150, seed_tag=882)
    assert abs(total - (v_full - v_empty)) < 0.15


# --- Pivotal single step: a chain with one decisive step --------------------


def pivotal_agent(ctx, task="pivot", fail_at=2, n=5):
    trace = []
    ok = True
    for i in range(n):

        def produce(step=i):
            if step == fail_at:
                return "bad" if ctx.rng.random() < 0.7 else "ok"
            return "ok"

        act = ctx.llm(f"s{i}", produce=produce, ctx=trace[-2:])
        trace.append(act)
        if act == "bad":
            ok = False
    return {"trace": trace, "ok": ok}


def test_pivotal_single_step_localised():
    traj = _record_failing(pivotal_agent, "pivot")
    result = attribute(traj, pivotal_agent, _ok, rollouts=120, method="both")
    assert result.point_of_commitment == 2
    assert result.culprit_index == 2
    # The pivotal step also carries the largest Shapley share.
    best = max(result.steps, key=lambda s: s.shapley)
    assert best.index == 2
