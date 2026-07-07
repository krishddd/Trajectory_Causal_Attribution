"""Idempotency-key cassette matching keeps *branching* agents sound.

The v0.1 replayer matched live calls to recorded steps by call *position*. When an
upstream ablation changed the control flow, a held step served the wrong step's
recorded output into a different operation (cross-contamination) — which made
Shapley coalition plans unsound for any agent whose step sequence depends on
earlier outputs. v0.2 matches by ``Step.op_key`` (kind + name + inputs), so a held
step is only served from the cassette when the same operation actually recurs.
"""

from agent_replay.attribution import attribute
from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay


def branching_agent(ctx, task="t"):
    """Step 1's *operation* depends on step 0's output (a routing decision)."""
    route = ctx.llm("router", produce=lambda: "A" if ctx.rng.random() < 0.5 else "B")
    if route == "A":
        x = ctx.tool("fetch_a", produce=lambda: "data_a")
    else:
        x = ctx.tool("fetch_b", produce=lambda: "data_b")
    y = ctx.llm("answer", produce=lambda: f"ans:{x}")
    return {"route": route, "x": x, "y": y}


def _record_route(target_route):
    for seed in range(50):
        traj = record(branching_agent, {}, session_id="branch", seed=seed)
        if traj.result["route"] == target_route:
            return traj
    raise AssertionError(f"no seed produced route {target_route}")


def test_key_matching_no_cross_contamination():
    traj = _record_route("B")  # recorded step 1 is fetch_b -> data_b
    # Hold steps 1 and 2 factual, resample step 0 (the router). When the router
    # flips to A, the live fetch_a call must NOT receive fetch_b's recorded output.
    plan = ReplayPlan(held={1, 2})
    for seed in range(40):
        r = replay(branching_agent, traj, plan, seed=seed)
        if r["route"] == "A":
            assert r["x"] == "data_a"  # served live, not cross-contaminated
        else:
            assert r["x"] == "data_b"


def test_position_matching_would_contaminate():
    """Documents the old behaviour: positional matching cross-contaminates."""
    traj = _record_route("B")
    plan = ReplayPlan(held={1, 2})
    contaminated = False
    for seed in range(40):
        r = replay(branching_agent, traj, plan, seed=seed, match="position")
        if r["route"] == "A" and r["x"] == "data_b":
            contaminated = True
            break
    assert contaminated  # the bug the key-matcher fixes


def test_diverged_flag_set_on_branch_change():
    from agent_replay.replayer import ReplayContext

    traj = _record_route("B")
    # Force the router to A while holding the (path-B) downstream steps.
    plan = ReplayPlan(held={1, 2}, forced={0: "A"})
    ctx = ReplayContext(traj, plan, seed=1)
    branching_agent(ctx, **traj.task)
    assert ctx.diverged  # fetch_a had no recorded counterpart


def test_shapley_runs_on_branching_agent():
    """Coalition plans (Shapley) execute without contamination on a branch agent."""
    traj = _record_route("B")

    def verifier(result):
        # Fail the run so attribution is defined; failure tied to the answer step.
        return 0.0 if result["y"].startswith("ans:data") else 1.0

    # Should complete and produce finite, well-formed values.
    result = attribute(
        traj, branching_agent, verifier, rollouts=30, method="shapley", permutation_pairs=6
    )
    assert len(result.steps) == len(traj)
    assert all(s.shapley is not None for s in result.steps)
