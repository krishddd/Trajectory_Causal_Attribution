"""Async agents: record/replay/attribute an ``async def`` agent transparently."""

import asyncio

from agent_replay.attribution import attribute
from agent_replay.recorder import arecord, record
from agent_replay.replayer import ReplayPlan, areplay, replay


async def _sleep0():
    # A trivial await point so the produce policies are genuinely async.
    await asyncio.sleep(0)


async def async_agent(ctx, task="t", fail_at=1, n=4, p=0.7):
    trace = []
    ok = True
    for i in range(n):

        async def produce(step=i):
            await _sleep0()
            if step == fail_at:
                return "bad" if ctx.rng.random() < p else "ok"
            return "ok"

        act = await ctx.tool(f"s{i}", produce=produce, ctx=trace[-2:])
        trace.append(act)
        if act == "bad":
            ok = False
    return {"trace": trace, "ok": ok}


def verifier(result):
    return 1.0 if result["ok"] else 0.0


def _record_failing():
    for seed in range(100):
        traj = record(async_agent, {}, session_id=f"a{seed}", seed=seed, verifier=verifier)
        if traj.outcome_score == 0.0:
            return traj
    raise AssertionError("no failing async recording")


def test_sync_record_detects_and_runs_async_agent():
    traj = record(async_agent, {}, session_id="a", seed=1, verifier=verifier)
    assert len(traj) == 4
    assert traj.outcome_score in (0.0, 1.0)


def test_arecord_native():
    traj = asyncio.run(arecord(async_agent, {}, session_id="a", seed=1, verifier=verifier))
    assert len(traj) == 4


def test_async_factual_replay_reproduces():
    traj = _record_failing()
    # Sync replay auto-dispatches to areplay for async agents.
    out = replay(async_agent, traj, ReplayPlan.factual(len(traj)), seed=traj.seed)
    assert out["trace"] == [s.output for s in traj.steps]


def test_async_areplay_native():
    traj = _record_failing()
    out = asyncio.run(areplay(async_agent, traj, ReplayPlan.factual(len(traj)), seed=5))
    assert out["trace"] == [s.output for s in traj.steps]


def test_attribute_async_agent_end_to_end():
    # The entire (synchronous) attribution pipeline works on an async agent.
    traj = _record_failing()
    result = attribute(traj, async_agent, verifier, rollouts=80, method="both", repair=True)
    assert result.failed
    assert result.point_of_commitment == 1
    assert result.culprit_index == 1
    assert result.repair is not None


def test_async_branch_safety():
    # Idempotency-key matching holds for async agents too.
    async def router(ctx, task="t"):
        route = await ctx.llm("router", produce=_route)
        if route == "A":
            x = await ctx.tool("fa", produce=_data_a)
        else:
            x = await ctx.tool("fb", produce=_data_b)
        return {"route": route, "x": x}

    async def _route():
        await _sleep0()
        return "A" if ctx_holder["ctx"].rng.random() < 0.5 else "B"

    ctx_holder = {}

    async def _data_a():
        await _sleep0()
        return "data_a"

    async def _data_b():
        await _sleep0()
        return "data_b"

    # Bind ctx for the produce closures via a recording wrapper.
    async def router2(ctx, task="t"):
        ctx_holder["ctx"] = ctx
        return await router(ctx, task)

    # Find a B-route recording.
    traj = None
    for seed in range(50):
        t = record(router2, {}, session_id=f"r{seed}", seed=seed)
        if t.result["route"] == "B":
            traj = t
            break
    assert traj is not None
    # Hold the path-B downstream steps, resample the router.
    plan = ReplayPlan(held={1, 2})
    for seed in range(20):
        r = replay(router2, traj, plan, seed=seed)
        if r["route"] == "A":
            assert r["x"] == "data_a"  # not cross-contaminated with data_b
