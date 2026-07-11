"""Deterministic virtual time & entropy (ctx.now / ctx.uuid + auto-instrument)."""

import time

from agent_replay import instrument
from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay


def test_ctx_now_recorded_and_replayed():
    def agent(ctx, task="t"):
        return {"t0": ctx.now(), "t1": ctx.now()}

    traj = record(agent, {}, session_id="now", seed=0)
    # Two now() calls captured as non-resamplable memory steps.
    now_steps = [s for s in traj.steps if s.name == "__now__"]
    assert len(now_steps) == 2
    assert all(s.resamplable is False for s in now_steps)
    # Factual replay serves the recorded timestamps verbatim.
    out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=99)
    assert out["t0"] == traj.result["t0"]
    assert out["t1"] == traj.result["t1"]


def test_ctx_uuid_deterministic_on_replay():
    def agent(ctx, task="t"):
        return {"id": ctx.uuid()}

    traj = record(agent, {}, session_id="uid", seed=0)
    out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=123)
    assert out["id"] == traj.result["id"]
    assert len(out["id"]) == 36  # canonical uuid string


def test_now_resampled_after_ablation_diverges_live():
    # When a now() call is resampled (not held), it re-reads live and is marked
    # diverged only if the op has no recorded counterpart; here it re-serves.
    def agent(ctx, task="t"):
        return {"t": ctx.now()}

    traj = record(agent, {}, session_id="n2", seed=0)
    # Resample-all plan: non-resamplable now() still serves the recorded value.
    out = replay(agent, traj, ReplayPlan(held=set()), seed=5)
    assert out["t"] == traj.result["t"]


def test_enable_virtual_time_patches_stdlib():
    instrument.enable_virtual_time()
    try:

        def agent(**task):
            # Unmodified stdlib calls, via ambient context.
            return {"a": time.time(), "b": time.time()}

        traj = record(agent, {}, session_id="vt", seed=0, pass_context=False)
        assert len([s for s in traj.steps if s.name == "__now__"]) == 2
        out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=1, pass_context=False)
        assert out["a"] == traj.result["a"]
    finally:
        instrument.disable_virtual_time()

    # After disable, time.time is the real clock again (returns ~now, no recording).
    assert instrument.current_context() is None
    assert abs(time.time() - time.time()) < 1.0


def test_virtual_time_passthrough_when_no_run():
    with instrument.virtual_time():
        t = time.time()  # no active run -> real clock
    assert t > 1_600_000_000  # a real unix timestamp


def test_uuid_virtual_time_roundtrip():
    import uuid as uuidlib

    with instrument.virtual_time():

        def agent(**task):
            return {"id": str(uuidlib.uuid4())}

        traj = record(agent, {}, session_id="vu", seed=0, pass_context=False)
        out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=7, pass_context=False)
        assert out["id"] == traj.result["id"]


def test_async_now():
    import asyncio

    async def agent(ctx, task="t"):
        v = await ctx.llm("s", produce=_mk())
        return {"t": ctx.now(), "v": v}

    def _mk():
        async def p():
            return "x"

        return p

    traj = record(agent, {}, session_id="an", seed=0)
    assert any(s.name == "__now__" for s in traj.steps)
    out = asyncio.run(
        __import__("agent_replay").areplay(agent, traj, ReplayPlan.factual(len(traj)), seed=3)
    )
    assert out["t"] == traj.result["t"]
