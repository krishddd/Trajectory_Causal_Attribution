"""Universal instrumentation: ambient context, decorators, and monkeypatching.

These verify the *mechanism* against a dummy module (framework SDKs are not a test
dependency); the ``RECIPES`` for real frameworks are exercised by the same
``patch`` path here.
"""

import sys
import types

import pytest

from agent_replay import attribute, instrument
from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay


def test_current_context_none_outside_run():
    assert instrument.current_context() is None


def test_record_step_decorator_captures_steps():
    @instrument.tool
    def search(q):
        return f"hits:{q}"

    def agent(**task):
        return {"answer": search(task["q"])}

    traj = record(agent, {"q": "why"}, session_id="deco", seed=0, pass_context=False)
    assert len(traj) == 1
    assert traj.steps[0].kind.value == "tool"
    assert traj.steps[0].name == "search"
    assert traj.result["answer"] == "hits:why"


def test_wrapper_passthrough_when_no_run():
    calls = {"n": 0}

    @instrument.llm
    def gen(x):
        calls["n"] += 1
        return x * 2

    # No active run -> transparent passthrough, no recording.
    assert gen(3) == 6
    assert calls["n"] == 1


def test_ambient_replay_roundtrip():
    def flaky(ctx_val):
        return "ok"

    flaky = instrument.wrap(flaky, "llm", "flaky")

    def agent(**task):
        return {"v": flaky(task["seed_marker"])}

    traj = record(agent, {"seed_marker": 1}, session_id="amb", seed=0, pass_context=False)
    # Factual replay via ambient context reproduces the recorded output.
    out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=5, pass_context=False)
    assert out["v"] == "ok"


def test_available_frameworks_nonempty():
    fws = instrument.available_frameworks()
    assert "openai" in fws and "langchain" in fws and "crewai" in fws


def _install_dummy_sdk():
    mod = types.ModuleType("dummy_sdk")

    class Client:
        def create(self, prompt="p"):
            return {"text": f"resp:{prompt}"}

    mod.Client = Client
    sys.modules["dummy_sdk"] = mod
    return mod


def test_patch_and_unpatch_dummy_sdk():
    mod = _install_dummy_sdk()
    try:
        assert instrument.patch("dummy_sdk.Client.create", "llm", "dummy.create")
        client = mod.Client()

        def agent(**task):
            return client.create(prompt=task["p"])

        traj = record(agent, {"p": "hi"}, session_id="patch", seed=0, pass_context=False)
        assert len(traj) == 1
        assert traj.steps[0].name == "dummy.create"
        assert traj.result == {"text": "resp:hi"}
    finally:
        instrument.unpatch("dummy_sdk.Client.create")
        del sys.modules["dummy_sdk"]

    # After unpatch the method is restored (no recording without an active run).
    assert not instrument.unpatch("dummy_sdk.Client.create")


def test_install_unknown_framework_raises():
    with pytest.raises(KeyError):
        instrument.install("not_a_framework")


def test_install_missing_sdk_is_best_effort():
    # A recipe pointing at an absent SDK is skipped, not raised (best-effort).
    instrument.RECIPES["_absent_"] = [("definitely_missing_pkg.Client.go", "llm", "absent.go")]
    try:
        patched = instrument.install("_absent_")
        assert patched == []  # skipped silently
    finally:
        del instrument.RECIPES["_absent_"]


def test_install_missing_sdk_strict_raises():
    instrument.RECIPES["_absent2_"] = [("definitely_missing_pkg.x", "llm", "x")]
    try:
        import pytest as _pytest

        with _pytest.raises(ImportError):
            instrument.install("_absent2_", strict=True)
    finally:
        del instrument.RECIPES["_absent2_"]


def test_record_agent_end_to_end():
    @instrument.tool
    def step(i):
        return "bad" if i == 1 else "ok"

    def agent(**task):
        trace = [step(i) for i in range(3)]
        return {"trace": trace, "ok": "bad" not in trace}

    traj = instrument.record_agent(agent, {}, session_id="auto", seed=0)
    assert len(traj) == 3
    result = attribute(
        traj, agent, lambda r: 1.0 if r["ok"] else 0.0, rollouts=20, pass_context=False
    )
    assert result.failed
