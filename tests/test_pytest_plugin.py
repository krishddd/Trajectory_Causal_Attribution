"""The pytest integration: flakiness-aware assertions with auto-attribution."""

import pytest

from agent_replay.pytest_plugin import (
    AgentFlakyError,
    assert_agent_passes,
    measure_failure_rate,
)


def _verifier(result):
    return 1.0 if result["ok"] else 0.0


def reliable_agent(ctx, task="t"):
    # Never fails.
    a = ctx.llm("s0", produce=lambda: "ok")
    return {"a": a, "ok": True}


def flaky_agent(ctx, task="t", fail_at=1, n=3, p=0.8):
    trace = []
    ok = True
    for i in range(n):

        def produce(step=i):
            if step == fail_at:
                return "bad" if ctx.rng.random() < p else "ok"
            return "ok"

        act = ctx.tool(f"s{i}", produce=produce, ctx=trace[-2:])
        trace.append(act)
        if act == "bad":
            ok = False
    return {"trace": trace, "ok": ok}


def test_measure_failure_rate_reliable():
    m = measure_failure_rate(reliable_agent, {}, _verifier, rollouts=10)
    assert m["p_fail"] == 0.0
    assert m["n_fail"] == 0
    assert len(m["trajectories"]) == 10


def test_measure_failure_rate_flaky():
    m = measure_failure_rate(flaky_agent, {}, _verifier, rollouts=20)
    assert m["p_fail"] > 0.5  # ~0.8 fail probability
    assert len(m["failures"]) == m["n_fail"]


def test_assert_agent_passes_reliable():
    # A reliable agent passes and returns its measurement.
    m = assert_agent_passes(reliable_agent, {}, _verifier, rollouts=15, p_fail_max=0.05)
    assert m["p_fail"] == 0.0


def test_assert_agent_passes_flaky_raises_with_attribution(tmp_path):
    report = str(tmp_path / "fail_report.html")
    with pytest.raises(AgentFlakyError) as exc:
        assert_agent_passes(
            flaky_agent,
            {},
            _verifier,
            rollouts=20,
            p_fail_max=0.05,
            attribution_rollouts=40,
            report_path=report,
        )
    err = exc.value
    # The failure message localises the culprit step and explains it.
    assert "failure budget" in str(err)
    assert "step 1" in str(err).lower() or "s1" in str(err)
    # Structured data is attached for programmatic inspection.
    assert err.p_fail > 0.05
    assert err.attribution is not None
    assert err.explanation is not None
    assert err.attribution.culprit_index == 1
    # HTML report artifact was written.
    assert (tmp_path / "fail_report.html").exists()


def test_assert_agent_passes_no_attribution_option():
    with pytest.raises(AgentFlakyError) as exc:
        assert_agent_passes(
            flaky_agent,
            {},
            _verifier,
            rollouts=15,
            p_fail_max=0.0,
            attribute_on_failure=False,
        )
    assert exc.value.attribution is None


def test_plugin_fixtures_registered(agent_replay_session, assert_agent):
    # The pytest11 entry point (mirrored via conftest under PYTHONPATH) exposes
    # these fixtures to any test session.
    assert agent_replay_session is not None
    assert callable(assert_agent)
