"""Gate CI on agent reliability, and get *which step broke* when it doesn't.

Run:  pytest examples/test_agent_with_pytest.py

`assert_agent_passes` runs the agent many times (agents are stochastic, so one
green run is not a pass), fails the test if the failure rate exceeds the budget,
and — crucially — puts a counterfactual attribution of a failing run into the
test output: the culprit step, why it failed, and the minimal fix.
"""

from agent_replay.pytest_plugin import assert_agent_passes


def verifier(result):
    return 1.0 if result["ok"] else 0.0


def my_agent(ctx, question="hi"):
    plan = ctx.llm("plan", produce=lambda: {"q": question})
    # A flaky retrieval step that corrupts the context 30% of the time.
    hits = ctx.tool("search", produce=lambda: "bad" if ctx.rng.random() < 0.3 else "good")
    draft = ctx.llm("answer", produce=lambda: hits, context=hits)
    return {"plan": plan, "draft": draft, "ok": draft == "good"}


def test_my_agent_is_reliable(tmp_path):
    # Require the agent to fail no more than 5% of the time over 40 rollouts.
    # If it exceeds that, the AssertionError names the culprit step and writes a report.
    assert_agent_passes(
        my_agent,
        {"question": "why did it fail?"},
        verifier,
        rollouts=40,
        p_fail_max=0.05,
        report_path=str(tmp_path / "agent_failure_report.html"),
    )
