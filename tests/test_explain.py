"""Explainability layer: traceable narrative over an attribution result."""

from _demo_agent import buggy_agent, make_recording, verifier
from agent_replay.attribution import attribute
from agent_replay.explain import (
    ROLE_DECISIVE,
    ROLE_LOCKED_IN,
    Explanation,
    explain,
)


def _result_and_traj():
    traj = make_recording("explain-test")
    result = attribute(traj, buggy_agent, verifier, rollouts=100, method="both", repair=True)
    return result, traj


def test_explanation_localises_decisive_step():
    result, traj = _result_and_traj()
    exp = explain(result, traj)
    assert exp.mode == "failure"
    assert f"step {result.culprit_index}" in exp.headline
    # The decisive role sits on the culprit; later steps are locked-in.
    decisive = [t for t in exp.trace if t.role == ROLE_DECISIVE]
    assert len(decisive) == 1 and decisive[0].index == result.culprit_index
    locked = [t for t in exp.trace if t.role == ROLE_LOCKED_IN]
    assert all(t.index > result.culprit_index for t in locked)


def test_explanation_names_recorded_action():
    result, traj = _result_and_traj()
    exp = explain(result, traj)
    decisive = next(t for t in exp.trace if t.role == ROLE_DECISIVE)
    # The decisive action is the recorded (bad) output of the culprit step.
    assert decisive.action == traj.steps[result.culprit_index].output
    assert decisive.action in exp.what


def test_explanation_why_has_numbers_and_fix():
    result, traj = _result_and_traj()
    exp = explain(result, traj)
    assert "rescue" in exp.why.lower()
    assert "point of commitment" in exp.why.lower()
    if result.repair and result.repair.valid:
        assert "repair" in exp.fix.lower() or "constrain" in exp.fix.lower()


def test_explanation_text_and_markdown_render():
    result, traj = _result_and_traj()
    exp = explain(result, traj)
    text = exp.to_text()
    assert "WHAT:" in text and "WHY:" in text and "Causal trace" in text
    md = exp.to_markdown()
    assert md.startswith("### ")
    assert "| Step | Role | Evidence |" in md
    # Text output is ASCII-safe (renders on any console).
    text.encode("ascii")


def test_explanation_dict_roundtrip():
    result, traj = _result_and_traj()
    exp = explain(result, traj)
    d = exp.to_dict()
    exp2 = Explanation.from_dict(d)
    assert exp2.headline == exp.headline
    assert len(exp2.trace) == len(exp.trace)
    assert exp2.trace[0].role == exp.trace[0].role


def test_result_explain_convenience_method():
    result, traj = _result_and_traj()
    exp = result.explain(traj)
    assert isinstance(exp, Explanation)


def test_explanation_without_trajectory_still_works():
    result, _ = _result_and_traj()
    exp = explain(result)  # no trajectory -> no action names, but still valid
    assert exp.headline
    assert len(exp.trace) == result.total_steps


def test_credit_mode_explanation():
    # A passing run analysed in credit mode explains which step secured success.
    def agent(ctx, task="t"):
        a = ctx.llm("s0", produce=lambda: "ok")
        b = ctx.tool("s1", produce=lambda: "safe" if ctx.rng.random() < 0.5 else "risky")
        return {"a": a, "b": b, "ok": b == "safe"}

    def verify(r):
        return 1.0 if r["ok"] else 0.0

    from agent_replay.recorder import record

    for seed in range(50):
        traj = record(agent, {}, session_id="credit", seed=seed, verifier=verify)
        if traj.outcome_score == 1.0:
            break
    result = attribute(traj, agent, verify, rollouts=60, on_success="credit")
    exp = explain(result, traj)
    assert exp.mode == "credit"
    assert "secured" in exp.headline.lower()
