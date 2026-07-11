"""Step-level faithfulness: the correct/wrong x faithful/unfaithful quadrants."""

from agent_replay.faithfulness import (
    CORRECT_FAITHFUL,
    CORRECT_UNFAITHFUL,
    WRONG_FAITHFUL,
    faithfulness,
)
from agent_replay.mock_agent import buggy_agent, verifier
from agent_replay.recorder import record


def _verify(r):
    return 1.0 if r["ok"] else 0.0


# --- wrong-faithful: the mock agent (fails, reasoning drives the failure) ----


def test_failed_run_is_wrong_faithful(recording):
    result = faithfulness(recording, buggy_agent, verifier, rollouts=60)
    assert not result.correct
    assert result.faithful  # masking the fatal step changes the outcome
    assert result.quadrant == WRONG_FAITHFUL
    assert "highest-" in (result.warning or "")


# --- correct-unfaithful: answer independent of reasoning (post-hoc) ----------


def posthoc_agent(ctx, task="t", n=3):
    for i in range(n):
        ctx.llm(f"reason_{i}", produce=lambda: "musing" if ctx.rng.random() < 0.5 else "pondering")
    return {"answer": "correct", "ok": True}  # answer ignores the reasoning


def test_posthoc_run_is_correct_unfaithful():
    traj = record(posthoc_agent, {}, session_id="ph", seed=0, verifier=_verify)
    result = faithfulness(traj, posthoc_agent, _verify, rollouts=40)
    assert result.correct
    assert not result.faithful  # masking reasoning never changes the (fixed) answer
    assert result.quadrant == CORRECT_UNFAITHFUL
    assert "post-hoc" in (result.warning or "")
    assert result.trajectory_faithfulness < 0.1


# --- correct-faithful: answer causally depends on a reasoning step -----------


def faithful_agent(ctx, task="t"):
    key = ctx.llm("derive", produce=lambda: "good" if ctx.rng.random() < 0.5 else "bad")
    return {"key": key, "ok": key == "good"}


def _record_correct(agent):
    for seed in range(100):
        traj = record(agent, {}, session_id=f"f{seed}", seed=seed, verifier=_verify)
        if traj.outcome_score == 1.0:
            return traj
    raise AssertionError("no correct recording")


def test_causal_run_is_correct_faithful():
    traj = _record_correct(faithful_agent)
    result = faithfulness(traj, faithful_agent, _verify, rollouts=80)
    assert result.correct
    assert result.faithful  # masking the derive step re-rolls the answer ~50%
    assert result.quadrant == CORRECT_FAITHFUL
    assert result.warning is None


def test_faithfulness_text_and_dict(recording):
    result = faithfulness(recording, buggy_agent, verifier, rollouts=40)
    text = result.to_text()
    assert "Faithfulness of" in text
    assert "Per-step" in text
    d = result.to_dict()
    assert d["quadrant"] == result.quadrant
    assert len(d["steps"]) == len(result.steps)


def test_only_reasoning_steps_scored_by_default(recording, fail_step):
    # The mock agent's fail step is a tool; default kinds=("llm",) skips it.
    result = faithfulness(recording, buggy_agent, verifier, rollouts=30)
    scored = {s.index for s in result.steps}
    assert fail_step not in scored  # tool step excluded from llm-only faithfulness
    # Including tools brings it in.
    result2 = faithfulness(recording, buggy_agent, verifier, rollouts=30, kinds=("llm", "tool"))
    assert fail_step in {s.index for s in result2.steps}
