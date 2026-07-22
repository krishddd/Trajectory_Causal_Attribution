"""v0.2 soundness features: success guard, credit mode, strict serialisation,
and the non-resamplable step flag."""

import pytest

from agent_replay.attribution import attribute
from agent_replay.errors import NonSerializableStepError, SuccessfulRunError
from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay


def _ok(result):
    return 1.0 if result["ok"] else 0.0


def passing_agent(ctx, task="t"):
    # Passes factually (b == "safe"), but re-deciding b risks failure.
    a = ctx.llm("s0", produce=lambda: "ok")
    b = ctx.tool("s1", produce=lambda: "safe" if ctx.rng.random() < 0.5 else "risky")
    return {"a": a, "b": b, "ok": b == "safe"}


def _record_passing():
    for seed in range(50):
        traj = record(passing_agent, {}, session_id="p", seed=seed, verifier=_ok)
        if traj.outcome_score == 1.0:
            return traj
    raise AssertionError("no passing seed found")


# --- passing-run guard (§2.2) ----------------------------------------------


def test_attribute_passing_run_raises_by_default():
    traj = _record_passing()
    assert traj.outcome_score == 1.0
    with pytest.raises(SuccessfulRunError):
        attribute(traj, passing_agent, _ok, rollouts=10)


def test_attribute_passing_run_credit_mode():
    traj = _record_passing()
    result = attribute(traj, passing_agent, _ok, rollouts=60, on_success="credit")
    assert result.mode == "credit"
    assert not result.failed
    # Credit scores are non-negative (failure risk introduced by re-decision).
    for s in result.steps:
        assert s.attribution >= -1e-9
    # Step 1 (b) is the risky decision that secured success.
    assert result.steps[1].attribution > 0.1
    assert result.culprit_index == 1


def test_attribute_invalid_on_success():
    traj = _record_passing()
    with pytest.raises(ValueError):
        attribute(traj, passing_agent, _ok, rollouts=10, on_success="bogus")


def test_credit_mode_shapley_sign_consistent():
    """Credit mode must flip the Shapley spine, not only the contrastive scores.

    Regression for the sign bug: with method='both'/'shapley' on a passing run the
    Shapley values were left failure-signed, so they disagreed with the credit-
    signed contrastive scores and _select_culprit picked the inverse "save point".
    """
    traj = _record_passing()

    both = attribute(traj, passing_agent, _ok, rollouts=60, method="both", on_success="credit")
    assert both.mode == "credit"
    # The risky decision (step 1) secured success: positive credit in BOTH signals,
    # and its Shapley value must not point the opposite way to its contrastive score.
    s1 = both.steps[1]
    assert s1.attribution > 0.1
    assert s1.shapley is not None and s1.shapley > 0.0
    # No step may show a contrastive credit and a Shapley "blame" of opposite sign
    # beyond noise — the two signals now share a sign convention.
    for s in both.steps:
        if s.shapley is not None and abs(s.attribution) > 0.15 and abs(s.shapley) > 0.15:
            assert (s.attribution > 0) == (s.shapley > 0)

    shap = attribute(traj, passing_agent, _ok, rollouts=60, method="shapley", on_success="credit")
    assert shap.mode == "credit"
    # The save point is the step that most secured success — the largest positive
    # credit — not the inverse the un-negated max used to pick.
    assert shap.culprit_index == 1
    assert shap.steps[1].attribution > 0.0


# --- strict serialisation (§2.4) -------------------------------------------


def test_non_serializable_output_raises():
    def bad_agent(ctx, task="t"):
        return ctx.llm("s0", produce=lambda: {1, 2, 3})  # a set is not JSON

    with pytest.raises(NonSerializableStepError):
        record(bad_agent, {}, session_id="bad", seed=0)


def test_non_serializable_can_be_disabled():
    def bad_agent(ctx, task="t"):
        ctx.llm("s0", produce=lambda: object())
        return {"ok": True}

    # Lax mode records best-effort without raising.
    traj = record(bad_agent, {}, session_id="lax", seed=0, strict_serialization=False)
    assert len(traj) == 1


# --- non-resamplable steps (§2.3) ------------------------------------------


def test_produce_none_is_non_resamplable():
    def obs_agent(ctx, task="t"):
        ctx.llm("observed", record_only=True)  # no produce policy
        return {"ok": True}

    traj = record(obs_agent, {}, session_id="obs", seed=0)
    assert traj.steps[0].resamplable is False


def test_non_resamplable_step_served_on_resample():
    captured = {"n": 0}

    def agent(ctx, task="t"):
        def produce():
            captured["n"] += 1
            return "live"

        # Marked non-resamplable: replay must serve recorded output, not call produce.
        val = ctx.llm("obs", produce=produce, resamplable=False)
        return {"val": val}

    traj = record(agent, {}, session_id="nr", seed=0)
    recorded_calls = captured["n"]
    # Resample everything: a normal step would re-run produce; this one must not.
    replay(agent, traj, ReplayPlan(held=set()), seed=99)
    assert captured["n"] == recorded_calls  # produce not called again on replay


def test_explicit_resamplable_flag_recorded():
    def agent(ctx, task="t"):
        ctx.llm("a", produce=lambda: "x", resamplable=True)
        ctx.llm("b", produce=lambda: "y", resamplable=False)
        return {"ok": True}

    traj = record(agent, {}, session_id="flags", seed=0)
    assert traj.steps[0].resamplable is True
    assert traj.steps[1].resamplable is False


def test_resamplable_survives_store_roundtrip(tmp_path):
    from agent_replay.store import CheckpointStore

    def agent(ctx, task="t"):
        ctx.llm("a", produce=lambda: "x")
        ctx.llm("b", produce=lambda: "y", resamplable=False)
        return {"ok": True}

    traj = record(agent, {}, session_id="rt", seed=0)
    db = str(tmp_path / "s.sqlite")
    with CheckpointStore(db) as store:
        store.save_trajectory(traj)
        loaded = store.load_trajectory("rt")
    assert loaded.steps[0].resamplable is True
    assert loaded.steps[1].resamplable is False


def test_credit_mode_html_report():
    traj = _record_passing()
    result = attribute(traj, passing_agent, _ok, rollouts=60, on_success="credit")
    from agent_replay.report import render_html

    html = render_html(result)
    assert "secured" in html.lower()
    assert "PASSED" in html
    assert "mode <code>credit" in html
