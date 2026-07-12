"""Per-step drift / entropy-of-autonomy curve (Multiverse deck, slide 2, Gap 6)."""

import math

from _demo_agent import buggy_agent, verifier
from agent_replay.drift import DriftResult, drift
from agent_replay.recorder import record
from agent_replay.stats import binary_entropy

# --- the entropy helper ------------------------------------------------------


def test_binary_entropy_bounds():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == 1.0
    assert math.isclose(binary_entropy(0.5), 1.0)
    # symmetric and interior values sit strictly between.
    assert math.isclose(binary_entropy(0.1), binary_entropy(0.9))
    assert 0.0 < binary_entropy(0.2) < 1.0


# --- entropy curve from the verifier alone -----------------------------------


def test_entropy_curve_available_without_state_scorer(recording):
    result = drift(recording, buggy_agent, verifier, rollouts=30)
    assert isinstance(result, DriftResult)
    assert len(result.points) == len(recording.steps)
    assert not result.health_available
    for p in result.points:
        assert 0.0 <= p.entropy <= 1.0
        assert 0.0 <= p.p_success <= 1.0
        assert p.health is None
        assert p.drift == 0.0


def test_fate_commits_around_the_failing_step(recording, fail_step):
    # The mock run is doomed once the fail step draws BAD: resampling *after* it
    # cannot rescue the run, so P(success) is 0 and entropy collapses there.
    result = drift(recording, buggy_agent, verifier, rollouts=40)
    # Steps strictly after the fail step are locked in (zero recoverability).
    for p in result.points:
        if p.index > fail_step:
            assert p.p_success == 0.0
            assert p.entropy == 0.0
    # The commitment index is the last still-open step; it does not run past the
    # locked-in tail.
    assert result.commitment_index is not None
    assert result.commitment_index <= fail_step


# --- alignment health overlay + silent-decay detection -----------------------


def test_state_scorer_adds_health_and_drift(recording):
    # A scorer that reports declining health as soon as a BAD action appears.
    def scorer(step):
        return 0.2 if step.output == "BAD" else 0.9

    result = drift(recording, buggy_agent, verifier, state_scorer=scorer, rollouts=20)
    assert result.health_available
    assert all(p.health is not None for p in result.points)
    # Health drops when BAD appears -> positive total drift, decay flagged.
    assert result.total_drift > 0.0
    assert result.decayed
    assert result.drift_onset_index is not None
    assert result.warning is not None


def test_stable_health_is_not_flagged():
    def flat_agent(ctx, task="t", n=4):
        for i in range(n):
            ctx.llm(f"reason_{i}", produce=lambda: "ok")
        return {"ok": True}

    traj = record(flat_agent, {}, session_id="flat", seed=0, verifier=lambda r: 1.0)
    result = drift(traj, flat_agent, lambda r: 1.0, state_scorer=lambda s: 0.9, rollouts=10)
    assert result.health_available
    assert result.total_drift == 0.0
    assert not result.decayed
    assert result.warning is None


# --- reporting surfaces ------------------------------------------------------


def test_to_text_and_to_dict(recording):
    result = drift(recording, buggy_agent, verifier, state_scorer=lambda s: 0.5, rollouts=15)
    text = result.to_text()
    assert "Drift curve for" in text
    assert "entropy of autonomy" in text
    d = result.to_dict()
    assert d["session_id"] == recording.session_id
    assert len(d["points"]) == len(result.points)
    assert d["health_available"] is True


def test_to_html_writes_svg(recording, tmp_path):
    result = drift(recording, buggy_agent, verifier, state_scorer=lambda s: 0.5, rollouts=10)
    out = tmp_path / "drift.html"
    result.to_html(str(out))
    body = out.read_text(encoding="utf-8")
    assert "<svg" in body
    assert "polyline" in body
    assert "Alignment health" in body  # health series present when scorer supplied


def test_determinism(recording):
    a = drift(recording, buggy_agent, verifier, rollouts=20)
    b = drift(recording, buggy_agent, verifier, rollouts=20)
    assert [p.p_success for p in a.points] == [p.p_success for p in b.points]
    assert a.entropy_auc == b.entropy_auc
