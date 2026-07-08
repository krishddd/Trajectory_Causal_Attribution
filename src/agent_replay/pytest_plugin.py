"""Pytest integration: assert agents pass, and explain *which step* broke when they don't.

Agents are stochastic, so a single green run is not a pass. This plugin provides a
**flakiness-aware** assertion — run the agent many times, measure the empirical
failure rate, and fail the test if it exceeds a budget. When it does, the plugin
automatically runs counterfactual attribution on a failing trajectory and puts the
plain-language explanation (which step, why, how to fix) straight into the test
failure message, optionally writing the full HTML report as a CI artifact.

Usage (no plugin registration needed — just import the helper)::

    from agent_replay.pytest_plugin import assert_agent_passes

    def test_my_agent():
        assert_agent_passes(my_agent, {"q": "..."}, my_verifier,
                            rollouts=30, p_fail_max=0.05)

Or use the ``agent_replay_session`` fixture for a temp-file checkpoint store that
persists the recorded trajectories as test artifacts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .attribution import attribute
from .recorder import record
from .stats import wilson_interval
from .types import Trajectory


class AgentFlakyError(AssertionError):
    """Raised by :func:`assert_agent_passes` when the failure budget is exceeded.

    Carries the structured results so a test harness can inspect them
    programmatically in addition to reading the formatted message.
    """

    def __init__(
        self,
        message: str,
        *,
        p_fail: float,
        failures: List[Trajectory],
        trajectories: List[Trajectory],
        attribution: Any = None,
        explanation: Any = None,
    ) -> None:
        super().__init__(message)
        self.p_fail = p_fail
        self.failures = failures
        self.trajectories = trajectories
        self.attribution = attribution
        self.explanation = explanation


def measure_failure_rate(
    agent_fn: Callable[..., Any],
    task: Optional[Dict[str, Any]],
    verifier: Callable[[Any], float],
    *,
    rollouts: int = 30,
    fail_threshold: float = 0.5,
    base_seed: int = 0,
    pass_context: bool = True,
    strict_serialization: bool = True,
) -> Dict[str, Any]:
    """Run ``agent_fn`` ``rollouts`` times and measure its empirical failure rate.

    Returns a dict with ``p_fail`` (point estimate), ``ci`` (Wilson 95% interval),
    ``failures`` (the failing trajectories) and ``trajectories`` (all of them).
    Each rollout uses a distinct seed, so this samples the agent's intrinsic
    flakiness rather than replaying one recording.
    """
    failures: List[Trajectory] = []
    trajectories: List[Trajectory] = []
    for k in range(rollouts):
        traj = record(
            agent_fn,
            task,
            session_id=f"test-{base_seed + k}",
            seed=base_seed + k,
            verifier=verifier,
            pass_context=pass_context,
            strict_serialization=strict_serialization,
        )
        trajectories.append(traj)
        score = traj.outcome_score
        if score is not None and score < fail_threshold:
            failures.append(traj)
    n_fail = len(failures)
    point, low, high = wilson_interval(n_fail, rollouts)
    return {
        "p_fail": point,
        "ci": (low, high),
        "n_fail": n_fail,
        "rollouts": rollouts,
        "failures": failures,
        "trajectories": trajectories,
    }


def assert_agent_passes(
    agent_fn: Callable[..., Any],
    task: Optional[Dict[str, Any]],
    verifier: Callable[[Any], float],
    *,
    rollouts: int = 30,
    p_fail_max: float = 0.05,
    fail_threshold: float = 0.5,
    base_seed: int = 0,
    pass_context: bool = True,
    strict_serialization: bool = True,
    attribute_on_failure: bool = True,
    attribution_rollouts: int = 60,
    method: str = "both",
    repair: bool = True,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Assert an agent's failure rate stays within budget; explain the culprit if not.

    Runs ``agent_fn`` ``rollouts`` times. If the observed failure rate exceeds
    ``p_fail_max``, raises :class:`AgentFlakyError` whose message includes a
    counterfactual attribution of one failing run — *which step* caused it, why,
    and the minimal fix — and optionally writes the HTML report to ``report_path``.

    Returns the measurement dict on success so callers can log the (passing) rate.
    """
    m = measure_failure_rate(
        agent_fn,
        task,
        verifier,
        rollouts=rollouts,
        fail_threshold=fail_threshold,
        base_seed=base_seed,
        pass_context=pass_context,
        strict_serialization=strict_serialization,
    )
    if m["p_fail"] <= p_fail_max:
        return m

    lo, hi = m["ci"]
    lines = [
        f"Agent exceeded its failure budget: {m['n_fail']}/{m['rollouts']} rollouts "
        f"failed (p_fail={m['p_fail']:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]) "
        f"> allowed {p_fail_max:.3f}.",
    ]

    result = explanation = None
    if attribute_on_failure and m["failures"]:
        failing = m["failures"][0]
        result = attribute(
            failing,
            agent_fn,
            verifier,
            rollouts=attribution_rollouts,
            method=method,
            fail_threshold=fail_threshold,
            repair=repair,
            pass_context=pass_context,
        )
        explanation = result.explain(failing)
        lines.append("")
        lines.append(explanation.to_text())
        if report_path:
            result.to_html(report_path, explanation=explanation)
            lines.append("")
            lines.append(f"Full report written to {report_path}")

    raise AgentFlakyError(
        "\n".join(lines),
        p_fail=m["p_fail"],
        failures=m["failures"],
        trajectories=m["trajectories"],
        attribution=result,
        explanation=explanation,
    )


# --- pytest fixtures (registered via the pytest11 entry point) --------------

try:  # pragma: no cover - only when pytest is installed
    import pytest

    @pytest.fixture
    def agent_replay_session(tmp_path):
        """A temp-file-backed :class:`~agent_replay.session.Session` for a test.

        The SQLite store lives under pytest's ``tmp_path`` so recorded
        trajectories survive as inspectable test artifacts.
        """
        from .session import Session

        sess = Session(str(tmp_path / "agent_replay.sqlite"))
        try:
            yield sess
        finally:
            sess.close()

    @pytest.fixture
    def assert_agent():
        """Fixture returning :func:`assert_agent_passes` for ergonomic use in tests."""
        return assert_agent_passes

except ImportError:  # pragma: no cover
    pass
