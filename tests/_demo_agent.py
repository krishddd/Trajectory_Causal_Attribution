"""Test-only fixture agent with a known, injectable failure.

This module is **not** part of the shipped ``agent_replay`` package — it lives in
the test tree so the installed tool ships zero bundled agents. It is the
ground-truth fixture the suite attributes against: a deterministic-yet-stochastic
multi-step agent whose failure originates at exactly one step, so tests can assert
that counterfactual attribution/drift/faithfulness localise it.

Because it is importable as a plain top-level module during the test run (pytest
puts ``tests/`` on ``sys.path``), it also backs the CLI tests via the
``_demo_agent:buggy_agent`` entrypoint string.

Failure model
-------------
The agent walks ``n_steps`` reasoning/tool steps. Every step is a benign "OK"
except the ``fail_step``, whose policy draws "BAD" with probability
``fail_prob``. The recording seed is chosen so the factual run draws "BAD" at
``fail_step`` (a genuine failure). Once "BAD" is produced the run is doomed:
downstream steps cannot undo it. The verifier fails the run iff any step emitted
"BAD".
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_FAIL_STEP = 3
DEFAULT_N_STEPS = 6
DEFAULT_FAIL_PROB = 0.7
# Seed under which the factual recording actually draws "BAD" at the fail step
# (random.Random(1).random() == 0.134 < DEFAULT_FAIL_PROB).
FACTUAL_SEED = 1


def buggy_agent(
    ctx: Any,
    task: str = "demo-task",
    n_steps: int = DEFAULT_N_STEPS,
    fail_step: int = DEFAULT_FAIL_STEP,
    fail_prob: float = DEFAULT_FAIL_PROB,
) -> Dict[str, Any]:
    """A multi-step agent that fails at ``fail_step`` with probability ``fail_prob``.

    Written once, run under both the recorder and the replayer unchanged.
    """
    state: Dict[str, Any] = {"task": task, "trace": [], "ok": True}

    for i in range(n_steps):

        def produce(step=i):
            # All randomness flows through ctx.rng to stay deterministic-by-seed.
            if step == fail_step:
                return "BAD" if ctx.rng.random() < fail_prob else "OK"
            return "OK"

        if i == fail_step:
            action = ctx.tool(f"tool_step_{i}", produce=produce, context=state["trace"][-3:])
        else:
            action = ctx.llm(f"reason_step_{i}", produce=produce, context=state["trace"][-3:])

        state["trace"].append(action)
        if action == "BAD":
            state["ok"] = False

    state["answer"] = "correct" if state["ok"] else "wrong"
    return state


def verifier(result: Dict[str, Any]) -> float:
    """Outcome score: 1.0 on success, 0.0 on failure (paper convention)."""
    return 1.0 if result.get("ok", False) else 0.0


def health_scorer(step: Any) -> float:
    """Intermediate-state scorer used by the drift-curve tests.

    Reports high alignment health for a benign "OK" step and low health once the
    agent commits a "BAD" action — the kind of per-step health signal the outcome
    verifier cannot see, letting ``agent_replay.drift`` chart the decay.
    """
    return 0.2 if getattr(step, "output", None) == "BAD" else 0.95


def make_recording(session_id: str = "mock-demo", seed: int = FACTUAL_SEED):
    """Record one factual, *failing* run of the fixture agent.

    Falls back to searching for a failing seed so the recording is guaranteed to
    be an actual failure even if the defaults are changed.
    """
    from agent_replay.recorder import record

    traj = record(
        buggy_agent, {"task": "demo-task"}, session_id=session_id, seed=seed, verifier=verifier
    )
    if traj.outcome_score and traj.outcome_score >= 0.5:
        for candidate in range(1, 200):
            traj = record(
                buggy_agent,
                {"task": "demo-task"},
                session_id=session_id,
                seed=candidate,
                verifier=verifier,
            )
            if traj.outcome_score is not None and traj.outcome_score < 0.5:
                break
    return traj
