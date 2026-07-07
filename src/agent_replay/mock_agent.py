"""A reference mock agent with a known, injectable failure.

This is the ground-truth fixture for the whole library: a deterministic-yet-
stochastic multi-step agent whose failure originates at exactly one step. Because
we *know* the culprit, the test-suite can assert that counterfactual attribution
localises it, and the CLI/demo have something meaningful to attribute without a
real LLM or API key.

Failure model
-------------
The agent walks ``n_steps`` reasoning/tool steps. Every step is a benign "OK"
except the ``fail_step``, whose policy draws "BAD" with probability
``fail_prob``. The recording seed is chosen so the factual run draws "BAD" at
``fail_step`` (a genuine failure). Once "BAD" is produced the run is doomed:
downstream steps cannot undo it. The verifier fails the run iff any step emitted
"BAD".

Why this yields a clean point-of-commitment: resampling step ``i <= fail_step``
re-rolls the fatal step and can rescue the run, so its attribution interval
excludes zero; resampling ``i > fail_step`` leaves the already-"BAD" fatal step
untouched, so the failure is locked in and attribution collapses to zero. The
*latest* significant step is therefore exactly ``fail_step``.
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


def make_recording(session_id: str = "mock-demo", seed: int = FACTUAL_SEED):
    """Record one factual, *failing* run of the mock agent.

    Falls back to searching for a failing seed so the recording is guaranteed to
    be an actual failure even if the defaults are changed.
    """
    from .recorder import record

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
