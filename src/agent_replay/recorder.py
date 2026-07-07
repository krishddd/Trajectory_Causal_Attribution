"""Recording: capture an agent run as a replayable trajectory.

The public contract is a small ``AgentContext`` handed to the user's agent
function. The agent performs its work through ``ctx.llm(...)``, ``ctx.tool(...)``
and ``ctx.memory(...)``, passing a ``produce`` callable that *is* the stochastic
policy for that step. During recording the policy is executed and its output is
stored; during replay the very same agent code runs, but the context decides —
per step — whether to serve the recorded output (deterministic replay) or to
re-run the policy (a counterfactual resample).

Crucially, the agent function is written **once** and works unchanged in both
modes, which is what makes attribution possible: the ablation engine simply
re-invokes it under different replay plans.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from .types import Step, StepKind, Trajectory


class AgentContext:
    """Base class shared by the recording and replaying contexts.

    All agent randomness must flow through ``self.rng`` so that a run is fully
    determined by ``(recorded cassette, seed)`` — the deterministic-replay
    invariant from the architecture document.
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self._idx = 0

    # -- public operation API -------------------------------------------------

    def llm(self, name: str, produce: Optional[Callable[[], Any]] = None, **inputs: Any) -> Any:
        """Record/replay an LLM call named ``name``."""
        return self._op(StepKind.LLM, name, produce, inputs)

    def tool(self, name: str, produce: Optional[Callable[[], Any]] = None, **inputs: Any) -> Any:
        """Record/replay a tool call named ``name``."""
        return self._op(StepKind.TOOL, name, produce, inputs)

    def memory(self, name: str, produce: Optional[Callable[[], Any]] = None, **inputs: Any) -> Any:
        """Record/replay a memory operation named ``name``."""
        return self._op(StepKind.MEMORY, name, produce, inputs)

    # -- subclass hook --------------------------------------------------------

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError


class RecordContext(AgentContext):
    """Context that executes every policy and captures the factual trajectory."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.steps: List[Step] = []

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
    ) -> Any:
        idx = self._idx
        self._idx += 1
        output = produce() if produce is not None else None
        step = Step(index=idx, kind=kind, name=name, inputs=dict(inputs), output=output)
        parent = self.steps[-1].step_hash if self.steps else ""
        step.compute_hashes(parent)
        self.steps.append(step)
        return output


def record(
    agent_fn: Callable[..., Any],
    task: Optional[Dict[str, Any]] = None,
    *,
    session_id: str,
    seed: int = 0,
    verifier: Optional[Callable[[Any], float]] = None,
) -> Trajectory:
    """Run ``agent_fn`` once, capturing a :class:`Trajectory`.

    ``agent_fn`` is called as ``agent_fn(ctx, **task)``. If a ``verifier`` is
    supplied, the returned result is scored and stored on the trajectory so the
    factual outcome is known without re-running.
    """
    task = dict(task or {})
    ctx = RecordContext(seed)
    result = agent_fn(ctx, **task)
    traj = Trajectory(
        session_id=session_id,
        task=task,
        steps=ctx.steps,
        seed=seed,
        result=result,
    )
    if verifier is not None:
        traj.outcome_score = float(verifier(result))
    return traj
