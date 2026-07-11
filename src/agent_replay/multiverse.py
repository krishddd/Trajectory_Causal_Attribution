"""The Multiverse: fork a recorded run into counterfactual branches.

Attribution already forks reality *transiently* for every ablation rollout. This
module makes forking first-class and **persisted**, matching the "AgentOps
Multiverse" the research describes: rewind to step ``i``, change one decision, and
run a new timeline forward — sharing the entire prefix (and, through the
content-addressable store, its storage) with the parent.

A fork records a *complete new trajectory*: the held prefix (served verbatim from
the parent cassette, so identical), the intervened step, and the live continuation
from that point on. The child's ``meta`` records ``parent_session`` / ``fork_step``
/ ``intervention`` so branches can be listed and diffed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ._ambient import bind_context, unbind_context
from .recorder import AgentContext, AsyncAgentContext, _check_serializable, _is_async
from .replayer import REMOVED, ReplayContext, ReplayPlan
from .types import Step, StepKind, Trajectory


def _coerce(output: Any) -> Any:
    """Represent a removed step's empty action as JSON-friendly ``None`` in the child."""
    return None if output is REMOVED else output


class _Unset:
    """Sentinel distinguishing "no do() intervention" from ``do=None``."""


UNSET = _Unset()


def _record_step(
    ctx: Any, kind: StepKind, name: str, inputs: Dict[str, Any], output: Any, resamplable: bool
) -> None:
    """Append a fully-formed step to a fork context's growing trajectory."""
    if ctx.strict_serialization:
        _check_serializable(kind, name, "inputs", inputs)
        _check_serializable(kind, name, "output", output)
    step = Step(
        index=len(ctx.steps),
        kind=kind,
        name=name,
        inputs=dict(inputs),
        output=output,
        resamplable=resamplable,
    )
    parent = ctx.steps[-1].step_hash if ctx.steps else ""
    step.compute_hashes(parent)
    ctx.steps.append(step)


class ForkContext(AgentContext):
    """Records a new trajectory while serving the parent cassette for held steps."""

    def __init__(
        self, parent: Trajectory, plan: ReplayPlan, seed: int, *, strict_serialization: bool = True
    ) -> None:
        super().__init__(seed)
        self._matcher = ReplayContext(parent, plan, seed)
        self.steps: List[Step] = []
        self.strict_serialization = strict_serialization

    def _op(self, kind, name, produce, inputs, resamplable=None):
        needs_produce, value = self._matcher._decide(kind, name, inputs)
        output = _coerce((produce() if produce is not None else None) if needs_produce else value)
        rs = resamplable if resamplable is not None else (produce is not None)
        _record_step(self, kind, name, inputs, output, rs)
        return output


class AsyncForkContext(AsyncAgentContext):
    """Async counterpart of :class:`ForkContext`."""

    def __init__(
        self, parent: Trajectory, plan: ReplayPlan, seed: int, *, strict_serialization: bool = True
    ) -> None:
        super().__init__(seed)
        self._matcher = ReplayContext(parent, plan, seed)
        self.steps: List[Step] = []
        self.strict_serialization = strict_serialization

    async def _aop(self, kind, name, produce, inputs, resamplable=None):
        needs_produce, value = self._matcher._decide(kind, name, inputs)
        output = _coerce(
            (await produce() if produce is not None else None) if needs_produce else value
        )
        rs = resamplable if resamplable is not None else (produce is not None)
        _record_step(self, kind, name, inputs, output, rs)
        return output

    def _value_op(self, name, produce):
        needs_produce, value = self._matcher._decide(StepKind.MEMORY, name, {})
        output = produce() if needs_produce else value
        _record_step(self, StepKind.MEMORY, name, {}, output, False)
        return output


def _plan(at_step: int, do: Any, remove: bool) -> ReplayPlan:
    held = set(range(at_step))
    forced = {at_step: do} if do is not UNSET else {}
    removed = {at_step} if remove else set()
    return ReplayPlan(held=held, forced=forced, removed=removed)


def _intervention_label(do: Any, remove: bool) -> str:
    if remove:
        return "remove"
    if do is not UNSET:
        return "do"
    return "resample"


def _finish(
    parent: Trajectory,
    ctx: Any,
    result: Any,
    at_step: int,
    do: Any,
    remove: bool,
    session_id: Optional[str],
    seed: int,
    verifier: Optional[Callable[[Any], float]],
) -> Trajectory:
    label = _intervention_label(do, remove)
    sid = session_id or f"{parent.session_id}::fork@{at_step}:{label}"
    child = Trajectory(
        session_id=sid,
        task=dict(parent.task),
        steps=ctx.steps,
        seed=seed,
        result=result,
        meta={"parent_session": parent.session_id, "fork_step": at_step, "intervention": label},
    )
    if verifier is not None:
        child.outcome_score = float(verifier(result))
    return child


def fork(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    at_step: int,
    *,
    do: Any = UNSET,
    remove: bool = False,
    seed: int = 0,
    session_id: Optional[str] = None,
    verifier: Optional[Callable[[Any], float]] = None,
    pass_context: bool = True,
    strict_serialization: bool = True,
) -> Trajectory:
    """Fork ``trajectory`` at ``at_step`` into a new counterfactual trajectory.

    Steps ``< at_step`` are held at their recorded actions; the step at ``at_step``
    is forced to ``do`` (a :func:`do`-intervention), dropped (``remove=True``), or
    resampled (default); everything after runs live. Returns a complete child
    :class:`~agent_replay.types.Trajectory` whose ``meta`` links back to the parent.
    """
    if _is_async(agent_fn):
        import asyncio

        return asyncio.run(
            afork(
                agent_fn,
                trajectory,
                at_step,
                do=do,
                remove=remove,
                seed=seed,
                session_id=session_id,
                verifier=verifier,
                pass_context=pass_context,
                strict_serialization=strict_serialization,
            )
        )
    ctx = ForkContext(
        trajectory, _plan(at_step, do, remove), seed, strict_serialization=strict_serialization
    )
    token = bind_context(ctx)
    try:
        result = agent_fn(ctx, **trajectory.task) if pass_context else agent_fn(**trajectory.task)
    finally:
        unbind_context(token)
    return _finish(trajectory, ctx, result, at_step, do, remove, session_id, seed, verifier)


async def afork(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    at_step: int,
    *,
    do: Any = UNSET,
    remove: bool = False,
    seed: int = 0,
    session_id: Optional[str] = None,
    verifier: Optional[Callable[[Any], float]] = None,
    pass_context: bool = True,
    strict_serialization: bool = True,
) -> Trajectory:
    """Async :func:`fork`."""
    ctx = AsyncForkContext(
        trajectory, _plan(at_step, do, remove), seed, strict_serialization=strict_serialization
    )
    token = bind_context(ctx)
    try:
        if pass_context:
            result = await agent_fn(ctx, **trajectory.task)
        else:
            result = await agent_fn(**trajectory.task)
    finally:
        unbind_context(token)
    return _finish(trajectory, ctx, result, at_step, do, remove, session_id, seed, verifier)


def resume(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    *,
    seed: int = 0,
    session_id: Optional[str] = None,
    verifier: Optional[Callable[[Any], float]] = None,
    pass_context: bool = True,
) -> Trajectory:
    """Replay the whole recorded prefix, then continue the run live.

    Durable recovery (Multiverse deck, slide 5): fast-forward through the recorded
    history and let the agent proceed live for any steps beyond the recorded
    horizon — useful to continue a crashed/truncated run without repeating side
    effects for the steps already on the cassette.
    """
    sid = session_id or f"{trajectory.session_id}::resume"
    return fork(
        agent_fn,
        trajectory,
        len(trajectory),
        seed=seed,
        session_id=sid,
        verifier=verifier,
        pass_context=pass_context,
    )


def diff(a: Trajectory, b: Trajectory) -> Dict[str, Any]:
    """Compare two trajectories step-by-step; return the first divergence + per-step diff.

    Two steps match when they are the same operation (idempotency key) *and*
    produced the same output — so ``diff`` pinpoints exactly where a fork's
    timeline departs from its parent (the "State Diff" in the research console).
    """
    steps: List[Dict[str, Any]] = []
    first_divergence: Optional[int] = None
    for i in range(max(len(a), len(b))):
        sa = a.steps[i] if i < len(a) else None
        sb = b.steps[i] if i < len(b) else None
        same = (
            sa is not None
            and sb is not None
            and sa.op_key() == sb.op_key()
            and sa.output == sb.output
        )
        if not same and first_divergence is None:
            first_divergence = i
        steps.append(
            {
                "index": i,
                "same": bool(same),
                "a": {"name": sa.name, "output": sa.output} if sa else None,
                "b": {"name": sb.name, "output": sb.output} if sb else None,
            }
        )
    return {
        "a": a.session_id,
        "b": b.session_id,
        "first_divergence": first_divergence,
        "n_diff": sum(1 for s in steps if not s["same"]),
        "steps": steps,
    }
