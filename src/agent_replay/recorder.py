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

import json
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ._ambient import bind_context, unbind_context
from .errors import NonSerializableStepError
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

    def llm(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        """Record/replay an LLM call named ``name``."""
        return self._op(StepKind.LLM, name, produce, inputs, resamplable, observe)

    def tool(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        """Record/replay a tool call named ``name``.

        Pass ``observe`` to record a distinct **action** and **observation**: the
        agent's chosen call is ``produce()`` (the action), and ``observe(action)``
        is what the environment returned (the observation, served downstream). With
        no ``observe`` the two coincide, exactly as before.
        """
        return self._op(StepKind.TOOL, name, produce, inputs, resamplable, observe)

    def memory(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        """Record/replay a memory operation named ``name``."""
        return self._op(StepKind.MEMORY, name, produce, inputs, resamplable, observe)

    # -- deterministic time & entropy -----------------------------------------

    def now(self) -> float:
        """Deterministic wall clock: record the real time; replay the recorded time.

        Route ``time.time()``-style reads through this (or auto-instrument them,
        see :func:`agent_replay.instrument.enable_virtual_time`) so a replayed run
        sees the same timestamps it recorded — otherwise resampled paths drift
        from wall-clock and any recorded output embedding a timestamp would change
        its idempotency key across runs.
        """
        return self._op(StepKind.MEMORY, "__now__", _real_now, {}, False)

    def uuid(self) -> str:
        """Deterministic uuid4 (string): record a real uuid; replay the recorded one."""
        return self._op(StepKind.MEMORY, "__uuid__", _real_uuid, {}, False)

    # -- subclass hook --------------------------------------------------------

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        raise NotImplementedError


# Bound at import (before any virtual-time patching) so ctx.now()/ctx.uuid()
# always read the true stdlib clock/entropy, even while instrument has patched
# ``time.time`` / ``uuid.uuid4`` to route through the context (avoids recursion).
_TRUE_TIME = time.time
_TRUE_UUID = uuid.uuid4


def _real_now() -> float:
    return _TRUE_TIME()


def _real_uuid() -> str:
    return str(_TRUE_UUID())


def _check_serializable(kind: StepKind, name: str, label: str, value: Any) -> None:
    """Raise :class:`NonSerializableStepError` if ``value`` is not JSON-native."""
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise NonSerializableStepError(
            f"{label} of step '{kind.value}:{name}' is not JSON-serialisable "
            f"({type(value).__name__}): {exc}. Recorded payloads must round-trip "
            f"as JSON; convert it to a plain dict/list/str, or pass "
            f"strict_serialization=False to record it best-effort (lossy)."
        ) from exc


class RecordContext(AgentContext):
    """Context that executes every policy and captures the factual trajectory."""

    def __init__(self, seed: int, *, strict_serialization: bool = True) -> None:
        super().__init__(seed)
        self.steps: List[Step] = []
        self.strict_serialization = strict_serialization

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        idx = self._idx
        self._idx += 1
        action = produce() if produce is not None else None
        # With an ``observe`` policy the action and observation differ: record the
        # action separately and serve the observation downstream. Otherwise they
        # coincide and ``action`` stays None ("the action is the output").
        if observe is not None:
            output = observe(action)
            recorded_action = action
        else:
            output = action
            recorded_action = None
        # A step with no policy cannot be re-drawn counterfactually. Default its
        # resamplability from whether a policy was supplied, unless overridden.
        if resamplable is None:
            resamplable = produce is not None
        if self.strict_serialization:
            _check_serializable(kind, name, "inputs", inputs)
            _check_serializable(kind, name, "output", output)
            if recorded_action is not None:
                _check_serializable(kind, name, "action", recorded_action)
        step = Step(
            index=idx,
            kind=kind,
            name=name,
            inputs=dict(inputs),
            output=output,
            resamplable=resamplable,
            action=recorded_action,
        )
        parent = self.steps[-1].step_hash if self.steps else ""
        step.compute_hashes(parent)
        self.steps.append(step)
        return output


class AsyncAgentContext:
    """Async counterpart of :class:`AgentContext` for ``async def`` agents.

    Exposes awaitable ``llm`` / ``tool`` / ``memory`` whose ``produce`` policy is
    an async callable (``await produce()``). The matching, hashing and plan
    semantics are identical to the sync path.
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self._idx = 0

    async def llm(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        return await self._aop(StepKind.LLM, name, produce, inputs, resamplable, observe)

    async def tool(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        return await self._aop(StepKind.TOOL, name, produce, inputs, resamplable, observe)

    async def memory(
        self,
        name: str,
        produce: Optional[Callable[[], Any]] = None,
        *,
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
        **inputs: Any,
    ) -> Any:
        return await self._aop(StepKind.MEMORY, name, produce, inputs, resamplable, observe)

    async def _aop(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        raise NotImplementedError

    # Deterministic time/entropy are instantaneous, so they stay synchronous even
    # on the async context (no ``await`` needed).
    def now(self) -> float:
        """Deterministic wall clock (see :meth:`AgentContext.now`)."""
        return self._value_op("__now__", _real_now)

    def uuid(self) -> str:
        """Deterministic uuid4 string (see :meth:`AgentContext.uuid`)."""
        return self._value_op("__uuid__", _real_uuid)

    def _value_op(self, name: str, produce: Callable[[], Any]) -> Any:
        raise NotImplementedError


class AsyncRecordContext(AsyncAgentContext):
    """Async recording context: awaits each policy and captures the trajectory."""

    def __init__(self, seed: int, *, strict_serialization: bool = True) -> None:
        super().__init__(seed)
        self.steps: List[Step] = []
        self.strict_serialization = strict_serialization

    async def _aop(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
        observe: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        idx = self._idx
        self._idx += 1
        action = await produce() if produce is not None else None
        if observe is not None:
            observed = observe(action)
            output = await observed if _is_awaitable(observed) else observed
            recorded_action = action
        else:
            output = action
            recorded_action = None
        if resamplable is None:
            resamplable = produce is not None
        if self.strict_serialization:
            _check_serializable(kind, name, "inputs", inputs)
            _check_serializable(kind, name, "output", output)
            if recorded_action is not None:
                _check_serializable(kind, name, "action", recorded_action)
        step = Step(
            index=idx,
            kind=kind,
            name=name,
            inputs=dict(inputs),
            output=output,
            resamplable=resamplable,
            action=recorded_action,
        )
        parent = self.steps[-1].step_hash if self.steps else ""
        step.compute_hashes(parent)
        self.steps.append(step)
        return output

    def _value_op(self, name: str, produce: Callable[[], Any]) -> Any:
        idx = self._idx
        self._idx += 1
        output = produce()
        if self.strict_serialization:
            _check_serializable(StepKind.MEMORY, name, "output", output)
        step = Step(
            index=idx, kind=StepKind.MEMORY, name=name, inputs={}, output=output, resamplable=False
        )
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
    strict_serialization: bool = True,
    pass_context: bool = True,
) -> Trajectory:
    """Run ``agent_fn`` once, capturing a :class:`Trajectory`.

    By default ``agent_fn`` is called as ``agent_fn(ctx, **task)`` (the explicit
    context style). With ``pass_context=False`` it is called as ``agent_fn(**task)``
    and steps are captured via the *ambient* context instead — the mode used by
    :mod:`agent_replay.instrument` to record framework agents that own their own
    call signature. The ambient context is published for the duration of the run
    in either case, so auto-instrumented callables work in both styles.

    ``async def`` agents are detected and run to completion via :func:`arecord`,
    so async agents flow through the whole (synchronous) attribution pipeline
    unchanged.

    If a ``verifier`` is supplied, the returned result is scored and stored on the
    trajectory so the factual outcome is known without re-running.

    With ``strict_serialization`` (default), any step input/output that is not
    JSON-serialisable raises :class:`~agent_replay.errors.NonSerializableStepError`
    at record time, rather than silently degrading to a string on store.
    """
    if _is_async(agent_fn):
        import asyncio

        return asyncio.run(
            arecord(
                agent_fn,
                task,
                session_id=session_id,
                seed=seed,
                verifier=verifier,
                strict_serialization=strict_serialization,
                pass_context=pass_context,
            )
        )
    task = dict(task or {})
    ctx = RecordContext(seed, strict_serialization=strict_serialization)
    token = bind_context(ctx)
    try:
        result = agent_fn(ctx, **task) if pass_context else agent_fn(**task)
    finally:
        unbind_context(token)
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


async def arecord(
    agent_fn: Callable[..., Any],
    task: Optional[Dict[str, Any]] = None,
    *,
    session_id: str,
    seed: int = 0,
    verifier: Optional[Callable[[Any], float]] = None,
    strict_serialization: bool = True,
    pass_context: bool = True,
) -> Trajectory:
    """Async recording: ``await agent_fn(ctx, **task)`` (or ``agent_fn(**task)``)."""
    task = dict(task or {})
    ctx = AsyncRecordContext(seed, strict_serialization=strict_serialization)
    token = bind_context(ctx)
    try:
        result = await (agent_fn(ctx, **task) if pass_context else agent_fn(**task))
    finally:
        unbind_context(token)
    traj = Trajectory(session_id=session_id, task=task, steps=ctx.steps, seed=seed, result=result)
    if verifier is not None:
        traj.outcome_score = float(verifier(result))
    return traj


def _is_awaitable(obj: Any) -> bool:
    """True if ``obj`` is awaitable (so an async ``observe`` policy is supported)."""
    import inspect

    return inspect.isawaitable(obj)


def _is_async(fn: Callable[..., Any]) -> bool:
    """True if calling ``fn`` yields a coroutine (an ``async def`` agent).

    Covers plain ``async def`` functions and ``functools.partial`` wrappers of
    them; a callable *object* with an async ``__call__`` is detected via its type.
    """
    import inspect

    if inspect.iscoroutinefunction(fn):
        return True
    dunder_call = inspect.getattr_static(type(fn), "__call__", None)
    return inspect.iscoroutinefunction(dunder_call)
