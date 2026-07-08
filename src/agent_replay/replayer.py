"""Deterministic replay with counterfactual interventions.

A :class:`ReplayPlan` decides, for every recorded step, whether the recorded
action is *held* at its factual value (served from the cassette) or *ablated*.
The core attribution move — "hold all prior steps at their factual recorded
actions, apply the intervention, and execute the trajectory forward" — is a plan
that holds ``{0..i-1}`` and resamples everything from ``i`` onward.

Matching live calls to recorded steps
--------------------------------------
Live replay calls are matched to recorded steps by **idempotency key**
(``Step.op_key`` = hash of kind + name + inputs), consuming recorded steps in
order, *not* by call position. This is the VCR/cassette semantics the research
prescribes ("recorded responses injected based on idempotency keys derived from
input hashes") and it is what keeps *branching* agents sound: when an upstream
ablation changes the control flow, a held step is served from the cassette only
when the very same operation actually recurs — otherwise the timeline has
diverged and the call is resampled live. Positional matching (the naive
alternative) would serve one step's recorded output into a different operation.

For a linear agent whose control flow never branches, key-in-order matching is
identical to positional matching, so this generalises the previous behaviour
without changing it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from ._ambient import bind_context, unbind_context
from .hashing import content_hash
from .recorder import AgentContext, AsyncAgentContext, _is_async
from .types import StepKind, Trajectory


class _Removed:
    """Distinct sentinel for a fully-removed step (unlike a legitimate ``None``)."""

    _instance: Optional["_Removed"] = None

    def __new__(cls) -> "_Removed":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<removed>"


REMOVED = _Removed()


class ReplayPlan:
    """A per-step decision table for a single counterfactual rollout.

    Indices refer to *recorded* step indices (a step's stable identity), not live
    call positions.

    * ``held`` — indices served from the recorded cassette (kept factual).
    * ``forced`` — indices whose action is overridden with a fixed value (``do``).
    * ``removed`` — indices dropped entirely (empty action).

    Any recorded step not covered above is *resampled*: its policy is executed
    live, producing a fresh draw for this rollout.
    """

    def __init__(
        self,
        held: Optional[Set[int]] = None,
        forced: Optional[Dict[int, Any]] = None,
        removed: Optional[Set[int]] = None,
    ) -> None:
        self.held: Set[int] = set(held or set())
        self.forced: Dict[int, Any] = dict(forced or {})
        self.removed: Set[int] = set(removed or set())

    @classmethod
    def factual(cls, n: int) -> "ReplayPlan":
        """Hold every step: a deterministic replay of the recorded run."""
        return cls(held=set(range(n)))

    @classmethod
    def ablate_from(cls, i: int, n: Optional[int] = None) -> "ReplayPlan":
        """Hold steps ``< i`` factual and resample step ``i`` and all downstream.

        This is the single-step contrastive intervention used in Phase 1. ``n``
        is accepted for call-site symmetry and ignored (the held set is the
        prefix ``{0..i-1}`` regardless of trajectory length).
        """
        return cls(held=set(range(i)))

    @classmethod
    def coalition(cls, members: Set[int]) -> "ReplayPlan":
        """Hold exactly ``members`` factual; resample the rest (Shapley value fn)."""
        return cls(held=set(members))

    def decision(self, idx: int) -> str:
        if idx in self.forced:
            return "force"
        if idx in self.removed:
            return "remove"
        if idx in self.held:
            return "hold"
        return "resample"


class ReplayContext(AgentContext):
    """Agent context that follows a :class:`ReplayPlan` against a trajectory.

    ``match`` selects how live calls bind to recorded steps: ``"key"`` (default,
    idempotency-key matching — branch-safe) or ``"position"`` (legacy call-index
    matching, kept for debugging and strictly-linear fast paths).
    """

    def __init__(
        self,
        trajectory: Trajectory,
        plan: ReplayPlan,
        seed: int,
        *,
        match: str = "key",
    ) -> None:
        super().__init__(seed)
        self.trajectory = trajectory
        self.plan = plan
        self.match = match
        self.diverged = False
        # key -> ordered list of recorded indices sharing that op key (handles
        # loops that repeat the same operation). Consumed in recorded order.
        self._by_key: Dict[str, List[int]] = {}
        for step in trajectory.steps:
            self._by_key.setdefault(step.op_key(), []).append(step.index)
        self._consumed: Set[int] = set()

    def _resolve(self, kind: StepKind, name: str, inputs: Dict[str, Any]) -> Optional[int]:
        """Return the recorded index this live call binds to, or ``None`` if none."""
        if self.match == "position":
            idx = self._idx
            return idx if idx < len(self.trajectory.steps) else None
        key = content_hash({"kind": kind.value, "name": name, "inputs": inputs})
        for idx in self._by_key.get(key, ()):
            if idx not in self._consumed:
                return idx
        return None

    def _decide(self, kind: StepKind, name: str, inputs: Dict[str, Any]):
        """Bind a live call to the cassette and decide its fate.

        Returns ``(needs_produce, value)``: when ``needs_produce`` is True the
        caller must run the live policy (sync ``produce()`` or async
        ``await produce()``); otherwise ``value`` is the recorded/forced result.
        Shared by the sync and async replay contexts so their matching logic can
        never drift apart.
        """
        rec_idx = self._resolve(kind, name, inputs)
        self._idx += 1
        if rec_idx is None:
            # No recorded counterpart: the timeline diverged (an upstream ablation
            # changed the control flow). Only live resampling is meaningful.
            self.diverged = True
            return True, None
        self._consumed.add(rec_idx)
        step = self.trajectory.steps[rec_idx]
        decision = self.plan.decision(rec_idx)
        if decision == "force":
            return False, self.plan.forced[rec_idx]
        if decision == "remove":
            return False, REMOVED
        if decision == "hold":
            return False, step.output
        # decision == "resample": a step with no genuine policy cannot be
        # re-drawn, so serve its recorded output rather than corrupt the rollout.
        if not step.resamplable:
            return False, step.output
        return True, None

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
    ) -> Any:
        needs_produce, value = self._decide(kind, name, inputs)
        if not needs_produce:
            return value
        return produce() if produce is not None else None


class AsyncReplayContext(AsyncAgentContext):
    """Async replay context: same cassette matching as :class:`ReplayContext`."""

    def __init__(
        self, trajectory: Trajectory, plan: ReplayPlan, seed: int, *, match: str = "key"
    ) -> None:
        super().__init__(seed)
        # Reuse the sync context purely as the matching state machine (its
        # ``_decide`` holds the single source of truth for binding + fate).
        self._m = ReplayContext(trajectory, plan, seed, match=match)

    @property
    def diverged(self) -> bool:
        return self._m.diverged

    async def _aop(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
        resamplable: Optional[bool] = None,
    ) -> Any:
        needs_produce, value = self._m._decide(kind, name, inputs)
        if not needs_produce:
            return value
        return await produce() if produce is not None else None


def replay(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    plan: ReplayPlan,
    *,
    seed: int,
    match: str = "key",
    pass_context: bool = True,
) -> Any:
    """Execute ``agent_fn`` once under ``plan`` and return its result.

    ``pass_context`` mirrors :func:`agent_replay.record`: ``True`` calls
    ``agent_fn(ctx, **task)``; ``False`` calls ``agent_fn(**task)`` and relies on
    the ambient context (auto-instrumented agents). The ambient context is always
    published for the duration of the run. ``async def`` agents are detected and
    run to completion via :func:`areplay`, so async agents work through the whole
    synchronous attribution pipeline unchanged.
    """
    if _is_async(agent_fn):
        import asyncio

        return asyncio.run(
            areplay(agent_fn, trajectory, plan, seed=seed, match=match, pass_context=pass_context)
        )
    ctx = ReplayContext(trajectory, plan, seed, match=match)
    token = bind_context(ctx)
    try:
        return agent_fn(ctx, **trajectory.task) if pass_context else agent_fn(**trajectory.task)
    finally:
        unbind_context(token)


async def areplay(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    plan: ReplayPlan,
    *,
    seed: int,
    match: str = "key",
    pass_context: bool = True,
) -> Any:
    """Async replay: ``await agent_fn(ctx, **task)`` (or ``agent_fn(**task)``)."""
    ctx = AsyncReplayContext(trajectory, plan, seed, match=match)
    token = bind_context(ctx)
    try:
        if pass_context:
            return await agent_fn(ctx, **trajectory.task)
        return await agent_fn(**trajectory.task)
    finally:
        unbind_context(token)
