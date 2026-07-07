"""Deterministic replay with counterfactual interventions.

A :class:`ReplayPlan` decides, for every step index, whether the recorded action
is *held* at its factual value (served from the cassette) or *ablated*. The core
attribution move — "hold all prior steps at their factual recorded actions,
apply the intervention, and execute the trajectory forward" — is expressed as a
plan that holds ``{0..i-1}`` and resamples everything from ``i`` onward.

Resampling step ``i`` necessarily re-rolls every downstream stochastic step
(the SCM is a sequential dependency chain); the plan models exactly that by
running the policy live for all non-held indices.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set

from .recorder import AgentContext
from .types import StepKind, Trajectory

# Sentinel returned when a step is fully removed (empty action).
REMOVED = None


class ReplayPlan:
    """A per-step decision table for a single counterfactual rollout.

    * ``held`` — indices served from the recorded cassette (kept factual).
    * ``forced`` — indices whose action is overridden with a fixed value (``do``).
    * ``removed`` — indices dropped entirely (empty action).

    Any index not covered by the above is *resampled*: its policy is executed
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
    def ablate_from(cls, i: int, n: int) -> "ReplayPlan":
        """Hold steps ``< i`` factual and resample step ``i`` and all downstream.

        This is the single-step contrastive intervention used in Phase 1.
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
    """Agent context that follows a :class:`ReplayPlan` against a trajectory."""

    def __init__(self, trajectory: Trajectory, plan: ReplayPlan, seed: int) -> None:
        super().__init__(seed)
        self.trajectory = trajectory
        self.plan = plan
        self.diverged = False

    def _op(
        self,
        kind: StepKind,
        name: str,
        produce: Optional[Callable[[], Any]],
        inputs: Dict[str, Any],
    ) -> Any:
        idx = self._idx
        self._idx += 1
        decision = self.plan.decision(idx)

        if decision == "force":
            return self.plan.forced[idx]
        if decision == "remove":
            return REMOVED
        if decision == "hold" and idx < len(self.trajectory.steps):
            # Serve the recorded output verbatim (deterministic replay).
            return self.trajectory.steps[idx].output

        # Resample: run the live policy. Beyond the recorded horizon this is the
        # only sane option, so note the divergence for diagnostics.
        if idx >= len(self.trajectory.steps):
            self.diverged = True
        if produce is None:
            return None
        return produce()


def replay(
    agent_fn: Callable[..., Any],
    trajectory: Trajectory,
    plan: ReplayPlan,
    *,
    seed: int,
) -> Any:
    """Execute ``agent_fn`` once under ``plan`` and return its result."""
    ctx = ReplayContext(trajectory, plan, seed)
    return agent_fn(ctx, **trajectory.task)
