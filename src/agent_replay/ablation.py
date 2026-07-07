"""The ablation engine: stochastic run-forward under a replay plan.

Because the agent policy is stochastic, a single intervention does not yield one
outcome but an *outcome distribution*. The engine executes a plan ``rollouts``
times with independent seeds and returns the per-rollout failure indicators, from
which the scorer estimates ``P(fail | ...)`` together with its uncertainty.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set

from .replayer import ReplayPlan, replay
from .types import Trajectory

# A verifier maps the agent's final result to a scalar outcome score in [0, 1],
# where higher is better. 0.0 means "failed" per the paper's convention.
Verifier = Callable[[Any], float]


class AblationEngine:
    """Runs counterfactual rollouts for an agent against a recorded trajectory."""

    def __init__(
        self,
        agent_fn: Callable[..., Any],
        trajectory: Trajectory,
        verifier: Verifier,
        *,
        fail_threshold: float = 0.5,
        base_seed: int = 1_000,
    ) -> None:
        self.agent_fn = agent_fn
        self.trajectory = trajectory
        self.verifier = verifier
        self.fail_threshold = fail_threshold
        self.base_seed = base_seed
        # Deterministic, unique seed stream per (plan-tag, rollout) so that
        # independent value-function evaluations never accidentally share draws.
        self._seed_salt = 0

    def is_fail(self, result: Any) -> bool:
        return float(self.verifier(result)) < self.fail_threshold

    def run_plan(
        self,
        plan: ReplayPlan,
        rollouts: int,
        *,
        seed_tag: int = 0,
    ) -> List[bool]:
        """Execute ``plan`` ``rollouts`` times; return per-rollout failure flags.

        ``seed_tag`` distinguishes otherwise-identical plans (e.g. the same
        coalition drawn in two different Shapley permutations) so their rollouts
        are statistically independent — the paper forbids caching a coalition's
        value precisely to preserve this per-evaluation variance.
        """
        fails: List[bool] = []
        for k in range(rollouts):
            seed = self.base_seed + 1_000_003 * (seed_tag + 1) + k
            result = replay(self.agent_fn, self.trajectory, plan, seed=seed)
            fails.append(self.is_fail(result))
        return fails

    def factual_fail(self, rollouts: int = 1) -> List[bool]:
        """Failure indicators for the fully-held (factual) plan.

        The factual run is deterministic given the cassette, so one rollout is
        sufficient, but the signature accepts more for symmetry.
        """
        plan = ReplayPlan.factual(len(self.trajectory))
        return self.run_plan(plan, rollouts, seed_tag=0)

    def ablate_from(
        self, index: int, rollouts: int, *, seed_tag: Optional[int] = None
    ) -> List[bool]:
        """Failure indicators for holding ``< index`` and resampling from ``index``."""
        plan = ReplayPlan.ablate_from(index, len(self.trajectory))
        tag = seed_tag if seed_tag is not None else index + 1
        return self.run_plan(plan, rollouts, seed_tag=tag)

    def coalition_value(
        self,
        members: Set[int],
        rollouts: int,
        *,
        seed_tag: int,
    ) -> float:
        """Estimate ``v(S)`` = P(fail) with ``members`` held factual.

        This is the Shapley value function. It is intentionally *not* memoised.
        """
        fails = self.run_plan(ReplayPlan.coalition(members), rollouts, seed_tag=seed_tag)
        return sum(1 for f in fails if f) / len(fails) if fails else 0.0
