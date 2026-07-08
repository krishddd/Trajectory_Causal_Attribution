"""The ablation engine: stochastic run-forward under a replay plan.

Because the agent policy is stochastic, a single intervention does not yield one
outcome but an *outcome distribution*. The engine executes a plan ``rollouts``
times with independent seeds and returns the per-rollout failure indicators, from
which the scorer estimates ``P(fail | ...)`` together with its uncertainty.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set

from .replayer import ReplayPlan, replay
from .stats import wilson_interval
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
        pass_context: bool = True,
    ) -> None:
        self.agent_fn = agent_fn
        self.trajectory = trajectory
        self.verifier = verifier
        self.fail_threshold = fail_threshold
        self.base_seed = base_seed
        self.pass_context = pass_context

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
            fails.append(self._one(plan, seed_tag, k))
        return fails

    def _one(self, plan: ReplayPlan, seed_tag: int, k: int) -> bool:
        """Run a single rollout ``k`` of ``plan`` and return its failure flag."""
        seed = self.base_seed + 1_000_003 * (seed_tag + 1) + k
        result = replay(
            self.agent_fn, self.trajectory, plan, seed=seed, pass_context=self.pass_context
        )
        return self.is_fail(result)

    def run_plan_adaptive(
        self,
        plan: ReplayPlan,
        *,
        seed_tag: int = 0,
        target_ci_width: float = 0.2,
        min_rollouts: int = 8,
        max_rollouts: int = 200,
        batch: int = 8,
    ) -> List[bool]:
        """Run rollouts in batches until the failure-rate CI is tight enough.

        Sequential stopping (the KernelSHAP / VRDS practice): keep sampling until
        the Wilson 95% interval on the failure proportion is narrower than
        ``target_ci_width`` (or ``max_rollouts`` is hit), never fewer than
        ``min_rollouts``. Typically resolves in far fewer rollouts than a fixed
        budget while preserving the same statistical guarantee, because a
        decisive proportion (near 0 or 1) reaches a tight interval quickly.
        """
        fails: List[bool] = []
        k = 0
        while k < max_rollouts:
            for _ in range(batch):
                if k >= max_rollouts:
                    break
                fails.append(self._one(plan, seed_tag, k))
                k += 1
            if k >= min_rollouts:
                n_fail = sum(1 for f in fails if f)
                _, low, high = wilson_interval(n_fail, len(fails))
                if (high - low) <= target_ci_width:
                    break
        return fails

    def factual_fail(self, rollouts: int = 1) -> List[bool]:
        """Failure indicators for the fully-held (factual) plan.

        The factual run is deterministic given the cassette, so one rollout is
        sufficient, but the signature accepts more for symmetry.
        """
        plan = ReplayPlan.factual(len(self.trajectory))
        return self.run_plan(plan, rollouts, seed_tag=0)

    def ablate_from(
        self,
        index: int,
        rollouts: int,
        *,
        seed_tag: Optional[int] = None,
        adaptive: bool = False,
        target_ci_width: float = 0.2,
        min_rollouts: int = 8,
    ) -> List[bool]:
        """Failure indicators for holding ``< index`` and resampling from ``index``.

        With ``adaptive`` the rollout count is chosen by sequential stopping
        (``rollouts`` becomes the cap); otherwise exactly ``rollouts`` are run.
        """
        plan = ReplayPlan.ablate_from(index, len(self.trajectory))
        tag = seed_tag if seed_tag is not None else index + 1
        if adaptive:
            return self.run_plan_adaptive(
                plan,
                seed_tag=tag,
                target_ci_width=target_ci_width,
                min_rollouts=min(min_rollouts, rollouts),
                max_rollouts=rollouts,
            )
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
