"""The ablation engine: stochastic run-forward under a replay plan.

Because the agent policy is stochastic, a single intervention does not yield one
outcome but an *outcome distribution*. The engine executes a plan ``rollouts``
times with independent seeds and returns the per-rollout failure indicators, from
which the scorer estimates ``P(fail | ...)`` together with its uncertainty.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

from .replayer import ReplayPlan, replay
from .stats import wilson_interval
from .types import Trajectory

# A verifier maps the agent's final result to a scalar outcome score in [0, 1],
# where higher is better. 0.0 means "failed" per the paper's convention.
Verifier = Callable[[Any], float]


class RolloutCache:
    """Shares prefix-hold / factual rollout results across analyses of one run.

    ``attribute``, ``drift`` and ``faithfulness`` each independently run the same
    ``ablate_from(i)`` / factual plan family on the same trajectory. Passing one
    cache to several of them lets the second analysis reuse the first's rollouts
    instead of re-executing the agent.

    **Only prefix-hold and factual plans are cached.** Shapley coalition values
    are *never* cached — the paper forbids reusing a coalition's value because
    each evaluation must stay an independent draw to preserve the marginal
    variance the Shapley CIs depend on — so :meth:`AblationEngine.coalition_value`
    bypasses the cache entirely.

    Correctness: a cached list is valid only for identical
    ``(trajectory content, plan, rollouts, fail_threshold, base_seed)``. The key
    includes ``trajectory.root_hash`` and those parameters, so mismatched configs
    never collide. Use one cache with a single ``(agent, verifier)`` pair.
    """

    def __init__(self) -> None:
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Tuple, compute: Callable[[], List[bool]]) -> List[bool]:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        value = compute()
        self._store[key] = value
        return value


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
        max_workers: int = 1,
        cache: Optional[RolloutCache] = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.trajectory = trajectory
        self.verifier = verifier
        self.fail_threshold = fail_threshold
        self.base_seed = base_seed
        self.pass_context = pass_context
        # Optional cross-analysis cache for prefix-hold / factual plans (never
        # coalition plans). See :class:`RolloutCache`.
        self.cache = cache
        # ``max_workers > 1`` runs rollouts on a thread pool. Each rollout is a
        # pure function of ``(plan, seed_tag, k)`` — its own ReplayContext, its own
        # seed, and an ambient context bound per worker thread (contextvars are
        # thread-isolated) — so parallel execution returns byte-identical results
        # to serial, just faster for I/O-bound produce fns (real LLM calls). It is
        # the single biggest wall-clock lever for real agents.
        self.max_workers = max(1, int(max_workers))

    def is_fail(self, result: Any) -> bool:
        return float(self.verifier(result)) < self.fail_threshold

    def _cache_key(self, kind: str, index: int, tag: int, rollouts: int) -> Tuple:
        """Identity of a cacheable prefix-hold/factual rollout batch."""
        return (
            self.trajectory.root_hash,
            kind,
            index,
            tag,
            rollouts,
            self.fail_threshold,
            self.base_seed,
        )

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
        return self._run_ks(plan, seed_tag, range(rollouts))

    def _run_ks(self, plan: ReplayPlan, seed_tag: int, ks: Iterable[int]) -> List[bool]:
        """Run rollouts for the given ``k`` indices, serial or on a thread pool.

        Results are returned in ``ks`` order regardless of completion order, so
        the output is deterministic (each rollout is pure in ``k``).
        """
        ks = list(ks)
        if self.max_workers <= 1 or len(ks) <= 1:
            return [self._one(plan, seed_tag, k) for k in ks]
        results: List[bool] = [False] * len(ks)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ks))) as ex:
            futures = {ex.submit(self._one, plan, seed_tag, k): i for i, k in enumerate(ks)}
            for fut, i in futures.items():
                results[i] = fut.result()
        return results

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
            hi = min(k + batch, max_rollouts)
            fails.extend(self._run_ks(plan, seed_tag, range(k, hi)))
            k = hi
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
        if self.cache is not None:
            key = self._cache_key("factual", -1, 0, rollouts)
            return self.cache.get_or_compute(key, lambda: self.run_plan(plan, rollouts, seed_tag=0))
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
            # Adaptive batches run a variable, data-dependent number of rollouts,
            # so they are not shared through the fixed-N cache.
            return self.run_plan_adaptive(
                plan,
                seed_tag=tag,
                target_ci_width=target_ci_width,
                min_rollouts=min(min_rollouts, rollouts),
                max_rollouts=rollouts,
            )
        if self.cache is not None:
            key = self._cache_key("ablate", index, tag, rollouts)
            return self.cache.get_or_compute(
                key, lambda: self.run_plan(plan, rollouts, seed_tag=tag)
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
