"""The attribution scorer: turn ablation rollouts into a causal verdict.

Two estimators are implemented, matching the paper:

**Phase 1 — single-step contrastive estimation.** For every candidate step ``i``
we hold ``< i`` factual, resample ``>= i``, and compute

    attribution(i) = P(fail | kept) - P(fail | ablated from i)

with a Wilson interval on the ablated failure rate and a bootstrap interval on
the difference. The *magnitude* of this score is deliberately **not** used to
pick the culprit: resampling an early step re-rolls the fatal late step too
(the butterfly-effect confound), inflating early scores. Instead the
**Point-of-Commitment Rule** selects the *latest* step whose interval still
strictly excludes zero — the final juncture at which re-deciding can still
rescue the run.

**Phase 2 — Shapley-value attribution.** To split credit fairly across
interacting steps we average each step's marginal contribution over sampled
permutations, using antithetic reverse-permutation pairing to cut variance.
Coalition values are never cached (that would collapse the marginal variance to
zero and produce falsely narrow intervals), and no truncation is used (it would
skip pivotal late steps).
"""

from __future__ import annotations

import random
from typing import List, Optional, Set

from .ablation import AblationEngine
from .errors import SuccessfulRunError
from .stats import bootstrap_diff_interval, mean, wilson_interval
from .types import (
    AttributionResult,
    ConfidenceInterval,
    StepAttribution,
    Trajectory,
)


def _rate(fails: List[bool]) -> float:
    return sum(1 for f in fails if f) / len(fails) if fails else 0.0


def contrastive_attribution(
    engine: AblationEngine,
    rollouts: int,
    *,
    bootstrap_seed: int = 7,
    adaptive: bool = False,
    target_ci_width: float = 0.2,
    min_rollouts: int = 8,
) -> List[StepAttribution]:
    """Phase 1: per-step contrastive scores with bootstrap difference intervals.

    The factual (kept) run is deterministic given the cassette, so it is
    evaluated once. The ablated distribution is estimated over ``rollouts``
    stochastic run-forwards. The reported interval is a bootstrap CI on the
    *difference* of failure rates — the quantity attribution actually is —
    which, unlike a Wilson interval transformed onto the difference, stays
    numerically exact in the boundary case where the ablated run always fails
    (``P(fail|ablated) = 1`` ⇒ attribution collapses cleanly to zero). Wilson
    proportion intervals remain available in :mod:`agent_replay.stats` for
    reporting ``P(fail|ablated)`` on its own.

    With ``adaptive`` each step uses sequential stopping (``rollouts`` is the
    per-step cap): rollouts accrue until the failure-rate interval is narrower
    than ``target_ci_width``. Decisive steps — the common case, where a step is
    either clearly pivotal or clearly irrelevant — resolve in far fewer rollouts.
    """
    traj = engine.trajectory
    kept_fails = engine.factual_fail(rollouts=1)
    p_kept = _rate(kept_fails)

    out: List[StepAttribution] = []
    for step in traj.steps:
        ablated = engine.ablate_from(
            step.index,
            rollouts,
            adaptive=adaptive,
            target_ci_width=target_ci_width,
            min_rollouts=min_rollouts,
        )
        p_abl = _rate(ablated)
        _, b_low, b_high = bootstrap_diff_interval(
            kept_fails, ablated, seed=bootstrap_seed + step.index
        )
        ci = ConfidenceInterval(
            point=p_kept - p_abl,
            low=b_low,
            high=b_high,
            method="bootstrap-diff",
        )
        n_abl_fail = sum(1 for f in ablated if f)
        _, w_low, w_high = wilson_interval(n_abl_fail, len(ablated))
        p_abl_ci = ConfidenceInterval(point=p_abl, low=w_low, high=w_high, method="wilson")
        out.append(
            StepAttribution(
                index=step.index,
                name=step.name,
                kind=step.kind.value,
                p_fail_kept=p_kept,
                p_fail_ablated=p_abl,
                attribution=p_kept - p_abl,
                ci=ci,
                resamplable=step.resamplable,
                p_fail_ablated_ci=p_abl_ci,
            )
        )
    return out


def point_of_commitment(steps: List[StepAttribution]) -> Optional[int]:
    """Latest step whose attribution CI strictly excludes zero.

    Beyond this point the failure is structurally locked in: resampling no longer
    systematically changes the outcome, so its interval brackets zero. Steps that
    are not resamplable cannot be a commitment locus (they were never truly
    ablated) and are skipped.
    """
    latest: Optional[int] = None
    for s in steps:
        if not s.resamplable:
            continue
        if s.ci.excludes_zero() and s.attribution > 0:
            if latest is None or s.index > latest:
                latest = s.index
    return latest


def shapley_attribution(
    engine: AblationEngine,
    rollouts: int,
    *,
    permutation_pairs: int = 8,
    seed: int = 13,
    adaptive: bool = False,
    target_ci_width: float = 0.2,
    min_pairs: int = 2,
    max_pairs: Optional[int] = None,
) -> List[StepAttribution]:
    """Phase 2: Monte-Carlo permutation Shapley values with antithetic pairing.

    For each permutation we walk the coalition from empty to full, adding one
    step at a time and recording the marginal change in the failure rate that
    step causes. Averaging those marginals over permutations yields each step's
    Shapley value; by the efficiency axiom they sum to ``v(full) - v(empty)``.

    Cost is ``pairs × 2 × (n + 1) × rollouts`` executions — the most expensive
    phase. With ``adaptive`` we stop adding permutation pairs once every step's
    bootstrap CI on its marginal mean is narrower than ``target_ci_width`` (after
    at least ``min_pairs``, up to ``max_pairs`` — defaulting to
    ``permutation_pairs``), the same sequential-stopping idea adaptive gives the
    contrastive phase. Pairs are drawn from a single RNG sequence, so the
    non-adaptive path is byte-identical to consuming a fixed pair count.
    """
    traj = engine.trajectory
    n = len(traj)
    rng = random.Random(seed)
    if adaptive:
        cap = max_pairs if max_pairs is not None else permutation_pairs
    else:
        cap = permutation_pairs

    # marginals[i] collects every observed marginal contribution of step i.
    marginals: List[List[float]] = [[] for _ in range(n)]
    tag = 0
    pairs_done = 0
    while pairs_done < cap:
        base = list(range(n))
        rng.shuffle(base)
        for perm in (base, list(reversed(base))):  # antithetic partner
            coalition: Set[int] = set()
            # v(empty): a fresh, uncached evaluation per permutation.
            v_prev = engine.coalition_value(set(), rollouts, seed_tag=tag)
            tag += 1
            for step_idx in perm:
                coalition.add(step_idx)
                v_curr = engine.coalition_value(coalition, rollouts, seed_tag=tag)
                tag += 1
                marginals[step_idx].append(v_curr - v_prev)
                v_prev = v_curr
        pairs_done += 1
        if adaptive and pairs_done >= min_pairs and n > 0:
            widest = 0.0
            for i in range(n):
                lo, hi = _bootstrap_mean_ci(marginals[i], seed=seed + i)
                widest = max(widest, hi - lo)
            if widest <= target_ci_width:
                break

    out: List[StepAttribution] = []
    for step in traj.steps:
        vals = marginals[step.index]
        phi = mean(vals)
        # Bootstrap a CI over the collected marginals for this step.
        low, high = _bootstrap_mean_ci(vals, seed=seed + step.index)
        out.append(
            StepAttribution(
                index=step.index,
                name=step.name,
                kind=step.kind.value,
                p_fail_kept=0.0,
                p_fail_ablated=0.0,
                attribution=phi,
                ci=ConfidenceInterval(point=phi, low=low, high=high, method="shapley-marginals"),
                shapley=phi,
                shapley_ci=ConfidenceInterval(
                    point=phi, low=low, high=high, method="shapley-marginals"
                ),
                resamplable=step.resamplable,
            )
        )
    return out


def _bootstrap_mean_ci(
    values: List[float], seed: int, iterations: int = 1000, alpha: float = 0.05
) -> tuple:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    def pct(q: float) -> float:
        if len(means) == 1:
            return means[0]
        pos = q * (len(means) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(means) - 1)
        frac = pos - lo
        return means[lo] * (1 - frac) + means[hi] * frac

    return (pct(alpha / 2), pct(1 - alpha / 2))


def _negate(step: StepAttribution) -> None:
    """Flip a step's scores and CIs in place (failure ⇄ credit).

    Credit mode reflects every attribution about zero so a positive score
    measures the failure risk a step's re-decision averts. Both the contrastive
    spine *and* the Shapley value (with their CIs) must flip together — otherwise
    a ``method="both"`` credit report would show credit-signed contrastive scores
    beside failure-signed Shapley values for the same step, and a
    ``method="shapley"`` credit run would let ``_select_culprit`` pick the step
    that *least* secured success as the "save point" (the exact inverse).
    """
    step.attribution = -step.attribution
    lo, hi = step.ci.low, step.ci.high
    step.ci = ConfidenceInterval(point=-step.ci.point, low=-hi, high=-lo, method=step.ci.method)
    if step.shapley is not None:
        step.shapley = -step.shapley
    if step.shapley_ci is not None:
        slo, shi = step.shapley_ci.low, step.shapley_ci.high
        step.shapley_ci = ConfidenceInterval(
            point=-step.shapley_ci.point, low=-shi, high=-slo, method=step.shapley_ci.method
        )


def attribute(
    trajectory: Trajectory,
    agent_fn,
    verifier,
    *,
    rollouts: int = 50,
    method: str = "contrastive",
    fail_threshold: float = 0.5,
    permutation_pairs: int = 8,
    base_seed: int = 1_000,
    repair: bool = False,
    repair_candidates: Optional[dict] = None,
    repair_propose_fn=None,
    on_success: str = "error",
    pass_context: bool = True,
    adaptive: bool = False,
    target_ci_width: float = 0.2,
    max_workers: int = 1,
    cache=None,
) -> AttributionResult:
    """Run the full attribution pipeline and return an :class:`AttributionResult`.

    ``method`` is ``"contrastive"`` (Phase 1 only), ``"shapley"`` (Phase 2 only),
    or ``"both"`` (contrastive localisation + Shapley credit split).

    With ``adaptive`` the contrastive phase uses sequential stopping (``rollouts``
    becomes the per-step cap and ``target_ci_width`` the precision target),
    typically cutting rollouts several-fold on decisive steps.

    Attribution is only meaningful for a *failing* trajectory. ``on_success``
    controls what happens when the factual run passed:

    * ``"error"`` (default) — raise :class:`~agent_replay.errors.SuccessfulRunError`.
    * ``"credit"`` — run the symmetric **credit** analysis: which step most
      secured the success (the latest step whose re-decision would introduce a
      significant failure risk). Scores are the sign-flipped contrastive
      differences and ``mode`` is set to ``"credit"``.
    """
    engine = AblationEngine(
        agent_fn,
        trajectory,
        verifier,
        fail_threshold=fail_threshold,
        base_seed=base_seed,
        pass_context=pass_context,
        max_workers=max_workers,
        cache=cache,
    )

    # Establish the factual outcome.
    factual_result = trajectory.result
    outcome_score = (
        trajectory.outcome_score
        if trajectory.outcome_score is not None
        else float(verifier(factual_result))
    )
    failed = outcome_score < fail_threshold

    if not failed:
        if on_success == "error":
            raise SuccessfulRunError(
                f"Run '{trajectory.session_id}' passed (score {outcome_score:.3f} >= "
                f"threshold {fail_threshold}); failure attribution is undefined. Pass "
                f"on_success='credit' for the symmetric 'which step secured success' "
                f"analysis, or check your verifier/fail_threshold."
            )
        if on_success != "credit":
            raise ValueError(f"on_success must be 'error' or 'credit', got {on_success!r}")

    mode = "credit" if not failed else "failure"

    contrastive: List[StepAttribution] = []
    shapley: List[StepAttribution] = []

    if method in ("contrastive", "both"):
        contrastive = contrastive_attribution(
            engine, rollouts, adaptive=adaptive, target_ci_width=target_ci_width
        )
    if method in ("shapley", "both"):
        shapley = shapley_attribution(
            engine,
            rollouts,
            permutation_pairs=permutation_pairs,
            seed=base_seed + 13,
            adaptive=adaptive,
            target_ci_width=target_ci_width,
        )

    # Merge: contrastive is the spine (gives P(fail|kept/ablated) + CIs); attach
    # Shapley values when both were computed.
    if method == "shapley":
        steps = shapley
    elif method == "both":
        steps = contrastive
        shap_by_idx = {s.index: s for s in shapley}
        for s in steps:
            sh = shap_by_idx.get(s.index)
            if sh is not None:
                s.shapley = sh.shapley
                s.shapley_ci = sh.shapley_ci
    else:
        steps = contrastive

    # In credit mode the raw scores are failure-oriented (p_kept - p_ablated <= 0
    # for contrastive; the Shapley marginals point the same way); flip every score
    # and CI so a positive "credit" measures the failure risk a step's re-decision
    # averts. This must cover the Shapley spine too — see _negate.
    if mode == "credit":
        for s in steps:
            _negate(s)

    # The point-of-commitment (or, in credit mode, the "save point") is defined by
    # the contrastive difference CIs; for Shapley-only runs it is undefined and the
    # culprit falls back to argmax phi.
    poc = point_of_commitment(steps) if method in ("contrastive", "both") else None

    culprit_index = _select_culprit(steps, poc)

    result = AttributionResult(
        session_id=trajectory.session_id,
        total_steps=len(trajectory),
        outcome_score=outcome_score,
        failed=failed,
        method=method,
        rollouts=rollouts,
        steps=steps,
        point_of_commitment=poc,
        culprit_index=culprit_index,
        mode=mode,
    )

    # Repair only applies to a genuine failure (flip fail→success).
    if repair and mode == "failure" and culprit_index is not None:
        from .repair import find_minimal_repair

        result.repair = find_minimal_repair(
            engine,
            culprit_index,
            rollouts=rollouts,
            candidates=repair_candidates,
            propose_fn=repair_propose_fn,
        )

    return result


def _select_culprit(steps: List[StepAttribution], poc: Optional[int]) -> Optional[int]:
    """Pick the responsible step: prefer point-of-commitment, else max score."""
    if not steps:
        return None
    if poc is not None:
        return poc
    # Shapley (or contrastive without a significant locus): highest attribution.
    best = max(steps, key=lambda s: s.shapley if s.shapley is not None else s.attribution)
    score = best.shapley if best.shapley is not None else best.attribution
    return best.index if score > 0 else None
