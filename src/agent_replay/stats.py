"""Statistical estimators used by the attribution scorer.

The paper insists on reporting *bounded* marginal contributions: a Wilson score
interval for the success proportion of the stochastic rollouts, and a bootstrap
interval on the difference between the factual and counterfactual failure
distributions. Both are implemented here with the standard library only.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

# 95% two-sided normal quantile.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(point, low, high)`` where ``point = successes / n``. Unlike the
    naive normal approximation, the Wilson interval stays inside ``[0, 1]`` and
    behaves sensibly for the small ``n`` typical of expensive agent rollouts.
    """
    if n == 0:
        return (0.0, 0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (phat, low, high)


def bootstrap_diff_interval(
    kept: Sequence[bool],
    ablated: Sequence[bool],
    iterations: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap interval for ``mean(kept) - mean(ablated)`` (failure rates).

    Both samples are Bernoulli failure indicators. We resample each with
    replacement ``iterations`` times and take the empirical percentiles of the
    difference. ``kept`` may be a single deterministic outcome (the factual run
    is deterministic given the cassette), in which case its mean is constant.
    """
    if not ablated:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    kept_list = [1.0 if b else 0.0 for b in kept] or [0.0]
    abl_list = [1.0 if b else 0.0 for b in ablated]
    point = _mean(kept_list) - _mean(abl_list)
    diffs: List[float] = []
    nk = len(kept_list)
    na = len(abl_list)
    for _ in range(iterations):
        ks = sum(kept_list[rng.randrange(nk)] for _ in range(nk)) / nk
        as_ = sum(abl_list[rng.randrange(na)] for _ in range(na)) / na
        diffs.append(ks - as_)
    diffs.sort()
    low = _percentile(diffs, alpha / 2.0)
    high = _percentile(diffs, 1.0 - alpha / 2.0)
    return (point, low, high)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def mean(xs: Sequence[float]) -> float:
    """Public arithmetic mean helper (returns 0.0 for an empty sequence)."""
    return _mean(xs)


def binary_entropy(p: float) -> float:
    """Shannon entropy (in bits) of a Bernoulli(``p``) outcome, in ``[0, 1]``.

    ``H(p) = -p*log2(p) - (1-p)*log2(1-p)``, with the usual ``0*log0 = 0``
    convention. Peaks at ``1.0`` when ``p == 0.5`` (maximally uncertain outcome)
    and is ``0.0`` at ``p in {0, 1}`` (the outcome is committed). This is the
    "entropy of autonomy" the drift curve tracks: how much of a run's fate is
    still open at a given step.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
