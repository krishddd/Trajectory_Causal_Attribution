"""Run several analyses of one trajectory while sharing their rollouts.

``attribute``, ``drift`` and ``faithfulness`` each re-run the same prefix-hold
rollout family on the same trajectory. Calling them separately pays for those
rollouts two or three times. :func:`analyze` runs the requested analyses with a
single shared :class:`~agent_replay.ablation.RolloutCache` and aligned seeds, so
the second and third analysis reuse the first's rollouts — the combined cost is
barely more than the most expensive one alone.

The shared cache covers the prefix-hold / factual plans that ``attribute`` (its
contrastive phase) and ``drift`` have in common; Shapley coalition rollouts are
never cached (they must stay independent draws). ``faithfulness`` shares only its
factual baseline, since its per-step masking plan differs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .ablation import RolloutCache
from .attribution import attribute
from .drift import drift
from .errors import SuccessfulRunError
from .faithfulness import faithfulness
from .types import Trajectory


def analyze(
    trajectory: Trajectory,
    agent_fn: Callable[..., Any],
    verifier: Callable[[Any], float],
    *,
    rollouts: int = 50,
    base_seed: int = 1_000,
    fail_threshold: float = 0.5,
    with_attribution: bool = True,
    with_drift: bool = True,
    with_faithfulness: bool = False,
    method: str = "contrastive",
    state_scorer: Optional[Callable[[Any], float]] = None,
    pass_context: bool = True,
) -> Dict[str, Any]:
    """Run attribution + drift (+ optionally faithfulness) sharing one rollout cache.

    All analyses use the same ``base_seed``, ``rollouts`` and ``fail_threshold`` so
    their identical prefix-hold plans hit the shared cache. Returns a dict with the
    requested results under ``"attribution"`` / ``"drift"`` / ``"faithfulness"``
    (attribution is ``None`` if the run passed) plus the ``"cache"`` (inspect
    ``cache.hits`` / ``cache.misses`` to see the reuse).
    """
    cache = RolloutCache()
    out: Dict[str, Any] = {"cache": cache}

    if with_attribution:
        try:
            out["attribution"] = attribute(
                trajectory,
                agent_fn,
                verifier,
                rollouts=rollouts,
                method=method,
                base_seed=base_seed,
                fail_threshold=fail_threshold,
                pass_context=pass_context,
                cache=cache,
            )
        except SuccessfulRunError:
            out["attribution"] = None  # passing run: nothing to attribute

    if with_drift:
        out["drift"] = drift(
            trajectory,
            agent_fn,
            verifier,
            rollouts=rollouts,
            base_seed=base_seed,
            fail_threshold=fail_threshold,
            state_scorer=state_scorer,
            pass_context=pass_context,
            cache=cache,
        )

    if with_faithfulness:
        out["faithfulness"] = faithfulness(
            trajectory,
            agent_fn,
            verifier,
            rollouts=rollouts,
            base_seed=base_seed,
            fail_threshold=fail_threshold,
            pass_context=pass_context,
            cache=cache,
        )

    return out
