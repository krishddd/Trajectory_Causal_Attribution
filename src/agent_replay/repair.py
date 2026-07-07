"""Minimal counterfactual repair for the culprit step.

Once a step is isolated as the causal locus, we search a candidate space of
replacement actions, inject each via a ``do`` intervention, and re-run the
trajectory forward. A candidate is a *valid repair* if it flips the failure
rate below threshold. Among valid repairs we pick the most **minimal** one —
the smallest behavioural drift from the original action — mirroring the paper's
minimality objective.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .ablation import AblationEngine
from .hashing import canonical_json
from .replayer import ReplayPlan
from .types import Repair


def minimality(original: Any, candidate: Any) -> float:
    """Similarity of ``candidate`` to ``original`` in ``[0, 1]`` (1 = identical).

    Uses a normalised token/edit similarity over the canonical JSON forms, a
    stand-in for the paper's token-level edit-distance minimality metric.
    """
    a = canonical_json(original)
    b = canonical_json(candidate)
    return SequenceMatcher(None, a, b).ratio()


def _default_candidates(original: Any) -> List[Any]:
    """A small, generic candidate space when the caller supplies none."""
    cands: List[Any] = []
    if isinstance(original, str):
        cands = ["OK", "", original.strip()]
    elif isinstance(original, dict):
        cleaned = {k: v for k, v in original.items() if v not in (None, "")}
        cands = [cleaned, {}]
    else:
        cands = [None, 0, ""]
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in cands:
        key = canonical_json(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def find_minimal_repair(
    engine: AblationEngine,
    step_index: int,
    *,
    rollouts: int = 50,
    candidates: Optional[Dict[int, List[Any]]] = None,
) -> Optional[Repair]:
    """Search for the most minimal valid repair of ``step_index``.

    ``candidates`` optionally maps a step index to a list of replacement actions;
    otherwise a generic candidate space is derived from the recorded action.
    """
    traj = engine.trajectory
    if step_index >= len(traj):
        return None
    original = traj.steps[step_index].output
    cand_list = (candidates or {}).get(step_index) or _default_candidates(original)

    best: Optional[Repair] = None
    for cand in cand_list:
        # Hold steps < culprit at their factual actions, force the culprit to
        # the candidate, and resample strictly downstream so the effect of the
        # repair can propagate through the rest of the trajectory.
        held = set(range(step_index))
        plan = ReplayPlan(held=held, forced={step_index: cand})
        fails = engine.run_plan(plan, rollouts, seed_tag=90_000 + step_index)
        p_fail = sum(1 for f in fails if f) / len(fails) if fails else 1.0
        valid = p_fail < engine.fail_threshold
        m = minimality(original, cand)
        repair = Repair(
            step_index=step_index,
            original_action=original,
            repaired_action=cand,
            p_fail_after=p_fail,
            minimality=m,
            valid=valid,
        )
        if valid and (best is None or m > best.minimality):
            best = repair
    return best
