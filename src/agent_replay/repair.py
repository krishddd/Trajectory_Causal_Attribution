"""Minimal counterfactual repair for the culprit step.

Once a step is isolated as the causal locus, we search a candidate space of
replacement actions, inject each via a ``do`` intervention, and re-run the
trajectory forward. A candidate is a *valid repair* if it flips the failure
rate below threshold. Among valid repairs we pick the most **minimal** one —
the smallest behavioural drift from the original action — mirroring the paper's
minimality objective.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from .ablation import AblationEngine
from .hashing import canonical_json
from .replayer import ReplayPlan
from .types import AttributionResult, Repair, Step, Trajectory

# A proposer maps (culprit step, full trajectory) to candidate replacement
# actions. The user supplies the model call (e.g. an LLM) so the core stays
# dependency-free; candidates are then validated causally like any other.
ProposeFn = Callable[[Step, Trajectory], List[Any]]


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
    propose_fn: Optional[ProposeFn] = None,
) -> Optional[Repair]:
    """Search for the most minimal valid repair of ``step_index``.

    Candidate replacement actions come from (in priority order): an explicit
    ``candidates[step_index]`` list; a user-supplied ``propose_fn(step, trajectory)``
    (e.g. an LLM that proposes fixes — the model call is the caller's, keeping the
    core dependency-free); otherwise a generic derived space. Every candidate is
    validated causally by re-running the trajectory forward, so a proposer can be
    creative without compromising soundness.
    """
    traj = engine.trajectory
    if step_index >= len(traj):
        return None
    step = traj.steps[step_index]
    original = step.output

    cand_list = (candidates or {}).get(step_index)
    if cand_list is None and propose_fn is not None:
        cand_list = list(propose_fn(step, traj))
    if not cand_list:
        cand_list = _default_candidates(original)

    # Baseline failure rate of the culprit held at its factual (bad) action, for
    # the guard/report to show what the repair improves over.
    base_fails = engine.run_plan(
        ReplayPlan(held=set(range(step_index + 1))), rollouts, seed_tag=80_000 + step_index
    )
    p_fail_before = sum(1 for f in base_fails if f) / len(base_fails) if base_fails else 1.0

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
            step_name=step.name,
            step_kind=step.kind.value,
            p_fail_before=p_fail_before,
        )
        if valid and (best is None or m > best.minimality):
            best = repair
    return best


def export_contrastive_pairs(
    results: List[AttributionResult],
    path: str,
    *,
    trajectories: Optional[Dict[str, Trajectory]] = None,
    valid_only: bool = True,
) -> int:
    """Write validated ``(wrong step -> minimal fix)`` pairs as JSONL for training.

    Each attribution result that carries a repair contributes one contrastive
    pair — the mathematically proven wrong action coupled with its minimal
    corrected counterpart — the high-signal, learning-ready supervision the
    source research describes (for DPO / preference optimisation). ``trajectories``
    optionally supplies the recorded context per session so the pair includes the
    culprit step's inputs. Returns the number of pairs written.
    """
    trajectories = trajectories or {}
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for result in results:
            repair = result.repair
            if repair is None or (valid_only and not repair.valid):
                continue
            context = None
            traj = trajectories.get(result.session_id)
            if traj is not None and repair.step_index < len(traj):
                context = traj.steps[repair.step_index].inputs
            pair = {
                "session_id": result.session_id,
                "step_index": repair.step_index,
                "step": f"{repair.step_kind}:{repair.step_name}",
                "context": context,
                "rejected": repair.original_action,
                "chosen": repair.repaired_action,
                "p_fail_before": repair.p_fail_before,
                "p_fail_after": repair.p_fail_after,
                "minimality": repair.minimality,
            }
            fh.write(json.dumps(pair, default=str) + "\n")
            n += 1
    return n
