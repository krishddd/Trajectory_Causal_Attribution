"""Multi-trajectory aggregation: find your agent's *systematic* weak step.

Attributing one failure tells you which step broke *that* run. But agents fail
the same way repeatedly, and the actionable question is: **across many failures
of the same task, is there a step that is consistently to blame?** A step that is
the culprit in 1 of 20 runs is bad luck; a step that is the culprit in 15 of 20 is
a design flaw.

This module pools per-run :class:`~agent_replay.types.AttributionResult`s by step
**name** (not index — indices shift between runs, names are the stable identity of
an operation, and the same named step may even recur within a run). For each
named step it reports how often it was the culprit, its mean attribution with a
bootstrap interval over runs, and ranks them so the top row is the agent's
systematic weak point.

No new estimation machinery: this is a pure reduction over `attribute` outputs,
so it inherits all of the engine's soundness (branch-safe matching, PoC rule,
non-resamplable handling).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .attribution import attribute
from .stats import bootstrap_mean_interval
from .types import AttributionResult, ConfidenceInterval, Trajectory


@dataclass
class StepAggregate:
    """Pooled blame for one named step across every run it appeared in."""

    name: str
    kind: str
    n_present: int  # runs containing at least one step with this name+kind
    n_culprit: int  # runs where this name was the attributed culprit
    n_poc: int  # runs where this name was the point of commitment
    mean_attribution: float  # mean over runs of the run's strongest score for this name
    ci: ConfidenceInterval  # bootstrap interval over the per-run points
    culprit_rate: float  # n_culprit / n_present — "when it runs, how often is it to blame"
    poc_rate: float  # n_poc / n_present
    points: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ci"] = self.ci.to_dict()
        return d


@dataclass
class AggregateResult:
    """The systematic-blame verdict over a batch of runs of the same task."""

    label: str
    n_runs: int  # failing runs actually analyzed
    n_skipped: int  # runs skipped (passed, or not in failure mode)
    steps: List[StepAggregate] = field(default_factory=list)
    systematic_culprit: Optional[str] = None  # "kind:name" most consistently to blame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n_runs": self.n_runs,
            "n_skipped": self.n_skipped,
            "systematic_culprit": self.systematic_culprit,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_text(self) -> str:
        lines = [
            f"Aggregate attribution for '{self.label}': {self.n_runs} failing runs"
            + (f" ({self.n_skipped} skipped)" if self.n_skipped else ""),
        ]
        if self.systematic_culprit is not None:
            lines.append(f"  Systematic weak step: {self.systematic_culprit}")
        else:
            lines.append("  No step was a consistent culprit.")
        lines.append("  step                     culprit   mean attr [95% CI]     poc")
        for s in self.steps:
            lines.append(
                f"    {s.kind}:{s.name:<18} "
                f"{s.n_culprit:>2}/{s.n_present:<2} {s.culprit_rate:4.0%}  "
                f"{s.mean_attribution:6.2f} [{s.ci.low:5.2f},{s.ci.high:5.2f}]  "
                f"{s.n_poc:>2}/{s.n_present:<2}"
            )
        return "\n".join(lines)


def _culprit_key(result: AttributionResult, index: Optional[int]) -> Optional[Tuple[str, str]]:
    """Map a step index in one result to its (kind, name) identity."""
    if index is None:
        return None
    for s in result.steps:
        if s.index == index:
            return (s.kind, s.name)
    return None


def aggregate(
    results: Sequence[AttributionResult],
    *,
    label: str = "aggregate",
    bootstrap_seed: int = 17,
) -> AggregateResult:
    """Pool a batch of per-run attribution results into systematic step blame.

    Only failure-mode results are analyzed; credit-mode or otherwise non-failure
    results are counted as skipped. Within a single run, a step name that recurs
    contributes its **strongest** (max) attribution — the run's worst showing of
    that operation. Steps are ranked by how many runs named them the culprit, then
    by mean attribution; the top-ranked name (if any run blamed it) is the
    ``systematic_culprit``.
    """
    analyzed = [r for r in results if r.mode == "failure" and r.failed]
    n_skipped = len(results) - len(analyzed)

    # (kind, name) -> accumulators
    present: Dict[Tuple[str, str], int] = {}
    culprit: Dict[Tuple[str, str], int] = {}
    poc: Dict[Tuple[str, str], int] = {}
    points: Dict[Tuple[str, str], List[float]] = {}

    for r in analyzed:
        # Per run, collapse repeated names to their strongest attribution.
        per_run_best: Dict[Tuple[str, str], float] = {}
        for s in r.steps:
            key = (s.kind, s.name)
            if key not in per_run_best or s.attribution > per_run_best[key]:
                per_run_best[key] = s.attribution
        for key, best in per_run_best.items():
            present[key] = present.get(key, 0) + 1
            points.setdefault(key, []).append(best)

        ck = _culprit_key(r, r.culprit_index)
        if ck is not None:
            culprit[ck] = culprit.get(ck, 0) + 1
        pk = _culprit_key(r, r.point_of_commitment)
        if pk is not None:
            poc[pk] = poc.get(pk, 0) + 1

    steps: List[StepAggregate] = []
    for key, n_present in present.items():
        kind, name = key
        pts = points[key]
        pt, low, high = bootstrap_mean_interval(pts, seed=bootstrap_seed)
        n_cul = culprit.get(key, 0)
        n_poc = poc.get(key, 0)
        steps.append(
            StepAggregate(
                name=name,
                kind=kind,
                n_present=n_present,
                n_culprit=n_cul,
                n_poc=n_poc,
                mean_attribution=pt,
                ci=ConfidenceInterval(point=pt, low=low, high=high, method="bootstrap-runs"),
                culprit_rate=n_cul / n_present if n_present else 0.0,
                poc_rate=n_poc / n_present if n_present else 0.0,
                points=pts,
            )
        )

    # Rank: most-blamed first, then strongest mean attribution.
    steps.sort(key=lambda s: (s.n_culprit, s.mean_attribution), reverse=True)
    systematic = None
    if steps and steps[0].n_culprit > 0:
        top = steps[0]
        systematic = f"{top.kind}:{top.name}"

    return AggregateResult(
        label=label,
        n_runs=len(analyzed),
        n_skipped=n_skipped,
        steps=steps,
        systematic_culprit=systematic,
    )


def aggregate_runs(
    trajectories: Sequence[Trajectory],
    agent_fn: Callable[..., Any],
    verifier: Callable[[Any], float],
    *,
    label: str = "aggregate",
    fail_threshold: float = 0.5,
    bootstrap_seed: int = 17,
    **attribute_kwargs: Any,
) -> AggregateResult:
    """Attribute each *failing* trajectory and aggregate the batch.

    Passing runs (outcome ``>= fail_threshold``) are skipped and counted — failure
    attribution is undefined for them. Remaining keyword arguments (``rollouts``,
    ``method``, ``adaptive``, ``max_workers``, …) are forwarded to
    :func:`agent_replay.attribute`.
    """
    results: List[AttributionResult] = []
    skipped = 0
    for traj in trajectories:
        score = (
            traj.outcome_score if traj.outcome_score is not None else float(verifier(traj.result))
        )
        if score >= fail_threshold:
            skipped += 1
            continue
        results.append(
            attribute(
                traj,
                agent_fn,
                verifier,
                fail_threshold=fail_threshold,
                **attribute_kwargs,
            )
        )
    agg = aggregate(results, label=label, bootstrap_seed=bootstrap_seed)
    agg.n_skipped += skipped
    return agg
