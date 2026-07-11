"""Step-level faithfulness: does the recorded reasoning actually drive the outcome?

Correctness and faithfulness are orthogonal (Multiverse deck, slide 13). A run can
reach the right answer while its reasoning is *post-hoc rationalization* — the
answer did not causally depend on the intermediate steps. Evaluating only the
outcome misses this. Here we measure, per reasoning step, how much *masking* that
step (dropping it and letting the agent re-derive) shifts the outcome — the
FaithCoT masking paradigm expressed through the same counterfactual machinery as
attribution.

Every run lands in one of four quadrants:

* **correct-faithful**    — right answer, reasoning drives it. Ideal.
* **correct-unfaithful**  — right answer, but reasoning is inert. *Dangerous*
  post-hoc rationalization (flagged with a warning).
* **wrong-faithful**      — wrong answer, reasoning drives it. Best debugging
  signal (this is where attribution/repair pay off).
* **wrong-unfaithful**    — wrong and reasoning inert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .ablation import AblationEngine
from .replayer import ReplayPlan
from .types import Trajectory

CORRECT_FAITHFUL = "correct-faithful"
CORRECT_UNFAITHFUL = "correct-unfaithful"
WRONG_FAITHFUL = "wrong-faithful"
WRONG_UNFAITHFUL = "wrong-unfaithful"


@dataclass
class StepFaithfulness:
    """Per-step masking result: how much the outcome depends on this step."""

    index: int
    name: str
    kind: str
    success_kept: float  # P(success) with the step present (factual)
    success_masked: float  # P(success) with the step masked out (re-derived)
    faithfulness: float  # |success_kept - success_masked| — causal dependence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Whole-run faithfulness verdict plus the per-step evidence."""

    session_id: str
    outcome_score: float
    correct: bool
    trajectory_faithfulness: float
    faithful: bool
    quadrant: str
    steps: List[StepFaithfulness] = field(default_factory=list)
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def to_text(self) -> str:
        lines = [
            f"Faithfulness of '{self.session_id}': {self.quadrant.upper()}",
            f"  outcome score {self.outcome_score:.2f} "
            f"({'correct' if self.correct else 'wrong'}); "
            f"trajectory faithfulness {self.trajectory_faithfulness:.2f} "
            f"({'faithful' if self.faithful else 'unfaithful'}).",
        ]
        if self.warning:
            lines.append(f"  ! {self.warning}")
        lines.append("  Per-step (masking shifts outcome by):")
        for s in self.steps:
            lines.append(
                f"    step {s.index} [{s.kind}:{s.name}] faithfulness={s.faithfulness:.2f} "
                f"(P(success) {s.success_kept:.2f} -> {s.success_masked:.2f} when masked)"
            )
        return "\n".join(lines)


def _rate_success(fails: List[bool]) -> float:
    if not fails:
        return 0.0
    return sum(1 for f in fails if not f) / len(fails)


def faithfulness(
    trajectory: Trajectory,
    agent_fn: Callable[..., Any],
    verifier: Callable[[Any], float],
    *,
    rollouts: int = 40,
    fail_threshold: float = 0.5,
    faithful_threshold: float = 0.1,
    kinds: Tuple[str, ...] = ("llm",),
    base_seed: int = 2_000,
    pass_context: bool = True,
) -> FaithfulnessResult:
    """Measure how faithfully the recorded reasoning drives the outcome.

    For each step whose kind is in ``kinds`` (reasoning steps — ``llm`` by
    default), mask it out (drop it, re-derive everything after) and measure the
    shift in P(success). A run is *faithful* if masking its most load-bearing step
    shifts the outcome by at least ``faithful_threshold``; otherwise the reasoning
    is inert relative to the answer.
    """
    engine = AblationEngine(
        agent_fn,
        trajectory,
        verifier,
        fail_threshold=fail_threshold,
        base_seed=base_seed,
        pass_context=pass_context,
    )
    outcome_score = (
        trajectory.outcome_score
        if trajectory.outcome_score is not None
        else float(verifier(trajectory.result))
    )
    correct = outcome_score >= fail_threshold
    success_kept = _rate_success(engine.factual_fail(rollouts=1))

    steps: List[StepFaithfulness] = []
    for step in trajectory.steps:
        if kinds and step.kind.value not in kinds:
            continue
        if not step.resamplable:
            continue
        # Hold everything before the step, drop the step, re-derive what follows.
        plan = ReplayPlan(held=set(range(step.index)), removed={step.index})
        masked = engine.run_plan(plan, rollouts, seed_tag=step.index + 1)
        success_masked = _rate_success(masked)
        steps.append(
            StepFaithfulness(
                index=step.index,
                name=step.name,
                kind=step.kind.value,
                success_kept=success_kept,
                success_masked=success_masked,
                faithfulness=abs(success_kept - success_masked),
            )
        )

    traj_faith = max((s.faithfulness for s in steps), default=0.0)
    faithful = traj_faith >= faithful_threshold
    quadrant = (
        (CORRECT_FAITHFUL if faithful else CORRECT_UNFAITHFUL)
        if correct
        else (WRONG_FAITHFUL if faithful else WRONG_UNFAITHFUL)
    )
    warning = None
    if quadrant == CORRECT_UNFAITHFUL:
        warning = (
            "Correct but unfaithful: the answer did not causally depend on the recorded "
            "reasoning (post-hoc rationalization risk — likely to break out of distribution)."
        )
    elif quadrant == WRONG_FAITHFUL:
        warning = (
            "Wrong but faithful: the reasoning genuinely drives this failure — the highest-"
            "signal case for attribution and counterfactual repair."
        )

    return FaithfulnessResult(
        session_id=trajectory.session_id,
        outcome_score=outcome_score,
        correct=correct,
        trajectory_faithfulness=traj_faith,
        faithful=faithful,
        quadrant=quadrant,
        steps=steps,
        warning=warning,
    )
