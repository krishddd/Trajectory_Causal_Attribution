"""Explainability: turn causal numbers into a traceable narrative.

The estimators are unchanged - this module only *interprets* an
:class:`~agent_replay.types.AttributionResult`. It answers, in plain language and
with the numbers that back each claim:

* **What** went wrong (the outcome and the decisive action),
* **Where** it went wrong (which step, its recorded action),
* **Why** that step is to blame (the counterfactual evidence and the
  point-of-commitment reasoning), and
* **How to fix** it (the minimal repair, when available),

plus a step-by-step **causal trace** that labels every step's role - benign,
contributing, decisive, or locked-in - so the failure is fully traceable from the
first action to the point of no return.

In *credit* mode (a passing run analysed with ``on_success="credit"``) the same
structure explains which step secured success instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .types import AttributionResult, StepAttribution, Trajectory

# Roles a step can play in the causal story.
ROLE_DECISIVE = "decisive"
ROLE_LOCKED_IN = "locked-in"
ROLE_CONTRIBUTING = "contributing"
ROLE_OBSERVED = "observed-only"
ROLE_BENIGN = "benign"


@dataclass
class StepTrace:
    """One step's entry in the causal trace, with the evidence behind its role."""

    index: int
    kind: str
    name: str
    role: str
    action: Any
    p_fail_kept: float
    p_fail_ablated: float
    rescue_rate: float
    ci_low: float
    ci_high: float
    shapley: Optional[float]
    note: str
    # Set only when the step split its action from its observation (deck slide 9);
    # None means action == observation (the common case).
    observation: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Explanation:
    """A structured, human-readable account of an attribution result."""

    session_id: str
    mode: str
    headline: str
    what: str
    where: str
    why: str
    fix: str
    confidence: str
    trace: List[StepTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trace"] = [t.to_dict() for t in self.trace]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Explanation":
        return cls(
            session_id=d["session_id"],
            mode=d.get("mode", "failure"),
            headline=d["headline"],
            what=d["what"],
            where=d["where"],
            why=d["why"],
            fix=d["fix"],
            confidence=d["confidence"],
            trace=[StepTrace(**t) for t in d.get("trace", [])],
        )

    def to_text(self) -> str:
        lines = [
            self.headline,
            "",
            f"WHAT:  {self.what}",
            f"WHERE: {self.where}",
            f"WHY:   {self.why}",
            f"FIX:   {self.fix}",
            "",
            f"Confidence: {self.confidence}",
            "",
            "Causal trace (first action -> point of no return):",
        ]
        for t in self.trace:
            marker = {
                ROLE_DECISIVE: ">>",
                ROLE_LOCKED_IN: " x",
                ROLE_CONTRIBUTING: " +",
                ROLE_OBSERVED: " ?",
                ROLE_BENIGN: "  ",
            }.get(t.role, "  ")
            split = ""
            if t.observation is not None:
                split = f" [action {t.action!r} -> observation {t.observation!r}]"
            lines.append(
                f"  {marker} step {t.index} [{t.kind}:{t.name}] {t.role} - {t.note}{split}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"### {self.headline}",
            "",
            f"- **What:** {self.what}",
            f"- **Where:** {self.where}",
            f"- **Why:** {self.why}",
            f"- **Fix:** {self.fix}",
            f"- **Confidence:** {self.confidence}",
            "",
            "| Step | Role | Evidence |",
            "|---|---|---|",
        ]
        for t in self.trace:
            note = t.note
            if t.observation is not None:
                note += f" (action `{t.action!r}` → observation `{t.observation!r}`)"
            lines.append(f"| {t.index} `{t.kind}:{t.name}` | {t.role} | {note} |")
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _step_of(trajectory: Optional["Trajectory"], index: Optional[int]):
    if trajectory is None or index is None or index >= len(trajectory.steps):
        return None
    return trajectory.steps[index]


def _action_of(trajectory: Optional["Trajectory"], index: Optional[int]) -> Any:
    """The recorded *action* of a step (its explicit action if split, else output)."""
    step = _step_of(trajectory, index)
    return None if step is None else step.action_value


def _observation_of(trajectory: Optional["Trajectory"], index: Optional[int]) -> Any:
    """The recorded observation, but only when it differs from the action."""
    step = _step_of(trajectory, index)
    if step is None or step.action is None:
        return None  # not split: action == observation
    return step.output


def _role(s: "StepAttribution", culprit_index: Optional[int], poc: Optional[int]) -> str:
    if not s.resamplable:
        return ROLE_OBSERVED
    if culprit_index is not None and s.index == culprit_index:
        return ROLE_DECISIVE
    if poc is not None and s.index > poc:
        return ROLE_LOCKED_IN
    if s.ci.excludes_zero() and s.attribution > 0:
        return ROLE_CONTRIBUTING
    return ROLE_BENIGN


def _note(s: "StepAttribution", role: str, mode: str) -> str:
    rescue = s.attribution  # failure mode: rescue rate; credit mode: risk introduced
    if role == ROLE_OBSERVED:
        return "observation-only step; cannot be re-decided, so not attributable."
    if role == ROLE_DECISIVE:
        if mode == "credit":
            return (
                f"re-deciding this step introduces failure {_fmt(s.p_fail_ablated)} of the "
                f"time - the latest step that still secures the outcome."
            )
        return (
            f"re-deciding this step rescues the run {_fmt(rescue)} of the time "
            f"(P(fail) drops {_fmt(s.p_fail_kept)}->{_fmt(s.p_fail_ablated)}); "
            f"it is the latest step where re-deciding still changes the outcome."
        )
    if role == ROLE_LOCKED_IN:
        return (
            f"outcome already committed here (P(fail|re-decide)={_fmt(s.p_fail_ablated)}); "
            f"re-deciding no longer helps."
        )
    if role == ROLE_CONTRIBUTING:
        return (
            f"re-deciding here also shifts the outcome ({_fmt(rescue)}), but the blame "
            f"resolves to a later step (butterfly effect)."
        )
    return "no significant causal effect on the outcome."


def explain(result: "AttributionResult", trajectory: Optional["Trajectory"] = None) -> Explanation:
    """Build a traceable :class:`Explanation` from an attribution result.

    Passing the recorded ``trajectory`` lets the explanation name the actual
    decisive *action* (its recorded output), not just the step position.
    """
    mode = result.mode
    steps = sorted(result.steps, key=lambda s: s.index)
    poc = result.point_of_commitment
    culprit = result.culprit

    trace = [
        StepTrace(
            index=s.index,
            kind=s.kind,
            name=s.name,
            role=(role := _role(s, result.culprit_index, poc)),
            action=_action_of(trajectory, s.index),
            p_fail_kept=s.p_fail_kept,
            p_fail_ablated=s.p_fail_ablated,
            rescue_rate=s.attribution,
            ci_low=s.ci.low,
            ci_high=s.ci.high,
            shapley=s.shapley,
            note=_note(s, role, mode),
            observation=_observation_of(trajectory, s.index),
        )
        for s in steps
    ]

    outcome = "failed" if result.failed else "passed"
    n_locked = sum(1 for t in trace if t.role == ROLE_LOCKED_IN)

    if culprit is None:
        headline = (
            f"No single step is causally responsible for the {outcome} run '{result.session_id}'."
        )
        what = (
            f"The run {outcome} (verifier score {_fmt(result.outcome_score)}), but no step's "
            f"counterfactual effect was statistically significant."
        )
        where = "No decisive step localised."
        why = (
            "Ablating each step in turn did not shift the failure distribution enough for its "
            "confidence interval to exclude zero - the failure may be diffuse, interaction-"
            "driven, or need more rollouts."
        )
        fix = "Increase rollouts, try method='shapley' for interactions, or inspect the trace."
        confidence = f"{result.rollouts} rollouts/step; no interval excluded zero."
        return Explanation(
            result.session_id, mode, headline, what, where, why, fix, confidence, trace
        )

    action = _action_of(trajectory, culprit.index)
    observation = _observation_of(trajectory, culprit.index)
    if action is None:
        action_txt = ""
    elif observation is not None:
        action_txt = f" Its recorded action was {action!r} (observation {observation!r})."
    else:
        action_txt = f" Its recorded action was {action!r}."
    score = culprit.shapley if culprit.shapley is not None else culprit.attribution

    if mode == "credit":
        headline = (
            f"Success of run '{result.session_id}' was most secured by step {culprit.index} "
            f"({culprit.kind}:{culprit.name})."
        )
        what = (
            f"The run passed (score {_fmt(result.outcome_score)}). Step {culprit.index} is the "
            f"latest decision whose re-rolling would most reintroduce failure.{action_txt}"
        )
        where = f"Step {culprit.index} - {culprit.kind}:{culprit.name}."
        why = (
            f"Holding step {culprit.index} at its recorded action keeps the run passing, but "
            f"re-deciding it (and everything after) fails {_fmt(culprit.p_fail_ablated)} of the "
            f"time - the largest such risk among all steps."
        )
        fix = "This step is load-bearing for success; guard or pin its behaviour."
    else:
        headline = (
            f"Failure of run '{result.session_id}' is attributed to step {culprit.index} "
            f"({culprit.kind}:{culprit.name})."
        )
        what = (
            f"The run failed (verifier score {_fmt(result.outcome_score)}). The decisive error "
            f"is step {culprit.index}.{action_txt}"
        )
        where = f"Step {culprit.index} - {culprit.kind}:{culprit.name}."
        why = (
            f"Keeping step {culprit.index}'s recorded action fails "
            f"{_fmt(culprit.p_fail_kept)} of the time; re-deciding it drops failure to "
            f"{_fmt(culprit.p_fail_ablated)} - a rescue of {_fmt(culprit.attribution)}. "
            f"It is the latest step where re-deciding still changes the outcome "
            f"(the point of commitment); the {n_locked} step(s) after it stay failing "
            f"regardless, so the run is locked into failure beyond here."
        )
        if culprit.shapley is not None:
            why += f" Its Shapley share of total blame is {_fmt(culprit.shapley)}."
        if result.repair is not None and result.repair.valid:
            r = result.repair
            fix = (
                f"Constrain step {r.step_index} from {r.original_action!r} toward "
                f"{r.repaired_action!r}: this validated minimal repair flips the outcome "
                f"(P(fail)->{_fmt(r.p_fail_after)}, minimality {_fmt(r.minimality)})."
            )
        else:
            fix = (
                f"Target step {culprit.index}: change its prompt/tool call so it no longer "
                f"produces {action!r}."
                if action is not None
                else f"Target step {culprit.index}'s decision."
            )

    ci_w = culprit.ci.high - culprit.ci.low
    confidence = (
        f"{result.rollouts} rollouts/step; culprit 95% CI "
        f"[{_fmt(culprit.ci.low)}, {_fmt(culprit.ci.high)}] (width {_fmt(ci_w)}), "
        f"score {_fmt(score)}."
    )

    return Explanation(result.session_id, mode, headline, what, where, why, fix, confidence, trace)
