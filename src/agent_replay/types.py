"""Core data types for agent-replay.

An agent execution is formalised (following the *Trajectory Causal Attribution*
paper) as a Structural Causal Model: a sequence of steps, each of which reads a
context, draws an action from a stochastic policy, and observes a result. These
dataclasses are the serialisable backbone that the recorder, store, replayer,
ablation engine and attribution scorer all share.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .hashing import content_hash, link_hash


class StepKind(str, Enum):
    """The three intercepted operation classes an agent performs."""

    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"


@dataclass
class Step:
    """A single recorded operation in a trajectory (one node of the SCM).

    ``resamplable`` records whether the step carries a genuine stochastic policy
    that can be re-run counterfactually. Steps captured by observation-only
    adapters (e.g. a LangChain callback that only sees the produced text) or with
    no ``produce`` policy are marked ``resamplable=False``: they cannot be truly
    ablated, so the replayer always serves their recorded output and the scorer
    surfaces them as non-attributable rather than silently reporting a spurious
    zero.
    """

    index: int
    kind: StepKind
    name: str
    inputs: Dict[str, Any]
    output: Any
    resamplable: bool = True
    parent_hash: str = ""
    step_hash: str = ""

    def op_key(self) -> str:
        """Content-addressable identity of the *operation* (kind + name + inputs).

        This is the idempotency key used to match a live replay call to its
        recorded counterpart, independent of call position — so that when an
        upstream ablation changes the control flow, a held step is only served
        from the cassette when the *same* operation actually recurs.
        """
        return content_hash({"kind": self.kind.value, "name": self.name, "inputs": self.inputs})

    # The Merkle node in the research (deck slide 7) is parent + action hash +
    # output hash. ``op_key`` is the action hash; ``output_hash`` is its counterpart,
    # letting callers ask "same decision, different observation?" and diff cheaply.
    def action_hash(self) -> str:
        """Alias for :meth:`op_key` — the content hash of the action (kind+name+inputs)."""
        return self.op_key()

    def output_hash(self) -> str:
        """Content hash of the recorded output (the observation)."""
        return content_hash(self.output)

    def compute_hashes(self, parent_hash: str) -> None:
        """Populate the Merkle-style ``parent_hash`` / ``step_hash`` fields.

        Each step hash chains off the previous step's hash together with the
        action and its output, giving the trajectory a tamper-evident,
        content-addressable identity (the Merkle-DAG idea from the architecture
        document, reduced to the essentials).
        """
        self.parent_hash = parent_hash
        self.step_hash = link_hash(parent_hash, self.action_hash(), self.output_hash())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step":
        return cls(
            index=d["index"],
            kind=StepKind(d["kind"]),
            name=d["name"],
            inputs=d["inputs"],
            output=d["output"],
            resamplable=d.get("resamplable", True),
            parent_hash=d.get("parent_hash", ""),
            step_hash=d.get("step_hash", ""),
        )


@dataclass
class Trajectory:
    """A full recorded agent run: the factual sequence used as the cassette."""

    session_id: str
    task: Dict[str, Any]
    steps: List[Step] = field(default_factory=list)
    seed: int = 0
    outcome_score: Optional[float] = None
    result: Any = None
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def root_hash(self) -> str:
        """The final step hash, which transitively commits the whole run."""
        return self.steps[-1].step_hash if self.steps else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "seed": self.seed,
            "outcome_score": self.outcome_score,
            "result": self.result,
            "created_at": self.created_at,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(
            session_id=d["session_id"],
            task=d["task"],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            seed=d.get("seed", 0),
            outcome_score=d.get("outcome_score"),
            result=d.get("result"),
            created_at=d.get("created_at", 0.0),
            meta=d.get("meta", {}),
        )


class InterventionKind(str, Enum):
    """The intervention algebra from the paper (do-calculus on one step).

    Only a subset is needed for step-ablation attribution; the remainder are
    exposed for repair and advanced use.
    """

    RESAMPLE = "resample"  # re-draw the action from the same policy (null intervention)
    DO = "do"  # force a specific action, overriding the policy
    MOCK_OBSERVE = "mock_observe"  # replace the observed result
    REMOVE = "remove"  # drop the step entirely (empty action)


@dataclass
class ConfidenceInterval:
    """A two-sided interval with the point estimate it brackets."""

    point: float
    low: float
    high: float
    method: str = ""

    def excludes_zero(self) -> bool:
        """True when the whole interval lies strictly on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepAttribution:
    """Per-step causal result: the contrastive score and its uncertainty."""

    index: int
    name: str
    kind: str
    p_fail_kept: float
    p_fail_ablated: float
    attribution: float
    ci: ConfidenceInterval
    shapley: Optional[float] = None
    shapley_ci: Optional[ConfidenceInterval] = None
    resamplable: bool = True
    screened: bool = False  # True when a pre-filter skipped causal evaluation

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ci"] = self.ci.to_dict()
        d["shapley_ci"] = self.shapley_ci.to_dict() if self.shapley_ci else None
        return d


@dataclass
class Repair:
    """A validated minimal counterfactual repair for the culprit step."""

    step_index: int
    original_action: Any
    repaired_action: Any
    p_fail_after: float
    minimality: float
    valid: bool
    step_name: str = ""
    step_kind: str = ""
    p_fail_before: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_guard(self) -> str:
        """Emit a deploy-time guard snippet that applies this repair at runtime.

        The snippet is a ready-to-paste hint: when the culprit step reproduces its
        failing action, substitute the validated minimal repair. It is advisory
        (the developer adapts it to their framework), turning the causal finding
        into a concrete recovery action.
        """
        label = f"{self.step_kind}:{self.step_name}".strip(":") or f"step {self.step_index}"
        return (
            f"# agent-replay guard for {label} "
            f"(validated: P(fail) {self.p_fail_before:.2f} -> {self.p_fail_after:.2f}, "
            f"minimality {self.minimality:.2f})\n"
            f"if step.name == {self.step_name!r} and step.output == {self.original_action!r}:\n"
            f"    step.output = {self.repaired_action!r}  # minimal counterfactual repair"
        )


@dataclass
class AttributionResult:
    """The full diagnostic artifact produced by :func:`agent_replay.attribute`."""

    session_id: str
    total_steps: int
    outcome_score: float
    failed: bool
    method: str
    rollouts: int
    steps: List[StepAttribution]
    point_of_commitment: Optional[int]
    culprit_index: Optional[int]
    mode: str = "failure"  # "failure" (attribute blame) or "credit" (attribute rescue)
    repair: Optional[Repair] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def culprit(self) -> Optional[StepAttribution]:
        if self.culprit_index is None:
            return None
        for s in self.steps:
            if s.index == self.culprit_index:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_steps": self.total_steps,
            "outcome_score": self.outcome_score,
            "failed": self.failed,
            "method": self.method,
            "rollouts": self.rollouts,
            "steps": [s.to_dict() for s in self.steps],
            "point_of_commitment": self.point_of_commitment,
            "culprit_index": self.culprit_index,
            "mode": self.mode,
            "repair": self.repair.to_dict() if self.repair else None,
            "meta": self.meta,
        }

    def explain(self, trajectory: Any = None) -> Any:
        """Return a traceable :class:`~agent_replay.explain.Explanation` of this result.

        Passing the recorded ``trajectory`` lets the explanation name the actual
        decisive action, not just the step position.
        """
        from .explain import explain as _explain

        return _explain(self, trajectory)

    def to_json(self, path: str, explanation: Any = None) -> str:
        """Write the JSON report (optionally embedding an explanation) and return the path."""
        import json

        data = self.to_dict()
        if explanation is not None:
            data["explanation"] = explanation.to_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def to_html(self, path: str, explanation: Any = None) -> str:
        """Write the standalone HTML report (optionally with a narrative panel)."""
        from .report import render_html

        html = render_html(self, explanation=explanation)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path
