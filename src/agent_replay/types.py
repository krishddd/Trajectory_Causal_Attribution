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
    """A single recorded operation in a trajectory (one node of the SCM)."""

    index: int
    kind: StepKind
    name: str
    inputs: Dict[str, Any]
    output: Any
    parent_hash: str = ""
    step_hash: str = ""

    def compute_hashes(self, parent_hash: str) -> None:
        """Populate the Merkle-style ``parent_hash`` / ``step_hash`` fields.

        Each step hash chains off the previous step's hash together with the
        action and its output, giving the trajectory a tamper-evident,
        content-addressable identity (the Merkle-DAG idea from the architecture
        document, reduced to the essentials).
        """
        self.parent_hash = parent_hash
        self.step_hash = link_hash(
            parent_hash,
            content_hash({"kind": self.kind.value, "name": self.name, "inputs": self.inputs}),
            content_hash(self.output),
        )

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
            "repair": self.repair.to_dict() if self.repair else None,
            "meta": self.meta,
        }

    def to_json(self, path: str) -> str:
        """Write the JSON report and return the path."""
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    def to_html(self, path: str) -> str:
        """Write the standalone HTML report and return the path."""
        from .report import render_html

        html = render_html(self)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path
