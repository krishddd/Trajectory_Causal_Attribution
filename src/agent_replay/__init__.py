"""agent-replay: counterfactual step-ablation attribution for AI agents.

Record an agent's trajectory (LLM / tool / memory operations) as a checkpointed,
replayable session, then discover *which step caused a failure* by re-running the
trajectory with individual steps ablated and measuring the shift in failure
probability:

    attribution(step i) = P(fail | step i kept) - P(fail | step i ablated)

Public API
----------
- :class:`Session`           - record and store agent runs (SQLite-backed).
- :func:`record`             - low-level one-shot recording.
- :func:`attribute`          - run the counterfactual attribution pipeline.
- :class:`CheckpointStore`   - the SQLite checkpoint/CAS store.
- :class:`Trajectory`, :class:`Step`, :class:`AttributionResult` - core types.
"""

from __future__ import annotations

from . import instrument
from .ablation import AblationEngine
from .attribution import attribute
from .errors import AgentReplayError, NonSerializableStepError, SuccessfulRunError
from .explain import Explanation, explain
from .pytest_plugin import assert_agent_passes, measure_failure_rate
from .recorder import AgentContext, AsyncAgentContext, arecord, record
from .repair import export_contrastive_pairs, find_minimal_repair
from .replayer import AsyncReplayContext, ReplayContext, ReplayPlan, areplay, replay
from .session import Session
from .store import CheckpointStore
from .types import (
    AttributionResult,
    ConfidenceInterval,
    InterventionKind,
    Repair,
    Step,
    StepAttribution,
    StepKind,
    Trajectory,
)

__version__ = "0.4.0"

__all__ = [
    "Session",
    "record",
    "attribute",
    "explain",
    "Explanation",
    "instrument",
    "assert_agent_passes",
    "measure_failure_rate",
    "replay",
    "arecord",
    "areplay",
    "ReplayPlan",
    "ReplayContext",
    "AsyncReplayContext",
    "AgentContext",
    "AsyncAgentContext",
    "AblationEngine",
    "CheckpointStore",
    "AgentReplayError",
    "NonSerializableStepError",
    "SuccessfulRunError",
    "find_minimal_repair",
    "export_contrastive_pairs",
    "Trajectory",
    "Step",
    "StepKind",
    "StepAttribution",
    "AttributionResult",
    "ConfidenceInterval",
    "InterventionKind",
    "Repair",
    "__version__",
]
