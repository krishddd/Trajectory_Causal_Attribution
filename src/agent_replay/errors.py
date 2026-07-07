"""Exception types for agent-replay."""

from __future__ import annotations


class AgentReplayError(Exception):
    """Base class for all agent-replay errors."""


class NonSerializableStepError(AgentReplayError):
    """A recorded step input/output is not JSON-serialisable.

    Recorded payloads must round-trip through the SQLite store as JSON. Storing a
    non-serialisable object would silently degrade it to a string via a ``repr``
    fallback, so on replay a *held* step would serve a string where the live
    agent produced an object. Recording fails fast instead (strict mode).
    """


class SuccessfulRunError(AgentReplayError):
    """Attribution was requested on a run that did not fail.

    Failure attribution is only meaningful for a failing trajectory. Pass
    ``on_success="credit"`` to :func:`agent_replay.attribute` to run the
    symmetric *credit* analysis (which step most secured the success) instead.
    """
