"""Ambient (implicit) agent context via ``contextvars``.

Threading an explicit ``ctx`` handle through an agent is the most transparent
API, but many frameworks (LangChain, CrewAI, AutoGen, …) own the call stack and
give you no place to pass one. The ambient context solves that: the recorder and
replayer publish the active :class:`~agent_replay.recorder.AgentContext` here for
the duration of a run, and instrumented callables (see
:mod:`agent_replay.instrument`) pick it up implicitly. It is a plain
``contextvars.ContextVar`` so it is correct across threads and ``asyncio`` tasks.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

_current: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "agent_replay_current_ctx", default=None
)


def current_context() -> Optional[Any]:
    """Return the active AgentContext, or ``None`` if not inside a run."""
    return _current.get()


def bind_context(ctx: Any) -> "contextvars.Token[Any]":
    """Publish ``ctx`` as the ambient context; returns a token for :func:`unbind`."""
    return _current.set(ctx)


def unbind_context(token: "contextvars.Token[Any]") -> None:
    """Restore the ambient context to its previous value."""
    _current.reset(token)
