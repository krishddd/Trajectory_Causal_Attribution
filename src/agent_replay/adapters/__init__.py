"""Optional framework adapters for agent-replay.

These are thin, dependency-guarded bridges so agent-replay can capture steps from
popular stacks without those stacks being required to install the core library:

- :mod:`agent_replay.adapters.langchain` - a LangChain callback handler.
- :mod:`agent_replay.adapters.openai_sdk` - an OpenAI Python SDK client wrapper.

Both import their target framework lazily and raise a clear error if it is
absent, so ``import agent_replay`` never fails on a machine without them.
"""

from __future__ import annotations

__all__ = ["langchain", "openai_sdk"]
