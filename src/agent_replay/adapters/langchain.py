"""LangChain adapter: a callback handler that records LLM and tool steps.

Attach :class:`AgentReplayCallbackHandler` to any LangChain runnable and each
LLM/tool invocation is captured as a step on an
:class:`~agent_replay.recorder.AgentContext`. The handler imports LangChain
lazily, so the core library never depends on it.

Usage
-----
    from agent_replay.adapters.langchain import AgentReplayCallbackHandler

    def agent(ctx, prompt):
        handler = AgentReplayCallbackHandler(ctx)
        return chain.invoke(prompt, config={"callbacks": [handler]})

Because LangChain drives the calls itself, the handler records the *observed*
outputs directly (there is no separate ``produce`` policy to re-run); this makes
it a faithful recorder. Full counterfactual resampling of LangChain steps
requires wrapping the model call with a ``produce`` closure via the native
``ctx.llm`` API instead — see the README.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _base_handler_cls() -> type:
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
    except Exception as exc:  # pragma: no cover - only hit without langchain
        raise ImportError(
            "The LangChain adapter requires 'langchain-core'. "
            "Install with: pip install 'agent-replay[langchain]'"
        ) from exc
    return BaseCallbackHandler


class AgentReplayCallbackHandler:
    """A LangChain callback handler recording steps onto an AgentContext.

    Implemented as a plain object that mixes in LangChain's ``BaseCallbackHandler``
    at construction time, so the module imports cleanly even when LangChain is
    not installed.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "AgentReplayCallbackHandler":
        base = _base_handler_cls()
        # Dynamically create a subclass that also inherits from LangChain's base.
        dynamic = type("AgentReplayCallbackHandlerImpl", (cls, base), {})
        instance = object.__new__(dynamic)
        return instance

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._pending: Dict[str, Dict[str, Any]] = {}

    # -- LLM callbacks --------------------------------------------------------

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", len(self._pending)))
        self._pending[run_id] = {"kind": "llm", "prompts": prompts}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        info = self._pending.pop(run_id, {"prompts": None})
        text = _extract_llm_text(response)
        self._ctx.llm("langchain_llm", produce=lambda: text, prompts=info.get("prompts"))

    # -- Tool callbacks -------------------------------------------------------

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", len(self._pending)))
        name = (serialized or {}).get("name", "tool")
        self._pending[run_id] = {"kind": "tool", "name": name, "input": input_str}

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        info = self._pending.pop(run_id, {"name": "tool", "input": None})
        out = output if isinstance(output, (str, int, float, bool, dict, list)) else str(output)
        self._ctx.tool(info.get("name", "tool"), produce=lambda: out, input=info.get("input"))


def _extract_llm_text(response: Any) -> str:
    """Pull the first generated text out of a LangChain ``LLMResult``."""
    try:
        generations = response.generations
        first = generations[0][0]
        return getattr(first, "text", None) or getattr(first.message, "content", str(first))
    except Exception:
        return str(response)
