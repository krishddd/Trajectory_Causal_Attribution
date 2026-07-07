"""OpenAI Python SDK adapter.

Wrap an ``openai.OpenAI`` client so that every ``chat.completions.create`` call
is routed through an :class:`~agent_replay.recorder.AgentContext`, recording the
request as an ``llm`` step (and, on replay, serving the recorded completion or
resampling it). The wrapper is duck-typed: it works with the real SDK client or
any object exposing ``chat.completions.create``.

Usage
-----
    from openai import OpenAI
    from agent_replay.adapters.openai_sdk import wrap_openai

    def agent(ctx, prompt):
        client = wrap_openai(OpenAI(), ctx, name="draft")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"]

During recording the real API is called and the response captured (reduced to a
JSON-serialisable dict); during replay the captured response is returned without
touching the network — the VCR/cassette pattern from the architecture document.
"""

from __future__ import annotations

from typing import Any, Callable


def _response_to_dict(resp: Any) -> Any:
    """Best-effort conversion of an SDK response object to a plain dict."""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    if isinstance(resp, (dict, list, str, int, float, bool)) or resp is None:
        return resp
    return {"repr": repr(resp)}


class _Completions:
    def __init__(self, real_create: Callable[..., Any], ctx: Any, name: str) -> None:
        self._real_create = real_create
        self._ctx = ctx
        self._name = name

    def create(self, **kwargs: Any) -> Any:
        def produce() -> Any:
            return _response_to_dict(self._real_create(**kwargs))

        # Record the request payload (minus anything unserialisable) as inputs.
        inputs = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature"),
        }
        return self._ctx.llm(self._name, produce=produce, **inputs)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class WrappedOpenAI:
    """A minimal ``client.chat.completions.create`` shim over the real client."""

    def __init__(self, client: Any, ctx: Any, name: str = "openai") -> None:
        real_create = client.chat.completions.create
        self.chat = _Chat(_Completions(real_create, ctx, name))


def wrap_openai(client: Any, ctx: Any, name: str = "openai") -> WrappedOpenAI:
    """Return a recording/replaying wrapper around an OpenAI client."""
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise TypeError(
            "wrap_openai expects an object exposing chat.completions.create (e.g. openai.OpenAI())"
        )
    return WrappedOpenAI(client, ctx, name)
