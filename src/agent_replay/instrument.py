"""Universal instrumentation: connect *any* framework to agent-replay.

Rather than a bespoke integration per framework, agent-replay exposes three small
primitives that everything else is built from:

1. :func:`current_context` — the ambient :class:`~agent_replay.recorder.AgentContext`
   published by :func:`agent_replay.record` / :func:`agent_replay.replay`.
2. :func:`record_step` (and the :func:`llm` / :func:`tool` / :func:`memory`
   shorthands) — a decorator that turns any function into a recorded step.
3. :func:`patch` / :func:`install` — monkeypatch a dotted callable (an SDK method)
   so *unmodified* code records automatically, driven by a data-only
   ``RECIPES`` registry.

Because the mechanism is generic, adding a new framework is just adding an entry
to ``RECIPES`` (or calling :func:`patch` yourself) — no new code paths. When no
run is active, every wrapper is a transparent pass-through, so instrumentation is
safe to leave installed in production.

Recording vs. resampling
-------------------------
A wrapped callable records a step whose ``produce`` policy *re-invokes the real
callable*. On replay, held steps are served from the cassette and only ablated
steps actually re-run — so counterfactual resampling calls the real model/tool
again (set ``resamplable=False`` for deterministic or side-effectful calls that
must not be re-executed).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._ambient import current_context
from .recorder import record  # re-exported convenience
from .types import StepKind

__all__ = [
    "current_context",
    "record_step",
    "llm",
    "tool",
    "memory",
    "wrap",
    "patch",
    "unpatch",
    "install",
    "uninstall",
    "installed",
    "available_frameworks",
    "record_agent",
    "enable_virtual_time",
    "disable_virtual_time",
    "virtual_time",
    "RECIPES",
]


CaptureFn = Callable[[tuple, dict], Dict[str, Any]]


# Keyword names reserved by the context ops; captured inputs colliding with these
# are renamed so ``op(name, produce=..., resamplable=..., **inputs)`` never clashes.
_RESERVED_INPUT_KEYS = frozenset({"name", "produce", "resamplable"})


def _default_capture(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Best-effort JSON-friendly snapshot of a call's arguments (for the cassette key).

    Positional args are captured as ``args``; keyword args are captured by name.
    Non-JSON-native values are reduced to a **stable** ``<TypeName>`` token rather
    than ``repr`` — ``repr`` embeds the object's memory address, which would make
    the idempotency key non-deterministic across processes and silently break
    replay of any patched instance method (where ``self`` is a captured arg).
    """

    def safe(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        return f"<{type(v).__name__}>"

    captured: Dict[str, Any] = {}
    if args:
        captured["args"] = [safe(a) for a in args]
    for k, v in kwargs.items():
        captured[str(k)] = safe(v)
    return captured


def _sanitize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Rename any input keys that would collide with the context op parameters."""
    if not any(k in _RESERVED_INPUT_KEYS for k in inputs):
        return inputs
    return {(f"{k}_" if k in _RESERVED_INPUT_KEYS else k): v for k, v in inputs.items()}


def record_step(
    kind: str = "tool",
    name: Optional[str] = None,
    *,
    resamplable: bool = True,
    capture: Optional[CaptureFn] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: record each call of the wrapped function as a step.

    ``kind`` is ``"llm"``, ``"tool"`` or ``"memory"``. When no run is active the
    wrapper calls through transparently.
    """
    StepKind(kind)  # validate

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name or getattr(fn, "__name__", kind)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_context()
            if ctx is None:
                return fn(*args, **kwargs)
            inputs = _sanitize_inputs((capture or _default_capture)(args, kwargs))
            op = getattr(ctx, kind)
            return op(
                step_name,
                produce=lambda: fn(*args, **kwargs),
                resamplable=resamplable,
                **inputs,
            )

        wrapper.__agent_replay_wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


def llm(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shorthand: ``@instrument.llm`` records the function as an ``llm`` step."""
    return record_step("llm")(fn)


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shorthand: ``@instrument.tool`` records the function as a ``tool`` step."""
    return record_step("tool")(fn)


def memory(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Shorthand: ``@instrument.memory`` records the function as a ``memory`` step."""
    return record_step("memory")(fn)


def wrap(
    fn: Callable[..., Any],
    kind: str = "tool",
    name: Optional[str] = None,
    *,
    resamplable: bool = True,
    capture: Optional[CaptureFn] = None,
) -> Callable[..., Any]:
    """Return an instrumented copy of ``fn`` without using decorator syntax."""
    return record_step(kind, name, resamplable=resamplable, capture=capture)(fn)


# --- monkeypatching a dotted target -----------------------------------------

_PATCHED: Dict[str, Tuple[Any, str, Any]] = {}


def _resolve(target: str) -> Tuple[Any, str]:
    """Resolve ``'pkg.mod.Class.method'`` to ``(owner_object, attribute_name)``."""
    module_path, _, attr_path = target.rpartition(".")
    if not module_path or not attr_path:
        raise ValueError(f"invalid patch target {target!r}, expected 'module.path.attr'")
    # Walk from the deepest importable module down to the attribute owner.
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(mod_name)
        except ImportError:
            continue
        for p in parts[split:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]
    raise ImportError(f"could not import any module prefix of {target!r}")


def patch(
    target: str,
    kind: str = "tool",
    name: Optional[str] = None,
    *,
    resamplable: bool = True,
    capture: Optional[CaptureFn] = None,
) -> bool:
    """Instrument a dotted callable in place. Returns True if patched.

    Idempotent and reversible via :func:`unpatch`. Raises if the target's module
    cannot be imported; callers that want best-effort behaviour should catch
    :class:`ImportError` (this is what :func:`install` does).
    """
    if target in _PATCHED:
        return True
    owner, attr = _resolve(target)
    original = getattr(owner, attr)
    step_name = name or target.rsplit(".", 1)[-1]
    wrapped = record_step(kind, step_name, resamplable=resamplable, capture=capture)(original)
    setattr(owner, attr, wrapped)
    _PATCHED[target] = (owner, attr, original)
    return True


def unpatch(target: str) -> bool:
    """Undo a :func:`patch`. Returns True if something was restored."""
    entry = _PATCHED.pop(target, None)
    if entry is None:
        return False
    owner, attr, original = entry
    setattr(owner, attr, original)
    return True


# --- framework recipes (data only) ------------------------------------------
#
# Each recipe is a list of (dotted_target, kind, step_name). Patching is
# best-effort: targets whose SDK is not installed are skipped. These paths follow
# each SDK's stable public call site; add your own with patch()/RECIPES[...].

RECIPES: Dict[str, List[Tuple[str, str, str]]] = {
    "openai": [
        ("openai.resources.chat.completions.Completions.create", "llm", "openai.chat"),
        ("openai.resources.responses.Responses.create", "llm", "openai.responses"),
    ],
    "anthropic": [
        ("anthropic.resources.messages.Messages.create", "llm", "anthropic.messages"),
    ],
    "litellm": [
        ("litellm.completion", "llm", "litellm.completion"),
    ],
    "cohere": [
        ("cohere.Client.chat", "llm", "cohere.chat"),
    ],
    "google-genai": [
        ("google.genai.models.Models.generate_content", "llm", "gemini.generate"),
    ],
    "mistralai": [
        ("mistralai.Mistral.chat.complete", "llm", "mistral.chat"),
    ],
    "langchain": [
        ("langchain_core.language_models.chat_models.BaseChatModel.invoke", "llm", "langchain.llm"),
        ("langchain_core.tools.BaseTool.invoke", "tool", "langchain.tool"),
    ],
    "llama-index": [
        ("llama_index.core.llms.LLM.chat", "llm", "llamaindex.chat"),
    ],
    "crewai": [
        ("crewai.LLM.call", "llm", "crewai.llm"),
    ],
    "autogen": [
        ("autogen_core.models.ChatCompletionClient.create", "llm", "autogen.create"),
    ],
}


def available_frameworks() -> List[str]:
    """List the framework recipe keys known to :func:`install`."""
    return sorted(RECIPES)


# --- deterministic virtual time & entropy -----------------------------------

_TIME_PATCHED: Dict[str, Tuple[Any, str, Any]] = {}


def enable_virtual_time() -> None:
    """Route ``time.time``, ``datetime.now`` and ``uuid.uuid4`` through the context.

    While a run is active these read the recorded value on replay (and record the
    real value on capture), so *unmodified* agents that call the stdlib clock or
    uuid become deterministic. When no run is active they call through to the real
    implementation, so it is safe to leave enabled. Idempotent; reverse with
    :func:`disable_virtual_time`.
    """
    import datetime as _dt
    import time as _time
    import uuid as _uuid

    # Capture the true stdlib callables now, before patching, so the no-context
    # passthrough (and any re-entrancy) never resolves back to the patched ones.
    real_time = _time.time
    real_uuid = _uuid.uuid4
    real_dt_now = _dt.datetime.now

    def clock() -> float:
        ctx = current_context()
        return ctx.now() if ctx is not None else real_time()

    def uuid4():
        ctx = current_context()
        return _uuid.UUID(ctx.uuid()) if ctx is not None else real_uuid()

    class _VDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            ctx = current_context()
            if ctx is None:
                return real_dt_now(tz)
            return _dt.datetime.fromtimestamp(ctx.now(), tz)

    _install_time_patch("time.time", _time, "time", clock)
    _install_time_patch("uuid.uuid4", _uuid, "uuid4", uuid4)
    _install_time_patch("datetime.datetime", _dt, "datetime", _VDatetime)


def disable_virtual_time() -> None:
    """Undo :func:`enable_virtual_time`."""
    for target in list(_TIME_PATCHED):
        owner, attr, original = _TIME_PATCHED.pop(target)
        setattr(owner, attr, original)


@contextlib.contextmanager
def virtual_time():
    """Context manager form of :func:`enable_virtual_time`."""
    enable_virtual_time()
    try:
        yield
    finally:
        disable_virtual_time()


def _install_time_patch(target: str, owner: Any, attr: str, replacement: Any) -> None:
    if target in _TIME_PATCHED:
        return
    _TIME_PATCHED[target] = (owner, attr, getattr(owner, attr))
    setattr(owner, attr, replacement)


def install(*frameworks: str, strict: bool = False) -> List[str]:
    """Patch the call sites for the named frameworks (best-effort).

    Targets whose SDK is not importable are skipped unless ``strict=True``.
    Returns the list of dotted targets that were successfully patched. Pass no
    names to attempt every recipe.
    """
    names = frameworks or tuple(RECIPES)
    patched: List[str] = []
    for fw in names:
        if fw not in RECIPES:
            raise KeyError(f"unknown framework {fw!r}; known: {available_frameworks()}")
        for target, kind, step_name in RECIPES[fw]:
            try:
                patch(target, kind, step_name)
                patched.append(target)
            except (ImportError, AttributeError, ValueError):
                if strict:
                    raise
    return patched


def uninstall(*frameworks: str) -> None:
    """Undo :func:`install` for the named frameworks (or all if none given)."""
    names = frameworks or tuple(RECIPES)
    for fw in names:
        for target, _, _ in RECIPES.get(fw, []):
            unpatch(target)


@contextlib.contextmanager
def installed(*frameworks: str, strict: bool = False):
    """Context manager: install recipes for the duration of the block."""
    patched = install(*frameworks, strict=strict)
    try:
        yield patched
    finally:
        for target in patched:
            unpatch(target)


def record_agent(
    agent_fn: Callable[..., Any],
    task: Optional[Dict[str, Any]] = None,
    *,
    session_id: str,
    frameworks: Tuple[str, ...] = (),
    seed: int = 0,
    verifier: Optional[Callable[[Any], float]] = None,
    strict_serialization: bool = True,
):
    """Record an auto-instrumented agent that takes no explicit ``ctx``.

    Installs the given framework recipes, then records ``agent_fn(**task)`` with
    the ambient context — the "connect any framework" entry point.
    """
    with installed(*frameworks):
        return record(
            agent_fn,
            task,
            session_id=session_id,
            seed=seed,
            verifier=verifier,
            strict_serialization=strict_serialization,
            pass_context=False,
        )
