"""Import trajectories recorded *elsewhere* — OpenTelemetry GenAI spans, a JSONL
of steps, or an in-memory list — so attribution can run on traces this library
did not record.

Observability platforms (LangSmith, Langfuse, AgentOps, OpenLLMetry, …) already
capture agent runs as spans. They can *display* a trace but not tell you which
step caused a failure. This module bridges that gap: it turns an external trace
into a first-class :class:`~agent_replay.types.Trajectory` you can diff, serve,
hash — and, given a per-kind/per-name **resample policy**, attribute.

The catch, stated honestly
--------------------------
Counterfactual attribution requires *re-executing* the agent's policy. An
imported trace is observation-only: it recorded what happened, not the callable
that produced it. So:

* Without resample functions, every imported step is ``resamplable=False`` — the
  trajectory is fully inspectable/replayable but attribution has nothing to
  perturb (every step is "observed-only").
* With a ``resample_fns`` map (``{step_name_or_kind: fn(ctx, inputs) -> output}``)
  the step becomes attributable: :func:`replayable_agent` builds an agent that
  re-issues the recorded operations in order, calling your policy on the steps
  you supplied one for. This is the "user-supplied resample_fn per step kind"
  contract the roadmap describes.

Because an imported agent has no live control flow, the reconstruction is linear:
each step is re-drawn from its *recorded* inputs. That is the correct semantics
for a trace whose branching logic we cannot see — and it is exactly what the
single-step contrastive estimator needs.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from .types import Step, StepKind, Trajectory

# A resample policy for one step: given the replay context and the step's recorded
# inputs, return a fresh output. Draw randomness from ``ctx.rng`` to stay
# deterministic-by-seed (the replay invariant).
ResampleFn = Callable[[Any, Dict[str, Any]], Any]

# How to fold the per-step outputs of a reconstructed run into a single result the
# verifier scores. The default keeps the full ordered output list.
Assemble = Callable[[List[Any]], Any]


def _default_assemble(outputs: List[Any]) -> Dict[str, Any]:
    return {"outputs": list(outputs)}


def _coerce_kind(kind: Any) -> StepKind:
    if isinstance(kind, StepKind):
        return kind
    k = str(kind).lower()
    if k in ("llm", "chat", "completion", "text_completion", "embeddings", "model"):
        return StepKind.LLM
    if k in ("tool", "function", "execute_tool", "retrieval", "api"):
        return StepKind.TOOL
    if k in ("memory", "store", "retrieve", "vector"):
        return StepKind.MEMORY
    # Unknown kinds default to TOOL (an opaque external operation).
    return StepKind.TOOL


def from_steps(
    steps: Sequence[Dict[str, Any]],
    *,
    session_id: str,
    task: Optional[Dict[str, Any]] = None,
    result: Any = None,
    outcome_score: Optional[float] = None,
    verifier: Optional[Callable[[Any], float]] = None,
    assemble: Assemble = _default_assemble,
    meta: Optional[Dict[str, Any]] = None,
) -> Trajectory:
    """Build a :class:`Trajectory` from a list of step dicts.

    Each step dict needs ``kind`` and ``name``; ``inputs`` and ``output`` are
    optional (default ``{}`` / ``None``). ``kind`` is coerced onto the three
    canonical classes (llm / tool / memory). Steps are Merkle-chained exactly as
    the recorder does, so imported trajectories hash and dedupe identically.

    Imported steps are ``resamplable=False`` (observation-only) unless the dict
    sets ``resamplable``; :func:`replayable_agent` flips that on for steps you
    give a resample policy. The trajectory ``result`` defaults to
    ``assemble([step outputs])`` so the factual outcome the verifier sees matches
    what a reconstructed replay produces.
    """
    built: List[Step] = []
    parent = ""
    for i, raw in enumerate(steps):
        step = Step(
            index=i,
            kind=_coerce_kind(raw.get("kind", "tool")),
            name=str(raw.get("name", f"step_{i}")),
            inputs=dict(raw.get("inputs", {}) or {}),
            output=raw.get("output"),
            resamplable=bool(raw.get("resamplable", False)),
        )
        step.compute_hashes(parent)
        parent = step.step_hash
        built.append(step)

    if result is None:
        result = assemble([s.output for s in built])

    traj = Trajectory(
        session_id=session_id,
        task=dict(task or {}),
        steps=built,
        seed=0,
        result=result,
        meta=dict(meta or {}),
    )
    if outcome_score is not None:
        traj.outcome_score = float(outcome_score)
    elif verifier is not None:
        traj.outcome_score = float(verifier(result))
    return traj


def from_jsonl(path: str, *, session_id: Optional[str] = None, **kwargs: Any) -> Trajectory:
    """Build a trajectory from a JSONL file.

    Two layouts are accepted:

    * **one step per line** — every line is a step object (``{"kind", "name",
      "inputs", "output"}``); pass ``session_id`` (and optionally ``task`` /
      ``verifier`` / ``assemble``) as keyword arguments.
    * **a single JSON object** (if the whole file parses as one object) with a
      ``steps`` list plus any of ``session_id`` / ``task`` / ``result`` /
      ``meta`` — a self-contained trajectory export.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    # Try whole-file single-object first.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "steps" in obj:
            sid = session_id or obj.get("session_id")
            if sid is None:
                raise ValueError("from_jsonl: no session_id in file or arguments")
            return from_steps(
                obj["steps"],
                session_id=sid,
                task=obj.get("task"),
                result=obj.get("result"),
                outcome_score=obj.get("outcome_score"),
                meta=obj.get("meta"),
                **kwargs,
            )
    except json.JSONDecodeError:
        pass
    # Fall back to one step per line.
    rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if session_id is None:
        raise ValueError("from_jsonl: session_id is required for a line-per-step file")
    return from_steps(rows, session_id=session_id, **kwargs)


# -- OpenTelemetry GenAI span mapping ---------------------------------------

# Request attributes worth keeping as step inputs (OTel GenAI semantic conventions).
_OTEL_INPUT_ATTRS = (
    "gen_ai.request.model",
    "gen_ai.system",
    "gen_ai.operation.name",
    "gen_ai.request.temperature",
    "gen_ai.request.max_tokens",
    "gen_ai.request.top_p",
    "gen_ai.tool.name",
    "gen_ai.prompt",
)
_OTEL_OUTPUT_ATTRS = (
    "gen_ai.completion",
    "gen_ai.response.finish_reasons",
    "gen_ai.tool.result",
)


def _otel_event_content(span: Dict[str, Any]) -> Dict[str, Any]:
    """Pull prompt/choice content out of a span's events, if present."""
    inputs: List[Any] = []
    outputs: List[Any] = []
    for ev in span.get("events", []) or []:
        nm = str(ev.get("name", ""))
        body = ev.get("attributes", ev.get("body", {})) or {}
        if nm in ("gen_ai.choice", "gen_ai.assistant.message"):
            outputs.append(body)
        elif nm.startswith("gen_ai.") and "message" in nm:
            inputs.append(body)
    out: Dict[str, Any] = {}
    if inputs:
        out["messages"] = inputs
    if outputs:
        out["choices"] = outputs
    return out


def _span_to_step(span: Dict[str, Any]) -> Dict[str, Any]:
    attrs = span.get("attributes", {}) or {}
    op = str(attrs.get("gen_ai.operation.name", "")).lower()
    tool_name = attrs.get("gen_ai.tool.name")
    is_tool = op in ("execute_tool", "invoke_tool") or tool_name is not None
    is_genai = any(str(k).startswith("gen_ai.") for k in attrs)
    kind = "tool" if is_tool else ("llm" if is_genai else "tool")

    name = tool_name or attrs.get("gen_ai.request.model") or span.get("name") or f"{kind}"

    inputs: Dict[str, Any] = {k: attrs[k] for k in _OTEL_INPUT_ATTRS if k in attrs}
    ev = _otel_event_content(span)
    if "messages" in ev:
        inputs["messages"] = ev["messages"]

    output: Any = None
    for k in _OTEL_OUTPUT_ATTRS:
        if k in attrs:
            output = attrs[k]
            break
    if output is None and "choices" in ev:
        output = ev["choices"] if len(ev["choices"]) > 1 else ev["choices"][0]

    return {"kind": kind, "name": str(name), "inputs": inputs, "output": output}


def from_otel_spans(
    spans: Sequence[Dict[str, Any]],
    *,
    session_id: str,
    task: Optional[Dict[str, Any]] = None,
    sort_by_start: bool = True,
    **kwargs: Any,
) -> Trajectory:
    """Build a trajectory from OpenTelemetry GenAI spans (best-effort).

    ``spans`` are plain dicts (as exported by the OTel SDK / an OTLP JSON dump):
    ``{"name", "attributes": {...}, "events": [...], "start_time"?}``. LLM spans
    (``gen_ai.*`` attributes) become ``llm`` steps; ``execute_tool`` spans or any
    span carrying ``gen_ai.tool.name`` become ``tool`` steps; request attributes
    (model, temperature, messages) are kept as inputs and the completion / tool
    result as the output. Spans are ordered by ``start_time`` when present.

    The mapping follows the OTel GenAI semantic conventions but is deliberately
    tolerant — unknown spans are imported as opaque ``tool`` steps rather than
    dropped, so nothing is silently lost. Post-process the returned trajectory's
    steps if your exporter uses non-standard attribute names.
    """
    ordered = list(spans)
    if sort_by_start:
        ordered.sort(key=lambda s: s.get("start_time", s.get("startTimeUnixNano", 0)) or 0)
    step_dicts = [_span_to_step(s) for s in ordered]
    return from_steps(step_dicts, session_id=session_id, task=task, **kwargs)


# -- making an imported trajectory attributable -----------------------------


def replayable_agent(
    trajectory: Trajectory,
    resample_fns: Optional[Dict[str, ResampleFn]] = None,
    *,
    assemble: Assemble = _default_assemble,
) -> Callable[..., Any]:
    """Turn an imported trajectory into an executable agent the engine can ablate.

    Returns an ``agent_fn(ctx, **task)`` that re-issues the recorded operations in
    order. For each step it looks up a resample policy in ``resample_fns`` by step
    **name** first, then by **kind** (``"llm"``/``"tool"``/``"memory"``); a match
    makes the step resamplable (its policy is ``fn(ctx, recorded_inputs)``), and
    the absence of one leaves it observation-only (served from the cassette).

    Resamplability is a property of the *stored* step (the ablation engine serves
    the recorded output for a non-resamplable step even under a resample plan), so
    this function sets ``step.resamplable`` on the trajectory in place to match the
    policies you supplied. Pass the returned agent and the same trajectory to
    :func:`agent_replay.attribute`.
    """
    resample_fns = resample_fns or {}
    resolved: List[Optional[ResampleFn]] = []
    for step in trajectory.steps:
        fn = resample_fns.get(step.name)
        if fn is None:
            fn = resample_fns.get(step.kind.value)
        step.resamplable = fn is not None
        resolved.append(fn)

    steps = list(trajectory.steps)

    def agent_fn(ctx: Any, **_task: Any) -> Any:
        outputs: List[Any] = []
        for step, fn in zip(steps, resolved):
            op = getattr(ctx, step.kind.value)  # ctx.llm / ctx.tool / ctx.memory
            if fn is not None:

                def produce(s: Step = step, f: ResampleFn = fn) -> Any:
                    return f(ctx, dict(s.inputs))

                out = op(step.name, produce=produce, resamplable=True, **step.inputs)
            else:
                out = op(step.name, produce=None, resamplable=False, **step.inputs)
            outputs.append(out)
        return assemble(outputs)

    return agent_fn
