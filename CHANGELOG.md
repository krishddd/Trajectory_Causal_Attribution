# Changelog

All notable changes to `agent-replay` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

## [0.3.0] — Universal adapters & explainability

### Added
- **Universal instrumentation (`agent_replay.instrument`)** — connect *any*
  framework, not just LangChain. Built on an ambient `contextvars` context so
  agents need no explicit `ctx`:
  - `@instrument.tool` / `@instrument.llm` / `@instrument.memory` decorators and
    `instrument.wrap(fn, kind, name)` to record any callable.
  - `instrument.patch(dotted_target, kind)` / `unpatch` / `install(*frameworks)` /
    `installed(...)` context manager — monkeypatch unmodified SDK call sites,
    driven by a data-only `RECIPES` registry (OpenAI, Anthropic, Cohere, Google
    GenAI, Mistral, LiteLLM, LangChain, LlamaIndex, CrewAI, AutoGen). Best-effort:
    absent SDKs are skipped. Adding a framework = one registry entry.
  - `instrument.record_agent(agent_fn, task, frameworks=(...))` — record an
    auto-instrumented agent that takes no `ctx`.
  - `record`/`replay`/`attribute`/`AblationEngine` gained `pass_context=` so the
    ablation engine re-runs auto-instrumented agents identically.
- **Explainability (`agent_replay.explain`)** — a traceable, plain-language
  narrative over an attribution result (methods unchanged; presentation only):
  - `explain(result, trajectory)` / `result.explain(trajectory)` →
    an `Explanation` with **what / where / why / fix / confidence** plus a
    per-step **causal trace** labelling each step *decisive*, *locked-in*,
    *contributing*, *observed-only*, or *benign* — tracing the run from first
    action to the point of no return, with the numbers behind every claim.
  - `Explanation.to_text()` (ASCII-safe), `.to_markdown()`, `.to_dict()`,
    `.from_dict()`.
  - The HTML report gains an **Explanation panel** with a colour-coded trace; the
    JSON report embeds an `explanation` section; the CLI prints the narrative
    (suppress with `--no-explain`) and embeds it in generated reports.
- Docs: `docs/frameworks.md` (three ways to connect any framework),
  `examples/connect_any_framework.py`.

### Changed
- Repository renamed to `Trajectory_Causal_Attribution` (importable package
  remains `agent_replay`).

## [0.2.0] — Soundness

The v0.2 milestone hardens the causal engine so attribution is correct for
**branching** agents (not just linear ones) and never reports silent noise. See
`docs/HANDOFF.md` §2 for the reproduced bugs this release fixes and
`docs/RESEARCH_NOTES.md` for the supporting literature.

### Fixed
- **Idempotency-key cassette matching (critical).** The replayer now binds live
  replay calls to recorded steps by content key (`Step.op_key` = hash of
  kind + name + inputs), consuming recorded steps in order, instead of by call
  *position*. When an upstream ablation changes the control flow, a held step is
  served from the cassette only when the *same* operation actually recurs;
  otherwise the timeline has diverged and the call is resampled live. This
  eliminates the cross-contamination that made Shapley coalition plans unsound
  for any agent whose step sequence depends on earlier outputs. Linear agents are
  unaffected (key-in-order matching == positional there). Legacy behaviour is
  still available via `replay(..., match="position")`.
- **Passing-run guard.** `attribute()` on a trajectory that did *not* fail now
  raises `SuccessfulRunError` by default instead of emitting negative
  "attribution" noise into a failure report.
- **Non-resamplable steps no longer corrupt rollouts.** A step recorded without a
  genuine policy (`produce=None`, or an observation-only adapter) is marked
  `resamplable=False`; on replay it always serves its recorded output rather than
  returning `None`, and the scorer surfaces it as non-attributable.

### Added
- **Credit mode.** `attribute(..., on_success="credit")` runs the symmetric
  analysis on a *successful* run: which step most secured success (the latest
  step whose re-decision would introduce a significant failure risk). Reports and
  CLI label this as a "save point".
- **Strict serialization.** Recording validates that every step input/output is
  JSON-serialisable and raises `NonSerializableStepError` at record time (opt out
  with `strict_serialization=False`), instead of silently degrading payloads to
  strings on store round-trip.
- **`resamplable` flag** on `Step`/`StepAttribution`, persisted in the store
  (with an automatic column migration for older databases) and surfaced as an
  "observed-only" badge in the HTML report.
- **CLI:** `agent-replay list`; and `attribute` flags `--fail-threshold`,
  `--base-seed`, `--on-success {error,credit}`.
- **Ground-truth tests** mirroring the Causal Agent Replay validation: AND/OR
  two-step interaction credit splits, a quantitative Shapley efficiency-axiom
  check, a pivotal-single-step localisation, and a branching-agent
  cross-contamination regression.
- New public exceptions: `AgentReplayError`, `NonSerializableStepError`,
  `SuccessfulRunError`.

### Changed
- `AttributionResult` gained a `mode` field (`"failure"` | `"credit"`).
- Dead code removed (`AblationEngine._seed_salt`); the `REMOVED` sentinel is now a
  distinct singleton rather than `None`.

## [0.1.0] — Initial release

Counterfactual step-ablation failure attribution: recorder, SQLite checkpoint
store with content-addressable blobs, deterministic replayer with intervention
plans, ablation engine, contrastive + Shapley attribution with the
Point-of-Commitment Rule, minimal counterfactual repair, HTML+JSON reports, CLI,
LangChain/OpenAI adapters, and a full pytest suite.
