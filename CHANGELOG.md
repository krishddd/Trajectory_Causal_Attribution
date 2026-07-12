# Changelog

All notable changes to `agent-replay` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

## [0.6.0] — Drift & the entropy of autonomy

Closes the last open item from the *Architecting the Agent Multiverse* deck
(`docs/MULTIVERSE_GAPS.md`): the per-step drift / entropy curve (Gap 6). The
library now covers the deck's full vision.

### Added
- **Per-step drift / entropy curve (Gap 6).** `drift(traj, agent, verifier,
  state_scorer=None)` charts a run's health as it unfolds. Always available from
  the verifier alone: for each step it holds the factual prefix and resamples the
  remainder to estimate `P(success)` and its **binary entropy** — the "entropy of
  autonomy" that collapses at the empirical point of commitment (cross-checking
  `attribute`). Given an optional intermediate-state `state_scorer(step) -> [0,1]`
  it overlays an **alignment-health** series and per-step **drift**, flags
  `decayed` runs, locates the drift onset, and warns on the deck's *silent
  alignment decay* signature (internal health degrading while the outcome still
  looks recoverable). `DriftResult.to_text()` / `.to_dict()` / `.to_html()` (a
  self-contained SVG curve). CLI `agent-replay drift --state-scorer …`.
- `stats.binary_entropy(p)` — Bernoulli entropy in bits, `[0, 1]`.
- `mock_agent.health_scorer` — a reference intermediate-state scorer.

## [0.5.0] — The Multiverse

Implements the gaps found against the *Architecting the Agent Multiverse* deck
(see `docs/MULTIVERSE_GAPS.md`): first-class forking, deterministic time/entropy,
faithfulness, and a console.

### Added
- **Multiverse forking (Gap 1).** `fork(agent, traj, at_step, do=/remove=)` records
  a complete counterfactual child trajectory — held prefix served from the parent
  cassette, the intervened step, then the live continuation — with `meta` linking
  `parent_session`/`fork_step`/`intervention`. `afork` (async), `diff(a, b)` (first
  divergence + per-step state diff), `CheckpointStore.branches(session)`, and CLI
  `fork` / `branches` / `diff`. Shared prefixes dedupe through the CAS blob store.
- **Durable resume (Gap 5).** `resume(agent, traj)` fast-forwards the recorded
  prefix and continues the run live beyond the recorded horizon.
- **Deterministic virtual time & entropy (Gap 2).** `ctx.now()` / `ctx.uuid()`
  (sync on both contexts) record real values as non-resamplable steps and replay
  them; `instrument.enable_virtual_time()` / `virtual_time()` patch `time.time`,
  `datetime.now` and `uuid.uuid4` so unmodified agents become deterministic.
- **Step-level faithfulness (Gap 3).** `faithfulness(traj, agent, verifier)` masks
  each reasoning step and measures the outcome shift, classifying runs into
  correct/wrong × faithful/unfaithful and flagging correct-unfaithful (post-hoc
  rationalization) and wrong-faithful (best debugging signal). CLI `faithfulness`.
- **Multiverse Console (Gap 4).** `agent-replay serve` — a zero-dependency
  `http.server` UI to browse sessions, per-step frozen state, and the branch graph.
- **Action/output hashes on the node (Gap 7).** `Step.action_hash()` /
  `Step.output_hash()` expose the deck's Merkle node structure.

### Changed
- `CheckpointStore(check_same_thread=...)` for the read-only console.

## [0.4.0] — Test your agent

The "test your agent" milestone: gate CI on agent reliability, cut attribution
cost, close the repair loop, and support async agents. See `docs/ANALYSIS.md` §4.

### Added
- **Pytest plugin (`agent_replay.pytest_plugin`).** `assert_agent_passes(agent,
  task, verifier, rollouts=N, p_fail_max=0.05)` is a flakiness-aware assertion
  (agents are stochastic; one green run is not a pass) that, on failure, runs
  counterfactual attribution and puts the plain-language explanation — which
  step, why, minimal fix — into the `AssertionError`, optionally writing the HTML
  report as a CI artifact. Also `measure_failure_rate()` (p_fail + Wilson CI),
  the `AgentFlakyError` (carries structured results), and `agent_replay_session`
  / `assert_agent` fixtures registered via a `pytest11` entry point.
- **Adaptive rollouts.** `attribute(adaptive=True, target_ci_width=0.2)` (and
  `AblationEngine.run_plan_adaptive`, `contrastive_attribution(adaptive=)`) use
  sequential stopping — rollouts accrue until the failure-rate interval is tight
  enough. Measured ~2.6× fewer rollouts on the 6-step mock (more on longer
  trajectories); the verdict is unchanged. CLI: `--adaptive` / `--target-ci-width`.
- **Repair v2 — closed-loop step-wise fixes.** A `propose_fn(step, trajectory)`
  hook lets a user-supplied model propose repair candidates (validated causally,
  core stays dependency-free); `attribute(repair_propose_fn=)`. `Repair.to_guard()`
  emits a deploy-time recovery snippet (CLI prints it under `[GUARD]`), and
  `export_contrastive_pairs(results, path)` writes validated
  `(rejected → chosen)` JSONL for DPO/preference fine-tuning. `Repair` now carries
  `step_name` / `step_kind` / `p_fail_before`.
- **Async agents.** `async def` agents are recorded/replayed/attributed
  transparently — `record`/`replay` detect coroutine agents and dispatch to the
  new `arecord`/`areplay` (with `AsyncAgentContext` / `AsyncReplayContext`), so
  the whole synchronous attribution pipeline (including Shapley, repair, and
  branch-safe key matching) works on async agents unchanged.

### Changed
- The replay matching logic is factored into a shared `_decide` used by both the
  sync and async replay contexts, so they can never drift apart.

## [0.3.1] — Instrumentation bug-fix audit

### Fixed
- **Non-deterministic idempotency keys for patched instance methods.**
  `instrument`'s default argument capture used `repr()` for non-JSON objects,
  which embeds the object's memory address. Any patched instance method (every
  SDK recipe, where `self` is a captured arg) therefore produced a key that
  changed across processes, so a cross-process `record` → `attribute` (e.g. via
  the CLI) silently resampled every "held" step instead of serving the cassette.
  Non-JSON values are now reduced to a stable `<TypeName>` token.
- **Reserved-keyword collision.** Instrumenting a callable whose argument is named
  `name`, `produce`, or `resamplable` raised `TypeError` (clash with the context
  op parameters). Such captured keys are now renamed.

### Added / Changed
- `Session.record()` forwards `strict_serialization=` and `pass_context=` (so the
  ergonomic facade also supports auto-instrumented, `ctx`-free agents).
- CLI `attribute` on a passing run prints a friendly hint (use `--on-success
  credit`) instead of an uncaught `SuccessfulRunError` traceback.
- Documented the concurrency limitation: the ambient context propagates to
  `asyncio` tasks but not to raw worker threads; recording is single-threaded.

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
