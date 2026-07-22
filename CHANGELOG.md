# Changelog

All notable changes to `agent-replay` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

## [0.11.0] — Shared rollouts & property-based guards

### Added
- **Cross-analysis rollout cache (`RolloutCache`, `analyze`).** `attribute`,
  `drift` and `faithfulness` each re-run the same prefix-hold rollout family on a
  run; passing a shared `RolloutCache` (via the new `cache=` argument on all
  three, or the turnkey `analyze(traj, agent, verifier, ...)` wrapper) lets later
  analyses reuse earlier rollouts, so the combined cost is barely more than the
  most expensive one alone. Only prefix-hold / factual plans are cached — Shapley
  **coalition** values are never cached (each must stay an independent draw to
  preserve marginal variance), and `coalition_value` bypasses the cache entirely.
  Cache keys include `trajectory.root_hash`, the plan, rollouts, `fail_threshold`
  and `base_seed`, so mismatched configs never collide; cached results are
  byte-identical to uncached ones.
- **Property-based tests (hypothesis).** `tests/test_properties.py` fuzzes the
  engine's core invariants over random branching agents and plans: factual replay
  fidelity, **no cassette cross-contamination** for any prefix or coalition plan
  (the soundness guarantee for branching agents), and the plan-algebra decision
  precedence. `hypothesis` added to the dev extra.

## [0.10.0] — The full intervention algebra

Completes the paper's do-calculus intervention set (`docs/HANDOFF.md` §3.4) via an
additive **action/observation split** on the step — no existing trajectory,
hash, or call site changes.

### Added
- **Action/observation split (`Step.action`).** The SCM distinguishes the policy's
  *action* from the *observation* it yields (deck slide 9). A new optional
  `Step.action` field records the action when it differs from the observation
  (`Step.output`); `Step.action_value` / `Step.observation` resolve either case,
  and `Step.action_value_hash()` enables "same decision, different observation?"
  queries. Fully additive: `action` defaults to `None` ("action == output"), so
  older trajectories, the Merkle hashes, and every call site are unchanged.
  - Recording API: `ctx.tool/llm/memory(..., observe=fn)` records a distinct
    action (`produce()`) and observation (`observe(action)`, served downstream).
    Async `observe` policies are awaited.
  - Store: nullable `action_hash` column (schema **v2**, idempotent migration);
    the action blob is only stored when a step actually splits.
- **`mock-observe` intervention.** `ReplayPlan(observed={i: value})` /
  `ReplayPlan.mock_observe(i, value)` replaces a step's *observation* downstream
  while keeping its recorded *action* — distinct from `do`/`force`, which
  overrides the action. This is how you test memory/context reliance: mock the
  observation of the step that produces the context ("edit-context").
- **`swap-model` intervention.** `ReplayPlan(model_override="…")` exposes
  `ctx.model_hint` to the replayed/forked run so a model-parameterized policy can
  answer "would a different model have succeeded from step *i*?".
- **`fork(..., observe=, model=)`** wire both new interventions into the
  Multiverse; children carry the `mock_observe` / `swap_model` intervention label
  (and `model_override` in `meta`).

## [0.9.0] — Systematic blame across many runs

### Added
- **Multi-trajectory aggregation (`agent_replay.aggregate`).** `aggregate_runs(
  trajectories, agent, verifier, ...)` attributes each *failing* run and pools the
  results by step **name** (indices shift between runs; names are the stable
  identity of an operation, and a name may recur within a run). For each named
  step it reports how often it was the culprit, its mean attribution with a
  bootstrap interval **over runs**, and the point-of-commitment rate; the ranking
  surfaces the agent's `systematic_culprit` — the difference between debugging one
  failure and finding a design flaw. Passing runs are skipped and counted.
  `aggregate(results)` pools already-computed `AttributionResult`s. Reuses
  `attribute` wholesale (no new estimation), so it inherits the engine's
  soundness. `AggregateResult.to_text()` / `.to_dict()`.
- **CLI `agent-replay aggregate`.** Pools attribution over selected sessions (or
  every session in a store) and prints the systematic-weak-step ranking;
  `--out` writes the aggregate as JSON.
- **`stats.bootstrap_mean_interval`** — public percentile-bootstrap interval for
  the mean of a sample (used for the pooled over-runs CIs).

## [0.8.0] — Prove it & import anything

Realizes the "prove it & import anything" milestone (`docs/ANALYSIS.md` §5): run
attribution on traces recorded *elsewhere*, and publish a step-accuracy number
against the Who&When LLM-judge baseline.

### Added
- **Trajectory import (`agent_replay.interop`).** Build a first-class
  `Trajectory` from traces this library did not record:
  - `interop.from_otel_spans(spans, ...)` — map OpenTelemetry **GenAI** spans
    (LangSmith / Langfuse / AgentOps / OpenLLMetry exports) onto llm/tool steps,
    tolerant of missing/non-standard attributes (unknown spans import as opaque
    tool steps rather than being dropped).
  - `interop.from_jsonl(path, ...)` — one-step-per-line *or* a single
    `{"steps": [...]}` object.
  - `interop.from_steps(step_dicts, ...)` — from an in-memory list; Merkle-chains
    and hashes imported steps identically to the recorder.
  - `interop.replayable_agent(traj, resample_fns={...})` — the bridge that makes
    an imported (observation-only) trace **attributable**: it reconstructs the
    recorded operations as an executable agent, turning on resamplability for the
    steps you supply a `fn(ctx, inputs) -> output` policy for. Steps without a
    policy stay observation-only (served from the cassette), exactly as the
    non-resamplable contract requires. Pass the agent + trajectory straight to
    `attribute`.
- **Who&When benchmark harness (`benchmarks/whowhen.py`).** Measures step-
  localization accuracy against the *Who&When* task (arXiv:2505.00212), where the
  strongest LLM-judge attributor reaches ~14.2%. Ships a deterministic synthetic
  generator with known ground truth (chain agents with a single responsible
  step, varied length/position) so it runs offline; on the default suite causal
  attribution localizes **100%** of culprits vs **12.5%** for max-magnitude
  blame *without* the Point-of-Commitment rule and the ~14.2% judge baseline —
  quantifying what the PoC rule buys. `evaluate()` accepts imported (real-
  dataset) trajectories via `interop`. Smoke-tested in CI.

### Changed
- README: mermaid pipeline / replay-decision / multiverse diagrams replace the
  ASCII architecture art; new "Import from anywhere" and benchmark sections.
- sdist now includes `benchmarks/`.

## [0.7.0] — Correctness, cost & credibility

A hardening release: one genuine correctness fix, the biggest wall-clock lever
(parallel rollouts), a cost cut for the most expensive phase (adaptive Shapley),
and several long-open stragglers from `docs/HANDOFF.md` / `docs/ANALYSIS.md`.

### Fixed
- **Credit-mode Shapley sign bug (correctness).** `attribute(on_success="credit")`
  previously negated only the contrastive scores, leaving Shapley values
  failure-signed. With `method="both"` a passing run then showed credit-signed
  contrastive scores beside failure-signed Shapley values for the *same* step,
  and with `method="shapley"` `_select_culprit` picked the step that *least*
  secured success as the "save point" — the exact inverse of credit mode.
  `_negate` now flips the Shapley value and its CI too, so every signal shares
  one sign convention. Regression test in `tests/test_soundness.py`.

### Added
- **Parallel rollouts (`AblationEngine(max_workers=N)`, `attribute(max_workers=N)`).**
  Rollouts run on a `ThreadPoolExecutor` — each rollout is a pure function of
  `(plan, seed_tag, k)` with its own `ReplayContext`, seed, and thread-bound
  ambient context, so parallel execution returns **byte-identical** results to
  serial (guarded by a test) while collapsing wall-clock for I/O-bound produce
  fns (real LLM calls). The single biggest practical speedup for real agents.
- **Adaptive Shapley (`shapley_attribution(adaptive=True)`, `attribute(adaptive=)`).**
  Sequential stopping now covers Phase 2, not just contrastive: permutation
  pairs accrue until every step's bootstrap CI on its marginal mean is narrower
  than `target_ci_width` (bounded by `min_pairs`/`max_pairs`), cutting the most
  expensive phase severalfold. The fixed-N path is byte-identical (pairs drawn
  from one RNG sequence).
- **Store hardening.** `CheckpointStore` opens on-disk databases in **WAL** mode
  (`synchronous=NORMAL`) so readers and parallel writers no longer block the
  whole file, and stamps a **`SCHEMA_VERSION`** via `PRAGMA user_version`
  (exposed as `store.schema_version`) so older databases can be detected and
  migrated forward. Migrations are idempotent.
- **Wilson CI on `P(fail | ablated)`.** Contrastive attribution now attaches a
  Wilson score interval to each step's ablated failure rate
  (`StepAttribution.p_fail_ablated_ci`); the HTML report shows it under the
  point estimate and the JSON report carries it. (Closes a HANDOFF §2.6 leftover:
  Wilson intervals were computed in `stats.py` but never surfaced.)

### Changed
- **OpenAI adapter warns on `temperature=0`.** `wrap_openai` now warns once when
  a call uses `temperature=0`: a deterministic policy makes counterfactual
  resampling zero-variance, collapsing attribution to ~0 silently. (HANDOFF §2.6.)
- **Faithfulness baseline is smoothed.** `faithfulness(..., kept_rollouts=)`
  (default `min(8, rollouts)`) averages `success_kept` over several held-full
  replays instead of a single one, so a *stochastic verifier* near the threshold
  no longer makes the quadrant assignment brittle. A no-op for deterministic
  verifiers, so existing behaviour is unchanged.

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

### Changed
- **The package ships zero bundled agents.** The demo/reference agent formerly at
  `agent_replay.mock_agent` has been removed from the installed package and moved
  into the test tree — `agent-replay` is now purely a tool you point at *your own*
  agent + verifier. README/CLI docs updated to show integrating your own agent.

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
