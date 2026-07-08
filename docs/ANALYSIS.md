# Complete project analysis — v0.3.1 (July 2026)

A full-project assessment of `agent-replay` / Trajectory Causal Attribution as an
**agent-testing and step-wise failure-fixing tool**: what it is today, how strong
each part is, where it sits in the landscape, and the highest-leverage
improvements next. Companions: `HANDOFF.md` (v0.1 audit + invariants),
`RESEARCH_NOTES.md` (July 2026 literature sweep), `CHANGELOG.md`.

---

## 1. What the project is today

A zero-dependency, MIT, pip-installable Python library (~2.4k lines src, 95
tests, CI green on Python 3.9–3.12) that answers the question observability
tools cannot: **which step caused the failure** — causally, not correlationally.

**Pipeline:** record (cassette + Merkle CAS in SQLite) → deterministic replay
with an intervention algebra (hold / resample / do / remove, idempotency-key
matched) → stochastic ablation rollouts → attribution
(`P(fail|kept) − P(fail|ablated)`, Point-of-Commitment rule, antithetic MC
Shapley with CIs) → minimal counterfactual repair → explainable, traceable
report (what/where/why/fix + per-step causal roles) in CLI text, HTML, JSON.

**Connectivity:** explicit `ctx` API; ambient-context decorators; monkeypatch
recipes for 10 SDKs (OpenAI, Anthropic, Cohere, Google GenAI, Mistral, LiteLLM,
LangChain, LlamaIndex, CrewAI, AutoGen); `patch()` for anything else.

## 2. Component scorecard

| Component | Maturity | Notes |
|---|---|---|
| Recorder / cassette / CAS store | ★★★★☆ | Strict serialization, migration, dedup. Missing: schema version stamp, WAL, payload size guards. |
| Replayer (interventions) | ★★★★☆ | Branch-safe key matching (v0.2), non-resamplable handling. Missing: `mock-observe` as distinct op (needs action/obs split), `edit-context`, `swap-model`. |
| Ablation engine | ★★★☆☆ | Correct but **serial and fixed-N** — the cost bottleneck. No adaptive stopping, no parallelism, no prefilter. |
| Attribution (contrastive + PoC) | ★★★★★ | Faithful to the research; ground-truth validated (pivotal step, AND/OR, efficiency axiom). |
| Shapley estimator | ★★★★☆ | Antithetic, no-cache, no-truncation. Missing: stratified sampling, CI-targeted auto-stop. |
| Repair | ★★★☆☆ | Valid + minimality works; candidate space is naive (generic mutations). No LLM-proposed candidates, no contrastive-pair export. |
| Explainability | ★★★★☆ | what/where/why/fix + causal roles; ASCII/MD/HTML/JSON. Missing: per-claim links back to raw rollout evidence. |
| Instrumentation | ★★★★☆ | Ambient ctx, decorators, recipes, patch; key-stability fixed (v0.3.1). Gaps: **no async**, raw-thread limitation, SDK response objects vs strict serialization. |
| Testing story *for users* | ★★☆☆☆ | The library tests itself well, but offers users no turnkey way to gate CI on agent regressions (see §4.1). |
| Benchmarked credibility | ★☆☆☆☆ | No public-benchmark numbers yet (Who&When harness not built). |

## 3. Position in the landscape (from the July 2026 sweep)

- **LLM-as-judge attribution** — ~14.2% step accuracy on Who&When (ICML'25
  Spotlight): the baseline this project beats by construction.
- **FAMAS** (spectrum analysis) and **zero-replay prediction** (event-KG +
  learned ranker, Recall@5 0.93): cheaper, less rigorous — natural *pre-filters*
  in front of this engine, not replacements.
- **CAR / CausalFlow**: the method implemented here; CAR's synthetic-SCM checks
  are now regression tests in this repo.
- **Observability platforms** (LangSmith, Langfuse, AgentOps, Braintrust):
  record and display traces but do not do counterfactual attribution — they are
  *feeders* (via OTel GenAI import) rather than competitors.
- Unique combination this repo has that none of the above ship together:
  causal step attribution **+** validated minimal repair **+** plain-language
  traceable explanation **+** zero-dep universal adapters.

## 4. Improvement plan (highest leverage first)

### 4.1 Make it an agent *testing* tool: `pytest` plugin (new, v0.4 headline)
The single biggest usability gap. Today a developer must hand-roll record →
attribute → assert. Ship `agent_replay.pytest_plugin`:
- `@pytest.fixture agent_replay_session` (tmp SQLite store);
- `assert_agent_passes(agent, task, verifier, rollouts=N, p_fail_max=0.05)` —
  a *flakiness-aware* assertion (agents are stochastic; a single green run is
  not a pass);
- on failure, automatically run attribution and print the explanation +
  write the HTML report as a test artifact — "the test tells you *which step*
  broke";
- cassette re-use between CI runs (record once locally, replay offline in CI —
  zero API cost, the VCR value prop).
Zero new deps (pytest is already the dev extra); entry-point
`[project.entry-points.pytest11]`.

### 4.2 Cut attribution cost 3–10×: adaptive rollouts (v0.4)
From RESEARCH_NOTES §4 (KernelSHAP variance practice, VRDS): per-step
sequential stopping — run rollouts until the bootstrap CI decisively
excludes/includes zero (`target_ci_width=`), allocate back-to-front for PoC
(most trajectories resolve after a suffix), and optional
`ThreadPoolExecutor` rollout parallelism for I/O-bound produce fns (safe:
each rollout has its own ReplayContext; ambient ctx binds per thread inside
the engine). This multiplies the affordability of everything else.

### 4.3 Step-wise fixes, closed loop: repair v2 (v0.4/v0.5)
Today's repair proves *a* fix exists; make it the "step-wise fixes" product:
- **LLM-proposed candidates:** `propose_fn(step, explanation) -> [candidates]`
  hook (user supplies the model call; keeps zero-dep), validated causally as now;
- **guard export:** emit the validated repair as a runtime guard snippet
  (`if step==culprit and action matches bad-pattern: constrain(...)`) — the
  deploy-time recovery the source research describes;
- **contrastive-pair export** (`wrong step → minimal fix` JSONL) for
  DPO/preference fine-tuning;
- *(training-knowledge, verify when search resets)* step-level verifier /
  process-reward-model scoring can rank candidate repairs before causal
  validation, cutting validation rollouts.

### 4.4 Async agents (v0.4)
Most production agents are async; the ambient context already propagates to
asyncio tasks — add `AsyncAgentContext` (`await ctx.llm(...)`, async produce),
`arecord`/`areplay`, and async wrappers in `instrument`. Mechanical but
unlocks the largest blocked user group.

### 4.5 Trajectory import + benchmark credibility (v0.5)
- `interop/otel.py`: build `Trajectory` from OTel GenAI spans (LangSmith/
  Langfuse/AgentOps exports) with user `resample_fn` per step kind;
- `benchmarks/whowhen/`: run the public Who&When dataset, publish
  accuracy@step vs the 14.2% judge baseline and cost-in-rollouts — the
  headline-claim evidence.

### 4.6 Smaller, known items
Staged prefilter funnel (spectrum heuristics → causal on top-M; RESEARCH_NOTES
§5); `mock-observe`/`edit-context` via action/observation split on `Step`;
SDK-response-object serialization helpers per recipe; WAL + schema version in
the store; per-claim evidence links in explanations; `agent-replay serve`
(stdlib HTTP session browser).

## 5. Suggested milestones

- **v0.4 — "test your agent":** pytest plugin + adaptive rollouts + async +
  repair `propose_fn`. (4.1–4.4)
- **v0.5 — "prove it & import anything":** OTel import, Who&When numbers,
  prefilter funnel, repair guard/DPO export, serve UI. (4.3, 4.5, 4.6)
- **v1.0 — hardening:** schema versioning, property-based tests on the plan
  algebra, accuracy-vs-cost benchmark suite, PyPI trusted publishing.

*Sources: the cited findings are from the live July 2026 sweep recorded in
`RESEARCH_NOTES.md` (Who&When 2505.00212, CAR 2606.08275, CausalFlow
2605.25338, FAMAS 2509.13782, zero-replay 2606.14805, VRDS 2210.16835,
KernelSHAP, OTel GenAI). Items marked "training-knowledge" (PRM-ranked repair)
should be re-verified with a fresh search before implementation.*
