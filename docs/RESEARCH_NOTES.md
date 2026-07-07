# Research digest → codebase improvements (July 2026)

Companion to `HANDOFF.md`. This maps the current state of the art around agent
failure attribution and replay onto **specific improvements in this codebase**.
Each section: what the research says → what to change in `agent-replay`.

---

## 1. The attribution landscape: where agent-replay sits

Three families of failure-attribution methods now exist:

| Family | Representative | Accuracy / cost | agent-replay's relation |
|---|---|---|---|
| **LLM-as-judge** (correlational) | Who&When benchmark methods (All-at-Once, Binary Search, Step-by-Step) | ~53.5% agent-level, **~14.2% step-level** — "fails to achieve practical usability" (ICML 2025 Spotlight) | The baseline we beat by being causal |
| **Spectrum analysis** (statistical, replay-light) | FAMAS (arXiv 2509.13782) | Best of 12 baselines on Who&When; uses repeated executions + suspiciousness ranking over agent/action activation patterns | A cheap *pre-filter* we should add (§5) |
| **Counterfactual replay** (causal) | CAR (arXiv 2606.08275), CausalFlow (arXiv 2605.25338) | CAR: recovers pivotal steps and two-step interactions on synthetic SCMs; Shapley efficiency 0.909 vs analytic 0.91 | The method this library implements |
| **Zero-replay prediction** (learned) | Knowledge-Based Zero-Replay Debugging (arXiv 2606.14805) | Branch Recall@5 0.73→0.93 at **zero** replay cost via event-KG + gradient-boosted ranker | A candidate-pruning stage (§5) |

**Implication for the codebase:** counterfactual replay is the accuracy gold
standard but the *most expensive* method (O(steps × rollouts) agent executions).
The competitive frontier is **hybrid**: cheap statistical/learned pre-filters to
shrink the candidate set, then causal replay only on survivors. agent-replay
currently ablates *every* step — that should become a staged funnel (§5).

## 2. CAR's own validation → tests we are missing

The CAR paper validates on **synthetic SCMs with known ground truth**: pivotal
single steps, two-step AND/OR interactions (expected Shapley ≈ 0.44/0.45/≈0),
and checks the **efficiency axiom quantitatively** (0.909 vs analytic 0.91).

**Codebase changes:**
- Add `tests/test_synthetic_scm.py`: fixture agents with *analytically known*
  attribution values — a pivotal-step chain, an AND-failure pair, an OR-failure
  pair. Assert Shapley recovers ≈0.5/0.5 splits on AND/OR (single-step ablation
  provably reports ≈2.0-total / ≈0.0-total there — the motivating case, untested
  in v0.1).
- Strengthen `test_shapley_efficiency_axiom` from "sum ∈ [0,1]" to
  `|sum(φ) − (v(full) − v(empty))| < 0.1` with `v(empty)` estimated
  independently — mirroring CAR's 0.909-vs-0.91 check.
- CAR "runs on hosted or free local models" — add an integration example against
  a local model (e.g. an Ollama-backed produce fn) to prove the same claim.

## 3. Determinism reality check (2025–2026 consensus)

Industry findings: OpenAI's `seed` param "improves reproducibility but doesn't
guarantee it"; Anthropic documents that temperature=0 is still non-deterministic;
measured accuracy variation up to 15% across "deterministic" runs. Root causes:
floating-point non-associativity and batch-size-dependent inference kernels.
Conclusion in the testing literature: **cassette record/replay is the only
reliable determinism layer** — exactly agent-replay's architecture. ✅

But the same literature flags the failure mode agent-replay has (HANDOFF §2.1):
pure HTTP/positional-level replay "freezes the whole agent loop." The fix that
works in practice: **record decisions, key them by request content, and let
divergent paths run live** — i.e. idempotency-key cassette matching. One 2026
approach (parameterized replay) goes further: extract variables from recorded
calls and re-resolve them against live values on replay.

**Codebase changes:**
- Confirms idempotency-key matching (HANDOFF §2.1) as v0.2 priority #1.
- Add optional **fuzzy cassette matching** tier: exact key → (opt-in) match on
  `(kind, name)` with nearest-inputs (hash prefix or user-supplied matcher) →
  live resample. Mirrors vcrpy's `match_on=` configurability.
- Document explicitly: `seed`/temperature=0 must **not** be trusted for factual
  reproduction; the cassette is the source of truth (README section).

## 4. Shapley estimator: proven upgrades beyond antithetic pairing

The estimation literature (VRDS, arXiv 2210.16835; stratified permutation
sampling; KernelSHAP variance work) gives three drop-in improvements:

1. **Stratified permutation sampling with optimal allocation** — group
   permutations into strata (by the position at which step *i* is inserted) and
   allocate samples ∝ within-stratum marginal-contribution std-dev. Complements
   antithetic pairing; same sample-complexity order with lower constant.
2. **Online variance tracking → auto-stop** — KernelSHAP practice: maintain
   running variance of each φ̂ᵢ, stop sampling permutations when all CIs reach a
   target width. This replaces the fixed `permutation_pairs=8` guess with a
   `target_ci_width=` parameter.
3. **Position-stratified early allocation for PoC** — because the culprit is the
   *latest* significant step, allocate contrastive rollouts back-to-front and
   stop when the latest CI excluding zero is bracketed by later steps whose CIs
   contain zero. Big win: most trajectories resolve after ablating a suffix.

**Codebase changes (`attribution.py`, `stats.py`):**
- `shapley_attribution(..., target_ci_width=None)` — adaptive stopping loop
  around the permutation walk (keep the no-caching invariant; variance tracked
  over per-permutation marginals which are already stored in `marginals`).
- `contrastive_attribution(..., adaptive=True)` — Wilson-CI–driven sequential
  rollouts per step (stop early when CI decisively excludes/includes 0), and a
  back-to-front step order with early exit for PoC localization.

## 5. Staged attribution funnel (borrowing from FAMAS + zero-replay)

FAMAS ranks suspicious actions from *repeated executions* without targeted
interventions; zero-replay debugging predicts replay outcomes from an event
knowledge graph (Recall@5 0.73→0.93, zero replay cost).

**Codebase change — new module `prefilter.py` (v0.3+):**
1. **Stage 0 (free):** structural heuristics on the recorded trajectory — steps
   whose output changed downstream context most (edit distance between
   consecutive contexts), last tool call before failure, error-shaped outputs.
2. **Stage 1 (cheap):** FAMAS-style suspiciousness — run K *full* resamples
   (coalition = ∅), correlate per-step action values with failure across runs
   (a spectrum/Tarantula-style score). Costs K rollouts total, not K×steps.
3. **Stage 2 (causal):** run the existing contrastive + Shapley machinery only
   on the top-M suspicious steps; report the rest as "screened out (score, CI
   not evaluated)". `attribute(..., prefilter="spectrum", top_m=5)`.
This preserves causal guarantees where it matters and cuts cost ~steps/M-fold.
Never silently: the report must show what was screened.

## 6. Interop: OpenTelemetry GenAI + LangGraph checkpoints

OTel GenAI semantic conventions (experimental as of March 2026) are becoming the
de-facto trace format for agents (LangGraph/CrewAI/AutoGen auto-instrumentation;
Langfuse/Braintrust/LangSmith all ingest OTLP). LangGraph natively supports
checkpointer-based time travel (replay from a checkpoint, fork state).

**Codebase changes (v0.4, extends HANDOFF v0.4.3–4):**
- `agent_replay.interop.otel`: build a `Trajectory` from OTel GenAI spans
  (`gen_ai.*` attributes → Step kind/name/inputs/output). This makes agent-replay
  *connectable to any instrumented agent* without code changes — attribution
  then requires only a user resample fn per step kind.
- `agent_replay.interop.langgraph`: adapter that (a) records from a LangGraph
  checkpointer stream, and (b) uses LangGraph's own time-travel fork API as the
  replay substrate for LangGraph agents (their checkpointer *is* a cassette).
- Emit side: optional OTel span export of attribution results (span events with
  `attribution.score`, `attribution.ci`) so results land in existing dashboards.

## 7. Evaluation: prove it on the public benchmark

Who&When (ICML 2025 Spotlight, open dataset: 127 annotated MAS failure logs with
decisive-error-step labels) is the field's shared yardstick; FAMAS and others
report on it. LLM-judge step accuracy ≈14%; agent-replay's causal method should
be benchmarked there to make the headline claim credible.

**Codebase change (v0.5):** `benchmarks/whowhen/` — loader for the public
dataset, trajectory import (via §6 interop), attribution run, and a scored
comparison table (accuracy@step, accuracy@agent, cost in rollouts). Caveat to
handle: Who&When traces aren't natively replayable, so this exercises the
imported-trajectory + user-resample-fn path — which is precisely the hardest
"connect to any agent" scenario and worth hardening first.

---

## Updated priority order (merging with HANDOFF §4)

1. **v0.2 (unchanged):** idempotency-key cassette; passing-run guard; resamplable
   flags; serialization strictness; **+ synthetic-SCM ground-truth tests (§2)**.
2. **v0.3:** adaptive rollouts via online variance/Wilson stopping (§4) —
   *promoted above* edit-context because it multiplies every later feature's
   affordability; then staged prefilter funnel (§5); then edit-context/mock-observe.
3. **v0.4:** OTel GenAI import + LangGraph checkpointer bridge (§6) — this is
   the concrete mechanism behind "connectable to all types of agents"; async;
   auto-instrumentation.
4. **v0.5:** Who&When benchmark harness (§7); report v2; repair v2.

## Sources

- [Who&When: Which Agent Causes Task Failures and When? (arXiv 2505.00212)](https://arxiv.org/abs/2505.00212) · [code/dataset](https://github.com/ag2ai/Agents_Failure_Attribution)
- [Causal Agent Replay (arXiv 2606.08275)](https://arxiv.org/abs/2606.08275)
- [CausalFlow: Causal Attribution and Counterfactual Repair (arXiv 2605.25338)](https://arxiv.org/abs/2605.25338)
- [FAMAS: spectrum-analysis failure attribution (arXiv 2509.13782)](https://arxiv.org/pdf/2509.13782)
- [Knowledge-Based Zero-Replay Debugging (arXiv 2606.14805)](https://arxiv.org/pdf/2606.14805)
- [Seeing the Whole Elephant: failure-attribution benchmark (arXiv 2604.22708)](https://arxiv.org/html/2604.22708v1)
- [POIROT: Interrogating Agents for Failure Detection (arXiv 2606.02282)](https://arxiv.org/pdf/2606.02282)
- [VRDS: Variance-reduced Shapley via stratified sampling (arXiv 2210.16835)](https://arxiv.org/pdf/2210.16835)
- [Sampling Permutations for Shapley Value Estimation (arXiv 2104.12199)](https://arxiv.org/pdf/2104.12199)
- [Understanding and improving KernelSHAP (Covert)](https://iancovert.com/blog/kernelshap/)
- [Deterministic Replay for AI agents (TianPan, 2026)](https://tianpan.co/blog/2026-04-12-deterministic-replay-debugging-non-deterministic-ai-agents)
- [Toggle OpenAI Model Determinism (lakeFS)](https://lakefs.io/blog/toggle-openai-model-determinism/) · [LLM consistency in 2025 (KeywordsAI)](https://www.keywordsai.co/blog/llm_consistency_2025)
- [VCR tests for LLMs (Nayak)](https://anaynayak.medium.com/eliminating-flaky-tests-using-vcr-tests-for-llms-a3feabf90bc5) · [vcr-langchain](https://github.com/amosjyng/vcr-langchain)
- [OTel GenAI semantic conventions overview (Uptrace, 2026)](https://uptrace.dev/blog/opentelemetry-ai-systems) · [agent-observability RFC](https://github.com/traceloop/openllmetry/issues/3460) · [Langfuse OTel](https://langfuse.com/integrations/native/opentelemetry)
- [Replayable Financial Agents: determinism-faithfulness harness (arXiv 2601.15322)](https://arxiv.org/pdf/2601.15322)
