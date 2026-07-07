# agent-replay — Verification Report & v0.2→v1.0 Upgrade Handoff

> **Purpose of this document.** This is a complete development-context handoff for the
> next engineer/model working on `agent-replay`. It records (1) what the current
> v0.1.0 implements and how faithfully it follows the research, (2) **verified bugs
> with reproduction evidence**, (3) the invariants any fix must preserve, and (4) an
> ordered upgrade roadmap with concrete designs. Read `docs/architecture.md` and the
> two research .docx files first; this file assumes their vocabulary (SCM,
> Point-of-Commitment, Shapley antithetic pairing, VCR cassette, Merkle CAS).

---

## 1. Verification verdict (v0.1.0)

**Status: core algorithm correct for linear (fixed step-sequence) agents. 56/56 tests
pass, ruff clean, wheel builds. One structural unsoundness for branching agents
(§2.1) and several semantic gaps.**

| Research requirement | Implemented? | Where | Verdict |
|---|---|---|---|
| SCM formalization (state→action→obs→outcome) | ✅ | `types.py` | Faithful |
| Attribution = P(fail\|kept) − P(fail\|ablated) | ✅ | `attribution.py:contrastive_attribution` | Faithful |
| Hold `<i` factual, resample `≥i`, N rollouts | ✅ | `replayer.py:ReplayPlan.ablate_from`, `ablation.py` | Faithful for prefix plans |
| Point-of-Commitment Rule (latest step, CI excludes 0) | ✅ | `attribution.py:point_of_commitment` | Faithful; verified to localize known culprit |
| Butterfly-confound avoidance (not max-magnitude) | ✅ | culprit selection prefers PoC | Faithful |
| Wilson interval on rollout proportion | ✅ (available) | `stats.py:wilson_interval` | Implemented + tested, but reported CI is bootstrap-diff only; Wilson not surfaced in reports |
| Bootstrap CI on the difference | ✅ | `stats.py:bootstrap_diff_interval` | Faithful |
| Shapley via MC permutations | ✅ | `attribution.py:shapley_attribution` | Correct math, **unsound for branching agents** (§2.1) |
| Antithetic reverse-permutation pairing | ✅ | `_antithetic_permutations` | Faithful |
| No coalition-value caching (preserve marginal variance) | ✅ | fresh `seed_tag` per evaluation | Faithful |
| No truncated MC Shapley | ✅ | full walk every permutation | Faithful |
| Minimal counterfactual repair + minimality metric | ✅ | `repair.py` | Faithful (edit-similarity proxy for token distance) |
| Intervention algebra: `resample` | ✅ | plan default | Faithful |
| Intervention algebra: `do` (force action) | ✅ | `ReplayPlan.forced` | Faithful |
| Intervention algebra: `mock-observe` | ⚠️ | enum exists, no distinct mechanics (aliased by `force`) | Gap — see §3.4 |
| Intervention algebra: `edit-context` | ❌ | not implemented | Gap — v0.3 |
| Intervention algebra: `swap-model` | ❌ | not implemented | Gap — v0.4 |
| VCR cassette / deterministic replay | ⚠️ | `replayer.py` — **positional index matching, not idempotency keys** | The paper specifies "recorded responses injected based on idempotency keys derived from input hashes". Current code matches by call *position*. Root cause of §2.1. |
| Merkle-linked steps + CAS dedup | ✅ | `hashing.py`, `store.py` | Faithful (application-level) |
| Event sourcing / append-only history | ✅ | SQLite steps table | Faithful |
| OS-level checkpoints (DeltaFS/CRIU/WASI) | ❌ by design | — | Documented out-of-scope; correct decision |
| HTML+JSON diagnostic report (3-phase format) | ✅ | `report.py`, `cli.py:_print_summary` | Faithful |

---

## 2. VERIFIED BUGS (with reproduction evidence)

### 2.1 ⛔ CRITICAL — Positional cassette matching cross-contaminates branching agents

**Evidence (reproduced):** a router agent whose step-1 tool depends on step-0's output.
Record with route=B (`fetch_b` at index 1). Resample step 0 only, hold {1,2}:

```
seed 1: route=A x=data_b  CROSS-CONTAMINATED=True   # fetch_a received fetch_b's output
seed 3: route=A x=data_b  CROSS-CONTAMINATED=True
```

`ReplayContext._op` serves `trajectory.steps[idx].output` purely by call index. When a
resampled upstream step changes the control flow, downstream *held* indices serve the
wrong step's recorded output into a different operation.

**Blast radius:**
- Prefix plans (`ablate_from(i)`: hold `{0..i-1}`) are **safe** — the held prefix
  deterministically reproduces the recorded control flow up to `i`.
  → Phase-1 contrastive attribution + PoC are sound even for branching agents. ✅
- **Coalition plans (Shapley) hold non-contiguous sets** → for branching agents,
  Shapley values are computed on contaminated rollouts. ⛔ Currently only correct for
  linear agents (like the bundled mock).
- Repair plans hold a prefix + force one step → safe. ✅

**Required fix (v0.2, matches the paper's own design):** idempotency-key matching.
- Cassette key = `content_hash({kind, name, inputs})` (already computable via
  `hashing.content_hash`; inputs are recorded on every `Step`).
- `ReplayContext._op` for a *held* decision: look up the key among not-yet-consumed
  recorded steps (multiset — same key may repeat; consume in recorded order).
  Key found → serve recorded output. Key miss → the timeline has diverged at this
  point; **fall back to live resample** and set `self.diverged = True`.
- Keep positional matching as a fast path when the plan is a pure prefix.
- `ReplayPlan` semantics change: `held` becomes "serve from cassette *if the same
  operation occurs*", which is exactly the paper's cassette semantics.
- Add regression test: the branching-router agent above; assert no
  cross-contamination for any coalition plan; assert Shapley on a branching agent
  with a known culprit still localizes it.
- Caveat to document: for `held` steps whose *inputs* change because an upstream
  resample altered them, the key will miss and the step re-runs live. That is the
  *correct* causal semantics (the step's factual action is only meaningful in its
  factual context).

### 2.2 HIGH — Attributing a PASSING run produces unguarded noise

**Evidence (reproduced):** attribute() on a trajectory with outcome 1.0 returns
`failed=False` with attributions like `[(0,-0.65),(1,-0.9),(2,-0.8),(3,-0.8)]` —
negative "attribution" values that are really *credit* (the kept steps prevent
failure), rendered in a failure report with no warning.

**Fix:** in `attribute()`, if `not failed`: either (a) raise `ValueError` unless
`allow_success=True`, or (b) switch to explicit **credit mode** — rename the score
"rescue credit = P(fail|ablated) − P(fail|kept)", flip the PoC rule to the latest step
whose *credit* CI excludes zero (the "save point"), and label the report accordingly.
Option (b) is a genuinely useful feature ("which step saved this run") — recommended.

### 2.3 MEDIUM — LangChain adapter silently breaks resampling

`AgentReplayCallbackHandler.on_llm_end` records with `produce=lambda: text` — the
*observed* text captured in a closure. Under a resample plan, "re-running the policy"
just returns the same recorded text: **all LangChain-recorded steps are effectively
deterministic**, so every attribution collapses toward 0 with no error. A user gets a
report full of zeros and no explanation.

**Fix:** tag steps recorded via observation-only adapters as `resamplable=False`
(new field on `Step`). `AblationEngine` must then either (a) skip them as candidates
and report them as "observed-only, not attributable", or (b) require the user to
supply a `resample_fn` per step name. The report must surface non-resamplable steps
explicitly. Same applies to any `produce=None` step (currently returns `None` on
resample — silently corrupting rollouts).

### 2.4 MEDIUM — Round-trip infidelity for non-JSON payloads is silent

`hashing.canonical_json(default=repr)` and `store.put_blob(json.dumps(default=str))`
mean a recorded numpy array / dataclass / datetime is stored as a string; on
`load_trajectory` + replay, held steps serve the *string* where the live agent
produced an object. No warning anywhere.

**Fix:** validate at record time — `RecordContext._op` should attempt
`json.loads(json.dumps(output))` round-trip; on mismatch raise
`NonSerializableStepError` with the step name (strict mode, default) or warn +
mark step `lossy=True` (lax mode). Also add optional pickle-blob storage for opaque
payloads (with a `--unsafe-pickle` opt-in given pickle's security model).

### 2.5 LOW — Dead/misleading code
- `ablation.py`: `self._seed_salt = 0` — written, never read. Remove.
- `replayer.py`: `ReplayPlan.ablate_from(i, n)` — `n` unused. Remove param or use it
  to pre-validate `i < n`.
- `types.py`: `InterventionKind.MOCK_OBSERVE` / `RESAMPLE` enum values are never
  referenced by the engine (plans use string decisions). Either wire the enum through
  `ReplayPlan.decision()` or drop the enum until §3.4.
- `replayer.py`: `REMOVED = None` sentinel is indistinguishable from a legitimate
  `None` output. Use a dedicated `class _Removed: ...` singleton.
- `attribution.py:attribute()`: `poc = ... if steps and steps[0].p_fail_kept is not None`
  — `p_fail_kept` is always a float (0.0 in shapley mode), so the guard is dead and
  PoC is (incidentally) evaluated on Shapley CIs in `method="shapley"`. Make it
  explicit: PoC is defined **only** for contrastive; in shapley-only mode set
  `point_of_commitment=None` deterministically.
- `stats.py:bootstrap_diff_interval` resamples `kept` even when it is a single
  deterministic observation (harmless, wasteful). Short-circuit `len(kept)==1`.
- Weak test: `test_shapley_efficiency_axiom` only asserts the sum ∈ [0,1]. Strengthen:
  compare `sum(φ)` against an independently estimated `v(full) − v(empty)` within a
  tolerance (e.g. ±0.15 at 40 rollouts).

### 2.6 LOW — Operational gaps
- `CheckpointStore` uses one connection, no WAL, not thread/process-safe.
- CLI does not expose `--fail-threshold`, `--base-seed`, `--bootstrap-iterations`.
- No `agent-replay list` / `inspect` subcommands (store has `list_sessions` already).
- OpenAI adapter resample calls the real paid API with the original `temperature`
  (temperature=0 ⇒ zero resample variance ⇒ zero attribution). Warn when
  temperature==0; document cost implications; add a cache/live/hybrid mode flag.

---

## 3. Invariants any change MUST preserve (regression gates)

1. **The estimand.** `attribution(i) = P(fail|kept) − P(fail|ablated-from-i)` with
   prefix-hold semantics for Phase 1. Never change what is estimated, only how.
2. **PoC over magnitude.** Culprit = latest step whose CI strictly excludes zero.
   `tests/test_attribution.py::test_point_of_commitment_localises_culprit` is the gate.
3. **No coalition caching, no Shapley truncation, antithetic pairing on.**
4. **Determinism:** run ≡ f(cassette, seed). All agent randomness via `ctx.rng`;
   engine seeds via `seed_tag`; identical plan+tag ⇒ identical rollouts.
5. **Write-once agent contract:** the same user function must run unchanged under
   `RecordContext` and `ReplayContext`. Never require record-only or replay-only code.
6. **Zero runtime dependencies** in the core (`pip install agent-replay` pulls nothing).
7. **Full suite green** (`pytest -q`), `ruff check` + `ruff format --check` clean,
   `python -m build` succeeds, Python 3.9–3.12.

---

## 4. Upgrade roadmap (ordered; each milestone independently shippable)

### v0.2 — Soundness ✅ SHIPPED (v0.2.0)
1. ✅ **Idempotency-key cassette matching** (§2.1) — `Step.op_key` binding in
   `replayer.py`; legacy `match="position"` retained. Branching agents now sound.
2. ✅ Passing-run guard + credit mode (§2.2) — `attribute(on_success=...)`,
   `SuccessfulRunError`, `AttributionResult.mode`.
3. ✅ `resamplable` step flag + non-resamplable serving/reporting (§2.3).
4. ✅ Serialization strictness (§2.4) — `NonSerializableStepError`.
5. ✅ Dead-code cleanup (§2.5) — `_seed_salt` removed, `_Removed` sentinel.
6. ✅ CLI flags + `list` command (§2.6).
7. ✅ New tests: `test_branching.py` (cross-contamination regression + position-mode
   contrast), `test_synthetic_scm.py` (AND/OR credit split, quantitative efficiency
   axiom, pivotal-step localisation), `test_soundness.py` (guard/credit/strict/flag).

   Result: **75 tests pass**, ruff clean, wheel builds. Remaining low-priority
   items from §2.5/§2.6 still open: `ablate_from`'s unused `n` (kept for API
   symmetry), OpenAI temperature==0 warning, WAL/thread-safety.

### v0.3 — Coverage of the paper's algebra
1. **`edit-context` intervention:** record `s_i` (a digest of the visible context) per
   step; allow plans to rewrite recorded *inputs* served to a held step (tests memory
   reliance). Requires cassette keys from §2.1 (edited inputs ⇒ new key ⇒ documented).
2. **`mock-observe` as first-class:** distinct from `do` — replaces the *observation*
   of a tool while keeping the action; needs action/observation split in `Step`
   (currently conflated in `output`). Schema change: `Step.action`, `Step.observation`
   with backward-compat loader.
3. **Adaptive rollouts (big cost win):** sequential testing (Wilson CI width target or
   SPRT) instead of fixed N — stop early per step when CI excludes/includes zero
   decisively. Typical 3–10× fewer agent executions; directly addresses the paper's
   "budget-bounded" estimation.
4. **Parallel rollouts:** `concurrent.futures` (thread pool for I/O-bound produce fns,
   process pool optional). Requires per-rollout independent `random.Random` (already
   true) and a thread-safe store (open connection per thread or queue writes).

### v0.4 — Real-agent ergonomics
1. **Async support:** `async def` agents, `await ctx.llm(...)`, async produce fns —
   most production agents are async. Mirror `AgentContext` → `AsyncAgentContext`.
2. **`swap-model` intervention:** a plan-level `model_override` handed to produce fns
   via `ctx.model_hint`, so agents parameterized on model id can be counterfactually
   upgraded ("would gpt-X have succeeded from step i?").
3. **Auto-instrumentation adapters** (the "connect any agent" goal):
   - OpenAI: monkeypatch mode (`agent_replay.instrument.openai()`) so *unmodified*
     agents record — no ctx-threading needed; ctx carried via `contextvars`.
   - Anthropic SDK adapter (messages.create) — same pattern.
   - LangGraph checkpointer bridge; LangChain resample via re-invoking the bound LLM
     with recorded prompts (fixes §2.3 properly).
   - MCP tool-call interceptor.
   - Generic `@record_step(kind, name)` decorator for any function.
4. **Trajectory import:** build a `Trajectory` from OpenTelemetry/OpenInference traces
   or a JSONL of steps, so attribution can run on trajectories recorded elsewhere
   (replay then needs user-supplied resample fns per step kind — ties into §2.3's
   `resample_fn` registry).

### v0.5 — Product surface
1. Report v2: show P(fail|ablated) Wilson CIs, per-step payloads (collapsible), the
   timeline with branch/fork display, non-resamplable step badges, credit mode.
2. `agent-replay serve` — tiny stdlib http.server UI listing sessions, drilling into
   steps, launching attributions (keep zero-dep: no flask).
3. Repair v2: LLM-generated repair candidates (user supplies a `propose_fn`),
   token-level minimality, contrastive-pair export (JSONL) for DPO/RLHF — the paper's
   "learning-ready supervision data".
4. Multi-trajectory aggregation: attribute over K failing sessions of the same task;
   report step-name-level (not index-level) aggregate blame.

### v1.0 — Scale & trust
- Schema versioning + migrations in the store; WAL mode; optional S3/CAS backend.
- Property-based tests (hypothesis) on plan algebra; fuzz the cassette matcher.
- Benchmark suite: attribution accuracy on synthetic AND/OR/chain failure agents at
  varying rollout budgets (accuracy-vs-cost curves like the paper's tables).
- Docs site; PyPI publish workflow (trusted publishing) appended to ci.yml.

---

## 5. Context for the next developer (how the pieces connect)

- Any agent integrates by routing nondeterministic ops through the context handle:
  `ctx.llm/tool/memory(name, produce=policy_fn, **inputs)`; randomness via `ctx.rng`.
  The SAME function runs under `RecordContext` (executes `produce`, captures `Step`s
  into a Merkle-linked list) and `ReplayContext` (consults a `ReplayPlan` per call:
  hold→serve cassette, resample→run `produce`, force→inject, remove→sentinel).
- `AblationEngine.run_plan(plan, N, seed_tag)` = N seeded executions → failure flags
  via `verifier(result) < fail_threshold`. `attribute()` orchestrates Phase 1
  (contrastive + PoC), Phase 2 (Shapley), optional repair, and returns
  `AttributionResult` → `report.py` (HTML) / `to_json`.
- Storage: `CheckpointStore` (SQLite) — sessions/steps reference deduplicated blobs
  by SHA-256 (`hashing.content_hash` over canonical JSON). `Session` is the facade.
- CLI (`cli.py`) loads the user's agent+verifier as `module:function` entrypoints —
  attribution *requires* re-executing the agent, so a stored trajectory alone is only
  enough for factual replay, never for counterfactuals.
- Everything is stdlib-only; keep it that way in core. Extras go under
  `[project.optional-dependencies]`.

**Definition of done for any milestone:** all §3 gates green + new regression tests
for each fixed bug + README/architecture.md updated + version bumped.
