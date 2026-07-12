# Missing-subject audit vs "Architecting the Agent Multiverse" (slide deck)

Source: `Architecting_the_Agent_Multiverse.pdf` (15 slides, NotebookLM deck of the
Chronos Protocol / CausalFlow research). This maps every subject in the deck to
the current library (v0.4.0) and specifies the **missing features** with enough
context to implement each. Companions: `HANDOFF.md`, `ANALYSIS.md`.

## Slide-by-slide coverage

| # | Slide subject | Status in v0.4.0 |
|---|---|---|
| 1 | Multiverse DAG of forked executions | ⚠️ partial — see Gap 1 |
| 2 | Entropy of autonomy (alignment decay over 70h) | ✅ `drift()` entropy/health curve — Gap 6 |
| 3 | Diagnostic trap (14% LLM-judge; hand vs brain) | ✅ the library's premise |
| 4 | Execution-as-Code ("model may change, history must not") | ✅ cassette + seeds |
| 5 | Workflow/Activity dichotomy, event ledger, replay mode | ✅ record/replay; ⚠️ resume — Gap 5 |
| 6 | OS-level substrates (DeltaBox/CRIU/Wasm table) | ✅ documented out-of-scope |
| 7 | Execution Merkle DAG, CAS, **O(1) branching**, node = parent/action/output/fs hashes | ⚠️ linear chain only — Gap 1; action/output hash split — Gap 7 |
| 8 | Network determinism: VCR, **virtual time**, **seeded entropy (uuid/rng)** | ⚠️ rng only — Gap 2 |
| 9 | SCM (C_t → A_t → O_t → Y) + attribution formula | ✅ |
| 10 | do() calculus tracks (null vs forced intervention) | ✅ |
| 11 | Stochastic confound + Point-of-Commitment | ✅ |
| 12 | Shapley (AND/OR gates) + antithetic pairing | ✅ ground-truth tested |
| 13 | **Step-level faithfulness: causality vs coverage quadrant** | ❌ — Gap 3 |
| 14 | Repair loop (isolate → inject → validate, minimality) | ✅ repair v2 |
| 15 | **Multiverse Console**: timeline graph, branch compare, REPL in frozen state, audit trail | ❌ — Gaps 1, 4, 5 |

## The gaps, with implementation context

### Gap 1 — The Multiverse itself: first-class forking & persisted branches
**Deck:** slides 1, 7, 15 — "forking an agent at Step 45 creates a new node with a
shared parent hash… instantly instantiates a parallel universe"; the console
compares Failed vs Fixed timelines side by side.
**Today:** forking exists *implicitly* (every ablation rollout is a transient
fork via `ReplayPlan`), but nothing is user-facing or persisted: sessions are
linear; a counterfactual run cannot be saved, named, listed, or diffed.
**Build (v0.5 candidate — highest value of these):**
- `fork(trajectory, at_step, *, do=None, seed) -> Trajectory` in a new
  `multiverse.py`: run `ReplayPlan(held=<prefix, optionally forced>)`, record the
  live continuation with a *branch-aware* recorder, return a new `Trajectory`
  whose `meta` carries `parent_session`, `fork_step`, `intervention`.
- Store: add `parent_session` / `fork_step` columns (nullable, migration like
  `resamplable`); `CheckpointStore.branches(session_id)` lists children.
  Storage is already O(1)-ish: forked steps dedupe through the CAS blobs.
- `diff(traj_a, traj_b)` → first divergent step + per-step action diff — the
  "State Diff" the research UI shows. CLI: `agent-replay fork` / `branches` / `diff`.

### Gap 2 — Virtual time & entropy (clock/uuid determinism)
**Deck:** slide 8 — "hook system calls (wasi:clocks/get_time) to inject logged
timestamps; seed RNGs deterministically".
**Today:** only `ctx.rng` is deterministic. An agent calling `datetime.now()`,
`time.time()` or `uuid.uuid4()` silently breaks replay fidelity (held steps are
fine, but resampled/live paths diverge from wall-clock and fresh uuids, and any
recorded output embedding a timestamp changes its idempotency key across runs).
**Build (small, high leverage):**
- `ctx.now()` and `ctx.uuid()` on both contexts: during recording they capture
  real values *as steps* (kind=`memory`, name=`__now__`/`__uuid__`,
  resamplable=False); during replay they serve the recorded values — no API
  change to the engine, they ride the existing cassette.
- `instrument.RECIPES["time"]` patching `time.time`/`datetime.datetime.now` and
  `uuid.uuid4` onto those ctx ops for unmodified agents (opt-in, since global).

### Gap 3 — Step-level faithfulness (causality vs coverage)
**Deck:** slide 13 — correctness ⊥ faithfulness quadrant (correct-but-unfaithful
= "dangerous post-hoc rationalization"; wrong-but-faithful = "ideal debugging
target"); FaithCoT-style masking.
**Today:** the verifier scores *outcome only*; nothing measures whether the
recorded reasoning steps actually drive the answer.
**Build (v0.5/v0.6, research-grade feature):** `faithfulness.py` —
- **Causality score** per reasoning step: mask/remove step i (existing `remove`
  intervention), hold everything else, N rollouts → shift in outcome
  distribution; AUC over progressive masking = trajectory faithfulness.
- Classify runs into the 2×2 (correct×faithful) using outcome score ×
  faithfulness score; flag correct-unfaithful runs in the report ("passed, but
  the reasoning did not cause the answer") — these are the silent risks.
- Reuses `AblationEngine` wholesale; mostly a new scorer + report section.

### Gap 4 — Multiverse Console (serve UI) with branch graph & frozen-state inspection
**Deck:** slide 15 — timeline graph with counterfactual branches, reverse-step
navigation, "Interactive REPL inside frozen states".
**Today:** static HTML report only.
**Build (already roadmapped as v0.5 `agent-replay serve`; the deck adds specifics):**
stdlib `http.server` browsing sessions → per-step state view (inputs/outputs =
the app-level "frozen state"), branch graph once Gap 1 lands, and a
"step back/forward" inspector. A true REPL into process memory is OS-substrate
territory (out of scope) — the app-level equivalent is dumping the step's
recorded inputs/outputs and context digest.

### Gap 5 — Durable resume (event-sourcing recovery)
**Deck:** slide 5 — replay = "fast-forward through days of successful operation
in milliseconds" to *resume* a crashed run, not only to analyse it.
**Today:** replay serves attribution; there is no "continue this run live from
step k" convenience.
**Build:** thin wrapper over Gap 1's `fork`: `resume(trajectory) =
fork(trajectory, at_step=len(trajectory), seed=...)` — hold the entire recorded
prefix, run live from the end. Cheap once fork exists.

### Gap 6 — Drift/entropy observability (nice-to-have)
**Deck:** slide 2 — alignment decay curve, context drift, silent failure.
**Today:** nothing tracks per-step health over long runs.
**Build (optional):** if the verifier can score *intermediate* state, record a
per-step score series and chart it in the report — the "entropy curve" for a
run. Low priority; useful mainly for long-horizon demos.

### Gap 7 — Minor: split action vs output hash in the Merkle node
**Deck:** slide 7 — node = parent + action hash + output hash (+ fs root hash).
**Today:** `step_hash = link(parent, op_key, output_hash)` — the parts exist but
only the combined hash is stored. Persisting `action_hash`/`output_hash`
separately enables "same decision, different observation" queries and cheaper
diffing (Gap 1). Trivial store migration.

## Priority recommendation
1. **Gap 2** (virtual time/uuid) — small, closes a real determinism hole.
2. **Gap 1** (fork/branches/diff) — the deck's headline; unlocks Gaps 4 & 5.
3. **Gap 5** (resume) — nearly free after Gap 1.
4. **Gap 4** (serve UI w/ branch graph) — already v0.5 roadmap.
5. **Gap 3** (faithfulness) — differentiating research feature.
6. Gaps 6–7 — opportunistic.

## Status (v0.6.0) — deck fully covered
- ✅ Gap 1 — `multiverse.fork`/`afork`/`diff`, `CheckpointStore.branches`, CLI.
- ✅ Gap 2 — `ctx.now`/`ctx.uuid`, `instrument.enable_virtual_time`.
- ✅ Gap 3 — `faithfulness()` + quadrant classification, CLI.
- ✅ Gap 4 — `serve.py` Multiverse Console, `agent-replay serve`.
- ✅ Gap 5 — `multiverse.resume`.
- ✅ Gap 6 — `drift()` per-step entropy-of-autonomy curve + optional
  intermediate-state `state_scorer` alignment-health overlay, silent-decay
  detection, `DriftResult.to_html` SVG chart, CLI `drift`. `stats.binary_entropy`.
- ✅ Gap 7 — `Step.action_hash`/`output_hash`.

Every slide subject in the deck now maps to a shipped capability. Gap 6 resolved
its "needs an intermediate-state scorer" caveat by (a) providing the
`state_scorer` hook for callers who have one and (b) always charting the
verifier-only entropy-of-autonomy curve when they don't.
