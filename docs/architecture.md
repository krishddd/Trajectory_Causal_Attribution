# Architecture & research mapping

This document maps the *Trajectory Causal Attribution* research (the
*Agent Time-Travel Debugger* / *Chronos Protocol* documents) onto the concrete
modules of `agent-replay`, and records the deliberate scope decisions.

## The question

When a long-horizon agent fails, observability tells you *that* it failed and
*where the harmful action executed* — but the fatal **decision** was usually made
earlier. The research frames failure attribution as a causal problem on a
**Structural Causal Model (SCM)** of the run:

```
trajectory = (s0, a0, o0), (s1, a1, o1), ..., (sT, aT, oT) -> outcome y
```

- `s_i` — the context the agent decides from at step `i` (prompt + history).
- `a_i` — the action drawn from the stochastic policy `π(a | s_i)`.
- `o_i` — the observation returned by the environment.
- `y`   — a scalar outcome from a user-supplied verifier (0.0 = fail).

To isolate step `i`'s causal impact we *intervene* on it and re-run forward. Since
`π` is stochastic, running forward `N` times yields an outcome **distribution**,
and the core estimand is:

```
attribution(step i) = P(fail | step i kept) − P(fail | step i ablated)
```

## Module map

| Research concept | Module | Notes |
|---|---|---|
| SCM: steps / trajectory | `types.py` | `Step`, `Trajectory`, `StepKind`. |
| Content-addressable Merkle state | `hashing.py`, `store.py` | SHA-256 CAS blobs, per-step `parent_hash → step_hash` chain. |
| Deterministic record | `recorder.py` | `RecordContext` runs each policy, captures outputs. |
| VCR/cassette replay | `replayer.py` | `ReplayContext` serves recorded outputs for *held* steps. |
| Intervention algebra | `replayer.py`, `types.py` | `resample`, `do` (force), `mock_observe`, `remove`. |
| Stochastic run-forward | `ablation.py` | `AblationEngine` runs `N` seeded rollouts per plan. |
| Single-step contrastive estimator | `attribution.py` | `contrastive_attribution`. |
| Point-of-Commitment Rule | `attribution.py` | `point_of_commitment`. |
| Shapley-value attribution | `attribution.py` | `shapley_attribution` + antithetic pairing. |
| Wilson + bootstrap intervals | `stats.py` | `wilson_interval`, `bootstrap_diff_interval`. |
| Minimal counterfactual repair | `repair.py` | `find_minimal_repair`, `minimality`. |
| Diagnostic report | `report.py`, `types.py` | HTML + JSON. |
| SQLite checkpoint store | `store.py` | sessions / steps / blobs / attributions. |
| CLI | `cli.py` | `record` / `replay` / `attribute` / `report`. |
| Adapters | `adapters/` | LangChain callback, OpenAI SDK wrapper. |

## The intervention algebra

A `ReplayPlan` decides, per step index, one of:

- **hold** — serve the recorded output verbatim (deterministic replay / cassette).
- **resample** — re-run the live policy (a fresh draw for this rollout).
- **force** (`do`) — override with a fixed action.
- **remove** — drop the step (empty action).

The single-step contrastive intervention for step `i` is
`ReplayPlan.ablate_from(i)`: **hold `< i`** at their factual recorded actions and
**resample `≥ i`**. Resampling step `i` necessarily re-rolls every downstream
stochastic step — the SCM is a sequential dependency chain — and the plan models
exactly that.

## Why magnitude fails, and the Point-of-Commitment Rule

Because resampling an early step re-rolls the fatal late step too, an early,
irrelevant step can produce a large, deceptive shift in the outcome distribution
(a butterfly effect). **Magnitude alone blames early steps.**

The fix: traverse the trajectory backwards and take the **latest** step whose
confidence interval on the attribution still strictly excludes zero. That is the
final juncture at which re-deciding the action can still systematically rescue the
run — the true causal locus. Beyond it, the failure is structurally locked in and
the interval brackets zero.

`agent-replay` reports a **bootstrap interval on the difference** of failure rates
as each step's CI (numerically exact at the boundary `P(fail|ablated)=1`), and
exposes the **Wilson score interval** for the ablated proportion on its own in
`stats.py`.

## Shapley attribution

Single-step ablation treats steps as independent, so it double-counts credit for
**AND-failures** (effect scores sum to ≈2.0) and zeroes it out for
**OR-failures**. Shapley values restore fair, additive credit:

- Sample random permutations; for each, walk the coalition empty → full, adding
  one step at a time and recording that step's **marginal** change in failure rate.
- **Antithetic reverse-permutation pairing** — each sampled permutation is paired
  with its exact reverse — reduces estimator variance.
- Coalition values are **never cached**: caching would collapse the per-step
  marginal variance to zero and yield falsely narrow intervals.
- **No truncation** — truncated Monte-Carlo Shapley is prone to skipping pivotal
  late steps that define the point of commitment.

By the efficiency axiom, `Σ φ_i = v(full) − v(empty)` — the difference in failure
probability between the fully factual and the fully resampled trajectory.

## Minimal counterfactual repair

Given the culprit step, `repair.py` injects candidate replacement actions via a
`do` intervention and re-runs forward. A candidate is a **valid repair** if it
drops the failure rate below threshold; among valid repairs we pick the one with
maximum **minimality** (least behavioural drift from the original action, measured
as a normalised edit similarity over canonical JSON). The result is a validated
`(wrong step → minimal fix)` pair — a deploy-time recovery hint and a
learning-ready supervision signal.

## Determinism model

A run is fully determined by `(recorded cassette, seed)`. All agent randomness
must flow through `ctx.rng` (a seeded `random.Random`). Held steps consume no
randomness (their outputs are served from the cassette); resampled steps draw
fresh values seeded per rollout, giving independent samples for the estimators.

## Scope: what is intentionally out

The source research also specifies OS-level substrates for capturing full
process/filesystem state at millisecond latency:

- **DeltaFS / DeltaCR** (DeltaBox) — overlayfs hot-layer switching + `fork()`
  memory templates for ~10 ms checkpoint / ~2 ms restore.
- **CRIU** — userspace process checkpoint/restore.
- **Firecracker microVMs** and **WASI-Virt** WebAssembly component virtualisation.
- **FUSE / Merkle-DAG filesystems**, network `datetime`/RNG virtualisation.

`agent-replay` implements the **application-level** essence: determinism comes
from recorded cassettes and seeded RNG, not kernel snapshots. The causal
attribution mathematics — the actual contribution of this library — does not
depend on those primitives, and the wrapper API stays framework- and
OS-agnostic. Teams that need true environment-level time travel can pair
`agent-replay`'s attribution engine with a microVM/Wasm substrate later without
changing the attribution code.
