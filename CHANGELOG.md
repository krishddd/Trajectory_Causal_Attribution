# Changelog

All notable changes to `agent-replay` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

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
