# agent-replay

**Find *which step* caused your AI agent to fail — with causal proof, not correlational guesswork.**

`agent-replay` records an AI agent's trajectory (LLM calls, tool calls, memory
operations) as a checkpointed, deterministically-replayable session, then
attributes a failure to a specific step by **counterfactual step-ablation**:
re-run the trajectory with one step perturbed and measure how the failure
probability shifts.

```
attribution(step i) = P(fail | step i kept) − P(fail | step i ablated)
```

This is the software-level realization of the *Trajectory Causal Attribution*
method: formalize the run as a Structural Causal Model, intervene on one step at
a time, and use the **Point-of-Commitment Rule** and **Shapley-value attribution**
to localize the true root cause — instead of blaming the step that mechanically
executed the harmful action (a category error) or an LLM-as-judge (≈14% accuracy
on the *Who&When* benchmark).

- 🎯 **Causal, not correlational** — real `do()`-calculus interventions on a replayable trajectory.
- 🧩 **Framework-agnostic** — a tiny decorator/wrapper API that works with *any* Python agent. Optional LangChain / OpenAI-SDK adapters included.
- 💾 **Checkpointed** — SQLite store with content-addressable, deduplicated blobs and a Merkle-linked step chain.
- 🔁 **Deterministic replay** — the VCR/cassette pattern: recorded steps are served verbatim; only ablated steps re-run.
- 📊 **Rigorous** — Wilson score + bootstrap confidence intervals, antithetic Shapley sampling, no coalition caching.
- 🛠 **Actionable** — searches for a *minimal counterfactual repair* and emits an HTML + JSON failure-attribution report.
- 🪶 **Zero runtime dependencies** — pure Python standard library.

---

## Install

```bash
pip install agent-replay
# optional integrations:
pip install "agent-replay[langchain]"
pip install "agent-replay[openai]"
```

Requires Python 3.9+.

---

## Quickstart (20-line integration)

```python
from agent_replay import Session, attribute

def my_agent(ctx, question):
    plan  = ctx.llm("plan",   produce=lambda: {"q": question}, prompt=question)
    hits  = ctx.tool("search", produce=lambda: ("bug" if ctx.rng.random() < 0.7 else "ok"), q=plan["q"])
    draft = ctx.llm("write",  produce=lambda: hits, context=hits)
    return {"answer": draft, "ok": draft == "ok"}

def verifier(result):                       # 1.0 == success, 0.0 == failure
    return 1.0 if result["ok"] else 0.0

with Session("demo.sqlite") as session:
    traj   = session.record(my_agent, {"question": "why did it fail?"}, seed=3, verifier=verifier)
    result = attribute(traj, my_agent, verifier, rollouts=60, method="both", repair=True)
    result.to_html("report.html")
    result.to_json("report.json")
    print("culprit step:", result.culprit_index)   # -> the step that caused the failure
```

The only thing your agent has to do is route its non-deterministic work through
the context handle — `ctx.llm(...)`, `ctx.tool(...)`, `ctx.memory(...)` — passing
a `produce` callable that *is* the policy for that step, and drawing any
randomness from `ctx.rng`. The **same function** is used for recording and for
every counterfactual rollout; that is what makes attribution possible.

---

## The CLI

```bash
# 1. record a factual run into a checkpoint store
agent-replay record --db demo.sqlite --session run1 \
    --agent agent_replay.mock_agent:buggy_agent \
    --verifier agent_replay.mock_agent:verifier --seed 1

# 2. deterministically replay it (fast-forward through the recorded decisions)
agent-replay replay --db demo.sqlite --session run1 \
    --agent agent_replay.mock_agent:buggy_agent \
    --verifier agent_replay.mock_agent:verifier

# 3. attribute the failure + generate reports + propose a repair
agent-replay attribute --db demo.sqlite --session run1 \
    --agent agent_replay.mock_agent:buggy_agent \
    --verifier agent_replay.mock_agent:verifier \
    --rollouts 60 --method both --repair --out report

# 4. regenerate the HTML report from a saved JSON report
agent-replay report --json report.json --out report.html
```

Sample output:

```
============================================================
TRAJECTORY CAUSAL ATTRIBUTION REPORT
============================================================
Session:      run1
Total steps:  6
Outcome:      FAILED (verifier score: 0.000)
Method:       both  (60 rollouts/step)
------------------------------------------------------------
Point-of-Commitment: step 3
[RESULT] Failure attributed to step 3 (tool tool_step_3) with score 0.700.
         CI [0.560, 0.820]
[REPAIR] step 3: 'BAD' -> 'OK' (valid, minimality 0.400, P(fail)->0.000)
============================================================
```

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │                Your agent                    │
                          │   ctx.llm(...) / ctx.tool(...) / ctx.memory   │
                          └───────────────┬──────────────────┬───────────┘
                                          │ record           │ replay (per rollout)
                                          ▼                  ▼
        ┌───────────────┐        ┌────────────────┐   ┌──────────────────┐
        │  RecordContext │  ───►  │   Trajectory   │   │  ReplayContext   │
        │ runs the policy│        │ (SCM: steps =  │   │ serves cassette  │
        │ captures steps │        │ state→action→  │   │ OR resamples per │
        └───────┬────────┘        │  obs→outcome)  │   │  ReplayPlan      │
                │                 └───────┬────────┘   └────────┬─────────┘
                ▼                         │                     │
    ┌──────────────────────┐             │                     ▼
    │   CheckpointStore     │◄────────────┘         ┌────────────────────────┐
    │  SQLite + CAS blobs   │  save / load          │     AblationEngine     │
    │  Merkle step chain    │                       │ N stochastic rollouts  │
    └──────────────────────┘                        │  per intervention      │
                                                    └───────────┬────────────┘
                                                                ▼
                                              ┌──────────────────────────────────┐
                                              │        AttributionScorer          │
                                              │  Phase 1: contrastive estimator   │
                                              │     + Point-of-Commitment Rule    │
                                              │  Phase 2: Shapley (antithetic)    │
                                              │  Wilson + bootstrap CIs           │
                                              └───────────────┬───────────────────┘
                                                              ▼
                                              ┌──────────────────────────────────┐
                                              │   Repair search (minimality)      │
                                              │   HTML + JSON attribution report  │
                                              └──────────────────────────────────┘
```

### How the causal attribution works

1. **Record** the factual run. Each step's inputs/outputs are stored (content-addressed, deduplicated) and chained into a Merkle-style hash sequence.
2. **Phase 1 — single-step contrastive estimation.** For every step `i`, hold steps `< i` at their factual recorded actions, **resample** step `i` and everything downstream, and run forward `N` times. Compute `attribution(i) = P(fail|kept) − P(fail|ablated)` with a Wilson interval on the ablated failure rate and a bootstrap interval on the difference.
3. **Point-of-Commitment Rule.** Because resampling an early step re-rolls the fatal late step too (a butterfly-effect confound), *magnitude alone blames early, irrelevant steps*. Instead we take the **latest** step whose interval still strictly excludes zero — the final juncture at which re-deciding can still rescue the run. That is the true causal locus.
4. **Phase 2 — Shapley attribution.** For interacting (AND/OR) failures, single-step ablation double-counts or zeroes-out credit. Shapley values split responsibility fairly by averaging each step's marginal contribution over sampled permutations, using **antithetic reverse-permutation pairing** for variance reduction. Coalition values are deliberately **never cached** (that would collapse marginal variance and yield falsely narrow intervals) and **no truncation** is used (it would skip pivotal late steps).
5. **Repair.** The culprit step's action is swapped for candidate repairs via a `do()` intervention; a candidate that flips the failure rate below threshold and has maximum **minimality** (least behavioural drift) is reported as the validated counterfactual repair.

### Scope note

The source research (*The Chronos Protocol* / *Agent Time-Travel Debugger*)
describes OS-level substrates — DeltaFS/DeltaCR millisecond checkpoints, CRIU,
Firecracker microVMs, WASI-Virt — for capturing full process/filesystem state.
`agent-replay` implements the **framework-agnostic, application-level** essence
of that vision: deterministic record/replay via recorded cassettes plus the full
causal-attribution mathematics, with zero external dependencies. The exotic
kernel primitives are intentionally out of scope; the attribution algorithm does
not depend on them.

---

## Public API

| Symbol | Purpose |
|---|---|
| `Session(db_path)` | Record and persist agent runs to a SQLite store. |
| `session.record(agent_fn, task, seed, verifier)` | Capture one factual `Trajectory`. |
| `attribute(traj, agent_fn, verifier, rollouts, method, repair)` | Run the attribution pipeline → `AttributionResult`. |
| `AttributionResult.to_html(path)` / `.to_json(path)` | Emit the failure-attribution report. |
| `record(...)` | Low-level one-shot recording (no store). |
| `replay(agent_fn, traj, plan, seed)` | Deterministic replay under a `ReplayPlan`. |
| `ReplayPlan.factual / .ablate_from / .coalition` | Build intervention plans. |
| `AblationEngine` | Run stochastic rollouts for a plan. |
| `find_minimal_repair(engine, step)` | Search for a minimal counterfactual repair. |
| `CheckpointStore` | The SQLite checkpoint / content-addressable store. |

`method` is `"contrastive"` (Phase 1), `"shapley"` (Phase 2), or `"both"`.

---

## Framework adapters

**OpenAI SDK** — wrap a client so every `chat.completions.create` becomes a recorded `llm` step:

```python
from openai import OpenAI
from agent_replay.adapters.openai_sdk import wrap_openai

def agent(ctx, prompt):
    client = wrap_openai(OpenAI(), ctx, name="draft")
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return resp["choices"][0]["message"]["content"]
```

**LangChain** — attach a callback handler:

```python
from agent_replay.adapters.langchain import AgentReplayCallbackHandler

def agent(ctx, prompt):
    handler = AgentReplayCallbackHandler(ctx)
    return chain.invoke(prompt, config={"callbacks": [handler]})
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                     # full suite (mock agent with a known-culprit step)
ruff check . && ruff format --check .
python -m build               # sdist + wheel
python examples/quickstart.py # generates report.html / report.json
```

CI (GitHub Actions) runs ruff lint + format checks, the pytest suite on Python
3.9–3.12 with coverage, and a distribution build.

---

## License

MIT © agent-replay contributors. See [LICENSE](LICENSE).
