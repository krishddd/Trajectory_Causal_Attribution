# Connecting any framework

agent-replay attributes failures for **any** Python agent. There are three ways
to feed it steps, from most explicit to most automatic. All three produce the
same `Trajectory` and work with the full attribution + explanation pipeline.

## 1. Explicit context (most transparent)

Pass the `ctx` handle and route non-deterministic work through it. Works with any
code; nothing is patched.

```python
def agent(ctx, question):
    plan = ctx.llm("plan", produce=lambda: call_model(question), prompt=question)
    hits = ctx.tool("search", produce=lambda: search(plan), q=plan)
    return {"answer": hits, "ok": ...}

traj = session.record(agent, {"question": "..."}, verifier=verifier)
```

## 2. Decorators / ambient context (no `ctx` threading)

Decorate your functions once; the agent takes no `ctx`. Steps are captured via the
*ambient* context that `record`/`replay` publish for the run.

```python
from agent_replay import instrument

@instrument.tool                    # or @instrument.llm / @instrument.memory
def search(query): ...

@instrument.llm
def answer(context): ...

def agent(question):                # no ctx parameter
    return {"answer": answer(search(question))}

traj = instrument.record_agent(agent, {"question": "..."}, session_id="s", verifier=verifier)
# attribution must re-run the same way:
result = attribute(traj, agent, verifier, pass_context=False)
```

`instrument.wrap(fn, kind, name)` does the same without decorator syntax.

## 3. Auto-instrument an SDK (unmodified framework code)

Monkeypatch a framework's call site so *existing* code records with no edits. A
data-only `RECIPES` registry ships call sites for common SDKs; patching is
best-effort (absent SDKs are skipped).

```python
from agent_replay import instrument

instrument.available_frameworks()
# ['anthropic', 'autogen', 'cohere', 'crewai', 'google-genai',
#  'langchain', 'litellm', 'llama-index', 'mistralai', 'openai']

with instrument.installed("openai", "langchain"):
    traj = instrument.record_agent(run_my_crew, {...}, session_id="s", verifier=v)
```

Not in the registry? Patch any dotted callable yourself — this *is* how the
recipes work, so it covers every framework:

```python
instrument.patch("my_framework.LLM.complete", kind="llm", name="myfw.complete")
# ... run/record ...
instrument.unpatch("my_framework.LLM.complete")
```

Or import a trajectory recorded elsewhere (OpenTelemetry GenAI spans, JSONL
traces) — see `docs/RESEARCH_NOTES.md` §6 for the planned `interop` importers.

### Resampling & side effects

A wrapped call's `produce` policy *re-invokes the real callable* during
counterfactual resampling. That is correct for idempotent model/tool calls. For
deterministic (temperature 0) or side-effectful calls that must not re-execute,
mark them non-resamplable:

```python
@instrument.record_step("tool", "charge_card", resamplable=False)
def charge_card(amount): ...
```

Non-resamplable steps are served from the cassette on replay and flagged
`observed-only` in the report — never silently scored zero.

## The LangChain callback adapter

For LangChain specifically, a callback handler is also provided (captures LLM and
tool events as observation-only steps):

```python
from agent_replay.adapters.langchain import AgentReplayCallbackHandler
chain.invoke(prompt, config={"callbacks": [AgentReplayCallbackHandler(ctx)]})
```

For full counterfactual resampling of LangChain steps, prefer option 2/3 (wrap the
bound model call) so re-running actually re-invokes the model.
