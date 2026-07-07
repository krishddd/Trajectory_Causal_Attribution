"""Connect *any* framework with auto-instrumentation, then get an explanation.

This shows the "no explicit ctx" style: functions are decorated (or SDK methods
are patched via ``instrument.install(...)``), the agent is recorded through the
*ambient* context, and the failure is explained in plain language.

Run:  python examples/connect_any_framework.py
"""

from agent_replay import attribute, instrument


# Decorate any callable — no framework required. For real SDKs you'd instead call
# instrument.install("openai", "anthropic", "langchain", ...) to patch them.
@instrument.tool
def search(query):
    # Pretend this hits a flaky retrieval tool.
    import random

    return "corrupt" if random.random() < 0.7 else "clean"


@instrument.llm
def answer(context):
    return "wrong" if context == "corrupt" else "correct"


def agent(question):  # note: no ctx parameter — uses the ambient context
    hits = search(question)
    draft = answer(hits)
    return {"draft": draft, "ok": draft == "correct"}


def verifier(result):
    return 1.0 if result["ok"] else 0.0


# record_agent installs recipes (none needed here) and records via ambient context.
traj = instrument.record_agent(agent, {"question": "why did it fail?"}, session_id="demo", seed=1)

# attribute must re-run the agent the same way -> pass_context=False.
result = attribute(
    traj, agent, verifier, rollouts=80, method="both", repair=True, pass_context=False
)

explanation = result.explain(traj)
print(explanation.to_text())

result.to_html("report.html", explanation=explanation)
result.to_json("report.json", explanation=explanation)
