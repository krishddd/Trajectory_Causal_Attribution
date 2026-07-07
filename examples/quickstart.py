"""20-line integration example: record an agent, attribute its failure.

Run:  python examples/quickstart.py
"""

from agent_replay import Session, attribute


def my_agent(ctx, question):
    plan = ctx.llm("plan", produce=lambda: {"q": question}, prompt=question)
    hits = ctx.tool(
        "search", produce=lambda: "bug" if ctx.rng.random() < 0.7 else "ok", q=plan["q"]
    )
    draft = ctx.llm("write", produce=lambda: hits, context=hits)
    return {"answer": draft, "ok": draft == "ok"}


def verifier(result):  # 1.0 == success, 0.0 == failure
    return 1.0 if result["ok"] else 0.0


with Session("demo.sqlite") as session:
    traj = session.record(my_agent, {"question": "why did it fail?"}, seed=3, verifier=verifier)
    result = attribute(traj, my_agent, verifier, rollouts=60, method="both", repair=True)
    result.to_html("report.html")
    result.to_json("report.json")
    print(
        f"Outcome: {'FAILED' if result.failed else 'PASSED'} | culprit step: {result.culprit_index}"
    )
