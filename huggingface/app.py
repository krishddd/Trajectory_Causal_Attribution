"""Hugging Face Space: an interactive agent-replay demo.

Builds a small stochastic multi-step agent with a single injected fault, records a
failing run, then attributes the failure to a step via counterfactual
step-ablation — showing the plain-language explanation and the full HTML report.

The package ships no bundled agents, so this Space defines its own demo agent
(the same shape as the test fixture) purely for illustration.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict

import gradio as gr

from agent_replay import __version__, attribute
from agent_replay.recorder import record


def make_agent(n_steps: int, fail_step: int, fail_prob: float) -> Callable[..., Any]:
    """A chain agent that fails iff it commits a BAD action at ``fail_step``."""

    def agent(ctx: Any, task: str = "demo") -> Dict[str, Any]:
        trace = []
        for i in range(n_steps):

            def produce(step: int = i) -> str:
                if step == fail_step:
                    return "BAD" if ctx.rng.random() < fail_prob else "OK"
                return "OK"

            op = ctx.tool if i == fail_step else ctx.llm
            trace.append(op(f"step_{i}", produce=produce))
        return {"trace": trace, "ok": "BAD" not in trace}

    return agent


def verifier(result: Dict[str, Any]) -> float:
    return 1.0 if result.get("ok", False) else 0.0


def run_demo(n_steps: int, fail_step: int, fail_prob: float, rollouts: int):
    n_steps, fail_step, rollouts = int(n_steps), int(fail_step), int(rollouts)
    if fail_step >= n_steps:
        fail_step = n_steps - 1
    agent = make_agent(n_steps, fail_step, float(fail_prob))

    # Search seeds for a run that actually fails, so there is something to attribute.
    traj = None
    for seed in range(400):
        t = record(agent, {"task": "demo"}, session_id=f"s{seed}", seed=seed, verifier=verifier)
        if t.outcome_score is not None and t.outcome_score < 0.5:
            traj = t
            break
    if traj is None:
        return "No failing run found — raise the fault probability and retry.", ""

    result = attribute(traj, agent, verifier, rollouts=rollouts, method="both", repair=True)
    explanation = result.explain(traj)

    html_path = os.path.join(tempfile.gettempdir(), "report.html")
    result.to_html(html_path, explanation=explanation)
    with open(html_path, encoding="utf-8") as fh:
        report_html = fh.read()

    verdict = (
        f"Ground-truth fault at step {fail_step}. "
        f"Attributed culprit: step {result.culprit_index}.\n\n" + explanation.to_text()
    )
    return verdict, report_html


with gr.Blocks(title="agent-replay") as demo:
    gr.Markdown(
        f"""
        # agent-replay — which step caused your agent to fail?

        Counterfactual **step-ablation** attribution: re-run a recorded agent
        trajectory with one step perturbed and measure how the failure probability
        shifts, then localize the culprit with the **Point-of-Commitment** rule.

        This demo injects a fault at a step you choose, records a failing run, and
        attributes it — the culprit should match the fault you set.

        `agent-replay {__version__}` · zero runtime dependencies ·
        [GitHub](https://github.com/krishddd/Trajectory_Causal_Attribution) ·
        [Docs](https://krishddd.github.io/Trajectory_Causal_Attribution/)
        """
    )
    with gr.Row():
        n_steps = gr.Slider(3, 10, value=6, step=1, label="Number of steps")
        fail_step = gr.Slider(0, 9, value=3, step=1, label="Fault at step")
        fail_prob = gr.Slider(0.2, 0.95, value=0.6, step=0.05, label="Fault probability")
        rollouts = gr.Slider(20, 120, value=60, step=10, label="Rollouts / step")
    run_btn = gr.Button("Record & attribute", variant="primary")
    verdict = gr.Textbox(label="Explanation", lines=16)
    report = gr.HTML(label="Attribution report")
    run_btn.click(
        run_demo,
        inputs=[n_steps, fail_step, fail_prob, rollouts],
        outputs=[verdict, report],
    )


if __name__ == "__main__":
    demo.launch()
