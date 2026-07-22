"""Who&When-style step-attribution benchmark.

*Who&When* (Zhang et al., ICML'25 Spotlight, arXiv:2505.00212) asks: given a
**failed** multi-agent trajectory, which step is responsible? Their strongest
LLM-as-judge attributor localizes the culprit step only ~14% of the time. This
harness measures how well counterfactual step-ablation attribution does on the
same task, on ground-truth-labelled trajectories, and reports the accuracy *and*
the rollout cost that buys it.

Because the public dataset is a network/licensing dependency, this script ships a
**synthetic generator** with known ground truth so the benchmark runs offline and
deterministically. Each case is a chain agent that is benign until a single
``fail_step`` commits a "BAD" action (with probability ``fail_prob``); the run
fails iff a BAD action occurred, so the *responsible* step is unambiguously
``fail_step``. Cases vary the chain length and the culprit's position (early /
middle / late) — the exact axis where magnitude-based blame fails and the
Point-of-Commitment rule wins.

To run the real dataset instead, export each trajectory to the JSONL layout
:mod:`agent_replay.interop` reads, supply a resample policy per step kind, and
pass the imported trajectories to :func:`evaluate` — the accuracy math is
identical.

Usage::

    python benchmarks/whowhen.py                 # default suite
    python benchmarks/whowhen.py --rollouts 80 --adaptive
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Allow running as a plain script (``python benchmarks/whowhen.py``) from a source
# checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_replay.attribution import attribute  # noqa: E402
from agent_replay.recorder import record  # noqa: E402
from agent_replay.types import Trajectory  # noqa: E402


def make_chain_agent(n_steps: int, fail_step: int, fail_prob: float) -> Callable[..., Any]:
    """A chain agent that fails iff it commits a BAD action at ``fail_step``."""

    def agent(ctx: Any, task: str = "t") -> Dict[str, Any]:
        trace: List[str] = []
        for i in range(n_steps):

            def produce(step: int = i) -> str:
                if step == fail_step:
                    return "BAD" if ctx.rng.random() < fail_prob else "OK"
                return "OK"

            kind = ctx.tool if i == fail_step else ctx.llm
            trace.append(kind(f"step_{i}", produce=produce))
        return {"trace": trace, "ok": "BAD" not in trace}

    return agent


def verifier(result: Dict[str, Any]) -> float:
    return 1.0 if result.get("ok", False) else 0.0


@dataclass
class Case:
    """One labelled benchmark instance."""

    name: str
    agent: Callable[..., Any]
    trajectory: Trajectory
    ground_truth: int  # the step that is actually responsible for the failure


def _record_failing(agent: Callable[..., Any], name: str) -> Trajectory:
    """Record a run of ``agent`` that actually fails (search seeds until it does)."""
    for seed in range(500):
        traj = record(agent, {"task": name}, session_id=name, seed=seed, verifier=verifier)
        if traj.outcome_score is not None and traj.outcome_score < 0.5:
            return traj
    raise RuntimeError(f"no failing seed found for {name}")


def default_suite() -> List[Case]:
    """A spread of chain lengths and culprit positions (early/middle/late)."""
    # fail_prob near 0.6 keeps a healthy rescue rate at the true step (the signal
    # the estimator localizes) while still guaranteeing a factual failure.
    specs: List[Tuple[str, int, int, float]] = [
        ("len4-early", 4, 0, 0.6),
        ("len4-late", 4, 3, 0.6),
        ("len6-early", 6, 1, 0.6),
        ("len6-mid", 6, 3, 0.6),
        ("len6-late", 6, 5, 0.6),
        ("len8-mid", 8, 4, 0.6),
        ("len8-late", 8, 7, 0.6),
        ("len10-mid", 10, 5, 0.6),
    ]
    cases: List[Case] = []
    for name, n, fs, p in specs:
        agent = make_chain_agent(n, fs, p)
        cases.append(Case(name, agent, _record_failing(agent, name), fs))
    return cases


# -- baselines (what you'd get without causal attribution) -------------------


def last_step_baseline(traj: Trajectory) -> int:
    """Blame the step that executed last — the naive 'it broke at the end' guess."""
    return len(traj) - 1


def max_magnitude_baseline(case: Case, rollouts: int) -> int:
    """Blame the highest-|attribution| step (no Point-of-Commitment rule).

    This is the confound the research warns about: resampling an early step also
    re-rolls the fatal late step, inflating early scores. Included to show what
    the PoC rule buys over raw magnitude.
    """
    result = attribute(case.trajectory, case.agent, verifier, rollouts=rollouts)
    return max(result.steps, key=lambda s: s.attribution).index


@dataclass
class BenchmarkReport:
    rollouts: int
    adaptive: bool
    n_cases: int
    causal_correct: int = 0
    last_correct: int = 0
    magnitude_correct: int = 0
    total_rollouts: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def causal_accuracy(self) -> float:
        return self.causal_correct / self.n_cases if self.n_cases else 0.0

    @property
    def last_accuracy(self) -> float:
        return self.last_correct / self.n_cases if self.n_cases else 0.0

    @property
    def magnitude_accuracy(self) -> float:
        return self.magnitude_correct / self.n_cases if self.n_cases else 0.0

    @property
    def mean_rollouts(self) -> float:
        return self.total_rollouts / self.n_cases if self.n_cases else 0.0

    def to_text(self) -> str:
        lines = [
            "Who&When-style step-attribution benchmark",
            f"  cases={self.n_cases}  rollouts/step={self.rollouts}  adaptive={self.adaptive}",
            "",
            f"  causal attribution (this tool)  : {self.causal_accuracy:6.1%}"
            f"  ({self.causal_correct}/{self.n_cases})",
            f"  max-magnitude (no PoC rule)     : {self.magnitude_accuracy:6.1%}"
            f"  ({self.magnitude_correct}/{self.n_cases})",
            f"  last-step baseline              : {self.last_accuracy:6.1%}"
            f"  ({self.last_correct}/{self.n_cases})",
            "  LLM-as-judge (Who&When lit.)    :  ~14.2%  (arXiv:2505.00212)",
            "",
            f"  mean rollouts/step actually run : {self.mean_rollouts:.1f}",
            "",
            "  per-case (culprit = ground truth):",
            "    case            truth  causal  magnitude  last",
        ]
        for r in self.rows:
            lines.append(
                f"    {r['name']:<14} {r['truth']:>5}  {r['causal']:>6}"
                f"  {r['magnitude']:>9}  {r['last']:>4}"
            )
        return "\n".join(lines)


def evaluate(cases: List[Case], *, rollouts: int = 60, adaptive: bool = False) -> BenchmarkReport:
    """Run causal attribution + baselines over ``cases`` and score localization."""
    report = BenchmarkReport(rollouts=rollouts, adaptive=adaptive, n_cases=len(cases))
    for case in cases:
        result = attribute(
            case.trajectory,
            case.agent,
            verifier,
            rollouts=rollouts,
            method="contrastive",
            adaptive=adaptive,
        )
        causal = result.culprit_index
        magnitude = max(result.steps, key=lambda s: s.attribution).index
        last = last_step_baseline(case.trajectory)

        report.causal_correct += int(causal == case.ground_truth)
        report.magnitude_correct += int(magnitude == case.ground_truth)
        report.last_correct += int(last == case.ground_truth)
        report.total_rollouts += result.rollouts
        report.rows.append(
            {
                "name": case.name,
                "truth": case.ground_truth,
                "causal": causal if causal is not None else "-",
                "magnitude": magnitude,
                "last": last,
            }
        )
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Who&When-style attribution benchmark")
    parser.add_argument("--rollouts", type=int, default=60, help="rollouts per step (or cap)")
    parser.add_argument("--adaptive", action="store_true", help="use sequential-stopping rollouts")
    args = parser.parse_args(argv)

    cases = default_suite()
    report = evaluate(cases, rollouts=args.rollouts, adaptive=args.adaptive)
    print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
