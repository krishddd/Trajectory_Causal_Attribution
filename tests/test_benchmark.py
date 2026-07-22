"""Smoke test for the Who&When benchmark harness (a tiny, fast subset)."""

import sys
from pathlib import Path

# The benchmarks/ directory is a scripts folder, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import whowhen  # noqa: E402


def _small_suite():
    specs = [("len4-mid", 4, 2, 0.6), ("len6-late", 6, 5, 0.6)]
    cases = []
    for name, n, fs, p in specs:
        agent = whowhen.make_chain_agent(n, fs, p)
        cases.append(whowhen.Case(name, agent, whowhen._record_failing(agent, name), fs))
    return cases


def test_benchmark_localises_and_beats_baselines():
    cases = _small_suite()
    report = whowhen.evaluate(cases, rollouts=50)
    # Causal attribution should localize every culprit in this easy suite...
    assert report.causal_accuracy == 1.0
    # ...and beat the naive magnitude baseline (PoC rule earns its keep).
    assert report.causal_accuracy >= report.magnitude_accuracy
    # The report renders and cites the judge baseline.
    text = report.to_text()
    assert "LLM-as-judge" in text
    assert "causal attribution" in text


def test_benchmark_adaptive_is_cheaper_or_equal():
    cases = _small_suite()
    fixed = whowhen.evaluate(cases, rollouts=80, adaptive=False)
    adaptive = whowhen.evaluate(cases, rollouts=80, adaptive=True)
    # Adaptive never runs more than the cap and usually far fewer.
    assert adaptive.mean_rollouts <= fixed.mean_rollouts
    assert adaptive.causal_accuracy == 1.0
