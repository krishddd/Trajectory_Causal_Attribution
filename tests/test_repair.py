"""Tests for minimal counterfactual repair."""

from _demo_agent import buggy_agent, verifier
from agent_replay.ablation import AblationEngine
from agent_replay.repair import find_minimal_repair, minimality


def test_minimality_identical_is_one():
    assert minimality("OK", "OK") == 1.0


def test_minimality_ordering():
    # "OK" is closer to "OKK" than to a totally different long string.
    near = minimality("OK", "OKK")
    far = minimality("OK", "totally different value")
    assert near > far


def test_find_repair_for_culprit(recording, fail_step):
    engine = AblationEngine(buggy_agent, recording, verifier)
    repair = find_minimal_repair(engine, fail_step, rollouts=60)
    assert repair is not None
    assert repair.valid
    assert repair.p_fail_after < 0.5
    assert repair.step_index == fail_step


def test_repair_with_explicit_candidates(recording, fail_step):
    engine = AblationEngine(buggy_agent, recording, verifier)
    repair = find_minimal_repair(engine, fail_step, rollouts=40, candidates={fail_step: ["OK"]})
    assert repair is not None
    assert repair.repaired_action == "OK"
    assert repair.valid


def test_repair_out_of_range_returns_none(recording):
    engine = AblationEngine(buggy_agent, recording, verifier)
    assert find_minimal_repair(engine, len(recording) + 5, rollouts=10) is None
