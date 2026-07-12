"""Tests for recording and hashing."""

from _demo_agent import DEFAULT_N_STEPS
from agent_replay.hashing import content_hash, link_hash
from agent_replay.recorder import record
from agent_replay.types import StepKind


def test_record_captures_all_steps(recording):
    assert len(recording) == DEFAULT_N_STEPS
    assert all(s.index == i for i, s in enumerate(recording.steps))


def test_record_marks_failure(recording):
    assert recording.outcome_score == 0.0
    assert recording.result["ok"] is False


def test_record_step_kinds(recording, fail_step):
    # The fail step is a tool call; the rest are llm reasoning steps.
    assert recording.steps[fail_step].kind == StepKind.TOOL
    assert recording.steps[0].kind == StepKind.LLM


def test_hash_chain_is_linked(recording):
    prev = ""
    for step in recording.steps:
        assert step.parent_hash == prev
        assert step.step_hash
        prev = step.step_hash
    assert recording.root_hash == recording.steps[-1].step_hash


def test_content_hash_is_deterministic():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash("x") != content_hash("y")


def test_link_hash_order_matters():
    assert link_hash("a", "b") != link_hash("b", "a")


def test_record_is_reproducible(agent, verify):
    t1 = record(agent, {"task": "t"}, session_id="a", seed=5, verifier=verify)
    t2 = record(agent, {"task": "t"}, session_id="b", seed=5, verifier=verify)
    assert [s.output for s in t1.steps] == [s.output for s in t2.steps]
    assert t1.root_hash == t2.root_hash
