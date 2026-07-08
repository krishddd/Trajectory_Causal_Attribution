"""Repair v2: propose_fn hook, guard export, contrastive-pair export."""

import json

from agent_replay.ablation import AblationEngine
from agent_replay.attribution import attribute
from agent_replay.mock_agent import buggy_agent, verifier
from agent_replay.repair import export_contrastive_pairs, find_minimal_repair


def test_propose_fn_candidates_are_validated(recording, fail_step):
    seen = {}

    def propose(step, traj):
        seen["called"] = True
        seen["step_index"] = step.index
        # Propose a good fix plus a still-bad decoy; only the good one validates.
        return ["OK", "BAD"]

    engine = AblationEngine(buggy_agent, recording, verifier)
    repair = find_minimal_repair(engine, fail_step, rollouts=40, propose_fn=propose)
    assert seen["called"] and seen["step_index"] == fail_step
    assert repair is not None and repair.valid
    assert repair.repaired_action == "OK"  # the decoy "BAD" did not validate


def test_repair_carries_step_identity_and_baseline(recording, fail_step):
    engine = AblationEngine(buggy_agent, recording, verifier)
    repair = find_minimal_repair(engine, fail_step, rollouts=50)
    assert repair is not None
    assert repair.step_name == recording.steps[fail_step].name
    assert repair.step_kind == recording.steps[fail_step].kind.value
    assert repair.p_fail_before > repair.p_fail_after  # repair improves on the culprit


def test_to_guard_snippet(recording, fail_step):
    engine = AblationEngine(buggy_agent, recording, verifier)
    repair = find_minimal_repair(engine, fail_step, rollouts=50)
    guard = repair.to_guard()
    assert "agent-replay guard" in guard
    assert repair.step_name in guard
    assert repr(repair.repaired_action) in guard
    # Snippet is syntactically plausible Python (if/assignment).
    assert "if step.name ==" in guard


def test_export_contrastive_pairs(recording, tmp_path):
    result = attribute(recording, buggy_agent, verifier, rollouts=60, repair=True)
    path = str(tmp_path / "pairs.jsonl")
    n = export_contrastive_pairs([result], path, trajectories={recording.session_id: recording})
    assert n == 1
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 1
    pair = rows[0]
    assert pair["rejected"] == "BAD"
    assert pair["chosen"] == result.repair.repaired_action
    assert pair["context"] is not None  # culprit step inputs included
    assert pair["p_fail_before"] >= pair["p_fail_after"]


def test_export_skips_results_without_valid_repair(recording, tmp_path):
    # A result computed without repair contributes no pairs.
    result = attribute(recording, buggy_agent, verifier, rollouts=40, repair=False)
    assert result.repair is None
    path = str(tmp_path / "empty.jsonl")
    n = export_contrastive_pairs([result], path)
    assert n == 0
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == ""
