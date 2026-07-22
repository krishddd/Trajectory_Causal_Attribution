"""Importing trajectories from external traces (JSONL / OTel) + attribution."""

import json

from agent_replay import attribute, interop
from agent_replay.replayer import ReplayPlan, replay


def _steps():
    return [
        {"kind": "llm", "name": "plan", "inputs": {"q": "why"}, "output": "route_a"},
        {"kind": "tool", "name": "search", "inputs": {"route": "a"}, "output": "BAD"},
        {"kind": "llm", "name": "write", "inputs": {"ctx": "BAD"}, "output": "wrong"},
    ]


def _verify(result):
    # Fails iff any step output is "BAD".
    return 0.0 if "BAD" in result["outputs"] else 1.0


def test_from_steps_builds_hashed_trajectory():
    traj = interop.from_steps(_steps(), session_id="imp", verifier=_verify)
    assert len(traj) == 3
    assert traj.steps[0].kind.value == "llm"
    assert traj.steps[1].kind.value == "tool"
    # Merkle chaining is populated (parent of step 1 == step 0's hash).
    assert traj.steps[1].parent_hash == traj.steps[0].step_hash
    assert traj.root_hash == traj.steps[-1].step_hash
    # Default result folds the recorded outputs; verifier sees the failure.
    assert traj.result == {"outputs": ["route_a", "BAD", "wrong"]}
    assert traj.outcome_score == 0.0
    # Imported steps are observation-only until a resample policy is supplied.
    assert all(not s.resamplable for s in traj.steps)


def test_from_jsonl_line_per_step(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in _steps()), encoding="utf-8")
    traj = interop.from_jsonl(str(p), session_id="imp", verifier=_verify)
    assert len(traj) == 3
    assert traj.outcome_score == 0.0


def test_from_jsonl_single_object(tmp_path):
    p = tmp_path / "traj.json"
    p.write_text(
        json.dumps({"session_id": "imp2", "task": {"q": "x"}, "steps": _steps()}),
        encoding="utf-8",
    )
    traj = interop.from_jsonl(str(p), verifier=_verify)
    assert traj.session_id == "imp2"
    assert traj.task == {"q": "x"}
    assert len(traj) == 3


def test_from_otel_spans_maps_llm_and_tool():
    spans = [
        {
            "name": "chat gpt-4o",
            "start_time": 1,
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.completion": "route_a",
            },
        },
        {
            "name": "execute_tool search",
            "start_time": 2,
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.result": "BAD",
            },
        },
    ]
    traj = interop.from_otel_spans(spans, session_id="otel")
    assert [s.kind.value for s in traj.steps] == ["llm", "tool"]
    assert traj.steps[0].name == "gpt-4o"
    assert traj.steps[1].name == "search"
    assert traj.steps[1].output == "BAD"


def test_otel_spans_sorted_by_start_time():
    spans = [
        {"name": "b", "start_time": 5, "attributes": {"gen_ai.completion": "second"}},
        {"name": "a", "start_time": 1, "attributes": {"gen_ai.completion": "first"}},
    ]
    traj = interop.from_otel_spans(spans, session_id="otel2")
    assert traj.steps[0].output == "first"
    assert traj.steps[1].output == "second"


def test_replayable_agent_factual_reproduces_result():
    traj = interop.from_steps(_steps(), session_id="imp", verifier=_verify)
    agent = interop.replayable_agent(traj)
    # No resample fns -> all observed-only -> factual replay reproduces the trace.
    out = replay(agent, traj, ReplayPlan.factual(len(traj)), seed=0)
    assert out == {"outputs": ["route_a", "BAD", "wrong"]}


def test_imported_trace_is_attributable_with_resample_fns():
    traj = interop.from_steps(_steps(), session_id="imp", verifier=_verify)

    # A resample policy for the culprit tool step: mostly recovers ("OK"), so
    # re-deciding it changes the outcome -> it should be attributed.
    def search_policy(ctx, inputs):
        return "OK" if ctx.rng.random() < 0.8 else "BAD"

    def write_policy(ctx, inputs):
        return "wrong" if "BAD" in str(inputs) else "right"

    agent = interop.replayable_agent(
        traj, resample_fns={"search": search_policy, "write": write_policy}
    )
    # replayable_agent flipped resamplability on the supplied steps.
    by_name = {s.name: s for s in traj.steps}
    assert by_name["search"].resamplable
    assert by_name["write"].resamplable
    assert not by_name["plan"].resamplable  # no policy supplied

    result = attribute(traj, agent, _verify, rollouts=80, method="contrastive")
    assert result.failed
    # The tool search step is the point of commitment for this failure.
    assert result.culprit_index == 1
