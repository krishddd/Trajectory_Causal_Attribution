"""Action/observation split + the mock-observe and swap-model interventions."""

from agent_replay import fork
from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay
from agent_replay.store import CheckpointStore
from agent_replay.types import Step, StepKind


def tool_agent(ctx, q="x"):
    # The action is the chosen query; the observation is the environment's result.
    r = ctx.tool(
        "search",
        produce=lambda: f"query:{q}",
        observe=lambda a: f"result-for-{a}",
        q=q,
    )
    return {"r": r}


def model_agent(ctx, q="x"):
    model = getattr(ctx, "model_hint", None) or "default"
    out = ctx.llm("gen", produce=lambda: f"{model}:answer")
    return {"out": out, "model": model}


# --- the split itself -------------------------------------------------------


def test_observe_records_distinct_action_and_observation():
    traj = record(tool_agent, {"q": "abc"}, session_id="t", seed=0)
    step = traj.steps[0]
    assert step.action == "query:abc"  # the recorded action
    assert step.output == "result-for-query:abc"  # the observation flows downstream
    assert step.observation == step.output
    assert step.action_value == "query:abc"
    assert traj.result == {"r": "result-for-query:abc"}


def test_action_defaults_to_output_when_no_observe():
    traj = record(model_agent, {}, session_id="m", seed=0)
    step = traj.steps[0]
    assert step.action is None  # not split
    assert step.action_value == step.output  # falls back to the observation


def test_action_survives_store_roundtrip(tmp_path):
    traj = record(tool_agent, {"q": "abc"}, session_id="t", seed=0)
    store = CheckpointStore(str(tmp_path / "s.db"))
    store.save_trajectory(traj)
    loaded = store.load_trajectory("t")
    assert loaded.steps[0].action == "query:abc"
    assert loaded.steps[0].output == "result-for-query:abc"
    # A run with no split loads action=None (back-compat with pre-split stores).
    m = record(model_agent, {}, session_id="m", seed=0)
    store.save_trajectory(m)
    assert store.load_trajectory("m").steps[0].action is None
    store.close()


def test_legacy_step_action_value_falls_back():
    # A Step built the old way (no action) behaves as action == output.
    s = Step(index=0, kind=StepKind.LLM, name="x", inputs={}, output="hello")
    assert s.action is None
    assert s.action_value == "hello"


# --- mock-observe (distinct from do/force) ----------------------------------


def test_mock_observe_overrides_observation_keeps_action():
    traj = record(tool_agent, {"q": "abc"}, session_id="t", seed=0)
    plan = ReplayPlan.mock_observe(0, "MOCKED-RESULT")
    out = replay(tool_agent, traj, plan, seed=0)
    assert out == {"r": "MOCKED-RESULT"}  # observation replaced downstream
    # The recorded action on the cassette is untouched.
    assert traj.steps[0].action == "query:abc"


def test_fork_mock_observe():
    traj = record(tool_agent, {"q": "abc"}, session_id="t", seed=0)
    child = fork(tool_agent, traj, 0, observe="MOCKED", session_id="child")
    assert child.result == {"r": "MOCKED"}
    assert child.meta["intervention"] == "mock_observe"


# --- swap-model -------------------------------------------------------------


def test_swap_model_hint_via_plan():
    traj = record(model_agent, {}, session_id="m", seed=0)
    assert traj.result["model"] == "default"
    out = replay(model_agent, traj, ReplayPlan(model_override="gpt-5"), seed=0)
    assert out["model"] == "gpt-5"
    assert out["out"] == "gpt-5:answer"


def test_fork_swap_model():
    traj = record(model_agent, {}, session_id="m", seed=0)
    child = fork(model_agent, traj, 0, model="gpt-5", session_id="upgraded")
    assert child.result["model"] == "gpt-5"
    assert child.meta["intervention"] == "swap_model"
    assert child.meta["model_override"] == "gpt-5"
