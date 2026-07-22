"""Tests for the OpenAI SDK adapter (recording wrapper + guardrails)."""

import warnings

import agent_replay.adapters.openai_sdk as openai_sdk
from agent_replay.adapters.openai_sdk import wrap_openai
from agent_replay.recorder import record


class _FakeCompletions:
    def create(self, **kwargs):
        return {"choices": [{"message": {"content": "hello"}}]}


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def _agent(ctx, prompt="hi", temperature=0.7):
    client = wrap_openai(_FakeClient(), ctx, name="draft")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp["choices"][0]["message"]["content"]


def test_wrap_openai_records_llm_step():
    traj = record(_agent, {"prompt": "hi"}, session_id="oa", seed=0)
    assert len(traj.steps) == 1
    assert traj.steps[0].kind.value == "llm"
    assert traj.result == "hello"


def test_zero_temperature_warns_once():
    # Reset the once-flag so the warning is observable in this test.
    openai_sdk._ZERO_TEMP_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record(
            lambda ctx: _agent(ctx, temperature=0),
            {},
            session_id="oa0",
            seed=0,
        )
    assert any("temperature=0" in str(w.message) for w in caught)


def test_nonzero_temperature_does_not_warn():
    openai_sdk._ZERO_TEMP_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record(lambda ctx: _agent(ctx, temperature=0.7), {}, session_id="oa1", seed=0)
    assert not any("temperature=0" in str(w.message) for w in caught)
