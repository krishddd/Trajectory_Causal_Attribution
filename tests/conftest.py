"""Shared pytest fixtures for the agent-replay suite."""

import pytest

from agent_replay.mock_agent import DEFAULT_FAIL_STEP, buggy_agent, make_recording, verifier


@pytest.fixture
def fail_step():
    return DEFAULT_FAIL_STEP


@pytest.fixture
def recording():
    """A recorded, genuinely-failing run of the reference mock agent."""
    return make_recording(session_id="test-mock")


@pytest.fixture
def agent():
    return buggy_agent


@pytest.fixture
def verify():
    return verifier
