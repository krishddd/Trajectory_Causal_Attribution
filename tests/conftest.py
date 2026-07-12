"""Shared pytest fixtures for the agent-replay suite."""

import pytest

from _demo_agent import DEFAULT_FAIL_STEP, buggy_agent, make_recording, verifier

# Make the plugin's fixtures available under PYTHONPATH (when the pytest11 entry
# point is not registered because the package is not pip-installed). When it *is*
# installed (CI), these simply mirror the entry-point fixtures.
from agent_replay.pytest_plugin import (  # noqa: E402
    agent_replay_session,  # noqa: F401
    assert_agent,  # noqa: F401
)


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
