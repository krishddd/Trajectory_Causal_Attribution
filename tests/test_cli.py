"""Tests for the command-line interface."""

import json

from agent_replay.cli import main

AGENT = "agent_replay.mock_agent:buggy_agent"
VERIFIER = "agent_replay.mock_agent:verifier"


def test_record_replay_attribute_flow(tmp_path, capsys):
    db = str(tmp_path / "cli.sqlite")

    # record
    rc = main(
        [
            "record",
            "--db",
            db,
            "--session",
            "s1",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--task",
            json.dumps({"task": "cli"}),
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Recorded session 's1'" in out
    assert "Root hash" in out

    # replay
    rc = main(["replay", "--db", db, "--session", "s1", "--agent", AGENT, "--verifier", VERIFIER])
    assert rc == 0
    assert "Deterministic replay" in capsys.readouterr().out

    # attribute -> writes reports
    base = str(tmp_path / "report")
    rc = main(
        [
            "attribute",
            "--db",
            db,
            "--session",
            "s1",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--rollouts",
            "60",
            "--method",
            "both",
            "--repair",
            "--out",
            base,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Failure attributed to step" in out
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()


def test_report_regeneration(tmp_path, capsys):
    db = str(tmp_path / "cli2.sqlite")
    main(
        [
            "record",
            "--db",
            db,
            "--session",
            "s",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--seed",
            "1",
        ]
    )
    capsys.readouterr()
    base = str(tmp_path / "rep")
    main(
        [
            "attribute",
            "--db",
            db,
            "--session",
            "s",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--rollouts",
            "40",
            "--out",
            base,
        ]
    )
    capsys.readouterr()

    html_out = str(tmp_path / "regen.html")
    rc = main(["report", "--json", base + ".json", "--out", html_out])
    assert rc == 0
    assert (tmp_path / "regen.html").exists()


def test_invalid_entrypoint_raises(tmp_path):
    db = str(tmp_path / "x.sqlite")
    try:
        main(["record", "--db", db, "--session", "s", "--agent", "not_a_valid_spec"])
    except ValueError as e:
        assert "invalid entrypoint" in str(e)
    else:
        raise AssertionError("expected ValueError")
