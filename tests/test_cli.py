"""Tests for the command-line interface."""

import json

from agent_replay.cli import main

AGENT = "_demo_agent:buggy_agent"
VERIFIER = "_demo_agent:verifier"


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


def test_list_command(tmp_path, capsys):
    db = str(tmp_path / "cli_list.sqlite")
    main(
        [
            "record",
            "--db",
            db,
            "--session",
            "sess-a",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--seed",
            "1",
        ]
    )
    capsys.readouterr()
    rc = main(["list", "--db", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sess-a" in out
    assert "steps" in out


def test_drift_command(tmp_path, capsys):
    db = str(tmp_path / "cli_drift.sqlite")
    main(
        [
            "record",
            "--db",
            db,
            "--session",
            "sd",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--seed",
            "1",
        ]
    )
    capsys.readouterr()
    out_html = str(tmp_path / "drift")
    rc = main(
        [
            "drift",
            "--db",
            db,
            "--session",
            "sd",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--state-scorer",
            "_demo_agent:health_scorer",
            "--rollouts",
            "15",
            "--out",
            out_html,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Drift curve for" in out
    assert "entropy of autonomy" in out
    assert (tmp_path / "drift.html").exists()


def test_aggregate_command(tmp_path, capsys):
    db = str(tmp_path / "agg.sqlite")
    # Record several failing runs of the fixture agent (fail_step 3 is systematic).
    for i, seed in enumerate([1, 5, 9, 13]):
        rc = main(
            [
                "record",
                "--db",
                db,
                "--session",
                f"run{i}",
                "--agent",
                AGENT,
                "--verifier",
                VERIFIER,
                "--seed",
                str(seed),
            ]
        )
        assert rc == 0
    capsys.readouterr()  # clear

    out_json = str(tmp_path / "agg")
    rc = main(
        [
            "aggregate",
            "--db",
            db,
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--rollouts",
            "40",
            "--label",
            "fixture",
            "--out",
            out_json,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Aggregate attribution for 'fixture'" in out
    assert "tool:tool_step_3" in out
    assert (tmp_path / "agg.json").exists()


def test_invalid_entrypoint_raises(tmp_path):
    db = str(tmp_path / "x.sqlite")
    try:
        main(["record", "--db", db, "--session", "s", "--agent", "not_a_valid_spec"])
    except ValueError as e:
        assert "invalid entrypoint" in str(e)
    else:
        raise AssertionError("expected ValueError")
