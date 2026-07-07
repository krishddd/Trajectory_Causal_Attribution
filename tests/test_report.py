"""Tests for JSON + HTML report generation."""

import json

from agent_replay.attribution import attribute
from agent_replay.mock_agent import buggy_agent, verifier
from agent_replay.report import render_html


def _result(recording):
    return attribute(recording, buggy_agent, verifier, rollouts=60, method="both", repair=True)


def test_json_report_roundtrips(recording, tmp_path):
    result = _result(recording)
    path = str(tmp_path / "r.json")
    result.to_json(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["session_id"] == recording.session_id
    assert data["culprit_index"] == result.culprit_index
    assert len(data["steps"]) == result.total_steps
    assert data["repair"] is not None


def test_html_report_is_standalone(recording, tmp_path):
    result = _result(recording)
    path = str(tmp_path / "r.html")
    result.to_html(path)
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    assert html.startswith("<!DOCTYPE html>")
    assert "Trajectory Causal Attribution Report" in html
    assert "culprit" in html
    # No external resource references.
    assert "http://" not in html and "https://" not in html


def test_render_html_marks_culprit(recording):
    result = _result(recording)
    html = render_html(result)
    assert f"step {result.culprit_index}" in html
    assert "FAILED" in html


def test_report_handles_no_culprit(recording):
    result = _result(recording)
    result.culprit_index = None
    result.point_of_commitment = None
    result.repair = None
    html = render_html(result)
    assert "No single step" in html
