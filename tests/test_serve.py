"""The Multiverse Console: render functions + a live smoke test."""

import threading
import urllib.request

from _demo_agent import buggy_agent, verifier
from agent_replay.multiverse import fork
from agent_replay.serve import make_server, render_index, render_session
from agent_replay.store import CheckpointStore


def _store_with_branch(recording, fail_step, tmp_path):
    db = str(tmp_path / "serve.sqlite")
    child = fork(
        buggy_agent, recording, fail_step, do="OK", seed=1, session_id="fixed", verifier=verifier
    )
    store = CheckpointStore(db)
    store.save_trajectory(recording)
    store.save_trajectory(child)
    return store


def test_render_index_lists_and_nests(recording, fail_step, tmp_path):
    store = _store_with_branch(recording, fail_step, tmp_path)
    html = render_index(store)
    assert recording.session_id in html
    assert "fixed" in html
    assert "fork@" in html  # branch tag shown
    assert "Multiverse Console" in html
    store.close()


def test_render_session_shows_frozen_state(recording, fail_step, tmp_path):
    store = _store_with_branch(recording, fail_step, tmp_path)
    html = render_session(store, recording.session_id)
    assert "Frozen state" in html
    assert "branches:" in html  # links to the child
    for s in recording.steps:
        assert s.name in html
    store.close()


def test_render_session_child_links_parent(recording, fail_step, tmp_path):
    store = _store_with_branch(recording, fail_step, tmp_path)
    html = render_session(store, "fixed")
    assert "forked from" in html
    assert recording.session_id in html
    store.close()


def test_render_missing_session(recording, tmp_path):
    store = CheckpointStore(str(tmp_path / "s.sqlite"))
    html = render_session(store, "ghost")
    assert "not found" in html
    store.close()


def test_live_server_smoke(recording, fail_step, tmp_path):
    store = _store_with_branch(recording, fail_step, tmp_path)
    store.close()  # the server opens its own connection
    httpd = make_server(str(tmp_path / "serve.sqlite"), port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            index = r.read().decode()
        assert "Multiverse Console" in index and "fixed" in index
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/session/{recording.session_id}", timeout=5
        ) as r:
            page = r.read().decode()
        assert "Frozen state" in page
    finally:
        httpd.shutdown()
        httpd.server_close()
