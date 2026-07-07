"""Tests for the SQLite checkpoint store and content-addressable blobs."""

import pytest

from agent_replay.store import CheckpointStore


def test_blob_roundtrip_and_dedup():
    store = CheckpointStore(":memory:")
    h1 = store.put_blob({"a": 1})
    h2 = store.put_blob({"a": 1})  # identical content -> same address
    h3 = store.put_blob({"a": 2})
    assert h1 == h2
    assert h1 != h3
    assert store.get_blob(h1) == {"a": 1}
    # Only two distinct blobs stored despite three puts.
    count = store._conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert count == 2
    store.close()


def test_get_missing_blob_raises():
    store = CheckpointStore(":memory:")
    with pytest.raises(KeyError):
        store.get_blob("deadbeef")
    store.close()


def test_save_and_load_trajectory(recording):
    store = CheckpointStore(":memory:")
    store.save_trajectory(recording)
    loaded = store.load_trajectory(recording.session_id)
    assert loaded.session_id == recording.session_id
    assert len(loaded) == len(recording)
    assert loaded.outcome_score == recording.outcome_score
    assert loaded.result == recording.result
    assert [s.output for s in loaded.steps] == [s.output for s in recording.steps]
    assert loaded.root_hash == recording.root_hash
    store.close()


def test_list_and_has_session(recording):
    store = CheckpointStore(":memory:")
    assert store.list_sessions() == []
    store.save_trajectory(recording)
    assert store.has_session(recording.session_id)
    assert not store.has_session("nope")
    assert recording.session_id in store.list_sessions()
    store.close()


def test_load_missing_session_raises():
    store = CheckpointStore(":memory:")
    with pytest.raises(KeyError):
        store.load_trajectory("ghost")
    store.close()


def test_file_backed_persistence(tmp_path, recording):
    db = str(tmp_path / "runs.sqlite")
    s1 = CheckpointStore(db)
    s1.save_trajectory(recording)
    s1.close()
    s2 = CheckpointStore(db)
    loaded = s2.load_trajectory(recording.session_id)
    assert len(loaded) == len(recording)
    s2.close()


def test_context_manager(recording):
    with CheckpointStore(":memory:") as store:
        store.save_trajectory(recording)
        assert store.has_session(recording.session_id)
