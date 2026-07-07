"""SQLite checkpoint store with content-addressable blob storage.

This is the persistent substrate: recorded trajectories, their steps, and the
attribution artifacts. Recorded values are stored once in a content-addressable
``blobs`` table (keyed by SHA-256), so repeated identical inputs/outputs across
steps and sessions are deduplicated — the essential, no-frills version of the
Merkle content-addressable storage described in the architecture document.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, List, Optional

from .hashing import content_hash
from .types import AttributionResult, Step, StepKind, Trajectory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    task_json      TEXT NOT NULL,
    seed           INTEGER NOT NULL,
    outcome_score  REAL,
    result_hash    TEXT,
    created_at     REAL NOT NULL,
    meta_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS blobs (
    hash  TEXT PRIMARY KEY,
    data  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    session_id   TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    name         TEXT NOT NULL,
    inputs_hash  TEXT NOT NULL,
    output_hash  TEXT NOT NULL,
    step_hash    TEXT NOT NULL,
    parent_hash  TEXT NOT NULL,
    PRIMARY KEY (session_id, idx),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS attributions (
    session_id   TEXT NOT NULL,
    method       TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (session_id, method),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


class CheckpointStore:
    """A thin, transactional wrapper over a SQLite database file."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- blob CAS -------------------------------------------------------------

    def put_blob(self, value: Any) -> str:
        """Store ``value`` (as canonical JSON) and return its content hash."""
        h = content_hash(value)
        self._conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, data) VALUES (?, ?)",
            (h, json.dumps(value, default=str)),
        )
        return h

    def get_blob(self, h: str) -> Any:
        row = self._conn.execute("SELECT data FROM blobs WHERE hash = ?", (h,)).fetchone()
        if row is None:
            raise KeyError(f"blob {h} not found")
        return json.loads(row[0])

    # -- trajectories ---------------------------------------------------------

    def save_trajectory(self, traj: Trajectory) -> None:
        """Persist a trajectory, deduplicating step payloads into the blob store."""
        result_hash = self.put_blob(traj.result)
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, task_json, seed, outcome_score, result_hash, created_at, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                traj.session_id,
                json.dumps(traj.task, default=str),
                traj.seed,
                traj.outcome_score,
                result_hash,
                traj.created_at,
                json.dumps(traj.meta, default=str),
            ),
        )
        self._conn.execute("DELETE FROM steps WHERE session_id = ?", (traj.session_id,))
        for step in traj.steps:
            inputs_hash = self.put_blob(step.inputs)
            output_hash = self.put_blob(step.output)
            self._conn.execute(
                """INSERT INTO steps
                   (session_id, idx, kind, name, inputs_hash, output_hash, step_hash, parent_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    traj.session_id,
                    step.index,
                    step.kind.value,
                    step.name,
                    inputs_hash,
                    output_hash,
                    step.step_hash,
                    step.parent_hash,
                ),
            )
        self._conn.commit()

    def load_trajectory(self, session_id: str) -> Trajectory:
        """Reconstruct a full trajectory from the store."""
        row = self._conn.execute(
            """SELECT task_json, seed, outcome_score, result_hash, created_at, meta_json
               FROM sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"session {session_id} not found")
        task_json, seed, outcome_score, result_hash, created_at, meta_json = row
        traj = Trajectory(
            session_id=session_id,
            task=json.loads(task_json),
            seed=seed,
            outcome_score=outcome_score,
            result=self.get_blob(result_hash),
            created_at=created_at,
            meta=json.loads(meta_json),
        )
        step_rows = self._conn.execute(
            """SELECT idx, kind, name, inputs_hash, output_hash, step_hash, parent_hash
               FROM steps WHERE session_id = ? ORDER BY idx""",
            (session_id,),
        ).fetchall()
        for idx, kind, name, inputs_hash, output_hash, step_hash, parent_hash in step_rows:
            traj.steps.append(
                Step(
                    index=idx,
                    kind=StepKind(kind),
                    name=name,
                    inputs=self.get_blob(inputs_hash),
                    output=self.get_blob(output_hash),
                    parent_hash=parent_hash,
                    step_hash=step_hash,
                )
            )
        return traj

    def list_sessions(self) -> List[str]:
        rows = self._conn.execute("SELECT session_id FROM sessions ORDER BY created_at").fetchall()
        return [r[0] for r in rows]

    def has_session(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    # -- attributions ---------------------------------------------------------

    def save_attribution(self, result: AttributionResult) -> None:
        import time

        self._conn.execute(
            """INSERT OR REPLACE INTO attributions
               (session_id, method, result_json, created_at) VALUES (?, ?, ?, ?)""",
            (
                result.session_id,
                result.method,
                json.dumps(result.to_dict(), default=str),
                time.time(),
            ),
        )
        self._conn.commit()

    def load_attribution(self, session_id: str, method: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT result_json FROM attributions WHERE session_id = ? AND method = ?",
            (session_id, method),
        ).fetchone()
        return json.loads(row[0]) if row else None
