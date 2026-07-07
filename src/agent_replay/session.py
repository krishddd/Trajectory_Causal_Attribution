"""The high-level ``Session`` facade tying recording and storage together.

``Session`` is the ergonomic entrypoint most users touch: open it on a SQLite
path, record agent runs into it, and load them back for attribution. It is a
thin orchestration layer over :mod:`recorder` and :mod:`store`.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from .recorder import record as _record
from .store import CheckpointStore
from .types import Trajectory


class Session:
    """A recording session backed by a checkpoint store.

    Example
    -------
    >>> with Session("runs.sqlite") as s:
    ...     traj = s.record(my_agent, task={"goal": "..."}, verifier=my_verifier)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.store = CheckpointStore(db_path)

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()

    # -- recording ------------------------------------------------------------

    def record(
        self,
        agent_fn: Callable[..., Any],
        task: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        seed: int = 0,
        verifier: Optional[Callable[[Any], float]] = None,
        persist: bool = True,
    ) -> Trajectory:
        """Record one run of ``agent_fn`` and (by default) persist it."""
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        traj = _record(agent_fn, task, session_id=session_id, seed=seed, verifier=verifier)
        if persist:
            self.store.save_trajectory(traj)
        return traj

    # -- retrieval ------------------------------------------------------------

    def load(self, session_id: str) -> Trajectory:
        return self.store.load_trajectory(session_id)

    def sessions(self) -> List[str]:
        return self.store.list_sessions()
