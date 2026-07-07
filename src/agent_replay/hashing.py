"""Canonical content hashing and Merkle-style linking.

Every recorded value is reduced to a deterministic SHA-256 over its canonical
JSON form. Equal values therefore share an address, which is what lets the
checkpoint store deduplicate blobs (content-addressable storage) and lets each
step chain off its parent, giving the trajectory a tamper-evident identity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to a stable, key-sorted JSON string.

    Non-JSON-native objects fall back to ``repr`` so hashing never crashes; the
    public API nonetheless documents that recorded inputs/outputs should be
    JSON-serialisable so they survive a round-trip through the SQLite store.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def content_hash(value: Any) -> str:
    """Return the hex SHA-256 of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def link_hash(*parts: str) -> str:
    """Hash an ordered sequence of hex digests into a single digest."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
