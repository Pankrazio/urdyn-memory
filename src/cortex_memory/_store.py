"""SQLite-backed persistence for canonical memories.

This module is the only place in Cortex that knows about SQL, cursors,
connections, or row layout. `Cortex` and the public API depend only on
`MemoryStore` and `Memory`; SQLite is an implementation detail that can
be replaced without changing either.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from ._errors import CortexStorageError
from ._memory import MEMORY_ID_PATTERN, VALID_EPISTEMIC_STATES, VALID_KINDS, Memory

STORE_SCHEMA_VERSION = 1
DB_FILENAME = "memory.db"


def db_path_for(cortex_dir: Path) -> Path:
    return cortex_dir / DB_FILENAME


class MemoryStore:
    """Boundary around the persisted memory table for a single workspace."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def create_or_open(cls, db_path: Path) -> "MemoryStore":
        """Open the store at `db_path`, creating it if it does not exist."""
        connection = sqlite3.connect(db_path)
        try:
            _ensure_schema(connection)
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise CortexStorageError(f"Cortex memory store at {db_path} is corrupted: {exc}") from exc
        except Exception:
            connection.close()
            raise
        return cls(connection)

    @classmethod
    def open_if_exists(cls, db_path: Path) -> "MemoryStore | None":
        """Open the store at `db_path`, or return None if it does not exist.

        Used by read-only operations so they never have the side effect
        of materializing an empty database file.
        """
        if not db_path.exists():
            return None
        return cls.create_or_open(db_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add(self, memory: Memory) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        memory.memory_id,
                        memory.content,
                        memory.kind,
                        memory.epistemic_state,
                        memory.recorded_at.isoformat(),
                    ),
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist memory {memory.memory_id!r}: {exc}") from exc

    def count(self) -> int:
        try:
            cursor = self._connection.execute("SELECT COUNT(*) FROM memories")
            (total,) = cursor.fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex memory store: {exc}") from exc
        return total

    def search(self, query: str, limit: int) -> list[Memory]:
        """Case-insensitive substring search, ranked by match count.

        Ties are broken by most-recent-first, then by `memory_id`, so
        the result order never depends on incidental database order.
        """
        needle = query.casefold()
        try:
            cursor = self._connection.execute(
                "SELECT memory_id, content, kind, epistemic_state, recorded_at FROM memories"
            )
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex memory store: {exc}") from exc

        matches: list[tuple[int, Memory]] = []
        for row in rows:
            memory = _row_to_memory(row)
            occurrences = memory.content.casefold().count(needle)
            if occurrences > 0:
                matches.append((occurrences, memory))

        matches.sort(key=lambda pair: pair[1].memory_id)
        matches.sort(key=lambda pair: pair[1].recorded_at, reverse=True)
        matches.sort(key=lambda pair: pair[0], reverse=True)

        return [memory for _, memory in matches[:limit]]


def _ensure_schema(connection: sqlite3.Connection) -> None:
    (version,) = connection.execute("PRAGMA user_version").fetchone()
    table_exists = (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
        ).fetchone()
        is not None
    )

    if version == 0:
        if table_exists:
            raise CortexStorageError(
                "Cortex memory store has no recognized schema version but already "
                "contains a 'memories' table; refusing to open a possibly corrupted store"
            )
        with connection:
            connection.execute(
                """
                CREATE TABLE memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    epistemic_state TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )
                """
            )
            connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
        return

    if version != STORE_SCHEMA_VERSION:
        raise CortexStorageError(
            f"Cortex memory store schema version {version} is not supported by this "
            f"version of Cortex (expected {STORE_SCHEMA_VERSION})"
        )

    if not table_exists:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version {version} but is "
            "missing the 'memories' table; refusing to silently recreate it"
        )


def _row_to_memory(row: tuple[str, str, str, str, str]) -> Memory:
    memory_id, content, kind, epistemic_state, recorded_at_raw = row

    if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
        raise CortexStorageError(f"Corrupted memory_id {memory_id!r} in Cortex memory store")
    if kind not in VALID_KINDS:
        raise CortexStorageError(f"Corrupted kind {kind!r} for memory {memory_id!r}")
    if epistemic_state not in VALID_EPISTEMIC_STATES:
        raise CortexStorageError(
            f"Corrupted epistemic_state {epistemic_state!r} for memory {memory_id!r}"
        )
    try:
        recorded_at = dt.datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CortexStorageError(
            f"Corrupted recorded_at value {recorded_at_raw!r} for memory {memory_id!r}"
        ) from exc
    return Memory(
        memory_id=memory_id,
        content=content,
        kind=kind,
        epistemic_state=epistemic_state,
        recorded_at=recorded_at,
    )
