"""SQLite-backed persistence for canonical memories, evidence, events, and
attempts.

This module is the only place in Cortex that knows about SQL, cursors,
connections, or row layout. `Cortex` and the public API depend only on
`MemoryStore`, `Memory`, `Evidence`, `Event`, and `Attempt`; SQLite is an
implementation detail that can be replaced without changing any of them.

Schema history:
  v1 - `memories` table only (content, kind, epistemic_state, recorded_at).
  v2 - adds `memories.supersedes`, `evidence`, `memory_evidence`, `events`.
  v3 - adds `attempts`, `attempt_evidence`. `memories.kind` and
       `memories.epistemic_state` gain new valid values (`lesson`,
       `root_cause`, `inferred`, `verified`) at the Python validation
       layer only — no column changes are needed for that.
  v4 - adds `skills`, `skill_steps`, `skill_conditions`, `skill_evidence`.

A v1, v2, or v3 database opened by this module is migrated forward to v4
in place, one step at a time, without touching existing rows.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path

from ._attempt import ATTEMPT_ID_PATTERN, VALID_OUTCOMES, Attempt
from ._errors import CortexStorageError
from ._event import EVENT_KIND_ATTEMPT_RECORDED, EVENT_KIND_MEMORY_RECORDED, EVENT_KIND_SKILL_PROMOTED, Event
from ._evidence import EVIDENCE_ID_PATTERN, Evidence
from ._memory import (
    EPISTEMIC_VERIFIED,
    KIND_LESSON,
    MEMORY_ID_PATTERN,
    VALID_EPISTEMIC_STATES,
    VALID_KINDS,
    Memory,
)
from ._skill import SKILL_CANDIDATE, SKILL_ID_PATTERN, SKILL_VERIFIED, VALID_SKILL_VERIFICATION_STATES, Skill

_SCHEMA_VERSION_V1 = 1
_SCHEMA_VERSION_V2 = 2
_SCHEMA_VERSION_V3 = 3
_SCHEMA_VERSION_V4 = 4
STORE_SCHEMA_VERSION = _SCHEMA_VERSION_V4
DB_FILENAME = "memory.db"

_V2_TABLES = ("memories", "evidence", "memory_evidence", "events")
_V3_TABLES = _V2_TABLES + ("attempts", "attempt_evidence")
_V4_TABLES = _V3_TABLES + ("skills", "skill_steps", "skill_conditions", "skill_evidence")

_CREATE_MEMORIES_V2_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        supersedes TEXT
    )
"""

_CREATE_EVIDENCE_SQL = """
    CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""

_CREATE_MEMORY_EVIDENCE_SQL = """
    CREATE TABLE memory_evidence (
        memory_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (memory_id, evidence_id)
    )
"""

_CREATE_EVENTS_SQL = """
    CREATE TABLE events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        occurred_at TEXT NOT NULL
    )
"""

_CREATE_ATTEMPTS_SQL = """
    CREATE TABLE attempts (
        attempt_id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        approach TEXT NOT NULL,
        outcome TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""

_CREATE_ATTEMPT_EVIDENCE_SQL = """
    CREATE TABLE attempt_evidence (
        attempt_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (attempt_id, evidence_id)
    )
"""

_CREATE_SKILLS_SQL = """
    CREATE TABLE skills (
        skill_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        purpose TEXT NOT NULL,
        verification_state TEXT NOT NULL,
        source_lesson_id TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""

_CREATE_SKILL_STEPS_SQL = """
    CREATE TABLE skill_steps (
        skill_id TEXT NOT NULL,
        step TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, position)
    )
"""

_CREATE_SKILL_CONDITIONS_SQL = """
    CREATE TABLE skill_conditions (
        skill_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, position)
    )
"""

_CREATE_SKILL_EVIDENCE_SQL = """
    CREATE TABLE skill_evidence (
        skill_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, evidence_id)
    )
"""


def db_path_for(cortex_dir: Path) -> Path:
    return cortex_dir / DB_FILENAME


class MemoryStore:
    """Boundary around the persisted memory/evidence/event/attempt tables
    for a single workspace."""

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

    # -- memories ---------------------------------------------------------

    def add(self, memory: Memory, events: Sequence[Event]) -> None:
        """Persist a new memory, its evidence links, and its events atomically.

        Raises `ValueError` if `memory.supersedes` names an unknown memory,
        or if any `memory.evidence_ids` entry names unknown evidence. Raises
        `CortexStorageError` on genuine storage corruption or I/O failure.
        """
        try:
            with self._connection:
                if memory.supersedes is not None:
                    if not self._memory_exists(memory.supersedes):
                        raise ValueError(f"Cannot supersede unknown memory {memory.supersedes!r}")
                    if self._has_superseder(memory.supersedes):
                        raise ValueError(
                            f"Memory {memory.supersedes!r} has already been superseded; "
                            "a memory can only be superseded once"
                        )
                for evidence_id in memory.evidence_ids:
                    if not self._evidence_exists(evidence_id):
                        raise ValueError(f"Unknown evidence reference {evidence_id!r}")

                self._connection.execute(
                    "INSERT INTO memories "
                    "(memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        memory.memory_id,
                        memory.content,
                        memory.kind,
                        memory.epistemic_state,
                        memory.recorded_at.isoformat(),
                        memory.supersedes,
                    ),
                )
                for position, evidence_id in enumerate(memory.evidence_ids):
                    self._connection.execute(
                        "INSERT INTO memory_evidence (memory_id, evidence_id, position) "
                        "VALUES (?, ?, ?)",
                        (memory.memory_id, evidence_id, position),
                    )
                for event in events:
                    self._connection.execute(
                        "INSERT INTO events (event_id, kind, subject_id, occurred_at) "
                        "VALUES (?, ?, ?, ?)",
                        (event.event_id, event.kind, event.subject_id, event.occurred_at.isoformat()),
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

    def search(self, query: str, limit: int, *, current_only: bool) -> list[Memory]:
        """Case-insensitive substring search, ranked by match count.

        Ties are broken by most-recent-first, then by `memory_id`, so
        the result order never depends on incidental database order.
        When `current_only` is True, superseded memories are excluded.
        """
        needle = query.casefold()
        rows = self._fetch_all_memory_rows()
        current_ids = self._current_ids_from_rows(rows) if current_only else None
        evidence_map = self._all_memory_evidence_map()

        matches: list[tuple[int, Memory]] = []
        for row in rows:
            memory = _row_to_memory(row, evidence_map.get(row[0], ()))
            if current_ids is not None and memory.memory_id not in current_ids:
                continue
            occurrences = memory.content.casefold().count(needle)
            if occurrences > 0:
                matches.append((occurrences, memory))

        matches.sort(key=lambda pair: pair[1].memory_id)
        matches.sort(key=lambda pair: pair[1].recorded_at, reverse=True)
        matches.sort(key=lambda pair: pair[0], reverse=True)

        return [memory for _, memory in matches[:limit]]

    def timeline(self, kind: str | None) -> list[Memory]:
        """Return memories in the order Cortex recorded them, oldest first.

        Ordering follows the append-only event log's own sequence, not
        `recorded_at` timestamps (which can collide) or incidental
        database row order.
        """
        try:
            cursor = self._connection.execute(
                "SELECT subject_id FROM events WHERE kind = ? ORDER BY sequence",
                (EVENT_KIND_MEMORY_RECORDED,),
            )
            subject_ids = [row[0] for row in cursor.fetchall()]
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex event log: {exc}") from exc

        evidence_map = self._all_memory_evidence_map()
        results: list[Memory] = []
        for memory_id in subject_ids:
            row = self._fetch_memory_row(memory_id)
            if row is None:
                raise CortexStorageError(
                    f"Event log references memory {memory_id!r} that does not exist"
                )
            memory = _row_to_memory(row, evidence_map.get(memory_id, ()))
            if kind is not None and memory.kind != kind:
                continue
            results.append(memory)
        return results

    def current_ids(self) -> set[str]:
        rows = self._fetch_all_memory_rows()
        return self._current_ids_from_rows(rows)

    # -- evidence -----------------------------------------------------------

    def add_evidence(self, evidence: Evidence) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO evidence (evidence_id, content, kind, recorded_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        evidence.evidence_id,
                        evidence.content,
                        evidence.kind,
                        evidence.recorded_at.isoformat(),
                    ),
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist evidence {evidence.evidence_id!r}: {exc}") from exc

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        try:
            row = self._connection.execute(
                "SELECT evidence_id, content, kind, recorded_at FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex evidence store: {exc}") from exc
        if row is None:
            return None
        return _row_to_evidence(row)

    # -- attempts -----------------------------------------------------------

    def add_attempt(self, attempt: Attempt, event: Event) -> None:
        """Persist a new attempt, its evidence links, and its event atomically.

        Raises `ValueError` if any `attempt.evidence_ids` entry names
        unknown evidence. Raises `CortexStorageError` on genuine storage
        corruption or I/O failure.
        """
        try:
            with self._connection:
                for evidence_id in attempt.evidence_ids:
                    if not self._evidence_exists(evidence_id):
                        raise ValueError(f"Unknown evidence reference {evidence_id!r}")

                self._connection.execute(
                    "INSERT INTO attempts (attempt_id, task, approach, outcome, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        attempt.attempt_id,
                        attempt.task,
                        attempt.approach,
                        attempt.outcome,
                        attempt.recorded_at.isoformat(),
                    ),
                )
                for position, evidence_id in enumerate(attempt.evidence_ids):
                    self._connection.execute(
                        "INSERT INTO attempt_evidence (attempt_id, evidence_id, position) "
                        "VALUES (?, ?, ?)",
                        (attempt.attempt_id, evidence_id, position),
                    )
                self._connection.execute(
                    "INSERT INTO events (event_id, kind, subject_id, occurred_at) "
                    "VALUES (?, ?, ?, ?)",
                    (event.event_id, event.kind, event.subject_id, event.occurred_at.isoformat()),
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist attempt {attempt.attempt_id!r}: {exc}") from exc

    def list_attempts(self) -> list[Attempt]:
        """Return every attempt in the order Cortex recorded them, oldest
        first. Attempts are append-only: nothing here is ever rewritten,
        including failed ones."""
        try:
            cursor = self._connection.execute(
                "SELECT subject_id FROM events WHERE kind = ? ORDER BY sequence",
                (EVENT_KIND_ATTEMPT_RECORDED,),
            )
            attempt_ids = [row[0] for row in cursor.fetchall()]
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex event log: {exc}") from exc

        evidence_map = self._all_attempt_evidence_map()
        results: list[Attempt] = []
        for attempt_id in attempt_ids:
            row = self._fetch_attempt_row(attempt_id)
            if row is None:
                raise CortexStorageError(
                    f"Event log references attempt {attempt_id!r} that does not exist"
                )
            results.append(_row_to_attempt(row, evidence_map.get(attempt_id, ())))
        return results

    # -- skills ---------------------------------------------------------------

    def add_skill(
        self,
        skill_id: str,
        *,
        name: str,
        purpose: str,
        steps: Sequence[str],
        conditions: Sequence[str],
        source_lesson_id: str,
        recorded_at: dt.datetime,
        event: Event,
    ) -> Skill:
        """Persist a new skill, its steps/conditions/evidence links, and its
        event atomically, and return the persisted `Skill`.

        `verification_state` and `evidence_ids` are never taken from the
        caller. They are derived here, inside the same transaction, from
        the CANONICAL Lesson memory actually persisted under
        `source_lesson_id` — its own `epistemic_state` and its own
        evidence links — so a caller cannot elevate a Skill's verification
        or redirect its provenance by handing `Cortex.promote()` a
        `Memory` object whose fields disagree with what Cortex itself has
        on record for that id (e.g. a forged `epistemic_state="verified"`
        or forged `evidence_ids` on an object that merely shares a real
        Lesson's `memory_id`). The persisted Lesson row is the only
        authority; nothing about the caller's object is trusted beyond
        which id to look up.

        Raises `ValueError` if `source_lesson_id` does not name an
        existing memory of kind `lesson`. Raises `CortexStorageError` on
        genuine storage corruption or I/O failure.
        """
        try:
            with self._connection:
                lesson_row = self._fetch_memory_row(source_lesson_id)
                if lesson_row is None or lesson_row[2] != KIND_LESSON:
                    raise ValueError(
                        f"Cannot promote unknown lesson {source_lesson_id!r}; "
                        "a skill can only be promoted from an existing lesson memory"
                    )
                canonical_lesson = _row_to_memory(lesson_row, self._memory_evidence_ids(source_lesson_id))

                verification_state = (
                    SKILL_VERIFIED if canonical_lesson.epistemic_state == EPISTEMIC_VERIFIED else SKILL_CANDIDATE
                )

                skill = Skill(
                    skill_id=skill_id,
                    name=name,
                    purpose=purpose,
                    steps=tuple(steps),
                    conditions=tuple(conditions),
                    verification_state=verification_state,
                    source_lesson_id=source_lesson_id,
                    evidence_ids=canonical_lesson.evidence_ids,
                    recorded_at=recorded_at,
                )

                self._connection.execute(
                    "INSERT INTO skills "
                    "(skill_id, name, purpose, verification_state, source_lesson_id, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        skill.skill_id,
                        skill.name,
                        skill.purpose,
                        skill.verification_state,
                        skill.source_lesson_id,
                        skill.recorded_at.isoformat(),
                    ),
                )
                for position, step in enumerate(skill.steps):
                    self._connection.execute(
                        "INSERT INTO skill_steps (skill_id, step, position) VALUES (?, ?, ?)",
                        (skill.skill_id, step, position),
                    )
                for position, condition in enumerate(skill.conditions):
                    self._connection.execute(
                        "INSERT INTO skill_conditions (skill_id, condition, position) VALUES (?, ?, ?)",
                        (skill.skill_id, condition, position),
                    )
                for position, evidence_id in enumerate(skill.evidence_ids):
                    self._connection.execute(
                        "INSERT INTO skill_evidence (skill_id, evidence_id, position) VALUES (?, ?, ?)",
                        (skill.skill_id, evidence_id, position),
                    )
                self._connection.execute(
                    "INSERT INTO events (event_id, kind, subject_id, occurred_at) "
                    "VALUES (?, ?, ?, ?)",
                    (event.event_id, event.kind, event.subject_id, event.occurred_at.isoformat()),
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist skill {skill_id!r}: {exc}") from exc
        return skill

    def get_skill(self, skill_id: str) -> Skill | None:
        row = self._fetch_skill_row(skill_id)
        if row is None:
            return None
        return _row_to_skill(
            row,
            steps=self._skill_steps(skill_id),
            conditions=self._skill_conditions(skill_id),
            evidence_ids=self._skill_evidence(skill_id),
        )

    def list_skills(self) -> list[Skill]:
        """Return every skill in the order Cortex recorded them, oldest
        first. Skills are append-only: promoting a new skill never rewrites
        or removes an earlier one."""
        try:
            cursor = self._connection.execute(
                "SELECT subject_id FROM events WHERE kind = ? ORDER BY sequence",
                (EVENT_KIND_SKILL_PROMOTED,),
            )
            skill_ids = [row[0] for row in cursor.fetchall()]
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex event log: {exc}") from exc

        results: list[Skill] = []
        for skill_id in skill_ids:
            row = self._fetch_skill_row(skill_id)
            if row is None:
                raise CortexStorageError(f"Event log references skill {skill_id!r} that does not exist")
            results.append(
                _row_to_skill(
                    row,
                    steps=self._skill_steps(skill_id),
                    conditions=self._skill_conditions(skill_id),
                    evidence_ids=self._skill_evidence(skill_id),
                )
            )
        return results

    def _fetch_skill_row(self, skill_id: str) -> tuple | None:
        try:
            return self._connection.execute(
                "SELECT skill_id, name, purpose, verification_state, source_lesson_id, recorded_at "
                "FROM skills WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex skill store: {exc}") from exc

    def _skill_steps(self, skill_id: str) -> tuple[str, ...]:
        try:
            rows = self._connection.execute(
                "SELECT step FROM skill_steps WHERE skill_id = ? ORDER BY position", (skill_id,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex skill store: {exc}") from exc
        return tuple(row[0] for row in rows)

    def _skill_conditions(self, skill_id: str) -> tuple[str, ...]:
        try:
            rows = self._connection.execute(
                "SELECT condition FROM skill_conditions WHERE skill_id = ? ORDER BY position", (skill_id,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex skill store: {exc}") from exc
        return tuple(row[0] for row in rows)

    def _skill_evidence(self, skill_id: str) -> tuple[str, ...]:
        try:
            rows = self._connection.execute(
                "SELECT evidence_id FROM skill_evidence WHERE skill_id = ? ORDER BY position", (skill_id,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex skill store: {exc}") from exc
        return tuple(row[0] for row in rows)

    def _memory_evidence_ids(self, memory_id: str) -> tuple[str, ...]:
        try:
            rows = self._connection.execute(
                "SELECT evidence_id FROM memory_evidence WHERE memory_id = ? ORDER BY position",
                (memory_id,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        return tuple(row[0] for row in rows)

    # -- internal helpers -----------------------------------------------------

    def _fetch_all_memory_rows(self) -> list[tuple]:
        try:
            cursor = self._connection.execute(
                "SELECT memory_id, content, kind, epistemic_state, recorded_at, supersedes "
                "FROM memories"
            )
            return cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex memory store: {exc}") from exc

    def _fetch_memory_row(self, memory_id: str) -> tuple | None:
        try:
            return self._connection.execute(
                "SELECT memory_id, content, kind, epistemic_state, recorded_at, supersedes "
                "FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex memory store: {exc}") from exc

    def _fetch_attempt_row(self, attempt_id: str) -> tuple | None:
        try:
            return self._connection.execute(
                "SELECT attempt_id, task, approach, outcome, recorded_at "
                "FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex attempt store: {exc}") from exc

    def _all_memory_evidence_map(self) -> dict[str, tuple[str, ...]]:
        try:
            rows = self._connection.execute(
                "SELECT memory_id, evidence_id FROM memory_evidence ORDER BY memory_id, position"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        return _group_ordered(rows)

    def _all_attempt_evidence_map(self) -> dict[str, tuple[str, ...]]:
        try:
            rows = self._connection.execute(
                "SELECT attempt_id, evidence_id FROM attempt_evidence ORDER BY attempt_id, position"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        return _group_ordered(rows)

    @staticmethod
    def _current_ids_from_rows(rows: list[tuple]) -> set[str]:
        all_ids = {row[0] for row in rows}
        superseded = {row[5] for row in rows if row[5] is not None}
        return all_ids - superseded

    def _memory_exists(self, memory_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return row is not None

    def _has_superseder(self, memory_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM memories WHERE supersedes = ?", (memory_id,)
        ).fetchone()
        return row is not None

    def _evidence_exists(self, evidence_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return row is not None


def _group_ordered(rows: list[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for owner_id, linked_id in rows:
        grouped.setdefault(owner_id, []).append(linked_id)
    return {owner_id: tuple(ids) for owner_id, ids in grouped.items()}


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _create_v4_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_CREATE_MEMORIES_V2_SQL)
    connection.execute(_CREATE_EVIDENCE_SQL)
    connection.execute(_CREATE_MEMORY_EVIDENCE_SQL)
    connection.execute(_CREATE_EVENTS_SQL)
    connection.execute(_CREATE_ATTEMPTS_SQL)
    connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
    connection.execute(_CREATE_SKILLS_SQL)
    connection.execute(_CREATE_SKILL_STEPS_SQL)
    connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
    connection.execute(_CREATE_SKILL_EVIDENCE_SQL)


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Upgrade a v1 store to v2 in a single all-or-nothing transaction.

    Also backfills a `memory_recorded` event for every memory that already
    existed under v1, ordered by its own `recorded_at`, so that pre-A3
    memories remain visible to `timeline()`/`state()` (which are event-log
    projections, not raw table scans).

    Python's `sqlite3` module does not open an implicit transaction before
    DDL statements, so `BEGIN` is issued explicitly here: without it, a
    failure partway through (e.g. a colliding table name) would leave
    earlier DDL statements permanently committed instead of rolled back.
    """
    if not _table_exists(connection, "memories"):
        raise CortexStorageError(
            "Cortex memory store is stamped with schema version 1 but is missing the "
            "'memories' table; refusing to migrate a possibly corrupted store"
        )
    connection.execute("BEGIN")
    with connection:
        connection.execute("ALTER TABLE memories ADD COLUMN supersedes TEXT")
        connection.execute(_CREATE_EVIDENCE_SQL)
        connection.execute(_CREATE_MEMORY_EVIDENCE_SQL)
        connection.execute(_CREATE_EVENTS_SQL)

        preexisting = connection.execute(
            "SELECT memory_id, recorded_at FROM memories ORDER BY recorded_at, memory_id"
        ).fetchall()
        for memory_id, recorded_at in preexisting:
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, EVENT_KIND_MEMORY_RECORDED, memory_id, recorded_at),
            )

        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION_V2}")


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Upgrade a v2 store to v3 by adding the `attempts`/`attempt_evidence`
    tables. There is no v2 data to backfill: attempts are a wholly new
    concept in A4, so nothing pre-existing needs reinterpreting.
    """
    missing = [name for name in _V2_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version 2 but is missing "
            f"table(s) {missing!r}; refusing to migrate a possibly corrupted store"
        )
    connection.execute("BEGIN")
    with connection:
        connection.execute(_CREATE_ATTEMPTS_SQL)
        connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION_V3}")


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Upgrade a v3 store to v4 by adding the `skills`/`skill_steps`/
    `skill_conditions`/`skill_evidence` tables. There is no v3 data to
    backfill: Skill is a wholly new concept in A5, so nothing pre-existing
    needs reinterpreting.
    """
    missing = [name for name in _V3_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version 3 but is missing "
            f"table(s) {missing!r}; refusing to migrate a possibly corrupted store"
        )
    connection.execute("BEGIN")
    with connection:
        connection.execute(_CREATE_SKILLS_SQL)
        connection.execute(_CREATE_SKILL_STEPS_SQL)
        connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
        connection.execute(_CREATE_SKILL_EVIDENCE_SQL)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION_V4}")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    (version,) = connection.execute("PRAGMA user_version").fetchone()

    if version == 0:
        if any(_table_exists(connection, name) for name in _V4_TABLES):
            raise CortexStorageError(
                "Cortex memory store has no recognized schema version but already "
                "contains data tables; refusing to open a possibly corrupted store"
            )
        connection.execute("BEGIN")
        with connection:
            _create_v4_schema(connection)
            connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
        return

    if version == _SCHEMA_VERSION_V1:
        _migrate_v1_to_v2(connection)
        version = _SCHEMA_VERSION_V2

    if version == _SCHEMA_VERSION_V2:
        _migrate_v2_to_v3(connection)
        version = _SCHEMA_VERSION_V3

    if version == _SCHEMA_VERSION_V3:
        _migrate_v3_to_v4(connection)
        version = _SCHEMA_VERSION_V4

    if version != STORE_SCHEMA_VERSION:
        raise CortexStorageError(
            f"Cortex memory store schema version {version} is not supported by this "
            f"version of Cortex (expected {STORE_SCHEMA_VERSION})"
        )

    missing = [name for name in _V4_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version {version} but is "
            f"missing table(s) {missing!r}; refusing to silently recreate them"
        )


def _row_to_memory(row: tuple, evidence_ids: tuple[str, ...]) -> Memory:
    memory_id, content, kind, epistemic_state, recorded_at_raw, supersedes = row

    if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
        raise CortexStorageError(f"Corrupted memory_id {memory_id!r} in Cortex memory store")
    if kind not in VALID_KINDS:
        raise CortexStorageError(f"Corrupted kind {kind!r} for memory {memory_id!r}")
    if epistemic_state not in VALID_EPISTEMIC_STATES:
        raise CortexStorageError(
            f"Corrupted epistemic_state {epistemic_state!r} for memory {memory_id!r}"
        )
    if supersedes is not None and (
        not isinstance(supersedes, str) or not MEMORY_ID_PATTERN.fullmatch(supersedes)
    ):
        raise CortexStorageError(f"Corrupted supersedes value {supersedes!r} for memory {memory_id!r}")
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
        supersedes=supersedes,
        evidence_ids=evidence_ids,
    )


def _row_to_evidence(row: tuple[str, str, str, str]) -> Evidence:
    evidence_id, content, kind, recorded_at_raw = row

    if not isinstance(evidence_id, str) or not EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
        raise CortexStorageError(f"Corrupted evidence_id {evidence_id!r} in Cortex evidence store")
    try:
        recorded_at = dt.datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CortexStorageError(
            f"Corrupted recorded_at value {recorded_at_raw!r} for evidence {evidence_id!r}"
        ) from exc
    return Evidence(evidence_id=evidence_id, content=content, kind=kind, recorded_at=recorded_at)


def _row_to_attempt(row: tuple[str, str, str, str, str], evidence_ids: tuple[str, ...]) -> Attempt:
    attempt_id, task, approach, outcome, recorded_at_raw = row

    if not isinstance(attempt_id, str) or not ATTEMPT_ID_PATTERN.fullmatch(attempt_id):
        raise CortexStorageError(f"Corrupted attempt_id {attempt_id!r} in Cortex attempt store")
    if outcome not in VALID_OUTCOMES:
        raise CortexStorageError(f"Corrupted outcome {outcome!r} for attempt {attempt_id!r}")
    try:
        recorded_at = dt.datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CortexStorageError(
            f"Corrupted recorded_at value {recorded_at_raw!r} for attempt {attempt_id!r}"
        ) from exc
    return Attempt(
        attempt_id=attempt_id,
        task=task,
        approach=approach,
        outcome=outcome,
        recorded_at=recorded_at,
        evidence_ids=evidence_ids,
    )


def _row_to_skill(
    row: tuple[str, str, str, str, str, str],
    *,
    steps: tuple[str, ...],
    conditions: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> Skill:
    skill_id, name, purpose, verification_state, source_lesson_id, recorded_at_raw = row

    if not isinstance(skill_id, str) or not SKILL_ID_PATTERN.fullmatch(skill_id):
        raise CortexStorageError(f"Corrupted skill_id {skill_id!r} in Cortex skill store")
    if verification_state not in VALID_SKILL_VERIFICATION_STATES:
        raise CortexStorageError(
            f"Corrupted verification_state {verification_state!r} for skill {skill_id!r}"
        )
    if not isinstance(source_lesson_id, str) or not MEMORY_ID_PATTERN.fullmatch(source_lesson_id):
        raise CortexStorageError(f"Corrupted source_lesson_id {source_lesson_id!r} for skill {skill_id!r}")
    try:
        recorded_at = dt.datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CortexStorageError(
            f"Corrupted recorded_at value {recorded_at_raw!r} for skill {skill_id!r}"
        ) from exc
    return Skill(
        skill_id=skill_id,
        name=name,
        purpose=purpose,
        steps=steps,
        conditions=conditions,
        verification_state=verification_state,
        source_lesson_id=source_lesson_id,
        evidence_ids=evidence_ids,
        recorded_at=recorded_at,
    )
