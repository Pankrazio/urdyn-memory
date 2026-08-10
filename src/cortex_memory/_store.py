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
  v5 - adds `memory_evidence.role` (A12.1): distinguishes Evidence the
       caller explicitly designated as SUPPORTING a memory from Evidence
       that is merely generically related/provenance. Existing rows are
       backfilled to `'related'` (never `'supporting'`: a pre-v5 row
       never recorded an explicit support assertion, and none is
       invented for it retroactively). Only `memories`/`memory_evidence`
       are affected; `attempt_evidence`/`skill_evidence` are unchanged --
       A12.1 deliberately does not extend this concept to Attempt or
       Skill (see `_workspace.py`'s A12.1 notes).
  v6 - adds `memory_conflicts` (A13.1): a single symmetric relation table
       recording that two Memories are explicitly asserted to conflict.
       No existing table changes shape. The pair `(memory_id_a,
       memory_id_b)` is stored canonically ordered (ascending) so that
       declaring A<->B and B<->A collapse onto the SAME row -- this is
       the entire mechanism behind idempotent duplicate/reverse-duplicate
       declarations (see `MemoryStore.add_conflict`). Deliberately no
       `conflict_id`, no evidence link, no status/resolution column:
       whether a conflict is "open" is derived at read time from
       `current_ids()`, never stored (see `_conflict.py`'s module
       docstring for the full reasoning).

A v1, v2, v3, v4, or v5 database opened by this module is migrated
forward to v6 in place, one step at a time, without touching existing
rows.

Search index (derived, not versioned):
  A FTS5 virtual table `search_index` provides candidate widening for
  `preflight()`/`guard()` (see `_retrieval.py`). It is deliberately NOT
  part of `STORE_SCHEMA_VERSION`: it holds no canonical truth, only a
  rebuildable projection of `memories.content`, `attempts.task`/
  `approach`, and `skills.name`/`purpose`/conditions, keyed by their own
  canonical ids. Folding it into the versioned migration chain would
  conflate "the data Cortex holds changed" with "a derived search aid
  was (re)built" -- two different kinds of change with different
  failure semantics (a missing/stale index degrades search; a missing
  canonical table is corruption). Instead, `_ensure_schema` creates and
  backfills it lazily, once, outside the version chain, safe to skip
  entirely if this SQLite build has no FTS5 support: `preflight()`/
  `guard()` still work in that case, using only the lexical channel
  that predates A7.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path

from ._attempt import ATTEMPT_ID_PATTERN, VALID_OUTCOMES, Attempt
from ._conflict import Conflict, canonical_pair
from ._errors import CortexStorageError
from ._event import EVENT_KIND_ATTEMPT_RECORDED, EVENT_KIND_MEMORY_RECORDED, EVENT_KIND_SKILL_PROMOTED, Event
from ._evidence import EVIDENCE_ID_PATTERN, VERIFICATION_EVIDENCE_KINDS, Evidence
from ._memory import (
    EPISTEMIC_VERIFIED,
    KIND_LESSON,
    MEMORY_ID_PATTERN,
    VALID_EPISTEMIC_STATES,
    VALID_KINDS,
    Memory,
)
from ._relevance import attempt_search_text, memory_search_text, skill_search_text
from ._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL
from ._skill import SKILL_CANDIDATE, SKILL_ID_PATTERN, SKILL_VERIFIED, VALID_SKILL_VERIFICATION_STATES, Skill

_SCHEMA_VERSION_V1 = 1
_SCHEMA_VERSION_V2 = 2
_SCHEMA_VERSION_V3 = 3
_SCHEMA_VERSION_V4 = 4
_SCHEMA_VERSION_V5 = 5
_SCHEMA_VERSION_V6 = 6
STORE_SCHEMA_VERSION = _SCHEMA_VERSION_V6
DB_FILENAME = "memory.db"

_V2_TABLES = ("memories", "evidence", "memory_evidence", "events")
_V3_TABLES = _V2_TABLES + ("attempts", "attempt_evidence")
_V4_TABLES = _V3_TABLES + ("skills", "skill_steps", "skill_conditions", "skill_evidence")
# v5 adds no new table, only a column on the existing `memory_evidence`
# table (see module docstring) -- the set of required tables is unchanged.
_V5_TABLES = _V4_TABLES
_V6_TABLES = _V5_TABLES + ("memory_conflicts",)

# Canonical values for `memory_evidence.role` (A12.1). Internal storage
# vocabulary only -- never exposed as SQL or as these literal strings in
# the public API; the public model expresses the same distinction via
# `Memory.evidence_ids` (all) vs `Memory.supporting_evidence_ids` (the
# `_ROLE_SUPPORTING` subset).
_ROLE_RELATED = "related"
_ROLE_SUPPORTING = "supporting"

# Derived search index: deliberately outside `_V4_TABLES`/`STORE_SCHEMA_VERSION`
# (see module docstring). `SEARCH_INDEX_TABLE` is exported for tests that
# need to simulate index loss/corruption against the real table name.
SEARCH_INDEX_TABLE = "search_index"

_CREATE_SEARCH_INDEX_SQL = f"""
    CREATE VIRTUAL TABLE {SEARCH_INDEX_TABLE} USING fts5(
        entity_type UNINDEXED,
        entity_id UNINDEXED,
        text
    )
"""

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

# v5 shape of the same table (see module docstring): used only to create
# a brand-new store directly at the current schema. A store migrating up
# from v1 still passes through `_CREATE_MEMORY_EVIDENCE_SQL` (the v2
# shape) at the v1->v2 step, then gains `role` via `_migrate_v4_to_v5`'s
# `ALTER TABLE` -- this constant is never used by that migration chain.
_CREATE_MEMORY_EVIDENCE_V5_SQL = f"""
    CREATE TABLE memory_evidence (
        memory_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        role TEXT NOT NULL DEFAULT '{_ROLE_RELATED}',
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

# v6 (A13.1): a single symmetric relation, keyed by the canonically
# ordered pair itself -- see the module docstring and `_conflict.py`. No
# `conflict_id`, no status, no evidence link: the pair's own primary key
# is both the identity and the uniqueness/idempotency guarantee, and
# `recorded_at` is written once and never updated by a later duplicate
# declaration (see `MemoryStore.add_conflict`).
#
# (A13.1.1) The `CHECK` is defence in depth, not the primary enforcement:
# `Cortex.record_conflict`/`add_conflict` already canonicalize the pair
# and reject self-conflict in Python, before any SQL runs. Placing the
# same invariant next to the canonical data means a row that is
# non-canonically ordered (`b < a`) or self-referential (`a == a`, which
# fails `a < b` too) cannot exist in the store AT ALL -- not even via a
# future internal call path that forgets to canonicalize, or a
# hand-edited database. This is a storage implementation detail: nothing
# about it is visible in, or promised by, the public API.
_CREATE_MEMORY_CONFLICTS_SQL = """
    CREATE TABLE memory_conflicts (
        memory_id_a TEXT NOT NULL,
        memory_id_b TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (memory_id_a, memory_id_b),
        CHECK (memory_id_a < memory_id_b)
    )
"""


def db_path_for(cortex_dir: Path) -> Path:
    return cortex_dir / DB_FILENAME


class MemoryStore:
    """Boundary around the persisted memory/evidence/event/attempt tables
    for a single workspace."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        # Computed fresh on every open (not cached across process
        # lifetimes): reflects whatever `_ensure_schema` just did, which
        # itself degrades gracefully if this SQLite build has no FTS5.
        self._fts_enabled = _table_exists(connection, SEARCH_INDEX_TABLE)

    @property
    def fts_enabled(self) -> bool:
        """Whether the derived FTS5 search index is available in this
        store. False on a SQLite build without FTS5 support -- an
        expected, handled condition, not an error. `preflight()`/
        `guard()` remain fully functional either way, using only the
        lexical channel when this is False."""
        return self._fts_enabled

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
        if any `memory.evidence_ids` entry names unknown evidence, if
        `memory.supporting_evidence_ids` is not a subset of
        `memory.evidence_ids`, or (A12.1.1) if `memory.epistemic_state`
        is `verified` without at least one supporting Evidence of a
        qualifying kind. Raises `CortexStorageError` on genuine storage
        corruption or I/O failure.

        WRITE-BOUNDARY NOTE (A12.1.1): `Cortex.remember()` is still the
        primary, user-facing enforcement point for the verified contract
        (see its docstring) -- this repeats the SAME rule, against the
        SAME `VERIFICATION_EVIDENCE_KINDS` constant, at the canonical
        storage boundary itself, so the invariant does not depend
        entirely on every future internal call path (a batch importer,
        a repair tool, anything constructing a `Memory` directly and
        calling `add()`) remembering to go through `remember()`. This is
        a WRITE-time check only -- it runs once, here, when a row is
        first inserted. It is deliberately NOT re-checked on any READ
        path (`search`/`timeline`/`state` below): a `verified` memory
        migrated from schema v4 legitimately has an empty
        `supporting_evidence_ids` (A12.1 grandfathering -- that concept
        did not exist when it was recorded) and must keep loading
        exactly as recorded. Applying this rule on read would silently
        contradict that grandfathering guarantee.
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
                if not set(memory.supporting_evidence_ids).issubset(memory.evidence_ids):
                    raise ValueError(
                        "supporting_evidence_ids must be a subset of evidence_ids"
                    )
                if memory.epistemic_state == EPISTEMIC_VERIFIED:
                    if not memory.supporting_evidence_ids:
                        raise ValueError(
                            "A memory cannot be persisted as verified without at least one "
                            "explicitly designated supporting Evidence"
                        )
                    supporting_kinds = {
                        self._evidence_kind(evidence_id) for evidence_id in memory.supporting_evidence_ids
                    }
                    if supporting_kinds.isdisjoint(VERIFICATION_EVIDENCE_KINDS):
                        raise ValueError(
                            "A memory can only be persisted as verified with supporting evidence "
                            f"of a qualifying kind (one of {sorted(VERIFICATION_EVIDENCE_KINDS)})"
                        )

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
                self._index_entity(ENTITY_MEMORY, memory.memory_id, memory_search_text(memory.content))
                supporting = frozenset(memory.supporting_evidence_ids)
                for position, evidence_id in enumerate(memory.evidence_ids):
                    role = _ROLE_SUPPORTING if evidence_id in supporting else _ROLE_RELATED
                    self._connection.execute(
                        "INSERT INTO memory_evidence (memory_id, evidence_id, position, role) "
                        "VALUES (?, ?, ?, ?)",
                        (memory.memory_id, evidence_id, position, role),
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
        supporting_map = self._all_memory_supporting_evidence_map()

        matches: list[tuple[int, Memory]] = []
        for row in rows:
            memory = _row_to_memory(row, evidence_map.get(row[0], ()), supporting_map.get(row[0], ()))
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
        supporting_map = self._all_memory_supporting_evidence_map()
        results: list[Memory] = []
        for memory_id in subject_ids:
            row = self._fetch_memory_row(memory_id)
            if row is None:
                raise CortexStorageError(
                    f"Event log references memory {memory_id!r} that does not exist"
                )
            memory = _row_to_memory(row, evidence_map.get(memory_id, ()), supporting_map.get(memory_id, ()))
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
                self._index_entity(
                    ENTITY_ATTEMPT, attempt.attempt_id, attempt_search_text(attempt.task, attempt.approach)
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
                canonical_lesson = _row_to_memory(
                    lesson_row,
                    self._memory_evidence_ids(source_lesson_id),
                    self._memory_supporting_evidence_ids(source_lesson_id),
                )

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
                self._index_entity(
                    ENTITY_SKILL, skill.skill_id, skill_search_text(skill.name, skill.purpose, skill.conditions)
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

    # -- conflicts ----------------------------------------------------------

    def add_conflict(self, memory_id_a: str, memory_id_b: str, recorded_at: dt.datetime) -> Conflict:
        """Idempotently persist a symmetric conflict relation between two
        existing memories and return the canonical `Conflict`.

        Raises `ValueError` if `memory_id_a == memory_id_b` (self-conflict)
        or if either id does not name an existing memory. Neither memory
        is required to be current: a conflict between historical memories
        is a legitimate fact about the past (see `_conflict.py`'s module
        docstring).

        Idempotency (A13.1 review decision 3/4/5): `record_conflict(A, B)`,
        `record_conflict(A, B)` again, and `record_conflict(B, A)` all
        resolve to the SAME row (identified by the canonically ordered
        pair, see `canonical_pair`) and never insert a second one. If the
        pair was already declared, the ALREADY-persisted `recorded_at` is
        returned -- the `recorded_at` argument passed on a later call is
        silently discarded, exactly like `_migrate_v4_to_v5`'s DEFAULT-based
        backfill discards any invented value: a repeat declaration is not
        a new transition, so it must not appear to change when the
        relation was first recorded.

        (A13.1.1) That idempotency is deliberately expressed as
        `ON CONFLICT(memory_id_a, memory_id_b) DO NOTHING` rather than
        `INSERT OR IGNORE`. The two behave identically for the case this
        method actually wants to absorb -- a repeated declaration of the
        SAME canonical pair -- but `OR IGNORE` suppresses EVERY constraint
        violation on the statement, including the table's own
        `CHECK (memory_id_a < memory_id_b)` and its `NOT NULL`s. That
        would turn a genuine canonical-integrity failure (a
        non-canonically ordered or self-referential pair reaching SQL)
        into a silent no-op that this method would then report as
        success by returning the pre-existing row. Targeting the primary
        key explicitly keeps "this exact relation was already declared"
        as the only condition that is ever absorbed; anything else still
        raises and is wrapped as `CortexStorageError` below.
        """
        if memory_id_a == memory_id_b:
            raise ValueError("A memory cannot conflict with itself")
        pair = canonical_pair(memory_id_a, memory_id_b)
        try:
            with self._connection:
                for memory_id in pair:
                    if not self._memory_exists(memory_id):
                        raise ValueError(f"Cannot record conflict for unknown memory {memory_id!r}")
                self._connection.execute(
                    "INSERT INTO memory_conflicts (memory_id_a, memory_id_b, recorded_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(memory_id_a, memory_id_b) DO NOTHING",
                    (pair[0], pair[1], recorded_at.isoformat()),
                )
                row = self._connection.execute(
                    "SELECT memory_id_a, memory_id_b, recorded_at FROM memory_conflicts "
                    "WHERE memory_id_a = ? AND memory_id_b = ?",
                    pair,
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist conflict {pair!r}: {exc}") from exc
        return _row_to_conflict(row)

    def list_conflicts(self) -> list[Conflict]:
        """Return every declared conflict relation, oldest first by
        `recorded_at`, tie-broken by the canonical pair itself so ordering
        never depends on incidental database row order (mirrors the
        tie-break discipline `search()` already applies to memories).

        (A13.1.1) Every participant is verified to name a memory this
        store actually holds, and a dangling one raises
        `CortexStorageError`. `_row_to_conflict` alone is not enough:
        it validates that each id is well-FORMED (32 hex, canonically
        ordered), which a corrupted or hand-edited row can satisfy while
        still pointing at a memory that does not exist. Returning such a
        row would hand callers a `Conflict` that claims to be a canonical
        relation between two recorded Memories when one of them is not
        recorded at all -- and, worse, `open_conflicts()` would quietly
        drop it (a nonexistent id is never in `current_ids()`), making
        storage corruption indistinguishable from the legitimate,
        expected "this conflict is no longer open" answer. Cortex does
        not repair or delete the row: it refuses to reinterpret it, the
        same standard `_all_memory_supporting_evidence_map` already
        applies to an unrecognized `memory_evidence.role`.
        """
        try:
            rows = self._connection.execute(
                "SELECT memory_id_a, memory_id_b, recorded_at FROM memory_conflicts "
                "ORDER BY recorded_at ASC, memory_id_a ASC, memory_id_b ASC"
            ).fetchall()
            known_ids = {
                row[0] for row in self._connection.execute("SELECT memory_id FROM memories").fetchall()
            }
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex conflict store: {exc}") from exc

        conflicts = [_row_to_conflict(row) for row in rows]
        for conflict in conflicts:
            for memory_id in conflict.memory_ids:
                if memory_id not in known_ids:
                    raise CortexStorageError(
                        f"Conflict {conflict.memory_ids!r} references memory {memory_id!r} "
                        "that does not exist in the Cortex memory store"
                    )
        return conflicts

    # -- search index (derived) --------------------------------------------

    def _index_entity(self, entity_type: str, entity_id: str, text: str) -> None:
        """Append one row to the derived search index, in the same
        transaction as the canonical write it accompanies. A no-op when
        this store has no FTS5 support: the canonical write is still
        the source of truth, this is purely an optional search aid.
        Must only be called from inside an already-open `with
        self._connection:` block, exactly like the canonical inserts it
        sits next to -- if that write rolls back, this row rolls back
        with it, so the index can never drift ahead of canonical data.
        """
        if not self._fts_enabled:
            return
        self._connection.execute(
            f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
            (entity_type, entity_id, text),
        )

    def search_candidates(self, query_tokens: frozenset[str], entity_type: str) -> list[tuple[str, str]]:
        """Return `(entity_id, text)` candidates of `entity_type` ranked
        best-first by BM25, ties broken by `entity_id` for deterministic
        ordering. Returns `[]` without querying at all if the FTS index
        is unavailable (`fts_enabled` is False) or `query_tokens` is
        empty -- both expected conditions, not failures. Every token is
        double-quoted in the generated MATCH expression so it is always
        treated as a literal term, never as FTS5 query syntax (an
        operator keyword, a column filter, punctuation): safe because
        `query_tokens` only ever contains `_relevance.tokens()` output,
        which cannot itself contain a quote character.
        """
        if not self._fts_enabled or not query_tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in sorted(query_tokens))
        try:
            rows = self._connection.execute(
                f"SELECT entity_id, text FROM {SEARCH_INDEX_TABLE} "
                f"WHERE entity_type = ? AND {SEARCH_INDEX_TABLE} MATCH ? "
                f"ORDER BY bm25({SEARCH_INDEX_TABLE}) ASC, entity_id ASC",
                (entity_type, match_query),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to search Cortex search index: {exc}") from exc
        return [(row[0], row[1]) for row in rows]

    def rebuild_search_index(self) -> None:
        """Fully rebuild the derived search index from canonical data,
        discarding whatever it currently holds first. Internal
        maintenance primitive, not a public Cortex command: the index is
        disposable and derived, so this is always safe to call, and is
        exactly what recovers a store opened once without FTS5 support
        (index missing) that is later opened by a build that has it.
        No-op if this store has no FTS5 support at all.
        """
        if not self._fts_enabled:
            return
        try:
            with self._connection:
                _rebuild_search_index(self._connection)
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to rebuild Cortex search index: {exc}") from exc

    def _memory_evidence_ids(self, memory_id: str) -> tuple[str, ...]:
        try:
            rows = self._connection.execute(
                "SELECT evidence_id FROM memory_evidence WHERE memory_id = ? ORDER BY position",
                (memory_id,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        return tuple(row[0] for row in rows)

    def _memory_supporting_evidence_ids(self, memory_id: str) -> tuple[str, ...]:
        """(A12.1.1) Validates every row's `role` explicitly -- see
        `_all_memory_supporting_evidence_map`'s docstring for why
        filtering by `role = 'supporting'` alone is not enough."""
        try:
            rows = self._connection.execute(
                "SELECT evidence_id, role FROM memory_evidence WHERE memory_id = ? ORDER BY position",
                (memory_id,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        result: list[str] = []
        for evidence_id, role in rows:
            if role not in (_ROLE_RELATED, _ROLE_SUPPORTING):
                raise CortexStorageError(
                    f"Corrupted memory_evidence.role {role!r} for memory {memory_id!r}"
                )
            if role == _ROLE_SUPPORTING:
                result.append(evidence_id)
        return tuple(result)

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

    def _all_memory_supporting_evidence_map(self) -> dict[str, tuple[str, ...]]:
        """Same shape as `_all_memory_evidence_map`, restricted to rows
        explicitly designated `role = 'supporting'` (A12.1). Order is the
        same `position` sequence used for the full `evidence_ids` list --
        a supporting Evidence keeps its relative position, not a
        renumbered one, so ordering stays consistent between
        `evidence_ids` and `supporting_evidence_ids` for the same memory.

        (A12.1.1) Reads every row's `role`, not just ones already
        matching `'supporting'` in SQL, so an unrecognized value (never
        written by this codebase, but not impossible for hand-edited or
        externally-touched data) is caught explicitly rather than
        silently treated as "not supporting" -- filtering with `WHERE
        role = 'supporting'` alone would make a corrupted role
        indistinguishable from a legitimate `'related'` one, silently
        dropping a real support designation with no error at all.
        """
        try:
            rows = self._connection.execute(
                "SELECT memory_id, evidence_id, role FROM memory_evidence ORDER BY memory_id, position"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex provenance links: {exc}") from exc
        supporting_rows: list[tuple[str, str]] = []
        for memory_id, evidence_id, role in rows:
            if role not in (_ROLE_RELATED, _ROLE_SUPPORTING):
                raise CortexStorageError(
                    f"Corrupted memory_evidence.role {role!r} for memory {memory_id!r}"
                )
            if role == _ROLE_SUPPORTING:
                supporting_rows.append((memory_id, evidence_id))
        return _group_ordered(supporting_rows)

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

    def _evidence_kind(self, evidence_id: str) -> str:
        """Only called from `add()`'s write-boundary verified check
        (A12.1.1), after `evidence_id` has already been confirmed to
        exist via `_evidence_exists` -- assumes the row is present."""
        (kind,) = self._connection.execute(
            "SELECT kind FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return kind


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


def _create_v6_schema(connection: sqlite3.Connection) -> None:
    """Create every table at its CURRENT (v6) shape, for a brand-new
    store only. Not part of the v1->v6 migration chain below, which
    upgrades each table incrementally instead."""
    connection.execute(_CREATE_MEMORIES_V2_SQL)
    connection.execute(_CREATE_EVIDENCE_SQL)
    connection.execute(_CREATE_MEMORY_EVIDENCE_V5_SQL)
    connection.execute(_CREATE_EVENTS_SQL)
    connection.execute(_CREATE_ATTEMPTS_SQL)
    connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
    connection.execute(_CREATE_SKILLS_SQL)
    connection.execute(_CREATE_SKILL_STEPS_SQL)
    connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
    connection.execute(_CREATE_SKILL_EVIDENCE_SQL)
    connection.execute(_CREATE_MEMORY_CONFLICTS_SQL)


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


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """Upgrade a v4 store to v5 (A12.1) by adding `memory_evidence.role`.

    Every pre-existing `memory_evidence` row is backfilled to `'related'`
    via the column's own `DEFAULT` -- never `'supporting'`: a row written
    before A12.1 never carried an explicit caller assertion of support,
    and none is invented for it here. This is why the column is declared
    `NOT NULL DEFAULT 'related'` rather than backfilled by a separate
    `UPDATE` -- the default IS the backfill, applied atomically by SQLite
    to every existing row as part of the single `ALTER TABLE`. No other
    table changes shape in this step.
    """
    missing = [name for name in _V4_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version 4 but is missing "
            f"table(s) {missing!r}; refusing to migrate a possibly corrupted store"
        )
    connection.execute("BEGIN")
    with connection:
        connection.execute(
            f"ALTER TABLE memory_evidence ADD COLUMN role TEXT NOT NULL DEFAULT '{_ROLE_RELATED}'"
        )
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION_V5}")


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    """Upgrade a v5 store to v6 (A13.1) by adding the `memory_conflicts`
    table. There is no v5 data to backfill: Conflict is a wholly new
    relation in A13.1, so nothing pre-existing needs reinterpreting --
    no pre-v6 row ever asserted a conflict, so there is nothing to infer
    retroactively. No other table changes shape in this step.
    """
    missing = [name for name in _V5_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version 5 but is missing "
            f"table(s) {missing!r}; refusing to migrate a possibly corrupted store"
        )
    connection.execute("BEGIN")
    with connection:
        connection.execute(_CREATE_MEMORY_CONFLICTS_SQL)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION_V6}")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    (version,) = connection.execute("PRAGMA user_version").fetchone()

    if version == 0:
        if any(_table_exists(connection, name) for name in _V6_TABLES):
            raise CortexStorageError(
                "Cortex memory store has no recognized schema version but already "
                "contains data tables; refusing to open a possibly corrupted store"
            )
        connection.execute("BEGIN")
        with connection:
            _create_v6_schema(connection)
            connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
        version = STORE_SCHEMA_VERSION

    if version == _SCHEMA_VERSION_V1:
        _migrate_v1_to_v2(connection)
        version = _SCHEMA_VERSION_V2

    if version == _SCHEMA_VERSION_V2:
        _migrate_v2_to_v3(connection)
        version = _SCHEMA_VERSION_V3

    if version == _SCHEMA_VERSION_V3:
        _migrate_v3_to_v4(connection)
        version = _SCHEMA_VERSION_V4

    if version == _SCHEMA_VERSION_V4:
        _migrate_v4_to_v5(connection)
        version = _SCHEMA_VERSION_V5

    if version == _SCHEMA_VERSION_V5:
        _migrate_v5_to_v6(connection)
        version = _SCHEMA_VERSION_V6

    if version != STORE_SCHEMA_VERSION:
        raise CortexStorageError(
            f"Cortex memory store schema version {version} is not supported by this "
            f"version of Cortex (expected {STORE_SCHEMA_VERSION})"
        )

    missing = [name for name in _V6_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version {version} but is "
            f"missing table(s) {missing!r}; refusing to silently recreate them"
        )

    _ensure_search_index(connection)


def _ensure_search_index(connection: sqlite3.Connection) -> None:
    """Create and backfill the derived FTS5 search index if it does not
    already exist, in its own transaction, separate from and after the
    canonical version chain above (see module docstring for why this is
    deliberately not part of `STORE_SCHEMA_VERSION`).

    Idempotent: a store that already has `search_index` (from a prior
    open) is left untouched here. A store opened once without FTS5
    support and later reopened by a build that has it will pick the
    index up automatically the next time this runs, fully backfilled --
    the same recovery `rebuild_search_index()` offers on demand.

    A missing FTS5 module in this SQLite build is an expected, handled
    condition: `_try_create_search_index` reports it and this function
    simply leaves the store without an index rather than failing to
    open. A failure while creating the index in some other way (e.g.
    disk I/O) is not swallowed here; it propagates like any other
    `sqlite3.DatabaseError` in this module, to be wrapped by the
    `CortexStorageError` handling in `MemoryStore.create_or_open`.
    """
    if _table_exists(connection, SEARCH_INDEX_TABLE):
        return
    connection.execute("BEGIN")
    with connection:
        if _try_create_search_index(connection):
            _rebuild_search_index(connection)


def _try_create_search_index(connection: sqlite3.Connection) -> bool:
    """Attempt to create the (empty) search index table. Returns False,
    without raising, only for the specific `sqlite3.OperationalError`
    SQLite raises when the FTS5 module itself is not compiled in
    ("no such module: fts5"). `CREATE VIRTUAL TABLE ... USING fts5`
    raises the *same* `OperationalError` class for unrelated problems --
    a malformed statement, a duplicate column, any other schema-level
    bug in `_CREATE_SEARCH_INDEX_SQL` -- so catching the class alone
    would silently relabel a genuine bug as "this SQLite build has no
    FTS5" and leave the store open with no index and no visible error.
    Matching on the "no such module" message (SQLite's own stable,
    long-standing wording for this specific condition) is what keeps
    that distinction; anything else propagates like any other
    `sqlite3.DatabaseError` in this module.
    """
    try:
        connection.execute(_CREATE_SEARCH_INDEX_SQL)
    except sqlite3.OperationalError as exc:
        if "no such module" in str(exc).lower():
            return False
        raise
    return True


def _rebuild_search_index(connection: sqlite3.Connection) -> None:
    """(Re)populate `search_index` from canonical data. Must be called
    only when the table exists and from inside an already-open
    transaction (see `_ensure_search_index` and
    `MemoryStore.rebuild_search_index`): clears every row first, so a
    partial/failed run rolls back to the previous complete index rather
    than leaving a half-populated one.
    """
    connection.execute(f"DELETE FROM {SEARCH_INDEX_TABLE}")

    memory_rows = connection.execute("SELECT memory_id, content FROM memories").fetchall()
    for memory_id, content in memory_rows:
        connection.execute(
            f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
            (ENTITY_MEMORY, memory_id, memory_search_text(content)),
        )

    attempt_rows = connection.execute("SELECT attempt_id, task, approach FROM attempts").fetchall()
    for attempt_id, task, approach in attempt_rows:
        connection.execute(
            f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
            (ENTITY_ATTEMPT, attempt_id, attempt_search_text(task, approach)),
        )

    skill_rows = connection.execute("SELECT skill_id, name, purpose FROM skills").fetchall()
    for skill_id, name, purpose in skill_rows:
        condition_rows = connection.execute(
            "SELECT condition FROM skill_conditions WHERE skill_id = ? ORDER BY position", (skill_id,)
        ).fetchall()
        conditions = tuple(row[0] for row in condition_rows)
        connection.execute(
            f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
            (ENTITY_SKILL, skill_id, skill_search_text(name, purpose, conditions)),
        )


def _row_to_memory(
    row: tuple, evidence_ids: tuple[str, ...], supporting_evidence_ids: tuple[str, ...] = ()
) -> Memory:
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
        supporting_evidence_ids=supporting_evidence_ids,
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


def _row_to_conflict(row: tuple[str, str, str]) -> Conflict:
    memory_id_a, memory_id_b, recorded_at_raw = row

    for memory_id in (memory_id_a, memory_id_b):
        if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
            raise CortexStorageError(f"Corrupted memory_id {memory_id!r} in Cortex conflict store")
    if not memory_id_a < memory_id_b:
        raise CortexStorageError(
            f"Corrupted conflict pair ordering ({memory_id_a!r}, {memory_id_b!r}) in Cortex conflict store"
        )
    try:
        recorded_at = dt.datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CortexStorageError(
            f"Corrupted recorded_at value {recorded_at_raw!r} for conflict "
            f"({memory_id_a!r}, {memory_id_b!r})"
        ) from exc
    return Conflict(memory_ids=(memory_id_a, memory_id_b), recorded_at=recorded_at)


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
