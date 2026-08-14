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
  v7 - adds `sources` and `source_observations` (A19.1): the stable
       identity of an observed project file, and the append-only history
       of what Cortex saw when it looked at it. No existing table changes
       shape, and NOTHING is backfilled: A19.1 is the first producer of
       Sources, so there is no pre-v7 data to reinterpret. In particular
       existing `file_reference` Evidence rows are left exactly as they
       are -- a pre-v7 Evidence was never an observation of a tracked
       Source, and inventing one for it would mean parsing its free text
       to guess a path, which is precisely what the structured columns
       here exist to make unnecessary.

A v1, v2, v3, v4, v5, or v6 database opened by this module is migrated
forward to v7 in place, one step at a time, without touching existing
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
from collections.abc import Callable, Sequence
from pathlib import Path

from ._attempt import ATTEMPT_ID_PATTERN, VALID_OUTCOMES, Attempt
from ._conflict import Conflict, canonical_pair
from ._errors import CortexStorageError
from ._event import EVENT_KIND_ATTEMPT_RECORDED, EVENT_KIND_MEMORY_RECORDED, EVENT_KIND_SKILL_PROMOTED, Event
from ._evidence import (
    EVIDENCE_ID_PATTERN,
    EVIDENCE_KIND_DOCUMENT_OBSERVATION,
    VERIFICATION_EVIDENCE_KINDS,
    Evidence,
)
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
from ._source import (
    DIGEST_PATTERN,
    SEED_ADDED,
    SEED_CHANGED,
    SEED_UNCHANGED,
    SOURCE_ID_PATTERN,
    Source,
    SourceObservation,
)

_SCHEMA_VERSION_V1 = 1
_SCHEMA_VERSION_V2 = 2
_SCHEMA_VERSION_V3 = 3
_SCHEMA_VERSION_V4 = 4
_SCHEMA_VERSION_V5 = 5
_SCHEMA_VERSION_V6 = 6
_SCHEMA_VERSION_V7 = 7
STORE_SCHEMA_VERSION = _SCHEMA_VERSION_V7
DB_FILENAME = "memory.db"

_V2_TABLES = ("memories", "evidence", "memory_evidence", "events")
_V3_TABLES = _V2_TABLES + ("attempts", "attempt_evidence")
_V4_TABLES = _V3_TABLES + ("skills", "skill_steps", "skill_conditions", "skill_evidence")
# v5 adds no new table, only a column on the existing `memory_evidence`
# table (see module docstring) -- the set of required tables is unchanged.
_V5_TABLES = _V4_TABLES
_V6_TABLES = _V5_TABLES + ("memory_conflicts",)
_V7_TABLES = _V6_TABLES + ("sources", "source_observations")

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


# v7 (A19.1). `path` is UNIQUE because it IS how a Source is addressed:
# one workspace-relative path is one tracked file, which is what makes a
# repeated seed resolve to the same identity instead of accumulating
# duplicate Sources for the same document.
#
# The `CHECK` is defence in depth, exactly like `memory_conflicts`'s:
# `resolve_seed_path` already guarantees a workspace-relative POSIX path
# in Python, before any SQL runs. Repeating the invariant next to the
# canonical data means an absolute path cannot exist in the store at all
# -- not via a future internal call path that forgets to normalize, and
# not via a hand-edited database. A stored absolute path would break the
# portability guarantee (a copied workspace still resolving its own
# Sources) silently, which is the kind of failure worth making
# impossible rather than merely unlikely.
_CREATE_SOURCES_SQL = """
    CREATE TABLE sources (
        source_id TEXT PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        first_observed_at TEXT NOT NULL,
        CHECK (path NOT LIKE '/%')
    )
"""

# `sequence` is an ordering column, not a second identity: it answers
# "which observation came last" without depending on timestamp ties, the
# same role `events.sequence` plays alongside `events.event_id`. The
# observation's IDENTITY is `evidence_id` (UNIQUE here, one Evidence per
# observation) -- see `_source.py` on why no separate observation id is
# minted. Neither `sequence` nor any other storage detail is exposed in
# the public model.
#
# Deliberately no `ON DELETE` clause and no foreign key: nothing in
# Cortex ever deletes a Source or an observation (A19.1 has no delete, no
# GC, no rename), and referential integrity is verified fail-closed on
# read instead (see `_load_source`/`list_sources`), consistent with how
# `memory_evidence` and `attempt_evidence` already work.
_CREATE_SOURCE_OBSERVATIONS_SQL = """
    CREATE TABLE source_observations (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL UNIQUE,
        digest TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        observed_at TEXT NOT NULL,
        CHECK (size_bytes >= 0)
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

    def add(self, memory: Memory, events: Sequence[Event]) -> Memory:
        """Persist a new memory, its evidence links, and its events
        atomically, and return the memory that is canonically on record
        afterwards.

        (A17) That return value is NOT always `memory`: if this store
        already holds a CURRENT memory exactly equivalent to it (see
        `_find_current_equivalent` for the exact definition), nothing is
        written at all -- no row, no evidence link, no event, no search
        index entry -- and the pre-existing memory is returned instead,
        with its own original `memory_id` and `recorded_at`. A repeated
        `remember()` of the same canonical memory is a retry of one
        operation, not a second fact, so it must not produce a second
        current record (which is what collapsed semantic retrieval into
        false ambiguity in A16's Human Acceptance) and must not
        fabricate a history entry claiming something happened. This is
        exactly the idempotency `add_conflict` already applies to a
        repeated conflict declaration, applied to the canonical memory
        write path; callers tell the two cases apart by comparing the
        returned `memory_id` with the one they submitted.

        The duplicate lookup runs inside the SAME transaction as the
        insert, opened with `BEGIN IMMEDIATE` so the check and the write
        cannot be interleaved with another connection's identical write
        (Python's sqlite3 would otherwise not take the write lock until
        the first INSERT, leaving a check-then-insert race between two
        concurrent processes remembering the same thing).

        Raises `ValueError` if `memory.supersedes` names an unknown memory,
        if any `memory.evidence_ids` entry names unknown evidence, if
        `memory.supporting_evidence_ids` is not a subset of
        `memory.evidence_ids`, (A12.1.1) if `memory.epistemic_state`
        is `verified` without at least one supporting Evidence of a
        qualifying kind, or (A18.1) if this call is about to insert a NEW
        Memory row (i.e. `_find_current_equivalent` found no current
        duplicate) and `events` contains no `EVENT_KIND_MEMORY_RECORDED`
        event whose `subject_id` equals `memory.memory_id` -- see the
        MEDIUM-1 note further down. Raises `CortexStorageError` on genuine storage
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

        (A17.1) These write-boundary checks -- everything above except
        `has_superseder` below -- run BEFORE the duplicate lookup, not
        after: a request that would have been rejected on the baseline
        (e.g. `verified` with no supporting Evidence) must stay rejected
        even if the store happens to already hold a CURRENT memory that
        is grandfathered into that exact same non-compliant shape (a
        `verified` memory persisted before A12.1 introduced this rule).
        Historical validity of an old row is not the same thing as
        current write admissibility of a new request that merely
        resembles it -- the duplicate lookup may only ever suppress a
        WRITE, never a VALIDATION. `_evidence_exists`/subset/verified are
        all pure checks over `memory`'s own fields (plus, for the
        qualifying-kind check, the kind of evidence IT names) -- none of
        them depend on whether a duplicate exists, so moving them earlier
        changes nothing about what they accept. `has_superseder` is the
        one exception and stays below the lookup: unlike the others, its
        answer can be TRUE only because of the very memory this request
        is about to be recognized as a duplicate of (at most one memory
        may ever supersede a given target, enforced by this same check
        every time a superseding memory was inserted -- so if a CURRENT
        equivalent M with `supersedes=X` exists, M is necessarily the
        only memory that could have set `has_superseder(X)`). Checking it
        before the lookup would misreport an identical, idempotent retry
        of a legitimate supersession as "X already superseded" -- see
        `test_supersession_history_is_not_rewritten_by_a_later_duplicate`.
        `_memory_exists(supersedes)`, by contrast, cannot be affected by
        the duplicate this way (it is a fact about `supersedes` itself,
        not about who points at it), so it moves up with the rest.
        """
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            with self._connection:
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
                if memory.supersedes is not None:
                    if not self._memory_exists(memory.supersedes):
                        raise ValueError(f"Cannot supersede unknown memory {memory.supersedes!r}")

                existing = self._find_current_equivalent(memory)
                if existing is not None:
                    return existing

                # (A18.1 / MEDIUM-1) Reaching here means a NEW canonical
                # Memory row is about to be inserted -- an exact duplicate
                # retry already returned above without writing anything,
                # so this check cannot turn a retry into a false
                # transition. `timeline()`/`state()` derive "this memory
                # exists"/"is current" entirely from the Event log (via
                # `EVENT_KIND_MEMORY_RECORDED`), while `recall()`/
                # `count()`/`current_ids()` derive it from this raw table.
                # A row inserted without its own `memory_recorded` event
                # would make those two sources disagree about the same
                # memory: visible to one, invisible to the other.
                if not any(
                    event.kind == EVENT_KIND_MEMORY_RECORDED and event.subject_id == memory.memory_id
                    for event in events
                ):
                    raise ValueError(
                        "A newly recorded memory must be accompanied by its own "
                        f"{EVENT_KIND_MEMORY_RECORDED!r} event whose subject_id equals "
                        "memory.memory_id"
                    )

                if memory.supersedes is not None:
                    if self._has_superseder(memory.supersedes):
                        raise ValueError(
                            f"Memory {memory.supersedes!r} has already been superseded; "
                            "a memory can only be superseded once"
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
        return memory

    def _find_current_equivalent(self, memory: Memory) -> Memory | None:
        """(A17) Return the CURRENT memory this store already holds that is
        exactly equivalent to `memory`, or None.

        "Exactly equivalent" means every canonical field that carries the
        memory's MEANING is identical: `content` (byte-for-byte, under
        SQLite's default BINARY collation -- no case folding, no
        trimming, no Unicode normalization, none of which Cortex applies
        anywhere else either), `kind`, `epistemic_state`, `supersedes`,
        and both provenance tuples (`evidence_ids` and
        `supporting_evidence_ids`, order included). `memory_id` and
        `recorded_at` are deliberately excluded: they describe the
        RECORDING OPERATION, not the memory's meaning -- a freshly
        generated uuid4 and a fresh timestamp differ on every call by
        construction, so including either would make "duplicate" an
        impossible state and this method dead code.

        Only CURRENT memories qualify. Re-asserting something that was
        superseded is a new fact about now -- the earlier record stopped
        being what Cortex believes, so restating it is a real transition
        and gets its own memory, exactly as it would have before A17. It
        is only the coexistence of two CURRENT equivalents that this
        prevents. Note that "current" is evaluated per candidate rather
        than against a whole-store `current_ids()` set: the equivalence
        query already narrows the field to a handful of rows, and asking
        `_has_superseder` about those is far cheaper than materializing
        every id in the store on every single write.

        Candidates are ordered oldest-first (tie-broken by `memory_id`,
        the same determinism discipline `search`/`list_conflicts` apply)
        so that a store which somehow already contains several current
        equivalents -- legacy duplicates recorded before A17, which are
        preserved untouched, never merged or deleted -- always collapses
        onto the SAME, oldest one rather than an incidental row order.
        """
        rows = self._connection.execute(
            "SELECT memory_id, content, kind, epistemic_state, recorded_at, supersedes "
            "FROM memories "
            "WHERE content = ? AND kind = ? AND epistemic_state = ? AND supersedes IS ? "
            "ORDER BY recorded_at ASC, memory_id ASC",
            (memory.content, memory.kind, memory.epistemic_state, memory.supersedes),
        ).fetchall()
        for row in rows:
            candidate_id = row[0]
            if self._has_superseder(candidate_id):
                continue
            candidate = _row_to_memory(
                row,
                self._memory_evidence_ids(candidate_id),
                self._memory_supporting_evidence_ids(candidate_id),
            )
            if (
                candidate.evidence_ids == memory.evidence_ids
                and candidate.supporting_evidence_ids == memory.supporting_evidence_ids
            ):
                return candidate
        return None

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

    # -- sources ------------------------------------------------------------

    def observe_source(
        self,
        *,
        path: str,
        digest: str,
        size_bytes: int,
        observed_at: dt.datetime,
        candidate_source_id: str,
        candidate_evidence_id: str,
        evidence_content: str,
    ) -> tuple[str, Source, Evidence]:
        """Record one observation of a project file, and return
        `(status, source, evidence)` where `status` is `added`,
        `unchanged`, or `changed`.

        This is ONE transaction, opened with `BEGIN IMMEDIATE`, covering
        the whole decision: resolving the Source by path, reading its
        latest observation, comparing digests, and (when they differ)
        inserting the Source, the Evidence and the observation together.
        Splitting it into `add_evidence()` followed by a separate
        observation write would leave two windows open: a concurrent
        identical seed could interleave between the digest check and the
        insert and produce two observations of the same unchanged file,
        and a failure between the two writes would strand an Evidence row
        belonging to no observation. `BEGIN IMMEDIATE` takes the write
        lock before the lookup, exactly as `add()` does for the canonical
        memory write path (see its docstring), so the check and the write
        cannot be interleaved by another process.

        IDEMPOTENCY is judged against the LATEST observation only, never
        against the whole history: if the file's current digest equals the
        one Cortex last saw, this is a re-seed of an unchanged file and
        nothing at all is written (`unchanged`). If it differs, a new
        observation is appended even if that exact digest appeared earlier
        in the file's history -- a file edited to A, then B, then back to
        A has genuinely been through three states, and collapsing the
        third onto the first would claim the second never happened.
        `candidate_source_id`/`candidate_evidence_id`/`evidence_content`
        are proposals: they are used only if this call actually writes,
        and silently discarded on `unchanged`, the same way
        `add_conflict` discards the `recorded_at` of a repeat
        declaration.

        Raises `ValueError` if any argument is malformed at the write
        boundary (non-canonical ids, a digest that is not the expected
        hex, a negative size, an absolute or empty path) -- repeating
        next to the canonical data the guarantees `_source.py` already
        enforces in Python, so no future internal call path can persist a
        Source that violates them. Raises `CortexStorageError` on genuine
        corruption (a Source with no observations, an observation whose
        Evidence no longer exists) or I/O failure.
        """
        if not SOURCE_ID_PATTERN.fullmatch(candidate_source_id):
            raise ValueError(f"Malformed source_id {candidate_source_id!r}")
        if not EVIDENCE_ID_PATTERN.fullmatch(candidate_evidence_id):
            raise ValueError(f"Malformed evidence_id {candidate_evidence_id!r}")
        if not DIGEST_PATTERN.fullmatch(digest):
            raise ValueError(f"Malformed source digest {digest!r}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError(f"Source size_bytes must be a non-negative integer, got {size_bytes!r}")
        if not path or path.startswith("/"):
            raise ValueError(f"Source path must be a non-empty workspace-relative path, got {path!r}")

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            with self._connection:
                row = self._connection.execute(
                    "SELECT source_id FROM sources WHERE path = ?", (path,)
                ).fetchone()

                if row is None:
                    status = SEED_ADDED
                    source_id = candidate_source_id
                    self._connection.execute(
                        "INSERT INTO sources (source_id, path, first_observed_at) VALUES (?, ?, ?)",
                        (source_id, path, observed_at.isoformat()),
                    )
                else:
                    source_id = row[0]
                    latest = self._connection.execute(
                        "SELECT digest, evidence_id FROM source_observations "
                        "WHERE source_id = ? ORDER BY sequence DESC LIMIT 1",
                        (source_id,),
                    ).fetchone()
                    if latest is None:
                        # A Source is only ever created together with its
                        # first observation, so this shape cannot be
                        # produced by any write path here. Reaching it
                        # means the store was edited outside Cortex;
                        # appending an observation would paper over that.
                        raise CortexStorageError(
                            f"Corrupted source {source_id!r}: it has no observations"
                        )
                    if latest[0] == digest:
                        evidence = self._require_observation_evidence(latest[1])
                        return (SEED_UNCHANGED, self._load_source(source_id), evidence)
                    status = SEED_CHANGED

                self._connection.execute(
                    "INSERT INTO evidence (evidence_id, content, kind, recorded_at) VALUES (?, ?, ?, ?)",
                    (
                        candidate_evidence_id,
                        evidence_content,
                        EVIDENCE_KIND_DOCUMENT_OBSERVATION,
                        observed_at.isoformat(),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO source_observations "
                    "(source_id, evidence_id, digest, size_bytes, observed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source_id, candidate_evidence_id, digest, size_bytes, observed_at.isoformat()),
                )
                evidence = self._require_observation_evidence(candidate_evidence_id)
                return (status, self._load_source(source_id), evidence)
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to observe source {path!r}: {exc}") from exc

    def list_sources(self) -> list[Source]:
        """Every tracked Source with its full observation history, ordered
        by path.

        Fails closed rather than reporting a partial picture: an
        observation whose `source_id` names no Source is reported as
        corruption instead of being silently dropped, because a dropped
        observation is invisible history -- the one thing this table
        exists to preserve.
        """
        try:
            (orphans,) = self._connection.execute(
                "SELECT COUNT(*) FROM source_observations o "
                "LEFT JOIN sources s ON s.source_id = o.source_id "
                "WHERE s.source_id IS NULL"
            ).fetchone()
            if orphans:
                raise CortexStorageError(
                    f"Cortex source store holds {orphans} observation(s) referring to an "
                    "unknown source; refusing to report a partial source history"
                )
            rows = self._connection.execute("SELECT source_id FROM sources ORDER BY path").fetchall()
            return [self._load_source(row[0]) for row in rows]
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex source store: {exc}") from exc

    def _load_source(self, source_id: str) -> Source:
        row = self._connection.execute(
            "SELECT source_id, path, first_observed_at FROM sources WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            raise CortexStorageError(f"Unknown source {source_id!r} in Cortex source store")

        # LEFT JOIN rather than an inner join on purpose: an observation
        # whose Evidence has vanished must surface as corruption, not
        # quietly disappear from the history.
        observation_rows = self._connection.execute(
            "SELECT o.source_id, o.evidence_id, o.digest, o.size_bytes, o.observed_at, e.evidence_id "
            "FROM source_observations o "
            "LEFT JOIN evidence e ON e.evidence_id = o.evidence_id "
            "WHERE o.source_id = ? ORDER BY o.sequence",
            (source_id,),
        ).fetchall()
        observations = []
        for observation_row in observation_rows:
            if observation_row[5] is None:
                raise CortexStorageError(
                    f"Corrupted source {source_id!r}: observation references unknown "
                    f"evidence {observation_row[1]!r}"
                )
            observations.append(_row_to_source_observation(observation_row[:5]))
        if not observations:
            raise CortexStorageError(f"Corrupted source {source_id!r}: it has no observations")
        return _row_to_source(row, tuple(observations))

    def _require_observation_evidence(self, evidence_id: str) -> Evidence:
        evidence = self.get_evidence(evidence_id)
        if evidence is None:
            raise CortexStorageError(
                f"Corrupted source observation: evidence {evidence_id!r} does not exist"
            )
        if evidence.kind != EVIDENCE_KIND_DOCUMENT_OBSERVATION:
            raise CortexStorageError(
                f"Corrupted source observation: evidence {evidence_id!r} has kind "
                f"{evidence.kind!r}, expected {EVIDENCE_KIND_DOCUMENT_OBSERVATION!r}"
            )
        return evidence

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


def _create_v7_schema(connection: sqlite3.Connection) -> None:
    """Create every table at its CURRENT (v7) shape, for a brand-new
    store only. Not part of the v1->v7 migration chain below, which
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
    connection.execute(_CREATE_SOURCES_SQL)
    connection.execute(_CREATE_SOURCE_OBSERVATIONS_SQL)


# (A20.R.2) Every `_migrate_vN_to_vN+1` below runs INSIDE the caller's
# transaction and owns neither its boundary nor the version stamp -- see
# `_ensure_schema`'s migration loop, which is the single entity that
# decides eligibility, opens `BEGIN IMMEDIATE`, stamps `PRAGMA
# user_version`, and commits. A helper that opened its own transaction
# could not observe the version under the caller's lock, which is
# precisely what makes the decision to run it safe.


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Upgrade a v1 store to v2.

    Also backfills a `memory_recorded` event for every memory that already
    existed under v1, ordered by its own `recorded_at`, so that pre-A3
    memories remain visible to `timeline()`/`state()` (which are event-log
    projections, not raw table scans).

    Runs inside the caller's transaction (see the note above): the
    all-or-nothing property this step has always had is provided by that
    transaction, so a failure partway through -- e.g. a colliding table
    name on a genuinely malformed store -- still rolls back every earlier
    statement here rather than leaving them committed.
    """
    if not _table_exists(connection, "memories"):
        raise CortexStorageError(
            "Cortex memory store is stamped with schema version 1 but is missing the "
            "'memories' table; refusing to migrate a possibly corrupted store"
        )
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
    connection.execute(_CREATE_ATTEMPTS_SQL)
    connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)


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
    connection.execute(_CREATE_SKILLS_SQL)
    connection.execute(_CREATE_SKILL_STEPS_SQL)
    connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
    connection.execute(_CREATE_SKILL_EVIDENCE_SQL)


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
    connection.execute(
        f"ALTER TABLE memory_evidence ADD COLUMN role TEXT NOT NULL DEFAULT '{_ROLE_RELATED}'"
    )


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
    connection.execute(_CREATE_MEMORY_CONFLICTS_SQL)


def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
    """Upgrade a v6 store to v7 (A19.1) by adding the `sources` and
    `source_observations` tables.

    Nothing is backfilled, and no existing row is read or rewritten.
    Source is a wholly new concept in A19.1: no pre-v7 row ever recorded
    an observation of a tracked file. In particular this migration does
    NOT scan existing `file_reference` Evidence to invent Sources for it
    -- that Evidence carries its path (if any) only inside free text
    written by whoever recorded it, and guessing structure out of it
    would fabricate canonical identities from prose. A pre-v7 workspace
    therefore arrives at v7 with zero Sources, exactly like a brand-new
    one.
    """
    missing = [name for name in _V6_TABLES if not _table_exists(connection, name)]
    if missing:
        raise CortexStorageError(
            f"Cortex memory store is stamped with schema version 6 but is missing "
            f"table(s) {missing!r}; refusing to migrate a possibly corrupted store"
        )
    connection.execute(_CREATE_SOURCES_SQL)
    connection.execute(_CREATE_SOURCE_OBSERVATIONS_SQL)


# (A20.R.2) The canonical migration chain, as data: the version a step
# applies to, mapped to the function that performs it and the version the
# store is stamped with once it succeeds. `_ensure_schema`'s loop is the
# only reader; nothing else decides which step is eligible.
#
# FUTURE MIGRATIONS: a new v7->v8 step is added by writing a helper that
# contains ONLY its corruption guard and its DDL/data transformation --
# no `BEGIN`, no commit, no `PRAGMA user_version` -- registering it here,
# and raising `STORE_SCHEMA_VERSION`. A helper must never decide for
# itself whether it is eligible to run: that decision belongs to the
# serialized boundary below, and taking it from an unlocked (therefore
# possibly stale) version is exactly the defect A20.R.2 repaired.
_MIGRATIONS: dict[int, tuple[Callable[[sqlite3.Connection], None], int]] = {
    _SCHEMA_VERSION_V1: (_migrate_v1_to_v2, _SCHEMA_VERSION_V2),
    _SCHEMA_VERSION_V2: (_migrate_v2_to_v3, _SCHEMA_VERSION_V3),
    _SCHEMA_VERSION_V3: (_migrate_v3_to_v4, _SCHEMA_VERSION_V4),
    _SCHEMA_VERSION_V4: (_migrate_v4_to_v5, _SCHEMA_VERSION_V5),
    _SCHEMA_VERSION_V5: (_migrate_v5_to_v6, _SCHEMA_VERSION_V6),
    _SCHEMA_VERSION_V6: (_migrate_v6_to_v7, _SCHEMA_VERSION_V7),
}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    (version,) = connection.execute("PRAGMA user_version").fetchone()

    if version == 0:
        # (A18.1 / PD-1) `BEGIN IMMEDIATE` takes SQLite's write lock
        # up front, before re-checking anything, so two processes
        # opening a not-yet-created `memory.db` at the same time cannot
        # both observe version 0 outside a lock and then both attempt
        # `_create_v6_schema` -- the same check-then-create race
        # `MemoryStore.add()` already closes for the canonical write
        # path (see its docstring), applied here to the very first
        # schema creation. The loser of the lock blocks until the
        # winner commits, then re-reads `PRAGMA user_version` UNDER the
        # lock: if the winner already created and stamped the schema,
        # the loser sees the real version and simply falls through to
        # the migration loop below -- which finds the store already at
        # `STORE_SCHEMA_VERSION` and does nothing -- instead of
        # re-running `_create_v7_schema` against tables that already
        # exist. (A20.R.2) That loop applies the same lock-then-re-read
        # invariant to the migration chain; this branch and that loop
        # are the same protocol at two points of the same open path.
        connection.execute("BEGIN IMMEDIATE")
        with connection:
            (version,) = connection.execute("PRAGMA user_version").fetchone()
            if version == 0:
                if any(_table_exists(connection, name) for name in _V7_TABLES):
                    raise CortexStorageError(
                        "Cortex memory store has no recognized schema version but already "
                        "contains data tables; refusing to open a possibly corrupted store"
                    )
                _create_v7_schema(connection)
                connection.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
                version = STORE_SCHEMA_VERSION

    # (A20.R.2) The canonical migration chain, one serialized step at a
    # time. THE INVARIANT: a process may decide which canonical migration
    # to execute only after taking SQLite's write lock and re-reading
    # `PRAGMA user_version` UNDER that lock.
    #
    # Before this loop existed, the version read above -- taken with no
    # lock held -- decided the whole chain, and each step then ran under
    # a plain `BEGIN` (DEFERRED, so it takes no lock until its first
    # write). Two processes opening the same v4 store both observed 4
    # outside any lock and both went on to run `_migrate_v4_to_v5`; the
    # loser took the lock only when it reached its `ALTER TABLE`, by
    # which point the winner had already committed, and it failed with
    # "duplicate column name: role" -- surfaced to the caller as a
    # corrupt store, failing an open that had nothing wrong with it.
    # Every step of the chain had this shape, and every one of them was
    # reproducibly hit. The lock was never what was missing: what was
    # missing is re-deciding once it is held.
    #
    # Each iteration therefore takes the lock FIRST, re-reads the real
    # version, and only then picks the step matching THAT version --
    # executing it and stamping the new version inside the SAME
    # transaction, so a step and the version that records it can never
    # disagree, not even across a crash (see below). A process whose
    # step was already performed by someone else simply sees the higher
    # version and moves on to the next one; when the store has reached
    # `STORE_SCHEMA_VERSION` there is nothing left to pick and the loop
    # ends.
    #
    # Note what this deliberately does NOT do: it never suppresses a
    # colliding DDL (no `IF NOT EXISTS`, no catching "duplicate
    # column"/"already exists"). Concurrency safety comes entirely from
    # the lock plus the re-read. That distinction is what keeps a store
    # stamped v4 that ALREADY has an incompatible `role` column failing
    # closed as it always has: under the lock its version is still 4, so
    # the step is genuinely eligible, runs, and collides -- a real
    # corrupt store, correctly rejected. Suppressing the error would
    # have closed the race by opening that store silently.
    #
    # One transaction per step, not one for the whole chain: it keeps
    # the write lock short (v1's event backfill scales with the store),
    # preserves the property that a completed step stays completed, and
    # leaves the canonical boundary fully committed before
    # `_ensure_search_index` opens its own `BEGIN IMMEDIATE` below --
    # these are sibling critical sections, never nested ones.
    #
    # The unlocked read above is kept purely as a fast path: a healthy
    # store already at `STORE_SCHEMA_VERSION` (the overwhelmingly common
    # case) skips the loop entirely and never takes the write lock. It
    # is an optimization, not what makes this correct -- `user_version`
    # only ever moves forward, so an unlocked read can be behind the
    # truth but never ahead of it, and being behind only means entering
    # a loop that re-decides everything under the lock anyway.
    while version != STORE_SCHEMA_VERSION and version in _MIGRATIONS:
        connection.execute("BEGIN IMMEDIATE")
        with connection:
            (version,) = connection.execute("PRAGMA user_version").fetchone()
            step = _MIGRATIONS.get(version)
            if step is None:
                # Either another process finished the whole chain while
                # this one waited for the lock (version is now
                # `STORE_SCHEMA_VERSION`), or the store is stamped with
                # a version this build knows no step for -- which the
                # validation below reports. Leaving the `with` block
                # commits this read-only transaction, releasing the
                # write lock rather than holding it for the rest of this
                # connection's life.
                break
            migrate, next_version = step
            migrate(connection)
            connection.execute(f"PRAGMA user_version = {next_version}")
            version = next_version

    if version != STORE_SCHEMA_VERSION:
        raise CortexStorageError(
            f"Cortex memory store schema version {version} is not supported by this "
            f"version of Cortex (expected {STORE_SCHEMA_VERSION})"
        )

    missing = [name for name in _V7_TABLES if not _table_exists(connection, name)]
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

    (A20.R) That idempotency has to hold ACROSS PROCESSES, not just
    across successive opens, so the existence check that decides whether
    to create the index runs a second time INSIDE the transaction, under
    SQLite's write lock. This is the same protocol `_ensure_schema`
    applies to canonical first creation (see its `version == 0` branch),
    applied to the derived index -- and for the same reason: the check
    below at function entry is unlocked, so two processes opening a store
    whose index does not exist yet can both pass it, and a plain `BEGIN`
    would not serialize them (Python's sqlite3 opens a DEFERRED
    transaction, which takes no lock until its first write). Both would
    then reach `CREATE VIRTUAL TABLE` and the loser would fail with
    "table search_index already exists" -- reported as a corrupt store,
    failing an open that has nothing wrong with it. `BEGIN IMMEDIATE`
    takes the write lock up front, so exactly one process can observe
    the index as absent and act on it; every other one blocks, re-reads
    the real state under the lock, sees the winner's index, and returns
    without a second CREATE and without a redundant initial rebuild.
    The unlocked check at entry is kept purely as a fast path for the
    overwhelmingly common case (index already present): it is now an
    optimization, not what makes this correct.

    The two critical sections stay deliberately separate: the canonical
    version chain and this derived projection are different boundaries
    with different failure semantics (see module docstring), and merging
    them would put a rebuildable search aid inside the transaction that
    defines what Cortex canonically holds.

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
    connection.execute("BEGIN IMMEDIATE")
    with connection:
        if _table_exists(connection, SEARCH_INDEX_TABLE):
            # Another process created it between the unlocked check
            # above and this lock. `with connection` commits this
            # (read-only) transaction on the way out, releasing the
            # write lock rather than leaving it open for the rest of
            # this connection's life.
            return
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


def _row_to_source(row: tuple[str, str, str], observations: tuple[SourceObservation, ...]) -> Source:
    source_id, path, first_observed_at_raw = row

    if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise CortexStorageError(f"Corrupted source_id {source_id!r} in Cortex source store")
    if not isinstance(path, str) or not path or path.startswith("/"):
        # A persisted absolute path would silently break the portability
        # guarantee (see `_CREATE_SOURCES_SQL`): refuse to hand it back as
        # if it were a valid workspace-relative Source.
        raise CortexStorageError(f"Corrupted path {path!r} for source {source_id!r}")
    try:
        first_observed_at = dt.datetime.fromisoformat(first_observed_at_raw)
    except (TypeError, ValueError) as exc:
        raise CortexStorageError(
            f"Corrupted first_observed_at value {first_observed_at_raw!r} for source {source_id!r}"
        ) from exc
    return Source(
        source_id=source_id,
        path=path,
        first_observed_at=first_observed_at,
        observations=observations,
    )


def _row_to_source_observation(row: tuple[str, str, str, int, str]) -> SourceObservation:
    source_id, evidence_id, digest, size_bytes, observed_at_raw = row

    if not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise CortexStorageError(f"Corrupted source_id {source_id!r} in Cortex source store")
    if not isinstance(evidence_id, str) or not EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
        raise CortexStorageError(
            f"Corrupted evidence_id {evidence_id!r} for an observation of source {source_id!r}"
        )
    if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
        raise CortexStorageError(
            f"Corrupted digest {digest!r} for an observation of source {source_id!r}"
        )
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise CortexStorageError(
            f"Corrupted size_bytes {size_bytes!r} for an observation of source {source_id!r}"
        )
    try:
        observed_at = dt.datetime.fromisoformat(observed_at_raw)
    except (TypeError, ValueError) as exc:
        raise CortexStorageError(
            f"Corrupted observed_at value {observed_at_raw!r} for an observation of "
            f"source {source_id!r}"
        ) from exc
    return SourceObservation(
        source_id=source_id,
        evidence_id=evidence_id,
        digest=digest,
        size_bytes=size_bytes,
        observed_at=observed_at,
    )


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
