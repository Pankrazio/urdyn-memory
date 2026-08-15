"""Persistence for the derived semantic index: entity vectors and the
model metadata they were generated with.

This is deliberately a SEPARATE SQLite file from `memory.db`
(`semantic_index.db`, next to it under `.cortex/`), not a table inside
the canonical store managed by `_store.py`. The semantic index is
DERIVED, OPTIONAL, REBUILDABLE, and REPLACEABLE (see `_semantic.py`'s
module docstring and the A7.4 report): it holds no canonical truth,
nothing here is ever required for a canonical write to succeed, and
deleting this entire file is always safe -- Cortex degrades to
lexical/FTS-only, exactly as if the semantic extra were never installed.
Keeping it in its own file (rather than a table in `memory.db`) makes
that guarantee trivial to reason about and trivial to test: "the
semantic index is missing" is just "this file does not exist", nothing
about the canonical schema version chain in `_store.py` is touched.

This module holds no dependency on `model2vec` or `numpy`: vectors are
opaque `bytes` blobs here, encoded/decoded by `_semantic.py`. That keeps
`_semantic_store.py` importable without the `[semantic]` extra installed
-- it is not imported at all from the normal `import cortex_memory` path
today, but nothing about its own implementation would force that if a
future caller needed to inspect index metadata without loading a model.

Status lifecycle: `begin_rebuild()` clears all vectors and writes a
`semantic_meta` row with `status='building'`, committed immediately.
Embedding the corpus happens OUTSIDE any database transaction (it calls
out to a model, which is slow and not itself transactional). Only after
every vector has been written does `finish_rebuild()` flip the status to
`'ready'` and commit. If the process is interrupted at any point between
`begin_rebuild()` and `finish_rebuild()` (killed, crashes, loses power),
the on-disk state is a `semantic_meta` row stuck at `status='building'`
-- `is_ready()` returns False for that, so the index is correctly
treated as incomplete/unavailable rather than silently used half-built.

[A27] FRESHNESS. That status answers "did the last full rebuild
finish", which is a question about a PROCESS. It says nothing about
whether the index still represents the canonical store, and A26 measured
what that gap costs: a workspace whose semantic index was simply absent
produced a plausible, incomplete `preflight()` with no signal that the
semantic channel had been skipped. The same silent shape applies to an
index that merely predates the memory a caller is asking about.

The freshness answer deliberately introduces NO new stored state -- no
dirty flag, no generation counter, no watermark column, and therefore no
schema change. It is COMPUTED, by comparing the ids canonical storage
currently holds for the three indexed pools against `indexed_ids()`
below. That comparison is authoritative because the canonical texts
feeding this index are append-only and immutable once written (no
`UPDATE`/`DELETE` reaches `memories`/`attempts`/`skills`; superseding
INSERTS a new memory rather than editing the old one), so "this id has a
vector" implies "that vector is the right one for it". A remembered flag
could not offer the same guarantee: canonical storage and this file are
separate SQLite databases with no atomic cross-store commit, so a crash
between a canonical commit and its dirty mark would recreate exactly the
silent staleness this is meant to remove. `SemanticState` (below) is
therefore always recomputed, never read back.

The invariant this rests on is stated here so it is not rediscovered by
accident: IF A SEMANTICALLY-INDEXED CANONICAL TEXT EVER BECOMES MUTABLE,
id-set coverage stops being sufficient and this mechanism must change in
the same commit.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

from ._errors import CortexStorageError

SEMANTIC_DB_FILENAME = "semantic_index.db"

STATUS_BUILDING = "building"
STATUS_READY = "ready"

# [A27] The four externally meaningful lifecycle states. Four, and not
# more, because each one has a DIFFERENT REMEDY: enable it, nothing,
# nothing (it repairs itself), or rebuild it. A state that would not
# change what the user or a future consumer does is decoration, not
# information, so `building`/`incompatible`/`model missing`/`corrupt` are
# all reported as UNAVAILABLE carrying a `detail`, rather than as four
# separate states with one shared remedy.
SEMANTIC_DISABLED = "disabled"
SEMANTIC_READY = "ready"
SEMANTIC_STALE = "stale"
SEMANTIC_UNAVAILABLE = "unavailable"

# Reasons attached to a state. Constants rather than inline literals so
# the CLI, the API and the tests all name the same condition, and plain
# ASCII because Cortex targets Windows consoles as a first-class case
# (a cp1252 terminal must never be the reason a state line fails to
# print).
DETAIL_NOT_SET_UP = "not set up"
DETAIL_EXTRA_MISSING = "the semantic extra is not installed"
DETAIL_BUILD_INCOMPLETE = "the last index build did not finish"
DETAIL_MODEL_MISMATCH = "the index was built with a different model"
DETAIL_MODEL_UNCACHED = "the model files are not in the local cache"
DETAIL_INDEX_UNREADABLE = "the index is unreadable"
DETAIL_REFRESH_FAILED = "the automatic refresh could not run"

_SETUP_HINT = "run: cortex semantic setup"
# One state needs a different remedy, and saying otherwise would send the
# user into a command that fails: `semantic_setup()` opens the existing
# index file before rebuilding it, so a file that is not a database at
# all raises instead of being repaired. Cortex will not delete it on the
# user's behalf -- derived and rebuildable is a reason it is SAFE to
# delete, not a licence for Cortex to do it unasked -- so the state says
# exactly what to do instead.
_REMEDY_BY_DETAIL = {
    DETAIL_INDEX_UNREADABLE: f"delete .cortex/{SEMANTIC_DB_FILENAME}, then {_SETUP_HINT}",
}

_CREATE_META_SQL = """
    CREATE TABLE semantic_meta (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        provider TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_revision TEXT,
        dimensions INTEGER NOT NULL,
        normalization TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL
    )
"""

_CREATE_VECTORS_SQL = """
    CREATE TABLE semantic_vectors (
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        vector BLOB NOT NULL,
        PRIMARY KEY (entity_type, entity_id)
    )
"""


def semantic_db_path_for(cortex_dir: Path) -> Path:
    return cortex_dir / SEMANTIC_DB_FILENAME


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticMeta:
    provider: str
    model_id: str
    model_revision: str | None
    dimensions: int
    normalization: str
    created_at: str
    status: str

    def matches(self, *, provider: str, model_id: str, normalization: str) -> bool:
        """Whether an already-built index was generated by the same
        model configuration a caller is about to query with. Checked
        BEFORE the model is loaded (provider/model_id/normalization are
        static constants in `_semantic.py`), so a mismatch is caught
        without paying to load a model that will not be used anyway.
        `dimensions` is deliberately not part of this comparison -- it is
        used later, purely to safely decode stored vector blobs
        (`_semantic.blob_to_vector`); for a fixed `model_id` and
        `normalization` it is expected to be constant, so it adds no
        further discriminating signal here.

        Revision is also intentionally NOT part of this comparison: it
        is recorded for diagnostics/auditing
        (`_semantic.py.resolve_local_revision`), but is often
        unresolvable (multiple cached revisions, no `main` ref) and must
        not itself force an otherwise-matching index to be treated as
        stale."""
        return provider == self.provider and model_id == self.model_id and normalization == self.normalization


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticState:
    """[A27] What the derived semantic index is, right now, relative to
    canonical state -- recomputed on every call (see the module
    docstring), never read back from storage.

    Carried on `Preflight`/`GuardResult` as well as reported by
    `cortex status`, so a consumer can always tell an ABSENT retrieval
    substrate from an ABSTAINING one. That distinction is the whole point:
    A26 produced an incomplete result that was indistinguishable from a
    complete one because both looked like "the semantic channel admitted
    nothing".

    `missing` is how many canonical records the index does not cover;
    `indexed` how many vectors it holds; `refreshed` how many vectors the
    call that produced this state just added (0 for a purely observational
    state, e.g. `cortex status`, which never refreshes anything).
    """

    status: str
    detail: str | None = None
    missing: int = 0
    indexed: int = 0
    refreshed: int = 0

    def is_usable(self) -> bool:
        """Whether the semantic channel can contribute candidates at all.
        A STALE index is usable: the vectors it does hold are valid for
        the records they cover (append-only canonical texts -- see the
        module docstring), so querying it strictly widens recall over
        lexical alone. It is reported as degraded, not withheld."""
        return self.status in (SEMANTIC_READY, SEMANTIC_STALE)

    def _remedy(self) -> str:
        return _REMEDY_BY_DETAIL.get(self.detail or "", _SETUP_HINT)

    def describe(self) -> str:
        """One line for `cortex status`. Reports observed state only --
        computing it never loads a model, never touches the network and
        never mutates anything."""
        if self.status == SEMANTIC_READY:
            return f"ready ({self.indexed} indexed)"
        if self.status == SEMANTIC_STALE:
            total = self.indexed + self.missing
            return f"stale ({self.missing} of {total} not indexed; refreshed automatically on next preflight)"
        if self.status == SEMANTIC_DISABLED:
            return f"disabled ({self.detail}; {self._remedy()})"
        return f"unavailable ({self.detail}; {self._remedy()})"

    def retrieval_mode(self) -> str:
        """One line for `preflight`/`guard`, printed in EVERY state --
        including a healthy one, and including an empty result.

        Printing it only when degraded would put the healthy case back
        into information-by-omission, which is the exact failure A26
        walked into: `No relevant experience found.` on its own cannot
        tell a reader whether semantic retrieval ran and found nothing, or
        never ran at all."""
        if self.status == SEMANTIC_DISABLED:
            return f"lexical only -- semantic retrieval is {self.detail} ({self._remedy()})"
        if self.status == SEMANTIC_UNAVAILABLE:
            return f"lexical only -- semantic retrieval is unavailable: {self.detail} ({self._remedy()})"
        if self.status == SEMANTIC_STALE:
            return (
                f"semantic + lexical -- DEGRADED: {self.missing} record(s) not indexed "
                f"({self.detail or DETAIL_REFRESH_FAILED}; {self._remedy()})"
            )
        if self.refreshed:
            return f"semantic + lexical (refreshed {self.refreshed})"
        return "semantic + lexical"


class SemanticIndexStore:
    """Boundary around the derived semantic index for a single workspace."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def create_or_open(cls, db_path: Path) -> "SemanticIndexStore":
        connection = sqlite3.connect(db_path)
        try:
            _ensure_schema(connection)
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise CortexStorageError(f"Cortex semantic index at {db_path} is corrupted: {exc}") from exc
        except Exception:
            connection.close()
            raise
        return cls(connection)

    @classmethod
    def open_if_exists(cls, db_path: Path) -> "SemanticIndexStore | None":
        if not db_path.exists():
            return None
        return cls.create_or_open(db_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SemanticIndexStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def meta(self) -> SemanticMeta | None:
        try:
            row = self._connection.execute(
                "SELECT provider, model_id, model_revision, dimensions, normalization, created_at, status "
                "FROM semantic_meta WHERE id = 1"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex semantic index metadata: {exc}") from exc
        if row is None:
            return None
        return SemanticMeta(*row)

    def is_ready(self) -> bool:
        meta = self.meta()
        return meta is not None and meta.status == STATUS_READY

    def begin_rebuild(
        self,
        *,
        provider: str,
        model_id: str,
        model_revision: str | None,
        dimensions: int,
        normalization: str,
        created_at: str,
    ) -> None:
        """Clear the index and mark it `building`, committed immediately
        so an interruption after this point leaves `is_ready()` False."""
        try:
            with self._connection:
                self._connection.execute("DELETE FROM semantic_vectors")
                self._connection.execute("DELETE FROM semantic_meta WHERE id = 1")
                self._connection.execute(
                    "INSERT INTO semantic_meta "
                    "(id, provider, model_id, model_revision, dimensions, normalization, created_at, status) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                    (provider, model_id, model_revision, dimensions, normalization, created_at, STATUS_BUILDING),
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to begin Cortex semantic index rebuild: {exc}") from exc

    def add_vectors(self, entity_type: str, rows: list[tuple[str, bytes]]) -> None:
        """Persist a batch of `(entity_id, vector_blob)` pairs for
        `entity_type`.

        Called from two places with two different surrounding contracts:
        by a full rebuild, between `begin_rebuild()` and
        `finish_rebuild()`; and (A27) by an incremental refresh of an
        index already at `status='ready'`, which deliberately does NOT
        bracket itself in those calls -- `begin_rebuild()` would erase
        every existing vector, and flipping the status would hand a
        second authority the right to declare readiness. The A7.4
        publication rule stays exactly as it was, owned by the rebuild
        path alone.

        `INSERT OR REPLACE` rather than `INSERT`: an incremental refresh
        derives its work list from a read taken slightly earlier, so two
        concurrent refreshes can legitimately compute overlapping lists
        and race to write the same `(entity_type, entity_id)`. Both are
        writing the SAME vector -- same model, same immutable canonical
        text -- so the loser of that race must be a no-op, not a
        `UNIQUE constraint failed` that aborts a `preflight()`. This is
        what makes duplicate refresh harmless instead of merely unlikely.
        """
        if not rows:
            return
        try:
            with self._connection:
                self._connection.executemany(
                    "INSERT OR REPLACE INTO semantic_vectors (entity_type, entity_id, vector) VALUES (?, ?, ?)",
                    [(entity_type, entity_id, blob) for entity_id, blob in rows],
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to persist Cortex semantic vectors: {exc}") from exc

    def finish_rebuild(self) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "UPDATE semantic_meta SET status = ? WHERE id = 1", (STATUS_READY,)
                )
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to finalize Cortex semantic index rebuild: {exc}") from exc

    def all_vectors(self, entity_type: str) -> list[tuple[str, bytes]]:
        try:
            rows = self._connection.execute(
                "SELECT entity_id, vector FROM semantic_vectors WHERE entity_type = ? ORDER BY entity_id",
                (entity_type,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex semantic vectors: {exc}") from exc
        return [(row[0], row[1]) for row in rows]

    def indexed_ids(self, entity_type: str) -> set[str]:
        """[A27] Which entity ids of `entity_type` this index covers.

        The whole freshness mechanism reads through here: ids only, no
        vector blobs, so asking "is this index current" costs a single
        index scan and never decodes, ranks or embeds anything. It is by
        construction cheaper than `all_vectors()`, which every retrieval
        call already pays for the same pool."""
        try:
            rows = self._connection.execute(
                "SELECT entity_id FROM semantic_vectors WHERE entity_type = ?", (entity_type,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex semantic index: {exc}") from exc
        return {row[0] for row in rows}

    def vector_count(self) -> int:
        try:
            (count,) = self._connection.execute("SELECT COUNT(*) FROM semantic_vectors").fetchone()
        except sqlite3.DatabaseError as exc:
            raise CortexStorageError(f"Failed to read Cortex semantic index: {exc}") from exc
        return count


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    with connection:
        if not _table_exists(connection, "semantic_meta"):
            connection.execute(_CREATE_META_SQL)
        if not _table_exists(connection, "semantic_vectors"):
            connection.execute(_CREATE_VECTORS_SQL)
