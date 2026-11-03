"""Backward compatibility with A5-A11 (schema v4) stores, and the A12.1
support relation.

Schema v4 only existed in Cortex development builds; implicit
migration (and its `role` backfill) is retired. Migration expectations run on
the AUTHENTIC v4 Brain written by the historical code (refused, intact). The
A12.1 reader invariants -- a `verified` memory without explicit support is
never demoted or rewritten by reading -- stay, on an explicitly built v7 row
(see the section comment below). The hand-written v4 builder was removed: on
the v7 store `Urdyn.init` creates it never produced a v4 store.
"""

import datetime as dt
import sqlite3
import uuid

import pytest

import legacy_brains as legacy
from urdyn import Urdyn, UrdynStorageError, UrdynStorageSchemaError
from urdyn import maintenance

def _v4(tmp_path):
    root, generated = legacy.cortex_brain(tmp_path, 4)
    return root, generated, legacy.db_path(root)


def test_v4_brain_is_refused_and_its_memories_survive_exactly(tmp_path):
    """Was `test_v4_database_opens_and_memory_survives`."""
    root, generated, db = _v4(tmp_path)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")
    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 4 and facts["integrity"] == "ok"
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


def test_v4_refusal_adds_no_role_column(tmp_path):
    """Was `test_v4_migration_reaches_v5_and_adds_role_column`."""
    root, _, db = _v4(tmp_path)
    assert "role" not in legacy.store_facts(db)["columns"]["memory_evidence"]
    for entry in (lambda: Urdyn.open(root).recall("synthetic"), lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.recover_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 4
    assert "role" not in facts["columns"]["memory_evidence"]


def test_v4_refusal_backfills_no_role(tmp_path):
    """Was `test_legacy_memory_evidence_rows_are_backfilled_to_related_not_supporting`
    (the v4->v5 backfill wrote `related`, never `supporting`). Retired with the
    migration: the backfill runs zero times -- the authentic v4 links stay
    exactly as written, with no role at all."""
    root, _, db = _v4(tmp_path)
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        links_before = sorted(connection.execute("SELECT * FROM memory_evidence").fetchall())
    finally:
        connection.close()
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).state()
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert sorted(connection.execute("SELECT * FROM memory_evidence").fetchall()) == links_before
    finally:
        connection.close()


def test_v4_refusal_is_safe_to_repeat(tmp_path):
    """Was `test_v4_migration_is_safe_to_repeat`."""
    root, _, _ = _v4(tmp_path)
    before = legacy.snapshot(root)
    messages = set()
    for _ in range(3):
        with pytest.raises(UrdynStorageSchemaError) as refused:
            Urdyn.open(root).recall("synthetic")
        messages.add(str(refused.value))
    assert len(messages) == 1
    assert legacy.snapshot(root) == before


# ---------------------------------------------------------------------------
# A12.1 reader invariant on the CURRENT schema. A `verified` memory whose
# only evidence link is `related` (no explicit support) was produced only by the
# retired v4->v5 migration; no public API of this build or of any release writes
# one (verification requires explicit supporting evidence). What remains valid
# is the reader's side: reading never demotes, rewrites or "repairs" such a row.
# The row is therefore built EXPLICITLY on a v7 store -- declared as such, with
# its `role` written out -- instead of through a fake v4 builder (which, on the
# v7 store `Urdyn.init` creates, never produced a v4 store at all).
# ---------------------------------------------------------------------------


def _v7_verified_memory_with_related_only_evidence(cx, *, content, evidence_kind="test_result"):
    memory_id, evidence_id = uuid.uuid4().hex, uuid.uuid4().hex
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    connection = sqlite3.connect(cx._db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
        assert "role" in [r[1] for r in connection.execute("PRAGMA table_info(memory_evidence)")]
        with connection:
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, 'lesson', 'verified', ?, NULL)", (memory_id, content, recorded_at))
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, 'memory_recorded', ?, ?)",
                (uuid.uuid4().hex, memory_id, recorded_at))
            connection.execute(
                "INSERT INTO evidence (evidence_id, content, kind, recorded_at) VALUES (?, ?, ?, ?)",
                (evidence_id, "qualifying evidence linked without explicit support", evidence_kind, recorded_at))
            connection.execute(
                "INSERT INTO memory_evidence (memory_id, evidence_id, position, role) VALUES (?, ?, 0, 'related')",
                (memory_id, evidence_id))
    finally:
        connection.close()
    return memory_id, evidence_id


def _row(cx, memory_id):
    connection = sqlite3.connect(f"file:{cx._db_path}?mode=ro", uri=True)
    try:
        return (connection.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone(),
                connection.execute("SELECT * FROM memory_evidence WHERE memory_id = ?", (memory_id,)).fetchall())
    finally:
        connection.close()


def test_a_verified_memory_with_only_related_evidence_stays_verified_on_read(tmp_path):
    """Was `test_legacy_verified_memory_stays_verified_with_empty_supporting_ids`."""
    cx = Urdyn.init(tmp_path, "dev")
    memory_id, evidence_id = _v7_verified_memory_with_related_only_evidence(cx, content="Migration rollback is atomic.")
    stored = _row(cx, memory_id)

    (lesson,) = [m for m in cx.state(kind="lesson") if m.memory_id == memory_id]

    assert lesson.epistemic_state == "verified"
    assert lesson.evidence_ids == (evidence_id,)
    assert lesson.supporting_evidence_ids == ()
    assert _row(cx, memory_id) == stored  # read wrote nothing


def test_a_verified_memory_with_only_related_evidence_still_surfaces_in_preflight(tmp_path):
    """Was `test_legacy_verified_lesson_still_surfaces_in_preflight_after_migration`:
    `verified_lessons` filters on `epistemic_state` only, never on explicit support."""
    cx = Urdyn.init(tmp_path, "dev")
    memory_id, _ = _v7_verified_memory_with_related_only_evidence(
        cx, content="Migration rollback is atomic under process crashes.")

    result = cx.preflight("Migration rollback is atomic under process crashes.")

    (surfaced,) = [m for m in result.verified_lessons if m.memory_id == memory_id]
    assert surfaced.supporting_evidence_ids == ()


def test_a_verified_memory_with_only_related_evidence_coexists_with_a_new_supported_lesson(tmp_path):
    """Was `test_v4_migrated_lesson_preserved_after_new_supporting_lesson_recorded`."""
    cx = Urdyn.init(tmp_path, "dev")
    old_id, _ = _v7_verified_memory_with_related_only_evidence(cx, content="Earlier verified claim.")
    stored = _row(cx, old_id)

    validation = cx.add_evidence("Fresh check passed.", kind="test_result")
    fresh = cx.learn("Fresh verified claim.", supporting_evidence=[validation], verified=True)

    lessons = {m.memory_id: m for m in cx.state(kind="lesson")}
    assert lessons[old_id].epistemic_state == "verified"
    assert lessons[old_id].supporting_evidence_ids == ()
    assert lessons[fresh.memory_id].supporting_evidence_ids == (validation.evidence_id,)
    assert _row(cx, old_id) == stored


def test_v4_sabotaged_store_fails_closed_untouched(tmp_path):
    """Was `test_v4_migration_does_not_destroy_data_on_failure`: a
    pre-added `role` column made the v4->v5 `ALTER TABLE` fail and roll back
    (still v4, row intact). On the AUTHENTIC v4 Brain the store is refused
    before any step: still v4, one `role` column (the sabotage, not a second
    one), memories intact, every byte kept."""
    root, generated = legacy.cortex_brain(tmp_path, 4)
    db = legacy.db_path(root)
    legacy.sabotage(db, "ALTER TABLE memory_evidence ADD COLUMN role TEXT")
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageError):
        Urdyn.open(root).recall("must survive")

    assert legacy.snapshot(root) == before
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 4
    assert facts["columns"]["memory_evidence"].count("role") == 1
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


# ---------------------------------------------------------------------------
# A12.1.1: v5 canonical-data corruption -- no silent reinterpretation
# ---------------------------------------------------------------------------


def test_invalid_role_value_raises_cleanly_instead_of_silently_dropping_support(tmp_path):
    """A `memory_evidence.role` value outside `{'related', 'supporting'}`
    (never written by this codebase, but not impossible for hand-edited
    or externally-touched data) must not be silently reinterpreted as
    "not supporting" -- that would drop a real support designation with
    no error at all. It must raise `UrdynStorageError`, the same way
    `_row_to_memory` already does for a corrupted `kind`/`epistemic_state`."""
    cx = Urdyn.init(tmp_path, "dev")
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("a lesson", supporting_evidence=[validation], verified=True)

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute(
            "UPDATE memory_evidence SET role = 'unknown_role' WHERE memory_id = ?",
            (lesson.memory_id,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.state(kind="lesson")


def test_user_version_5_with_missing_role_column_raises_cleanly(tmp_path):
    """`PRAGMA user_version == 5` alone must not be
    trusted as proof the schema is actually shaped like v5: if
    `memory_evidence` is missing its `role` column (a corrupted or
    incomplete upgrade), every read/write path that touches
    `role` fails loudly as `UrdynStorageError` -- the same standard of
    schema integrity already applied to every other version (table-level
    checks in `_ensure_schema`, `sqlite3.DatabaseError` wrapped
    consistently everywhere else in this module). Nothing recreates the
    table automatically and no data is lost.

    Under the storage contract (`docs/storage-contract.md`: `open`
    requires a valid canonical DB, never creates or migrates) the refusal
    comes earlier, at `Urdyn.open` itself, as `UrdynStorageCorruptError`;
    a handle opened before the damage refuses its reads the same way. The
    damaged table is not rebuilt and the bytes are unchanged."""
    from urdyn import UrdynStorageCorruptError

    cx = Urdyn.init(tmp_path, "dev")
    original = cx.remember("seed content that must survive", kind="note")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("DROP TABLE memory_evidence")
        connection.execute(
            "CREATE TABLE memory_evidence (memory_id TEXT NOT NULL, evidence_id TEXT NOT NULL, "
            "position INTEGER NOT NULL, PRIMARY KEY (memory_id, evidence_id))"
        )
        connection.commit()
        (version,) = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()
    # A19.1 bumped STORE_SCHEMA_VERSION to 7, so `Urdyn.init` above
    # already created this store directly at v7; the assertion tracks
    # that literal, not the v5 this test predates.
    assert version == 7  # sabotage did not touch the version stamp
    damaged = cx._db_path.read_bytes()

    with pytest.raises(UrdynStorageCorruptError):
        Urdyn.open(tmp_path)
    with pytest.raises(UrdynStorageCorruptError):
        cx.recall("seed")

    assert cx._db_path.read_bytes() == damaged
    # the canonical row itself was never touched or lost by the sabotage,
    # and the damaged table was not silently rebuilt with its `role` column
    connection = sqlite3.connect(cx._db_path)
    try:
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (original.memory_id,)
        ).fetchone()
        columns = {name for _, name, *_ in connection.execute("PRAGMA table_info(memory_evidence)")}
    finally:
        connection.close()
    assert row == (original.memory_id, "seed content that must survive")
    assert "role" not in columns
