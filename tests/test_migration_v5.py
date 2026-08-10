"""Backward-compatibility tests: opening and upgrading an A5-A11 (schema
v4) store to A12.1 (schema v5).

Mirrors the schema-migration test style already established for v1->v2
(test_migration.py), v1/v2->v3 (test_migration_v3.py), and v3->v4
(test_migration_v4.py): the older schema is built by hand rather than
trusted from current code, so a real migration path is exercised.

v5 adds exactly one column, `memory_evidence.role`, backfilled to
`'related'` for every pre-existing row (A12.1's explicit support
relation -- see `_store.py`'s module docstring). No new table.
"""

import datetime as dt
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex, CortexStorageError

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

# The v4 shape: no `role` column. This is exactly what A12.1 adds.
_CREATE_MEMORY_EVIDENCE_V4_SQL = """
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


def _table_exists(connection, name):
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _build_v4_database(
    db_path,
    *,
    memory_id=None,
    content="a legacy verified lesson from before A12.1",
    kind="lesson",
    epistemic_state="user_asserted",
    evidence_id=None,
    evidence_kind="test_result",
):
    """Insert one more memory into a v4-schema database (no `role` column
    on `memory_evidence`), creating the schema first if this is the first
    call for `db_path`. If `evidence_id` is given, links it to the memory
    exactly as pre-A12.1 code would have: a bare `memory_evidence` row
    with no support/related distinction."""
    memory_id = memory_id or uuid.uuid4().hex
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            if not _table_exists(connection, "memories"):
                connection.execute(_CREATE_MEMORIES_V2_SQL)
                connection.execute(_CREATE_EVIDENCE_SQL)
                connection.execute(_CREATE_MEMORY_EVIDENCE_V4_SQL)
                connection.execute(_CREATE_EVENTS_SQL)
                connection.execute(_CREATE_ATTEMPTS_SQL)
                connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
                connection.execute(_CREATE_SKILLS_SQL)
                connection.execute(_CREATE_SKILL_STEPS_SQL)
                connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
                connection.execute(_CREATE_SKILL_EVIDENCE_SQL)
                connection.execute("PRAGMA user_version = 4")
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, content, kind, epistemic_state, recorded_at, None),
            )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, "memory_recorded", memory_id, recorded_at),
            )
            if evidence_id is not None:
                connection.execute(
                    "INSERT INTO evidence (evidence_id, content, kind, recorded_at) VALUES (?, ?, ?, ?)",
                    (evidence_id, "legacy qualifying evidence", evidence_kind, recorded_at),
                )
                connection.execute(
                    "INSERT INTO memory_evidence (memory_id, evidence_id, position) VALUES (?, ?, ?)",
                    (memory_id, evidence_id, 0),
                )
    finally:
        connection.close()
    return memory_id


def test_v4_database_opens_and_memory_survives(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v4_database(cx._db_path)

    results = cx.recall("legacy verified lesson")

    assert len(results) == 1
    assert results[0].memory_id == memory_id


def test_v4_migration_reaches_v5_and_adds_role_column(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_v4_database(cx._db_path)

    # migration v4 -> v5 happens transparently on first open
    cx.recall("legacy verified lesson")

    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        columns = [row[1] for row in connection.execute("PRAGMA table_info(memory_evidence)").fetchall()]
    finally:
        connection.close()

    assert version == 5
    assert "role" in columns


def test_legacy_verified_memory_stays_verified_with_empty_supporting_ids(tmp_path):
    """[A12.1 grandfathering, acceptance-critical] A memory recorded
    `verified` under the pre-A12.1 contract must remain `verified` after
    migration -- never invalidated, downgraded, or rewritten -- even
    though it has no explicit supporting Evidence (that concept did not
    exist when it was recorded). Its generic `evidence_ids` must survive
    unchanged; `supporting_evidence_ids` is empty, not invented."""
    cx = Cortex.init(tmp_path, "dev")
    evidence_id = uuid.uuid4().hex
    memory_id = _build_v4_database(
        cx._db_path,
        content="Migration rollback is atomic.",
        epistemic_state="verified",
        evidence_id=evidence_id,
        evidence_kind="test_result",
    )

    (lesson,) = [m for m in cx.state(kind="lesson") if m.memory_id == memory_id]

    assert lesson.epistemic_state == "verified"
    assert lesson.evidence_ids == (evidence_id,)
    assert lesson.supporting_evidence_ids == ()


def test_legacy_verified_lesson_still_surfaces_in_preflight_after_migration(tmp_path):
    """[A12.1.1 section 5/9] `preflight().verified_lessons` filters only
    on `epistemic_state == "verified"` (see `_workspace.py`'s
    `Cortex.preflight`) -- it has never depended on
    `supporting_evidence_ids` and must not start doing so now. A
    v4-migrated verified lesson with empty `supporting_evidence_ids`
    must keep appearing in preflight exactly as it did before A12.1,
    for a lexically matching task."""
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v4_database(
        cx._db_path,
        content="Migration rollback is atomic under process crashes.",
        epistemic_state="verified",
        evidence_id=uuid.uuid4().hex,
        evidence_kind="test_result",
    )

    result = cx.preflight("Migration rollback is atomic under process crashes.")

    matched = {m.memory_id for m in result.verified_lessons}
    assert memory_id in matched
    (surfaced,) = [m for m in result.verified_lessons if m.memory_id == memory_id]
    assert surfaced.supporting_evidence_ids == ()


def test_legacy_memory_evidence_rows_are_backfilled_to_related_not_supporting(tmp_path):
    """The migration must never invent an explicit support assertion a
    pre-A12.1 caller never made."""
    cx = Cortex.init(tmp_path, "dev")
    evidence_id = uuid.uuid4().hex
    memory_id = _build_v4_database(
        cx._db_path,
        epistemic_state="verified",
        evidence_id=evidence_id,
    )

    cx.recall("legacy")  # trigger the migration

    connection = sqlite3.connect(cx._db_path)
    try:
        (role,) = connection.execute(
            "SELECT role FROM memory_evidence WHERE memory_id = ? AND evidence_id = ?",
            (memory_id, evidence_id),
        ).fetchone()
    finally:
        connection.close()

    assert role == "related"


def test_v4_migrated_lesson_preserved_after_new_supporting_lesson_recorded(tmp_path):
    """The legacy row and a freshly-recorded, properly-supported lesson
    coexist without interference after migration."""
    cx = Cortex.init(tmp_path, "dev")
    legacy_evidence_id = uuid.uuid4().hex
    legacy_id = _build_v4_database(
        cx._db_path,
        content="Legacy verified claim.",
        epistemic_state="verified",
        evidence_id=legacy_evidence_id,
    )

    validation = cx.add_evidence("Fresh check passed.", kind="test_result")
    fresh = cx.learn("Fresh verified claim.", supporting_evidence=[validation], verified=True)

    lessons = {m.memory_id: m for m in cx.state(kind="lesson")}
    assert lessons[legacy_id].epistemic_state == "verified"
    assert lessons[legacy_id].supporting_evidence_ids == ()
    assert lessons[fresh.memory_id].supporting_evidence_ids == (validation.evidence_id,)


def test_v4_migration_is_safe_to_repeat(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v4_database(cx._db_path)

    first = cx.recall("legacy verified lesson")
    second = cx.recall("legacy verified lesson")

    assert [m.memory_id for m in first] == [memory_id]
    assert [m.memory_id for m in second] == [memory_id]


def test_v4_migration_does_not_destroy_data_on_failure(tmp_path):
    """Migration atomicity (same standard as the v3->v4 test): sabotage
    the single `ALTER TABLE` `_migrate_v4_to_v5` issues by pre-adding a
    colliding `role` column by hand, forcing SQLite to reject the real
    `ALTER TABLE ... ADD COLUMN role` with a duplicate-column error
    partway through the migration's own transaction. The schema version
    must not advance and the pre-existing canonical row must survive
    untouched."""
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v4_database(cx._db_path, content="must survive a failed migration")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("ALTER TABLE memory_evidence ADD COLUMN role TEXT")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("must survive")

    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
    finally:
        connection.close()

    assert version == 4
    assert row == (memory_id, "must survive a failed migration")


# ---------------------------------------------------------------------------
# A12.1.1: v5 canonical-data corruption -- no silent reinterpretation
# ---------------------------------------------------------------------------


def test_invalid_role_value_raises_cleanly_instead_of_silently_dropping_support(tmp_path):
    """A `memory_evidence.role` value outside `{'related', 'supporting'}`
    (never written by this codebase, but not impossible for hand-edited
    or externally-touched data) must not be silently reinterpreted as
    "not supporting" -- that would drop a real support designation with
    no error at all. It must raise `CortexStorageError`, the same way
    `_row_to_memory` already does for a corrupted `kind`/`epistemic_state`."""
    cx = Cortex.init(tmp_path, "dev")
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

    with pytest.raises(CortexStorageError):
        cx.state(kind="lesson")


def test_user_version_5_with_missing_role_column_raises_cleanly(tmp_path):
    """[A12.1.1 section 4] `PRAGMA user_version == 5` alone must not be
    trusted as proof the schema is actually shaped like v5: if
    `memory_evidence` is missing its `role` column (a corrupted or
    incomplete upgrade), every read/write path that touches
    `role` fails loudly as `CortexStorageError` -- the same standard of
    schema integrity already applied to every other version (table-level
    checks in `_ensure_schema`, `sqlite3.DatabaseError` wrapped
    consistently everywhere else in this module). Nothing recreates the
    table automatically and no data is lost."""
    cx = Cortex.init(tmp_path, "dev")
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
    assert version == 5  # sabotage did not touch the version stamp

    reopened = Cortex.open(tmp_path)
    with pytest.raises(CortexStorageError):
        reopened.recall("seed")

    # the canonical row itself was never touched or lost by the sabotage
    connection = sqlite3.connect(cx._db_path)
    try:
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (original.memory_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row == (original.memory_id, "seed content that must survive")
