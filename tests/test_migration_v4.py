"""Backward-compatibility tests: opening and upgrading an A4 (schema v3)
store to A5 (schema v4).

Mirrors the schema-migration test style already established for v1->v2
(test_migration.py) and v1/v2->v3 (test_migration_v3.py): the older
schema is built by hand rather than trusted from current code, so a real
migration path is exercised.
"""

import datetime as dt
import sqlite3
import uuid

import pytest

from urdyn import Urdyn, UrdynStorageError

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


def _table_exists(connection, name):
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _build_v3_database(db_path, *, memory_id=None, content="a lesson from A4", kind="lesson"):
    """Insert one more memory into a v3-schema database, creating the
    schema first if this is the first call for `db_path`."""
    memory_id = memory_id or uuid.uuid4().hex
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            if not _table_exists(connection, "memories"):
                connection.execute(_CREATE_MEMORIES_V2_SQL)
                connection.execute(_CREATE_EVIDENCE_SQL)
                connection.execute(_CREATE_MEMORY_EVIDENCE_SQL)
                connection.execute(_CREATE_EVENTS_SQL)
                connection.execute(_CREATE_ATTEMPTS_SQL)
                connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
                connection.execute("PRAGMA user_version = 3")
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, content, kind, "user_asserted", recorded_at, None),
            )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, "memory_recorded", memory_id, recorded_at),
            )
    finally:
        connection.close()
    return memory_id


def test_v3_database_opens_and_memory_survives(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v3_database(cx._db_path)

    results = cx.recall("a lesson from A4")

    assert len(results) == 1
    assert results[0].memory_id == memory_id
    assert results[0].kind == "lesson"


def test_v3_migration_reaches_v4_and_creates_skill_tables(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    _build_v3_database(cx._db_path)

    # migration v3 -> v4 -> v5 -> v6 happens transparently on first open
    cx.recall("a lesson from A4")

    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert version == 7
    assert {"skills", "skill_steps", "skill_conditions", "skill_evidence"} <= tables


def test_v3_migrated_lesson_can_be_promoted_to_a_skill(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v3_database(cx._db_path, content="Use only the newly issued refresh token.")

    (lesson,) = cx.state(kind="lesson")
    assert lesson.memory_id == memory_id

    skill = cx.promote(lesson, name="Safely modify refresh-token rotation", purpose="p", steps=["s1"])

    assert skill.source_lesson_id == memory_id
    # the migrated lesson was recorded user_asserted (v3 had no concept of
    # promoting anything), so the resulting skill must stay a candidate
    assert skill.verification_state == "candidate"


def test_v3_migration_is_safe_to_repeat(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v3_database(cx._db_path)

    first = cx.recall("a lesson from A4")
    second = cx.recall("a lesson from A4")

    assert [m.memory_id for m in first] == [memory_id]
    assert [m.memory_id for m in second] == [memory_id]


def test_v3_migration_does_not_destroy_data_on_failure(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v3_database(cx._db_path, content="must survive a failed migration")

    # Sabotage the migration partway through: `_migrate_v3_to_v4` creates
    # `skills`, then `skill_steps`, then `skill_conditions`, then
    # `skill_evidence`, in that order. Pre-creating a colliding
    # `skill_conditions` table lets the migration successfully create the
    # first two new tables *inside its own transaction* before failing on
    # the third -- a real partial-write scenario, not a precondition
    # failure before anything was written (which pre-colliding `skills`,
    # the very first table, would only demonstrate).
    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("CREATE TABLE skill_conditions (bogus_column TEXT)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.recall("must survive")

    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        tables = {
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        conditions_columns = [
            r[1] for r in connection.execute("PRAGMA table_info(skill_conditions)").fetchall()
        ]
    finally:
        connection.close()

    assert version == 3
    assert row == (memory_id, "must survive a failed migration")
    # the transaction must have rolled back entirely: `skills` and
    # `skill_steps`, created earlier in the same failed transaction, must
    # not have survived, and the sabotage table must be untouched (still
    # its bogus shape, not the real skill_conditions schema).
    assert "skills" not in tables
    assert "skill_steps" not in tables
    assert "skill_evidence" not in tables
    assert conditions_columns == ["bogus_column"]
