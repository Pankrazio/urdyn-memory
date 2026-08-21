"""Backward-compatibility tests: opening and upgrading an A3 (schema v2)
store to A4 (schema v3), and the full A2->A4 chain from schema v1.

These tests build the v2 database file by hand (mirroring the exact v2
`_ensure_schema` layout from A3, before `attempts`/`attempt_evidence`
existed), rather than trusting current code to simulate it, so that a
real migration path is exercised.
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

_V1_CREATE_MEMORIES_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""


def _table_exists(connection, name):
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _build_v2_database(db_path, *, memory_id=None, content="a decision from A3", supersedes=None):
    """Insert one more memory into a v2-schema database, creating the
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
                connection.execute("PRAGMA user_version = 2")
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, content, "decision", "user_asserted", recorded_at, supersedes),
            )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, "memory_recorded", memory_id, recorded_at),
            )
    finally:
        connection.close()
    return memory_id


def _build_v1_database(db_path, *, memory_id=None, content="a memory from before A3"):
    memory_id = memory_id or uuid.uuid4().hex
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(_V1_CREATE_MEMORIES_SQL)
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, content, "decision", "user_asserted", dt.datetime.now(dt.timezone.utc).isoformat()),
            )
            connection.execute("PRAGMA user_version = 1")
    finally:
        connection.close()
    return memory_id


def test_v2_database_opens_and_memory_survives(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v2_database(cx._db_path)

    results = cx.recall("a decision from A3")

    assert len(results) == 1
    assert results[0].memory_id == memory_id
    assert results[0].supersedes is None
    assert results[0].evidence_ids == ()


def test_v2_migration_preserves_supersession(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    old_id = _build_v2_database(cx._db_path, content="PostgreSQL was selected.")
    new_id = _build_v2_database(cx._db_path, content="SQLite was selected for V1.", supersedes=old_id)

    history = cx.timeline(kind="decision")
    assert {m.memory_id for m in history} == {old_id, new_id}
    current = cx.state(kind="decision")
    assert [m.memory_id for m in current] == [new_id]


def test_v2_migration_preserves_memory_count(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    _build_v2_database(cx._db_path)

    assert cx._count_memories() == 1


def test_v2_migration_is_safe_to_repeat(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v2_database(cx._db_path)

    first = cx.recall("a decision from A3")
    second = cx.recall("a decision from A3")

    assert [m.memory_id for m in first] == [memory_id]
    assert [m.memory_id for m in second] == [memory_id]


def test_v2_migrated_workspace_can_record_new_a4_experience(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    _build_v2_database(cx._db_path, content="SQLite was selected for V1.")

    attempt = cx.record_attempt(task="task", approach="approach", outcome="failed")
    lesson = cx.learn("a lesson learned after migration")

    assert cx.preflight("task approach").known_failures[0].attempt_id == attempt.attempt_id
    assert lesson.kind == "lesson"
    # the pre-existing A3 memory must still be there alongside the new A4 data
    assert cx._count_memories() == 2


def test_v2_migration_does_not_destroy_data_on_failure(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    memory_id = _build_v2_database(cx._db_path, content="must survive a failed migration")

    # Sabotage the migration: pre-create a colliding 'attempts' table so
    # the migration's own CREATE TABLE fails partway through.
    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("CREATE TABLE attempts (bogus_column TEXT)")
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
    finally:
        connection.close()

    assert version == 2
    assert row == (memory_id, "must survive a failed migration")


def test_full_chain_v1_database_reaches_v3_and_supports_a4_features(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    old_id = _build_v1_database(cx._db_path, content="a memory from A2")

    # migration v1 -> v2 -> v3 -> v4 happens transparently on first open
    attempt = cx.record_attempt(task="a task", approach="an approach", outcome="succeeded")

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

    # A5 adds schema v4 (Skill persistence), A12.1 adds v5 (explicit
    # support role), A13.1 adds v6 (Conflict relation) and A19.1 adds v7
    # (Source observations) on top of A4's v3; a v1 database now migrates
    # all the way through, not just to v3.
    assert version == 7
    assert {"memories", "evidence", "memory_evidence", "events", "attempts", "attempt_evidence"} <= tables

    results = cx.recall("a memory from A2")
    assert [m.memory_id for m in results] == [old_id]
    assert cx.preflight("a task an approach").known_failures == ()


def test_future_schema_version_is_rejected_without_data_loss(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("must not be destroyed by an unknown future schema")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("PRAGMA user_version = 99")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.recall("must not be destroyed")

    connection = sqlite3.connect(cx._db_path)
    try:
        count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        connection.close()
    assert count == 1
