"""Backward-compatibility tests: opening and upgrading an A2 (schema v1) store.

These tests build the v1 database file by hand (mirroring the exact v1
`_ensure_schema` layout from before A3), rather than trusting current code
to simulate it, so that a real migration path is exercised.
"""

import datetime as dt
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex, CortexStorageError

_V1_CREATE_MEMORIES_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""


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


def test_v1_database_opens_and_memory_survives(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v1_database(cx._db_path)

    results = cx.recall("a memory from before A3")

    assert len(results) == 1
    assert results[0].memory_id == memory_id
    assert results[0].content == "a memory from before A3"
    assert results[0].kind == "decision"
    assert results[0].epistemic_state == "user_asserted"


def test_v1_migration_preserves_a3_defaults_for_old_memory(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_v1_database(cx._db_path)

    (memory,) = cx.recall("a memory from before A3")

    assert memory.supersedes is None
    assert memory.evidence_ids == ()


def test_v1_migration_preserves_memory_count(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_v1_database(cx._db_path)

    assert cx._count_memories() == 1


def test_v1_migration_is_safe_to_repeat(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v1_database(cx._db_path)

    first_open_results = cx.recall("a memory from before A3")
    second_open_results = cx.recall("a memory from before A3")

    assert [m.memory_id for m in first_open_results] == [memory_id]
    assert [m.memory_id for m in second_open_results] == [memory_id]


def test_v1_migrated_memory_participates_in_new_supersession(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_id = _build_v1_database(cx._db_path, content="PostgreSQL was selected.")

    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old_id)

    history = cx.timeline(kind="decision")
    assert [m.memory_id for m in history] == [old_id, new.memory_id]
    current = cx.state(kind="decision")
    assert [m.memory_id for m in current] == [new.memory_id]


def test_v1_migration_does_not_destroy_data_on_failure(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory_id = _build_v1_database(cx._db_path, content="must survive a failed migration")

    # Sabotage the migration: pre-create a colliding 'evidence' table so the
    # migration's own CREATE TABLE fails partway through the transaction.
    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("CREATE TABLE evidence (bogus_column TEXT)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("must survive")

    # the transaction must have rolled back entirely: no supersedes column,
    # original row untouched, still stamped as v1.
    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        columns = [row[1] for row in connection.execute("PRAGMA table_info(memories)").fetchall()]
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
    finally:
        connection.close()

    assert version == 1
    assert "supersedes" not in columns
    assert row == (memory_id, "must survive a failed migration")


def test_future_schema_version_is_rejected_without_data_loss(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("must not be destroyed by an unknown future schema")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("PRAGMA user_version = 99")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("must not be destroyed")

    connection = sqlite3.connect(cx._db_path)
    try:
        count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        connection.close()
    assert count == 1
