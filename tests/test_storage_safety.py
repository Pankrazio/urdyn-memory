"""Safety tests for the persisted memory store."""

import sqlite3

import pytest

from cortex_memory import Cortex, CortexStorageError


def test_unsupported_schema_version_is_rejected_without_data_loss(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("a memory that must not be destroyed")

    db_path = tmp_path / ".cortex" / "memory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("memory")

    # the underlying row must still be present; only the version guard tripped
    connection = sqlite3.connect(db_path)
    count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    connection.close()
    assert count == 1


def test_garbage_db_file_fails_explicitly_instead_of_raw_traceback(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    db_path = tmp_path / ".cortex" / "memory.db"
    db_path.write_bytes(b"this is not a sqlite database")

    with pytest.raises(CortexStorageError):
        cx.remember("should not silently overwrite garbage")

    # the garbage file must not have been silently destroyed/recreated
    assert db_path.read_bytes() == b"this is not a sqlite database"


def test_recall_does_not_create_store_file_when_none_exists(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    cx.recall("anything")

    assert not (tmp_path / ".cortex" / "memory.db").exists()


def test_status_does_not_create_store_file_when_none_exists(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    assert cx._count_memories() == 0
    assert not (tmp_path / ".cortex" / "memory.db").exists()


def test_corrupted_kind_value_is_rejected_explicitly(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("a memory")

    db_path = tmp_path / ".cortex" / "memory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE memories SET kind = 'not-a-real-kind'")
    connection.commit()
    connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("memory")


def test_corrupted_memory_id_is_rejected_explicitly(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("a memory")

    db_path = tmp_path / ".cortex" / "memory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE memories SET memory_id = 'not-a-valid-id'")
    connection.commit()
    connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("memory")


def test_missing_memories_table_with_valid_schema_version_is_rejected(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("a memory")

    db_path = tmp_path / ".cortex" / "memory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("DROP TABLE memories")
    connection.commit()
    connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("memory")

    with pytest.raises(CortexStorageError):
        cx._count_memories()

    with pytest.raises(CortexStorageError):
        cx.remember("must not silently recreate the table")


def test_sqlite_error_during_read_is_translated_to_cortex_error(tmp_path):
    """A corruption that only manifests mid-read (table present, but with
    an altered/incompatible column layout) must still surface as
    CortexStorageError, not a raw sqlite3 exception."""
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("a memory")

    db_path = tmp_path / ".cortex" / "memory.db"
    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE memories DROP COLUMN kind")
    connection.commit()
    connection.close()

    with pytest.raises(CortexStorageError):
        cx.recall("memory")
