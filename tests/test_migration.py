"""Backward compatibility with an A2 (schema v1) store.

Schema v1 only ever existed in unreleased Cortex development builds, and
Urdyn never migrates a legacy schema implicitly (see
`docs/compatibility.md`). These tests run on the AUTHENTIC v1 Brain written
by the historical code (tests/legacy_brains.py, SHA-256 checked) and pin that
the store is refused fail-closed while the old memory is preserved exactly:
nothing is invented, repeating is harmless, nothing is written, and a
sabotaged store fails closed without being touched.
"""

import sqlite3

import pytest

import legacy_brains as legacy
from urdyn import Urdyn, UrdynStorageError, UrdynStorageSchemaError
from urdyn import maintenance


def _v1(tmp_path):
    root, generated = legacy.cortex_brain(tmp_path, 1)
    return root, generated, legacy.db_path(root)


def test_v1_brain_is_refused_and_its_memories_survive_exactly(tmp_path):
    root, generated, db = _v1(tmp_path)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")

    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 1 and facts["integrity"] == "ok"
    assert {row[0] for row in facts["memories"]} == set(generated["written"])
    # Exactly the v1 row shape the historical code wrote: no column added.
    assert facts["columns"]["memories"] == ["memory_id", "content", "kind", "epistemic_state", "recorded_at"]


def test_v1_refusal_invents_no_supersession_and_no_evidence(tmp_path):
    root, _, db = _v1(tmp_path)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root)
    facts = legacy.store_facts(db)
    assert "supersedes" not in facts["columns"]["memories"]
    assert not {"evidence", "memory_evidence", "events"} & set(facts["tables"])


def test_v1_memory_count_is_unchanged_by_every_entry(tmp_path):
    root, generated, db = _v1(tmp_path)
    for entry in (lambda: Urdyn.open(root), lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.inspect_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    assert len(legacy.store_facts(db)["memories"]) == len(generated["written"])


def test_v1_refusal_is_safe_to_repeat(tmp_path):
    root, _, db = _v1(tmp_path)
    before = legacy.snapshot(root)
    errors = []
    for _ in range(3):
        with pytest.raises(UrdynStorageSchemaError) as refused:
            Urdyn.open(root)
        errors.append(str(refused.value))
    assert len(set(errors)) == 1
    assert legacy.snapshot(root) == before


def test_v1_brain_cannot_be_written_into(tmp_path):
    """The original wrote a NEW memory superseding the migrated one. A v1 store
    must never receive a write from this build: no handle exists to write with,
    and the FTS maintenance entry (the one write path that bypasses open) is
    refused as well; the history stays exactly what the historical code wrote."""
    root, generated, db = _v1(tmp_path)
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).remember("SQLite was selected for V1.", kind="decision")
    with pytest.raises(UrdynStorageSchemaError):
        maintenance.recover_search_index(root)
    assert legacy.snapshot(root) == before
    assert {row[0] for row in legacy.store_facts(db)["memories"]} == set(generated["written"])


def test_v1_sabotaged_store_fails_closed_untouched(tmp_path):
    """Original: a colliding `evidence` table made the v1->v2 step fail, and the
    step had to roll back entirely (still v1, no `supersedes`, row intact).
    Now refused before any step, with the same three facts and every byte kept."""
    root, generated, db = _v1(tmp_path)
    legacy.sabotage(db, "CREATE TABLE evidence (bogus_column TEXT)")
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageError):
        Urdyn.open(root).recall("must survive")
    assert legacy.snapshot(root) == before
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 1
    assert "supersedes" not in facts["columns"]["memories"]
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


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
