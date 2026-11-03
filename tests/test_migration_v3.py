"""Backward compatibility with A3 (schema v2) and the A2->A4 chain.

Schemas v1-v6 only existed in Cortex development builds; implicit
migration is retired. The tests run on AUTHENTIC Brains written by the
historical code (tests/legacy_brains.py) and pin refusal with the data
intact; the still-valid A4 expectation runs on AUTHENTIC released Brains.
The hand-written builders these tests used were removed: on the v7 store
`Urdyn.init` creates they never produced a legacy store.
"""

import sqlite3

import pytest

import legacy_brains as legacy
from urdyn import Urdyn, UrdynStorageError, UrdynStorageSchemaError
from urdyn import maintenance

# The five tests below replace tests that built a "v2" store with a
# hand-written builder on the v7 store `Urdyn.init` creates: the builder only
# created its v2 schema when `memories` was absent, so it inserted into the
# CURRENT tables and no v2 store ever existed.
# Migration expectations now run on the AUTHENTIC v2 Brain (refused, intact);
# the one still-valid v7 expectation runs on AUTHENTIC released Brains.


def _v2(tmp_path):
    root, generated = legacy.cortex_brain(tmp_path, 2)
    return root, generated, legacy.db_path(root)


def test_v2_brain_is_refused_and_its_memories_survive_exactly(tmp_path):
    """Was `test_v2_database_opens_and_memory_survives`."""
    root, generated, db = _v2(tmp_path)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")
    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 2 and facts["integrity"] == "ok"
    assert {row[0] for row in facts["memories"]} == set(generated["written"])
    assert not {"attempts", "attempt_evidence"} & set(facts["tables"])


def test_v2_refusal_preserves_the_stored_supersession(tmp_path):
    """Was `test_v2_migration_preserves_supersession`: the supersession the
    historical code recorded (one memory superseding another) is still exactly
    what the store holds after the refusal."""
    root, generated, db = _v2(tmp_path)
    assert generated["supersession"] is True
    before = legacy.store_facts(db)["memories"]
    superseding = [row for row in before if row[5] is not None]
    assert len(superseding) == 1 and superseding[0][5] in generated["written"]
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).state()
    assert legacy.store_facts(db)["memories"] == before


def test_v2_memory_count_is_unchanged_by_refusals(tmp_path):
    """Was `test_v2_migration_preserves_memory_count`."""
    root, generated, db = _v2(tmp_path)
    for entry in (lambda: Urdyn.open(root), lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.recover_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    assert len(legacy.store_facts(db)["memories"]) == len(generated["written"])


def test_v2_refusal_is_safe_to_repeat(tmp_path):
    """Was `test_v2_migration_is_safe_to_repeat`."""
    root, _, _ = _v2(tmp_path)
    before = legacy.snapshot(root)
    messages = set()
    for _ in range(3):
        with pytest.raises(UrdynStorageSchemaError) as refused:
            Urdyn.open(root).recall("synthetic")
        messages.add(str(refused.value))
    assert len(messages) == 1
    assert legacy.snapshot(root) == before


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_released_brain_accepts_new_a4_experience_beside_its_memories(tmp_path, release):
    """Was `test_v2_migrated_workspace_can_record_new_a4_experience`. The valid
    part -- new A4 experience (attempt, lesson) recorded next to memories an
    EARLIER build wrote, both usable -- on the AUTHENTIC Brain of a release."""
    root, generated = legacy.release_brain(tmp_path, release)
    cx = Urdyn.open(root)
    written = {m.memory_id: m.content for m in cx.timeline()}
    assert set(written) == set(generated["written"])

    attempt = cx.record_attempt(task="task", approach="approach", outcome="failed")
    lesson = cx.learn("a lesson learned after the release")

    assert cx.preflight("task approach").known_failures[0].attempt_id == attempt.attempt_id
    assert lesson.kind == "lesson"
    after = {m.memory_id: m.content for m in Urdyn.open(root).timeline()}
    assert after == {**written, lesson.memory_id: lesson.content}
    assert cx._count_memories() == len(generated["written"]) + 1


def test_v2_sabotaged_store_fails_closed_untouched(tmp_path):
    """Was `test_v2_migration_does_not_destroy_data_on_failure`: a
    colliding `attempts` table made the v2->v3 step fail and roll back (still
    v2, row intact). On the AUTHENTIC v2 Brain the store is now refused before
    any step: still v2, every memory intact, every byte kept."""
    root, generated = legacy.cortex_brain(tmp_path, 2)
    db = legacy.db_path(root)
    legacy.sabotage(db, "CREATE TABLE attempts (bogus_column TEXT)")
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageError):
        Urdyn.open(root).recall("must survive")

    assert legacy.snapshot(root) == before
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 2
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


def test_v1_brain_never_reaches_a_current_schema(tmp_path):
    """Was `test_full_chain_v1_database_reaches_v3_and_supports_a4_features`
    (implicit v1->v7 chain on first write). On the AUTHENTIC v1 Brain no entry
    -- open, init, a write, the FTS maintenance entry -- moves it: version,
    tables and memories stay exactly as the historical code wrote them."""
    root, generated = legacy.cortex_brain(tmp_path, 1)
    db = legacy.db_path(root)
    facts = legacy.store_facts(db)
    for entry in (lambda: Urdyn.open(root).record_attempt(task="a task", approach="an approach", outcome="succeeded"),
                  lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.recover_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 1
    assert not {"attempts", "attempt_evidence", "events"} & set(facts["tables"])


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
