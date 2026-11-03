"""Backward compatibility with A4 (schema v3) stores.

Schema v3 only existed in Cortex development builds; implicit
migration is retired. Tests run on the AUTHENTIC v3 Brain written by the
historical code (refused, intact) and, for the still-valid promotion
expectation, on AUTHENTIC lessons written by released Urdyn versions. The
hand-written v3 builder was removed: on the v7 store `Urdyn.init` creates it
never produced a v3 store.
"""

import sqlite3

import pytest

import legacy_brains as legacy
from urdyn import Urdyn, UrdynStorageError, UrdynStorageSchemaError
from urdyn import maintenance

def _v3(tmp_path):
    root, generated = legacy.cortex_brain(tmp_path, 3)
    return root, generated, legacy.db_path(root)


def test_v3_brain_is_refused_and_its_memories_survive_exactly(tmp_path):
    """Was `test_v3_database_opens_and_memory_survives`."""
    root, generated, db = _v3(tmp_path)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")
    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 3 and facts["integrity"] == "ok"
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


def test_v3_refusal_creates_no_skill_tables(tmp_path):
    """Was `test_v3_migration_reaches_v4_and_creates_skill_tables`."""
    root, _, db = _v3(tmp_path)
    for entry in (lambda: Urdyn.open(root).recall("synthetic"), lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.recover_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 3
    assert not {"skills", "skill_steps", "skill_conditions", "skill_evidence"} & set(facts["tables"])


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_lesson_recorded_by_a_release_promotes_to_a_candidate_skill(tmp_path, release):
    """Was `test_v3_migrated_lesson_can_be_promoted_to_a_skill`. The valid part:
    a lesson an EARLIER build recorded as user-asserted can be promoted by this
    build, and the skill stays a candidate (no epistemic promotion). Runs on the
    AUTHENTIC lesson written by each release's own `learn`."""
    root, generated = legacy.release_experience_brain(tmp_path, release)
    assert generated["lesson_state"] == "user_asserted"
    cx = Urdyn.open(root)
    (lesson,) = cx.state(kind="lesson")
    assert lesson.memory_id == generated["lesson"]

    skill = cx.promote(lesson, name="Safely modify refresh-token rotation", purpose="p", steps=["s1"])

    assert skill.source_lesson_id == generated["lesson"]
    assert skill.verification_state == "candidate"
    (still,) = Urdyn.open(root).state(kind="lesson")
    assert (still.memory_id, still.content, still.epistemic_state) == (lesson.memory_id, lesson.content, "user_asserted")


def test_v3_refusal_is_safe_to_repeat(tmp_path):
    """Was `test_v3_migration_is_safe_to_repeat`."""
    root, _, _ = _v3(tmp_path)
    before = legacy.snapshot(root)
    messages = set()
    for _ in range(3):
        with pytest.raises(UrdynStorageSchemaError) as refused:
            Urdyn.open(root).recall("synthetic")
        messages.add(str(refused.value))
    assert len(messages) == 1
    assert legacy.snapshot(root) == before


def test_v3_sabotaged_store_fails_closed_untouched(tmp_path):
    """Was `test_v3_migration_does_not_destroy_data_on_failure`: a
    colliding `skill_conditions` made the v3->v4 step fail after creating
    `skills`/`skill_steps`, and the whole step had to roll back. On the
    AUTHENTIC v3 Brain the store is refused before any step: still v3, row
    intact, none of the v4 tables created, the sabotage table untouched."""
    root, generated = legacy.cortex_brain(tmp_path, 3)
    db = legacy.db_path(root)
    legacy.sabotage(db, "CREATE TABLE skill_conditions (bogus_column TEXT)")
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageError):
        Urdyn.open(root).recall("must survive")

    assert legacy.snapshot(root) == before
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 3
    assert {row[0] for row in facts["memories"]} == set(generated["written"])
    assert not {"skills", "skill_steps", "skill_evidence"} & set(facts["tables"])
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert [r[1] for r in connection.execute("PRAGMA table_info(skill_conditions)")] == ["bogus_column"]
    finally:
        connection.close()
