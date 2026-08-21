"""Tests for the `Skill` primitive and `Cortex.promote()`."""

import dataclasses
import datetime as dt
import sqlite3

import pytest

from cortex_memory import Cortex, CortexStorageError
from cortex_memory._memory import Memory
from cortex_memory._store import MemoryStore


def _verified_lesson(cx, content="Use only the newly issued refresh token."):
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    return cx.learn(content, supporting_evidence=[validation], verified=True)


def test_promote_assigns_stable_valid_id(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    skill = cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=["Inspect the rotation flow.", "Persist only the newly issued token."],
    )

    assert isinstance(skill.skill_id, str)
    assert skill.skill_id


def test_promote_from_verified_lesson_yields_verified_skill(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.verification_state == "verified"


def test_promote_from_candidate_lesson_yields_candidate_skill(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = cx.learn("Refresh tokens might need special handling.")

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.verification_state == "candidate"


def test_promote_from_candidate_lesson_with_weak_evidence_yields_candidate_skill(tmp_path):
    """A skill must not become verified just because *some* evidence was
    cited somewhere -- the same weak-evidence rule `remember()` already
    enforces for Memory must hold for Skill too, without a second,
    laxer verification path being introduced for it."""
    cx = Cortex.init(tmp_path, "dev")
    opinion = cx.add_evidence("I think this works.", kind="user_statement")
    lesson = cx.learn("Refresh tokens might need special handling.", evidence=[opinion])

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.verification_state == "candidate"


def test_promote_has_no_verified_override_parameter(tmp_path):
    """Verification is derived from the source lesson's own epistemic
    state, not something the caller can assert directly -- there is no
    `verified=` (or similar) parameter to bypass that."""
    cx = Cortex.init(tmp_path, "dev")
    lesson = cx.learn("Refresh tokens might need special handling.")

    with pytest.raises(TypeError):
        cx.promote(lesson, name="n", purpose="p", steps=["s1"], verified=True)


def test_promote_preserves_steps_ordering(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    steps = ["Inspect the rotation flow.", "Persist only the newly issued token.", "Run auth tests."]

    skill = cx.promote(lesson, name="n", purpose="p", steps=steps)

    assert skill.steps == tuple(steps)


def test_promote_rejects_empty_steps(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=[])


def test_promote_rejects_blank_step(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=["ok", "   "])


def test_promote_rejects_empty_name(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="   ", purpose="p", steps=["s1"])


def test_promote_rejects_empty_purpose(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="   ", steps=["s1"])


def test_promote_rejects_non_lesson_memory(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    decision = cx.remember("SQLite was selected.", kind="decision")

    with pytest.raises(ValueError):
        cx.promote(decision, name="n", purpose="p", steps=["s1"])


def test_promote_rejects_string_as_steps(tmp_path):
    """`str` satisfies `Sequence[str]`, so without an explicit guard
    `steps="pytest"` would silently become `("p", "y", "t", "e", "s",
    "t")` -- five bogus one-character steps instead of a rejected call."""
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps="pytest")


def test_promote_rejects_bytes_as_steps(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=b"pytest")


def test_promote_rejects_string_as_conditions(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=["s1"], conditions="only for auth")


def test_promote_rejects_bytes_as_conditions(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=["s1"], conditions=b"only for auth")


def test_promote_accepts_list_and_tuple_of_steps(tmp_path):
    """Normal `list[str]`/`tuple[str, ...]` input must keep working exactly
    as before -- the str/bytes guard must not be over-broad."""
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    from_list = cx.promote(lesson, name="n1", purpose="p", steps=["step one", "step two"])
    other_lesson = cx.learn("Another lesson.")
    from_tuple = cx.promote(other_lesson, name="n2", purpose="p", steps=("step one", "step two"))

    assert from_list.steps == ("step one", "step two")
    assert from_tuple.steps == ("step one", "step two")


def test_promote_conditions_default_to_empty(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.conditions == ()


def test_promote_preserves_conditions_ordering(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    conditions = ["Only applies to authentication refresh flows.", "Not applicable to first-time login."]

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"], conditions=conditions)

    assert skill.conditions == tuple(conditions)


def test_promote_rejects_blank_condition(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    with pytest.raises(ValueError):
        cx.promote(lesson, name="n", purpose="p", steps=["s1"], conditions=["  "])


def test_promote_records_source_lesson_provenance(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.source_lesson_id == lesson.memory_id
    assert skill.evidence_ids == lesson.evidence_ids


def test_promote_ignores_forged_verification_state_on_caller_object(tmp_path):
    """The persisted Lesson is authoritative, not the
    `Memory` object the caller happens to pass in. Build a second object
    that shares a real, persisted CANDIDATE lesson's `memory_id` but
    claims `epistemic_state="verified"`; the resulting Skill must still
    be `candidate`, derived from what Cortex actually has on record for
    that id, not from the forged object's claim."""
    cx = Cortex.init(tmp_path, "dev")
    candidate = cx.learn("Refresh tokens might need special handling.")
    assert candidate.epistemic_state == "user_asserted"

    forged = dataclasses.replace(candidate, epistemic_state="verified")

    skill = cx.promote(forged, name="n", purpose="p", steps=["s1"])

    assert skill.verification_state == "candidate"


def test_promote_ignores_forged_evidence_ids_on_caller_object(tmp_path):
    """Provenance is derived from the persisted Lesson's own
    evidence links, not from whatever `evidence_ids` the caller's object
    happens to carry. A forged object sharing a real Lesson's memory_id
    but claiming different evidence must not redirect the Skill's
    provenance to that fabricated evidence."""
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    fake_evidence_id = "c" * 32

    forged = dataclasses.replace(lesson, evidence_ids=(fake_evidence_id,))

    skill = cx.promote(forged, name="n", purpose="p", steps=["s1"])

    assert skill.evidence_ids == lesson.evidence_ids
    assert fake_evidence_id not in skill.evidence_ids


def test_promote_does_not_mutate_or_supersede_the_lesson(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    (still_there,) = [m for m in cx.state(kind="lesson") if m.memory_id == lesson.memory_id]
    assert still_there.content == lesson.content
    assert still_there.epistemic_state == lesson.epistemic_state
    assert still_there.supersedes is None


def test_promote_rejects_fabricated_lesson_reference(tmp_path):
    """A `Memory` object that Cortex never actually persisted must not be
    promotable, even if it claims kind='lesson' -- otherwise provenance
    could point at nothing."""
    cx = Cortex.init(tmp_path, "dev")
    fabricated = Memory(
        memory_id="a" * 32,
        content="never actually persisted",
        kind="lesson",
        epistemic_state="verified",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.promote(fabricated, name="n", purpose="p", steps=["s1"])

    assert cx.skills() == []


def test_failed_promotion_does_not_partially_persist(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    fabricated = Memory(
        memory_id="a" * 32,
        content="never actually persisted",
        kind="lesson",
        epistemic_state="verified",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.promote(fabricated, name="n", purpose="p", steps=["s1"])

    assert cx.skills() == []
    with MemoryStore.create_or_open(cx._db_path) as store:
        (count,) = store._connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'skill_promoted'"
        ).fetchone()
    assert count == 0


def test_failed_promotion_does_not_survive_partial_writes(tmp_path, monkeypatch):
    """Stronger than `test_failed_promotion_does_not_partially_persist`:
    that one fails on a precondition check before any row is written.
    Here we force a genuine mid-transaction failure by making the
    *third* table `add_skill()` writes to (`skill_conditions`) collide
    on the primary key it is about to insert. `skills` and `skill_steps`
    must already have been written successfully inside the same
    transaction when that happens, so if they are gone afterwards the
    transaction genuinely rolled back rather than never having started.

    `skill_id` is made deterministic the same way `test_supersession.py`
    already does for `memory_id` (monkeypatching `uuid.uuid4`), not via
    any new production hook.
    """
    import uuid as uuid_module

    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)

    fixed_skill_id = "b" * 32
    monkeypatch.setattr(uuid_module, "uuid4", lambda: uuid_module.UUID(fixed_skill_id))

    connection = sqlite3.connect(cx._db_path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO skill_conditions (skill_id, condition, position) VALUES (?, ?, ?)",
                (fixed_skill_id, "sabotage", 0),
            )
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.promote(lesson, name="n", purpose="p", steps=["s1"], conditions=["only condition"])

    assert cx.skills() == []
    connection = sqlite3.connect(cx._db_path)
    try:
        skills_count = connection.execute(
            "SELECT COUNT(*) FROM skills WHERE skill_id = ?", (fixed_skill_id,)
        ).fetchone()[0]
        steps_count = connection.execute(
            "SELECT COUNT(*) FROM skill_steps WHERE skill_id = ?", (fixed_skill_id,)
        ).fetchone()[0]
        events_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'skill_promoted'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert skills_count == 0
    assert steps_count == 0
    assert events_count == 0


def test_get_skill_resolves_persisted_skill(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    original = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    resolved = cx.get_skill(original.skill_id)

    assert resolved == original


def test_get_skill_rejects_unknown_id(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.get_skill("0" * 32)


def test_skills_lists_in_recorded_order(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    first = cx.promote(lesson, name="first", purpose="p", steps=["s1"])
    second_lesson = cx.learn("Another lesson.")
    second = cx.promote(second_lesson, name="second", purpose="p", steps=["s1"])

    items = cx.skills()

    assert [s.skill_id for s in items] == [first.skill_id, second.skill_id]


def test_skills_on_empty_workspace_returns_empty_list(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    assert cx.skills() == []


def test_skill_persists_across_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    original = cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="p",
        steps=["s1", "s2"],
        conditions=["only for auth refresh"],
    )
    del cx

    reopened = Cortex.open(tmp_path)
    (skill,) = reopened.skills()

    assert skill.skill_id == original.skill_id
    assert skill.name == original.name
    assert skill.steps == original.steps
    assert skill.conditions == original.conditions
    assert skill.verification_state == original.verification_state
    assert skill.source_lesson_id == original.source_lesson_id
    assert skill.evidence_ids == original.evidence_ids


def test_skill_object_is_immutable(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    with pytest.raises(dataclasses.FrozenInstanceError):
        skill.name = "changed"


def test_corrupted_verification_state_is_rejected_explicitly(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("UPDATE skills SET verification_state = 'not-a-real-state'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.get_skill(skill.skill_id)


def test_corrupted_skill_id_is_rejected_explicitly(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = _verified_lesson(cx)
    cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("UPDATE skills SET skill_id = 'not-a-valid-id'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.get_skill("not-a-valid-id")
