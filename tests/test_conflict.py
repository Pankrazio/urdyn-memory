"""Tests for the A13.1 canonical `Conflict` relation.

A `Conflict` is a symmetric, explicit, structural statement that two
Memories cannot both be treated as a coherent description of the same
state. It is NOT a judgment: Urdyn never chooses a side, never mutates
either Memory, never changes an `epistemic_state`, and never implies
invalidation or supersession (see `_conflict.py`'s module docstring).

Semantics under test throughout this file:

    CONFLICT != INVALIDATION / SUPERSESSION / DISPROVEN

`open_conflicts()` is a DERIVED projection (both participants current),
never a stored status. `conflicts()` is the full, append-only canonical
history, never rewritten by a later resolution.
"""

import datetime as dt
import sqlite3

import pytest

import legacy_brains as legacy
from urdyn import Conflict, Urdyn


# -- 1. basic recording ------------------------------------------------------


def test_record_basic_conflict(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    conflict = cx.record_conflict(a, b)

    assert isinstance(conflict, Conflict)
    assert set(conflict.memory_ids) == {a.memory_id, b.memory_id}
    assert isinstance(conflict.recorded_at, dt.datetime)


# -- 2. symmetric canonical ordering -----------------------------------------


def test_conflict_memory_ids_are_canonically_ordered(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    forward = cx.record_conflict(a, b)
    lo, hi = sorted([a.memory_id, b.memory_id])

    assert forward.memory_ids == (lo, hi)


# -- 3. self-conflict rejected ------------------------------------------------


def test_self_conflict_is_rejected(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")

    with pytest.raises(ValueError):
        cx.record_conflict(a, a)


# -- 4/5/6. duplicate / reverse-duplicate idempotency -------------------------


def test_duplicate_declaration_is_idempotent(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    first = cx.record_conflict(a, b)
    second = cx.record_conflict(a, b)

    assert first == second
    assert len(cx.conflicts()) == 1


def test_reverse_duplicate_declaration_is_idempotent(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    forward = cx.record_conflict(a, b)
    reverse = cx.record_conflict(b, a)

    assert forward == reverse
    assert len(cx.conflicts()) == 1


def test_duplicate_declaration_does_not_change_recorded_at(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    first = cx.record_conflict(a, b)
    second = cx.record_conflict(a, b)
    third = cx.record_conflict(b, a)

    assert first.recorded_at == second.recorded_at == third.recorded_at


# -- 7/8/9/10. epistemic interaction ------------------------------------------


def test_verified_vs_verified_conflict_is_allowed(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence_a = cx.add_evidence("Test suite passed under strategy A.", kind="test_result")
    evidence_b = cx.add_evidence("Test suite failed under strategy A.", kind="test_result")
    a = cx.learn("Migration strategy A is safe.", supporting_evidence=[evidence_a], verified=True)
    b = cx.learn("Migration strategy A is unsafe.", supporting_evidence=[evidence_b], verified=True)

    conflict = cx.record_conflict(a, b)

    assert set(conflict.memory_ids) == {a.memory_id, b.memory_id}


def test_both_memories_remain_current_after_conflict(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    cx.record_conflict(a, b)

    current_ids = {m.memory_id for m in cx.state()}
    assert a.memory_id in current_ids
    assert b.memory_id in current_ids


def test_epistemic_states_are_unchanged_by_conflict(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("Checked directly.", kind="test_result")
    a = cx.learn("Migration strategy A is safe.", supporting_evidence=[evidence], verified=True)
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    cx.record_conflict(a, b)

    (reloaded_a,) = [m for m in cx.timeline() if m.memory_id == a.memory_id]
    (reloaded_b,) = [m for m in cx.timeline() if m.memory_id == b.memory_id]
    assert reloaded_a.epistemic_state == "verified"
    assert reloaded_b.epistemic_state == "user_asserted"


def test_user_asserted_vs_verified_conflict_is_allowed(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("Checked directly.", kind="test_result")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.learn("Migration strategy A is unsafe.", supporting_evidence=[evidence], verified=True)

    conflict = cx.record_conflict(a, b)

    assert set(conflict.memory_ids) == {a.memory_id, b.memory_id}


# -- 11. cross-kind ------------------------------------------------------------


def test_cross_kind_conflict_is_allowed(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    decision = cx.remember("Use SQLite.", kind="decision")
    invariant = cx.remember("Runtime must support concurrent multi-writer access.", kind="invariant")

    conflict = cx.record_conflict(decision, invariant)

    assert set(conflict.memory_ids) == {decision.memory_id, invariant.memory_id}


# -- 12/13. current vs historical ----------------------------------------------


def test_conflict_with_non_current_participant_is_accepted(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember("Python 3.13 is required.", kind="environment", supersedes=a.memory_id)
    b = cx.remember("Node 18 is required.", kind="environment")

    conflict = cx.record_conflict(a, b)

    assert set(conflict.memory_ids) == {a.memory_id, b.memory_id}


def test_historical_conflict_is_excluded_from_open(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember("Python 3.13 is required.", kind="environment", supersedes=a.memory_id)
    b = cx.remember("Node 18 is required.", kind="environment")

    cx.record_conflict(a, b)

    assert len(cx.conflicts()) == 1
    assert cx.open_conflicts() == []


# -- 14/15. resolution via supersession / invalidation -------------------------


def test_supersession_closes_open_projection_but_preserves_history(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    cx.record_conflict(a, b)
    assert len(cx.open_conflicts()) == 1

    cx.remember("Migration strategy A was replaced by strategy B.", kind="decision", supersedes=b.memory_id)

    assert len(cx.conflicts()) == 1
    assert cx.open_conflicts() == []


def test_invalidation_closes_open_projection_but_preserves_history(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    cx.record_conflict(a, b)
    assert len(cx.open_conflicts()) == 1

    cx.remember(
        "The claim that strategy A is unsafe is no longer trusted.",
        kind="invalidation",
        supersedes=b.memory_id,
    )

    assert len(cx.conflicts()) == 1
    assert cx.open_conflicts() == []


# -- 16. canonical history preserved -------------------------------------------


def test_conflict_history_survives_resolution_of_both_sides(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    original = cx.record_conflict(a, b)

    cx.remember("A replacement decision.", kind="decision", supersedes=a.memory_id)
    cx.remember("Another replacement decision.", kind="decision", supersedes=b.memory_id)

    (preserved,) = cx.conflicts()
    assert preserved == original
    assert cx.open_conflicts() == []


# -- 17. reopen ------------------------------------------------------------------


def test_conflict_survives_reopening(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    original = cx.record_conflict(a, b)
    del cx

    reopened = Urdyn.open(tmp_path)

    assert reopened.conflicts() == [original]
    assert reopened.open_conflicts() == [original]


# -- 18. copied workspace ---------------------------------------------------------


def test_conflict_survives_a_copied_workspace(tmp_path):
    import shutil

    source = tmp_path / "source"
    cx = Urdyn.init(source, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    cx.record_conflict(a, b)
    del cx

    destination = tmp_path / "copy"
    shutil.copytree(source / ".urdyn", destination / ".urdyn")

    copied = Urdyn.open(destination)
    assert len(copied.conflicts()) == 1
    assert len(copied.open_conflicts()) == 1


# -- 19. deterministic ordering ---------------------------------------------------


def test_conflicts_are_ordered_deterministically_by_recorded_at(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("A", kind="note")
    b = cx.remember("B", kind="note")
    c = cx.remember("C", kind="note")
    d = cx.remember("D", kind="note")

    first = cx.record_conflict(a, b)
    second = cx.record_conflict(c, d)

    ordered = cx.conflicts()
    assert ordered[0] == first
    assert ordered[1] == second


# -- 20/21. unknown / forged Memory --------------------------------------------


def test_conflict_with_unknown_memory_id_is_rejected(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")

    from urdyn import Memory

    forged = Memory(
        memory_id="0" * 32,
        content="never actually persisted",
        kind="decision",
        epistemic_state="user_asserted",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.record_conflict(a, forged)


def test_forged_memory_object_cannot_bypass_canonical_lookup(tmp_path):
    """A `Memory` object sharing a REAL id but disagreeing with the
    canonical fields (a stale/forged copy) must not be able to conjure a
    conflict with a memory that never existed under a different id. Only
    `memory_id` is trusted -- the same discipline `promote()` already
    applies to `lesson.memory_id`."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")

    from urdyn import Memory

    forged_but_nonexistent_id = Memory(
        memory_id="f" * 32,
        content="forged content, never persisted under this id",
        kind="decision",
        epistemic_state="verified",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.record_conflict(a, forged_but_nonexistent_id)

    assert cx.conflicts() == []


# -- 22-26. migration v5->v6 -----------------------------------------------------


def _v5(tmp_path):
    root, generated = legacy.cortex_brain(tmp_path, 5)
    return root, generated, legacy.db_path(root)


def test_v5_brain_is_refused_and_its_memories_survive_exactly(tmp_path):
    """Was `test_v5_database_migrates_to_v6_and_memory_survives` (its
    hand-written v5 builder only ever wrote into the v7 store `Urdyn.init`
    creates). On the AUTHENTIC v5 Brain: refused, every byte and memory kept."""
    from urdyn import UrdynStorageSchemaError

    root, generated, db = _v5(tmp_path)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")
    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == 5 and facts["integrity"] == "ok"
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


def test_v5_refusal_creates_no_memory_conflicts_table(tmp_path):
    """Was `test_v5_migration_reaches_current_schema_and_adds_memory_conflicts_table`."""
    from urdyn import UrdynStorageSchemaError, maintenance

    root, _, db = _v5(tmp_path)
    for entry in (lambda: Urdyn.open(root).recall("synthetic"), lambda: Urdyn.init(root, "general"),
                  lambda: maintenance.recover_search_index(root)):
        with pytest.raises(UrdynStorageSchemaError):
            entry()
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 5
    assert "memory_conflicts" not in facts["tables"]


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_memories_recorded_by_a_release_can_participate_in_a_new_conflict(tmp_path, release):
    """Was `test_v5_migrated_memories_can_participate_in_a_new_conflict`.
    The valid part -- memories an EARLIER build recorded can be put in a new
    Conflict by this one, which is then open and persistent, without touching
    the memories -- on the AUTHENTIC notes written by each release."""
    root, generated = legacy.release_experience_brain(tmp_path, release)
    cx = Urdyn.open(root)
    notes = {m.memory_id: m for m in cx.timeline() if m.memory_id in generated["notes"]}
    assert set(notes) == set(generated["notes"])
    stored = sorted((m.memory_id, m.content, m.epistemic_state) for m in notes.values())
    a, b = (notes[i] for i in generated["notes"])

    conflict = cx.record_conflict(a, b)

    assert set(conflict.memory_ids) == set(generated["notes"])
    reopened = Urdyn.open(root)
    assert [set(c.memory_ids) for c in reopened.open_conflicts()] == [set(generated["notes"])]
    assert sorted((m.memory_id, m.content, m.epistemic_state)
                  for m in reopened.timeline() if m.memory_id in notes) == stored


def test_v5_refusal_is_repeatable(tmp_path):
    """Was `test_v5_migration_is_repeatable`."""
    from urdyn import UrdynStorageSchemaError

    root, _, _ = _v5(tmp_path)
    before = legacy.snapshot(root)
    messages = set()
    for _ in range(3):
        with pytest.raises(UrdynStorageSchemaError) as refused:
            Urdyn.open(root).recall("synthetic")
        messages.add(str(refused.value))
    assert len(messages) == 1
    assert legacy.snapshot(root) == before


def test_v5_sabotaged_store_fails_closed_untouched(tmp_path):
    """Was `test_v5_migration_does_not_destroy_data_on_failure`: a
    colliding `memory_conflicts` made the v5->v6 step fail and roll back
    (still v5, row intact). On the AUTHENTIC v5 Brain the store is refused
    before any step: still v5, memories intact, the sabotage table untouched,
    every byte kept."""
    from urdyn import UrdynStorageError

    root, generated = legacy.cortex_brain(tmp_path, 5)
    db = legacy.db_path(root)
    legacy.sabotage(db, "CREATE TABLE memory_conflicts (bogus_column TEXT)")
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageError):
        Urdyn.open(root).recall("must survive")

    assert legacy.snapshot(root) == before
    facts = legacy.store_facts(db)
    assert facts["user_version"] == 5
    assert {row[0] for row in facts["memories"]} == set(generated["written"])
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert [r[1] for r in connection.execute("PRAGMA table_info(memory_conflicts)")] == ["bogus_column"]
    finally:
        connection.close()


@pytest.mark.parametrize("version", legacy.CORTEX_VERSIONS)
def test_no_cortex_version_reaches_the_current_chain(tmp_path, version):
    """Was `test_v1_to_current_full_migration_chain_still_works`
    (implicit v1->v7, `memory_conflicts` created on the way). On every
    AUTHENTIC Cortex Brain v1-v6 the open is refused: the version, the tables
    (`memory_conflicts` present only if the historical v6 code created it) and
    the memories stay exactly as written."""
    from urdyn import UrdynStorageSchemaError

    root, generated = legacy.cortex_brain(tmp_path, version)
    db = legacy.db_path(root)
    facts = legacy.store_facts(db)
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root).recall("synthetic")

    assert legacy.snapshot(root) == before
    assert legacy.store_facts(db) == facts
    assert facts["user_version"] == version
    assert ("memory_conflicts" in facts["tables"]) == (version == 6)
    assert {row[0] for row in facts["memories"]} == set(generated["written"])


# -- 27. no Event emitted for a conflict declaration -----------------------------


def test_no_event_is_emitted_by_conflict_declaration(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    connection = sqlite3.connect(cx._db_path)
    try:
        (before,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    finally:
        connection.close()

    cx.record_conflict(a, b)

    connection = sqlite3.connect(cx._db_path)
    try:
        (after,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    finally:
        connection.close()

    assert after == before


# -- 28. no mutation of Memory objects -------------------------------------------


def test_memory_objects_are_unchanged_by_conflict_declaration(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    a_before, b_before = a, b

    cx.record_conflict(a, b)

    assert a == a_before
    assert b == b_before
    (reloaded_a,) = [m for m in cx.timeline() if m.memory_id == a.memory_id]
    (reloaded_b,) = [m for m in cx.timeline() if m.memory_id == b.memory_id]
    assert reloaded_a.supersedes is None
    assert reloaded_b.supersedes is None


# -- storage/schema surface -------------------------------------------------------


def test_store_schema_version_is_7(tmp_path):
    from urdyn._store import STORE_SCHEMA_VERSION

    assert STORE_SCHEMA_VERSION == 7


# -- malformed v6 schema (A12.1.1-style integrity discipline) --------------------


def test_user_version_6_with_missing_memory_conflicts_table_raises_cleanly(tmp_path):
    """`PRAGMA user_version == 6` alone must not be trusted as proof the
    schema is actually shaped like v6: if `memory_conflicts` is missing
    (a corrupted or incomplete upgrade), opening the store fails loudly as
    `UrdynStorageError` -- no auto-repair, no silent recreation, same
    standard already applied to every other version in `_ensure_schema`.

    Under the storage contract (`docs/storage-contract.md`: `open`
    requires a valid canonical DB and never creates tables) the refusal
    comes earlier, at `Urdyn.open` itself, as `UrdynStorageCorruptError`;
    a handle opened before the damage refuses its reads the same way.
    Nothing is repaired: the table stays missing and the bytes are
    unchanged."""
    from urdyn import UrdynStorageCorruptError

    cx = Urdyn.init(tmp_path, "dev")
    original = cx.remember("seed content that must survive", kind="note")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("DROP TABLE memory_conflicts")
        connection.commit()
    finally:
        connection.close()
    damaged = cx._db_path.read_bytes()

    with pytest.raises(UrdynStorageCorruptError):
        Urdyn.open(tmp_path)
    with pytest.raises(UrdynStorageCorruptError):
        cx.recall("seed")

    assert cx._db_path.read_bytes() == damaged
    connection = sqlite3.connect(cx._db_path)
    try:
        row = connection.execute(
            "SELECT memory_id, content FROM memories WHERE memory_id = ?", (original.memory_id,)
        ).fetchone()
        recreated = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'memory_conflicts'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert row == (original.memory_id, "seed content that must survive")
    assert recreated == 0


def test_orphan_conflict_raises_cleanly_instead_of_being_returned(tmp_path):
    """[A13.1.1] A `memory_conflicts` row whose ids are well-FORMED (32
    hex, canonically ordered) but name a memory the store does not hold
    must not be returned as a canonical `Conflict`. Returning it would
    assert a relation between two recorded Memories when one was never
    recorded -- and `open_conflicts()` would then silently drop it (a
    nonexistent id is never current), making corruption look exactly
    like the legitimate "no longer open" answer."""
    from urdyn import UrdynStorageError

    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    cx.record_conflict(a, b)

    ghost_id = "c" * 32
    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("DELETE FROM memory_conflicts")
        lo, hi = sorted([a.memory_id, ghost_id])
        connection.execute(
            "INSERT INTO memory_conflicts (memory_id_a, memory_id_b, recorded_at) VALUES (?, ?, ?)",
            (lo, hi, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.conflicts()
    with pytest.raises(UrdynStorageError):
        cx.open_conflicts()


def test_storage_rejects_a_non_canonically_ordered_pair(tmp_path):
    """[A13.1.1] Defence in depth: the canonical pair invariant lives on
    the table itself (`CHECK (memory_id_a < memory_id_b)`), not only in
    the Python write path, so a reversed pair cannot exist in the store
    even if written by something other than `record_conflict()`."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    lo, hi = sorted([a.memory_id, b.memory_id])

    connection = sqlite3.connect(cx._db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO memory_conflicts (memory_id_a, memory_id_b, recorded_at) VALUES (?, ?, ?)",
                (hi, lo, dt.datetime.now(dt.timezone.utc).isoformat()),
            )
    finally:
        connection.close()


def test_storage_rejects_a_self_referential_pair(tmp_path):
    """[A13.1.1] `a == a` fails `a < b` too, so self-conflict is
    impossible at the storage level as well as in `record_conflict()`."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")

    connection = sqlite3.connect(cx._db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO memory_conflicts (memory_id_a, memory_id_b, recorded_at) VALUES (?, ?, ?)",
                (a.memory_id, a.memory_id, dt.datetime.now(dt.timezone.utc).isoformat()),
            )
    finally:
        connection.close()


def test_idempotency_does_not_silence_a_non_duplicate_constraint_violation(tmp_path):
    """[A13.1.1] The idempotent INSERT must absorb ONLY a duplicate of
    the canonical pair, never any other constraint failure. Simulates a
    future internal call path that reaches SQL without canonicalizing
    (by making `canonical_pair` return a reversed pair): that violates
    the table's CHECK, and must surface as a clean `UrdynStorageError`
    rather than being swallowed the way a blanket `INSERT OR IGNORE`
    would swallow it."""
    from urdyn import UrdynStorageError
    from urdyn import _store as store_module

    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    def _reversed_pair(x, y):
        lo, hi = sorted([x, y])
        return (hi, lo)  # deliberately non-canonical

    original = store_module.canonical_pair
    store_module.canonical_pair = _reversed_pair
    try:
        with pytest.raises(UrdynStorageError):
            cx.record_conflict(a, b)
    finally:
        store_module.canonical_pair = original

    # nothing was persisted by the rejected write
    assert cx.conflicts() == []


# -- recorded_at contract (A13.1.1 point 2) --------------------------------------


def test_conflict_recorded_at_matches_the_canonical_temporal_type(tmp_path):
    """[A13.1.1] `Conflict.recorded_at` must be the same public temporal
    representation every other canonical Urdyn primitive uses -- a
    timezone-aware `datetime`, not an ISO string or a naive datetime.
    Compared here against a real `Memory` rather than asserted in the
    abstract, so this test fails if EITHER model drifts."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    conflict = cx.record_conflict(a, b)

    assert type(conflict.recorded_at) is type(a.recorded_at)
    assert isinstance(conflict.recorded_at, dt.datetime)
    assert conflict.recorded_at.tzinfo is not None
    assert a.recorded_at.tzinfo is not None


def test_conflict_recorded_at_round_trips_exactly_through_storage(tmp_path):
    """Record -> reopen -> equality: the reloaded `Conflict` must be
    byte-identical to the one returned at write time, `recorded_at`
    included (no precision loss, no tz drift), so `==` is a reliable
    identity check across processes."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    original = cx.record_conflict(a, b)
    del cx

    (reloaded,) = Urdyn.open(tmp_path).conflicts()

    assert reloaded == original
    assert reloaded.recorded_at == original.recorded_at
    assert reloaded.recorded_at.tzinfo is not None


def test_conflict_recorded_at_supports_ordering_against_memory_timestamps(tmp_path):
    """`recorded_at` is directly comparable with other canonical
    primitives' timestamps -- it would not be if Conflict had introduced
    a different temporal type or a naive datetime."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")

    conflict = cx.record_conflict(a, b)

    assert conflict.recorded_at >= a.recorded_at
    assert conflict.recorded_at >= b.recorded_at


def test_duplicate_recorded_at_is_stable_across_a_reopen(tmp_path):
    """[A13.1.1 point 8] The idempotency contract must hold across
    processes, not just within one: a reversed re-declaration issued by a
    freshly reopened workspace returns the ORIGINAL `recorded_at` and
    still leaves exactly one canonical row."""
    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    original = cx.record_conflict(a, b)
    del cx

    reopened = Urdyn.open(tmp_path)
    (reloaded_a,) = [m for m in reopened.timeline() if m.memory_id == a.memory_id]
    (reloaded_b,) = [m for m in reopened.timeline() if m.memory_id == b.memory_id]
    again = reopened.record_conflict(reloaded_b, reloaded_a)  # reversed, new process

    assert again == original
    assert again.recorded_at == original.recorded_at
    assert len(reopened.conflicts()) == 1


def test_corrupted_conflict_pair_ordering_raises_cleanly(tmp_path):
    """A `memory_conflicts` row with `memory_id_a >= memory_id_b` must not
    be silently accepted as a valid, differently-ordered pair -- that
    would break the deduplication/idempotency guarantee the ordering
    exists to provide. It must raise `UrdynStorageError`.

    [A13.1.1] The READ path enforces this independently of the table's
    own `CHECK (memory_id_a < memory_id_b)`, so the sabotage here first
    recreates `memory_conflicts` WITHOUT that constraint -- the shape a
    store hand-built or touched by an external tool could legitimately
    have. `test_storage_rejects_a_non_canonically_ordered_pair` covers
    the storage layer; this covers the reconstruction layer. Both must
    hold on their own: that is what makes it defence in depth rather
    than one check written twice."""
    from urdyn import UrdynStorageError

    cx = Urdyn.init(tmp_path, "dev")
    a = cx.remember("Migration strategy A is safe.", kind="decision")
    b = cx.remember("Migration strategy A is unsafe.", kind="decision")
    cx.record_conflict(a, b)

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("DROP TABLE memory_conflicts")
        connection.execute(
            "CREATE TABLE memory_conflicts (memory_id_a TEXT NOT NULL, memory_id_b TEXT NOT NULL, "
            "recorded_at TEXT NOT NULL, PRIMARY KEY (memory_id_a, memory_id_b))"
        )
        lo, hi = sorted([a.memory_id, b.memory_id])
        connection.execute(
            "INSERT INTO memory_conflicts (memory_id_a, memory_id_b, recorded_at) VALUES (?, ?, ?)",
            (hi, lo, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.conflicts()
