"""Tests for A12.1: the explicit support relation.

`Memory.evidence_ids` (generic, possibly-irrelevant provenance) is now
distinct from `Memory.supporting_evidence_ids` (the subset the caller
explicitly designated as supporting THIS memory). Since A12.1, a new
`verified` memory requires at least one item in `supporting_evidence_ids`
of a qualifying kind -- a qualifying-kind Evidence cited only as generic
`evidence` is no longer enough, closing the accidental
false-verification gap A12.0 identified.

This file intentionally also documents, rather than hides, what A12.1
does NOT fix: directionality (a FAILED test explicitly designated as
supporting is still accepted) and semantic relevance (an explicitly
designated but topically unrelated Evidence is still accepted). See the
"CURRENT LIMITATION" tests below.
"""

import pytest

from cortex_memory import Cortex


# ---------------------------------------------------------------------------
# domain model: default field, subset invariant, roundtrip
# ---------------------------------------------------------------------------


def test_memory_supporting_evidence_ids_defaults_to_empty(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("a plain memory with no provenance")

    assert memory.supporting_evidence_ids == ()


def test_supporting_evidence_is_automatically_folded_into_evidence_ids(tmp_path):
    """Supporting implies related: the caller never has to cite the same
    Evidence in both `evidence` and `supporting_evidence`."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    memory = cx.learn(
        "Use only the newly issued refresh token.", supporting_evidence=[validation], verified=True
    )

    assert memory.evidence_ids == (validation.evidence_id,)
    assert memory.supporting_evidence_ids == (validation.evidence_id,)


def test_supporting_evidence_ids_is_always_a_subset_of_evidence_ids(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    context = cx.add_evidence("some contextual note", kind="user_statement")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    memory = cx.learn(
        "Use only the newly issued refresh token.",
        evidence=[context],
        supporting_evidence=[validation],
        verified=True,
    )

    assert set(memory.supporting_evidence_ids).issubset(memory.evidence_ids)
    assert memory.evidence_ids == (context.evidence_id, validation.evidence_id)


def test_supporting_and_generic_evidence_ordering_is_deterministic(tmp_path):
    """[A12.1.1 section 9] `evidence_ids` is the single master order (the
    only order the storage layer's `position` column can reconstruct on
    reload): `evidence`'s own order comes first, supporting-only ids are
    appended afterward, in the order given, without duplicates.
    `supporting_evidence_ids` is NOT kept in the caller's own
    `supporting_evidence` order -- it follows its members' relative
    order WITHIN `evidence_ids` instead, so this in-process result is
    identical, ordering included, to a fresh reload of the same row
    (see `test_supporting_evidence_ordering_survives_reopen` below)."""
    cx = Cortex.init(tmp_path, "dev")
    a = cx.add_evidence("a", kind="user_statement")
    b = cx.add_evidence("b", kind="test_result")
    c = cx.add_evidence("c", kind="test_result")

    memory = cx.remember(
        "ordered evidence",
        epistemic_state="verified",
        evidence=[a, b],
        supporting_evidence=[c, b],
    )

    assert memory.evidence_ids == (a.evidence_id, b.evidence_id, c.evidence_id)
    # caller passed supporting_evidence=[c, b] (c first), but the
    # canonical order follows evidence_ids (b before c)
    assert memory.supporting_evidence_ids == (b.evidence_id, c.evidence_id)


def test_supporting_evidence_ordering_survives_reopen(tmp_path):
    """[A12.1.1 section 9, acceptance-critical] The object `remember()`
    returns and the object a fresh reload produces for the same row must
    have byte-identical `supporting_evidence_ids` ordering -- this was
    NOT true before A12.1.1 (the in-process object preserved the
    caller's raw `supporting_evidence` order, while storage always
    reconstructs order via the single shared `position` column, i.e.
    relative order within `evidence_ids`), an accidental,
    algorithm-dependent divergence rather than a deliberate contract."""
    cx = Cortex.init(tmp_path, "dev")
    a = cx.add_evidence("a", kind="user_statement")
    b = cx.add_evidence("b", kind="test_result")
    c = cx.add_evidence("c", kind="test_result")

    original = cx.remember(
        "ordered evidence, reload check",
        epistemic_state="verified",
        evidence=[a, c],
        supporting_evidence=[b, c],  # caller order deliberately reversed vs. evidence_ids
    )
    del cx

    reopened = Cortex.open(tmp_path)
    (reloaded,) = [m for m in reopened.state() if m.memory_id == original.memory_id]

    assert reloaded.evidence_ids == original.evidence_ids
    assert reloaded.supporting_evidence_ids == original.supporting_evidence_ids


def test_duplicate_supporting_evidence_reference_is_deduplicated(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("checked", kind="test_result")

    memory = cx.learn("a lesson", supporting_evidence=[validation, validation], verified=True)

    assert memory.supporting_evidence_ids == (validation.evidence_id,)
    assert memory.evidence_ids == (validation.evidence_id,)


def _memory_evidence_rows(db_path, memory_id):
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(
            "SELECT evidence_id, role FROM memory_evidence WHERE memory_id = ? ORDER BY position",
            (memory_id,),
        ).fetchall()
    finally:
        connection.close()


def test_overlap_evidence_a_b_supporting_b(tmp_path):
    """[A12.1.1 section 10] `evidence=[A, B], supporting_evidence=[B]`:
    B must appear exactly once in `evidence_ids` (not duplicated because
    it is cited in both pools) and exactly once as a `memory_evidence`
    row, with `role='supporting'` winning over `related` for that id."""
    cx = Cortex.init(tmp_path, "dev")
    a = cx.add_evidence("a", kind="user_statement")
    b = cx.add_evidence("b", kind="test_result")

    memory = cx.remember(
        "overlap case 1", epistemic_state="verified", evidence=[a, b], supporting_evidence=[b]
    )

    assert memory.evidence_ids == (a.evidence_id, b.evidence_id)
    assert memory.supporting_evidence_ids == (b.evidence_id,)

    rows = _memory_evidence_rows(cx._db_path, memory.memory_id)
    assert rows == [(a.evidence_id, "related"), (b.evidence_id, "supporting")]

    reopened = Cortex.open(tmp_path)
    (reloaded,) = [m for m in reopened.state() if m.memory_id == memory.memory_id]
    assert reloaded.evidence_ids == memory.evidence_ids
    assert reloaded.supporting_evidence_ids == memory.supporting_evidence_ids


def test_overlap_evidence_b_supporting_b_b(tmp_path):
    """[A12.1.1 section 10] `evidence=[B], supporting_evidence=[B, B]`:
    duplicated both within `supporting_evidence` itself and against
    `evidence` -- B must still appear exactly once everywhere."""
    cx = Cortex.init(tmp_path, "dev")
    b = cx.add_evidence("b", kind="test_result")

    memory = cx.remember(
        "overlap case 2", epistemic_state="verified", evidence=[b], supporting_evidence=[b, b]
    )

    assert memory.evidence_ids == (b.evidence_id,)
    assert memory.supporting_evidence_ids == (b.evidence_id,)

    rows = _memory_evidence_rows(cx._db_path, memory.memory_id)
    assert rows == [(b.evidence_id, "supporting")]

    reopened = Cortex.open(tmp_path)
    (reloaded,) = [m for m in reopened.state() if m.memory_id == memory.memory_id]
    assert reloaded.evidence_ids == memory.evidence_ids
    assert reloaded.supporting_evidence_ids == memory.supporting_evidence_ids


def test_overlap_evidence_a_c_supporting_b_c(tmp_path):
    """[A12.1.1 section 10] `evidence=[A, C], supporting_evidence=[B, C]`:
    B is supporting-only (folded in via supporting-implies-related), C
    overlaps both pools, A is generic-only. No duplicate rows; master
    order (evidence_ids) determines `supporting_evidence_ids`'s order."""
    cx = Cortex.init(tmp_path, "dev")
    a = cx.add_evidence("a", kind="user_statement")
    b = cx.add_evidence("b", kind="test_result")
    c = cx.add_evidence("c", kind="test_result")

    memory = cx.remember(
        "overlap case 3", epistemic_state="verified", evidence=[a, c], supporting_evidence=[b, c]
    )

    assert memory.evidence_ids == (a.evidence_id, c.evidence_id, b.evidence_id)
    assert memory.supporting_evidence_ids == (c.evidence_id, b.evidence_id)

    rows = _memory_evidence_rows(cx._db_path, memory.memory_id)
    assert rows == [
        (a.evidence_id, "related"),
        (c.evidence_id, "supporting"),
        (b.evidence_id, "supporting"),
    ]

    reopened = Cortex.open(tmp_path)
    (reloaded,) = [m for m in reopened.state() if m.memory_id == memory.memory_id]
    assert reloaded.evidence_ids == memory.evidence_ids
    assert reloaded.supporting_evidence_ids == memory.supporting_evidence_ids


def test_remember_rejects_unknown_supporting_evidence_reference(tmp_path):
    """Same atomicity guarantee `evidence` already has: a fabricated
    supporting Evidence id must reject the whole memory, not persist it
    partially."""
    import datetime as dt

    from cortex_memory._evidence import Evidence

    cx = Cortex.init(tmp_path, "dev")
    fabricated = Evidence(
        evidence_id="b" * 32,
        content="never actually persisted",
        kind="test_result",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.remember(
            "a memory built on fabricated supporting evidence",
            epistemic_state="verified",
            supporting_evidence=[fabricated],
        )

    assert cx.recall("fabricated", include_superseded=True) == []


def test_supporting_evidence_survives_reopen(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    context = cx.add_evidence("context note", kind="user_statement")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    original = cx.learn(
        "Use only the newly issued refresh token.",
        evidence=[context],
        supporting_evidence=[validation],
        verified=True,
    )
    del cx

    reopened = Cortex.open(tmp_path)
    (lesson,) = reopened.state(kind="lesson")

    assert lesson.memory_id == original.memory_id
    assert lesson.evidence_ids == (context.evidence_id, validation.evidence_id)
    assert lesson.supporting_evidence_ids == (validation.evidence_id,)


def test_supporting_evidence_survives_copied_workspace(tmp_path):
    import shutil

    original_root = tmp_path / "original"
    original_root.mkdir()
    cx = Cortex.init(original_root, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    original = cx.learn(
        "Use only the newly issued refresh token.", supporting_evidence=[validation], verified=True
    )
    del cx

    copy_root = tmp_path / "copy"
    shutil.copytree(original_root, copy_root)

    copied = Cortex.open(copy_root)
    (lesson,) = copied.state(kind="lesson")

    assert lesson.memory_id == original.memory_id
    assert lesson.supporting_evidence_ids == (validation.evidence_id,)


# ---------------------------------------------------------------------------
# the new verified gate: generic evidence alone is no longer enough
# ---------------------------------------------------------------------------


def test_generic_qualifying_evidence_alone_no_longer_verifies(tmp_path):
    """[A12.1 core capability] A qualifying-kind Evidence (test_result)
    cited only as generic `evidence` -- never explicitly designated
    supporting -- must not verify a new memory, even though the exact
    same call would have succeeded before A12.1."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.remember(
            "SQLite migrations are safe under process crashes.",
            kind="lesson",
            epistemic_state="verified",
            evidence=[validation],
        )


def test_supporting_qualifying_evidence_verifies(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    memory = cx.remember(
        "SQLite migrations are safe under process crashes.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[validation],
    )

    assert memory.epistemic_state == "verified"


def test_verified_requires_supporting_evidence_even_if_generic_evidence_exists(tmp_path):
    """[A12.1 section 6, dogfood case] `evidence_ids=(test_id,),
    supporting_evidence_ids=()` must be REJECTED for `verified`, even
    though `evidence_ids` alone contains a qualifying kind."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.remember(
            "a claim",
            epistemic_state="verified",
            evidence=[validation],
            supporting_evidence=(),
        )


def test_non_qualifying_supporting_evidence_does_not_verify(tmp_path):
    """[A12.1 section 21] The qualifying-kind check applies to the
    SUPPORTING pool specifically -- a generic qualifying Evidence cannot
    substitute for a non-qualifying supporting one."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    opinion = cx.add_evidence("I think this works.", kind="user_statement")

    with pytest.raises(ValueError):
        cx.remember(
            "a claim",
            epistemic_state="verified",
            evidence=[validation],
            supporting_evidence=[opinion],
        )


def test_non_qualifying_supporting_evidence_is_still_allowed_on_non_verified_memory(tmp_path):
    """[A12.1 section 21] Support and verification-qualification are
    distinct concepts: a caller may designate a non-qualifying Evidence
    as supporting a candidate (non-verified) memory -- 'supporting'
    means 'the caller asserts this backs the claim', not 'this
    Evidence, alone, would satisfy the verified gate'."""
    cx = Cortex.init(tmp_path, "dev")
    opinion = cx.add_evidence("I think this works.", kind="user_statement")

    memory = cx.remember(
        "a candidate claim",
        supporting_evidence=[opinion],
    )

    assert memory.epistemic_state == "user_asserted"
    assert memory.supporting_evidence_ids == (opinion.evidence_id,)


def test_qualifying_generic_and_non_qualifying_supporting_does_not_verify(tmp_path):
    """The inverse of the accepted case: a qualifying Evidence sitting in
    generic `evidence` cannot rescue a non-qualifying `supporting_evidence`."""
    cx = Cortex.init(tmp_path, "dev")
    reference = cx.add_evidence("src/auth/refresh.py", kind="file_reference")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.remember(
            "a claim",
            epistemic_state="verified",
            evidence=[validation],
            supporting_evidence=[reference],
        )


def test_learn_verified_uses_the_same_gate_as_remember(tmp_path):
    """`learn()` delegates to `remember()`; no second verification gate."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.learn("a lesson", evidence=[validation], verified=True)


# ---------------------------------------------------------------------------
# downstream: Preflight and Skill benefit without any structural change
# ---------------------------------------------------------------------------


def test_preflight_no_longer_surfaces_a_generically_related_false_verification(tmp_path):
    """[A12.0 probe 1 + probe 5, closed] The exact false-verification
    scenario A12.0 demonstrated empirically can no longer be constructed
    at all: the `remember()` call itself is rejected, so there is no
    falsely-verified lesson left for `preflight()` to surface."""
    cx = Cortex.init(tmp_path, "dev")
    css_test = cx.add_evidence("CSS mobile layout tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.remember(
            "SQLite migrations are safe under process crashes.",
            kind="lesson",
            epistemic_state="verified",
            evidence=[css_test],
        )

    result = cx.preflight("SQLite migrations are safe under process crashes")
    assert result.verified_lessons == ()


def test_skill_promoted_from_properly_supported_lesson_is_verified(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    lesson = cx.learn(
        "Use only the newly issued refresh token.", supporting_evidence=[validation], verified=True
    )

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    assert skill.verification_state == "verified"


def test_skill_authority_is_fully_traceable_to_the_supporting_evidence(tmp_path):
    """[A12.1.1 section 11] Skill deliberately does NOT get its own
    `supporting_evidence_ids` field (A12.1 section 26) -- but an auditor
    must still be able to walk the full chain: Skill -> `source_lesson_id`
    -> the CANONICAL source Lesson (fetched fresh, not trusted from any
    forged/stale object -- see `test_promote_ignores_forged_*` in
    test_skill.py) -> that Lesson's own `supporting_evidence_ids`,
    landing exactly on the Evidence that justified the verification in
    the first place. If this chain were broken, a Skill's `verified`
    label would be authority with no traceable provenance."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    lesson = cx.learn(
        "Use only the newly issued refresh token.", supporting_evidence=[validation], verified=True
    )

    skill = cx.promote(lesson, name="n", purpose="p", steps=["s1"])

    (canonical_lesson,) = [m for m in cx.state(kind="lesson") if m.memory_id == skill.source_lesson_id]
    assert canonical_lesson.supporting_evidence_ids == (validation.evidence_id,)
    assert skill.evidence_ids == canonical_lesson.evidence_ids


def test_skill_cannot_inherit_false_authority_via_generic_evidence_path(tmp_path):
    """[A12.1 section 24/A12.0 section 16] The Skill-propagation risk
    A12.0 flagged as the highest-severity path is closed at the source:
    a Lesson can no longer become falsely `verified` from generically
    related qualifying Evidence, so no Skill promoted from it can either."""
    cx = Cortex.init(tmp_path, "dev")
    unrelated_test = cx.add_evidence("CSS mobile layout tests passed.", kind="test_result")

    with pytest.raises(ValueError):
        cx.learn(
            "SQLite migrations are safe under process crashes.",
            evidence=[unrelated_test],
            verified=True,
        )

    # no falsely-verified lesson exists to promote from in the first place
    assert cx.state(kind="lesson") == []


# ---------------------------------------------------------------------------
# CURRENT LIMITATION: A12.1 does not judge directionality or relevance
# ---------------------------------------------------------------------------


def test_current_limitation_directionality_failed_test_can_still_be_designated_supporting(tmp_path):
    """[A12.0 section 9, A12.1 section 9 -- documented, not hidden]
    Cortex does not parse Evidence content for PASS/FAIL or any other
    keyword. A FAILED test explicitly designated as supporting a
    positive claim is still accepted: A12.1 requires an explicit
    assertion, it does not validate that assertion's truth."""
    cx = Cortex.init(tmp_path, "dev")
    failed_test = cx.add_evidence("Migration atomicity test FAILED.", kind="test_result")

    memory = cx.remember(
        "Migration atomicity is safe.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[failed_test],
    )

    assert memory.epistemic_state == "verified"


def test_current_limitation_negative_user_confirmation_can_still_verify_a_positive_claim(tmp_path):
    """[A12.0 section 9] Same principle for `user_confirmation`: content
    negativity is not interpreted."""
    cx = Cortex.init(tmp_path, "dev")
    denial = cx.add_evidence("I confirm this is NOT correct.", kind="user_confirmation")

    memory = cx.remember(
        "The login flow works correctly.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[denial],
    )

    assert memory.epistemic_state == "verified"


def test_current_limitation_topically_irrelevant_evidence_can_still_be_designated_supporting(tmp_path):
    """[A12.0 section 10, A12.1 section 10] Explicit support is not
    semantic relevance: once the caller deliberately designates Evidence
    as supporting, Cortex does not judge whether it is actually about
    the same topic as the claim."""
    cx = Cortex.init(tmp_path, "dev")
    css_test = cx.add_evidence("CSS mobile layout tests passed.", kind="test_result")

    memory = cx.remember(
        "SQLite migrations are safe under process crashes.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[css_test],
    )

    assert memory.epistemic_state == "verified"


def test_current_limitation_same_evidence_can_support_contradictory_claims(tmp_path):
    """[A12.0 section 15/39, out of scope for A12] Conflict/non-
    contradiction detection is explicitly not part of this tracer."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("tests passed: 12/12", kind="test_result")

    safe = cx.remember(
        "Migration strategy is safe.", kind="lesson", epistemic_state="verified", supporting_evidence=[validation]
    )
    unsafe = cx.remember(
        "Migration strategy is unsafe.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[validation],
    )

    assert safe.epistemic_state == "verified"
    assert unsafe.epistemic_state == "verified"


# ---------------------------------------------------------------------------
# A12.1.1: write-boundary hardening -- the verified contract must not
# depend solely on Cortex.remember() being the only call path that ever
# reaches MemoryStore.add()
# ---------------------------------------------------------------------------


def test_store_add_rejects_verified_memory_with_no_supporting_evidence(tmp_path):
    """[A12.1.1] A `Memory` constructed directly with
    `epistemic_state="verified"` and no supporting evidence at all, then
    persisted via `MemoryStore.add()` directly (bypassing
    `Cortex.remember()`'s gate entirely), must still be rejected. The
    canonical write boundary is the backstop, not just the orchestrator
    method -- a future internal call path (a batch importer, a repair
    tool) that constructs a `Memory` and calls `add()` directly must not
    be able to silently produce a falsely-verified record."""
    import datetime as dt
    import uuid

    from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED, Event
    from cortex_memory._memory import Memory
    from cortex_memory._store import MemoryStore

    cx = Cortex.init(tmp_path, "dev")
    memory_id = uuid.uuid4().hex
    now = dt.datetime.now(dt.timezone.utc)
    forged = Memory(
        memory_id=memory_id,
        content="Bypassed remember() entirely.",
        kind="lesson",
        epistemic_state="verified",
        recorded_at=now,
        evidence_ids=(),
        supporting_evidence_ids=(),
    )
    event = Event(event_id=uuid.uuid4().hex, kind=EVENT_KIND_MEMORY_RECORDED, subject_id=memory_id, occurred_at=now)

    with pytest.raises(ValueError):
        with MemoryStore.create_or_open(cx._db_path) as store:
            store.add(forged, [event])

    assert cx.state(kind="lesson") == []


def test_store_add_rejects_verified_memory_with_only_non_qualifying_supporting_evidence(tmp_path):
    """[A12.1.1] Same write-boundary defense, for the non-qualifying-kind
    case: `MemoryStore.add()` must reuse the exact same
    `VERIFICATION_EVIDENCE_KINDS` rule `remember()` uses, not a
    divergent or weaker one."""
    import datetime as dt
    import uuid

    from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED, Event
    from cortex_memory._memory import Memory
    from cortex_memory._store import MemoryStore

    cx = Cortex.init(tmp_path, "dev")
    opinion = cx.add_evidence("I think this works.", kind="user_statement")
    memory_id = uuid.uuid4().hex
    now = dt.datetime.now(dt.timezone.utc)
    forged = Memory(
        memory_id=memory_id,
        content="Bypassed remember() with a non-qualifying supporting Evidence.",
        kind="lesson",
        epistemic_state="verified",
        recorded_at=now,
        evidence_ids=(opinion.evidence_id,),
        supporting_evidence_ids=(opinion.evidence_id,),
    )
    event = Event(event_id=uuid.uuid4().hex, kind=EVENT_KIND_MEMORY_RECORDED, subject_id=memory_id, occurred_at=now)

    with pytest.raises(ValueError):
        with MemoryStore.create_or_open(cx._db_path) as store:
            store.add(forged, [event])


def test_store_add_still_accepts_legitimate_verified_memory_built_by_hand(tmp_path):
    """Sanity complement: the write-boundary check must not reject a
    correctly-formed verified `Memory`, only ones missing qualifying
    supporting evidence -- confirming this is the same rule as
    `remember()`'s, not a stricter, divergent one."""
    import datetime as dt
    import uuid

    from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED, Event
    from cortex_memory._memory import Memory
    from cortex_memory._store import MemoryStore

    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    memory_id = uuid.uuid4().hex
    now = dt.datetime.now(dt.timezone.utc)
    well_formed = Memory(
        memory_id=memory_id,
        content="A properly supported verified memory.",
        kind="lesson",
        epistemic_state="verified",
        recorded_at=now,
        evidence_ids=(validation.evidence_id,),
        supporting_evidence_ids=(validation.evidence_id,),
    )
    event = Event(event_id=uuid.uuid4().hex, kind=EVENT_KIND_MEMORY_RECORDED, subject_id=memory_id, occurred_at=now)

    with MemoryStore.create_or_open(cx._db_path) as store:
        store.add(well_formed, [event])

    (lesson,) = [m for m in cx.state(kind="lesson") if m.memory_id == memory_id]
    assert lesson.epistemic_state == "verified"


def test_store_add_write_check_does_not_apply_to_reads_of_legacy_data(tmp_path):
    """[A12.1.1 write vs read distinction] The write-boundary check lives
    only in `add()`. It must never run on read paths, or a v4-migrated
    verified memory with empty `supporting_evidence_ids` (A12.1
    grandfathering) would break on every read. This is exercised more
    thoroughly in `test_migration_v5.py`; this test only pins the
    principle at the unit level using the public API."""
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("a lesson", supporting_evidence=[validation], verified=True)

    # reading it back repeatedly must never re-trigger the write check
    for _ in range(3):
        (reloaded,) = [m for m in cx.state(kind="lesson") if m.memory_id == lesson.memory_id]
        assert reloaded.epistemic_state == "verified"
