"""End-to-end tests for the A10.1 explicit-capture contract:

    explicit failure Evidence
        -> failed Attempt (linked)
    explicit verification Evidence
        -> succeeded Attempt (linked)
        -> verified Lesson (backed by the verification Evidence)
    process ends
        -> new process
        -> preflight(task) recovers known failure + verified lesson
           + recommended validation, with provenance intact

This is deliberately NOT a new primitive: every step below is composed
from public API that already existed before A10 (`add_evidence`,
`record_attempt`, `learn`, `preflight`, `get_evidence`). A10.1's finding
is that "capture" -- explicit, deliberate creation of canonical Evidence
that later supports an Attempt/Lesson -- was already fully supported by
that composition; these tests freeze it as a stable, tested contract
rather than adding any new surface.

The remaining tests here freeze three boundaries A10.0 identified as
load-bearing for that claim, none of which A10.1 is allowed to weaken:
Evidence alone is not knowledge (never surfaced by `preflight()` on its
own), Evidence is not part of the append-only Event history, and Evidence
is not part of the FTS/semantic retrieval surface -- only Attempt/Memory/
Skill are. `Attempt`/`Memory`/`Skill` remain the retrieval surface;
Evidence is supporting material resolved by id from their `evidence_ids`.
"""

from __future__ import annotations

from urdyn import Urdyn
from urdyn._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL
from urdyn._store import SEARCH_INDEX_TABLE

TASK = "Prevent a database migration from leaving partially applied state after a failure."


def test_capture_workflow_end_to_end(tmp_path):
    process_a = Urdyn.init(tmp_path, "dev")

    # STEP A -- explicit failure Evidence: a raw observation, no
    # conclusion drawn from it yet.
    failure_evidence = process_a.add_evidence(
        "A forced migration failure left only part of the schema update applied.",
        kind="error_observation",
    )

    # STEP B -- failed Attempt, linked to the failure Evidence at
    # creation time (no post-hoc linking API is needed for this
    # workflow: Evidence is created before the Attempt that cites it).
    failed_attempt = process_a.record_attempt(
        task=TASK,
        approach="Apply schema changes as individual autocommitted statements.",
        outcome="failed",
        evidence=[failure_evidence],
    )
    assert failed_attempt.evidence_ids == (failure_evidence.evidence_id,)

    # STEP C -- explicit verification Evidence: also a raw observation
    # (a test result), not yet a Lesson.
    verification_evidence = process_a.add_evidence(
        "A forced failure during the transactional migration rolled back every "
        "schema change; all rollback tests passed.",
        kind="test_result",
    )

    # STEP D -- successful Attempt, a separate independent record. The
    # first (failed) Attempt is never rewritten.
    succeeded_attempt = process_a.record_attempt(
        task=TASK,
        approach="Wrap all schema changes in a single transaction with rollback on failure.",
        outcome="succeeded",
        evidence=[verification_evidence],
    )
    assert succeeded_attempt.evidence_ids == (verification_evidence.evidence_id,)
    assert succeeded_attempt.attempt_id != failed_attempt.attempt_id

    # STEP E -- verified Lesson, backed by the verification Evidence
    # (a `test_result`, one of `VERIFICATION_EVIDENCE_KINDS`). No root
    # cause is derived automatically.
    lesson = process_a.learn(
        "Persistent schema migrations should execute atomically so a failure cannot leave partial state.",
        supporting_evidence=[verification_evidence],
        verified=True,
    )
    assert lesson.epistemic_state == "verified"
    assert lesson.evidence_ids == (verification_evidence.evidence_id,)

    # process A terminates; nothing below may depend on its Python state
    del process_a

    # a brand-new process/session discovers the same workspace
    process_b = Urdyn.open(tmp_path)

    result = process_b.preflight(TASK)

    assert not result.is_empty()
    assert len(result.known_failures) == 1
    assert result.known_failures[0].attempt_id == failed_attempt.attempt_id
    assert result.known_failures[0].approach == "Apply schema changes as individual autocommitted statements."

    assert len(result.verified_lessons) == 1
    assert result.verified_lessons[0].memory_id == lesson.memory_id
    assert (
        result.verified_lessons[0].content
        == "Persistent schema migrations should execute atomically so a failure cannot leave partial state."
    )

    assert len(result.recommended_validation) == 1
    assert result.recommended_validation[0].evidence_id == verification_evidence.evidence_id
    assert result.recommended_validation[0].content == (
        "A forced failure during the transactional migration rolled back every "
        "schema change; all rollback tests passed."
    )

    # STEP F -- Evidence provenance round-trip through the resolved ids,
    # not through any object retained from process A.
    resolved_failure = process_b.get_evidence(failure_evidence.evidence_id)
    assert resolved_failure.evidence_id == failure_evidence.evidence_id
    assert resolved_failure.content == failure_evidence.content
    assert resolved_failure.kind == "error_observation"
    assert resolved_failure.recorded_at == failure_evidence.recorded_at

    resolved_verification = process_b.get_evidence(verification_evidence.evidence_id)
    assert resolved_verification.evidence_id == verification_evidence.evidence_id
    assert resolved_verification.content == verification_evidence.content
    assert resolved_verification.kind == "test_result"
    assert resolved_verification.recorded_at == verification_evidence.recorded_at


def test_standalone_evidence_is_never_surfaced_by_preflight(tmp_path):
    """Evidence != Knowledge: raw Evidence, captured but never linked to
    an Attempt or a Memory, must not become part of what `preflight()`
    returns -- not as a known failure, not as a lesson, not as
    recommended validation. Capture never promotes anything to
    knowledge on its own; only an explicit `record_attempt()`/`learn()`
    call does."""
    cx = Urdyn.init(tmp_path, "dev")

    cx.add_evidence(
        "A forced migration failure left only part of the schema update applied.",
        kind="error_observation",
    )
    cx.add_evidence(
        "A forced failure during the transactional migration rolled back every "
        "schema change; all rollback tests passed.",
        kind="test_result",
    )

    result = cx.preflight(TASK)

    assert result.is_empty()


def test_add_evidence_does_not_append_an_event(tmp_path):
    """Evidence != Event: `add_evidence()` writes only to the `evidence`
    table. It must not append anything to the append-only `events` log
    that backs `timeline()`/`list_attempts()`/`list_skills()` ordering --
    Evidence has no `VALID_EVENT_KINDS` entry and is not ordered through
    the event log the way Memory/Attempt/Skill are."""
    import sqlite3

    cx = Urdyn.init(tmp_path, "dev")
    cx.add_evidence("some raw observation", kind="command_output")

    connection = sqlite3.connect(cx._db_path)
    try:
        (count,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    finally:
        connection.close()

    assert count == 0


def test_evidence_content_is_never_indexed_for_retrieval(tmp_path):
    """Evidence is supporting material, not the retrieval surface:
    `preflight()`/`guard()` select Attempt/Memory/Skill candidates and
    resolve Evidence afterwards, by id, through their `evidence_ids`.
    Evidence content itself must never appear as a row in the derived
    FTS `search_index` table -- confirming there is no fourth,
    Evidence-shaped entity type alongside `ENTITY_MEMORY`/
    `ENTITY_ATTEMPT`/`ENTITY_SKILL`."""
    import sqlite3

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence(
        "A forced migration failure left only part of the schema update applied.",
        kind="error_observation",
    )

    connection = sqlite3.connect(cx._db_path)
    try:
        rows = connection.execute(
            f"SELECT entity_type FROM {SEARCH_INDEX_TABLE} WHERE entity_id = ?",
            (evidence.evidence_id,),
        ).fetchall()
        entity_types = connection.execute(
            f"SELECT DISTINCT entity_type FROM {SEARCH_INDEX_TABLE}"
        ).fetchall()
    finally:
        connection.close()

    assert rows == []
    assert {row[0] for row in entity_types} <= {ENTITY_MEMORY, ENTITY_ATTEMPT, ENTITY_SKILL}


def test_add_evidence_with_identical_content_yields_distinct_ids(tmp_path):
    """Repeated real-world work produces repeated observations (e.g. the
    same failing command run twice). Capture must not deduplicate by
    content: two observations made at different times are different
    evidence, even if their text happens to coincide -- content equality
    is not event equality."""
    cx = Urdyn.init(tmp_path, "dev")

    first = cx.add_evidence("1 failed, 8 passed", kind="test_result")
    second = cx.add_evidence("1 failed, 8 passed", kind="test_result")

    assert first.evidence_id != second.evidence_id
    assert cx.get_evidence(first.evidence_id).content == cx.get_evidence(second.evidence_id).content


def test_capture_workflow_does_not_depend_on_the_semantic_channel(tmp_path, monkeypatch):
    """The tracer's acceptance must not silently depend on the optional
    semantic retrieval channel (A7.4). Reusing the frozen task verbatim
    as the `preflight()` query is not, on its own, enough to admit the
    Lesson: this exact wording pair only clears ~2 shared tokens against
    the lexical majority threshold, below `_relevance.is_relevant`'s bar
    (see `_retrieval.py`'s module docstring on why concise wording can
    dilute past that threshold). What actually admits the Lesson here,
    with or without the semantic extra, is the SAME shared-provenance
    channel `test_preflight_matches_the_a4_real_utility_scenario`
    exercises: the succeeded Attempt matches the task lexically (it
    reuses the task text verbatim), and the Lesson shares that Attempt's
    own Evidence -- present since A4, well before A7.4 introduced the
    semantic channel. Simulating the `[semantic]` extra being absent
    must not change the outcome, because this admission path never
    consults it."""
    from urdyn import _workspace

    monkeypatch.setattr(_workspace, "_load_semantic_module", lambda: None)

    cx = Urdyn.init(tmp_path, "dev")
    failure_evidence = cx.add_evidence(
        "A forced migration failure left only part of the schema update applied.",
        kind="error_observation",
    )
    cx.record_attempt(
        task=TASK,
        approach="Apply schema changes as individual autocommitted statements.",
        outcome="failed",
        evidence=[failure_evidence],
    )
    verification_evidence = cx.add_evidence(
        "A forced failure during the transactional migration rolled back every "
        "schema change; all rollback tests passed.",
        kind="test_result",
    )
    cx.record_attempt(
        task=TASK,
        approach="Wrap all schema changes in a single transaction with rollback on failure.",
        outcome="succeeded",
        evidence=[verification_evidence],
    )
    cx.learn(
        "Persistent schema migrations should execute atomically so a failure cannot leave partial state.",
        supporting_evidence=[verification_evidence],
        verified=True,
    )

    result = cx.preflight(TASK)

    assert len(result.known_failures) == 1
    assert len(result.verified_lessons) == 1
    assert len(result.recommended_validation) == 1
