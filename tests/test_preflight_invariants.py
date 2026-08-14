"""Tests for `Preflight.invariants` (A9.1): the one field in `Preflight`
that bypasses task relevance entirely.

Two properties are exercised here that are NOT covered by
`test_preflight.py` (which is about the lexical/FTS/semantic matching
that everything else in `Preflight` goes through):

1. an invariant with zero lexical overlap with the task must still
   appear -- unlike every other `Preflight` field, no relevance channel
   is consulted for it;
2. `pending`/`question`/`environment` must NEVER appear in `Preflight`,
   even when their content is a near-exact match for the task -- these
   kinds are deliberately excluded from `preflight()` ("deep memory,
   thin context": preflight is task preparation, not a project
   dashboard).
"""

from cortex_memory import Cortex
from cortex_memory import _workspace


def test_unrelated_invariant_is_included_in_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember(".cortex/ must remain gitignored.", kind="invariant")

    result = cx.preflight("Optimize database connection pooling.")

    assert len(result.invariants) == 1
    assert result.invariants[0].content == ".cortex/ must remain gitignored."
    assert not result.is_empty()


def test_superseded_invariant_is_excluded_from_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    new = cx.remember(
        "Canonical IDs must not depend on SQLite row IDs.", kind="invariant", supersedes=old.memory_id
    )

    result = cx.preflight("Optimize database connection pooling.")

    assert [m.memory_id for m in result.invariants] == [new.memory_id]


def test_preflight_on_empty_workspace_has_no_invariants(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    result = cx.preflight("some task nobody has attempted")

    assert result.invariants == ()
    assert result.is_empty()


def test_preflight_reports_multiple_current_invariants(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    second = cx.remember("Canonical IDs must not depend on SQLite row IDs.", kind="invariant")
    third = cx.remember("Derived indexes must remain rebuildable.", kind="invariant")

    result = cx.preflight("Refactor the CLI argument parser.")

    assert {m.memory_id for m in result.invariants} == {
        first.memory_id,
        second.memory_id,
        third.memory_id,
    }


def test_preflight_pending_does_not_leak_into_the_a9_1_fields(tmp_path):
    """[A22.1] A pending IS now surfaced -- in its own `pending` field,
    and only when relevant to the task (see
    `tests/test_a22_pending_preflight.py`). What A9.1 asserted here and
    what A22.1 keeps asserting is the part that never changed: a pending
    is not a root cause, not a lesson, not a known failure, and it does
    not inherit the invariants' unconditional inclusion."""
    cx = Cortex.init(tmp_path, "dev")
    task = "Optimize database connection pooling for the storage layer."
    pending = cx.remember(task, kind="pending")

    result = cx.preflight(task)

    assert [m.memory_id for m in result.pending] == [pending.memory_id]
    # nothing pending leaks into any existing field
    assert result.known_failures == ()
    assert result.root_causes == ()
    assert result.verified_lessons == ()
    assert result.invariants == ()
    # ...and it is task-relevant, not project-wide like an invariant
    assert cx.preflight("Rename the changelog heading").pending == ()


def test_preflight_question_never_appears_even_when_lexically_identical(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    task = "Optimize database connection pooling for the storage layer."
    cx.remember(task, kind="question")

    result = cx.preflight(task)

    assert not hasattr(result, "questions")
    assert result.known_failures == ()
    assert result.root_causes == ()
    assert result.verified_lessons == ()


def test_preflight_environment_never_appears_even_when_lexically_identical(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    task = "Optimize database connection pooling for the storage layer."
    cx.remember(task, kind="environment")

    result = cx.preflight(task)

    assert not hasattr(result, "environment")
    assert result.known_failures == ()
    assert result.root_causes == ()
    assert result.verified_lessons == ()


def test_preflight_result_has_no_question_environment_fields_at_all():
    """Shape guarantee: A9.1 gained exactly one field (`invariants`), not
    four, and A22.1 gained exactly one more (`pending`) -- deliberately
    not `questions`/`environment`, which have no admission channel and
    are still reached only through `state()`/`timeline()`."""
    import dataclasses

    from cortex_memory import Preflight

    field_names = {f.name for f in dataclasses.fields(Preflight)}

    assert "questions" not in field_names
    assert "environment" not in field_names
    assert "invariants" in field_names
    assert "pending" in field_names


def test_invariants_work_without_the_semantic_extra(tmp_path, monkeypatch):
    """Invariant retrieval never calls into `_semantic.py` at all (see
    `Cortex.preflight`): it is a plain `state(kind="invariant")`-style
    fetch, filtered only by current state. Simulating the `[semantic]`
    extra being absent (`_load_semantic_module` returning None, exactly
    what happens on a real install without it) must not change that."""
    monkeypatch.setattr(_workspace, "_load_semantic_module", lambda: None)
    cx = Cortex.init(tmp_path, "dev")
    cx.remember(".cortex/ must remain gitignored.", kind="invariant")

    result = cx.preflight("Optimize database connection pooling.")

    assert len(result.invariants) == 1


def test_guard_result_has_no_invariants_field():
    """`guard()` deliberately does not get invariants in A9.1."""
    import dataclasses

    from cortex_memory import GuardResult

    field_names = {f.name for f in dataclasses.fields(GuardResult)}

    assert "invariants" not in field_names
