"""Tests for `Urdyn.record_attempt()`."""

import sqlite3

import pytest

from urdyn import Urdyn, UrdynStorageError


def test_record_attempt_assigns_stable_valid_id(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    attempt = cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Modify token refresh handling directly.",
        outcome="failed",
    )

    assert isinstance(attempt.attempt_id, str)
    assert attempt.attempt_id


def test_record_failed_attempt(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    attempt = cx.record_attempt(task="task", approach="approach", outcome="failed")

    assert attempt.outcome == "failed"
    assert attempt.task == "task"
    assert attempt.approach == "approach"


def test_record_successful_attempt(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    attempt = cx.record_attempt(task="task", approach="approach", outcome="succeeded")

    assert attempt.outcome == "succeeded"


def test_record_partial_attempt(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    attempt = cx.record_attempt(task="task", approach="approach", outcome="partial")

    assert attempt.outcome == "partial"


def test_record_attempt_rejects_unknown_outcome(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.record_attempt(task="task", approach="approach", outcome="not-an-outcome")


def test_record_attempt_rejects_empty_task(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.record_attempt(task="   ", approach="approach", outcome="failed")


def test_record_attempt_rejects_empty_approach(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.record_attempt(task="task", approach="  ", outcome="failed")


def test_attempt_persists_across_reopening(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    original = cx.record_attempt(task="task", approach="approach", outcome="failed")
    del cx

    reopened = Urdyn.open(tmp_path)
    (attempt,) = reopened.preflight("task approach").known_failures

    assert attempt.attempt_id == original.attempt_id
    assert attempt.task == original.task
    assert attempt.approach == original.approach
    assert attempt.outcome == original.outcome


def test_record_attempt_rejects_unknown_evidence_reference(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    from urdyn._evidence import Evidence
    import datetime as dt

    fabricated = Evidence(
        evidence_id="a" * 32,
        content="never actually persisted",
        kind="error_observation",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.record_attempt(task="task", approach="approach", outcome="failed", evidence=[fabricated])

    # the failed write must not have partially persisted an attempt or event
    assert cx.preflight("task approach").known_failures == ()


def test_record_attempt_deduplicates_repeated_evidence_reference(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("the same evidence, referenced twice", kind="error_observation")

    attempt = cx.record_attempt(task="task", approach="approach", outcome="failed", evidence=[evidence, evidence])

    assert attempt.evidence_ids == (evidence.evidence_id,)


def test_corrupted_attempt_id_is_rejected_explicitly(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="task", approach="approach", outcome="failed")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("UPDATE attempts SET attempt_id = 'not-a-valid-id'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.preflight("task approach")


def test_corrupted_outcome_is_rejected_explicitly(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="task", approach="approach", outcome="failed")

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("UPDATE attempts SET outcome = 'not-a-real-outcome'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UrdynStorageError):
        cx.preflight("task approach")


def test_attempt_object_is_immutable(tmp_path):
    import dataclasses

    cx = Urdyn.init(tmp_path, "dev")
    attempt = cx.record_attempt(task="task", approach="approach", outcome="failed")

    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.outcome = "succeeded"


def test_urdyn_has_no_way_to_mutate_an_existing_attempt(tmp_path):
    """There must be no API path -- public or otherwise on Urdyn -- that
    rewrites a previously recorded attempt's outcome. The only way to
    record a different result is a new, independent `record_attempt`."""
    cx = Urdyn.init(tmp_path, "dev")

    public_api = {name for name in dir(cx) if not name.startswith("_")}
    mutating_names = {name for name in public_api if "update" in name or "edit" in name or "mutate" in name}
    assert mutating_names == set()


def test_multiple_attempts_are_all_preserved(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    first = cx.record_attempt(task="task one", approach="approach one", outcome="failed")
    second = cx.record_attempt(task="task one", approach="approach two", outcome="succeeded")

    failures = cx.preflight("task one").known_failures
    assert [a.attempt_id for a in failures] == [first.attempt_id]

    # a later success must not erase or rewrite the earlier failure
    assert second.attempt_id != first.attempt_id
