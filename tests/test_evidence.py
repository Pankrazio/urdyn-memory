"""Tests for `Urdyn.add_evidence()`, `Urdyn.get_evidence()`, and provenance."""

import pytest

from urdyn import Urdyn


def test_add_evidence_assigns_stable_valid_id(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    evidence = cx.add_evidence("The user said SQLite should be used for V1.")

    assert isinstance(evidence.evidence_id, str)
    assert evidence.evidence_id


def test_add_evidence_defaults_to_user_statement_kind(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    evidence = cx.add_evidence("some statement")

    assert evidence.kind == "user_statement"


def test_add_evidence_accepts_explicit_kind(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    evidence = cx.add_evidence("tests passed: 77/77", kind="test_result")

    assert evidence.kind == "test_result"


def test_add_evidence_accepts_user_confirmation_kind(tmp_path):
    """`user_confirmation` is distinct from `user_statement`: it is what
    lets a verified memory be backed by something a user explicitly
    confirmed, as opposed to a bare unchecked opinion."""
    cx = Urdyn.init(tmp_path, "dev")

    evidence = cx.add_evidence("I ran it and the bug is gone.", kind="user_confirmation")

    assert evidence.kind == "user_confirmation"


def test_add_evidence_rejects_unknown_kind(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.add_evidence("something", kind="not-a-kind")


def test_add_evidence_rejects_empty_content(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.add_evidence("   ")


def test_evidence_content_persists_across_reopening(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("The user said SQLite should be used for V1.")
    del cx

    reopened = Urdyn.open(tmp_path)
    fetched = reopened.get_evidence(evidence.evidence_id)

    assert fetched.evidence_id == evidence.evidence_id
    assert fetched.content == evidence.content
    assert fetched.kind == evidence.kind


def test_get_evidence_rejects_unknown_id(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.get_evidence("0" * 32)


def test_get_evidence_on_empty_workspace_rejects(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.get_evidence("0" * 32)


def test_memory_records_evidence_association(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("The user said SQLite should be used for V1.")

    memory = cx.remember("SQLite was selected for V1.", kind="decision", evidence=[evidence])

    assert memory.evidence_ids == (evidence.evidence_id,)


def test_memory_evidence_association_persists_across_reopening(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("The user said SQLite should be used for V1.")
    original = cx.remember("SQLite was selected for V1.", kind="decision", evidence=[evidence])
    del cx

    reopened = Urdyn.open(tmp_path)
    (memory,) = reopened.recall("SQLite")

    assert memory.memory_id == original.memory_id
    assert memory.evidence_ids == (evidence.evidence_id,)
    provenance = reopened.get_evidence(memory.evidence_ids[0])
    assert provenance.content == "The user said SQLite should be used for V1."


def test_memory_with_no_evidence_has_empty_evidence_ids(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    memory = cx.remember("a plain memory with no provenance")

    assert memory.evidence_ids == ()


def test_remember_deduplicates_repeated_evidence_reference(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("the same evidence, referenced twice")

    memory = cx.remember("a memory", evidence=[evidence, evidence])

    assert memory.evidence_ids == (evidence.evidence_id,)


def test_remember_rejects_unknown_evidence_reference(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.add_evidence("real evidence")
    from urdyn._evidence import Evidence
    import datetime as dt

    fabricated = Evidence(
        evidence_id="a" * 32,
        content="never actually persisted",
        kind="user_statement",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(ValueError):
        cx.remember("a memory built on fabricated evidence", evidence=[fabricated])

    # the memory must not have been partially persisted
    assert cx.recall("fabricated", include_superseded=True) == []
