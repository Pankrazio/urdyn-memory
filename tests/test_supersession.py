"""Tests for memory supersession: old memory preserved, new memory current."""

import pytest

from cortex_memory import Cortex


def test_supersession_preserves_old_memory(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")

    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    history = cx.timeline(kind="decision")
    assert [m.memory_id for m in history] == [old.memory_id, new.memory_id]
    assert history[0].content == "PostgreSQL was selected."


def test_new_memory_records_supersedes_link(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")

    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    assert new.supersedes == old.memory_id
    assert old.supersedes is None


def test_recall_excludes_superseded_memory_by_default(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    results = cx.recall("selected")

    assert len(results) == 1
    assert results[0].content == "SQLite was selected for V1."


def test_recall_includes_superseded_memory_when_requested(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    results = cx.recall("selected", include_superseded=True)

    assert len(results) == 2
    assert {m.content for m in results} == {"PostgreSQL was selected.", "SQLite was selected for V1."}


def test_state_returns_only_current_memory(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    current = cx.state(kind="decision")

    assert [m.memory_id for m in current] == [new.memory_id]


def test_supersede_unknown_memory_is_rejected(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("SQLite was selected for V1.", kind="decision", supersedes="0" * 32)

    # nothing must have been persisted
    assert cx.timeline() == []


def test_self_supersession_is_rejected(tmp_path, monkeypatch):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember("PostgreSQL was selected.", kind="decision")

    import uuid as uuid_module

    monkeypatch.setattr(uuid_module, "uuid4", lambda: uuid_module.UUID(first.memory_id))

    with pytest.raises(ValueError):
        cx.remember("self-superseding memory", kind="decision", supersedes=first.memory_id)


def test_supersession_survives_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)
    del cx

    reopened = Cortex.open(tmp_path)

    history = reopened.timeline(kind="decision")
    assert [m.memory_id for m in history] == [old.memory_id, new.memory_id]
    current = reopened.state(kind="decision")
    assert [m.memory_id for m in current] == [new.memory_id]


def test_failed_supersession_does_not_partially_persist(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("orphaned decision", kind="decision", supersedes="0" * 32)

    # no memory, and no dangling event, must have been written
    assert cx._count_memories() == 0
    assert cx.timeline(kind="decision") == []


def test_memory_cannot_be_superseded_twice(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    with pytest.raises(ValueError):
        cx.remember("MySQL was selected for V1.", kind="decision", supersedes=old.memory_id)

    # the first supersession must remain the only one
    history = cx.timeline(kind="decision")
    assert len(history) == 2


def test_multiple_supersessions_form_a_chain(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember("PostgreSQL was selected.", kind="decision")
    b = cx.remember("MySQL was selected.", kind="decision", supersedes=a.memory_id)
    c = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=b.memory_id)

    history = cx.timeline(kind="decision")
    assert [m.memory_id for m in history] == [a.memory_id, b.memory_id, c.memory_id]

    current = cx.state(kind="decision")
    assert [m.memory_id for m in current] == [c.memory_id]
