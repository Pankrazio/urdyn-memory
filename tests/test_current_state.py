"""Tests for `Urdyn.state()`: the current-state projection over history."""

from urdyn import Urdyn


def test_state_on_empty_workspace_returns_empty_list(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    assert cx.state() == []


def test_state_returns_all_memories_when_nothing_superseded(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    first = cx.remember("first")
    second = cx.remember("second")

    current = cx.state()

    assert {m.memory_id for m in current} == {first.memory_id, second.memory_id}


def test_state_excludes_superseded_memory(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    current = cx.state()

    assert [m.memory_id for m in current] == [new.memory_id]


def test_state_filters_by_kind(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    note = cx.remember("a note", kind="note")
    decision = cx.remember("a decision", kind="decision")

    assert [m.memory_id for m in cx.state(kind="note")] == [note.memory_id]
    assert [m.memory_id for m in cx.state(kind="decision")] == [decision.memory_id]


def test_state_survives_reopening(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)
    del cx

    reopened = Urdyn.open(tmp_path)

    assert [m.memory_id for m in reopened.state(kind="decision")] == [new.memory_id]
