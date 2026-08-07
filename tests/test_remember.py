"""Tests for `Cortex.remember()`."""

import datetime as dt

import pytest

from cortex_memory import Cortex


def test_remember_returns_memory_with_content(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("SQLite was selected for the first storage implementation.")

    assert memory.content == "SQLite was selected for the first storage implementation."


def test_remember_assigns_stable_valid_id(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("first memory")

    assert isinstance(memory.memory_id, str)
    assert memory.memory_id


def test_remember_assigns_utc_recorded_at(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    before = dt.datetime.now(dt.timezone.utc)
    memory = cx.remember("timestamped memory")
    after = dt.datetime.now(dt.timezone.utc)

    assert memory.recorded_at.tzinfo is not None
    assert before <= memory.recorded_at <= after


def test_remember_defaults_to_user_asserted_epistemic_state(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("something the user told Cortex")

    assert memory.epistemic_state == "user_asserted"


def test_remember_defaults_to_note_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("plain memory")

    assert memory.kind == "note"


def test_remember_accepts_explicit_valid_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember("we chose SQLite", kind="decision")

    assert memory.kind == "decision"


def test_remember_rejects_unknown_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("something", kind="not-a-kind")


def test_remember_rejects_empty_content(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("")


def test_remember_rejects_whitespace_only_content(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("   \n\t  ")


def test_remember_persists_to_disk(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    cx.remember("durable memory")

    assert (tmp_path / ".cortex" / "memory.db").is_file()


def test_multiple_memories_are_all_retained(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    cx.remember("first")
    cx.remember("second")
    cx.remember("third")

    assert cx._count_memories() == 3


def test_remember_assigns_distinct_ids(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    first = cx.remember("one")
    second = cx.remember("two")

    assert first.memory_id != second.memory_id
