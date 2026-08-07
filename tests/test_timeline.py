"""Tests for `Cortex.timeline()`."""

from cortex_memory import Cortex


def test_timeline_on_empty_workspace_returns_empty_list(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    assert cx.timeline() == []


def test_timeline_orders_memories_oldest_first(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember("first")
    second = cx.remember("second")
    third = cx.remember("third")

    history = cx.timeline()

    assert [m.memory_id for m in history] == [first.memory_id, second.memory_id, third.memory_id]


def test_timeline_filters_by_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    note = cx.remember("a note", kind="note")
    decision = cx.remember("a decision", kind="decision")

    notes = cx.timeline(kind="note")
    decisions = cx.timeline(kind="decision")

    assert [m.memory_id for m in notes] == [note.memory_id]
    assert [m.memory_id for m in decisions] == [decision.memory_id]


def test_timeline_includes_superseded_and_current_memories(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    history = cx.timeline(kind="decision")

    assert [m.memory_id for m in history] == [old.memory_id, new.memory_id]


def test_timeline_ordering_is_deterministic_across_calls(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("first")
    cx.remember("second")
    cx.remember("third")

    first_call = cx.timeline()
    second_call = cx.timeline()

    assert [m.memory_id for m in first_call] == [m.memory_id for m in second_call]


def test_timeline_survives_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember("first")
    second = cx.remember("second")
    del cx

    reopened = Cortex.open(tmp_path)
    history = reopened.timeline()

    assert [m.memory_id for m in history] == [first.memory_id, second.memory_id]


def test_timeline_order_is_insertion_order_even_with_identical_timestamps(tmp_path, monkeypatch):
    import datetime as real_dt

    import cortex_memory._workspace as workspace_module

    cx = Cortex.init(tmp_path, "dev")
    frozen_now = real_dt.datetime.now(real_dt.timezone.utc)

    # Replace only the `dt` name as seen inside `_workspace`, so `_store`'s
    # own `dt.datetime.fromisoformat` (same shared `datetime` module) is
    # unaffected. This isolates the clock freeze to memory creation.
    monkeypatch.setattr(workspace_module, "dt", _FakeDatetimeModule(frozen_now, real_dt))

    first = cx.remember("first")
    second = cx.remember("second")
    third = cx.remember("third")

    assert first.recorded_at == second.recorded_at == third.recorded_at

    history = cx.timeline()
    assert [m.memory_id for m in history] == [first.memory_id, second.memory_id, third.memory_id]


class _FrozenClock:
    """Stand-in for `datetime.datetime` whose `now()` always returns the
    same fixed instant, so we can test timeline ordering under a
    timestamp collision without relying on real clock precision."""

    def __init__(self, frozen_now):
        self._frozen_now = frozen_now

    def now(self, tz=None):
        return self._frozen_now


class _FakeDatetimeModule:
    """Minimal stand-in for the `datetime` module, exposing only what
    `_workspace.py` uses (`dt.datetime.now(dt.timezone.utc)`)."""

    def __init__(self, frozen_now, real_dt_module):
        self.datetime = _FrozenClock(frozen_now)
        self.timezone = real_dt_module.timezone
