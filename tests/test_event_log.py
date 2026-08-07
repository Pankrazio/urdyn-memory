"""White-box tests for the internal append-only event log backing `timeline()`.

`Event` is not part of the public API (see `cortex_memory.__init__`), but it
is a real, persisted primitive with its own identity and ordering
guarantees, so it is tested directly against its owning module.
"""

from cortex_memory import Cortex
from cortex_memory._event import EVENT_ID_PATTERN, EVENT_KIND_MEMORY_RECORDED, EVENT_KIND_MEMORY_SUPERSEDED
from cortex_memory._store import MemoryStore


def _events(cx):
    with MemoryStore.create_or_open(cx._db_path) as store:
        cursor = store._connection.execute(
            "SELECT event_id, kind, subject_id FROM events ORDER BY sequence"
        )
        return cursor.fetchall()


def test_remember_appends_one_event(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    memory = cx.remember("first")

    rows = _events(cx)

    assert len(rows) == 1
    event_id, kind, subject_id = rows[0]
    assert EVENT_ID_PATTERN.fullmatch(event_id)
    assert kind == EVENT_KIND_MEMORY_RECORDED
    assert subject_id == memory.memory_id


def test_supersession_appends_two_events(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("PostgreSQL was selected.", kind="decision")
    new = cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

    rows = _events(cx)

    assert len(rows) == 3
    kinds_and_subjects = [(kind, subject_id) for _, kind, subject_id in rows]
    assert kinds_and_subjects == [
        (EVENT_KIND_MEMORY_RECORDED, old.memory_id),
        (EVENT_KIND_MEMORY_RECORDED, new.memory_id),
        (EVENT_KIND_MEMORY_SUPERSEDED, old.memory_id),
    ]


def test_event_ids_are_distinct(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("first")
    cx.remember("second")

    rows = _events(cx)
    event_ids = [row[0] for row in rows]

    assert len(event_ids) == len(set(event_ids))


def test_events_are_append_only_across_restart(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("first")
    before_restart = _events(cx)
    del cx

    reopened = Cortex.open(tmp_path)
    reopened.remember("second")
    after_restart = _events(reopened)

    assert after_restart[: len(before_restart)] == before_restart
    assert len(after_restart) == len(before_restart) + 1


def test_failed_remember_does_not_append_a_dangling_event(tmp_path):
    import pytest

    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("orphaned", kind="decision", supersedes="0" * 32)

    assert _events(cx) == []
