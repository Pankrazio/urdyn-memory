"""A18.1 foundation hardening: MEDIUM-1 (canonical Memory must have
canonical Event history) and PD-1 (first `memory.db` creation race).

Both fixes live in `MemoryStore` (`_store.py`):

- MEDIUM-1: `MemoryStore.add()` now rejects a write that would insert a
  NEW Memory row without an accompanying `EVENT_KIND_MEMORY_RECORDED`
  event whose `subject_id` equals the new memory's id. An exact
  duplicate retry (which writes nothing) is unaffected.
- PD-1: `_ensure_schema()`'s first-creation path (`user_version == 0`)
  now serializes under `BEGIN IMMEDIATE` and re-checks the version
  after acquiring the lock, so two processes racing to create
  `memory.db` for the first time cannot both attempt
  `_create_v6_schema()`.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex
from cortex_memory._errors import CortexStorageError
from cortex_memory._event import (
    EVENT_KIND_MEMORY_RECORDED,
    EVENT_KIND_MEMORY_SUPERSEDED,
    Event,
)
from cortex_memory._memory import Memory
from cortex_memory._store import STORE_SCHEMA_VERSION, MemoryStore, db_path_for


def _events(cx, kind=None):
    with sqlite3.connect(cx._db_path) as connection:
        if kind is None:
            rows = connection.execute("SELECT kind, subject_id FROM events ORDER BY sequence").fetchall()
        else:
            rows = connection.execute(
                "SELECT kind, subject_id FROM events WHERE kind = ? ORDER BY sequence", (kind,)
            ).fetchall()
    return rows


def _raw_memory_count(cx) -> int:
    with sqlite3.connect(cx._db_path) as connection:
        (count,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
    return count


def _fts_memory_count(cx) -> int:
    with sqlite3.connect(cx._db_path) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM search_index WHERE entity_type = 'memory'"
        ).fetchone()
    return count


def _make_memory(*, content, kind="note", supersedes=None, epistemic_state="user_asserted") -> Memory:
    return Memory(
        memory_id=uuid.uuid4().hex,
        content=content,
        kind=kind,
        epistemic_state=epistemic_state,
        recorded_at=dt.datetime.now(dt.timezone.utc),
        supersedes=supersedes,
        evidence_ids=(),
        supporting_evidence_ids=(),
    )


# ---------------------------------------------------------------------------
# MEDIUM-1: canonical Memory must have canonical Event history
# ---------------------------------------------------------------------------


class TestMediumOneEventRequirement:
    def test_a_direct_add_with_no_events_is_rejected(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("baseline fact", kind="note")
        before_raw = _raw_memory_count(cx)
        before_events = len(_events(cx))
        before_fts = _fts_memory_count(cx)

        memory = _make_memory(content="a ghost memory with no event")
        with pytest.raises(ValueError, match="memory_recorded"):
            with MemoryStore.create_or_open(cx._db_path) as store:
                store.add(memory, ())

        assert _raw_memory_count(cx) == before_raw
        assert len(_events(cx)) == before_events
        assert _fts_memory_count(cx) == before_fts
        # store must still be fully usable afterwards
        again = cx.remember("store still works after rejected ghost write", kind="note")
        assert again.memory_id

    def test_b_event_with_wrong_subject_id_is_rejected(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("baseline fact", kind="note")
        before_raw = _raw_memory_count(cx)
        before_events = len(_events(cx))

        memory = _make_memory(content="mismatched subject")
        wrong_subject_event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_MEMORY_RECORDED,
            subject_id=uuid.uuid4().hex,  # NOT memory.memory_id
            occurred_at=memory.recorded_at,
        )
        with pytest.raises(ValueError, match="memory_recorded"):
            with MemoryStore.create_or_open(cx._db_path) as store:
                store.add(memory, [wrong_subject_event])

        assert _raw_memory_count(cx) == before_raw
        assert len(_events(cx)) == before_events

    def test_b_event_with_wrong_kind_is_rejected(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("baseline fact", kind="note")
        before_raw = _raw_memory_count(cx)
        before_events = len(_events(cx))

        memory = _make_memory(content="wrong kind of event")
        wrong_kind_event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_MEMORY_SUPERSEDED,  # not memory_recorded
            subject_id=memory.memory_id,
            occurred_at=memory.recorded_at,
        )
        with pytest.raises(ValueError, match="memory_recorded"):
            with MemoryStore.create_or_open(cx._db_path) as store:
                store.add(memory, [wrong_kind_event])

        assert _raw_memory_count(cx) == before_raw
        assert len(_events(cx)) == before_events

    def test_c_direct_write_with_correct_event_is_visible_everywhere(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        memory = _make_memory(content="a properly recorded direct write", kind="note")
        event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_MEMORY_RECORDED,
            subject_id=memory.memory_id,
            occurred_at=memory.recorded_at,
        )
        with MemoryStore.create_or_open(cx._db_path) as store:
            store.add(memory, [event])

        assert memory.memory_id in {m.memory_id for m in cx.recall("properly recorded")}
        assert memory.memory_id in {m.memory_id for m in cx.timeline()}
        assert memory.memory_id in {m.memory_id for m in cx.state()}

        reopened = Cortex.open(tmp_path)
        assert memory.memory_id in {m.memory_id for m in reopened.recall("properly recorded")}
        assert memory.memory_id in {m.memory_id for m in reopened.timeline()}
        assert memory.memory_id in {m.memory_id for m in reopened.state()}

    def test_d_exact_duplicate_retry_needs_no_event(self, tmp_path):
        """The A17 fast-path must stay reachable with `events=()`: a
        retry that resolves to an already-current equivalent writes
        nothing, so it must not be forced to supply an event it will
        never use."""
        cx = Cortex.init(tmp_path, "dev")
        first = cx.remember("duplicate-prone content", kind="note")

        duplicate = _make_memory(content="duplicate-prone content", kind="note")
        with MemoryStore.create_or_open(cx._db_path) as store:
            returned = store.add(duplicate, ())

        assert returned.memory_id == first.memory_id
        assert _raw_memory_count(cx) == 1
        assert len(_events(cx)) == 1

    def test_e_supersession_history_unchanged(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        original = cx.remember("root cause A", kind="root_cause")
        updated = cx.remember("root cause B", kind="root_cause", supersedes=original.memory_id)

        recorded_events = _events(cx, kind=EVENT_KIND_MEMORY_RECORDED)
        superseded_events = _events(cx, kind=EVENT_KIND_MEMORY_SUPERSEDED)
        assert len(recorded_events) == 2
        assert len(superseded_events) == 1
        assert superseded_events[0][1] == original.memory_id
        assert cx.state()[0].memory_id == updated.memory_id

    def test_f_supersession_retry_is_idempotent(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        original = cx.remember("target of supersession", kind="root_cause")
        first = cx.remember("the replacement fact", kind="root_cause", supersedes=original.memory_id)
        second = cx.remember("the replacement fact", kind="root_cause", supersedes=original.memory_id)

        assert first.memory_id == second.memory_id
        assert len(_events(cx, kind=EVENT_KIND_MEMORY_SUPERSEDED)) == 1
        assert len(_events(cx, kind=EVENT_KIND_MEMORY_RECORDED)) == 2

    def test_g_injected_event_insert_failure_rolls_back_everything(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("pre-existing baseline", kind="note")
        before_raw = _raw_memory_count(cx)
        before_events = len(_events(cx))
        before_fts = _fts_memory_count(cx)

        class _FailOnEventInsert:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, params=()):
                if "INSERT INTO events" in sql:
                    raise sqlite3.OperationalError("simulated event insert failure")
                return self._real.execute(sql, params) if params != () else self._real.execute(sql)

            def __enter__(self):
                self._real.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._real.__exit__(exc_type, exc, tb)

            def close(self):
                self._real.close()

        store = MemoryStore.create_or_open(cx._db_path)
        store._connection = _FailOnEventInsert(store._connection)

        memory = _make_memory(content="should never survive")
        event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_MEMORY_RECORDED,
            subject_id=memory.memory_id,
            occurred_at=memory.recorded_at,
        )
        with pytest.raises(CortexStorageError):
            store.add(memory, [event])
        store._connection.close()

        assert _raw_memory_count(cx) == before_raw
        assert len(_events(cx)) == before_events
        assert _fts_memory_count(cx) == before_fts
        # store remains usable
        again = cx.remember("store still usable after injected failure", kind="note")
        assert again.memory_id


# ---------------------------------------------------------------------------
# PD-1: first `memory.db` creation race
# ---------------------------------------------------------------------------

_PROC_COUNT = 6


def _pd1_worker_remember(workspace_dir, content, kind, barrier, queue):
    try:
        barrier.wait(timeout=30)
        from cortex_memory import Cortex as _Cortex

        memory = _Cortex.open(workspace_dir).remember(content, kind=kind)
        queue.put(("ok", memory.memory_id))
    except BaseException as exc:  # noqa: BLE001 - reported to parent, not swallowed
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_workers(tmp_path, contents, *, timeout=60):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(contents))
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_pd1_worker_remember, args=(tmp_path, content, "note", barrier, queue))
        for content in contents
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for process in processes:
        process.join(timeout=timeout)
        assert process.exitcode == 0, f"worker process exited with {process.exitcode}"
    return results


def _integrity_check(db_path) -> str:
    with sqlite3.connect(db_path) as connection:
        (result,) = connection.execute("PRAGMA integrity_check").fetchone()
        (user_version,) = connection.execute("PRAGMA user_version").fetchone()
    return result, user_version


class TestPD1FirstCreationRace:
    def test_a_identical_first_write_from_many_processes(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        db_path = db_path_for(tmp_path / ".cortex")
        assert not db_path.exists()

        results = _run_workers(tmp_path, ["the same first fact"] * _PROC_COUNT)

        errors = [r for r in results if r[0] == "error"]
        assert not errors, errors
        memory_ids = {r[1] for r in results}
        assert len(memory_ids) == 1

        assert db_path.exists()
        integrity, user_version = _integrity_check(db_path)
        assert integrity == "ok"
        assert user_version == STORE_SCHEMA_VERSION

        reopened = Cortex.open(tmp_path)
        assert reopened._count_memories() == 1
        assert len(reopened.timeline()) == 1
        assert len(reopened.state()) == 1
        assert reopened.state()[0].memory_id == next(iter(memory_ids))

    def test_b_distinct_first_writes_from_many_processes(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        db_path = db_path_for(tmp_path / ".cortex")
        assert not db_path.exists()

        contents = [f"distinct first fact number {i}" for i in range(_PROC_COUNT)]
        results = _run_workers(tmp_path, contents)

        errors = [r for r in results if r[0] == "error"]
        assert not errors, errors
        memory_ids = {r[1] for r in results}
        assert len(memory_ids) == _PROC_COUNT

        integrity, user_version = _integrity_check(db_path)
        assert integrity == "ok"
        assert user_version == STORE_SCHEMA_VERSION

        reopened = Cortex.open(tmp_path)
        assert reopened._count_memories() == _PROC_COUNT
        assert {m.memory_id for m in reopened.timeline()} == memory_ids
        assert {m.memory_id for m in reopened.state()} == memory_ids

    def test_c_reopen_after_contention_is_fully_functional(self, tmp_path):
        Cortex.init(tmp_path, "dev")
        _run_workers(tmp_path, ["reopen check fact"] * _PROC_COUNT)

        cx = Cortex.open(tmp_path)
        cx.state()
        cx.timeline()
        followup = cx.remember("a normal write after contention", kind="note")
        assert followup.memory_id
        assert cx._count_memories() == 2

    def test_d_read_only_does_not_materialize_the_database(self, tmp_path):
        Cortex.init(tmp_path, "dev")
        db_path = db_path_for(tmp_path / ".cortex")
        cx = Cortex.open(tmp_path)

        assert cx.state() == []
        assert cx.timeline() == []
        assert cx.recall("anything") == []
        assert cx._count_memories() == 0
        assert not db_path.exists()

    def test_e_existing_db_concurrency_still_works(self, tmp_path):
        """A17's original concurrency guarantee on an ALREADY-created
        store must not regress: seed the store first (so this test is
        purely about the `add()` duplicate/write path, not PD-1), then
        hit it with concurrent identical writes."""
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("seed so memory.db already exists", kind="note")

        results = _run_workers(tmp_path, ["existing-db concurrent fact"] * _PROC_COUNT)

        errors = [r for r in results if r[0] == "error"]
        assert not errors, errors
        memory_ids = {r[1] for r in results}
        assert len(memory_ids) == 1
        assert cx._count_memories() == 2


# ---------------------------------------------------------------------------
# Interaction: first-creation race x identical concurrent remember x
# canonical Event requirement x A17 duplicate idempotency, all at once.
# ---------------------------------------------------------------------------


def test_interaction_first_creation_concurrent_identical_remember(tmp_path):
    Cortex.init(tmp_path, "dev")
    db_path = db_path_for(tmp_path / ".cortex")
    assert not db_path.exists()

    results = _run_workers(tmp_path, ["the one true first fact"] * _PROC_COUNT)

    errors = [r for r in results if r[0] == "error"]
    assert not errors, errors
    memory_ids = {r[1] for r in results}
    assert len(memory_ids) == 1
    canonical_id = next(iter(memory_ids))

    integrity, user_version = _integrity_check(db_path)
    assert integrity == "ok"
    assert user_version == STORE_SCHEMA_VERSION

    cx = Cortex.open(tmp_path)
    recorded_events = _events(cx, kind=EVENT_KIND_MEMORY_RECORDED)
    assert len(recorded_events) == 1
    assert recorded_events[0][1] == canonical_id

    assert [m.memory_id for m in cx.recall("the one true first fact")] == [canonical_id]
    assert [m.memory_id for m in cx.timeline()] == [canonical_id]
    assert [m.memory_id for m in cx.state()] == [canonical_id]
