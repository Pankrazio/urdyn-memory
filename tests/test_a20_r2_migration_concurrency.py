"""A20.R.2: the canonical migration chain must be serialized by SQLite,
not by timing.

`_ensure_schema()` used to read `PRAGMA user_version` ONCE, outside any
lock, and let that single reading decide the whole v1->v7 chain; each
step then ran under a plain (DEFERRED) `BEGIN`, which takes no lock until
its first write. Two processes opening the same store therefore both
observed version N with no lock held and both went on to execute the
N->N+1 migration. The loser took the lock only when it reached its first
DDL -- by which point the winner had committed -- and collided:

    duplicate column name: supersedes      (v1->v2)
    table attempts already exists          (v2->v3)
    table skills already exists            (v3->v4)
    duplicate column name: role            (v4->v5)
    table memory_conflicts already exists  (v5->v6)
    table sources already exists           (v6->v7)

reported to the caller as `CortexStorageError: ... is corrupted` -- an
open failing on a store with nothing wrong with it, and a message
inviting the user to discard an intact database. Every step of the chain
had this shape; against the pre-repair baseline 10 rounds out of 10 (six
real processes each, starting from every version v1..v6) produced at
least one failure.

The repair is the same protocol A18.1 applied to first creation and
A20.R.1 applied to the derived index, now applied to the migration chain:
take the write lock FIRST (`BEGIN IMMEDIATE`), re-read `PRAGMA
user_version` UNDER it, and only then pick the step matching THAT
version -- one step per transaction, with the version stamp written
inside the same transaction as the DDL it records.

Two things this deliberately is NOT:

- it is not `IF NOT EXISTS` / catching "duplicate column". Concurrency
  safety comes from the lock plus the re-read, which is what keeps a
  genuinely malformed store failing closed (see
  `TestMalformedStoreStillFailsClosed`).
- it is not one transaction for the whole chain. Per-step keeps the
  write lock short, preserves "a completed step stays completed", and
  leaves the canonical boundary fully committed before
  `_ensure_search_index` opens its own -- sibling critical sections,
  never nested ones.

The race is open-time, so it hits READS too: every public operation
reaches `_ensure_schema` through `create_or_open`/`open_if_exists`.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex, CortexStorageError
from cortex_memory._store import (
    _CREATE_ATTEMPT_EVIDENCE_SQL,
    _CREATE_ATTEMPTS_SQL,
    _CREATE_EVENTS_SQL,
    _CREATE_EVIDENCE_SQL,
    _CREATE_MEMORIES_V2_SQL,
    _CREATE_MEMORY_CONFLICTS_SQL,
    _CREATE_MEMORY_EVIDENCE_SQL,
    _CREATE_MEMORY_EVIDENCE_V5_SQL,
    _CREATE_SKILL_CONDITIONS_SQL,
    _CREATE_SKILL_EVIDENCE_SQL,
    _CREATE_SKILL_STEPS_SQL,
    _CREATE_SKILLS_SQL,
    _ROLE_RELATED,
    SEARCH_INDEX_TABLE,
    STORE_SCHEMA_VERSION,
    MemoryStore,
    db_path_for,
)

_PROC_COUNT = 6

_V1_MEMORIES_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""

_V7_TABLE_NAMES = (
    "memories",
    "evidence",
    "memory_evidence",
    "events",
    "attempts",
    "attempt_evidence",
    "skills",
    "skill_steps",
    "skill_conditions",
    "skill_evidence",
    "memory_conflicts",
    "sources",
    "source_observations",
)


# ---------------------------------------------------------------------------
# Building a store that is genuinely AT an old version -- not a current
# store with its version number rewritten. The DDL constants are imported
# from the module under test so that a store built here is byte-for-byte
# the shape the corresponding migration step expects to find.
# ---------------------------------------------------------------------------


def _ddl_for_version(version):
    v2 = [_CREATE_MEMORIES_V2_SQL, _CREATE_EVIDENCE_SQL, _CREATE_MEMORY_EVIDENCE_SQL, _CREATE_EVENTS_SQL]
    v3 = v2 + [_CREATE_ATTEMPTS_SQL, _CREATE_ATTEMPT_EVIDENCE_SQL]
    v4 = v3 + [
        _CREATE_SKILLS_SQL,
        _CREATE_SKILL_STEPS_SQL,
        _CREATE_SKILL_CONDITIONS_SQL,
        _CREATE_SKILL_EVIDENCE_SQL,
    ]
    # v5 changes the shape of `memory_evidence` itself (it gains `role`),
    # so a real v5 store is not v4 plus a table.
    v5 = [
        _CREATE_MEMORIES_V2_SQL,
        _CREATE_EVIDENCE_SQL,
        _CREATE_MEMORY_EVIDENCE_V5_SQL,
        _CREATE_EVENTS_SQL,
        _CREATE_ATTEMPTS_SQL,
        _CREATE_ATTEMPT_EVIDENCE_SQL,
        _CREATE_SKILLS_SQL,
        _CREATE_SKILL_STEPS_SQL,
        _CREATE_SKILL_CONDITIONS_SQL,
        _CREATE_SKILL_EVIDENCE_SQL,
    ]
    v6 = v5 + [_CREATE_MEMORY_CONFLICTS_SQL]
    return {1: [_V1_MEMORIES_SQL], 2: v2, 3: v3, 4: v4, 5: v5, 6: v6}[version]


def _build_store_at_version(db_path, version, *, count=3):
    """Create a store stamped at `version` holding canonical data valid
    for that version, and return the memory ids it holds.

    v1 predates `evidence`/`events` entirely, so a v1 store carries
    memories only -- which is exactly what makes it the interesting
    starting point: it is the one step of the chain that BACKFILLS data
    rather than only adding tables.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    memory_ids = []
    try:
        with connection:
            for statement in _ddl_for_version(version):
                connection.execute(statement)
            for index in range(count):
                memory_id = uuid.uuid4().hex
                memory_ids.append(memory_id)
                recorded_at = f"2026-08-{10 + index:02d}T09:00:00+00:00"
                content = f"canonical fact number {index} recorded under schema v{version}"
                if version == 1:
                    connection.execute(
                        "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (memory_id, content, "note", "user_asserted", recorded_at),
                    )
                    continue
                connection.execute(
                    "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, "
                    "supersedes) VALUES (?, ?, ?, ?, ?, NULL)",
                    (memory_id, content, "note", "user_asserted", recorded_at),
                )
                connection.execute(
                    "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                    (uuid.uuid4().hex, "memory_recorded", memory_id, recorded_at),
                )
                evidence_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO evidence (evidence_id, content, kind, recorded_at) VALUES (?, ?, ?, ?)",
                    (evidence_id, f"observed support for fact {index}", "user_statement", recorded_at),
                )
                if version >= 5:
                    connection.execute(
                        "INSERT INTO memory_evidence (memory_id, evidence_id, position, role) "
                        "VALUES (?, ?, 0, ?)",
                        (memory_id, evidence_id, _ROLE_RELATED),
                    )
                else:
                    connection.execute(
                        "INSERT INTO memory_evidence (memory_id, evidence_id, position) VALUES (?, ?, 0)",
                        (memory_id, evidence_id),
                    )
            connection.execute(f"PRAGMA user_version = {version}")
    finally:
        connection.close()
    return memory_ids


def _prepare_workspace(tmp_path, version, *, count=3):
    Cortex.init(tmp_path, "dev")
    db_path = db_path_for(tmp_path / ".cortex")
    assert not db_path.exists(), "Cortex.init must not materialize the store this test builds by hand"
    return db_path, _build_store_at_version(db_path, version, count=count)


def _db_facts(db_path):
    connection = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "tables": tables,
            "memory_evidence_columns": [
                row[1] for row in connection.execute("PRAGMA table_info(memory_evidence)")
            ],
            "memory_ids": sorted(row[0] for row in connection.execute("SELECT memory_id FROM memories")),
            "memory_recorded_events": connection.execute(
                "SELECT COUNT(*) FROM events WHERE kind = 'memory_recorded'"
            ).fetchone()[0],
            "memory_evidence_rows": connection.execute(
                "SELECT COUNT(*) FROM memory_evidence"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def _assert_healthy_v7(db_path, expected_ids, *, extra_memories=0, expected_links=None):
    """Every property the migration must hold on to, asserted together:
    a store that migrated but silently lost a relationship, or landed on
    a version its schema does not match, is not a success."""
    facts = _db_facts(db_path)
    assert facts["integrity_check"] == "ok"
    assert facts["user_version"] == STORE_SCHEMA_VERSION
    assert set(_V7_TABLE_NAMES) <= facts["tables"]
    assert "role" in facts["memory_evidence_columns"]
    assert set(expected_ids) <= set(facts["memory_ids"])
    assert len(facts["memory_ids"]) == len(expected_ids) + extra_memories
    # The v1->v2 backfill writes one `memory_recorded` event per
    # pre-existing memory. Executed twice it would double them, which is
    # the one way this defect could ever have damaged canonical data.
    assert facts["memory_recorded_events"] == len(expected_ids) + extra_memories
    if expected_links is not None:
        assert facts["memory_evidence_rows"] == expected_links
    return facts


# ---------------------------------------------------------------------------
# Real-process helpers. Threads would not reproduce this: the boundary
# under test is SQLite's inter-PROCESS write lock, and a Python-level
# lock would mask exactly the defect. `spawn` gives each worker its own
# interpreter state and its own connections, as separate `cortex`
# invocations really have.
# ---------------------------------------------------------------------------


def _worker_read(workspace_dir, _payload, barrier, queue):
    try:
        barrier.wait(timeout=60)
        from cortex_memory import Cortex as _Cortex

        cortex = _Cortex.open(workspace_dir)
        queue.put(("ok", f"state={len(cortex.state())} timeline={len(cortex.timeline())}"))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _worker_remember(workspace_dir, content, barrier, queue):
    try:
        barrier.wait(timeout=60)
        from cortex_memory import Cortex as _Cortex

        queue.put(("ok", _Cortex.open(workspace_dir).remember(content, kind="note").memory_id))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _worker_recall(workspace_dir, _payload, barrier, queue):
    try:
        barrier.wait(timeout=60)
        from cortex_memory import Cortex as _Cortex

        queue.put(("ok", f"recall={len(_Cortex.open(workspace_dir).recall('canonical'))}"))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_workers(workspace_dir, target, payloads, *, timeout=120):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(payloads))
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=target, args=(workspace_dir, payload, barrier, queue)) for payload in payloads
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for process in processes:
        process.join(timeout=timeout)
        assert process.exitcode == 0, f"worker process exited with {process.exitcode}"
    return results


def _assert_no_worker_failed(results):
    errors = [message for status, message in results if status == "error"]
    assert not errors, errors


# ---------------------------------------------------------------------------
# The primary regression: an existing v4 store, six concurrent processes.
# ---------------------------------------------------------------------------


class TestConcurrentV4Migration:
    def test_a_six_concurrent_readers_migrate_v4_to_v7_exactly_once(self, tmp_path):
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        results = _run_workers(tmp_path, _worker_read, [None] * _PROC_COUNT)

        _assert_no_worker_failed(results)
        assert {payload for _, payload in results} == {"state=3 timeline=3"}
        _assert_healthy_v7(db_path, memory_ids, expected_links=len(memory_ids))

        # Pre-existing links are backfilled to 'related', never invented
        # as 'supporting' -- the migration's own contract, which must
        # survive being executed under contention.
        reopened = Cortex.open(tmp_path)
        for memory in reopened.state():
            assert memory.supporting_evidence_ids == ()
            assert len(memory.evidence_ids) == 1

    def test_b_concurrent_writes_during_migration(self, tmp_path):
        """The invariant lives in store initialization, not in the
        caller, so a burst of writers must be as safe as a burst of
        readers -- and their writes must all land."""
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        contents = [f"written during the v4 migration by worker {i}" for i in range(_PROC_COUNT)]
        results = _run_workers(tmp_path, _worker_remember, contents)

        _assert_no_worker_failed(results)
        assert len({payload for _, payload in results}) == _PROC_COUNT
        _assert_healthy_v7(db_path, memory_ids, extra_memories=_PROC_COUNT)

    def test_c_concurrent_recalls_reach_a_usable_search_index(self, tmp_path):
        """A20.R.1's derived boundary sits immediately after this one on
        the same open path: the canonical chain must be committed before
        the index is built, and the index must end up usable."""
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        results = _run_workers(tmp_path, _worker_recall, [None] * _PROC_COUNT)

        _assert_no_worker_failed(results)
        facts = _assert_healthy_v7(db_path, memory_ids)
        assert SEARCH_INDEX_TABLE in facts["tables"]

        store = MemoryStore.open_if_exists(db_path)
        with store:
            assert store.fts_enabled is True
        assert len(Cortex.open(tmp_path).recall("canonical")) == len(memory_ids)


# ---------------------------------------------------------------------------
# The full chain: v1 -> v7 under contention. This is the only starting
# point that exercises a data migration (the event backfill), so it is
# where a repair that let two processes both run a step would show up as
# damaged canonical data rather than only as a failed open.
# ---------------------------------------------------------------------------


class TestConcurrentV1FullChain:
    def test_a_six_concurrent_readers_traverse_the_whole_chain(self, tmp_path):
        db_path, memory_ids = _prepare_workspace(tmp_path, 1)

        results = _run_workers(tmp_path, _worker_read, [None] * _PROC_COUNT)

        _assert_no_worker_failed(results)
        assert {payload for _, payload in results} == {"state=3 timeline=3"}
        # v1 held no evidence links at all, and the chain must not invent any.
        _assert_healthy_v7(db_path, memory_ids, expected_links=0)

        reopened = Cortex.open(tmp_path)
        assert sorted(m.memory_id for m in reopened.timeline()) == sorted(memory_ids)

    def test_b_the_event_backfill_happens_exactly_once(self, tmp_path):
        """The discriminating assertion for canonical damage: v1->v2
        inserts one `memory_recorded` event per pre-existing memory. Two
        processes both executing that step would double every event, and
        `timeline()`/`state()` are event-log projections, so the damage
        would be silent and permanent."""
        db_path, memory_ids = _prepare_workspace(tmp_path, 1, count=5)

        _assert_no_worker_failed(_run_workers(tmp_path, _worker_read, [None] * _PROC_COUNT))

        facts = _assert_healthy_v7(db_path, memory_ids)
        assert facts["memory_recorded_events"] == 5
        connection = sqlite3.connect(db_path)
        try:
            duplicated = connection.execute(
                "SELECT subject_id FROM events GROUP BY subject_id, kind HAVING COUNT(*) > 1"
            ).fetchall()
        finally:
            connection.close()
        assert duplicated == []

    def test_c_concurrent_writes_across_the_whole_chain(self, tmp_path):
        db_path, memory_ids = _prepare_workspace(tmp_path, 1)

        contents = [f"written during the v1 chain by worker {i}" for i in range(_PROC_COUNT)]
        results = _run_workers(tmp_path, _worker_remember, contents)

        _assert_no_worker_failed(results)
        assert len({payload for _, payload in results}) == _PROC_COUNT
        _assert_healthy_v7(db_path, memory_ids, extra_memories=_PROC_COUNT)


# ---------------------------------------------------------------------------
# The locking protocol itself, deterministically -- no timing, no sleeps.
# ---------------------------------------------------------------------------


class _StaleVersionConnection(sqlite3.Connection):
    """Models the LOSING process exactly: its unlocked entry read of
    `PRAGMA user_version` reports the old version, and by the time it
    could act on it another opener has already migrated the store all
    the way to v7.

    The hook fires on the FIRST unlocked read only, so any re-read
    performed under the lock sees the real state -- which is precisely
    what the repair must consult, and what the pre-repair code never
    asked for.
    """

    winner = None
    fired = False

    def execute(self, sql, *args, **kwargs):
        normalized = " ".join(sql.split()).lower()
        if normalized == "pragma user_version" and not _StaleVersionConnection.fired:
            _StaleVersionConnection.fired = True
            cursor = super().execute(sql, *args, **kwargs)
            rows = cursor.fetchall()
            _StaleVersionConnection.winner()  # another opener migrates, right here
            return _FrozenCursor(rows)
        return super().execute(sql, *args, **kwargs)


class _FrozenCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _RecordingConnection(sqlite3.Connection):
    """Records the statements `_ensure_schema` issues, in order, so the
    ordering the invariant demands can be asserted directly."""

    log = None

    def execute(self, sql, *args, **kwargs):
        _RecordingConnection.log.append(" ".join(sql.split()))
        return super().execute(sql, *args, **kwargs)


class TestMigrationProtocol:
    def test_a_a_stale_version_never_decides_the_migration(self, tmp_path):
        """THE discriminating test of this repair, and it needs no
        concurrency at all: the losing process is reproduced exactly,
        deterministically, by making its unlocked read stale.

        Against the pre-repair baseline this fails with
        "duplicate column name: role" -- the reported incident -- because
        the decision taken before the lock is never revisited after it.
        """
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        def winner_migrates_the_store():
            with MemoryStore.create_or_open(db_path):
                pass

        _StaleVersionConnection.winner = winner_migrates_the_store
        _StaleVersionConnection.fired = False
        connection = sqlite3.connect(db_path, factory=_StaleVersionConnection)
        try:
            from cortex_memory._store import _ensure_schema

            _ensure_schema(connection)  # must not raise: the store is intact
            assert _StaleVersionConnection.fired, "the unlocked read never happened; test is vacuous"
            # No transaction may outlive the boundary: one left open here
            # would stall every other process for this connection's life.
            assert connection.in_transaction is False
        finally:
            connection.close()
            _StaleVersionConnection.winner = None

        _assert_healthy_v7(db_path, memory_ids, expected_links=len(memory_ids))

    def test_b_every_migration_step_decides_under_the_write_lock(self, tmp_path):
        """Fixes the invariant itself: each canonical DDL is preceded, in
        its own transaction, by `BEGIN IMMEDIATE` and by a re-read of
        `PRAGMA user_version` taken after that lock was acquired.

        Asserted on the statement stream rather than on internals, so it
        constrains the protocol without pinning how the loop is written.
        """
        db_path, _ = _prepare_workspace(tmp_path, 1)

        _RecordingConnection.log = []
        connection = sqlite3.connect(db_path, factory=_RecordingConnection)
        try:
            from cortex_memory._store import _ensure_schema

            _ensure_schema(connection)
        finally:
            connection.close()
        log = _RecordingConnection.log
        _RecordingConnection.log = None

        stamps = [i for i, sql in enumerate(log) if sql.startswith("PRAGMA user_version =")]
        assert len(stamps) == 6, f"expected one stamp per v1->v7 step, saw {len(stamps)}: {log}"

        for stamp in stamps:
            begins = [i for i, sql in enumerate(log[:stamp]) if sql.upper().startswith("BEGIN")]
            assert begins, "a migration step ran outside any transaction"
            begin = begins[-1]
            assert log[begin].upper() == "BEGIN IMMEDIATE", (
                f"step at {stamp} was opened with {log[begin]!r}: a DEFERRED transaction takes no "
                "lock until its first write, so the decision to run the step was unprotected"
            )
            rereads = [i for i in range(begin, stamp) if log[i] == "PRAGMA user_version"]
            assert rereads, (
                f"step at {stamp} never re-read PRAGMA user_version between acquiring the lock at "
                f"{begin} and stamping: the decision rests on a version read before the lock"
            )
            # and the version that decided the step is the one read under the lock
            assert rereads[-1] < stamp

    def test_c_a_healthy_v7_store_takes_no_write_lock_and_is_not_mutated(self, tmp_path):
        """The common case must stay cheap: an up-to-date store must not
        serialize every opener behind a write lock it has no use for,
        and must not be touched."""
        cortex = Cortex.init(tmp_path, "dev")
        cortex.remember("a fact recorded at the current schema version", kind="note")
        db_path = db_path_for(tmp_path / ".cortex")
        before = _db_facts(db_path)

        _RecordingConnection.log = []
        connection = sqlite3.connect(db_path, factory=_RecordingConnection)
        try:
            from cortex_memory._store import _ensure_schema

            _ensure_schema(connection)
            assert connection.in_transaction is False
        finally:
            connection.close()
        log = _RecordingConnection.log
        _RecordingConnection.log = None

        assert not [sql for sql in log if sql.upper().startswith("BEGIN")], log
        assert not [sql for sql in log if sql.startswith("PRAGMA user_version =")], log
        assert _db_facts(db_path) == before


# ---------------------------------------------------------------------------
# Serialized migration from every remaining starting version. Concurrency
# is pinned by v1 and v4 above; what these check is that each helper is
# still correct after losing ownership of its transaction and its version
# stamp.
# ---------------------------------------------------------------------------


class TestEveryStartingVersionStillMigrates:
    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
    def test_a_serial_open_migrates_and_preserves_data(self, tmp_path, version):
        db_path, memory_ids = _prepare_workspace(tmp_path, version)

        cortex = Cortex.open(tmp_path)
        assert sorted(m.memory_id for m in cortex.state()) == sorted(memory_ids)

        _assert_healthy_v7(db_path, memory_ids, expected_links=0 if version == 1 else len(memory_ids))

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
    def test_b_reopening_a_migrated_store_changes_nothing(self, tmp_path, version):
        """Migration must be a one-time transition, not something the
        open path redoes or re-stamps on every use."""
        db_path, _ = _prepare_workspace(tmp_path, version)
        Cortex.open(tmp_path).state()
        after_first = _db_facts(db_path)

        Cortex.open(tmp_path).state()

        assert _db_facts(db_path) == after_first


# ---------------------------------------------------------------------------
# Fail-closed: concurrency safety must not have been bought by making
# Cortex more permissive about stores that really are malformed.
# ---------------------------------------------------------------------------


class TestMalformedStoreStillFailsClosed:
    def test_a_v4_store_with_an_incompatible_role_column_is_rejected(self, tmp_path):
        """A store stamped v4 whose `memory_evidence` already carries a
        `role` column in an incompatible shape (nullable, no default) is
        NOT a store someone else migrated -- it is a corrupt one. Under
        the lock its version is still 4, so the step is genuinely
        eligible, runs, and collides. Suppressing that collision with
        `IF NOT EXISTS`, or by catching "duplicate column", would have
        closed the race by opening this store silently.
        """
        db_path, _ = _prepare_workspace(tmp_path, 4)
        connection = sqlite3.connect(db_path)
        try:
            with connection:
                connection.execute("ALTER TABLE memory_evidence ADD COLUMN role TEXT")
                connection.execute("PRAGMA user_version = 4")
        finally:
            connection.close()

        with pytest.raises(CortexStorageError) as excinfo:
            Cortex.open(tmp_path).state()
        assert "duplicate column name: role" in str(excinfo.value)

    def test_b_a_store_missing_the_tables_its_version_claims_is_rejected(self, tmp_path):
        db_path, _ = _prepare_workspace(tmp_path, 4)
        connection = sqlite3.connect(db_path)
        try:
            with connection:
                connection.execute("DROP TABLE skills")
        finally:
            connection.close()

        with pytest.raises(CortexStorageError) as excinfo:
            Cortex.open(tmp_path).state()
        assert "possibly corrupted store" in str(excinfo.value)

    def test_c_a_version_this_build_knows_no_step_for_is_rejected(self, tmp_path):
        """The loop must not spin, and must not silently accept, a store
        stamped by a future Cortex."""
        db_path, _ = _prepare_workspace(tmp_path, 4)
        connection = sqlite3.connect(db_path)
        try:
            with connection:
                connection.execute("PRAGMA user_version = 99")
        finally:
            connection.close()

        with pytest.raises(CortexStorageError) as excinfo:
            Cortex.open(tmp_path).state()
        assert "not supported by this version of Cortex" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Crash / rollback: the DDL and the version that records it live in the
# same transaction, so they can never be observed disagreeing.
# ---------------------------------------------------------------------------


class _FailBeforeStampConnection(sqlite3.Connection):
    """Fails a migration step after its DDL has run but before the
    version stamp -- the exact window in which a crashed process could
    leave a half-migrated schema behind."""

    def execute(self, sql, *args, **kwargs):
        if " ".join(sql.split()) == "PRAGMA user_version = 5":
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return super().execute(sql, *args, **kwargs)


class TestCrashMidMigration:
    def test_a_failure_before_the_version_stamp_rolls_the_step_back(self, tmp_path):
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        connection = sqlite3.connect(db_path, factory=_FailBeforeStampConnection)
        try:
            from cortex_memory._store import _ensure_schema

            with pytest.raises(sqlite3.OperationalError):
                _ensure_schema(connection)
        finally:
            connection.close()

        facts = _db_facts(db_path)
        assert facts["user_version"] == 4
        assert "role" not in facts["memory_evidence_columns"]
        assert facts["integrity_check"] == "ok"
        assert facts["memory_ids"] == sorted(memory_ids)

    def test_b_a_later_open_completes_the_migration(self, tmp_path):
        """A store left behind by a crashed migration is not damaged, it
        is merely still old: the next open must simply retry."""
        db_path, memory_ids = _prepare_workspace(tmp_path, 4)

        connection = sqlite3.connect(db_path, factory=_FailBeforeStampConnection)
        try:
            from cortex_memory._store import _ensure_schema

            with pytest.raises(sqlite3.OperationalError):
                _ensure_schema(connection)
        finally:
            connection.close()

        assert sorted(m.memory_id for m in Cortex.open(tmp_path).state()) == sorted(memory_ids)
        _assert_healthy_v7(db_path, memory_ids, expected_links=len(memory_ids))
