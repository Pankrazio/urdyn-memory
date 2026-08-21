"""A20.R.1: concurrent first initialization of the derived FTS5 search
index must be serialized by SQLite, not by timing.

`_ensure_search_index()` used to decide whether to create the index
OUTSIDE any lock and then open a plain (DEFERRED) transaction, which
takes no lock until its first write. Two processes opening a store whose
index does not exist yet could therefore both pass that check and both
reach `CREATE VIRTUAL TABLE search_index`; the loser failed with
"table search_index already exists", surfaced as
`CortexStorageError: ... is corrupted` -- an open failing on a store with
nothing wrong with it, and a message inviting the user to discard a
perfectly intact database.

The repair re-checks under `BEGIN IMMEDIATE`, the same protocol
`_ensure_schema` already applies to canonical first creation (A18.1),
kept as a SEPARATE critical section because the canonical version chain
and this rebuildable projection are different boundaries with different
failure semantics (see `_store.py`'s module docstring).

Two exposures existed, and the second was by far the larger:

- fresh database, 6 concurrent first writers: ~19% of rounds failed.
  A18.1's canonical `BEGIN IMMEDIATE` staggers the processes, which
  narrows -- but never closes -- the window on the derived check.
- ESTABLISHED store already at v7 whose index is absent (a pre-A7
  store, or one first opened by a SQLite build without FTS5), 6
  concurrent openers: ~82% of rounds failed, up to 4 of 6 processes in
  the same round, because nothing serializes them beforehand. That is
  the real upgrade path, and `TestExistingStoreIndexAbsent` below is
  the discriminating test of this repair.

The race is open-time, so it hits READS too (`recall`/`state`/
`timeline`/`preflight`/`guard` all reach `_ensure_schema`), not only
`remember()`.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex, CortexStorageError
from cortex_memory._retrieval import ENTITY_MEMORY
from cortex_memory._store import SEARCH_INDEX_TABLE, STORE_SCHEMA_VERSION, MemoryStore, db_path_for

_PROC_COUNT = 6


# ---------------------------------------------------------------------------
# Real-process helpers. Threads would not reproduce any of this: the
# boundary being tested is SQLite's inter-PROCESS write lock, and a
# Python-level lock would mask exactly the defect under test. `spawn` is
# used so each worker builds its own interpreter state and its own
# connections, as separate `cortex` invocations really do.
# ---------------------------------------------------------------------------


def _worker_remember(workspace_dir, content, barrier, queue):
    try:
        barrier.wait(timeout=30)
        from cortex_memory import Cortex as _Cortex

        memory = _Cortex.open(workspace_dir).remember(content, kind="note")
        queue.put(("ok", memory.memory_id))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_workers(workspace_dir, contents, *, timeout=60):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(contents))
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker_remember, args=(workspace_dir, content, barrier, queue))
        for content in contents
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


def _db_facts(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return {
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "memories": connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "events": connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "index_rows": connection.execute(
                f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE} WHERE entity_type = ?", (ENTITY_MEMORY,)
            ).fetchone()[0],
        }
    finally:
        connection.close()


def _index_is_usable(db_path, terms):
    """The index must be QUERYABLE, not merely present: a table that
    exists but was left empty or half-built by a losing process would
    pass an existence check and still have lost the projection."""
    store = MemoryStore.open_if_exists(db_path)
    with store:
        assert store.fts_enabled is True
        return store.search_candidates(frozenset(terms), ENTITY_MEMORY)


def _drop_search_index(db_path):
    """Reduce a valid, fully-migrated store to the state a pre-A7 store
    (or one first opened without FTS5 support) is really in: canonical
    data at the current schema version, derived projection absent."""
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(f"DROP TABLE {SEARCH_INDEX_TABLE}")
        (version,) = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()
    assert version == STORE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# The discriminating scenario: established v7 store, index absent.
# ---------------------------------------------------------------------------


class TestExistingStoreIndexAbsent:
    def test_a_six_concurrent_openers_initialize_the_index_exactly_once(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("canonical fact that predates the derived index", kind="note")
        db_path = db_path_for(tmp_path / ".cortex")
        _drop_search_index(db_path)

        contents = [f"post-upgrade fact number {i}" for i in range(_PROC_COUNT)]
        results = _run_workers(tmp_path, contents)

        _assert_no_worker_failed(results)
        assert len({memory_id for _, memory_id in results}) == _PROC_COUNT

        facts = _db_facts(db_path)
        assert facts["integrity_check"] == "ok"
        assert facts["user_version"] == STORE_SCHEMA_VERSION
        assert facts["memories"] == _PROC_COUNT + 1
        assert facts["events"] == _PROC_COUNT + 1
        # Exactly one initial build, fully backfilled and never
        # truncated by a second process racing behind the winner.
        assert facts["index_rows"] == _PROC_COUNT + 1

        assert _index_is_usable(db_path, {"predates"}) != []

    def test_b_read_only_openers_do_not_fail_on_a_missing_index(self, tmp_path):
        """The race is open-time, so it never needed a write to trigger:
        `open_if_exists` reaches the same `_ensure_schema`. Reads must
        not fail because another process is building the projection."""
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a fact to recall after the index is rebuilt", kind="note")
        db_path = db_path_for(tmp_path / ".cortex")
        _drop_search_index(db_path)

        results = _run_workers(tmp_path, ["a concurrent write alongside the readers"] * _PROC_COUNT)
        _assert_no_worker_failed(results)

        reopened = Cortex.open(tmp_path)
        assert len(reopened.state()) == 2
        assert len(reopened.timeline()) == 2
        assert [m.content for m in reopened.recall("recall")] == [
            "a fact to recall after the index is rebuilt"
        ]


# ---------------------------------------------------------------------------
# Fresh database: first creation of canonical schema AND derived index
# in the same burst.
# ---------------------------------------------------------------------------


class TestFreshDatabase:
    def test_a_identical_first_writes_from_six_processes(self, tmp_path):
        Cortex.init(tmp_path, "dev")
        db_path = db_path_for(tmp_path / ".cortex")
        assert not db_path.exists()

        results = _run_workers(tmp_path, ["the same first fact"] * _PROC_COUNT)

        _assert_no_worker_failed(results)
        # A17 canonical idempotency is unaffected by the repair: six
        # identical remembers remain ONE fact, not six.
        memory_ids = {memory_id for _, memory_id in results}
        assert len(memory_ids) == 1

        facts = _db_facts(db_path)
        assert facts["integrity_check"] == "ok"
        assert facts["user_version"] == STORE_SCHEMA_VERSION
        assert facts["memories"] == 1
        assert facts["events"] == 1
        assert facts["index_rows"] == 1

        assert [entity_id for entity_id, _ in _index_is_usable(db_path, {"first"})] == [
            next(iter(memory_ids))
        ]

    def test_b_distinct_first_writes_from_six_processes(self, tmp_path):
        """Pressure-tests schema init, derived init, canonical writes and
        search indexing at once: every process both creates and writes."""
        Cortex.init(tmp_path, "dev")
        db_path = db_path_for(tmp_path / ".cortex")
        assert not db_path.exists()

        contents = [f"distinct first fact number {i}" for i in range(_PROC_COUNT)]
        results = _run_workers(tmp_path, contents)

        _assert_no_worker_failed(results)
        memory_ids = {memory_id for _, memory_id in results}
        assert len(memory_ids) == _PROC_COUNT

        facts = _db_facts(db_path)
        assert facts["integrity_check"] == "ok"
        assert facts["user_version"] == STORE_SCHEMA_VERSION
        assert facts["memories"] == _PROC_COUNT
        assert facts["events"] == _PROC_COUNT
        assert facts["index_rows"] == _PROC_COUNT

        # every distinct memory is individually findable through the index
        for index in range(_PROC_COUNT):
            found = _index_is_usable(db_path, {f"{index}"})
            assert len(found) == 1


# ---------------------------------------------------------------------------
# The locking protocol itself, deterministically -- no timing, no sleeps.
# ---------------------------------------------------------------------------


class TestLockingProtocol:
    def test_a_loser_rechecks_under_the_lock_and_never_attempts_create(self, tmp_path, monkeypatch):
        """Models the losing process exactly: its UNLOCKED entry check
        reports the index absent, but by the time it holds the write
        lock the winner has already created it. It must re-read the real
        state under the lock and return without touching the projection.

        `_try_create_search_index` is replaced with a fuse: reaching it
        at all means the re-check did not happen, which is the defect.
        """
        import cortex_memory._store as store_module

        cx = Cortex.init(tmp_path, "dev")
        cx.remember("index already built by the winner", kind="note")

        calls = {"exists": 0}
        real_table_exists = store_module._table_exists

        def stale_first_check(connection, name):
            if name == SEARCH_INDEX_TABLE:
                calls["exists"] += 1
                if calls["exists"] == 1:
                    return False  # the unlocked check, already stale
            return real_table_exists(connection, name)

        def must_not_be_reached(connection):
            raise AssertionError("CREATE attempted despite the index existing under the lock")

        monkeypatch.setattr(store_module, "_table_exists", stale_first_check)
        monkeypatch.setattr(store_module, "_try_create_search_index", must_not_be_reached)

        connection = sqlite3.connect(cx._db_path)
        try:
            store_module._ensure_search_index(connection)
            assert calls["exists"] >= 2, "the existence check did not run again under the lock"
            # The write lock must not be held past the early return: a
            # transaction left open here would stall every other process
            # for the whole life of this connection.
            assert connection.in_transaction is False
        finally:
            connection.close()

        other = sqlite3.connect(cx._db_path, timeout=1)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.rollback()
        finally:
            other.close()

    def test_b_winner_leaves_no_open_transaction_either(self, tmp_path):
        import cortex_memory._store as store_module

        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a fact", kind="note")
        _drop_search_index(cx._db_path)

        connection = sqlite3.connect(cx._db_path)
        try:
            store_module._ensure_search_index(connection)
            assert connection.in_transaction is False
            assert store_module._table_exists(connection, SEARCH_INDEX_TABLE)
        finally:
            connection.close()

    def test_c_no_fts5_support_still_opens_and_holds_no_lock(self, tmp_path, monkeypatch):
        """The graceful-degradation path runs inside the new IMMEDIATE
        transaction now, so it must still commit out of it cleanly
        instead of leaving the store's write lock held with no index to
        show for it."""
        import cortex_memory._store as store_module

        monkeypatch.setattr(store_module, "_try_create_search_index", lambda connection: False)

        cx = Cortex.init(tmp_path, "dev")
        cx.remember("recorded without any FTS5 support", kind="note")

        store = MemoryStore.open_if_exists(cx._db_path)
        with store:
            assert store.fts_enabled is False
            assert store._connection.in_transaction is False

        other = sqlite3.connect(cx._db_path, timeout=1)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.rollback()
        finally:
            other.close()


# ---------------------------------------------------------------------------
# Failure of the winner mid-initialization.
# ---------------------------------------------------------------------------


class TestWinnerFailureRollsBack:
    def test_a_failed_initial_backfill_leaves_no_partial_projection(self, tmp_path, monkeypatch):
        """If the winner dies between CREATE and the initial backfill,
        SQLite's transactional DDL must take the whole projection with
        it -- all six FTS5 shadow objects included -- so no later opener
        can mistake a half-built index for a complete one. Canonical
        data is untouched, and the next open simply builds it again.
        No retry machinery is involved.
        """
        import cortex_memory._store as store_module

        cx = Cortex.init(tmp_path, "dev")
        cx.remember("canonical content that must survive intact", kind="note")
        _drop_search_index(cx._db_path)

        def explode(connection):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(store_module, "_rebuild_search_index", explode)

        with pytest.raises(CortexStorageError):
            MemoryStore.create_or_open(cx._db_path)

        connection = sqlite3.connect(cx._db_path)
        try:
            leftovers = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE ?", (f"{SEARCH_INDEX_TABLE}%",)
                ).fetchall()
            ]
            (memory_count,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
            (integrity,) = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()

        assert leftovers == []
        assert memory_count == 1
        assert integrity == "ok"

        # unpatched, the very next open rebuilds it completely
        monkeypatch.undo()
        assert _index_is_usable(cx._db_path, {"canonical"}) != []


# ---------------------------------------------------------------------------
# The repair must not become a validity claim.
# ---------------------------------------------------------------------------


class TestExistingObjectIsNotValidated:
    def test_a_an_incompatible_search_index_is_not_treated_as_usable(self, tmp_path):
        """Re-checking existence under a lock answers "does an object
        with this name exist", never "is it the projection Cortex
        expects". Deliberately unchanged by this repair, and the reason
        `CREATE ... IF NOT EXISTS` was NOT adopted: it would suppress
        the collision silently at the DDL level too. A structurally
        wrong `search_index` must still fail loudly when queried, not
        quietly return no candidates.
        """
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a real canonical fact", kind="note")

        connection = sqlite3.connect(cx._db_path)
        try:
            with connection:
                connection.execute(f"DROP TABLE {SEARCH_INDEX_TABLE}")
                connection.execute(f"CREATE TABLE {SEARCH_INDEX_TABLE} (wrong_column TEXT)")
        finally:
            connection.close()

        store = MemoryStore.open_if_exists(cx._db_path)
        with store:
            with pytest.raises(CortexStorageError):
                store.search_candidates(frozenset({"canonical"}), ENTITY_MEMORY)

        # canonical reads are unaffected by a broken derived projection
        assert len(Cortex.open(tmp_path).state()) == 1


# ---------------------------------------------------------------------------
# Migration path: the derived boundary is reached from every version.
# ---------------------------------------------------------------------------


_CREATE_MEMORIES_V2_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY, content TEXT NOT NULL, kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL, recorded_at TEXT NOT NULL, supersedes TEXT
    )
"""
_CREATE_EVIDENCE_SQL = """
    CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY, content TEXT NOT NULL, kind TEXT NOT NULL, recorded_at TEXT NOT NULL
    )
"""
_CREATE_MEMORY_EVIDENCE_SQL = """
    CREATE TABLE memory_evidence (
        memory_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (memory_id, evidence_id)
    )
"""
_CREATE_EVENTS_SQL = """
    CREATE TABLE events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
        subject_id TEXT NOT NULL, occurred_at TEXT NOT NULL
    )
"""
_CREATE_ATTEMPTS_SQL = """
    CREATE TABLE attempts (
        attempt_id TEXT PRIMARY KEY, task TEXT NOT NULL, approach TEXT NOT NULL, outcome TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""
_CREATE_ATTEMPT_EVIDENCE_SQL = """
    CREATE TABLE attempt_evidence (
        attempt_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (attempt_id, evidence_id)
    )
"""
_CREATE_SKILLS_SQL = """
    CREATE TABLE skills (
        skill_id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL,
        verification_state TEXT NOT NULL, source_lesson_id TEXT NOT NULL, recorded_at TEXT NOT NULL
    )
"""
_CREATE_SKILL_STEPS_SQL = """
    CREATE TABLE skill_steps (
        skill_id TEXT NOT NULL, step TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, position)
    )
"""
_CREATE_SKILL_CONDITIONS_SQL = """
    CREATE TABLE skill_conditions (
        skill_id TEXT NOT NULL, condition TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, position)
    )
"""
_CREATE_SKILL_EVIDENCE_SQL = """
    CREATE TABLE skill_evidence (
        skill_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, evidence_id)
    )
"""


def _build_standalone_v4_database(db_path, *, content):
    """A complete hand-written v4 database, never touching current
    creation code -- the same approach `test_search_index.py` and
    `test_migration_v4.py` take. Used to prove the derived-index
    boundary behaves identically on a store that arrived at v7 through
    the MIGRATION chain rather than through fresh creation.
    """
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    memory_id = uuid.uuid4().hex
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            for statement in (
                _CREATE_MEMORIES_V2_SQL,
                _CREATE_EVIDENCE_SQL,
                _CREATE_MEMORY_EVIDENCE_SQL,
                _CREATE_EVENTS_SQL,
                _CREATE_ATTEMPTS_SQL,
                _CREATE_ATTEMPT_EVIDENCE_SQL,
                _CREATE_SKILLS_SQL,
                _CREATE_SKILL_STEPS_SQL,
                _CREATE_SKILL_CONDITIONS_SQL,
                _CREATE_SKILL_EVIDENCE_SQL,
            ):
                connection.execute(statement)
            connection.execute("PRAGMA user_version = 4")
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, 'note', 'user_asserted', ?, NULL)",
                (memory_id, content, recorded_at),
            )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) "
                "VALUES (?, 'memory_recorded', ?, ?)",
                (uuid.uuid4().hex, memory_id, recorded_at),
            )
    finally:
        connection.close()
    return memory_id


def test_migrated_store_with_absent_index_initializes_it_once_under_concurrency(tmp_path):
    """A store that reached v7 through the v4->v7 MIGRATION chain, not
    through fresh creation, must hit the same repaired derived-index
    boundary. The migration itself is performed serially first (a single
    open), so this test isolates the derived boundary -- concurrent
    execution of the migration chain is a separate, pre-existing defect
    at a different critical section (covered separately)
    and is deliberately not what is asserted here.
    """
    Cortex.init(tmp_path, "dev")
    db_path = db_path_for(tmp_path / ".cortex")
    legacy_id = _build_standalone_v4_database(db_path, content="a pre-A7 lesson worth finding")

    migrated = Cortex.open(tmp_path)
    assert len(migrated.state()) == 1
    _drop_search_index(db_path)

    contents = [f"post-migration fact number {i}" for i in range(_PROC_COUNT)]
    results = _run_workers(tmp_path, contents)

    _assert_no_worker_failed(results)

    facts = _db_facts(db_path)
    assert facts["integrity_check"] == "ok"
    assert facts["user_version"] == STORE_SCHEMA_VERSION
    assert facts["memories"] == _PROC_COUNT + 1
    assert facts["events"] == _PROC_COUNT + 1
    assert facts["index_rows"] == _PROC_COUNT + 1

    # the migrated pre-A7 memory is backfilled into the index exactly once
    assert [entity_id for entity_id, _ in _index_is_usable(db_path, {"pre"})] == [legacy_id]
