"""Legacy Cortex stores are refused, never migrated -- also under concurrency.

History. A20.R.2 serialized the implicit v1->v7 migration chain that every
open used to run (`_ensure_schema`: `BEGIN IMMEDIATE`, re-read of `PRAGMA
user_version` under the lock, one step per transaction). Implicit migration
has since been removed from every public path: schemas v1-v6 only ever
existed in unreleased development builds named Cortex (`.cortex/`), never in
a tagged release (see `docs/compatibility.md`). The internal chain is still
in `_store.py`; no public entry may reach it.

What this file pins now, on AUTHENTIC Cortex Brains written by the historical
code itself (tests/data/legacy_cortex_brains, SHA-256 checked before use):

* as written (`.cortex/`) a legacy Brain is not recognized (`UrdynNotFoundError`
  or `UrdynManifestError`), and its bytes are never touched;
* the same bytes under `.urdyn/` are refused fail-closed
  (`UrdynStorageSchemaError`) by every public entry -- open, init, every
  operation, the FTS maintenance entry, backup -- with no byte changed, no
  side file left, no table created, no event backfilled;
* refusals hold under six concurrent processes (readers, writers, recalls,
  maintenance), and come from a read-only inspection (proved with read-only
  files and a spy on the internal migration chain);
* the three malformed-store cases A20.R.2 kept failing closed still do, earlier.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sqlite3
import stat
import tarfile
from pathlib import Path

import pytest

from urdyn import (
    Urdyn,
    UrdynManifestError,
    UrdynNotFoundError,
    UrdynStorageSchemaError,
)
from urdyn import backup, maintenance
from urdyn._store import db_path_for

_PROC_COUNT = 6
_FIXTURES = Path(__file__).parent / "data" / "legacy_cortex_brains"
_VERSIONS = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Authentic legacy Brains
# ---------------------------------------------------------------------------


def _verified_archive(version):
    archive = _FIXTURES / f"v{version}.tar.gz"
    sums = dict(line.split()[::-1] for line in (_FIXTURES / "SHA256SUMS").read_text().splitlines() if line.strip())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == sums[archive.name], "fixture is not the authentic one"
    return archive


def _legacy_brain(tmp_path, version, *, layout="urdyn"):
    """Extract the authentic Cortex Brain of `version`. `layout="cortex"`
    keeps it exactly as written; `layout="urdyn"` moves the same bytes under
    `.urdyn/` (what a user renaming the directory would get)."""
    target = tmp_path / f"v{version}-{layout}"
    target.mkdir()
    with tarfile.open(_verified_archive(version)) as archive:
        archive.extractall(target, filter="data")
    root = target / "brain"
    generated = json.loads((target / "generate.json").read_text())
    if layout == "urdyn":
        (root / ".cortex").rename(root / ".urdyn")
    return root, generated


def _snapshot(root):
    """Every entry under the Brain root (names, types, bytes) plus its parent's
    entry names: a refusal must leave all of it exactly as it was."""
    entries = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        entries[relative] = path.read_bytes() if path.is_file() else "dir"
    # `.urdyn-locks/` is the lifecycle lock directory every coordinated entry
    # creates next to the Brain (derived, EMPTY operational files).
    # It is excluded from the comparison only after checking it holds nothing
    # else; anything else appearing in the parent (e.g. `.urdyn-maintenance`) counts.
    locks = root.parent / ".urdyn-locks"
    if locks.exists():
        for lock in locks.iterdir():
            assert lock.name == ".gitignore" or (lock.name.startswith(".urdyn-lock-") and lock.stat().st_size == 0)
    return entries, sorted(p.name for p in root.parent.iterdir() if p.name != ".urdyn-locks")


def _store_facts(db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "tables": sorted(r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")),
            "memories": sorted(connection.execute("SELECT memory_id, content FROM memories").fetchall()),
            "integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
        }
    finally:
        connection.close()


def _legacy_db(root):
    return db_path_for(root / ".urdyn")


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



# ---------------------------------------------------------------------------
# Real-process workers (spawn: separate interpreters and connections, as
# separate `urdyn` invocations have).
# ---------------------------------------------------------------------------


def _worker(workspace_dir, action, barrier, queue):
    try:
        barrier.wait(timeout=60)
        from urdyn import Urdyn as _Urdyn, maintenance as _maintenance

        if action == "read":
            cx = _Urdyn.open(workspace_dir)
            queue.put(("ok", f"state={len(cx.state())}"))
        elif action == "recall":
            queue.put(("ok", f"recall={len(_Urdyn.open(workspace_dir).recall('canonical'))}"))
        elif action == "maintenance":
            queue.put(("ok", _maintenance.recover_search_index(workspace_dir).mode))
        else:
            queue.put(("ok", _Urdyn.open(workspace_dir).remember(action, kind="note").memory_id))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        queue.put(("error", type(exc).__name__))


def _run_workers(workspace_dir, actions, *, timeout=120):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(actions))
    queue = ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(str(workspace_dir), a, barrier, queue)) for a in actions]
    for process in processes:
        process.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for process in processes:
        process.join(timeout=timeout)
        assert process.exitcode == 0, f"worker process exited with {process.exitcode}"
    return results


# ---------------------------------------------------------------------------
# 1. Concurrent openers of a legacy store: all refused, nothing changes.
#    Replaces TestConcurrentV4Migration (a, b, c) and TestConcurrentV1FullChain
#    (a, b, c), which required the store to be migrated exactly once.
# ---------------------------------------------------------------------------


class TestConcurrentOpenersAreRefused:
    @pytest.mark.parametrize("kind", ["read", "write", "recall", "maintenance"])
    @pytest.mark.parametrize("version", _VERSIONS)
    def test_six_concurrent_processes_are_all_refused_and_nothing_changes(self, tmp_path, version, kind):
        root, generated = _legacy_brain(tmp_path, version)
        before = _snapshot(root)
        facts = _store_facts(_legacy_db(root))
        actions = [f"written by worker {i}" if kind == "write" else kind for i in range(_PROC_COUNT)]

        results = _run_workers(root, actions)

        assert results == [("error", "UrdynStorageSchemaError")] * _PROC_COUNT
        assert _snapshot(root) == before  # no byte, no journal/WAL, no maintenance copy, no new entry
        assert _store_facts(_legacy_db(root)) == facts
        assert {memory_id for memory_id, _ in facts["memories"]} == set(generated["written"])

    def test_no_event_is_backfilled_into_a_v1_store(self, tmp_path):
        """A20.R.2's sharpest invariant was "the v1->v2 event backfill runs
        exactly once". Retired migration means it runs ZERO times: a v1 store
        keeps having no `events` table after any number of concurrent opens."""
        root, _ = _legacy_brain(tmp_path, 1)
        assert "events" not in _store_facts(_legacy_db(root))["tables"]
        _run_workers(root, ["read"] * _PROC_COUNT)
        _run_workers(root, ["written during the old chain"] * _PROC_COUNT)
        assert "events" not in _store_facts(_legacy_db(root))["tables"]


# ---------------------------------------------------------------------------
# 2. No public entry reaches the internal migration chain.
#    Replaces TestMigrationProtocol (a, b): they pinned the locking protocol
#    of `_ensure_schema`, which no public path calls any more.
# ---------------------------------------------------------------------------


class TestNoPublicEntryMigrates:
    @pytest.mark.parametrize("version", _VERSIONS)
    def test_every_public_entry_refuses_without_touching_the_migration_chain(self, tmp_path, monkeypatch, version):
        import urdyn._store as store_module

        reached = []

        def spy(*args, **kwargs):
            reached.append(args)
            raise AssertionError("the internal migration chain was reached")

        for name in ("_ensure_schema", "_migrate_v1_to_v2", "_migrate_v2_to_v3", "_migrate_v3_to_v4",
                     "_migrate_v4_to_v5", "_migrate_v5_to_v6", "_migrate_v6_to_v7"):
            monkeypatch.setattr(store_module, name, spy)

        root, _ = _legacy_brain(tmp_path, version)
        before = _snapshot(root)
        entries = {
            "open": lambda: Urdyn.open(root),
            "init": lambda: Urdyn.init(root, "general"),
            "discover": lambda: Urdyn.discover(root),
            "inspect_search_index": lambda: maintenance.inspect_search_index(root),
            "recover_search_index": lambda: maintenance.recover_search_index(root),
        }
        for name, call in entries.items():
            with pytest.raises(UrdynStorageSchemaError):
                call()
        with pytest.raises(backup.BackupError) as refused_backup:
            backup.create_backup(root, tmp_path / f"backup-v{version}.zip")
        assert isinstance(refused_backup.value.__cause__, UrdynStorageSchemaError)
        assert not (tmp_path / f"backup-v{version}.zip").exists()
        assert reached == []
        assert _snapshot(root) == before

    @pytest.mark.parametrize("version", _VERSIONS)
    def test_the_refusal_is_a_read_only_inspection(self, tmp_path, version):
        """With the store, its directory and the root read-only, the refusal is
        still the schema refusal -- not a failed write -- so the path that
        refuses never tried to write."""
        root, _ = _legacy_brain(tmp_path, version)
        urdyn_dir = root / ".urdyn"
        modes = {p: stat.S_IMODE(p.stat().st_mode) for p in (root, urdyn_dir, *urdyn_dir.iterdir())}
        try:
            for path in modes:
                path.chmod(0o555 if path.is_dir() else 0o444)
            before = _snapshot(root)
            with pytest.raises(UrdynStorageSchemaError):
                Urdyn.open(root)
            with pytest.raises(UrdynStorageSchemaError):
                maintenance.inspect_search_index(root)
            assert _snapshot(root) == before
        finally:
            for path, mode in modes.items():
                path.chmod(mode)


# ---------------------------------------------------------------------------
# 3. Serial use, every starting version.
#    Replaces TestEveryStartingVersionStillMigrates (a x6, b x6) and
#    TestCrashMidMigration (a, b): no step runs, so there is no partial
#    migration to roll back or to complete; what remains to pin is that the
#    refusal is total, repeatable and leaves no trace.
# ---------------------------------------------------------------------------


class TestEveryStartingVersionIsRefused:
    @pytest.mark.parametrize("version", _VERSIONS)
    def test_a_legacy_brain_as_written_is_not_recognized_and_untouched(self, tmp_path, version):
        root, generated = _legacy_brain(tmp_path, version, layout="cortex")
        before = _snapshot(root)
        with pytest.raises(UrdynNotFoundError):
            Urdyn.open(root)
        with pytest.raises(UrdynNotFoundError):
            Urdyn.init(root, "general")
        with pytest.raises(UrdynManifestError):
            maintenance.recover_search_index(root)
        assert _snapshot(root) == before
        db = root / ".cortex" / "memory.db"
        assert {m for m, _ in _store_facts(db)["memories"]} == set(generated["written"])

    @pytest.mark.parametrize("version", _VERSIONS)
    def test_repeated_refusals_leave_the_data_exactly_as_written(self, tmp_path, version):
        root, generated = _legacy_brain(tmp_path, version)
        before = _snapshot(root)
        facts = _store_facts(_legacy_db(root))
        assert facts["user_version"] == version and facts["integrity"] == "ok"
        for _ in range(3):
            with pytest.raises(UrdynStorageSchemaError):
                Urdyn.open(root).state()
            with pytest.raises(UrdynStorageSchemaError):
                Urdyn.init(root, "general")
        assert _snapshot(root) == before
        assert _store_facts(_legacy_db(root)) == facts
        assert {m for m, _ in facts["memories"]} == set(generated["written"])
        # No hot journal or WAL to roll back: a plain read-write open finds nothing pending.
        assert sorted(p.name for p in (root / ".urdyn").iterdir()) == ["manifest.json", "memory.db"]


# ---------------------------------------------------------------------------
# 4. Locking protocol of a healthy current store (kept unchanged from A20.R.2).
# ---------------------------------------------------------------------------


class _RecordingConnection(sqlite3.Connection):
    """Records the statements `_ensure_schema` issues, in order, so the
    ordering the invariant demands can be asserted directly."""

    log = None

    def execute(self, sql, *args, **kwargs):
        _RecordingConnection.log.append(" ".join(sql.split()))
        return super().execute(sql, *args, **kwargs)


class TestMigrationProtocol:
    def test_c_a_healthy_v7_store_takes_no_write_lock_and_is_not_mutated(self, tmp_path):
        """The common case must stay cheap: an up-to-date store must not
        serialize every opener behind a write lock it has no use for,
        and must not be touched."""
        urdyn = Urdyn.init(tmp_path, "dev")
        urdyn.remember("a fact recorded at the current schema version", kind="note")
        db_path = db_path_for(tmp_path / ".urdyn")
        before = _db_facts(db_path)

        _RecordingConnection.log = []
        connection = sqlite3.connect(db_path, factory=_RecordingConnection)
        try:
            from urdyn._store import _ensure_schema

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
# 5. Fail-closed for malformed stores (kept, adapted to authentic fixtures).
#    A20.R.2 required them to be rejected by the migration step; they are now
#    rejected before any step, by the schema review, and stay byte-identical.
# ---------------------------------------------------------------------------


def _malformed(tmp_path, statements):
    root, _ = _legacy_brain(tmp_path, 4)
    connection = sqlite3.connect(_legacy_db(root))
    try:
        with connection:
            for statement in statements:
                connection.execute(statement)
    finally:
        connection.close()
    return root


class TestMalformedStoreStillFailsClosed:
    def test_a_v4_store_with_an_incompatible_role_column_is_rejected(self, tmp_path):
        """A store stamped v4 whose `memory_evidence` already carries an
        incompatible `role` column is a corrupt store, not a migrated one: it
        must never be opened silently. Refused at the schema review,
        before any step could collide; the column and the stamp stay as found."""
        root = _malformed(tmp_path, ["ALTER TABLE memory_evidence ADD COLUMN role TEXT", "PRAGMA user_version = 4"])
        before = _snapshot(root)
        with pytest.raises(UrdynStorageSchemaError):
            Urdyn.open(root).state()
        with pytest.raises(UrdynStorageSchemaError):
            maintenance.recover_search_index(root)
        assert _snapshot(root) == before
        connection = sqlite3.connect(f"file:{_legacy_db(root)}?mode=ro", uri=True)
        try:
            columns = [r[1] for r in connection.execute("PRAGMA table_info(memory_evidence)")]
            assert columns.count("role") == 1
            assert connection.execute("PRAGMA user_version").fetchone() == (4,)
        finally:
            connection.close()

    def test_b_a_store_missing_the_tables_its_version_claims_is_rejected(self, tmp_path):
        root = _malformed(tmp_path, ["DROP TABLE skills"])
        before = _snapshot(root)
        with pytest.raises(UrdynStorageSchemaError):
            Urdyn.open(root).state()
        assert _snapshot(root) == before
        assert "skills" not in _store_facts(_legacy_db(root))["tables"]  # not recreated

    def test_c_a_version_this_build_knows_no_step_for_is_rejected(self, tmp_path):
        """A store stamped by a future Urdyn must not be accepted or touched."""
        root = _malformed(tmp_path, ["PRAGMA user_version = 99"])
        before = _snapshot(root)
        with pytest.raises(UrdynStorageSchemaError):
            Urdyn.open(root).state()
        assert _snapshot(root) == before
        assert _store_facts(_legacy_db(root))["user_version"] == 99
