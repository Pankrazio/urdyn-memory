"""Exceptional recovery of a damaged FTS5 search index.

`open()`, `verify_storage()` and `@guarded` stay unchanged: they refuse a
Brain whose FTS5 projection is damaged. `urdyn.maintenance` is the only way
past that refusal, and only for that cause. Covered: recovery of a corrupted
index; canonical corruption (alone and mixed with index damage); wrong schema
version and wrong identity binding; an ordinary table (or unknown object)
named like the index; errors and process crashes before and after the
commit; failed safety copy; runtime without FTS5; WAL content in the safety
copy; lifecycle-lock contention; concurrent writers and a concurrent restore.

Synthetic Brains in temporary directories only.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid

import pytest

from urdyn import Urdyn, UrdynStorageCorruptError, UrdynStorageError, UrdynStorageSchemaError, UrdynStorageUnavailableError
from urdyn import backup, maintenance
from urdyn._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL, ENTITY_SOURCE
from urdyn._store import SEARCH_INDEX_TABLE, MemoryStore

TERMS = {ENTITY_MEMORY: "zebrafact", ENTITY_ATTEMPT: "quokkatask", ENTITY_SKILL: "ibexskill",
         ENTITY_SOURCE: "narwhalnote"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _brain(root, *, seed=True, extra=40):
    cx = Urdyn.create(root, "dev")
    cx.remember(f"The deploy {TERMS[ENTITY_MEMORY]} needs a migration first.", kind="note")
    for i in range(extra):
        cx.remember(f"synthetic canonical fact number {i} about the pipeline", kind="note")
    cx.record_attempt(task=f"Run the {TERMS[ENTITY_ATTEMPT]} migration", approach="direct", outcome="failed")
    lesson = cx.learn("Migrations must be applied before the deploy.")
    cx.promote(lesson, name=f"Apply {TERMS[ENTITY_SKILL]} migrations", purpose="Safe schema changes",
               steps=["apply", "verify"], conditions=["before deploy"])
    if seed:
        (root / "notes.md").write_text(f"Project note: the {TERMS[ENTITY_SOURCE]} service.\n", encoding="utf-8")
        cx.seed(["notes.md"])
    return cx


def _sql(db_path, statement, params=()):
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            return connection.execute(statement, params).fetchall()
    finally:
        connection.close()


def _damage_index(db_path):
    _sql(db_path, f"DELETE FROM {SEARCH_INDEX_TABLE}_data WHERE id > 10")
    messages = [r[0] for r in _sql(db_path, "PRAGMA quick_check")]
    assert messages and all(m.startswith("fts5:") for m in messages), messages


def _damage_canonical_page(db_path, table="memories"):
    """Give a canonical table's root page an invalid b-tree page type."""
    (rootpage,) = _sql(db_path, "SELECT rootpage FROM sqlite_master WHERE name = ?", (table,))[0]
    assert rootpage > 1
    (page_size,) = _sql(db_path, "PRAGMA page_size")[0]
    with open(db_path, "r+b") as stream:
        stream.seek((rootpage - 1) * page_size)
        stream.write(b"\x07")
    try:
        messages = [r[0] for r in _sql(db_path, "PRAGMA quick_check")]
    except sqlite3.DatabaseError:
        return  # SQLite refuses to even scan it: canonical damage, not FTS
    assert any(not m.startswith("fts5:") for m in messages if m != "ok"), messages


def _canonical(db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dump = [line for line in connection.iterdump()
                if SEARCH_INDEX_TABLE not in line and not line.startswith("PRAGMA writable_schema")]
        identity = connection.execute("SELECT * FROM brain_identity").fetchall()
    finally:
        connection.close()
    return dump, identity


def _files(root):
    urdyn_dir = root / ".urdyn"
    return {p.name: p.read_bytes() for p in sorted(urdyn_dir.iterdir()) if p.is_file()}


def _maintenance_dir(root):
    return root.parent / maintenance.MAINTENANCE_DIRNAME


def _assert_refused_by_open(root):
    with pytest.raises(UrdynStorageCorruptError):
        Urdyn.open(root)


def _assert_nothing_changed(root, before_files):
    assert _files(root) == before_files


def _assert_healthy(root, canonical_before):
    cx = Urdyn.open(root)
    assert cx.search_index_state().status == "usable"
    assert _canonical(cx._db_path) == canonical_before
    with MemoryStore.open_existing(cx._db_path) as store:
        for entity_type, term in TERMS.items():
            if entity_type == ENTITY_SOURCE and not (root / "notes.md").exists():
                continue
            assert len(store.search_candidates(frozenset({term}), entity_type)) == 1, entity_type


# --------------------------------------------------------------------------
# 1. Recovery of a corrupted FTS index
# --------------------------------------------------------------------------


def test_a_damaged_index_is_refused_by_ordinary_paths_and_recovered_by_maintenance(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    canonical = _canonical(cx._db_path)
    manifest = (root / ".urdyn" / "manifest.json").read_bytes()
    root_entries = sorted(p.name for p in root.iterdir())
    _damage_index(cx._db_path)

    _assert_refused_by_open(root)
    with pytest.raises(UrdynStorageCorruptError):
        cx.state()  # @guarded: no general bypass
    with pytest.raises(UrdynStorageCorruptError):
        cx.verify_storage()
    assert maintenance.inspect_search_index(root).status == "invalid"

    recovery = maintenance.recover_search_index(root)

    assert recovery.mode == "recovered" and recovery.state.status == "usable"
    _assert_healthy(root, canonical)
    assert (root / ".urdyn" / "manifest.json").read_bytes() == manifest
    # The safety copy: outside the Brain root, private, Git-ignored, a faithful copy.
    copy = recovery.backup_path
    assert copy.parent == _maintenance_dir(root)
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert stat.S_IMODE(copy.parent.stat().st_mode) == 0o700
    assert (copy.parent / ".gitignore").read_text().splitlines()[-1] == "*"
    assert _canonical(copy) == canonical
    assert any(m.startswith("fts5:") for (m,) in _sql(copy, "PRAGMA quick_check"))
    # Nothing was added to the Brain root (an ordinary backup refuses unknown entries there).
    assert sorted(p.name for p in root.iterdir()) == root_entries


def test_the_handle_methods_use_the_same_entry_and_check_the_handle_identity(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    canonical = _canonical(cx._db_path)
    _damage_index(cx._db_path)
    assert cx.search_index_state().status == "invalid"
    assert cx.rebuild_search_index().status == "usable"
    _assert_healthy(root, canonical)


def test_recovery_on_a_healthy_brain_is_the_ordinary_rebuild_without_a_copy(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    recovery = maintenance.recover_search_index(root)
    assert (recovery.mode, recovery.backup_path, recovery.state.status) == ("rebuilt", None, "usable")
    assert not _maintenance_dir(root).exists()


# --------------------------------------------------------------------------
# 2. Refusals: canonical corruption, schema, binding, unexpected objects
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index_damaged_too", [False, True])
def test_canonical_corruption_is_refused_and_nothing_is_touched(tmp_path, index_damaged_too):
    root = tmp_path / "brain"
    cx = _brain(root)
    if index_damaged_too:
        _damage_index(cx._db_path)
    _damage_canonical_page(cx._db_path)
    before = _files(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.recover_search_index(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.inspect_search_index(root)
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


@pytest.mark.parametrize("table", ["memories", "evidence", "brain_identity"])
def test_mixed_fts_and_canonical_messages_are_refused(tmp_path, table):
    """quick_check REPORTS (does not raise) canonical damage next to FTS5
    damage: one non-FTS message is enough to refuse."""
    import struct

    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    (rootpage,) = _sql(cx._db_path, "SELECT rootpage FROM sqlite_master WHERE name = ?", (table,))[0]
    (page_size,) = _sql(cx._db_path, "PRAGMA page_size")[0]
    with open(cx._db_path, "r+b") as stream:  # first-freeblock offset pointing past the page
        stream.seek((rootpage - 1) * page_size + 1)
        stream.write(struct.pack(">H", page_size - 2))
    messages = [r[0] for r in _sql(cx._db_path, "PRAGMA quick_check(10000)")]
    assert any(m.startswith("fts5:") for m in messages) and any(not m.startswith("fts5:") for m in messages)
    before = _files(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.recover_search_index(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.inspect_search_index(root)
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


def test_a_wrong_schema_version_is_refused(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    _sql(cx._db_path, "PRAGMA user_version = 6")
    before = _files(root)
    with pytest.raises(UrdynStorageSchemaError):
        maintenance.recover_search_index(root)
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


def test_a_wrong_identity_binding_is_refused(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    manifest_path = root / ".urdyn" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["cortex_id"] = uuid.uuid4().hex
    manifest_path.write_text(json.dumps(data))
    before = _files(root)
    with pytest.raises(UrdynStorageError):
        maintenance.recover_search_index(root)
    with pytest.raises(UrdynStorageError):
        cx.rebuild_search_index()  # the handle's identity no longer matches the manifest
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


def test_an_ordinary_table_named_like_the_index_is_never_dropped(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    _sql(cx._db_path, f"DROP TABLE {SEARCH_INDEX_TABLE}")
    _sql(cx._db_path, f"CREATE TABLE {SEARCH_INDEX_TABLE} (entity_type, entity_id, text)")
    _sql(cx._db_path, f"INSERT INTO {SEARCH_INDEX_TABLE} VALUES ('memory', 'x', 'a row someone stored here')")
    before = _files(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.recover_search_index(root)
    with pytest.raises(UrdynStorageCorruptError):
        cx.rebuild_search_index()
    _assert_nothing_changed(root, before)
    assert _sql(cx._db_path, f"SELECT text FROM {SEARCH_INDEX_TABLE}") == [("a row someone stored here",)]


def test_an_unknown_object_next_to_a_damaged_index_is_refused(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    _sql(cx._db_path, f"CREATE TABLE {SEARCH_INDEX_TABLE}_notes (x)")
    _damage_index(cx._db_path)
    before = _files(root)
    with pytest.raises(UrdynStorageCorruptError):
        maintenance.recover_search_index(root)
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


def test_without_fts5_nothing_is_copied_or_written(tmp_path, monkeypatch):
    import urdyn._store as store_module

    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    before = _files(root)
    monkeypatch.setattr(store_module, "fts5_runtime_available", lambda: False)
    with pytest.raises(UrdynStorageUnavailableError):
        maintenance.recover_search_index(root)
    _assert_nothing_changed(root, before)
    assert not _maintenance_dir(root).exists()


# --------------------------------------------------------------------------
# 3. Safety copy failures: nothing changed, no partial copy left
# --------------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["not_a_directory", "read_only", "copy_mismatch"])
def test_a_failed_safety_copy_changes_nothing(tmp_path, monkeypatch, failure):
    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    before = _files(root)
    directory = _maintenance_dir(root)
    if failure == "not_a_directory":
        directory.write_text("not a directory")
    elif failure == "read_only":
        directory.mkdir(mode=0o700)
        (directory / ".gitignore").write_text("*\n")
        directory.chmod(0o500)
    else:
        original = maintenance._canonical_fingerprint
        calls = []

        def differs_on_the_copy(connection):
            calls.append(1)
            value = original(connection)
            return value + "x" if len(calls) == 1 else value  # first call fingerprints the copy

        monkeypatch.setattr(maintenance, "_canonical_fingerprint", differs_on_the_copy)
    try:
        with pytest.raises(UrdynStorageUnavailableError):
            maintenance.recover_search_index(root)
    finally:
        if failure == "read_only":
            directory.chmod(0o700)
    _assert_nothing_changed(root, before)
    _assert_refused_by_open(root)
    if directory.is_dir():
        assert [p.name for p in directory.iterdir() if p.suffix == ".db"] == []


# --------------------------------------------------------------------------
# 4. Errors before the commit: rolled back, index still damaged, copy kept
# --------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["rebuild", "validation"])
def test_an_error_before_the_commit_rolls_back(tmp_path, monkeypatch, where):
    import urdyn._store as store_module

    root = tmp_path / "brain"
    cx = _brain(root)
    canonical = _canonical(cx._db_path)
    _damage_index(cx._db_path)
    before = _files(root)
    if where == "rebuild":
        def explode(connection):
            raise sqlite3.OperationalError("disk I/O error")
        monkeypatch.setattr(store_module, "_rebuild_search_index", explode)
    else:
        original = store_module._rebuild_search_index

        def incomplete(connection):
            original(connection)
            connection.execute(f"DELETE FROM {SEARCH_INDEX_TABLE} WHERE rowid = (SELECT MIN(rowid) FROM {SEARCH_INDEX_TABLE})")
        monkeypatch.setattr(store_module, "_rebuild_search_index", incomplete)
    with pytest.raises(UrdynStorageError):
        maintenance.recover_search_index(root)
    _assert_nothing_changed(root, before)
    _assert_refused_by_open(root)
    copies = list(_maintenance_dir(root).glob("*.db"))
    assert len(copies) == 1 and _canonical(copies[0]) == canonical
    monkeypatch.undo()
    assert maintenance.recover_search_index(root).state.status == "usable"
    _assert_healthy(root, canonical)


def test_a_commit_failure_without_commit_reports_and_changes_nothing(tmp_path, monkeypatch):
    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    before = _files(root)

    def failing_commit(connection):
        raise sqlite3.OperationalError("disk I/O error during commit")

    monkeypatch.setattr(maintenance, "_commit", failing_commit)
    with pytest.raises(UrdynStorageError, match="could not be confirmed"):
        maintenance.recover_search_index(root)
    _assert_nothing_changed(root, before)
    _assert_refused_by_open(root)


def test_a_commit_that_succeeds_but_reports_an_error_is_confirmed(tmp_path, monkeypatch):
    root = tmp_path / "brain"
    cx = _brain(root)
    canonical = _canonical(cx._db_path)
    _damage_index(cx._db_path)

    def commit_then_fail(connection):
        connection.commit()
        raise sqlite3.OperationalError("I/O error reported after the commit")

    monkeypatch.setattr(maintenance, "_commit", commit_then_fail)
    assert maintenance.recover_search_index(root).state.status == "usable"
    _assert_healthy(root, canonical)


# --------------------------------------------------------------------------
# 5. Process crashes before and after the commit
# --------------------------------------------------------------------------

_CHILD = r"""
import os, sys
import urdyn._store as store_module
from urdyn import maintenance
root, when = sys.argv[1], sys.argv[2]
if when == "during_rebuild":
    original = store_module._rebuild_search_index
    def dying(connection):
        original(connection)
        os._exit(17)
    store_module._rebuild_search_index = dying
elif when == "after_drop":
    # Dies right after DROP + integrity check + CREATE, before populating.
    store_module._rebuild_search_index = lambda connection: os._exit(17)
elif when == "after_commit":
    def commit_and_die(connection):
        connection.commit()
        os._exit(17)
    maintenance._commit = commit_and_die
maintenance.recover_search_index(root)
os._exit(0)
"""


@pytest.mark.parametrize("when", ["after_drop", "during_rebuild", "after_commit"])
def test_a_process_killed_during_recovery(tmp_path, when):
    root = tmp_path / "brain"
    cx = _brain(root)
    canonical = _canonical(cx._db_path)
    _damage_index(cx._db_path)
    before = _files(root)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.run([sys.executable, "-c", _CHILD, str(root), when], env=env,
                           capture_output=True, text=True, timeout=60)
    assert child.returncode == 17, child.stderr
    if when == "after_commit":
        _assert_healthy(root, canonical)
    else:
        # The dead process leaves a hot journal; SQLite rolls it back on the
        # next write access. Until then readers see the pre-crash state.
        assert (root / ".urdyn" / "memory.db-journal").exists()
        _assert_refused_by_open(root)  # still damaged: nothing was published
        assert _files(root)["manifest.json"] == before["manifest.json"]
        assert _canonical(cx._db_path) == canonical
        assert maintenance.recover_search_index(root).state.status == "usable"
        assert not (root / ".urdyn" / "memory.db-journal").exists()
        _assert_healthy(root, canonical)
    # One verified safety copy per attempt that reached the copy step.
    assert len(list(_maintenance_dir(root).glob("*.db"))) == (1 if when == "after_commit" else 2)


# --------------------------------------------------------------------------
# 6. WAL: the safety copy includes committed WAL content
# --------------------------------------------------------------------------


def test_the_safety_copy_includes_wal_content(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    writer = sqlite3.connect(cx._db_path)
    try:
        assert writer.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        with writer:
            writer.execute(f"DELETE FROM {SEARCH_INDEX_TABLE}_data WHERE id > 10")
            writer.execute("UPDATE memories SET content = content WHERE 0")  # no-op touch
        wal = cx._db_path.with_name(cx._db_path.name + "-wal")
        assert wal.exists() and wal.stat().st_size > 0
        main_file_canonical = sqlite3.connect(f"file:{cx._db_path}?mode=ro&nolock=1&immutable=1", uri=True)
        try:
            # Read the main file alone (WAL ignored): the damage is only in the WAL.
            assert main_file_canonical.execute(f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE}_data").fetchone()[0] > 11
        finally:
            main_file_canonical.close()
        recovery = maintenance.recover_search_index(root)
    finally:
        writer.close()
    copy = recovery.backup_path
    assert recovery.mode == "recovered"
    # The copy holds the WAL state (the damage), not the stale main file.
    assert _sql(copy, f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE}_data")[0][0] <= 11
    assert Urdyn.open(root).search_index_state().status == "usable"


# --------------------------------------------------------------------------
# 7. Concurrency: lock contention, writers, restore
# --------------------------------------------------------------------------


def test_recovery_waits_for_the_lifecycle_lock(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    _damage_index(cx._db_path)
    code = ("import sys\nfrom urdyn._lifecycle import lifecycle\n"
            "with lifecycle(sys.argv[1]):\n    print('held', flush=True)\n    sys.stdin.readline()\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    holder = subprocess.Popen([sys.executable, "-c", code, str(root)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, env=env)
    result = {}
    try:
        assert holder.stdout.readline().strip() == "held"
        worker = threading.Thread(target=lambda: result.setdefault("r", maintenance.recover_search_index(root)))
        worker.start()
        time.sleep(0.7)
        assert worker.is_alive() and "r" not in result  # blocked on the lock, nothing done yet
        assert not _maintenance_dir(root).exists()
        holder.stdin.write("\n")
        holder.stdin.flush()
        worker.join(timeout=60)
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=10)
    assert result["r"].state.status == "usable"


def _concurrent_worker(root, action, barrier, queue):
    try:
        barrier.wait(timeout=30)
        from urdyn import Urdyn as _Urdyn, maintenance as _maintenance, backup as _backup

        if action == "recover":
            queue.put(("ok", action, _maintenance.recover_search_index(root).mode))
        elif action.startswith("restore:"):
            queue.put(("ok", action, _backup.restore_backup(root, action.split(":", 1)[1])))
        else:
            queue.put(("ok", action, _Urdyn.open(root).remember(action, kind="note").memory_id))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        queue.put(("error", action, f"{type(exc).__name__}: {exc}"))


def _run(root, actions, timeout=120):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(len(actions)), ctx.Queue()
    processes = [ctx.Process(target=_concurrent_worker, args=(str(root), a, barrier, queue)) for a in actions]
    for p in processes:
        p.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for p in processes:
        p.join(timeout=timeout)
        assert p.exitcode == 0
    return results


def test_concurrent_writers_either_are_refused_before_or_succeed_after_recovery(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root)
    count_before = len(cx.timeline())
    _damage_index(cx._db_path)
    writes = [f"concurrent write {i} capybaraterm" for i in range(5)]
    results = _run(root, ["recover", *writes])
    recover = [r for r in results if r[1] == "recover"][0]
    assert recover[0] == "ok", recover
    for status, action, detail in results:
        if action != "recover" and status == "error":
            assert detail.startswith("UrdynStorageCorruptError"), detail  # refused while damaged, never half-done
    succeeded = [r for r in results if r[1] != "recover" and r[0] == "ok"]
    reopened = Urdyn.open(root)
    assert len(reopened.timeline()) == count_before + len(succeeded)
    assert reopened.search_index_state().status == "usable"


def test_a_concurrent_restore_and_recovery_are_serialized(tmp_path):
    root = tmp_path / "brain"
    cx = _brain(root, seed=False)
    canonical = _canonical(cx._db_path)
    archive = tmp_path / "healthy.zip"
    backup.create_backup(root, archive)
    _damage_index(cx._db_path)
    results = _run(root, ["recover", f"restore:{archive}"])
    assert all(status == "ok" for status, _, _ in results), results
    reopened = Urdyn.open(root)
    assert reopened.search_index_state().status == "usable"
    assert _canonical(reopened._db_path) == canonical
