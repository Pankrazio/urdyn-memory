"""The canonical store is bound to the Brain's logical identity.

Contract under test (see `docs/storage-contract.md`, "Brain identity"):

* A new Brain stores, inside `memory.db` itself (table `brain_identity`,
  additive: `user_version` stays 7 so that earlier builds, 0.4.0 included,
  keep opening it, which a rollback to an earlier build requires), the same
  32-hex identity its manifest carries (`cortex_id`). The binding is
  immutable; its presence is the contract marker.
* Opening, every guarded operation, recovery and backup/restore compare the
  binding with the manifest / the open handle and refuse a discordance
  BEFORE any canonical data is used, without writing anything.
* Absent or malformed binding (corruption) is reported differently from a
  well-formed binding that names another Brain (discordance).
* Legacy Brains have no binding: they stay usable exactly as before and are
  never converted, bound or rewritten implicitly. Documented limit: a
  binding table removed by deliberate SQL looks exactly like a legacy store.
* Moving, backing up and restoring keep the identity; a full copy is a
  replica of the same identity (not detectable, by design); a copied
  manifest or a copied database is detected.

Synthetic Brains in temporary directories only. Crashes are real process
deaths (`os._exit`) at the named step, not exceptions raised after the fact.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from urdyn import Urdyn
from urdyn import _errors
from urdyn._errors import (UrdynError, UrdynStorageCorruptError, UrdynStorageError,
                           UrdynStorageSchemaError)
from urdyn.backup import BackupError, create_backup, inspect_backup, recover_interrupted_restore, restore_backup

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _identity_error():
    # Resolved at call time so that, run against a build without the identity
    # contract, each test fails on its own assertion instead of the whole
    # module failing to import.
    return _errors.UrdynStorageIdentityError


def tree(path: Path) -> dict:
    """Every entry under `path` (and `path` itself): kind, size, sha256."""
    result = {}
    if not os.path.lexists(path):
        return result
    for entry in [path, *sorted(path.rglob("*"))]:
        key = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            result[key] = ("link", os.readlink(entry))
        elif entry.is_dir():
            result[key] = ("dir",)
        else:
            data = entry.read_bytes()
            result[key] = ("file", len(data), hashlib.sha256(data).hexdigest())
    return result


def manifest_id(root: Path) -> str:
    return json.loads((root / ".urdyn/manifest.json").read_text())["cortex_id"]


def db_facts(db_path: Path):
    """(user_version, identity rows or None if there is no binding table),
    read-only."""
    connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        try:
            rows = connection.execute("SELECT * FROM brain_identity").fetchall()
        except sqlite3.OperationalError:
            rows = None
        return version, rows
    finally:
        connection.close()


def drop_identity_triggers(connection):
    for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'brain_identity'").fetchall():
        connection.execute(f'DROP TRIGGER "{name}"')


def make_legacy_unbound(root: Path) -> None:
    """Replace the Brain's store with what builds before the identity binding
    created: the v7 layout, no identity binding (the same `_ensure_schema` path
    store creation used then)."""
    from urdyn._store import _ensure_schema
    db_path = root / ".urdyn/memory.db"
    db_path.unlink()
    with db_path.open("xb"):
        pass
    connection = sqlite3.connect(db_path)
    try:
        _ensure_schema(connection)
    finally:
        connection.close()
    assert db_facts(db_path) == (7, None)


def child(code: str, *args, expected: int = 0):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-c", code, *map(str, args)], env=env,
                             capture_output=True, timeout=60)
    assert process.returncode == expected, process.stderr.decode()
    return process


@pytest.fixture
def two_brains(tmp_path):
    a = tmp_path / "brain-a"
    b = tmp_path / "brain-b"
    Urdyn.create(a).remember("memory of A")
    Urdyn.create(b).remember("memory of B")
    return a, b


# ---------------------------------------------------------------------------
# identity of new Brains
# ---------------------------------------------------------------------------


def test_new_brain_binds_its_canonical_store_to_the_manifest_identity(tmp_path):
    root = tmp_path / "brain"
    engine = Urdyn.create(root)
    version, rows = db_facts(root / ".urdyn/memory.db")
    assert version == 7  # additive binding: STORE_SCHEMA_VERSION unchanged (rollback compatibility)
    assert rows == [(1, manifest_id(root))]
    assert engine.urdyn_id == manifest_id(root)


def test_two_independent_brains_have_distinct_bound_identities(two_brains):
    a, b = two_brains
    assert manifest_id(a) != manifest_id(b)
    for root in (a, b):
        assert db_facts(root / ".urdyn/memory.db")[1] == [(1, manifest_id(root))]


def test_init_of_an_absent_directory_binds_the_identity_too(tmp_path):
    root = tmp_path / "fresh"
    Urdyn.init(root, "dev")
    assert db_facts(root / ".urdyn/memory.db") == (7, [(1, manifest_id(root))])


def test_binding_is_immutable_inside_sqlite(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root)
    db_path = root / ".urdyn/memory.db"
    assert db_facts(db_path) == (7, [(1, manifest_id(root))])  # not vacuous: the table exists
    before = tree(root)
    connection = sqlite3.connect(db_path)
    try:
        for statement in ("UPDATE brain_identity SET brain_id = '" + "0" * 32 + "'",
                          "DELETE FROM brain_identity",
                          "INSERT INTO brain_identity VALUES (2, '" + "1" * 32 + "')",
                          "INSERT INTO brain_identity VALUES (1, '" + "1" * 32 + "')",
                          "INSERT OR REPLACE INTO brain_identity VALUES (1, '" + "1" * 32 + "')",
                          "UPDATE OR REPLACE brain_identity SET brain_id = '" + "1" * 32 + "'"):
            with pytest.raises(sqlite3.DatabaseError):
                with connection:
                    connection.execute(statement)
    finally:
        connection.close()
    assert tree(root) == before
    Urdyn.open(root).state()


def test_manifest_copied_from_another_brain_is_refused_without_writes(two_brains):
    a, b = two_brains
    shutil.copyfile(a / ".urdyn/manifest.json", b / ".urdyn/manifest.json")
    before = tree(b)
    for attempt in (lambda: Urdyn.open(b), lambda: Urdyn.init(b), lambda: Urdyn.discover(b)):
        with pytest.raises(_identity_error()):
            attempt()
        assert tree(b) == before
    assert Urdyn.open(a).state()[0].content == "memory of A"


def test_database_copied_without_its_manifest_is_refused_without_writes(two_brains):
    a, b = two_brains
    shutil.copyfile(b / ".urdyn/memory.db", a / ".urdyn/memory.db")
    before = tree(a)
    with pytest.raises(_identity_error()):
        Urdyn.open(a)
    assert tree(a) == before


def test_discordance_is_not_reported_as_corruption(two_brains):
    a, b = two_brains
    shutil.copyfile(a / ".urdyn/manifest.json", b / ".urdyn/manifest.json")
    with pytest.raises(UrdynStorageError) as caught:
        Urdyn.open(b)
    assert isinstance(caught.value, _identity_error())
    assert not isinstance(caught.value, UrdynStorageCorruptError)


def test_full_copy_is_a_replica_of_the_same_identity(tmp_path):
    original = tmp_path / "brain"
    Urdyn.create(original).remember("shared past")
    replica = tmp_path / "replica"
    shutil.copytree(original, replica)
    assert Urdyn.open(replica).urdyn_id == Urdyn.open(original).urdyn_id
    assert Urdyn.open(replica).state()[0].content == "shared past"


def test_moved_brain_keeps_its_identity(tmp_path):
    root = tmp_path / "here"
    identity = Urdyn.create(root).urdyn_id
    moved = tmp_path / "elsewhere" / "renamed"
    moved.parent.mkdir()
    root.rename(moved)
    engine = Urdyn.open(moved)
    assert engine.urdyn_id == identity
    engine.remember("after the move")
    assert db_facts(moved / ".urdyn/memory.db") == (7, [(1, identity)])


def _tamper_binding(db_path: Path, how: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        # External manipulation, not Urdyn: it bypasses the table's own CHECKs
        # and triggers (which keep Urdyn's code from writing such values).
        connection.execute("PRAGMA ignore_check_constraints = ON")
        with connection:
            drop_identity_triggers(connection)
            if how == "row-missing":
                connection.execute("DELETE FROM brain_identity")
            elif how == "table-missing":
                connection.execute("DROP TABLE brain_identity")
            elif how == "malformed":
                connection.execute("UPDATE brain_identity SET brain_id = 'not-an-identity'")
            elif how == "uppercase":
                connection.execute("UPDATE brain_identity SET brain_id = upper(brain_id)")
            elif how == "duplicated":
                (identity,) = connection.execute("SELECT brain_id FROM brain_identity").fetchone()
                connection.execute("DROP TABLE brain_identity")
                connection.execute("CREATE TABLE brain_identity (singleton INTEGER, brain_id TEXT)")
                connection.executemany("INSERT INTO brain_identity VALUES (?, ?)", [(1, identity), (1, identity)])
            elif how == "non-text":
                connection.execute("UPDATE brain_identity SET brain_id = x'00'")
            else:
                raise AssertionError(how)
    finally:
        connection.close()


@pytest.mark.parametrize("how", ["row-missing", "malformed", "uppercase", "duplicated", "non-text"])
def test_absent_or_malformed_binding_is_corruption_and_nothing_is_written(tmp_path, how):
    root = tmp_path / "brain"
    Urdyn.create(root).remember("kept")
    _tamper_binding(root / ".urdyn/memory.db", how)
    before = tree(root)
    with pytest.raises(UrdynStorageCorruptError) as caught:
        Urdyn.open(root)
    assert not isinstance(caught.value, _identity_error())
    assert tree(root) == before


def test_dropped_binding_table_reads_as_legacy_which_is_a_documented_limit(tmp_path):
    """Not a guarantee but the stated boundary: without a version bump (kept
    for rollback compatibility), the binding's presence is the only marker, so a
    store whose table was removed by deliberate SQL is a legacy store. It is
    opened as such -- unbound, untouched -- never reported as verified."""
    root = tmp_path / "brain"
    Urdyn.create(root).remember("kept")
    _tamper_binding(root / ".urdyn/memory.db", "table-missing")
    before = tree(root)
    assert Urdyn.open(root).state()[0].content == "kept"
    assert tree(root) == before
    assert db_facts(root / ".urdyn/memory.db") == (7, None)


def test_store_level_check_names_the_three_cases_distinctly(tmp_path, two_brains):
    from urdyn._store import MemoryStore
    a, b = two_brains
    messages = set()
    with pytest.raises(_identity_error()) as mismatch:
        MemoryStore.verify_existing(a / ".urdyn/memory.db", brain_id=manifest_id(b))
    messages.add(str(mismatch.value))
    for how in ("row-missing", "malformed"):
        root = tmp_path / how
        Urdyn.create(root)
        _tamper_binding(root / ".urdyn/memory.db", how)
        with pytest.raises(UrdynStorageCorruptError) as caught:
            MemoryStore.verify_existing(root / ".urdyn/memory.db", brain_id=manifest_id(root))
        messages.add(str(caught.value))
    assert len(messages) == 3, messages


# ---------------------------------------------------------------------------
# legacy (v7, unbound) and unknown schemas
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_brain(tmp_path):
    root = tmp_path / "legacy"
    Urdyn.create(root)
    make_legacy_unbound(root)
    return root


def test_legacy_unbound_brain_stays_usable_and_is_never_converted(legacy_brain):
    root = legacy_brain
    manifest_before = (root / ".urdyn/manifest.json").read_bytes()
    before = tree(root)
    engine = Urdyn.open(root)
    assert tree(root) == before, "opening a legacy Brain must not write"
    engine.remember("legacy keeps working")
    assert engine.recall("legacy")[0].content == "legacy keeps working"
    Urdyn.init(root)  # re-init of an existing legacy Brain: no conversion either
    assert db_facts(root / ".urdyn/memory.db") == (7, None)
    assert (root / ".urdyn/manifest.json").read_bytes() == manifest_before


def test_legacy_brain_still_refuses_a_handle_whose_manifest_changed(legacy_brain, tmp_path):
    engine = Urdyn.open(legacy_brain)
    other = tmp_path / "other"
    Urdyn.create(other)
    shutil.copyfile(other / ".urdyn/manifest.json", legacy_brain / ".urdyn/manifest.json")
    with pytest.raises(UrdynStorageError):
        engine.state()


@pytest.mark.parametrize("version", [6, 8, 9, 0])
def test_unknown_or_historical_schema_is_refused_without_writes(tmp_path, version):
    root = tmp_path / "brain"
    Urdyn.create(root)
    connection = sqlite3.connect(root / ".urdyn/memory.db")
    try:
        with connection:
            connection.execute(f"PRAGMA user_version = {version}")
    finally:
        connection.close()
    before = tree(root)
    with pytest.raises(UrdynStorageError):
        Urdyn.open(root)
    assert tree(root) == before


# ---------------------------------------------------------------------------
# open handles
# ---------------------------------------------------------------------------


def test_live_handle_refuses_a_foreign_brain_swapped_in_with_a_copied_manifest(tmp_path, two_brains):
    a, b = two_brains
    engine = Urdyn.open(a)
    shutil.copyfile(a / ".urdyn/manifest.json", b / ".urdyn/manifest.json")
    a.rename(tmp_path / "a-moved-away")
    b.rename(a)
    before = tree(a)
    for call in (lambda: engine.remember("must not land in B"), lambda: engine.state(),
                 lambda: engine.recall("memory")):
        with pytest.raises(_identity_error()):
            call()
    assert tree(a) == before


def test_live_handle_refuses_a_store_replaced_under_an_unchanged_manifest(two_brains):
    a, b = two_brains
    engine = Urdyn.open(a)
    shutil.copyfile(b / ".urdyn/memory.db", a / ".urdyn/memory.db")
    before = tree(a)
    with pytest.raises(_identity_error()):
        engine.remember("must not land in B's store")
    assert tree(a) == before


# ---------------------------------------------------------------------------
# creation: interruptions, recovery, retry
# ---------------------------------------------------------------------------

_CRASH_PRELUDE = '''import os, sys
from urdyn import Urdyn
from urdyn import _store, _workspace
from urdyn._store import MemoryStore
step = sys.argv[2]
def die(*_a, **_k):
    os._exit(74)
'''


def _crash_create(root: Path, step: str):
    code = _CRASH_PRELUDE + '''
if step == "before-store":
    MemoryStore.create_new = classmethod(lambda cls, path: die())
elif step == "inside-store-transaction":
    original_bind = _store._bind_identity
    def bind_then_die(connection, brain_id):
        original_bind(connection, brain_id)
        die()
    _store._bind_identity = bind_then_die
elif step == "before-publish":
    original_rename = os.rename
    def rename(src, dst, *a, **k):
        if str(src).endswith(".urdyn.creating"):
            die()
        return original_rename(src, dst, *a, **k)
    _workspace.os.rename = rename
elif step == "after-publish":
    original_rename = os.rename
    def rename(src, dst, *a, **k):
        original_rename(src, dst, *a, **k)
        if str(src).endswith(".urdyn.creating"):
            die()
    _workspace.os.rename = rename
Urdyn.create(sys.argv[1])
'''
    child(code, root, step, expected=74)


@pytest.mark.parametrize("step", ["before-store", "inside-store-transaction"])
def test_interrupted_preparation_is_never_presented_as_a_brain(tmp_path, step):
    root = tmp_path / "brain"
    _crash_create(root, step)
    stage = root / ".urdyn.creating"
    assert stage.is_dir() and not (root / ".urdyn").exists()
    identity = json.loads((stage / "manifest.json").read_text())["cortex_id"]
    before = tree(root)
    with pytest.raises(UrdynError):
        Urdyn.open(root)
    with pytest.raises(UrdynError):
        Urdyn.recover_creation(root)
    with pytest.raises(UrdynError):
        Urdyn.create(root)  # retry refuses: the interrupted stage is never overwritten
    assert tree(root) == before
    assert json.loads((stage / "manifest.json").read_text())["cortex_id"] == identity


def test_complete_but_unpublished_stage_is_recovered_with_its_own_identity(tmp_path):
    root = tmp_path / "brain"
    _crash_create(root, "before-publish")
    stage = root / ".urdyn.creating"
    identity = json.loads((stage / "manifest.json").read_text())["cortex_id"]
    assert db_facts(stage / "memory.db") == (7, [(1, identity)])
    with pytest.raises(UrdynError):
        Urdyn.open(root)
    engine = Urdyn.recover_creation(root)
    assert engine.urdyn_id == identity == manifest_id(root)
    assert db_facts(root / ".urdyn/memory.db") == (7, [(1, identity)])
    assert Urdyn.recover_creation(root).urdyn_id == identity  # repeating is harmless


def test_crash_right_after_publication_leaves_a_consistent_brain(tmp_path):
    root = tmp_path / "brain"
    _crash_create(root, "after-publish")
    engine = Urdyn.open(root)
    assert db_facts(root / ".urdyn/memory.db") == (7, [(1, engine.urdyn_id)])
    engine.remember("works after the crash")


def test_recovery_refuses_a_stage_whose_manifest_and_store_disagree(tmp_path, two_brains):
    a, _ = two_brains
    root = tmp_path / "brain"
    _crash_create(root, "before-publish")
    stage = root / ".urdyn.creating"
    shutil.copyfile(a / ".urdyn/manifest.json", stage / "manifest.json")
    before = tree(root)
    with pytest.raises(_identity_error()):
        Urdyn.recover_creation(root)
    assert tree(root) == before
    assert not (root / ".urdyn").exists()


def test_store_creation_requires_the_manifest_it_is_bound_to(tmp_path):
    from urdyn._store import MemoryStore
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(UrdynError):
        MemoryStore.create_new(stage / "memory.db")
    assert not (stage / "memory.db").exists()


# ---------------------------------------------------------------------------
# backup and restore
# ---------------------------------------------------------------------------


def _archive_db(archive: Path, tmp_path: Path) -> Path:
    out = tmp_path / ("extracted-" + archive.name)
    out.mkdir()
    with zipfile.ZipFile(archive) as source:
        (out / "memory.db").write_bytes(source.read(".urdyn/memory.db"))
    return out / "memory.db"


def _forge_archive(path: Path, manifest: bytes, database: bytes | None) -> None:
    files = {".urdyn/manifest.json": manifest}
    if database is not None:
        files[".urdyn/memory.db"] = database
    metadata = {"format": 1, "urdyn_memory": "0.4.0",
                "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("backup.json", json.dumps(metadata, sort_keys=True))
        for name, data in files.items():
            archive.writestr(name, data)


def test_backup_carries_the_binding(tmp_path):
    root = tmp_path / "brain"
    identity = Urdyn.create(root).urdyn_id
    archive = tmp_path / "b.zip"
    create_backup(root, archive)
    assert db_facts(_archive_db(archive, tmp_path)) == (7, [(1, identity)])


def test_restore_to_a_new_destination_keeps_the_backup_identity(tmp_path):
    root = tmp_path / "brain"
    identity = Urdyn.create(root).urdyn_id
    archive = tmp_path / "b.zip"
    create_backup(root, archive)
    restored = tmp_path / "restored"
    restore_backup(restored, archive)
    assert Urdyn.open(restored).urdyn_id == identity
    assert db_facts(restored / ".urdyn/memory.db") == (7, [(1, identity)])


def test_restore_over_the_same_brain_keeps_its_identity(tmp_path):
    root = tmp_path / "brain"
    engine = Urdyn.create(root)
    engine.remember("before backup")
    archive = tmp_path / "b.zip"
    create_backup(root, archive)
    engine.remember("after backup")
    restore_backup(root, archive)
    reopened = Urdyn.open(root)
    assert reopened.urdyn_id == engine.urdyn_id
    assert [m.content for m in reopened.state()] == ["before backup"]


def test_restore_of_another_brain_over_an_existing_one_is_refused(tmp_path, two_brains):
    a, b = two_brains
    archive = tmp_path / "b.zip"
    create_backup(b, archive)
    before = tree(a)
    with pytest.raises(BackupError):
        restore_backup(a, archive)
    assert tree(a) == before
    assert not os.path.lexists(tmp_path / ".brain-a.stage")
    assert not os.path.lexists(tmp_path / ".brain-a.rollback")
    assert Urdyn.open(a).state()[0].content == "memory of A"


def test_restore_of_another_brain_over_a_legacy_one_is_refused_too(tmp_path, legacy_brain):
    other = tmp_path / "other"
    Urdyn.create(other)
    archive = tmp_path / "other.zip"
    create_backup(other, archive)
    before = tree(legacy_brain)
    with pytest.raises(BackupError):
        restore_backup(legacy_brain, archive)
    assert tree(legacy_brain) == before


def test_restore_over_a_brain_whose_identity_cannot_be_read_retains_it(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root).remember("damaged but retained")
    archive_source = tmp_path / "source"
    identity = Urdyn.create(archive_source).urdyn_id
    archive = tmp_path / "source.zip"
    create_backup(archive_source, archive)
    (root / ".urdyn/manifest.json").write_text("{not json")
    damaged = tree(root)
    restore_backup(root, archive)  # restore stays the repair path for a damaged Brain
    assert Urdyn.open(root).urdyn_id == identity
    assert tree(tmp_path / ".brain.rollback") == damaged


def test_restore_of_a_legacy_backup_is_not_converted(tmp_path, legacy_brain):
    Urdyn.open(legacy_brain).remember("legacy fact")
    archive = tmp_path / "legacy.zip"
    create_backup(legacy_brain, archive)
    restored = tmp_path / "restored"
    restore_backup(restored, archive)
    assert db_facts(restored / ".urdyn/memory.db") == (7, None)
    assert Urdyn.open(restored).state()[0].content == "legacy fact"
    assert manifest_id(restored) == manifest_id(legacy_brain)


def test_manifest_only_archive_gets_a_new_empty_store_bound_to_that_manifest(tmp_path):
    root = tmp_path / "brain"
    identity = Urdyn.create(root).urdyn_id
    manifest = (root / ".urdyn/manifest.json").read_bytes()
    archive = tmp_path / "m07.zip"
    _forge_archive(archive, manifest, None)
    restored = tmp_path / "restored"
    restore_backup(restored, archive)
    assert Urdyn.open(restored).state() == []
    assert db_facts(restored / ".urdyn/memory.db") == (7, [(1, identity)])


def test_archive_pairing_a_manifest_with_another_brains_store_is_refused(tmp_path, two_brains):
    a, b = two_brains
    archive = tmp_path / "forged.zip"
    _forge_archive(archive, (a / ".urdyn/manifest.json").read_bytes(), (b / ".urdyn/memory.db").read_bytes())
    # inspect_backup is not serialized and keeps its contract: invalid content
    # surfaces as the storage error itself (restore wraps it in BackupError).
    with pytest.raises(_identity_error()):
        inspect_backup(archive)
    restored = tmp_path / "restored"
    with pytest.raises(BackupError):
        restore_backup(restored, archive)
    assert not os.path.lexists(restored)
    before = tree(a)
    with pytest.raises(BackupError):
        restore_backup(a, archive)
    assert tree(a) == before


def test_archive_whose_binding_was_stripped_is_refused(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root).remember("x")
    stripped = tmp_path / "stripped"
    shutil.copytree(root, stripped)
    _tamper_binding(stripped / ".urdyn/memory.db", "row-missing")
    archive = tmp_path / "stripped.zip"
    _forge_archive(archive, (stripped / ".urdyn/manifest.json").read_bytes(),
                   (stripped / ".urdyn/memory.db").read_bytes())
    with pytest.raises(UrdynStorageCorruptError):
        inspect_backup(archive)
    before = tree(root)
    with pytest.raises(BackupError):
        restore_backup(root, archive)
    assert tree(root) == before


@pytest.mark.parametrize("crash_at", [1, 2])
def test_interrupted_restore_activation_recovers_a_consistent_brain(tmp_path, crash_at):
    """Real process death at the first (`root -> rollback`) or second
    (`stage -> root`) activation rename; recovery must leave exactly one
    consistent Brain of the same identity, never a mixed one."""
    root = tmp_path / "brain"
    engine = Urdyn.create(root)
    engine.remember("in backup")
    archive = tmp_path / "b.zip"
    create_backup(root, archive)
    engine.remember("only in the live Brain")
    code = '''import os, sys
from urdyn import backup
original = backup._replace
calls = []
def replace(source, target):
    calls.append(1)
    if len(calls) == int(sys.argv[3]):
        os._exit(74)
    return original(source, target)
backup._replace = replace
backup.restore_backup(sys.argv[1], sys.argv[2])
'''
    child(code, root, archive, crash_at, expected=74)
    recover_interrupted_restore(root)
    recovered = Urdyn.open(root)
    assert recovered.urdyn_id == engine.urdyn_id
    assert db_facts(root / ".urdyn/memory.db") == (7, [(1, engine.urdyn_id)])
    assert {m.content for m in recovered.state()} <= {"in backup", "only in the live Brain"}
