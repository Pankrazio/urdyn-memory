"""Regressions for two engine defects found while qualifying 0.4.1.

H. Creating a canonical store let a raw `sqlite3.OperationalError` (or an
   `OSError`) escape `MemoryStore.create_new`, outside the public
   `UrdynStorageError` contract. Now it is typed, keeps the original
   exception as `__cause__`, says nothing about paths or SQL, and the
   staging area is preserved and never published.

I. The lifecycle lock name hashed `str(path).encode()`, which raises
   `UnicodeEncodeError` for a POSIX path holding bytes that are not valid
   UTF-8 (Python carries them as surrogate escapes): no Brain could be
   created or opened there. Now the name hashes the original path bytes;
   every decodable path keeps exactly the lock name it had before.

Every test that names a defect fails without the fix and passes with it;
the compatibility tests pass on both and pin what must not change.
Temporary directories and synthetic Brains only.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import subprocess
import sys
import uuid

import pytest

from urdyn import Urdyn, UrdynStorageError
from urdyn._cli import main
from urdyn._lifecycle import LOCK_DIRNAME, LOCK_PREFIX, lifecycle, normalize_location
from urdyn._manifest import LEGACY_WORKSPACE_ID_KEY, SCHEMA_VERSION, write_manifest
from urdyn._store import SEARCH_INDEX_TABLE, MemoryStore, db_path_for

# A CREATE VIRTUAL TABLE that SQLite rejects with an OperationalError that is
# not "no such module: fts5" (duplicate column): a schema bug, never an
# FTS5-less build.
_BROKEN_SEARCH_INDEX_SQL = (
    f"CREATE VIRTUAL TABLE {SEARCH_INDEX_TABLE} "
    "USING fts5(entity_type UNINDEXED, entity_type UNINDEXED, text)"
)
_UNDECODABLE = b"proj\x9bect"  # 0x9B is not valid UTF-8 on its own


def _lock_file(location) -> "os.PathLike":
    """The lock file the build under test uses for `location`, computed from
    the raw bytes of the normalized path (what the OS sees)."""
    path = normalize_location(location)
    digest = hashlib.sha256(os.fsencode(str(path))).hexdigest()
    return path.parent / LOCK_DIRNAME / (LOCK_PREFIX + digest)


# --------------------------------------------------------------------------
# H. Typed error from store creation.
# --------------------------------------------------------------------------


def test_h_create_failure_is_a_typed_storage_error_that_keeps_its_cause(tmp_path, monkeypatch):
    import urdyn._store as store_module

    monkeypatch.setattr(store_module, "_CREATE_SEARCH_INDEX_SQL", _BROKEN_SEARCH_INDEX_SQL)
    root = tmp_path / "brain"

    with pytest.raises(UrdynStorageError) as info:
        Urdyn.create(root, "dev")

    error = info.value
    assert not isinstance(error, sqlite3.Error)
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    assert "no such module" not in str(error.__cause__).lower()  # not the FTS5-less case
    # Nothing sensitive in the public message: no location, no SQL.
    assert str(tmp_path) not in str(error)
    assert "CREATE" not in str(error) and "fts5" not in str(error)
    # Create contract: nothing published, the staging is kept for inspection.
    assert not (root / ".urdyn").exists()
    assert (root / ".urdyn.creating").is_dir()


def test_h_cli_create_failure_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    import urdyn._store as store_module

    monkeypatch.setattr(store_module, "_CREATE_SEARCH_INDEX_SQL", _BROKEN_SEARCH_INDEX_SQL)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["create", "dev"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("urdyn: error: ")
    assert "Traceback" not in captured.err
    assert not (tmp_path / ".urdyn").exists()


def test_h_create_new_never_overwrites_an_existing_store_file(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    write_manifest(stage, {"schema_version": SCHEMA_VERSION, "profile": "dev",
                           LEGACY_WORKSPACE_ID_KEY: uuid.uuid4().hex})
    db_path = db_path_for(stage)
    original = b"bytes that must survive untouched"
    db_path.write_bytes(original)

    with pytest.raises(UrdynStorageError) as info:
        MemoryStore.create_new(db_path)

    assert isinstance(info.value.__cause__, FileExistsError)
    assert str(tmp_path) not in str(info.value)
    assert db_path.read_bytes() == original


def test_h_missing_fts5_module_still_degrades_without_an_error(tmp_path, monkeypatch):
    """Compatibility: the one expected FTS failure stays non-fatal."""
    import urdyn._store as store_module

    def no_fts5(connection):
        return False  # what `_try_create_search_index` returns for "no such module: fts5"

    monkeypatch.setattr(store_module, "_try_create_search_index", no_fts5)
    cx = Urdyn.create(tmp_path / "brain", "dev")
    cx.remember("still recorded without a search index", kind="note")
    assert [m.content for m in cx.state()] == ["still recorded without a search index"]


# --------------------------------------------------------------------------
# I. Lifecycle lock on a path that is not valid UTF-8.
# --------------------------------------------------------------------------


def test_i_a_brain_works_at_a_path_that_is_not_valid_utf8(tmp_path):
    root = tmp_path / os.fsdecode(_UNDECODABLE)

    cx = Urdyn.create(root, "dev")
    memory = cx.remember("content at an undecodable location", kind="note")
    reopened = Urdyn.open(root)

    assert [m.memory_id for m in reopened.state()] == [memory.memory_id]
    assert os.path.isdir(os.fsencode(tmp_path) + b"/" + _UNDECODABLE + b"/.urdyn")


def test_i_lock_file_is_named_from_the_raw_path_bytes(tmp_path):
    root = tmp_path / os.fsdecode(_UNDECODABLE)
    Urdyn.create(root, "dev")

    raw = os.fsencode(str(normalize_location(root)))
    assert _UNDECODABLE in raw
    expected = LOCK_PREFIX + hashlib.sha256(raw).hexdigest()
    assert expected in os.listdir(tmp_path / LOCK_DIRNAME)


@pytest.mark.parametrize("name", ["brain", "with space", "café", "日本語", "emoji-\U0001f9e0"])
def test_i_decodable_paths_keep_the_lock_name_of_the_previous_build(tmp_path, name):
    """Compatibility: two builds running side by side must still agree on
    the lock of every location the previous build could already lock."""
    root = tmp_path / name
    with lifecycle(root):
        pass
    previous = LOCK_PREFIX + hashlib.sha256(str(normalize_location(root)).encode()).hexdigest()
    assert previous in os.listdir(tmp_path / LOCK_DIRNAME)


def test_i_distinct_locations_never_share_a_lock(tmp_path):
    names = [b"proj\x9bect", b"proj\x9cect", b"project", b"proj\xc2\x9bect"]  # last: valid UTF-8 for U+009B
    locks = set()
    for raw in names:
        root = tmp_path / os.fsdecode(raw)
        with lifecycle(root):
            pass
        locks.add(_lock_file(root).name)
    assert len(locks) == len(names)
    assert len(os.listdir(tmp_path / LOCK_DIRNAME)) == len(names) + 1  # + the directory's .gitignore


def test_i_lock_on_an_undecodable_path_excludes_another_process(tmp_path):
    """The lock the child holds is exactly the file the parent computes, and
    it is really held: a non-blocking flock fails until the child releases."""
    root = tmp_path / os.fsdecode(_UNDECODABLE)
    Urdyn.create(root, "dev")
    code = (
        "import sys\n"
        "from urdyn._lifecycle import lifecycle\n"
        "with lifecycle(sys.argv[1]):\n"
        "    print('held', flush=True)\n"
        "    sys.stdin.readline()\n"
        "print('released', flush=True)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.Popen([sys.executable, "-c", code, str(root)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    fd = None
    try:
        assert child.stdout.readline().strip() == "held", child.stderr.read()
        fd = os.open(_lock_file(root), os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.stdin.write("\n")
        child.stdin.flush()
        assert child.stdout.readline().strip() == "released"
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
