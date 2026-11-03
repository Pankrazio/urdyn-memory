"""Lifecycle lock files versus Git.

Contracts (see `docs/storage-contract.md`):

* Lifecycle lock files are derived, empty, operational files. They live in a
  self-ignoring directory `<parent>/.urdyn-locks/` (its own `.gitignore`
  ignores everything in it), so a Brain nested inside a repository never
  leaves a lock file that `git add -A` would commit. The user's `.gitignore`
  files and the Git index are never modified for this.
* A lock file Git already tracks, or a legacy `.urdyn-lock-*` left by an
  older build, is recognised and reported -- never deleted or untracked.
* The lock keeps excluding across processes; a symlinked lock directory is
  refused.

Temporary directories and synthetic Git repositories only.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from urdyn import Urdyn, UrdynStorageUnavailableError
from urdyn import backup
from urdyn._lifecycle import lifecycle

LOCK_DIR = ".urdyn-locks"


@pytest.fixture(autouse=True)
def _isolated_git_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args], cwd=cwd,
                          env={**os.environ, "LC_ALL": "C"}, capture_output=True, text=True, check=check)


def _would_be_added(repo: Path) -> list[str]:
    """Exactly what `git add -A` would stage: untracked and not ignored."""
    return sorted(_git(repo, "ls-files", "--others", "--exclude-standard").stdout.split())


def _index(repo: Path) -> bytes:
    index = repo / ".git" / "index"
    return index.read_bytes() if index.exists() else b""


def _nested(tmp_path):
    repo = tmp_path / "repo"
    proj = repo / "apps" / "proj"
    proj.mkdir(parents=True)
    (proj / "README.md").write_text("x\n")
    _git(repo, "init", "-q", ".")
    return repo, proj


def _legacy_lock_name(location: Path) -> str:
    return ".urdyn-lock-" + hashlib.sha256(str(location).encode()).hexdigest()


# ---------------------------------------------------------------------------
# lifecycle lock files
# ---------------------------------------------------------------------------


def test_nested_project_leaves_no_lifecycle_file_for_git(tmp_path):
    repo, proj = _nested(tmp_path)

    Urdyn.create(proj, "general").remember("n")

    assert _would_be_added(repo) == ["apps/proj/.gitignore", "apps/proj/README.md"]
    assert not list((repo / "apps").glob(".urdyn-lock-*"))
    lock_dir = repo / "apps" / LOCK_DIR
    assert (lock_dir / ".gitignore").read_bytes().splitlines()[-1] == b"*"
    _git(repo, "add", "-A")
    assert not [p for p in _git(repo, "ls-files").stdout.split() if "urdyn-lock" in p or LOCK_DIR in p]


def test_lock_file_is_empty_and_private(tmp_path):
    Urdyn.create(tmp_path / "brain", "general").remember("n")

    locks = [p for p in (tmp_path / LOCK_DIR).iterdir() if p.name != ".gitignore"]
    assert [p.name for p in locks] == [_legacy_lock_name(tmp_path / "brain")]
    assert locks[0].stat().st_size == 0
    assert locks[0].stat().st_mode & 0o777 == 0o600


def test_restore_lock_files_also_stay_out_of_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", ".")
    brain = repo / "brain"
    Urdyn.create(brain, "general").remember("n")
    backup.create_backup(brain, tmp_path / "b.zip")

    backup.restore_backup(brain, tmp_path / "b.zip")

    assert not [p for p in _would_be_added(repo) if "urdyn-lock" in p or LOCK_DIR in p]


def test_lock_still_excludes_another_process(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    code = (
        "import sys, time\n"
        "from urdyn._lifecycle import lifecycle\n"
        "with lifecycle(sys.argv[1]):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(1.0)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.Popen([sys.executable, "-c", code, str(root)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "held"
        started = time.monotonic()
        Urdyn.open(root)
        waited = time.monotonic() - started
    finally:
        child.wait(timeout=10)
    assert waited >= 0.5


def test_lock_is_reentrant_and_released(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    with lifecycle(root):
        with lifecycle(root):
            Urdyn.open(root).state()
    Urdyn.open(root).state()


def test_symlinked_lock_directory_is_refused_and_nothing_is_written_through_it(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    os.symlink(elsewhere, parent / LOCK_DIR)

    with pytest.raises(UrdynStorageUnavailableError):
        Urdyn.create(parent / "brain", "general")

    assert list(elsewhere.iterdir()) == []
    assert not (parent / "brain" / ".urdyn").exists()


def test_lock_directory_without_hard_links_still_gets_its_gitignore(tmp_path, monkeypatch):
    from urdyn import _lifecycle

    def no_links(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(_lifecycle.os, "link", no_links)
    Urdyn.create(tmp_path / "brain", "general").remember("n")

    assert (tmp_path / LOCK_DIR / ".gitignore").read_bytes().splitlines()[-1] == b"*"
    assert sorted(p.name for p in (tmp_path / LOCK_DIR).iterdir() if not p.name.startswith(".urdyn-lock-")) == [".gitignore"]


def test_existing_lock_directory_gitignore_is_never_rewritten(tmp_path):
    (tmp_path / LOCK_DIR).mkdir()
    custom = b"# kept by the user\n*\n"
    (tmp_path / LOCK_DIR / ".gitignore").write_bytes(custom)

    Urdyn.create(tmp_path / "brain", "general").remember("n")

    assert (tmp_path / LOCK_DIR / ".gitignore").read_bytes() == custom


def test_an_already_tracked_lock_file_is_reported_and_the_index_is_not_touched(tmp_path):
    repo, proj = _nested(tmp_path)
    legacy = repo / "apps" / _legacy_lock_name(proj)
    legacy.write_bytes(b"")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lock committed by mistake")
    index = _index(repo)

    cx = Urdyn.create(proj, "general")

    message = cx._git_protection.describe()
    assert "tracked" in message and ".urdyn-lock-" in message
    assert _index(repo) == index
    assert legacy.exists()


def test_a_legacy_lock_file_left_untracked_is_reported_not_deleted(tmp_path):
    repo, proj = _nested(tmp_path)
    legacy = repo / "apps" / _legacy_lock_name(proj)
    legacy.write_bytes(b"")

    cx = Urdyn.create(proj, "general")

    message = cx._git_protection.describe()
    assert "not ignored" in message and ".urdyn-lock-" in message
    assert legacy.exists()


def test_no_lock_warning_for_a_clean_nested_project(tmp_path):
    repo, proj = _nested(tmp_path)

    cx = Urdyn.create(proj, "general")

    assert "Warning" not in cx._git_protection.describe()
