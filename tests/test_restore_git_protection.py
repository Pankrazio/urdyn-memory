"""Restore keeps the Git protection of the Brain.

Contracts (see `docs/backup-and-recovery.md`):

* `restore_backup` publishes the restored Brain with the same Git protection
  as `create`: the existing `.gitignore` of the Brain root is carried over
  byte for byte, `.urdyn/` is ensured and verified with Git, and the staging
  area is protected BEFORE any canonical byte is written to it. When the
  protection cannot be put in place nothing is activated and the previous
  Brain stays where it was. I/O failures keep restore's raw OSError.
* The temporary check area of `inspect_backup` is ignored while it holds a
  copy of canonical data.

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
# restore and the Git protection of the Brain
# ---------------------------------------------------------------------------


def _dedicated_brain_in_repo(tmp_path, gitignore: bytes | None = None):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("x\n")
    _git(repo, "init", "-q", ".")
    brain = repo / "brain"
    if gitignore is not None:
        brain.mkdir()
        (brain / ".gitignore").write_bytes(gitignore)
    Urdyn.create(brain, "general").remember("canonical note")
    archive = tmp_path / "brain.zip"
    backup.create_backup(brain, archive)
    return repo, brain, archive


def test_restore_keeps_the_restored_brain_ignored_by_git(tmp_path):
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path)
    index = _index(repo)

    backup.restore_backup(brain, archive)

    staged = _would_be_added(repo)
    # No canonical byte and no lock file; the only Urdyn-written files Git
    # sees are `.gitignore` files (brain/ and the retained .brain.rollback/),
    # as for `create` .
    assert not [p for p in staged if "/.urdyn/" in p or "urdyn-lock" in p], staged
    assert [p for p in staged if p.startswith(".brain.")] == [".brain.rollback/.gitignore"]
    assert _git(repo, "check-ignore", "-q", "brain/.urdyn/memory.db", check=False).returncode == 0
    assert _index(repo) == index


def test_restore_carries_the_existing_gitignore_over_byte_for_byte(tmp_path):
    custom = b"# dedicated Brain directory\r\n*.tmp\r\n.urdyn/\r\n"
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path, gitignore=custom)

    backup.restore_backup(brain, archive)

    assert (brain / ".gitignore").read_bytes() == custom
    assert Urdyn.open(brain).recall("canonical")


def test_restore_staging_is_protected_while_it_holds_canonical_data(tmp_path, monkeypatch):
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path)
    seen = []
    original = backup._validate_urdyn

    def observe(root):
        if Path(root).name == ".brain.stage":
            seen.append(_would_be_added(repo))
        return original(root)

    monkeypatch.setattr(backup, "_validate_urdyn", observe)
    backup.restore_backup(brain, archive)

    assert seen, "the staging area was not validated"
    assert [p for p in seen[0] if p.startswith(".brain.stage/")] == [".brain.stage/.gitignore"], seen[0]


def test_restore_to_a_new_location_outside_git_writes_the_entry(tmp_path):
    source = tmp_path / "src"
    Urdyn.create(source, "general").remember("n")
    backup.create_backup(source, tmp_path / "b.zip")

    backup.restore_backup(tmp_path / "restored", tmp_path / "b.zip")

    assert (tmp_path / "restored" / ".gitignore").read_bytes() == b".urdyn/\n"


def test_restore_refuses_when_git_tracks_the_brain_and_keeps_the_previous_brain(tmp_path):
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path)
    _git(repo, "add", "-f", "brain/.urdyn/manifest.json")
    _git(repo, "commit", "-q", "-m", "tracked by mistake")
    before = (brain / ".urdyn" / "memory.db").read_bytes()
    index = _index(repo)

    with pytest.raises(backup.BackupError, match="tracks"):
        backup.restore_backup(brain, archive)

    assert (brain / ".urdyn" / "memory.db").read_bytes() == before
    assert not (repo / ".brain.stage").exists() and not (repo / ".brain.rollback").exists()
    assert _index(repo) == index


def test_restore_refuses_a_gitignore_that_is_not_a_regular_file(tmp_path):
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path)
    target = tmp_path / "outside.gitignore"
    target.write_bytes(b"x\n")
    (brain / ".gitignore").unlink()
    os.symlink(target, brain / ".gitignore")
    before = (brain / ".urdyn" / "memory.db").read_bytes()

    with pytest.raises(backup.BackupError):
        backup.restore_backup(brain, archive)

    assert target.read_bytes() == b"x\n"
    assert (brain / ".urdyn" / "memory.db").read_bytes() == before
    assert not (repo / ".brain.stage").exists()


def test_restore_refuses_an_explicit_user_negation(tmp_path):
    repo, brain, archive = _dedicated_brain_in_repo(tmp_path)
    (brain / ".gitignore").write_bytes(b"!.urdyn/\n")
    before = (brain / ".urdyn" / "memory.db").read_bytes()

    with pytest.raises(backup.BackupError, match="un-ignores"):
        backup.restore_backup(brain, archive)

    assert (brain / ".gitignore").read_bytes() == b"!.urdyn/\n"
    assert (brain / ".urdyn" / "memory.db").read_bytes() == before


def test_backup_check_area_is_ignored_while_it_holds_canonical_data(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", ".")
    source = tmp_path / "src"
    Urdyn.create(source, "general").remember("n")
    seen = []
    original = backup._validate_urdyn

    def observe(root):
        if Path(root).name == "brain-default":
            seen.append(_would_be_added(repo))
        return original(root)

    monkeypatch.setattr(backup, "_validate_urdyn", observe)
    backup.create_backup(source, repo / "b.zip")  # archive deliberately inside a work tree

    assert seen
    assert not [p for p in seen[0] if "urdyn-check-" in p], seen[0]


def _restore_fsync_calls(tmp_path, name):
    root = tmp_path / name
    Urdyn.create(root, "general").remember("Original durable memory")
    archive = tmp_path / f"{name}.zip"
    backup.create_backup(root, archive)
    return root, archive


def test_every_restore_fsync_boundary_fails_safe_including_the_new_ones(tmp_path, monkeypatch):
    """An earlier test injected a failure at fsync calls 1..6 of a
    restore. The Git protection adds a durability boundary (the carried-over
    `.gitignore` of the staging area), so the boundaries are counted here
    instead of hard-coded, and EVERY one of them is injected: the restore
    raises the raw OSError and the previous Brain survives, byte for byte,
    at the root or in the rollback."""
    import urdyn.backup as b

    root, archive = _restore_fsync_calls(tmp_path / "count", "brain")
    original = b.os.fsync
    calls = 0

    def counting(fd):
        nonlocal calls
        calls += 1
        return original(fd)

    monkeypatch.setattr(b.os, "fsync", counting)
    backup.restore_backup(root, archive)
    monkeypatch.setattr(b.os, "fsync", original)
    boundaries = calls
    assert boundaries >= 7  # the 6 historical boundaries + the staging .gitignore

    def tree_digest(path):
        return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}

    for step in range(1, boundaries + 1):
        base = tmp_path / f"step{step}"
        base.mkdir()
        root, archive = _restore_fsync_calls(base, "brain")
        Urdyn.open(root).remember("after the backup")
        before = tree_digest(root)
        seen = 0

        def failing(fd):
            nonlocal seen
            seen += 1
            if seen == step:
                raise OSError("injected fsync failure")
            return original(fd)

        monkeypatch.setattr(b.os, "fsync", failing)
        with pytest.raises(OSError):
            backup.restore_backup(root, archive)
        monkeypatch.setattr(b.os, "fsync", original)
        rollback = base / ".brain.rollback"
        assert any(p.exists() and tree_digest(p) == before for p in (root, rollback)), step
