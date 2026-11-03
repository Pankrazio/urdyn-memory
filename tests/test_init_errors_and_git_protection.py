"""Initialization contract, exception boundaries, and Git protection
of the Brain directory.

Contracts (see `docs/storage-contract.md`):

* `init` creates ONLY in an absent or empty directory, otherwise opens an
  existing Brain (deliberate and fail-closed). A non-empty directory
  without a Brain is refused as "not found", with the explicit remedy
  (`create`) -- never as a corrupt manifest that does not exist.
* Not found / present-but-invalid / inaccessible / interrupted are distinct:
  `UrdynNotFoundError`, `UrdynManifestError` (or a storage error for the DB),
  `UrdynStorageUnavailableError`.
* Creating or (re)initializing a Brain puts `.urdyn/` in the directory's
  `.gitignore` without altering existing bytes, refuses to write through a
  symlink, never overrides an explicit user negation, never touches the Git
  index, and -- when Git is available -- refuses to proceed if Git still
  tracks or does not ignore the Brain's files. Nothing is created when the
  protection cannot be put in place.

Temporary directories and synthetic Git repositories only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from urdyn import (
    Urdyn,
    UrdynError,
    UrdynManifestError,
    UrdynNotFoundError,
    UrdynStorageUnavailableError,
)
from urdyn._cli import main as cli_main

HAS_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAS_GIT, reason="git executable not available")
FALSE_MANIFEST_CLAIM = "exists but has no manifest"


@pytest.fixture(autouse=True)
def _isolated_git_config(monkeypatch, tmp_path):
    # Synthetic repositories only, never the developer's global Git config.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args], cwd=cwd,
                          env={**os.environ, "LC_ALL": "C"}, capture_output=True, text=True, check=check)


def _tree(root: Path):
    """Every entry under `root` (relative path -> kind/size/digest)."""
    out = {}
    for path in sorted(root.rglob("*")) if root.exists() else []:
        rel = path.relative_to(root).as_posix()
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if path.is_symlink():
            out[rel] = "link:" + os.readlink(path)
        elif path.is_dir():
            out[rel] = "dir"
        else:
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _project(tmp_path, name="proj"):
    root = tmp_path / name
    root.mkdir()
    (root / "README.md").write_text("synthetic project\n")
    return root


def _index(repo):
    return _git(repo, "ls-files", "-s").stdout


# ---------------------------------------------------------------------------
# Group A -- initialization
# ---------------------------------------------------------------------------


def test_init_in_a_populated_directory_is_refused_as_not_found_with_the_remedy(tmp_path):
    root = _project(tmp_path)
    before = _tree(root)
    with pytest.raises(UrdynNotFoundError) as info:
        Urdyn.init(root, "dev")
    message = str(info.value)
    assert FALSE_MANIFEST_CLAIM not in message
    assert "urdyn create" in message and "Urdyn.create" in message
    assert _tree(root) == before  # nothing created, nothing lost


def test_cli_init_in_a_populated_directory_fails_clearly_and_creates_nothing(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    before = _tree(root)
    monkeypatch.chdir(root)
    assert cli_main(["init", "dev"]) == 1
    err = capsys.readouterr().err
    assert FALSE_MANIFEST_CLAIM not in err and "urdyn create" in err
    assert _tree(root) == before


@pytest.mark.parametrize("prepare", ["absent", "empty"])
def test_init_still_creates_in_an_absent_or_empty_directory(tmp_path, prepare):
    root = tmp_path / "brain"
    if prepare == "empty":
        root.mkdir()
    cx = Urdyn.init(root, "general")
    assert (root / ".urdyn" / "manifest.json").is_file() and (root / ".urdyn" / "memory.db").is_file()
    assert cx.profile == "general"


def test_init_on_a_valid_brain_reopens_it_without_rewriting_it(tmp_path):
    root = tmp_path / "brain"
    first = Urdyn.init(root, "general")
    first.remember("synthetic note")
    canonical = {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")}
    again = Urdyn.init(root, "general")
    assert again.urdyn_id == first.urdyn_id
    assert {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")} == canonical


@pytest.mark.parametrize("case", ["no_manifest", "malformed", "not_a_directory"])
def test_init_never_initializes_over_an_invalid_brain(tmp_path, case):
    root = _project(tmp_path)
    if case == "not_a_directory":
        (root / ".urdyn").write_text("not a directory\n")
    else:
        (root / ".urdyn").mkdir()
        if case == "malformed":
            (root / ".urdyn" / "manifest.json").write_text("{not json")
    before = _tree(root)
    with pytest.raises(UrdynManifestError) as info:
        Urdyn.init(root, "general")
    assert not isinstance(info.value, UrdynNotFoundError)
    assert _tree(root) == before


def test_interrupted_creation_is_reported_as_such_and_preserved(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    (root / ".urdyn.creating").mkdir()
    (root / ".urdyn.creating" / "manifest.json").write_text("{}")
    before = _tree(root)
    for call in (lambda: Urdyn.open(root), lambda: Urdyn.init(root, "general"), lambda: Urdyn.create(root, "general")):
        with pytest.raises(UrdynError) as info:
            call()
        assert not isinstance(info.value, UrdynNotFoundError)
        assert FALSE_MANIFEST_CLAIM not in str(info.value)
    with pytest.raises(UrdynStorageUnavailableError, match="recover_creation"):
        Urdyn.open(root)
    assert _tree(root) == before


def test_interrupted_restore_is_not_reported_as_a_missing_brain(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    os.rename(root, tmp_path / ".brain.rollback")  # the state a crash mid-restore can leave
    with pytest.raises(UrdynStorageUnavailableError, match="recover_interrupted_restore"):
        Urdyn.open(root)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission bits do not bind root")
def test_inaccessible_directory_is_a_storage_access_error(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    os.chmod(root, 0)
    try:
        for call in (lambda: Urdyn.open(root), lambda: Urdyn.init(root, "general")):
            with pytest.raises(UrdynStorageUnavailableError):
                call()
    finally:
        os.chmod(root, 0o755)
    assert Urdyn.open(root).profile == "general"  # untouched


def test_urdyn_symlink_is_refused_and_nothing_is_written_outside(tmp_path):
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / ".urdyn")
    with pytest.raises(UrdynManifestError, match="symlink"):
        Urdyn.init(root, "general")
    assert list(outside.iterdir()) == []


# ---------------------------------------------------------------------------
# Group B -- exception boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prepare", ["absent", "empty", "populated"])
def test_open_without_a_brain_is_not_found(tmp_path, prepare):
    root = tmp_path / "proj"
    if prepare != "absent":
        root.mkdir()
    if prepare == "populated":
        (root / "README.md").write_text("x\n")
    before = _tree(root)
    with pytest.raises(UrdynNotFoundError) as info:
        Urdyn.open(root)
    assert FALSE_MANIFEST_CLAIM not in str(info.value)
    assert _tree(root) == before and (root.exists() == (prepare != "absent"))


def test_open_requires_the_exact_root_and_says_not_found(tmp_path):
    Urdyn.create(tmp_path / "brain", "general")
    nested = tmp_path / "brain" / "src"
    nested.mkdir()
    with pytest.raises(UrdynNotFoundError):
        Urdyn.open(nested)


@pytest.mark.parametrize("case", ["no_manifest", "malformed", "wrong_schema"])
def test_present_but_invalid_brain_is_a_manifest_error_not_missing(tmp_path, case):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    manifest = root / ".urdyn" / "manifest.json"
    if case == "no_manifest":
        manifest.unlink()
    elif case == "malformed":
        manifest.write_text("{not json")
    else:
        manifest.write_text(manifest.read_text().replace('"schema_version": 1', '"schema_version": 99'))
    with pytest.raises(UrdynManifestError) as info:
        Urdyn.open(root)
    assert not isinstance(info.value, UrdynNotFoundError)


def test_discover_points_to_the_remedy_that_works(tmp_path):
    root = _project(tmp_path)
    with pytest.raises(UrdynNotFoundError, match="urdyn create"):
        Urdyn.discover(root)


# ---------------------------------------------------------------------------
# Group C -- Git protection
# ---------------------------------------------------------------------------


def test_create_outside_git_writes_the_entry_and_says_it_is_unverified_by_git(tmp_path):
    root = _project(tmp_path)
    cx = Urdyn.create(root, "general")
    assert (root / ".gitignore").read_bytes() == b".urdyn/\n"
    assert cx._git_protection.status == "not-a-repository"


@needs_git
def test_create_in_a_repo_without_gitignore_is_verified_ignored_and_index_untouched(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    index = _index(root)
    cx = Urdyn.create(root, "general")
    assert cx._git_protection.status == "ignored"
    for probe in (".urdyn/memory.db", ".urdyn/manifest.json"):
        assert _git(root, "check-ignore", "-q", probe, check=False).returncode == 0
    assert ".urdyn" not in _git(root, "status", "--porcelain", "--untracked-files=all").stdout
    assert _index(root) == index


@needs_git
def test_custom_rules_are_preserved_byte_for_byte(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    original = b"build/\r\n# keep me\r\n*.log"  # CRLF, comment, no trailing newline
    (root / ".gitignore").write_bytes(original)
    Urdyn.create(root, "general")
    content = (root / ".gitignore").read_bytes()
    assert content == original + b"\r\n.urdyn/\r\n"
    assert _git(root, "check-ignore", "-q", "build/x", check=False).returncode == 0  # user rule still works


@needs_git
@pytest.mark.parametrize("line", [".urdyn/", ".urdyn", "/.urdyn/", "/.urdyn"])
def test_an_existing_rule_is_not_duplicated(tmp_path, line):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    (root / ".gitignore").write_text(f"node_modules/\n{line}\n")
    before = (root / ".gitignore").read_bytes()
    Urdyn.create(root, "general")
    assert (root / ".gitignore").read_bytes() == before


def test_repeated_init_is_idempotent(tmp_path):
    root = tmp_path / "brain"
    Urdyn.init(root, "general")
    first = (root / ".gitignore").read_bytes()
    Urdyn.init(root, "general")
    Urdyn.init(root, "general")
    assert (root / ".gitignore").read_bytes() == first == b".urdyn/\n"


def test_init_reopen_adds_the_missing_entry_to_an_older_brain(tmp_path):
    root = tmp_path / "brain"
    Urdyn.create(root, "general")
    (root / ".gitignore").unlink()  # e.g. a Brain created by an earlier development build
    canonical = {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")}
    Urdyn.init(root, "general")
    assert (root / ".gitignore").read_bytes() == b".urdyn/\n"
    assert {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")} == canonical
    Urdyn.open(root)  # open never writes it
    (root / ".gitignore").unlink()
    Urdyn.open(root)
    assert not (root / ".gitignore").exists()


@needs_git
def test_nested_project_inside_a_repository(tmp_path):
    repo = _project(tmp_path, "repo")
    _git(repo, "init", "-q", ".")
    (repo / ".gitignore").write_text("dist/\n")
    repo_rules = (repo / ".gitignore").read_bytes()
    sub = repo / "packages" / "app"
    sub.mkdir(parents=True)
    (sub / "notes.md").write_text("x\n")
    cx = Urdyn.create(sub, "general")
    assert cx._git_protection.status == "ignored"
    assert (repo / ".gitignore").read_bytes() == repo_rules  # the repository's own rules untouched
    assert _git(repo, "check-ignore", "-q", "packages/app/.urdyn/memory.db", check=False).returncode == 0


@needs_git
def test_an_explicit_user_negation_is_not_overridden(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    (root / ".gitignore").write_text("*.log\n!.urdyn/\n")
    before = _tree(root)
    with pytest.raises(UrdynError, match="un-ignores"):
        Urdyn.create(root, "general")
    assert _tree(root) == before


@needs_git
def test_already_tracked_brain_files_are_reported_and_the_index_is_not_touched(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    (root / ".urdyn").mkdir()
    (root / ".urdyn" / "memory.db").write_text("synthetic\n")
    _git(root, "add", "-f", ".urdyn/memory.db")
    _git(root, "commit", "-qm", "synthetic")
    shutil.rmtree(root / ".urdyn")
    index, before = _index(root), _tree(root)
    with pytest.raises(UrdynError) as info:
        Urdyn.create(root, "general")
    assert "tracks" in str(info.value) and ".urdyn/memory.db" in str(info.value)
    assert _index(root) == index and _tree(root) == before  # no .urdyn, no .gitignore, index intact


@needs_git
def test_reinit_of_a_brain_whose_files_are_tracked_fails_and_keeps_the_brain(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    Urdyn.create(root, "general").remember("synthetic note")
    _git(root, "add", "-f", ".urdyn/manifest.json")
    canonical = {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")}
    index = _index(root)
    with pytest.raises(UrdynError, match="tracks"):
        Urdyn.init(root, "general")
    assert _index(root) == index
    assert {k: v for k, v in _tree(root).items() if k.startswith(".urdyn")} == canonical
    assert len(Urdyn.open(root).timeline()) == 1  # still usable


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission bits do not bind root")
def test_gitignore_write_failure_creates_nothing(tmp_path):
    root = _project(tmp_path)
    (root / ".gitignore").write_text("build/\n")
    os.chmod(root / ".gitignore", 0o444)
    before = _tree(root)
    try:
        with pytest.raises(UrdynError, match="could not"):
            Urdyn.create(root, "general")
        assert _tree(root) == before  # no .urdyn, no .urdyn.creating
    finally:
        os.chmod(root / ".gitignore", 0o644)
    Urdyn.create(root, "general")  # and nothing blocks a later, successful attempt


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_gitignore_that_is_not_a_regular_file_is_not_written_through(tmp_path, kind):
    root = _project(tmp_path)
    outside = tmp_path / "outside.gitignore"
    outside.write_text("keep\n")
    if kind == "symlink":
        os.symlink(outside, root / ".gitignore")
    else:
        (root / ".gitignore").mkdir()
    before = _tree(root)
    with pytest.raises(UrdynError, match="not a regular file|symbolic link"):
        Urdyn.create(root, "general")
    assert outside.read_text() == "keep\n" and _tree(root) == before


def test_without_git_the_protection_is_written_but_not_claimed_as_verified(tmp_path, monkeypatch, capsys):
    import urdyn._gitignore as gitignore

    monkeypatch.setattr(gitignore.shutil, "which", lambda name: None)
    root = tmp_path / "brain"
    root.mkdir()
    monkeypatch.chdir(root)
    assert cli_main(["init", "general"]) == 0
    out = capsys.readouterr().out
    assert (root / ".gitignore").read_bytes() == b".urdyn/\n"
    assert "not verified" in out and "verified with git" not in out


@needs_git
def test_cli_reports_verified_protection_and_its_limits(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    monkeypatch.chdir(root)
    assert cli_main(["create", "general"]) == 0
    out = capsys.readouterr().out
    assert "ignored by Git" in out and "not encryption" in out


@needs_git
def test_repository_configured_programs_are_not_run_by_the_check(tmp_path):
    root = _project(tmp_path)
    _git(root, "init", "-q", ".")
    marker = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    _git(root, "config", "core.fsmonitor", str(hook))
    Urdyn.create(root, "general")
    assert not marker.exists()


def test_recover_creation_publishes_with_the_protection(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    stage = root / ".urdyn.creating"
    # A complete but unpublished staging, as a crash right before the rename leaves it.
    Urdyn.create(tmp_path / "donor", "general")
    shutil.copytree(tmp_path / "donor" / ".urdyn", stage)
    cx = Urdyn.recover_creation(root)
    assert cx._git_protection is not None
    assert (root / ".gitignore").read_bytes() == b".urdyn/\n"


def test_backup_still_accepts_the_protected_brain(tmp_path):
    from urdyn.backup import create_backup, inspect_backup

    root = tmp_path / "brain"
    Urdyn.create(root, "general").remember("synthetic note")
    archive = tmp_path / "backup.zip"
    create_backup(root, archive)
    assert inspect_backup(archive)
