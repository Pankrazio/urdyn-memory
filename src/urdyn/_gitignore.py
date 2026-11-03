"""Git safety net for the Brain directory (`.urdyn/`).

Restored and hardened after 0.4.0. Urdyn <= 0.4.0 added `.urdyn/` to the
workspace's `.gitignore` on `init`; an unreleased development build stopped
calling it. The 0.4.0 helper also rewrote the whole file through text mode
(CRLF line endings were converted), followed a `.gitignore` symlink out of the
workspace, appended `.urdyn/` after an explicit user `!.urdyn/` (silently
overriding it), and could not see Brain files Git already tracks.

What this module guarantees, and what it does not:

* The entry is APPENDED with the file's own line ending; existing bytes are
  never rewritten. An existing `.urdyn` rule is not duplicated. A symlink or
  non-regular `.gitignore` is never written through. An explicit user
  negation of `.urdyn` is never overridden: creation is refused instead.
* When a `git` executable is available and the workspace is inside a work
  tree, Git itself is asked (`ls-files`, `check-ignore`) whether the Brain's
  files are tracked or not ignored; either makes the caller refuse. The Git
  index, config and work tree are never modified; repository-configured
  programs (`core.fsmonitor`) are disabled for these read-only queries.
* Without Git, or when Git cannot answer, the entry is still written but the
  protection is reported as NOT verified -- never claimed.
* `.gitignore` only prevents ACCIDENTAL commits from this work tree. It is not
  encryption, access control, nor protection of data already pushed; it does
  not bind other tools, `git add -f`, or other clones' configuration.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
import subprocess
from pathlib import Path

from ._errors import UrdynError

_ENTRY = ".urdyn/"
_IGNORE_LINES = frozenset({b".urdyn", b".urdyn/", b"/.urdyn", b"/.urdyn/"})
_NEGATION_LINES = frozenset(b"!" + line for line in _IGNORE_LINES)
# Paths Git is asked about. They need not exist: a directory pattern such as
# `.urdyn/` matches paths below it whether or not they exist yet.
_PROBES = (".urdyn/manifest.json", ".urdyn/memory.db")
_GIT_TIMEOUT_S = 10

GIT_IGNORED = "ignored"
GIT_NOT_A_REPOSITORY = "not-a-repository"
GIT_UNVERIFIED = "unverified"


class GitProtectionError(UrdynError):
    """The Brain directory could not be protected from Git as promised: the
    entry could not be written, or Git still tracks / does not ignore it.
    Raised BEFORE anything is created or published."""


@dataclasses.dataclass(frozen=True)
class GitProtection:
    """What is known about the protection after it was put in place."""

    status: str  # GIT_IGNORED | GIT_NOT_A_REPOSITORY | GIT_UNVERIFIED
    detail: str = ""
    # Lifecycle lock files next to the Brain that Git tracks or would
    # commit (left by an older build). Reported, never deleted or untracked.
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        limits = ".gitignore only prevents accidental commits; it is not encryption or access control"
        if self.status == GIT_IGNORED:
            text = f".urdyn/ is ignored by Git (verified with git check-ignore). {limits}."
        elif self.status == GIT_NOT_A_REPOSITORY:
            text = f".urdyn/ is listed in .gitignore; this directory is not inside a Git work tree. {limits}."
        else:
            text = f".urdyn/ is listed in .gitignore but the protection is not verified ({self.detail}). {limits}."
        return " ".join([text, *(f"Warning: {w}" for w in self.warnings)])


def ensure_gitignore_entry(workspace: Path) -> bool:
    """Append `.urdyn/` to `workspace/.gitignore` unless a rule is present.
    Returns whether the file was written. Raises `GitProtectionError`."""
    path = workspace / ".gitignore"
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        info = None
    except OSError as exc:
        raise GitProtectionError(f"Urdyn could not inspect {path}: {exc.strerror}") from exc
    try:
        if info is None:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            data = _ENTRY.encode() + b"\n"
        else:
            if stat.S_ISLNK(info.st_mode):
                raise GitProtectionError(f"{path} is a symbolic link; Urdyn does not write through it")
            if not stat.S_ISREG(info.st_mode):
                raise GitProtectionError(f"{path} is not a regular file; Urdyn does not modify it")
            with open(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                content = stream.read()
            lines = {line.strip() for line in content.split(b"\n")}
            if lines & _NEGATION_LINES:
                raise GitProtectionError(
                    f"{path} explicitly un-ignores .urdyn ({b', '.join(sorted(lines & _NEGATION_LINES)).decode()}); "
                    "Urdyn will not override that rule. Remove it, or create the Brain outside this work tree"
                )
            if lines & _IGNORE_LINES:
                return False
            eol = b"\r\n" if b"\r\n" in content else b"\n"
            data = (eol if content and not content.endswith(b"\n") else b"") + _ENTRY.encode() + eol
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with open(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except GitProtectionError:
        raise
    except OSError as exc:
        raise GitProtectionError(f"Urdyn could not add {_ENTRY} to {path}: {exc.strerror}") from exc
    return True


def _git(git: str, workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [git, "-c", "core.fsmonitor=false", *args],
        cwd=workspace,
        env={**os.environ, "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
    )


def _first_line(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip().splitlines()
    return (text[0] if text else "no output")[:160]


def _inside_work_tree(git: str, cwd: Path) -> bool:
    result = _git(git, cwd, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == b"true"


def _names(raw: bytes) -> list[str]:
    return [n.decode("utf-8", "replace") for n in raw.split(b"\0") if n]


def refuse_if_tracked(workspace: Path) -> None:
    """Raise `GitProtectionError` when Git tracks files under
    `workspace/.urdyn`, even if `workspace` does not exist (yet). Used by
    restore, which publishes a Brain at a location it does not create through
    `create`. Read-only; never touches the index. Without Git: no-op."""
    git = shutil.which("git")
    if git is None:
        return
    cwd, spec = (workspace, ".urdyn") if workspace.is_dir() else (workspace.parent, f"{workspace.name}/.urdyn")
    try:
        if not cwd.is_dir() or not _inside_work_tree(git, cwd):
            return
        tracked = _git(git, cwd, "ls-files", "-z", "--", spec)
    except (OSError, subprocess.TimeoutExpired):
        return
    if tracked.returncode == 0 and tracked.stdout:
        names = _names(tracked.stdout)[:5]
        raise GitProtectionError(
            f"Git already tracks files under .urdyn/ in {workspace} ({', '.join(names)}); Urdyn does not change "
            "the Git index: untrack them yourself (for example `git rm --cached -r .urdyn`), or use another location"
        )


def _lock_file_warnings(git: str, workspace: Path) -> tuple[str, ...]:
    """Lock files in the Brain's parent that Git tracks, or legacy
    `.urdyn-lock-*` files (older builds kept them directly in the parent)
    that Git would commit. Read-only."""
    parent = workspace.parent
    specs = (":(glob).urdyn-lock-*", ":(glob).urdyn-locks/**")
    try:
        if not _inside_work_tree(git, parent):
            return ()
        tracked = _git(git, parent, "ls-files", "-z", "--", *specs)
        loose = _git(git, parent, "ls-files", "-z", "--others", "--exclude-standard", "--", *specs)
    except (OSError, subprocess.TimeoutExpired):
        return ()
    warnings = []
    if tracked.returncode == 0 and tracked.stdout:
        warnings.append(
            f"Git tracks Urdyn lifecycle lock files in {parent} ({', '.join(_names(tracked.stdout)[:3])}). They are "
            "empty and hold no Brain data, but should not be versioned; Urdyn does not change the index "
            "(untrack them yourself with `git rm --cached`)."
        )
    if loose.returncode == 0 and loose.stdout:
        warnings.append(
            f"legacy Urdyn lock files in {parent} are not ignored by Git ({', '.join(_names(loose.stdout)[:3])}). "
            "This build keeps its locks in .urdyn-locks/; the old files are empty and can be deleted once no "
            "older Urdyn process uses this Brain. Urdyn does not delete them."
        )
    return tuple(warnings)


def protect_workspace(workspace: Path) -> GitProtection:
    """Put the `.gitignore` entry in place and verify it with Git when
    possible. Raises `GitProtectionError` -- before anything is created --
    when Git already tracks Brain files, when the entry cannot be written,
    or when Git still does not ignore the Brain's files."""
    git = shutil.which("git")
    repository = False
    unverified = "git executable not found" if git is None else ""
    if git is not None:
        try:
            inside = _git(git, workspace, "rev-parse", "--is-inside-work-tree")
            if inside.returncode == 0 and inside.stdout.strip() == b"true":
                repository = True
            elif inside.returncode != 0 and b"not a git repository" in inside.stderr.lower():
                pass
            else:
                unverified = f"git could not inspect this directory: {_first_line(inside.stderr or inside.stdout)}"
            if repository:
                tracked = _git(git, workspace, "ls-files", "-z", "--", ".urdyn")
                if tracked.returncode != 0:
                    repository, unverified = False, f"git ls-files failed: {_first_line(tracked.stderr)}"
                elif tracked.stdout:
                    names = [n.decode("utf-8", "replace") for n in tracked.stdout.split(b"\0") if n][:5]
                    raise GitProtectionError(
                        f"Git already tracks files under .urdyn/ in {workspace} ({', '.join(names)}); a Brain here "
                        "would write canonical data into tracked files. Urdyn does not change the Git index: untrack "
                        "them yourself (for example `git rm --cached -r .urdyn`), or create the Brain elsewhere"
                    )
        except (OSError, subprocess.TimeoutExpired) as exc:
            repository, unverified = False, f"git could not be run: {exc}"
    ensure_gitignore_entry(workspace)
    if not repository:
        return GitProtection(GIT_UNVERIFIED, unverified) if unverified else GitProtection(GIT_NOT_A_REPOSITORY)
    try:
        for probe in _PROBES:
            result = _git(git, workspace, "check-ignore", "-q", "--", probe)
            if result.returncode == 1:
                raise GitProtectionError(
                    f"Git does not ignore {probe} in {workspace} even with the .gitignore entry (another rule "
                    "re-includes it); Urdyn will not create a Brain whose files Git would commit"
                )
            if result.returncode != 0:
                return GitProtection(GIT_UNVERIFIED, f"git check-ignore failed: {_first_line(result.stderr)}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GitProtection(GIT_UNVERIFIED, f"git could not be run: {exc}")
    return GitProtection(GIT_IGNORED, warnings=_lock_file_warnings(git, workspace))
