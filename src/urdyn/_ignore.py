"""A minimal, dependency-free gitignore-pattern matcher.

WHY THIS EXISTS. Urdyn's project-file discovery (`_source.py`) walks the
workspace tree looking for documentation-like files. A repository already
states which of its files are not part of itself -- in `.gitignore` and
`.git/info/exclude` -- and that statement is the closest thing to an
explicit privacy boundary a project ever publishes. Honouring it is what
makes automatic discovery safe to run without asking: a file the user
already told git to forget is a file Urdyn must never surface, index, or
observe.

WHY NOT SHELL OUT TO GIT. `git check-ignore` would be authoritative, but
git is an OPTIONAL dependency of Urdyn -- a workspace need not be a
repository at all, and a subprocess per candidate path (or even one batch
call per scan) would put a process spawn on the watcher's hot path and
make discovery fail differently on a machine without git installed. These
files are plain text with a small, well-documented grammar; reading them
directly keeps discovery pure, fast, and identical everywhere.

DOCUMENTED LIMITATIONS. This is a faithful subset, not a reimplementation
of git's `wildmatch`:

  * Only ignore files at the WORKSPACE ROOT are read (`.gitignore`,
    `.git/info/exclude`, and the XDG global `~/.config/git/ignore`).
    Nested per-directory `.gitignore` files are NOT consulted -- a known
    limitation. The effect is under-ignoring (a nested rule is missed),
    never over-ignoring, and the mandatory exclusion list in `_source.py`
    covers the directories that matter most in practice.
  * `core.excludesFile` configured to a non-default location is not
    honoured: reading it would require parsing git config, and the XDG
    default is what an unconfigured machine uses.
  * Backslash escaping of metacharacters (`\\#`, `\\!`, `\\ `) is handled
    only for a leading `#`/`!` and trailing spaces; a literal `[` or `*`
    escaped with a backslash is treated as the metacharacter.
  * `**` is best-effort: `**/x`, `x/**`, and `a/**/b` are supported.
  * Re-inclusion (`!pattern`) cannot resurrect a file whose parent
    directory is excluded -- which is git's own documented behaviour, and
    is what makes directory pruning during a walk correct.

Everything here is READ-ONLY. Writing `.urdyn/` into `.gitignore` lives
in `_gitignore.py` and stays there: one module writes one line, this one
never writes at all.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

__all__ = [
    "IgnoreRules",
    "load_ignore_rules",
    "parse_ignore_text",
]


@dataclasses.dataclass(frozen=True, slots=True)
class _Rule:
    """One compiled ignore line.

    `regex` matches against a workspace-relative POSIX path with no
    leading slash. `dir_only` restricts the rule to directories (a
    trailing `/` in the source pattern). `negated` re-includes rather
    than excludes; because rules are applied in file order with
    last-match-wins, a negation only undoes what an EARLIER rule did.
    """

    regex: re.Pattern[str]
    negated: bool
    dir_only: bool


def _translate(pattern: str) -> str:
    """Translate one gitignore glob body into a regex source string.

    Segment-aware on purpose: `*` and `?` must not cross a `/`, which is
    exactly what `fnmatch.translate` gets wrong for path patterns.
    """
    out: list[str] = []
    i = 0
    length = len(pattern)
    while i < length:
        char = pattern[i]
        if char == "*":
            # `**` spanning whole segments; otherwise a within-segment `*`.
            if pattern.startswith("**", i):
                j = i + 2
                if j < length and pattern[j] == "/":
                    # `**/` -> zero or more leading directories.
                    out.append("(?:.*/)?")
                    i = j + 1
                    continue
                if j >= length:
                    # trailing `**` -> everything below here
                    out.append(".*")
                    i = j
                    continue
                out.append(".*")
                i = j
                continue
            out.append("[^/]*")
            i += 1
            continue
        if char == "?":
            out.append("[^/]")
            i += 1
            continue
        if char == "[":
            end = i + 1
            if end < length and pattern[end] in "!^":
                end += 1
            if end < length and pattern[end] == "]":
                end += 1
            while end < length and pattern[end] != "]":
                end += 1
            if end >= length:
                # Unterminated class: treat the `[` as a literal.
                out.append(re.escape("["))
                i += 1
                continue
            body = pattern[i + 1 : end]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body + "]")
            i = end + 1
            continue
        out.append(re.escape(char))
        i += 1
    return "".join(out)


def _strip_trailing_spaces(line: str) -> str:
    """Drop unescaped trailing whitespace, as git does."""
    idx = len(line)
    while idx > 0 and line[idx - 1] in " \t":
        # A backslash-escaped space is significant and stops the trim.
        if idx >= 2 and line[idx - 2] == "\\":
            break
        idx -= 1
    return line[:idx]


def _compile(line: str) -> _Rule | None:
    raw = _strip_trailing_spaces(line.rstrip("\n").rstrip("\r"))
    if not raw:
        return None
    if raw.startswith("#"):
        return None
    negated = False
    if raw.startswith("!"):
        negated = True
        raw = raw[1:]
    elif raw.startswith("\\#") or raw.startswith("\\!"):
        raw = raw[1:]
    if not raw:
        return None

    dir_only = raw.endswith("/")
    if dir_only:
        raw = raw[:-1]
    if not raw:
        return None

    anchored = "/" in raw
    if raw.startswith("/"):
        raw = raw[1:]
        anchored = True
    if not raw:
        return None

    body = _translate(raw)
    prefix = "" if anchored else "(?:.*/)?"
    # A directory pattern also matches everything beneath it; the walk
    # prunes such directories, but `is_ignored` must agree when asked
    # about a path directly.
    regex = re.compile(f"^{prefix}{body}(?:/.*)?$")
    return _Rule(regex=regex, negated=negated, dir_only=dir_only)


def parse_ignore_text(text: str) -> list[_Rule]:
    """Compile the lines of one ignore file, in order."""
    rules: list[_Rule] = []
    for line in text.splitlines():
        rule = _compile(line)
        if rule is not None:
            rules.append(rule)
    return rules


class IgnoreRules:
    """The cumulative ignore ruleset for one workspace.

    Immutable once built. `is_ignored` is pure and allocation-light: the
    discovery walk calls it once per directory entry, so it must stay
    cheap enough to run on every scan.
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: list[_Rule] | tuple[_Rule, ...] = ()) -> None:
        self._rules = tuple(rules)

    def __bool__(self) -> bool:
        return bool(self._rules)

    def _match_one(self, relative_posix: str, *, is_dir: bool) -> bool:
        """Last-match-wins over the whole ruleset for one exact path."""
        ignored = False
        for rule in self._rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.match(relative_posix):
                ignored = not rule.negated
        return ignored

    def is_ignored(self, relative_posix: str, *, is_dir: bool = False) -> bool:
        """True if `relative_posix` (workspace-relative, POSIX, no leading
        slash) is excluded.

        Every ancestor directory is evaluated first: git cannot re-include
        a file under an excluded directory, so an ignored ancestor is
        decisive and short-circuits. This is also what makes pruning a
        directory during the walk equivalent to testing each file in it.
        """
        if not self._rules or not relative_posix:
            return False
        parts = relative_posix.split("/")
        for depth in range(1, len(parts)):
            if self._match_one("/".join(parts[:depth]), is_dir=True):
                return True
        return self._match_one(relative_posix, is_dir=is_dir)


def _read_text(path: Path) -> str | None:
    """Best-effort text read. Any failure -- missing, unreadable, not
    UTF-8, a directory -- is silently `None`: an ignore file Urdyn cannot
    read must never be the reason discovery fails."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _global_ignore_path() -> Path:
    """The XDG default git global excludes file (`core.excludesFile` is
    not read -- see the module docstring)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "git" / "ignore"


def load_ignore_rules(workspace_root: Path) -> IgnoreRules:
    """Build the ruleset for `workspace_root`, lowest precedence first.

    Order matches git's: the global excludes file, then
    `.git/info/exclude` (only when a `.git` directory is actually present
    -- repository detection by directory presence, never by invoking
    git), then the root `.gitignore`. Later files win because rules are
    applied last-match-wins.

    Never raises. A workspace with no ignore files at all yields an empty
    ruleset that ignores nothing.
    """
    rules: list[_Rule] = []

    global_text = _read_text(_global_ignore_path())
    if global_text is not None:
        rules.extend(parse_ignore_text(global_text))

    git_dir = workspace_root / ".git"
    try:
        has_git = git_dir.is_dir()
    except OSError:
        has_git = False
    if has_git:
        exclude_text = _read_text(git_dir / "info" / "exclude")
        if exclude_text is not None:
            rules.extend(parse_ignore_text(exclude_text))

    gitignore_text = _read_text(workspace_root / ".gitignore")
    if gitignore_text is not None:
        rules.extend(parse_ignore_text(gitignore_text))

    return IgnoreRules(rules)
