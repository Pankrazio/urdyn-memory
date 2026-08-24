"""The canonical `Source` model: the stable identity of an external thing
Urdyn has observed, plus the individual observations made of it.

A Source is NOT knowledge. Seeding a project file records that Urdyn saw
that file, in that state, at that moment -- nothing more. It creates no
Memory, asserts no belief, and grants no authority: the
`document_observation` Evidence each observation carries is deliberately
excluded from `VERIFICATION_EVIDENCE_KINDS` (see `_evidence.py`), so a
Memory can never become `verified` by citing a seeded file. Turning a
document into a belief stays an explicit act performed by the caller
through `remember()`/`learn()`, with the observation cited as provenance.

The three levels this keeps apart:

  Source     -- "this file exists in this project and Urdyn tracks it"
                (identity: a workspace-relative path)
  Observation -- "at this moment the file said this, and its bytes hashed
                to this digest" (a fact about one point in time,
                append-only; the text itself lives in its Evidence)
  Memory     -- "this is what we believe" (never produced here)

IDENTITY. A Source is identified by its `source_id` (a canonical 32-hex
Urdyn identity, like every other primitive) and addressed by `path`,
which is always RELATIVE to the workspace root and always POSIX-style.
Absolute paths are never persisted: a workspace copied or moved to a
different absolute location must keep resolving its own Sources, so the
canonical record cannot depend on where the workspace happened to live
when the file was seeded.

An observation deliberately has NO id of its own. Every observation
already creates exactly one Evidence, whose `evidence_id` is a stable
canonical identity for the same fact -- minting a second UUID 1:1 with it
would create two names for one thing, and force every future consumer to
decide which one is "the" identity. `SourceObservation.evidence_id` IS
the observation's identity.

WHAT IS STORED. The observation's Evidence holds the document's text
VERBATIM, exactly as it was decoded when the file was read. This is what
makes it an Evidence at all: every other kind (`command_output`,
`test_result`, `error_observation`) carries what was observed, and an
Evidence holding only a digest would be the one that merely describes
its own observation instead of containing it -- "there was a file whose
bytes hashed to X" cannot be cited, audited, or reconstructed once the
file has changed or is gone.

The digest is a companion to the text, not a replacement for it: it is
what makes a repeated seed decidable without comparing whole documents,
and what lets a reader confirm that a stored snapshot is the text that
produced it.

The cost is deliberate and bounded: only UTF-8 text, only up to
`MAX_SEED_FILE_BYTES`, and only STORED when the content actually changed
-- a re-seed of an unchanged file writes nothing at all. (Discovery
reads candidates too, to answer "is this binary/empty?", but reading is
not recording: nothing a caller did not explicitly seed is ever
persisted.) `.urdyn/` does
therefore hold a local copy of the documents it was asked to observe,
which the CLI states plainly at seed time rather than leaving implied.

THREAT MODEL for path handling. The checks below (`resolve_seed_path`)
defend against a path that reaches OUTSIDE the workspace -- via `..`,
via an absolute path, or via a symlink pointing elsewhere -- and against
seeding something that is not ordinary text (a device, a FIFO, a binary,
a file too large to be a project document), plus a minimal denylist of
names that habitually hold secrets. They do NOT defend against a local
attacker who can modify the filesystem between the moment Urdyn resolves
a path and the moment it reads the bytes (a TOCTOU swap): such an
attacker already has write access to the repository being seeded, and
closing that window would require holding an open file descriptor across
every check for a threat that is not the one this feature has. The
denylist is a safety net against an accidental `urdyn seed .env`, never
a claim to detect secrets: Urdyn does not scan file content for them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fnmatch
import hashlib
import os
import re
import stat
from pathlib import Path

from ._errors import UrdynSourceError
from ._evidence import Evidence
from ._ignore import IgnoreRules as _IgnoreRules
from ._ignore import load_ignore_rules

SOURCE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

# Bare lowercase hex, no algorithm prefix: the algorithm is fixed by
# `_DIGEST_ALGORITHM` below and documented, so nothing ever has to parse
# the stored value to know how to interpret it. Changing algorithm later
# is a schema migration, not a format negotiation embedded in the data.
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_DIGEST_ALGORITHM = "sha256"

# The three outcomes a single `seed()` of one path can have. `unchanged`
# is not a failure and not a no-op the caller should ignore: it means the
# file is still exactly as Urdyn last observed it.
SEED_ADDED = "added"
SEED_UNCHANGED = "unchanged"
SEED_CHANGED = "changed"
VALID_SEED_STATUSES = frozenset({SEED_ADDED, SEED_UNCHANGED, SEED_CHANGED})

# A project document that does not fit in 1 MiB is not the kind of thing
# this feature exists to track: it is a dataset, a generated artifact, or
# a vendored blob. Refusing it keeps a stray `urdyn seed` from reading a
# huge file into memory just to hash it.
MAX_SEED_FILE_BYTES = 1_048_576

# Names that habitually hold credentials, matched against the file's own
# name (so a nested `config/.env` is caught too). Minimal on purpose --
# see the module docstring: this is a guard against an accidental seed,
# not a secret scanner, and it is applied even to explicitly named paths
# because the mistake it catches is precisely a mistyped explicit path.
_SECRET_NAME_PATTERNS = (
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
)

# (A19.1) Root-only manifest/description files for `urdyn seed` with no
# arguments, in the `dev` profile only. These stay ROOT-ONLY on purpose:
# they are conventions about the top of a repository (`pyproject.toml`
# three directories down is a vendored copy or a test fixture, not this
# project's manifest), so globbing them recursively would add noise, not
# context.
_DISCOVERY_ROOT_PATTERNS = (
    "README*",
    "LICENSE*",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "requirements*.txt",
    "AGENTS.md",
    "CLAUDE.md",
)

# (A53) Names automatic discovery refuses to PROPOSE, even though an
# explicit `urdyn seed <path>` for the same name still works (that stays
# governed by `_SECRET_NAME_PATTERNS` above, applied via
# `resolve_seed_path` to explicit and discovered paths alike). Recursive
# discovery widens what a bare `urdyn seed` can surface to any `.md`/
# `.txt` file in the tree, which means a file merely NAMED like private
# material (`secrets.txt`, `team-password-notes.md`) can now be offered
# even though its name matches no credential-file pattern. Automatic
# discovery is a suggestion nobody asked for by path, so it holds itself
# to a stricter, discovery-only bar; a deliberate `urdyn seed secrets.txt`
# is a different act and is not affected by this list.
_DISCOVERY_SENSITIVE_NAME_PATTERNS = (
    "*secret*",
    "*password*",
    "*passwd*",
    "*credential*",
    "*token*",
)


def _is_discovery_sensitive_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in _DISCOVERY_SENSITIVE_NAME_PATTERNS)


# (A53) Documentation-like files ARE discovered recursively. The earlier
# design globbed only a flat `docs/*.md`, which silently missed the very
# common `docs/research/notes.md` shape and made "did Urdyn see my new
# note?" depend on how deep the author filed it.
#
# What keeps the recursive walk bounded -- and therefore keeps this from
# becoming "scan the whole repository":
#
#   1. EXTENSION SCOPE. Only these suffixes are considered. Source code
#      is deliberately NOT discovered: `.py`/`.js`/... are the bulk of a
#      repository and are not the project-describing documents this
#      feature exists to track. An explicit `urdyn seed src/main.py`
#      still works and is unchanged.
#   2. DIRECTORY PRUNING. `MANDATORY_EXCLUDED_DIR_NAMES` below is never
#      descended into, regardless of what any ignore file says.
#   3. IGNORE RULES. `.gitignore` / `.git/info/exclude` (see `_ignore.py`)
#      prune directories and reject files. What the project already told
#      git to forget, Urdyn never sees.
#   4. PER-FILE GATES. Every surviving candidate passes the same
#      `resolve_seed_path` + `read_seed_candidate` checks an explicit
#      seed does: inside the workspace, regular file, not a credential
#      name, under `MAX_SEED_FILE_BYTES`, real UTF-8 text, non-empty.
#   5. A HARD VISIT CAP (`MAX_DISCOVERY_VISITS`).
_DISCOVERY_RECURSIVE_SUFFIXES = (".md", ".txt")

# Never descended into, whatever `.gitignore` says (it may say nothing:
# a workspace need not be a git repository at all, and `.venv/` is
# routinely absent from `.gitignore` because nobody ever needed git to
# know about it). These are the directories where a recursive walk would
# otherwise spend all of its time and find nothing that describes the
# project: dependency trees, build output, virtualenvs, tool caches.
#
# `.urdyn` itself is NOT listed here -- it is passed in as
# `urdyn_dirname` by every caller (see `discover_candidate_paths`), so
# the one place that knows the directory's real name stays
# `_workspace.URDYN_DIRNAME` rather than a second copy of the string.
MANDATORY_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
    }
)

# Same policy, for directory names that are only expressible as a glob.
MANDATORY_EXCLUDED_DIR_PATTERNS = ("*.egg-info",)

# Hard ceiling on directory entries examined in ONE discovery pass. A
# pathological workspace (a generated tree, a mount point with millions
# of files, a symlink farm) must make discovery return early, not hang or
# blow up: the walk stops and returns what it found so far, which is
# still a valid -- merely incomplete -- suggestion. Sized well above any
# plausible real repository so that hitting it is diagnostic of something
# unusual rather than a limit normal projects live against.
MAX_DISCOVERY_VISITS = 50_000


@dataclasses.dataclass(frozen=True, slots=True)
class SourceObservation:
    """One observation of a `Source`: what Urdyn saw, and when.

    Identified by `evidence_id` -- the Evidence recorded alongside it (see
    the module docstring on why there is no separate observation id).
    Append-only: an observation is never updated or deleted, so a file
    that changes accumulates observations rather than overwriting them.
    """

    source_id: str
    evidence_id: str
    digest: str
    size_bytes: int
    observed_at: dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """A project file Urdyn tracks, with its full observation history.

    `observations` is ordered oldest-first and is never empty: a Source is
    only ever created together with its first observation, in one
    transaction. Carrying the complete relation on the model (rather than
    a "latest" projection) follows the same shape as `Memory.evidence_ids`
    and `Skill.steps` -- the canonical record loads whole, and no second
    call is needed to see history.

    `first_observed_at` is the `sources` row's own column: when Urdyn
    first recorded this identity. It is not recomputed from
    `observations`.
    """

    source_id: str
    path: str
    first_observed_at: dt.datetime
    observations: tuple[SourceObservation, ...]

    @property
    def latest_observation(self) -> SourceObservation:
        """The most recent observation. Always present (see class docstring)."""
        return self.observations[-1]


@dataclasses.dataclass(frozen=True, slots=True)
class SeedResult:
    """What one `seed()` of one path did.

    A derived result type, not canonical data: `status` describes THIS
    CALL, not a property of the Source (a Source reloaded tomorrow could
    not say whether the call that recorded it was the first one) -- the
    same reasoning that keeps A17's "was this newly recorded?" out of
    `Memory`.

    `evidence` is the Evidence belonging to the observation this call
    resolved to: the newly created one for `added`/`changed`, the
    already-persisted one for `unchanged`. Passing it to
    `remember(..., evidence=[...])` is how a caller explicitly derives a
    belief from a seeded file.
    """

    status: str
    source: Source
    evidence: Evidence


@dataclasses.dataclass(frozen=True, slots=True)
class SeedCandidate:
    """A file that passed every path/content check and is ready to be
    persisted. Purely internal to the seed pipeline.

    `text` is the decoded document, carried here so the seed pipeline
    reads the file exactly ONCE: `read_seed_candidate` already has to
    decode the bytes to reject non-UTF-8 input, and re-reading them later
    to build the Evidence would both cost a second read and open a window
    in which the two could disagree about what was observed.

    `size_bytes` is the size of the BYTES on disk, not `len(text)`: it
    describes the file that was read, and for any non-ASCII document the
    two differ.

    The text carries no path, digest or timestamp: those are structured
    columns (see `_store.py`), and embedding them in the payload would
    force a future reader to parse them back out of a document.
    """

    path: str
    digest: str
    size_bytes: int
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class SeedCandidateReport:
    """What `urdyn seed` with no arguments found, split by what seeding
    each candidate WOULD do.

    A derived view, not canonical data -- the same reasoning that keeps
    `SeedResult.status` off `Source`. The split is computed by comparing
    each candidate's current digest against
    `Source.latest_observation.digest`, i.e. exactly the comparison
    `seed()` itself would make, so the three groups predict the three
    `SEED_*` statuses without performing any of them.

      new       -- eligible and never seeded (would be `added`)
      changed   -- already a tracked Source whose content has since
                   changed (would be `changed`)
      unchanged -- already a tracked Source, byte-identical to the last
                   observation (would be `unchanged`, writing nothing)

    Nothing is written to produce this. Excluded and ineligible files are
    deliberately NOT enumerated: listing everything the walk rejected
    would turn a suggestion into a report on the whole repository, and
    "why was my file not offered?" is a question for `urdyn seed <path>`,
    which answers it with the exact refusal reason.
    """

    new: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()

    @property
    def paths(self) -> list[str]:
        """Every candidate, sorted -- the flat list `seed_candidates()`
        has always returned."""
        return sorted((*self.new, *self.changed, *self.unchanged))

    def __bool__(self) -> bool:
        return bool(self.new or self.changed or self.unchanged)


def compute_digest(data: bytes) -> str:
    """The canonical digest of a file's bytes (see `DIGEST_PATTERN`)."""
    return hashlib.new(_DIGEST_ALGORITHM, data).hexdigest()


def _is_secret_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _SECRET_NAME_PATTERNS)


def resolve_seed_path(workspace: Path, urdyn_dirname: str, raw: str | Path) -> str:
    """Validate one path a caller asked to seed and return it as a
    workspace-relative POSIX path.

    Raises `UrdynSourceError` -- never a bare `OSError` or `ValueError`
    -- for every way a path can be unacceptable, so a batch caller can
    skip one rejected file without also swallowing programming errors.

    Applied to EXPLICIT paths as well as discovered ones: the allowlist
    is what differs between the two (an explicitly named file is a
    deliberate choice and needs no allowlist), never these checks.
    """
    workspace_root = workspace.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate

    try:
        # `resolve()` collapses `..` AND follows symlinks, so a single
        # containment check below covers traversal and symlink escape
        # together: whatever the path is spelled like, this is the file
        # that would actually be read. A self-referential or excessively
        # deep symlink makes `resolve()` raise `RuntimeError`, not
        # `OSError` -- caught here alongside it so such a path is refused
        # like any other unresolvable one, never left to propagate as an
        # unhandled exception out of a caller that only expects
        # `UrdynSourceError` (e.g. the watcher's discovery scan).
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise UrdynSourceError(f"Cannot resolve path {str(raw)!r}: {exc}") from exc

    if not resolved.is_relative_to(workspace_root):
        raise UrdynSourceError(
            f"Refusing to seed {str(raw)!r}: it resolves outside the Urdyn workspace"
        )

    relative = resolved.relative_to(workspace_root)
    if not relative.parts:
        raise UrdynSourceError(f"Refusing to seed {str(raw)!r}: it is the workspace root itself")
    if relative.parts[0] == urdyn_dirname:
        raise UrdynSourceError(
            f"Refusing to seed {str(raw)!r}: it is inside Urdyn's own {urdyn_dirname}/ directory"
        )
    if _is_secret_name(resolved.name):
        raise UrdynSourceError(
            f"Refusing to seed {str(raw)!r}: its name matches a credential pattern "
            "Urdyn will not record"
        )

    try:
        info = resolved.stat()
    except OSError as exc:
        raise UrdynSourceError(f"Cannot read {str(raw)!r}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise UrdynSourceError(f"Refusing to seed {str(raw)!r}: it is not a regular file")
    if info.st_size > MAX_SEED_FILE_BYTES:
        raise UrdynSourceError(
            f"Refusing to seed {str(raw)!r}: it is {info.st_size} bytes, over the "
            f"{MAX_SEED_FILE_BYTES}-byte limit for a project document"
        )

    return relative.as_posix()


def read_seed_candidate(workspace: Path, relative_path: str) -> SeedCandidate:
    """Read an already-validated workspace-relative path ONCE, and return
    its digest, its size on disk, and the text decoded from it.

    Raises `UrdynSourceError` if the file is binary or is not valid
    UTF-8 text: Urdyn tracks project documents, and a digest of a blob
    it could not have read as text would be a provenance claim about
    something it does not understand.

    An EMPTY file is refused as well. There is no observation worth
    keeping: its Evidence would have to hold empty content, which the
    rest of the write API already refuses (`add_evidence` rejects blank
    content), and relaxing that invariant across every Evidence to admit
    one degenerate document would be the wrong trade. An empty README is
    also far more often a file nobody has written yet than a document
    someone means to track.
    """
    target = (workspace.resolve() / relative_path).resolve()
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise UrdynSourceError(f"Cannot read {relative_path!r}: {exc}") from exc

    if len(data) > MAX_SEED_FILE_BYTES:
        # The file grew between `resolve_seed_path`'s stat and this read.
        raise UrdynSourceError(
            f"Refusing to seed {relative_path!r}: it is {len(data)} bytes, over the "
            f"{MAX_SEED_FILE_BYTES}-byte limit for a project document"
        )
    if b"\x00" in data:
        raise UrdynSourceError(f"Refusing to seed {relative_path!r}: it looks like a binary file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UrdynSourceError(
            f"Refusing to seed {relative_path!r}: it is not valid UTF-8 text"
        ) from exc
    if not text.strip():
        # Checked on the DECODED text, not on `len(data)`: a file holding
        # only whitespace has bytes but still records nothing observed,
        # and `add_evidence`'s own blank-content rule is a `strip()` too.
        raise UrdynSourceError(
            f"Refusing to seed {relative_path!r}: it is empty, so there is no observation to record"
        )

    return SeedCandidate(
        path=relative_path,
        digest=compute_digest(data),
        size_bytes=len(data),
        text=text,
    )


def _is_excluded_dir_name(name: str, urdyn_dirname: str) -> bool:
    """True for a directory the walk must never descend into (§2 of the
    `_DISCOVERY_RECURSIVE_SUFFIXES` note above)."""
    if name == urdyn_dirname or name in MANDATORY_EXCLUDED_DIR_NAMES:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in MANDATORY_EXCLUDED_DIR_PATTERNS)


def _walk_discoverable_files(
    workspace_root: Path,
    urdyn_dirname: str,
    ignore_rules: _IgnoreRules,
) -> list[str]:
    """Breadth-first walk of `workspace_root`, yielding workspace-relative
    POSIX paths whose suffix is in `_DISCOVERY_RECURSIVE_SUFFIXES` and
    which survive directory pruning and ignore rules.

    Path/content eligibility is NOT decided here -- that stays in
    `resolve_seed_path`/`read_seed_candidate`, applied by the caller, so
    there is exactly one definition of "seedable" in the codebase.

    Symlink handling. A symlinked directory is followed only if it still
    resolves inside the workspace, and only if its resolved identity
    (`st_dev`, `st_ino`) is not already one of its OWN ANCESTORS -- which
    is what makes a symlink cycle (`loop/back -> loop`) terminate instead
    of recursing forever.

    Deliberately an ancestor check, not a global "visited" set: two
    distinct paths that alias the same directory (`linked -> real`) are
    BOTH reachable spellings of a real file inside the workspace, and
    which one a global set happened to reach first would depend on
    `scandir` order -- making discovery's output non-deterministic in
    exactly the way the sorted result is supposed to prevent. A symlinked
    FILE is not resolved here at all: `resolve_seed_path` already refuses
    one that escapes, and doing it twice would just be a second policy.
    """
    found: list[str] = []
    visits = 0
    queue: list[tuple[Path, str, frozenset[tuple[int, int]]]] = []

    try:
        root_info = workspace_root.stat()
    except OSError:
        return found
    queue.append((workspace_root, "", frozenset({(root_info.st_dev, root_info.st_ino)})))

    while queue:
        directory, prefix, ancestors = queue.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            # Unreadable directory (permissions, a vanished mount): skip
            # it, never fail the whole scan for one bad subtree.
            continue

        for entry in entries:
            visits += 1
            if visits > MAX_DISCOVERY_VISITS:
                # Degrade gracefully: what was found so far is still a
                # valid suggestion, just not an exhaustive one.
                return found

            relative = f"{prefix}{entry.name}"
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue

            if is_dir:
                if _is_excluded_dir_name(entry.name, urdyn_dirname):
                    continue
                if ignore_rules.is_ignored(relative, is_dir=True):
                    continue
                try:
                    info = entry.stat()  # follows symlinks: identity of the target
                except OSError:
                    continue
                identity = (info.st_dev, info.st_ino)
                if identity in ancestors:
                    continue  # a cycle: this directory contains itself
                try:
                    resolved = Path(entry.path).resolve()
                except (OSError, RuntimeError):
                    continue
                if not resolved.is_relative_to(workspace_root):
                    continue
                queue.append((Path(entry.path), f"{relative}/", ancestors | {identity}))
                continue

            if not entry.name.endswith(_DISCOVERY_RECURSIVE_SUFFIXES):
                continue
            if ignore_rules.is_ignored(relative, is_dir=False):
                continue
            if _is_discovery_sensitive_name(entry.name):
                continue
            found.append(relative)

    return found


def discover_seed_candidates(
    workspace: Path,
    urdyn_dirname: str,
    *,
    ignore_rules: _IgnoreRules | None = None,
) -> list[SeedCandidate]:
    """The `dev` profile's project-context candidates, fully read and
    validated, sorted by workspace-relative POSIX path for determinism.

    Two sources, unioned: the root-only manifest/description globs
    (`_DISCOVERY_ROOT_PATTERNS`) and a bounded recursive walk for
    documentation-like files (`_DISCOVERY_RECURSIVE_SUFFIXES`). Both are
    filtered by `.gitignore`/`.git/info/exclude` and by the mandatory
    directory exclusions, and every survivor must pass the same checks an
    explicitly seeded path does.

    Writes nothing, ever -- `urdyn seed` with no arguments is a
    suggestion, and `test_discovery_writes_nothing` is the guard. It DOES
    read candidate content (via `read_seed_candidate`), because "is this
    binary / empty / not UTF-8?" is not answerable from a stat and a
    candidate the seed pipeline would immediately refuse is not a
    candidate worth proposing. The cost is bounded by the same things
    that bound the walk: the 1 MiB size cap applied before the read, the
    extension allowlist, and directory pruning.

    Deterministic and idempotent: the same filesystem yields the same
    sorted list.

    Any failure of an individual path -- unresolvable, oversized, binary,
    a credential name -- is silently skipped rather than raised.

    `ignore_rules` may be passed by a caller that already built them (the
    watcher re-uses one set across a scan); by default they are loaded
    fresh from the workspace.
    """
    workspace_root = workspace.resolve()
    if ignore_rules is None:
        ignore_rules = load_ignore_rules(workspace_root)

    matches: set[str] = set()
    for pattern in _DISCOVERY_ROOT_PATTERNS:
        for match in workspace_root.glob(pattern):
            relative = match.name
            if ignore_rules.is_ignored(relative, is_dir=match.is_dir()):
                continue
            matches.add(relative)
    matches.update(_walk_discoverable_files(workspace_root, urdyn_dirname, ignore_rules))

    seen: set[str] = set()
    candidates: list[SeedCandidate] = []
    for match in sorted(matches):
        try:
            relative = resolve_seed_path(workspace_root, urdyn_dirname, match)
            if relative in seen:
                continue
            candidate = read_seed_candidate(workspace_root, relative)
        except UrdynSourceError:
            continue
        seen.add(relative)
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: candidate.path)


def discover_candidate_paths(
    workspace: Path,
    urdyn_dirname: str,
    *,
    ignore_rules: _IgnoreRules | None = None,
) -> list[str]:
    """`discover_seed_candidates`, reduced to just the sorted paths --
    the shape every existing caller wants."""
    return [
        candidate.path
        for candidate in discover_seed_candidates(
            workspace, urdyn_dirname, ignore_rules=ignore_rules
        )
    ]
