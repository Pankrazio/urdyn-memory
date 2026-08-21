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

The cost is deliberate and bounded: only files a caller explicitly named
(or explicitly chose from discovery) are read, only UTF-8 text, only up
to `MAX_SEED_FILE_BYTES`, and only when the content actually changed --
a re-seed of an unchanged file writes nothing at all. `.urdyn/` does
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
import re
import stat
from pathlib import Path

from ._errors import UrdynSourceError
from ._evidence import Evidence

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

# (A19.1) Conservative allowlist for `urdyn seed` with no arguments, in
# the `dev` profile only. Deliberately NOT a recursive crawl: these are
# the files that describe a project to a newcomer, at the two locations
# where projects conventionally put them (the root, and a flat `docs/`).
# A recursive walk would turn a bounded, predictable command into one
# whose output depends on the size and shape of the whole repository.
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
_DISCOVERY_DOCS_PATTERN = ("docs", "*.md")


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


def discover_candidate_paths(workspace: Path, urdyn_dirname: str) -> list[str]:
    """The `dev` profile's conservative project-context allowlist, as
    workspace-relative POSIX paths, sorted for determinism.

    Reads no file content and writes nothing. A match that fails any
    check in `resolve_seed_path` is silently skipped rather than raised:
    this is a suggestion of what COULD be seeded, so an unreadable or
    oversized `README.pdf` simply is not a candidate.
    """
    workspace_root = workspace.resolve()
    matches: set[str] = set()

    for pattern in _DISCOVERY_ROOT_PATTERNS:
        for match in workspace_root.glob(pattern):
            matches.add(str(match))
    docs_dirname, docs_pattern = _DISCOVERY_DOCS_PATTERN
    for match in (workspace_root / docs_dirname).glob(docs_pattern):
        matches.add(str(match))

    candidates = []
    for match in matches:
        try:
            candidates.append(resolve_seed_path(workspace_root, urdyn_dirname, match))
        except UrdynSourceError:
            continue
    return sorted(set(candidates))
