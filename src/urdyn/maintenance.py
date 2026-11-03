"""Explicit maintenance entry for the derived FTS5 search index.

`Urdyn.open()`, `verify_storage()` and `@guarded` are unchanged: their
`PRAGMA quick_check` rejects a Brain whose derived FTS5 index is damaged
(SQLite >= 3.44 checks FTS5 in `quick_check`), even when canonical data is
intact. This module is the ONE narrow way past that refusal, and only for
that cause:

* it takes the Brain's lifecycle lock and checks location, manifest, schema
  version and identity binding exactly as an ordinary operation would;
* it accepts a failed `quick_check` only if EVERY message names the FTS5
  `search_index` table and every canonical table passes its own
  table-scoped `quick_check`; anything else is refused as it is today;
* it removes only a recognized FTS projection (the `search_index` FTS5
  virtual table and its shadow tables) -- an ordinary table with that name
  or any other unexpected object is refused, nothing removed;
* before touching anything it writes a consistent SQLite safety copy (the
  online backup API, which reads committed WAL content too), verified
  against the source, in `<parent>/.urdyn-maintenance/`, never overwriting;
* DROP, canonical integrity check, rebuild, full validation and comparison
  of canonical data happen in ONE `BEGIN IMMEDIATE` transaction, committed
  only after all of them pass; a fresh connection then re-verifies the
  Brain through the ordinary `verify_existing` path.

Any doubt ends in a typed error with nothing changed (or, when the commit
itself fails, an explicit statement of what could be confirmed).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import os
import sqlite3
import uuid
from pathlib import Path

from . import _store
from ._errors import (
    UrdynStorageCorruptError,
    UrdynStorageError,
    UrdynStorageSchemaError,
    UrdynStorageUnavailableError,
)
from ._lifecycle import lifecycle, normalize_location, sync_directory
from ._manifest import LEGACY_WORKSPACE_ID_KEY, read_manifest
from ._store import (
    SEARCH_INDEX_TABLE,
    _index_objects,
    _require_recognized_projection,
    STORE_SCHEMA_VERSION,
    MemoryStore,
    SearchIndexState,
    db_path_for,
)

URDYN_DIRNAME = ".urdyn"  # same literal as `_workspace.URDYN_DIRNAME` (pinned by tests)
MAINTENANCE_DIRNAME = ".urdyn-maintenance"
_MAINTENANCE_GITIGNORE = b"# Created by Urdyn: maintenance safety copies of Brain data. Never commit.\n*\n"
_QUICK_CHECK_LIMIT = 10000
_STORAGE_INTEGRITY_FAILED = "Storage integrity check failed"


@dataclasses.dataclass(frozen=True, slots=True)
class SearchIndexRecovery:
    """Outcome of `recover_search_index`. `mode` is `rebuilt` (ordinary
    checks passed; the phase-1 transactional rebuild ran, no safety copy
    needed) or `recovered` (the exceptional path for a damaged FTS5
    projection ran; `backup_path` is the verified safety copy taken first)."""

    state: SearchIndexState
    mode: str
    backup_path: Path | None = None


class _Abort(Exception):
    """Internal: a check failed inside the recovery transaction."""


def _commit(connection: sqlite3.Connection) -> None:
    """The single commit point of the exceptional recovery (one name, so the
    failure handling around it can be exercised)."""
    connection.commit()


# --------------------------------------------------------------------------
# Session: lock + the same checks an ordinary operation makes
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Session:
    workspace: Path
    db_path: Path
    brain_id: str
    connection: sqlite3.Connection
    damaged_index: bool  # True only for an isolated FTS5 projection failure


@contextlib.contextmanager
def _session(path, *, expected: tuple[str, str] | None):
    workspace = normalize_location(path)
    with lifecycle(workspace):
        urdyn_dir = workspace / URDYN_DIRNAME
        manifest = read_manifest(urdyn_dir)
        brain_id = manifest[LEGACY_WORKSPACE_ID_KEY]
        if expected is not None and (brain_id, manifest["profile"]) != expected:
            raise UrdynStorageError("Brain identity changed; reopen explicitly")
        db_path = db_path_for(urdyn_dir)
        damaged_index = False
        try:
            MemoryStore.verify_existing(db_path, brain_id=brain_id)  # the ordinary, unchanged check
        except UrdynStorageCorruptError as refused:
            if str(refused) != _STORAGE_INTEGRITY_FAILED:
                raise
            _verify_isolated_index_damage(db_path, brain_id, refused)
            damaged_index = True
        connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            yield _Session(workspace, db_path, brain_id, connection, damaged_index)
        finally:
            connection.close()


def _verify_isolated_index_damage(db_path: Path, brain_id: str, refused: Exception) -> None:
    """Re-run the ordinary checks the failed `quick_check` cut short, then
    accept the failure only if it is provably confined to the FTS5 index.
    Any other outcome re-raises the ORIGINAL refusal."""
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        if connection.execute("PRAGMA user_version").fetchone()[0] != STORE_SCHEMA_VERSION:
            raise UrdynStorageSchemaError("Storage requires explicit schema review")
        _store._verify_identity_binding(connection, db_path, brain_id)
        messages = [row[0] for row in connection.execute(f"PRAGMA quick_check({_QUICK_CHECK_LIMIT})")]
        if len(messages) >= _QUICK_CHECK_LIMIT or not messages:
            raise refused
        if not all(_is_search_index_message(message) for message in messages):
            raise refused
        objects = _index_objects(connection)
        _require_recognized_projection(objects)
        if not _canonical_tables_pass_quick_check(connection, {name for _, name, _, _ in objects}):
            raise refused
        missing = [name for name in _store._V7_TABLES if not _store._table_exists(connection, name)]
        if missing:
            raise UrdynStorageCorruptError("Canonical storage tables are missing")
        _store._verify_columns(connection)
    except sqlite3.DatabaseError as exc:
        raise refused from exc
    finally:
        connection.close()


def _canonical_tables_pass_quick_check(connection, index_names) -> bool:
    """Independent of the message text: every table outside the FTS
    projection must pass its own table-scoped `quick_check`."""
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'")]
    for name in tables:
        if name in index_names:
            continue
        quoted = name.replace('"', '""')
        if connection.execute(f'PRAGMA quick_check("{quoted}")').fetchall() != [("ok",)]:
            return False
    return True


def _is_search_index_message(message: str) -> bool:
    """A `quick_check` message produced by FTS5 about the `search_index`
    table (form observed with SQLite 3.53: `fts5: corruption found reading
    blob N from table "search_index"`). Anything else is not ours to accept."""
    return message.lower().startswith("fts5:") and f'"{SEARCH_INDEX_TABLE}"' in message


def _canonical_fingerprint(connection) -> str:
    """Hash of every non-index object definition and every row of every
    canonical table (order-independent within a table)."""
    digest = hashlib.sha256()
    index_names = {name for _, name, _, _ in _index_objects(connection)}
    schema = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    for type_, name, tbl_name, sql in schema:
        if name in index_names or tbl_name in index_names:
            continue
        digest.update(repr((type_, name, tbl_name, sql)).encode("utf-8", "surrogateescape"))
        if type_ == "table":
            quoted = name.replace('"', '""')
            rows = sorted(repr(row) for row in connection.execute(f'SELECT * FROM "{quoted}"'))
            digest.update(repr((name, len(rows))).encode())
            for row in rows:
                digest.update(row.encode("utf-8", "surrogateescape"))
    digest.update(repr(connection.execute("PRAGMA user_version").fetchone()).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Safety copy
# --------------------------------------------------------------------------


def _maintenance_dir(workspace: Path) -> Path:
    directory = workspace.parent / MAINTENANCE_DIRNAME
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    if directory.is_symlink() or not directory.is_dir():
        raise UrdynStorageUnavailableError("The maintenance directory is not a real directory")
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        try:
            fd = os.open(gitignore, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            try:
                os.write(fd, _MAINTENANCE_GITIGNORE)
            finally:
                os.close(fd)
        except FileExistsError:
            pass
    return directory


def _write_safety_copy(session: _Session) -> tuple[Path, str]:
    """Consistent copy of the whole database through the online backup API,
    to a NEW file (exclusive create, never overwrites), verified against the
    source and fsynced. Returns its path and the canonical fingerprint both
    share. On any failure the partial copy this call created is removed and
    nothing in the Brain has been touched."""
    try:
        directory = _maintenance_dir(session.workspace)
        tag = hashlib.sha256(os.fsencode(str(session.workspace))).hexdigest()[:16]
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = directory / f"search-index-recovery-{tag}-{stamp}-{uuid.uuid4().hex[:8]}.db"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    except OSError as exc:
        raise UrdynStorageUnavailableError("The safety copy could not be created; nothing was changed") from exc
    destination = None
    try:
        destination = sqlite3.connect(target)
        session.connection.backup(destination)
        destination.close()
        destination = None
        copy = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            copy_fingerprint = _canonical_fingerprint(copy)
            # Use the original manifest's identity for both validations; the
            # safety copy has no adjacent manifest. None is valid only when
            # the binding table is absent (a legacy-unbound Brain).
            copy_identity = _store._verify_identity_binding(copy, target, session.brain_id)
        finally:
            copy.close()
        source_fingerprint = _canonical_fingerprint(session.connection)
        source_identity = _store._verify_identity_binding(session.connection, session.db_path, session.brain_id)
        if copy_fingerprint != source_fingerprint or copy_identity != source_identity:
            raise _Abort("the safety copy does not match the Brain")
        with open(target, "rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(directory)
        return target, source_fingerprint
    except BaseException as exc:
        if destination is not None:
            destination.close()
        with contextlib.suppress(OSError):
            os.unlink(target)
        if isinstance(exc, (sqlite3.DatabaseError, OSError, _Abort)):
            raise UrdynStorageUnavailableError("The safety copy could not be created; nothing was changed") from exc
        raise


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def inspect_search_index(path, *, _expected: tuple[str, str] | None = None) -> SearchIndexState:
    """Diagnose the search index of the Brain at `path`, even when
    `Urdyn.open()` refuses it because only the FTS5 index is damaged.
    Never writes. Canonical damage is refused exactly as `open()` refuses it."""
    with _session(path, expected=_expected) as session:
        return MemoryStore(session.connection).inspect_search_index()


def recover_search_index(path, *, _expected: tuple[str, str] | None = None) -> SearchIndexRecovery:
    """Rebuild the search index of the Brain at `path`.

    Ordinary checks pass: the phase-1 transactional rebuild (validated
    before commit). Ordinary checks fail ONLY because the FTS5 projection is
    damaged: safety copy first, then DROP + canonical check + rebuild +
    validation + canonical comparison in one transaction, committed last,
    re-verified afterwards. Anything else is refused with nothing changed."""
    if not _store.fts5_runtime_available():
        raise UrdynStorageUnavailableError("This SQLite runtime has no FTS5: the search index cannot be rebuilt")
    with _session(path, expected=_expected) as session:
        if not session.damaged_index:
            state = MemoryStore(session.connection).recreate_search_index()
            return SearchIndexRecovery(state=state, mode="rebuilt")
        backup_path, fingerprint = _write_safety_copy(session)
        state = _recover_damaged_index(session, fingerprint, backup_path)
        return SearchIndexRecovery(state=state, mode="recovered", backup_path=backup_path)


def _recover_damaged_index(session: _Session, fingerprint: str, backup_path: Path) -> SearchIndexState:
    connection = session.connection
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.DatabaseError as exc:
        raise UrdynStorageUnavailableError(
            f"The Brain is busy; nothing was changed (safety copy kept: {backup_path.name})") from exc
    try:
        if _canonical_fingerprint(connection) != fingerprint:
            raise _Abort("canonical data changed since the safety copy")
        _require_recognized_projection(_index_objects(connection))
        if _store._table_exists(connection, SEARCH_INDEX_TABLE):
            connection.execute(f"DROP TABLE {SEARCH_INDEX_TABLE}")
        if _index_objects(connection):
            raise _Abort("search index objects remain after removing it")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise _Abort("canonical integrity check failed with the search index removed")
        connection.execute(_store._CREATE_SEARCH_INDEX_SQL)
        _store._rebuild_search_index(connection)
        state = _store._validate_search_index(connection, runtime=True)
        if not state.is_usable():
            raise _Abort(f"rebuilt index failed validation ({state.detail})")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise _Abort("integrity check failed after the rebuild")
        if _canonical_fingerprint(connection) != fingerprint:
            raise _Abort("canonical data differs after the rebuild")
        if connection.execute("PRAGMA user_version").fetchone()[0] != STORE_SCHEMA_VERSION:
            raise _Abort("schema version changed")
        _store._verify_identity_binding(connection, session.db_path, session.brain_id)
    except BaseException as exc:
        connection.rollback()
        if isinstance(exc, _Abort):
            raise UrdynStorageError(f"Search index recovery refused: {exc}; rolled back, nothing changed") from None
        if isinstance(exc, sqlite3.DatabaseError):
            raise UrdynStorageError("Search index recovery failed; rolled back, nothing changed") from exc
        raise
    try:
        _commit(connection)
    except BaseException as exc:
        with contextlib.suppress(sqlite3.Error):
            connection.rollback()
        if _confirmed(session, fingerprint):
            return state
        raise UrdynStorageError(
            "Search index recovery outcome could not be confirmed after a commit failure; "
            f"the Brain was not changed or not verifiably repaired (safety copy: {backup_path})"
        ) from exc
    if not _confirmed(session, fingerprint):
        raise UrdynStorageError(
            f"Search index recovery committed but the Brain did not re-verify (safety copy: {backup_path})")
    return state


def _confirmed(session: _Session, fingerprint: str) -> bool:
    """Re-verify on a FRESH connection through the ordinary checks."""
    try:
        MemoryStore.verify_existing(session.db_path, brain_id=session.brain_id)
        connection = sqlite3.connect(session.db_path.resolve().as_uri() + "?mode=rw", uri=True)
        try:
            store = MemoryStore(connection)
            return store.inspect_search_index().is_usable() and _canonical_fingerprint(connection) == fingerprint
        finally:
            connection.close()
    except (UrdynStorageError, sqlite3.Error, OSError):
        return False
