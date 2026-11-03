"""Engine-owned snapshots (archive format version 1) and coordinated restore.

The per-destination lifecycle lock is held for canonical snapshot/activation.
SQLite validation is delegated to the canonical store owner.
"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from ._errors import UrdynManifestError, UrdynStorageError
from ._manifest import LEGACY_WORKSPACE_ID_KEY, read_manifest
from ._gitignore import GitProtectionError, protect_workspace, refuse_if_tracked
from ._workspace import Urdyn
from ._store import MemoryStore, db_path_for
from ._lifecycle import lifecycle, lifecycle_many, normalize_location, sync_directory
from functools import wraps

FORMAT_VERSION = 1
ENGINE_VERSION = "0.4.0"
MANIFEST_FILE = ".urdyn/manifest.json"
DATABASE_FILE = ".urdyn/memory.db"
ARCHIVE_MANIFEST = "backup.json"
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
CHUNK = 1024 * 1024


class BackupError(ValueError):
    """A backup cannot be safely created or restored."""


def _paths(root):
    root = normalize_location(root)
    if root.is_symlink():
        raise BackupError("Invalid Brain location")
    return root, root.parent / ("." + root.name + ".stage"), root.parent / ("." + root.name + ".rollback")


def _serialized(operation):
    @wraps(operation)
    def call(root, *args, **kwargs):
        paths = _paths(root)
        locks = paths if operation.__name__ in {"restore_backup", "recover_interrupted_restore"} else paths[:1]
        with lifecycle_many(locks):
            try:
                return operation(root, *args, **kwargs)
            except UrdynStorageError as exc:
                raise BackupError(str(exc)) from exc
    return call

def _replace(source, target):
    os.replace(source, target)
    sync_directory(Path(target).parent)

@_serialized
def recover_interrupted_restore(root):
    root, stage, rollback = _paths(root)
    if rollback.exists() and not root.exists():
        _validate_urdyn(rollback)
        _replace(rollback, root)
    if root.exists():
        _validate_urdyn(root)
    if stage.exists():
        if stage.is_symlink() or not stage.is_dir():
            raise BackupError("Invalid recovery staging area")
        if not root.exists():
            raise BackupError("Brain recovery requires manual inspection")
    if not root.exists() and rollback.exists():
        raise BackupError("Could not recover previous Brain")


def _canonical_files(root):
    Urdyn.open(root)
    if (root / DATABASE_FILE).exists():
        _check_sqlite(root / DATABASE_FILE)
    urdyn_dir = root / ".urdyn"
    if urdyn_dir.is_symlink() or not urdyn_dir.is_dir():
        raise BackupError("Invalid Urdyn directory")
    allowed = {MANIFEST_FILE, DATABASE_FILE, ".urdyn/semantic_index.db", ".gitignore"}
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise BackupError("Symlink in Brain")
        if path.is_file():
            if relative not in allowed or not stat.S_ISREG(path.stat().st_mode):
                raise BackupError("Unknown file in Brain")
            actual.add(relative)
        elif path.is_dir():
            if relative != ".urdyn":
                raise BackupError("Unknown directory in Brain")
        else:
            raise BackupError("Unsupported file in Brain")
    if MANIFEST_FILE not in actual:
        raise BackupError("Missing Urdyn manifest")
    # The semantic index and .gitignore can be reconstructed. SQLite sidecars
    # are deliberately rejected: an active writer must never be copied.
    return [name for name in (MANIFEST_FILE, DATABASE_FILE) if name in actual]


def _check_sqlite(path):
    MemoryStore.verify_existing(Path(path))


def _validate_urdyn(root):
    engine = Urdyn.open(root)
    engine.state()
    engine.timeline()
    engine.sources()
    engine.skills()
    engine.conflicts()
    if (root / DATABASE_FILE).exists():
        _check_sqlite(root / DATABASE_FILE)


# What a check area holds while it exists: a copy of canonical data.
_CHECK_AREA_GITIGNORE = b"# Created by Urdyn: temporary backup check area.\n*\n"


def _protect_stage(root, stage):
    """Give the staging area the Git protection the restored Brain must keep,
    BEFORE any canonical byte is written to it (the same rule `create` applies).

    The root's existing `.gitignore` is carried over byte for byte -- the
    archive does not contain it, so activating the stage would otherwise drop
    it and leave `.urdyn/` committable -- then `.urdyn/` is ensured and
    verified with Git exactly as `create` does. Git tracking files at the
    final location refuses the restore. Raises `BackupError` for a refusal and
    the raw `OSError` for an I/O failure; nothing has been activated yet, so
    the previous Brain stays in place."""
    current = root / ".gitignore"
    try:
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                raise BackupError(f"{current} is not a regular file; restore will not carry it over or replace it")
            with open(os.open(current, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
                data = source.read()
            with open(os.open(stage / ".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644),
                      "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        refuse_if_tracked(root)
        protect_workspace(stage)
    except GitProtectionError as exc:
        # An I/O failure keeps restore's contract for I/O failures (the raw
        # OSError, as for every other restore step); a policy refusal is a
        # BackupError.
        if isinstance(exc.__cause__, OSError):
            raise exc.__cause__ from exc
        raise BackupError(str(exc)) from exc


def _archive_identity(archive):
    """The logical identity the archive restores (its manifest's)."""
    try:
        identity = json.loads(archive.read(MANIFEST_FILE)).get(LEGACY_WORKSPACE_ID_KEY)
    except (ValueError, AttributeError) as exc:
        raise BackupError("Invalid backup manifest") from exc
    if not isinstance(identity, str):
        raise BackupError("Invalid backup manifest")
    return identity


def _refuse_replacing_another_brain(root, incoming):
    """A restore never replaces a Brain with a DIFFERENT Brain.

    Compared on the logical identity both manifests carry, so the rule is the
    same for bound (v8) and legacy (v7) Brains. An absent root, or one whose
    manifest is absent or malformed (identity unknown), is not refused:
    restore stays the repair path for a damaged Brain, and the previous root
    is retained as `.<name>.rollback` exactly as before. A manifest that
    exists but cannot be read (permissions, I/O) refuses the restore: what
    cannot be inspected is not replaced. Restoring another Brain's backup
    remains possible into a new location; replacing in place is not offered
    (no implicit override)."""
    try:
        current = read_manifest(root / ".urdyn")[LEGACY_WORKSPACE_ID_KEY]
    except UrdynManifestError:
        return
    if current != incoming:
        raise BackupError(
            "The backup belongs to a different Brain than the one at the destination; restore refused "
            "and nothing was changed. Restore it into a new location instead.")


def _digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            size += len(block)
            if size > MAX_FILE_BYTES:
                raise BackupError("Brain file exceeds backup limit")
            digest.update(block)
    return size, digest.hexdigest()


@_serialized
def create_backup(root, destination):
    root, _, _ = _paths(root)
    files = _canonical_files(root)
    description = {}
    for name in files:
        size, digest = _digest_file(root / name)
        description[name] = {"size": size, "sha256": digest}
    destination = Path(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise BackupError("Backup output must be outside the Brain")
    if destination.exists():
        raise BackupError("Backup output already exists")
    metadata = {"format": FORMAT_VERSION, "urdyn_memory": ENGINE_VERSION,
                "files": description}
    owned = False
    try:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            owned = True
            archive.writestr(ARCHIVE_MANIFEST, json.dumps(metadata, sort_keys=True))
            for name in files:
                archive.write(root / name, arcname=name)
        for name in files:
            size, digest = _digest_file(root / name)
            if (size, digest) != (description[name]["size"], description[name]["sha256"]):
                raise BackupError("Brain changed during backup")
        if destination.stat().st_size > MAX_ARCHIVE_BYTES:
            raise BackupError("Backup exceeds size limit")
        inspect_backup(destination)
        with destination.open("rb") as result:
            os.fsync(result.fileno())
        sync_directory(destination.parent)
        return json.dumps({"format": FORMAT_VERSION, "files": files}, sort_keys=True)
    except Exception:
        if owned:
            destination.unlink(missing_ok=True)
        raise


def _checked_archive(source):
    source = Path(source)
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise BackupError("Backup exceeds size limit")
    archive = zipfile.ZipFile(source, "r")
    try:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) not in (2, 3) or len(names) != len(set(names)):
            raise BackupError("Unexpected backup members")
        if ARCHIVE_MANIFEST not in names or MANIFEST_FILE not in names:
            raise BackupError("Missing backup member")
        if set(names) - {ARCHIVE_MANIFEST, MANIFEST_FILE, DATABASE_FILE}:
            raise BackupError("Unexpected backup member")
        for member in members:
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or member.file_size > MAX_FILE_BYTES:
                raise BackupError("Invalid backup member")
        if archive.getinfo(ARCHIVE_MANIFEST).file_size > 4096:
            raise BackupError("Invalid backup metadata")
        metadata = json.loads(archive.read(ARCHIVE_MANIFEST))
        if not isinstance(metadata, dict) or metadata.get("format") != FORMAT_VERSION:
            raise BackupError("Incompatible backup format")
        if metadata.get("urdyn_memory") != ENGINE_VERSION:
            raise BackupError("Incompatible Urdyn version")
        files = metadata.get("files")
        if not isinstance(files, dict) or set(files) != set(names) - {ARCHIVE_MANIFEST}:
            raise BackupError("Invalid backup file list")
        for name, info in files.items():
            if not isinstance(info, dict) or set(info) != {"size", "sha256"}:
                raise BackupError("Invalid file metadata")
            if type(info["size"]) is not int or info["size"] < 0 or info["size"] > MAX_FILE_BYTES:
                raise BackupError("Invalid file size")
            digest = info["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise BackupError("Invalid file digest")
            if archive.getinfo(name).file_size != info["size"]:
                raise BackupError("File size mismatch")
        return archive, metadata
    except Exception:
        archive.close()
        raise


def inspect_backup(source):
    archive, metadata = _checked_archive(source)
    with archive:
        for name, info in metadata["files"].items():
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                while block := stream.read(CHUNK):
                    size += len(block)
                    if size > MAX_FILE_BYTES:
                        raise BackupError("Backup entry exceeds limit")
                    digest.update(block)
            if size != info["size"] or digest.hexdigest() != info["sha256"]:
                raise BackupError("Backup integrity mismatch")
        with tempfile.TemporaryDirectory(prefix="urdyn-check-", dir=Path(source).parent) as temporary:
            # The check area may sit inside a work tree (wherever the
            # archive is): ignore it before it holds any canonical byte.
            (Path(temporary) / ".gitignore").write_bytes(_CHECK_AREA_GITIGNORE)
            check_root = Path(temporary) / "brain-default"
            (check_root / ".urdyn").mkdir(parents=True)
            for name in metadata["files"]:
                with archive.open(name) as input_file, (check_root / name).open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file, CHUNK)
            if DATABASE_FILE not in metadata["files"]:
                with MemoryStore.create_new(db_path_for(check_root / ".urdyn")): pass
            _validate_urdyn(check_root)
            has_memories = bool(Urdyn.open(check_root).state())
    return json.dumps({"format": FORMAT_VERSION, "urdyn_memory": ENGINE_VERSION,
                       "has_memories": has_memories}, sort_keys=True)


@_serialized
def restore_backup(root, source):
    root, stage, rollback = _paths(root)
    if stage.exists() or rollback.exists():
        raise BackupError("Previous restore state needs attention")
    inspect_backup(source)
    archive, metadata = _checked_archive(source)
    owned_stage = False
    activation_started = False
    try:
        incoming = _archive_identity(archive)
        _refuse_replacing_another_brain(root, incoming)
        stage.mkdir(mode=0o700)
        owned_stage = True
        _protect_stage(root, stage)
        (stage / ".urdyn").mkdir(mode=0o700)
        with archive:
            for name, info in metadata["files"].items():
                target = stage / name
                digest = hashlib.sha256()
                with archive.open(name) as input_file, target.open("xb") as output_file:
                    while block := input_file.read(CHUNK):
                        digest.update(block)
                        output_file.write(block)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                if digest.hexdigest() != info["sha256"]:
                    raise BackupError("Backup changed during restore")
        if DATABASE_FILE not in metadata["files"]:
            with MemoryStore.create_new(db_path_for(stage / ".urdyn")): pass
        _validate_urdyn(stage)  # includes the store <-> manifest binding
        if read_manifest(stage / ".urdyn")[LEGACY_WORKSPACE_ID_KEY] != incoming:
            raise BackupError("Backup changed during restore")
        sync_directory(stage / ".urdyn")
        sync_directory(stage)
        activation_started = True
        if root.exists():
            _replace(root, rollback)
        try:
            _replace(stage, root)
            _validate_urdyn(root)
        except Exception:
            if root.exists():
                _replace(root, stage)
            if rollback.exists():
                _replace(rollback, root)
            raise
        return json.dumps({"restored": True, "previous_brain_retained": rollback.exists()})
    finally:
        archive.close()
        if owned_stage and not activation_started and stage.exists() and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
