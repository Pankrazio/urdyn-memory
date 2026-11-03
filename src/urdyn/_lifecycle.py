"""Brain lifecycle serialization, independent of SQLite transactions.

Cooperating threads/processes take the external per-location lock before any
canonical operation or directory replacement. POSIX lock released on crash.
"""
import fcntl
import functools
import hashlib
import os
import threading
from contextlib import contextmanager, ExitStack
from pathlib import Path

# Lock files are derived, empty, operational files. They live in a
# self-ignoring directory next to the Brain root (its own `.gitignore` ignores
# everything inside, itself included -- the pattern of `.pytest_cache`), so a
# Brain nested in a repository never leaves a file `git add -A` would commit,
# without touching the user's `.gitignore` files or the Git index. Still in
# the parent, so a restore that replaces the root does not move it.
LOCK_DIRNAME = ".urdyn-locks"
LOCK_PREFIX = ".urdyn-lock-"
_LOCK_DIR_GITIGNORE = b"# Created by Urdyn: lifecycle lock files only (empty, no Brain data).\n*\n"

_registry = {}
_registry_guard = threading.Lock()
_local = threading.local()
_fds = set()
_fork_guard = threading.Lock()
_inference_slots = {}

def _after_fork():
    global _registry, _registry_guard, _local, _fds, _fork_guard, _inference_slots
    # Inherited descriptors must not keep a parent's lifecycle lock alive.
    for fd in _fds:
        os.close(fd)
    _fds = set()
    _registry = {}
    _inference_slots = {}
    _registry_guard = threading.Lock()
    _local = threading.local()
    _fork_guard = threading.Lock()

if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_fork_guard.acquire, after_in_parent=_fork_guard.release,
                        after_in_child=_after_fork)


def normalize_location(path):
    path = Path(path).absolute()
    from ._errors import UrdynStorageError, UrdynStorageUnavailableError
    try:
        if path.is_symlink():
            raise UrdynStorageError("Symlink Brain locations are not supported")
        return path.parent.resolve() / path.name
    except OSError as exc:
        raise UrdynStorageUnavailableError("Brain location unavailable") from exc



@contextmanager
def lifecycle_many(paths):
    keys = sorted({normalize_location(path) for path in paths}, key=str)
    held = getattr(_local, "held", set())
    if held and any(str(key) not in held and str(key) < max(held) for key in keys):
        from ._errors import UrdynStorageError
        raise UrdynStorageError("Lifecycle lock order violation")
    with ExitStack() as stack:
        for key in keys:
            stack.enter_context(lifecycle(key))
        yield


def _ensure_lock_dir_gitignore(dir_fd):
    """Create `.gitignore` in the lock directory once; never rewrite one that
    exists. Best effort: it only keeps EMPTY lock files out of Git, so failing
    to write it must never make the lifecycle lock unavailable (lock files
    Git would then see are reported by `_gitignore`). Written to a temporary
    name and hard-linked into place, so a crash never publishes a partial
    file; no fsync (not canonical data, and no new durability boundary inside
    restore). A leftover temporary file is ignored by the next one."""
    try:
        os.stat(".gitignore", dir_fd=dir_fd, follow_symlinks=False)
        return
    except FileNotFoundError:
        pass
    except OSError:
        return
    temporary = f".gitignore.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
        try:
            os.write(fd, _LOCK_DIR_GITIGNORE)
        finally:
            os.close(fd)
        try:
            os.link(temporary, ".gitignore", src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
        except FileExistsError:
            pass  # another process published it first
        except OSError:
            # Filesystems without hard links (FAT/exFAT, some FUSE mounts):
            # exclusive creation instead; still never overwrites an existing one.
            try:
                fd = os.open(".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
            except FileExistsError:
                pass
            else:
                try:
                    os.write(fd, _LOCK_DIR_GITIGNORE)
                finally:
                    os.close(fd)
    except OSError:
        pass
    finally:
        try:
            os.unlink(temporary, dir_fd=dir_fd)
        except OSError:
            pass


def _open_lock_file(parent, name):
    """Open (creating if needed) `parent/.urdyn-locks/name` for flock. The
    lock directory must be a real directory, never a symlink."""
    try:
        os.mkdir(parent / LOCK_DIRNAME, 0o700)
    except FileExistsError:
        pass
    dir_fd = os.open(parent / LOCK_DIRNAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _ensure_lock_dir_gitignore(dir_fd)
        return os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


@contextmanager
def lifecycle(path):
    path = normalize_location(path)
    key = str(path)
    with _registry_guard:
        mutex = _registry.setdefault(key, threading.RLock())
    with mutex:
        held = getattr(_local, 'held', set())
        if key in held:
            yield
            return
        from ._errors import UrdynStorageUnavailableError
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UrdynStorageUnavailableError("Lifecycle parent unavailable") from exc
        # A path the OS gave as undecodable bytes reaches us with
        # surrogate escapes; "surrogateescape" hashes those original bytes.
        # Every other path encodes exactly as before, so lock names already
        # in use (and a process of an older build) still agree.
        name = LOCK_PREFIX + hashlib.sha256(key.encode("utf-8", "surrogateescape")).hexdigest()
        with _fork_guard:
            try:
                fd = _open_lock_file(path.parent, name)
            except OSError as exc:
                raise UrdynStorageUnavailableError("Lifecycle lock unavailable") from exc
            _fds.add(fd)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise UrdynStorageUnavailableError("Lifecycle lock unavailable") from exc
            _local.held = held | {key}
            yield
        finally:
            _local.held = held
            with _fork_guard:
                _fds.discard(fd)
                os.close(fd)

@contextmanager
def inference_slot(path, timeout, encoder=None):
    """In-process mutual exclusion of semantic inference for one
    Brain and one encoder, OUTSIDE the lifecycle lock: calls of this process
    never run the same encoder for the same Brain concurrently, which is what
    the lifecycle lock used to guarantee as a side effect (a client backend
    may not be thread-safe). `encoder` names the encoder (None: the
    built-in one), so a rebuild with another backend never waits for this
    one. It is not a lifecycle lock: canonical operations never take it, and
    it is never held while the lifecycle lock is. Yields False, without
    waiting any longer, if the slot stayed busy for `timeout` seconds; the
    call that holds it is not interrupted. Per process only: each process
    has its own encoder instances."""
    key = (str(normalize_location(path)), encoder)
    with _registry_guard:
        slot = _inference_slots.setdefault(key, threading.Lock())
    if not slot.acquire(timeout=timeout):
        yield False
        return
    try:
        yield True
    finally:
        slot.release()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def guarded(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        with lifecycle(self._path):
            from ._manifest import read_manifest, LEGACY_WORKSPACE_ID_KEY
            from ._errors import UrdynStorageError
            manifest = read_manifest(self._path / '.urdyn')
            if (manifest[LEGACY_WORKSPACE_ID_KEY], manifest['profile']) != (self._urdyn_id, self._profile):
                raise UrdynStorageError('Brain identity changed; reopen explicitly')
            self.verify_storage()
            return method(self, *args, **kwargs)
    return call
