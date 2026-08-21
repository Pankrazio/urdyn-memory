"""The Dev profile's automatic filesystem watcher.

Turns "the user seeded a file once" into "Cortex keeps noticing when it
changes", for the `dev` profile only. This module owns exactly four
things: a persistent on/off switch, a detached background process, a
bounded polling/reconciliation loop, and the OS-held lock that makes both
liveness and single-instance ownership answerable without a PID registry.

WHAT THIS IS NOT. It is not a second storage path: every write goes
through `Cortex.seed()`, which already owns validation, digest comparison
and the `Source`/`SourceObservation`/`document_observation` Evidence
contract (see `_source.py`). It is not a knowledge writer: nothing here
can create a Memory, mark anything verified, or otherwise reach past the
Source/Evidence boundary -- that boundary is enforced by `seed()` itself,
not re-checked here. And it is not an event-driven watcher: correctness
rests on comparing current filesystem state against what Cortex last
recorded, never on the delivery of any particular OS notification (kernel
event queues are lossy under load; a periodic reconciliation is not).

SCOPE, BY CONSTRUCTION. `Cortex.watcher_scope()` returns tracked `Source`
paths unioned with the `dev` discovery allowlist -- both bounded,
non-recursive sets. This module never walks a directory tree: every scan
is "stat these specific paths", not "find files under this root". A
directory denylist (`.git/`, `node_modules/`, ...) is therefore not
needed as the exclusion mechanism -- scope structurally cannot contain
such a path, since neither `sources()` (itself gated by
`resolve_seed_path`, which already refuses anything under `.cortex/` or
outside the workspace) nor the allowlist globs can ever match one. The one
denylist this module DOES apply is `_looks_like_temp_name`, because the
allowlist's `README*`-style globs can transiently match an editor's own
backup file (`README.md~`); see its docstring.

LIFECYCLE. A workspace has a persistent switch (`watcher.json`,
`{"enabled": bool}`) and a live process holding an advisory lock
(`watcher.lock`) for as long as it runs. The two are independent on
purpose: `enabled=true` with no live holder is `stale`, not a
contradiction, and is exactly the state every normal Cortex CLI command
recovers from via `supervise()` -- the reboot story is "the machine
restarts, the watcher does not, and the next `cortex` command notices and
restarts it", not a login item or a system service.

Baseline is a cache, not state: this module persists no baseline file.
The durable answer to "what did Cortex last see" is already canonical --
`Source.latest_observation.digest` -- so losing the in-memory map on
crash or restart costs one reconciliation pass, never history (see
`_reconcile_baseline`).
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from ._errors import CortexError, CortexManifestError, CortexSourceError
from ._manifest import PROFILE_DEV, read_manifest
from ._source import SEED_UNCHANGED
from ._workspace import CORTEX_DIRNAME, Cortex

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only off Linux
    fcntl = None  # type: ignore[assignment]

WATCHER_CONFIG_FILENAME = "watcher.json"
WATCHER_LOCK_FILENAME = "watcher.lock"
WATCHER_LOG_FILENAME = "watcher.log"

# -- tuning constants, all derived from measurement, not guesswork ----------

# Per-path settle/debounce window: the minimum measured to
# collapse Ctrl-S spam into one observation, with margin for slower atomic
# saves. Coalescing is per path, never global -- a save-all across many
# files stays that many observations, each on its own settle clock.
SETTLE_SECONDS = 1.0

# Adaptive scan interval: cheap while a project is actively being
# edited, backing off automatically once it goes quiet. `_ACTIVE_WINDOW`/
# `_DEEP_IDLE_WINDOW` classify "how long since anything last changed";
# `_SCAN_COST_FACTOR` bounds the watcher at roughly 2% of one core by
# construction, so a very large scope cannot make polling itself expensive.
_ACTIVE_INTERVAL = 2.0
_IDLE_INTERVAL = 10.0
_DEEP_IDLE_INTERVAL = 60.0
_ACTIVE_WINDOW = 120.0
_DEEP_IDLE_WINDOW = 1800.0
_SCAN_COST_FACTOR = 50
_MIN_INTERVAL = 2.0
_MAX_INTERVAL = 60.0

# Shutdown must be responsive even while the loop is sleeping through a
# 60-second deep-idle interval, so the sleep is chunked rather than one
# long `time.sleep()` call.
_SHUTDOWN_POLL_SECONDS = 0.2

# Rate limit: bounds sustained write-lock pressure from a
# burst of changes independently of settle/debounce.
_MAX_WRITES_PER_SECOND = 5

# `watcher.log` is a bounded diagnostic aid, not an archive: truncated and
# restarted rather than rotated once it crosses this size.
_LOG_MAX_BYTES = 1_048_576

# How long `stop_watcher` waits for the held lock to actually free up
# after sending SIGTERM, before reporting a timeout instead of a firm
# "stopped".
_STOP_WAIT_SECONDS = 5.0

# Temp/editor artifacts that can transiently match the discovery
# allowlist's `README*`-style globs (an emacs save produces `README.md~`,
# which DOES match `README*`). Scope from tracked Sources is never
# filtered by this -- an explicit `cortex seed foo.orig` is a deliberate
# choice, not an accident this guards against.
_TEMP_NAME_PATTERNS = (
    "*.swp", "*.swo", "*.swx", "*~", ".#*", "#*#",
    "*.tmp", ".*.tmp", "*.orig", "*.rej", "4913", "*.part", "*.crdownload",
)


def _looks_like_temp_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _TEMP_NAME_PATTERNS)


# -- persistent config: watcher.json -----------------------------------------


def _config_path(cortex_dir: Path) -> Path:
    return cortex_dir / WATCHER_CONFIG_FILENAME


def read_config(cortex_dir: Path) -> dict:
    """Read `watcher.json`, defaulting to disabled for any absent or
    unreadable config -- a corrupt config must degrade to "off", never
    crash an unrelated `cortex` command."""
    try:
        raw = _config_path(cortex_dir).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {"enabled": False}
    if not isinstance(data, dict) or not isinstance(data.get("enabled"), bool):
        return {"enabled": False}
    return data


def write_config(cortex_dir: Path, data: dict) -> None:
    """Write `watcher.json` atomically (tmp + `os.replace`), the same
    corruption-avoidance pattern `_manifest.py` uses for the canonical
    manifest."""
    config_path = _config_path(cortex_dir)
    tmp_path = cortex_dir / f".{WATCHER_CONFIG_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, config_path)


# -- liveness lock: watcher.lock ---------------------------------------------

LOCK_STOPPED = "stopped"
LOCK_RUNNING = "running"
LOCK_STALE = "stale"


@dataclasses.dataclass(frozen=True, slots=True)
class LockProbe:
    """The outcome of one liveness probe against `watcher.lock`.

    `state` is derived from the OS lock itself, never from a
    stored PID: `LOCK_RUNNING` means some process holds the lock right
    now, `LOCK_STALE` means the lock is free but a previous holder left
    metadata behind without clearing it (a crash), and `LOCK_STOPPED`
    means the lock is free and empty (never started, or a clean stop).
    `metadata` is the last holder-written payload, best-effort -- present
    for `running`/`stale`, `None` for `stopped` or on any parse failure.
    """

    state: str
    metadata: dict | None


def _lock_path(cortex_dir: Path) -> Path:
    return cortex_dir / WATCHER_LOCK_FILENAME


def probe_lock(cortex_dir: Path) -> LockProbe:
    """Non-destructively determine whether a watcher currently holds the
    lock, without ever becoming the holder.

    Opens (creating if absent) and attempts a non-blocking exclusive
    `flock`. Success means nobody holds it right now, so it is released
    immediately and the leftover content (if any) is read to distinguish
    a clean stop from a crash. Failure means a live process holds it.
    """
    if fcntl is None:
        return LockProbe(state=LOCK_STOPPED, metadata=None)

    lock_path = _lock_path(cortex_dir)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            metadata = _read_lock_metadata(fd)
            return LockProbe(state=LOCK_RUNNING, metadata=metadata)

        try:
            metadata = _read_lock_metadata(fd)
            state = LOCK_STALE if metadata else LOCK_STOPPED
            return LockProbe(state=state, metadata=metadata)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_lock_metadata(fd: int) -> dict | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 65536)
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def peek_lock_metadata(cortex_dir: Path) -> dict | None:
    """Read whatever metadata is currently in `watcher.lock`, WITHOUT
    ever attempting to acquire it.

    This exists only because `probe_lock` cannot be used to wait for a
    just-spawned child to come up: probing acquires the lock (however
    briefly) to test it, and a probe landing in the narrow window before
    the child's own first `flock()` attempt would win that race and make
    the child see "already held" -- exiting immediately under the
    fail-fast contract below, even though the only other claimant was a
    transient reader. Waiting on this function instead (see
    `_wait_until_running`) can never contend with anyone: a fresh child
    only ever writes metadata AFTER it already holds the lock, so seeing
    it here needs no acquisition of its own.
    """
    if fcntl is None:
        return None
    try:
        fd = os.open(_lock_path(cortex_dir), os.O_RDONLY)
    except OSError:
        return None
    try:
        return _read_lock_metadata(fd)
    finally:
        os.close(fd)


def _write_lock_metadata(fd: int, metadata: dict) -> None:
    payload = json.dumps(metadata).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload)
    os.ftruncate(fd, len(payload))
    os.fsync(fd)


class _HeldLock:
    """The lock, held for the lifetime of one watcher process.

    Kept as a thin wrapper around a raw fd rather than a `Path`+bool so
    the fd -- the thing the kernel actually tracks -- cannot be dropped
    and silently release the lock early; only `release()` closes it.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def update_metadata(self, metadata: dict) -> None:
        _write_lock_metadata(self._fd, metadata)

    def clear_metadata(self) -> None:
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.fsync(self._fd)

    def release(self) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


def try_acquire_lock(cortex_dir: Path) -> _HeldLock | None:
    """Attempt to become the watcher for this workspace. Returns `None`
    immediately (never blocks, never retries) if another process already
    holds the lock -- the fail-fast contract: the loser never
    opens the store and never produces a second observation stream."""
    if fcntl is None:
        return None
    lock_path = _lock_path(cortex_dir)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return _HeldLock(fd)


# -- log: watcher.log ---------------------------------------------------------


def _log_path(cortex_dir: Path) -> Path:
    return cortex_dir / WATCHER_LOG_FILENAME


def log_line(cortex_dir: Path, message: str) -> None:
    """Append one bounded diagnostic line. Best-effort: a logging failure
    must never be the reason the watcher stops observing changes."""
    try:
        log_path = _log_path(cortex_dir)
        if log_path.exists() and log_path.stat().st_size >= _LOG_MAX_BYTES:
            log_path.write_text("", encoding="utf-8")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    except OSError:
        pass


# -- fingerprint and scope ----------------------------------------------------


def _stat_fingerprint(full_path: Path) -> tuple[int, int] | None:
    """The cheap tier-1 fingerprint: `(mtime_ns, size)` from a
    non-symlink-following stat. Returns `None` for anything missing or not
    a regular file -- a symlink swapped in for a tracked path reads as
    "changed", and the eventual `Cortex.seed()` call rejects it through
    the existing `resolve_seed_path` regular-file check; this module adds
    no new symlink policy of its own -- including NOT filtering out
    non-regular dirents here: doing so would make them invisible to the
    settle/observe pipeline instead of reaching `Cortex.seed()`'s refusal
    (the earlier shape of this function did exactly that, silently
    hiding e.g. a symlinked path forever instead of ever refusing it)."""
    try:
        info = os.stat(full_path, follow_symlinks=False)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _scan_scope(cortex: Cortex) -> list[str]:
    """The current watched paths: `Cortex.watcher_scope()`, filtered for
    editor/temp artifacts that can transiently match the allowlist (see
    `_looks_like_temp_name`). Recomputed on every call -- cheap by
    construction, since neither half of `watcher_scope()` walks a tree."""
    scope = cortex.watcher_scope()
    return sorted(path for path in scope if not _looks_like_temp_name(Path(path).name))


def _is_transient_source_error(exc: CortexSourceError) -> bool:
    """True if `exc` wraps an `OSError` (EACCES, EMFILE, EIO, ESTALE, a
    file vanishing between discovery and read, ...) rather than a
    content-dependent refusal (oversize, binary, invalid UTF-8, symlink
    escape/loop, secret name, ...).

    `_source.py` raises `CortexSourceError` for both, but every one of its
    `except OSError as exc: raise CortexSourceError(...) from exc` sites
    (and `_seed_one`'s own `except RuntimeError as exc: raise
    CortexSourceError(...) from exc` for a self-referential symlink,
    which must stay classified as permanent) sets `__cause__` via `raise
    ... from exc`, while every content-dependent refusal raises bare. That
    chained cause is enough to tell "the SAME bytes will never seed
    successfully" apart from "this attempt could not read the bytes at
    all" without CortexSourceError growing a subclass or `_source.py`
    growing a second exception type."""
    return isinstance(exc.__cause__, OSError)


def _seed_one(cortex: Cortex, path: str):
    """`cortex.seed([path])[0]`, with one `pathlib` detail folded into
    the `CortexSourceError` refusal both callers already handle: a
    self-referential or excessively deep symlink makes `Path.resolve()`
    raise `RuntimeError` ("Symlink loop from ..."), not `OSError` --
    `resolve_seed_path` catches only the latter. A single bad path must
    never crash the watcher (the same principle applied to oversized/
    binary files), so it is folded in here rather than left to
    propagate as an unhandled exception out of the observe loop."""
    try:
        (result,) = cortex.seed([path])
    except RuntimeError as exc:
        raise CortexSourceError(f"Refusing to seed {path!r}: {exc}") from exc
    return result


# -- baseline / reconciliation -------------------------------------------------


def _reconcile_baseline(
    cortex: Cortex,
    scope: list[str],
    *,
    retro_observe: bool,
    cortex_dir: Path,
    stats: "_RuntimeStats",
    should_stop: Callable[[], bool],
) -> dict[str, tuple[int, int]]:
    """Build the in-memory baseline map for `scope`.

    `retro_observe=False` (first-ever enable): every path is baselined at
    its CURRENT on-disk fingerprint with no comparison and no read --
    "start observing from now", producing zero observations even if a
    scope path happens to already be a tracked Source with drifted
    content. This is `cortex init dev`'s contract: enabling is not the
    same claim as having watched continuously.

    `retro_observe=True` (process (re)start while already enabled): every
    scope path that IS a tracked Source is re-seeded -- `Cortex.seed()`
    hashes it and compares against `Source.latest_observation.digest`,
    recording exactly one observation if it differs and nothing if it
    does not. A scope path that is NOT yet a tracked Source is baselined
    like the enable case, never retroactively observed -- acquiring
    pre-existing content has always been `cortex seed`'s job, not the
    watcher's. This is what recovers a change made while the watcher was
    down, from canonical data rather than from a lost in-memory cache.

    `should_stop` is polled once per path: a large tracked scope can make
    this pass take seconds, and a `stop`/disable landing mid-pass must be
    able to cut it short rather than finish reconciling (and potentially
    observing) paths against a switch that has already been flipped off.
    The partial baseline built so far is returned as-is -- correct by
    construction, since every entry in it already reflects a path that
    finished being reconciled.

    Baseline causality: a path's baseline entry is set to the fingerprint
    that was actually read BEFORE the seed attempt, never a fresh
    post-write stat -- the file can change again while `seed()` runs, and
    adopting that newer state as "already seen" would lose it silently.
    On a transient/retryable failure -- a `CortexError` that is not a
    `CortexSourceError` at all, or a `CortexSourceError` wrapping an
    `OSError` rather than a permanent, content-dependent refusal (see
    `_is_transient_source_error`) -- the path is left OUT of the
    returned baseline entirely, so the observe loop treats it as
    unbaselined and retries it on its very next scan instead of silently
    dropping the pending change.
    """
    tracked_paths = {source.path for source in cortex.sources()}
    baseline: dict[str, tuple[int, int]] = {}
    for path in scope:
        if should_stop():
            break
        full_path = cortex.path / path
        fingerprint = _stat_fingerprint(full_path)
        if fingerprint is None:
            continue
        if retro_observe and path in tracked_paths:
            try:
                result = _seed_one(cortex, path)
                if result.status != SEED_UNCHANGED:
                    stats.observation_count += 1
                    stats.last_observation_path = path
                    stats.last_observation_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    log_line(cortex_dir, f"reconciled {result.status} {path}")
            except CortexSourceError as exc:
                stats.last_error = str(exc)
                if _is_transient_source_error(exc):
                    # I/O failure, not a content-dependent refusal (see
                    # `_is_transient_source_error`): nothing was recorded,
                    # so this path must not be baselined at all -- same
                    # as the `CortexError` branch below -- letting the
                    # next reconciliation retry it instead of treating a
                    # never-attempted read as already handled.
                    log_line(cortex_dir, f"error {path}: {exc}")
                    continue
                # Permanent, content-dependent refusal: the SAME bytes
                # will never seed successfully, so the baseline still
                # advances to the fingerprint that was attempted --
                # otherwise every future scan would re-refuse it forever.
                log_line(cortex_dir, f"refused {path}: {exc}")
            except CortexError as exc:
                # Transient/environment failure (e.g. a locked store): no
                # observation was actually recorded, so this path must
                # not be baselined at all -- see the docstring above.
                stats.last_error = str(exc)
                log_line(cortex_dir, f"error {path}: {exc}")
                continue
        baseline[path] = fingerprint
    return baseline


# -- rate limiting --------------------------------------------------------------


class _RateLimiter:
    """Caps writes to `max_per_second` over a rolling one-second window,
    sleeping just long enough to stay under it rather than dropping or
    batching anything."""

    def __init__(self, max_per_second: int) -> None:
        self._max_per_second = max_per_second
        self._recent: deque[float] = deque()

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        while self._recent and now - self._recent[0] >= 1.0:
            self._recent.popleft()
        if len(self._recent) >= self._max_per_second:
            sleep_for = 1.0 - (now - self._recent[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._recent and now - self._recent[0] >= 1.0:
                self._recent.popleft()
        self._recent.append(time.monotonic())


# -- adaptive interval ------------------------------------------------------


def _target_interval(seconds_since_last_change: float, last_scan_duration: float) -> float:
    if seconds_since_last_change < _ACTIVE_WINDOW:
        base = _ACTIVE_INTERVAL
    elif seconds_since_last_change < _DEEP_IDLE_WINDOW:
        base = _IDLE_INTERVAL
    else:
        base = _DEEP_IDLE_INTERVAL
    low = max(_MIN_INTERVAL, last_scan_duration * _SCAN_COST_FACTOR)
    return min(_MAX_INTERVAL, max(base, low))


# -- the running loop ----------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _RuntimeStats:
    observation_count: int = 0
    last_observation_path: str | None = None
    last_observation_at: str | None = None
    last_error: str | None = None


def _run_loop(
    cortex: Cortex,
    lock: _HeldLock,
    cortex_dir: Path,
    stats: _RuntimeStats,
    should_stop: Callable[[], bool],
    baseline: dict[str, tuple[int, int]],
) -> None:
    """The polling/settle/observe loop, starting from an already-computed
    `baseline` -- built by the caller via `_reconcile_baseline`, with
    `retro_observe` set according to whether this is a first-ever enable
    (False) or a process (re)start (True). Runs until `should_stop()`
    returns True (the SIGTERM/SIGINT handlers, or `watcher.json` having
    been disabled since this process started -- see `_child_main`'s
    `_should_stop`) or an unrecoverable condition (workspace identity
    changed, manifest gone) is hit.
    """
    pending: dict[str, tuple[tuple[int, int], float]] = {}
    limiter = _RateLimiter(_MAX_WRITES_PER_SECOND)
    last_change_monotonic = time.monotonic()

    while not should_stop():
        scan_start = time.monotonic()
        try:
            manifest = read_manifest(cortex_dir)
        except CortexManifestError as exc:
            log_line(cortex_dir, f"error: workspace manifest unreadable, exiting: {exc}")
            return
        if manifest["cortex_id"] != cortex.cortex_id:
            log_line(cortex_dir, "workspace identity changed, exiting")
            return

        try:
            scope = _scan_scope(cortex)
        except OSError as exc:
            log_line(cortex_dir, f"error: scope resolution failed: {exc}")
            scope = list(baseline)

        current_paths = set(scope)
        for stale_path in set(baseline) - current_paths:
            baseline.pop(stale_path, None)
        for stale_path in set(pending) - current_paths:
            pending.pop(stale_path, None)

        now = time.monotonic()
        for path in scope:
            fingerprint = _stat_fingerprint(cortex.path / path)
            if fingerprint is None:
                pending.pop(path, None)
                continue
            if baseline.get(path) == fingerprint:
                pending.pop(path, None)
                continue

            prior = pending.get(path)
            if prior is not None and prior[0] == fingerprint and now - prior[1] >= SETTLE_SECONDS:
                limiter.wait_if_needed()
                # `fingerprint` (== `prior[0]`) is the state that SETTLED
                # and was handed to `_seed_one` below -- the baseline must
                # advance to exactly that value on success, never to a
                # fresh post-write stat, since the file can already have
                # changed again while `_seed_one` was running.
                baseline_advances = False
                try:
                    result = _seed_one(cortex, path)
                    if result.status != SEED_UNCHANGED:
                        stats.observation_count += 1
                        stats.last_observation_path = path
                        stats.last_observation_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        log_line(cortex_dir, f"observed {result.status} {path}")
                    baseline_advances = True
                except CortexSourceError as exc:
                    stats.last_error = str(exc)
                    if _is_transient_source_error(exc):
                        # I/O failure, not a content-dependent refusal
                        # (see `_is_transient_source_error`): nothing was
                        # recorded, so the baseline must NOT advance --
                        # same as the `CortexError` branch below -- so
                        # the next scan retries this exact change instead
                        # of treating a never-attempted read as handled.
                        log_line(cortex_dir, f"error {path}: {exc}")
                    else:
                        # Permanent, content-dependent refusal: retrying
                        # the SAME bytes can never succeed, so the
                        # baseline still advances to the attempted
                        # fingerprint -- otherwise every scan would
                        # re-refuse it forever. A later edit produces a
                        # new fingerprint and is evaluated fresh.
                        log_line(cortex_dir, f"refused {path}: {exc}")
                        baseline_advances = True
                except CortexError as exc:
                    # Transient/environment failure (e.g. a locked
                    # store): nothing was actually recorded, so the
                    # baseline must NOT advance -- leaving `pending`
                    # unchanged below means the next scan retries this
                    # exact change instead of silently losing it.
                    stats.last_error = str(exc)
                    log_line(cortex_dir, f"error {path}: {exc}")

                if baseline_advances:
                    baseline[path] = fingerprint
                    pending.pop(path, None)
                last_change_monotonic = now
                lock.update_metadata(_lock_metadata(cortex, stats))
            else:
                pending[path] = (fingerprint, now)
                last_change_monotonic = now

        last_scan_duration = time.monotonic() - scan_start
        interval = _target_interval(now - last_change_monotonic, last_scan_duration)
        _interruptible_sleep(interval, should_stop)


def _interruptible_sleep(total_seconds: float, should_stop) -> None:
    deadline = time.monotonic() + total_seconds
    while not should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_SHUTDOWN_POLL_SECONDS, remaining))


def _lock_metadata(cortex: Cortex, stats: _RuntimeStats) -> dict:
    return {
        "pid": os.getpid(),
        "started_at": _PROCESS_STARTED_AT,
        "cortex_id": cortex.cortex_id,
        "cortex_version": _cortex_version(),
        "executable": sys.executable,
        "observation_count": stats.observation_count,
        "last_observation_path": stats.last_observation_path,
        "last_observation_at": stats.last_observation_at,
        "last_error": stats.last_error,
    }


def _cortex_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return "unknown"
    try:
        return version("cortex-memory")
    except PackageNotFoundError:
        return "unknown"


_PROCESS_STARTED_AT: str = ""

# Bound on how long a spawn call waits for the new child to confirm it
# actually holds the lock before returning control to the CLI command
# that triggered it -- purely a UX/determinism nicety (acquiring the lock
# and writing its first metadata takes milliseconds), never load-bearing:
# a timeout still returns the same action code, since the child keeps
# starting up regardless of whether anyone waited for it.
_SPAWN_CONFIRM_TIMEOUT = 2.0


# -- process launcher ---------------------------------------------------------


def spawn_watcher(cortex: Cortex, *, fresh: bool) -> int:
    """Launch the detached watcher child for `cortex` and return its pid.

    An argument list, never a shell string: no injection surface. Run
    with `sys.executable` so the active interpreter/venv is what actually
    runs, never a `cortex` shim that might not be on `PATH` at all.
    `start_new_session=True` gives the child its own session/process
    group so it survives the launching terminal closing;
    `stdin`/`stdout`/`stderr` are all `DEVNULL` so it can neither
    block on nor write into a terminal that later goes away. The child
    re-resolves and re-opens the workspace itself from `--workspace`.
    """
    argv = [sys.executable, "-m", "cortex_memory._watcher", "--workspace", str(cortex.path)]
    if fresh:
        argv.append("--fresh")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return process.pid


def _wait_until_running(cortex_dir: Path, expected_pid: int, timeout: float = _SPAWN_CONFIRM_TIMEOUT) -> bool:
    """Wait for the freshly spawned `expected_pid` to publish its lock
    metadata -- i.e. to have finished reconciling its baseline and
    started the observe loop (see `_child_main`'s publish-after-baseline
    ordering). Polls via `peek_lock_metadata`, never `probe_lock`: the
    latter's brief acquire-and-release would race the child's own first
    `flock()` attempt (see that function's docstring). Matching
    `expected_pid` specifically -- not just "some metadata exists" --
    rules out a false-positive read of a PREVIOUS holder's stale
    metadata left behind by a crash.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metadata = peek_lock_metadata(cortex_dir)
        if metadata is not None and metadata.get("pid") == expected_pid:
            return True
        time.sleep(0.05)
    metadata = peek_lock_metadata(cortex_dir)
    return metadata is not None and metadata.get("pid") == expected_pid


def _terminate_and_wait(cortex_dir: Path, pid: int, timeout: float = _STOP_WAIT_SECONDS) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe_lock(cortex_dir).state != LOCK_RUNNING:
            return
        time.sleep(0.1)


# -- CLI-facing operations ------------------------------------------------------

ACTION_STARTED = "started"
ACTION_ALREADY_RUNNING = "already_running"
ACTION_RESTARTED = "restarted"
ACTION_UNAVAILABLE = "unavailable"
ACTION_STOPPED = "stopped"
ACTION_ALREADY_STOPPED = "already_stopped"
ACTION_STOP_TIMEOUT = "stop_timeout"


@dataclasses.dataclass(frozen=True, slots=True)
class WatcherAction:
    """What one control operation (`enable_and_start`/`supervise`/
    `stop_watcher`) did, for the CLI layer to render into a message --
    this module returns facts, `_cli.py` owns wording."""

    code: str
    pid: int | None = None


ACTION_UNAVAILABLE_RESULT = WatcherAction(ACTION_UNAVAILABLE)


def _require_dev_profile(cortex: Cortex) -> None:
    if cortex.profile != PROFILE_DEV:
        raise ValueError(
            f"The project watcher is only available in the {PROFILE_DEV!r} profile; "
            f"this workspace is {cortex.profile!r}"
        )


def enable_and_start(cortex: Cortex) -> WatcherAction:
    """`cortex init dev`'s and `cortex watch start`'s shared entry point:
    idempotently make the Dev watcher enabled and running.

    A first-ever call (config absent or `enabled=false`) flips the switch
    and launches a FRESH watcher -- zero-observation baseline.
    Every subsequent call is exactly the same supervision opportunity any
    other Cortex command performs (`supervise()`): a healthy running
    watcher is left alone, a stale one is restarted with reconciliation.
    Raises `ValueError` outside the `dev` profile. Never raises for an
    unavailable platform backend -- returns `ACTION_UNAVAILABLE` instead,
    for the CLI to render as a professional status line, not a crash.
    """
    _require_dev_profile(cortex)
    if fcntl is None:
        return ACTION_UNAVAILABLE_RESULT

    cortex_dir = cortex.path / CORTEX_DIRNAME
    config = read_config(cortex_dir)
    if not config.get("enabled", False):
        write_config(cortex_dir, {"enabled": True})
        pid = spawn_watcher(cortex, fresh=True)
        _wait_until_running(cortex_dir, pid)
        return WatcherAction(ACTION_STARTED, pid=pid)

    action = supervise(cortex)
    return action if action is not None else WatcherAction(ACTION_ALREADY_RUNNING)


def supervise(cortex: Cortex) -> WatcherAction | None:
    """The recovery hook every normal Cortex command performs:
    a cheap no-op unless this is a `dev` workspace with the watcher
    enabled, in which case it restarts a stale watcher (with
    reconciliation) or a version-mismatched one, and otherwise leaves a
    healthy one running untouched.

    Returns `None` when there was nothing to check (not `dev`, not
    enabled, or the platform backend is unavailable) so callers can tell
    "nothing to report" apart from "checked, and it is healthy".
    """
    if fcntl is None or cortex.profile != PROFILE_DEV:
        return None
    cortex_dir = cortex.path / CORTEX_DIRNAME
    config = read_config(cortex_dir)
    if not config.get("enabled", False):
        return None

    probe = probe_lock(cortex_dir)
    if probe.state == LOCK_RUNNING:
        metadata = probe.metadata or {}
        pid = metadata.get("pid")
        pid = pid if isinstance(pid, int) else None
        if metadata.get("cortex_version") != _cortex_version():
            # Package-upgrade policy: stop the mismatched holder
            # and start a fresh one of the current version, logging once.
            if pid is not None:
                _terminate_and_wait(cortex_dir, pid)
            log_line(cortex_dir, f"version mismatch ({metadata.get('cortex_version')!r}), restarting")
            new_pid = spawn_watcher(cortex, fresh=False)
            _wait_until_running(cortex_dir, new_pid)
            return WatcherAction(ACTION_RESTARTED, pid=new_pid)
        return WatcherAction(ACTION_ALREADY_RUNNING, pid=pid)

    # `stopped` or `stale`: nothing healthy holds the lock right now.
    new_pid = spawn_watcher(cortex, fresh=False)
    _wait_until_running(cortex_dir, new_pid)
    return WatcherAction(ACTION_RESTARTED, pid=new_pid)


def stop_watcher(cortex: Cortex) -> WatcherAction:
    """`cortex watch stop`: persistently disable the watcher and stop any
    running process. Disables the config FIRST, before signaling, so a
    command racing this one can never resurrect it through `supervise()`
    -- `stop` is deliberately persistent: there is no separate
    temporary-stop state to fall back into. Raises `ValueError` outside
    the `dev` profile.
    """
    _require_dev_profile(cortex)
    if fcntl is None:
        return ACTION_UNAVAILABLE_RESULT
    cortex_dir = cortex.path / CORTEX_DIRNAME
    write_config(cortex_dir, {"enabled": False})

    probe = probe_lock(cortex_dir)
    if probe.state != LOCK_RUNNING:
        return WatcherAction(ACTION_ALREADY_STOPPED)

    pid = (probe.metadata or {}).get("pid")
    pid = pid if isinstance(pid, int) else None
    if pid is not None:
        _terminate_and_wait(cortex_dir, pid)

    if probe_lock(cortex_dir).state == LOCK_RUNNING:
        return WatcherAction(ACTION_STOP_TIMEOUT, pid=pid)
    return WatcherAction(ACTION_STOPPED, pid=pid)


def _missing_sources_count(cortex: Cortex) -> int:
    return sum(1 for source in cortex.sources() if not (cortex.path / source.path).is_file())


def status_lines(cortex: Cortex, *, detailed: bool = False) -> list[str]:
    """The `Watcher: ...` line(s) for `cortex status` (one line,
    `detailed=False`) and `cortex watch status` (the fuller dashboard,
    `detailed=True`).

    Outside the `dev` profile, `detailed=False` returns an empty list --
    non-dev `cortex status` output is byte-for-byte unchanged by this
    module's presence -- while `detailed=True` (an explicit `cortex
    watch status` call) says
    plainly that the watcher does not apply here, rather than printing
    nothing for a command the user explicitly ran.
    """
    if cortex.profile != PROFILE_DEV:
        return [f"Watcher: not available outside the {PROFILE_DEV!r} profile"] if detailed else []
    if fcntl is None:
        return ["Watcher: unavailable (background watching needs Linux in this release)"]

    cortex_dir = cortex.path / CORTEX_DIRNAME
    config = read_config(cortex_dir)
    if not config.get("enabled", False):
        return ["Watcher: disabled"]

    probe = probe_lock(cortex_dir)
    if probe.state == LOCK_STALE:
        return ["Watcher: stale (enabled but not running; it will restart on the next command)"]
    if probe.state == LOCK_STOPPED:
        return ["Watcher: stopped"]

    metadata = probe.metadata or {}
    pid = metadata.get("pid")
    summary = f"Watcher: running (pid {pid})" if isinstance(pid, int) else "Watcher: running"
    if not detailed:
        return [summary]

    lines = [summary]
    started_at = metadata.get("started_at")
    if isinstance(started_at, str) and started_at:
        lines.append(f"  started: {started_at}")
    lines.append(f"  observations this run: {metadata.get('observation_count', 0)}")
    last_path = metadata.get("last_observation_path")
    if last_path:
        lines.append(f"  last observation: {metadata.get('last_observation_at')} {last_path}")
    last_error = metadata.get("last_error")
    if last_error:
        lines.append(f"  last error: {last_error}")
    missing = _missing_sources_count(cortex)
    if missing:
        plural = "s" if missing != 1 else ""
        lines.append(f"  tracked source{plural} missing on disk: {missing}")
    return lines


# -- child process entry point -------------------------------------------------


def _child_main(argv: list[str]) -> int:
    global _PROCESS_STARTED_AT
    import argparse

    parser = argparse.ArgumentParser(prog="cortex-memory-watcher")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    cortex_dir = workspace / CORTEX_DIRNAME

    lock = try_acquire_lock(cortex_dir)
    if lock is None:
        # Another watcher already owns this workspace. Exit quietly and
        # immediately -- no store is opened, so no second observation
        # stream can exist.
        return 3

    _PROCESS_STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats = _RuntimeStats()
    log_line(cortex_dir, f"start pid={os.getpid()} fresh={args.fresh}")

    stop_requested = False

    def _handle_signal(signum, frame) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    # Registered before any work happens (including opening the workspace
    # and reconciling), so a stop request during a slow reconciliation
    # pass is at least recorded immediately, not lost to Python's default
    # SIGTERM handling.
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _should_stop() -> bool:
        # `stop_watcher` writes `watcher.json` BEFORE it can even learn
        # this process's pid (the pid is only published in
        # `watcher.lock`'s metadata once, after reconciliation -- see the
        # `lock.update_metadata` call below), so during that window a
        # racing `stop`/disable can find no pid to signal at all. Signal
        # delivery must therefore never be the ONLY way this process
        # learns it should stop: every poll of `should_stop` -- inside a
        # long `_reconcile_baseline` pass, at the top of `_run_loop`'s
        # scan, and every `_SHUTDOWN_POLL_SECONDS` tick of its sleep --
        # also rereads the persistent switch directly, so a disable is
        # always noticed on its own, independent of whether SIGTERM ever
        # reached (or could reach) this pid.
        return stop_requested or not read_config(cortex_dir).get("enabled", False)

    try:
        cortex = Cortex.open(workspace)
    except CortexError as exc:
        log_line(cortex_dir, f"error: cannot open workspace, exiting: {exc}")
        lock.release()
        return 1

    try:
        # `--fresh` (first-ever enable) baselines with `retro_observe=False`:
        # zero observations, even for a scope path that happens to already
        # be a tracked Source. Any other start (restart, crash recovery,
        # `watch start` after `stop`) reconciles tracked Sources against
        # their canonical latest digest -- see `_reconcile_baseline`.
        scope = _scan_scope(cortex)
        baseline = _reconcile_baseline(
            cortex,
            scope,
            retro_observe=not args.fresh,
            cortex_dir=cortex_dir,
            stats=stats,
            should_stop=_should_stop,
        )
        log_line(
            cortex_dir,
            f"baseline established: {len(baseline)} path(s), "
            f"{stats.observation_count} reconciled observation(s)",
        )
        # Metadata (and therefore "running" as any prober -- including
        # `_wait_until_running` -- observes it) is published only AFTER
        # the baseline is fixed, never before: a caller that treats
        # "confirmed running" as "safe to start caring about changes from
        # this point" must never be able to race the reconciliation pass
        # itself. Publishing it earlier (right after acquiring the lock)
        # was tried and measurably wrong -- a write landing in that
        # window could be silently absorbed into the baseline instead of
        # being seen as a change.
        lock.update_metadata(_lock_metadata(cortex, stats))
        _run_loop(cortex, lock, cortex_dir, stats, _should_stop, baseline)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        log_line(cortex_dir, f"error: unhandled exception, exiting: {exc}")
    finally:
        log_line(cortex_dir, f"stop pid={os.getpid()} observations={stats.observation_count}")
        lock.clear_metadata()
        lock.release()

    return 0


def main(argv: list[str] | None = None) -> int:
    return _child_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
