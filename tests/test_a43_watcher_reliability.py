"""Focused regression tests for Dev watcher reliability behaviors: stop
during a slow restart, baseline causality, retryable-failure handling,
self-referential symlinks, and the transient-vs-permanent I/O
classification in the settle/observe path.

Each test targets ONE failure mechanism rather than the watcher's
behavior in general -- `test_a43_dev_watcher.py` already covers the
broad surface. The stop-during-restart and self-referential-symlink
cases are inherently cross-process races / discovery-path bugs and are
reproduced against a real detached child (the same style
`test_a43_dev_watcher.py` uses). The baseline-causality, retryable-
failure, and transient-I/O-classification cases are in-process,
single-file causality bugs in `_run_loop`'s settle/observe step; the
first two are reproduced deterministically by driving `_run_loop`
directly against a real `Cortex` with a narrowly monkeypatched `seed`,
rather than racing real wall-clock timing against a subprocess for an
outcome that does not actually depend on cross-process scheduling. The
transient-I/O cases use a real `os.chmod` permission failure instead of
fault injection, for the same reason `test_symlink_loop_does_not_hang_
and_is_refused` uses a real symlink: this is an environment failure, not
a logic fault, so the test should hit the real one.
"""

from __future__ import annotations

import os
import signal
import threading
import time

from cortex_memory import Cortex
from cortex_memory import _watcher
from cortex_memory._errors import CortexSourceError, CortexStorageError


def _wait_for(predicate, timeout=8.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _observation_count(cx: Cortex, path: str) -> int:
    for source in cx.sources():
        if source.path == path:
            return len(source.observations)
    return 0


def _init_dev(tmp_path, **files) -> Cortex:
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return Cortex.init(tmp_path, "dev")


def _running_pid(cx: Cortex) -> int | None:
    probe = _watcher.probe_lock(cx.path / ".cortex")
    if probe.state != _watcher.LOCK_RUNNING:
        return None
    pid = (probe.metadata or {}).get("pid")
    return pid if isinstance(pid, int) else None


def _run_loop_in_thread(cx, lock, stats, baseline, stop_event):
    """Drive `_run_loop` in a background thread, in-process, against a
    real `Cortex` and a real held lock -- no subprocess, so a
    monkeypatched `cx.seed` can inject a deterministic race. The caller
    MUST release/clear `lock` itself once the thread has stopped: this
    module reuses `os.getpid()` (the TEST process) as the "holder" pid in
    `watcher.lock` metadata for as long as the lock is held, and
    `tests/conftest.py`'s leaked-watcher cleanup force-kills whatever pid
    it finds there for any lock it sees as still RUNNING.
    """
    thread = threading.Thread(
        target=_watcher._run_loop,
        args=(cx, lock, cx.path / ".cortex", stats, stop_event.is_set, baseline),
        daemon=True,
    )
    thread.start()
    return thread


# -- stop/disable during a slow restart must actually stop the watcher -------


def test_stop_during_slow_restart_reconciliation_actually_stops(tmp_path):
    """`stop_watcher` writes `watcher.json` to disabled and best-effort
    signals the current holder's pid -- but during a RESTART's
    reconciliation (retro_observe=True: every tracked Source is re-seeded
    against its canonical digest) the freshly spawned child holds the
    lock for measurable real time before it ever publishes that pid in
    `watcher.lock`'s metadata (`_child_main` publishes only after the
    baseline is fixed). A `stop` landing in that window finds no pid to
    signal at all. Without a persistent-switch recheck, nothing else made
    the child notice `enabled: false`: it finished reconciling and then
    kept observing indefinitely. Instead, the child rereads the
    persistent switch on its own and exits regardless of whether any
    signal reached it.
    """
    file_count = 300
    files = {f"docs/f{i}.md": f"content {i}\n" for i in range(file_count)}
    cx = _init_dev(tmp_path, **files)
    cx.seed(list(files))  # track all of them as Sources up front

    action = _watcher.enable_and_start(cx)
    assert action.code == _watcher.ACTION_STARTED
    first_pid = _running_pid(cx)
    assert first_pid is not None

    # Crash it, then change every tracked file so the eventual restart's
    # retro_observe reconciliation has `file_count` real seed()-and-commit
    # calls to perform -- measured at several real seconds for this file
    # count, comfortably longer than `_wait_until_running`'s 2s confirm
    # timeout below.
    os.kill(first_pid, signal.SIGKILL)
    assert _wait_for(lambda: _watcher.probe_lock(cx.path / ".cortex").state == _watcher.LOCK_STALE)
    for name in files:
        (tmp_path / name).write_text("changed\n", encoding="utf-8")

    restart_action = _watcher.supervise(cx)  # blocks up to ~2s; child keeps reconciling regardless
    assert restart_action.code == _watcher.ACTION_RESTARTED
    second_pid = restart_action.pid
    assert second_pid is not None
    assert second_pid != first_pid

    # `supervise` already spent ~2s waiting for a confirmation it never
    # got, so a several-second reconciliation is almost certainly still
    # in flight right now -- exactly the pre-publish window this test
    # targets.
    _watcher.stop_watcher(cx)
    assert _watcher.read_config(cx.path / ".cortex")["enabled"] is False

    # Whatever `stop_watcher` itself reported (its own outcome depends on
    # exactly when the race lands and is not what this test pins down),
    # the child must actually go away -- and stay away.
    assert _wait_for(
        lambda: _watcher.probe_lock(cx.path / ".cortex").state != _watcher.LOCK_RUNNING, timeout=15.0
    )
    assert _running_pid(cx) is None

    # A brand-new, never-tracked file is unambiguous: only an ONGOING
    # `_run_loop` pass -- one that failed to notice it was disabled --
    # could ever pick it up. The restart's own reconciliation (which ran
    # before this file even existed) cannot explain an observation here.
    new_file = tmp_path / "docs" / "post_stop_new.md"
    new_file.write_text("must never be observed\n", encoding="utf-8")
    time.sleep(4.0)  # several scan+settle cycles, had the child wrongly still been alive
    assert _observation_count(cx, "docs/post_stop_new.md") == 0
    assert all(source.path != "docs/post_stop_new.md" for source in cx.sources())

    lines = _watcher.status_lines(cx, detailed=True)
    assert any("disabled" in line for line in lines)


# -- baseline causality must track what was actually processed ---------------


def test_write_during_observation_is_not_silently_lost(tmp_path, monkeypatch):
    """If the watched file changes to B in the gap right after `seed()`
    finishes recording an earlier state, the baseline must not silently
    adopt B's fingerprint as "already seen" -- B must still be detected
    and recorded as its own observation on a later scan.

    Reproduced deterministically: `cx.seed` is wrapped so that, on its
    first call, it performs the REAL seed and then immediately rewrites
    the file underneath it -- landing the race in the exact gap the old
    code's post-write re-stat (`baseline[path] = _stat_fingerprint(...)`)
    used to read from. Racing real wall-clock writes against a real
    subprocess for the same few-millisecond window would not reliably
    reproduce this without either an enormous sample size or brittle
    timing, for no benefit: the bug is a single-process logic error in
    `_run_loop`, not a cross-process one.
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    monkeypatch.setattr(_watcher, "SETTLE_SECONDS", 0.0)
    target = tmp_path / "README.md"

    real_seed = cx.seed
    switched = threading.Event()

    def _racy_seed(paths):
        result = real_seed(paths)  # actually reads/records the CURRENT bytes
        if not switched.is_set():
            switched.set()
            target.write_text("B\n", encoding="utf-8")  # races in right after
        return result

    monkeypatch.setattr(cx, "seed", _racy_seed)

    lock = _watcher.try_acquire_lock(cx.path / ".cortex")
    assert lock is not None
    stats = _watcher._RuntimeStats()
    baseline = {"README.md": _watcher._stat_fingerprint(target)}
    target.write_text("intermediate\n", encoding="utf-8")

    stop_event = threading.Event()
    thread = _run_loop_in_thread(cx, lock, stats, baseline, stop_event)
    try:
        assert _wait_for(lambda: _observation_count(cx, "README.md") >= 1, timeout=8.0)
        assert _wait_for(lambda: _observation_count(cx, "README.md") >= 2, timeout=8.0)
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        lock.clear_metadata()
        lock.release()

    source = [s for s in cx.sources() if s.path == "README.md"][0]
    contents = [cx.get_evidence(o.evidence_id).content for o in source.observations]
    assert contents == ["intermediate\n", "B\n"]


# -- a retryable failure must not advance the baseline ------------------------


def test_retryable_failure_does_not_advance_baseline(tmp_path, monkeypatch):
    """When `seed()` fails with a transient/environment error (a locked
    store is the concrete case this targets), the baseline must NOT
    advance: the next scan has to retry the same change instead of
    treating it as already handled and losing it forever.

    `cx.seed` is wrapped to raise `CortexStorageError` on its first call
    for this path and succeed on every call after -- deterministic fault
    injection, not a real SQLite lock contended from another connection
    (reachable in practice, e.g. a concurrent `cortex` command; this test
    isolates the WATCHER's reaction to that class of failure).
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    monkeypatch.setattr(_watcher, "SETTLE_SECONDS", 0.0)
    target = tmp_path / "README.md"

    real_seed = cx.seed
    attempts: list[object] = []
    should_fail = threading.Event()
    should_fail.set()

    def _flaky_seed(paths):
        attempts.append(paths)
        if should_fail.is_set():
            should_fail.clear()
            raise CortexStorageError("Failed to observe source 'README.md': database is locked")
        return real_seed(paths)

    monkeypatch.setattr(cx, "seed", _flaky_seed)

    lock = _watcher.try_acquire_lock(cx.path / ".cortex")
    assert lock is not None
    stats = _watcher._RuntimeStats()
    baseline = {"README.md": _watcher._stat_fingerprint(target)}
    target.write_text("changed\n", encoding="utf-8")

    stop_event = threading.Event()
    thread = _run_loop_in_thread(cx, lock, stats, baseline, stop_event)
    try:
        assert _wait_for(lambda: len(attempts) >= 1, timeout=8.0)
        # The first attempt failed and must not have recorded anything.
        assert _observation_count(cx, "README.md") == 0
        # A retryable failure must not have been mistaken for "handled":
        # the loop keeps retrying the same unresolved change on its own.
        assert _wait_for(lambda: _observation_count(cx, "README.md") == 1, timeout=8.0)
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        lock.clear_metadata()
        lock.release()

    assert len(attempts) >= 2  # first failed, a later one actually recorded it
    assert _observation_count(cx, "README.md") == 1
    source = [s for s in cx.sources() if s.path == "README.md"][0]
    evidence = cx.get_evidence(source.latest_observation.evidence_id)
    assert evidence.content == "changed\n"


# -- a self-referential symlink in the discovery path must not crash-loop ----


def test_self_referential_symlink_in_discovery_does_not_crash_loop(tmp_path):
    """A self-referential symlink matching the `docs/` discovery
    allowlist makes `Path.resolve()` raise `RuntimeError` deep inside
    `discover_candidate_paths` -- a path `_scan_scope` reaches on EVERY
    scan (via `Cortex.watcher_scope()`), not only through an
    already-tracked Source's `_seed_one` call (see
    `test_symlink_loop_does_not_hang_and_is_refused` in
    `test_a43_dev_watcher.py`, which only exercises the latter and would
    still pass without this fix).

    Without it, this propagated uncaught out of `_child_main`'s very
    first `_scan_scope()` call, exiting the child before it ever
    published its lock metadata -- every subsequent `supervise()` (i.e.
    every normal Cortex command) would see a free lock and respawn a
    doomed child again, each attempt costing `_wait_until_running`'s ~2s
    confirm timeout.
    """
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    (tmp_path / "docs").mkdir()
    loop_path = tmp_path / "docs" / "self.md"
    loop_path.symlink_to(loop_path)

    action = _watcher.enable_and_start(cx)
    assert action.code == _watcher.ACTION_STARTED
    pid = _running_pid(cx)
    assert pid is not None

    # Several "normal Cortex commands" worth of supervision: a healthy
    # watcher must never be restarted, and none of this may cost anywhere
    # near the ~2s a doomed respawn attempt would.
    for _ in range(3):
        start = time.monotonic()
        supervise_action = _watcher.supervise(cx)
        elapsed = time.monotonic() - start
        assert supervise_action.code == _watcher.ACTION_ALREADY_RUNNING
        assert supervise_action.pid == pid
        assert elapsed < 1.0

    assert _running_pid(cx) == pid  # never respawned
    assert "docs/self.md" not in cx.watcher_scope()  # scope resolves without raising
    assert all(source.path != "docs/self.md" for source in cx.sources())

    # and the watcher must still be doing its actual job
    (tmp_path / "README.md").write_text("still working\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)


def _log_text(cx: Cortex) -> str:
    log_path = cx.path / ".cortex" / _watcher.WATCHER_LOG_FILENAME
    return log_path.read_text(encoding="utf-8") if log_path.exists() else ""


# -- a transient I/O failure must not be mistaken for a permanent,
#    content-dependent refusal ------------------------------------------------
#
# `_source.py` raises the same `CortexSourceError` both for content the
# watcher must never retry (oversize, binary, invalid UTF-8, an escaping or
# looping symlink) and for an `OSError` it hit while trying to resolve/stat/
# read a path (EACCES, EMFILE, EIO, ESTALE, the file vanishing between
# discovery and read). Treating every `CortexSourceError` as the former and
# advancing the baseline anyway would silently lose the in-flight change.


def test_tracked_path_transient_permission_failure_does_not_advance_baseline(tmp_path):
    """A file that is already a tracked Source, edited while temporarily
    unreadable (EACCES): the edit must not be lost while the watcher keeps
    running -- it must be retried and recorded once the file is readable
    again, without requiring a restart.
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    cx.seed(["README.md"])
    _watcher.enable_and_start(cx)
    assert _running_pid(cx) is not None
    target = tmp_path / "README.md"

    target.write_text("B\n", encoding="utf-8")
    os.chmod(target, 0o000)
    try:
        assert _wait_for(lambda: "README.md" in _log_text(cx), timeout=8.0)
        assert _observation_count(cx, "README.md") == 1  # B not recorded while unreadable
        assert _running_pid(cx) is not None  # watcher did not crash or exit
        log = _log_text(cx)
        # The classification itself, not just its eventual effect: a
        # transient I/O failure must be logged as a retryable `error`,
        # never as the permanent-refusal `refused` line.
        assert "error README.md:" in log
        assert "refused README.md:" not in log
    finally:
        os.chmod(target, 0o644)

    assert _wait_for(lambda: _observation_count(cx, "README.md") == 2, timeout=8.0)
    source = [s for s in cx.sources() if s.path == "README.md"][0]
    assert source.latest_observation is not None
    evidence = cx.get_evidence(source.latest_observation.evidence_id)
    assert evidence.content == "B\n"


def test_discovery_path_transient_permission_failure_recovers_without_restart(tmp_path):
    """A file that is NOT yet a tracked Source -- only reachable through
    the discovery allowlist -- created unreadable and made readable later:
    it must eventually be seeded while the SAME watcher process keeps
    running, with no restart. `_reconcile_baseline(retro_observe=True)`
    only re-seeds paths already tracked as a Source, so a path that failed
    on its very first attempt would otherwise never be retried even
    across a restart.
    """
    cx = _init_dev(tmp_path, **{"README.md": "keep watcher busy\n"})
    (tmp_path / "docs").mkdir()
    _watcher.enable_and_start(cx)
    assert _running_pid(cx) is not None

    target = tmp_path / "docs" / "note.md"
    target.write_text("first content\n", encoding="utf-8")
    os.chmod(target, 0o000)
    try:
        assert _wait_for(lambda: "docs/note.md" in _log_text(cx), timeout=8.0)
        assert _observation_count(cx, "docs/note.md") == 0
        assert all(s.path != "docs/note.md" for s in cx.sources())
        assert _running_pid(cx) is not None
        log = _log_text(cx)
        assert "error docs/note.md:" in log
        assert "refused docs/note.md:" not in log
    finally:
        os.chmod(target, 0o644)

    assert _wait_for(lambda: _observation_count(cx, "docs/note.md") == 1, timeout=8.0)
    source = [s for s in cx.sources() if s.path == "docs/note.md"][0]
    evidence = cx.get_evidence(source.latest_observation.evidence_id)
    assert evidence.content == "first content\n"


def test_multiple_consecutive_transient_failures_then_success(tmp_path, monkeypatch):
    """Several consecutive transient failures on the same pending change
    must all leave the baseline untouched, with no crash and no observation
    recorded, until the underlying condition clears -- then exactly one
    observation is recorded.
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    monkeypatch.setattr(_watcher, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(_watcher, "_MIN_INTERVAL", 0.05)
    monkeypatch.setattr(_watcher, "_ACTIVE_INTERVAL", 0.05)
    target = tmp_path / "README.md"

    real_seed = cx.seed
    attempts: list[object] = []
    fail_count = 3
    calls_before_success = threading.Event()

    def _flaky_seed(paths):
        attempts.append(paths)
        if len(attempts) <= fail_count:
            raise CortexSourceError(f"Cannot read {paths[0]!r}: [Errno 13] Permission denied") from PermissionError(
                13, "Permission denied"
            )
        calls_before_success.set()
        return real_seed(paths)

    monkeypatch.setattr(cx, "seed", _flaky_seed)

    lock = _watcher.try_acquire_lock(cx.path / ".cortex")
    assert lock is not None
    stats = _watcher._RuntimeStats()
    baseline = {"README.md": _watcher._stat_fingerprint(target)}
    target.write_text("B\n", encoding="utf-8")

    stop_event = threading.Event()
    thread = _run_loop_in_thread(cx, lock, stats, baseline, stop_event)
    try:
        assert _wait_for(lambda: len(attempts) > fail_count, timeout=8.0)
        assert _wait_for(lambda: _observation_count(cx, "README.md") == 1, timeout=8.0)
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        lock.clear_metadata()
        lock.release()

    assert _observation_count(cx, "README.md") == 1
    source = [s for s in cx.sources() if s.path == "README.md"][0]
    evidence = cx.get_evidence(source.latest_observation.evidence_id)
    assert evidence.content == "B\n"


def test_permanent_content_rejection_still_advances_baseline_no_retry_loop(tmp_path, monkeypatch):
    """Non-regression: a genuinely permanent, content-dependent refusal
    (invalid UTF-8, no `OSError` cause) must still advance the baseline
    after exactly one attempt -- transient-I/O handling must not turn
    every refusal into an infinite retry loop.
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    cx.seed(["README.md"])
    monkeypatch.setattr(_watcher, "SETTLE_SECONDS", 0.0)
    target = tmp_path / "README.md"

    real_seed = cx.seed
    attempts: list[object] = []

    def _counting_seed(paths):
        attempts.append(paths)
        return real_seed(paths)

    monkeypatch.setattr(cx, "seed", _counting_seed)

    lock = _watcher.try_acquire_lock(cx.path / ".cortex")
    assert lock is not None
    stats = _watcher._RuntimeStats()
    baseline = {"README.md": _watcher._stat_fingerprint(target)}
    target.write_bytes(b"\xff\xfe not valid utf-8")

    stop_event = threading.Event()
    thread = _run_loop_in_thread(cx, lock, stats, baseline, stop_event)
    try:
        assert _wait_for(lambda: len(attempts) >= 1, timeout=8.0)
        time.sleep(1.0)  # several more scan cycles worth
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        lock.clear_metadata()
        lock.release()

    assert len(attempts) == 1  # refused once, never retried
    assert _observation_count(cx, "README.md") == 1  # still only the original seed
    assert stats.last_error is not None and "UTF-8" in stats.last_error


def test_reconcile_baseline_transient_failure_excluded_from_baseline(tmp_path):
    """The restart path (`_reconcile_baseline(retro_observe=True)`) must
    apply the same transient/permanent distinction: a tracked Source that
    is transiently unreadable at restart time must be excluded from the
    returned baseline so the very next `_run_loop` scan retries it, not
    silently baselined as if it had been (re)observed.
    """
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    cx.seed(["README.md"])
    target = tmp_path / "README.md"
    target.write_text("B\n", encoding="utf-8")
    os.chmod(target, 0o000)
    try:
        stats = _watcher._RuntimeStats()
        baseline = _watcher._reconcile_baseline(
            cx,
            ["README.md"],
            retro_observe=True,
            cortex_dir=cx.path / ".cortex",
            stats=stats,
            should_stop=lambda: False,
        )
    finally:
        os.chmod(target, 0o644)

    assert "README.md" not in baseline
    assert _observation_count(cx, "README.md") == 1  # B was never recorded
    assert stats.last_error is not None and "Permission denied" in stats.last_error
