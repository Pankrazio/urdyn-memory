"""Shared test fixtures.

`urdyn init dev` (A43) starts a real detached background watcher
process. Any test anywhere in this suite that runs it -- directly, or
via `cli_main(["init", "dev"])` -- must not be allowed to leak that
process past its own teardown. This is enforced once, globally, rather
than requiring every test file to opt in: a test file written before
A43 (or one written after it but focused on something unrelated) has no
reason to know a background process might now exist under its `tmp_path`.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from urdyn import _watcher


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _kill_watchers_under(root: Path) -> None:
    if not root.exists():
        return
    for lock_path in root.rglob(_watcher.WATCHER_LOCK_FILENAME):
        urdyn_dir = lock_path.parent
        probe = _watcher.probe_lock(urdyn_dir)
        if probe.state != _watcher.LOCK_RUNNING:
            continue
        pid = (probe.metadata or {}).get("pid")
        if not isinstance(pid, int):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        _wait_until(lambda cd=urdyn_dir: _watcher.probe_lock(cd).state != _watcher.LOCK_RUNNING)


@pytest.fixture(autouse=True)
def _no_leaked_watchers(tmp_path):
    """Force-kill any watcher still holding a lock anywhere under this
    test's `tmp_path` at teardown. Never targets a process outside that
    directory tree, so it cannot affect anything this test did not
    itself create."""
    yield
    _kill_watchers_under(tmp_path)
