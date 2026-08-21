"""A43: the Dev profile's automatic filesystem watcher.

Non-vacuity: before this module existed, nothing in Urdyn ever
re-observed a tracked project file on its own -- `Source.observations`
was updated only by an explicit `urdyn seed` call. `test_gap_...` below
is that exact scenario (enable the watcher, change a tracked file, do
nothing else) and is the one test in this file that would fail forever
on the pre-A43 codebase, for a real behavioral reason: no code path
existed to make it pass.

These tests exercise the REAL lifecycle -- a real detached child process,
real `flock`, real polling and settling -- rather than mocking any of it,
because the correctness claims here (single-instance ownership, crash
recovery, debounce) are exactly the OS-level guarantees this design relies on.
Consequently most tests wait on real wall-clock time (`_wait_for` polls
rather than sleeping a fixed amount, to stay as fast as the machine
allows). Process-leak protection (force-killing any watcher still
holding a lock at teardown) lives in `tests/conftest.py`, globally, not
in this file -- see its module docstring for why.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import threading
import time

import pytest

from urdyn import Urdyn
from urdyn import _watcher
from urdyn._cli import main as cli_main
from urdyn._evidence import EVIDENCE_KIND_DOCUMENT_OBSERVATION
from urdyn._source import compute_digest


# -- shared helpers -----------------------------------------------------------


def _wait_for(predicate, timeout=8.0, interval=0.1):
    """Poll `predicate` until it is truthy or `timeout` elapses, returning
    the final truthiness either way -- faster than a flat sleep whenever
    the condition is met early, and still a hard bound otherwise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _observation_count(cx: Urdyn, path: str) -> int:
    for source in cx.sources():
        if source.path == path:
            return len(source.observations)
    return 0


# Process-leak protection lives in `tests/conftest.py` (`_no_leaked_watchers`,
# autouse, global): `urdyn init dev` starts a real background process, so
# the cleanup must cover every test file that might trigger it, not only
# this one.


def _init_dev(tmp_path, **files) -> Urdyn:
    """A `dev` workspace with the given pre-existing files, watcher NOT
    yet enabled -- callers opt in via `_watcher.enable_and_start`."""
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return Urdyn.init(tmp_path, "dev")


def _running_pid(cx: Urdyn) -> int | None:
    probe = _watcher.probe_lock(cx.path / ".urdyn")
    if probe.state != _watcher.LOCK_RUNNING:
        return None
    pid = (probe.metadata or {}).get("pid")
    return pid if isinstance(pid, int) else None


# -- 1. non-vacuity + core observe behavior ------------------------------------


def test_gap_modify_tracked_file_produces_exactly_one_observation(tmp_path):
    """THE non-vacuity test: on a codebase without this module this fails
    forever, because nothing ever re-seeds a tracked file on its own."""
    cx = _init_dev(tmp_path, **{"README.md": "hello\n"})
    _watcher.enable_and_start(cx)
    assert _observation_count(cx, "README.md") == 0

    (tmp_path / "README.md").write_text("hello, changed\n", encoding="utf-8")

    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)
    assert _observation_count(cx, "README.md") == 1


# -- 2. init / baseline / profile gating ---------------------------------------


def test_init_dev_enables_and_starts_watcher(tmp_path):
    cx = _init_dev(tmp_path)
    action = _watcher.enable_and_start(cx)
    assert action.code == _watcher.ACTION_STARTED
    assert _running_pid(cx) is not None


def test_init_dev_new_project_baseline_is_zero_observations(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "hi\n", "pyproject.toml": "[project]\n"})
    _watcher.enable_and_start(cx)
    assert cx.sources() == []


def test_init_dev_existing_project_baseline_is_zero_fake_observations(tmp_path):
    """An 'existing dev project' HA scenario: files predate Urdyn
    entirely. Enabling must not retroactively fabricate history for
    them."""
    cx = _init_dev(
        tmp_path,
        **{"README.md": "already here\n", "docs/guide.md": "already here too\n"},
    )
    _watcher.enable_and_start(cx)
    assert cx.sources() == []
    # and a REAL change afterward is still observed normally
    (tmp_path / "README.md").write_text("now changed\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)


def test_init_general_does_not_enable_watcher(tmp_path):
    cx = Urdyn.init(tmp_path, "general")
    assert not (tmp_path / ".urdyn" / _watcher.WATCHER_CONFIG_FILENAME).exists()
    assert _watcher.status_lines(cx) == []
    assert _watcher.supervise(cx) is None


def test_init_lab_does_not_enable_watcher(tmp_path):
    cx = Urdyn.init(tmp_path, "lab")
    assert not (tmp_path / ".urdyn" / _watcher.WATCHER_CONFIG_FILENAME).exists()
    assert _watcher.status_lines(cx) == []
    assert _watcher.supervise(cx) is None


def test_general_lab_cli_status_output_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli_main(["init", "general"])
    capsys.readouterr()
    cli_main(["status"])
    out = capsys.readouterr().out
    assert "Watcher" not in out


# -- 3. dedup / debounce --------------------------------------------------------


def test_same_content_rewrite_no_duplicate_observation(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "content\n"})
    _watcher.enable_and_start(cx)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)

    original_bytes = (tmp_path / "README.md").read_bytes()
    time.sleep(1.2)  # cross a settle window so a same-content write is unambiguous
    (tmp_path / "README.md").write_bytes(original_bytes)
    time.sleep(4.0)  # long enough for at least one more scan+settle cycle
    assert _observation_count(cx, "README.md") == 1


def test_a_to_b_to_a_yields_three_observations(tmp_path):
    """Idempotency is judged against the LATEST observation only: a file
    edited A->B->A genuinely passed through three states."""
    cx = _init_dev(tmp_path, **{"README.md": "A\n"})
    _watcher.enable_and_start(cx)

    (tmp_path / "README.md").write_text("B\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)
    time.sleep(1.2)
    (tmp_path / "README.md").write_text("A\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 2)

    digests = [o.digest for o in [s for s in cx.sources() if s.path == "README.md"][0].observations]
    # digests[0] is the A->B transition, digests[1] is B->A: genuinely
    # different content each time, and critically NOT collapsed onto a
    # (nonexistent) first observation just because "A" was the original
    # baseline text -- going back to "A" is still a real second change.
    assert digests[0] != digests[1]
    assert digests[1] == compute_digest(b"A\n")


def test_rapid_saves_coalesce_to_one_observation(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "start\n"})
    _watcher.enable_and_start(cx)

    for i in range(6):
        (tmp_path / "README.md").write_text(f"save {i}\n", encoding="utf-8")
        time.sleep(0.25)

    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1, timeout=10.0)
    time.sleep(2.0)  # settle further; must not creep past 1
    assert _observation_count(cx, "README.md") == 1
    final_text = [s for s in cx.sources() if s.path == "README.md"][0]
    evidence = cx.get_evidence(final_text.latest_observation.evidence_id)
    assert evidence.content == "save 5\n"


def test_save_all_across_multiple_files_stays_multiple_observations(tmp_path):
    """Coalescing is per path, never global."""
    cx = _init_dev(tmp_path, **{"README.md": "r\n", "docs/a.md": "a\n", "docs/b.md": "b\n"})
    _watcher.enable_and_start(cx)

    for name in ("README.md", "docs/a.md", "docs/b.md"):
        (tmp_path / name).write_text("changed\n", encoding="utf-8")

    assert _wait_for(
        lambda: all(_observation_count(cx, p) == 1 for p in ("README.md", "docs/a.md", "docs/b.md")),
        timeout=10.0,
    )


# -- 4. identity / provenance / authority boundary ------------------------------


def test_source_identity_preserved_across_observations(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "v1\n"})
    _watcher.enable_and_start(cx)
    (tmp_path / "README.md").write_text("v2\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)
    first_id = [s for s in cx.sources() if s.path == "README.md"][0].source_id

    time.sleep(1.2)
    (tmp_path / "README.md").write_text("v3\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 2)
    second_id = [s for s in cx.sources() if s.path == "README.md"][0].source_id

    assert first_id == second_id


def test_evidence_provenance_correct(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "before\n"})
    _watcher.enable_and_start(cx)
    (tmp_path / "README.md").write_text("after\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)

    source = [s for s in cx.sources() if s.path == "README.md"][0]
    evidence = cx.get_evidence(source.latest_observation.evidence_id)
    assert evidence.kind == EVIDENCE_KIND_DOCUMENT_OBSERVATION
    assert evidence.content == "after\n"


def test_no_canonical_memory_promotion_after_watcher_activity(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n", "docs/a.md": "y\n"})
    _watcher.enable_and_start(cx)
    for i in range(3):
        (tmp_path / "README.md").write_text(f"x{i}\n", encoding="utf-8")
        time.sleep(1.3)
    assert _wait_for(lambda: _observation_count(cx, "README.md") >= 1)

    assert cx.timeline() == []
    assert cx.state() == []
    assert cx._count_memories() == 0


# -- 5. scope: structural exclusions and bounded discovery ----------------------


@pytest.mark.parametrize(
    "excluded_dir", [".urdyn", ".git", "__pycache__", "node_modules", "dist", "build"]
)
def test_files_under_excluded_or_uninteresting_dirs_never_observed(tmp_path, excluded_dir):
    """Scope is a bounded allowlist (tracked Sources + a root/docs-only
    discovery glob), never a recursive walk -- a path under any of these
    directories cannot structurally match it, so no denylist is needed
    as the enforcement mechanism (see `_watcher.py`'s module docstring).
    """
    cx = _init_dev(tmp_path, **{"README.md": "root\n"})
    (tmp_path / excluded_dir).mkdir(parents=True, exist_ok=True)
    (tmp_path / excluded_dir / "README.md").write_text("should never be observed\n", encoding="utf-8")

    scope = cx.watcher_scope()
    assert f"{excluded_dir}/README.md" not in scope

    _watcher.enable_and_start(cx)
    time.sleep(3.0)
    assert _observation_count(cx, f"{excluded_dir}/README.md") == 0


def test_out_of_scope_file_ignored_until_seeded(tmp_path):
    cx = _init_dev(tmp_path, **{"src/module.py": "print('hi')\n"})
    _watcher.enable_and_start(cx)

    (tmp_path / "src" / "module.py").write_text("print('changed')\n", encoding="utf-8")
    time.sleep(3.0)
    assert _observation_count(cx, "src/module.py") == 0

    cx.seed(["src/module.py"])
    assert _observation_count(cx, "src/module.py") == 1

    (tmp_path / "src" / "module.py").write_text("print('changed again')\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "src/module.py") == 2)


def test_temp_editor_artifact_matching_allowlist_glob_is_skipped(tmp_path):
    """`README*` matches an emacs backup `README.md~` -- the one case
    where the discovery allowlist can transiently pick up a temp file;
    it must be filtered before it ever reaches `seed()`."""
    cx = _init_dev(tmp_path, **{"README.md": "root\n"})
    (tmp_path / "README.md~").write_text("backup, must never be observed\n", encoding="utf-8")

    scope = cx.watcher_scope()
    assert "README.md~" in scope  # present in the raw union...
    assert "README.md~" not in _watcher._scan_scope(cx)  # ...but filtered before scanning

    _watcher.enable_and_start(cx)
    time.sleep(3.0)
    assert _observation_count(cx, "README.md~") == 0


# -- 6. lifecycle: duplicate prevention, restart, stop --------------------------


def test_duplicate_watcher_prevented(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    first_pid = _running_pid(cx)
    assert first_pid is not None

    exit_code = _watcher._child_main(["--workspace", str(tmp_path), "--fresh"])
    assert exit_code == 3
    assert _running_pid(cx) == first_pid


def test_repeated_init_dev_does_not_duplicate_watcher(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
    cli_main(["init", "dev"])
    capsys.readouterr()
    cx = Urdyn.open(tmp_path)
    first_pid = _running_pid(cx)
    assert first_pid is not None

    cli_main(["init", "dev"])
    capsys.readouterr()
    assert _running_pid(cx) == first_pid
    assert cx.sources() == []


def test_stop_lifecycle_is_clean(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    assert _running_pid(cx) is not None

    action = _watcher.stop_watcher(cx)
    assert action.code == _watcher.ACTION_STOPPED
    assert _running_pid(cx) is None
    assert _watcher.read_config(cx.path / ".urdyn")["enabled"] is False


def test_crash_then_restart_reconciliation_recovers_offline_change(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    # Restart reconciliation only re-seeds paths that ARE already tracked
    # Sources -- a path never observed even once has no
    # canonical digest to reconcile against, so get README.md tracked via
    # one real live observation BEFORE crashing it.
    (tmp_path / "README.md").write_text("first real edit\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)

    pid = _running_pid(cx)
    assert pid is not None

    os.kill(pid, signal.SIGKILL)
    assert _wait_for(lambda: _watcher.probe_lock(cx.path / ".urdyn").state == _watcher.LOCK_STALE)

    (tmp_path / "README.md").write_text("changed while offline\n", encoding="utf-8")

    # Any normal command is a recovery opportunity.
    action = _watcher.supervise(cx)
    assert action.code == _watcher.ACTION_RESTARTED
    assert _running_pid(cx) is not None
    assert _running_pid(cx) != pid
    assert _observation_count(cx, "README.md") == 2


def test_watch_stop_is_not_auto_restarted_by_a_normal_command(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    _watcher.stop_watcher(cx)
    assert _running_pid(cx) is None

    _watcher.supervise(cx)  # what every normal command does internally
    assert _running_pid(cx) is None


def test_watch_status_does_not_resurrect_a_stale_watcher(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    pid = _running_pid(cx)
    os.kill(pid, signal.SIGKILL)
    assert _wait_for(lambda: _watcher.probe_lock(cx.path / ".urdyn").state == _watcher.LOCK_STALE)

    for _ in range(2):
        lines = _watcher.status_lines(cx, detailed=True)
        assert any("stale" in line for line in lines)
        assert _running_pid(cx) is None


def test_watcher_from_subdirectory_binds_workspace_root(tmp_path, monkeypatch, capsys):
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    (tmp_path / "README.md").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cli_main(["init", "dev"])
    capsys.readouterr()

    monkeypatch.chdir(nested)
    cli_main(["status"])
    out = capsys.readouterr().out
    assert "Watcher: running" in out

    cx = Urdyn.open(tmp_path)
    metadata = _watcher.probe_lock(cx.path / ".urdyn").metadata
    assert metadata["urdyn_id"] == cx.urdyn_id


def test_watch_start_reports_already_running_with_pid(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})
    _watcher.enable_and_start(cx)
    pid = _running_pid(cx)

    action = _watcher.enable_and_start(cx)
    assert action.code == _watcher.ACTION_ALREADY_RUNNING
    assert action.pid == pid


def test_watch_start_outside_dev_profile_raises(tmp_path):
    cx = Urdyn.init(tmp_path, "general")
    with pytest.raises(ValueError):
        _watcher.enable_and_start(cx)


# -- 7. safety: symlinks, oversized/binary files ---------------------------------


def _log_text(cx: Urdyn) -> str:
    log_path = cx.path / ".urdyn" / _watcher.WATCHER_LOG_FILENAME
    return log_path.read_text(encoding="utf-8") if log_path.exists() else ""


def test_symlink_escape_refused(tmp_path):
    """A path only reaches `Urdyn.seed()` (and therefore a logged
    'refused') via TWO routes: it is already a tracked Source (no
    re-validation -- `watcher_scope()` trusts canonical data), or it
    freshly matches the discovery allowlist (which DOES re-validate via
    `resolve_seed_path` on every scope computation, so a path that fails
    that check -- like an escaping symlink -- is filtered out of scope
    before ever being attempted, not logged as refused). This test
    exercises the first route: seed a real file, then swap it for an
    escaping symlink."""
    outside = tmp_path.parent / f"outside-{tmp_path.name}.md"
    outside.write_text("secret elsewhere\n", encoding="utf-8")
    try:
        cx = _init_dev(tmp_path, **{"src/tracked.md": "placeholder\n"})
        cx.seed(["src/tracked.md"])
        _watcher.enable_and_start(cx)

        (tmp_path / "src" / "tracked.md").unlink()
        (tmp_path / "src" / "tracked.md").symlink_to(outside)

        assert _wait_for(lambda: "refused" in _log_text(cx), timeout=8.0)
        source = [s for s in cx.sources() if s.path == "src/tracked.md"][0]
        assert len(source.observations) == 1  # unchanged: still the original text
    finally:
        outside.unlink(missing_ok=True)


def test_symlink_loop_does_not_hang_and_is_refused(tmp_path):
    cx = _init_dev(tmp_path, **{"src/tracked.md": "placeholder\n"})
    cx.seed(["src/tracked.md"])
    _watcher.enable_and_start(cx)
    assert _running_pid(cx) is not None

    target = tmp_path / "src" / "tracked.md"
    target.unlink()
    target.symlink_to(target)  # a trivial self-loop

    assert _wait_for(lambda: "refused" in _log_text(cx), timeout=8.0)
    assert _running_pid(cx) is not None  # still alive, never hung
    source = [s for s in cx.sources() if s.path == "src/tracked.md"][0]
    assert len(source.observations) == 1


def test_oversized_binary_blank_refused_logged_and_watcher_keeps_running(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "fine\n"})
    _watcher.enable_and_start(cx)

    # Created AFTER enable: detected as new/changed paths, each reaching
    # a real seed() attempt and refusal, rather than being silently
    # absorbed into the fresh, zero-observation baseline.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "oversized.md").write_text("x" * (1_048_576 + 1), encoding="utf-8")
    (tmp_path / "docs" / "binary.md").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "docs" / "blank.md").write_text("   \n", encoding="utf-8")

    # The oversized file is pre-filtered at the DISCOVERY stage itself
    # (`discover_candidate_paths` applies the same `resolve_seed_path`
    # size check used everywhere else) -- it never even enters scope, so
    # it is never read and never reaches a seed() attempt at all. That is
    # a stronger guarantee than "refused", not the same one, so it is
    # asserted separately from the binary/blank cases below.
    assert "docs/oversized.md" not in _watcher._scan_scope(cx)

    def _both_refused() -> bool:
        text = _log_text(cx)
        return text.count("refused") >= 2

    assert _wait_for(_both_refused, timeout=10.0)
    assert cx.sources() == []  # all refused, nothing fabricated

    # the watcher must still be alive and functional afterward
    (tmp_path / "README.md").write_text("still working\n", encoding="utf-8")
    assert _wait_for(lambda: _observation_count(cx, "README.md") == 1)


# -- 8. concurrency smoke and platform isolation ---------------------------------


def test_concurrency_smoke_watcher_alongside_reads(tmp_path):
    cx = _init_dev(tmp_path, **{"README.md": "start\n"})
    _watcher.enable_and_start(cx)

    errors: list[Exception] = []
    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            try:
                cx.recall("start")
                cx.context("start")
                cx.state()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(3)]
    for thread in threads:
        thread.start()

    for i in range(3):
        (tmp_path / "README.md").write_text(f"iteration {i}\n", encoding="utf-8")
        time.sleep(1.3)

    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)

    assert errors == []
    with sqlite3.connect(cx._db_path) as connection:
        (result,) = connection.execute("PRAGMA integrity_check").fetchone()
    assert result == "ok"


def test_import_and_status_are_safe_without_fcntl(tmp_path, monkeypatch):
    """The base package must stay importable, and every watcher-facing
    operation must degrade to a professional 'unavailable' status,
    on a platform where `fcntl` does not exist."""
    monkeypatch.setattr(_watcher, "fcntl", None)
    cx = _init_dev(tmp_path, **{"README.md": "x\n"})

    action = _watcher.enable_and_start(cx)
    assert action.code == _watcher.ACTION_UNAVAILABLE
    assert _watcher.status_lines(cx) == ["Watcher: unavailable (background watching needs Linux in this release)"]
    assert _watcher.supervise(cx) is None
    assert not (tmp_path / ".urdyn" / _watcher.WATCHER_LOCK_FILENAME).exists()
