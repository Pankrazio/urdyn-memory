"""Permanent compatibility with Brains written by the released Urdyn
0.1.0, 0.2.0, 0.3.0 and 0.4.0 (schema v7, no `brain_identity`: legacy-unbound).

The fixtures are AUTHENTIC: written by the code of each release tag
(tests/data/released_brains, SHA-256 checked before use). For every
release this build must: open the Brain; return exactly the memories, history
and supersession the release recorded; keep it legacy-unbound (an
ordinary open never binds, converts or rewrites it); report its FTS index
usable; and change NOT ONE BYTE while reading -- also under six concurrent
reader processes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import sqlite3

import pytest

import legacy_brains as legacy
from urdyn import Urdyn
from urdyn._cli import main

_PROC_COUNT = 6


def _events(db):
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT kind, subject_id FROM events ORDER BY sequence").fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_released_brain_opens_with_its_memories_history_and_supersession(tmp_path, release):
    root, generated = legacy.release_brain(tmp_path, release)
    cx = Urdyn.open(root)
    cx.verify_storage()

    timeline = cx.timeline()
    state = cx.state()
    assert [m.memory_id for m in state] == generated["state"]
    assert sorted(m.memory_id for m in timeline) == sorted(generated["written"])
    superseding = [m for m in timeline if m.supersedes is not None]
    assert len(superseding) == 1
    assert superseding[0].supersedes in generated["written"]
    assert superseding[0].supersedes not in {m.memory_id for m in state}
    # Every memory the release recorded has its `memory_recorded` event, once.
    recorded = [subject for kind, subject in _events(legacy.db_path(root)) if kind == "memory_recorded"]
    assert sorted(recorded) == sorted(generated["written"])
    assert [m.content for m in cx.recall("pytest")] == ["Synthetic fact beta: tests run with pytest."]


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_released_brain_stays_legacy_unbound(tmp_path, release):
    root, _ = legacy.release_brain(tmp_path, release)
    manifest = json.loads((root / ".urdyn" / "manifest.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{32}", manifest["cortex_id"])
    assert "brain_identity" not in legacy.store_facts(legacy.db_path(root))["tables"]

    cx = Urdyn.open(root)
    assert cx.urdyn_id == manifest["cortex_id"]
    cx.state()
    cx.search_index_state()
    assert "brain_identity" not in legacy.store_facts(legacy.db_path(root))["tables"]


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_released_brain_index_is_usable_for_this_build(tmp_path, release):
    root, generated = legacy.release_brain(tmp_path, release)
    state = Urdyn.open(root).search_index_state()
    assert state.status == "usable", state.detail
    counts = {entity_type: (expected, indexed) for entity_type, expected, indexed in state.entities}
    assert counts["memory"] == (len(generated["written"]), len(generated["written"]))


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_reading_a_released_brain_changes_no_byte(tmp_path, release, monkeypatch, capsys):
    root, _ = legacy.release_brain(tmp_path, release)
    before = legacy.snapshot(root)

    cx = Urdyn.open(root)
    cx.verify_storage()
    cx.state()
    cx.timeline()
    cx.recall("build")
    cx.conflicts()
    cx.skills()
    cx.preflight("build the project with ninja")
    cx.guard("run the build")
    cx.context("build the project").render()
    cx.search_index_state()
    Urdyn.discover(root)
    monkeypatch.chdir(root)
    assert main(["status"]) == 0
    assert main(["search-index", "status"]) == 0
    capsys.readouterr()

    assert legacy.snapshot(root) == before


def _reader(workspace, barrier, queue):
    try:
        barrier.wait(timeout=60)
        from urdyn import Urdyn as _Urdyn

        cx = _Urdyn.open(workspace)
        queue.put(("ok", (len(cx.state()), len(cx.timeline()), len(cx.recall("synthetic")))))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_concurrent_readers_of_a_released_brain_change_no_byte(tmp_path, release):
    root, generated = legacy.release_brain(tmp_path, release)
    before = legacy.snapshot(root)
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(_PROC_COUNT), ctx.Queue()
    processes = [ctx.Process(target=_reader, args=(str(root), barrier, queue)) for _ in range(_PROC_COUNT)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    expected = (len(generated["state"]), len(generated["written"]), len(generated["state"]))
    assert results == [("ok", expected)] * _PROC_COUNT
    assert legacy.snapshot(root) == before
