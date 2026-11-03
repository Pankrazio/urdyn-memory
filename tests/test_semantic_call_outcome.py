"""A query must not hide a failure of the semantic
channel behind a healthy-looking retrieval line.

`semantic_state()` describes the INDEX (built, compatible, covering canonical
state) and keeps its meaning. What a `Preflight`/`CompiledContext`/
`GuardResult` carries as `retrieval` describes what THIS call could actually
use: if a semantic read fails while the index was declared usable, the call
is answered by the lexical/FTS channels alone and says so, as the existing
`unavailable` state already means ("lexical only").

Fake encoders only, temporary workspaces only.
"""

from __future__ import annotations

import inspect
import shutil
import sqlite3
import threading

import numpy as np
import pytest

import urdyn._semantic_store as semantic_store_module
from urdyn import Urdyn
from urdyn._semantic_store import SEMANTIC_READY, SEMANTIC_UNAVAILABLE

from test_semantic import fake_semantic  # noqa: F401  (pytest fixture)

semantic = pytest.importorskip("urdyn._semantic")

VECTORS_INVALID = getattr(semantic_store_module, "DETAIL_QUERY_VECTORS_INVALID", "<constant missing>")
QUERY_FAILED = getattr(semantic_store_module, "DETAIL_QUERY_FAILED", "<constant missing>")
NAN6 = np.full(6, np.nan, np.float32).tobytes()


def _workspace(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    # "alphabet" shares the fake model's "alpha" concept but no lexical token
    # with "alpha", so this root cause is reachable ONLY semantically.
    cx.remember("alpha root cause", kind="root_cause")
    cx.record_attempt(task="beta deploy", approach="beta script", outcome="failed")
    cx.semantic_setup()
    return cx


def _corrupt(cx, entity_type):
    with sqlite3.connect(cx._semantic_db_path) as con:
        con.execute("UPDATE semantic_vectors SET vector = ? WHERE entity_type = ?", (NAN6, entity_type))


def _shape(preflight):
    return ([m.memory_id for m in preflight.root_causes], [m.memory_id for m in preflight.verified_lessons],
            [a.attempt_id for a in preflight.known_failures], [m.memory_id for m in preflight.pending])


def _canonical(cx):
    return [(m.memory_id, m.content, m.kind, m.supersedes) for m in cx.timeline()]


def _lexical_reference(cx, task):
    """The same canonical store answered with no semantic index at all."""
    moved = cx._semantic_db_path.with_suffix(".aside")
    shutil.move(cx._semantic_db_path, moved)
    try:
        return cx.preflight(task)
    finally:
        shutil.move(moved, cx._semantic_db_path)


def test_healthy_index_still_contributes_semantically(tmp_path, fake_semantic):
    cx = _workspace(tmp_path)
    result = cx.preflight("alphabet soup")
    assert result.retrieval.status == SEMANTIC_READY and result.retrieval.detail is None
    assert [m.content for m in result.root_causes] == ["alpha root cause"]  # semantic-only admission
    assert result.retrieval.retrieval_mode() == "semantic + lexical"


def test_corrupted_vector_is_reported_and_the_answer_is_exactly_lexical(tmp_path, fake_semantic):
    cx = _workspace(tmp_path)
    before = _canonical(cx)
    _corrupt(cx, "memory")
    index_bytes = cx._semantic_db_path.read_bytes()

    result = cx.preflight("alphabet soup")

    assert result.retrieval.status == SEMANTIC_UNAVAILABLE
    assert result.retrieval.detail == VECTORS_INVALID
    assert result.retrieval.retrieval_mode().startswith("lexical only")
    assert _shape(result) == _shape(_lexical_reference(cx, "alphabet soup"))  # no semantic leftovers
    # Index-level state keeps its meaning: built, compatible, covering.
    assert cx.semantic_state().status == SEMANTIC_READY
    # Nothing repaired, nothing canonical touched.
    assert cx._semantic_db_path.read_bytes() == index_bytes
    assert _canonical(cx) == before


def test_one_failing_pool_makes_the_whole_call_lexical_only(tmp_path, fake_semantic):
    cx = _workspace(tmp_path)
    _corrupt(cx, "attempt")  # the memory pool is intact and would admit "alpha root cause"
    result = cx.preflight("alphabet soup")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == VECTORS_INVALID
    assert _shape(result) == _shape(_lexical_reference(cx, "alphabet soup"))
    assert result.root_causes == ()  # the healthy pool's semantic admission is not mixed in


def test_context_and_guard_report_the_same_failure(tmp_path, fake_semantic):
    cx = _workspace(tmp_path)
    _corrupt(cx, "memory")
    _corrupt(cx, "skill")
    compiled = cx.context("alphabet soup")
    assert compiled.retrieval.status == SEMANTIC_UNAVAILABLE and compiled.retrieval.detail == VECTORS_INVALID
    assert "lexical only" in compiled.render()
    _corrupt(cx, "attempt")
    guarded = cx.guard("beta deploy")
    assert guarded.retrieval.status == SEMANTIC_UNAVAILABLE and guarded.retrieval.detail == VECTORS_INVALID


def test_channel_vanishing_after_preparation_is_reported(tmp_path, fake_semantic, monkeypatch):
    cx = _workspace(tmp_path)
    monkeypatch.setattr(Urdyn, "_semantic_context", lambda self: None)  # e.g. a rebuild began mid-call
    result = cx.preflight("alphabet soup")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == QUERY_FAILED


def test_unexpected_semantic_exception_is_reported_not_swallowed(tmp_path, fake_semantic, monkeypatch):
    cx = _workspace(tmp_path)
    monkeypatch.setattr(semantic, "semantic_admitted_ids", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = cx.preflight("alphabet soup")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == QUERY_FAILED


def test_lexical_pass_does_not_refresh_or_embed_again(tmp_path, fake_semantic, monkeypatch):
    cx = _workspace(tmp_path)
    cx.remember("gamma note")  # index becomes stale -> the first pass refreshes once
    _corrupt(cx, "memory")
    refreshes = []
    original = Urdyn._semantic_refresh
    monkeypatch.setattr(Urdyn, "_semantic_refresh", lambda self, pools: refreshes.append(1) or original(self, pools))
    result = cx.preflight("alphabet soup")
    assert len(refreshes) == 1
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE


def test_project_evidence_trace_is_not_duplicated_by_the_lexical_pass(tmp_path, fake_semantic):
    cx = _workspace(tmp_path)
    (tmp_path / "notes.md").write_text("alpha deployment notes for the beta script\n")
    cx.seed(["notes.md"])
    cx.semantic_setup()
    healthy_trace: list = []
    cx.context("beta script", _project_evidence_trace=healthy_trace)
    _corrupt(cx, "memory")
    trace: list = []
    compiled = cx.context("beta script", _project_evidence_trace=trace)
    assert compiled.retrieval.status == SEMANTIC_UNAVAILABLE
    keys = [(t.source_path, t.chunk_index) for t in trace]
    assert len(keys) == len(set(keys))


def test_concurrent_calls_report_their_own_outcome(tmp_path, fake_semantic):
    healthy = _workspace(tmp_path / "healthy")
    broken = _workspace(tmp_path / "broken")
    _corrupt(broken, "memory")
    seen = {}
    barrier = threading.Barrier(2)

    def run(name, cx):
        barrier.wait()
        for _ in range(20):
            seen.setdefault(name, set()).add(cx.preflight("alphabet soup").retrieval.status)

    threads = [threading.Thread(target=run, args=("healthy", healthy)), threading.Thread(target=run, args=("broken", broken))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"healthy": {SEMANTIC_READY}, "broken": {SEMANTIC_UNAVAILABLE}}


def test_public_signatures_are_unchanged():
    assert list(inspect.signature(Urdyn.preflight).parameters) == ["self", "task"]
    assert list(inspect.signature(Urdyn.guard).parameters) == ["self", "action"]
    assert list(inspect.signature(Urdyn.context).parameters) == ["self", "task", "budget", "_project_evidence_trace"]
