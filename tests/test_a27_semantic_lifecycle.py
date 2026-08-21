"""A27: the lifecycle of the DERIVED semantic index -- when it is
current, when it is not, and what Urdyn does and says about it.

WHY THIS FILE EXISTS. A26 ran a full dev loop -- record experience in one
session, ask for it in the next -- and concluded that semantic retrieval
did not generalize. A26.1 disproved that: `urdyn semantic setup` had
never been run, `semantic_index.db` did not exist, `_semantic_context()`
returned None, and every semantic pool returned an empty admission set.
That is bit-for-bit what a pool returns when it RAN and correctly
abstained, so `preflight()` produced a plausible, incomplete answer with
nothing anywhere saying the semantic channel had been skipped. Rebuilding
the index offline, over the same canonical data and the same thresholds,
took the same query from 1 useful item to 6.

So the defect was never calibration. It was that derived state had no
lifecycle: nothing knew whether the index still represented canonical
storage, and nothing could say so. These tests pin the contract that
replaces it:

  * freshness is COMPUTED by comparing canonical ids to indexed ids, so
    it survives crashes, races and copied workspaces -- there is no
    stored dirty flag that could be wrong (see `_semantic_store.py`);
  * a stale index is repaired incrementally, offline, at the consumer
    boundary, so the A26 journey works with no manual rebuild;
  * every degraded condition is REPORTED, never inferred from silence.

The fake one-hot backend from `test_semantic.py` is reused throughout:
concept words map to orthogonal axes, so which candidates the semantic
channel admits is exact arithmetic rather than a property of a real
model. The lifecycle under test is model-independent by design, and the
real model is exercised separately in `test_semantic_real_model.py`.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import threading

import pytest

from urdyn import Urdyn
from urdyn._cli import main
from urdyn._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_INDEX_UNREADABLE,
    DETAIL_MODEL_MISMATCH,
    DETAIL_MODEL_UNCACHED,
    DETAIL_NOT_SET_UP,
    DETAIL_REFRESH_FAILED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    SemanticIndexStore,
)
from test_semantic import fake_semantic  # noqa: F401  (pytest fixture)

# The A26 experience, in the shape A26 recorded it: one failed attempt,
# one root cause, one verified lesson, one still-open pending, all about
# the same "alpha" concept, plus a Session-2 task worded so that NO
# lexical or FTS channel can match it -- it shares no significant token
# with anything stored. Only the semantic channel can bridge it, which is
# what makes every assertion below a statement about the semantic
# lifecycle rather than about lexical matching that would have worked
# anyway.
A26_TASK = "wire up a completely differently worded piece of work concerning alpha"
UNRELATED_TASK = "restyle the marketing footer stylesheet and tidy its colour tokens"


def _verified_lesson(cx, content):
    evidence = cx.add_evidence("pytest -q: 14 passed", kind="test_result")
    return cx.learn(content, supporting_evidence=[evidence], verified=True)


def _record_a26_experience(cx):
    """The canonical Session-1 capture, recorded ONLY through the public
    API, exactly as a developer would."""
    evidence = cx.add_evidence("pytest -q: 14 passed, 0 failed", kind="test_result")
    cx.record_attempt(
        task="alpha ingestion keeps dropping records",
        approach="alpha retry loop without idempotency",
        outcome="failed",
        evidence=[evidence],
    )
    cx.remember(
        "alpha writes were not idempotent, so the retry duplicated them",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[evidence],
    )
    cx.learn(
        "alpha retries must be idempotent before any retry loop is introduced",
        supporting_evidence=[evidence],
        verified=True,
    )
    cx.remember("alpha dead-letter handling is still unfinished", kind="pending")


def _vector_snapshot(cx):
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        return {
            entity_type: dict(store.all_vectors(entity_type))
            for entity_type in (ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL)
        }


def _canonical_snapshot(cx):
    """Every canonical row a derived refresh must not touch, read through
    a separate connection so nothing about Urdyn's own store objects can
    make the comparison vacuous."""
    connection = sqlite3.connect(cx._db_path)
    try:
        return {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("memories", "attempts", "skills", "evidence", "events", "memory_evidence")
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# A. never enabled
# ---------------------------------------------------------------------------


def test_never_enabled_workspace_reports_disabled(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content", kind="note")

    state = cx.semantic_state()

    assert state.status == SEMANTIC_DISABLED
    assert state.detail == DETAIL_NOT_SET_UP
    assert not state.is_usable()


def test_disabled_state_never_consults_the_semantic_runtime(tmp_path, monkeypatch):
    """A workspace that never opted in must answer from the filesystem
    alone -- no ONNX import, no model, no cache probe. Proven by making
    the runtime lookup itself explode: if the answer needed it, this
    would raise instead of returning."""
    import urdyn._workspace as workspace

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content", kind="note")
    monkeypatch.setattr(
        workspace, "_load_semantic_module", lambda: pytest.fail("status must not load the semantic runtime")
    )

    assert cx.semantic_state().status == SEMANTIC_DISABLED


def test_never_enabled_preflight_keeps_lexical_value_and_says_it_is_lexical(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "Migrations run before the deployment starts")

    result = cx.preflight("Migrations run before the deployment starts")

    # the lexical channel is untouched: this still works exactly as it did
    assert [m.memory_id for m in result.verified_lessons] == [lesson.memory_id]
    # ...but the caller can now tell that it was the ONLY channel
    assert result.retrieval.status == SEMANTIC_DISABLED
    assert "lexical only" in result.retrieval.retrieval_mode()


def test_cli_status_reports_disabled(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Urdyn.init(tmp_path, "dev")

    assert main(["status"]) == 0

    assert "Semantic: disabled (not set up; run: urdyn semantic setup)" in capsys.readouterr().out


def test_cli_init_announces_that_semantic_is_not_enabled(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["init", "dev"]) == 0

    out = capsys.readouterr().out
    assert "urdyn semantic setup" in out


# ---------------------------------------------------------------------------
# B. setup, and what does / does not make an index stale
# ---------------------------------------------------------------------------


def test_setup_makes_the_index_ready(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _record_a26_experience(cx)

    cx.semantic_setup()

    state = cx.semantic_state()
    assert state.status == SEMANTIC_READY
    assert state.missing == 0
    assert state.indexed == 4  # 1 attempt + 3 memories (root cause, lesson, pending), 0 skills
    assert state.is_usable()


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda cx: cx.remember("beta note", kind="note"), id="remember"),
        pytest.param(lambda cx: _verified_lesson(cx, "beta lesson"), id="learn"),
        pytest.param(
            lambda cx: cx.record_attempt(task="beta task", approach="beta approach", outcome="failed"),
            id="attempt",
        ),
        pytest.param(lambda cx: cx.remember("beta work left", kind="pending"), id="pending"),
        pytest.param(lambda cx: cx.remember("beta rule", kind="invariant"), id="invariant"),
        pytest.param(
            lambda cx: cx.remember("beta withdrawn", kind="invalidation"), id="invalidation"
        ),
    ],
)
def test_semantically_relevant_writes_make_the_index_stale(tmp_path, fake_semantic, write):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()

    write(cx)

    state = cx.semantic_state()
    assert state.status == SEMANTIC_STALE
    assert state.missing == 1


def test_supersession_makes_the_index_stale_through_its_new_memory(tmp_path, fake_semantic):
    """Superseding INSERTS; it never edits the superseded row. That is
    what makes id coverage a sufficient freshness test, so it is asserted
    rather than assumed."""
    cx = Urdyn.init(tmp_path, "dev")
    original = cx.remember("alpha behaviour as first understood", kind="note")
    cx.semantic_setup()

    cx.remember("alpha behaviour, corrected", kind="note", supersedes=original.memory_id)

    assert cx.semantic_state().missing == 1


def test_promoting_a_skill_makes_the_index_stale(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "alpha procedure worth keeping")
    cx.semantic_setup()

    cx.promote(lesson, name="alpha procedure", purpose="handle alpha", steps=["do alpha"])

    assert cx.semantic_state().status == SEMANTIC_STALE


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda cx: cx.add_evidence("some log line", kind="command_output"), id="evidence"),
        pytest.param(lambda cx: cx.record_conflict(cx.state()[0], cx.state()[1]), id="conflict"),
    ],
)
def test_canonical_writes_that_feed_no_semantic_pool_do_not_make_it_stale(tmp_path, fake_semantic, write):
    """Freshness must track the ACTUAL semantic input set. Evidence and
    Conflicts are canonical data that no pool indexes, so writing them
    must not cost a rebuild -- a freshness signal that fired on every
    canonical write would be correct-but-useless, and would put a model
    load in front of workflows that need none."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha first", kind="note")
    cx.remember("beta second", kind="note")
    cx.semantic_setup()

    write(cx)

    assert cx.semantic_state().status == SEMANTIC_READY


def test_a_deduplicated_remember_does_not_make_the_index_stale(tmp_path, fake_semantic):
    """A17: a repeated `remember()` of a current equivalent writes no row
    and no event. It must therefore cost no rebuild either."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content", kind="note")
    cx.semantic_setup()

    cx.remember("alpha content", kind="note")

    assert cx.semantic_state().status == SEMANTIC_READY


def test_freshness_is_decided_without_loading_the_model(tmp_path, fake_semantic, monkeypatch):
    """`urdyn status` runs this on every invocation, so staleness must
    be decidable from stored ids alone."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    cx.remember("beta added later", kind="note")

    monkeypatch.setattr(
        semantic, "load_model_for_index", lambda meta: pytest.fail("freshness must not load a model")
    )
    monkeypatch.setattr(semantic, "embed", lambda *a, **k: pytest.fail("freshness must not embed"))

    assert cx.semantic_state().status == SEMANTIC_STALE


def test_cli_status_reports_stale_with_counts(tmp_path, monkeypatch, capsys, fake_semantic):
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    cx.remember("beta later", kind="note")

    assert main(["status"]) == 0

    assert "Semantic: stale (1 of 2 not indexed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# C. automatic incremental refresh
# ---------------------------------------------------------------------------


def test_stale_preflight_refreshes_only_what_is_missing(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha already indexed", kind="note")
    cx.semantic_setup()
    before = _vector_snapshot(cx)
    cx.remember("beta added after the build", kind="note")

    result = cx.preflight("something about beta entirely reworded")

    assert result.retrieval.status == SEMANTIC_READY
    assert result.retrieval.refreshed == 1  # the new memory only, not the corpus
    after = _vector_snapshot(cx)
    assert after[ENTITY_MEMORY] | before[ENTITY_MEMORY] == after[ENTITY_MEMORY]
    for entity_id, vector in before[ENTITY_MEMORY].items():
        assert after[ENTITY_MEMORY][entity_id] == vector  # untouched, byte for byte
    assert cx.semantic_state().status == SEMANTIC_READY


def test_multiple_pending_writes_are_refreshed_in_one_pass(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)

    result = cx.preflight(A26_TASK)

    assert result.retrieval.refreshed == 4
    assert result.retrieval.status == SEMANTIC_READY


def test_a_current_index_does_no_write_work_and_loads_no_model_to_decide_that(
    tmp_path, fake_semantic, monkeypatch
):
    cx = Urdyn.init(tmp_path, "dev")
    _record_a26_experience(cx)
    cx.semantic_setup()
    before = _vector_snapshot(cx)

    calls = []
    original_add = SemanticIndexStore.add_vectors
    monkeypatch.setattr(
        SemanticIndexStore,
        "add_vectors",
        lambda self, entity_type, rows: calls.append(entity_type) or original_add(self, entity_type, rows),
    )

    result = cx.preflight(A26_TASK)

    assert calls == []  # nothing to refresh means nothing is written
    assert result.retrieval.refreshed == 0
    assert result.retrieval.retrieval_mode() == "semantic + lexical"
    assert _vector_snapshot(cx) == before


def test_guard_gets_the_same_lifecycle(tmp_path, fake_semantic):
    """`guard()` consumes the semantic channel directly, so it degrades
    the same silent way and gets the same treatment -- one helper, not a
    second implementation."""
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "alpha procedure worth keeping")
    cx.semantic_setup()
    cx.promote(lesson, name="alpha procedure", purpose="handle alpha safely", steps=["do alpha"])

    result = cx.guard("run a differently worded operation touching alpha")

    assert result.retrieval.refreshed >= 1
    assert result.retrieval.status == SEMANTIC_READY
    assert [s.name for s in result.applicable_skills] == ["alpha procedure"]


def test_refresh_leaves_canonical_state_byte_identical(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)
    before = _canonical_snapshot(cx)

    cx.preflight(A26_TASK)

    assert _canonical_snapshot(cx) == before


# ---------------------------------------------------------------------------
# D. failure: degraded, never falsely current
# ---------------------------------------------------------------------------


def test_model_missing_offline_is_unavailable_not_stale(tmp_path, fake_semantic, monkeypatch):
    """An index whose model cannot be loaded here is not "missing some
    vectors" -- it cannot be queried at all, and topping it up is
    impossible. Different condition, different remedy, different state."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    cx.remember("beta later", kind="note")
    monkeypatch.setattr(semantic, "artifacts_available", lambda meta: False)

    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_MODEL_UNCACHED
    assert "lexical only" in state.retrieval_mode()

    result = cx.preflight("alpha seed")  # lexically matchable, so still useful
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE
    assert len(result.verified_lessons) + len(result.known_failures) == 0
    assert result.retrieval.detail == DETAIL_MODEL_UNCACHED


def test_a_failed_refresh_reports_degraded_and_never_ready(tmp_path, fake_semantic, monkeypatch):
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    cx.remember("beta added later", kind="note")

    def _explode(*args, **kwargs):
        raise RuntimeError("embedding backend fell over")

    monkeypatch.setattr(semantic, "embed", _explode)

    result = cx.preflight("alpha seed")

    assert result.retrieval.status == SEMANTIC_STALE
    assert result.retrieval.detail == DETAIL_REFRESH_FAILED
    assert result.retrieval.missing == 1
    assert "DEGRADED" in result.retrieval.retrieval_mode()
    # the lexical answer is still delivered in full
    assert len(cx.recall("alpha seed")) == 1


def test_canonical_memory_survives_a_failed_refresh(tmp_path, fake_semantic, monkeypatch):
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)
    before = _canonical_snapshot(cx)
    monkeypatch.setattr(semantic, "embed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    cx.preflight(A26_TASK)

    assert _canonical_snapshot(cx) == before
    assert cx.semantic_state().status == SEMANTIC_STALE


def test_partial_refresh_progress_is_kept_and_still_reads_as_stale(tmp_path, fake_semantic, monkeypatch):
    """The publication boundary, tested where it is testable: a refresh
    that dies between pools has committed real vectors, and must still be
    classified stale afterwards. Coverage is recomputed, so there is no
    moment at which partial progress could be announced as current."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    cx.record_attempt(task="alpha task", approach="alpha approach", outcome="failed")
    cx.remember("alpha memory", kind="note")

    real_embed = semantic.embed
    calls = {"n": 0}

    def _embed_then_die(model, texts):
        calls["n"] += 1
        if calls["n"] > 1:  # the attempt pool goes through; the memory pool does not
            raise RuntimeError("interrupted between pools")
        return real_embed(model, texts)

    monkeypatch.setattr(semantic, "embed", _embed_then_die)

    result = cx.preflight(A26_TASK)

    assert result.retrieval.status == SEMANTIC_STALE
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        assert len(store.indexed_ids(ENTITY_ATTEMPT)) == 1  # progress was kept
        assert store.indexed_ids(ENTITY_MEMORY) == set()
    monkeypatch.undo()
    assert cx.preflight(A26_TASK).retrieval.status == SEMANTIC_READY  # and completes on retry


def test_an_incompatible_index_is_never_topped_up(tmp_path, fake_semantic):
    """A16.2.1 measured that two artifacts of the SAME model produce
    different vectors. Adding this build's vectors to an index built by
    another one would mix vector spaces, so an incompatible index must
    report UNAVAILABLE and be rebuilt explicitly -- never refreshed."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store._connection.execute("UPDATE semantic_meta SET model_id = 'some/other-model#x' WHERE id = 1")
        store._connection.commit()
    cx.remember("beta added later", kind="note")
    before = _vector_snapshot(cx)

    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_MODEL_MISMATCH

    cx.preflight("anything at all about beta")

    assert _vector_snapshot(cx) == before  # not one vector added


def test_an_interrupted_build_reports_incomplete_not_stale(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store._connection.execute("UPDATE semantic_meta SET status = 'building' WHERE id = 1")
        store._connection.commit()

    state = cx.semantic_state()

    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_BUILD_INCOMPLETE


def test_a_corrupted_index_degrades_and_is_not_deleted(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "Migrations run before the deployment starts")
    cx.semantic_setup()
    cx._semantic_db_path.write_bytes(b"this is not a database at all, not even close")

    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_INDEX_UNREADABLE

    result = cx.preflight("Migrations run before the deployment starts")
    assert [m.memory_id for m in result.verified_lessons] == [lesson.memory_id]
    assert cx._semantic_db_path.exists()  # derived and rebuildable, but never deleted for you


def test_setup_repairs_the_unusable_states_it_can_and_says_so_where_it_cannot(tmp_path, fake_semantic):
    """`urdyn semantic setup` is the advertised remedy, so the advert
    has to be true. It is, for every state reachable by ordinary use --
    and it is NOT for a file that is not a database at all, which setup
    cannot open in order to rebuild. Urdyn does not delete that file for
    you (derived and rebuildable makes it SAFE to delete, not Urdyn's
    call to make), so the state carries the one remedy that works."""
    from urdyn._errors import UrdynStorageError

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha seed", kind="note")
    cx.semantic_setup()

    # a model-mismatched index: rebuilt in place, no user intervention
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store._connection.execute("UPDATE semantic_meta SET model_id = 'other/model#x' WHERE id = 1")
        store._connection.commit()
    cx.semantic_setup()
    assert cx.semantic_state().status == SEMANTIC_READY

    # an interrupted build: likewise
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store._connection.execute("UPDATE semantic_meta SET status = 'building' WHERE id = 1")
        store._connection.commit()
    cx.semantic_setup()
    assert cx.semantic_state().status == SEMANTIC_READY

    # a corrupted FILE: setup cannot open it, and the state says exactly
    # what to do rather than pointing at a command that would fail
    cx._semantic_db_path.write_bytes(b"garbage")
    state = cx.semantic_state()
    assert state.detail == DETAIL_INDEX_UNREADABLE
    assert "delete .urdyn/semantic_index.db" in state.describe()
    with pytest.raises(UrdynStorageError):
        cx.semantic_setup()

    cx._semantic_db_path.unlink()
    cx.semantic_setup()
    assert cx.semantic_state().status == SEMANTIC_READY


# ---------------------------------------------------------------------------
# E. concurrency: the authority must survive any interleaving
# ---------------------------------------------------------------------------


def test_two_concurrent_stale_consumers_both_succeed(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)

    results = []
    barrier = threading.Barrier(2)

    def _run():
        worker = Urdyn.discover(tmp_path)
        barrier.wait(timeout=30)
        results.append(worker.preflight(A26_TASK))

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(results) == 2
    for result in results:
        assert result.retrieval.status == SEMANTIC_READY
    assert cx.semantic_state().status == SEMANTIC_READY
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        assert store.vector_count() == 4  # each vector written once, no duplicates


def test_a_canonical_write_during_refresh_leaves_a_recomputable_answer(tmp_path, fake_semantic, monkeypatch):
    """The key concurrency property is NOT "no exceptions": it is that
    freshness can always be recomputed correctly afterwards. A write that
    lands mid-refresh is simply not covered yet, and the next question
    asked says so."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    cx.remember("alpha first", kind="note")

    real_embed = semantic.embed
    fired = {"done": False}

    def _embed_then_write(model, texts):
        if not fired["done"]:
            fired["done"] = True
            Urdyn.discover(tmp_path).remember("beta landed mid-refresh", kind="note")
        return real_embed(model, texts)

    monkeypatch.setattr(semantic, "embed", _embed_then_write)

    cx.preflight(A26_TASK)
    monkeypatch.undo()

    # recomputed from scratch, against canonical state as it now is
    state = cx.semantic_state()
    assert state.status == SEMANTIC_STALE
    assert state.missing == 1
    assert cx.preflight(A26_TASK).retrieval.status == SEMANTIC_READY


def test_explicit_setup_over_a_stale_index_rebuilds_it_completely(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha first", kind="note")
    cx.semantic_setup()
    cx.remember("beta second", kind="note")
    assert cx.semantic_state().status == SEMANTIC_STALE

    cx.semantic_setup()

    assert cx.semantic_state().status == SEMANTIC_READY


# ---------------------------------------------------------------------------
# F. the A26 journey itself
# ---------------------------------------------------------------------------


def test_a26_journey_recovers_experience_with_no_manual_rebuild(tmp_path, fake_semantic):
    """THE regression this whole subtask exists for.

    Session 1 enables semantic retrieval once and records experience.
    Session 2 knows nothing, runs exactly one command -- `preflight` --
    and must be told what Session 1 learned. On the A27 baseline the
    index still described the empty workspace of step 2, every semantic
    pool abstained, and this task (which shares no significant token with
    anything stored) returned nothing at all.
    """
    # --- Session 1 -------------------------------------------------
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()  # one-time, explicit opt-in
    _record_a26_experience(cx)
    assert cx.semantic_state().status == SEMANTIC_STALE  # and nobody rebuilds it

    # --- Session 2: a fresh handle, one command --------------------
    session2 = Urdyn.discover(tmp_path)
    result = session2.preflight(A26_TASK)

    assert result.retrieval.status == SEMANTIC_READY
    assert result.retrieval.refreshed == 4
    assert result.retrieval.retrieval_mode() == "semantic + lexical (refreshed 4)"

    assert [m.content for m in result.verified_lessons] == [
        "alpha retries must be idempotent before any retry loop is introduced"
    ]
    assert [m.content for m in result.pending] == ["alpha dead-letter handling is still unfinished"]
    assert [a.task for a in result.known_failures] == ["alpha ingestion keeps dropping records"]
    assert [m.content for m in result.root_causes] == [
        "alpha writes were not idempotent, so the retry duplicated them"
    ]
    assert [e.content for e in result.recommended_validation] == ["pytest -q: 14 passed, 0 failed"]

    # the derived repair changed nothing canonical
    assert session2.semantic_state().status == SEMANTIC_READY


def test_an_unrelated_task_still_abstains_after_an_automatic_refresh(tmp_path, fake_semantic):
    """Freshness must not become a synonym for permissiveness: the point
    is that relevant experience is reachable, not that everything is."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)

    result = cx.preflight(UNRELATED_TASK)

    assert result.retrieval.status == SEMANTIC_READY  # the substrate WAS current
    assert result.verified_lessons == ()
    assert result.root_causes == ()
    assert result.known_failures == ()
    assert result.pending == ()


def test_cli_a26_journey_reports_semantic_and_lexical(tmp_path, monkeypatch, capsys, fake_semantic):
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    _record_a26_experience(cx)

    assert main(["preflight", A26_TASK]) == 0

    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Retrieval: semantic + lexical (refreshed 4)"
    assert "VERIFIED LESSONS" in out
    assert main(["status"]) == 0
    assert "Semantic: ready" in capsys.readouterr().out


def test_an_empty_result_still_reports_how_it_was_produced(tmp_path, monkeypatch, capsys, fake_semantic):
    """A26's actual output was a plausible near-empty answer. "Nothing
    found" must never again be printable without saying what looked."""
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()

    assert main(["preflight", UNRELATED_TASK]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "Retrieval: semantic + lexical",
        "No relevant experience found.",
    ]


def test_an_empty_result_on_a_never_enabled_workspace_says_lexical_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Urdyn.init(tmp_path, "dev")

    assert main(["preflight", UNRELATED_TASK]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "Retrieval: lexical only -- semantic retrieval is not set up (run: urdyn semantic setup)",
        "No relevant experience found.",
    ]


# ---------------------------------------------------------------------------
# G. what A27 deliberately did NOT change
# ---------------------------------------------------------------------------


def test_no_canonical_schema_migration_was_needed(tmp_path):
    from urdyn._store import STORE_SCHEMA_VERSION

    assert STORE_SCHEMA_VERSION == 7


def test_the_derived_schema_is_unchanged_too(tmp_path, fake_semantic):
    """Freshness is computed from the primary key `semantic_vectors` has
    carried since A7.4 -- no column, no table, no migration, and nothing
    for a future `urdyn rebuild` to have to re-establish."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha", kind="note")
    cx.semantic_setup()

    connection = sqlite3.connect(cx._semantic_db_path)
    try:
        tables = {
            row[0]: row[1]
            for row in connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        connection.close()

    assert set(tables) == {"semantic_meta", "semantic_vectors"}
    assert "dirty" not in tables["semantic_meta"]
    assert "generation" not in tables["semantic_meta"]
    assert "watermark" not in tables["semantic_meta"]


def test_semantic_policy_is_untouched():
    """A26.1 proved the thresholds were never the problem. A27 must not
    have quietly moved one."""
    from urdyn._semantic import LESSON_SEMANTIC_FLOOR, SEMANTIC_POLICY, SET_ADMISSION_LIMIT

    assert SEMANTIC_POLICY[ENTITY_ATTEMPT] == dataclasses.replace(
        SEMANTIC_POLICY[ENTITY_ATTEMPT], absolute_floor=0.50, margin_floor=0.08
    )
    assert (SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor, SEMANTIC_POLICY[ENTITY_MEMORY].margin_floor) == (
        0.20,
        0.08,
    )
    assert (SEMANTIC_POLICY[ENTITY_SKILL].absolute_floor, SEMANTIC_POLICY[ENTITY_SKILL].margin_floor) == (
        0.55,
        0.10,
    )
    assert (LESSON_SEMANTIC_FLOOR, SET_ADMISSION_LIMIT) == (0.30, 2)
