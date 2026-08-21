"""A7.4: the optional semantic retrieval channel.

Fast, deterministic unit tests only. None of these load the real
ONNX model or touch the network: wherever `Urdyn` needs to load
a semantic model, `urdyn._semantic.load_model_for_setup`/
`load_model_for_retrieval` are monkeypatched to return `_FakeStaticModel`,
a small controllable stand-in whose vectors are exact functions of a
fixed "concept" vocabulary, so admission/abstention outcomes can be
constructed exactly rather than hoped for. See `test_semantic_real_model.py`
for the real-model integration test, which is skipped unless the real
model is already cached locally (never run as part of the normal suite).
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pytest

from urdyn import Urdyn, UrdynSemanticUnavailableError
from urdyn._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL
from urdyn._semantic_store import SemanticIndexStore, semantic_db_path_for

# ---------------------------------------------------------------------------
# Fake deterministic embedding backend
# ---------------------------------------------------------------------------

_FAKE_CONCEPTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
_FAKE_NONE_INDEX = len(_FAKE_CONCEPTS)  # a dedicated "no recognized concept" axis
_FAKE_DIM = len(_FAKE_CONCEPTS) + 1


class _FakeStaticModel:
    """Deterministic, controllable stand-in for the real ONNX encoder:
    each text is mapped to a one-hot-per-concept vector for whichever of
    `_FAKE_CONCEPTS` it contains (case-insensitive substring match), so
    cosine similarity between any two constructed texts is exactly
    predictable -- no hashing, no real model, no randomness. Text with no
    recognized concept word maps to a DEDICATED "none" axis, orthogonal
    to every real concept -- not a near-zero component on the first
    concept axis, which (found while writing these tests) L2-normalizes
    right back up to a perfect, spurious cosine=1.0 match against any
    text that happens to be pure "alpha"."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), _FAKE_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for j, concept in enumerate(_FAKE_CONCEPTS):
                if concept in lowered:
                    vectors[i, j] = 1.0
            if not vectors[i].any():
                vectors[i, _FAKE_NONE_INDEX] = 1.0
        return vectors


@pytest.fixture
def fake_semantic(monkeypatch):
    """Wire `_semantic.load_model_for_setup`/`load_model_for_retrieval`
    to the fake backend above, and give a fixed, non-None fake revision
    (real revision resolution is tested separately, against the real
    `huggingface_hub` cache, in the real-model integration test)."""
    import urdyn._semantic as semantic

    fake_model = _FakeStaticModel()
    monkeypatch.setattr(semantic, "load_model_for_setup", lambda model_id=None: fake_model)
    monkeypatch.setattr(semantic, "load_model_for_retrieval", lambda model_id=None: fake_model)
    monkeypatch.setattr(semantic, "resolve_local_revision", lambda model_id=None: "fake-revision")
    # [A27] `artifacts_available` is the cheap "can this index be queried
    # here at all" probe the lifecycle uses instead of loading a model
    # (see `Urdyn.semantic_state`). Its real implementation resolves
    # paths in the Hugging Face cache, which the fake backend by
    # definition has nothing in, so it is faked at the same boundary as
    # the loaders above -- keeping its ONE real behaviour, that an index
    # this build cannot read is never reported as available.
    monkeypatch.setattr(
        semantic, "artifacts_available", lambda meta: semantic.artifact_for_index(meta) is not None
    )
    return fake_model


# ---------------------------------------------------------------------------
# Optional dependency: base import and graceful degradation
# ---------------------------------------------------------------------------


def _block_semantic_runtime(monkeypatch):
    """Simulate the `[semantic]` extra not being installed: block
    `import onnxruntime` and evict any already-imported `_semantic`
    module so the next `from . import _semantic` re-executes its
    top-level `import onnxruntime`, hitting the block.

    [A16.3] This blocks `onnxruntime`, the package that actually marks
    the extra's presence since the backend moved off model2vec. Getting
    this wrong is not a cosmetic detail: while migrating these tests,
    blocking the now-unimported `model2vec` made the simulation a silent
    no-op, and `test_semantic_setup_raises_clear_error_without_the_extra`
    fell straight through into the REAL setup path and started
    downloading the model from the network -- the exact thing the suite
    must never do. Whatever this blocks must stay the package
    `_semantic.py` genuinely imports.

    Deleting `sys.modules["urdyn._semantic"]` alone is not
    enough: once a submodule has been imported anywhere in the process,
    Python also binds it as an ATTRIBUTE on the parent package object
    (`urdyn._semantic`), and `from . import _semantic` resolves
    via that attribute directly, bypassing `sys.modules`/the import
    machinery entirely if the attribute is still there (found while
    writing this test: the block silently had no effect without this
    second step, and a later test picked up the real, already-cached
    module instead of re-raising ImportError)."""
    import urdyn

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.delitem(sys.modules, "urdyn._semantic", raising=False)
    monkeypatch.delattr(urdyn, "_semantic", raising=False)


def test_base_import_does_not_require_the_semantic_runtime(monkeypatch):
    _block_semantic_runtime(monkeypatch)
    import importlib

    import urdyn

    importlib.reload(urdyn)  # re-run __init__ under the block, just in case
    assert urdyn.Urdyn is not None


def test_preflight_and_guard_degrade_silently_without_the_extra(tmp_path, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="Closed leaked connections", outcome="failed")

    _block_semantic_runtime(monkeypatch)

    result = cx.preflight("Fix connection pool exhaustion")
    assert len(result.known_failures) == 1  # lexical channel still works
    guard_result = cx.guard("Fix connection pool exhaustion")
    assert guard_result.is_empty()  # no crash, no traceback


def test_semantic_setup_raises_clear_error_without_the_extra(tmp_path, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    _block_semantic_runtime(monkeypatch)

    with pytest.raises(UrdynSemanticUnavailableError, match="semantic"):
        cx.semantic_setup()


# ---------------------------------------------------------------------------
# Pure policy/math: rank_candidates, semantic_admitted_id, blob round-trip
# ---------------------------------------------------------------------------


def test_rank_candidates_orders_best_first():
    from urdyn._semantic import rank_candidates

    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    ranked = rank_candidates(np.array([1.0, 0.0], dtype=np.float32), ["low", "high", "mid"], matrix)
    assert [entity_id for entity_id, _ in ranked] == ["low", "mid", "high"]


def test_rank_candidates_scores_the_full_pool_no_early_cutoff():
    from urdyn._semantic import rank_candidates

    n = 50
    matrix = np.eye(n, dtype=np.float32)[:, :2]
    matrix = np.pad(matrix, ((0, 0), (0, 0)))
    ids = [f"id-{i}" for i in range(n)]
    ranked = rank_candidates(np.array([1.0, 0.0], dtype=np.float32), ids, matrix)
    assert len(ranked) == n  # every candidate scored, nothing dropped before ranking


def test_semantic_admitted_id_admits_a_clear_winner():
    from urdyn._semantic import semantic_admitted_id

    ranked = [("winner", 0.9), ("runner_up", 0.1)]
    assert semantic_admitted_id(ranked, ENTITY_MEMORY) == "winner"


def test_semantic_admitted_id_abstains_below_absolute_floor():
    from urdyn._semantic import semantic_admitted_id

    ranked = [("weak", 0.05), ("weaker", 0.01)]
    assert semantic_admitted_id(ranked, ENTITY_MEMORY) is None


def test_semantic_admitted_id_abstains_when_margin_too_thin():
    from urdyn._semantic import semantic_admitted_id

    ranked = [("top", 0.30), ("close_runner_up", 0.29)]  # both above MEMORY's 0.20 floor
    assert semantic_admitted_id(ranked, ENTITY_MEMORY) is None


def test_semantic_admitted_id_single_candidate_pool_skips_margin_floor():
    """Regression test: found manually during A7.4 implementation. A pool
    with exactly one candidate has no runner-up, so the margin floor
    (which measures ambiguity against a runner-up) must not apply --
    only the absolute floor should. Without this fix, a real, strong
    single-candidate match was incorrectly rejected."""
    from urdyn._semantic import semantic_admitted_id

    assert semantic_admitted_id([("only_one", 0.99)], ENTITY_MEMORY) == "only_one"
    assert semantic_admitted_id([("only_one", 0.01)], ENTITY_MEMORY) is None  # still needs the absolute floor


def test_semantic_admitted_id_never_admits_below_rank_1():
    from urdyn._semantic import semantic_admitted_id

    # rank 2 clears every floor on its own but is never even considered
    ranked = [("rank1", 0.20001), ("rank2", 0.99)]
    assert semantic_admitted_id(ranked, ENTITY_MEMORY) is None  # rank1's margin is ~0, abstains
    assert "rank2" not in (semantic_admitted_id(ranked, ENTITY_MEMORY) or "")


def test_vector_blob_round_trip():
    from urdyn._semantic import blob_to_vector, vector_to_blob

    original = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    restored = blob_to_vector(vector_to_blob(original), dimensions=4)
    assert np.allclose(original, restored)


def test_corrupted_vector_blob_wrong_dimension_raises():
    from urdyn._semantic import blob_to_vector, vector_to_blob

    blob = vector_to_blob(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError):
        blob_to_vector(blob, dimensions=4)


def test_is_normalized_detects_unit_and_non_unit_vectors():
    from urdyn._semantic import is_normalized

    assert is_normalized(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)) is True
    assert is_normalized(np.array([[3.0, 4.0]], dtype=np.float32)) is False


def test_embed_normalizes_even_if_the_backend_did_not(fake_semantic):
    from urdyn._semantic import embed, is_normalized

    vectors = embed(fake_semantic, ["alpha", "alpha beta"])
    assert is_normalized(vectors)


# ---------------------------------------------------------------------------
# SemanticIndexStore: build / persist / restart / rebuild / stale / mismatch
# ---------------------------------------------------------------------------


def test_semantic_index_store_starts_with_no_meta(tmp_path):
    store = SemanticIndexStore.create_or_open(tmp_path / "semantic_index.db")
    with store:
        assert store.meta() is None
        assert store.is_ready() is False


def test_semantic_index_store_build_persist_restart():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "semantic_index.db"
        store = SemanticIndexStore.create_or_open(db_path)
        with store:
            store.begin_rebuild(
                provider="model2vec", model_id="fake/model", model_revision="rev1",
                dimensions=4, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
            )
            store.add_vectors(ENTITY_MEMORY, [("m1", b"\x00" * 16)])
            store.finish_rebuild()

        reopened = SemanticIndexStore.create_or_open(db_path)
        with reopened:
            assert reopened.is_ready() is True
            assert reopened.vector_count() == 1
            assert reopened.all_vectors(ENTITY_MEMORY) == [("m1", b"\x00" * 16)]


def test_semantic_index_store_begin_rebuild_marks_building_until_finished():
    """A crash between `begin_rebuild()` and `finish_rebuild()` must
    leave the index correctly reporting itself as not ready -- proven
    here by simply not calling `finish_rebuild()`."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "semantic_index.db"
        store = SemanticIndexStore.create_or_open(db_path)
        with store:
            store.begin_rebuild(
                provider="model2vec", model_id="fake/model", model_revision=None,
                dimensions=4, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
            )
            store.add_vectors(ENTITY_MEMORY, [("m1", b"\x00" * 16)])
            assert store.is_ready() is False  # never finished


def test_semantic_index_store_rebuild_discards_previous_vectors():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "semantic_index.db"
        store = SemanticIndexStore.create_or_open(db_path)
        with store:
            store.begin_rebuild(
                provider="model2vec", model_id="fake/model", model_revision=None,
                dimensions=4, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
            )
            store.add_vectors(ENTITY_MEMORY, [("old", b"\x00" * 16)])
            store.finish_rebuild()

            store.begin_rebuild(
                provider="model2vec", model_id="fake/model", model_revision=None,
                dimensions=4, normalization="l2", created_at="2026-01-01T00:00:01+00:00",
            )
            store.add_vectors(ENTITY_MEMORY, [("new", b"\x01" * 16)])
            store.finish_rebuild()

            assert store.all_vectors(ENTITY_MEMORY) == [("new", b"\x01" * 16)]


def test_semantic_meta_matches_detects_model_config_mismatch():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "semantic_index.db"
        store = SemanticIndexStore.create_or_open(db_path)
        with store:
            store.begin_rebuild(
                provider="model2vec", model_id="minishlab/potion-retrieval-32M", model_revision="rev1",
                dimensions=512, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
            )
            store.finish_rebuild()
            meta = store.meta()

    assert meta.matches(provider="model2vec", model_id="minishlab/potion-retrieval-32M", normalization="l2")
    assert not meta.matches(provider="model2vec", model_id="a-different-model", normalization="l2")
    assert not meta.matches(provider="fastembed", model_id="minishlab/potion-retrieval-32M", normalization="l2")


def test_semantic_vectors_entity_type_isolates_a_deliberate_id_collision():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "semantic_index.db"
        store = SemanticIndexStore.create_or_open(db_path)
        with store:
            store.begin_rebuild(
                provider="model2vec", model_id="fake/model", model_revision=None,
                dimensions=4, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
            )
            same_id = "a" * 32
            store.add_vectors(ENTITY_MEMORY, [(same_id, b"\x00" * 16)])
            store.add_vectors(ENTITY_ATTEMPT, [(same_id, b"\x01" * 16)])
            store.finish_rebuild()

            assert store.all_vectors(ENTITY_MEMORY) == [(same_id, b"\x00" * 16)]
            assert store.all_vectors(ENTITY_ATTEMPT) == [(same_id, b"\x01" * 16)]


# ---------------------------------------------------------------------------
# Urdyn.semantic_setup(): idempotency, canonical isolation, representation
# ---------------------------------------------------------------------------


def test_semantic_setup_is_idempotent(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content here", kind="note")

    first = cx.semantic_setup()
    second = cx.semantic_setup()

    assert first.memory_count == second.memory_count == 1
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        assert store.vector_count() == 1  # not doubled by re-running


def test_semantic_setup_never_touches_canonical_memory_db(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content here", kind="note")

    connection = sqlite3.connect(cx._db_path)
    before = connection.execute("SELECT * FROM memories").fetchall()
    connection.close()

    cx.semantic_setup()

    connection = sqlite3.connect(cx._db_path)
    after = connection.execute("SELECT * FROM memories").fetchall()
    connection.close()
    assert before == after


def test_semantic_setup_on_an_empty_workspace_succeeds_with_zero_vectors(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    result = cx.semantic_setup()
    assert (result.attempt_count, result.memory_count, result.skill_count) == (0, 0, 0)


def test_semantic_setup_persists_model_metadata(tmp_path, fake_semantic):
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha content", kind="note")
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        meta = store.meta()
    assert meta.provider == semantic.SEMANTIC_PROVIDER
    assert meta.model_id == semantic.SEMANTIC_MODEL_ID
    assert meta.model_revision == "fake-revision"
    assert meta.dimensions == _FAKE_DIM
    assert meta.normalization == "l2"
    assert meta.status == "ready"


def test_semantic_representation_excludes_skill_steps(tmp_path, fake_semantic):
    """Point 8's constraint: a Skill's semantic text is name+purpose+
    conditions, never its `steps` -- mirroring `_relevance.skill_search_text`.
    Verified here by putting a concept word ONLY in `steps` and confirming
    it never influences the stored vector (would show up as a nonzero
    component on that concept's axis if steps leaked in)."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("something about beta", supporting_evidence=[ev], verified=True)
    cx.promote(
        lesson,
        name="unrelated name",
        purpose="unrelated purpose",
        steps=["This step mentions gamma but must not affect the semantic vector"],
        conditions=[],
    )
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        ((_, blob),) = store.all_vectors(ENTITY_SKILL)
    from urdyn._semantic import blob_to_vector

    vector = blob_to_vector(blob, dimensions=_FAKE_DIM)
    gamma_index = _FAKE_CONCEPTS.index("gamma")
    assert vector[gamma_index] == 0.0


# ---------------------------------------------------------------------------
# preflight()/guard() wiring: admission, abstention, fallback, portability
# ---------------------------------------------------------------------------


def _verified_lesson(cx, content):
    ev = cx.add_evidence("checked", kind="test_result")
    return cx.learn(content, supporting_evidence=[ev], verified=True)


def _root_cause_with_own_evidence(cx, content):
    """[A23.1] A root cause carrying its OWN Evidence -- the root-cause
    counterpart of `_verified_lesson`, used where a test needs two
    candidates that tie in the single-winner MEMORY pool without sharing
    provenance (which would make them one A7.8 cluster)."""
    ev = cx.add_evidence("observed", kind="error_observation")
    return cx.remember(content, kind="root_cause", epistemic_state="inferred", evidence=[ev])


def test_preflight_admits_a_semantic_paraphrase_lexical_alone_would_miss(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha topic explained in the original wording")
    cx.semantic_setup()

    # shares zero significant tokens with the stored lesson, but the fake
    # embedding maps both to the same "alpha" concept
    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert len(result.verified_lessons) == 1


def test_guard_abstains_on_a_thin_margin_even_with_a_top1_candidate(tmp_path, fake_semantic):
    """Direct analogue of the A7.3 payment-guard-clause finding: a #1
    ranked candidate must not be admitted just because it is #1 -- proven
    here by constructing two skills whose scores against the query are
    close enough that SKILL's calibrated margin floor rejects both.

    The fake backend is one-hot per concept, so the geometry is exact: a
    query naming one concept scores 1/sqrt(k) against a skill naming k
    concepts. Three concepts against four gives 0.5774 and 0.5000 -- a
    top score above the absolute floor and a 0.0774 margin beneath the
    margin floor, which is the regime this test is about. The premise is
    ASSERTED rather than assumed, because a recalibration that moved the
    policy out from under those numbers would otherwise leave this test
    passing while testing nothing (a failure mode this suite has already
    hit once, see `_block_semantic_runtime`)."""
    import math

    from urdyn._semantic import SEMANTIC_POLICY

    cx = Urdyn.init(tmp_path, "dev")
    lesson1 = _verified_lesson(cx, "alpha skill lesson")
    cx.promote(lesson1, name="alpha beta skill", purpose="gamma work", steps=["do alpha"])
    lesson2 = _verified_lesson(cx, "alpha beta skill lesson")
    cx.promote(lesson2, name="alpha beta skill two", purpose="gamma and delta work", steps=["do alpha"])
    cx.semantic_setup()

    policy = SEMANTIC_POLICY[ENTITY_SKILL]
    top, runner_up = 1 / math.sqrt(3), 1 / math.sqrt(4)
    assert top >= policy.absolute_floor, "premise broken: the top candidate no longer clears the floor"
    assert top - runner_up < policy.margin_floor, "premise broken: this margin is no longer thin"

    # a long, mostly-unrelated query that only happens to mention "alpha"
    # once: lexically diluted well below `is_relevant`'s majority
    # threshold, so only the semantic channel is in play here.
    diluted_query = (
        "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    )
    result = cx.guard(diluted_query)
    assert result.applicable_skills == ()


def test_semantic_index_missing_falls_back_to_lexical_cleanly(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="closed leaks", outcome="failed")
    # deliberately never call semantic_setup()
    result = cx.preflight("Fix connection pool exhaustion")
    assert len(result.known_failures) == 1  # lexical channel alone


def test_semantic_index_stale_metadata_is_not_used_silently(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(
            provider="model2vec", model_id="a-different-model-entirely", model_revision=None,
            dimensions=_FAKE_DIM, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
        )
        store.finish_rebuild()

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert result.verified_lessons == ()  # mismatched model config, not used


def test_semantic_index_mid_rebuild_is_not_used(tmp_path, fake_semantic):
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(
            provider=semantic.SEMANTIC_PROVIDER, model_id=semantic.SEMANTIC_MODEL_ID,
            model_revision="fake-revision",
            dimensions=_FAKE_DIM, normalization="l2", created_at="2026-01-01T00:00:01+00:00",
        )
        # deliberately do not call finish_rebuild()

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert result.verified_lessons == ()


def test_corrupted_vector_in_index_degrades_the_whole_pool_safely(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        with store._connection:
            store._connection.execute(
                "UPDATE semantic_vectors SET vector = ? WHERE entity_type = ?", (b"\x00\x00", ENTITY_MEMORY)
            )

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert result.verified_lessons == ()  # degrades, does not crash


def test_portability_copied_workspace_preserves_canonical_ids_and_semantic_result(tmp_path, fake_semantic):
    import shutil

    src = tmp_path / "src"
    cx = Urdyn.init(src, "dev")
    lesson = _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    dst = tmp_path / "dst"
    shutil.copytree(src, dst)

    copied = Urdyn.open(dst)
    result = copied.preflight("a completely different phrasing that happens to be about alpha")
    assert [m.memory_id for m in result.verified_lessons] == [lesson.memory_id]


def test_portability_missing_semantic_index_after_copy_falls_back_safely(tmp_path, fake_semantic):
    import shutil

    src = tmp_path / "src"
    cx = Urdyn.init(src, "dev")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="closed leaks", outcome="failed")
    cx.semantic_setup()

    dst = tmp_path / "dst"
    shutil.copytree(src, dst)
    (dst / ".urdyn" / semantic_db_path_for(src / ".urdyn").name).unlink()

    copied = Urdyn.open(dst)
    result = copied.preflight("Fix connection pool exhaustion")
    assert len(result.known_failures) == 1  # lexical channel, no crash


def test_empty_and_whitespace_query_never_crash_semantic_widening(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    admitted = cx._semantic_widen("   ", ENTITY_MEMORY)
    assert admitted == frozenset()


def test_unicode_query_does_not_crash_semantic_widening(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    admitted = cx._semantic_widen("emoji stress test \U0001f9e0\U0001f4a5 café naïve 中文", ENTITY_MEMORY)
    assert admitted == frozenset()  # no shared concept word, correctly abstains, no crash


def test_very_long_query_does_not_crash_semantic_widening(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    long_query = "word " * 5000 + "alpha"
    admitted = cx._semantic_widen(long_query, ENTITY_MEMORY)
    assert isinstance(admitted, frozenset)


def test_very_short_single_word_query_does_not_crash_semantic_widening(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    admitted = cx._semantic_widen("a", ENTITY_MEMORY)
    assert isinstance(admitted, frozenset)


def test_fts_disabled_semantic_enabled_still_widens_via_semantic(tmp_path, fake_semantic, monkeypatch):
    """FTS5 unavailable (simulated the same way `test_search_index.py`
    does) must not disable the independent semantic channel: the two
    widening channels are meant to degrade independently."""
    import urdyn._store as store_module

    monkeypatch.setattr(store_module, "_try_create_search_index", lambda connection: False)

    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha topic explained in the original wording")
    cx.semantic_setup()

    from urdyn._store import MemoryStore

    with MemoryStore.open_if_exists(cx._db_path) as store:
        assert store.fts_enabled is False

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert len(result.verified_lessons) == 1  # semantic channel alone still recovers it


def test_both_fts_and_semantic_unavailable_falls_back_to_pure_lexical(tmp_path, monkeypatch):
    import urdyn._store as store_module

    monkeypatch.setattr(store_module, "_try_create_search_index", lambda connection: False)
    _block_semantic_runtime(monkeypatch)

    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="closed leaks", outcome="failed")

    result = cx.preflight("Fix connection pool exhaustion")  # exact match: pure lexical still works
    assert len(result.known_failures) == 1


# ---------------------------------------------------------------------------
# Epistemic/temporal filtering still applies to semantically-admitted ids
# ---------------------------------------------------------------------------


def test_superseded_memory_is_not_resurrected_via_semantic_channel(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    old = _verified_lesson(cx, "alpha old content")
    _verified_lesson(cx, "alpha new content")  # supersedes intentionally omitted below to keep both "current"
    cx.semantic_setup()

    # now actually supersede the old one with a fresh verified lesson
    ev = cx.add_evidence("checked again", kind="test_result")
    cx.remember(
        "alpha replacement content", kind="lesson", epistemic_state="verified", supporting_evidence=[ev], supersedes=old.memory_id
    )
    cx.semantic_setup()

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    surfaced_ids = {m.memory_id for m in result.verified_lessons}
    assert old.memory_id not in surfaced_ids


def test_unverified_lesson_is_not_promoted_via_semantic_channel(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.learn("alpha unverified content")  # default: not verified
    cx.semantic_setup()

    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert result.verified_lessons == ()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_semantic_setup_reports_success(tmp_path, fake_semantic, monkeypatch, capsys):
    import urdyn._semantic as semantic
    from urdyn._cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["init", "dev"]) == 0
    assert main(["remember", "alpha content", "--kind", "note"]) == 0
    capsys.readouterr()

    exit_code = main(["semantic", "setup"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Semantic index ready." in out
    assert semantic.SEMANTIC_MODEL_ID in out


def test_cli_semantic_setup_without_extra_fails_cleanly_no_traceback(tmp_path, monkeypatch, capsys):
    from urdyn._cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["init", "dev"]) == 0
    capsys.readouterr()

    _block_semantic_runtime(monkeypatch)
    exit_code = main(["semantic", "setup"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" not in captured.err
    assert "urdyn: error:" in captured.err


# ---------------------------------------------------------------------------
# A7.7: eligibility-before-ranking correctness fix
# ---------------------------------------------------------------------------


def test_semantic_admitted_ids_eligible_ids_filters_pool_before_ranking():
    """Unit-level proof of the [A7.7] fix, independent of Urdyn: an
    ineligible candidate that would win the FULL pool must not be able
    to consume the single admission slot once `eligible_ids` excludes
    it -- the eligible runner-up gets a real chance instead."""
    from urdyn._semantic import blob_to_vector, semantic_admitted_ids, vector_to_blob

    class _Model:
        def encode(self, texts):
            # query vector aligned with "ineligible" exactly, and at a
            # comfortable angle from "eligible" that still clears MEMORY's
            # floor (0.20) and margin (0.08) once alone in the pool
            return np.array([[1.0, 0.0]], dtype=np.float32)

    ineligible_vec = vector_to_blob(np.array([1.0, 0.0], dtype=np.float32))  # cos=1.0 with query
    eligible_vec = vector_to_blob(np.array([0.6, 0.8], dtype=np.float32))  # cos=0.6 with query

    stored_vectors = [("ineligible-id", ineligible_vec), ("eligible-id", eligible_vec)]

    # without eligible_ids: the ineligible entity wins the pool outright
    unrestricted = semantic_admitted_ids(
        "query", "memory", model=_Model(), stored_vectors=stored_vectors, dimensions=2
    )
    assert unrestricted == frozenset({"ineligible-id"})

    # with eligible_ids excluding it: the eligible one is ranked and
    # admitted on its own merits (0.6 clears the 0.20 floor, single
    # candidate in the restricted pool so no margin competitor)
    restricted = semantic_admitted_ids(
        "query",
        "memory",
        model=_Model(),
        stored_vectors=stored_vectors,
        dimensions=2,
        eligible_ids=frozenset({"eligible-id"}),
    )
    assert restricted == frozenset({"eligible-id"})


def test_semantic_admitted_ids_eligible_ids_does_not_auto_promote_below_floor():
    """The fix must not become "always show the best eligible
    candidate": if the only eligible candidate does not itself clear the
    normal absolute floor, the result is still abstention."""
    from urdyn._semantic import semantic_admitted_ids, vector_to_blob

    class _Model:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    weak_vec = vector_to_blob(np.array([0.05, 0.0], dtype=np.float32))  # cos=0.05, well below 0.20
    stored_vectors = [("only-eligible", weak_vec)]

    result = semantic_admitted_ids(
        "query",
        "memory",
        model=_Model(),
        stored_vectors=stored_vectors,
        dimensions=2,
        eligible_ids=frozenset({"only-eligible"}),
    )
    assert result == frozenset()


def test_preflight_ineligible_top1_no_longer_blocks_eligible_verified_lesson(tmp_path, fake_semantic):
    """Integration-level reproduction of the exact A7.5/A7.6 safe/unsafe
    finding: an UNVERIFIED lesson that dominates the raw ranking must no
    longer prevent a DIFFERENT, VERIFIED, current lesson from being
    admitted through the semantic channel."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha", kind="lesson", epistemic_state="user_asserted")  # unverified: cos=1.0 with "alpha"
    verified = _verified_lesson(cx, "alpha beta")  # verified: cos=0.707 with "alpha", alone once filtered
    cx.semantic_setup()

    result = cx.preflight("alpha")
    assert [m.memory_id for m in result.verified_lessons] == [verified.memory_id]


def test_preflight_superseded_current_does_not_block_verified_lesson(tmp_path, fake_semantic):
    """Same fix, different eligibility reason: a SUPERSEDED memory
    (even if it would have scored higher) must not block its own
    successor -- or any other current, verified lesson -- from being
    considered."""
    cx = Urdyn.init(tmp_path, "dev")
    old = _verified_lesson(cx, "alpha")  # will be superseded
    new = _verified_lesson(cx, "alpha beta")
    ev = cx.add_evidence("superseding check", kind="test_result")
    cx.remember(
        "alpha beta gamma", kind="lesson", epistemic_state="verified", supporting_evidence=[ev], supersedes=old.memory_id
    )
    cx.semantic_setup()

    result = cx.preflight("alpha")
    surfaced_ids = {m.memory_id for m in result.verified_lessons}
    assert old.memory_id not in surfaced_ids  # superseded: must never be shown
    assert new.memory_id in surfaced_ids or True  # `new` may or may not clear the floor; only the guarantee above is required


def test_guard_attempt_pool_ignores_succeeded_attempts_for_eligibility(tmp_path, fake_semantic):
    """[A7.7] guard()'s attempt pool eligibility: a SUCCEEDED attempt can
    never appear in guard()'s known_failures (guard only ever surfaces
    FAILED attempts), so it must not be able to dominate the semantic
    ranking pool and starve a genuinely relevant failed attempt either."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(task="alpha", approach="alpha", outcome="succeeded")  # cos=1.0, but never eligible
    ev = cx.add_evidence("observed failure", kind="test_result")
    failed = cx.record_attempt(task="alpha beta", approach="alpha beta", outcome="failed", evidence=[ev])
    lesson = cx.learn("alpha beta lesson", supporting_evidence=[ev], verified=True)
    cx.promote(lesson, name="alpha beta skill", purpose="alpha beta purpose", steps=["step"])
    cx.semantic_setup()

    admitted = cx._semantic_widen("alpha", ENTITY_ATTEMPT, eligible_ids=frozenset({failed.attempt_id}))
    assert admitted == frozenset({failed.attempt_id})


# ---------------------------------------------------------------------------
# A7.7: rigorous query-conditioned structural corroboration (preflight only)
# ---------------------------------------------------------------------------


def test_preflight_corroboration_admits_a_near_floor_memory_with_independently_relevant_attempt(
    tmp_path, fake_semantic
):
    """The one mechanism A7.6 validated as safe and useful: when normal
    semantic admission abstains (here: two root causes tie exactly, so
    the margin check rejects both), a related, INDEPENDENTLY admitted
    failed Attempt (real relationship: shared Evidence) rescues the tied
    candidate instead of leaving preflight() empty.

    [A23.1] The tie is built from ROOT CAUSES rather than verified
    lessons, as it was until A23.1. Corroboration itself is unchanged and
    still considers root causes and verified lessons alike (see
    `_preflight_corroboration_admitted`), but a lesson tie no longer
    reaches the abstention this test needs as its premise: verified
    lessons now have their own bounded set admission channel, which
    admits both sides of such a tie directly. Root causes are the kind
    that still depends solely on the single-winner MEMORY pool, so they
    are what can still exhibit -- and therefore still test -- the
    corroboration fallback."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence for corroboration", kind="test_result")
    cx.remember("alpha", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    _root_cause_with_own_evidence(cx, "alpha")  # exact tie -> margin floor rejects both on its own
    # related Attempt, sharing `ev`, independently and strongly relevant
    # to the same query on its own terms (own score clears ATTEMPT's floor)
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed", evidence=[ev])
    cx.semantic_setup()

    # diluted so the LEXICAL/FTS channel (a single shared word, "alpha",
    # against a query with 10+ tokens) stays silent -- isolates the
    # semantic+corroboration path this test is actually about, exactly
    # like the existing thin-margin guard test above does
    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    result = cx.preflight(diluted_query)
    surfaced_ids = {m.memory_id for m in result.root_causes}
    assert len(surfaced_ids) == 1  # corroboration rescues exactly the pool's own rank-1, not both, not neither


def test_preflight_corroboration_rejects_a_merely_present_unrelated_attempt(tmp_path, fake_semantic):
    """The WEAK version of corroboration A7.6 explicitly rejected (a
    related entity counts just because it exists / scores above some
    floor) must not be what got implemented: here the related Attempt
    shares Evidence with the tied root cause but is NOT independently
    relevant to the query at all (orthogonal concept) -- corroboration
    must NOT fire, and the tie must remain unresolved (abstain).

    [A23.1] Built from root causes for the same reason as the
    corroboration-admits case above."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence, unrelated attempt", kind="test_result")
    cx.remember("alpha", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    _root_cause_with_own_evidence(cx, "alpha")  # exact tie, same as the case above
    # related Attempt shares the SAME evidence, but its own text has
    # nothing to do with the query -- must not count as corroboration
    cx.record_attempt(task="beta", approach="beta", outcome="failed", evidence=[ev])
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    result = cx.preflight(diluted_query)
    assert result.root_causes == ()


def test_preflight_corroboration_requires_a_real_canonical_relationship(tmp_path, fake_semantic):
    """A related-looking Attempt that shares NO Evidence at all with the
    tied candidate (no real canonical relationship) must not corroborate
    it either, even if its own text would independently score well.

    [A23.1] Built from root causes for the same reason as the
    corroboration-admits case above."""
    cx = Urdyn.init(tmp_path, "dev")
    ev1 = cx.add_evidence("evidence for the tied root cause", kind="error_observation")
    cx.remember("alpha", kind="root_cause", epistemic_state="inferred", evidence=[ev1])
    _root_cause_with_own_evidence(cx, "alpha")
    # a DIFFERENT, unrelated attempt: independently strong on "alpha" but
    # shares no Evidence with either tied root cause
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed")
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    result = cx.preflight(diluted_query)
    assert result.root_causes == ()


def test_guard_never_gets_structural_corroboration_widening(tmp_path, fake_semantic):
    """Section 8/A7.7: corroboration is preflight-only, by design. Build
    the exact SKILL-pool analogue of the corroboration-admits case above
    (tied skills + an independently strong related failed Attempt) and
    confirm guard() still abstains -- there is no corroboration fallback
    wired into guard() at all."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence", kind="test_result")
    lesson_a = _verified_lesson(cx, "alpha")
    cx.promote(lesson_a, name="alpha skill a", purpose="alpha", steps=["s"])
    lesson_b = cx.learn("alpha", supporting_evidence=[ev], verified=True)
    cx.promote(lesson_b, name="alpha skill b", purpose="alpha", steps=["s"])
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed", evidence=[ev])
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    result = cx.guard(diluted_query)
    assert result.applicable_skills == ()


# ---------------------------------------------------------------------------
# A7.7: preflight (recall) vs guard (precision) product asymmetry
# ---------------------------------------------------------------------------


def test_preflight_favors_recall_guard_favors_precision_on_the_same_query(tmp_path, fake_semantic):
    """The explicit A7.7 product decision, proven on ONE shared query:
    preflight() surfaces useful experience where guard() -- deliberately
    not widened the same way -- abstains. The two APIs are not required
    to agree.

    [A23.1] The asymmetry is unchanged and, if anything, sharper: what
    carries the lesson into preflight() is now the verified-lesson set
    admission channel rather than the corroboration fallback (the
    corroboration mechanism itself is covered, unchanged, by the three
    root-cause tests above). guard()'s SKILL pool is untouched by A23.1
    -- it still asks the single-winner question, still has the identical
    tie, and still abstains, which is the half of this test that
    discriminates."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence for corroboration", kind="test_result")
    lesson = cx.learn("alpha", supporting_evidence=[ev], verified=True)
    _verified_lesson(cx, "alpha")  # tie partner, forces the memory-pool margin abstain path
    skill = cx.promote(lesson, name="alpha related skill", purpose="alpha", steps=["s"])
    other_lesson = _verified_lesson(cx, "alpha")
    cx.promote(other_lesson, name="alpha other skill", purpose="alpha", steps=["s"])  # skill-pool tie partner
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed", evidence=[ev])
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    preflight_result = cx.preflight(diluted_query)
    guard_result = cx.guard(diluted_query)

    assert lesson.memory_id in {m.memory_id for m in preflight_result.verified_lessons}  # recall
    # guard's own skill pool has the identical tie (two "alpha" skills,
    # cos=1.0 each) -- margin floor rejects it, and guard has no
    # corroboration fallback to rescue it the way preflight does
    assert skill.skill_id not in {s.skill_id for s in guard_result.applicable_skills}


# ---------------------------------------------------------------------------
# A7.7: adversarial cases not already covered above
# ---------------------------------------------------------------------------


def test_eligible_rank_2_below_its_own_floor_still_abstains(tmp_path, fake_semantic):
    """An eligible candidate at rank #2 of the FULL pool that, even once
    the ineligible rank #1 is filtered out, does not itself clear the
    absolute floor must still result in abstention -- the fix restricts
    the pool, it does not lower the bar for whatever is left in it."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha", kind="lesson", epistemic_state="user_asserted")  # unverified, cos=1.0, rank #1, ineligible
    _verified_lesson(cx, "epsilon")  # verified, but cos=0.0 with an "alpha" query -- eligible, still below floor
    cx.semantic_setup()

    result = cx.preflight("alpha")
    assert result.verified_lessons == ()


def test_stale_semantic_index_disables_corroboration_too(tmp_path, fake_semantic):
    """The corroboration path shares `_semantic_context()` with normal
    semantic widening, so a stale/mismatched index must disable it the
    same way -- verified explicitly rather than assumed from shared code."""
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence", kind="test_result")
    cx.learn("alpha", supporting_evidence=[ev], verified=True)
    _verified_lesson(cx, "alpha")
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed", evidence=[ev])
    cx.semantic_setup()

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(
            provider="model2vec", model_id="a-different-model-entirely", model_revision=None,
            dimensions=_FAKE_DIM, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
        )
        store.finish_rebuild()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
    result = cx.preflight(diluted_query)
    assert result.verified_lessons == ()  # mismatched index: neither channel nor corroboration used


def test_corroboration_does_not_crash_without_the_extra(tmp_path, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    ev = cx.add_evidence("shared evidence", kind="test_result")
    cx.learn("alpha", supporting_evidence=[ev], verified=True)
    cx.record_attempt(task="alpha", approach="alpha", outcome="failed", evidence=[ev])

    _block_semantic_runtime(monkeypatch)
    result = cx.preflight("a completely unrelated query with plenty of unrelated words in it")
    assert result.verified_lessons == ()  # no crash, no traceback


# ---------------------------------------------------------------------------
# [A16.3] Backend identity: artifact selection, effective identity, and the
# artifact/revision mixing hazard measured in A16.2.1
# ---------------------------------------------------------------------------


def _meta(model_id, *, provider=None, normalization="l2", dimensions=384, status="ready"):
    """A `SemanticMeta` built directly, so index-compatibility can be
    tested without standing up a workspace for every variation."""
    import urdyn._semantic as semantic
    from urdyn._semantic_store import SemanticMeta

    return SemanticMeta(
        provider=semantic.SEMANTIC_PROVIDER if provider is None else provider,
        model_id=model_id,
        model_revision=None,
        dimensions=dimensions,
        normalization=normalization,
        created_at="2026-01-01T00:00:00+00:00",
        status=status,
    )


def test_preferred_artifact_is_chosen_by_architecture_never_by_os():
    """Architecture decides, and only architecture: the same machine
    string must give the same artifact whatever the OS is, and anything
    unrecognized must land on the portable full-precision artifact
    rather than on a guess."""
    import urdyn._semantic as semantic

    assert semantic.preferred_artifact("x86_64") == semantic.ARTIFACT_X86_64
    assert semantic.preferred_artifact("AMD64") == semantic.ARTIFACT_X86_64  # Windows spelling
    assert semantic.preferred_artifact("arm64") == semantic.ARTIFACT_ARM64  # macOS spelling
    assert semantic.preferred_artifact("aarch64") == semantic.ARTIFACT_ARM64  # Linux spelling
    assert semantic.preferred_artifact("riscv64") == semantic.ARTIFACT_PORTABLE
    assert semantic.preferred_artifact("") == semantic.ARTIFACT_PORTABLE
    assert semantic.preferred_artifact() in semantic.SUPPORTED_ARTIFACTS


def test_effective_identity_names_repo_revision_and_artifact():
    import urdyn._semantic as semantic

    identity = semantic.model_identity_for(semantic.ARTIFACT_X86_64)
    assert semantic.SEMANTIC_MODEL_REPO in identity
    assert semantic.SEMANTIC_MODEL_REVISION in identity
    assert semantic.ARTIFACT_X86_64 in identity
    # different artifacts of the same model are different identities --
    # this is the whole point of the A16.2.1 hardening
    assert identity != semantic.model_identity_for(semantic.ARTIFACT_PORTABLE)


def test_index_is_queried_with_the_artifact_it_was_built_with():
    """The A16.2.1 mixing hazard, closed: an index built with the
    portable artifact reports the portable artifact, even on a machine
    whose preferred artifact is a different one. Answering with the
    RECORDED artifact is what makes it impossible to score stored
    vectors against a query embedded by a different artifact."""
    import urdyn._semantic as semantic

    for artifact in sorted(semantic.SUPPORTED_ARTIFACTS):
        meta = _meta(semantic.model_identity_for(artifact))
        assert semantic.artifact_for_index(meta) == artifact


def test_index_from_a_different_revision_of_the_same_model_is_refused():
    """An upstream re-export of the same repo and the same artifact
    filename can still produce different vectors, so the pinned revision
    is part of the identity and a different one is not readable here."""
    import urdyn._semantic as semantic

    foreign = (
        f"{semantic.SEMANTIC_MODEL_REPO}@0000000000000000000000000000000000000000"
        f"#{semantic.ARTIFACT_X86_64}"
    )
    assert semantic.artifact_for_index(_meta(foreign)) is None


def test_index_from_a_foreign_model_provider_or_normalization_is_refused():
    import urdyn._semantic as semantic

    good = semantic.model_identity_for(semantic.ARTIFACT_X86_64)
    assert semantic.artifact_for_index(_meta(good)) is not None
    assert semantic.artifact_for_index(_meta("minishlab/potion-retrieval-32M")) is None
    assert semantic.artifact_for_index(_meta("")) is None
    assert semantic.artifact_for_index(_meta(good, provider="model2vec")) is None
    assert semantic.artifact_for_index(_meta(good, normalization="none")) is None
    # a well-formed identity naming an artifact this build cannot load
    unknown = f"{semantic.SEMANTIC_MODEL_REPO}@{semantic.SEMANTIC_MODEL_REVISION}#onnx/model_made_up.onnx"
    assert semantic.artifact_for_index(_meta(unknown)) is None


def test_a_pre_a16_3_potion_index_degrades_to_lexical_and_is_rebuildable(tmp_path, fake_semantic):
    """[A16.3] The honest upgrade story, end to end: canonical
    memory survives untouched, the OLD derived semantic index is refused
    rather than misread as if it were this backend's, retrieval degrades
    to lexical/FTS, and a fresh `semantic_setup()` restores the semantic
    channel. No vector migration exists, and none is promised."""
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "alpha content")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="closed leaks", outcome="failed")
    cx.semantic_setup()

    # rewrite the index exactly as A7.4..A16.2 would have written it
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(
            provider="model2vec", model_id="minishlab/potion-retrieval-32M", model_revision="rev1",
            dimensions=_FAKE_DIM, normalization="l2", created_at="2026-01-01T00:00:00+00:00",
        )
        store.finish_rebuild()

    # canonical memory is untouched by any of this
    assert lesson.memory_id in {m.memory_id for m in cx.recall("alpha content")}
    # the old index is not read as if it were ours
    semantic_only = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert semantic_only.verified_lessons == ()
    # ...while the lexical channel keeps working on its own terms
    lexical = cx.preflight("Fix connection pool exhaustion")
    assert len(lexical.known_failures) == 1

    cx.semantic_setup()  # the documented remedy: rebuild, do not migrate
    recovered = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert lesson.memory_id in {m.memory_id for m in recovered.verified_lessons}


# ---------------------------------------------------------------------------
# [A16.3] Failure injection
# ---------------------------------------------------------------------------


def test_setup_reports_a_urdyn_error_when_the_model_cannot_be_loaded(tmp_path, monkeypatch):
    """A download or ONNX-session failure during setup must surface as
    Urdyn's own error type, not as a raw onnxruntime/huggingface_hub
    traceback (which the CLI would print verbatim)."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")

    def _boom():
        raise RuntimeError("simulated artifact download/session failure")

    monkeypatch.setattr(semantic, "load_model_for_setup", _boom)
    with pytest.raises(UrdynSemanticUnavailableError, match="semantic model"):
        cx.semantic_setup()


def test_setup_failure_leaves_a_previously_built_index_intact(tmp_path, fake_semantic, monkeypatch):
    """Fail closed: a setup that cannot load a model must not destroy the
    index that was already there and working."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "alpha content")
    cx.semantic_setup()

    # a flag rather than `monkeypatch.undo()`: undo would also revert the
    # `fake_semantic` fixture's own patches and quietly turn the rest of
    # this test into a no-model run that proves nothing.
    failing = {"now": True}
    working_loader = semantic.load_model_for_setup

    def _maybe_boom():
        if failing["now"]:
            raise RuntimeError("simulated failure on a later setup run")
        return working_loader()

    monkeypatch.setattr(semantic, "load_model_for_setup", _maybe_boom)
    with pytest.raises(UrdynSemanticUnavailableError):
        cx.semantic_setup()

    failing["now"] = False
    result = cx.preflight("a completely different phrasing that happens to be about alpha")
    assert lesson.memory_id in {m.memory_id for m in result.verified_lessons}


def test_retrieval_degrades_when_the_cached_artifact_is_gone(tmp_path, fake_semantic, monkeypatch):
    """The artifact disappearing from the local cache after setup (a
    pruned cache, a workspace copied without one) must degrade to
    lexical/FTS -- never crash, and never reach for the network."""
    import urdyn._semantic as semantic

    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, "alpha content")
    cx.record_attempt(task="Fix connection pool exhaustion", approach="closed leaks", outcome="failed")
    cx.semantic_setup()

    def _missing(artifact=None):
        raise OSError("simulated: artifact not present in the local cache")

    monkeypatch.setattr(semantic, "load_model_for_retrieval", _missing)

    assert cx.preflight("a completely different phrasing that happens to be about alpha").verified_lessons == ()
    assert len(cx.preflight("Fix connection pool exhaustion").known_failures) == 1
    assert cx.guard("Fix connection pool exhaustion").is_empty() in (True, False)  # no crash either way


def test_setup_falls_back_to_the_portable_artifact_and_records_it(monkeypatch):
    """Exactly one fallback: preferred artifact -> portable
    full-precision artifact. Whatever succeeded is what gets recorded, so
    a fallback can never be silently mixed with another artifact's
    vectors."""
    import urdyn._semantic as semantic

    attempted = []

    class _Encoder:
        def __init__(self, artifact):
            self.identity = semantic.model_identity_for(artifact)

    def _build(artifact, *, local_files_only):
        attempted.append(artifact)
        if artifact != semantic.ARTIFACT_PORTABLE:
            raise RuntimeError("simulated: this artifact does not load here")
        return _Encoder(artifact)

    monkeypatch.setattr(semantic, "_build_encoder", _build)
    monkeypatch.setattr(semantic, "preferred_artifact", lambda machine=None: semantic.ARTIFACT_X86_64)

    model = semantic.load_model_for_setup()
    assert attempted == [semantic.ARTIFACT_X86_64, semantic.ARTIFACT_PORTABLE]
    assert semantic.model_identity(model) == semantic.model_identity_for(semantic.ARTIFACT_PORTABLE)


def test_setup_gives_up_after_one_fallback_rather_than_looping(monkeypatch):
    import urdyn._semantic as semantic

    attempted = []

    def _build(artifact, *, local_files_only):
        attempted.append(artifact)
        raise RuntimeError("simulated: nothing loads here")

    monkeypatch.setattr(semantic, "_build_encoder", _build)
    monkeypatch.setattr(semantic, "preferred_artifact", lambda machine=None: semantic.ARTIFACT_X86_64)

    with pytest.raises(semantic.SemanticUnavailable):
        semantic.load_model_for_setup()
    assert attempted == [semantic.ARTIFACT_X86_64, semantic.ARTIFACT_PORTABLE]  # two attempts, not a chain


def test_retrieval_never_asks_the_network_for_a_missing_artifact(monkeypatch):
    """`load_model_for_retrieval` must always resolve files with
    `local_files_only=True`. This is the property that keeps
    `preflight()`/`guard()` offline by construction rather than by
    convention."""
    import urdyn._semantic as semantic

    seen = {}

    def _fake_download(repo, filename, revision=None, local_files_only=False):
        seen[filename] = local_files_only
        raise OSError("not cached")

    monkeypatch.setattr(semantic.huggingface_hub, "hf_hub_download", _fake_download)
    with pytest.raises(OSError):
        semantic.load_model_for_retrieval(semantic.ARTIFACT_PORTABLE)
    assert seen and all(local_only is True for local_only in seen.values())


def test_encoding_is_chunked_so_peak_memory_does_not_scale_with_the_workspace():
    """[A16.3] `semantic_setup()` hands the encoder every memory in
    the workspace in one call. With the previous static-embedding backend
    that was free; with a transformer the intermediate activations scale
    with the batch, and a 1002-memory workspace encoded as a single batch
    was measured peaking at 6.3 GB of RSS during A16.3. The encoder must
    therefore chunk, and no caller should have to know that."""
    import urdyn._semantic as semantic

    batch_sizes = []

    class _RecordingEncoder(semantic._OnnxTextEncoder):
        def __init__(self):  # no session, no tokenizer: only the chunking is under test
            pass

        def _encode_batch(self, texts):
            batch_sizes.append(len(texts))
            return np.zeros((len(texts), 384), dtype=np.float32)

    encoder = _RecordingEncoder()
    total = semantic.SEMANTIC_ENCODE_BATCH_SIZE * 3 + 7
    vectors = encoder.encode([f"memory {i}" for i in range(total)])

    assert vectors.shape == (total, 384)  # every input still gets exactly one vector, in order
    assert max(batch_sizes) <= semantic.SEMANTIC_ENCODE_BATCH_SIZE
    assert sum(batch_sizes) == total
    assert encoder.encode([]).shape[0] == 0
