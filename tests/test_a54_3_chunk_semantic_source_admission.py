"""Regression coverage for Strategy B' (A54.3): chunk-level semantic
representation with Source-level admission.

A54.2/A54.2.1 measured, on a real dogfooded multi-document workspace, that
ONE embedding per whole document (truncated at the model's own 128-token limit)
cannot represent a passage buried past that window, and that reranking an
already-admitted candidate set (Strategy D/E) has a demonstrated recall
ceiling -- the single best-matching chunk in the whole corpus, for a real
paraphrased query, belonged to a document the pipeline did not admit at all.

The chosen fix (`Urdyn._context_evidence_semantic_admitted`,
`_workspace.ENTITY_SOURCE_CHUNK`, `_chunk.chunk_semantic_id`): every derived
chunk of a seeded Source's current Evidence gets its OWN embedding, but
admission stays at SOURCE granularity -- a Source's semantic score is the
MAXIMUM among its own current chunks' scores, ranked and capped exactly as
the old whole-document score was (`EVIDENCE_SEMANTIC_FLOOR`/
`EVIDENCE_ADMISSION_LIMIT`, both UNCHANGED, not recalibrated this session).
This is what prevents same-source flooding by construction: at most
`EVIDENCE_ADMISSION_LIMIT` distinct SOURCES are ever admitted, never a flood
of one long document's own chunks.

Chunks remain exactly what A54's `_chunk.py` module docstring already
requires: derived, non-canonical, rebuildable, carrying no authority of
their own, always reducible back to their parent Evidence's own canonical
id. Nothing here persists a chunk as Evidence, and nothing promotes Evidence
to Memory.
"""

from __future__ import annotations

import pytest

from urdyn import Urdyn
from urdyn._chunk import chunk_semantic_id, parse_chunk_semantic_id
from urdyn._context import SECTION_PROJECT_EVIDENCE
from urdyn._preflight import ProjectEvidenceTrace
from test_semantic_real_model import _offline, skip_without_model

real_model = pytest.mark.real_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workspace(tmp_path):
    return Urdyn.init(tmp_path, "dev")


def _seed(cx, tmp_path, relative_path: str, content: str):
    full_path = tmp_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    (result,) = cx.seed([relative_path])
    return result


def _project_evidence_items(result):
    for section in result.sections:
        if section.heading == SECTION_PROJECT_EVIDENCE:
            return section.items
    return ()


def _paragraphs(*parts: str) -> str:
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 1/8 -- deterministic derived chunk identity: never a row id, never a
# path, always a plain function of (evidence_id, chunk_index).
# ---------------------------------------------------------------------------


def test_chunk_semantic_id_round_trips():
    assert parse_chunk_semantic_id(chunk_semantic_id("abc123", 7)) == ("abc123", 7)


def test_chunk_semantic_id_is_deterministic():
    assert chunk_semantic_id("abc123", 3) == chunk_semantic_id("abc123", 3)


def test_chunk_semantic_id_distinguishes_chunks_of_the_same_evidence():
    assert chunk_semantic_id("abc123", 0) != chunk_semantic_id("abc123", 1)


def test_chunk_semantic_id_distinguishes_different_evidence():
    assert chunk_semantic_id("abc123", 0) != chunk_semantic_id("xyz789", 0)


# ---------------------------------------------------------------------------
# Fixtures shared by the real-model tests below. `_LONG_DOC` buries its one
# relevant paragraph among several unrelated ones; `_WEAK_COMPETITOR` is
# only loosely on-topic (not deliberately crafted to sound maximally
# on-topic the way A54.2's adversarial controlled fixture was) -- a
# realistic shape given the real corpus's own competitor documents scored
# 0.44-0.56, not the 0.75 an adversarial vacuous document can reach.
# ---------------------------------------------------------------------------

_QUERY = "What sustained throughput number did the benchmark record for the sharded pipeline?"

_LONG_DOC = _paragraphs(
    "# Platform Notes",
    "The onboarding process for new engineers takes about two weeks and covers repository "
    "layout, code review norms, and the standard local dev setup used across every team.",
    "Authentication for internal tools goes through the shared SSO provider, which issues "
    "short-lived tokens refreshed automatically by the client library during an active session.",
    "Logging conventions require a structured JSON line per request, tagged with a trace id "
    "that downstream services propagate unchanged through the entire call chain.",
    "The sharded pipeline benchmark recorded a sustained throughput of 18,400 requests per "
    "second across six hours of continuous load, with every worker's reading logged for audit.",
    "Deployment happens through the standard rolling-update controller, which drains "
    "connections from one instance at a time before replacing it under normal load.",
    "Code review requires at least one approval from a different team for any change "
    "touching the shared platform libraries, and two for the authentication subsystem.",
    "Incident response follows a standard severity ladder, and every incident above "
    "severity two gets a written postmortem within a week of resolution.",
    "The internal wiki is the source of truth for on-call rotations and the current list "
    "of services each team owns across the platform's several dozen deployed apps.",
)

_WEAK_COMPETITOR = _paragraphs(
    "# Team Directory",
    "The platform team owns roughly a dozen internal services, including several pipeline "
    "components used for general data processing across the organization's other teams.",
)


@real_model
@skip_without_model
def test_long_document_with_buried_relevant_chunk_outranks_a_weaker_competitor(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        long_doc = _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        _seed(cx, tmp_path, "docs/team-directory.md", _WEAK_COMPETITOR)
        cx.semantic_setup()

        trace: list[ProjectEvidenceTrace] = []
        result = cx.context(_QUERY, budget=100000, _project_evidence_trace=trace)

        selected_ids = {item.entity_id for item in _project_evidence_items(result)}
        assert long_doc.evidence.evidence_id in selected_ids, (
            "the document holding the actually relevant, buried paragraph must be admitted"
        )
        rendered = "\n".join(item.content for item in _project_evidence_items(result))
        assert "18,400 requests per second" in rendered


@real_model
@skip_without_model
def test_document_level_embedding_of_the_same_long_document_would_have_lost(tmp_path):
    """Confirms the fixture above is actually decisive, not accidentally
    easy: the OLD whole-document score (still directly computable by
    embedding `evidence.content` in full) is measurably lower than the
    NEW max-over-chunks score, which is the entire reason chunk-level
    admission is needed here."""
    with _offline():
        cx = _workspace(tmp_path)
        long_doc = _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        cx.semantic_setup()
        context = cx._semantic_context()
        assert context is not None
        semantic, model, meta = context

        whole_doc_vec = semantic.embed(model, [long_doc.evidence.content])[0]
        query_vec = semantic.embed(model, [_QUERY])[0]
        whole_doc_score = float(whole_doc_vec @ query_vec)

        trace: list[ProjectEvidenceTrace] = []
        cx.context(_QUERY, budget=100000, _project_evidence_trace=trace)
        max_chunk_score = max(t.semantic_score for t in trace if t.source_path == "docs/platform-notes.md")

        assert max_chunk_score > whole_doc_score + 0.05, (
            f"chunk-level max ({max_chunk_score:.4f}) should clearly exceed the whole-document "
            f"score ({whole_doc_score:.4f}) for this fixture to demonstrate anything"
        )


# ---------------------------------------------------------------------------
# One strong chunk is enough -- and many mediocre chunks do not out-vote it.
# `_MANY_MEDIOCRE` has ten paragraphs, each loosely on-topic (mentions
# "pipeline" and "benchmark" in passing) but none stating an actual figure;
# `_ONE_STRONG` has a single paragraph that states the figure directly.
# If aggregation were SUM or AVERAGE-with-count-bonus instead of MAX, the
# ten-paragraph document could plausibly out-score the one-paragraph one on
# raw accumulated similarity; under MAX it cannot, because only each
# document's own single best chunk is ever compared.
# ---------------------------------------------------------------------------

_MANY_MEDIOCRE = _paragraphs(
    "# Pipeline Notes",
    *(
        f"Pipeline benchmark round {i} ran without incident and results were filed in the "
        "usual shared tracking sheet for later reference by whoever needs them next."
        for i in range(1, 11)
    ),
)

_ONE_STRONG = _paragraphs(
    "# Findings",
    "The benchmark recorded a sustained throughput of 18,400 requests per second for the "
    "sharded pipeline.",
)


@real_model
@skip_without_model
def test_one_strong_chunk_beats_ten_mediocre_chunks(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        _seed(cx, tmp_path, "docs/many-mediocre.md", _MANY_MEDIOCRE)
        _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)
        cx.semantic_setup()

        trace: list[ProjectEvidenceTrace] = []
        cx.context(_QUERY, budget=100000, _project_evidence_trace=trace)

        scores = {}
        for t in trace:
            scores.setdefault(t.source_path, t.semantic_score)
        assert scores["docs/one-strong.md"] > scores.get("docs/many-mediocre.md", 0.0), (
            f"one precise chunk must outscore ten mediocre ones under MAX aggregation: {scores}"
        )


# ---------------------------------------------------------------------------
# No same-source flooding at admission: a single long, broadly relevant
# document must not occupy every admission slot on its own -- admission
# aggregates to ONE score per Source before the cap is ever applied.
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_admission_never_selects_more_than_the_cap_distinct_sources(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        _seed(cx, tmp_path, "docs/team-directory.md", _WEAK_COMPETITOR)
        _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)
        cx.semantic_setup()

        # Observable contract, through the public retrieval path rather
        # than a private helper: at most `EVIDENCE_ADMISSION_LIMIT`
        # distinct Sources ever carry the 'semantic' channel in the trace.
        trace: list[ProjectEvidenceTrace] = []
        cx.context(_QUERY, budget=100000, _project_evidence_trace=trace)
        semantically_admitted_sources = {t.source_path for t in trace if "semantic" in t.channels}
        from urdyn import _semantic

        assert len(semantically_admitted_sources) <= _semantic.EVIDENCE_ADMISSION_LIMIT, (
            f"semantic admission must never exceed its own cap across distinct Sources, "
            f"got {semantically_admitted_sources}"
        )


# ---------------------------------------------------------------------------
# Provenance: a chunk's admission never changes what a rendered PROJECT
# EVIDENCE item claims to be. Source != Evidence != Memory holds regardless
# of chunk-level semantic scoring.
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_authority_invariant_holds_under_chunk_level_admission(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        seeded = _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        cx.semantic_setup()

        result = cx.context(_QUERY, budget=100000)

        items = _project_evidence_items(result)
        assert items
        for item in items:
            assert item.kind == "evidence"
            assert item.authority == "document_observation"
            if item.entity_id == seeded.evidence.evidence_id:
                assert item.source_path == "docs/platform-notes.md"


def test_no_evidence_promoted_to_memory_by_chunk_admission(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)

    # Even without semantic setup at all (lexical-only), and even after a
    # context() call that reads PROJECT EVIDENCE, no Memory is ever created
    # as a side effect -- promotion is exclusively `Urdyn.promote()`, an
    # explicit, unrelated call this test never makes.
    cx.context(_QUERY, budget=100000)

    assert cx.recall(_QUERY) == []


# ---------------------------------------------------------------------------
# Current observation only: a superseded Source's chunks must never
# contaminate admission, even though the (append-only) semantic index still
# holds their vectors.
# ---------------------------------------------------------------------------

_ORIGINAL_CONTENT = _paragraphs(
    "# Findings v1",
    "The benchmark recorded a sustained throughput of 18,400 requests per second for the "
    "sharded pipeline.",
)

_UPDATED_CONTENT = _paragraphs(
    "# Findings v2",
    "The office plant-watering rotation has moved to a shared calendar, and whoever is "
    "listed for a given week is responsible for every plant on that floor of the building.",
)


@real_model
@skip_without_model
def test_source_update_excludes_the_stale_observations_chunks_from_admission(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        seeded = _seed(cx, tmp_path, "docs/findings.md", _ORIGINAL_CONTENT)
        cx.semantic_setup()

        first_result = cx.context(_QUERY, budget=100000)
        assert seeded.evidence.evidence_id in {
            item.entity_id for item in _project_evidence_items(first_result)
        }

        full_path = tmp_path / "docs/findings.md"
        full_path.write_text(_UPDATED_CONTENT, encoding="utf-8")
        (updated,) = cx.seed(["docs/findings.md"])
        assert updated.evidence.evidence_id != seeded.evidence.evidence_id

        second_result = cx.context(_QUERY, budget=100000)
        second_ids = {item.entity_id for item in _project_evidence_items(second_result)}
        assert seeded.evidence.evidence_id not in second_ids, (
            "the SUPERSEDED observation's evidence id must never be admitted again"
        )
        assert updated.evidence.evidence_id not in second_ids, (
            "the CURRENT observation no longer discusses the topic, so it should not be "
            "admitted either -- this asserts the new content, not the old, decides the outcome"
        )


@real_model
@skip_without_model
def test_source_update_produces_fresh_chunk_ids_for_the_new_observation(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        seeded = _seed(cx, tmp_path, "docs/findings.md", _ORIGINAL_CONTENT)
        cx.semantic_setup()

        full_path = tmp_path / "docs/findings.md"
        full_path.write_text(_UPDATED_CONTENT, encoding="utf-8")
        (updated,) = cx.seed(["docs/findings.md"])
        cx.context("anything", budget=100000)  # triggers incremental refresh

        from urdyn._chunk import chunk_evidence
        from urdyn._semantic_store import SemanticIndexStore
        from urdyn._workspace import ENTITY_SOURCE_CHUNK

        semantic_db = tmp_path / ".urdyn" / "semantic_index.db"
        with SemanticIndexStore.open_if_exists(semantic_db) as index:
            indexed = index.indexed_ids(ENTITY_SOURCE_CHUNK)

        old_chunk_ids = {
            chunk_semantic_id(seeded.evidence.evidence_id, c.chunk_index)
            for c in chunk_evidence(seeded.evidence)
        }
        new_chunk_ids = {
            chunk_semantic_id(updated.evidence.evidence_id, c.chunk_index)
            for c in chunk_evidence(updated.evidence)
        }
        # Append-only: the old observation's chunk vectors are NOT deleted...
        assert old_chunk_ids <= indexed
        # ...but the new observation gets its own, freshly embedded ids.
        assert new_chunk_ids <= indexed


# ---------------------------------------------------------------------------
# Incremental refresh: an unchanged document's chunks are never re-embedded.
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_incremental_refresh_does_not_touch_unchanged_chunks(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)
        first_setup = cx.semantic_setup()
        assert first_setup.source_evidence_count == 2

        # Nothing changed: the next context() call's freshness check must
        # find zero missing ids and therefore refresh nothing.
        state_before = cx.semantic_state()
        assert state_before.status == "ready"
        cx.context(_QUERY, budget=100000)
        state_after = cx.semantic_state()
        assert state_after.status == "ready"
        assert state_after.refreshed == 0


@real_model
@skip_without_model
def test_first_query_after_source_update_pays_the_incremental_embed_cost_the_next_does_not(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        seeded = _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        cx.semantic_setup()

        full_path = tmp_path / "docs/platform-notes.md"
        full_path.write_text(_LONG_DOC + "\nOne more paragraph added to force a new observation.\n", encoding="utf-8")
        (updated,) = cx.seed(["docs/platform-notes.md"])
        assert updated.evidence.evidence_id != seeded.evidence.evidence_id

        # First preflight/context after the update: the index is STALE
        # for the new observation's chunks and must refresh them.
        first_state = cx.semantic_state()
        assert first_state.status == "stale"
        assert first_state.missing > 0

        cx.context(_QUERY, budget=100000)  # pays the refresh cost

        # Second call, nothing changed since: must be READY with nothing
        # left to refresh.
        second_state = cx.semantic_state()
        assert second_state.status == "ready"


# ---------------------------------------------------------------------------
# Rebuild reproducibility: a full `semantic_setup()` rebuild produces the
# same chunk ids (not necessarily identical FLOATS, since ONNX inference is
# not bit-exact across runs on all hardware, but the same coverage) both
# times.
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_full_rebuild_is_deterministic_in_chunk_coverage(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)
        _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)

        from urdyn._semantic_store import SemanticIndexStore
        from urdyn._workspace import ENTITY_SOURCE_CHUNK

        semantic_db = tmp_path / ".urdyn" / "semantic_index.db"

        cx.semantic_setup()
        with SemanticIndexStore.open_if_exists(semantic_db) as index:
            first_ids = index.indexed_ids(ENTITY_SOURCE_CHUNK)

        cx.semantic_setup()  # idempotent full rebuild
        with SemanticIndexStore.open_if_exists(semantic_db) as index:
            second_ids = index.indexed_ids(ENTITY_SOURCE_CHUNK)

        assert first_ids == second_ids
        assert len(first_ids) > 2  # more than one chunk per document, confirms real chunking


# ---------------------------------------------------------------------------
# Lexical-only path stays fully unchanged: no semantic setup call at all.
# ---------------------------------------------------------------------------


def test_lexical_only_path_unchanged_without_semantic_setup(tmp_path):
    cx = _workspace(tmp_path)
    seeded = _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)

    result = cx.context(_QUERY, budget=100000)

    selected_ids = {item.entity_id for item in _project_evidence_items(result)}
    assert seeded.evidence.evidence_id in selected_ids
    assert result.retrieval is not None
    assert result.retrieval.retrieval_mode().startswith("lexical only")


# ---------------------------------------------------------------------------
# Semantic setup count: `source_evidence_count` reports DOCUMENTS, not
# chunks -- a public-API field whose meaning must survive this change.
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_setup_result_reports_document_count_not_chunk_count(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        _seed(cx, tmp_path, "docs/platform-notes.md", _LONG_DOC)  # 9 chunks
        _seed(cx, tmp_path, "docs/one-strong.md", _ONE_STRONG)  # 1 chunk

        result = cx.semantic_setup()

        assert result.source_evidence_count == 2, (
            f"must count 2 documents, not their combined ~10 chunks; got {result.source_evidence_count}"
        )
