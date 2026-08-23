"""A52.1: PROJECT EVIDENCE candidates that individually exceed a realistic
budget -- the gap A52 left open.

Real-world dogfooding of the A52 checkout inside a pre-existing workspace
(`urdyn-platform`, originally seeded under `urdyn-memory 0.1.0`, no reseed,
no manual migration) reproduced a second-order gap: `urdyn context --budget
6000 "<query genuinely covered by the seeded docs>"` now found 6 relevant
PROJECT EVIDENCE candidates (A52's fix), but the compiled context still read
"No compiled items fit within the budget." -- 0 of 6 selected; 6 omitted for
budget.

Root cause, verified against the unmodified A52 HEAD (`4f33b66`) before any
fix in this module: `compile_context` represents a PROJECT EVIDENCE
candidate as the ENTIRE current-observation `Evidence.content`, verbatim,
one candidate per Source (see `_context.py`'s module docstring, "never
truncated"). A real architecture document a few KB long, easily admitted by
`evidence_is_relevant` (lexical majority over the WHOLE document, or FTS, or
a semantic embedding computed over its own leading ~128 tokens), costs more
CHARACTERS on its own than a 6000-character budget has room for -- and
`compile_context`'s admission is a deterministic PREFIX scan that stops at
the first candidate that does not fit (A29.1's "prefix monotonicity"), so
once every candidate individually exceeds the residual budget, all of them
are counted "omitted for budget" and none is ever tried.

The fix (`_chunk.py`) introduces a purely DERIVED, never-persisted retrieval
representation: `Evidence.content` is deterministically split into
paragraph-aware chunks, recomputed fresh on every `context()` call directly
from canonical `Evidence.content` (so it can never go stale across a Source
update, and needs no "rebuild" step -- there is nothing cached to rebuild).
Chunks of an ALREADY task-relevant document (the existing `evidence_is_relevant`
gate is unchanged) are ranked by their OWN lexical overlap with the query, so
`compile_context` can admit the specific paragraph that matters instead of
being forced to admit-or-reject the entire document as one indivisible unit.
`Source != Evidence != Memory` is unaffected: a chunk still renders with
`authority="document_observation"`, never `epistemic_state`, and nothing
here creates, promotes, or touches a Memory.
"""

from __future__ import annotations

import pytest

from urdyn import Urdyn
from urdyn._chunk import DEFAULT_CHUNK_MAX_CHARS, EvidenceChunk, chunk_evidence, chunk_text_spans, rank_evidence_chunks
from urdyn._context import SECTION_PROJECT_EVIDENCE
from urdyn._evidence import Evidence
from urdyn._relevance import tokens as _tokens

real_model = pytest.mark.real_model

# Mirrors the exact query from the real `urdyn-platform` dogfood session
# (A52.1's session brief), with the same deliberately rich, multi-concept
# vocabulary that made a real seeded architecture document score as
# relevant on its own leading paragraph.
_QUERY = (
    "What architectural constraints should an implementation of a Firefox browser adapter respect "
    "regarding canonical memory authority, dependency direction, component orthogonality, failure "
    "containment, compatibility, and replacement?"
)


def _project_evidence_items(result):
    for section in result.sections:
        if section.heading == SECTION_PROJECT_EVIDENCE:
            return section.items
    return ()


def _relevant_paragraph(topic: str) -> str:
    # Deliberately reuses the query's own significant vocabulary, the same
    # "controlled lexical overlap" style `test_a52_project_evidence_retrieval.py`
    # already relies on -- this is the paragraph a real reader would call
    # "the part that answers the question".
    return (
        f"# {topic}\n\n"
        "Every adapter must respect canonical memory authority: only the core may write belief "
        "state, and dependency direction flows one way -- adapters depend on the core, never the "
        "reverse. Component orthogonality requires that a Firefox browser adapter never shares "
        "mutable state directly with another component. Failure containment means one adapter's "
        "crash must never take down the core or another adapter. Compatibility and replacement "
        "require that a browser adapter implementation stay swappable without changing the core.\n"
    )


def _irrelevant_filler(paragraph_count: int, seed: int) -> str:
    # Shares essentially no significant vocabulary with `_QUERY`: plain
    # release/packaging boilerplate, repeated with a changing index so each
    # paragraph is a distinct, real paragraph boundary rather than one
    # giant duplicated block a chunker could collapse.
    return "\n\n".join(
        f"Release note {seed}.{i}: this changelog entry only bumps the package version, updates "
        "the PyPI metadata, and fixes an unrelated typo in the README footer text."
        for i in range(paragraph_count)
    )


def _long_relevant_doc(topic: str, *, filler_paragraphs: int = 40, seed: int = 0) -> str:
    """A realistic-sized architecture document: one genuinely relevant
    section followed by enough irrelevant filler to push the WHOLE
    document past a 6000-character budget on its own -- the exact shape
    the real `urdyn-platform` dogfood reproduced."""
    doc = _relevant_paragraph(topic) + "\n\n" + _irrelevant_filler(filler_paragraphs, seed)
    assert len(doc) > 6000, "fixture must reproduce the real oversized-candidate shape"
    return doc


def _workspace(tmp_path):
    return Urdyn.init(tmp_path)


def _seed(cx, tmp_path, relative_path: str, content: str):
    full_path = tmp_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    (result,) = cx.seed([relative_path])
    return result


# ---------------------------------------------------------------------------
# Exact reproduction of the reported dogfood symptom: 6 relevant, oversized
# candidates, budget=6000, 0 selected before the fix.
# ---------------------------------------------------------------------------


def test_six_oversized_relevant_documents_still_yield_a_selection_under_6000_budget(tmp_path):
    cx = _workspace(tmp_path)
    for i in range(6):
        _seed(cx, tmp_path, f"docs/architecture/doc-{i}.md", _long_relevant_doc(f"Architecture Note {i}", seed=i))

    result = cx.context(_QUERY, budget=6000)

    assert result.used <= 6000
    items = _project_evidence_items(result)
    assert items, "expected at least one PROJECT EVIDENCE item to survive the 6000-char budget"
    assert "No compiled items fit within the budget." not in result.render()
    assert "-- 0 of" not in result.render()

    # The content actually selected must come from the RELEVANT paragraph,
    # never from the irrelevant filler -- selecting *something* that fits
    # the budget is not the same as selecting something USEFUL.
    for item in items:
        shared = _tokens(_QUERY) & _tokens(item.content)
        assert len(shared) >= 6, f"selected item does not look relevant to the query: {item.content!r}"
        assert "changelog entry" not in item.content
        assert "PyPI metadata" not in item.content


def test_relevant_section_of_long_document_selected_irrelevant_section_not(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _long_relevant_doc("Authority Model", filler_paragraphs=60))

    result = cx.context(_QUERY, budget=6000)

    items = _project_evidence_items(result)
    assert items
    rendered_content = "\n".join(item.content for item in items)
    assert "canonical memory authority" in rendered_content
    assert "changelog entry" not in rendered_content
    assert "PyPI metadata" not in rendered_content


def test_project_evidence_never_exceeds_the_stated_budget(tmp_path):
    cx = _workspace(tmp_path)
    for i in range(6):
        _seed(cx, tmp_path, f"docs/architecture/doc-{i}.md", _long_relevant_doc(f"Architecture Note {i}", seed=i))

    for budget in (500, 1500, 3000, 6000, 12000):
        result = cx.context(_QUERY, budget=budget)
        assert result.used <= budget


# ---------------------------------------------------------------------------
# Small documents are completely unaffected: single-chunk pass-through,
# byte-identical to A52 behavior.
# ---------------------------------------------------------------------------


def test_small_document_still_produces_exactly_one_whole_document_item(tmp_path):
    cx = _workspace(tmp_path)
    small_doc = (
        "# Authority Model\n\n"
        "Every adapter must respect canonical memory authority when reading state. Dependency "
        "direction flows one way: adapters depend on the core, never the reverse. Firefox browser "
        "adapters must respect component orthogonality and failure containment.\n"
    )
    assert len(small_doc) <= DEFAULT_CHUNK_MAX_CHARS
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", small_doc)

    result = cx.context(_QUERY, budget=100000)

    items = _project_evidence_items(result)
    assert len(items) == 1
    assert items[0].content == small_doc
    assert items[0].entity_id == seeded.evidence.evidence_id


# ---------------------------------------------------------------------------
# Memory + oversized Project Evidence coexist under a constrained budget.
# ---------------------------------------------------------------------------


def test_memory_and_oversized_project_evidence_coexist_under_constrained_budget(tmp_path):
    cx = _workspace(tmp_path)
    invariant = cx.remember(
        "Firefox browser adapters must respect canonical memory authority for dependency direction",
        kind="invariant",
    )
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _long_relevant_doc("Authority Model"))

    result = cx.context(_QUERY, budget=6000)

    by_heading = {section.heading: {item.entity_id for item in section.items} for section in result.sections}
    assert invariant.memory_id in by_heading.get("CONSTRAINTS", set())
    assert by_heading.get(SECTION_PROJECT_EVIDENCE, set()), "expected PROJECT EVIDENCE to also survive the budget"
    assert result.used <= 6000


# ---------------------------------------------------------------------------
# Authority invariant preserved for chunked items, and Memories stay at zero.
# ---------------------------------------------------------------------------


def test_chunked_project_evidence_authority_is_never_epistemic_state(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _long_relevant_doc("Authority Model"))

    result = cx.context(_QUERY, budget=6000)

    items = _project_evidence_items(result)
    assert items
    for item in items:
        assert item.authority == "document_observation"
        assert item.authority not in {"verified", "user_asserted"}
        assert item.kind == "evidence"


def test_seeding_large_documents_alone_still_records_zero_memories(tmp_path):
    cx = _workspace(tmp_path)
    for i in range(6):
        _seed(cx, tmp_path, f"docs/architecture/doc-{i}.md", _long_relevant_doc(f"Architecture Note {i}", seed=i))

    assert cx.timeline() == []
    result = cx.context(_QUERY, budget=6000)
    assert _project_evidence_items(result)
    assert cx.timeline() == []


# ---------------------------------------------------------------------------
# Provenance preserved: the source path is still visible per chunk, and the
# entity_id still resolves back to the real canonical Evidence.
# ---------------------------------------------------------------------------


def test_chunked_project_evidence_still_renders_its_source_path(tmp_path):
    cx = _workspace(tmp_path)
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _long_relevant_doc("Authority Model"))

    result = cx.context(_QUERY, budget=6000)

    items = _project_evidence_items(result)
    assert items
    for item in items:
        assert item.source_path == "docs/architecture/authority-model.md"
        assert item.entity_id == seeded.evidence.evidence_id
    rendered = result.render()
    assert "docs/architecture/authority-model.md" in rendered


# ---------------------------------------------------------------------------
# Source update: a stale (v1) chunk representation must never leak into a
# context compiled after the Source moved to v2 -- there is nothing cached
# to go stale, since chunks are recomputed from the CURRENT observation on
# every call, but this asserts the observable behavior directly.
# ---------------------------------------------------------------------------


def test_source_update_to_oversized_document_never_mixes_v1_and_v2_chunks(tmp_path):
    cx = _workspace(tmp_path)
    v1_content = _long_relevant_doc("Authority Model", filler_paragraphs=50, seed=1)
    v1 = _seed(cx, tmp_path, "docs/architecture/authority-model.md", v1_content)

    result_v1 = cx.context(_QUERY, budget=6000)
    items_v1 = _project_evidence_items(result_v1)
    assert items_v1
    assert all(item.entity_id == v1.evidence.evidence_id for item in items_v1)

    # v2 keeps the relevant paragraph (still matches `_QUERY`) but changes
    # every filler paragraph's seed, so a leaked v1 chunk would be
    # detectable by its distinct filler wording.
    v2_content = _long_relevant_doc("Authority Model", filler_paragraphs=50, seed=2)
    v2 = _seed(cx, tmp_path, "docs/architecture/authority-model.md", v2_content)
    assert v2.evidence.evidence_id != v1.evidence.evidence_id

    result_v2 = cx.context(_QUERY, budget=6000)
    items_v2 = _project_evidence_items(result_v2)
    assert items_v2
    assert all(item.entity_id == v2.evidence.evidence_id for item in items_v2)
    for item in items_v2:
        assert "Release note 1." not in item.content

    # v1's Evidence still canonically exists (append-only history).
    assert cx.get_evidence(v1.evidence.evidence_id).content == v1_content


# ---------------------------------------------------------------------------
# Determinism / "rebuild": chunking is a pure function of canonical
# `Evidence.content` -- calling it twice over the same Evidence always
# reconstructs byte-identical chunks, with no cache to go stale.
# ---------------------------------------------------------------------------


def test_chunking_is_deterministic_and_always_rebuildable_from_canonical_content():
    evidence = Evidence(
        evidence_id="a" * 32,
        content=_long_relevant_doc("Authority Model"),
        kind="document_observation",
        recorded_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    first = chunk_evidence(evidence)
    second = chunk_evidence(evidence)
    assert first == second
    assert all(chunk.text == evidence.content[chunk.start : chunk.end] for chunk in first)


# ---------------------------------------------------------------------------
# Unit-level coverage of the chunker itself.
# ---------------------------------------------------------------------------


def test_chunk_text_spans_returns_single_span_for_short_text():
    text = "short document"
    spans = chunk_text_spans(text, max_chars=1200)
    assert spans == [(0, len(text))]


def test_chunk_text_spans_splits_long_text_on_paragraph_boundaries():
    paragraphs = [f"Paragraph number {i} with some real content in it." for i in range(80)]
    text = "\n\n".join(paragraphs)
    spans = chunk_text_spans(text, max_chars=200)
    assert len(spans) > 1
    for start, end in spans:
        assert end - start <= 200
        assert start < end
    # Spans are in order and never overlap.
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= end


def test_chunk_text_spans_hard_splits_a_single_oversized_paragraph():
    text = "word " * 2000  # one giant paragraph, no blank-line boundary at all
    spans = chunk_text_spans(text, max_chars=300)
    assert len(spans) > 1
    for start, end in spans:
        assert end - start <= 300


def test_chunk_evidence_offsets_reconstruct_the_original_text():
    content = _long_relevant_doc("Authority Model")
    evidence = Evidence(
        evidence_id="b" * 32,
        content=content,
        kind="document_observation",
        recorded_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    chunks = chunk_evidence(evidence)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == content[chunk.start : chunk.end]
        assert chunk.evidence_id == evidence.evidence_id
        assert chunk.chunk_count == len(chunks)


def test_rank_evidence_chunks_orders_relevant_chunks_first_and_drops_irrelevant_ones():
    query_tokens = _tokens(_QUERY)
    relevant_text = _relevant_paragraph("Authority Model")
    irrelevant_text = _irrelevant_filler(3, seed=9)
    chunks = (
        EvidenceChunk(evidence_id="c" * 32, chunk_index=0, chunk_count=2, start=0, end=len(irrelevant_text), text=irrelevant_text),
        EvidenceChunk(
            evidence_id="c" * 32,
            chunk_index=1,
            chunk_count=2,
            start=len(irrelevant_text),
            end=len(irrelevant_text) + len(relevant_text),
            text=relevant_text,
        ),
    )
    ranked = rank_evidence_chunks(query_tokens, chunks)
    assert ranked == (chunks[1],)


def test_rank_evidence_chunks_falls_back_to_leading_chunk_when_none_score_lexically():
    # Simulates a document admitted only via FTS/semantic on the whole
    # text, where no individual chunk clears the lexical majority bar on
    # its own -- the pool must still surface something rather than nothing.
    query_tokens = frozenset({"zzz_no_overlap_at_all"})
    chunks = tuple(
        EvidenceChunk(evidence_id="d" * 32, chunk_index=i, chunk_count=3, start=i, end=i + 1, text=f"paragraph {i}")
        for i in range(3)
    )
    ranked = rank_evidence_chunks(query_tokens, chunks)
    assert ranked == (chunks[0],)
