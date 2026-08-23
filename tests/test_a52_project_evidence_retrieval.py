"""Regression coverage for PROJECT EVIDENCE retrieval in `context()`.

`urdyn seed` records a document as a `Source`/`SourceObservation` and
canonical `Evidence`. PROJECT EVIDENCE is a distinct `context()` candidate
pool sourced from `MemoryStore.list_current_source_evidence()` (the current,
i.e. latest, observation of every seeded Source), so a superseded observation
is never a candidate. Candidates use the same lexical, FTS, and semantic
retrieval channels as the other pools.

The authority invariant remains explicit: `Source != Evidence != Memory`.
Project Evidence is rendered with `authority=evidence.kind` (always
`document_observation`), never `epistemic_state`, so compiled context cannot
present raw document text as verified knowledge. `preflight()` and `recall()`
remain unchanged; this is a `context()`-only pool, like Decision memories.

These tests cover retrieval, exclusion, budget handling, coexistence with
Memory, current-observation filtering, semantic lifecycle, rendering,
backward compatibility, and CLI behavior.
"""

from __future__ import annotations

import pytest

from urdyn import Urdyn
from urdyn._cli import main
from urdyn._context import SECTION_PROJECT_EVIDENCE, compile_context
from test_semantic_real_model import _offline, skip_without_model

real_model = pytest.mark.real_model

# Deliberately high, controlled lexical overlap with `_TASK` (mirroring
# `test_a29_context_compiler.py`'s style): after stopword removal `_TASK`
# has 11 significant tokens, so admission needs a strict majority (6).
_TASK = "What should a Firefox browser adapter respect about canonical authority and dependency direction"

_RELEVANT_DOC = """# Authority Model

Every adapter must respect canonical authority when reading memory state.
Dependency direction flows one way: adapters depend on the core, never
the reverse.
"""

_UNRELATED_DOC = """# Release Notes

This release fixes a typo in the changelog and bumps the version number
for the next PyPI publish.
"""

# A second, later revision of `_RELEVANT_DOC`'s path: deliberately shares
# NO significant vocabulary with `_TASK`, so if a superseded observation
# were ever wrongly reintroduced as current, this test would catch it by
# the pool going empty instead of by any stale text leaking through.
_RELEVANT_DOC_V2 = """# Authority Model

See the onboarding wiki for the full picture; this file is retained for
history only.
"""


def _workspace(tmp_path):
    return Urdyn.init(tmp_path)


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


# ---------------------------------------------------------------------------
# 1 / 4 -- relevant project Evidence retrieved, lexical-only mode works
# ---------------------------------------------------------------------------


def test_relevant_seeded_document_surfaces_as_project_evidence(tmp_path):
    cx = _workspace(tmp_path)
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=100000)

    items = _project_evidence_items(result)
    assert {item.entity_id for item in items} == {seeded.evidence.evidence_id}
    assert items[0].content == _RELEVANT_DOC


# ---------------------------------------------------------------------------
# 2 -- irrelevant Evidence excluded
# ---------------------------------------------------------------------------


def test_irrelevant_seeded_document_excluded(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/release-notes.md", _UNRELATED_DOC)

    result = cx.context(_TASK, budget=100000)

    assert _project_evidence_items(result) == ()
    assert result.is_empty()


def test_relevant_included_irrelevant_excluded_together(tmp_path):
    cx = _workspace(tmp_path)
    relevant = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)
    _seed(cx, tmp_path, "docs/release-notes.md", _UNRELATED_DOC)

    result = cx.context(_TASK, budget=100000)

    items = _project_evidence_items(result)
    assert {item.entity_id for item in items} == {relevant.evidence.evidence_id}


# ---------------------------------------------------------------------------
# 3 -- budget respected
# ---------------------------------------------------------------------------


def test_project_evidence_omitted_when_it_does_not_fit_budget(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=1)

    assert _project_evidence_items(result) == ()
    assert result.omitted > 0
    assert "No compiled items fit within the budget." in result.render()


def test_project_evidence_included_when_budget_allows(tmp_path):
    cx = _workspace(tmp_path)
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=100000)

    assert result.used <= 100000
    items = _project_evidence_items(result)
    assert {item.entity_id for item in items} == {seeded.evidence.evidence_id}


# ---------------------------------------------------------------------------
# 6 -- Memory + Evidence coexist correctly
# ---------------------------------------------------------------------------


def test_memory_and_project_evidence_coexist(tmp_path):
    cx = _workspace(tmp_path)
    invariant = cx.remember(
        "Firefox browser adapters must respect canonical authority for dependency direction", kind="invariant"
    )
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=100000)

    by_heading = {section.heading: {item.entity_id for item in section.items} for section in result.sections}
    assert invariant.memory_id in by_heading["CONSTRAINTS"]
    assert seeded.evidence.evidence_id in by_heading[SECTION_PROJECT_EVIDENCE]


# ---------------------------------------------------------------------------
# 7 / 8 -- Source update behavior / stale observation handling
# ---------------------------------------------------------------------------


def test_source_update_only_current_observation_surfaces(tmp_path):
    cx = _workspace(tmp_path)
    v1 = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)
    assert v1.status == "added"

    result_v1 = cx.context(_TASK, budget=100000)
    assert {item.entity_id for item in _project_evidence_items(result_v1)} == {v1.evidence.evidence_id}

    v2 = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC_V2)
    assert v2.status == "changed"
    assert v2.evidence.evidence_id != v1.evidence.evidence_id

    # The old (v1) observation's Evidence is NOT reintroduced even though
    # its own wording would still satisfy `_TASK` -- only the CURRENT
    # (v2) observation is ever a candidate, and v2's wording does not
    # match `_TASK`, so the pool must now be empty.
    result_v2 = cx.context(_TASK, budget=100000)
    assert _project_evidence_items(result_v2) == ()

    # v1's Evidence still canonically exists (append-only history) --
    # this is "excluded from the current pool", never "deleted".
    assert cx.get_evidence(v1.evidence.evidence_id).content == _RELEVANT_DOC


def test_source_update_never_duplicates_across_two_observations(tmp_path):
    """Even when BOTH observations would independently satisfy `_TASK`,
    at most one PROJECT EVIDENCE item exists per Source -- current-only,
    never a growing history of matches for the same document."""
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)
    v2 = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC + "\n\nSee also the adapter guide.")
    assert v2.status == "changed"

    result = cx.context(_TASK, budget=100000)

    items = _project_evidence_items(result)
    assert len(items) == 1
    assert items[0].entity_id == v2.evidence.evidence_id


# ---------------------------------------------------------------------------
# 9 / 11 / 13 -- zero-Evidence unchanged, Memories stay 0, no promotion to Memory
# ---------------------------------------------------------------------------


def test_no_seeded_documents_behaves_exactly_as_before(tmp_path):
    cx = _workspace(tmp_path)
    invariant = cx.remember(
        "Firefox browser adapters must respect canonical authority for dependency direction", kind="invariant"
    )

    result = cx.context(_TASK, budget=100000)

    assert _project_evidence_items(result) == ()
    assert {section.heading for section in result.sections} == {"CONSTRAINTS"}
    assert {item.entity_id for section in result.sections for item in section.items} == {invariant.memory_id}


def test_seeding_project_docs_alone_records_zero_memories(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)
    _seed(cx, tmp_path, "docs/release-notes.md", _UNRELATED_DOC)

    assert cx.timeline() == []

    result = cx.context(_TASK, budget=100000)
    assert _project_evidence_items(result) != ()
    # Still zero Memories after a context compilation that DID surface
    # the seeded material -- retrieving Evidence must never itself
    # create or promote anything.
    assert cx.timeline() == []


def test_project_evidence_authority_is_never_epistemic_state(tmp_path):
    cx = _workspace(tmp_path)
    seeded = _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=100000)

    (item,) = _project_evidence_items(result)
    assert item.entity_id == seeded.evidence.evidence_id
    assert item.authority == "document_observation"
    assert item.authority not in {"verified", "user_asserted"}
    assert item.kind == "evidence"


# ---------------------------------------------------------------------------
# 12 -- provenance preserved (source path visible, and CLI-reachable id)
# ---------------------------------------------------------------------------


def test_project_evidence_renders_its_source_path(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/authority-model.md", _RELEVANT_DOC)

    result = cx.context(_TASK, budget=100000)

    (item,) = _project_evidence_items(result)
    assert item.source_path == "docs/architecture/authority-model.md"
    rendered = result.render()
    assert "docs/architecture/authority-model.md" in rendered
    assert SECTION_PROJECT_EVIDENCE in rendered


# ---------------------------------------------------------------------------
# 10 -- backward compatibility: `compile_context` without `project_evidence`
# ---------------------------------------------------------------------------


def test_compile_context_without_project_evidence_argument_still_works():
    result = compile_context(
        task=_TASK,
        budget=100000,
        invariants=(),
        invariants_excluded=0,
        pending=(),
        lessons=(),
        decisions=(),
        root_causes=(),
        known_failures=(),
        recommended_validation_candidates=(),
        open_conflicts=[],
        retrieval=None,
    )
    assert result.is_empty()
    assert result.sections == ()


# ---------------------------------------------------------------------------
# End-to-end CLI coverage
# ---------------------------------------------------------------------------


def test_cli_reproduction_seed_then_context_surfaces_the_document(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "dev"]) == 0
    capsys.readouterr()

    (tmp_path / "docs" / "architecture").mkdir(parents=True)
    (tmp_path / "docs" / "architecture" / "authority-model.md").write_text(_RELEVANT_DOC, encoding="utf-8")
    assert main(["seed", "docs/architecture/authority-model.md"]) == 0
    capsys.readouterr()

    assert main(["context", "--budget", "6000", _TASK]) == 0
    out = capsys.readouterr().out
    assert "No compiled context for this task." not in out
    assert "0 of 0 selected" not in out
    assert SECTION_PROJECT_EVIDENCE in out
    assert "docs/architecture/authority-model.md" in out


# ---------------------------------------------------------------------------
# 5 -- semantic mode works (real model only)
# ---------------------------------------------------------------------------

_SEMANTIC_DOC = """# Component Orthogonality

Two components must never share mutable state directly with each other.
All coordination between them happens through explicit, versioned
interfaces, so either side can be swapped for a different implementation
without breaking the other.
"""

# Deliberately near-zero direct token overlap with `_SEMANTIC_DOC`'s own
# wording -- a genuine paraphrase, the same shape as
# `test_a29_context_compiler.py`'s `test_semantic_paraphrase_is_admitted_into_context`.
_SEMANTIC_PARAPHRASE_TASK = (
    "is it safe to replace one module with a totally different one as long as the rest of the "
    "system only ever talks to it through its public interface"
)


@real_model
@skip_without_model
def test_semantic_admits_paraphrased_seeded_document_into_context(tmp_path):
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        cx.semantic_setup()
        seeded = _seed(cx, tmp_path, "docs/architecture/component-orthogonality.md", _SEMANTIC_DOC)

        result = cx.context(_SEMANTIC_PARAPHRASE_TASK, budget=100000)

        selected_ids = {item.entity_id for section in result.sections for item in section.items}
        assert seeded.evidence.evidence_id in selected_ids


@real_model
@skip_without_model
def test_semantic_setup_reports_project_evidence_count(tmp_path):
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        _seed(cx, tmp_path, "docs/architecture/component-orthogonality.md", _SEMANTIC_DOC)

        setup_result = cx.semantic_setup()

        assert setup_result.source_evidence_count == 1
        assert setup_result.memory_count == 0


@real_model
@skip_without_model
def test_cli_semantic_setup_prints_project_evidence(tmp_path, monkeypatch, capsys):
    with _offline():
        monkeypatch.chdir(tmp_path)
        assert main(["init", "dev"]) == 0
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "component-orthogonality.md").write_text(_SEMANTIC_DOC, encoding="utf-8")
        assert main(["seed", "docs/component-orthogonality.md"]) == 0
        capsys.readouterr()

        assert main(["semantic", "setup"]) == 0
        out = capsys.readouterr().out
        assert "1 project evidence" in out
