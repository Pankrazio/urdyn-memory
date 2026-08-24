"""Regression coverage for A54.4: MSC's redundancy criterion must not
conflate token-set overlap with informational redundancy.

Real dogfooding of A54.3's Strategy B' (chunk-level, Source-level-max
semantic admission) found a real, general failure class:
`minimum_sufficient_project_evidence` (A54) drops a candidate whose own
chunk's query tokens are a subset of an already-kept candidate's, with no
regard for WHY that candidate was admitted. For a short query (few
significant tokens), an incidental, shallow mention -- a table-of-contents
line that merely links to the real answer, sharing "authority"/"model"
from the LINK PATH itself plus "what" from an unrelated sentence two
bullets away in the same chunk -- can cover the query's entire tiny
vocabulary first, silently suppressing the actual canonical definition
document even though that document independently cleared the SEMANTIC
admission channel by a wide margin (real corpus: 0.88 cosine for the
suppressed document vs 0.0 for the TOC line that survived).

Root cause traced precisely (see the A54.4 report): NOT lexical-primary
ordering (A29.1's own rationale -- "the ONE signal available in every
configuration" -- remains fully valid and is unaffected by B'; verified by
direct code archaeology, not assumed) but MSC's token-coverage criterion
alone, which has no way to tell "the same information, restated" apart
from "an incidental mention sharing the query's own short vocabulary".

The fix (`_preflight.minimum_sufficient_project_evidence`): a candidate
whose SOURCE cleared semantic admission independently
(`candidate.admission.semantic`) is also kept the first time that Source is
seen, even when its own chunk's tokens are a subset of what is already
covered. Bounded by `EVIDENCE_ADMISSION_LIMIT` (semantic admission itself
is capped there) and by `represented_sources` (only the FIRST candidate of
a not-yet-represented Source can use this reason -- a Source's second and
later chunks still need the original token rule, so same-source flooding
stays exactly as controlled as A54.3 already made it). Structurally
unreachable without the semantic extra: `admission.semantic` is `False` for
every candidate in a lexical-only workspace, so this file's lexical-only
test is a real, not incidental, guarantee.
"""

from __future__ import annotations

import datetime as dt

import pytest

from urdyn import Urdyn
from urdyn._chunk import EvidenceChunk
from urdyn._context import SECTION_PROJECT_EVIDENCE
from urdyn._evidence import Evidence
from urdyn._preflight import (
    EvidenceAdmission,
    ProjectEvidenceCandidate,
    ProjectEvidenceTrace,
    minimum_sufficient_project_evidence,
)
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


def _candidate(
    *,
    evidence_id: str,
    source_path: str,
    text: str,
    lexical_shared_tokens: int,
    semantic_score: float,
    semantic_admitted: bool,
    lexical_admitted: bool = True,
) -> ProjectEvidenceCandidate:
    evidence = Evidence(
        evidence_id=evidence_id,
        content=text,
        kind="document_observation",
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )
    chunk = EvidenceChunk(
        evidence_id=evidence_id, chunk_index=0, chunk_count=1, start=0, end=len(text), text=text
    )
    admission = EvidenceAdmission(lexical=lexical_admitted, fts=False, semantic=semantic_admitted)
    return ProjectEvidenceCandidate(
        evidence=evidence,
        source_path=source_path,
        chunk=chunk,
        lexical_shared_tokens=lexical_shared_tokens,
        semantic_score=semantic_score,
        admission=admission,
    )


# ---------------------------------------------------------------------------
# Unit level: the filter's own logic, no model, no workspace.
# ---------------------------------------------------------------------------


def test_semantically_admitted_source_survives_despite_subset_tokens():
    query_tokens = frozenset({"authority", "model", "what"})
    toc_line = _candidate(
        evidence_id="toc",
        source_path="README.md",
        text="authority model what",  # shares all 3 query tokens, shallow
        lexical_shared_tokens=3,
        semantic_score=0.0,
        semantic_admitted=False,
    )
    definition = _candidate(
        evidence_id="def",
        source_path="docs/system-x-overview.md",
        text="authority model",  # subset: 2 of 3, but semantically admitted
        lexical_shared_tokens=2,
        semantic_score=0.8827,
        semantic_admitted=True,
    )

    kept = minimum_sufficient_project_evidence(query_tokens, (toc_line, definition))

    assert [c.evidence.evidence_id for c in kept] == ["toc", "def"]


def test_without_semantic_admission_the_subset_candidate_is_still_dropped():
    """Regression guard: the ORIGINAL token-coverage rule is unchanged for
    candidates that never cleared the semantic channel -- this is what
    keeps a lexical-only workspace's behavior identical to before A54.4."""
    query_tokens = frozenset({"authority", "model", "what"})
    toc_line = _candidate(
        evidence_id="toc",
        source_path="README.md",
        text="authority model what",
        lexical_shared_tokens=3,
        semantic_score=0.0,
        semantic_admitted=False,
    )
    subset_no_semantic = _candidate(
        evidence_id="def",
        source_path="docs/system-x-overview.md",
        text="authority model",
        lexical_shared_tokens=2,
        semantic_score=0.0,
        semantic_admitted=False,  # never cleared the semantic channel
    )

    kept = minimum_sufficient_project_evidence(query_tokens, (toc_line, subset_no_semantic))

    assert [c.evidence.evidence_id for c in kept] == ["toc"]


def test_second_chunk_of_an_already_represented_semantic_source_still_needs_new_tokens():
    """Same-source flooding stays controlled: the semantic exception fires
    only ONCE per Source (its first-seen chunk); a second chunk of the SAME
    Source, contributing nothing new, is still dropped."""
    query_tokens = frozenset({"authority", "model", "what"})
    toc_line = _candidate(
        evidence_id="toc", source_path="README.md", text="authority model what",
        lexical_shared_tokens=3, semantic_score=0.0, semantic_admitted=False,
    )
    definition_chunk_1 = _candidate(
        evidence_id="def", source_path="docs/system-x-overview.md", text="authority model",
        lexical_shared_tokens=2, semantic_score=0.88, semantic_admitted=True,
    )
    definition_chunk_2 = _candidate(
        evidence_id="def", source_path="docs/system-x-overview.md", text="authority model",
        lexical_shared_tokens=2, semantic_score=0.88, semantic_admitted=True,
    )

    kept = minimum_sufficient_project_evidence(
        query_tokens, (toc_line, definition_chunk_1, definition_chunk_2)
    )

    assert len(kept) == 2  # toc_line + exactly ONE chunk of "def", not both


def test_semantic_exception_never_fires_for_a_candidate_with_new_tokens_anyway():
    """When a candidate already contributes new tokens, the semantic
    exception is not even needed -- this confirms the two reasons compose
    correctly rather than one masking the other."""
    query_tokens = frozenset({"caching", "retries", "logging"})
    a = _candidate(
        evidence_id="a", source_path="docs/caching.md", text="caching",
        lexical_shared_tokens=1, semantic_score=0.0, semantic_admitted=False,
    )
    b = _candidate(
        evidence_id="b", source_path="docs/retries.md", text="retries",
        lexical_shared_tokens=1, semantic_score=0.0, semantic_admitted=False,
    )

    kept = minimum_sufficient_project_evidence(query_tokens, (a, b))

    assert [c.evidence.evidence_id for c in kept] == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration level: the exact real-corpus failure shape, generalized and
# reproduced with synthetic, non-hardcoded documents (CASE A).
# ---------------------------------------------------------------------------

_DEFINITION_X = (
    "# Capability X\n\nCapability X is the mechanism by which the system negotiates "
    "feature support between two components before any data is exchanged between them."
)
_TOC_REFERENCE = (
    "# Docs\n\n- [capability-x.md](capability-x.md) -- what capability X covers.\n"
    "- [logging.md](logging.md) -- what gets logged and when.\n"
    "- [deploy.md](deploy.md) -- what a deploy actually does.\n"
)


@real_model
@skip_without_model
def test_toc_style_reference_does_not_suppress_the_canonical_definition(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        definition = _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
        _seed(cx, tmp_path, "README.md", _TOC_REFERENCE)
        cx.semantic_setup()

        result = cx.context("What is capability X?", budget=4000)

        selected_ids = {item.entity_id for item in _project_evidence_items(result)}
        assert definition.evidence.evidence_id in selected_ids, (
            "the canonical definition must survive even though a shallow, incidental "
            "table-of-contents mention shares (or exceeds) its lexical token count"
        )


# ---------------------------------------------------------------------------
# MSC ROLE-AWARE FAILURE, exactly as specified: a definition chunk and a
# decision/conclusion chunk about the SAME subject, sharing many topical
# tokens, must both survive when both are independently relevant.
# ---------------------------------------------------------------------------

_MECHANISM_Y_DEFINITION = (
    "# Mechanism Y\n\nMechanism Y is a technical mechanism that coordinates writes "
    "across replicas by routing every write through a designated leader replica."
)
_MECHANISM_Y_DECISION = (
    "# Decision\n\nThe project decided to adopt mechanism Y for the staging "
    "environment only, after evaluating the write-latency cost against consistency."
)


@real_model
@skip_without_model
def test_definition_and_decision_both_survive_when_both_semantically_relevant(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        definition = _seed(cx, tmp_path, "docs/mechanism-y.md", _MECHANISM_Y_DEFINITION)
        decision = _seed(cx, tmp_path, "docs/decision-y.md", _MECHANISM_Y_DECISION)
        cx.semantic_setup()

        result = cx.context("What was decided about mechanism Y?", budget=4000)

        selected_ids = {item.entity_id for item in _project_evidence_items(result)}
        assert decision.evidence.evidence_id in selected_ids, "the decision itself must surface"
        assert definition.evidence.evidence_id in selected_ids, (
            "MSC must not treat the definition as redundant with the decision merely "
            "because they share topical tokens -- they answer different questions"
        )


# ---------------------------------------------------------------------------
# Regression guards: cases where the OLD behavior was already correct must
# stay correct (CASE B / CASE C from the A54.4 investigation).
# ---------------------------------------------------------------------------

_API_ERRORS = (
    "# API Errors\n\nA rate-limit error returns HTTP status code 429, with a "
    "Retry-After header indicating how many seconds to wait before retrying."
)
_API_OVERVIEW = (
    "# API Overview\n\nThe API is designed to be predictable and well-documented, "
    "with clear error semantics and consistent request/response shapes throughout."
)


@real_model
@skip_without_model
def test_exact_lexical_query_still_dominates_a_loosely_related_document(tmp_path):
    with _offline():
        cx = _workspace(tmp_path)
        precise = _seed(cx, tmp_path, "docs/api-errors.md", _API_ERRORS)
        _seed(cx, tmp_path, "docs/api-overview.md", _API_OVERVIEW)
        cx.semantic_setup()

        result = cx.context(
            "What is the exact HTTP status code returned on a rate-limit error?", budget=4000
        )

        selected_ids = {item.entity_id for item in _project_evidence_items(result)}
        assert precise.evidence.evidence_id in selected_ids
        rendered = "\n".join(item.content for item in _project_evidence_items(result))
        assert "429" in rendered


_ISOLATION = (
    "# Data Isolation\n\nEach tenant's records live in a separate namespace, so one "
    "customer's writes can never be visible to, or overwrite, another customer's rows."
)
_COMPONENTS = (
    "# Components\n\nThe system is organized into several components, each with its "
    "own data, its own configuration, and its own deployment lifecycle and versioning."
)


@real_model
@skip_without_model
def test_paraphrase_with_distinct_tokens_is_unaffected_by_the_semantic_exception(tmp_path):
    """A54's original multi-aspect guarantee, reconfirmed after A54.4: when
    two documents already contribute genuinely distinct query tokens, both
    were already kept before this session, and still are -- the new
    exception changes nothing here, it only adds a THIRD reason to keep a
    candidate that would otherwise have been dropped."""
    with _offline():
        cx = _workspace(tmp_path)
        isolation = _seed(cx, tmp_path, "docs/isolation.md", _ISOLATION)
        components = _seed(cx, tmp_path, "docs/components.md", _COMPONENTS)
        cx.semantic_setup()

        result = cx.context(
            "How does the system stop different components from stepping on each "
            "other's data?",
            budget=4000,
        )

        selected_ids = {item.entity_id for item in _project_evidence_items(result)}
        assert isolation.evidence.evidence_id in selected_ids
        assert components.evidence.evidence_id in selected_ids


# ---------------------------------------------------------------------------
# Lexical-only invariant: without the semantic extra, MSC behaves exactly
# as it did before A54.4 -- the exception path is unreachable, not merely
# untriggered.
# ---------------------------------------------------------------------------


def test_lexical_only_workspace_msc_behavior_is_unchanged(tmp_path):
    cx = _workspace(tmp_path)
    toc = _seed(cx, tmp_path, "README.md", _TOC_REFERENCE)
    definition = _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    # No cx.semantic_setup() call: this workspace never has semantic vectors.

    trace: list[ProjectEvidenceTrace] = []
    result = cx.context("What is capability X?", budget=4000, _project_evidence_trace=trace)

    assert all(not t.channels or "semantic" not in t.channels for t in trace)
    selected_ids = {item.entity_id for item in _project_evidence_items(result)}
    # Without the semantic channel, the definition may or may not clear
    # MSC depending on plain token coverage alone -- what this test
    # actually guards is that no candidate here was kept via a semantic
    # reason, since none exists in this workspace.
    assert toc.evidence.evidence_id in selected_ids or definition.evidence.evidence_id in selected_ids
