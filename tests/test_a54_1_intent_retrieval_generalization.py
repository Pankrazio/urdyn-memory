"""Regression coverage for A54.1 -- intent retrieval generalization.

A54 introduced `_chunk._weighted_overlap`: a shared token drawn from a
small, fixed `_INTENT_TERMS` vocabulary counted `_INTENT_TERM_WEIGHT`
times instead of once, to stop a topically-dense definition document from
out-ranking a short, precise conclusion document. Real dogfooding of that
fix on a multi-document workspace (replaying a historical failing query)
falsified it: an entirely unrelated document that happened to contain the
single word "conclude" in an aside about something else got that lone
token boosted 3x, with ZERO other shared vocabulary -- enough for
`minimum_sufficient_project_evidence` to keep it as if it were relevant.
Confirmed by grep against the real corpus: the word never appears
anywhere in either document that actually discusses the query's real
topic, so no amount of growing `_INTENT_TERMS` could have fixed this
specific case.

The fix (`_chunk._weighted_overlap`, see its docstring): the weight only
applies when the candidate ALSO shares at least one non-intent (topical)
token with the query -- independent evidence the candidate is even about
the right subject. An intent word matched in total topical isolation now
counts as an ordinary token (weight 1), exactly as it did before A54
introduced the weighting at all.

This file tests two things `test_a54_intent_sensitive_retrieval.py`
does not:

1. The false-positive this fix removes (§ FALSE POSITIVE below).
2. Honest paraphrase generalization (§ PARAPHRASE GENERALIZATION below):
   verbs already in `_INTENT_TERMS` (concluded/decided/evaluated/found)
   get the mechanism's help; verbs deliberately NOT added to that list
   this session (established/determined/assessed -- confirmed by real
   dogfooding to be absent from the corpus that originally exposed this
   bug, so adding them would not even have helped there) get NO special
   help and the tests say so explicitly, rather than asserting success
   that would not reflect what the code actually does. Growing the list
   ad hoc to make these pass was explicitly ruled out; see the module
   docstring in `_chunk.py`.
"""

from __future__ import annotations

import pytest

from urdyn import Urdyn
from urdyn._chunk import EvidenceChunk, score_evidence_chunks
from urdyn._context import SECTION_PROJECT_EVIDENCE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _chunk(text: str) -> EvidenceChunk:
    return EvidenceChunk(evidence_id="e1", chunk_index=0, chunk_count=1, start=0, end=len(text), text=text)


# ---------------------------------------------------------------------------
# FALSE POSITIVE -- an intent word matched with zero topical overlap must
# not be boosted. Text below is a synthetic reconstruction, generic and
# not tied to any real project, of the exact shape that falsified A54's
# original fix during dogfooding review: an intent word appearing in a
# sentence about an entirely unrelated topic.
# ---------------------------------------------------------------------------

_INCIDENTAL_INTENT_SENTENCE = (
    "It must not be possible to read the project's public documentation "
    "and conclude that it endorses a specific third-party vendor."
)


def test_lone_intent_token_with_no_topical_overlap_is_not_boosted():
    query_tokens = frozenset({"aware", "differentiation", "hydration", "model", "conclude"})
    candidate = _chunk(_INCIDENTAL_INTENT_SENTENCE)

    ((score, _),) = score_evidence_chunks(query_tokens, (candidate,))

    # Exactly the raw shared-token count (just "conclude") -- no A54
    # weight applied, since there is no topical token alongside it.
    assert score == 1


def test_intent_token_still_boosted_when_topical_overlap_present():
    # Regression guard: the fix must not disable weighting altogether --
    # only when the intent token has no topical company.
    query_tokens = frozenset({"widget", "throughput", "concluded"})
    candidate = _chunk("The team concluded widget throughput was acceptable.")

    ((score, _),) = score_evidence_chunks(query_tokens, (candidate,))

    assert score == 5  # 2 topical ("widget", "throughput") + 3x "concluded"


def test_incidental_intent_word_does_not_outrank_the_real_topical_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(
        cx,
        tmp_path,
        "docs/architecture/capability-x.md",
        "# Capability X\n\nCapability X is a technical mechanism for model-aware hydration, "
        "coordinating differentiation between destination models.",
    )
    _seed(
        cx,
        tmp_path,
        "docs/architecture/unrelated-boundary.md",
        f"# Boundary\n\n{_INCIDENTAL_INTENT_SENTENCE}",
    )

    result = cx.context(
        "What did the research conclude about capability X's model-aware hydration differentiation?",
        budget=100000,
    )

    paths = [item.source_path for item in _project_evidence_items(result)]
    assert "docs/architecture/unrelated-boundary.md" not in paths, (
        f"a document whose ONLY connection to the query is one incidental intent word "
        f"must not be admitted ahead of (or instead of) the genuinely on-topic one; got {paths}"
    )
    assert "docs/architecture/capability-x.md" in paths


# ---------------------------------------------------------------------------
# PARAPHRASE GENERALIZATION -- an isolated fixture where the definition
# and conclusion documents are engineered to TIE on raw topical overlap
# (both share exactly the query's topical vocabulary, nothing more), so
# the ONLY thing that can break the tie is whether the target verb is
# weighted. This isolates the verb itself: no other intent word (like
# "evaluation") is present anywhere to confound the result, unlike
# `test_a54_intent_sensitive_retrieval.py`'s CASE A, whose query and
# documents already contain "evaluation" (itself in `_INTENT_TERMS`).
# ---------------------------------------------------------------------------

_DEFINITION_X = """# Capability X

Capability X is a technical mechanism that allows a system to handle
concurrent load by partitioning work across independent workers under
heavy load. Capability X operates by assigning each incoming unit of
load a shard key and routing that load to the worker responsible for the
shard, so that capability X performs well as workers handle more load.
Capability X was designed so that performance under load stays stable as
load increases, and its performance characteristics under load are
documented here in detail for engineers who need to reason about how
capability X performs when load grows.
"""


def _conclusion_doc(verb: str) -> str:
    return f"# Findings\n\nThe report {verb} that capability X performs poorly under load beyond twelve workers.\n"


def _query(verb: str) -> str:
    return f"What did the report {verb} about how capability X performs under load?"


# Verified empirically (raw shared-token count) that this exact pairing
# ties at 7-7 between `_DEFINITION_X` and `_conclusion_doc(verb)` for
# EVERY verb below, regardless of vocabulary membership -- so a resolved
# tie in favor of the conclusion document can only be attributed to the
# A54.1 weighting, never to an accidental raw-count edge.
_IN_VOCABULARY_VERBS = ("concluded", "decided", "evaluated", "found")
_OUT_OF_VOCABULARY_VERBS = ("established", "determined", "assessed")


@pytest.mark.parametrize("verb", _IN_VOCABULARY_VERBS)
def test_in_vocabulary_paraphrase_breaks_the_tie_toward_the_conclusion(tmp_path, verb):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    _seed(cx, tmp_path, "docs/research/capability-x-findings.md", _conclusion_doc(verb))

    result = cx.context(_query(verb), budget=100000)

    paths = [item.source_path for item in _project_evidence_items(result)]
    assert paths and paths[0] == "docs/research/capability-x-findings.md", (
        f"verb {verb!r} is in `_INTENT_TERMS`; the tie must resolve toward the document "
        f"that actually states the conclusion, got order {paths}"
    )


@pytest.mark.parametrize("verb", _OUT_OF_VOCABULARY_VERBS)
def test_out_of_vocabulary_paraphrase_is_an_honestly_documented_gap(tmp_path, verb):
    """These verbs were deliberately NOT added to `_INTENT_TERMS` this
    session (see `_chunk.py`'s module docstring): real dogfooding showed
    that growing the list is not the fix for the failure that motivated
    A54.1, and there was no evidence these specific verbs generalize
    better than any other synonym would.

    This test asserts what the code ACTUALLY does for them today --
    falling through to the tie's next deterministic key
    (`source_path`, alphabetical) -- which currently favors the WRONG
    document, `capability-x.md`. This is the same class of bug A54
    originally fixed, just for a verb family outside the current
    vocabulary. It is recorded here as a known, honest limitation (see
    the A54.1 report's MUST/CANDIDATE follow-ups), not silently masked
    by an assertion that would only pass by accident.
    """
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    _seed(cx, tmp_path, "docs/research/capability-x-findings.md", _conclusion_doc(verb))

    result = cx.context(_query(verb), budget=100000)

    paths = [item.source_path for item in _project_evidence_items(result)]
    assert paths and paths[0] == "docs/architecture/capability-x.md", (
        f"if this starts failing, verb {verb!r} started resolving correctly on its own -- "
        "update this test's expectation and the report's limitations section, do not just "
        "delete the assertion"
    )
