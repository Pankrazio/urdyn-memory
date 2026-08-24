"""Regression coverage for Minimum Sufficient Context in PROJECT EVIDENCE
selection (A54).

Independent mechanism from `test_a54_intent_sensitive_retrieval.py`'s
intent weighting: this file never asserts on WHICH document/chunk ranks
first, only on HOW MANY otherwise-eligible, already-ranked candidates
`Urdyn.context()` actually offers to the budget scan.

`compile_context`'s budget admission (A29.1) is a deterministic prefix
scan over already-ranked candidates: given a generous enough budget, it
used to admit every relevant PROJECT EVIDENCE chunk up to the byte limit,
including chunks that only restate a topic a higher-ranked chunk already
covered. Real signal did not grow with the extra budget spent on them --
observed directly during A54 dogfooding review, comparing a small budget
against a much larger one over the same corpus.

`_preflight.minimum_sufficient_project_evidence` (see its docstring) is a
redundancy filter that runs BEFORE `compile_context` ever sees the
candidate list: a candidate is kept only if its own text contains at
least one query token not already covered by a higher-ranked, already-kept
candidate. `compile_context`'s own prefix-scan algorithm is completely
unchanged by this -- the filter only changes what is offered as a
candidate, uniformly regardless of `budget`.
"""

from __future__ import annotations

from urdyn import Urdyn
from urdyn._context import SECTION_PROJECT_EVIDENCE
from urdyn._preflight import ProjectEvidenceTrace

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


# ---------------------------------------------------------------------------
# The saturation fixture: one seeded document, two chunks that together
# cover the query well, and eight more chunks that only restate a topic
# those two already cover -- no chunk here shares NO vocabulary with the
# query (all ten would clear `evidence_is_relevant`'s admission gate),
# but only the first two contribute anything the others do not.
# ---------------------------------------------------------------------------

_QUERY = "What is the sustained widget throughput benchmark result and which worker count was tested?"

_SUFFICIENT_1 = (
    "The sustained widget throughput benchmark recorded a result of 18,400 widgets per second, "
    "and every worker involved in the run logged its own reading to a shared spreadsheet for "
    "later cross-team review by anyone who needed the raw numbers afterward."
)
_SUFFICIENT_2 = (
    "The benchmark tested a worker count of sixteen for this widget throughput run, chosen after "
    "an earlier smaller trial suggested sixteen was the point past which adding still more "
    "workers stopped changing the throughput figure the benchmark could observe."
)
_REDUNDANT = [
    "The widget throughput benchmark was considered strong overall by the team, and it was "
    "referenced again in several follow-up planning conversations about future capacity.",
    "Widget throughput was measured carefully across the whole benchmark run, with extra "
    "attention paid to keeping the measurement window consistent from start to finish.",
    "Widget throughput benchmark numbers were reviewed by everyone, and nobody on the "
    "reviewing team raised any concern about how the figures had been collected that day.",
    "The widget throughput benchmark was archived in the same shared location the team "
    "already used for every earlier benchmark run of exactly this same general kind.",
    "Widget throughput stayed steady for the full duration of the benchmark run, and "
    "several people later cited it as evidence the setup was stable under real load.",
    "This widget throughput benchmark matched figures reported earlier, so no further "
    "investigation into a possible discrepancy was considered necessary at the time.",
    "Widget throughput benchmark data was exported for the quarterly review, formatted the "
    "same way it always is so the reviewing committee could compare it directly.",
    "The team discussed the widget throughput benchmark during the retro, and agreed it "
    "was worth mentioning again in the next quarterly planning document as well.",
]

_SATURATION_DOC = "# Findings\n\n" + "\n\n".join([_SUFFICIENT_1, _SUFFICIENT_2] + _REDUNDANT) + "\n"

# Budgets chosen against the actual rendered cost of this fixture's
# chunks (not round numbers): SMALL fits only the single top-ranked
# chunk, MEDIUM fits both sufficient chunks exactly, LARGE is a hundred
# times MEDIUM so the prefix-scan alone would have had ample room to
# admit all ten chunks if nothing filtered the candidate list first.
_SMALL_BUDGET = 400
_MEDIUM_BUDGET = 900
_LARGE_BUDGET = 100_000


def _seed_saturation_doc(cx, tmp_path):
    return _seed(cx, tmp_path, "docs/findings.md", _SATURATION_DOC)


# ---------------------------------------------------------------------------
# 1 -- a large budget does not force the eight redundant chunks in.
# ---------------------------------------------------------------------------


def test_large_budget_does_not_admit_redundant_chunks(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    result = cx.context(_QUERY, budget=_LARGE_BUDGET)

    items = _project_evidence_items(result)
    assert len(items) == 2, f"expected exactly the 2 sufficient chunks, got {len(items)}: {items}"


# ---------------------------------------------------------------------------
# 2 -- context stays within budget (basic sanity, every budget tier).
# ---------------------------------------------------------------------------


def test_context_always_stays_within_budget(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    for budget in (_SMALL_BUDGET, _MEDIUM_BUDGET, _LARGE_BUDGET):
        result = cx.context(_QUERY, budget=budget)
        assert result.used <= budget


# ---------------------------------------------------------------------------
# 3 -- a strong single-aspect budget can stop after sufficient support:
# a small budget that only fits the single top-ranked chunk still gets
# the one that best supports the query, not an arbitrary redundant one.
# ---------------------------------------------------------------------------


def test_small_budget_selects_the_top_ranked_sufficient_chunk(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    result = cx.context(_QUERY, budget=_SMALL_BUDGET)

    items = _project_evidence_items(result)
    assert len(items) == 1
    assert "18,400 widgets per second" in items[0].content


# ---------------------------------------------------------------------------
# 4 -- medium budget: both genuinely sufficient chunks fit and are both
# selected; the filter does not under-select when there IS room for the
# full sufficient set.
# ---------------------------------------------------------------------------


def test_medium_budget_selects_both_sufficient_chunks(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    result = cx.context(_QUERY, budget=_MEDIUM_BUDGET)

    items = _project_evidence_items(result)
    assert len(items) == 2
    rendered = "\n".join(item.content for item in items)
    assert "18,400 widgets per second" in rendered
    assert "worker count of sixteen" in rendered


# ---------------------------------------------------------------------------
# 5 -- medium and large budgets select the SAME chunks: once the
# sufficient set is exhausted, spending more budget adds nothing, exactly
# the "minimum SUFFICIENT, not minimum possible" distinction the filter
# is required to make.
# ---------------------------------------------------------------------------


def test_medium_and_large_budgets_agree_once_sufficient_set_is_exhausted(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    medium = cx.context(_QUERY, budget=_MEDIUM_BUDGET)
    large = cx.context(_QUERY, budget=_LARGE_BUDGET)

    medium_paths = {(item.entity_id, item.chunk_index) for item in _project_evidence_items(medium)}
    large_paths = {(item.entity_id, item.chunk_index) for item in _project_evidence_items(large)}
    assert medium_paths == large_paths


# ---------------------------------------------------------------------------
# 6 -- diagnostic trace: redundant candidates are marked NOT sufficient
# (cut before the budget scan), never conflated with "cut for budget".
# ---------------------------------------------------------------------------


def test_trace_marks_redundant_candidates_as_not_sufficient(tmp_path):
    cx = _workspace(tmp_path)
    _seed_saturation_doc(cx, tmp_path)

    trace: list[ProjectEvidenceTrace] = []
    cx.context(_QUERY, budget=_LARGE_BUDGET, _project_evidence_trace=trace)

    sufficient_count = sum(1 for row in trace if row.sufficient)
    assert sufficient_count == 2
    assert len(trace) == 10  # every admitted chunk of the seeded document is still traced


# ---------------------------------------------------------------------------
# MULTI-ASPECT SAFETY -- the filter must never collapse a genuinely
# multi-aspect query down to "only the first source". Three distinct
# documents, each the ONLY source for its own aspect of the query: all
# three must survive, because each contributes query tokens none of the
# others do.
# ---------------------------------------------------------------------------

# "request" is not preamble filler: each note's OWN content naturally
# talks about a widget request (a cache lookup, a retried send, a logged
# call), so it is a legitimately shared token, not one artificially
# inserted to inflate admission. Each note ends up sharing exactly three
# of the query's five significant tokens ("widget", "request", and its
# own aspect word) -- enough to clear the majority admission threshold
# without any cross-referencing preamble that would leak the OTHER two
# aspects' vocabulary into every note and make them look redundant to
# each other from the query's point of view.
_MULTI_ASPECT_QUERY = "widget caching retries logging request"

_CACHING_DOC = (
    "# Widget Caching\n\n"
    "Widget caching stores a recently computed widget result in memory so a repeated request "
    "for the same widget does not recompute it, trading memory for latency on cache hits."
)
_RETRIES_DOC = (
    "# Widget Retries\n\n"
    "Widget retries resend a failed widget request up to three times with a short backoff "
    "between attempts before the caller is finally told the widget request failed."
)
_LOGGING_DOC = (
    "# Widget Logging\n\n"
    "Widget logging records every widget request and its outcome to a structured log line "
    "so an operator can reconstruct what happened to any one widget after the fact."
)


def test_multi_aspect_query_still_gathers_all_three_aspects(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/caching.md", _CACHING_DOC)
    _seed(cx, tmp_path, "docs/retries.md", _RETRIES_DOC)
    _seed(cx, tmp_path, "docs/logging.md", _LOGGING_DOC)

    result = cx.context(_MULTI_ASPECT_QUERY, budget=_LARGE_BUDGET)

    paths = {item.source_path for item in _project_evidence_items(result)}
    assert paths == {"docs/caching.md", "docs/retries.md", "docs/logging.md"}
