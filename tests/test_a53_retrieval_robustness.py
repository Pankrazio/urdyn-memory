"""Retrieval robustness for PROJECT EVIDENCE: cross-document ORDERING.

A52/A52.1 established that a seeded Source's current observation is admitted
into `context()` through three independent boolean channels (lexical majority,
FTS widening, semantic floor) and that an admitted document is split into
relevance-RANKED chunks. Neither step ever decided anything about the order in
which DIFFERENT admitted documents compete for the budget: the candidate list
came out of `MemoryStore.list_current_source_evidence()` ordered
alphabetically by workspace-relative path, and `compile_context` admits by a
strict PREFIX scan that stops at the first candidate that does not fit.

Alphabetical order is an attribute of the FILENAME, not of the query. Combined
with a prefix-stop budget scan it means the alphabetically-first admitted
document gets first claim on the whole budget regardless of how strongly it
matches, and a later, far better-matching document can contribute nothing.

Every corpus below is synthetic and deliberately generic (widgets, sharding,
throughput). The `docs/architecture/*` versus `docs/research/*` layout is kept
because the DIRECTORY-NAME alphabetical relationship is the structural trigger
being regression-tested -- not because any real project is being described.
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


def _selected_source_paths(result) -> set[str]:
    return {item.source_path for item in _project_evidence_items(result)}


def _chunks_per_source(result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _project_evidence_items(result):
        counts[item.source_path] = counts.get(item.source_path, 0) + 1
    return counts


def _paragraphs(*chunks: str) -> str:
    return "\n\n".join(chunks) + "\n"


# ---------------------------------------------------------------------------
# Symptom A / D -- wrong-source dominance and source monopolization.
#
# `docs/architecture/aggregate-widget-overview.md` sorts before
# `docs/research/widget-throughput-findings.md`. The overview is broad and only
# moderately on-topic; the findings document is precisely what the query asks
# for. Under a budget that fits roughly two chunks, alphabetical order lets the
# overview take everything.
# ---------------------------------------------------------------------------

_THROUGHPUT_QUERY = (
    "Which widget throughput benchmark result did the sharded pipeline evaluation "
    "record under sustained load?"
)

_BROAD_OVERVIEW = _paragraphs(
    "# Widget Platform Overview",
    "The widget platform is organised as a pipeline of stages. Each stage owns its own "
    "queue and can be scaled independently, which is the property most teams care about "
    "when they first read this overview document end to end.",
    "A pipeline stage may be sharded. Sharding a stage splits its queue across several "
    "workers so that no single worker becomes the bottleneck for the whole widget "
    "pipeline during a busy period of the day.",
    "Throughput is one of several properties the platform reports. The others are "
    "latency, queue depth and error rate, and each is reported per stage rather than "
    "for the pipeline as a whole, which makes comparison across deployments awkward.",
    "Load is applied to the widget pipeline by the traffic generator that ships with the "
    "platform. The generator can replay a recorded trace or synthesise a flat request "
    "pattern, and most teams start with the flat pattern because it is simpler.",
    "A benchmark harness exists in the repository but it is not wired into continuous "
    "integration. Running it is a manual step and the result is written to a scratch "
    "directory that nobody collects, so the numbers are rarely compared over time.",
    "An evaluation of the platform is scheduled once per release. The evaluation covers "
    "documentation, packaging and operational readiness, and it does not record any "
    "performance number of its own for the widget pipeline.",
    "Which stage to scale first is usually decided by inspecting queue depth rather than "
    "by any formal method. This works well enough while the pipeline is small and stops "
    "working once several stages are sharded at the same time.",
)

_PRECISE_FINDINGS = _paragraphs(
    "# Widget Throughput Findings",
    "The sharded pipeline evaluation did record a widget throughput benchmark result "
    "under sustained load: 18,400 widgets per second sustained for six hours, which is "
    "the number this document exists to state.",
    "Test rig and calibration notes are kept in an appendix that is not reproduced here.",
)


def test_symptom_a_precise_source_is_represented_alongside_the_broad_one(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    result = cx.context(_THROUGHPUT_QUERY, budget=1400)

    assert result.used <= 1400
    counts = _chunks_per_source(result)
    assert counts.get("docs/research/widget-throughput-findings.md", 0) >= 1, (
        "the document that literally answers the query contributed nothing; "
        f"selection was {counts}"
    )


def test_symptom_a_answering_paragraph_actually_reaches_the_compiled_context(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    result = cx.context(_THROUGHPUT_QUERY, budget=1400)

    rendered = "\n".join(item.content for item in _project_evidence_items(result))
    assert "18,400 widgets per second" in rendered


# ---------------------------------------------------------------------------
# Symptom B -- perturbation instability.
#
# Two near-paraphrases of one question differing by a single non-essential
# qualifier ("actually"). The threshold in `_relevance.is_relevant` is
# `len(query_tokens) // 2 + 1`, so one extra token can move it, changing which
# documents clear admission. The assertion below is deliberately about the
# COMPILED result rather than the admitted set: the requirement is that the
# compiled context does not change catastrophically, quantified as Jaccard
# overlap >= 0.5 between the two selected source-path sets.
# ---------------------------------------------------------------------------

# Nine significant tokens (threshold 5) versus ten (threshold 6). The single
# added word is a pure qualifier that changes no intent.
_PERTURBATION_QUERY_A = "How does the sharded widget pipeline handle sustained load spikes today?"
_PERTURBATION_QUERY_B = "How does the sharded widget pipeline actually handle sustained load spikes today?"

# Shares all nine of query A's significant tokens.
_SPIKE_HANDLING = _paragraphs(
    "# Load Spike Handling",
    "How the sharded widget pipeline will handle sustained load spikes today: work is "
    "admitted into a bounded queue per shard and the overflow is shed. Shedding is more "
    "aggressive for sustained spikes than for short bursts.",
    "Shed work is retried once from the spillover buffer.",
)

# Shares exactly five of query A's significant tokens (sharded, widget,
# pipeline, load, spikes) -- enough at threshold 5, not enough at threshold 6.
_SHARD_BACKGROUND = _paragraphs(
    "# Shard Background",
    "A shard belongs to one widget pipeline worker and owns its own queue. Traffic is "
    "distributed across shards by a hash of the widget key, so a sharded pipeline spreads "
    "load evenly as long as the key space remains uniform across every partition.",
    "Non-uniform key spaces produce hot shards. Hot shards are the usual reason that load "
    "spikes become visible latency inside a sharded widget pipeline, and they are also the "
    "usual reason an operator reaches for a manual rebalance of the widget pipeline.",
    "Rebalancing a sharded pipeline moves key ranges between shards while the widget "
    "pipeline keeps running, and no request is dropped while a range is in flight between "
    "one shard and another, which is the property the rebalance protocol guarantees.",
)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def test_symptom_b_one_filler_word_does_not_change_the_selected_sources(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/shard-background.md", _SHARD_BACKGROUND)
    _seed(cx, tmp_path, "docs/research/spike-handling.md", _SPIKE_HANDLING)

    # 1000 characters fits exactly one of these documents' leading chunks.
    selected_a = _selected_source_paths(cx.context(_PERTURBATION_QUERY_A, budget=1000))
    selected_b = _selected_source_paths(cx.context(_PERTURBATION_QUERY_B, budget=1000))

    assert selected_a, "baseline query selected no project evidence at all"
    assert selected_b, "perturbed query selected no project evidence at all"
    overlap = _jaccard(selected_a, selected_b)
    assert overlap >= 0.5, (
        "a single filler word changed the compiled context catastrophically: "
        f"{selected_a} vs {selected_b} (jaccard {overlap:.2f})"
    )


# ---------------------------------------------------------------------------
# Symptom C -- intent mismatch (definition versus conclusion).
#
# Both documents are about the same subject. The definition document shares
# more raw subject vocabulary; the conclusion document is the one that answers
# "what did we conclude". Alphabetically the definition sorts first.
# ---------------------------------------------------------------------------

_CONCLUSION_QUERY = (
    "What did we conclude about widget sharding as a strategic opportunity for the platform?"
)

_SHARDING_DEFINITION = _paragraphs(
    "# Widget Sharding: Definition",
    "Widget sharding is the technique of partitioning the widget keyspace across several "
    "independent workers. The platform implements it with a consistent hash ring, and "
    "each worker owns a contiguous arc of that ring.",
    "A shard boundary in the widget platform is not a transaction boundary. Anything that "
    "must be atomic across two widget shards has to be expressed as a saga, which is what "
    "the platform sharding guide describes in its second half.",
    "Rebalancing moves arcs between workers. The platform performs rebalancing online, and "
    "widget sharding therefore never requires a maintenance window, which is the property "
    "most often cited about this design.",
    "What sharding does not do is reduce total work. It distributes work, and the platform "
    "documentation about widget sharding is careful to say so in the opening paragraph of "
    "every chapter.",
)

_SHARDING_CONCLUSION = _paragraphs(
    "# Sharding Review: Conclusion",
    "We conclude that widget sharding is a strategic opportunity for the platform rather "
    "than a maintenance chore: it is what lets us sell the platform into accounts whose "
    "volume we cannot serve today.",
    "The review board recorded no dissent on this conclusion.",
)


def test_symptom_c_conclusion_document_is_not_crowded_out_by_the_definition(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/widget-sharding-definition.md", _SHARDING_DEFINITION)
    _seed(cx, tmp_path, "docs/research/widget-sharding-review.md", _SHARDING_CONCLUSION)

    result = cx.context(_CONCLUSION_QUERY, budget=1400)

    counts = _chunks_per_source(result)
    assert counts.get("docs/research/widget-sharding-review.md", 0) >= 1, (
        f"the document holding the conclusion contributed nothing; selection was {counts}"
    )
    rendered = "\n".join(item.content for item in _project_evidence_items(result))
    assert "strategic opportunity for the platform" in rendered


# ---------------------------------------------------------------------------
# Symptom D -- source monopolization across a multi-aspect query.
#
# One alphabetically-early source has many moderately relevant chunks; two
# alphabetically-later sources each hold one highly relevant chunk about a
# distinct aspect of the same question.
# ---------------------------------------------------------------------------

_MULTI_ASPECT_QUERY = (
    "What is the retry policy and what is the encryption policy for widget pipeline "
    "records in the archive?"
)

_MODERATE_BULK = _paragraphs(
    "# Archive Notes",
    "The archive stores widget pipeline records after they leave the live pipeline. Every "
    "record keeps its original identifier so that a record in the archive can be traced "
    "back to the widget that produced it.",
    "Records in the archive are grouped into daily segments. A segment is sealed at "
    "midnight and no record is ever appended to a sealed segment of the widget archive "
    "afterwards, which simplifies the retention policy considerably.",
    "The archive exposes a read path for records and a policy hook that is consulted "
    "before a record is served. What that hook does for widget pipeline records is "
    "deployment specific and is not fixed by the archive itself.",
    "Segment compaction rewrites older widget records into larger files. Compaction is a "
    "policy of the archive operator, not of the pipeline, and the records themselves are "
    "unchanged by it apart from their physical location.",
    "The archive reports a record count and a byte count per segment. Neither number "
    "distinguishes widget pipeline records from other records, so the archive policy "
    "documentation recommends a separate tally if that distinction matters.",
    "What the archive does not provide is a search index over records. Callers that need "
    "one build it themselves from the widget pipeline record stream, and the archive "
    "policy is explicitly silent on how they should do that.",
)

_RETRY_ASPECT = _paragraphs(
    "# Retry Policy",
    "The retry policy for widget pipeline records in the archive is three attempts with "
    "exponential backoff, after which the record is parked in the dead-letter segment.",
)

_ENCRYPTION_ASPECT = _paragraphs(
    "# Encryption Policy",
    "The encryption policy for widget pipeline records in the archive is envelope "
    "encryption per segment, with the data key rotated every time a segment is sealed.",
)


def test_symptom_d_high_relevance_sources_survive_a_bulky_moderate_source(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/archive-notes.md", _MODERATE_BULK)
    _seed(cx, tmp_path, "docs/research/retry-policy.md", _RETRY_ASPECT)
    _seed(cx, tmp_path, "docs/research/encryption-policy.md", _ENCRYPTION_ASPECT)

    result = cx.context(_MULTI_ASPECT_QUERY, budget=1600)

    selected = _selected_source_paths(result)
    assert "docs/research/retry-policy.md" in selected or "docs/research/encryption-policy.md" in selected, (
        f"only the bulky moderate-relevance source survived; selection was {selected}"
    )


# ---------------------------------------------------------------------------
# Ordering invariants the fix must keep.
# ---------------------------------------------------------------------------


def test_ordering_is_by_lexical_overlap_with_path_only_as_a_final_tiebreak(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    trace: list[ProjectEvidenceTrace] = []
    cx.context(_THROUGHPUT_QUERY, budget=1400, _project_evidence_trace=trace)

    assert trace, "instrumentation collected nothing"
    scores = [row.lexical_shared_tokens for row in trace]
    assert scores == sorted(scores, reverse=True), f"candidates are not relevance-ordered: {scores}"
    assert trace[0].source_path == "docs/research/widget-throughput-findings.md"
    # Ties fall back to path/chunk_index, so the order stays total and stable.
    keys = [(-r.lexical_shared_tokens, -r.semantic_score, r.source_path, r.chunk_index) for r in trace]
    assert keys == sorted(keys)


def test_trace_reports_admission_channels_and_budget_omission(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/archive-notes.md", _MODERATE_BULK)
    _seed(cx, tmp_path, "docs/research/retry-policy.md", _RETRY_ASPECT)
    _seed(cx, tmp_path, "docs/research/encryption-policy.md", _ENCRYPTION_ASPECT)

    trace: list[ProjectEvidenceTrace] = []
    result = cx.context(_MULTI_ASPECT_QUERY, budget=1600, _project_evidence_trace=trace)

    assert [row.position for row in trace] == list(range(len(trace)))
    assert all(row.channels for row in trace), "an admitted candidate reported no admission channel"
    assert all(set(row.channels) <= {"lexical", "fts", "semantic"} for row in trace)
    # `selected` must agree exactly with what the compiled context shows.
    traced_selected = {(row.source_path, row.chunk_index) for row in trace if row.selected}
    compiled_selected = {
        (item.source_path, item.chunk_index or 0) for item in _project_evidence_items(result)
    }
    assert traced_selected == compiled_selected
    assert any(not row.selected for row in trace), "fixture no longer exercises budget omission"


def test_trace_parameter_never_changes_the_compiled_result(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    without = cx.context(_THROUGHPUT_QUERY, budget=1400)
    with_trace = cx.context(_THROUGHPUT_QUERY, budget=1400, _project_evidence_trace=[])
    assert without.render() == with_trace.render()
    assert without.used == with_trace.used


def test_project_evidence_selection_is_deterministic_across_repeated_calls(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    first = cx.context(_THROUGHPUT_QUERY, budget=1400).render()
    second = cx.context(_THROUGHPUT_QUERY, budget=1400).render()
    assert first == second


def test_project_evidence_selection_stays_prefix_monotonic_across_budgets(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/aggregate-widget-overview.md", _BROAD_OVERVIEW)
    _seed(cx, tmp_path, "docs/research/widget-throughput-findings.md", _PRECISE_FINDINGS)

    def _keys(budget):
        return [
            (item.source_path, item.chunk_index)
            for item in _project_evidence_items(cx.context(_THROUGHPUT_QUERY, budget=budget))
        ]

    small, medium, large = _keys(900), _keys(1800), _keys(8000)
    assert medium[: len(small)] == small
    assert large[: len(medium)] == medium
