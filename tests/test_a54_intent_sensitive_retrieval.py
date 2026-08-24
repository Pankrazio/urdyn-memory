"""Regression coverage for intent-sensitive PROJECT EVIDENCE ranking (A54).

A52/A53 established that an admitted seeded document's chunks are ranked
by raw shared-token COUNT (`_chunk.score_evidence_chunks`), and that
different documents compete for budget in that same order
(`_preflight.ordered_project_evidence`). Neither step ever distinguished
WHAT KIND of shared token a match was: a long, topically-dense document
that paraphrases a subject many different ways racks up more raw overlap
than a short, precise document that directly answers the query -- even
when the query itself names the kind of answer it wants (a conclusion, a
decision, an evaluation result, the current state) and only the short
document actually contains it.

This was reproduced twice during real, independent dogfooding sessions: a
dense technical definition kept outranking a later research conclusion on a
query that explicitly asked what that research concluded. The fix
(`_chunk._weighted_overlap`, `_chunk._INTENT_TERMS`) weights a shared
token drawn from a small, fixed, domain-generic intent vocabulary more
heavily than an ordinary topical token -- still deterministic, still
requiring the token to appear in the query itself, still touching
RANKING only and never the boolean admission gate (`evidence_is_relevant`
is unchanged and is exercised directly below).

Every corpus here is synthetic and generic (capability X, mechanism Y,
system W, system Z) -- mirroring `test_a53_retrieval_robustness.py`'s own
convention -- not tied to any real project, vendor, or dogfood file.
"""

from __future__ import annotations

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


def _selected_source_paths(result) -> tuple[str, ...]:
    return tuple(item.source_path for item in _project_evidence_items(result))


# ---------------------------------------------------------------------------
# Unit-level: the weighting primitive itself, isolated from retrieval.
# ---------------------------------------------------------------------------


def _chunk(text: str) -> EvidenceChunk:
    return EvidenceChunk(evidence_id="e1", chunk_index=0, chunk_count=1, start=0, end=len(text), text=text)


def test_shared_intent_token_outweighs_equal_generic_overlap():
    # Two candidates share the same NUMBER of generic tokens with the
    # query; only the second also shares an intent word the query itself
    # asks for ("concluded"). The intent-bearing one must score higher.
    query_tokens = frozenset({"widget", "throughput", "concluded"})
    generic_only = _chunk("The widget throughput document explains widget throughput in general terms.")
    with_intent = _chunk("The team concluded widget throughput was acceptable.")

    ((generic_score, _),) = score_evidence_chunks(query_tokens, (generic_only,))
    ((intent_score, _),) = score_evidence_chunks(query_tokens, (with_intent,))

    assert intent_score > generic_score


def test_intent_word_absent_from_query_never_boosts_anything():
    # The candidate contains "concluded", but the query never asked about
    # a conclusion -- the shared-vocabulary set excludes it entirely, so
    # weighting has nothing to act on and the plain count is unchanged.
    query_tokens = frozenset({"widget", "throughput"})
    candidate = _chunk("The team concluded widget throughput was acceptable.")

    ((score, _),) = score_evidence_chunks(query_tokens, (candidate,))

    assert score == 2  # exactly "widget" + "throughput", no boost applied


# ---------------------------------------------------------------------------
# CASE A -- definition vs. conclusion.
#
# Verified ranking-reorder case: BOTH documents are admitted for the
# conclusion query, and the fix changes which one is ordered first (a true
# ranking regression test, not merely an admission-gate difference).
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

_CONCLUSION_X = """# Findings

The evaluation concluded that capability X performs poorly under load
beyond twelve workers.
"""


def test_case_a_definition_query_prefers_the_definition_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    _seed(cx, tmp_path, "docs/research/capability-x-findings.md", _CONCLUSION_X)

    result = cx.context("What is capability X?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths[0] == "docs/architecture/capability-x.md"


def test_case_a_conclusion_query_prefers_the_conclusion_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    _seed(cx, tmp_path, "docs/research/capability-x-findings.md", _CONCLUSION_X)

    result = cx.context(
        "What did the later evaluation conclude about how capability X performs under load?",
        budget=100000,
    )

    paths = _selected_source_paths(result)
    assert set(paths) == {"docs/architecture/capability-x.md", "docs/research/capability-x-findings.md"}, (
        "both documents must still be admitted -- this case is about RANKING, not admission"
    )
    assert paths[0] == "docs/research/capability-x-findings.md", (
        f"the document that literally states the conclusion must rank first; order was {paths}"
    )


# ---------------------------------------------------------------------------
# CASE B -- definition vs. decision.
# ---------------------------------------------------------------------------

_DEFINITION_Y = """# Mechanism Y

Mechanism Y is a technical mechanism that coordinates writes across
replicas by routing every write through a designated leader replica for
mechanism Y. Mechanism Y guarantees that a write is visible on every
replica before mechanism Y acknowledges it, so mechanism Y trades some
write latency for the consistency guarantee mechanism Y provides. The
leader election protocol mechanism Y depends on is documented separately
from how mechanism Y itself routes and acknowledges writes.
"""

_DECISION_Y = """# Decision

The project decided to adopt mechanism Y for the staging environment
only, after evaluating the write-latency cost against the consistency
guarantee it provides.
"""


def test_case_b_definition_query_prefers_the_mechanism_definition(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/mechanism-y.md", _DEFINITION_Y)
    _seed(cx, tmp_path, "docs/decisions/mechanism-y-adoption.md", _DECISION_Y)

    result = cx.context("How does mechanism Y work?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/architecture/mechanism-y.md"


def test_case_b_decision_query_prefers_the_decision_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/mechanism-y.md", _DEFINITION_Y)
    _seed(cx, tmp_path, "docs/decisions/mechanism-y-adoption.md", _DECISION_Y)

    result = cx.context("What was decided about mechanism Y?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/decisions/mechanism-y-adoption.md"


# ---------------------------------------------------------------------------
# CASE C -- historical/original description vs. current state.
# ---------------------------------------------------------------------------

_ORIGINAL_W = """# System W Original Design

System W was originally designed as a dual-path system: a write path
that accepted incoming records for system W and a read path that served
system W queries from a separate replica. System W's write path for
records handled validation, normalization and indexing before a record
became visible to the read path of system W, and system W's design
document describes each of these stages of system W in detail.
"""

_CURRENT_W = """# System W State

The current state of system W is read-only: the write path described in
the original design was removed, and system W now only serves queries
from the latest snapshot.
"""


def test_case_c_broad_query_prefers_the_design_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/system-w-design.md", _ORIGINAL_W)
    _seed(cx, tmp_path, "docs/state/system-w-current.md", _CURRENT_W)

    result = cx.context("What is system W?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/architecture/system-w-design.md"


def test_case_c_current_state_query_prefers_the_current_state_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/system-w-design.md", _ORIGINAL_W)
    _seed(cx, tmp_path, "docs/state/system-w-current.md", _CURRENT_W)

    result = cx.context("What is the current state of system W?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/state/system-w-current.md"


# ---------------------------------------------------------------------------
# CASE D -- description vs. evaluation result.
# ---------------------------------------------------------------------------

_DESCRIPTION_Z = """# System Z

System Z is a caching layer that sits in front of the primary database
for system Z's callers and serves reads from an in-memory store when
system Z has a warm entry, falling back to the primary database when
system Z has a cold entry. System Z invalidates a cached entry whenever
the corresponding row in the primary database changes, which is the
property system Z relies on to stay consistent with system Z's backing
store.
"""

_EVALUATION_Z = """# Evaluation

The evaluation set out to find how system Z performs under load, and
found that system Z reduces read latency by 40 percent but increases
memory usage significantly under a cold cache.
"""


def test_case_d_broad_query_prefers_the_description_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/system-z.md", _DESCRIPTION_Z)
    _seed(cx, tmp_path, "docs/research/system-z-evaluation.md", _EVALUATION_Z)

    result = cx.context("What is system Z?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/architecture/system-z.md"


def test_case_d_evaluation_query_prefers_the_evaluation_document(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/system-z.md", _DESCRIPTION_Z)
    _seed(cx, tmp_path, "docs/research/system-z-evaluation.md", _EVALUATION_Z)

    result = cx.context("What did the evaluation find about system Z?", budget=100000)

    paths = _selected_source_paths(result)
    assert paths and paths[0] == "docs/research/system-z-evaluation.md"


# ---------------------------------------------------------------------------
# Admission gate must remain unaffected by intent weighting: a document
# that merely CONTAINS an intent word, but is otherwise off-topic, must
# still fail admission -- weighting only reorders already-admitted
# candidates, it can never let an irrelevant document in.
# ---------------------------------------------------------------------------

_UNRELATED_WITH_INTENT_WORD = """# Notes

The evaluation of the annual facilities schedule concluded last week
with no changes recommended.
"""


def test_intent_word_alone_does_not_bypass_admission(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/system-z.md", _DESCRIPTION_Z)
    _seed(cx, tmp_path, "docs/notes/facilities.md", _UNRELATED_WITH_INTENT_WORD)

    result = cx.context("What did the evaluation find about system Z?", budget=100000)

    paths = _selected_source_paths(result)
    assert "docs/notes/facilities.md" not in paths


# ---------------------------------------------------------------------------
# Authority invariant: intent-weighted ranking never changes what a
# PROJECT EVIDENCE item claims to be. Source != Evidence != Memory holds
# regardless of which document an intent-bearing query happens to prefer.
# ---------------------------------------------------------------------------


def test_authority_invariant_holds_regardless_of_intent_match(tmp_path):
    cx = _workspace(tmp_path)
    _seed(cx, tmp_path, "docs/architecture/capability-x.md", _DEFINITION_X)
    _seed(cx, tmp_path, "docs/research/capability-x-findings.md", _CONCLUSION_X)

    result = cx.context(
        "What did the later evaluation conclude about how capability X performs under load?",
        budget=100000,
    )

    for item in _project_evidence_items(result):
        assert item.kind == "evidence"
        assert item.authority == "document_observation"
