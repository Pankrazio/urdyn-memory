"""Retrieval chunks: a derived, never-persisted view of a Source's
current-observation `Evidence.content`, split into candidate segments a
budgeted `compile_context` can admit or reject individually.

Seeded Sources participate in `context()` retrieval, but their canonical
Evidence can contain an entire document. Representing each candidate as the
whole document, verbatim, would be correct for the canonical record -- Evidence
holds the document's full text, and nothing here ever changes that -- but it
made every real, multi-paragraph project document compete for budget as one
indivisible unit. A document a few KB long, easily admitted as task-relevant
by `_preflight.evidence_is_relevant`, costs more budget on its own than most
`context()` calls have room for, and `compile_context`'s admission is a
deterministic PREFIX scan (A29.1) that stops at the first candidate that
does not fit -- so several oversized, genuinely relevant documents can all
be "omitted for budget" while the compiled context stays empty.

CANONICAL EVIDENCE != RETRIEVAL CHUNK. `EvidenceChunk` is not a new kind of
evidence, not a new table, and not a new source of authority: it is a pure,
deterministic slice of `Evidence.content`, computed FRESH every time
`chunk_evidence` is called and never written to storage. This is what makes
it trivially "rebuildable" -- there is nothing cached to go stale, so a
Source update (a new current observation, a new `evidence_id`) can never
leak a chunk of the SUPERSEDED text: whichever Evidence the caller passes in
is the only text ever chunked.

`chunk_index`/`chunk_count`/`start`/`end` exist for provenance and audit,
not identity: a chunk is never looked up on its own by any id of its own
(see `_context.ContextItem.entity_id`, which is always the parent
Evidence's real, canonical id -- multiple chunks of the same Evidence
render under the SAME entity_id, distinguished only by
`ContextItem.chunk_index`/`chunk_count` in the rendering). `start`/`end`
are the exact character offsets into the parent `Evidence.content` this
chunk's `text` came from, so `chunk.text == evidence.content[chunk.start:chunk.end]`
always holds -- the chunk is a literal substring, never reformatted,
summarized, or paraphrased.

RANKING is a separate, deliberately excluded concern: this module only
splits text into candidate segments and can order them by lexical overlap
with a query (`rank_evidence_chunks`), reusing the SAME deterministic
lexical primitive (`_relevance.tokens`) every other relevance channel in
this codebase already uses. It does not decide whether the PARENT document
is relevant at all -- that gate is unchanged, still
`_preflight.evidence_is_relevant`, still evaluated over the WHOLE document
via all three existing channels (lexical majority, FTS, semantic). No new
embedding backend, no new persisted index: chunk ranking only ever narrows
an ALREADY-admitted document down to its most relevant segments.

INTENT-WEIGHTED OVERLAP (A54). A flat "count of shared tokens" score
structurally favors a long, topically-dense document over a short,
precise one that directly answers the query: a document that paraphrases
"capability X" ten different ways racks up shared generic vocabulary
(the topic's own words) even when it never actually contains the
CONCLUSION, DECISION or EVALUATION the query is asking for, while the
short document that states that conclusion in one sentence shares fewer
tokens overall and loses the ranking despite being the right answer. This
was reproduced during A54 dogfooding review: a dense definition
out-scored a precise later-research conclusion on a query that explicitly
asked what that research concluded.

`_INTENT_TERMS` is a small, fixed vocabulary of words that signal WHAT
KIND of answer a query wants (concluding, deciding, evaluating, a result,
a current/latest state) rather than WHAT TOPIC it is about. A shared
token drawn from this set counts `_INTENT_TERM_WEIGHT` times instead of
once -- still an integer, still deterministic, still requiring the token
to appear in both the query AND the candidate (an intent word absent from
the query never boosts anything). This only re-orders candidates that
ADMISSION already let through; no threshold here is an admission gate,
and a lexical-only workspace (no semantic extra installed) ranks exactly
as correctly as one with it, since this is pure vocabulary weighting with
no embedding involved.

TOPICAL CO-OCCURRENCE REQUIREMENT (A54.1). Real-corpus dogfooding
(replaying a historical failing query from A54's own fix review) falsified
the plain version above: a completely unrelated document that happened to
contain the single word "conclude" in an aside about something else
entirely still got its lone shared token boosted 3x by
`_INTENT_TERM_WEIGHT`, with ZERO other shared vocabulary -- and that
inflated score was enough for `_preflight.minimum_sufficient_project_evidence`
to treat it as worth keeping, purely because nothing else in the corpus
happened to contain that exact word. Confirmed by grep: the word never
appears anywhere in either of the two documents that actually discuss the
query's real topic.

Growing `_INTENT_TERMS` to cover more reporting verbs (establish,
determine, assess, ...) cannot fix this: it was independently confirmed
that the ACTUAL relevant documents in that corpus use no reporting verb
at all for their conclusion -- they simply state it as prose. The bug was
never "the wrong word was on the list", it was "a lone intent-word match,
with no accompanying topical evidence that the candidate is even about
the right SUBJECT, was treated as meaningful signal at all".

The fix: `_weighted_overlap` now applies `_INTENT_TERM_WEIGHT` only when
the shared tokens include at least one TOPICAL (non-intent) token too --
i.e. only when there is independent evidence the candidate is about the
query's subject in the first place. An intent word matched in isolation
now counts as an ordinary shared token (weight 1, the same as any other
single incidental word, and the same as this codebase's behavior before
A54 introduced the weighting at all) -- never specially promoted on its
own. This keeps every A54 CASE A-D fixture unchanged (each of those
documents shares real topical vocabulary WITH its intent word, by
construction), while removing the exact mechanism that promoted a
topically-empty match. No new vocabulary, no per-query classification,
no embedding: intent cues remain a REFINEMENT applied on top of
independently-established topical relevance, never a substitute for it.
"""

from __future__ import annotations

import dataclasses
import re

from ._evidence import Evidence
from ._relevance import tokens as _tokens

# Chosen so that several relevant chunks from different documents can
# coexist under a typical `context()` budget alongside
# CONSTRAINTS/OPEN RISKS/LESSONS/DECISIONS,
# while staying large enough that a chunk is still a coherent, readable
# unit of a real document (a paragraph or a small group of them), not a
# sentence fragment. Not derived from `DEFAULT_CONTEXT_BUDGET` (4000) in
# `_context.py`: chunk size is a property of how documents are split, budget
# is a property of how much total context an agent wants -- conflating the
# two would make one control tune the other by accident.
DEFAULT_CHUNK_MAX_CHARS = 1200

_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n")

# [A54] Words that name WHAT KIND of answer a query wants, not what topic
# it is about -- see the module docstring's "INTENT-WEIGHTED OVERLAP"
# section. Deliberately small, fixed, and generic across domains: no word
# tied to any one project, vendor, or dogfood corpus. Only inflection
# forms of each intent verb/noun are included (not synonyms in general),
# to keep the set auditable at a glance.
_INTENT_TERMS = frozenset(
    {
        "conclude", "concludes", "concluded", "concluding", "conclusion", "conclusions",
        "decide", "decides", "decided", "deciding", "decision", "decisions",
        "evaluate", "evaluates", "evaluated", "evaluating", "evaluation", "evaluations",
        "result", "results", "finding", "findings", "found",
        "current", "currently", "latest", "recent", "recently",
        "definition", "definitions", "define", "defines", "defined", "defining",
    }
)

# Integer, not a tuned float: a shared intent token counts as this many
# ordinary shared tokens. Large enough to move a short, precise,
# intent-matching document ahead of a longer topically-dense one in the
# A54 reproduction (see module docstring), without being so large that a
# single incidental intent-word match can outweigh a genuinely large
# topical overlap on its own.
_INTENT_TERM_WEIGHT = 3


def _weighted_overlap(query_tokens: frozenset[str], candidate_tokens: set[str]) -> int:
    shared = query_tokens & candidate_tokens
    intent_shared = shared & _INTENT_TERMS
    topical_shared = shared - _INTENT_TERMS
    if not topical_shared:
        # [A54.1] No independent evidence this candidate is even about
        # the query's subject -- an intent word matched in isolation is
        # an ordinary shared token, not a signal of relevance on its own.
        # See the module docstring's "TOPICAL CO-OCCURRENCE REQUIREMENT".
        return len(shared)
    return len(shared) + (_INTENT_TERM_WEIGHT - 1) * len(intent_shared)


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """One deterministic, literal slice of a parent Evidence's `content`.

    `text` is always exactly `evidence.content[start:end]` -- see the
    module docstring on why this is a substring, never a rewrite. A
    single-chunk document (`chunk_count == 1`) always has `text` equal to
    the ENTIRE `evidence.content`, byte for byte: `chunk_text_spans`
    special-cases text under `max_chars` to skip paragraph splitting
    entirely, so a small seeded document renders identically to the
    unchunked representation.
    """

    evidence_id: str
    chunk_index: int
    chunk_count: int
    start: int
    end: int
    text: str


def chunk_text_spans(text: str, *, max_chars: int = DEFAULT_CHUNK_MAX_CHARS) -> list[tuple[int, int]]:
    """Deterministic `(start, end)` character-offset spans covering `text`,
    each at most `max_chars` long.

    Paragraph-aware: text under `max_chars` is returned as a single span
    covering the whole thing (no splitting at all, so a short document is
    never touched); longer text is split on blank-line paragraph
    boundaries, and each paragraph becomes its OWN span -- never merged
    with a neighboring paragraph, even if both are tiny. This is
    deliberate, not the simplest way to minimize chunk count: a chunk is
    the unit `rank_evidence_chunks` admits or drops as a whole, so merging
    a genuinely relevant paragraph with an adjacent irrelevant one would
    force them to share one admission decision and let irrelevant text
    ride along inside a "relevant" candidate, silently spending its
    budget. Splitting on the document's OWN paragraph boundaries is
    respecting structure the author already chose, not artificial
    fragmentation. A single paragraph that alone exceeds `max_chars` (a
    huge unbroken block, a code fence) is hard-split on its own,
    preferring the last whitespace at or before the limit so a chunk
    boundary does not usually land mid-word.

    No overlap between chunks: each character of `text` belongs to
    exactly one span. Overlap would let the same sentence compete for
    budget twice under two different chunk identities, which is not
    needed for splitting an oversized candidate into admissible pieces
    and would only complicate deduplication and ranking for no
    measurable retrieval benefit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    for start, end in _paragraph_spans(text):
        if end - start > max_chars:
            spans.extend(_hard_split_span(text, start, end, max_chars))
        else:
            spans.append((start, end))
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Non-blank paragraph spans of `text`, in order, skipping the
    blank-line runs between them. Never empty for non-empty `text`."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _PARAGRAPH_SEPARATOR.finditer(text):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or [(0, len(text))]


def _hard_split_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split one oversized paragraph span into `<= max_chars` pieces,
    preferring to break at whitespace so a boundary does not usually fall
    mid-word. Always makes forward progress even if no whitespace is
    found (`break_at` falls back to the hard `limit`)."""
    spans: list[tuple[int, int]] = []
    pos = start
    while end - pos > max_chars:
        limit = pos + max_chars
        break_at = text.rfind(" ", pos + 1, limit)
        if break_at == -1:
            break_at = text.rfind("\n", pos + 1, limit)
        if break_at == -1:
            break_at = limit
        spans.append((pos, break_at))
        pos = break_at
        while pos < end and text[pos] in " \n":
            pos += 1
    spans.append((pos, end))
    return spans


def chunk_evidence(evidence: Evidence, *, max_chars: int = DEFAULT_CHUNK_MAX_CHARS) -> tuple[EvidenceChunk, ...]:
    """The full, ordered set of chunks for one Evidence's `content`, fresh
    every call -- see the module docstring on why nothing here is cached
    or persisted."""
    spans = chunk_text_spans(evidence.content, max_chars=max_chars)
    count = len(spans)
    return tuple(
        EvidenceChunk(
            evidence_id=evidence.evidence_id,
            chunk_index=index,
            chunk_count=count,
            start=start,
            end=end,
            text=evidence.content[start:end],
        )
        for index, (start, end) in enumerate(spans)
    )


_CHUNK_SEMANTIC_ID_SEPARATOR = "#"


def chunk_semantic_id(evidence_id: str, chunk_index: int) -> str:
    """[A54.3] Deterministic, derived identity for one chunk's semantic
    (embedding) representation -- a plain, reconstructible function of
    the same `(evidence_id, chunk_index)` pair `EvidenceChunk` already
    carries, never a row id and never a path. `evidence_id` is a
    `uuid.uuid4().hex` string (see `_workspace.py`), which never contains
    `_CHUNK_SEMANTIC_ID_SEPARATOR`, so this round-trips exactly through
    `parse_chunk_semantic_id`.

    This id names a SEMANTIC INDEX row only -- see `_semantic_store.py` --
    never a canonical record: a chunk has no identity or authority of its
    own (module docstring), and nothing outside the semantic pool ever
    looks this id up."""
    return f"{evidence_id}{_CHUNK_SEMANTIC_ID_SEPARATOR}{chunk_index}"


def parse_chunk_semantic_id(semantic_id: str) -> tuple[str, int]:
    """Inverse of `chunk_semantic_id`: recovers the parent Evidence's own
    canonical id and the chunk's index within it. `rpartition` (not
    `split`) on purpose: robust even in the purely theoretical case of an
    id containing the separator, since only the LAST occurrence can be
    the chunk-index delimiter this function itself always writes."""
    evidence_id, _, index = semantic_id.rpartition(_CHUNK_SEMANTIC_ID_SEPARATOR)
    return evidence_id, int(index)


def score_evidence_chunks(
    query_tokens: frozenset[str], chunks: tuple[EvidenceChunk, ...]
) -> tuple[tuple[int, EvidenceChunk], ...]:
    """`rank_evidence_chunks`, but KEEPING each surviving chunk's own
    lexical score instead of discarding it.

    The score was always computed here -- ranking is defined by it -- and
    was simply thrown away at the return statement, which left every
    caller downstream with no way to compare a chunk of one document
    against a chunk of another. Exposing the number that already decides
    WITHIN-document order is not a new channel: it is the same integer,
    surfaced instead of dropped, so cross-document ordering can use it
    too (see `_preflight.ordered_project_evidence`).

    The score itself is `_weighted_overlap` (see the module docstring's
    "INTENT-WEIGHTED OVERLAP" section, A54): plain shared-token count,
    except a shared token drawn from `_INTENT_TERMS` counts extra. Zero
    exactly when there is no overlap at all, same as the plain count did,
    so every "zero overlap" branch below is unaffected by the weighting.

    Ordering, the zero-overlap leading-chunk fallback, and the
    single-chunk pass-through are all exactly `rank_evidence_chunks`'s,
    which is now a thin wrapper over this function -- there is one
    implementation of chunk ranking, not two that could drift.
    """
    if len(chunks) <= 1:
        return tuple((_weighted_overlap(query_tokens, _tokens(chunk.text)), chunk) for chunk in chunks)
    scored = [(_weighted_overlap(query_tokens, _tokens(chunk.text)), chunk) for chunk in chunks]
    if all(score == 0 for score, _ in scored):
        return ((0, chunks[0]),)
    relevant = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: (-pair[0], pair[1].chunk_index))
    return tuple(relevant)


def rank_evidence_chunks(
    query_tokens: frozenset[str], chunks: tuple[EvidenceChunk, ...]
) -> tuple[EvidenceChunk, ...]:
    """Order an ALREADY task-relevant document's chunks by their OWN
    lexical overlap with `query_tokens`, dropping chunks with zero
    overlap -- this is what keeps an irrelevant section of an otherwise
    relevant document from ever being selected, regardless of budget.

    If EVERY chunk scores zero (the parent document was admitted only
    through FTS or semantic similarity, channels that do not necessarily
    concentrate their signal in any one paragraph's raw vocabulary), falls
    back to the single leading chunk (`chunk_index == 0`) rather than
    returning nothing: the document was already judged relevant by the
    caller, and the leading chunk is the same text a semantic embedding
    (truncated to its own leading tokens, see `_semantic.py`) actually
    scored in the first place.

    A single-chunk document is returned unchanged: no ranking to do, and
    no risk of a document that fits its own budget being second-guessed
    out of the pool by its own lexical score.
    """
    return tuple(chunk for _score, chunk in score_evidence_chunks(query_tokens, chunks))
