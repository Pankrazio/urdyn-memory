"""[A52.1] Retrieval chunks: a DERIVED, never-persisted view of a Source's
current-observation `Evidence.content`, split into candidate segments a
budgeted `compile_context` can admit or reject individually.

WHY THIS EXISTS. A52 made a seeded Source's current Evidence participate in
`context()` retrieval, but represented each candidate as the ENTIRE document,
verbatim, one candidate per Source (see `_context.py`'s module docstring:
"never truncated"). That is correct for the CANONICAL record -- Evidence
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
"""

from __future__ import annotations

import dataclasses
import re

from ._evidence import Evidence
from ._relevance import tokens as _tokens

# [A52.1] Chosen so that several relevant chunks from DIFFERENT documents
# can coexist under a typical `context()` budget (the reported dogfood
# session used 6000) alongside CONSTRAINTS/OPEN RISKS/LESSONS/DECISIONS,
# while staying large enough that a chunk is still a coherent, readable
# unit of a real document (a paragraph or a small group of them), not a
# sentence fragment. Not derived from `DEFAULT_CONTEXT_BUDGET` (4000) in
# `_context.py`: chunk size is a property of how documents are split, budget
# is a property of how much total context an agent wants -- conflating the
# two would make one control tune the other by accident.
DEFAULT_CHUNK_MAX_CHARS = 1200

_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n")


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """One deterministic, literal slice of a parent Evidence's `content`.

    `text` is always exactly `evidence.content[start:end]` -- see the
    module docstring on why this is a substring, never a rewrite. A
    single-chunk document (`chunk_count == 1`) always has `text` equal to
    the ENTIRE `evidence.content`, byte for byte: `chunk_text_spans`
    special-cases text under `max_chars` to skip paragraph splitting
    entirely, so a small seeded document renders identically to how A52
    already rendered it.
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
    if len(chunks) <= 1:
        return chunks
    scored = [(len(query_tokens & _tokens(chunk.text)), chunk) for chunk in chunks]
    if all(score == 0 for score, _ in scored):
        return (chunks[0],)
    relevant = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: (-pair[0], pair[1].chunk_index))
    return tuple(chunk for _score, chunk in relevant)
