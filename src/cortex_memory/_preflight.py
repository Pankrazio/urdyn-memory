"""Preflight: selecting prior experience relevant to an upcoming task.

Deterministic, not a search engine and not the Context Compiler: no
score is ever returned to a caller, no semantic similarity, no
summarization, no token budgeting. Two independent, deterministic
retrieval channels feed the same admission decision (see
`_retrieval.py` for why): the structured lexical majority rule
(`_relevance.is_relevant`), and FTS5/BM25 candidate widening for
queries a natural phrasing dilutes past that rule's threshold. Either
channel admitting a candidate is enough; a candidate need not clear
both.

Relevance is decided two ways:

1. Direct vocabulary overlap with the task (an attempt's own task/approach
   text, a lesson's or root cause's own content), through either channel.
2. Shared provenance: a root cause or lesson that cites the same Evidence
   as an attempt already judged relevant is included even if its own
   wording shares no vocabulary with the task at all. A root cause like
   "the old token was reused after rotation" has nothing lexically in
   common with "authentication refresh logic" — it is relevant because it
   explains the *same observed failure*, which Evidence is what actually
   connects it to the matching attempt.

A third, OPTIONAL channel (A7.4) can widen candidate recall further: a
calibrated semantic-similarity admission (`_semantic.py`, brute-force
cosine over static embeddings), passed in here as
`attempt_semantic_admitted`/`memory_semantic_admitted` -- pre-computed
sets of at most one id each, already filtered through the same
abstention policy that decides "is this candidate semantically relevant
enough" (see `_semantic.py`'s module docstring). This module never
computes or sees a similarity score itself: it only checks id membership,
exactly like the FTS channel above. Defaults to empty, so a workspace
without the `[semantic]` extra installed (or without `cortex semantic
setup` having been run) behaves exactly as it did before A7.4.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ._attempt import OUTCOME_FAILED, OUTCOME_SUCCEEDED, Attempt
from ._evidence import RECOMMENDED_VALIDATION_EVIDENCE_KINDS, Evidence
from ._memory import Memory
from ._relevance import attempt_search_text as _attempt_search_text
from ._relevance import is_relevant as _is_relevant
from ._relevance import memory_search_text as _memory_search_text
from ._relevance import tokens as _tokens
from ._retrieval import fts_admitted_ids as _fts_admitted_ids

_VALIDATION_EVIDENCE_KINDS = RECOMMENDED_VALIDATION_EVIDENCE_KINDS


@dataclasses.dataclass(frozen=True, slots=True)
class Preflight:
    """Prior experience relevant to a task, grouped by what it means for
    an agent about to start work. A category is empty, not absent, when
    Cortex has nothing relevant on record for it.

    `invariants` (A9.1) is the one exception to "relevant to a task": it
    is every CURRENT project-wide operational invariant (`Memory` of kind
    `invariant`), included unconditionally regardless of `task`'s own
    wording. An invariant is not a piece of experience that may or may
    not bear on this particular task the way a lesson or root cause
    does -- it is a constraint the project holds at all times, so it
    bypasses the lexical/FTS/semantic admission channels entirely and is
    filtered only by current state (superseded invariants are excluded,
    exactly like every other kind). Deliberately placed last and
    defaulted to `()` to keep the pre-A9.1 four-field shape a valid,
    unbroken prefix of this one.

    `open_invalidations` (A11.3) is every CURRENT `Memory` of kind
    `invalidation` relevant to `task`, using the same relevance-matching
    machinery as `root_causes` and `verified_lessons`. Unlike `invariants`,
    an invalidation is not project-wide by default and does not bypass
    relevance matching. It answers a question none of the other fields
    can: not "what do we currently know" but "what prior knowledge had
    its current authority explicitly withdrawn without a replacement".
    An agent must not confuse an empty `root_causes`/`verified_lessons`/
    `invariants` (Cortex has nothing on record) with a non-empty
    `open_invalidations` (Cortex had something on record and explicitly
    stopped trusting it). Never contains the Memory that was invalidated
    -- only the invalidation itself. Deliberately placed last and
    defaulted to `()` for the same backward-compatibility reason as
    `invariants`.
    """

    task: str
    known_failures: tuple[Attempt, ...]
    root_causes: tuple[Memory, ...]
    verified_lessons: tuple[Memory, ...]
    recommended_validation: tuple[Evidence, ...]
    invariants: tuple[Memory, ...] = ()
    open_invalidations: tuple[Memory, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.known_failures
            or self.root_causes
            or self.verified_lessons
            or self.recommended_validation
            or self.invariants
            or self.open_invalidations
        )


def build_preflight(
    task: str,
    *,
    attempts: list[Attempt],
    root_cause_memories: list[Memory],
    verified_lesson_memories: list[Memory],
    evidence_lookup: Callable[[str], Evidence],
    attempt_fts_candidates: list[tuple[str, str]] = (),
    memory_fts_candidates: list[tuple[str, str]] = (),
    attempt_semantic_admitted: frozenset[str] = frozenset(),
    memory_semantic_admitted: frozenset[str] = frozenset(),
    invariant_memories: list[Memory] = (),
    invalidation_memories: list[Memory] = (),
) -> Preflight:
    """Pure selection logic, operating on data already fetched from
    storage. Takes no dependency on SQLite so it can be tested and
    reasoned about independently of the storage boundary.

    `attempt_fts_candidates`/`memory_fts_candidates` are `(entity_id,
    text)` pairs already ranked best-first by BM25 for this `task` (see
    `MemoryStore.search_candidates`), or `()` if FTS5 is unavailable --
    channel B then contributes nothing and matching falls back to the
    lexical channel alone, exactly as before A7. Only ids also present
    in `attempts`/`root_cause_memories`/`verified_lesson_memories` can
    ever be admitted through them: those lists are already filtered to
    current state and, for lessons, to `verified` -- a candidate the FTS
    channel ranks highly but that was excluded from those lists (a
    superseded memory, an unverified lesson) is never even considered,
    the same way a lexical match on such an item already isn't.

    `evidence_lookup` is expected to raise if given an id that does not
    resolve: every id passed to it here came from an attempt's or
    memory's own `evidence_ids`, so a miss means the persisted provenance
    link is dangling, not that the caller asked about something that
    was never expected to exist.

    `invariant_memories` (A9.1) bypasses relevance matching entirely: the
    caller is expected to have already filtered it to current, kind
    `invariant` memories (see `Cortex.preflight`), and every one of them
    is included in the result regardless of `task`. No lexical/FTS/
    semantic channel is consulted for this field.

    `invalidation_memories` (A11.3) is expected to already be filtered to
    current, kind `invalidation` memories, exactly like
    `root_cause_memories`/`verified_lesson_memories`. Unlike
    `invariant_memories`, it goes through the SAME `_memory_matches`
    relevance gate as root causes/lessons -- an invalidation is not
    project-wide by default. `memory_semantic_admitted` is shared with
    root causes/lessons only insofar as the caller chooses to union
    independently-computed admission sets before calling this function;
    this function itself does not know or care where that set's members
    came from, so it cannot introduce cross-pool competition on its own.
    """
    query_tokens = frozenset(_tokens(task))
    attempt_admitted = _fts_admitted_ids(query_tokens, list(attempt_fts_candidates))
    memory_admitted = _fts_admitted_ids(query_tokens, list(memory_fts_candidates))

    def _attempt_matches(attempt: Attempt) -> bool:
        if _is_relevant(query_tokens, _attempt_search_text(attempt.task, attempt.approach)):
            return True
        if attempt.attempt_id in attempt_admitted:
            return True
        return attempt.attempt_id in attempt_semantic_admitted

    known_failures = tuple(a for a in attempts if a.outcome == OUTCOME_FAILED and _attempt_matches(a))
    relevant_successes = tuple(a for a in attempts if a.outcome == OUTCOME_SUCCEEDED and _attempt_matches(a))

    relevant_attempt_evidence_ids = frozenset(
        evidence_id for attempt in (*known_failures, *relevant_successes) for evidence_id in attempt.evidence_ids
    )

    def _memory_matches(memory: Memory) -> bool:
        if _is_relevant(query_tokens, _memory_search_text(memory.content)):
            return True
        if memory.memory_id in memory_admitted:
            return True
        if memory.memory_id in memory_semantic_admitted:
            return True
        return not relevant_attempt_evidence_ids.isdisjoint(memory.evidence_ids)

    root_causes = tuple(memory for memory in root_cause_memories if _memory_matches(memory))
    verified_lessons = tuple(memory for memory in verified_lesson_memories if _memory_matches(memory))
    open_invalidations = tuple(memory for memory in invalidation_memories if _memory_matches(memory))

    candidate_evidence_ids: list[str] = []
    for source in (*verified_lessons, *relevant_successes):
        for evidence_id in source.evidence_ids:
            if evidence_id not in candidate_evidence_ids:
                candidate_evidence_ids.append(evidence_id)

    recommended_validation = []
    for evidence_id in candidate_evidence_ids:
        evidence = evidence_lookup(evidence_id)
        if evidence.kind in _VALIDATION_EVIDENCE_KINDS:
            recommended_validation.append(evidence)

    return Preflight(
        task=task,
        known_failures=known_failures,
        root_causes=root_causes,
        verified_lessons=verified_lessons,
        recommended_validation=tuple(recommended_validation),
        invariants=tuple(invariant_memories),
        open_invalidations=open_invalidations,
    )
