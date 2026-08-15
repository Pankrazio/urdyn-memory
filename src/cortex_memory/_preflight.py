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
id sets, already filtered through whichever abstention policy decides
"is this candidate semantically relevant enough" for the pool they came
from (see `_semantic.py`'s module docstring). Historically at most one
id each; since A23.1 the caller's verified-lesson pool contributes a
BOUNDED SET instead of a single winner, which changes nothing here --
this module has never done anything with these but test id membership,
and deliberately does not know how many ids any pool decided to admit
or why. This module never
computes or sees a similarity score itself: it only checks id membership,
exactly like the FTS channel above. Defaults to empty, so a workspace
without the `[semantic]` extra installed (or without `cortex semantic
setup` having been run) behaves exactly as it did before A7.4.

`open_conflicts` (A14.1) is a fourth, orthogonal signal, not a fourth
relevance channel: it never decides whether a Memory is shown, only
whether Cortex additionally discloses that a shown Memory is party to an
open canonical `Conflict` (see `_conflict.py`). See `PreflightConflict`
and `Preflight.open_conflicts` for the relevance rule and why it
deliberately reuses this module's existing machinery instead of adding a
new semantic pool.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

from ._attempt import OUTCOME_FAILED, OUTCOME_SUCCEEDED, Attempt
from ._conflict import Conflict
from ._errors import CortexStorageError
from ._evidence import RECOMMENDED_VALIDATION_EVIDENCE_KINDS, Evidence
from ._memory import Memory
from ._relevance import attempt_search_text as _attempt_search_text
from ._relevance import is_relevant as _is_relevant
from ._relevance import memory_search_text as _memory_search_text
from ._relevance import tokens as _tokens
from ._retrieval import fts_admitted_ids as _fts_admitted_ids

_VALIDATION_EVIDENCE_KINDS = RECOMMENDED_VALIDATION_EVIDENCE_KINDS


@dataclasses.dataclass(frozen=True, slots=True)
class PreflightConflict:
    """A derived, read-only pairing of a canonical `Conflict` with the two
    Memories it names, scoped to a single `Preflight` result.

    NOT a canonical primitive: `Conflict` itself (see `_conflict.py`)
    deliberately carries only `memory_ids`/`recorded_at`, because the
    conflict relation is a fact about two ids, not about their content.
    But a `Preflight` result is meant to be self-sufficient -- a consumer
    that already has one should not need a second lookup (`state()`,
    `timeline()`, or a new `get_memory()`) just to read what the two
    participants actually say. This type closes exactly that gap, for
    THIS result only: it is never persisted, never returned by any other
    API, and changes nothing about what `Cortex.conflicts()`/
    `Cortex.open_conflicts()` return.

    `memories` holds the SAME `Memory` objects admitted elsewhere in this
    `Preflight` when a participant also qualifies for `root_causes`/
    `verified_lessons`/`open_invalidations` -- never a copy, never
    derived or summarized text. Always ordered like `conflict.memory_ids`:
    `memories[0]` is the Memory named by `memory_ids[0]`, `memories[1]`
    by `memory_ids[1]`, regardless of which order the two Memories were
    created or declared in (see `_conflict.canonical_pair`).
    """

    conflict: Conflict
    memories: tuple[Memory, Memory]


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

    `pending` (A22.1) is every CURRENT `Memory` of kind `pending`
    relevant to `task`, using the same relevance-matching machinery as
    `root_causes`/`verified_lessons`/`open_invalidations`. Like
    `open_invalidations` and UNLIKE `invariants`, unfinished operational
    work is not project-wide by default: a pending item is surfaced only
    when it is relevant to THIS task, never as an unconditional dump of
    everything still open. It answers yet another question none of the
    other fields can: not "what do we currently know" and not "what did
    we stop trusting", but "what work is still open that bears on what
    you are about to do". A pending is NOT elevated in authority by its
    kind -- it keeps whatever `epistemic_state` it was recorded with
    (`user_asserted` by default), and being shown here is a statement
    about relevance, never about truth. Closure needs no new mechanism:
    a completed or cancelled pending is superseded by whatever Memory
    records the resolution (see `_memory.py`'s KIND_PENDING note), which
    makes it non-current, which removes it from this field for exactly
    the same reason it removes it from every other one. Deliberately
    placed last and defaulted to `()` for the same
    backward-compatibility reason as `invariants`/`open_invalidations`/
    `open_conflicts`.

    `open_conflicts` (A14.1) is every OPEN canonical `Conflict` (see
    `Cortex.open_conflicts()` -- both participants current) relevant to
    `task`, paired with the two Memories it names as a `PreflightConflict`
    (see that class for why). A participant is relevant under the SAME
    rule used for `root_causes`/`verified_lessons`/`open_invalidations`
    (lexical, FTS, semantic, or evidence-provenance rescue), OR because
    it is already being shown in one of those three fields -- deliberately
    NOT `invariants`, whose always-include rule would otherwise make
    every conflict touching an invariant appear regardless of `task`. A
    conflict is included if EITHER participant is relevant -- one-sided
    sufficiency, not both: a Memory Cortex is about to show as
    individually authoritative must never be shown without also
    surfacing that Cortex knows it is contradicted, even if the other
    side of that contradiction says nothing the task's own wording
    matches. This never changes `verified`/epistemic status, never
    resolves which side is correct, and never removes anything from
    `root_causes`/`verified_lessons`/`open_invalidations` -- it is a
    strictly additive signal alongside them. Deliberately placed last and
    defaulted to `()` for the same backward-compatibility reason as
    `invariants`/`open_invalidations`.
    """

    task: str
    known_failures: tuple[Attempt, ...]
    root_causes: tuple[Memory, ...]
    verified_lessons: tuple[Memory, ...]
    recommended_validation: tuple[Evidence, ...]
    invariants: tuple[Memory, ...] = ()
    open_invalidations: tuple[Memory, ...] = ()
    open_conflicts: tuple[PreflightConflict, ...] = ()
    pending: tuple[Memory, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.known_failures
            or self.root_causes
            or self.verified_lessons
            or self.recommended_validation
            or self.invariants
            or self.open_invalidations
            or self.open_conflicts
            or self.pending
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
    pending_memories: list[Memory] = (),
    open_conflicts: list[Conflict] = (),
    conflict_participants: Mapping[str, Memory] = {},  # noqa: B006 -- never mutated, see below
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

    `pending_memories` (A22.1) is expected to already be filtered to
    current, kind `pending` memories, exactly like
    `invalidation_memories`, and goes through the SAME `_memory_matches`
    relevance gate -- no new threshold, no pending-specific score, no
    boost, and specifically no "return everything still open" shortcut.
    A22 measured that the existing gate already both admits the relevant
    pending and rejects the unrelated one for the same task, so the only
    thing missing was this parameter. The caller is expected to have
    computed whatever semantic admission it wants for pending in its OWN
    disjoint pool (see `Cortex.preflight`); this function only checks id
    membership in `memory_semantic_admitted` and therefore cannot itself
    create competition between pending and any other category.

    `open_conflicts` (A14.1) is expected to already be filtered to OPEN
    conflicts (see `Cortex.open_conflicts()`) -- this function has no
    concept of "current" and does not re-derive it. `conflict_participants`
    must map every id named by every conflict in `open_conflicts` to its
    (current) `Memory`; a miss raises `CortexStorageError` rather than
    silently dropping the conflict (see the fail-closed check below).
    Deliberately takes NO new semantic admission set for conflicts: a
    participant inherits relevance from `root_causes`/`verified_lessons`/
    `open_invalidations` if it is already in one of them (which already
    reflects whatever the semantic channel decided for that field), and
    otherwise falls back to the same lexical/FTS/semantic/evidence-rescue
    `_memory_matches` gate used everywhere else in this function -- never
    a dedicated conflict-only semantic pool (see A14.0.1's
    cross-conflict-suppression finding for why one was rejected).
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
    pending = tuple(memory for memory in pending_memories if _memory_matches(memory))

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

    # [A14.1] Fail-closed integrity: every participant of every OPEN
    # conflict must resolve in `conflict_participants`. `open_conflicts()`
    # already guarantees both participants are current, and the caller is
    # expected to have built `conflict_participants` from that SAME
    # current-state read -- so a miss here is never "this conflict just
    # isn't relevant", it is `conflict_participants`/`open_conflicts`
    # disagreeing about what is current, an internal inconsistency.
    # Silently dropping such a conflict would make storage corruption
    # indistinguishable from "Cortex checked and found nothing wrong",
    # which is exactly the false certainty A14 exists to prevent -- so
    # this raises instead of skipping, checked BEFORE relevance so a
    # corrupted-but-irrelevant conflict cannot slip through unexamined.
    for conflict in open_conflicts:
        for memory_id in conflict.memory_ids:
            if memory_id not in conflict_participants:
                raise CortexStorageError(
                    f"Open conflict {conflict.memory_ids!r} references memory {memory_id!r} "
                    "that is missing from the current participant map"
                )

    # [A14.1] Rule A': a conflict participant is relevant if it clears
    # the same lexical/FTS/semantic/evidence-rescue gate as everything
    # else (`_memory_matches`), OR the SAME Memory object already
    # appears in `root_causes`/`verified_lessons`/`open_invalidations`.
    # Deliberately excludes `invariants`: those are always-include
    # regardless of `task`, so treating invariant membership as a
    # relevance signal would make every conflict touching an invariant
    # appear on every task, defeating relevance entirely (measured in
    # A14.0.1's invariant-contagion probe). This is also how semantic
    # admission reaches conflicts without a dedicated pool: if the
    # semantic channel already admitted a participant into one of the
    # three gated fields, it is already in `_conflict_gated_ids` below.
    _conflict_gated_ids = frozenset(
        memory.memory_id for memory in (*root_causes, *verified_lessons, *open_invalidations)
    )

    def _conflict_participant_relevant(memory_id: str) -> bool:
        if memory_id in _conflict_gated_ids:
            return True
        return _memory_matches(conflict_participants[memory_id])

    # One-sided sufficiency (A14.0.1 §8): a Memory Cortex is about to show
    # as individually authoritative must not be shown without also
    # surfacing that it is contradicted, even when the other side of that
    # contradiction says nothing the task's own wording matches.
    preflight_conflicts = tuple(
        PreflightConflict(
            conflict=conflict,
            memories=(
                conflict_participants[conflict.memory_ids[0]],
                conflict_participants[conflict.memory_ids[1]],
            ),
        )
        for conflict in open_conflicts
        if _conflict_participant_relevant(conflict.memory_ids[0])
        or _conflict_participant_relevant(conflict.memory_ids[1])
    )

    return Preflight(
        task=task,
        known_failures=known_failures,
        root_causes=root_causes,
        verified_lessons=verified_lessons,
        recommended_validation=tuple(recommended_validation),
        invariants=tuple(invariant_memories),
        open_invalidations=open_invalidations,
        open_conflicts=preflight_conflicts,
        pending=pending,
    )
