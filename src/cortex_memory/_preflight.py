"""Preflight: selecting prior experience relevant to an upcoming task.

Deterministic and lexical: given experience Cortex already has on record
(attempts, verified lessons, root causes), select the subset worth
surfacing for a task. This is not a search engine and not the Context
Compiler: no ranking scores, no semantic similarity, no summarization, no
token budgeting.

Relevance is decided two ways:

1. Direct vocabulary overlap with the task (an attempt's own task/approach
   text, a lesson's or root cause's own content).
2. Shared provenance: a root cause or lesson that cites the same Evidence
   as an attempt already judged relevant is included even if its own
   wording shares no vocabulary with the task at all. A root cause like
   "the old token was reused after rotation" has nothing lexically in
   common with "authentication refresh logic" — it is relevant because it
   explains the *same observed failure*, which Evidence is what actually
   connects it to the matching attempt.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable

from ._attempt import OUTCOME_FAILED, OUTCOME_SUCCEEDED, Attempt
from ._evidence import Evidence
from ._memory import Memory

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A small, fixed stopword list to keep single common words (e.g. "the",
# "to") from making unrelated experience look relevant. Not a language
# model, not configurable: just noise reduction for keyword overlap.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "to", "of", "for", "on", "in", "and", "or", "is",
        "are", "was", "were", "be", "been", "with", "this", "that", "it",
        "its", "as", "by", "at", "from", "into", "during", "not", "do",
        "does", "did",
    }
)

_VALIDATION_EVIDENCE_KINDS = frozenset({"test_result", "command_output"})


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOPWORDS}


def _is_relevant(query_tokens: frozenset[str], text: str) -> bool:
    """A candidate is relevant if it shares a strict majority of the
    query's significant vocabulary — more than half, not just a fixed
    count of two. A flat "two shared words" rule lets two completely
    unrelated attempts both match a long, generic query (e.g. "update",
    "error", and "change" all appearing somewhere) just because each
    happens to share exactly those two common engineering words; scaling
    the requirement with query length keeps that from happening, while a
    one-word query still only needs that one word."""
    if not query_tokens:
        return False
    shared = query_tokens & _tokens(text)
    threshold = len(query_tokens) // 2 + 1
    return len(shared) >= threshold


@dataclasses.dataclass(frozen=True, slots=True)
class Preflight:
    """Prior experience relevant to a task, grouped by what it means for
    an agent about to start work. A category is empty, not absent, when
    Cortex has nothing relevant on record for it.
    """

    task: str
    known_failures: tuple[Attempt, ...]
    root_causes: tuple[Memory, ...]
    verified_lessons: tuple[Memory, ...]
    recommended_validation: tuple[Evidence, ...]

    def is_empty(self) -> bool:
        return not (
            self.known_failures or self.root_causes or self.verified_lessons or self.recommended_validation
        )


def build_preflight(
    task: str,
    *,
    attempts: list[Attempt],
    root_cause_memories: list[Memory],
    verified_lesson_memories: list[Memory],
    evidence_lookup: Callable[[str], Evidence],
) -> Preflight:
    """Pure selection logic, operating on data already fetched from
    storage. Takes no dependency on SQLite so it can be tested and
    reasoned about independently of the storage boundary.

    `evidence_lookup` is expected to raise if given an id that does not
    resolve: every id passed to it here came from an attempt's or
    memory's own `evidence_ids`, so a miss means the persisted provenance
    link is dangling, not that the caller asked about something that
    was never expected to exist.
    """
    query_tokens = frozenset(_tokens(task))

    def _attempt_matches(attempt: Attempt) -> bool:
        return _is_relevant(query_tokens, f"{attempt.task} {attempt.approach}")

    known_failures = tuple(a for a in attempts if a.outcome == OUTCOME_FAILED and _attempt_matches(a))
    relevant_successes = tuple(a for a in attempts if a.outcome == OUTCOME_SUCCEEDED and _attempt_matches(a))

    relevant_attempt_evidence_ids = frozenset(
        evidence_id for attempt in (*known_failures, *relevant_successes) for evidence_id in attempt.evidence_ids
    )

    def _memory_matches(memory: Memory) -> bool:
        if _is_relevant(query_tokens, memory.content):
            return True
        return not relevant_attempt_evidence_ids.isdisjoint(memory.evidence_ids)

    root_causes = tuple(memory for memory in root_cause_memories if _memory_matches(memory))
    verified_lessons = tuple(memory for memory in verified_lesson_memories if _memory_matches(memory))

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
    )
