"""Guard: is there a known risk or an applicable, deliberately-recorded
procedure directly relevant to an action about to be taken?

`guard(action)` is deliberately narrower than `preflight(task)`. Preflight
answers "what prior experience is worth knowing before starting this
task" and surfaces known failures, root causes, and verified lessons on
lexical relevance alone. Guard answers a stricter question: "is there a
Skill — a procedure Urdyn was deliberately taught — that applies here?"

A `known_failure` must satisfy all three of:
  1. it failed;
  2. it shares Evidence with a Skill that is itself applicable to `action`;
  3. its own task/approach text is lexically relevant to `action`.

(2) alone is not enough: Evidence is sometimes shared by coincidence (the
same CI run, the same generic note) between two otherwise unrelated
pieces of experience, and admitting a failure on shared-Evidence alone
would let an unrelated failure leak into a warning about a completely
different action just because it happened to cite the same note as the
matching Skill. (3) alone is not enough either — that is exactly what
`preflight()` already does, and it is not what makes `guard()` a
different, more selective function. Both conditions together, plus (1),
are what keeps guard selective instead of being `preflight()` under
another name: a different, stricter admission rule, not just a renamed
return type.

"Lexically relevant" in (3) is decided by the same channels `preflight()`
uses (`_relevance.py`'s structured majority rule, FTS5/BM25 candidate
widening via `_retrieval.py`, and A7.4's optional calibrated semantic
channel via `_semantic.py`) — not a second, separate engine. What keeps
`guard()` more conservative than `preflight()` is not a stricter version
of relevance itself, but that relevance is only ever one of three
conditions `guard()` requires together; widening which candidates can
satisfy condition (3) does not loosen (1) or (2) — a Skill's own
applicability match uses the SAME per-pool semantic abstention policy as
`preflight()`, but that policy was itself calibrated more conservatively
for the `skill` pool specifically (see `_semantic.SEMANTIC_POLICY`), so
`guard()`'s applicable-skill matching stays harder
to satisfy than `preflight()`'s memory matching even though both go
through the same code path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ._attempt import OUTCOME_FAILED, Attempt
from ._evidence import RECOMMENDED_VALIDATION_EVIDENCE_KINDS, Evidence
from ._relevance import attempt_search_text, is_relevant, skill_search_text, tokens
from ._retrieval import fts_admitted_ids
from ._semantic_store import SemanticState
from ._skill import Skill


@dataclasses.dataclass(frozen=True, slots=True)
class GuardResult:
    """Advisory output of `guard()`, grouped the same way `Preflight` is.

    Guard never blocks, mutates, or executes anything — it only reports
    what Urdyn found. `applicable_skills` carries each Skill's own
    `verification_state`, so a consumer can tell a verified procedure from
    an unverified one instead of Urdyn quietly implying certainty it
    doesn't have.

    `retrieval` (A27) reports which retrieval substrate actually answered
    — semantic plus lexical, or lexical alone because the semantic index
    was never enabled, is unusable here, or could not be brought up to
    date. It is `None` only when a caller built this object directly
    without going through `Urdyn.guard()`. Deliberately placed last and
    defaulted so the existing field shape stays an unbroken prefix, and
    deliberately excluded from `is_empty()`: it describes HOW the answer
    was produced, never WHAT was found.
    """

    action: str
    known_failures: tuple[Attempt, ...]
    applicable_skills: tuple[Skill, ...]
    recommended_validation: tuple[Evidence, ...]
    retrieval: SemanticState | None = None

    def is_empty(self) -> bool:
        return not (self.known_failures or self.applicable_skills or self.recommended_validation)


def build_guard_result(
    action: str,
    *,
    skills: list[Skill],
    attempts: list[Attempt],
    evidence_lookup: Callable[[str], Evidence],
    skill_fts_candidates: list[tuple[str, str]] = (),
    attempt_fts_candidates: list[tuple[str, str]] = (),
    skill_semantic_admitted: frozenset[str] = frozenset(),
    attempt_semantic_admitted: frozenset[str] = frozenset(),
    retrieval: SemanticState | None = None,
) -> GuardResult:
    """Pure selection logic, operating on data already fetched from
    storage. Takes no dependency on SQLite so it can be tested and
    reasoned about independently of the storage boundary.

    `skills` and `attempts` are expected in the order Urdyn recorded
    them, so the result stays deterministic across calls.

    `skill_fts_candidates`/`attempt_fts_candidates` are `(entity_id,
    text)` pairs already ranked best-first by BM25 for this `action`
    (see `MemoryStore.search_candidates`), or `()` if FTS5 is
    unavailable -- matching then falls back to the lexical channel
    alone, exactly as before A7. As in `preflight()`, only ids already
    present in `skills`/`attempts` can be admitted through them; nothing
    about current-state or evidence-sharing filtering changes.
    """
    query_tokens = frozenset(tokens(action))
    skill_admitted = fts_admitted_ids(query_tokens, list(skill_fts_candidates))
    attempt_admitted = fts_admitted_ids(query_tokens, list(attempt_fts_candidates))

    def _skill_matches(skill: Skill) -> bool:
        haystack = skill_search_text(skill.name, skill.purpose, skill.conditions)
        if is_relevant(query_tokens, haystack):
            return True
        if skill.skill_id in skill_admitted:
            return True
        return skill.skill_id in skill_semantic_admitted

    applicable_skills = tuple(skill for skill in skills if _skill_matches(skill))

    relevant_evidence_ids = frozenset(
        evidence_id for skill in applicable_skills for evidence_id in skill.evidence_ids
    )

    def _attempt_matches(attempt: Attempt) -> bool:
        if is_relevant(query_tokens, attempt_search_text(attempt.task, attempt.approach)):
            return True
        if attempt.attempt_id in attempt_admitted:
            return True
        return attempt.attempt_id in attempt_semantic_admitted

    known_failures = tuple(
        attempt
        for attempt in attempts
        if attempt.outcome == OUTCOME_FAILED
        and not relevant_evidence_ids.isdisjoint(attempt.evidence_ids)
        and _attempt_matches(attempt)
    )

    candidate_evidence_ids: list[str] = []
    for skill in applicable_skills:
        for evidence_id in skill.evidence_ids:
            if evidence_id not in candidate_evidence_ids:
                candidate_evidence_ids.append(evidence_id)

    recommended_validation = []
    for evidence_id in candidate_evidence_ids:
        evidence = evidence_lookup(evidence_id)
        if evidence.kind in RECOMMENDED_VALIDATION_EVIDENCE_KINDS:
            recommended_validation.append(evidence)

    return GuardResult(
        action=action,
        known_failures=known_failures,
        applicable_skills=applicable_skills,
        recommended_validation=tuple(recommended_validation),
        retrieval=retrieval,
    )
