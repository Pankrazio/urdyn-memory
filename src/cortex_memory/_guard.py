"""Guard: is there a known risk or an applicable, deliberately-recorded
procedure directly relevant to an action about to be taken?

`guard(action)` is deliberately narrower than `preflight(task)`. Preflight
answers "what prior experience is worth knowing before starting this
task" and surfaces known failures, root causes, and verified lessons on
lexical relevance alone. Guard answers a stricter question: "is there a
Skill — a procedure Cortex was deliberately taught — that applies here?"

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
return type. Lexical relevance uses the same deterministic engine
`preflight()` uses (`_relevance.py`), not a second one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from ._attempt import OUTCOME_FAILED, Attempt
from ._evidence import RECOMMENDED_VALIDATION_EVIDENCE_KINDS, Evidence
from ._relevance import is_relevant, tokens
from ._skill import Skill


@dataclasses.dataclass(frozen=True, slots=True)
class GuardResult:
    """Advisory output of `guard()`, grouped the same way `Preflight` is.

    Guard never blocks, mutates, or executes anything — it only reports
    what Cortex found. `applicable_skills` carries each Skill's own
    `verification_state`, so a consumer can tell a verified procedure from
    an unverified one instead of Cortex quietly implying certainty it
    doesn't have.
    """

    action: str
    known_failures: tuple[Attempt, ...]
    applicable_skills: tuple[Skill, ...]
    recommended_validation: tuple[Evidence, ...]

    def is_empty(self) -> bool:
        return not (self.known_failures or self.applicable_skills or self.recommended_validation)


def build_guard_result(
    action: str,
    *,
    skills: list[Skill],
    attempts: list[Attempt],
    evidence_lookup: Callable[[str], Evidence],
) -> GuardResult:
    """Pure selection logic, operating on data already fetched from
    storage. Takes no dependency on SQLite so it can be tested and
    reasoned about independently of the storage boundary.

    `skills` and `attempts` are expected in the order Cortex recorded
    them, so the result stays deterministic across calls.
    """
    query_tokens = frozenset(tokens(action))

    def _skill_matches(skill: Skill) -> bool:
        haystack = f"{skill.name} {skill.purpose} {' '.join(skill.conditions)}"
        return is_relevant(query_tokens, haystack)

    applicable_skills = tuple(skill for skill in skills if _skill_matches(skill))

    relevant_evidence_ids = frozenset(
        evidence_id for skill in applicable_skills for evidence_id in skill.evidence_ids
    )

    def _attempt_matches(attempt: Attempt) -> bool:
        return is_relevant(query_tokens, f"{attempt.task} {attempt.approach}")

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
    )
