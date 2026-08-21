"""The canonical `Skill` model.

A `Skill` is not a `Memory`. `Memory` (and its `lesson` kind) answers "what
does Urdyn know" — a conclusion, stated as a sentence. A `Skill` answers a
different question, "how do I do this" — an ordered procedure with a name
and an applicability, meant to be followed, not just recalled. Bending
`Memory` to carry `steps`/`conditions` fields that only a Skill ever
populates would deform Memory's contract for every other kind; Skill gets
its own primitive instead.

A Skill is never created from nothing: it is deliberately *promoted* from
an existing Lesson (see `Urdyn.promote`), which is where its
`source_lesson_id` and `evidence_ids` provenance come from. Promotion is
never automatic and a Skill is never mutated in place — a later, better
procedure is a new Skill, not an edit of an old one.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

SKILL_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

SKILL_CANDIDATE = "candidate"
SKILL_VERIFIED = "verified"
VALID_SKILL_VERIFICATION_STATES = frozenset({SKILL_CANDIDATE, SKILL_VERIFIED})


@dataclasses.dataclass(frozen=True, slots=True)
class Skill:
    """A single recorded procedure, persisted by Urdyn.

    `steps` is the ordered procedure itself — read top to bottom, not a
    bag of tips. `conditions` is when the procedure applies (its
    applicability), which may be empty if nothing beyond the name/purpose
    narrows it.

    `verification_state` is `candidate` or `verified`, never a confidence
    score. A Skill is `verified` only when promoted from a Lesson that was
    itself `verified` — a Skill promoted from an unverified (candidate)
    Lesson stays `candidate`, no matter how the promotion is worded. This
    mirrors `Memory.epistemic_state`'s honesty rule instead of inventing a
    second one.

    `source_lesson_id` is the memory_id of the Lesson this Skill was
    promoted from — the provenance a consumer can follow to see exactly
    why Urdyn considers this procedure valid. `evidence_ids` is copied
    from that Lesson's own evidence, as canonically persisted at
    promotion time. The copied provenance itself remains stable because
    the source Lesson is immutable — but that is a claim about the
    record, not about the world: immutable provenance does not mean this
    procedure remains valid forever. A Skill's real-world validity can go
    stale later (dependencies, environment, code, or assumptions change)
    even though the record of why Urdyn once considered it valid never
    does. A5 does not implement Skill revalidation; any future change in
    validity must be represented through new Urdyn history/state, not
    by mutating this Skill.
    """

    skill_id: str
    name: str
    purpose: str
    steps: tuple[str, ...]
    conditions: tuple[str, ...]
    verification_state: str
    source_lesson_id: str
    evidence_ids: tuple[str, ...]
    recorded_at: dt.datetime
