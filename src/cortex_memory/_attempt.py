"""The canonical `Attempt` model.

An `Attempt` is an observable record of trying to do something: what task
was being worked on, what approach was taken, and what happened. It is not
knowledge or a conclusion (that is what `Memory`/`Lesson` are for) — it is
a fact about what was done, kept even when the outcome was failure.

Attempts are append-only: a failed attempt is never rewritten to look
successful. A later successful attempt at the same task is a separate,
independent `Attempt` record.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

ATTEMPT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_PARTIAL = "partial"
VALID_OUTCOMES = frozenset({OUTCOME_SUCCEEDED, OUTCOME_FAILED, OUTCOME_PARTIAL})


@dataclasses.dataclass(frozen=True, slots=True)
class Attempt:
    """A single recorded attempt at a task, persisted by Cortex.

    `evidence_ids` is the same provenance mechanism used by `Memory`: the
    ids of `Evidence` supporting what happened (e.g. an error message or a
    test result), not a copy of that evidence's content.
    """

    attempt_id: str
    task: str
    approach: str
    outcome: str
    recorded_at: dt.datetime
    evidence_ids: tuple[str, ...] = ()
