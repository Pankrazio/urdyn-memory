"""The canonical `Event` primitive: an append-only record that something happened.

An `Event` is not a `Memory` or an `Attempt`. Those are recorded knowledge
and recorded experience; an event is a fact about Cortex's own history (a
memory was recorded, a memory was superseded, an attempt was recorded).
Events back the deterministic ordering used by `timeline()` and
`list_attempts()`, and are never mutated once written.

Internal to Cortex: not part of the public API. Its identity is a uuid4
hex string, stable and independent of any database row id.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

EVENT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

EVENT_KIND_MEMORY_RECORDED = "memory_recorded"
EVENT_KIND_MEMORY_SUPERSEDED = "memory_superseded"
EVENT_KIND_ATTEMPT_RECORDED = "attempt_recorded"
EVENT_KIND_SKILL_PROMOTED = "skill_promoted"
VALID_EVENT_KINDS = frozenset(
    {
        EVENT_KIND_MEMORY_RECORDED,
        EVENT_KIND_MEMORY_SUPERSEDED,
        EVENT_KIND_ATTEMPT_RECORDED,
        EVENT_KIND_SKILL_PROMOTED,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """A single append-only history record.

    `subject_id` is the memory_id the event is about. `occurred_at` is
    when Cortex recorded the event, in UTC.
    """

    event_id: str
    kind: str
    subject_id: str
    occurred_at: dt.datetime
