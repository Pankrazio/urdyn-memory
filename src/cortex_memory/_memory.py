"""The canonical `Memory` model.

A `Memory` is Cortex's public unit of recorded knowledge. It carries no
storage details (no row ids, no SQL, no file paths) so that the backend
can evolve without changing what callers see.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

MEMORY_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

DEFAULT_KIND = "note"
VALID_KINDS = frozenset({"note", "decision"})

EPISTEMIC_USER_ASSERTED = "user_asserted"
VALID_EPISTEMIC_STATES = frozenset({EPISTEMIC_USER_ASSERTED})


@dataclasses.dataclass(frozen=True, slots=True)
class Memory:
    """A single canonical memory persisted by Cortex.

    `recorded_at` is when Cortex recorded the memory, in UTC. It is not
    a claim about when the underlying fact occurred or was observed.
    """

    memory_id: str
    content: str
    kind: str
    epistemic_state: str
    recorded_at: dt.datetime
