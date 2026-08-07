"""The canonical `Evidence` model.

Evidence is distinct from Memory: it is raw support for a belief (a user
statement, a command output, a test result, a file/document reference, a
tool output), not the belief itself. Recording evidence never implies
verification: a memory derived from a user statement remains
`user_asserted`, not `verified`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re

EVIDENCE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")

DEFAULT_EVIDENCE_KIND = "user_statement"
VALID_EVIDENCE_KINDS = frozenset(
    {
        "user_statement",
        "command_output",
        "test_result",
        "file_reference",
        "tool_output",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """A single piece of evidence persisted by Cortex."""

    evidence_id: str
    content: str
    kind: str
    recorded_at: dt.datetime
