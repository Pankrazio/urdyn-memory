"""The canonical `Evidence` model.

Evidence is distinct from Memory: it is raw support for a belief (a user
statement, a command output, a test result, a file/document reference, a
tool output), not the belief itself. Recording evidence never implies
verification: a memory derived from a user statement remains
`user_asserted`, not `verified`.

`error_observation` is a technical failure observed during an `Attempt`
(an error message, a stack trace). It is deliberately modeled as an
Evidence kind rather than a separate `Error` class: an observed error is
raw support ("this is what happened"), not a claim ("this is why it
happened") — the latter is what a `root_cause` memory is for.

`user_confirmation` is distinct from `user_statement`: a statement is
what someone said or believes ("I think this works"), which is not
itself a check of anything. A confirmation is the user explicitly
attesting to a specific, checkable fact ("I ran it and the bug is gone").
This distinction exists only so `verified` (see `_memory.py`) can require
evidence that actually confirms something, without building any
identity/authentication machinery around who is confirming it.
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
        "user_confirmation",
        "command_output",
        "test_result",
        "file_reference",
        "tool_output",
        "error_observation",
    }
)

# Evidence kinds strong enough to justify calling a memory `verified`:
# each represents an actual check that occurred (a test ran, a command or
# tool produced output, the user explicitly confirmed a specific fact).
# Excluded: `user_statement` (an assertion, not a check), `file_reference`
# (points at a file without confirming anything about it was inspected),
# and `error_observation` (documents a failure, not a confirmation that
# something is correct).
VERIFICATION_EVIDENCE_KINDS = frozenset(
    {
        "test_result",
        "command_output",
        "tool_output",
        "user_confirmation",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """A single piece of evidence persisted by Cortex."""

    evidence_id: str
    content: str
    kind: str
    recorded_at: dt.datetime
