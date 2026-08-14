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

# Named because two modules outside this one now depend on this exact
# kind: `_source.py`/`_store.py` record every Source observation as one
# `document_observation` Evidence (A19.1). A magic string repeated across
# modules is how the observation contract quietly drifts from the
# verification contract below.
#
# Deliberately NOT `file_reference`, whose meaning below is unchanged: a
# reference points AT a file without attesting that anything was read
# out of it. Seeding does read it -- it opens the file, decodes its
# bytes, rejects binaries and oversized documents, and keeps the text it
# actually saw. Reusing `file_reference` for that would collapse two
# different epistemic facts ("Cortex has a pointer to this file" and
# "Cortex read this document, and this is what it said") onto one kind,
# permanently, since pre-A19.1 rows already carry the old meaning.
EVIDENCE_KIND_DOCUMENT_OBSERVATION = "document_observation"

VALID_EVIDENCE_KINDS = frozenset(
    {
        "user_statement",
        "user_confirmation",
        "command_output",
        "test_result",
        "file_reference",
        EVIDENCE_KIND_DOCUMENT_OBSERVATION,
        "tool_output",
        "error_observation",
    }
)

# Evidence kinds strong enough to justify calling a memory `verified`:
# each represents an actual check that occurred (a test ran, a command or
# tool produced output, the user explicitly confirmed a specific fact).
# Excluded: `user_statement` (an assertion, not a check), `file_reference`
# (points at a file without confirming anything about it was inspected),
# `document_observation` (Cortex genuinely read the document, but reading
# a README does not check that what it claims is true -- the observation
# is faithful, the claims inside it are not thereby verified), and
# `error_observation` (documents a failure, not a confirmation that
# something is correct).
VERIFICATION_EVIDENCE_KINDS = frozenset(
    {
        "test_result",
        "command_output",
        "tool_output",
        "user_confirmation",
    }
)

# Evidence kinds worth surfacing as "recommended validation" to run before
# trusting a piece of experience (preflight) or acting on a warning (guard).
# Narrower than `VERIFICATION_EVIDENCE_KINDS`: a `user_confirmation` can
# justify calling a memory verified, but it is not itself something an
# agent can re-run to check its own work the way a test or command is.
RECOMMENDED_VALIDATION_EVIDENCE_KINDS = frozenset({"test_result", "command_output"})


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """A single piece of evidence persisted by Cortex."""

    evidence_id: str
    content: str
    kind: str
    recorded_at: dt.datetime
