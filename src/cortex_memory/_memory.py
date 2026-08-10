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
KIND_LESSON = "lesson"
KIND_ROOT_CAUSE = "root_cause"

# Operational-memory kinds (A9.1): specializations of Memory, not new
# canonical primitives. Current operational truth is derived exactly the
# way any other Memory kind's current truth is derived -- `supersedes` +
# `state(kind)` -- rather than through a dedicated dataclass or lifecycle
# enum. See each kind's usage in `_workspace.py`/`_preflight.py` for its
# exact semantics:
#   - `pending`: unfinished operational work that is still current. A
#     completed/cancelled pending is superseded by a memory of whatever
#     kind fits the resolution (typically `note` or `decision`) -- here
#     `supersedes` represents "this operational item is closed", not a
#     belief revision, which is a second, distinct meaning layered onto
#     the same mechanism. Deliberately no `pending_status`/done/cancelled
#     enum: closure is "no longer current", exactly like every other kind.
#   - `question`: an unresolved question that is currently open. Its
#     `epistemic_state` describes the proposition "this question is
#     currently open", not any future answer. Resolution is supersession
#     by a `decision` or `note` carrying the answer -- never `verified`,
#     which has no meaning for a question.
#   - `invariant`: a PROJECT-WIDE operational constraint that must not be
#     violated (e.g. ".cortex/ must remain gitignored"). A9.1 deliberately
#     restricts this kind to project-wide constraints only -- a
#     constraint that applies to a narrow subsystem is a FUTURE
#     POSSIBILITY pending a real `scope` model, not something to force
#     into this kind now. An invariant can be superseded if the project
#     deliberately revises the constraint.
#   - `environment`: a current project environment/toolchain fact (e.g.
#     "Python 3.12 is required"). Superseded when the fact changes; no
#     automatic staleness detection.
KIND_PENDING = "pending"
KIND_QUESTION = "question"
KIND_INVARIANT = "invariant"
KIND_ENVIRONMENT = "environment"

# A11.1: another specialization of Memory, not a new canonical primitive.
# An `invalidation` explicitly withdraws current authority from a prior
# Memory -- it does not assert that Memory was false, only that it must
# no longer be treated as current/authoritative. It normally supersedes
# the Memory whose authority is being withdrawn, using `supersedes`
# exactly as any other kind does: history stays append-only (the old
# Memory is preserved, never rewritten), and a later positive Memory may
# in turn supersede the invalidation once a replacement is known. Until
# then, `state(kind=<original kind>)` simply has nothing current in that
# lineage -- no separate "stale"/"invalidated" flag is introduced.
KIND_INVALIDATION = "invalidation"

VALID_KINDS = frozenset(
    {
        "note",
        "decision",
        KIND_LESSON,
        KIND_ROOT_CAUSE,
        KIND_PENDING,
        KIND_QUESTION,
        KIND_INVARIANT,
        KIND_ENVIRONMENT,
        KIND_INVALIDATION,
    }
)

EPISTEMIC_USER_ASSERTED = "user_asserted"
EPISTEMIC_INFERRED = "inferred"
EPISTEMIC_VERIFIED = "verified"
VALID_EPISTEMIC_STATES = frozenset({EPISTEMIC_USER_ASSERTED, EPISTEMIC_INFERRED, EPISTEMIC_VERIFIED})


@dataclasses.dataclass(frozen=True, slots=True)
class Memory:
    """A single canonical memory persisted by Cortex.

    `recorded_at` is when Cortex recorded the memory, in UTC. It is not
    a claim about when the underlying fact occurred or was observed.

    `supersedes` is the memory_id of an older memory this one replaces,
    or None. Superseding never mutates or deletes the older memory; it
    only marks this memory as the newer one in that lineage.

    `evidence_ids` is the stable provenance trail: the ids of Evidence
    this memory was derived from, in the order they were given. It is
    a reference, not a copy of evidence content. It answers "where did
    this come from", not "what proves it" — see `supporting_evidence_ids`.

    `epistemic_state` distinguishes how a memory came to be believed:
    `user_asserted` (stated, not checked), `inferred` (concluded without
    direct observation, e.g. a root cause deduced from symptoms), or
    `verified` (backed by evidence that represents an actual check, e.g.
    a test result). The exact requirement for `verified` differs by when
    the memory was recorded -- see `supporting_evidence_ids` below for
    the current (A12.1) contract and its pre-A12.1 legacy exception.

    `supporting_evidence_ids` (A12.1) is the subset of `evidence_ids`
    the caller explicitly designated as supporting THIS memory, as
    opposed to merely related/contextual provenance. It is always a
    subset of `evidence_ids` by construction (`remember()` folds any
    supporting evidence into the generic provenance trail automatically
    -- a caller never has to cite the same Evidence twice). A `verified`
    memory recorded from A12.1 onward always has at least one supporting
    Evidence of a qualifying kind; generic `evidence_ids` alone is no
    longer enough, even if it happens to contain a qualifying kind (see
    `remember()`'s docstring for the exact gate). A memory recorded
    before A12.1 shipped may be `verified` with an empty
    `supporting_evidence_ids` -- this is "verified under the pre-A12.1
    contract", preserved as-is, never rewritten or downgraded.

    Explicit support is not truth: Cortex does not semantically judge
    whether the designated Evidence actually proves this memory's
    content, only that the caller deliberately asserted it does, with
    a kind that represents an actual check having occurred.
    """

    memory_id: str
    content: str
    kind: str
    epistemic_state: str
    recorded_at: dt.datetime
    supersedes: str | None = None
    evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
