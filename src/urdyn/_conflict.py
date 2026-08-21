"""The canonical `Conflict` model: an explicit, structural statement that
two current or historical Memories cannot both be treated as a coherent
description of the same state.

A `Conflict` is not a judgment. Urdyn does not understand the content of
either Memory well enough to decide which one is right, so it never
chooses, deletes, downgrades, or invalidates anything on account of a
conflict being recorded. It only preserves the caller's explicit
assertion that these two Memories are mutually incompatible, in a form
that is canonical (not reconstructable from search/similarity), auditable
(survives restart and copy), and portable (identified only by the two
existing Memory ids, never a storage-specific id).

The relation is symmetric by construction: `record_conflict(A, B)` and
`record_conflict(B, A)` describe the same fact, so `memory_ids` is always
canonically ordered (ascending) rather than kept in call order -- this is
what makes A<->B and B<->A collapse to the same identity instead of being
treated as two different relations.

Deliberately excluded from A13.1:
no `conflict_id` (the ordered pair of Memory ids is itself a sufficient,
storage-independent identity), no evidence, no status/resolution field, no
severity/confidence. Whether a conflict is currently "open" is never
stored -- it is derived by checking `Urdyn.state()`/`current_ids()`
membership for both `memory_ids` at read time, exactly like every other
current-state projection in Urdyn.
"""

from __future__ import annotations

import dataclasses
import datetime as dt


def canonical_pair(memory_a_id: str, memory_b_id: str) -> tuple[str, str]:
    """Order two memory ids deterministically so that the pair identifies
    the conflict relation regardless of call order. `(A, B)` and `(B, A)`
    always produce the same result -- this is the entire mechanism behind
    A13.1's symmetric identity and duplicate/reverse-duplicate idempotency.
    """
    return (memory_a_id, memory_b_id) if memory_a_id < memory_b_id else (memory_b_id, memory_a_id)


@dataclasses.dataclass(frozen=True, slots=True)
class Conflict:
    """A single canonical conflict relation between two Memories.

    `memory_ids` is always the canonically ordered pair (see
    `canonical_pair`) -- it is the relation's entire identity, not just a
    payload field. `recorded_at` is when Urdyn first recorded this
    relation; a repeated `record_conflict()` call for the same pair (in
    either order) never changes it (see `Urdyn.record_conflict`'s
    docstring for the idempotency contract).

    Deliberately carries nothing else: no `conflict_id`, no evidence, no
    status/resolution. Whether this conflict is currently operative is not
    a property of this object -- see `Urdyn.open_conflicts()`.
    """

    memory_ids: tuple[str, str]
    recorded_at: dt.datetime
