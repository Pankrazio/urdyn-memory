"""Tests locking down `Preflight` as a stable public contract, independent
of the lexical matching strategy that currently implements it.

The point of these tests is not behavior but shape: `cx.preflight(task)`
must keep returning the same kind of thing (a `Preflight` of Attempt/
Memory/Evidence tuples) even if the internal matching strategy is later
replaced with FTS, BM25, or semantic retrieval. Nothing about "shared
lexical tokens" should be observable from outside `_preflight.py`.
"""

import dataclasses
import datetime as dt

import cortex_memory
from cortex_memory import Cortex, Preflight, PreflightConflict
from cortex_memory._conflict import Conflict
from cortex_memory._memory import Memory


def test_preflight_is_exported_but_matching_internals_are_not():
    assert "Preflight" in cortex_memory.__all__
    assert not hasattr(cortex_memory, "build_preflight")
    assert not hasattr(cortex_memory, "_is_relevant")
    assert not hasattr(cortex_memory, "_tokens")


def test_preflight_dataclass_shape_has_no_matching_strategy_leakage():
    field_names = {f.name for f in dataclasses.fields(Preflight)}

    # `invariants` (A9.1), `open_invalidations` (A11.3), `open_conflicts`
    # (A14.1) and `pending` (A22.1) are deliberate, documented additions
    # to this contract: unlike the other fields, `invariants` is populated
    # without any matching strategy at all (see `Preflight.invariants`'s
    # docstring), `open_invalidations` and `pending` go through the SAME
    # matching strategy as `root_causes`/`verified_lessons` (see their
    # docstrings), and `open_conflicts`
    # reuses that same strategy PLUS membership in those already-admitted
    # fields (see `Preflight.open_conflicts`'s docstring) -- none of the
    # four reintroduces the leakage this test guards against, since
    # nothing about "shared lexical tokens" becomes observable from the
    # field names or types themselves. In particular `pending` names a
    # canonical `Memory` kind, not a retrieval channel: that a pending is
    # selected by relevance rather than dumped is a property of the
    # values, invisible in the shape.
    assert field_names == {
        "task",
        "known_failures",
        "root_causes",
        "verified_lessons",
        "recommended_validation",
        "invariants",
        "open_invalidations",
        "open_conflicts",
        "pending",
    }


def test_preflight_conflict_is_a_derived_view_not_the_canonical_conflict():
    """`PreflightConflict` is a Preflight-scoped derived pairing, not the
    canonical `Conflict` primitive (see `_conflict.py`): it must carry the
    canonical object PLUS the two Memories, and nothing else invented for
    rendering convenience (no severity/status/reason/score)."""
    field_names = {f.name for f in dataclasses.fields(PreflightConflict)}

    assert field_names == {"conflict", "memories"}


def test_preflight_invariants_field_defaults_to_empty_tuple():
    """The pre-A9.1 four-argument construction of `Preflight` must remain
    valid: `invariants` defaults to `()` rather than being required, so
    any existing caller that built a `Preflight` without naming it does
    not break."""
    preflight = Preflight(
        task="a task",
        known_failures=(),
        root_causes=(),
        verified_lessons=(),
        recommended_validation=(),
    )

    assert preflight.invariants == ()
    assert preflight.open_invalidations == ()
    assert preflight.open_conflicts == ()


def test_preflight_open_invalidations_field_defaults_to_empty_tuple():
    """The pre-A11.3 five-argument construction of `Preflight` (task +
    the four original fields + `invariants`) must remain valid:
    `open_invalidations` defaults to `()` rather than being required."""
    preflight = Preflight(
        task="a task",
        known_failures=(),
        root_causes=(),
        verified_lessons=(),
        recommended_validation=(),
        invariants=(),
    )

    assert preflight.open_invalidations == ()
    assert preflight.open_conflicts == ()


def test_preflight_open_conflicts_field_defaults_to_empty_tuple():
    """The pre-A14.1 six-argument construction of `Preflight` (task + the
    four original fields + `invariants` + `open_invalidations`) must
    remain valid: `open_conflicts` defaults to `()` rather than being
    required."""
    preflight = Preflight(
        task="a task",
        known_failures=(),
        root_causes=(),
        verified_lessons=(),
        recommended_validation=(),
        invariants=(),
        open_invalidations=(),
    )

    assert preflight.open_conflicts == ()


def test_preflight_with_only_open_conflicts_is_not_empty(tmp_path):
    """A `Preflight` bearing only a conflict signal is not "nothing found"
    -- see `Preflight.is_empty()`'s docstring and A14.1's false-certainty
    property."""
    memory_a = Memory(
        memory_id="a" * 32,
        content="A",
        kind="note",
        epistemic_state="user_asserted",
        recorded_at=dt.datetime.now(dt.timezone.utc),
        supersedes=None,
        evidence_ids=(),
        supporting_evidence_ids=(),
    )
    memory_b = dataclasses.replace(memory_a, memory_id="b" * 32, content="B")
    conflict = Conflict(memory_ids=("a" * 32, "b" * 32), recorded_at=dt.datetime.now(dt.timezone.utc))

    preflight = Preflight(
        task="a task",
        known_failures=(),
        root_causes=(),
        verified_lessons=(),
        recommended_validation=(),
        open_conflicts=(PreflightConflict(conflict=conflict, memories=(memory_a, memory_b)),),
    )

    assert preflight.is_empty() is False


def test_cortex_preflight_signature_takes_only_a_task_string(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    # a real call with just a task string must work; there is no second
    # positional/keyword parameter a caller is expected to supply that
    # would hint at a specific retrieval strategy (e.g. no `min_tokens=`,
    # no `strategy=`, no `top_k=`).
    result = cx.preflight("some task")

    assert isinstance(result, Preflight)
