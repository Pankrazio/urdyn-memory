"""Tests locking down `Preflight` as a stable public contract, independent
of the lexical matching strategy that currently implements it.

The point of these tests is not behavior but shape: `cx.preflight(task)`
must keep returning the same kind of thing (a `Preflight` of Attempt/
Memory/Evidence tuples) even if the internal matching strategy is later
replaced with FTS, BM25, or semantic retrieval. Nothing about "shared
lexical tokens" should be observable from outside `_preflight.py`.
"""

import dataclasses

import cortex_memory
from cortex_memory import Cortex, Preflight


def test_preflight_is_exported_but_matching_internals_are_not():
    assert "Preflight" in cortex_memory.__all__
    assert not hasattr(cortex_memory, "build_preflight")
    assert not hasattr(cortex_memory, "_is_relevant")
    assert not hasattr(cortex_memory, "_tokens")


def test_preflight_dataclass_shape_has_no_matching_strategy_leakage():
    field_names = {f.name for f in dataclasses.fields(Preflight)}

    # `invariants` (A9.1) and `open_invalidations` (A11.3) are deliberate,
    # documented additions to this contract: unlike the other fields,
    # `invariants` is populated without any matching strategy at all (see
    # `Preflight.invariants`'s docstring), and `open_invalidations` goes
    # through the SAME matching strategy as `root_causes`/
    # `verified_lessons` (see `Preflight.open_invalidations`'s docstring)
    # -- neither reintroduces the leakage this test guards against, since
    # nothing about "shared lexical tokens" becomes observable from the
    # field names or types themselves.
    assert field_names == {
        "task",
        "known_failures",
        "root_causes",
        "verified_lessons",
        "recommended_validation",
        "invariants",
        "open_invalidations",
    }


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


def test_cortex_preflight_signature_takes_only_a_task_string(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    # a real call with just a task string must work; there is no second
    # positional/keyword parameter a caller is expected to supply that
    # would hint at a specific retrieval strategy (e.g. no `min_tokens=`,
    # no `strategy=`, no `top_k=`).
    result = cx.preflight("some task")

    assert isinstance(result, Preflight)
