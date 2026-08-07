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

    assert field_names == {
        "task",
        "known_failures",
        "root_causes",
        "verified_lessons",
        "recommended_validation",
    }


def test_cortex_preflight_signature_takes_only_a_task_string(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    # a real call with just a task string must work; there is no second
    # positional/keyword parameter a caller is expected to supply that
    # would hint at a specific retrieval strategy (e.g. no `min_tokens=`,
    # no `strategy=`, no `top_k=`).
    result = cx.preflight("some task")

    assert isinstance(result, Preflight)
