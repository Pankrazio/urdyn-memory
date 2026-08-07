"""Tests for `Cortex.recall()`."""

import pytest

from cortex_memory import Cortex


def test_recall_finds_exact_match(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("SQLite was selected for the first storage implementation.")

    results = cx.recall("SQLite")

    assert len(results) == 1
    assert "SQLite" in results[0].content


def test_recall_finds_partial_match(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("The public API must not expose raw SQL.")

    results = cx.recall("public API")

    assert len(results) == 1


def test_recall_is_case_insensitive(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("SQLite was selected for the first storage implementation.")

    results = cx.recall("sqlite")

    assert len(results) == 1


def test_recall_with_no_matches_returns_empty_list(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("unrelated content")

    results = cx.recall("nonexistent-term-xyz")

    assert results == []


def test_recall_on_empty_workspace_returns_empty_list(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    results = cx.recall("anything")

    assert results == []


def test_recall_rejects_empty_query(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.recall("")


def test_recall_rejects_whitespace_only_query(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.recall("   ")


def test_recall_respects_limit(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    for i in range(5):
        cx.remember(f"repeated memory number {i}")

    results = cx.recall("repeated", limit=2)

    assert len(results) == 2


def test_recall_rejects_non_positive_limit(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("something")

    with pytest.raises(ValueError):
        cx.recall("something", limit=0)


def test_recall_ranks_more_occurrences_higher(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("apple")
    cx.remember("apple apple apple")
    cx.remember("apple apple")

    results = cx.recall("apple")

    assert [m.content for m in results] == [
        "apple apple apple",
        "apple apple",
        "apple",
    ]


def test_recall_tie_breaks_by_most_recent_first(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember("banana one")
    second = cx.remember("banana two")

    results = cx.recall("banana")

    assert [m.memory_id for m in results] == [second.memory_id, first.memory_id]


def test_recall_ordering_is_deterministic_across_calls(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("orange fruit")
    cx.remember("orange orange fruit")
    cx.remember("plain orange")

    first_call = cx.recall("orange")
    second_call = cx.recall("orange")

    assert [m.memory_id for m in first_call] == [m.memory_id for m in second_call]


def test_recall_returns_multiple_relevant_results(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("SQLite was selected for the first storage implementation.")
    cx.remember("The public API must not expose raw SQL.")

    sqlite_results = cx.recall("SQLite")
    api_results = cx.recall("public API")

    assert len(sqlite_results) == 1
    assert len(api_results) == 1
