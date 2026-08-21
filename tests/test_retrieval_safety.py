"""FTS5 query safety: a task/action string is never concatenated
directly into FTS5 query syntax. Every token reaching
`MemoryStore.search_candidates` already passed through
`_relevance.tokens()` (alnum-only, casefolded), and every token is
individually double-quoted before being placed in a MATCH expression,
so it is always treated as a literal term -- never as an FTS5 operator,
column filter, or malformed syntax -- regardless of what punctuation,
quoting, or Unicode the original user text contained.

These tests exercise both the public API (`cx.preflight()`/
`cx.guard()`, which real callers use) and `MemoryStore.search_candidates`
directly (the actual SQL boundary), so a regression that reintroduces
raw string concatenation would be caught at both levels.
"""

import pytest

from urdyn import Urdyn
from urdyn._relevance import tokens
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._store import MemoryStore


@pytest.mark.parametrize(
    "task",
    [
        'Fix the "broken" login flow',
        "Fix (urgent) login bug before release",
        "What's wrong with the user's login session?",
        "Fix login bug -- it's blocking release!",
        "Fix the bug near the login screen",  # "near" is an FTS5 operator keyword
        "Réparer le bug de connexion à la base de données",  # non-ASCII input
        "修复登录错误",  # CJK input: tokenizes to nothing under [a-z0-9]+, must not crash
        "Fix login bug \U0001f525\U0001f41b",  # emoji
        "???",  # punctuation only, tokenizes to nothing
        "the a to of for on in and or is",  # stopwords only, tokenizes to nothing
    ],
)
def test_preflight_and_guard_never_raise_on_pathological_task_text(tmp_path, task):
    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(
        task="Fix login bug near the session handler", approach="Patch it", outcome="failed"
    )

    preflight_result = cx.preflight(task)
    guard_result = cx.guard(task)

    assert preflight_result.task == task
    assert guard_result.action == task


def test_search_candidates_returns_empty_for_empty_token_set(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Something searchable.", kind="note")

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        assert store.search_candidates(frozenset(), ENTITY_MEMORY) == []


def test_search_candidates_handles_a_token_that_is_an_fts5_operator_keyword(tmp_path):
    """`near` and `not` are meaningful in raw FTS5 query syntax. As
    ordinary tokens (quoted, never concatenated raw) they must behave
    like any other literal word, matching content that contains them
    and never raising."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("The bug is near the login screen.", kind="note")

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        candidates = store.search_candidates(frozenset({"near"}), ENTITY_MEMORY)

    assert len(candidates) == 1


def test_search_candidates_query_containing_double_quote_character_is_impossible(tmp_path):
    """Defense-in-depth: `_relevance.tokens()` can never produce a token
    containing a double quote (its pattern is `[a-z0-9]+`), so the
    quoting `search_candidates` applies can never itself be broken out
    of by a token derived from user text."""
    assert all('"' not in token for token in tokens('Fix "quoted" login bug'))


def test_preflight_on_unicode_only_task_returns_empty_not_error(tmp_path):
    """A task with no ASCII alphanumeric content tokenizes to nothing.
    Both channels admit nothing for an empty token set; the call
    returns a normal empty `Preflight`, not an exception -- Urdyn
    cannot yet tokenize non-ASCII scripts, and that is a known,
    gracefully-handled limitation, not a crash."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Something in the workspace.", kind="note")

    result = cx.preflight("修复登录错误")

    assert result.is_empty()
