"""Tests for `Cortex.preflight()`: selecting relevant prior experience.

`test_preflight_matches_the_a4_real_utility_scenario` mirrors the A4
milestone's own acceptance scenario: an agent in one session records a
failed attempt, its root cause, a fix, and a verified lesson; a second,
unrelated `Cortex` handle (simulating a different agent/session) must be
able to retrieve that experience through `preflight()` alone.
"""

import pytest

from cortex_memory import Cortex


def test_preflight_rejects_empty_task(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.preflight("   ")


def test_preflight_on_empty_workspace_returns_empty_result(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    result = cx.preflight("some task nobody has attempted")

    assert result.is_empty()
    assert result.known_failures == ()
    assert result.root_causes == ()
    assert result.verified_lessons == ()
    assert result.recommended_validation == ()


def test_preflight_matches_the_a4_real_utility_scenario(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    # The root cause's own wording ("old token was reused after rotation")
    # shares no vocabulary with the task ("authentication refresh logic").
    # It is connected to the matching failed attempt only through the
    # error evidence both cite -- exactly the provenance graph the A4
    # milestone describes (Attempt -> Evidence <- root cause / lesson).
    error_evidence = cx.add_evidence(
        "Refresh token was invalidated during rotation.", kind="error_observation"
    )
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Modify token refresh handling directly.",
        outcome="failed",
        evidence=[error_evidence],
    )
    cx.remember(
        "Old token was reused after rotation.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[error_evidence],
    )

    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Persist and use only the newly issued refresh token.",
        outcome="succeeded",
        evidence=[validation],
    )
    cx.learn(
        "Use only the newly issued refresh token.",
        evidence=[validation],
        verified=True,
    )

    # a second, independent handle stands in for a fresh agent/session
    agent_b = Cortex.discover(tmp_path)
    result = agent_b.preflight("Modify authentication refresh logic")

    assert not result.is_empty()
    assert len(result.known_failures) == 1
    assert "Modify token refresh handling directly." == result.known_failures[0].approach
    assert len(result.root_causes) == 1
    assert "Old token was reused after rotation." == result.root_causes[0].content
    assert len(result.verified_lessons) == 1
    assert "Use only the newly issued refresh token." == result.verified_lessons[0].content
    assert len(result.recommended_validation) == 1
    assert "Authentication tests passed." == result.recommended_validation[0].content


def test_preflight_matches_a_paraphrased_related_task(tmp_path):
    """A future task worded differently from the original but about the
    same thing must still surface the relevant failure."""
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Modify authentication refresh logic",
        approach="Directly mutate the token in place",
        outcome="failed",
    )

    result = cx.preflight("Fix authentication refresh token handling")

    assert len(result.known_failures) == 1


def test_preflight_generic_engineering_vocabulary_does_not_cross_match(tmp_path):
    """Common words like 'update', 'fix', 'error', 'test', and 'change'
    are so frequent in engineering tasks that two completely unrelated
    attempts can each share a couple of them with any given query. That
    must not be enough on its own to call either one relevant."""
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Update the database connection pool",
        approach="Fix the timeout error by changing the config",
        outcome="failed",
    )
    cx.record_attempt(
        task="Update the frontend button styles",
        approach="Fix the layout error by changing the CSS",
        outcome="failed",
    )

    result = cx.preflight("Update login error handling and change the test setup")

    assert result.known_failures == ()


def test_preflight_excludes_unrelated_task(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Modify token refresh handling directly.",
        outcome="failed",
    )

    result = cx.preflight("Refactor CSS button styles")

    assert result.is_empty()


def test_preflight_excludes_unverified_lesson_candidate(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.learn("Refresh tokens for authentication might need special handling.")

    result = cx.preflight("authentication refresh tokens")

    assert result.verified_lessons == ()


def test_preflight_excludes_successful_attempt_from_known_failures(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Persist the newly issued refresh token.",
        outcome="succeeded",
    )

    result = cx.preflight("authentication refresh logic")

    assert result.known_failures == ()


def test_preflight_excludes_superseded_lesson(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    old = cx.learn("Refresh tokens for authentication need care.", evidence=[validation], verified=True)
    cx.learn(
        "Use only the newly issued refresh token for authentication.",
        evidence=[validation],
        verified=True,
        supersedes=old.memory_id,
    )

    result = cx.preflight("authentication refresh token")

    assert len(result.verified_lessons) == 1
    assert result.verified_lessons[0].memory_id != old.memory_id


def test_shared_evidence_correlation_does_not_leak_across_unrelated_topics(tmp_path):
    """Two attempts that happen to cite the *same generic* evidence must
    not cross-pollinate: a root cause tied to one unrelated failure must
    not show up for a query that only matches the other."""
    cx = Cortex.init(tmp_path, "dev")
    shared_note = cx.add_evidence("CI run #4021 output", kind="command_output")

    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Modify token refresh handling directly.",
        outcome="failed",
        evidence=[shared_note],
    )
    cx.record_attempt(
        task="Refactor CSS button styles.",
        approach="Rewrite the stylesheet from scratch.",
        outcome="failed",
        evidence=[shared_note],
    )
    cx.remember(
        "The build pipeline was flaky that day.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[shared_note],
    )

    # This query is only about CSS; it must not pull in the unrelated
    # root cause just because it shares evidence with the CSS attempt.
    result = cx.preflight("Refactor CSS button styles")

    assert len(result.known_failures) == 1
    assert result.known_failures[0].task == "Refactor CSS button styles."
    # the root cause DOES legitimately correlate here, since it shares
    # evidence with the one attempt that *did* match this query
    assert len(result.root_causes) == 1


def test_preflight_missing_evidence_reference_fails_explicitly(tmp_path):
    """A dangling evidence reference on a *successful* attempt is
    reachable through `recommended_validation`'s resolution path, which
    must fail loudly on corruption rather than silently drop the entry."""
    import sqlite3

    from cortex_memory import CortexStorageError

    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("CI output", kind="command_output")
    cx.record_attempt(task="task", approach="approach", outcome="succeeded", evidence=[evidence])

    connection = sqlite3.connect(cx._db_path)
    try:
        connection.execute("DELETE FROM evidence WHERE evidence_id = ?", (evidence.evidence_id,))
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CortexStorageError):
        cx.preflight("task approach")


def test_preflight_ordering_is_deterministic_across_calls(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(task="authentication refresh", approach="first try", outcome="failed")
    cx.record_attempt(task="authentication refresh", approach="second try", outcome="failed")

    first_call = cx.preflight("authentication refresh")
    second_call = cx.preflight("authentication refresh")

    assert [a.attempt_id for a in first_call.known_failures] == [a.attempt_id for a in second_call.known_failures]


def test_preflight_survives_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Modify token refresh handling directly.",
        outcome="failed",
    )
    del cx

    reopened = Cortex.open(tmp_path)
    result = reopened.preflight("authentication refresh logic")

    assert len(result.known_failures) == 1
