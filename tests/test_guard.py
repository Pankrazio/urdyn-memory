"""Tests for `Cortex.guard()` and `GuardResult`.

`test_guard_finds_known_failure_applicable_skill_and_validation` mirrors
the A5 milestone's own REAL UTILITY TEST scenario, and
`test_guard_is_not_an_alias_of_preflight` locks down the property that
makes `guard()` meaningfully different from `preflight()`: guard is
anchored on applicable Skills, not on lexical relevance alone.
"""

import dataclasses

import pytest

from cortex_memory import Cortex


def _build_refresh_token_experience(cx):
    error_evidence = cx.add_evidence(
        "Refresh token was invalidated during rotation.", kind="error_observation"
    )
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Reuse the previous refresh token after rotation.",
        outcome="failed",
        evidence=[error_evidence],
    )

    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Persist and use only the newly issued refresh token.",
        outcome="succeeded",
        evidence=[validation],
    )
    lesson = cx.learn(
        "After token rotation, use only the newly issued refresh token.",
        evidence=[error_evidence],
        supporting_evidence=[validation],
        verified=True,
    )
    skill = cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=[
            "Inspect the refresh-token rotation flow.",
            "Persist only the newly issued refresh token.",
            "Do not reuse the previous token.",
            "Run authentication refresh tests.",
        ],
    )
    return error_evidence, validation, lesson, skill


def test_guard_rejects_empty_action(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.guard("   ")


def test_guard_on_empty_workspace_returns_empty_result(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    result = cx.guard("Modify refresh-token persistence logic")

    assert result.is_empty()


def test_guard_finds_known_failure_applicable_skill_and_validation(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)

    result = cx.guard("Modify refresh-token persistence logic")

    assert not result.is_empty()
    assert len(result.known_failures) == 1
    assert result.known_failures[0].approach == "Reuse the previous refresh token after rotation."
    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].name == "Safely modify refresh-token rotation"
    assert result.applicable_skills[0].verification_state == "verified"
    assert len(result.recommended_validation) == 1
    assert result.recommended_validation[0].content == "Authentication tests passed."


def test_guard_on_unrelated_action_returns_empty(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)

    result = cx.guard("Change CSS button color")

    assert result.is_empty()


def test_guard_excludes_skill_with_no_lexical_relevance(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)

    result = cx.guard("Refactor CSS button styles")

    assert result.applicable_skills == ()


def test_guard_is_not_an_alias_of_preflight(tmp_path):
    """A failed attempt that lexically matches the action but has no
    associated Skill must surface through preflight but NOT through
    guard: guard is anchored on applicable skills, preflight is not."""
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(
        task="Refactor the database connection pool.",
        approach="Rewrite the pooling logic from scratch.",
        outcome="failed",
    )

    preflight_result = cx.preflight("Refactor the database connection pool")
    guard_result = cx.guard("Refactor the database connection pool")

    assert len(preflight_result.known_failures) == 1
    assert guard_result.is_empty()


def test_guard_ordering_is_deterministic_across_calls(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)

    first_call = cx.guard("Modify refresh-token persistence logic")
    second_call = cx.guard("Modify refresh-token persistence logic")

    assert [s.skill_id for s in first_call.applicable_skills] == [
        s.skill_id for s in second_call.applicable_skills
    ]
    assert [a.attempt_id for a in first_call.known_failures] == [
        a.attempt_id for a in second_call.known_failures
    ]


def test_guard_survives_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)
    del cx

    reopened = Cortex.open(tmp_path)
    result = reopened.guard("Modify refresh-token persistence logic")

    assert not result.is_empty()
    assert len(result.applicable_skills) == 1


def test_guard_from_independent_handle_after_promotion(tmp_path):
    """The A5 real-utility scenario: one agent/session records the
    experience and promotes a skill; a second, independent handle (a
    fresh agent) discovers the workspace and gets a warning purely from
    `guard()`, without any shared conversation state."""
    cx = Cortex.init(tmp_path, "dev")
    _build_refresh_token_experience(cx)

    agent_b = Cortex.discover(tmp_path)
    result = agent_b.guard("Modify refresh-token persistence logic")

    assert len(result.known_failures) == 1
    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].verification_state == "verified"
    assert len(result.recommended_validation) == 1

    unrelated = agent_b.guard("Change CSS button color")
    assert unrelated.is_empty()


def test_guard_result_is_immutable(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    result = cx.guard("some action")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.action = "changed"


def test_guard_shared_evidence_does_not_leak_across_unrelated_skills(tmp_path):
    """Two genuinely unrelated failed attempts (CSS and AUTH) happen to
    cite the exact same generic Evidence (e.g. the same CI run note).
    Each has its own associated Skill. Guarding the CSS action must
    surface only the CSS failure: shared Evidence alone must not be
    enough to leak the unrelated AUTH failure in, since AUTH's own
    task/approach text has nothing lexically in common with the CSS
    action either."""
    cx = Cortex.init(tmp_path, "dev")
    shared_note = cx.add_evidence("CI run #4021 output", kind="command_output")

    css_attempt = cx.record_attempt(
        task="Refactor CSS button styles.",
        approach="Rewrite the stylesheet from scratch.",
        outcome="failed",
        evidence=[shared_note],
    )
    auth_attempt = cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Reuse the previous refresh token after rotation.",
        outcome="failed",
        evidence=[shared_note],
    )

    css_lesson = cx.learn(
        "Rewriting the stylesheet from scratch breaks unrelated components.",
        evidence=[shared_note],
    )
    cx.promote(
        css_lesson,
        name="Refactor CSS button styles safely",
        purpose="Change button styling without breaking unrelated components.",
        steps=["Change one selector at a time.", "Check unrelated components after each change."],
    )

    auth_lesson = cx.learn(
        "After token rotation, use only the newly issued refresh token.",
        evidence=[shared_note],
    )
    cx.promote(
        auth_lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=["Persist only the newly issued refresh token."],
    )

    # this action is only about the CSS skill; it must not pull in the
    # unrelated auth attempt just because they share the same evidence
    result = cx.guard("Refactor CSS button styles")

    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].name == "Refactor CSS button styles safely"
    assert [a.attempt_id for a in result.known_failures] == [css_attempt.attempt_id]
    assert auth_attempt.attempt_id not in [a.attempt_id for a in result.known_failures]

    # symmetric check: guarding the AUTH action must not pull in CSS either
    auth_result = cx.guard("Modify refresh-token persistence logic")
    assert [a.attempt_id for a in auth_result.known_failures] == [auth_attempt.attempt_id]
    assert css_attempt.attempt_id not in [a.attempt_id for a in auth_result.known_failures]


def test_guard_excludes_failure_that_shares_evidence_but_is_not_lexically_relevant(tmp_path):
    """Even when a failed attempt genuinely shares Evidence with an
    applicable Skill, it must also be lexically relevant to the action
    itself to be reported -- shared Evidence is necessary but not
    sufficient. Here the failed attempt's own task/approach text has
    nothing to do with the guarded action, even though it cites the same
    Evidence as the (lexically applicable) Skill."""
    cx = Cortex.init(tmp_path, "dev")
    shared_note = cx.add_evidence("CI run #4021 output", kind="command_output")

    unrelated_attempt = cx.record_attempt(
        task="Deploy the nightly batch job.",
        approach="Trigger the cron runner manually.",
        outcome="failed",
        evidence=[shared_note],
    )
    lesson = cx.learn(
        "After token rotation, use only the newly issued refresh token.",
        evidence=[shared_note],
    )
    cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=["Persist only the newly issued refresh token."],
    )

    result = cx.guard("Modify refresh-token persistence logic")

    assert len(result.applicable_skills) == 1
    assert unrelated_attempt.attempt_id not in [a.attempt_id for a in result.known_failures]
    assert result.known_failures == ()


def test_guard_candidate_skill_is_still_reported_but_labeled_candidate(tmp_path):
    """Guard does not hide unverified procedures -- it reports them
    honestly labeled as `candidate`, since epistemic honesty means never
    pretending a candidate is verified, not suppressing candidates
    entirely."""
    cx = Cortex.init(tmp_path, "dev")
    lesson = cx.learn("Refresh tokens for authentication rotation might need special handling.")
    cx.promote(
        lesson,
        name="Investigate refresh-token rotation handling",
        purpose="Understand refresh token rotation edge cases before changing anything.",
        steps=["Read the rotation code.", "Check for reuse of old tokens."],
    )

    result = cx.guard("Investigate refresh-token rotation handling")

    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].verification_state == "candidate"
