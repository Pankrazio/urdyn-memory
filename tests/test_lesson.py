"""Tests for `Cortex.learn()` and lesson verification semantics."""

import pytest

from cortex_memory import Cortex


def test_learn_defaults_to_lesson_kind_and_candidate_state(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    lesson = cx.learn("Use only the newly issued refresh token.")

    assert lesson.kind == "lesson"
    assert lesson.epistemic_state == "user_asserted"


def test_learn_verified_requires_evidence(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.learn("Use only the newly issued refresh token.", verified=True)


def test_learn_verified_with_evidence_is_accepted(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    lesson = cx.learn(
        "Use only the newly issued refresh token.",
        evidence=[validation],
        verified=True,
    )

    assert lesson.epistemic_state == "verified"
    assert lesson.evidence_ids == (validation.evidence_id,)


def test_learn_verified_rejects_bare_user_statement(tmp_path):
    """Verified means backed by an actual check, not just backed by
    *something*. An unchecked opinion must not be enough to call a
    lesson verified, even though it is a perfectly valid Evidence."""
    cx = Cortex.init(tmp_path, "dev")
    opinion = cx.add_evidence("I think this solution works.", kind="user_statement")

    with pytest.raises(ValueError):
        cx.learn("This solution always works.", evidence=[opinion], verified=True)


def test_learn_verified_rejects_bare_file_reference(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    reference = cx.add_evidence("src/auth/refresh.py", kind="file_reference")

    with pytest.raises(ValueError):
        cx.learn("This solution always works.", evidence=[reference], verified=True)


def test_learn_verified_accepts_user_confirmation(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    confirmation = cx.add_evidence(
        "I ran the auth flow manually and the bug is gone.", kind="user_confirmation"
    )

    lesson = cx.learn("Use only the newly issued refresh token.", evidence=[confirmation], verified=True)

    assert lesson.epistemic_state == "verified"


def test_learn_verified_accepts_command_output(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    output = cx.add_evidence("exit code 0, deploy succeeded", kind="command_output")

    lesson = cx.learn("Use only the newly issued refresh token.", evidence=[output], verified=True)

    assert lesson.epistemic_state == "verified"


def test_learn_verified_accepts_mixed_evidence_if_any_qualifies(tmp_path):
    """A weak (user_statement) piece of evidence alongside a strong
    (test_result) one must not disqualify the verification: the rule is
    'at least one qualifying piece', not 'every piece must qualify'."""
    cx = Cortex.init(tmp_path, "dev")
    opinion = cx.add_evidence("I think this solution works.", kind="user_statement")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    lesson = cx.learn(
        "Use only the newly issued refresh token.", evidence=[opinion, validation], verified=True
    )

    assert lesson.epistemic_state == "verified"


def test_recording_a_successful_attempt_does_not_verify_an_unrelated_lesson(tmp_path):
    """A successful Attempt is not, by itself, Evidence for anything: it
    must be explicitly cited as such. Simply existing in the same
    workspace must not upgrade any candidate lesson's epistemic state."""
    cx = Cortex.init(tmp_path, "dev")
    candidate = cx.learn("Refresh tokens might need special handling.")

    cx.record_attempt(task="Update authentication refresh logic.", approach="the fix", outcome="succeeded")

    (still_candidate,) = [m for m in cx.timeline(kind="lesson") if m.memory_id == candidate.memory_id]
    assert still_candidate.epistemic_state == "user_asserted"


def test_remember_verified_requires_evidence_generically(tmp_path):
    """The verified-requires-evidence rule is a `remember()` invariant, not
    a `learn()`-only special case: it must hold for any kind of memory."""
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("Old token was reused after rotation.", kind="root_cause", epistemic_state="verified")


def test_remember_accepts_inferred_epistemic_state(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    root_cause = cx.remember(
        "Old token was reused after rotation.", kind="root_cause", epistemic_state="inferred"
    )

    assert root_cause.epistemic_state == "inferred"


def test_remember_rejects_unknown_epistemic_state(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("something", epistemic_state="not-a-real-state")


def test_lesson_candidate_can_be_superseded_by_verified_version(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    candidate = cx.learn("Refresh tokens might need special handling.")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")

    verified = cx.learn(
        "Use only the newly issued refresh token.",
        evidence=[validation],
        verified=True,
        supersedes=candidate.memory_id,
    )

    history = cx.timeline(kind="lesson")
    assert [m.memory_id for m in history] == [candidate.memory_id, verified.memory_id]
    current = cx.state(kind="lesson")
    assert [m.memory_id for m in current] == [verified.memory_id]


def test_lesson_persists_across_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    original = cx.learn("Use only the newly issued refresh token.", evidence=[validation], verified=True)
    del cx

    reopened = Cortex.open(tmp_path)
    (lesson,) = reopened.state(kind="lesson")

    assert lesson.memory_id == original.memory_id
    assert lesson.epistemic_state == "verified"
    assert lesson.evidence_ids == (validation.evidence_id,)
