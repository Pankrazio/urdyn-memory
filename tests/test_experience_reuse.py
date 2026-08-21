"""End-to-end tests for the A5 experience-reuse tracer bullet:

    Attempt (failed) -> Attempt (succeeded) -> verified Lesson
        -> explicit promotion -> verified Skill -> process ends

    new agent -> guard(action) -> known failure + applicable skill
        + recommended validation

This mirrors how `test_preflight.py`'s A4 scenario test works: one
`Urdyn` handle records everything and is discarded, and a second,
independent handle stands in for a fresh agent/session with no shared
conversation state.
"""

import shutil

from urdyn import Urdyn


def test_attempt_lesson_skill_guard_end_to_end(tmp_path):
    process_a = Urdyn.init(tmp_path, "dev")

    error_evidence = process_a.add_evidence(
        "Refresh token was invalidated during rotation.", kind="error_observation"
    )
    process_a.record_attempt(
        task="Update authentication refresh logic.",
        approach="Reuse the previous refresh token after rotation.",
        outcome="failed",
        evidence=[error_evidence],
    )

    validation = process_a.add_evidence("Authentication tests passed.", kind="test_result")
    process_a.record_attempt(
        task="Update authentication refresh logic.",
        approach="Persist and use only the newly issued refresh token.",
        outcome="succeeded",
        evidence=[validation],
    )

    lesson = process_a.learn(
        "After token rotation, use only the newly issued refresh token.",
        evidence=[error_evidence],
        supporting_evidence=[validation],
        verified=True,
    )

    skill = process_a.promote(
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
    assert skill.verification_state == "verified"

    # process A terminates; nothing below may depend on its Python state
    del process_a

    # a brand-new agent/session discovers the workspace with no shared state
    process_b = Urdyn.discover(tmp_path)

    result = process_b.guard("Modify refresh-token persistence logic")
    assert len(result.known_failures) == 1
    assert result.known_failures[0].approach == "Reuse the previous refresh token after rotation."
    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].skill_id == skill.skill_id
    assert result.applicable_skills[0].verification_state == "verified"
    assert result.applicable_skills[0].steps == skill.steps
    assert len(result.recommended_validation) == 1
    assert result.recommended_validation[0].content == "Authentication tests passed."

    # a third, equally independent handle confirms an unrelated action
    # gets no warning at all
    process_c = Urdyn.discover(tmp_path)
    unrelated = process_c.guard("Change CSS button color")
    assert unrelated.is_empty()


def test_workspace_is_portable_after_promotion(tmp_path):
    """Copying `.urdyn/` to a new path must preserve skill ids, steps,
    conditions, provenance, verification state, and guard behavior."""
    original_root = tmp_path / "original"
    original_root.mkdir()
    cx = Urdyn.init(original_root, "dev")

    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    lesson = cx.learn(
        "After token rotation, use only the newly issued refresh token.",
        supporting_evidence=[validation],
        verified=True,
    )
    original_skill = cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=["Inspect rotation.", "Persist only the new token."],
        conditions=["Only applies to authentication refresh flows."],
    )
    del cx

    copied_root = tmp_path / "copied"
    shutil.copytree(original_root, copied_root)

    reopened = Urdyn.open(copied_root)
    (copied_skill,) = reopened.skills()

    assert copied_skill.skill_id == original_skill.skill_id
    assert copied_skill.name == original_skill.name
    assert copied_skill.steps == original_skill.steps
    assert copied_skill.conditions == original_skill.conditions
    assert copied_skill.verification_state == original_skill.verification_state
    assert copied_skill.source_lesson_id == original_skill.source_lesson_id
    assert copied_skill.evidence_ids == original_skill.evidence_ids

    result = reopened.guard("Modify refresh-token persistence logic")
    assert len(result.applicable_skills) == 1
    assert result.applicable_skills[0].skill_id == original_skill.skill_id
