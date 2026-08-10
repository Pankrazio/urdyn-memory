"""The FTS5 candidate-widening channel must never bypass epistemic or
temporal filtering: an unverified Lesson, a superseded Memory, or an
unverified Skill are excluded (or honestly labeled) exactly as they
already were for the lexical channel alone (see `test_preflight.py`'s
`test_preflight_excludes_unverified_lesson_candidate` and
`test_preflight_excludes_superseded_lesson`). These tests specifically
target the NEW risk surface A7 introduces: long, diluted queries that
only the widened channel could plausibly admit, to prove the new
channel inherits the same filtering rather than bypassing it.

The mechanism that makes this safe by construction: `build_preflight`/
`build_guard_result` only ever check FTS-admitted ids against
candidates already present in `attempts`/`root_cause_memories`/
`verified_lesson_memories`/`skills` -- lists `_workspace.py` filters to
current state and (for lessons) `verified` *before* either channel is
ever consulted (see `_preflight.py`'s and `_guard.py`'s own
docstrings). These tests exist to verify that construction holds under
FTS5, not to re-derive it from scratch.
"""

from cortex_memory import Cortex

_DILUTED_TASK = (
    "I was reviewing the deployment checklist and also wanted to check on how to "
    "fix that database connection pool exhaustion problem we keep running into "
    "under load, could you help me understand what has already been tried there"
)


def test_fts_channel_does_not_surface_unverified_lesson_as_verified(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    lesson = cx.learn("Database connection pool exhaustion happens under sustained load.")

    result = cx.preflight(_DILUTED_TASK)

    assert lesson.memory_id not in [m.memory_id for m in result.verified_lessons]
    assert result.verified_lessons == ()


def test_fts_channel_does_not_resurrect_a_superseded_verified_lesson(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Confirmed via load test.", kind="test_result")
    old_lesson = cx.learn(
        "Database connection pool exhaustion happens under sustained load.",
        supporting_evidence=[validation],
        verified=True,
    )
    new_lesson = cx.learn(
        "Close database connections explicitly to avoid pool exhaustion under sustained load.",
        supporting_evidence=[validation],
        verified=True,
        supersedes=old_lesson.memory_id,
    )

    result = cx.preflight(
        "I was reviewing the deployment checklist and also wanted to check on how "
        "to explicitly close database connections to avoid that pool exhaustion "
        "problem under sustained load instead of leaving it unresolved, could you "
        "help me understand what has already been tried there before I dig in myself"
    )

    assert old_lesson.memory_id not in [m.memory_id for m in result.verified_lessons]
    assert [m.memory_id for m in result.verified_lessons] == [new_lesson.memory_id]


def test_fts_channel_reports_a_candidate_skill_honestly_not_as_verified(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    candidate_lesson = cx.learn("Redis cache connections might need a retry policy under load.")
    candidate_skill = cx.promote(
        candidate_lesson,
        name="Investigate redis retry policy",
        purpose="Understand redis connection retries under load before changing anything.",
        steps=["Read the retry code."],
    )

    result = cx.guard(
        "I was going over our infra notes and wondered whether we already looked "
        "into redis connection retry policy handling under load, has anyone "
        "investigated that before"
    )

    assert [s.skill_id for s in result.applicable_skills] == [candidate_skill.skill_id]
    assert result.applicable_skills[0].verification_state == "candidate"
