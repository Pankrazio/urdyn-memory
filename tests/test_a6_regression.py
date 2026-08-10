"""A6 regression: recovering experience about Cortex's own retrieval
limitation, through natural paraphrases, via the public API only.

A6 (real-world validation) observed that `preflight()`/`guard()` could
return empty for a naturally-phrased task even though Cortex had
directly relevant experience on record -- a failed Attempt, a Root
Cause, a verified Lesson, and a verified Skill -- because the lexical
matcher required a strict majority of the QUERY's own vocabulary to be
present in the candidate, and a naturally-phrased task dilutes that
ratio. A7.0 diagnosed this and A7 fixes it with FTS5/BM25 candidate
widening (see `_retrieval.py`).

This test builds a workspace's worth of experience *about that exact
problem* -- Cortex's own retrieval sensitivity to task wording -- the
same way A4/A5's own tests build the refresh-token scenario: entirely
through the public API, never touching `_relevance.py`, `is_relevant`,
or any formula/threshold. The content recorded here is reusable
knowledge ("preflight/guard can miss things when worded differently"),
not a ready-made patch.

Every query below is verified, inline, to be a genuine miss for the
unmodified lexical channel alone (`is_relevant`, unchanged since A4) --
this is the "baseline must show a real miss" requirement from A7's
brief -- and then verified to succeed through `cx.preflight()`/
`cx.guard()`, the real public API a consuming agent actually calls.
"""

from cortex_memory import Cortex
from cortex_memory._relevance import is_relevant
from cortex_memory._relevance import tokens as _tokens


def _build_retrieval_sensitivity_experience(cx):
    error_evidence = cx.add_evidence(
        "preflight() returned nothing for a rephrased task even though a directly "
        "relevant lesson was already recorded",
        kind="error_observation",
    )
    failed_attempt = cx.record_attempt(
        task="Diagnose why preflight and guard found nothing relevant for a rephrased task",
        approach="Assumed nothing relevant had been recorded before",
        outcome="failed",
        evidence=[error_evidence],
    )
    root_cause = cx.remember(
        "preflight and guard rely too much on exact query wording.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[error_evidence],
    )
    validation = cx.add_evidence(
        "Re-ran preflight() close to the lesson's own wording: it was found immediately.",
        kind="test_result",
    )
    lesson = cx.learn(
        "preflight() and guard() can miss relevant experience that exists when the "
        "task is worded differently than how it was recorded.",
        evidence=[error_evidence],
        supporting_evidence=[validation],
        verified=True,
    )
    skill = cx.promote(
        lesson,
        name="Reword before trusting an empty guard result",
        purpose="preflight or guard can miss experience that is relevant and recorded.",
        steps=["Reword the task and retry.", "Do not assume nothing is known."],
    )
    return failed_attempt, root_cause, validation, lesson, skill


def test_a6_lesson_and_skill_are_genuine_misses_for_the_old_lexical_channel_alone(tmp_path):
    """Baseline check: these natural paraphrases are real misses for the
    matcher that predates A7 (unchanged `is_relevant`), not queries
    picked because they happen to already work. This is the "baseline
    demonstrates a real miss" requirement -- proof the fix is doing
    something, not just proof the new code runs."""
    cx = Cortex.init(tmp_path, "dev")
    _, _, _, lesson, skill = _build_retrieval_sensitivity_experience(cx)

    natural_task = (
        "it feels like preflight and guard aren't surfacing relevant experience, is "
        "that a known limitation when a task ends up worded differently compared to "
        "how it was before"
    )
    natural_action = (
        "guard hasn't been warning me about anything even though I feel like it "
        "should already know this could miss relevant experience before I end up "
        "trusting an empty result"
    )
    skill_haystack = f"{skill.name} {skill.purpose} {' '.join(skill.conditions)}"

    assert is_relevant(frozenset(_tokens(natural_task)), lesson.content) is False
    assert is_relevant(frozenset(_tokens(natural_action)), skill_haystack) is False


def test_a6_natural_paraphrase_recovers_known_failure_and_root_cause_via_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    failed_attempt, root_cause, _, _, _ = _build_retrieval_sensitivity_experience(cx)

    # a second, independent handle stands in for a fresh agent/session, exactly
    # as the A4/A5 real-utility scenarios do
    agent_b = Cortex.discover(tmp_path)
    result = agent_b.preflight(
        "why do preflight and guard find nothing relevant when a task is rephrased"
    )

    assert not result.is_empty()
    assert [a.attempt_id for a in result.known_failures] == [failed_attempt.attempt_id]
    assert [m.memory_id for m in result.root_causes] == [root_cause.memory_id]


def test_a6_natural_paraphrase_recovers_verified_lesson_and_validation_via_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _, _, validation, lesson, _ = _build_retrieval_sensitivity_experience(cx)

    agent_b = Cortex.discover(tmp_path)
    result = agent_b.preflight(
        "it feels like preflight and guard aren't surfacing relevant experience, is "
        "that a known limitation when a task ends up worded differently compared to "
        "how it was before"
    )

    assert not result.is_empty()
    assert [m.memory_id for m in result.verified_lessons] == [lesson.memory_id]
    assert [e.evidence_id for e in result.recommended_validation] == [validation.evidence_id]


def test_a6_natural_paraphrase_recovers_applicable_skill_and_validation_via_guard(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _, _, validation, _, skill = _build_retrieval_sensitivity_experience(cx)

    agent_b = Cortex.discover(tmp_path)
    result = agent_b.guard(
        "guard hasn't been warning me about anything even though I feel like it "
        "should already know this could miss relevant experience before I end up "
        "trusting an empty result"
    )

    assert not result.is_empty()
    assert [s.skill_id for s in result.applicable_skills] == [skill.skill_id]
    assert result.applicable_skills[0].verification_state == "verified"
    assert [e.evidence_id for e in result.recommended_validation] == [validation.evidence_id]


def test_a6_unrelated_task_stays_empty(tmp_path):
    """Hard negative: a completely unrelated task must not be affected
    by the wider candidate net FTS5 casts."""
    cx = Cortex.init(tmp_path, "dev")
    _build_retrieval_sensitivity_experience(cx)

    agent_b = Cortex.discover(tmp_path)
    preflight_result = agent_b.preflight("Change CSS button color to blue")
    guard_result = agent_b.guard("Change CSS button color to blue")

    assert preflight_result.is_empty()
    assert guard_result.is_empty()


def test_a6_borderline_negative_does_not_leak_unrelated_guard_clause_experience(tmp_path):
    """Borderline negative: a query about a *different* guard clause
    (payment input validation) shares surface vocabulary ('guard',
    'validation') with the retrieval-sensitivity experience but is
    genuinely unrelated. Widened candidate generation must not let it
    leak in just because both mention 'guard'."""
    cx = Cortex.init(tmp_path, "dev")
    _, _, _, lesson, skill = _build_retrieval_sensitivity_experience(cx)

    payment_evidence = cx.add_evidence(
        "Payment form accepted an empty card number in staging", kind="error_observation"
    )
    cx.record_attempt(
        task="Add input validation to the payment form guard clause",
        approach="Reject empty card numbers before submission",
        outcome="failed",
        evidence=[payment_evidence],
    )
    payment_lesson = cx.learn(
        "Reject empty card numbers in the payment form before submission.",
        evidence=[payment_evidence],
    )
    cx.promote(
        payment_lesson,
        name="Validate payment form input",
        purpose="Reject empty or malformed card numbers before submission.",
        steps=["Add a guard clause that checks the card number is non-empty."],
        conditions=["Submitting a payment form"],
    )

    agent_b = Cortex.discover(tmp_path)
    borderline = "add validation to the guard clause in the payment form before we deploy"
    preflight_result = agent_b.preflight(borderline)
    guard_result = agent_b.guard(borderline)

    assert lesson.memory_id not in [m.memory_id for m in preflight_result.verified_lessons]
    assert skill.skill_id not in [s.skill_id for s in guard_result.applicable_skills]
