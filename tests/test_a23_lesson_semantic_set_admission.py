"""A23.1: bounded semantic SET admission for verified Lessons.

A23 (read-only analysis) reproduced, on the real model, that two
complementary verified lessons scoring 0.3782 and 0.3206 against the same
task were BOTH rejected -- not because either was irrelevant (the MEMORY
absolute floor is 0.20 and both cleared it comfortably) but because they
were 0.0576 apart, under a 0.08 `margin_floor`. A realistic four-lesson
workspace returned zero or one of four. The margin asks "is #1 separated
enough from #2 to be trusted as THE answer", which is the right question
only when at most one candidate can be correct.

Nothing in the Lesson model says that. There is no `alternative_of`, no
scope exclusivity, no winner semantics: exclusivity is expressed
canonically by `Conflict` and `supersedes`, and authority by
`epistemic_state`/supporting Evidence. And `preflight()` itself already
treats lessons as a SET on its other two channels -- lexical majority and
FTS/BM25 both admit unboundedly many.

A23.1 therefore changes the ADMISSION POLICY for one category, not the
similarity engine: verified lessons get their own disjoint pool
(the A11.3/A22.1 pattern) whose admission is
`_semantic.set_admitted_ids` -- the same ranking, no margin, a
Lesson-specific floor and a cap.

[A23.2] Both constants were then CALIBRATED on a prospective corpus
frozen before it was scored (78 lessons, 67 bilingual scenes, 13 of them
scenes where emitting nothing is the correct answer), and validated once
against the untouched A23.1.1 corpus: cap 3 -> 2 (rank #3 holds 2% of
genuinely relevant lessons, so the third slot bought no recall and cost
0.09 precision) and a Lesson floor of 0.30 (the 0.20 floor emitted a
lesson on 92% of the no-answer scenes; 0.30 does so on 25%, and stays
below the 0.327 median score of a genuinely relevant lesson).

The tests here use a deterministic fake encoder that maps a text to an
exact cosine against the query, so every case below is about the POLICY
and never about model luck. The real-model counterpart lives at the
bottom, skipped unless the pinned artifacts are already cached locally.

WHAT MUST NOT CHANGE, asserted here as well as in the suites it belongs
to: the global `margin_floor` (untouched -- MEMORY, ATTEMPT and SKILL
pools still ask the single-winner question), authority and current-state
eligibility, the lexical/FTS channels, and pending/invalidation pool
isolation.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from urdyn import Urdyn
from urdyn._semantic import (
    LESSON_SEMANTIC_FLOOR, SEMANTIC_POLICY, SET_ADMISSION_LIMIT,
)
from urdyn._retrieval import ENTITY_MEMORY

_FLOOR = LESSON_SEMANTIC_FLOOR                          # the Lesson pool's own, A23.2
_MEMORY_FLOOR = SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor   # untouched by A23
_MARGIN = SEMANTIC_POLICY[ENTITY_MEMORY].margin_floor

# A task whose vocabulary deliberately shares NOTHING with any lesson text
# below, so the lexical majority rule and the FTS channel both stay
# silent and these tests isolate the semantic channel -- the same
# technique the existing A7.7/A7.8 tests use.
TASK = "kindly outline whichever precautions merit attention throughout tomorrow morning"


class _ScoredModel:
    """Deterministic encoder placing every text on the unit circle at an
    exact cosine from the query.

    `cosines` maps a text to its intended similarity with the query; the
    query itself is mapped to 1.0, i.e. the vector (1, 0). Anything not
    listed is orthogonal to the query (cosine 0.0) rather than
    near-parallel -- the trap `test_semantic.py`'s own fake documents:
    a near-zero vector L2-normalizes back up into a spurious perfect
    match.
    """

    def __init__(self, cosines: dict[str, float]) -> None:
        self._cosines = dict(cosines)
        self._cosines[TASK] = 1.0

    def encode(self, texts):
        vectors = np.zeros((len(texts), 2), dtype=np.float32)
        for i, text in enumerate(texts):
            cosine = self._cosines.get(text)
            if cosine is None:
                vectors[i] = (0.0, 1.0)
            else:
                vectors[i] = (cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)))
        return vectors


@pytest.fixture
def scored_semantic(monkeypatch):
    """Install a `_ScoredModel` built from a caller-supplied cosine map.
    Returns the installer so each test states the exact score geometry it
    is about."""
    import urdyn._semantic as semantic

    def install(cosines: dict[str, float]):
        model = _ScoredModel(cosines)
        monkeypatch.setattr(semantic, "load_model_for_setup", lambda *a, **k: model)
        monkeypatch.setattr(semantic, "load_model_for_retrieval", lambda *a, **k: model)
        monkeypatch.setattr(semantic, "resolve_local_revision", lambda *a, **k: "fake-revision")
        return model

    return install


def _verified_lesson(cx, content, *, supersedes=None):
    evidence = cx.add_evidence(f"pytest -q :: passed :: {content}", kind="test_result")
    return cx.learn(
        content, evidence=[evidence], supporting_evidence=[evidence], verified=True, supersedes=supersedes
    )


def _lesson_ids(result):
    return {memory.memory_id for memory in result.verified_lessons}


# ---------------------------------------------------------------------------
# 12 -- THE DISCRIMINATING TEST AGAINST THE CURRENT POLICY
# ---------------------------------------------------------------------------


def test_two_complementary_verified_lessons_inside_the_margin_are_both_admitted(
    tmp_path, scored_semantic
):
    """A23's finding, as a policy test.

    Both lessons clear the EXISTING absolute floor comfortably; the gap
    between them (0.0576) is below the EXISTING margin floor. On
    2096197 this returns zero lessons, because the winner-margin policy
    reads a thin gap as "the pool is ambiguous" -- which is a statement
    about which single answer to trust, in a category that has no single
    answer. Both must now be admitted.

    The score geometry is the real one A23 measured, not a constructed
    edge case.
    """
    a_content = "Normalize environment values before applying numeric validation."
    b_content = "Reject booleans explicitly before integer validation; bool subclasses int."
    scored_semantic({a_content: 0.3782, b_content: 0.3206})

    cx = Urdyn.init(tmp_path, "dev")
    lesson_a = _verified_lesson(cx, a_content)
    lesson_b = _verified_lesson(cx, b_content)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    # premise, so this can never pass vacuously: both above the floor,
    # and the gap genuinely inside the margin the old policy applied
    assert 0.3782 >= _FLOOR and 0.3206 >= _FLOOR, "premise: both clear the calibrated Lesson floor"
    assert 0.3782 - 0.3206 < _MARGIN
    assert _lesson_ids(result) == {lesson_a.memory_id, lesson_b.memory_id}


def test_the_two_strongest_of_three_relevant_lessons_are_admitted(tmp_path, scored_semantic):
    """Set admission is not "one at a time": with three relevant lessons
    above the floor the two strongest are admitted together.

    [A23.2] The third is dropped by the calibrated cap, and that is a
    measured trade rather than a limitation to apologise for: on the
    prospective corpus rank #3 held 2% of all genuinely relevant lessons,
    so the third slot added no recall while costing a third of the
    precision."""
    strongest = "Normalize environment values before numeric validation."
    second = "Reject booleans explicitly when validating integers."
    third = "Register each new numeric option in the permitted-keys list."
    scored_semantic({strongest: 0.55, second: 0.50, third: 0.45})

    cx = Urdyn.init(tmp_path, "dev")
    a = _verified_lesson(cx, strongest)
    b = _verified_lesson(cx, second)
    c = _verified_lesson(cx, third)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert 0.45 >= _FLOOR, "premise: the third is dropped by the CAP, not by the floor"
    assert _lesson_ids(result) == {a.memory_id, b.memory_id}
    assert c.memory_id not in _lesson_ids(result)


# ---------------------------------------------------------------------------
# 13 -- THE DISCRIMINATING CAP TEST
# ---------------------------------------------------------------------------


def test_semantic_lesson_admission_is_capped_at_the_calibrated_limit(tmp_path, scored_semantic):
    """Four lessons, all above the floor, all mutually within the margin.
    Exactly the top `SET_ADMISSION_LIMIT` are admitted semantically -- the
    cap is what replaces the margin's incidental bound on how much a
    single channel may add, and A23 measured that removing the margin
    WITHOUT a cap cost materially more false positives."""
    a, b, c, d = (
        "Normalize environment values before numeric validation.",
        "Reject booleans explicitly when validating integers.",
        "Register each new numeric option in the permitted-keys list.",
        "Validate configuration after merging defaults with overrides.",
    )
    scored_semantic({a: 0.60, b: 0.55, c: 0.50, d: 0.45})

    cx = Urdyn.init(tmp_path, "dev")
    lesson_a = _verified_lesson(cx, a)
    lesson_b = _verified_lesson(cx, b)
    lesson_c = _verified_lesson(cx, c)
    lesson_d = _verified_lesson(cx, d)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert SET_ADMISSION_LIMIT == 2, "this test states the cap it is asserting"
    assert 0.45 >= _FLOOR, "premise: the excluded lessons are excluded by the CAP, not by the floor"
    assert _lesson_ids(result) == {lesson_a.memory_id, lesson_b.memory_id}
    assert lesson_c.memory_id not in _lesson_ids(result)
    assert lesson_d.memory_id not in _lesson_ids(result)


def test_capped_admission_is_stable_across_repeated_calls(tmp_path, scored_semantic):
    """Determinism, asserted as reproducibility rather than as a claim
    about tie-break order: A23.1 introduces no ordering of its own (see
    `set_admitted_ids`), so the same workspace and the same task must
    yield the identical selection every time."""
    contents = {
        "Normalize environment values before numeric validation.": 0.60,
        "Reject booleans explicitly when validating integers.": 0.55,
        "Register each new numeric option in the permitted-keys list.": 0.50,
        "Validate configuration after merging defaults with overrides.": 0.45,
    }
    scored_semantic(contents)

    cx = Urdyn.init(tmp_path, "dev")
    for content in contents:
        _verified_lesson(cx, content)
    cx.semantic_setup()

    results = [_lesson_ids(cx.preflight(TASK)) for _ in range(5)]

    assert all(selection == results[0] for selection in results)
    assert len(results[0]) == SET_ADMISSION_LIMIT


# ---------------------------------------------------------------------------
# 14 -- THE FALSE-POSITIVE GUARD: the floor still does its own job
# ---------------------------------------------------------------------------


def test_a_lesson_below_the_existing_floor_is_never_admitted(tmp_path, scored_semantic):
    """Set admission did not lower the bar: A23.1 reuses the EXISTING
    absolute floor untouched, so a weak candidate is rejected exactly as
    before -- and the cap is not what rejects it here."""
    strong = "Normalize environment values before numeric validation."
    weak = "Prefer descriptive commit messages when touching shared modules."
    # 0.27 sits ABOVE the untouched MEMORY floor and BELOW the calibrated
    # Lesson floor: the band A23.2 measured as almost pure noise.
    scored_semantic({strong: 0.55, weak: 0.27})

    cx = Urdyn.init(tmp_path, "dev")
    admitted = _verified_lesson(cx, strong)
    rejected = _verified_lesson(cx, weak)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _MEMORY_FLOOR < 0.27 < _FLOOR <= 0.55, (
        "premise: only the Lesson-specific floor separates these two -- the MEMORY floor would not"
    )
    assert _lesson_ids(result) == {admitted.memory_id}
    assert rejected.memory_id not in _lesson_ids(result)


def test_a_task_with_no_relevant_lesson_emits_nothing_from_the_noise_band(tmp_path, scored_semantic):
    """[A23.2] The calibration's hard case, as a deterministic regression.

    A task Urdyn genuinely has nothing for -- but a workspace full of
    engineering lessons that all sit in the band this model puts almost
    any prescriptive sentence into. A23.2 measured that band directly: on
    13 scenes whose correct answer was to emit nothing, a 0.20 floor
    emitted at least one lesson in 92% of them, because the median FALSE
    positive scores 0.350 while the median genuinely relevant lesson
    scores 0.327.

    The scores here are placed inside that measured band (0.21-0.29)
    rather than copied from any one corpus scene, so this test asserts the
    policy's behaviour on the band and not one model output.
    """
    contents = {
        "Pin the build tool version in the lockfile.": 0.29,
        "Assert on behaviour rather than on the wording of a log line.": 0.26,
        "Document why a workaround exists, not only what it does.": 0.24,
        "Emit a metric for every retry.": 0.21,
    }
    scored_semantic(contents)

    cx = Urdyn.init(tmp_path, "dev")
    for content in contents:
        _verified_lesson(cx, content)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert all(_MEMORY_FLOOR <= score < _FLOOR for score in contents.values()), (
        "premise: every candidate is in the band that the MEMORY floor admits and the Lesson floor rejects"
    )
    assert result.verified_lessons == (), (
        "preflight must be able to say it has nothing; this is the abstention A23.2 restored"
    )


def test_an_unrelated_workspace_still_surfaces_nothing(tmp_path, scored_semantic):
    """The whole point of abstention survives: several lessons, none
    relevant, nothing admitted. Bounded set retrieval is not
    recall-at-any-cost."""
    contents = {
        "Use design tokens instead of hard-coded hex colours.": 0.04,
        "Refresh a derived index when its source changes.": 0.11,
        "Write migrations so they are safe to run twice.": -0.05,
    }
    scored_semantic(contents)

    cx = Urdyn.init(tmp_path, "dev")
    for content in contents:
        _verified_lesson(cx, content)
    cx.semantic_setup()

    assert cx.preflight(TASK).verified_lessons == ()


# ---------------------------------------------------------------------------
# 7 -- AUTHORITY AND CURRENT STATE ARE NOT WIDENED
# ---------------------------------------------------------------------------


def test_an_unverified_lesson_is_not_admitted_by_a_high_score(tmp_path, scored_semantic):
    """Relevance is not authority. A candidate lesson scoring 1.0 -- the
    highest score expressible -- must stay out, because eligibility is
    decided before ranking and A23.1 did not touch it."""
    candidate = "Normalize environment values before numeric validation."
    verified = "Reject booleans explicitly when validating integers."
    scored_semantic({candidate: 1.0, verified: 0.55})

    cx = Urdyn.init(tmp_path, "dev")
    unverified = cx.learn(candidate)  # user_asserted, not verified
    trusted = _verified_lesson(cx, verified)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _lesson_ids(result) == {trusted.memory_id}
    assert unverified.memory_id not in _lesson_ids(result)


def test_a_superseded_lesson_is_not_admitted_by_a_high_score(tmp_path, scored_semantic):
    """Current state is not authority either, and neither is a score: a
    superseded verified lesson stays out however well it ranks, while
    its replacement is admitted normally."""
    old_content = "Coerce configuration values with int(value) and accept what Python accepts."
    new_content = "Reject non-numeric configuration strings instead of coercing them."
    scored_semantic({old_content: 0.95, new_content: 0.40})

    cx = Urdyn.init(tmp_path, "dev")
    old = _verified_lesson(cx, old_content)
    new = _verified_lesson(cx, new_content, supersedes=old.memory_id)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _lesson_ids(result) == {new.memory_id}
    assert old.memory_id not in _lesson_ids(result)
    # history is preserved -- the old lesson is gone from operational
    # retrieval, not from the record
    assert old.memory_id in {m.memory_id for m in cx.timeline()}


# ---------------------------------------------------------------------------
# 9 -- LEXICAL / FTS UNION AND THE CAP'S EXACT SCOPE
# ---------------------------------------------------------------------------


def test_a_lesson_admitted_both_lexically_and_semantically_appears_once(tmp_path, scored_semantic):
    """The union is over ids, so a lesson both channels admit is reported
    once -- `build_preflight` filters its own list, it does not
    concatenate per-channel results."""
    content = "normalize environment values before numeric validation"
    scored_semantic({content: 0.9, "normalize environment values": 1.0})

    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, content)
    cx.semantic_setup()

    result = cx.preflight("normalize environment values")

    assert [m.memory_id for m in result.verified_lessons] == [lesson.memory_id]


def test_the_semantic_cap_does_not_truncate_lexically_relevant_lessons(tmp_path, scored_semantic):
    """The cap bounds ONE channel, never the field.

    Five lessons all lexically relevant to the same task must all be
    surfaced: reading `SET_ADMISSION_LIMIT` as "preflight shows at most
    three lessons" would silently narrow the pre-A23.1 lexical contract,
    which is a regression, not this tracer's goal.
    """
    contents = [
        "environment values numeric validation normalize",
        "environment values numeric validation booleans",
        "environment values numeric validation allowlist",
        "environment values numeric validation merge",
        "environment values numeric validation defaults",
    ]
    scored_semantic({})  # nothing scores at all: the lexical channel alone must carry this

    cx = Urdyn.init(tmp_path, "dev")
    lessons = [_verified_lesson(cx, content) for content in contents]
    cx.semantic_setup()

    result = cx.preflight("environment values numeric validation")

    assert len(result.verified_lessons) == 5 > SET_ADMISSION_LIMIT
    assert _lesson_ids(result) == {lesson.memory_id for lesson in lessons}


def test_without_a_semantic_index_lexical_preflight_is_unchanged(tmp_path):
    """A23.1 is additive to semantic recall. With no `semantic setup` at
    all -- no fake model, no index -- preflight behaves exactly as it did
    before: the lexical channel works, and the semantic channel
    contributes nothing rather than raising."""
    cx = Urdyn.init(tmp_path, "dev")
    lesson = _verified_lesson(cx, "environment values numeric validation normalize")
    _verified_lesson(cx, "unrelated stylesheet button colour tokens")

    result = cx.preflight("environment values numeric validation")

    assert _lesson_ids(result) == {lesson.memory_id}


# ---------------------------------------------------------------------------
# 11 -- POOL ISOLATION
# ---------------------------------------------------------------------------


def test_a_root_cause_does_not_consume_lesson_set_capacity(tmp_path, scored_semantic):
    """The pools are disjoint in the direction A23.1 introduces: the
    LESSON pool contains lessons only, so a root cause outranking every
    one of them cannot take a slot from the cap.

    (The converse -- a lesson participating in the MEMORY pool -- is kept:
    verified lessons stay eligible there so A7.8's shared-Evidence cluster
    rescue keeps working, where a lesson below the floor is admitted
    because its sibling root cause cleared it. A23.1 adds a channel; it
    removes none. What A23.4 later bounded is not that eligibility but
    what a lesson's own SCORE may buy there -- see A23.4's admission rule
    below.)
    """
    root_cause_content = "The loader accepted a boolean where an integer was expected."
    lessons = {
        "Normalize environment values before numeric validation.": 0.55,
        "Reject booleans explicitly when validating integers.": 0.50,
    }
    scored_semantic({root_cause_content: 0.99, **lessons})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed at load time", kind="error_observation")
    root_cause = cx.remember(
        root_cause_content, kind="root_cause", epistemic_state="inferred", evidence=[evidence]
    )
    lesson_ids = {_verified_lesson(cx, content).memory_id for content in lessons}
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _lesson_ids(result) == lesson_ids, "the top-scoring root cause took a lesson slot"
    assert root_cause.memory_id in {m.memory_id for m in result.root_causes}


def test_lesson_set_admission_does_not_disturb_the_pending_pool(tmp_path, scored_semantic):
    """A22.1's disjoint pending pool keeps its own admission: two lessons
    winning the lesson pool must not cost the pending its slot, and the
    pending must not cost the lessons theirs."""
    pending_content = "The numeric override still needs an entry in the permitted-keys list."
    lesson_a = "Normalize environment values before numeric validation."
    lesson_b = "Reject booleans explicitly when validating integers."
    scored_semantic({pending_content: 0.60, lesson_a: 0.55, lesson_b: 0.52})

    cx = Urdyn.init(tmp_path, "dev")
    pending = cx.remember(pending_content, kind="pending")
    a = _verified_lesson(cx, lesson_a)
    b = _verified_lesson(cx, lesson_b)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert [m.memory_id for m in result.pending] == [pending.memory_id]
    assert _lesson_ids(result) == {a.memory_id, b.memory_id}


def test_lesson_set_admission_does_not_disturb_the_invalidation_pool(tmp_path, scored_semantic):
    """The same, for A11.3's disjoint invalidation pool."""
    invalidation_content = "The rule about coercing configuration values is no longer trusted."
    lesson_a = "Normalize environment values before numeric validation."
    lesson_b = "Reject booleans explicitly when validating integers."
    scored_semantic({invalidation_content: 0.60, lesson_a: 0.55, lesson_b: 0.52})

    cx = Urdyn.init(tmp_path, "dev")
    invalidation = cx.remember(invalidation_content, kind="invalidation")
    a = _verified_lesson(cx, lesson_a)
    b = _verified_lesson(cx, lesson_b)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert [m.memory_id for m in result.open_invalidations] == [invalidation.memory_id]
    assert _lesson_ids(result) == {a.memory_id, b.memory_id}


# ---------------------------------------------------------------------------
# 17 -- SIMILARITY STILL DOES NOT ADJUDICATE
# ---------------------------------------------------------------------------


def test_two_contradictory_lessons_surface_together_with_their_conflict(tmp_path, scored_semantic):
    """A23 measured that the margin, faced with two genuinely
    contradictory lessons, admitted one and hid the other -- presenting
    one side of a contradiction as the only truth. That was never a
    design requirement: `_conflict.py` is explicit that Urdyn never
    chooses a side, and `open_conflicts` exists to disclose both.

    Set admission does not adjudicate either. It surfaces both eligible
    lessons and leaves the canonical `Conflict` channel to say they
    disagree. This asserts the existing conflict behaviour is intact, not
    a new one.
    """
    coerce_content = "Always coerce configuration values with int(value)."
    strict_content = "Never coerce configuration values; reject non-numeric strings."
    scored_semantic({coerce_content: 0.55, strict_content: 0.50})

    cx = Urdyn.init(tmp_path, "dev")
    coerce_lesson = _verified_lesson(cx, coerce_content)
    strict_lesson = _verified_lesson(cx, strict_content)
    cx.record_conflict(coerce_lesson, strict_lesson)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _lesson_ids(result) == {coerce_lesson.memory_id, strict_lesson.memory_id}
    assert len(result.open_conflicts) == 1
    surfaced = {m.memory_id for m in result.open_conflicts[0].memories}
    assert surfaced == {coerce_lesson.memory_id, strict_lesson.memory_id}
    # neither side was downgraded, resolved or re-ranked by similarity
    assert all(m.epistemic_state == "verified" for m in result.verified_lessons)


# ---------------------------------------------------------------------------
# [A23.4] THE LESSON POLICY IS AUTHORITATIVE END-TO-END
#
# A23.2's validation found, and A23.3 reproduced in five distinct shapes,
# that the calibrated Lesson floor and cap held only INSIDE the Lesson
# pool. Verified lessons are also eligible in the MEMORY pool (A7.8 needs
# them there, see below), and that pool admits on the MEMORY floor of 0.20
# with no cap -- so a lesson the Lesson pool had rejected could still reach
# `preflight()` through it. Measured at 0/67 scenes on the dense
# prospective corpus and reproduced on the real model in a 4-lesson
# workspace: a small-pool phenomenon, because a mediocre candidate has to
# stay isolated enough to keep its margin.
#
# The repair does NOT remove lessons from the MEMORY pool -- A23.3 measured
# that doing so breaks three A7.8 tests. It separates the two ways a
# candidate reaches that pool's result:
#
#   score-borne representative -> obeys its OWN category's floor;
#   provenance-borne sibling   -> may still cross category boundaries,
#                                 once a valid representative has
#                                 established the experience's relevance.
# ---------------------------------------------------------------------------


def test_an_isolated_below_floor_lesson_is_not_admitted_through_the_memory_pool(
    tmp_path, scored_semantic
):
    """The leak in its simplest form, and the sparse workspace A23.2 found
    it in: ONE mediocre lesson, nothing else to compete with it, scoring in
    the band between the two floors. The Lesson pool rejects it; before
    A23.4 the MEMORY pool admitted it anyway, because 0.25 clears 0.20 and
    an isolated candidate has no competitor to lose the margin to."""
    weak = "Prefer descriptive commit messages when touching shared modules."
    scored_semantic({weak: 0.25})

    cx = Urdyn.init(tmp_path, "dev")
    _verified_lesson(cx, weak)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert _MEMORY_FLOOR < 0.25 < _FLOOR, (
        "premise: the score is admissible under MEMORY policy and inadmissible under Lesson policy"
    )
    assert result.verified_lessons == ()


def test_a_below_floor_lesson_cannot_establish_relevance_for_its_root_cause_sibling(
    tmp_path, scored_semantic
):
    """Cross-category contamination, the shape A23.3 added to the four
    A23.2 knew about: a lesson in the noise band does not merely leak
    itself, it can become the representative of its own provenance cluster
    and carry a root cause scoring 0.05 into the result with it.

    A rejected representative establishes nothing. Neither memory is
    relevant to this task, and neither may appear."""
    weak_lesson = "Prefer descriptive commit messages when touching shared modules."
    weak_root_cause = "The release notes were generated from the wrong branch."
    scored_semantic({weak_lesson: 0.25, weak_root_cause: 0.05})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed during the release", kind="error_observation")
    validation = cx.add_evidence("pytest -q :: passed", kind="test_result")
    cx.learn(weak_lesson, evidence=[evidence], supporting_evidence=[validation], verified=True)
    cx.remember(
        weak_root_cause, kind="root_cause", epistemic_state="inferred", evidence=[evidence]
    )
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert result.verified_lessons == ()
    assert result.root_causes == (), "a rejected representative dragged its sibling in"


def test_a_strong_root_cause_still_rescues_its_low_scoring_lesson_sibling(
    tmp_path, scored_semantic
):
    """A7.8's contract, in the direction the dedicated Lesson pool CANNOT
    reproduce: the lesson scores 0.0, so no floor and no cap can ever
    admit it on similarity. It appears because its root cause -- the same
    experience, the same Evidence -- established relevance, and an admitted
    cluster is admitted in full.

    This is why A23.4 is a category-boundary rule and not
    "every emitted lesson must score >= 0.30"."""
    root_cause_content = "The migration applied half its statements before failing."
    lesson_content = "Wrap multi-statement migrations in a single transaction."
    scored_semantic({root_cause_content: 0.70, lesson_content: 0.0})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed at deploy time", kind="error_observation")
    validation = cx.add_evidence("pytest -q :: passed", kind="test_result")
    root_cause = cx.remember(
        root_cause_content, kind="root_cause", epistemic_state="inferred", evidence=[evidence]
    )
    lesson = cx.learn(
        lesson_content, evidence=[evidence], supporting_evidence=[validation], verified=True
    )
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert 0.0 < _FLOOR, "premise: the lesson's own score cannot reach the Lesson floor"
    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}
    assert _lesson_ids(result) == {lesson.memory_id}


def test_a_lesson_clearing_its_own_floor_still_rescues_its_weak_root_cause_sibling(
    tmp_path, scored_semantic
):
    """The same rescue in the other direction, which is the one A23.4
    could plausibly have broken: a lesson may still be the representative
    that establishes relevance -- it just has to clear its OWN floor to do
    it. Its root cause sibling, which has no other channel, comes with
    it."""
    lesson_content = "Wrap multi-statement migrations in a single transaction."
    root_cause_content = "The migration applied half its statements before failing."
    scored_semantic({lesson_content: 0.70, root_cause_content: 0.05})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed at deploy time", kind="error_observation")
    validation = cx.add_evidence("pytest -q :: passed", kind="test_result")
    lesson = cx.learn(
        lesson_content, evidence=[evidence], supporting_evidence=[validation], verified=True
    )
    root_cause = cx.remember(
        root_cause_content, kind="root_cause", epistemic_state="inferred", evidence=[evidence]
    )
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert 0.70 >= _FLOOR, "premise: the representative clears its own category's floor"
    assert _lesson_ids(result) == {lesson.memory_id}
    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}


def test_a_shared_evidence_cluster_of_lessons_cannot_bypass_the_lesson_floor(
    tmp_path, scored_semantic
):
    """Two lessons drawn from one experience, both in the noise band. The
    cluster has no member of another category, so there is no
    cross-category rescue to preserve here -- admitting it would only be
    the MEMORY floor deciding a question that belongs to the Lesson
    pool."""
    first = "Prefer descriptive commit messages when touching shared modules."
    second = "Keep the changelog entry in the same commit as the change."
    scored_semantic({first: 0.25, second: 0.24})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed during the release", kind="error_observation")
    validation = cx.add_evidence("pytest -q :: passed", kind="test_result")
    for content in (first, second):
        cx.learn(content, evidence=[evidence], supporting_evidence=[validation], verified=True)
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert all(_MEMORY_FLOOR < score < _FLOOR for score in (0.25, 0.24))
    assert result.verified_lessons == ()


def test_a_shared_evidence_cluster_of_lessons_cannot_bypass_the_semantic_cap(
    tmp_path, scored_semantic
):
    """The second half of the same defect: the cap was bypassable too.
    A7.8 admits a winning cluster IN FULL, which is right for
    reconstructing one experience across categories and wrong as a way to
    return three lessons where the calibrated policy allows two. All three
    clear the floor here, so this is purely about who counts the slots."""
    contents = {
        "Wrap multi-statement migrations in a single transaction.": 0.60,
        "Take a restorable snapshot before a destructive migration.": 0.55,
        "Rehearse the migration against a copy of production data.": 0.50,
    }
    scored_semantic(contents)

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed at deploy time", kind="error_observation")
    validation = cx.add_evidence("pytest -q :: passed", kind="test_result")
    by_content = {
        content: cx.learn(
            content, evidence=[evidence], supporting_evidence=[validation], verified=True
        )
        for content in contents
    }
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert all(score >= _FLOOR for score in contents.values()), (
        "premise: the cap, not the floor, is what has to reject the third"
    )
    assert len(result.verified_lessons) == SET_ADMISSION_LIMIT
    assert _lesson_ids(result) == {
        by_content["Wrap multi-statement migrations in a single transaction."].memory_id,
        by_content["Take a restorable snapshot before a destructive migration."].memory_id,
    }


def test_the_calibrated_lesson_floor_is_anchored_on_both_sides(tmp_path, scored_semantic):
    """[A23.4] A23.2 found the calibrated 0.30 could drift to 0.31 or 0.32
    with the whole focused suite still green: the cap was pinned by an
    exact assertion, the floor was not.

    Anchored here behaviourally rather than by asserting the constant, and
    at +/- 0.001 rather than exactly at the boundary: the fake encoder's
    float32 round trip was measured at ~1e-8 (0.30 comes back as
    0.30000001), so 0.001 is three orders of magnitude clear of it while
    still catching any drift of 0.01.

    Each side gets its OWN workspace, deliberately. Put both lessons in
    one, and the MEMORY pool rejects the pair on a 0.002 margin -- which
    would make the lower side look excluded for a reason that has nothing
    to do with the Lesson floor. Isolated is also the shape the leak
    actually takes."""
    above = "Wrap multi-statement migrations in a single transaction."
    below = "Keep the changelog entry in the same commit as the change."
    scored_semantic({above: 0.3010, below: 0.2990})

    admitting = Urdyn.init(tmp_path / "above", "dev")
    admitted = _verified_lesson(admitting, above)
    admitting.semantic_setup()

    rejecting = Urdyn.init(tmp_path / "below", "dev")
    _verified_lesson(rejecting, below)
    rejecting.semantic_setup()

    assert _lesson_ids(admitting.preflight(TASK)) == {admitted.memory_id}, (
        "the calibrated floor drifted upward"
    )
    assert rejecting.preflight(TASK).verified_lessons == (), (
        "the calibrated floor drifted downward, or the Lesson pool is not authoritative"
    )


def test_the_memory_floor_still_admits_a_non_lesson_in_the_same_band(tmp_path, scored_semantic):
    """The Lesson floor did not become global. A root cause scoring 0.25 --
    the exact band the tests above reject a lesson in -- is still admitted
    end-to-end under the unchanged MEMORY policy, which is what makes this
    a category boundary rather than a raised threshold."""
    root_cause_content = "The loader accepted a boolean where an integer was expected."
    scored_semantic({root_cause_content: 0.25})

    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence("observed at load time", kind="error_observation")
    root_cause = cx.remember(
        root_cause_content, kind="root_cause", epistemic_state="inferred", evidence=[evidence]
    )
    cx.semantic_setup()

    result = cx.preflight(TASK)

    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}


# ---------------------------------------------------------------------------
# The policy function itself, isolated from Urdyn
# ---------------------------------------------------------------------------


def test_set_admitted_ids_applies_its_own_floor_and_no_margin():
    from urdyn._semantic import set_admitted_ids

    ranked = [("a", 0.3782), ("b", 0.3206), ("c", 0.27)]

    assert set_admitted_ids(ranked, floor=_FLOOR) == frozenset({"a", "b"})


def test_set_admitted_ids_caps_and_handles_degenerate_input():
    from urdyn._semantic import set_admitted_ids

    ranked = [("a", 0.6), ("b", 0.55), ("c", 0.5), ("d", 0.45)]

    assert set_admitted_ids(ranked, floor=_FLOOR) == frozenset({"a", "b"})
    assert set_admitted_ids(ranked, floor=_FLOOR, limit=1) == frozenset({"a"})
    assert set_admitted_ids(ranked, floor=_FLOOR, limit=0) == frozenset()
    assert set_admitted_ids(ranked, floor=0.9) == frozenset()
    assert set_admitted_ids([], floor=_FLOOR) == frozenset()


def test_the_lesson_floor_is_not_the_memory_floor():
    """[A23.2] The two floors answer different questions and are
    calibrated separately; wiring the Lesson pool back onto the MEMORY
    constant would silently undo this calibration."""
    from urdyn._semantic import set_admitted_ids

    ranked = [("noise", 0.27)]

    assert 0.27 >= _MEMORY_FLOOR
    assert set_admitted_ids(ranked, floor=_FLOOR) == frozenset()
    assert set_admitted_ids(ranked, floor=_MEMORY_FLOOR) == frozenset({"noise"})


def test_the_single_winner_policy_is_untouched():
    """The global winner+margin helper must still behave exactly as it
    did: A23.1 adds an alternative policy, it does not modify the one the
    MEMORY, ATTEMPT and SKILL pools still rely on."""
    from urdyn._semantic import semantic_admitted_id

    thin = [("a", 0.3782), ("b", 0.3206)]
    wide = [("a", 0.60), ("b", 0.20)]

    assert semantic_admitted_id(thin, ENTITY_MEMORY) is None
    assert semantic_admitted_id(wide, ENTITY_MEMORY) == "a"
    assert SEMANTIC_POLICY[ENTITY_MEMORY].margin_floor == 0.08
    assert SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor == 0.20


# ---------------------------------------------------------------------------
# Real-model integration: complementary lessons on the shipped backend.
# Skipped -- with a reason, never silently -- unless the pinned artifacts
# are already in the local Hugging Face cache. Never downloads anything.
# ---------------------------------------------------------------------------


def _real_model_available() -> bool:
    try:
        import huggingface_hub
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401

        from urdyn import _semantic
    except ImportError:
        return False
    for artifact in (_semantic.preferred_artifact(), _semantic.ARTIFACT_PORTABLE):
        try:
            for filename in (artifact, _semantic.TOKENIZER_FILENAME):
                huggingface_hub.hf_hub_download(
                    _semantic.SEMANTIC_MODEL_REPO,
                    filename,
                    revision=_semantic.SEMANTIC_MODEL_REVISION,
                    local_files_only=True,
                )
            return True
        except Exception:
            continue
    return False


_SKIP_REASON = (
    "the real ONNX semantic model is not cached locally (and/or the 'semantic' "
    "extra is not installed) -- run 'urdyn semantic setup' in a scratch "
    "workspace once to populate the Hugging Face cache, then re-run this file; "
    "never downloaded automatically by the test suite itself"
)
real_model = pytest.mark.real_model
skip_without_model = pytest.mark.skipif(not _real_model_available(), reason=_SKIP_REASON)


def _offline():
    class _Offline:
        def __enter__(self):
            self._previous = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            return self

        def __exit__(self, *exc_info):
            if self._previous is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = self._previous

    return _Offline()


_REAL_LESSON_A = (
    "Normalize environment values before applying numeric validation, because every "
    "environment variable arrives as a string."
)
_REAL_LESSON_B = (
    "Reject booleans explicitly before integer validation, because bool is a subclass "
    "of int in Python."
)
_REAL_UNRELATED = "Use design tokens instead of hard-coded hex colours in stylesheets."


def _real_workspace(cx):
    lesson_a = _verified_lesson(cx, _REAL_LESSON_A)
    lesson_b = _verified_lesson(cx, _REAL_LESSON_B)
    unrelated = _verified_lesson(cx, _REAL_UNRELATED)
    return lesson_a, lesson_b, unrelated


@real_model
@skip_without_model
def test_real_model_complementary_lessons_survive_together(tmp_path):
    """The A23 reproduction on the shipped backend, as a regression test.

    Two genuinely complementary verified lessons and a natural paraphrase
    of the task. On 2096197 the two scored 0.4372 and 0.3800 -- both far
    above the 0.20 floor -- and were rejected together for being 0.0572
    apart. No exact score is asserted here (scores are a property of the
    model, and the existing real-model tests only assert outcomes); what
    is asserted is that multiple genuinely useful lessons can now survive
    together, and that the unrelated one still does not.
    """
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        lesson_a, lesson_b, unrelated = _real_workspace(cx)
        cx.semantic_setup()

        result = cx.preflight("validate integer config value from env var")

        surfaced = _lesson_ids(result)
        assert lesson_a.memory_id in surfaced
        assert lesson_b.memory_id in surfaced
        assert unrelated.memory_id not in surfaced


@real_model
@skip_without_model
def test_real_model_unrelated_task_still_abstains(tmp_path):
    """The other half: bounded set retrieval must not become "always show
    something".

    The workspace holds only the two engineering lessons -- deliberately
    not the stylesheet one, which a styling question is genuinely about
    (measured: it scores 0.2235 against this task and is admitted, on
    2096197 exactly as here, because the absolute floor and the margin
    both pass it; that is pre-existing behaviour and not what this test
    is for).
    """
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        _verified_lesson(cx, _REAL_LESSON_A)
        _verified_lesson(cx, _REAL_LESSON_B)
        cx.semantic_setup()

        result = cx.preflight("Change the marketing landing page hero image to a dark variant.")

        assert result.verified_lessons == ()


@real_model
@skip_without_model
def test_real_model_cross_language_recovers_a_lesson_without_the_old_false_positive(tmp_path):
    """A16.3's multilingual backend is where the semantic channel matters
    most, because the lexical channel cannot help across languages at
    all. A cheap probe only, not a cross-language milestone.

    THE CASE THAT MOTIVATED A23.2'S CALIBRATION. Cross-language compresses
    this pool's scores asymmetrically: the Italian task pulls the directly
    relevant lesson to 0.6414 but leaves the second genuinely useful one
    at 0.1178, while the unrelated stylesheet lesson lands at 0.2363. Under
    A23.1's 0.20 floor that stylesheet lesson was admitted, and this test
    asserted it as a measured cost. The calibrated 0.30 Lesson floor now
    rejects it.

    What is NOT claimed: that the floor separated relevance. It did not --
    the second genuinely useful lesson (0.1178) stays out too, and A23.2
    measured that false positives have a HIGHER median score than genuine
    ones. The floor removed a low-band false positive, which is the one
    thing an absolute floor reliably does.
    """
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        lesson_a, lesson_b, unrelated = _real_workspace(cx)
        cx.semantic_setup()

        result = cx.preflight(
            "validare un valore di configurazione intero letto da variabile d'ambiente"
        )

        surfaced = _lesson_ids(result)
        assert lesson_a.memory_id in surfaced, "the directly relevant lesson must survive"
        assert unrelated.memory_id not in surfaced, "the A23.1 false positive must now be rejected"
        assert lesson_b.memory_id not in surfaced, (
            "still below the floor cross-language -- a signal limit, not a policy effect"
        )
