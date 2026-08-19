"""A31.2: dedicated semantic SET admission for Invariants in `context()`.

A29.1 gave `context()` its discriminating behaviour -- current
project-wide Invariants are filtered for task relevance instead of being
projected unconditionally the way `preflight()` projects them -- but let
that filtering reuse the MEMORY pool's single-winner admission
(`_semantic.semantic_admitted_id`: rank #1 only, absolute floor, margin
against rank #2). A30 found a real workspace where a relevant Invariant
never reached the CONSTRAINTS section; A31 located the exclusion exactly
(the semantic channel is the only one that can admit an Invariant whose
wording does not overlap the task, and that channel has capacity one);
A31.1 then measured the policy on a prospective corpus frozen before it
was scored -- 59 invariants over 22 domains in 6 project pools, 44 task
scenes in three languages, 11 of them scenes where emitting nothing is
correct -- with an offline harness verified to reproduce baseline
455fffa's selection on all 44 scenes.

What that measurement found is the reason this file exists, and it is not
the case that started the investigation:

- the MEMORY absolute floor rejected NOTHING in this pool (the rank #1
  candidate always cleared 0.40), so admission was decided by the margin
  alone;
- the margin alone left the CONSTRAINTS section empty on 27 of the 33
  scenes that had a genuinely applicable Invariant, and lost 21 of the 24
  critical constraints;
- the margin asks "is #1 separated enough from #2 to be THE answer",
  which is the wrong question here: 15 of 44 scenes have two or more
  co-relevant Invariants, and co-relevant constraints score close to each
  other by definition.

A31.2 therefore changes the ADMISSION POLICY of one category, exactly as
A23.1 did for Lessons: Invariants get `_semantic.set_admitted_ids` with
their own calibrated floor (0.35, the highest value staying under the
0.383 median relevant score, and the lowest preserving abstention exactly
as before) and cap (2, the last slot whose margin adds more signal than
noise), and NO margin check.

WHAT MUST NOT CHANGE, asserted here as well as in the suites it belongs
to: `preflight()`'s unconditional invariants (A9.1), the lexical / FTS /
provenance channels and their union, the Lesson floor (A23.2) and the
single-winner policy of every pool that still asks that question
(MEMORY, ATTEMPT, SKILL, and `context()`'s own Decision pool),
current-state eligibility, the budget, and section ordering.

WHAT THIS DOES NOT FIX, measured rather than hoped: precision (false
positives go from 2 to 22 as true positives go from 5 to 24), 15 of 24
critical constraints are still missed, and the original A30 wording is
still not recovered -- it ranks 8th of 10 for its own task, and admitting
it would cost the abstention this floor exists to protect. `preflight`
remains the complete, unconditional view; the compiler is a selection.

Like `test_a23_lesson_semantic_set_admission.py`, the policy tests below
use a deterministic fake encoder that places each text at an exact cosine
from the query, so every one of them is about the POLICY and never about
model luck. The `real_model` tests at the bottom replay a small subset of
the A31.1 corpus on the shipped backend -- the full 44-scene corpus stays
a calibration artefact, not a fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cortex_memory import Cortex
from cortex_memory._context import SECTION_CONSTRAINTS
from cortex_memory._retrieval import ENTITY_MEMORY
from cortex_memory._semantic import (
    INVARIANT_ADMISSION_LIMIT, INVARIANT_SEMANTIC_FLOOR, LESSON_SEMANTIC_FLOOR, SEMANTIC_POLICY,
)
from test_semantic_real_model import _offline, skip_without_model

real_model = pytest.mark.real_model

_FLOOR = INVARIANT_SEMANTIC_FLOOR                                # the Invariant pool's own, A31.1
_MEMORY_FLOOR = SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor    # untouched by A31.2
_MARGIN = SEMANTIC_POLICY[ENTITY_MEMORY].margin_floor            # untouched, and no longer consulted here

# A task whose vocabulary deliberately shares NOTHING with any invariant
# text below, so the lexical majority rule and the FTS channel both stay
# silent and these tests isolate the semantic channel -- the technique the
# A7.7/A7.8/A23.1 tests already use.
TASK = "kindly outline whichever precautions merit attention throughout tomorrow morning"


class _ScoredModel:
    """Deterministic encoder placing every text on the unit circle at an
    exact cosine from the query.

    `cosines` maps a text to its intended similarity with the query; the
    query itself is mapped to 1.0. Anything not listed is orthogonal to
    the query rather than near-parallel -- a near-zero vector would
    L2-normalize back up into a spurious perfect match.
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
    """Install a `_ScoredModel` built from a caller-supplied cosine map,
    so each test states the exact score geometry it is about."""
    import cortex_memory._semantic as semantic

    def install(cosines: dict[str, float]):
        model = _ScoredModel(cosines)
        monkeypatch.setattr(semantic, "load_model_for_setup", lambda *a, **k: model)
        monkeypatch.setattr(semantic, "load_model_for_retrieval", lambda *a, **k: model)
        monkeypatch.setattr(semantic, "resolve_local_revision", lambda *a, **k: "fake-revision")
        return model

    return install


_BIG_BUDGET = 200_000


def _constraint_ids(compiled) -> set[str]:
    return {
        item.entity_id
        for section in compiled.sections
        if section.heading == SECTION_CONSTRAINTS
        for item in section.items
    }


# ---------------------------------------------------------------------------
# THE DISCRIMINATING TEST AGAINST THE PREVIOUS POLICY
# ---------------------------------------------------------------------------


def test_a_relevant_invariant_is_admitted_when_a_near_tie_previously_silenced_the_section(
    tmp_path, scored_semantic
):
    """A31.1's margin-collapse case, as a policy test.

    The score geometry is the real one measured on scene `S20` of the
    frozen corpus, not a constructed edge case: a dispatcher task whose
    applicable constraint ("the dispatcher loop performs no blocking I/O")
    scores 0.5664, with a non-applicable delivery guarantee 0.0132 behind
    it. Both clear every floor in the codebase comfortably. On baseline
    455fffa the compiler returns ZERO constraints for that task -- not
    because it has none, but because the runner-up is too close for a
    policy that assumes exactly one candidate can be right.

    Behavioural, not structural: what is asserted is the compiled
    CONSTRAINTS section, not the existence of a helper.
    """
    relevant = "The dispatcher loop performs no blocking I/O other than the queue poll itself."
    near_tie = "Delivery is at-least-once and every handler must be idempotent."
    scored_semantic({relevant: 0.5664, near_tie: 0.5532})

    cx = Cortex.init(tmp_path, "dev")
    admitted = cx.remember(relevant, kind="invariant")
    cx.remember(near_tie, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    # premises, so this can never pass vacuously
    assert 0.5664 >= _FLOOR and 0.5532 >= _FLOOR, "premise: both clear the calibrated Invariant floor"
    assert 0.5664 - 0.5532 < _MARGIN, "premise: the gap is inside the margin the old policy applied"
    assert admitted.memory_id in _constraint_ids(compiled)


def test_two_co_relevant_invariants_are_admitted_together(tmp_path, scored_semantic):
    """The other half of the same change: set admission means genuinely
    co-relevant constraints enter TOGETHER, without having to defeat each
    other. A CONSTRAINTS section that can hold one constraint is a
    property of the policy, never of the domain."""
    first = "A job runs on exactly one worker at a time, enforced by a database row lock."
    second = "A state transition is written in the same transaction as the side effect it describes."
    scored_semantic({first: 0.52, second: 0.48})

    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember(first, kind="invariant")
    b = cx.remember(second, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert 0.52 - 0.48 < _MARGIN, "premise: the old policy would have rejected both"
    assert _constraint_ids(compiled) == {a.memory_id, b.memory_id}
    assert compiled.invariants_excluded == 0


# ---------------------------------------------------------------------------
# THE CAP AND THE FLOOR, EACH ISOLATED
# ---------------------------------------------------------------------------


def test_semantic_invariant_admission_is_capped_at_the_calibrated_limit(tmp_path, scored_semantic):
    """Four invariants, all above the floor, all mutually within the
    margin: exactly `INVARIANT_ADMISSION_LIMIT` are admitted. The cap is
    what replaces the margin's incidental bound on how much one channel
    may add, and A31.1 measured the third slot costing 2.5 false positives
    per true positive. It is a guard, not a target to fill."""
    texts = {
        "A job runs on exactly one worker at a time.": 0.60,
        "Cancellation is advisory and never interrupts a running handler.": 0.55,
        "Scheduled times are stored and compared as UTC instants.": 0.50,
        "The queue table is the single source of truth for job state.": 0.45,
    }
    scored_semantic(texts)

    cx = Cortex.init(tmp_path, "dev")
    recorded = [cx.remember(text, kind="invariant") for text in texts]
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert INVARIANT_ADMISSION_LIMIT == 2, "this test states the cap it is asserting"
    assert all(score >= _FLOOR for score in texts.values()), (
        "premise: the excluded invariants are excluded by the CAP, not by the floor"
    )
    assert _constraint_ids(compiled) == {recorded[0].memory_id, recorded[1].memory_id}
    assert compiled.invariants_excluded == 2


def test_an_invariant_below_the_calibrated_floor_is_not_admitted(tmp_path, scored_semantic):
    """Removing the margin did not remove abstention: the floor is now the
    only thing that can produce it, and it still does.

    0.32 is placed in the band A31.1 measured as almost pure noise for
    this pool -- above the untouched MEMORY floor, above the Lesson floor,
    below the Invariant one. That the Lesson floor would admit it is
    asserted deliberately: it is what makes the Invariant floor a
    category-specific contract rather than a reused constant.
    """
    strong = "A job runs on exactly one worker at a time, enforced by a database row lock."
    weak = "Scheduled times are stored and compared as UTC instants."
    scored_semantic({strong: 0.55, weak: 0.32})

    cx = Cortex.init(tmp_path, "dev")
    admitted = cx.remember(strong, kind="invariant")
    rejected = cx.remember(weak, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert _MEMORY_FLOOR < 0.32 and LESSON_SEMANTIC_FLOOR <= 0.32 < _FLOOR, (
        "premise: only the Invariant-specific floor rejects this candidate"
    )
    assert _constraint_ids(compiled) == {admitted.memory_id}
    assert rejected.memory_id not in _constraint_ids(compiled)
    assert compiled.invariants_excluded == 1


def test_an_invariant_sharing_the_task_topic_without_binding_it_stays_out(tmp_path, scored_semantic):
    """The false-positive pressure case A31.1 designed for: a real,
    current, well-written constraint from the same project that scores in
    the noise band because it shares the task's subject matter without
    binding the task. Neither its currency nor its authority buys it
    admission.

    This is what the floor can do. What it cannot do is separate
    applicability from similarity -- A31.1 measured 22 false positives at
    this operating point, several of them scoring above genuinely
    applicable constraints. That limitation is documented, not tested
    away here.
    """
    binding = "Cancellation is advisory and never interrupts a handler that is already running."
    topical = "The CLI writes its machine-readable report to stdout and diagnostics to stderr."
    scored_semantic({binding: 0.51, topical: 0.29})

    cx = Cortex.init(tmp_path, "dev")
    admitted = cx.remember(binding, kind="invariant")
    noise = cx.remember(topical, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert _constraint_ids(compiled) == {admitted.memory_id}
    assert noise.memory_id not in _constraint_ids(compiled)


def test_an_unrelated_task_admits_no_invariant_at_all(tmp_path, scored_semantic):
    """Abstention survives the change, and A31.1 measured it exactly: on
    the 11 corpus scenes where emitting nothing is correct, this policy is
    right on 10 -- the same 10 as the previous policy, with the same
    single false alarm. Recall was bought from the margin, not from the
    floor."""
    texts = {
        "A job runs on exactly one worker at a time.": 0.24,
        "Cancellation is advisory and never interrupts a running handler.": 0.19,
        "Scheduled times are stored and compared as UTC instants.": 0.11,
    }
    scored_semantic(texts)

    cx = Cortex.init(tmp_path, "dev")
    for text in texts:
        cx.remember(text, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert _constraint_ids(compiled) == set()
    assert compiled.invariants_excluded == 3


# ---------------------------------------------------------------------------
# CURRENT STATE, THE OTHER CHANNELS, AND THE OTHER CATEGORIES
# ---------------------------------------------------------------------------


def test_a_superseded_invariant_is_not_admitted_by_a_high_score(tmp_path, scored_semantic):
    """Current state is not relevance, and no score can override it: the
    eligible pool is restricted BEFORE ranking (A7.7), so a superseded
    invariant scoring 0.99 is not a candidate at all, while its
    replacement is admitted normally. This property belongs to the pool,
    not to the policy -- it must hold at every floor and cap."""
    old_text = "Every destination lies at or below the extraction root."
    new_text = "Every destination lies STRICTLY below the extraction root; the root is not a legal destination."
    scored_semantic({old_text: 0.99, new_text: 0.40})

    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember(old_text, kind="invariant")
    new = cx.remember(new_text, kind="invariant", supersedes=old.memory_id)
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert _constraint_ids(compiled) == {new.memory_id}
    assert old.memory_id not in _constraint_ids(compiled)
    # history is preserved -- the old invariant left operational
    # retrieval, not the record
    assert old.memory_id in {m.memory_id for m in cx.timeline()}


def test_lexical_admission_is_unchanged_without_a_semantic_index(tmp_path):
    """A31.2 is additive to semantic recall only. With no `semantic setup`
    at all -- no fake model, no index -- `context()` behaves exactly as
    before: the lexical channel admits what it always admitted, and the
    semantic channel contributes nothing rather than raising.

    The honest half of that contract, measured in A31.1: without the
    semantic channel this pool is close to inert (zero invariants admitted
    across all 44 corpus scenes), because an invariant enters lexically
    only when the task restates its vocabulary. That is what lexical
    retrieval cannot do, not a regression to fix here -- and `preflight`
    still shows every current invariant unconditionally.
    """
    task = "Add retry handling for failed background jobs in the queue"
    relevant = "Retry handling for failed background jobs must never reorder the queue"
    unrelated = "All commit messages must be written in English"

    cx = Cortex.init(tmp_path, "dev")
    admitted = cx.remember(relevant, kind="invariant")
    excluded = cx.remember(unrelated, kind="invariant")

    compiled = cx.context(task, budget=_BIG_BUDGET)

    assert _constraint_ids(compiled) == {admitted.memory_id}
    assert excluded.memory_id not in _constraint_ids(compiled)


def test_a_lexically_relevant_invariant_below_the_floor_is_still_admitted(tmp_path, scored_semantic):
    """The channels are unioned, never intersected: an invariant whose
    semantic score is far below the floor still enters through the
    lexical channel, exactly as before. The new policy narrows one
    channel's output, never the field."""
    task = "Add retry handling for failed background jobs in the queue"
    lexical = "Retry handling for failed background jobs must never reorder the queue"
    scored_semantic({lexical: 0.05})

    cx = Cortex.init(tmp_path, "dev")
    invariant = cx.remember(lexical, kind="invariant")
    cx.semantic_setup()

    compiled = cx.context(task, budget=_BIG_BUDGET)

    assert 0.05 < _FLOOR, "premise: the semantic channel rejects this outright"
    assert _constraint_ids(compiled) == {invariant.memory_id}


def test_the_semantic_cap_does_not_truncate_lexically_relevant_invariants(tmp_path, scored_semantic):
    """The cap bounds ONE channel, never the section. Four invariants all
    lexically relevant to the same task must all be compiled: reading
    `INVARIANT_ADMISSION_LIMIT` as "context shows at most two constraints"
    would silently narrow the pre-existing lexical contract."""
    task = "Add retry handling for failed background jobs in the queue"
    texts = [
        "Retry handling for failed background jobs must never reorder the queue",
        "Retry handling for failed background jobs must stay idempotent in the queue",
        "Retry handling for failed background jobs must not block the queue loop",
        "Retry handling for failed background jobs must record every queue attempt",
    ]
    scored_semantic({})  # nothing scores at all: the lexical channel alone must carry this

    cx = Cortex.init(tmp_path, "dev")
    recorded = [cx.remember(text, kind="invariant") for text in texts]
    cx.semantic_setup()

    compiled = cx.context(task, budget=_BIG_BUDGET)

    assert len(_constraint_ids(compiled)) == 4 > INVARIANT_ADMISSION_LIMIT
    assert _constraint_ids(compiled) == {m.memory_id for m in recorded}


def test_preflight_still_projects_every_current_invariant(tmp_path, scored_semantic):
    """A9.1's contract is untouched, and the separation is the point:
    `preflight()` is the complete checklist of current invariants,
    `context()` is a task-relevant selection under a budget. The new
    policy has exactly one caller and it is not preflight -- an invariant
    the compiler abstains from must still be visible there.
    """
    admitted_text = "A job runs on exactly one worker at a time."
    below_floor = "Scheduled times are stored and compared as UTC instants."
    unrelated = "The CLI exit code is 0 on success and 2 on a usage error."
    scored_semantic({admitted_text: 0.55, below_floor: 0.31, unrelated: 0.02})

    cx = Cortex.init(tmp_path, "dev")
    recorded = [cx.remember(text, kind="invariant") for text in (admitted_text, below_floor, unrelated)]
    cx.semantic_setup()

    result = cx.preflight(TASK)
    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert {m.memory_id for m in result.invariants} == {m.memory_id for m in recorded}
    assert _constraint_ids(compiled) == {recorded[0].memory_id}
    assert compiled.invariants_excluded == 2


def test_the_decision_pool_keeps_the_single_winner_policy(tmp_path, scored_semantic):
    """The new policy must not leak into the sibling pool it sits next to.
    Two Decisions inside the margin are still BOTH rejected -- a Decision
    records what was chosen among alternatives, so "which single candidate
    is the intended one" remains the right question there. The same
    geometry admits both Invariants."""
    invariant_a = "A job runs on exactly one worker at a time."
    invariant_b = "Cancellation is advisory and never interrupts a running handler."
    decision_a = "Background retries use exponential backoff capped at six attempts."
    decision_b = "Background retries use a fixed delay of thirty seconds."
    scored_semantic({invariant_a: 0.55, invariant_b: 0.52, decision_a: 0.55, decision_b: 0.52})

    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember(invariant_a, kind="invariant")
    b = cx.remember(invariant_b, kind="invariant")
    cx.remember(decision_a, kind="decision")
    cx.remember(decision_b, kind="decision")
    cx.semantic_setup()

    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    decision_ids = {
        item.entity_id
        for section in compiled.sections
        if section.heading == "DECISIONS"
        for item in section.items
    }
    assert _constraint_ids(compiled) == {a.memory_id, b.memory_id}
    assert decision_ids == set(), "the Decision pool still applies the single-winner margin"


def test_lesson_and_pending_admission_are_untouched(tmp_path, scored_semantic):
    """The Lesson floor calibrated in A23.2 and A22.1's disjoint pending
    pool keep their own behaviour: a lesson at 0.32 is admitted (its floor
    is 0.30) while an invariant at the same score is not (its floor is
    0.35). Two categories, two calibrations, neither reading the other's
    constant."""
    lesson_text = "Retried writes must carry a stable idempotency key."
    pending_text = "The dead-letter table still has no retention policy."
    invariant_text = "Scheduled times are stored and compared as UTC instants."
    scored_semantic({lesson_text: 0.32, pending_text: 0.32, invariant_text: 0.32})

    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("pytest -> 3 passed", kind="test_result")
    lesson = cx.learn(lesson_text, verified=True, supporting_evidence=[evidence])
    pending = cx.remember(pending_text, kind="pending")
    invariant = cx.remember(invariant_text, kind="invariant")
    cx.semantic_setup()

    result = cx.preflight(TASK)
    compiled = cx.context(TASK, budget=_BIG_BUDGET)

    assert LESSON_SEMANTIC_FLOOR <= 0.32 < _FLOOR, "premise: the two floors disagree about this score"
    assert lesson.memory_id in {m.memory_id for m in result.verified_lessons}
    assert pending.memory_id in {m.memory_id for m in result.pending}
    assert invariant.memory_id not in _constraint_ids(compiled)


# ---------------------------------------------------------------------------
# BUDGET, DETERMINISM, CANONICAL STATE
# ---------------------------------------------------------------------------


def test_an_admitted_invariant_still_competes_for_budget(tmp_path, scored_semantic):
    """Admission is not inclusion. A second admitted invariant that does
    not fit is reported as omitted FOR BUDGET, never as excluded for
    irrelevance -- the distinction A30 relied on to diagnose this defect
    in the first place, and the reason invariants are not budget-exempt.
    """
    first = "A job runs on exactly one worker at a time, enforced by a database row lock that survives a crash."
    second = "Cancellation is advisory and never interrupts a handler that is already running to completion."
    scored_semantic({first: 0.60, second: 0.55})

    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember(first, kind="invariant")
    b = cx.remember(second, kind="invariant")
    cx.semantic_setup()

    generous = cx.context(TASK, budget=_BIG_BUDGET)
    assert _constraint_ids(generous) == {a.memory_id, b.memory_id}, "premise: both are admitted"

    # Exactly the heading plus the first constraint's own rendered line,
    # taken from the generous render so the arithmetic cannot drift from
    # the renderer.
    first_line = next(line for line in generous.render().splitlines() if a.memory_id in line)
    tight = cx.context(TASK, budget=len(SECTION_CONSTRAINTS) + 1 + len(first_line) + 1)

    assert _constraint_ids(tight) == {a.memory_id}
    assert tight.omitted >= 1
    assert tight.invariants_excluded == 0, "the second one was dropped by the budget, not by relevance"
    assert _constraint_ids(tight) <= _constraint_ids(generous), "prefix monotonicity"


def test_invariant_admission_is_deterministic(tmp_path, scored_semantic):
    """Same task, same canonical state, same index: the same selection and
    the same rendered bytes. `set_admitted_ids` never re-sorts and never
    breaks ties itself, so the cap's contents are exactly as deterministic
    as the ranking that produced them."""
    texts = {
        "A job runs on exactly one worker at a time.": 0.60,
        "Cancellation is advisory and never interrupts a running handler.": 0.55,
        "Scheduled times are stored and compared as UTC instants.": 0.50,
    }
    scored_semantic(texts)

    cx = Cortex.init(tmp_path, "dev")
    for text in texts:
        cx.remember(text, kind="invariant")
    cx.semantic_setup()

    renders = [cx.context(TASK, budget=_BIG_BUDGET).render() for _ in range(5)]

    assert all(render == renders[0] for render in renders)


def test_context_never_mutates_canonical_state(tmp_path, scored_semantic):
    """`CompiledContext` is derived: repeated compilation must leave the
    event log and the current-state view byte-identical."""
    texts = {
        "A job runs on exactly one worker at a time.": 0.60,
        "Scheduled times are stored and compared as UTC instants.": 0.25,
    }
    scored_semantic(texts)

    cx = Cortex.init(tmp_path, "dev")
    for text in texts:
        cx.remember(text, kind="invariant")
    cx.semantic_setup()

    before = ([m.memory_id for m in cx.timeline()], [m.memory_id for m in cx.state()])
    for _ in range(3):
        cx.context(TASK, budget=_BIG_BUDGET)
    after = ([m.memory_id for m in cx.timeline()], [m.memory_id for m in cx.state()])

    assert before == after


# ---------------------------------------------------------------------------
# A31.1 CORPUS REGRESSION SUBSET (real model only)
#
# A small, justified subset of the frozen corpus -- margin collapse,
# multi-invariant, abstention, multilingual, long task, superseded -- not
# the 44 scenes, which remain a calibration artefact. Texts and tasks are
# verbatim from `a31.1-v1`. Skipped, with a reason, unless the pinned
# artifacts are already cached locally; never downloads anything.
# ---------------------------------------------------------------------------

_P3 = {
    "P3-I01": "Jobs of equal priority are dispatched in submission order; the scheduler never reorders them, so a client that observes submission order can predict execution order.",
    "P3-I02": "Delivery is at-least-once and every handler must be idempotent: after a worker crash a job is redelivered rather than silently dropped.",
    "P3-I04": "A job's state transition is written in the same database transaction as the record of the side effect it describes; there is no window in which a job is marked done without its effect being durable.",
    "P3-I05": "Worker threads never hold the scheduler lock while executing a handler; that lock protects the queue structures only.",
    "P3-I07": "Cancellation is advisory: it prevents future dispatch but never interrupts a handler that is already running.",
    "P3-I08": "Retry backoff is exponential with full jitter and a hard cap of six attempts; after that the job moves to the dead-letter table and is never retried automatically.",
    "P3-I10": "The dispatcher loop performs no blocking I/O other than the queue poll itself; anything that can wait runs in the worker pool.",
}
_P3_SUPERSEDED = "Failed jobs are retried after a fixed thirty-second delay, with no limit on the number of attempts."

_S20_TASK = (
    "The dispatcher currently blocks while it writes the audit row for a job it has just picked up, "
    "and under load the whole loop stalls behind that write. Move that work off the hot path, but do "
    "not change the moment at which a job becomes visible as running to the other workers."
)
_S21_TASK = "Aggiungere un comando per annullare un job che sta gia girando"


def _p3_workspace(cx):
    """The P3 pool of the frozen corpus, including its real supersession
    (`P3-I11` replaced by `P3-I08`), recorded through the public API."""
    superseded = cx.remember(_P3_SUPERSEDED, kind="invariant")
    recorded = {}
    for invariant_id, text in _P3.items():
        supersedes = superseded.memory_id if invariant_id == "P3-I08" else None
        recorded[invariant_id] = cx.remember(text, kind="invariant", supersedes=supersedes)
    return recorded, superseded


@real_model
@skip_without_model
def test_real_model_margin_collapse_scene_now_delivers_its_constraint(tmp_path):
    """Corpus scene `S20`, on the shipped backend: the non-vacuity anchor
    of this tracer, measured on baseline 455fffa before any code was
    written (CONSTRAINTS empty, `invariants_excluded` 10, with `P3-I10` at
    0.5664 and the runner-up 0.0132 behind).

    What is asserted is the outcome, never a score: the section is no
    longer empty and it carries the applicable constraint. Deliberately
    NOT asserted, because A31.1 measured it and this tracer does not fix
    it: the second slot on this scene goes to a false positive, and the
    scene's critical constraint `P3-I04` is not recovered. Ranking quality
    is a separate problem from admission policy.
    """
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        recorded, superseded = _p3_workspace(cx)
        cx.semantic_setup()

        compiled = cx.context(_S20_TASK, budget=_BIG_BUDGET)

        selected = _constraint_ids(compiled)
        assert selected, "the compiler must no longer deliver an empty CONSTRAINTS section here"
        assert recorded["P3-I10"].memory_id in selected
        assert len(selected) <= INVARIANT_ADMISSION_LIMIT
        assert superseded.memory_id not in selected


@real_model
@skip_without_model
def test_real_model_italian_task_recovers_an_english_invariant(tmp_path):
    """Corpus scene `S21`: an Italian task against English invariants,
    where the lexical channel cannot help at all. Its single expected
    constraint is also a critical one -- an implementation that reads
    cancellation as "kill the running handler" violates it."""
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        recorded, _ = _p3_workspace(cx)
        cx.semantic_setup()

        compiled = cx.context(_S21_TASK, budget=_BIG_BUDGET)

        assert recorded["P3-I07"].memory_id in _constraint_ids(compiled)


@real_model
@skip_without_model
def test_real_model_unrelated_task_in_a_full_pool_still_abstains(tmp_path):
    """The abstention half, on a realistic pool: a task about neither
    scheduling nor persistence, against ten current invariants that all
    describe a queue. The floor must still be able to say nothing
    applies."""
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        _p3_workspace(cx)
        cx.semantic_setup()

        compiled = cx.context(
            "Replace the marketing site hero image with a dark variant and update the alt text",
            budget=_BIG_BUDGET,
        )

        assert _constraint_ids(compiled) == set()
        assert compiled.invariants_excluded == len(_P3)


_P1_I02 = (
    "Every destination returned by build_plan, in ExtractionPlan.files and in ExtractionPlan.directories "
    "alike, lies STRICTLY below ExtractionPlan.root: the root itself is not a legal destination. build_plan "
    "raises UnsafeEntryError instead of ever returning an uncontained destination, so no consumer needs to "
    "re-check containment."
)
_P1_I01 = (
    "build_plan is a pure function of (Manifest, root string): it performs no filesystem access, does not "
    "stat, does not create directories and does not write. Callers may build a plan for a root that does not "
    "exist yet."
)
_P1_OTHERS = (
    "Manifest entry paths are POSIX-style and slash-separated; a backslash anywhere in an entry path is rejected by the manifest validator.",
    "The order of entries in ExtractionPlan.files and ExtractionPlan.directories is the manifest declaration order; the planner never sorts them.",
    "Every error raised by the package derives from SafeExtractError; no standard-library exception ever escapes the public API unwrapped.",
    "The CLI writes its machine-readable report to stdout and every human-facing diagnostic to stderr, so stdout stays pipeable.",
    "safeextract depends on the Python standard library only; no third-party runtime dependency may be introduced.",
    "Log files are opened in append mode and never truncated, so a concurrent run cannot lose another run's lines.",
)

_S03_TASK = (
    "We want to ship the extraction step in two phases. In this first phase I need a command that takes a "
    "manifest and an extraction root and prints, for every entry, the destination it would be written to, "
    "together with a summary of how many files and directories the run would produce. Nothing should be "
    "created on disk in this phase, not even the extraction root itself. All the safety checks we already "
    "have must keep applying, and the report must not claim more safety than we actually provide."
)


@real_model
@skip_without_model
def test_real_model_long_task_delivers_its_critical_constraint(tmp_path):
    """Corpus scene `S03`, the long realistic formulation of the A30
    domain, and the slice where A31.1 measured the largest gain (11 true
    positives against 5 false positives over 9 long scenes, from a
    baseline of 1). Its critical constraint is the containment guarantee.

    The A30 wording itself (`P1-I01`, the purity constraint) is present in
    this workspace and is NOT expected to be recovered: it ranks 8th of 10
    for its own task, and admitting it would require a floor that A31.1
    measured as destroying abstention. That miss is documented in the
    module docstring rather than frozen into an assertion, so a future
    improvement in ranking is free to fix it.
    """
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        containment = cx.remember(_P1_I02, kind="invariant")
        cx.remember(_P1_I01, kind="invariant")
        for text in _P1_OTHERS:
            cx.remember(text, kind="invariant")
        cx.semantic_setup()

        compiled = cx.context(_S03_TASK, budget=_BIG_BUDGET)

        assert containment.memory_id in _constraint_ids(compiled)


@real_model
@skip_without_model
def test_real_model_context_still_reuses_a27_auto_refresh(tmp_path):
    """The A27 lifecycle is unchanged and still owns index freshness: an
    invariant recorded after `semantic setup` leaves the index stale, and
    the first `context()` call refreshes it once through the existing
    consumer-boundary path -- this tracer added no second preparation
    step, no second encoding pass and no second index."""
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        cx.semantic_setup()
        invariant = cx.remember(_P3["P3-I07"], kind="invariant")

        assert cx.semantic_state().status == "stale"

        compiled = cx.context(_S21_TASK, budget=_BIG_BUDGET)

        assert compiled.retrieval is not None
        assert compiled.retrieval.refreshed > 0
        assert invariant.memory_id in _constraint_ids(compiled)
        assert cx.semantic_state().status == "ready"
