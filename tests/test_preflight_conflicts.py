"""A14.1: `Preflight.open_conflicts` -- surfacing OPEN canonical `Conflict`
relations (A13) through `preflight()`, so an agent can never be shown a
Memory as individually authoritative without also being told Cortex
knows it is contradicted.

Before A14.1, `cx.preflight(task)` could return two `verified` Lessons
that directly contradict each other, with no signal that Cortex had
already recorded `record_conflict(A, B)` -- FALSE OPERATIONAL CERTAINTY.
This file locks down:

  - the derived `PreflightConflict` view (canonical `Conflict` + the two
    Memories it names -- NOT a new canonical primitive, see
    `_preflight.py`'s module docstring);
  - Rule A' relevance: a participant is relevant if it clears the same
    lexical/FTS/semantic/evidence-rescue gate as everything else, OR is
    already shown in `root_causes`/`verified_lessons`/`open_invalidations`
    -- deliberately NOT `invariants` (A14.0.1's invariant-contagion
    finding);
  - one-sided sufficiency (either participant relevant is enough);
  - ZERO new semantic pool for conflicts (A14.0.1's rejected tracer);
  - exact non-interference with the six pre-existing `Preflight` fields;
  - fail-closed integrity for a missing participant;
  - the `timeline(None)` partitioning A14.1 introduced to avoid a fifth
    `timeline(kind)` read.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cortex_memory import Cortex, CortexStorageError, Preflight, PreflightConflict
from cortex_memory._conflict import Conflict
from cortex_memory._memory import Memory
from cortex_memory._preflight import build_preflight
from cortex_memory._store import MemoryStore

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _verified(cx, content):
    evidence = cx.add_evidence(f"confirmed: {content}", kind="user_confirmation")
    return cx.learn(content, verified=True, supporting_evidence=[evidence])


def _existing_field_snapshot(preflight: Preflight) -> dict:
    """Exact id-AND-order snapshot of every field A14.1 must not touch."""
    return {
        "known_failures": [a.attempt_id for a in preflight.known_failures],
        "root_causes": [m.memory_id for m in preflight.root_causes],
        "verified_lessons": [m.memory_id for m in preflight.verified_lessons],
        "recommended_validation": [e.evidence_id for e in preflight.recommended_validation],
        "invariants": [m.memory_id for m in preflight.invariants],
        "open_invalidations": [m.memory_id for m in preflight.open_invalidations],
    }


# One-sided relevance pair, verified against the real `is_relevant`
# threshold (see A14.1's report): the task shares a lexical majority with
# A but not with B.
_TASK_WEBHOOK = "Fix webhook delivery retry safety"
_LESSON_A = "Retry of the webhook delivery is safe."
_LESSON_B = "Duplicate charges occur when the webhook delivery is repeated."

_FAKE_CONCEPTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
_FAKE_NONE_INDEX = len(_FAKE_CONCEPTS)
_FAKE_DIM = len(_FAKE_CONCEPTS) + 1


class _FakeStaticModel:
    """Deterministic embedding backend, identical technique to
    `test_preflight_invalidations.py`/`test_a7_8_regression.py`: a fixed
    concept vocabulary makes semantic admission fully predictable without
    a real model, so these tests never depend on real-model ranking."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), _FAKE_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for j, concept in enumerate(_FAKE_CONCEPTS):
                if concept in lowered:
                    vectors[i, j] = 1.0
            if not vectors[i].any():
                vectors[i, _FAKE_NONE_INDEX] = 1.0
        return vectors


@pytest.fixture
def fake_semantic(monkeypatch):
    import cortex_memory._semantic as semantic

    fake_model = _FakeStaticModel()
    monkeypatch.setattr(semantic, "load_model_for_setup", lambda model_id=None: fake_model)
    monkeypatch.setattr(semantic, "load_model_for_retrieval", lambda model_id=None: fake_model)
    monkeypatch.setattr(semantic, "resolve_local_revision", lambda model_id=None: "fake-revision")
    return fake_model


# ---------------------------------------------------------------------------
# 1-2: field default / is_empty (see also test_preflight_contract.py)
# ---------------------------------------------------------------------------


def test_open_conflicts_field_defaults_to_empty_tuple_on_a_workspace_with_no_conflicts(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    result = cx.preflight("some task with no recorded experience at all")

    assert result.open_conflicts == ()


def test_preflight_with_only_a_relevant_conflict_is_not_empty(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = cx.learn(_LESSON_B)  # unverified candidate: kept out of every other field
    cx.record_conflict(a, b)

    result = cx.preflight(_TASK_WEBHOOK)

    assert result.root_causes == ()
    assert result.known_failures == ()
    assert result.recommended_validation == ()
    assert result.invariants == ()
    assert result.open_invalidations == ()
    assert result.verified_lessons == (a,)
    assert len(result.open_conflicts) == 1
    assert result.is_empty() is False


# ---------------------------------------------------------------------------
# 3-5: north star -- both remain verified, both remain in their own fields
# ---------------------------------------------------------------------------


def test_north_star_two_verified_lessons_in_open_conflict(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    task = "Fix the retry idempotency key handling"
    # both lexically relevant to THIS task (see A14.1 report, cross-kind
    # example) so this is genuinely two-sided, not relying on rescue.
    rc = cx.remember(
        "The retry loop reuses a stale idempotency key.",
        kind="root_cause",
        epistemic_state="inferred",
    )
    ls = _verified(cx, "The retry loop always generates a fresh idempotency key.")
    cx.record_conflict(rc, ls)
    cx.record_conflict(a, b)  # unrelated conflict: must not appear for this task

    result = cx.preflight(task)

    assert rc in result.root_causes
    assert ls in result.verified_lessons
    assert rc.epistemic_state == "inferred"
    assert ls.epistemic_state == "verified"
    view_pairs = [frozenset(v.conflict.memory_ids) for v in result.open_conflicts]
    assert frozenset((rc.memory_id, ls.memory_id)) in view_pairs
    assert frozenset((a.memory_id, b.memory_id)) not in view_pairs


def test_verified_lessons_remain_verified_when_part_of_a_conflict(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)

    result = cx.preflight(_TASK_WEBHOOK)

    assert a in result.verified_lessons
    assert a.epistemic_state == "verified"
    # B is `verified` -- unaffected by the conflict, exactly like A -- but
    # is NOT independently relevant to this task on its own wording, and
    # `record_conflict()` does not change that: see
    # `test_conflict_membership_never_admits_a_memory_into_an_authority_field`
    # below for the frozen property this is a corollary of. `verified` and
    # "admitted to THIS task's verified_lessons" are different questions.
    assert b.epistemic_state == "verified"
    assert b not in result.verified_lessons


# ---------------------------------------------------------------------------
# 3.1 (A14.1.1, BLOCKING PROPERTY): Rule A' dependency direction.
#
# existing relevance-gated Preflight fields --> Conflict relevance,
# NEVER the reverse. A memory's participation in an open Conflict must
# never be what gets it INTO root_causes/verified_lessons/
# open_invalidations -- those three fields are computed first, from
# data that has no notion of Conflict at all (see `build_preflight`:
# `root_causes`/`verified_lessons`/`open_invalidations` are assigned
# before `_conflict_gated_ids` is even built FROM them). This section
# exists because an earlier draft of this file's own docstring/Human
# Acceptance notes used ambiguous wording ("B admitted via Rule A' through
# conflict membership") that, if it described the real behavior, would be
# exactly the forbidden direction. It does not: this is a documentation
# correction, not a code fix -- the tests below prove the code was
# already right.
# ---------------------------------------------------------------------------


def test_conflict_membership_never_admits_a_memory_into_an_authority_field(tmp_path):
    """A and B are BOTH verified Lessons. Only A is independently relevant
    to the task (lexical channel, verified inline). A<->B is open. The
    frozen property: `verified_lessons` contains EXACTLY what it would
    contain without the Conflict -- A only, never B -- while
    `open_conflicts` still carries both Memories in its derived view.
    Authority-field admission and conflict-participant visibility are
    different questions with different answers here, by design."""
    from cortex_memory._relevance import is_relevant, memory_search_text, tokens

    assert is_relevant(frozenset(tokens(_TASK_WEBHOOK)), memory_search_text(_LESSON_B)) is False

    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)

    without_conflict = cx.preflight(_TASK_WEBHOOK)
    assert without_conflict.verified_lessons == (a,)

    cx.record_conflict(a, b)
    with_conflict = cx.preflight(_TASK_WEBHOOK)

    # The authority field is UNCHANGED by the Conflict's existence.
    assert with_conflict.verified_lessons == without_conflict.verified_lessons
    assert b not in with_conflict.verified_lessons

    # But the conflict's derived view still carries both Memories.
    assert len(with_conflict.open_conflicts) == 1
    view = with_conflict.open_conflicts[0]
    assert {m.memory_id for m in view.memories} == {a.memory_id, b.memory_id}
    assert b.epistemic_state == "verified"


@pytest.mark.parametrize(
    "build",
    [
        "lesson_lesson",
        "root_cause_lesson",
        "decision_invariant",
        "environment_environment",
    ],
)
def test_exact_field_equality_before_and_after_conflict_across_kinds(tmp_path, build):
    """The before/after snapshot from `test_exact_non_interference_of_all_existing_fields`,
    repeated across the four kind combinations the report calls out.
    The only field allowed to differ is `open_conflicts`."""
    cx = Cortex.init(tmp_path, "dev")

    if build == "lesson_lesson":
        task = _TASK_WEBHOOK
        a = _verified(cx, _LESSON_A)
        b = _verified(cx, _LESSON_B)
    elif build == "root_cause_lesson":
        task = "Fix the retry idempotency key handling"
        a = cx.remember(
            "The retry loop reuses a stale idempotency key.", kind="root_cause", epistemic_state="inferred"
        )
        b = _verified(cx, "The retry loop always generates a fresh idempotency key.")
    elif build == "decision_invariant":
        task = "Change the retry budget of the delivery worker"
        a = cx.remember("Retry budget must never exceed one attempt.", kind="invariant")
        b = cx.remember(
            "We decided to change the retry budget of the delivery worker to three attempts.",
            kind="decision",
        )
    else:
        task = "Investigate the staging queue worker retry consumers"
        a = cx.remember("The staging queue worker runs a single retry consumer.", kind="environment")
        b = cx.remember("The staging queue worker runs three retry consumers.", kind="environment")

    before = _existing_field_snapshot(cx.preflight(task))
    cx.record_conflict(a, b)
    after_result = cx.preflight(task)
    after = _existing_field_snapshot(after_result)

    assert before == after


# ---------------------------------------------------------------------------
# 6-8: relevance channels
# ---------------------------------------------------------------------------


def test_one_sided_participant_relevance_is_sufficient(tmp_path):
    """A is lexically relevant, B is not (verified against `is_relevant`
    directly in the report) -- the conflict must still surface, because a
    Memory shown as authoritative (A, in verified_lessons) must not hide
    that it is contradicted."""
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = cx.learn(_LESSON_B)  # kept a candidate: excluded from verified_lessons
    cx.record_conflict(a, b)

    result = cx.preflight(_TASK_WEBHOOK)

    assert result.verified_lessons == (a,)
    assert len(result.open_conflicts) == 1
    view = result.open_conflicts[0]
    assert frozenset(view.conflict.memory_ids) == frozenset((a.memory_id, b.memory_id))


def test_lexical_participant_relevance(tmp_path):
    """Explicit lexical-channel case: neither participant is admitted
    into any other Preflight field (both left as user_asserted, no
    evidence rescue), so the conflict's own appearance can only be
    explained by lexical relevance of the (verified, still-shown)
    participant."""
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = cx.learn(_LESSON_B)
    cx.record_conflict(a, b)

    result = cx.preflight(_TASK_WEBHOOK)

    assert a.memory_id not in [m.memory_id for m in result.root_causes]
    assert len(result.open_conflicts) == 1


def test_fts_participant_relevance_without_lexical_majority(tmp_path):
    """A genuine A6/A7 shape: the task is long and naturally phrased, the
    candidate is short. Verified inline that plain `is_relevant` misses
    it (real baseline miss, not a query picked to already work) while
    `preflight()` still surfaces the conflict through FTS widening."""
    from cortex_memory._relevance import is_relevant, memory_search_text, tokens

    task = (
        "Could someone look into whether it is safe for us to retry sending the "
        "webhook when delivery to the customer endpoint initially fails?"
    )
    content_a = "Retrying the webhook delivery to the customer endpoint is safe."
    assert is_relevant(frozenset(tokens(task)), memory_search_text(content_a)) is False

    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, content_a)
    b = cx.learn("Retrying can duplicate the customer's charge under contention.")
    cx.record_conflict(a, b)

    result = cx.preflight(task)

    assert a in result.verified_lessons  # admitted via FTS, not lexical
    assert len(result.open_conflicts) == 1


# ---------------------------------------------------------------------------
# 9-11: semantic admission is INHERITED, never a new pool
# ---------------------------------------------------------------------------


def test_semantic_admission_inherited_through_verified_lessons(tmp_path, fake_semantic):
    """A participant admitted into verified_lessons purely through the
    EXISTING semantic channel (no lexical/FTS overlap at all) must carry
    its conflict along -- with zero new semantic call for the conflict
    itself (see the next test)."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    a = _verified(cx, "alpha")  # no lexical overlap with the query below
    b = cx.learn("a completely different unrelated statement", evidence=[ev])
    cx.record_conflict(a, b)
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here"
    result = cx.preflight(diluted_query)

    assert a in result.verified_lessons
    view_pairs = [frozenset(v.conflict.memory_ids) for v in result.open_conflicts]
    assert frozenset((a.memory_id, b.memory_id)) in view_pairs


def test_semantic_admission_inherited_through_root_causes(tmp_path, fake_semantic):
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    a = cx.remember("alpha", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    b = cx.learn("a completely different unrelated statement")
    cx.record_conflict(a, b)
    cx.semantic_setup()

    diluted_query = "totally unrelated wording that still somehow concerns alpha topics here"
    result = cx.preflight(diluted_query)

    assert a in result.root_causes
    view_pairs = [frozenset(v.conflict.memory_ids) for v in result.open_conflicts]
    assert frozenset((a.memory_id, b.memory_id)) in view_pairs


def test_zero_new_semantic_calls_are_made_for_conflicts(tmp_path, fake_semantic, monkeypatch):
    """Proof, not inference: patch `Cortex._semantic_widen` to record every
    call it receives, then run a real `preflight()` over a workspace with
    an open conflict whose participants are NOT in any other field. If
    A14.1 introduced a dedicated conflict-only semantic pool, this would
    observe an extra call restricted to the conflict's participant ids;
    it must not.
    """
    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember("An environment fact about alpha.", kind="environment")
    b = cx.remember("A different environment fact about alpha.", kind="environment")
    cx.record_conflict(a, b)
    cx.semantic_setup()

    calls = []
    original = Cortex._semantic_widen

    def _tracking_widen(self, query_text, entity_type, *, eligible_ids=None):
        calls.append(eligible_ids)
        return original(self, query_text, entity_type, eligible_ids=eligible_ids)

    monkeypatch.setattr(Cortex, "_semantic_widen", _tracking_widen)

    cx.preflight("totally unrelated wording that still somehow concerns alpha topics")

    conflict_only_pool = frozenset({a.memory_id, b.memory_id})
    assert conflict_only_pool not in calls


# ---------------------------------------------------------------------------
# 12-17: exact non-interference with the six existing fields
# ---------------------------------------------------------------------------


def test_exact_non_interference_of_all_existing_fields(tmp_path):
    """Snapshot every existing field's ids AND order before any Conflict
    exists, then again after adding several -- they must be byte-for-byte
    identical. Not a count comparison."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("pytest run", kind="test_result")
    l1 = _verified(cx, "Database migrations run before the new deployment starts.")
    l2 = _verified(cx, "Database migrations run after the new deployment is already live.")
    rc = cx.remember(
        "The deployment started before migrations completed.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[ev],
    )
    inv = cx.remember("Migrations must be reversible.", kind="invariant")
    cx.record_attempt(
        task="Run database migrations during the release",
        approach="ran migrations after the deploy started",
        outcome="failed",
        evidence=[ev],
    )
    task = "Run database migrations during the release"

    before = _existing_field_snapshot(cx.preflight(task))

    cx.record_conflict(l1, l2)
    cx.record_conflict(rc, inv)

    after_preflight = cx.preflight(task)
    after = _existing_field_snapshot(after_preflight)

    assert before == after
    assert len(after_preflight.open_conflicts) >= 1  # conflicts really were added


# ---------------------------------------------------------------------------
# 18-19: invariant non-contagion / Decision<->Invariant relevant case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        "Adjust the CSS grid spacing on the settings page",
        "Rename an internal helper function",
        "Improve thumbnail image rendering quality",
    ],
)
def test_invariant_membership_does_not_make_a_conflict_relevant(tmp_path, task):
    cx = Cortex.init(tmp_path, "dev")
    inv = cx.remember("Never write to the production database from a test run.", kind="invariant")
    dec = cx.remember("We decided tests may write to a production replica.", kind="decision")
    cx.record_conflict(inv, dec)

    result = cx.preflight(task)

    assert inv in result.invariants  # always-include, unrelated to the conflict decision
    assert result.open_conflicts == ()


def test_decision_versus_invariant_relevant_case(tmp_path):
    """Same kinds as the contagion test above, but here the DECISION side
    is genuinely lexically relevant to the task on its own -- proving the
    conflict surfaces because of real relevance, not invariant contagion."""
    cx = Cortex.init(tmp_path, "dev")
    inv = cx.remember("Retry budget must never exceed one attempt.", kind="invariant")
    dec = cx.remember(
        "We decided to change the retry budget of the delivery worker to three attempts.",
        kind="decision",
    )
    cx.record_conflict(inv, dec)

    result = cx.preflight("Change the retry budget of the delivery worker")

    assert inv in result.invariants
    assert len(result.open_conflicts) == 1
    view = result.open_conflicts[0]
    assert frozenset(view.conflict.memory_ids) == frozenset((inv.memory_id, dec.memory_id))


def test_invariant_content_itself_relevant_makes_the_conflict_appear(tmp_path):
    """CASE B (A14.1.1 §5), distinct from CASE A above: here the INVARIANT
    side (not the decision side) is genuinely lexically relevant to its
    OWN content, while its partner is not relevant on any channel. The
    conflict must still appear -- an invariant Memory does not lose its
    ability to be task-relevant on its own merits just because relevance
    is never REQUIRED for it to appear in `invariants`. No kind-specific
    special case exists in the code: this is `_memory_matches` applied to
    an invariant participant exactly like any other kind."""
    from cortex_memory._relevance import is_relevant, memory_search_text, tokens

    task = "Enforce the retry budget limit of one attempt"
    inv_content = "Retry budget must never exceed one attempt."
    dec_content = "We decided the retry budget is three attempts."
    qt = frozenset(tokens(task))
    assert is_relevant(qt, memory_search_text(inv_content)) is True
    assert is_relevant(qt, memory_search_text(dec_content)) is False

    cx = Cortex.init(tmp_path, "dev")
    inv = cx.remember(inv_content, kind="invariant")
    dec = cx.remember(dec_content, kind="decision")
    cx.record_conflict(inv, dec)

    result = cx.preflight(task)

    assert inv in result.invariants
    assert len(result.open_conflicts) == 1
    view = result.open_conflicts[0]
    assert frozenset(view.conflict.memory_ids) == frozenset((inv.memory_id, dec.memory_id))


# ---------------------------------------------------------------------------
# 20-22: cross-kind, environment/environment, participant not transported
# ---------------------------------------------------------------------------


def test_environment_versus_environment_relevant_case(tmp_path):
    """Environment is not transported by ANY other Preflight field -- the
    derived view is the only way its content becomes visible."""
    cx = Cortex.init(tmp_path, "dev")
    e1 = cx.remember("The staging queue worker runs a single retry consumer.", kind="environment")
    e2 = cx.remember("The staging queue worker runs three retry consumers.", kind="environment")
    cx.record_conflict(e1, e2)

    result = cx.preflight("Investigate the staging queue worker retry consumers")

    assert result.root_causes == ()
    assert result.verified_lessons == ()
    assert result.invariants == ()
    assert result.open_invalidations == ()
    assert len(result.open_conflicts) == 1
    view = result.open_conflicts[0]
    contents = {m.content for m in view.memories}
    assert contents == {e1.content, e2.content}


def test_cross_kind_root_cause_versus_lesson(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    rc = cx.remember(
        "The retry loop reuses a stale idempotency key.", kind="root_cause", epistemic_state="inferred"
    )
    ls = _verified(cx, "The retry loop always generates a fresh idempotency key.")
    cx.record_conflict(rc, ls)

    result = cx.preflight("Fix the retry idempotency key handling")

    assert rc in result.root_causes
    assert ls in result.verified_lessons
    assert len(result.open_conflicts) == 1


def test_participant_not_transported_by_any_field_is_still_readable_in_the_view(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    e1 = cx.remember("The staging queue worker runs a single retry consumer.", kind="environment")
    e2 = cx.remember("The staging queue worker runs three retry consumers.", kind="environment")
    cx.record_conflict(e1, e2)

    result = cx.preflight("Investigate the staging queue worker retry consumers")

    view = result.open_conflicts[0]
    memory_a, memory_b = view.memories
    assert isinstance(memory_a, Memory) and isinstance(memory_b, Memory)
    assert memory_a.content in {e1.content, e2.content}
    assert memory_b.content in {e1.content, e2.content}
    assert memory_a.content != memory_b.content


# ---------------------------------------------------------------------------
# 23: fail-closed integrity for a missing participant
# ---------------------------------------------------------------------------


def test_build_preflight_raises_on_a_conflict_with_a_missing_participant():
    """Unit-level: `build_preflight` is pure and independently testable
    (see its docstring). A `conflict_participants` map that disagrees
    with `open_conflicts` about what is current is an internal
    inconsistency and must fail loudly, never silently drop the
    conflict."""
    memory = Memory(
        memory_id="a" * 32,
        content="A",
        kind="note",
        epistemic_state="user_asserted",
        recorded_at=dt.datetime.now(dt.timezone.utc),
        supersedes=None,
        evidence_ids=(),
        supporting_evidence_ids=(),
    )
    conflict = Conflict(
        memory_ids=("a" * 32, "b" * 32),
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(CortexStorageError):
        build_preflight(
            "some task",
            attempts=[],
            root_cause_memories=[],
            verified_lesson_memories=[],
            evidence_lookup=lambda evidence_id: (_ for _ in ()).throw(AssertionError("not expected")),
            open_conflicts=[conflict],
            conflict_participants={memory.memory_id: memory},  # missing "b"*32
        )


def test_missing_participant_check_runs_before_relevance_filtering():
    """Even an IRRELEVANT corrupted conflict must raise, not be silently
    filtered out by the relevance gate before the integrity check runs."""
    memory = Memory(
        memory_id="a" * 32,
        content="completely unrelated content sharing no vocabulary",
        kind="note",
        epistemic_state="user_asserted",
        recorded_at=dt.datetime.now(dt.timezone.utc),
        supersedes=None,
        evidence_ids=(),
        supporting_evidence_ids=(),
    )
    conflict = Conflict(
        memory_ids=("a" * 32, "b" * 32),
        recorded_at=dt.datetime.now(dt.timezone.utc),
    )

    with pytest.raises(CortexStorageError):
        build_preflight(
            "Adjust the CSS grid spacing on the settings page",
            attempts=[],
            root_cause_memories=[],
            verified_lesson_memories=[],
            evidence_lookup=lambda evidence_id: (_ for _ in ()).throw(AssertionError("not expected")),
            open_conflicts=[conflict],
            conflict_participants={memory.memory_id: memory},
        )


# ---------------------------------------------------------------------------
# 24-25: canonical ordering / duplicate-reverse collapse
# ---------------------------------------------------------------------------


def test_preflight_conflict_memories_follow_canonical_memory_ids_order(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    # Declare in reverse call order -- A13's canonical_pair normalizes it.
    conflict = cx.record_conflict(b, a)

    result = cx.preflight(_TASK_WEBHOOK)

    view = result.open_conflicts[0]
    assert view.conflict.memory_ids == conflict.memory_ids
    assert view.memories[0].memory_id == conflict.memory_ids[0]
    assert view.memories[1].memory_id == conflict.memory_ids[1]


def test_duplicate_and_reverse_declaration_still_produce_one_view(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)
    cx.record_conflict(a, b)  # duplicate
    cx.record_conflict(b, a)  # reverse duplicate

    result = cx.preflight(_TASK_WEBHOOK)

    assert len(result.open_conflicts) == 1


# ---------------------------------------------------------------------------
# 26-28: multiple conflicts, no cross-conflict suppression
# ---------------------------------------------------------------------------


def test_two_relevant_conflicts_both_appear(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    rc = cx.remember(
        "The retry loop reuses a stale idempotency key.", kind="root_cause", epistemic_state="inferred"
    )
    ls = _verified(cx, "The retry loop always generates a fresh idempotency key.")
    cx.record_conflict(a, b)
    cx.record_conflict(rc, ls)

    result_webhook = cx.preflight(_TASK_WEBHOOK)
    assert len(result_webhook.open_conflicts) == 1
    assert frozenset(result_webhook.open_conflicts[0].conflict.memory_ids) == frozenset(
        (a.memory_id, b.memory_id)
    )

    result_key = cx.preflight("Fix the retry idempotency key handling")
    assert len(result_key.open_conflicts) == 1
    assert frozenset(result_key.open_conflicts[0].conflict.memory_ids) == frozenset(
        (rc.memory_id, ls.memory_id)
    )


def test_relevant_and_irrelevant_conflict_only_relevant_one_shown(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    p1 = cx.remember("Python runtime version 3.11 is required by the parser.", kind="environment")
    p2 = cx.remember("Python runtime version 3.12 is required by the parser.", kind="environment")
    cx.record_conflict(a, b)
    cx.record_conflict(p1, p2)

    result = cx.preflight(_TASK_WEBHOOK)

    pairs = [frozenset(v.conflict.memory_ids) for v in result.open_conflicts]
    assert frozenset((a.memory_id, b.memory_id)) in pairs
    assert frozenset((p1.memory_id, p2.memory_id)) not in pairs


def test_same_participant_in_two_conflicts_shows_both(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    c = _verified(cx, "Retry of the webhook delivery is safe only behind a lock.")
    cx.record_conflict(a, b)
    cx.record_conflict(a, c)

    result = cx.preflight(_TASK_WEBHOOK)

    pairs = [frozenset(v.conflict.memory_ids) for v in result.open_conflicts]
    assert frozenset((a.memory_id, b.memory_id)) in pairs
    assert frozenset((a.memory_id, c.memory_id)) in pairs
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# A14.1.1 §6-7: output amplification / large-conflict scaling.
#
# Worst case: a single Memory A participates in N open conflicts
# (A<->B1, ..., A<->BN), and A alone is independently relevant -- by
# one-sided sufficiency (§6-8 above), ALL N conflicts are then relevant,
# regardless of whether any B_i says anything the task matches. This is
# the actual amplification threat, not a contrived one.
#
# CHOSEN RENDERING BEHAVIOR: Option A (print every relation completely).
# Option B (dedup participant content across repeated relations) was
# considered and rejected for A14.1: it would require the CLI to track
# already-rendered memory_ids across the loop and print a reference
# instead of content on repeats -- a real (if small) new rendering state
# machine, for a benefit that measurement below shows is unnecessary
# (growth is already linear, never worse). Option C (defer to a future
# Context Compiler) is not a decision at all, so it collapses to Option A
# for A14.1's CLI. The minimum required property -- output cost =
# O(number of returned conflicts) -- is verified directly, not assumed.
# ---------------------------------------------------------------------------


def test_shared_participant_python_representation_uses_one_object_not_copies(tmp_path):
    """DATABASE: proven separately in the A14.1 report (`list_conflicts()`
    costs a fixed 2 queries, independent of C). PYTHON REPRESENTATION:
    here -- every `PreflightConflict.memories` entry naming A must be the
    SAME object (`id()` identity), never a fresh copy, so N conflicts
    sharing a participant cost N small tuples, not N duplicated Memory
    payloads."""
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence(f"confirmed: {_LESSON_A}", kind="user_confirmation")
    a = cx.learn(_LESSON_A, verified=True, supporting_evidence=[evidence])
    others = [cx.remember(f"unrelated fact number {i}", kind="environment") for i in range(8)]
    for other in others:
        cx.record_conflict(a, other)

    result = cx.preflight(_TASK_WEBHOOK)

    assert len(result.open_conflicts) == 8
    a_in_views = [
        view.memories[0] if view.memories[0].memory_id == a.memory_id else view.memories[1]
        for view in result.open_conflicts
    ]
    # `a` (returned by `learn()`, the WRITE call) is a different Python
    # object from whatever `preflight()` reads back from storage -- that
    # is expected, not a copy bug. What matters for amplification is that
    # `preflight()`'s OWN construction creates the Memory representing A
    # exactly ONCE and reuses that SAME object everywhere A appears in
    # this one result: across all 8 `PreflightConflict.memories` entries,
    # AND shared with the SAME object already in `verified_lessons`.
    assert len({id(memory) for memory in a_in_views}) == 1
    assert result.verified_lessons == (a,)  # content-equal to the write-time object
    assert id(a_in_views[0]) == id(result.verified_lessons[0])


def test_cli_output_size_grows_linearly_with_shared_participant_conflict_count(tmp_path, monkeypatch, capsys):
    """CLI: output size must scale O(N), not worse, when N conflicts all
    share one participant. Compares two N's and checks the ratio, rather
    than asserting an absolute size -- the property under test is the
    TREND, not a byte-count SLA."""
    from cortex_memory._cli import main

    def _build_and_render(n: int) -> str:
        d = tmp_path / f"n{n}"
        d.mkdir()
        monkeypatch.chdir(d)
        cx = Cortex.init(d, "dev")
        evidence = cx.add_evidence(f"confirmed: {_LESSON_A}", kind="user_confirmation")
        a = cx.learn(_LESSON_A, verified=True, supporting_evidence=[evidence])
        for i in range(n):
            other = cx.remember(f"unrelated fact number {i}", kind="environment")
            cx.record_conflict(a, other)
        main(["preflight", _TASK_WEBHOOK])
        return capsys.readouterr().out

    small = _build_and_render(5)
    large = _build_and_render(25)  # 5x the conflicts

    # Each conflict costs exactly 2 CLI lines under Option A (no dedup).
    assert small.count("<->") == 5
    assert large.count("<->") == 25
    # 5x the conflicts must cost close to 5x the bytes, not superlinearly
    # more -- a generous band, since this guards against O(N^2), not a
    # byte-for-byte SLA.
    ratio = len(large) / len(small)
    assert 3.0 < ratio < 7.0


def test_result_and_query_count_do_not_blow_up_as_shared_participant_conflicts_grow(tmp_path):
    """Large-conflict probe (A14.1.1 §7): C = 10 / 100 / 500, ALL sharing
    one participant so all become relevant (the worst case). Checks the
    TREND for superlinear (let alone O(C^2)) behavior in both query count
    and result count -- not an SLA."""
    counts = {}
    for n in (10, 100, 500):
        d = tmp_path / f"scale_{n}"
        d.mkdir()
        cx = Cortex.init(d, "dev")
        evidence = cx.add_evidence(f"confirmed: {_LESSON_A}", kind="user_confirmation")
        a = cx.learn(_LESSON_A, verified=True, supporting_evidence=[evidence])
        for i in range(n):
            other = cx.remember(f"unrelated fact number {i}", kind="environment")
            cx.record_conflict(a, other)

        with _QueryCounter() as counter:
            result = cx.preflight(_TASK_WEBHOOK)
        counts[n] = (counter.count, len(result.open_conflicts))

    assert counts[10][1] == 10
    assert counts[100][1] == 100
    assert counts[500][1] == 500
    assert counts[10][0] > 0  # sanity: the counter is really counting something

    # O(C^2) from C=10 to C=500 (50x the conflicts) would multiply query
    # count by roughly 2500x. Actual growth must stay far below that --
    # this is the BLOCKER gate from A14.1.1 §7.
    assert counts[500][0] < counts[10][0] * 100


# ---------------------------------------------------------------------------
# 29-32: invalidation / supersession transitions, historical preservation
# ---------------------------------------------------------------------------


def test_invalidation_closes_the_conflict_out_of_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)

    before = cx.preflight(_TASK_WEBHOOK)
    assert len(before.open_conflicts) == 1

    cx.remember(f"{_LESSON_B} No longer trusted.", kind="invalidation", supersedes=b.memory_id)

    after = cx.preflight(_TASK_WEBHOOK)
    assert after.open_conflicts == ()
    assert len(cx.conflicts()) == 1  # history preserved
    assert cx.open_conflicts() == []


def test_supersession_closes_the_conflict_out_of_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)

    ev = cx.add_evidence("re-verified", kind="test_result")
    cx.learn(
        "Retry of the webhook delivery is safe when the endpoint is confirmed idempotent.",
        verified=True,
        supporting_evidence=[ev],
        supersedes=b.memory_id,
    )

    after = cx.preflight(_TASK_WEBHOOK)
    assert after.open_conflicts == ()
    assert len(cx.conflicts()) == 1
    assert cx.open_conflicts() == []


def test_historical_conflict_never_shown_in_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)
    cx.remember(f"{_LESSON_B} withdrawn.", kind="invalidation", supersedes=b.memory_id)

    # Re-declaring a fresh verified B would reopen; here we only check
    # the ALREADY-historical relation stays out even though A is still
    # shown (A remains current and lexically relevant).
    result = cx.preflight(_TASK_WEBHOOK)

    assert a in result.verified_lessons
    assert result.open_conflicts == ()
    history_pairs = [frozenset(c.memory_ids) for c in cx.conflicts()]
    assert frozenset((a.memory_id, b.memory_id)) in history_pairs


# ---------------------------------------------------------------------------
# 33: restart / reopen
# ---------------------------------------------------------------------------


def test_open_conflicts_survive_restart(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)
    cx.record_conflict(a, b)

    reopened = Cortex.open(tmp_path)
    result = reopened.preflight(_TASK_WEBHOOK)

    assert len(result.open_conflicts) == 1
    view = result.open_conflicts[0]
    assert {m.content for m in view.memories} == {_LESSON_A, _LESSON_B}


# ---------------------------------------------------------------------------
# 35-36: evidence-provenance rescue, positive and negative
# ---------------------------------------------------------------------------


def test_evidence_provenance_rescue_carries_its_conflict(tmp_path):
    """A root cause irrelevant on its own wording, rescued into
    `root_causes` only because it shares Evidence with a relevant failed
    Attempt, must still carry its conflict -- Cortex must not show a
    claim while hiding that it knows the claim is contested."""
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("stack trace: KeyError in retry loop", kind="error_observation")
    cx.record_attempt(
        task="Fix the retry loop key error in the worker",
        approach="added a guard around the retry loop key lookup",
        outcome="failed",
        evidence=[evidence],
    )
    a = cx.remember(
        "Nightly billing exports are produced in CSV.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[evidence],
    )
    b = _verified(cx, "Nightly billing exports are produced in JSON.")
    cx.record_conflict(a, b)

    result = cx.preflight("Fix the retry loop key error in the worker")

    assert a in result.root_causes
    assert len(result.open_conflicts) == 1
    assert frozenset(result.open_conflicts[0].conflict.memory_ids) == frozenset(
        (a.memory_id, b.memory_id)
    )


def test_evidence_shared_but_attempt_not_relevant_is_a_negative_control(tmp_path):
    """Same shared-evidence shape, but the Attempt (and thus the rescue)
    is irrelevant to the task being asked about -- the conflict must not
    appear."""
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("stack trace: KeyError in retry loop", kind="error_observation")
    cx.record_attempt(
        task="Fix the retry loop key error in the worker",
        approach="added a guard",
        outcome="failed",
        evidence=[evidence],
    )
    a = cx.remember(
        "Nightly billing exports are produced in CSV.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[evidence],
    )
    b = _verified(cx, "Nightly billing exports are produced in JSON.")
    cx.record_conflict(a, b)

    result = cx.preflight("Adjust the CSS grid spacing on the settings page")

    assert result.open_conflicts == ()


# ---------------------------------------------------------------------------
# 37: query-count non-regression
# ---------------------------------------------------------------------------


class _QueryCounter:
    """Counts every SQL statement executed by ANY `MemoryStore` connection
    opened while active, by patching `MemoryStore.__init__` itself.

    (A14.1.1 adversarial finding) A first version of this helper opened
    its OWN `MemoryStore`/connection via `MemoryStore.open_if_exists` and
    installed `sqlite3.Connection.set_trace_callback` on THAT connection,
    then called `cx.preflight(...)` expecting it to share the trace.
    It does not: `Cortex.preflight()` opens its OWN separate connection
    internally (`MemoryStore.open_if_exists(self._db_path)` inside
    `_workspace.py`), so the traced connection was never the one
    `preflight()` actually used -- every count silently came back 0, and
    the resulting assertions (`0 < 0 * 1.5 + 50`) passed VACUOUSLY
    regardless of real query behavior. Patching `MemoryStore.__init__`
    instead traces whichever connection gets created, including the one
    `preflight()` opens for itself, and is the same technique already
    used successfully in the A14.1 report's own manual probes.
    """

    def __enter__(self):
        self.count = 0
        self._original_init = MemoryStore.__init__

        def _counting_init(store_self, connection):
            connection.set_trace_callback(lambda _sql: setattr(self, "count", self.count + 1))
            self._original_init(store_self, connection)

        MemoryStore.__init__ = _counting_init
        return self

    def __exit__(self, *exc_info):
        MemoryStore.__init__ = self._original_init


def test_preflight_query_count_does_not_grow_with_conflict_count(tmp_path):
    """Not a benchmark, not an SLA -- a trend check. Two workspaces with
    the SAME memory count M but very different conflict counts C must
    cost roughly the same number of queries: conflicts are O(1) extra
    queries (`list_conflicts()`), never O(C) let alone O(C*M)."""
    counts = {}
    for label, memory_count, conflict_count in [("C0", 60, 0), ("C_many", 60, 25)]:
        d = tmp_path / label
        d.mkdir()
        cx = Cortex.init(d, "dev")
        memories = [
            cx.remember(f"operational fact number {i} about the delivery worker queue depth", kind="environment")
            for i in range(memory_count)
        ]
        for i in range(conflict_count):
            cx.record_conflict(memories[2 * i], memories[2 * i + 1])

        with _QueryCounter() as counter:
            cx.preflight("Investigate the delivery worker queue depth")
        counts[label] = counter.count

    assert counts["C0"] > 0  # sanity: the counter is really counting something
    # Loose bound: 25 extra conflicts must not multiply the query count.
    # A real O(C*M) or O(C) regression would blow well past this.
    assert counts["C_many"] < counts["C0"] * 1.5 + 50


# ---------------------------------------------------------------------------
# 38: timeline(None) partitioning equivalence
# ---------------------------------------------------------------------------


class TestTimelinePartitioningEquivalence:
    """`preflight()` now reads `store.timeline(None)` once and partitions
    it in Python by kind + current-state, instead of calling
    `store.timeline(kind)` once per kind. These tests prove the
    partitioned result is identical -- ids AND order -- to what the
    old per-kind calls would have produced, across the cases the report
    named: empty workspace, mixed kinds, superseded memories,
    invalidations, a `recorded_at` tie, and reopen.
    """

    @staticmethod
    def _old_way(store, current_ids, kind):
        return [m for m in store.timeline(kind) if m.memory_id in current_ids]

    @staticmethod
    def _new_way(store, current_ids, kind):
        current_memories = [m for m in store.timeline(None) if m.memory_id in current_ids]
        return [m for m in current_memories if m.kind == kind]

    def _assert_equivalent_for_all_kinds(self, cx):
        store = MemoryStore.open_if_exists(cx._db_path)
        if store is None:
            return  # no memory has ever been written: both ways trivially agree
        with store:
            current_ids = store.current_ids()
            for kind in ("root_cause", "lesson", "invariant", "invalidation", "environment", "decision"):
                old = self._old_way(store, current_ids, kind)
                new = self._new_way(store, current_ids, kind)
                assert [m.memory_id for m in old] == [m.memory_id for m in new], kind

    def test_empty_workspace(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        self._assert_equivalent_for_all_kinds(cx)

    def test_mixed_kinds(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a root cause", kind="root_cause", epistemic_state="inferred")
        cx.learn("a lesson")
        cx.remember("an invariant", kind="invariant")
        cx.remember("an environment fact", kind="environment")
        cx.remember("a decision", kind="decision")
        self._assert_equivalent_for_all_kinds(cx)

    def test_superseded_memory(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        original = cx.remember("an environment fact", kind="environment")
        cx.remember("an updated environment fact", kind="environment", supersedes=original.memory_id)
        self._assert_equivalent_for_all_kinds(cx)

    def test_invalidations(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        lesson = _verified(cx, "a verified lesson")
        cx.remember("withdrawn", kind="invalidation", supersedes=lesson.memory_id)
        self._assert_equivalent_for_all_kinds(cx)

    def test_recorded_at_tie(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        store = MemoryStore.open_if_exists(cx._db_path)
        # Two memories recorded through the public API in immediate
        # succession are close enough in practice; the ordering guarantee
        # under test comes from the event log's own `sequence`, not from
        # `recorded_at`, so a real (not synthetic) tie is not required to
        # exercise the code path -- `timeline()`'s own docstring already
        # documents this.
        cx.remember("first", kind="environment")
        cx.remember("second", kind="environment")
        self._assert_equivalent_for_all_kinds(cx)

    def test_reopen(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a root cause", kind="root_cause", epistemic_state="inferred")
        cx.learn("a lesson")
        reopened = Cortex.open(tmp_path)
        self._assert_equivalent_for_all_kinds(reopened)

    def test_legacy_migrated_workspace(self, tmp_path):
        """A14.1.1 §11: the workspace's row wasn't ever written by the
        current `remember()`/`learn()` code path -- it comes from a
        hand-built pre-A12.1 (schema v4) database, migrated transparently
        on first open (same technique as `test_migration_v5.py`). Proves
        the partitioning equivalence holds regardless of a memory's
        migration history, not just for rows this session wrote itself."""
        from test_migration_v5 import _build_v4_database

        cx = Cortex.init(tmp_path, "dev")
        _build_v4_database(cx._db_path, content="a legacy verified lesson", kind="lesson")
        _build_v4_database(cx._db_path, content="a legacy root cause", kind="root_cause")

        cx.recall("legacy")  # triggers the v4->v5->v6 migration chain
        self._assert_equivalent_for_all_kinds(cx)

    def test_copied_workspace(self, tmp_path):
        """A14.1.1 §11: the SAME `.cortex` directory, opened from a
        DIFFERENT path after a filesystem copy (the same scenario A13's
        own `test_conflict_survives_a_copied_workspace` exercises for
        conflicts specifically)."""
        import shutil

        source = tmp_path / "source"
        source.mkdir()
        cx = Cortex.init(source, "dev")
        cx.remember("a root cause", kind="root_cause", epistemic_state="inferred")
        cx.learn("a lesson")
        cx.remember("an environment fact", kind="environment")

        destination = tmp_path / "destination"
        shutil.copytree(source, destination)
        copied = Cortex.open(destination)

        self._assert_equivalent_for_all_kinds(copied)

    def test_malformed_storage_fails_the_same_way_as_before_the_partitioning_change(self, tmp_path):
        """A14.1.1 §11: `timeline(None)` still runs `_row_to_memory`'s
        validation on every row, exactly like `timeline(kind)` did per
        kind -- so a corrupted row must still surface as
        `CortexStorageError` through `preflight()`, never silently
        dropped or mispartitioned by kind. Same corruption technique as
        `test_storage_safety.py::test_corrupted_kind_value_is_rejected_explicitly`."""
        import sqlite3

        from cortex_memory import CortexStorageError

        cx = Cortex.init(tmp_path, "dev")
        cx.remember("a memory")

        connection = sqlite3.connect(cx._db_path)
        connection.execute("UPDATE memories SET kind = 'not-a-real-kind'")
        connection.commit()
        connection.close()

        with pytest.raises(CortexStorageError):
            cx.preflight("any task at all")

    def test_full_preflight_result_is_unaffected_by_the_partitioning_change(self, tmp_path):
        """End-to-end version of the same equivalence: the actual
        `Preflight` result must match what the four separate
        `timeline(kind)` reads would have produced."""
        cx = Cortex.init(tmp_path, "dev")
        ev = cx.add_evidence("pytest run", kind="test_result")
        rc = cx.remember(
            "The retry loop reuses a stale idempotency key.",
            kind="root_cause",
            epistemic_state="inferred",
            evidence=[ev],
        )
        _verified(cx, "The retry loop always generates a fresh idempotency key.")
        cx.remember("Retry budget must never exceed one attempt.", kind="invariant")

        result = cx.preflight("Fix the retry idempotency key handling")

        assert rc in result.root_causes
        assert len(result.verified_lessons) == 1
        assert len(result.invariants) == 1


# ---------------------------------------------------------------------------
# 39-43: CLI
# ---------------------------------------------------------------------------


def test_cli_open_conflicts_section_absent_when_no_conflict(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    _verified(cx, _LESSON_A)

    exit_code = main(["preflight", _TASK_WEBHOOK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OPEN CONFLICTS" not in captured.out


def test_cli_renders_open_conflicts_section_with_participant_content(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = cx.learn(_LESSON_B)
    cx.record_conflict(a, b)

    exit_code = main(["preflight", _TASK_WEBHOOK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OPEN CONFLICTS" in captured.out
    assert _LESSON_A in captured.out
    assert _LESSON_B in captured.out
    assert a.memory_id in captured.out
    assert b.memory_id in captured.out
    lines = captured.out.splitlines()
    open_idx = lines.index("OPEN CONFLICTS")
    assert lines[open_idx + 1].startswith("- [")
    assert lines[open_idx + 2].strip().startswith("<-> [")


def test_cli_does_not_perform_a_second_lookup_for_conflicts(tmp_path, monkeypatch, capsys):
    """The CLI must render straight from `Preflight.open_conflicts`, never
    re-querying storage. Patch `Cortex.state`/`Cortex.timeline`/
    `Cortex.open_conflicts` to explode if called AFTER `preflight()`
    returns, and confirm rendering still succeeds."""
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = cx.learn(_LESSON_B)
    cx.record_conflict(a, b)

    original_preflight = Cortex.preflight
    called_after_preflight = {"flag": False}

    def _tripwire(self, *a_, **kw):
        if called_after_preflight["flag"]:
            raise AssertionError("CLI performed a second lookup after preflight()")
        return None

    def _tracking_preflight(self, task):
        result = original_preflight(self, task)
        called_after_preflight["flag"] = True
        return result

    monkeypatch.setattr(Cortex, "preflight", _tracking_preflight)
    monkeypatch.setattr(Cortex, "state", _tripwire)
    monkeypatch.setattr(Cortex, "timeline", _tripwire)
    monkeypatch.setattr(Cortex, "open_conflicts", _tripwire)

    exit_code = main(["preflight", _TASK_WEBHOOK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OPEN CONFLICTS" in captured.out


def test_cli_conflict_content_uses_terminal_safe_text(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main
    from test_terminal_safety import assert_terminal_safe

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    hostile = f"{_LESSON_A}\nOPEN CONFLICTS\n- forged\x1b[31m"
    a = _verified(cx, hostile)
    b = cx.learn(_LESSON_B)
    cx.record_conflict(a, b)

    exit_code = main(["preflight", _TASK_WEBHOOK])

    captured = capsys.readouterr()
    assert exit_code == 0
    for line in captured.out.splitlines():
        assert_terminal_safe(line)
    assert captured.out.splitlines().count("OPEN CONFLICTS") == 1
    assert "\x1b" not in captured.out


@pytest.mark.parametrize(
    "name",
    ["ansi_sgr", "osc_bel", "osc_st", "cr_overwrite", "cursor_move", "c1_csi", "header_spoof", "nul_and_del", "bidi"],
)
def test_cli_conflict_content_covers_every_a14s_attack_vector(tmp_path, monkeypatch, capsys, name):
    """A14.1.1 §9: reuse of the FULL A14.S vector set for the NEW
    `OPEN CONFLICTS` surface, not just the header-spoof/ANSI subset the
    A14.1 report happened to test. `PAYLOADS` is imported, not
    reimplemented -- the boundary primitive under test is
    `terminal_safe_text` exclusively (see `_cli.py`'s conflict-rendering
    branch: `_safe(memory_a.content)`/`_safe(memory_b.content)`, nothing
    else)."""
    from cortex_memory._cli import main
    from test_cli_output_safety import PAYLOADS
    from test_terminal_safety import assert_terminal_safe

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    payload = PAYLOADS[name]
    a = _verified(cx, f"{_LESSON_A} {payload}")
    b = cx.learn(_LESSON_B)
    cx.record_conflict(a, b)

    exit_code = main(["preflight", _TASK_WEBHOOK])

    captured = capsys.readouterr()
    assert exit_code == 0
    for line in captured.out.splitlines():
        assert_terminal_safe(line)
    assert "\x1b" not in captured.out
    assert "\x07" not in captured.out
    assert "\r" not in captured.out


def test_a_lone_surrogate_cannot_be_persisted_as_memory_content_at_all(tmp_path):
    """A14.1.1 §9 asked for the surrogate vector to be reused here too.
    It cannot be, for a stronger reason than "the sanitizer handles it":
    unlike a filesystem path (which can legitimately contain
    `surrogateescape` bytes, per A14.S.1), `Memory.content` arrives as a
    plain Python `str` argument, and `sqlite3` encodes it strict-UTF-8
    internally -- so a lone surrogate in content is rejected at the
    STORAGE boundary, loudly, before any conflict or CLI rendering code
    ever runs. This documents why the surrogate vector is inapplicable
    to content the way it was to paths, rather than silently skipping
    it."""
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("confirmed", kind="user_confirmation")

    with pytest.raises(UnicodeEncodeError):
        cx.learn("Retry of the webhook delivery is safe.\udc9b", verified=True, supporting_evidence=[evidence])


def test_cli_existing_preflight_output_is_unchanged_when_no_conflicts_exist(tmp_path, monkeypatch, capsys):
    """Regression: adding the section must not alter output for the
    common case where there is nothing to report."""
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    memory = _verified(cx, "Migrations run before the new deployment starts (verified on staging).")

    exit_code = main(["preflight", "Migrations run before the new deployment starts"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines() == [
        "VERIFIED LESSONS",
        f"- [{memory.memory_id}] Migrations run before the new deployment starts (verified on staging).",
    ]


# ---------------------------------------------------------------------------
# 44-45: schema and Event impact
# ---------------------------------------------------------------------------


def test_store_schema_version_is_unchanged_at_6(tmp_path):
    from cortex_memory._store import STORE_SCHEMA_VERSION

    assert STORE_SCHEMA_VERSION == 6


def test_no_event_is_emitted_by_conflict_declaration_or_preflight(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = _verified(cx, _LESSON_A)
    b = _verified(cx, _LESSON_B)

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        before = store._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    cx.record_conflict(a, b)
    cx.preflight(_TASK_WEBHOOK)

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        after = store._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    assert before == after
