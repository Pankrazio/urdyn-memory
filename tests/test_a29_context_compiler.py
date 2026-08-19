"""A29.1: `Cortex.context()` / `CompiledContext`, the first Context
Compiler tracer.

`preflight()` answers "what does Cortex know that bears on this task",
unbounded. `context()` answers a narrower question -- "what must an
agent respect right now to start this task" -- under an explicit
character budget. These tests deliberately assert behaviors `preflight()`
cannot satisfy at all (a Decision surfacing, a current Invariant being
excluded, a real budget deciding what survives), not just a different
rendering of the same data -- see `test_context_shows_decision_preflight_never_can`
and `test_context_filters_invariants_preflight_shows_unconditionally` for
the two non-vacuity anchors.

Every non-`real_model` test here runs on the lexical channel alone
(no `semantic setup`), exactly like `test_a22_pending_preflight.py`: all
category texts share a deliberately controlled, high-overlap vocabulary
with `_TASK` so admission is deterministic without any model. The
`real_model`-marked tests at the bottom exercise the semantic channel and
the A27 auto-refresh reuse, gated behind the same local-cache probe the
rest of the suite already uses.
"""

from __future__ import annotations

import datetime as dt

import pytest

from cortex_memory import Cortex
from cortex_memory._cli import main
from cortex_memory._context import (
    SECTION_CONSTRAINTS,
    SECTION_DECISIONS,
    SECTION_HISTORY,
    SECTION_OPEN_RISKS,
    ContextItem,
    _render_item,
    compile_context,
)
from cortex_memory._memory import Memory
from test_cli_output_safety import assert_output_terminal_safe
from test_semantic_real_model import _offline, skip_without_model

real_model = pytest.mark.real_model

# Deliberately high, controlled lexical overlap with `_TASK` (7
# significant tokens after stopwords -> admission threshold 4), the same
# style `test_a22_pending_preflight.py` uses for its A21 anchor.
_TASK = "Add retry handling for failed background jobs in the queue"

_INVARIANT_RELEVANT = "Retry handling for failed background jobs must never reorder the queue"
_INVARIANT_UNRELATED = "All commit messages must be written in English"

_PENDING_RELEVANT = "Retry handling for failed background jobs has no backoff yet"
_PENDING_UNRELATED = "Update the README screenshots before the next release"

_LESSON_VERIFIED = "Failed background jobs must retry with idempotent handling to avoid duplicate side effects"
_LESSON_CANDIDATE = "Background jobs retry handling should log a warning on each attempt"

_DECISION_RELEVANT = "Retry handling for failed background jobs uses exponential backoff capped at 5 attempts"

_ROOT_CAUSE = "Background jobs failed to retry because the queue handling code swallowed the exception"

_STANDALONE_ATTEMPT_TASK = "Retry failed background jobs without blocking the queue handling loop"
_STANDALONE_ATTEMPT_APPROACH = "Tried retrying synchronously inside the queue handling loop"

_UNRELATED_LESSON = "Bake the cake at 180 degrees for 40 minutes"


def _workspace(tmp_path):
    return Cortex.init(tmp_path)


def _populate(cx):
    """One realistic, fully-CLI-shaped workspace covering every category
    `context()` composes, plus deliberately unrelated noise and one
    superseded invariant. Returns the ids a test typically needs to
    assert against."""
    invariant = cx.remember(_INVARIANT_RELEVANT, kind="invariant")
    cx.remember(_INVARIANT_UNRELATED, kind="invariant")

    pending = cx.remember(_PENDING_RELEVANT, kind="pending")
    cx.remember(_PENDING_UNRELATED, kind="pending")

    test_evidence = cx.add_evidence("pytest tests/test_retry.py -> 3 passed", kind="test_result")
    lesson = cx.learn(_LESSON_VERIFIED, verified=True, supporting_evidence=[test_evidence])
    cx.learn(_LESSON_CANDIDATE, verified=False)

    decision = cx.remember(_DECISION_RELEVANT, kind="decision")

    rc_evidence = cx.add_evidence("Traceback: exception swallowed in queue handler", kind="error_observation")
    root_cause = cx.remember(_ROOT_CAUSE, kind="root_cause", evidence=[rc_evidence])
    absorbed_attempt = cx.record_attempt(
        task=_TASK, approach="Added retry inside the except branch", outcome="failed", evidence=[rc_evidence]
    )
    standalone_evidence = cx.add_evidence("Traceback: deadlock in retry loop", kind="error_observation")
    standalone_attempt = cx.record_attempt(
        task=_STANDALONE_ATTEMPT_TASK,
        approach=_STANDALONE_ATTEMPT_APPROACH,
        outcome="failed",
        evidence=[standalone_evidence],
    )

    unrelated_evidence = cx.add_evidence("cake came out well", kind="test_result")
    cx.learn(_UNRELATED_LESSON, verified=True, supporting_evidence=[unrelated_evidence])

    return {
        "invariant": invariant,
        "pending": pending,
        "lesson": lesson,
        "decision": decision,
        "root_cause": root_cause,
        "absorbed_attempt": absorbed_attempt,
        "standalone_attempt": standalone_attempt,
    }


# ---------------------------------------------------------------------------
# A / F / G / H -- relevant compilation, per category
# ---------------------------------------------------------------------------


def test_relevant_compilation_covers_every_category(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    result = cx.context(_TASK)

    by_heading = {section.heading: section.items for section in result.sections}
    assert {item.entity_id for item in by_heading[SECTION_CONSTRAINTS]} == {ids["invariant"].memory_id}
    assert {item.entity_id for item in by_heading[SECTION_OPEN_RISKS]} == {ids["pending"].memory_id}
    assert {item.entity_id for item in by_heading["LESSONS"]} == {ids["lesson"].memory_id}
    assert {item.entity_id for item in by_heading[SECTION_DECISIONS]} == {ids["decision"].memory_id}
    history_ids = {item.entity_id for item in by_heading[SECTION_HISTORY]}
    assert ids["root_cause"].memory_id in history_ids
    assert ids["standalone_attempt"].attempt_id in history_ids
    assert not result.is_empty()


def test_relevant_pending_included_unrelated_pending_excluded(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    result = cx.context(_TASK)

    risk_ids = {item.entity_id for section in result.sections if section.heading == SECTION_OPEN_RISKS for item in section.items}
    assert risk_ids == {ids["pending"].memory_id}


# ---------------------------------------------------------------------------
# B -- abstention on an unrelated task
# ---------------------------------------------------------------------------


def test_unrelated_task_abstains_instead_of_filling_the_budget(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    result = cx.context("Rotate the TLS certificate for the nginx reverse proxy", budget=100000)

    assert result.is_empty()
    assert result.sections == ()


# ---------------------------------------------------------------------------
# N -- empty context rendering distinguishes no-candidates from budget
# omission (A34)
# ---------------------------------------------------------------------------


def test_no_relevant_candidates_renders_no_context_message(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    result = cx.context("Rotate the TLS certificate for the nginx reverse proxy", budget=100000)

    assert result.omitted == 0
    assert "No compiled context for this task." in result.render()


def test_relevant_candidate_omitted_for_budget_does_not_claim_no_context(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(_INVARIANT_RELEVANT, kind="invariant")

    result = cx.context(_TASK, budget=1)

    assert result.is_empty()
    assert result.omitted > 0
    rendered = result.render()
    assert "No compiled context for this task." not in rendered
    assert "No compiled items fit within the budget." in rendered
    assert "0 of 1 selected; 1 omitted for budget" in rendered


def test_selected_item_renders_normally_no_abstention_message(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(_INVARIANT_RELEVANT, kind="invariant")

    result = cx.context(_TASK, budget=100000)

    rendered = result.render()
    assert not result.is_empty()
    assert "No compiled context for this task." not in rendered
    assert "No compiled items fit within the budget." not in rendered


def test_cli_context_reports_budget_omission_not_absence_of_context(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    cx.remember(_INVARIANT_RELEVANT, kind="invariant")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["context", _TASK, "--budget", "1"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No compiled context for this task." not in out
    assert "No compiled items fit within the budget." in out


# ---------------------------------------------------------------------------
# C -- current-state filtering: superseded is excluded even under a huge budget
# ---------------------------------------------------------------------------


def test_superseded_memory_never_reintroduced_even_under_a_large_budget(tmp_path):
    cx = _workspace(tmp_path)
    # Strongly relevant on its OWN wording (so it WOULD be admitted if
    # current-state filtering did not exclude it), but superseded.
    old_invariant = cx.remember(
        "Retry handling for failed background jobs must always run synchronously in the queue", kind="invariant"
    )
    new_invariant = cx.remember(
        "Retry handling for failed background jobs may run on a bounded worker pool in the queue",
        kind="invariant",
        supersedes=old_invariant.memory_id,
    )

    result = cx.context(_TASK, budget=100000)

    all_ids = {item.entity_id for section in result.sections for item in section.items}
    assert old_invariant.memory_id not in all_ids
    assert new_invariant.memory_id in all_ids


# ---------------------------------------------------------------------------
# E -- verified Lesson selection, unverified never promoted
# ---------------------------------------------------------------------------


def test_only_verified_lesson_is_selected_not_the_candidate(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    result = cx.context(_TASK, budget=100000)

    lesson_ids = {item.entity_id for section in result.sections if section.heading == "LESSONS" for item in section.items}
    assert lesson_ids == {ids["lesson"].memory_id}
    for section in result.sections:
        for item in section.items:
            assert item.content != _LESSON_CANDIDATE


# ---------------------------------------------------------------------------
# G / U -- the Decision discriminant: absent from preflight, present in context
# ---------------------------------------------------------------------------


def test_context_shows_decision_preflight_never_can(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    preflight_result = cx.preflight(_TASK)
    context_result = cx.context(_TASK, budget=100000)

    assert not hasattr(preflight_result, "decisions")
    preflight_ids = set()
    for field in ("known_failures", "root_causes", "verified_lessons", "invariants", "pending"):
        preflight_ids.update(getattr(m, "memory_id", getattr(m, "attempt_id", None)) for m in getattr(preflight_result, field))
    assert ids["decision"].memory_id not in preflight_ids

    context_ids = {item.entity_id for section in context_result.sections for item in section.items}
    assert ids["decision"].memory_id in context_ids


# ---------------------------------------------------------------------------
# §29/§35 -- the Invariant discriminant: preflight is unconditional, context is not
# ---------------------------------------------------------------------------


def test_context_filters_invariants_preflight_shows_unconditionally(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    preflight_result = cx.preflight(_TASK)
    context_result = cx.context(_TASK, budget=100000)

    preflight_invariant_ids = {m.memory_id for m in preflight_result.invariants}
    # preflight includes every CURRENT invariant, relevant or not (A9.1)
    assert len(preflight_invariant_ids) == 2

    context_invariant_ids = {
        item.entity_id for section in context_result.sections if section.heading == SECTION_CONSTRAINTS for item in section.items
    }
    assert context_invariant_ids == {ids["invariant"].memory_id}
    assert context_result.invariants_excluded == 1


# ---------------------------------------------------------------------------
# I -- Conflict visibility, never resolved, atomic with its item under budget
# ---------------------------------------------------------------------------


def test_conflict_between_two_selected_items_is_disclosed_both_ways(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)
    cx.record_conflict(ids["lesson"], ids["decision"])

    result = cx.context(_TASK, budget=100000)

    lesson_item = next(item for section in result.sections for item in section.items if item.entity_id == ids["lesson"].memory_id)
    decision_item = next(item for section in result.sections for item in section.items if item.entity_id == ids["decision"].memory_id)
    assert ids["decision"].memory_id in lesson_item.conflicts_with
    assert ids["lesson"].memory_id in decision_item.conflicts_with


def test_conflict_marker_never_appears_without_its_item_under_tight_budget(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)
    cx.record_conflict(ids["lesson"], ids["decision"])

    # A budget that admits the invariant/pending but cannot possibly fit
    # the lesson AND its conflict marker together.
    result = cx.context(_TASK, budget=180)

    rendered = result.render()
    if "CONFLICTS WITH" in rendered:
        assert f"[{ids['lesson'].memory_id}]" in rendered or f"[{ids['decision'].memory_id}]" in rendered


# ---------------------------------------------------------------------------
# J -- provenance / auditability
# ---------------------------------------------------------------------------


def test_every_item_id_is_a_canonical_32_hex_id_resolvable_in_timeline(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)
    result = cx.context(_TASK, budget=100000)

    memory_ids = {m.memory_id for m in cx.timeline()}
    for section in result.sections:
        for item in section.items:
            assert len(item.entity_id) == 32
            int(item.entity_id, 16)  # must be valid hex
            if item.kind not in ("attempt", "evidence"):
                assert item.entity_id in memory_ids


# ---------------------------------------------------------------------------
# K -- deterministic ordering, repeatable
# ---------------------------------------------------------------------------


def test_section_order_is_fixed_and_render_is_repeatable(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    first = cx.context(_TASK, budget=100000)
    second = cx.context(_TASK, budget=100000)

    assert first.render() == second.render()
    fixed_order = [SECTION_CONSTRAINTS, SECTION_OPEN_RISKS, "LESSONS", SECTION_DECISIONS, SECTION_HISTORY, "VALIDATION"]
    headings = [section.heading for section in first.sections]
    assert headings == [heading for heading in fixed_order if heading in headings]


# ---------------------------------------------------------------------------
# L / M -- budget determinism and SMALL/MEDIUM/LARGE prefix monotonicity
# ---------------------------------------------------------------------------


def test_used_never_exceeds_budget_across_a_range(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    for budget in (1, 50, 100, 250, 500, 1000, 4000, 20000):
        result = cx.context(_TASK, budget=budget)
        assert result.used <= budget


def test_small_medium_large_are_a_monotonic_prefix(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    small = cx.context(_TASK, budget=90)
    medium = cx.context(_TASK, budget=400)
    large = cx.context(_TASK, budget=100000)

    small_ids = [item.entity_id for section in small.sections for item in section.items]
    medium_ids = [item.entity_id for section in medium.sections for item in section.items]
    large_ids = [item.entity_id for section in large.sections for item in section.items]

    assert small_ids == medium_ids[: len(small_ids)]
    assert medium_ids == large_ids[: len(medium_ids)]
    assert len(small_ids) <= len(medium_ids) <= len(large_ids)
    # SMALL must actually be smaller than LARGE for this to be a real test.
    assert len(small_ids) < len(large_ids)


def test_large_budget_never_admits_unrelated_noise(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    result = cx.context(_TASK, budget=100000)

    for section in result.sections:
        for item in section.items:
            assert "cake" not in item.content
            assert "README" not in item.content
            assert "commit messages" not in item.content


# ---------------------------------------------------------------------------
# N / O / P -- provenance-based redundancy control
# ---------------------------------------------------------------------------


def test_attempt_sharing_evidence_with_root_cause_is_absorbed_not_duplicated(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    result = cx.context(_TASK, budget=100000)

    all_item_ids = {item.entity_id for section in result.sections for item in section.items}
    assert ids["absorbed_attempt"].attempt_id not in all_item_ids

    root_cause_item = next(
        item for section in result.sections if section.heading == SECTION_HISTORY for item in section.items
        if item.entity_id == ids["root_cause"].memory_id
    )
    assert ids["absorbed_attempt"].attempt_id in root_cause_item.provenance


def test_distinct_attempt_with_no_shared_evidence_remains_a_standalone_candidate(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    result = cx.context(_TASK, budget=100000)

    history_items = [item for section in result.sections if section.heading == SECTION_HISTORY for item in section.items]
    standalone = next(item for item in history_items if item.entity_id == ids["standalone_attempt"].attempt_id)
    assert standalone.kind == "attempt"
    assert standalone.provenance == ()


# ---------------------------------------------------------------------------
# R -- explicit lexical-only mode
# ---------------------------------------------------------------------------


def test_lexical_only_mode_is_explicit_and_still_finds_relevant_experience(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    result = cx.context(_TASK, budget=100000)

    assert result.retrieval is not None
    assert "lexical only" in result.retrieval.retrieval_mode()
    assert not result.is_empty()
    assert "Retrieval: lexical only" in result.render()


# ---------------------------------------------------------------------------
# T -- no canonical mutation
# ---------------------------------------------------------------------------


def test_context_never_mutates_canonical_state(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    before_timeline = [(m.memory_id, m.content, m.supersedes) for m in cx.timeline()]
    before_state = [m.memory_id for m in cx.state()]
    before_count = cx._count_memories()

    cx.context(_TASK, budget=50)
    cx.context(_TASK, budget=4000)
    cx.context("an entirely unrelated task about lemon cakes", budget=4000)

    after_timeline = [(m.memory_id, m.content, m.supersedes) for m in cx.timeline()]
    after_state = [m.memory_id for m in cx.state()]
    after_count = cx._count_memories()

    assert before_timeline == after_timeline
    assert before_state == after_state
    assert before_count == after_count


# ---------------------------------------------------------------------------
# X -- terminal safety of render()
# ---------------------------------------------------------------------------


def test_render_output_is_terminal_safe(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(f"{_INVARIANT_RELEVANT}\x1b[2Jinjected", kind="invariant")

    result = cx.context(_TASK, budget=100000)

    assert_output_terminal_safe(result.render())


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


def test_zero_or_negative_budget_is_rejected(tmp_path):
    cx = _workspace(tmp_path)
    with pytest.raises(ValueError):
        cx.context(_TASK, budget=0)
    with pytest.raises(ValueError):
        cx.context(_TASK, budget=-1)


def test_empty_task_is_rejected(tmp_path):
    cx = _workspace(tmp_path)
    with pytest.raises(ValueError):
        cx.context("   ")


# ---------------------------------------------------------------------------
# V / W -- CLI
# ---------------------------------------------------------------------------


def test_cli_context_command(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["context", _TASK])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Retrieval:" in out
    assert SECTION_DECISIONS in out
    assert_output_terminal_safe(out)


def test_cli_context_budget_flag_shrinks_output(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    main(["context", _TASK, "--budget", "100000"])
    large_out = capsys.readouterr().out

    exit_code = main(["context", _TASK, "--budget", "80"])
    small_out = capsys.readouterr().out

    assert exit_code == 0
    assert len(small_out) < len(large_out)


def test_cli_context_on_empty_workspace(tmp_path, monkeypatch, capsys):
    Cortex.init(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["context", "Do anything at all"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Retrieval:" in out
    assert "No compiled context for this task." in out


# ---------------------------------------------------------------------------
# Journey: a realistic single-call developer session (A28-style, lexical only)
# ---------------------------------------------------------------------------


def test_realistic_developer_journey_single_context_call(tmp_path):
    """A new AI session, one `preflight()` and one `context()` call on
    the same realistic workspace, both public API: the compiled context
    must be a strict subset of what preflight shows on the SAME
    categories (never inventing anything preflight would not also admit
    for root causes/lessons/pending), while also carrying the Decision
    preflight cannot, and while excluding the invariant preflight shows
    unconditionally."""
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    preflight_result = cx.preflight(_TASK)
    compiled = cx.context(_TASK)

    compiled_ids = {item.entity_id for section in compiled.sections for item in section.items}
    assert compiled_ids & {m.memory_id for m in preflight_result.root_causes} <= {
        m.memory_id for m in preflight_result.root_causes
    }
    assert {m.memory_id for m in preflight_result.verified_lessons} == {ids["lesson"].memory_id}
    assert {m.memory_id for m in preflight_result.pending} == {ids["pending"].memory_id}

    assert ids["decision"].memory_id in compiled_ids
    assert compiled.invariants_excluded == 1
    assert not compiled.is_empty()
    assert len(compiled.render()) < 4000


# ---------------------------------------------------------------------------
# A36 -- golden contract: item representation IS the selection cost
# ---------------------------------------------------------------------------
#
# A35 established that `compile_context`'s admission decisions are driven
# by the exact character count `_render_item` produces for a candidate
# (see `_context.py`'s `DEFAULT_CONTEXT_BUDGET` comment): the rendered
# text a candidate would occupy on screen IS its budget cost, with no
# canonical, renderer-independent cost model standing between the two.
# One consequence: even a single extra character in `_render_item`'s
# output can push a candidate past the budget boundary. The golden tests
# below freeze every representation `_render_item` produces today,
# byte-for-byte, so an edit to that function is caught here even when it
# "only" touches spacing or a label -- and the test after them
# demonstrates the coupling directly, on the real `compile_context` entry
# point. This protects against ACCIDENTAL drift; it does not promise this
# text is a stable public format -- a deliberate future change updates
# these strings (and, consciously, whatever budget behavior follows from
# it).

_FIXED_RECORDED_AT = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


def _fixed_memory(memory_id: str, content: str, *, epistemic_state: str = "user_asserted") -> Memory:
    return Memory(
        memory_id=memory_id,
        content=content,
        kind="invariant",
        epistemic_state=epistemic_state,
        recorded_at=_FIXED_RECORDED_AT,
    )


def _compile_single_invariant(memory: Memory, budget: int):
    return compile_context(
        task=_TASK,
        budget=budget,
        invariants=(memory,),
        invariants_excluded=0,
        pending=(),
        lessons=(),
        decisions=(),
        root_causes=(),
        known_failures=(),
        recommended_validation_candidates=(),
        open_conflicts=[],
        retrieval=None,
    )


def test_golden_render_item_with_authority_no_extras():
    """The form every Invariant/Pending/Decision/Lesson/Evidence line
    shares: `authority` set, no provenance, no conflict."""
    item = ContextItem(
        entity_id="0" * 32,
        kind="invariant",
        content="Retry handling for failed background jobs must never reorder the queue",
        authority="user_asserted",
    )
    assert _render_item(item) == (
        "- [00000000000000000000000000000000] (user_asserted) "
        "Retry handling for failed background jobs must never reorder the queue"
    )


def test_golden_render_item_without_authority_attempt_form():
    """The one kind with `authority is None`: a standalone Attempt."""
    item = ContextItem(
        entity_id="1" * 32,
        kind="attempt",
        content="Retry failed background jobs without blocking the queue -- Tried retrying synchronously",
        authority=None,
    )
    assert _render_item(item) == (
        "- [11111111111111111111111111111111] "
        "Retry failed background jobs without blocking the queue -- Tried retrying synchronously"
    )


def test_golden_render_item_with_provenance_citation():
    item = ContextItem(
        entity_id="2" * 32,
        kind="root_cause",
        content="Background jobs failed to retry because the queue handling code swallowed the exception",
        authority="user_asserted",
        provenance=("3" * 32, "4" * 32),
    )
    assert _render_item(item) == (
        "- [22222222222222222222222222222222] (user_asserted) "
        "Background jobs failed to retry because the queue handling code swallowed the exception\n"
        "  from attempt [33333333333333333333333333333333], [44444444444444444444444444444444]"
    )


def test_golden_render_item_with_conflict_marker():
    item = ContextItem(
        entity_id="5" * 32,
        kind="decision",
        content="Retry handling for failed background jobs uses exponential backoff capped at 5 attempts",
        authority="user_asserted",
        conflicts_with=("6" * 32,),
    )
    assert _render_item(item) == (
        "- [55555555555555555555555555555555] (user_asserted) "
        "Retry handling for failed background jobs uses exponential backoff capped at 5 attempts\n"
        "  CONFLICTS WITH [66666666666666666666666666666666]"
    )


def test_golden_render_item_with_provenance_and_conflict_combined():
    """A RootCause is the one kind reachable through `compile_context`
    that can carry BOTH an absorbed-Attempt citation and a Conflict
    marker at once: `conflicts_with` is populated from
    `_conflict_partner_map` for every Memory-kind candidate independent
    of `provenance` (see `_memory_item` in `_context.py`). A real
    reachable combination, not a hypothetical one."""
    item = ContextItem(
        entity_id="7" * 32,
        kind="root_cause",
        content="Background jobs failed to retry because the queue handling code swallowed the exception",
        authority="inferred",
        provenance=("8" * 32,),
        conflicts_with=("9" * 32,),
    )
    assert _render_item(item) == (
        "- [77777777777777777777777777777777] (inferred) "
        "Background jobs failed to retry because the queue handling code swallowed the exception\n"
        "  from attempt [88888888888888888888888888888888]\n"
        "  CONFLICTS WITH [99999999999999999999999999999999]"
    )


def test_budget_boundary_admits_item_exactly_at_its_rendered_cost():
    """Anchors `CompiledContext.used` to `_render_item`'s actual byte
    cost, not an independently-computed formula: a budget equal to the
    section-header-plus-item cost admits the item with `used == budget`;
    one character less and the SAME item is entirely excluded, never
    partially rendered (see the PREFIX MONOTONICITY note in
    `compile_context`)."""
    memory = _fixed_memory("a" * 32, "Retry handling for failed background jobs must never reorder the queue")

    exact_cost = _compile_single_invariant(memory, 100000).used
    assert exact_cost > 0

    at_boundary = _compile_single_invariant(memory, exact_cost)
    assert at_boundary.used == exact_cost
    assert at_boundary.omitted == 0
    assert {item.entity_id for section in at_boundary.sections for item in section.items} == {memory.memory_id}

    below_boundary = _compile_single_invariant(memory, exact_cost - 1)
    assert below_boundary.sections == ()
    assert below_boundary.omitted == 1


def test_one_character_longer_representation_flips_budget_admission():
    """The behavioral core of the A35 decision, exercised on the real
    `compile_context` entry point: holding the budget FIXED at exactly
    the cost of one item's current rendering, appending a single
    character to that item's stored content -- one more character in
    `_render_item`'s output -- is enough to move it from admitted to
    entirely omitted. This is why the golden tests above are not
    cosmetic: changing what `_render_item` emits changes what a budgeted
    caller actually receives."""
    short_memory = _fixed_memory("b" * 32, "Retry handling for failed background jobs must never reorder queues")
    long_memory = _fixed_memory("b" * 32, "Retry handling for failed background jobs must never reorder queues.")

    budget = _compile_single_invariant(short_memory, 100000).used

    with_short_content = _compile_single_invariant(short_memory, budget)
    with_long_content = _compile_single_invariant(long_memory, budget)

    assert {item.entity_id for section in with_short_content.sections for item in section.items} == {
        short_memory.memory_id
    }
    assert with_long_content.sections == ()
    assert with_long_content.omitted == 1


# ---------------------------------------------------------------------------
# Q / S -- semantic channel and A27 auto-refresh reuse (real model only)
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_semantic_paraphrase_is_admitted_into_context(tmp_path):
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        cx.semantic_setup()
        evidence = cx.add_evidence("pytest -> 3 passed", kind="test_result")
        lesson = cx.learn(
            "Every retried write must carry a stable idempotency key so the server can "
            "recognize and collapse a duplicate caused by a client-side timeout.",
            supporting_evidence=[evidence],
            verified=True,
        )

        # Zero lexical overlap with the lesson's own wording.
        paraphrase = "how should the payment endpoint classify a request that timed out once and then completed on a retry"
        result = cx.context(paraphrase, budget=100000)

        selected_ids = {item.entity_id for section in result.sections for item in section.items}
        assert lesson.memory_id in selected_ids


@real_model
@skip_without_model
def test_context_reuses_a27_auto_refresh_without_manual_rebuild(tmp_path):
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        cx.semantic_setup()

        evidence = cx.add_evidence("pytest -> 3 passed", kind="test_result")
        lesson = cx.learn(_LESSON_VERIFIED, verified=True, supporting_evidence=[evidence])

        state_before = cx.semantic_state()
        assert state_before.status == "stale"

        result = cx.context(_TASK, budget=100000)

        assert result.retrieval is not None
        assert result.retrieval.refreshed > 0
        assert lesson.memory_id in {item.entity_id for section in result.sections for item in section.items}
        assert cx.semantic_state().status == "ready"
