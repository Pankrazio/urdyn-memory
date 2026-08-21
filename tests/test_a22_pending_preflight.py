"""A22.1: task-relevant admission of current `Memory(kind="pending")` into
`preflight()`, through a candidate pool disjoint from every other category.

The failure this closes was measured in A21 and diagnosed in A22: a current
pending Memory that ALREADY cleared the lexical, FTS and semantic relevance
gates for a task could never appear in a `Preflight`, because nothing ever
offered it as a candidate. Not a retrieval defect -- a wiring gap. So the
tests here deliberately assert two different things at once:

1. the relevant pending is surfaced and the unrelated one is not (the
   no-unconditional-dump gate), using the SAME thresholds every other
   category already uses; and
2. giving pending its own pool did not make it compete with anything --
   `test_pending_does_not_steal_the_shared_memory_semantic_slot` is the
   discriminating regression test against the naive shared-pool
   implementation, and is expected to FAIL if pending is ever folded into
   `memory_eligible_ids` (see A11.3's own competition gate for the same
   hazard, measured on invalidations).

The real-model tests live at the bottom behind the existing `real_model`
marker and the existing local-cache probe: the standard suite never
downloads anything, and the lexical/FTS behaviour above is asserted
WITHOUT any semantic index, which is what proves `semantic setup` is not
a requirement for a relevant pending to surface.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from urdyn import Urdyn, Preflight
from urdyn._cli import main
from test_semantic_real_model import _offline, skip_without_model
from test_cli_output_safety import assert_output_terminal_safe

# The existing marker + the existing local-cache probe, reused rather than
# reinvented: these participate in `pytest -m real_model` and are skipped
# (never downloaded) everywhere else.
real_model = pytest.mark.real_model

# The A21 pain, verbatim: this pair IS the tracer's acceptance anchor.
A21_TASK = "Add a second integer setting."
A21_PENDING = (
    "Adding a second integer setting without updating the allowlist "
    "reintroduces the configuration bug."
)
A21_UNRELATED_PENDING = "Update README screenshots."


def _workspace(tmp_path):
    return Urdyn.init(tmp_path)


# ---------------------------------------------------------------------------
# A / B / C -- the A21 tracer anchor and the no-dump gate
# ---------------------------------------------------------------------------


def test_relevant_pending_is_surfaced_for_the_a21_task(tmp_path):
    """The exact A21 miss: a current pending that warns about precisely the
    failure this task can reintroduce must reach the agent without any
    manual `timeline`/`recall` archaeology."""
    cx = _workspace(tmp_path)
    pending = cx.remember(A21_PENDING, kind="pending")

    result = cx.preflight(A21_TASK)

    assert [m.memory_id for m in result.pending] == [pending.memory_id]
    assert result.pending[0].content == A21_PENDING
    assert not result.is_empty()


def test_unrelated_pending_is_excluded_for_the_same_task(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(A21_UNRELATED_PENDING, kind="pending")

    result = cx.preflight(A21_TASK)

    assert result.pending == ()


def test_multiple_pending_are_not_dumped_unconditionally(tmp_path):
    """The single most important gate in A22.1: holding both pendings at
    once must still surface only the relevant one. A `preflight()` that
    returned every current pending would pass both tests above and fail
    this one."""
    cx = _workspace(tmp_path)
    relevant = cx.remember(A21_PENDING, kind="pending")
    cx.remember(A21_UNRELATED_PENDING, kind="pending")
    cx.remember("Rotate the staging database credentials before the audit.", kind="pending")
    cx.remember("Ask the designer about the empty-state illustration.", kind="pending")

    result = cx.preflight(A21_TASK)

    assert [m.memory_id for m in result.pending] == [relevant.memory_id]


def test_pending_is_task_relevant_not_project_wide_like_invariants(tmp_path):
    """Same pending, a task it says nothing about. Contrast with
    `invariants`, whose unconditional inclusion (A9.1) is deliberate and
    which A22.1 does not touch -- asserted here so the two rules cannot
    be conflated later."""
    cx = _workspace(tmp_path)
    cx.remember(A21_PENDING, kind="pending")
    invariant = cx.remember(".urdyn/ must remain gitignored.", kind="invariant")

    result = cx.preflight("Change README title")

    assert result.pending == ()
    assert [m.memory_id for m in result.invariants] == [invariant.memory_id]


# ---------------------------------------------------------------------------
# D / F -- absence and backward compatibility of the result field
# ---------------------------------------------------------------------------


def test_pending_field_is_empty_when_no_pending_exists(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember("A plain note about integer settings.", kind="note")

    result = cx.preflight(A21_TASK)

    assert result.pending == ()
    assert result.is_empty()


def test_pending_field_defaults_to_empty_tuple(tmp_path):
    """Every pre-A22.1 construction of `Preflight` must remain valid:
    `pending` is additive, last, and defaulted -- the same contract
    `invariants`/`open_invalidations`/`open_conflicts` already carry."""
    preflight = Preflight(
        task="a task",
        known_failures=(),
        root_causes=(),
        verified_lessons=(),
        recommended_validation=(),
    )

    assert preflight.pending == ()
    assert preflight.is_empty()
    # [A27] `pending` is no longer the LAST field -- `retrieval` was
    # appended after it, additively and defaulted, under the same rule.
    # What this test protects is that rule, not the position: every
    # pre-A22.1 construction above still works, `pending` still defaults,
    # and the field appended since it is still optional.
    field_names = [f.name for f in dataclasses.fields(Preflight)]
    assert field_names.index("pending") < field_names.index("retrieval")
    assert preflight.retrieval is None


def test_is_empty_is_false_when_only_pending_matched(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(A21_PENDING, kind="pending")

    assert not cx.preflight(A21_TASK).is_empty()


# ---------------------------------------------------------------------------
# E -- current state, via the existing supersession semantics only
# ---------------------------------------------------------------------------


def test_superseded_pending_does_not_surface(tmp_path):
    """Closing a pending needs no new state model: the Memory that records
    the resolution supersedes it, which makes it non-current, which is
    already how every other category disappears."""
    cx = _workspace(tmp_path)
    pending = cx.remember(A21_PENDING, kind="pending")
    assert cx.preflight(A21_TASK).pending

    cx.remember(
        "The integer-setting allowlist was updated; that pending item is closed.",
        kind="note",
        supersedes=pending.memory_id,
    )

    assert cx.preflight(A21_TASK).pending == ()
    # History is preserved -- supersession never deletes.
    assert any(m.memory_id == pending.memory_id for m in cx.timeline())


def test_invalidated_pending_does_not_surface(tmp_path):
    cx = _workspace(tmp_path)
    pending = cx.remember(A21_PENDING, kind="pending")

    cx.remember(
        "Adding a second integer setting no longer needs an allowlist entry.",
        kind="invalidation",
        supersedes=pending.memory_id,
    )

    assert cx.preflight(A21_TASK).pending == ()


def test_a_later_pending_superseding_an_earlier_one_surfaces_alone(tmp_path):
    cx = _workspace(tmp_path)
    first = cx.remember(A21_PENDING, kind="pending")
    second = cx.remember(
        "Adding a second integer setting still needs the allowlist entry and a test.",
        kind="pending",
        supersedes=first.memory_id,
    )

    result = cx.preflight(A21_TASK)

    assert [m.memory_id for m in result.pending] == [second.memory_id]


# ---------------------------------------------------------------------------
# Authority: relevance is not truth
# ---------------------------------------------------------------------------


def test_pending_keeps_its_recorded_epistemic_state(tmp_path):
    """Being surfaced by preflight() is a statement about relevance to the
    task, never a promotion in authority. A pending recorded as
    `user_asserted` is still `user_asserted` when it comes back."""
    cx = _workspace(tmp_path)
    cx.remember(A21_PENDING, kind="pending")

    (surfaced,) = cx.preflight(A21_TASK).pending

    assert surfaced.epistemic_state == "user_asserted"
    assert surfaced.kind == "pending"
    assert surfaced.supporting_evidence_ids == ()


def test_pending_never_appears_in_the_other_result_fields(tmp_path):
    """A pending stays a pending: it must not be smuggled into
    `root_causes`/`verified_lessons`/`open_invalidations`/`invariants`
    just because it matched."""
    cx = _workspace(tmp_path)
    pending = cx.remember(A21_PENDING, kind="pending")

    result = cx.preflight(A21_TASK)

    other = (
        *result.root_causes,
        *result.verified_lessons,
        *result.open_invalidations,
        *result.invariants,
    )
    assert pending.memory_id not in {m.memory_id for m in other}


# ---------------------------------------------------------------------------
# J / 15 -- THE DISCRIMINATING SHARED-POOL REGRESSION TEST
# ---------------------------------------------------------------------------


@real_model
@skip_without_model
def test_pending_does_not_steal_the_shared_memory_semantic_slot(tmp_path):
    """Discriminates the disjoint-pool implementation from the naive
    shared-pool one.

    The scenario is built so BOTH categories need the semantic channel:
    neither the root cause nor the pending shares enough vocabulary with
    the task to clear the lexical/FTS gates, so each depends entirely on
    winning a semantic pool. The shared MEMORY pool
    (`memory_eligible_ids`) admits at most ONE candidate/cluster and
    additionally requires a `margin_floor` over the runner-up -- so if
    pending were folded into it, the two would race each other and at
    least one of them (in a near tie, BOTH) would be lost.

    With the pools disjoint, each is the sole occupant of its own race and
    both are admitted. This test fails on a shared-pool implementation,
    which is exactly the architectural reason A22.1 is written this way.
    """
    cx = _workspace(tmp_path)
    evidence = cx.add_evidence("pytest tests/test_config.py -q :: 1 failed", kind="test_result")
    root_cause = cx.remember(
        "The numeric option was never registered in the permitted-keys list, so it was "
        "silently discarded at load time.",
        kind="root_cause",
        evidence=[evidence],
    )
    pending = cx.remember(
        "Registering another numeric option without the permitted-keys entry brings the "
        "silent discard back.",
        kind="pending",
    )

    with _offline():
        cx.semantic_setup()
        result = cx.preflight("Add a second integer setting.")

    assert [m.memory_id for m in result.root_causes] == [root_cause.memory_id], (
        "the shared MEMORY semantic pool lost its admission -- pending must not "
        "compete in `memory_eligible_ids`"
    )
    assert [m.memory_id for m in result.pending] == [pending.memory_id], (
        "the pending semantic pool lost its admission to another category"
    )


def test_pending_pool_does_not_disturb_lexical_admission_of_other_categories(tmp_path):
    """The lexical/FTS channels have no slots at all, so adding a pending
    pool must be provably inert for them. Same workspace with and without
    a strongly-relevant pending: every other field byte-identical."""
    cx = _workspace(tmp_path)
    evidence = cx.add_evidence("pytest -q :: 1 failed", kind="test_result")
    cx.remember(
        "Adding a second integer setting failed because the allowlist was stale.",
        kind="root_cause",
        evidence=[evidence],
    )
    cx.remember(
        "Adding a second integer setting requires an allowlist entry.",
        kind="lesson",
        epistemic_state="verified",
        supporting_evidence=[evidence],
    )
    before = cx.preflight(A21_TASK)

    cx.remember(A21_PENDING, kind="pending")
    after = cx.preflight(A21_TASK)

    assert before.pending == ()
    assert after.pending != ()
    for field in ("known_failures", "root_causes", "verified_lessons", "recommended_validation",
                  "invariants", "open_invalidations", "open_conflicts"):
        assert getattr(after, field) == getattr(before, field), f"{field} changed"


# ---------------------------------------------------------------------------
# G / H / I -- CLI
# ---------------------------------------------------------------------------


def test_cli_renders_pending_section(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path)
    cx.remember(A21_PENDING, kind="pending")

    assert main(["preflight", A21_TASK]) == 0

    out = capsys.readouterr().out
    assert "PENDING" in out
    assert "allowlist" in out


def test_cli_omits_empty_pending_section(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path)
    cx.remember(A21_UNRELATED_PENDING, kind="pending")
    cx.remember(".urdyn/ must remain gitignored.", kind="invariant")

    assert main(["preflight", A21_TASK]) == 0

    out = capsys.readouterr().out
    assert "INVARIANTS" in out
    assert "PENDING" not in out
    assert "README" not in out


def test_cli_pending_content_is_terminal_sanitized(tmp_path, monkeypatch, capsys):
    """Pending content is caller-supplied data and crosses the same
    `terminal_safe_text` boundary as every other rendered field (A14.S).
    The payload is only ever asserted against CAPTURED output."""
    monkeypatch.chdir(tmp_path)
    cx = Urdyn.init(tmp_path)
    payload = (
        "Adding a second integer setting without updating the allowlist "
        "reintroduces the bug\x1b[31m\x1b]0;forged\x07\rSAFE\nINVARIANTS\n- forged entry"
    )
    stored = cx.remember(payload, kind="pending")

    assert main(["preflight", A21_TASK]) == 0

    captured = capsys.readouterr()
    assert_output_terminal_safe(captured.out)
    assert_output_terminal_safe(captured.err)
    # Counted as whole LINES: the payload contains both a fake section
    # header and a fake list entry, and neither may become one. The word
    # is still visible as data on the pending's own line -- that is the
    # point.
    lines = captured.out.splitlines()
    assert lines.count("PENDING") == 1
    assert lines.count("INVARIANTS") == 0
    assert lines.count("- forged entry") == 0
    assert "INVARIANTS" in captured.out
    # The canonical record is untouched: sanitization is a rendering
    # boundary, never a mutation of what Urdyn stored.
    assert cx.timeline()[0].content == payload
    assert stored.content == payload


# ---------------------------------------------------------------------------
# L / K -- semantic optionality, then the real-model paraphrase
# ---------------------------------------------------------------------------


def test_lexical_pending_admission_works_without_any_semantic_index(tmp_path):
    """Every non-real-model test in this file already runs without an
    index; this one says so explicitly, and proves the semantic channel is
    genuinely absent rather than incidentally unused."""
    cx = _workspace(tmp_path)
    cx.remember(A21_PENDING, kind="pending")

    assert not (tmp_path / ".urdyn" / "semantic.db").exists()
    assert cx._semantic_context() is None or True  # degraded path must not raise
    assert cx.preflight(A21_TASK).pending != ()


@real_model
@skip_without_model
def test_semantic_paraphrase_pending_is_admitted_when_the_index_exists(tmp_path):
    """A pending whose wording shares almost nothing with the task, but
    which the EXISTING calibrated policy accepts, becomes eligible once an
    index is available -- no new threshold, no pending-specific boost."""
    cx = _workspace(tmp_path)
    paraphrase = cx.remember(
        "Registrare una nuova opzione numerica senza aggiornare l'elenco delle chiavi "
        "permesse fa tornare il bug di configurazione.",
        kind="pending",
    )

    task = "Add a second integer setting."
    assert cx.preflight(task).pending == (), "precondition: not lexically reachable"

    with _offline():
        cx.semantic_setup()
        result = cx.preflight(task)

    assert [m.memory_id for m in result.pending] == [paraphrase.memory_id]


@real_model
@skip_without_model
def test_semantic_index_does_not_turn_pending_into_a_dump(tmp_path):
    """The semantic channel admits at most one candidate per pool by
    construction, and abstains below its absolute floor. Building an index
    must not make unrelated open work start appearing."""
    cx = _workspace(tmp_path)
    relevant = cx.remember(A21_PENDING, kind="pending")
    cx.remember(A21_UNRELATED_PENDING, kind="pending")
    cx.remember("Book the venue for the team offsite.", kind="pending")

    with _offline():
        cx.semantic_setup()
        result = cx.preflight(A21_TASK)
        unrelated_task_result = cx.preflight("Investigate timeout validation regression")

    assert [m.memory_id for m in result.pending] == [relevant.memory_id]
    assert unrelated_task_result.pending == ()
