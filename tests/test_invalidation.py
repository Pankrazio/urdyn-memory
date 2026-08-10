"""Tests for the A11.1 `invalidation` Memory kind.

`invalidation` is NOT a new canonical primitive: it is another
specialization of `Memory`, exactly like the A9.1 operational kinds
(`pending`/`question`/`invariant`/`environment`). It reuses `supersedes`
and the existing current-state projection (`Cortex.state`) unchanged.

Semantics under test throughout this file:

    INVALIDATED != FALSE / DISPROVEN

An `invalidation` Memory means "this prior Memory must no longer be
treated as current/authoritative knowledge" -- it does NOT mean "this
prior Memory was proven wrong". Cortex has no `disproven`/`false` concept
at all; these tests exist partly to demonstrate that recording an
invalidation never mutates, retags, or otherwise alters the original
Memory's own fields (which remain exactly as recorded), only its current-
state projection changes.
"""

import pytest

from cortex_memory import Cortex


# -- basic recording + persistence --------------------------------------


def test_remember_accepts_invalidation_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    inv = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    assert inv.kind == "invalidation"
    assert inv.supersedes == old.memory_id


def test_invalidation_survives_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    original = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )
    del cx

    reopened = Cortex.open(tmp_path)
    (current,) = reopened.state(kind="invalidation")

    assert current.memory_id == original.memory_id
    assert current.supersedes == old.memory_id


# -- withdrawal of authority without a known replacement -----------------


def test_environment_invalidated_without_replacement_has_no_current_environment(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    assert cx.state(kind="environment") == []


def test_invalidation_itself_is_current_in_its_own_kind(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    inv = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    current = cx.state(kind="invalidation")
    assert [m.memory_id for m in current] == [inv.memory_id]


def test_original_memory_is_preserved_unmodified_in_history(tmp_path):
    """Invalidating a Memory must never mutate the original: its own
    `content`/`epistemic_state` stay exactly as recorded. Only its
    current-state projection changes."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    (preserved,) = [m for m in cx.timeline(kind="environment") if m.memory_id == old.memory_id]
    assert preserved.content == "Python 3.12 is required."
    assert preserved.epistemic_state == old.epistemic_state
    assert preserved.supersedes is None


# -- generic state() contract (A11.1 section 9) ---------------------------


def test_generic_state_includes_the_invalidation_and_excludes_the_invalidated_memory(tmp_path):
    """`state()` (no kind filter) is the projection of "what is current".
    An invalidation IS itself current, useful operational knowledge
    ("do not trust this anymore") -- it must not be silently hidden from
    the generic projection just because of its kind."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    inv = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    current_ids = {m.memory_id for m in cx.state()}

    assert inv.memory_id in current_ids
    assert old.memory_id not in current_ids


# -- recall() contract (A11.1 section 10) ---------------------------------


def test_generic_recall_can_surface_a_current_invalidation(tmp_path):
    """`recall()` is generic lexical search over current memories; an
    invalidation's content ("we no longer trust X") is itself current
    knowledge a caller may legitimately be searching for. It must remain
    identifiable via `memory.kind == "invalidation"`, but must not be
    specially excluded from the existing generic retrieval surface."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required for this project.", kind="environment")
    cx.remember(
        "The Python 3.12 runtime requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    results = cx.recall("no longer trusted and must be revalidated")

    assert len(results) == 1
    assert results[0].kind == "invalidation"


# -- preflight() interaction (A11.1 section 11) ---------------------------


def test_invalidated_verified_lesson_disappears_from_preflight(tmp_path):
    """Purely a consequence of the existing current-state filter already
    applied by `Cortex.preflight` -- no new logic is added for this."""
    cx = Cortex.init(tmp_path, "dev")
    task = "Update authentication refresh logic."
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    lesson = cx.learn(
        "Update authentication refresh logic using only the newly issued refresh token.",
        supporting_evidence=[validation],
        verified=True,
    )

    before = cx.preflight(task)
    assert lesson.memory_id in {m.memory_id for m in before.verified_lessons}

    cx.remember(
        "This lesson about refresh-token handling is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=lesson.memory_id,
    )

    after = cx.preflight(task)
    assert lesson.memory_id not in {m.memory_id for m in after.verified_lessons}


# -- invariant interaction (A11.1 section 12) ------------------------------


def test_invalidated_invariant_disappears_from_preflight_invariants(tmp_path):
    """A9.1's `Preflight.invariants` already includes only CURRENT
    invariants (bypassing task relevance entirely). Invalidating an
    invariant removes it from that field for free, through the same
    current-state filter -- no new A9<->A11 integration logic is added."""
    cx = Cortex.init(tmp_path, "dev")
    invariant = cx.remember(".cortex/ must remain gitignored.", kind="invariant")

    before = cx.preflight("Refactor the CLI argument parser.")
    assert invariant.memory_id in {m.memory_id for m in before.invariants}

    cx.remember(
        "This invariant is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=invariant.memory_id,
    )

    after = cx.preflight("Refactor the CLI argument parser.")
    assert invariant.memory_id not in {m.memory_id for m in after.invariants}
    # the invalidation itself must not leak into `invariants`: it is not
    # of kind `invariant`, and A11.1 adds no new Preflight field for it.
    assert all(m.kind == "invariant" for m in after.invariants)


# -- complete replacement chain (A11.1 sections 13/15) ---------------------


def test_complete_chain_original_invalidation_replacement(tmp_path):
    a = None
    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember("Python 3.12 is required.", kind="environment")
    b = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=a.memory_id,
    )
    c = cx.remember("Python 3.13 is required.", kind="environment", supersedes=b.memory_id)

    assert [m.memory_id for m in cx.state(kind="environment")] == [c.memory_id]
    assert cx.state(kind="invalidation") == []

    current_ids = {m.memory_id for m in cx.state()}
    assert current_ids == {c.memory_id}

    history_ids = {m.memory_id for m in cx.timeline()}
    assert history_ids == {a.memory_id, b.memory_id, c.memory_id}


# -- timeline behavior, cross-kind nuance (A11.1 section 14) ---------------


def test_kind_filtered_timeline_excludes_the_invalidation(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember("Python 3.12 is required.", kind="environment")
    b = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=a.memory_id,
    )
    c = cx.remember("Python 3.13 is required.", kind="environment", supersedes=b.memory_id)

    environment_history = cx.timeline(kind="environment")
    invalidation_history = cx.timeline(kind="invalidation")

    assert [m.memory_id for m in environment_history] == [a.memory_id, c.memory_id]
    assert [m.memory_id for m in invalidation_history] == [b.memory_id]


def test_global_timeline_preserves_full_cross_kind_chain_in_order(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    a = cx.remember("Python 3.12 is required.", kind="environment")
    b = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=a.memory_id,
    )
    c = cx.remember("Python 3.13 is required.", kind="environment", supersedes=b.memory_id)

    global_history = cx.timeline()

    assert [m.memory_id for m in global_history] == [a.memory_id, b.memory_id, c.memory_id]


# -- adversarial: double invalidation / already-superseded (sections 16-17) --


def test_double_invalidation_of_the_same_memory_is_rejected(tmp_path):
    """No invalidation-specific validation is added: this is the same
    generic single-supersession guarantee `test_supersession.py` already
    locks down for every other kind."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    with pytest.raises(ValueError):
        cx.remember(
            "A second, independent doubt about the same requirement.",
            kind="invalidation",
            supersedes=old.memory_id,
        )


def test_invalidating_an_already_superseded_memory_is_rejected(tmp_path):
    """The caller must act on the current head of the lineage. An
    ordinary revision (not an invalidation) already occupies the
    supersession slot, so invalidating the original afterwards fails --
    generic behavior, not special-cased for `invalidation`."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember("Python 3.13 is required.", kind="environment", supersedes=old.memory_id)

    with pytest.raises(ValueError):
        cx.remember(
            "Doubt about the original requirement, raised too late.",
            kind="invalidation",
            supersedes=old.memory_id,
        )


def test_invalidation_of_unknown_memory_is_rejected(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("Doubt about something that was never recorded.", kind="invalidation", supersedes="0" * 32)


# -- invalidation of an invalidation (A11.1 section 18) ---------------------


def test_invalidation_of_an_invalidation_is_allowed_by_the_generic_model(tmp_path):
    """Not assumed, verified: nothing in `MemoryStore.add` checks the KIND
    of the memory being superseded, only that it exists and has no
    superseder yet. A second invalidation superseding a first one
    ("even our earlier doubt has itself been superseded") is therefore
    ALLOWED BY THE GENERIC MODEL. A11.1 introduces no new restriction
    here; any future restriction is a FUTURE POSSIBILITY, not decided now."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    first_doubt = cx.remember(
        "The Python 3.12 requirement is no longer trusted.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    second_doubt = cx.remember(
        "Our earlier doubt was itself based on a misreading; re-opening for revalidation.",
        kind="invalidation",
        supersedes=first_doubt.memory_id,
    )

    assert second_doubt.supersedes == first_doubt.memory_id
    assert cx.state(kind="invalidation") == [second_doubt]


# -- standalone invalidation, no supersedes (A11.1 section 35) -------------


def test_invalidation_without_supersedes_is_legitimate(tmp_path):
    """A11.1 must not hardcode an obligation that an invalidation always
    have `supersedes`: Cortex may not hold the original Memory at all
    (e.g. doubt cast on an external assumption never itself recorded).
    `remember()`'s existing, generic validation already allows any kind
    to be recorded without `supersedes`; nothing about `invalidation`
    requires special-casing this."""
    cx = Cortex.init(tmp_path, "dev")

    standalone = cx.remember(
        "An external assumption this project relied on is no longer trusted.",
        kind="invalidation",
    )

    assert standalone.supersedes is None
    assert [m.memory_id for m in cx.state(kind="invalidation")] == [standalone.memory_id]


# -- epistemic state of the invalidation itself (A11.1 section 19) ---------


def test_invalidation_can_be_recorded_user_asserted(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    inv = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )

    assert inv.epistemic_state == "user_asserted"


def test_verified_invalidation_does_not_retag_the_original_memory(tmp_path):
    """`verified` here describes the claim "we should withdraw authority
    from the old requirement" -- it is backed by a real check (a
    command_output showing the old requirement no longer holds in CI).
    It does NOT mean "Python 3.12 is required" was proven false: Cortex
    has no such concept, and the original Memory's own fields stay
    exactly as recorded (asserted below)."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    ci_evidence = cx.add_evidence(
        "CI now runs successfully against Python 3.13 only; 3.12 jobs were removed.",
        kind="command_output",
    )

    inv = cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
        epistemic_state="verified",
        supporting_evidence=[ci_evidence],
    )

    assert inv.epistemic_state == "verified"
    # the ORIGINAL memory is untouched: still whatever it always was,
    # never retagged as "false"/"disproven" -- no such field exists.
    (preserved,) = [m for m in cx.timeline(kind="environment") if m.memory_id == old.memory_id]
    assert preserved.epistemic_state == "user_asserted"
    assert preserved.content == "Python 3.12 is required."


def test_verified_invalidation_requires_qualifying_evidence_like_any_other_kind(tmp_path):
    """No new verification rule is introduced for `invalidation`: the
    generic `remember()` rule (verified requires >=1 qualifying Evidence)
    applies unchanged."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")

    with pytest.raises(ValueError):
        cx.remember(
            "The Python 3.12 requirement is no longer trusted.",
            kind="invalidation",
            supersedes=old.memory_id,
            epistemic_state="verified",
        )


# -- storage/schema surface (A11.1 section 17) ------------------------------


def test_store_schema_version_is_unchanged(tmp_path):
    """A11.1 itself introduced no schema change (kind='invalidation' is a
    pure Python-level `VALID_KINDS` addition, like A9.1's kinds). The
    literal anchor below tracks whatever A12.1/A13.1 (or later) legitimately
    bumped it to since -- see `test_migration_v5.py`/`test_conflict.py` for
    those bumps."""
    from cortex_memory._store import STORE_SCHEMA_VERSION

    assert STORE_SCHEMA_VERSION == 6


# -- copied workspace (A11.1 section 19 test list) --------------------------


def test_invalidation_survives_a_copied_workspace(tmp_path):
    import shutil

    source = tmp_path / "source"
    cx = Cortex.init(source, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember(
        "The Python 3.12 requirement is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )
    del cx

    destination = tmp_path / "copy"
    shutil.copytree(source / ".cortex", destination / ".cortex")

    copied = Cortex.open(destination)
    assert copied.state(kind="environment") == []
    assert len(copied.state(kind="invalidation")) == 1


# -- CLI (A11.1 section 30): the existing `remember --kind` surface --------


def test_cli_remember_accepts_invalidation_kind(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    remember_exit = main(["remember", "Python 3.12 is required.", "--kind", "environment"])
    captured = capsys.readouterr()
    old_id = captured.out.strip().split("[")[1].split("]")[0]

    exit_code = main(
        [
            "remember",
            "The Python 3.12 requirement is no longer trusted and must be revalidated.",
            "--kind",
            "invalidation",
            "--supersedes",
            old_id,
        ]
    )

    captured = capsys.readouterr()
    assert remember_exit == 0
    assert exit_code == 0
    assert "(invalidation)" in captured.out
