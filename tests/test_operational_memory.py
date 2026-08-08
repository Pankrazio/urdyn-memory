"""Tests for A9.1 operational-memory kinds: `pending`, `question`,
`invariant`, `environment`.

These are NOT a new canonical primitive. They are specializations of the
existing `Memory` model via `Memory.kind`, and their "current operational
state" is the same derived projection every other kind already gets:
`Cortex.state(kind=...)` over `Memory.supersedes`. These tests exist to
verify that the existing supersession/current-state machinery -- built
and tested for `note`/`decision`/`lesson`/`root_cause` -- behaves
correctly for the new kinds too, including SAME-kind supersession
(environment/invariant revision) and CROSS-kind supersession (a
`pending`/`question` closed by a memory of a different kind), which is
not exercised anywhere else in the suite.
"""

import pytest

from cortex_memory import Cortex


# -- basic recording + persistence --------------------------------------


@pytest.mark.parametrize("kind", ["pending", "question", "invariant", "environment"])
def test_remember_accepts_new_operational_kind(tmp_path, kind):
    cx = Cortex.init(tmp_path, "dev")

    memory = cx.remember(f"a {kind} fact", kind=kind)

    assert memory.kind == kind
    assert memory.content == f"a {kind} fact"


@pytest.mark.parametrize("kind", ["pending", "question", "invariant", "environment"])
def test_new_operational_kind_survives_reopening(tmp_path, kind):
    cx = Cortex.init(tmp_path, "dev")
    original = cx.remember(f"a {kind} fact", kind=kind)
    del cx

    reopened = Cortex.open(tmp_path)
    current = reopened.state(kind=kind)

    assert [m.memory_id for m in current] == [original.memory_id]


@pytest.mark.parametrize("kind", ["pending", "question", "invariant", "environment"])
def test_state_filters_new_kind_from_other_kinds(tmp_path, kind):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("an unrelated note", kind="note")
    target = cx.remember(f"a {kind} fact", kind=kind)

    current = cx.state(kind=kind)

    assert [m.memory_id for m in current] == [target.memory_id]


def test_unknown_kind_is_still_rejected(tmp_path):
    """The new kinds must not have loosened kind validation into
    accepting arbitrary strings."""
    cx = Cortex.init(tmp_path, "dev")

    with pytest.raises(ValueError):
        cx.remember("something", kind="blocked")

    with pytest.raises(ValueError):
        cx.remember("something", kind="technical_debt")

    with pytest.raises(ValueError):
        cx.remember("something", kind="risk")


# -- same-kind supersession (environment, invariant) ---------------------


def test_environment_revision_leaves_only_newest_current(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    new = cx.remember("Python 3.13 is required.", kind="environment", supersedes=old.memory_id)

    current = cx.state(kind="environment")
    history = cx.timeline(kind="environment")

    assert [m.memory_id for m in current] == [new.memory_id]
    assert [m.memory_id for m in history] == [old.memory_id, new.memory_id]


def test_invariant_revision_leaves_only_newest_current(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Migrations may be applied out of order.", kind="invariant")
    new = cx.remember(
        "Migrations must be applied strictly in order.", kind="invariant", supersedes=old.memory_id
    )

    current = cx.state(kind="invariant")
    history = cx.timeline(kind="invariant")

    assert [m.memory_id for m in current] == [new.memory_id]
    assert [m.memory_id for m in history] == [old.memory_id, new.memory_id]


# -- cross-kind supersession (pending, question) --------------------------
#
# Not assumed to work: `MemoryStore.add` validates `supersedes` only
# against "does a memory with this id exist and is it not already
# superseded" (see `_store.py`), with no kind-equality check anywhere in
# that path. These tests exercise the real storage layer to confirm that
# behavior rather than taking it on faith.


def test_pending_completed_by_note_disappears_from_current_pending(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    pending = cx.remember("Run Dev Validation #2.", kind="pending")

    closing_note = cx.remember(
        "Dev Validation #2 completed.", kind="note", supersedes=pending.memory_id
    )

    assert cx.state(kind="pending") == []
    # the closing memory itself is not a pending item
    assert [m.memory_id for m in cx.state(kind="note")] == [closing_note.memory_id]
    # history is preserved under the ORIGINAL kind, not silently reclassified
    history = cx.timeline(kind="pending")
    assert [m.memory_id for m in history] == [pending.memory_id]


def test_pending_completed_by_decision_disappears_from_current_pending(tmp_path):
    """The task's own example uses `note`; `decision` is the other kind
    the design calls out as an appropriate closure kind."""
    cx = Cortex.init(tmp_path, "dev")
    pending = cx.remember("Choose a storage backend.", kind="pending")

    cx.remember("SQLite was chosen as the storage backend.", kind="decision", supersedes=pending.memory_id)

    assert cx.state(kind="pending") == []


def test_question_resolved_by_decision_disappears_from_current_questions(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    question = cx.remember(
        "Should large workspaces use a different semantic retrieval strategy?", kind="question"
    )

    answer = cx.remember(
        "Large workspaces will keep the same strategy until evidence says otherwise.",
        kind="decision",
        supersedes=question.memory_id,
    )

    assert cx.state(kind="question") == []
    assert [m.memory_id for m in cx.state(kind="decision")] == [answer.memory_id]
    history = cx.timeline(kind="question")
    assert [m.memory_id for m in history] == [question.memory_id]


def test_cross_kind_supersession_preserves_full_history_across_both_kinds(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    pending = cx.remember("Write the A9.1 report.", kind="pending")
    closing_note = cx.remember("A9.1 report written.", kind="note", supersedes=pending.memory_id)

    full_history = cx.timeline()

    assert {m.memory_id for m in full_history} == {pending.memory_id, closing_note.memory_id}


# -- question + epistemic_state -------------------------------------------


def test_question_can_be_recorded_user_asserted(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    question = cx.remember("What is causing the flaky test?", kind="question", epistemic_state="user_asserted")

    assert question.epistemic_state == "user_asserted"
    assert [m.memory_id for m in cx.state(kind="question")] == [question.memory_id]


def test_question_can_be_recorded_inferred(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    question = cx.remember(
        "Is the retry logic the actual cause of the timeout?", kind="question", epistemic_state="inferred"
    )

    assert question.epistemic_state == "inferred"
    assert [m.memory_id for m in cx.state(kind="question")] == [question.memory_id]


def test_question_resolution_is_supersession_not_verification(tmp_path):
    """Resolving a question is done by superseding it with the answer,
    never by re-marking the question itself `verified` -- `verified` has
    no meaning for a question (there is nothing to check), and doing so
    would let a second, parallel "is this resolved" signal drift out of
    sync with the supersession graph, which is the only one Cortex
    trusts."""
    cx = Cortex.init(tmp_path, "dev")
    question = cx.remember("Which database should we use?", kind="question")

    with pytest.raises(ValueError):
        # verified requires qualifying evidence and is semantically wrong
        # for a question regardless; this must fail exactly like it would
        # for any other kind, not be special-cased.
        cx.remember("resolved", kind="question", epistemic_state="verified")

    # the original question is still open; nothing about it changed
    assert [m.memory_id for m in cx.state(kind="question")] == [question.memory_id]


# -- deterministic ordering, same guarantee as existing kinds --------------


# -- adversarial ------------------------------------------------------


def test_new_kind_memory_cannot_be_superseded_twice(tmp_path):
    """Same guarantee `test_supersession.py` already locks down for
    `decision`, re-checked for a new operational kind: nothing about
    A9.1 was supposed to touch this constraint, but it is not assumed."""
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember("Python 3.13 is required.", kind="environment", supersedes=old.memory_id)

    with pytest.raises(ValueError):
        cx.remember("Python 3.14 is required.", kind="environment", supersedes=old.memory_id)

    history = cx.timeline(kind="environment")
    assert len(history) == 2


def test_operational_memory_survives_a_copied_workspace(tmp_path):
    """A `.cortex/` directory copied wholesale to a new location (e.g. a
    workspace clone) must still reconstruct the same current operational
    state -- nothing about it may depend on the original path."""
    import shutil

    source = tmp_path / "source"
    cx = Cortex.init(source, "dev")
    old = cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    cx.remember("Canonical IDs must not depend on SQLite row IDs.", kind="invariant", supersedes=old.memory_id)
    del cx

    destination = tmp_path / "copy"
    shutil.copytree(source / ".cortex", destination / ".cortex")

    copied = Cortex.open(destination)
    current = copied.state(kind="invariant")

    assert len(current) == 1
    assert current[0].content == "Canonical IDs must not depend on SQLite row IDs."


def test_operational_kinds_preserve_deterministic_timeline_order(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    first = cx.remember("first pending item", kind="pending")
    second = cx.remember("second pending item", kind="pending")
    third = cx.remember("third pending item", kind="pending")

    history = cx.timeline(kind="pending")

    assert [m.memory_id for m in history] == [first.memory_id, second.memory_id, third.memory_id]
