"""A11.3: `Preflight.open_invalidations` -- surfacing CURRENT, task-relevant
`invalidation` Memory through `preflight()`.

Before A11.3, an agent calling `cx.preflight(task)` could not distinguish
"Cortex has nothing on record about this" from "Cortex had something on
record and explicitly withdrew its authority". This file locks down the
new field's contract and, most importantly, the SEMANTIC COMPETITION GATE
(A11.2's original proposal -- sharing the root-cause/lesson semantic
eligible-id pool with invalidations -- was found unsafe before it was ever
implemented; A11.3 uses a disjoint, independently-ranked pool instead, and
the tests below prove both why the naive approach is unsafe and that the
real implementation avoids it).
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex_memory import Cortex

# ---------------------------------------------------------------------------
# Fake deterministic embedding backend -- identical technique to
# test_semantic.py / test_a7_8_regression.py, reused rather than
# reinvented so the competition gate below is directly comparable to the
# A7.8 regression it mirrors.
# ---------------------------------------------------------------------------

_FAKE_CONCEPTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
_FAKE_NONE_INDEX = len(_FAKE_CONCEPTS)
_FAKE_DIM = len(_FAKE_CONCEPTS) + 1


class _FakeStaticModel:
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


_DILUTED_QUERY = "totally unrelated wording that still somehow concerns alpha topics here and nothing else"


# ---------------------------------------------------------------------------
# SEMANTIC COMPETITION GATE
# ---------------------------------------------------------------------------


def test_semantic_competition_gate_naive_shared_pool_would_suppress_root_cause(tmp_path, fake_semantic):
    """Calls the existing, UNMODIFIED `_preflight_memory_semantic_widen`
    directly with a naive shared `memory_eligible_ids` pool -- exactly
    what a naive reading of A11.2's proposal would do -- to reproduce,
    with real code, why A11.3 does NOT share this pool between root
    causes/lessons and invalidations. Mirrors
    `test_preflight_cluster_still_loses_to_a_stronger_unrelated_competitor`
    in test_a7_8_regression.py, which already proves the same mechanism
    suppresses one ordinary lesson in favor of a stronger unrelated one;
    this test proves an unrelated invalidation can do the exact same harm
    if allowed into the same pool. Does NOT exercise `Cortex.preflight()`'s
    real code path -- that is proven safe separately, below."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    # weak candidate: "alpha beta" scores ~0.707 against a pure "alpha" query
    root_cause = cx.remember("alpha beta", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    # unrelated invalidation, no shared evidence, strong pure "alpha" match (score 1.0)
    unrelated_env = cx.remember("some environment fact", kind="environment")
    invalidation = cx.remember("alpha", kind="invalidation", supersedes=unrelated_env.memory_id)
    cx.semantic_setup()

    baseline = cx._preflight_memory_semantic_widen(
        _DILUTED_QUERY,
        root_cause_memories=[root_cause],
        verified_lesson_memories=[],
        memory_eligible_ids=frozenset({root_cause.memory_id}),
    )
    assert baseline == frozenset({root_cause.memory_id})

    naive = cx._preflight_memory_semantic_widen(
        _DILUTED_QUERY,
        root_cause_memories=[root_cause],
        verified_lesson_memories=[],
        memory_eligible_ids=frozenset({root_cause.memory_id, invalidation.memory_id}),
    )
    # REGRESSION reproduced: sharing the pool lets the unrelated, more
    # strongly-scoring invalidation win the pool's single admission slot
    # outright (clears both absolute floor and margin against the root
    # cause) -- the previously-admitted root cause is no longer in the
    # returned set at all.
    assert root_cause.memory_id not in naive
    assert naive == frozenset({invalidation.memory_id})


def test_semantic_competition_gate_real_preflight_preserves_root_cause_admission(tmp_path, fake_semantic):
    """The real `Cortex.preflight()` code path, using disjoint pools,
    does NOT reproduce the regression above: the root cause is still
    admitted, and the invalidation is independently admitted into its
    own field, because the two are never in the same race."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    root_cause = cx.remember("alpha beta", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    unrelated_env = cx.remember("some environment fact", kind="environment")
    invalidation = cx.remember("alpha", kind="invalidation", supersedes=unrelated_env.memory_id)
    cx.semantic_setup()

    result = cx.preflight(_DILUTED_QUERY)

    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}


def test_semantic_competition_gate_preserves_verified_lesson_admission(tmp_path, fake_semantic):
    """Same gate, exercised against `verified_lessons` instead of
    `root_causes`, since both pools feed the same shared machinery."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("alpha beta", evidence=[ev], supporting_evidence=[validation], verified=True)
    unrelated_env = cx.remember("some environment fact", kind="environment")
    invalidation = cx.remember("alpha", kind="invalidation", supersedes=unrelated_env.memory_id)
    cx.semantic_setup()

    result = cx.preflight(_DILUTED_QUERY)

    assert {m.memory_id for m in result.verified_lessons} == {lesson.memory_id}
    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}


def test_preflight_regression_existing_fields_unaffected_by_irrelevant_and_close_invalidations(
    tmp_path, fake_semantic
):
    """PREFLIGHT REGRESSION (mandatory): compares exact IDs, not counts.
    A pre-existing Root Cause + Verified Lesson + Recommended Validation
    scenario must produce byte-identical `root_causes`/`verified_lessons`/
    `recommended_validation` before and after (A) an irrelevant current
    invalidation and (B) a semantically-close-but-operationally-different
    current invalidation are added to the workspace."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    root_cause = cx.remember("alpha beta", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("alpha beta", evidence=[ev], supporting_evidence=[validation], verified=True)
    cx.semantic_setup()

    baseline = cx.preflight(_DILUTED_QUERY)
    assert {m.memory_id for m in baseline.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in baseline.verified_lessons} == {lesson.memory_id}
    assert {e.evidence_id for e in baseline.recommended_validation} == {validation.evidence_id}

    # A. completely irrelevant invalidation (a different concept axis)
    unrelated_env = cx.remember("Node.js version pinned.", kind="environment")
    cx.remember("gamma", kind="invalidation", supersedes=unrelated_env.memory_id)
    cx.semantic_setup()
    after_a = cx.preflight(_DILUTED_QUERY)
    assert {m.memory_id for m in after_a.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in after_a.verified_lessons} == {lesson.memory_id}
    assert {e.evidence_id for e in after_a.recommended_validation} == {validation.evidence_id}
    assert after_a.open_invalidations == ()

    # B. semantically close (same "alpha" concept axis, strong score) but
    # operationally different -- no shared Evidence with the root cause/lesson.
    another_env = cx.remember("Another environment fact.", kind="environment")
    invalidation_b = cx.remember("alpha", kind="invalidation", supersedes=another_env.memory_id)
    cx.semantic_setup()
    after_b = cx.preflight(_DILUTED_QUERY)
    assert {m.memory_id for m in after_b.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in after_b.verified_lessons} == {lesson.memory_id}
    assert {e.evidence_id for e in after_b.recommended_validation} == {validation.evidence_id}
    assert {m.memory_id for m in after_b.open_invalidations} == {invalidation_b.memory_id}


# ---------------------------------------------------------------------------
# North Star / negative controls (lexical only -- no semantic extra needed)
# ---------------------------------------------------------------------------


def test_open_invalidations_north_star_environment(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    invalidation = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}
    assert result.is_empty() is False
    # no resurrection: the old environment Memory itself is not present
    # anywhere in the result.
    assert all(m.kind != "environment" for m in result.open_invalidations)


def test_open_invalidations_negative_control_unrelated_css_task(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )

    result = cx.preflight("Improve mobile CSS layout.")

    assert result.open_invalidations == ()


def test_open_invalidations_negative_control_unrelated_payment_task(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_decision = cx.remember("Use SQLite for V1 storage.", kind="decision")
    cx.remember(
        "The SQLite storage decision is no longer settled; database migration strategy is not trusted.",
        kind="invalidation",
        supersedes=old_decision.memory_id,
    )

    result = cx.preflight("Handle payment retry idempotency.")

    assert result.open_invalidations == ()


def test_two_invalidations_only_the_relevant_one_surfaces(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    relevant = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )
    old_api = cx.remember("Service X endpoint supports API v2.", kind="note")
    cx.remember(
        "Do not trust the old Service X API v2 endpoint anymore; it must be reconfirmed.",
        kind="invalidation",
        supersedes=old_api.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert {m.memory_id for m in result.open_invalidations} == {relevant.memory_id}


def test_ordering_of_multiple_relevant_invalidations_is_oldest_first(tmp_path):
    """Same ordering convention already used for `root_causes`/
    `verified_lessons`: the order `store.timeline(kind)` recorded them
    in, not a new ranking score."""
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    first = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )
    old_env_2 = cx.remember("Python 3.12 is also required in the CI runtime image.", kind="environment")
    second = cx.remember(
        "The Python 3.12 CI runtime requirement must also be revalidated.",
        kind="invalidation",
        supersedes=old_env_2.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert [m.memory_id for m in result.open_invalidations] == [first.memory_id, second.memory_id]


# ---------------------------------------------------------------------------
# invariant / lesson / decision dogfood
# ---------------------------------------------------------------------------


def test_invalidated_invariant_does_not_inherit_always_include_policy(tmp_path):
    """Negative control (mandatory): an invalidation superseding an
    invariant must NOT be auto-included the way the invariant itself
    would have been -- it goes through ordinary relevance matching."""
    cx = Cortex.init(tmp_path, "dev")
    invariant = cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    cx.remember(
        "This invariant is no longer trusted and must be revalidated.",
        kind="invalidation",
        supersedes=invariant.memory_id,
    )

    result = cx.preflight("Refactor the CLI argument parser.")

    assert result.invariants == ()
    assert result.open_invalidations == ()


def test_invalidated_invariant_surfaces_when_relevant(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    invariant = cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    invalidation = cx.remember(
        "The .cortex gitignore invariant is no longer trusted.",
        kind="invalidation",
        supersedes=invariant.memory_id,
    )

    result = cx.preflight("Is the .cortex gitignore invariant still trusted?")

    assert invariant.memory_id not in {m.memory_id for m in result.invariants}
    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}


def test_invalidated_verified_lesson_disappears_and_invalidation_surfaces(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    lesson = cx.learn(
        "Update authentication refresh logic by using only the newly issued refresh token.",
        supporting_evidence=[validation],
        verified=True,
    )
    task = "Update authentication refresh logic."
    before = cx.preflight(task)
    assert lesson.memory_id in {m.memory_id for m in before.verified_lessons}

    invalidation = cx.remember(
        "The refresh token lesson for authentication refresh logic is no longer trusted.",
        kind="invalidation",
        supersedes=lesson.memory_id,
    )

    after = cx.preflight(task)
    assert lesson.memory_id not in {m.memory_id for m in after.verified_lessons}
    assert {m.memory_id for m in after.open_invalidations} == {invalidation.memory_id}


def test_invalidated_decision_surfaces_in_open_invalidations(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    decision = cx.remember("Use SQLite for V1 storage.", kind="decision")
    invalidation = cx.remember(
        "The SQLite storage decision is no longer settled; persistence design must be reconsidered.",
        kind="invalidation",
        supersedes=decision.memory_id,
    )

    result = cx.preflight("Design persistence changes for V1 storage.")

    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}
    # no new "decision" section is introduced; the decision itself never
    # surfaces anywhere in Preflight, before or after A11.3.
    assert not hasattr(result, "decisions")


# ---------------------------------------------------------------------------
# standalone / closed / chained / epistemic state
# ---------------------------------------------------------------------------


def test_standalone_invalidation_surfaces_when_relevant(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    standalone = cx.remember(
        "An external assumption about payment gateway idempotency is no longer trusted.",
        kind="invalidation",
    )

    result = cx.preflight("Handle payment gateway idempotency guarantees.")

    assert {m.memory_id for m in result.open_invalidations} == {standalone.memory_id}


def test_closed_invalidation_disappears_from_open_invalidations(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required for this project.", kind="environment")
    invalidation = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )
    before = cx.preflight("Update the Python runtime.")
    assert {m.memory_id for m in before.open_invalidations} == {invalidation.memory_id}

    cx.remember(
        "Python 3.13 is required for this project.", kind="environment", supersedes=invalidation.memory_id
    )

    after = cx.preflight("Update the Python runtime.")
    assert after.open_invalidations == ()


def test_chained_invalidation_only_current_head_surfaces(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old = cx.remember("Python 3.12 is required for this project.", kind="environment")
    first = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old.memory_id,
    )
    second = cx.remember(
        "Our earlier doubt about the Python 3.12 runtime requirement needs revalidation too.",
        kind="invalidation",
        supersedes=first.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert {m.memory_id for m in result.open_invalidations} == {second.memory_id}


def test_open_invalidations_admits_user_asserted(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_api = cx.remember("Service X endpoint supports API v2.", kind="note")
    invalidation = cx.remember(
        "Do not trust the old Service X API v2 endpoint anymore.",
        kind="invalidation",
        supersedes=old_api.memory_id,
    )

    result = cx.preflight("Reconfirm whether the Service X API v2 endpoint is still trusted.")

    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}
    assert result.open_invalidations[0].epistemic_state == "user_asserted"


def test_open_invalidations_admits_verified(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_decision = cx.remember("Use SQLite for V1 storage.", kind="decision")
    ci_evidence = cx.add_evidence("CI now shows the SQLite backend failing under load.", kind="command_output")
    invalidation = cx.remember(
        "The SQLite storage decision is no longer trusted; CI shows it failing under load.",
        kind="invalidation",
        supersedes=old_decision.memory_id,
        epistemic_state="verified",
        supporting_evidence=[ci_evidence],
    )

    result = cx.preflight("Reconsider the SQLite storage decision given CI failures under load.")

    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}
    assert result.open_invalidations[0].epistemic_state == "verified"


# ---------------------------------------------------------------------------
# is_empty(), backward compatibility, no-semantic-extra fallback
# ---------------------------------------------------------------------------


def test_is_empty_is_false_when_only_open_invalidations_is_populated(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert result.known_failures == ()
    assert result.root_causes == ()
    assert result.verified_lessons == ()
    assert result.recommended_validation == ()
    assert result.invariants == ()
    assert result.open_invalidations != ()
    assert result.is_empty() is False


def test_open_invalidations_works_without_the_semantic_extra(tmp_path, monkeypatch):
    """Lexical/FTS admission alone must be sufficient for a clear case --
    semantic is optional widening, never a requirement."""
    from cortex_memory import _workspace

    monkeypatch.setattr(_workspace, "_load_semantic_module", lambda: None)
    cx = Cortex.init(tmp_path, "dev")
    old_env = cx.remember("Python 3.12 is required for this project.", kind="environment")
    invalidation = cx.remember(
        "The Python 3.12 runtime requirement must be revalidated.",
        kind="invalidation",
        supersedes=old_env.memory_id,
    )

    result = cx.preflight("Update the Python runtime.")

    assert {m.memory_id for m in result.open_invalidations} == {invalidation.memory_id}


# ---------------------------------------------------------------------------
# storage / schema surface
# ---------------------------------------------------------------------------


def test_store_schema_version_still_unchanged(tmp_path):
    """A11.3 itself introduced no schema change, same reasoning as
    `test_invalidation.py`'s equivalent anchor -- see there for why the
    literal below now reflects A13.1's legitimate v6 bump instead."""
    from cortex_memory._store import STORE_SCHEMA_VERSION

    assert STORE_SCHEMA_VERSION == 6


# ---------------------------------------------------------------------------
# Guard is untouched
# ---------------------------------------------------------------------------


def test_guard_result_has_no_invalidation_field(tmp_path):
    """A11.2's explicit decision: Guard awareness is deferred, not part
    of A11.3. `GuardResult`'s shape must be unchanged."""
    import dataclasses

    from cortex_memory import GuardResult

    field_names = {f.name for f in dataclasses.fields(GuardResult)}

    assert field_names == {"action", "known_failures", "applicable_skills", "recommended_validation"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_preflight_shows_open_invalidations_section_when_present(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "Python 3.12 is required for this project.", "--kind", "environment"])
    captured = capsys.readouterr()
    old_id = captured.out.strip().split("[")[1].split("]")[0]
    main(
        [
            "remember",
            "The Python 3.12 runtime requirement must be revalidated.",
            "--kind",
            "invalidation",
            "--supersedes",
            old_id,
        ]
    )
    capsys.readouterr()

    exit_code = main(["preflight", "Update the Python runtime."])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OPEN INVALIDATIONS" in captured.out
    assert "must be revalidated" in captured.out


def test_cli_preflight_omits_open_invalidations_section_when_empty(tmp_path, monkeypatch, capsys):
    from cortex_memory._cli import main

    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["preflight", "some task nobody has attempted"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OPEN INVALIDATIONS" not in captured.out
