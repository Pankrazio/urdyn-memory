"""A7.4 real-model integration test: `minishlab/potion-retrieval-32M` via
`model2vec`, against a REAL Cortex workspace built through the public API.

Skipped entirely -- with an explicit reason, never silently -- unless
the model is already cached locally (checked via
`huggingface_hub.scan_cache_dir()`, never downloaded here) and the
`[semantic]` extra is importable. This file is never part of a normal
`uv run pytest` run's *requirements*; it participates only when the
environment already has what it needs, exactly the "real model
integration test, run explicitly before Human Review" split section 32
of the A7.4 brief asks for.

This is also where A7.4's calibration is checked against genuinely NEW
queries: `test_holdout_new_semantic_positive_paraphrases_not_used_in_calibration`
uses paraphrases written fresh for this test, never seen while picking
`_semantic.SEMANTIC_POLICY`'s floors (see `_semantic.py`'s module
docstring and the A7.4 report for how those floors were calibrated, on
the frozen A7.3 corpus).
"""

from __future__ import annotations

import os

import pytest

from cortex_memory import Cortex

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_MODEL_ID = "minishlab/potion-retrieval-32M"


def _real_model_available() -> bool:
    try:
        import huggingface_hub
        import model2vec  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    try:
        cache_info = huggingface_hub.scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == _MODEL_ID and repo.repo_type == "model" for repo in cache_info.repos)


pytestmark = pytest.mark.real_model
_SKIP_REASON = (
    f"real model2vec model {_MODEL_ID!r} is not cached locally (and/or the "
    "'semantic' extra is not installed) -- run 'cortex semantic setup' in a "
    "scratch workspace once to populate the Hugging Face cache, then re-run "
    "this file; never downloaded automatically by the test suite itself"
)
skip_without_model = pytest.mark.skipif(not _real_model_available(), reason=_SKIP_REASON)


def _offline():
    """Context manager forcing HF_HUB_OFFLINE for the duration of a
    block, to prove retrieval genuinely needs no network -- restores
    whatever was there before on exit."""

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


def _build_human_acceptance_workspace(tmp_path):
    """Real Cortex workspace, built only through the public API, holding
    the exact Human Acceptance lesson/skill from A7.2/A7.3 plus the
    entities needed for the two mandatory adversarial regression cases:
    the payment-guard-clause false positive and the CSS hard negative.
    Content is copied verbatim from the frozen A7.3 corpus
    (`/tmp/cortex-a7-semantic-eval/corpus/corpus.py`, entities `d1-*`,
    `hn-css`), not re-derived or reworded.
    """
    cx = Cortex.init(tmp_path, "dev")

    cx.record_attempt(
        task="Diagnose why preflight and guard found nothing relevant for a rephrased task",
        approach="Assumed nothing relevant had been recorded before",
        outcome="failed",
    )
    root_cause_evidence = cx.add_evidence(
        "preflight and guard rely too much on exact query wording.", kind="user_statement"
    )
    verification = cx.add_evidence(
        "Reproduced: a rephrased task missed experience recorded under different wording.",
        kind="user_confirmation",
    )
    lesson = cx.learn(
        "preflight() and guard() can miss relevant experience that exists when the "
        "task is worded differently than how it was recorded.",
        evidence=[root_cause_evidence],
        supporting_evidence=[verification],
        verified=True,
    )
    skill = cx.promote(
        lesson,
        name="Reword before trusting an empty guard result",
        purpose="preflight or guard can miss experience that is relevant and recorded.",
        steps=["Rephrase the task in different words", "Run preflight/guard again before concluding nothing exists"],
    )

    # hard negative: an entirely unrelated attempt, present so the
    # semantic index pool is not artificially tiny/single-candidate.
    cx.record_attempt(
        task="Change CSS button color to blue",
        approach="Updated the stylesheet",
        outcome="succeeded",
    )

    cx.semantic_setup()
    return cx, lesson, skill


@skip_without_model
def test_human_acceptance_recall_matches_the_calibrated_policy(tmp_path):
    """The 4 verbatim Human Acceptance queries from A7.2/A7.3, classified
    explicitly (A7.7 section 14: "classifica esplicitamente gli abstain
    intenzionali", not "pretend a failed query passed"):

      - `ha-preflight-2`, `ha-guard-1`: RECOVERED (A7.4 baseline, unchanged).
      - `ha-preflight-1`: INTENTIONAL ABSTENTION. A7.6/A7.7 measured its
        nearest real negative neighbor (`bn-9`, a hard negative from the
        A7.3 corpus) at a score only 0.0018 apart -- see
        `test_near_floor_path_stays_rejected_bn9_and_ha_preflight_1_are_not_separable`
        below for the permanent evidence this is not a robust basis for a
        "near-floor" admission rule, per the explicit A7.7 instruction to
        prefer losing this recall case over shipping a threshold that
        fragile. Rigorous structural corroboration (A7.7's one preflight
        widening mechanism) does not rescue it either: its related failed
        Attempt scores even lower, independently, than the lesson itself.
      - `ha-guard-2`: INTENTIONAL ABSTENTION. guard() is deliberately not
        widened at all in A7.7 (no corroboration, unchanged admission
        policy from A7.4) -- A7.6 found this query mathematically
        indistinguishable, on every measured signal, from the payment
        false positive below.

    This test locks down the real, calibrated, and now-architecturally-
    explained outcome, not an aspirational one -- a future change that
    silently drops `ha-preflight-2`/`ha-guard-1` is a real regression;
    one that still doesn't recover `ha-preflight-1`/`ha-guard-2` is not.
    """
    with _offline():
        cx, lesson, skill = _build_human_acceptance_workspace(tmp_path)

        preflight_1 = cx.preflight(
            "Investigate why previous engineering knowledge sometimes disappears when I "
            "describe the same problem using different words"
        )
        assert lesson.memory_id not in {m.memory_id for m in preflight_1.verified_lessons}, (
            "ha-preflight-1 is a documented, calibrated abstention (see module docstring); "
            "if this now passes, the calibration should be revisited and this assertion updated"
        )

        preflight_2 = cx.preflight(
            "Check whether Cortex already learned anything useful about missing prior "
            "knowledge when the wording of a task changes"
        )
        assert lesson.memory_id in {m.memory_id for m in preflight_2.verified_lessons}

        guard_1 = cx.guard(
            "Before relying on an empty preflight result, check whether useful prior "
            "experience may have been missed because my description changed"
        )
        assert skill.skill_id in {s.skill_id for s in guard_1.applicable_skills}

        guard_2 = cx.guard(
            "I am about to trust a supposedly empty memory lookup after describing the "
            "issue differently from before"
        )
        assert skill.skill_id not in {s.skill_id for s in guard_2.applicable_skills}, (
            "ha-guard-2 is a documented, calibrated abstention (see module docstring)"
        )


@skip_without_model
def test_near_floor_path_stays_rejected_bn9_and_ha_preflight_1_are_not_separable(tmp_path):
    """A7.7 section 7's permanent evidence trail: a "near-floor + minimal
    lexical signal" admission rule for the memory pool was considered
    (it would recover `ha-preflight-1`) and explicitly REJECTED, because
    its only real-corpus separation from a genuine hard negative
    (`bn-9` from A7.3: "a test failed intermittently in the parser
    suite, was this seen previously") is a 0.0018 score gap -- nowhere
    near a robust basis for a product threshold. This test locks in that
    rejection: BOTH queries must behave the same way (neither is
    admitted), proving the current shipped policy does not quietly
    special-case one over the other.
    """
    with _offline():
        cx, lesson, _skill = _build_human_acceptance_workspace(tmp_path)

        ha_preflight_1 = cx.preflight(
            "Investigate why previous engineering knowledge sometimes disappears when I "
            "describe the same problem using different words"
        )
        bn_9 = cx.preflight("a test failed intermittently in the parser suite, was this seen previously")

        ha_preflight_1_recovered = lesson.memory_id in {m.memory_id for m in ha_preflight_1.verified_lessons}
        bn_9_leaked = lesson.memory_id in {m.memory_id for m in bn_9.verified_lessons}

        assert not bn_9_leaked, "bn-9 is a hard negative: it must never surface d1-lesson, regardless of score"
        assert not ha_preflight_1_recovered, (
            "ha-preflight-1 is intentionally NOT recovered (see A7.7 report section 7): its score is "
            "too close to bn-9's to safely separate with a near-floor rule, so this documents the "
            "accepted trade-off (losing this recall case) rather than a fragile threshold. If this "
            "assertion is ever changed to expect recovery, `bn_9_leaked` above must still be False."
        )


@skip_without_model
def test_payment_guard_clause_false_positive_stays_rejected(tmp_path):
    """The A7.3 false positive: an unrelated payment-form query must not
    surface the reword-related Skill as applicable, despite sharing the
    single word "guard". This is the SKILL pool's calibration floor
    (absolute=0.40, margin=0.38) doing its job."""
    with _offline():
        cx, _lesson, skill = _build_human_acceptance_workspace(tmp_path)

        result = cx.guard("add input validation to the guard clause in the payment form before we deploy")
        assert skill.skill_id not in {s.skill_id for s in result.applicable_skills}


@skip_without_model
def test_css_hard_negative_stays_empty(tmp_path):
    with _offline():
        cx, lesson, skill = _build_human_acceptance_workspace(tmp_path)

        result = cx.preflight("Change the CSS button color to blue")
        assert lesson.memory_id not in {m.memory_id for m in result.verified_lessons}


@skip_without_model
def test_safe_unsafe_contradiction_never_surfaces_the_wrong_lesson_as_authoritative(tmp_path):
    """The A7.3 contradiction finding, reproduced end-to-end: for this
    query, the real model ranks the UNVERIFIED contradictory memory
    ("...is safe...") above the correct, verified one ("...is unsafe...")
    in the FULL, unfiltered memory pool (semantic score 0.6269 for the
    correct lesson, but not rank #1 overall -- exactly A7.3's
    `contradiction-2` result).

    Two structural facts keep this safe:

    1. The contradictory memory is never `verified` (no test/confirmation
       evidence backs it here), so it can never appear in
       `Preflight.verified_lessons` regardless of how it ranks --
       `preflight()`'s existing epistemic filtering (predates A7.4)
       already refuses it authority.
    2. [A7.7] `_semantic_widen`'s candidate pool is now restricted to
       ELIGIBLE ids BEFORE ranking (the eligibility-before-ranking
       correctness fix confirmed and reproduced in A7.5/A7.6): the
       unverified contradictory memory (and any other ineligible
       candidate) is excluded from the ranking pool entirely, so it can
       no longer outrank the correct lesson and consume the pool's
       single admission slot the way it did before A7.7.

    RESULT: unlike A7.4 (where this exact case degraded to silence --
    neither lesson shown), the CORRECT verified lesson is now recovered,
    while the incorrect unverified one is never surfaced. This is the
    fix working as intended, not a coincidence: verified with the real
    model, not simulated.
    """
    with _offline():
        cx, _lesson, _skill = _build_human_acceptance_workspace(tmp_path)

        unsafe_evidence = cx.add_evidence(
            "observed session hijack after reusing old token", kind="user_confirmation"
        )
        unsafe_lesson = cx.learn(
            "After rotating a refresh token, reusing the old token is unsafe and can "
            "lead to session hijacking; always use the newly issued token.",
            supporting_evidence=[unsafe_evidence],
            verified=True,
        )
        contradiction_evidence = cx.add_evidence("grace-window claim, unverified", kind="user_statement")
        contradiction_lesson = cx.learn(
            "Reusing the old refresh token right after rotation is safe as long as it "
            "is used within a short grace window before it expires.",
            evidence=[contradiction_evidence],
        )
        cx.semantic_setup()

        result = cx.preflight("What is the current guidance on reusing an old refresh token right after rotation")
        surfaced_ids = {m.memory_id for m in result.verified_lessons}

        assert contradiction_lesson.memory_id not in surfaced_ids, (
            "the UNVERIFIED, contradictory memory must never be surfaced as an authoritative "
            "verified lesson -- this is the one guarantee that must always hold"
        )
        assert unsafe_lesson.memory_id in surfaced_ids, (
            "the correct, verified lesson should now be recovered thanks to the A7.7 "
            "eligibility-before-ranking fix -- if this regresses, the fix itself likely regressed"
        )


@skip_without_model
def test_holdout_new_semantic_positive_paraphrases_not_used_in_calibration(tmp_path):
    """Genuinely new paraphrases, written for this test and never looked
    at while picking `SEMANTIC_POLICY`'s floors -- a real holdout check,
    not a re-run of the calibration set.

    HONEST RESULT (see A7.4 report section 18): both paraphrases below
    score BELOW their pool's calibrated absolute floor (memory: 0.071 vs
    a 0.20 floor; skill: 0.366 vs a 0.40 floor) and are correctly
    abstained from -- neither is admitted. This is not the hoped-for
    outcome; it is the actual one, kept as the assertion instead of
    being tuned away. It demonstrates a real, honestly-reported limit on
    how far this policy's recall generalizes: it reliably recovers
    paraphrases with a similarity profile close to the calibration
    positives (`ha-preflight-2`, `ha-guard-1`), not arbitrary natural
    rewordings further from that profile. Per the A7.4 brief's own
    instruction ("una volta scelta la policy sul calibration set, NON
    ritoccarla query per query sul holdout; se il holdout fallisce,
    riporta il fallimento"), the floors were NOT retuned after seeing
    this result.
    """
    with _offline():
        cx, lesson, skill = _build_human_acceptance_workspace(tmp_path)

        preflight_holdout = cx.preflight(
            "sometimes cortex acts like it has no memory of something we already dealt "
            "with, just because I asked about it in new words"
        )
        assert lesson.memory_id not in {m.memory_id for m in preflight_holdout.verified_lessons}

        guard_holdout = cx.guard(
            "worried that a blank result from guard just means my phrasing was off, not "
            "that nothing relevant was ever recorded"
        )
        assert skill.skill_id not in {s.skill_id for s in guard_holdout.applicable_skills}


@skip_without_model
def test_preflight_recovers_via_corroboration_where_guard_still_abstains(tmp_path):
    """A7.7's clearest real-model demonstration of the explicit product
    decision (preflight = controlled recall, guard = conservative
    precision): the SAME text (the `holdout-guard` paraphrase from
    A7.4) is queried through BOTH APIs. `guard()`'s own skill-pool
    score for it (0.3662) sits below the skill floor (0.40) and gets no
    widening help -- unchanged from A7.4, abstains. `preflight()`'s
    memory-pool score for the same text is also below floor, but its
    rigorous structural corroboration path (A7.7, preflight-only) finds
    a REAL, canonically-linked failed Attempt (shares Evidence with
    `d1-lesson`) that is independently and strongly relevant to this
    exact text on its own terms, and rescues it. The two APIs
    legitimately disagree on the same input -- that is the intended
    design, not an inconsistency to fix.
    """
    with _offline():
        cx, lesson, skill = _build_human_acceptance_workspace(tmp_path)

        text = (
            "worried that a blank result from guard just means my phrasing was off, not "
            "that nothing relevant was ever recorded"
        )
        preflight_result = cx.preflight(text)
        guard_result = cx.guard(text)

        assert lesson.memory_id in {m.memory_id for m in preflight_result.verified_lessons}, (
            "preflight() should recover this via rigorous structural corroboration"
        )
        assert skill.skill_id not in {s.skill_id for s in guard_result.applicable_skills}, (
            "guard() must stay conservative on the identical text -- no corroboration widening for guard()"
        )


@skip_without_model
def test_offline_after_setup_no_network_required(tmp_path):
    """`cortex semantic setup` is the only entry point allowed to touch
    the network; every retrieval call after that must work with
    HF_HUB_OFFLINE=1 forced -- this is the real, end-to-end version of
    what `_semantic.load_model_for_retrieval` claims to guarantee."""
    cx, lesson, _skill = _build_human_acceptance_workspace(tmp_path)  # setup happens with network allowed

    with _offline():
        result = cx.preflight(
            "Check whether Cortex already learned anything useful about missing prior "
            "knowledge when the wording of a task changes"
        )
    assert lesson.memory_id in {m.memory_id for m in result.verified_lessons}


@skip_without_model
def test_model_revision_is_resolved_from_the_real_cache(tmp_path):
    cx, _lesson, _skill = _build_human_acceptance_workspace(tmp_path)
    from cortex_memory._semantic_store import SemanticIndexStore

    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        meta = store.meta()
    assert meta.model_revision is not None
    assert len(meta.model_revision) == 40  # a git commit hash, as huggingface_hub reports it
