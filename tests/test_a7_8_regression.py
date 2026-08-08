"""A7.8 regression: a real Human Acceptance miss, and its fix.

A7 Final Human Acceptance failed on a real, spontaneous query from
Leonardo against a real workspace (`/tmp/cortex_a7_final`):

    "I'm running a database migration and I want to make sure that if
    something fails, the system doesn't end up with only part of the
    update applied."

`preflight()` returned nothing, even though a directly relevant
`root_cause` and a `verified` `lesson` -- drawn from the SAME incident,
sharing Evidence -- were on record (a near-wording control query
recovered both immediately, proving this was a retrieval miss, not a
data-loss bug).

Diagnosis (see the A7.8 report): the memory pool's semantic admission
(`_semantic.semantic_admitted_ids`) is single-winner-plus-margin,
treating every candidate as if it competes with every other candidate.
A root cause and the verified lesson drawn from it are not competitors
-- they are the SAME experience described from two angles, and will
structurally score very close to each other against any query genuinely
about that incident, collapsing the margin between them. The existing
attempt-mediated shared-provenance rescue in `build_preflight`
(`_preflight.py`) could not help either: it only fires once the
ATTEMPT itself is independently judged relevant, and the attempt
pool's absolute floor (0.50) was calibrated defensively, never against
a real positive case reachable by ordinary, conversationally-phrased
task wording.

The fix (`Cortex._preflight_memory_semantic_widen`, in `_workspace.py`)
extends the SAME shared-Evidence trust `build_preflight` already grants
attempt-to-memory rescue to memory-to-memory: candidates are clustered
by shared `evidence_ids`, a cluster competes on margin as a single unit
against the best-scoring candidate OUTSIDE it (never against its own
sibling), and an admitted cluster is admitted in full.

These tests lock in that fix. `test_semantic_real_model.py` and
`test_semantic.py` already cover (unchanged by this fix -- see the
A7.8 report's corpus comparison table): the 4 Human Acceptance
queries, the 2 A7.4 holdout paraphrases, the payment-guard-clause false
positive, the CSS hard negative, the safe/unsafe refresh-token
contradiction, and the bn-9 near-floor negative. This file does not
repeat them.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cortex_memory import Cortex

# ---------------------------------------------------------------------------
# Fast, deterministic tests: the clustering mechanism itself, isolated with
# a fake embedding backend (same technique as test_semantic.py).
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


def _verified_lesson(cx, content):
    ev = cx.add_evidence("checked", kind="test_result")
    return cx.learn(content, evidence=[ev], verified=True)


_DILUTED_QUERY = (
    "totally unrelated wording that still somehow concerns alpha topics here and nothing else"
)


def test_preflight_admits_sibling_root_cause_and_lesson_tied_on_score(tmp_path, fake_semantic):
    """The exact tie the plain single-winner-plus-margin policy would
    reject (margin=0 < the 0.08 floor) must now be admitted TOGETHER
    when the two candidates share Evidence: they are the same
    experience (cause + prescription), not two candidates competing for
    the pool's one admission slot."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    root_cause = cx.remember("alpha", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("alpha", evidence=[ev, validation], verified=True)
    cx.semantic_setup()

    result = cx.preflight(_DILUTED_QUERY)

    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in result.verified_lessons} == {lesson.memory_id}


def test_preflight_does_not_cluster_memories_that_do_not_share_evidence(tmp_path, fake_semantic):
    """Regression safety: an exact tie between two memories that do NOT
    share Evidence must stay rejected exactly as before -- clustering
    must never merge candidates just because they happen to tie."""
    cx = Cortex.init(tmp_path, "dev")
    root_cause = cx.remember("alpha", kind="root_cause", epistemic_state="inferred")
    _verified_lesson(cx, "alpha")  # its own, unrelated evidence
    cx.semantic_setup()

    result = cx.preflight(_DILUTED_QUERY)

    assert result.root_causes == ()
    assert result.verified_lessons == ()


def test_preflight_admits_low_scoring_sibling_via_cluster_membership(tmp_path, fake_semantic):
    """A cluster member whose OWN score would not individually clear the
    absolute floor is still admitted once its sibling (sharing
    Evidence) clears floor+margin as the cluster's representative --
    the same trust `build_preflight`'s existing attempt-to-memory
    shared-provenance rule already grants with no independent score
    check on the rescued side, applied symmetrically to memory-to-
    memory."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    # "alpha beta" scores ~0.707 against a pure "alpha" query -- clears the 0.20 floor comfortably
    root_cause = cx.remember("alpha beta", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    # "gamma" alone scores 0.0 against a pure "alpha" query -- well below the memory floor on its own
    validation = cx.add_evidence("checked", kind="test_result")
    lesson = cx.learn("gamma", evidence=[ev, validation], verified=True)
    cx.semantic_setup()

    result = cx.preflight(_DILUTED_QUERY)

    assert {m.memory_id for m in result.root_causes} == {root_cause.memory_id}
    assert {m.memory_id for m in result.verified_lessons} == {lesson.memory_id}


def test_preflight_cluster_still_loses_to_a_stronger_unrelated_competitor(tmp_path, fake_semantic):
    """A sibling cluster is not an unconditional pass: if a genuinely
    unrelated candidate (different Evidence, different concept) beats
    the cluster's representative score by more than the margin floor,
    the cluster still loses, exactly as plain single-winner admission
    already would for two unrelated candidates."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("observed failure", kind="error_observation")
    validation = cx.add_evidence("checked", kind="test_result")
    cx.remember("alpha beta", kind="root_cause", epistemic_state="inferred", evidence=[ev])
    cx.learn("alpha beta", evidence=[ev, validation], verified=True)
    # unrelated, exact match on the query's own concept -- strictly stronger, no shared evidence
    strong_unrelated = _verified_lesson(cx, "alpha")
    cx.semantic_setup()

    result = cx.preflight("totally unrelated wording that still somehow concerns alpha and nothing else")

    surfaced = {m.memory_id for m in result.verified_lessons}
    assert surfaced == {strong_unrelated.memory_id}


# ---------------------------------------------------------------------------
# Real-model integration test: the actual Human Acceptance failure, fixed.
# Skipped unless the real model is already cached locally -- never
# downloaded by the test suite itself (same policy as
# test_semantic_real_model.py).
# ---------------------------------------------------------------------------

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


def _build_database_migration_experience(cx):
    """Verbatim content from the real `/tmp/cortex_a7_final` Human
    Acceptance workspace (verified via Cortex's own public API during
    A7.8 diagnosis), not reworded or simplified for this test."""
    failure_evidence = cx.add_evidence(
        "A database schema upgrade failed halfway through and left part of the new "
        "structure applied while the remaining changes were missing.",
        kind="error_observation",
    )
    cx.record_attempt(
        task="Upgrade the persisted database schema safely",
        approach="Apply the schema changes sequentially without one enclosing transaction",
        outcome="failed",
        evidence=[failure_evidence],
    )
    root_cause = cx.remember(
        "The database upgrade was not atomic, so a failure could leave partially "
        "applied persistent state.",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[failure_evidence],
    )
    validation = cx.add_evidence(
        "A forced failure during the transactional upgrade rolled back every schema "
        "change and preserved the previous database state.",
        kind="test_result",
    )
    lesson = cx.learn(
        "Persistent schema upgrades should execute atomically so that a failure "
        "cannot leave only part of the migration applied.",
        evidence=[failure_evidence, validation],
        verified=True,
    )
    cx.promote(
        lesson,
        name="Perform safe persistent schema upgrades",
        purpose="Evolve persisted storage without leaving a partially upgraded database after failure.",
        steps=[
            "Run the upgrade inside an explicit transaction.",
            "Force a failure path and verify complete rollback.",
            "Verify the previous schema remains usable after rollback.",
        ],
    )
    return root_cause, lesson


@skip_without_model
def test_human_spontaneous_database_migration_query_now_recovers_useful_experience(tmp_path):
    """THE regression case: the exact, frozen, spontaneous query
    Leonardo typed during A7 Final Human Acceptance. Before the A7.8
    fix this returned `Preflight.is_empty() == True`. It must now
    surface the root cause AND the verified lesson together -- not just
    one of the two -- since they are the same experience and the fix
    exists precisely so neither is left out arbitrarily."""
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        root_cause, lesson = _build_database_migration_experience(cx)
        cx.semantic_setup()

        result = cx.preflight(
            "I'm running a database migration and I want to make sure that if "
            "something fails, the system doesn't end up with only part of the "
            "update applied."
        )

        assert not result.is_empty(), "this is the real Human Acceptance miss -- must not regress to empty"
        assert root_cause.memory_id in {m.memory_id for m in result.root_causes}
        assert lesson.memory_id in {m.memory_id for m in result.verified_lessons}


@skip_without_model
def test_human_spontaneous_query_control_still_recovers_known_failure_too(tmp_path):
    """The near-wording control query from A7.8 diagnosis: it was
    already a full HIT (known failure + root cause + verified lesson +
    recommended validation) before this fix, via the attempt's own
    semantic admission clearing its much higher floor. Must remain a
    full HIT after the fix -- the fix only ever widens the memory pool,
    it does not change attempt-pool admission."""
    with _offline():
        cx = Cortex.init(tmp_path, "dev")
        root_cause, lesson = _build_database_migration_experience(cx)
        cx.semantic_setup()

        result = cx.preflight(
            "Prevent a database schema upgrade from leaving partially applied "
            "state when the upgrade fails."
        )

        assert not result.is_empty()
        assert len(result.known_failures) == 1
        assert root_cause.memory_id in {m.memory_id for m in result.root_causes}
        assert lesson.memory_id in {m.memory_id for m in result.verified_lessons}
