"""Real-model integration test for the A16.3 semantic backend:
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` executed
through ONNX Runtime, against REAL Urdyn workspaces built only through
the public API.

Skipped entirely -- with an explicit reason, never silently -- unless the
pinned artifacts are ALREADY in the local Hugging Face cache (checked
with `local_files_only=True`, so asking the question never downloads the
answer) and the `[semantic]` extra is importable. This file is never part
of a normal `uv run pytest` run's *requirements*; it participates only
when the environment already has what it needs. Every test here runs with
`HF_HUB_OFFLINE=1` forced, which is what makes "retrieval works offline"
an actually-tested property rather than a claim.

WHAT CHANGED IN A16.3, AND WHAT DID NOT. The backend swap is motivated by
cross-language recall, and that is what the bulk of this file now covers:
A16 measured the previous model2vec/potion backend at 0-30% recall for
cross-language queries against 83-84% for this one. The A7.x behaviours
that were properties of the SYSTEM rather than of that particular model
are kept and still asserted here: an unverified memory never gains
authority, a hard negative never surfaces, the payment-guard-clause false
positive stays rejected. The A7.4 assertions that encoded
potion-SPECIFIC score outcomes (which exact acceptance-testing paraphrase
landed just under which calibrated floor) are NOT carried over pretending
to be model-independent -- they measured one model's geometry, and this
is a different model.
"""

from __future__ import annotations

import os

import pytest

from urdyn import Urdyn

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


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


pytestmark = pytest.mark.real_model
_SKIP_REASON = (
    "the real ONNX semantic model is not cached locally (and/or the 'semantic' "
    "extra is not installed) -- run 'urdyn semantic setup' in a scratch "
    "workspace once to populate the Hugging Face cache, then re-run this file; "
    "never downloaded automatically by the test suite itself"
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


# ---------------------------------------------------------------------------
# Cross-language workspaces
# ---------------------------------------------------------------------------

# A recorded experience in Urdyn is normally a root cause AND the verified
# lesson drawn from it, sharing the Evidence that ties them together -- that
# is the shape `preflight()` is built around, so it is the shape asserted
# here. (Written with only the prescriptive lesson stored, several of these
# queries land under the shipped margin floor and abstain: the diagnostic
# wording of a question is much closer to a root cause than to a rule. That
# is the calibrated policy behaving correctly on a workspace that is missing
# half the experience, not a backend defect -- measured while writing this
# file, and worth knowing before anyone "fixes" it.)
_EN_ROOT_CAUSE = (
    "The payment client retried a timed-out request without an idempotency key, so the "
    "server processed the same charge twice when the first response was merely slow, not failed."
)
_EN_LESSON = (
    "Every retried write must carry a stable idempotency key so the server can "
    "recognize and collapse a duplicate caused by a client-side timeout."
)
_IT_ROOT_CAUSE = (
    "Il client di pagamento ripeteva una richiesta andata in timeout senza una idempotency key, "
    "così il server elaborava lo stesso addebito due volte quando la prima risposta era solo lenta, non fallita."
)
_IT_LESSON = (
    "Ogni scrittura ripetuta deve portare una idempotency key stabile, così il "
    "server può riconoscere e collassare un duplicato causato da un timeout lato client."
)
# Deliberate noise so no pool is a single-candidate pool (which skips the
# margin floor) and so "found the right one" means something.
_NOISE = (
    "Config files containing secrets must be written with restrictive permissions, "
    "never left at the umask default.",
    "A derived index must be refreshed whenever its source data changes, not on a "
    "fixed schedule that ignores how often the source actually changes.",
    "Change the CSS button color to blue.",
)


def _workspace_with(tmp_path, root_cause_text, lesson_text):
    cx = Urdyn.init(tmp_path, "dev")
    evidence = cx.add_evidence(
        "Reproduced: the same charge was processed twice after a client-side timeout.",
        kind="user_confirmation",
    )
    root_cause = cx.remember(root_cause_text, kind="root_cause", evidence=[evidence])
    lesson = cx.learn(lesson_text, evidence=[evidence], supporting_evidence=[evidence], verified=True)
    for noise in _NOISE:
        note_evidence = cx.add_evidence(f"note: {noise}", kind="user_statement")
        cx.remember(noise, kind="root_cause", evidence=[note_evidence])
    cx.semantic_setup()
    return cx, root_cause, lesson


def _surfaces(result, *memories):
    """Whether the recorded EXPERIENCE was surfaced at all -- through the
    root cause, the lesson, or both. Asserting the experience rather than
    one specific channel is what keeps this a retrieval regression test
    instead of a re-assertion of A7.8's clustering rules."""
    surfaced = {m.memory_id for m in result.root_causes} | {m.memory_id for m in result.verified_lessons}
    return any(memory.memory_id in surfaced for memory in memories)


_MEMORY_LANGUAGES = {"EN": (_EN_ROOT_CAUSE, _EN_LESSON), "IT": (_IT_ROOT_CAUSE, _IT_LESSON)}


@skip_without_model
@pytest.mark.parametrize(
    "memory_language, query",
    [
        # same-language baselines: whatever else changes, these must hold
        ("EN", "Why did the customer get charged twice after we retried a slow payment request?"),
        ("IT", "Perché il cliente è stato addebitato due volte dopo aver ripetuto una richiesta di pagamento lenta?"),
        # the actual reason A16 exists: the query is in a different
        # language than the memory it must find
        ("EN", "Perché il cliente è stato addebitato due volte dopo aver ripetuto una richiesta di pagamento lenta?"),
        ("IT", "Why did the customer get charged twice after we retried a slow payment request?"),
        ("EN", "Pourquoi le client a-t-il été débité deux fois alors qu'on a juste relancé une requête de paiement lente ?"),
        ("EN", "¿Por qué al cliente se le cobró dos veces después de que solo reintentamos una solicitud de pago lenta?"),
    ],
    ids=["en_to_en", "it_to_it", "en_memory_it_query", "it_memory_en_query", "en_memory_fr_query", "en_memory_es_query"],
)
def test_cross_language_recall(tmp_path, memory_language, query):
    """The A16 requirement itself, as regression coverage rather than as a
    benchmark: an experience recorded in one language is found by a
    question asked in another. The previous backend failed most of these
    outright (A16 measured 0% for both cross-language directions on its
    frozen corpus); asserting recovery here is what stops a future change
    from quietly giving that up again."""
    root_cause_text, lesson_text = _MEMORY_LANGUAGES[memory_language]
    with _offline():
        cx, root_cause, lesson = _workspace_with(tmp_path, root_cause_text, lesson_text)
        assert _surfaces(cx.preflight(query), root_cause, lesson), (
            f"{memory_language} memory should be reachable from this query"
        )


@skip_without_model
def test_natural_paraphrase_same_language(tmp_path):
    """Different wording, same language, no shared distinctive vocabulary
    with the stored text -- the original A7 motivation, still required."""
    with _offline():
        cx, root_cause, lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)
        result = cx.preflight(
            "A payment went through more than once even though we only meant to resend "
            "it after it seemed stuck -- what should we have done?"
        )
        assert _surfaces(result, root_cause, lesson)


@skip_without_model
def test_unrelated_query_abstains(tmp_path):
    """Abstention is the correct answer far more often than a guess is.
    An unrelated styling question must not surface an engineering
    experience, in any language."""
    with _offline():
        cx, root_cause, lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)
        assert not _surfaces(cx.preflight("How do I update the stylesheet to change a button's color?"), root_cause, lesson)
        assert not _surfaces(
            cx.preflight("Come aggiorno il foglio di stile per cambiare il colore di un pulsante?"), root_cause, lesson
        )


@skip_without_model
def test_adjacent_request_handling_query_is_admitted_documented_limit(tmp_path):
    """A MEASURED LIMIT, asserted as it actually behaves rather than as
    one would like it to (the A7.4 precedent: record the real outcome, do
    not tune it away).

    A question about a rate limiter dropping legitimate requests under
    load surfaces the retry/duplicate-charge root cause: the real score
    is 0.3695 against a 0.20 floor, with a 0.2345 margin against a 0.08
    floor, so it is admitted comfortably rather than marginally. The two
    texts are not as unrelated as the labels suggest -- both are about
    requests failing and being retried under load -- and this is the
    backend's honest judgement, not a threshold accident.

    It is asserted here so the behaviour is VISIBLE and any future change
    to it is deliberate: if a later backend or policy stops admitting
    this, that is arguably an improvement, and this test should be
    updated on purpose rather than silently drifting."""
    with _offline():
        cx, root_cause, lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)
        result = cx.preflight(
            "why is our token bucket rate limiter dropping legitimate requests during peak traffic"
        )
        assert _surfaces(result, root_cause, lesson)


@skip_without_model
def test_unverified_contradiction_never_gains_authority(tmp_path):
    """A system property, not a model property: an unverified memory can
    never appear as a verified lesson, however well it ranks. Carried
    over from A7.4 unchanged because it must hold for ANY backend."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        verified_evidence = cx.add_evidence("observed session hijack after reusing old token", kind="user_confirmation")
        correct = cx.learn(
            "After rotating a refresh token, reusing the old token is unsafe and can lead to "
            "session hijacking; always use the newly issued token.",
            supporting_evidence=[verified_evidence],
            verified=True,
        )
        loose_evidence = cx.add_evidence("grace-window claim, unverified", kind="user_statement")
        contradiction = cx.learn(
            "Reusing the old refresh token right after rotation is safe as long as it is used "
            "within a short grace window before it expires.",
            evidence=[loose_evidence],
        )
        cx.semantic_setup()

        result = cx.preflight("What is the current guidance on reusing an old refresh token right after rotation")
        surfaced = {m.memory_id for m in result.verified_lessons}
        assert contradiction.memory_id not in surfaced, (
            "an UNVERIFIED memory must never be surfaced as an authoritative verified lesson -- "
            "this is the one guarantee that must hold no matter which model is in use"
        )
        assert correct.memory_id in surfaced


@skip_without_model
def test_superseded_memory_is_not_resurrected(tmp_path):
    """Also a system property: supersession outranks similarity."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        old_evidence = cx.add_evidence("initial diagnosis", kind="user_confirmation")
        old = cx.learn(
            "The intermittent timeout is caused by slow DNS resolution in the client.",
            supporting_evidence=[old_evidence],
            verified=True,
        )
        new_evidence = cx.add_evidence("corrected after further investigation", kind="test_result")
        cx.remember(
            "The intermittent timeout is caused by connection pool exhaustion, not DNS.",
            kind="lesson",
            epistemic_state="verified",
            supporting_evidence=[new_evidence],
            supersedes=old.memory_id,
        )
        cx.semantic_setup()

        result = cx.preflight("perché le richieste vanno in timeout a intermittenza")
        assert old.memory_id not in {m.memory_id for m in result.verified_lessons}


@skip_without_model
def test_payment_guard_clause_false_positive_stays_rejected(tmp_path):
    """[A7.3 / A16.3.1] The historical false positive -- an unrelated
    payment-form query surfacing the reword-related Skill because both
    happen to say "guard" -- must not be admitted.

    A16.3 broke this: the query scored 0.3393 under the old backend
    (below the then-0.40 SKILL floor) and 0.4555 under this one (above
    it), and the pool holds a single candidate, so A7.4's margin floor is
    deliberately not applied and the absolute floor decided alone.

    A16.3.1 fixed it by RE-CALIBRATING that floor to this backend's score
    geometry rather than by special-casing this query: the floor was
    chosen from a frozen corpus that deliberately EXCLUDED this scenario,
    and this query was then used only as an independent acceptance check.
    The companion test below is the other half of the claim -- the same
    Skill is still found when the query genuinely calls for it, so this
    is abstention, not the Skill having become unreachable."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
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
            steps=["Rephrase the task in different words", "Run preflight/guard again"],
        )
        cx.record_attempt(task="Change CSS button color to blue", approach="Updated the stylesheet", outcome="succeeded")
        cx.semantic_setup()

        result = cx.guard("add input validation to the guard clause in the payment form before we deploy")
        assert skill.skill_id not in {s.skill_id for s in result.applicable_skills}

        # the other half: abstention above, retrieval here, same Skill and
        # same workspace -- a floor high enough to reject everything would
        # pass the assertion above and be worthless.
        genuine = cx.guard("guard returned nothing for this task, is that trustworthy?")
        assert skill.skill_id in {s.skill_id for s in genuine.applicable_skills}


# ---------------------------------------------------------------------------
# Backend identity, offline behaviour, idempotency
# ---------------------------------------------------------------------------


@skip_without_model
def test_setup_records_the_full_effective_identity(tmp_path):
    """[A16.3] The index must name the repository, the pinned revision and
    the exact artifact it was built with -- that triple is what stops it
    from being read back by a different artifact or a different upstream
    revision."""
    from urdyn import _semantic
    from urdyn._semantic_store import SemanticIndexStore

    with _offline():
        cx, _root_cause, _lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)

        with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
            meta = store.meta()

    assert meta.provider == _semantic.SEMANTIC_PROVIDER
    assert meta.model_revision == _semantic.SEMANTIC_MODEL_REVISION
    assert len(meta.model_revision) == 40  # a real git commit hash
    assert _semantic.SEMANTIC_MODEL_REPO in meta.model_id
    assert _semantic.SEMANTIC_MODEL_REVISION in meta.model_id
    artifact = _semantic.artifact_for_index(meta)
    assert artifact in _semantic.SUPPORTED_ARTIFACTS
    assert artifact in meta.model_id
    assert meta.dimensions == 384  # verified against the real model output, not a constant
    assert meta.status == "ready"


@skip_without_model
def test_offline_after_setup_no_network_required(tmp_path):
    """`urdyn semantic setup` is the only entry point allowed to touch
    the network; every retrieval call after that must work with
    HF_HUB_OFFLINE=1 forced."""
    cx, root_cause, lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)  # setup happens with network allowed

    with _offline():
        result = cx.preflight("Perché il cliente è stato addebitato due volte dopo un timeout?")
    assert _surfaces(result, root_cause, lesson)


@skip_without_model
def test_repeated_setup_is_idempotent_and_still_correct(tmp_path):
    with _offline():
        cx, root_cause, lesson = _workspace_with(tmp_path, _EN_ROOT_CAUSE, _EN_LESSON)
        cx.semantic_setup()
        cx.semantic_setup()

        result = cx.preflight("Perché il cliente è stato addebitato due volte dopo un timeout?")
        assert _surfaces(result, root_cause, lesson)

        from urdyn._semantic_store import SemanticIndexStore

        with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
            assert store.is_ready()
            # one row per indexed entity, not accumulating duplicates
            assert store.vector_count() == 2 + len(_NOISE)  # root cause + lesson + noise


@skip_without_model
def test_oversize_and_malformed_text_stay_data(tmp_path):
    """Hostile input is data, not a hazard: very long content and
    malformed Unicode are bounded by the tokenizer's 128-token truncation
    and must not break setup, retrieval, or canonical storage."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        evidence = cx.add_evidence("bulk import", kind="user_statement")
        huge = cx.remember("payment timeout retry duplicate charge " * 5000, kind="root_cause", evidence=[evidence])
        cx.remember("weird \u200b zero-width, \u202e reversed, \U0001f600 emoji, ünïcödé", kind="note", evidence=[evidence])
        cx.semantic_setup()

        result = cx.preflight("duplicate charge after a retried payment")
        assert isinstance(result.root_causes, tuple)
        assert huge.memory_id in {m.memory_id for m in cx.recall("payment timeout")}


# ---------------------------------------------------------------------------
# [A16.3.1] SKILL and ATTEMPT pool calibration against this backend.
#
# These assert PROPERTIES -- admitted / not admitted -- never the cosine
# that produced them. A test pinned to an exact score would have to be
# rewritten by every backend change and would say nothing about whether
# retrieval is still correct, which is the mistake A16.3 had to unpick in
# the A7.4 assertions it inherited.
# ---------------------------------------------------------------------------


def _skill_workspace(cx, name, purpose, conditions=()):
    """Promote one Skill through the public API, with the Evidence chain
    `promote` requires. Returns the Skill."""
    evidence = cx.add_evidence(f"observed: {purpose}", kind="user_statement")
    confirmation = cx.add_evidence(f"reproduced: {purpose}", kind="user_confirmation")
    lesson = cx.learn(purpose, evidence=[evidence], supporting_evidence=[confirmation], verified=True)
    return cx.promote(lesson, name=name, purpose=purpose, steps=["apply it"], conditions=list(conditions))


@skip_without_model
def test_a_clearly_applicable_skill_is_still_admitted(tmp_path):
    """The raised SKILL floor must not have turned the pool off: a query
    that restates what the Skill is for still reaches it."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        skill = _skill_workspace(
            cx,
            "Use an idempotency key for retryable writes",
            "A retried API write must not create duplicate side effects.",
            ["The endpoint performs a non-idempotent write", "The client may retry after a timeout"],
        )
        cx.semantic_setup()

        result = cx.guard("use an idempotency key so a retried write does not duplicate")
        assert skill.skill_id in {s.skill_id for s in result.applicable_skills}


@skip_without_model
def test_a_cross_language_applicable_skill_is_admitted(tmp_path):
    """An Italian Skill answering an English query -- the whole point of
    the A16.3 backend -- survives the A16.3.1 recalibration."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        skill = _skill_workspace(
            cx,
            "Verificare la connessione al database prima del rilascio",
            "Un deploy con credenziali sbagliate fallisce soltanto una volta in produzione.",
            ["Prima di un rilascio in produzione"],
        )
        cx.semantic_setup()

        result = cx.guard("check that the database credentials actually work before releasing")
        assert skill.skill_id in {s.skill_id for s in result.applicable_skills}


@skip_without_model
def test_a_lone_skill_whose_conditions_do_not_hold_is_not_admitted(tmp_path):
    """The single-candidate case the A16.3.1 corpus was built around: a
    Skill whose PURPOSE is close to the query but whose CONDITIONS name a
    different environment. There is no runner-up, so A7.4's margin floor
    is skipped by design and the absolute floor is the only thing between
    this Skill and an agent being told it applies."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        skill = _skill_workspace(
            cx,
            "Serialize workers with a Postgres advisory lock",
            "Two worker processes must not process the same row at the same time.",
            ["The deployment uses PostgreSQL", "Several worker processes share one database"],
        )
        cx.semantic_setup()

        result = cx.guard("we run a single-process SQLite desktop app, how do we stop two writes colliding?")
        assert skill.skill_id not in {s.skill_id for s in result.applicable_skills}


@skip_without_model
def test_a_relevant_failed_attempt_is_admitted(tmp_path):
    """[A16.3.1] The ATTEMPT pool was validated and left unchanged; this
    pins the recall half of that claim, cross-language.

    Asked through `preflight()`, not `guard()`: guard's known_failures
    additionally require the attempt to share Evidence with an applicable
    Skill (see `_guard.py`), so a guard-based assertion here would be
    answering a question about that rule rather than about the ATTEMPT
    pool's calibration."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        attempt = cx.record_attempt(
            task="Debug intermittent 502 responses from the nginx proxy",
            approach="Increased proxy_read_timeout, which did not stop the 502s",
            outcome="failed",
        )
        cx.semantic_setup()

        result = cx.preflight("il proxy nginx restituisce 502 a intermittenza")
        assert attempt.attempt_id in {a.attempt_id for a in result.known_failures}


@skip_without_model
def test_a_near_miss_attempt_is_not_admitted(tmp_path):
    """Same tool, different failure mode: prior history about a build
    killed for memory must not be offered for a build that cannot reach
    the registry. Nothing about the two is operationally transferable."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        attempt = cx.record_attempt(
            task="Fix the failing Docker build on CI",
            approach="Raised the container memory limit because the build was OOM killed",
            outcome="failed",
        )
        cx.semantic_setup()

        result = cx.preflight("the docker build cannot pull the base image from the registry")
        assert attempt.attempt_id not in {a.attempt_id for a in result.known_failures}


# ---------------------------------------------------------------------------
# [A16.3.2] What the two final policy values actually buy.
#
# Each of these pins a behaviour that a plausible-looking change to
# SEMANTIC_POLICY would silently take away: the SKILL floor's position
# between 0.50 and 0.55, and the ATTEMPT margin being low enough to admit
# a clear winner but not so low that an ambiguous pool resolves itself.
# ---------------------------------------------------------------------------


@skip_without_model
def test_a_skill_is_rejected_when_its_conditions_name_a_different_runtime(tmp_path):
    """The case that decides SKILL's floor between 0.50 and 0.55.

    The Skill is about threads sharing one process's memory; the question
    is about two separate processes sharing Redis. The remedy is genuinely
    different, but the wording is close enough that this scored 0.5252 in
    the A16.3.2 corpus -- above 0.50, below 0.55. It is the reason the
    floor is where it is, so if someone lowers it to 0.50 this test is
    what says no."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        skill = _skill_workspace(
            cx,
            "Guard the shared counter with an atomic compare-and-swap",
            "Two threads incrementing the same counter lose updates when the read and the write interleave.",
            ["Multiple OS threads share one process memory space"],
        )
        cx.semantic_setup()

        result = cx.guard(
            "two separate server processes increment a counter stored in Redis and the total comes out wrong"
        )
        assert skill.skill_id not in {s.skill_id for s in result.applicable_skills}


@skip_without_model
def test_a_multi_candidate_attempt_pool_admits_the_right_history(tmp_path):
    """[A16.3.2] The closure of the question A16.3.1 had to leave open:
    with two plausible prior attempts in the pool, the one that actually
    bears on the task is admitted. Under the old 0.35 margin every
    multi-candidate positive was rejected, so this could not have passed."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        latency = cx.record_attempt(
            task="Reduce the TLS handshake latency for mobile clients",
            approach="Enabled session resumption, which only helped repeat connections",
            outcome="failed",
        )
        expiry = cx.record_attempt(
            task="Fix the expired TLS certificate on the staging gateway",
            approach="Renewed it by hand and then forgot to update the automation",
            outcome="failed",
        )
        cx.semantic_setup()

        surfaced = {a.attempt_id for a in cx.preflight("staging is down again because the certificate expired").known_failures}
        assert expiry.attempt_id in surfaced
        assert latency.attempt_id not in surfaced


@skip_without_model
def test_an_ambiguous_attempt_pool_still_abstains(tmp_path):
    """The other half of lowering the margin to 0.08: it must not become a
    formality. Both attempts here are about notification email, in Italian,
    against an English query -- the top two land 0.0250 apart, and the
    top-ranked one is the WRONG one. A pool that cannot tell its own
    candidates apart must abstain rather than pick, which is exactly what
    the margin floor is for."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        duplicate = cx.record_attempt(
            task="Correggere l'invio di email duplicate agli utenti",
            approach="Il job veniva rieseguito senza marcare le email già inviate",
            outcome="failed",
        )
        spam = cx.record_attempt(
            task="Correggere le email di notifica che finivano nello spam",
            approach="Configurati SPF e DKIM sul dominio di invio",
            outcome="failed",
        )
        cx.semantic_setup()

        surfaced = {
            a.attempt_id
            for a in cx.preflight("users complained they received the same notification email twice").known_failures
        }
        assert duplicate.attempt_id not in surfaced
        assert spam.attempt_id not in surfaced


# ---------------------------------------------------------------------------
# [A16.3.3] The SKILL margin gate, on the real model.
# ---------------------------------------------------------------------------


@skip_without_model
def test_a_multi_candidate_skill_pool_admits_the_right_skill(tmp_path):
    """[A16.3.3] Two Skills that are both about indexing, one of which
    answers the question. Under the inherited 0.38 margin this abstained
    -- it was the shape of case A16.3.2 measured being rejected 8 times
    out of 9 -- and it is the reason the margin was recalibrated."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        covering = _skill_workspace(
            cx,
            "Add a covering index so the query never touches the table",
            "An index that lacks the selected columns still costs one heap lookup per matched row.",
            ["The query selects few columns from many rows"],
        )
        selectivity = _skill_workspace(
            cx,
            "Order composite index columns by selectivity",
            "A composite index whose leading column barely narrows anything is close to useless.",
            ["The query filters on several columns at once"],
        )
        cx.semantic_setup()

        surfaced = {
            s.skill_id
            for s in cx.guard("the query uses the index but still reads the table for every row").applicable_skills
        }
        assert covering.skill_id in surfaced
        assert selectivity.skill_id not in surfaced


@skip_without_model
def test_a_skill_pool_that_cannot_tell_its_candidates_apart_abstains(tmp_path):
    """The other half of lowering the SKILL margin: it must still stop the
    pool from guessing. Both Skills here are about token lifetime, the
    question is about clock drift, and the WRONG one ranks first by 0.0519
    -- under the margin floor, so nothing is admitted. Admitting rank #1
    regardless is exactly the A7.3 mistake."""
    with _offline():
        cx = Urdyn.init(tmp_path, "dev")
        skew = _skill_workspace(
            cx,
            "Allow a small clock skew when validating token expiry",
            "A few seconds of drift between servers rejects tokens that are still valid.",
            ["Tokens are validated on a different host than the one that issued them"],
        )
        revoke = _skill_workspace(
            cx,
            "Keep a revocation list for tokens that must stop working early",
            "A stateless token stays valid until it expires, even after the user logs out.",
            ["A user can log out or be disabled before the token expires"],
        )
        cx.semantic_setup()

        surfaced = {
            s.skill_id
            for s in cx.guard(
                "users get token expired errors immediately after logging in on one of our nodes"
            ).applicable_skills
        }
        assert skew.skill_id not in surfaced
        assert revoke.skill_id not in surfaced
