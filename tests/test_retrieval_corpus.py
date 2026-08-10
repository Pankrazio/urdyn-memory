"""General evaluation corpus for the A7 retrieval fix, independent of
the A6 case itself (see `test_a6_regression.py` for that mandatory
regression case). A different domain (database connection pool
exhaustion) is used here deliberately, so the fix is not only proven
on the one scenario it was diagnosed from.

Every query below was verified empirically (see A7's implementation
notes) before being locked into this file: positives that are asserted
to be recovered genuinely are; the one asserted as a documented miss
genuinely misses even with the new fused retrieval, not just the old
one. Nothing here was adjusted to make FTS5 "look better" -- the
corpus records what the fix does and does not do.
"""

from cortex_memory import Cortex


def _build_connection_pool_experience(cx):
    error_evidence = cx.add_evidence(
        "Connection pool exhausted during a load test, requests started timing out.",
        kind="error_observation",
    )
    failed_attempt = cx.record_attempt(
        task="Fix database connection pool exhaustion under load",
        approach="Increased the pool size without addressing connection leaks",
        outcome="failed",
        evidence=[error_evidence],
    )
    validation = cx.add_evidence(
        "Pool size increase alone did not fix the timeouts under sustained load.",
        kind="test_result",
    )
    lesson = cx.learn(
        "Close database connections explicitly after each request instead of "
        "relying on garbage collection to reclaim the pool.",
        evidence=[error_evidence],
        supporting_evidence=[validation],
        verified=True,
    )
    skill = cx.promote(
        lesson,
        name="Fix connection pool exhaustion safely",
        purpose="Stop pool exhaustion by closing connections explicitly instead of "
        "just growing the pool.",
        steps=[
            "Audit request handlers for unclosed connections.",
            "Add explicit connection.close() after each request.",
        ],
    )
    return failed_attempt, validation, lesson, skill


def test_corpus_near_original_and_short_paraphrases_are_recovered(tmp_path):
    """Near-verbatim and short positives: the easiest cases, already
    working before A7 -- must keep working."""
    cx = Cortex.init(tmp_path, "dev")
    _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    for task in (
        "Fix database connection pool exhaustion under load",
        "fix connection pool exhaustion",
    ):
        preflight_result = agent_b.preflight(task)
        guard_result = agent_b.guard(task)
        assert not preflight_result.is_empty(), task
        assert len(preflight_result.known_failures) == 1, task
        assert len(preflight_result.verified_lessons) == 1, task
        assert not guard_result.is_empty(), task
        assert len(guard_result.applicable_skills) == 1, task


def test_corpus_long_noisy_paraphrase_recovers_known_failure(tmp_path):
    """A long task with realistic surrounding chatter (deployment
    checklist, "could you help me understand") around the same
    vocabulary the attempt itself uses -- the A6 dilution shape,
    reproduced in an unrelated domain. Only the failed Attempt is
    asserted here (via `preflight()`); the Lesson/Skill do not clear
    the symmetric threshold for this particular query, which is an
    honest outcome, not a bug: `guard()` staying more conservative than
    `preflight()` here is the intended asymmetry (see `_guard.py`)."""
    cx = Cortex.init(tmp_path, "dev")
    failed_attempt, _, _, _ = _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    task = (
        "I was reviewing the deployment checklist and also wanted to check on how "
        "to fix that database connection pool exhaustion problem we keep running "
        "into under load, could you help me understand what has already been "
        "tried there"
    )
    result = agent_b.preflight(task)
    guard_result = agent_b.guard(task)

    assert [a.attempt_id for a in result.known_failures] == [failed_attempt.attempt_id]
    # guard() staying empty here, for the very query that already moved
    # preflight(), is the conservatism asymmetry sec. 33.5 requires made
    # concrete, not just asserted in prose.
    assert guard_result.is_empty()


def test_corpus_moderate_synonym_paraphrase_is_a_documented_residual_gap(tmp_path):
    """Documents the residual semantic gap A7 does not close (see A7.0
    report, section 27): a short paraphrase that swaps in synonyms and
    different surface forms ("exhausting" for "exhaustion") without
    reusing enough of the candidate's own exact vocabulary shares too
    little with a short candidate for either channel to admit it.
    Closing this would require stemming or semantic retrieval, both
    explicitly out of scope for A7. This is a `pytest.mark` -free
    assertion of the CURRENT, honest behavior, not a target to
    eventually flip to "found" without adding a new retrieval channel.
    """
    cx = Cortex.init(tmp_path, "dev")
    _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    task = "we keep seeing database connection pool exhaustion whenever traffic gets high"
    result = agent_b.preflight(task)

    assert result.is_empty()


def test_corpus_borderline_negatives_do_not_leak(tmp_path):
    """Queries that share real vocabulary with the stored experience
    ('pool', 'connection') or with Cortex's own retrieval vocabulary
    ('preflight', 'guard', 'relevant', 'validation', 'experience') but
    describe a genuinely different problem. Widened candidate
    generation must not let any of them surface the connection-pool
    experience."""
    cx = Cortex.init(tmp_path, "dev")
    _, _, lesson, skill = _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    borderline_queries = (
        "check whether preflight and guard have enough relevant validation "
        "experience for this new pool of features",
        "redis cache connections keep failing intermittently, is that connected "
        "to something we've seen before",
        "increase the thread pool size for the background job scheduler",
        "getting a connection refused error when SSH-ing into the staging box",
    )
    for query in borderline_queries:
        preflight_result = agent_b.preflight(query)
        guard_result = agent_b.guard(query)
        assert lesson.memory_id not in [m.memory_id for m in preflight_result.verified_lessons], query
        assert skill.skill_id not in [s.skill_id for s in guard_result.applicable_skills], query


def test_corpus_hard_negatives_are_empty(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    hard_negatives = (
        "Change CSS button color to blue",
        "Update the onboarding documentation with new screenshots",
        "Rename the login endpoint to /auth/login",
    )
    for query in hard_negatives:
        assert agent_b.preflight(query).is_empty(), query
        assert agent_b.guard(query).is_empty(), query


def test_corpus_short_generic_attempt_does_not_leak_into_a_long_unrelated_query(tmp_path):
    """Adversarial case found during A7.1 review: a very short, entirely
    generic attempt (its whole task+approach text is words the original
    `is_relevant` docstring itself names as too weak alone -- "fix",
    "error", "update", "test") is genuinely unrelated to a long, natural
    query that happens to use the same generic words for something
    else entirely. The FTS-widening symmetric threshold, evaluated on
    the CANDIDATE's own short length alone, can drop low enough to admit
    this by coincidence; the fix caps how far the threshold can drop
    below the original query-length-scaled bar (see `_retrieval.py`)."""
    cx = Cortex.init(tmp_path, "dev")
    cx.record_attempt(task="Fix the error", approach="Update the test", outcome="failed")
    agent_b = Cortex.discover(tmp_path)

    query = (
        "I was going through the release notes and wanted to fix an unrelated error "
        "in the onboarding flow, then update the test suite for the notification "
        "service before the demo tomorrow, does any of this ring a bell from "
        "something we already looked at"
    )
    result = agent_b.preflight(query)

    assert result.known_failures == ()


def test_corpus_relevant_attempt_is_not_excluded_by_a_fixed_top_k_rank_cutoff(tmp_path):
    """Adversarial case found during A7.1 review: with enough other
    attempts that share dense, high-frequency vocabulary with a long
    natural query (a realistic situation once a workspace has more than
    a handful of related-but-distinct attempts), a fixed top-K BM25
    rank cutoff can push a genuinely relevant, threshold-qualifying
    attempt past the cutoff and exclude it entirely -- turning "which
    candidates are even considered" into an unintended second recall
    gate, on top of the (already validated) shared-token threshold that
    is supposed to be the actual admission decision. The fix widens
    candidate consideration to no longer treat rank position as a
    relevance signal by itself."""
    cx = Cortex.init(tmp_path, "dev")
    noise_specs = [
        ("Checklist database connection pool exhaustion fix", "Reviewed checklist"),
        ("Database connection pool exhaustion checklist review", "Updated checklist"),
        ("Fix database connection pool under load", "Restarted service"),
        ("Connection pool exhaustion under load", "Bumped timeout"),
        ("Database connection pool exhaustion checklist", "Wrote runbook"),
        ("Exhaustion pool database connection load fix", "Patched driver"),
        ("Fix connection pool exhaustion load database", "Restarted the pool"),
        ("Pool exhaustion connection database load", "Rotated credentials"),
        ("Database connection pool load", "Reviewed config"),
        ("Database pool exhaustion during load", "Scaled replicas"),
    ]
    for task, approach in noise_specs:
        cx.record_attempt(task=task, approach=approach, outcome="failed")

    correct_attempt = cx.record_attempt(
        task="Fix database connection pool exhaustion under load",
        approach="Increased the pool size without addressing connection leaks",
        outcome="failed",
    )
    agent_b = Cortex.discover(tmp_path)

    query = (
        "I was reviewing the deployment checklist and also wanted to check on how "
        "to fix that database connection pool exhaustion problem we keep running "
        "into under load, could you help me understand what has already been "
        "tried there"
    )
    result = agent_b.preflight(query)

    assert correct_attempt.attempt_id in [a.attempt_id for a in result.known_failures]


def test_corpus_ordering_is_deterministic_across_repeated_calls(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    _build_connection_pool_experience(cx)
    agent_b = Cortex.discover(tmp_path)

    task = "Fix database connection pool exhaustion under load"
    first = agent_b.preflight(task)
    second = agent_b.preflight(task)

    assert [a.attempt_id for a in first.known_failures] == [a.attempt_id for a in second.known_failures]
    assert [m.memory_id for m in first.verified_lessons] == [m.memory_id for m in second.verified_lessons]
