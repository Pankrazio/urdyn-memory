"""Candidate widening via FTS5/BM25 for `preflight()`/`guard()`.

The structured lexical signal in `_relevance.py` (`is_relevant`) remains
the primary, unchanged relevance decision: a strict majority of the
QUERY's own vocabulary must appear in the candidate. That rule is exact
and well-tested, but it structurally penalizes long, naturally-phrased
queries against short, concise canonical memories: the threshold grows
with query length, so a task phrased with ordinary surrounding words
dilutes the shared/query ratio below threshold even when the query
substantially covers the candidate's own, much smaller vocabulary. This
is the A6 failure mode: relevant experience existed, but a naturally
worded task missed it.

FTS5/BM25 is a second, independent channel a candidate can be admitted
through instead of the lexical majority rule. Its admission rule is
deliberately not a confidence score: BM25 values are never compared to
an arbitrary constant. BM25 RANK was tried as the admission criterion
first, and rejected: found during A7.1 review, a fixed top-K rank
cutoff (rank was being used as if it were itself a relevance signal)
let enough other lexically dense, topically-adjacent candidates
outrank a genuinely relevant, threshold-qualifying one that it was
pushed past the cutoff and excluded outright -- turning "which
candidates are even considered" into an unintended second recall gate
riding on top of the threshold check below, which is supposed to be
the actual admission decision. Rank is not used for admission at all
now: every FTS match (i.e. every candidate sharing at least one
significant token with the query -- MATCH already guarantees that) is
evaluated against the same shared-token threshold. `_FTS_CANDIDATE_LIMIT`
still bounds how many matches are evaluated per call, but purely as a
defensive resource bound against a pathological result set, not as a
relevance heuristic -- set generously relative to the tens-to-hundreds
of records a per-project Urdyn workspace is expected to hold (see
A7.1's populated-workspace test), so it is not expected to bind in
normal use.

A flat "shares N words" floor was tried first for the threshold check
itself, and also rejected: it let a completely unrelated attempt back
in through
`test_preflight_generic_engineering_vocabulary_does_not_cross_match`,
because two generic engineering words ("update", "error") clear a
small fixed floor trivially. The floor used here instead is the same
majority formula `is_relevant` already uses, generalized to be
symmetric: shared tokens must be a strict majority of the SMALLER of
the query's and the candidate's own vocabulary --
`min(len(query_tokens), len(candidate_tokens)) // 2 + 1`. When the
candidate's own text is at least as long as the query (the
generic-vocabulary case: two attempts with as much text as the query
itself), this is identical to `is_relevant`'s own threshold, so nothing
is admitted here that the original rule would not already have
rejected. It only relaxes admission when the candidate is markedly
SHORTER than the query -- exactly the A6 shape, a concise stored
memory versus a long, naturally-phrased task -- where the threshold
scales down with the candidate's own small vocabulary instead of
staying pinned to the query's inflated length.

The symmetric formula alone is still not enough: found during A7.1
review, a candidate that is both SHORT and made ENTIRELY of generic
words ("Fix the error" / "Update the test" -- literally the words
`is_relevant`'s own docstring names as too weak alone) can have 100%
of its own tiny vocabulary coincidentally present in a long, genuinely
unrelated query, since generic engineering words are common by
definition. Shortness was meant to be a proxy for "this candidate's
own vocabulary is a meaningful, specific bar to clear" -- it stops
being a good proxy when the candidate has nothing but generic words to
begin with, since clearing 100% of an all-generic vocabulary carries
no real signal. `_symmetric_majority_threshold` therefore caps how far
candidate-shortness alone is allowed to lower the bar: the threshold
can never drop below half of what `is_relevant`'s own query-length-scaled
threshold would have required, no matter how short or generic the
candidate is. This
still fully preserves the A6 recovery cases (their shared-token counts
clear this floor with room to spare) while rejecting the all-generic
short-candidate case (whose shared count comes only from genericness,
not from covering a meaningful share of the query).

Entity type strings (`ENTITY_MEMORY`/`ENTITY_ATTEMPT`/`ENTITY_SKILL`/
`ENTITY_SOURCE`) are internal vocabulary shared only between this
module, `_store.py`'s FTS index, and `_semantic.py`'s pool policy;
nothing about them is part of the public API.

[A52] `ENTITY_SOURCE` widens the same lexical/FTS admission this module
already provides to the CURRENT (latest-observation) Evidence of a
seeded Source -- see `_workspace.py`'s `_semantic_pool_entries` and
`Urdyn.context()`. It is deliberately the fourth pool, not a
repurposing of `ENTITY_MEMORY`: a document observation is canonical
Evidence, never a Memory (see `_evidence.py`'s module docstring), and
mixing it into the Memory pool would let raw, unverified document text
compete for the same FTS/semantic admission slots as verified
experience.
"""

from __future__ import annotations

from ._relevance import tokens as _tokens

ENTITY_MEMORY = "memory"
ENTITY_ATTEMPT = "attempt"
ENTITY_SKILL = "skill"
ENTITY_SOURCE = "source"

# A defensive bound on how many FTS matches are evaluated per call, not
# a relevance cutoff: every match is judged on the same threshold
# regardless of its rank, so this only protects against a pathological
# result set, and is set well above realistic per-project workspace
# sizes so it is not expected to bind in normal use.
_FTS_CANDIDATE_LIMIT = 200


def _symmetric_majority_threshold(query_token_count: int, candidate_token_count: int) -> int:
    original_threshold = query_token_count // 2 + 1
    symmetric_threshold = min(query_token_count, candidate_token_count) // 2 + 1
    relaxation_cap = original_threshold // 2
    return max(symmetric_threshold, relaxation_cap)


def fts_admitted_ids(
    query_tokens: frozenset[str], ranked_candidates: list[tuple[str, str]]
) -> frozenset[str]:
    """Select the subset of FTS-ranked candidates admitted through the
    widening channel.

    `ranked_candidates` is `(entity_id, text)` pairs already ordered
    best-first by BM25 rank for one entity type (see
    `MemoryStore.search_candidates`); the order itself is not used for
    admission (see module docstring), only as a stable, deterministic
    iteration order. Every candidate up to `_FTS_CANDIDATE_LIMIT` (a
    defensive bound, not a relevance cutoff) is evaluated against the
    symmetric majority threshold described in the module docstring. An
    empty `query_tokens` or `ranked_candidates` admits nothing.
    """
    if not query_tokens:
        return frozenset()
    admitted = []
    for entity_id, text in ranked_candidates[:_FTS_CANDIDATE_LIMIT]:
        candidate_tokens = _tokens(text)
        shared = query_tokens & candidate_tokens
        threshold = _symmetric_majority_threshold(len(query_tokens), len(candidate_tokens))
        if len(shared) >= threshold:
            admitted.append(entity_id)
    return frozenset(admitted)
