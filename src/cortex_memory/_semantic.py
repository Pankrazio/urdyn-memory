"""Optional semantic retrieval channel: static embeddings (model2vec) plus
brute-force cosine similarity, used to widen candidate recall in
`preflight()`/`guard()` the same way `_retrieval.py`'s FTS5/BM25 channel
already does.

This module is the ONLY place in Cortex that imports `model2vec` or
`numpy`. It is never imported from `__init__.py`, `_workspace.py` at
module scope, or any other module reachable from a plain `import
cortex_memory` -- callers (`_workspace.py`, `_cli.py`) import it lazily,
inside functions, wrapped in `try/except ImportError`, so a base install
without the `cortex-memory[semantic]` extra never even attempts to load
`model2vec`/`numpy` and behaves exactly as it did before A7.4.

Two loading entry points exist on purpose:
  - `load_model_for_setup` allows a network download if the model is not
    yet cached (used only by `cortex semantic setup`).
  - `load_model_for_retrieval` forces `HF_HUB_OFFLINE=1` for the duration
    of the call and passes `force_download=False`, so a normal
    `preflight()`/`guard()` call can never trigger a network request --
    required because model2vec's own `StaticModel.from_pretrained`
    defaults to `force_download=True`, which unconditionally calls
    `huggingface_hub.snapshot_download` (a network round-trip) even when
    the model is already cached. Passing `force_download=False`
    explicitly is what makes `_resolve_folder` return the cached path via
    `maybe_get_cached_model_path` without touching the network at all;
    `HF_HUB_OFFLINE=1` is a second, redundant guarantee of the same
    property, kept in case that internal resolution behavior ever changes
    upstream. This was found and verified empirically during A7.4
    implementation, not assumed.

Admission ("is this candidate semantically relevant enough to widen
recall with") is calibrated per entity-type pool (attempt / memory /
skill) from real per-query brute-force rankings over the frozen A7.3
evaluation corpus, ranked separately per pool exactly as Cortex itself
pools candidates -- see `SEMANTIC_POLICY` below and the A7.4 report for
the full calibration data. Calibration found that no single
similarity/margin threshold cleanly separates every Human Acceptance
query from every adversarial negative in this corpus (concretely: the
payment-guard-clause false positive from A7.3 and Human Acceptance query
`ha-guard-2` are mathematically inseparable on this signal alone, since
the false positive scores higher on both absolute similarity and margin).
The shipped policy is deliberately calibrated toward precision: it
recovers a real subset of the Human Acceptance gap without reintroducing
the known false positive, and abstains rather than guessing on the
genuinely ambiguous remainder. This is a documented trade-off, not a
claim that every Human Acceptance query is recovered by the semantic
channel alone.

[A7.7] Eligibility fix: `semantic_admitted_ids` accepts an optional
`eligible_ids` filter, applied to the candidate pool BEFORE ranking
(not after). A7.5/A7.6 found and reproduced a real correctness bug: an
entity the consumer could never use anyway (not current, not verified,
a superseded memory) could still win the pool's single admission slot
by outranking every usable candidate, permanently starving them of
consideration -- eligibility was checked only downstream, against the
one id this module had already committed to. `eligible_ids` is a
generic id filter, nothing more: this module still has no idea what
"verified" or "current" mean, or which consumer is asking. Restricting
the pool to eligible ids before ranking does not mean "always admit the
best eligible candidate" -- the exact same absolute/margin policy below
still applies to whatever wins the now-restricted pool; abstention is
still the outcome whenever nothing eligible clears it.
claim that every Human Acceptance query is recovered by the semantic
channel alone.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import huggingface_hub
import numpy as np
from model2vec import StaticModel

SEMANTIC_PROVIDER = "model2vec"
SEMANTIC_MODEL_ID = "minishlab/potion-retrieval-32M"
SEMANTIC_NORMALIZATION = "l2"

ENTITY_ATTEMPT = "attempt"
ENTITY_MEMORY = "memory"
ENTITY_SKILL = "skill"


@dataclasses.dataclass(frozen=True, slots=True)
class PoolPolicy:
    """Calibrated abstention floors for one entity-type pool. A candidate
    is admitted only if it ranks #1 in its pool AND clears both floors --
    see module docstring for how these were derived."""

    absolute_floor: float
    margin_floor: float


# Calibrated from the real per-pool brute-force rankings recorded in the
# A7.4 report (frozen A7.3 corpus, `minishlab/potion-retrieval-32M`).
# ATTEMPT: no Human Acceptance query targets this pool; the floor is set
#   high, comfortably above every false candidate observed (highest was
#   0.42/0.29), purely for architectural symmetry with the other pools.
# MEMORY: recovers `ha-preflight-2` (0.2812/0.0977); deliberately does
#   NOT recover `ha-preflight-1` (0.1925/0.0602), which is
#   indistinguishable on this signal from a genuine hard negative
#   (`hn-q-8`, 0.2025/0.0621) scoring higher on both axes.
# SKILL: recovers `ha-guard-1` (0.5217/0.4198); deliberately does NOT
#   recover `ha-guard-2` (0.2285/0.1792), which is dominated on both axes
#   by the payment-guard-clause false positive (0.3393/0.1951) that A7.3
#   found and this tracer must not reintroduce.
SEMANTIC_POLICY: dict[str, PoolPolicy] = {
    ENTITY_ATTEMPT: PoolPolicy(absolute_floor=0.50, margin_floor=0.35),
    ENTITY_MEMORY: PoolPolicy(absolute_floor=0.20, margin_floor=0.08),
    ENTITY_SKILL: PoolPolicy(absolute_floor=0.40, margin_floor=0.38),
}

_model_cache: dict[str, StaticModel] = {}


class SemanticUnavailable(Exception):
    """Raised internally when the semantic channel cannot be used right
    now (extra not installed, model not set up, index missing/stale).
    Never propagates out of `_workspace.py`'s public methods -- every
    caller there catches this and degrades to lexical/FTS-only, exactly
    as if A7.4 did not exist."""


def load_model_for_setup(model_id: str = SEMANTIC_MODEL_ID) -> StaticModel:
    """Load (downloading if not already cached) the semantic model.
    Only ever called from `cortex semantic setup` -- allowed to touch
    the network."""
    if model_id not in _model_cache:
        _model_cache[model_id] = StaticModel.from_pretrained(model_id, force_download=False)
    return _model_cache[model_id]


def load_model_for_retrieval(model_id: str = SEMANTIC_MODEL_ID) -> StaticModel:
    """Load the semantic model for a normal retrieval call. Never
    touches the network (see module docstring): raises whatever
    model2vec/huggingface_hub raises if the model is not already cached,
    which callers must treat as `SemanticUnavailable` and degrade from,
    never as a crash."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = StaticModel.from_pretrained(model_id, force_download=False)
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
    _model_cache[model_id] = model
    return model


def model_dimensions(model: StaticModel) -> int:
    return int(model.encode(["_cortex_semantic_dimension_probe_"]).shape[1])


def resolve_local_revision(model_id: str = SEMANTIC_MODEL_ID) -> str | None:
    """Best-effort local commit hash for the currently cached snapshot of
    `model_id`, via `huggingface_hub.scan_cache_dir()` -- a supported
    public API, not a scrape of the cache's internal file layout. Returns
    None (an explicit, documented "unresolvable" state, not an error) if
    the repo is not cached, or if more than one revision is cached
    without a clear `main` ref to disambiguate: forcing a guess in that
    case would be worse than admitting the revision is not known."""
    try:
        cache_info = huggingface_hub.scan_cache_dir()
    except Exception:
        return None
    for repo in cache_info.repos:
        if repo.repo_id != model_id or repo.repo_type != "model":
            continue
        revisions = list(repo.revisions)
        if len(revisions) == 1:
            return revisions[0].commit_hash
        for revision in revisions:
            if "main" in revision.refs:
                return revision.commit_hash
        return None
    return None


def embed(model: StaticModel, texts: Sequence[str]) -> np.ndarray:
    """Embed `texts` and L2-normalize each row so cosine similarity
    reduces to a plain dot product. model2vec's own vectors are already
    close to unit norm (verified empirically -- see `is_normalized`),
    but normalization is applied explicitly here rather than assumed,
    so this module's correctness never depends on that being true of
    whichever model is configured."""
    vectors = np.asarray(model.encode(list(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def is_normalized(vectors: np.ndarray, *, atol: float = 1e-3) -> bool:
    if vectors.size == 0:
        return True
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.allclose(norms, 1.0, atol=atol))


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dimensions: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.shape[0] != dimensions:
        raise ValueError(
            f"Corrupted semantic vector: expected {dimensions} float32 dimensions, "
            f"got {vector.shape[0]} (blob is {len(blob)} bytes)"
        )
    return vector


def rank_candidates(
    query_vector: np.ndarray, ids: Sequence[str], matrix: np.ndarray
) -> list[tuple[str, float]]:
    """Rank every candidate in `matrix` against `query_vector` by cosine
    similarity (both assumed already L2-normalized), best first. Scores
    the FULL pool every time -- no early top-K cutoff (A7.1 already found
    that rank-based cutoffs applied before scoring can silently drop a
    relevant candidate; the same lesson applies here)."""
    if len(ids) == 0:
        return []
    scores = matrix @ query_vector
    order = np.argsort(-scores)
    return [(ids[i], float(scores[i])) for i in order]


def semantic_rank_eligible(
    query_text: str,
    *,
    model: StaticModel,
    stored_vectors: list[tuple[str, bytes]],
    dimensions: int,
    eligible_ids: frozenset[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank `stored_vectors` against `query_text`, best first -- the
    pure ranking primitive underneath `semantic_admitted_ids`, exposed
    separately so a caller (e.g. `_workspace.py`'s preflight
    corroboration path) can inspect the top-ranked ELIGIBLE candidate
    and its score even when it does not clear the admission policy
    below. `eligible_ids`, if given, restricts the pool BEFORE ranking
    (see module docstring's [A7.7] note) -- a plain id filter, carrying
    no meaning about why those ids were chosen. Returns `[]` for
    blank/whitespace-only query text or an empty (post-filter) pool.
    """
    if eligible_ids is not None:
        stored_vectors = [(eid, blob) for eid, blob in stored_vectors if eid in eligible_ids]
    if not stored_vectors or not query_text.strip():
        return []
    ids = [entity_id for entity_id, _ in stored_vectors]
    matrix = np.stack([blob_to_vector(blob, dimensions) for _, blob in stored_vectors])
    query_vector = embed(model, [query_text])[0]
    return rank_candidates(query_vector, ids, matrix)


def semantic_admitted_ids(
    query_text: str,
    entity_type: str,
    *,
    model: StaticModel,
    stored_vectors: list[tuple[str, bytes]],
    dimensions: int,
    eligible_ids: frozenset[str] | None = None,
) -> frozenset[str]:
    """End-to-end widening call: embed `query_text`, decode `stored_vectors`
    (`(entity_id, blob)` pairs for one entity-type pool, as persisted by
    `_semantic_store.SemanticIndexStore`), rank, and apply the calibrated
    abstention policy. Returns a set of at most one id -- see
    `semantic_admitted_id`. Returns an empty set for blank/whitespace-only
    query text or an empty pool, same as the FTS channel does.

    `eligible_ids`, if given, restricts the pool to those ids BEFORE
    ranking -- the [A7.7] correctness fix (see module docstring): a
    candidate the caller could never use anyway no longer occupies the
    pool's single winner-take-all slot. This is NOT "always admit the
    best eligible candidate" -- `semantic_admitted_id` still applies the
    exact same absolute/margin policy to whatever wins the (possibly
    restricted) pool, so abstention remains the outcome whenever nothing
    eligible clears it.
    """
    ranked = semantic_rank_eligible(
        query_text, model=model, stored_vectors=stored_vectors, dimensions=dimensions, eligible_ids=eligible_ids
    )
    admitted = semantic_admitted_id(ranked, entity_type)
    return frozenset({admitted}) if admitted is not None else frozenset()


def semantic_admitted_id(ranked: list[tuple[str, float]], entity_type: str) -> str | None:
    """Return the single admitted candidate id for this pool, or None if
    nothing clears the calibrated abstention policy -- abstention is a
    normal, expected outcome, not a failure (see module docstring:
    "semantic candidate #1 is not automatically semantic admission").
    Only the #1-ranked candidate is ever eligible: lower ranks are, by
    construction, even less likely to clear the bar than #1 already
    would have to.

    A pool with a single candidate has no runner-up to be ambiguous
    against, so the margin floor is not applied in that case -- only the
    absolute floor is (found during A7.4 implementation: with the naive
    "margin = 0 when there is no runner-up" definition, a real,
    comfortably-scoring match in a small workspace with only one
    candidate in its pool was incorrectly rejected outright).
    """
    if not ranked:
        return None
    policy = SEMANTIC_POLICY[entity_type]
    top_id, top_score = ranked[0]
    if top_score < policy.absolute_floor:
        return None
    if len(ranked) == 1:
        return top_id
    margin = top_score - ranked[1][1]
    if margin >= policy.margin_floor:
        return top_id
    return None
