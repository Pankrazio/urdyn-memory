"""Optional semantic retrieval channel: local ONNX sentence embeddings
plus brute-force cosine similarity, used to widen candidate recall in
`preflight()`/`guard()` the same way `_retrieval.py`'s FTS5/BM25 channel
already does.

This module is the ONLY place in Cortex that imports `onnxruntime`,
`tokenizers`, `huggingface_hub` or `numpy`. It is never imported from
`__init__.py`, `_workspace.py` at module scope, or any other module
reachable from a plain `import cortex_memory` -- callers
(`_workspace.py`, `_cli.py`) import it lazily, inside functions, wrapped
in `try/except ImportError`, so a base install without the
`cortex-memory[semantic]` extra never even attempts to load them and
behaves exactly as it did before A7.4. Nothing outside this module ever
sees an `InferenceSession`, a `Tokenizer`, an artifact filename or a
Hugging Face cache path: the rest of Cortex only ever gets
`embed(model, texts) -> normalized vectors`.

[A16.3] BACKEND: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
executed directly through ONNX Runtime, replacing the model2vec/potion
backend used from A7.4 to A16.2. The reason is cross-language recall:
A16 measured the previous backend at 0-30% recall for cross-language
queries (an Italian question against an English memory and vice versa),
against 83-84% for this model on the same frozen corpus, with zero false
retrievals and no authority leakage. A16.2.1 additionally verified,
numerically, that driving the OFFICIAL fp32 artifact through the recipe
below reproduces the reference implementation exactly (per-text cosine
1.00000, top-1 agreement 1.000, rank correlation 1.000), which is why
Cortex owns this small encoder instead of taking a heavier third-party
runtime dependency that would have added ~200 MB of RSS for the same
vectors.

The embedding recipe is the model's OWN published SentenceTransformer
configuration, not an invention: mean pooling over the attention mask
(`1_Pooling/config.json`: `pooling_mode_mean_tokens: true`,
`pooling_mode_cls_token: false`) and truncation at 128 tokens
(`sentence_bert_config.json`: `max_seq_length: 128`). The model's own
`modules.json` declares no `Normalize` module, so the model does not
L2-normalize its output; `embed()` below does it explicitly, exactly as
it always has, which is what keeps cosine similarity a plain dot
product.

`SEMANTIC_MAX_SEQ_LENGTH` is pinned here as a constant rather than read
from the repository at runtime ON PURPOSE: it is part of Cortex's
embedding contract, and a value silently changed upstream must not be
able to silently change the vector space of an already-indexed
workspace. The same reasoning pins the model revision (see
`SEMANTIC_MODEL_REVISION`). Values that CAN drift without a contract
change (the padding token id, the output dimensionality) are read from
the real artifacts at setup instead of trusted.

Two loading entry points exist on purpose:
  - `load_model_for_setup` allows a network download if the artifact is
    not yet cached (used only by `cortex semantic setup`).
  - `load_model_for_retrieval` passes `local_files_only=True`, so a
    normal `preflight()`/`guard()` call can never trigger a network
    request: a missing artifact raises, and every caller in
    `_workspace.py` treats that as `SemanticUnavailable` and degrades to
    lexical/FTS, rather than reaching for the network.

[A16.3] EFFECTIVE MODEL IDENTITY. A16.2.1 measured a real hazard: two
different ONNX artifacts of the SAME model produce slightly different
vectors (per-text cosine 0.9947 between the quantized and the
full-precision artifact), enough to change the top-ranked candidate for
~1 query in 14. An index built with one artifact must therefore never be
queried with another. What is persisted as the index's `model_id` is
consequently not the bare repository name but the full effective
identity `repo@revision#artifact`, so that the artifact and the upstream
revision are both part of what `SemanticMeta.matches()` compares -- no
schema change was needed, because `model_id` is free text. Retrieval
then loads THE ARTIFACT THE INDEX RECORDS (see `artifact_for_index`),
which removes the mixing hazard entirely instead of merely detecting it:
an index built with the ARM64 artifact is queried with the ARM64
artifact, or, if that artifact cannot be loaded here, not at all.

Admission ("is this candidate semantically relevant enough to widen
recall with") is calibrated per entity-type pool (attempt / memory /
skill) -- see `SEMANTIC_POLICY` below. A16.2.1 re-validated the MEMORY
pool's shipped floors against this backend on a frozen 14-scenario,
4-language holdout and found them to hold unchanged (83% recall, zero
false retrievals, and the margin floor demonstrably rejecting all three
wrong top-1 candidates, whose margins topped out at 0.0499 against a
0.08 floor), which is why A16.3 changed the backend without touching
MEMORY. It left the other two pools unmeasured, and A16.3.1 then closed
that gap on a frozen SKILL/ATTEMPT corpus: ATTEMPT held, SKILL did not
and its absolute floor moved. The lesson worth keeping is that these
floors are properties of the model's score geometry, so "the backend
changed" and "the policy is still calibrated" are separate claims that
need separate evidence, per pool.

[A23.1] TWO ADMISSION POLICIES, ONE SIMILARITY ENGINE. `SEMANTIC_POLICY`
above is a per-POOL calibration (which floors), not a per-pool decision
about how many candidates a pool may return. That second question
belongs to the category CONSUMING the pool: `semantic_admitted_id`
answers "which single candidate is the intended one" (winner + margin,
unchanged since A7.4), while `set_admitted_ids` answers "which
candidates are relevant enough to include" (every candidate above the
SAME absolute floor, capped, no margin). Nothing about the scores, the
model, the index or the floors differs between them. A23 measured why
the distinction is necessary: applying the single-winner question to
verified Lessons -- a category whose own model expresses no exclusivity,
and which the lexical/FTS channels already admit as a set -- rejected
two complementary lessons for being 0.0576 apart while both sat far
above the floor.

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
"""

from __future__ import annotations

import dataclasses
import platform
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import huggingface_hub
import numpy as np
import onnxruntime
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from ._semantic_store import SemanticMeta

SEMANTIC_PROVIDER = "onnxruntime"
SEMANTIC_NORMALIZATION = "l2"

# Canonical model repository, pinned to the exact upstream revision
# validated in A16.2.1. Pinning the revision (rather than tracking
# `main`) is what stops an upstream re-export from silently changing the
# vector space of workspaces that are already indexed.
SEMANTIC_MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

# Part of the embedding contract -- see module docstring for why this is
# pinned here instead of read from the repository at runtime.
SEMANTIC_MAX_SEQ_LENGTH = 128

TOKENIZER_FILENAME = "tokenizer.json"

# How many texts are pushed through the ONNX graph at once. This is a
# MEMORY bound, not a speed knob, and it is why it exists at all: the
# previous backend was a static embedding lookup, so encoding an entire
# workspace in one call cost nothing beyond the vectors themselves. A
# transformer's intermediate activations scale with the batch, and
# `semantic_setup()` legitimately hands the encoder every memory in the
# workspace at once -- measured during A16.3, a 1002-memory workspace
# encoded as a single batch peaked at 6.3 GB of RSS. Chunking keeps peak
# memory flat in the size of the workspace, which is what makes this
# usable on a normal laptop rather than only on a small workspace.
SEMANTIC_ENCODE_BATCH_SIZE = 32

# Official artifacts published by the model repository itself. Cortex
# picks one internally; there is deliberately no user-facing choice.
ARTIFACT_PORTABLE = "onnx/model.onnx"
ARTIFACT_X86_64 = "onnx/model_quint8_avx2.onnx"
ARTIFACT_ARM64 = "onnx/model_qint8_arm64.onnx"
SUPPORTED_ARTIFACTS = frozenset({ARTIFACT_PORTABLE, ARTIFACT_X86_64, ARTIFACT_ARM64})

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


# All thresholds below were calibrated empirically on frozen evaluation
# corpora, each frozen BEFORE it was scored. The corpora are not shipped
# with the package; the measured outcomes are recorded here so the
# operating point can be audited and re-derived.
#
# Originally calibrated in A7.4 on the frozen A7.3 corpus. Scores are a
# property of the MODEL, not of Cortex, so a backend change invalidates
# these numbers until they are re-measured pool by pool.
# ATTEMPT: absolute floor unchanged since A7.4 and re-validated twice --
#   it sits just above the highest false candidate seen in either corpus
#   (0.4625), which is what makes it the gate doing the real work. The
#   margin floor moved from 0.35 to 0.08 in A16.3.2. A16.3.1 had measured
#   0.35 rejecting every multi-candidate positive it saw (0/5) while
#   stopping no negative, but could not act: its holdout contained no
#   multi-candidate scene, so no replacement was independently validated.
#   A16.3.2 built that holdout -- 12 new multi-candidate scenes in unused
#   domains -- ran all four frozen candidates (0.35/0.20/0.10/0.08) over
#   it once, and got zero false admissions from all four with
#   multi-candidate recall of 2/6/8/9 respectively. 0.08 is therefore the
#   most permissive value that still cost nothing, not the loosest value
#   that could be defended: it is doing real work at that level, since it
#   is what rejected the one wrong top-1 candidate that cleared the
#   absolute floor (a runner-up 0.0250 behind it). Below the margin the
#   pool is ambiguous and abstention is still the answer.
# MEMORY: A16.2.1 holdout, 63 positive queries across 4 languages: 83%
#   recall, 0 false retrievals; every wrong top-1 candidate had a margin
#   of at most 0.0499, i.e. below this floor, so the floor is doing real
#   work rather than being nominally satisfied.
# SKILL: raised from 0.40 to 0.55 in A16.3.1. Under the A16.3 backend
#   this pool's calibration scores are bimodal -- restatements of a skill
#   land at 0.66-0.73, genuinely-applicable-but-differently-worded ones
#   at 0.34-0.40 -- while the worst false candidate reaches 0.4570. The
#   old 0.40 floor therefore sat INSIDE the false-positive band without
#   buying any recall: no calibration positive scored between 0.40 and
#   0.66, so every value in the empty 0.457-0.6585 gap scores identically
#   and 0.55 is simply its centre, chosen for distance from both edges
#   rather than fitted to any one query. It removes all three calibration
#   false positives (0.4134, 0.4555, 0.4570) at a measured cost of two
#   holdout positives (0.4268, 0.5059) -- the trade A16.3.1 accepted
#   deliberately, since a Skill wrongly reported as applicable misleads an
#   agent more than a missing one does. The A7.3 payment-guard-clause
#   false positive (0.4555) was deliberately held out of calibration so
#   the floor could not be fitted to it; it is rejected here while the
#   same Skill's genuine query (0.5713) is still admitted.
#   A16.3.2 then re-decided this value prospectively, on a second corpus
#   of 14 scenes in domains the first one never used, running 0.40 / 0.50
#   / 0.55 over it once. 0.55 was the only one with zero false
#   admissions. The interesting part is 0.50, which A16.3.1 had flagged
#   post-hoc as possibly better and deliberately not adopted: it admitted
#   the same two false candidates as 0.40 (0.5031 and 0.5252, both
#   landing in the 0.50-0.55 band), so the caution was right and the
#   hypothesis is now measured rather than open.
#   The margin floor moved from 0.38 to 0.10 in A16.3.3 -- the last value
#   in any pool still carrying a Potion-era number. A16.3.2 had measured
#   0.38 rejecting 8 of 9 multi-candidate positives, including matches
#   scoring 0.79 with a 0.23 lead over the runner-up, which is the shape
#   of an answer a policy should admit rather than suppress. A16.3.3 ran
#   0.38 / 0.20 / 0.10 / 0.08 over a third corpus of 13 new
#   multi-candidate scenes: all four gave zero false admissions,
#   multi-candidate recall went 1 / 4 / 9 / 9, and the last two tied, so
#   the more conservative was taken. What makes 0.10 defensible rather
#   than merely permitted: at that value exactly ONE positive in the
#   corpus is rejected by the margin instead of by the absolute floor,
#   and in that one the top-ranked candidate was the WRONG Skill, 0.0519
#   ahead of the right one -- the gate abstained exactly where abstaining
#   was correct. Every other rejection there is now the absolute floor's
#   doing, which is where the remaining recall question lives.
SEMANTIC_POLICY: dict[str, PoolPolicy] = {
    ENTITY_ATTEMPT: PoolPolicy(absolute_floor=0.50, margin_floor=0.08),
    ENTITY_MEMORY: PoolPolicy(absolute_floor=0.20, margin_floor=0.08),
    ENTITY_SKILL: PoolPolicy(absolute_floor=0.55, margin_floor=0.10),
}

# [A23.1/A23.2] The SET admission policy for verified Lessons: how deep
# the pool may go, and how relevant a candidate must be. Internal policy
# constants, deliberately not exposed through the CLI, the public API,
# the manifest or an environment variable -- a caller able to raise them
# would be able to turn preflight into a dump of the workspace.
#
# Both were CALIBRATED IN A23.2 on a prospective corpus frozen before it
# was scored (78 verified lessons over 19 domains, 67 task scenes across
# two languages, 13 of them scenes where the correct answer is to emit
# nothing), then validated once against the untouched A23.1.1 diagnostic
# corpus. They are a V1 operating point, not architectural truth: like
# every number in `SEMANTIC_POLICY`, they are properties of this model's
# score geometry and must be re-measured, not reasoned about, if the
# backend changes.
#
# CAP = 2, and the third slot is what the measurement rejected: rank #3
#   holds 2% (English) and 0% (Italian) of all genuinely relevant lessons,
#   so at this floor k=3 and k=2 have IDENTICAL recall (0.37) while
#   precision falls from 0.37 to 0.26 and output grows from 1.4 to 2.0
#   rows per task. The oracle ceiling confirms it: a perfect top-3 policy
#   would reach recall 0.41 against top-2's 0.40. Two is also the smallest
#   cap that still satisfies A23's original requirement -- two
#   complementary lessons CAN appear together.
# FLOOR = 0.30, and what fixes it is the median: the median score of a
#   genuinely relevant lesson is 0.327 (English) / 0.346 (Italian), so any
#   floor at or above ~0.35 sits ABOVE the median true positive and
#   discards more than half of what is relevant by construction. 0.30 is
#   the highest value that stays under that ceiling while still recovering
#   both wordings of the original A23 reproduction (whose weaker lesson
#   scores 0.3206) and still rejecting the A23.1 journey's measured false
#   positive (0.2750).
#   What it buys, on scenes where the correct answer is to emit NOTHING:
#   the 0.20 floor emitted at least one lesson in 92% of them, this one in
#   25%. That is the job an absolute floor can actually do.
#   What it cannot do is separate relevance: measured over the same
#   corpus, false positives have a HIGHER median score (0.350) than
#   genuinely relevant lessons do (0.327). The floor is a precision
#   control, never semantic understanding -- see A23.2 for the
#   rank-inversion cases it provably cannot fix.
#
# Deliberately NOT `SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor`: that
# floor belongs to the single-winner MEMORY pool, is calibrated for a
# different question, and is untouched by A23.2.
LESSON_SEMANTIC_FLOOR = 0.30
SET_ADMISSION_LIMIT = 2

# [A31.2] The SET admission policy for project-wide Invariants in the
# COMPILED CONTEXT. Same internal-constant discipline as the Lesson pair
# above: not exposed through the CLI, the public API, the manifest or an
# environment variable. Kept separate from `LESSON_SEMANTIC_FLOOR` /
# `SET_ADMISSION_LIMIT` on purpose -- reusing those would make an
# Invariant-specific contract implicit and would couple two calibrations
# that were measured independently, on different corpora, for different
# categories.
#
# CALIBRATED IN A31.1 on a prospective corpus frozen before it was scored
# (59 invariants over 22 domains grouped into 6 project pools, 44 task
# scenes across three languages, 11 of them scenes where the correct
# answer is to emit nothing, 8 held out by a rule fixed before scoring).
# A V1 operating point, a property of this model's score geometry: it
# must be re-measured, not reasoned about, if the backend changes.
#
# FLOOR = 0.35, anchored to the median with A23.2's own criterion: the
#   median score of a genuinely relevant invariant is 0.383, so any floor
#   at or above 0.40 sits above it and discards more than half of what is
#   relevant by construction. 0.35 is the highest measured value staying
#   under that ceiling, and the lowest one that preserves abstention
#   EXACTLY as the previous policy did (10 of 11 empty-set scenes correct,
#   the same single false alarm). Below it abstention collapses fast: at
#   0.30, 9 correct; at 0.20, 2.
# CAP = 2, because the second slot is the only one whose margin adds more
#   signal than noise (+11 true positives against +9 false positives);
#   the third costs 2.5 false positives per true positive and drops
#   precision under 0.5. Cap 1 is excluded for an independent reason: it
#   leaves critical false negatives at 21 of 24, identical to the
#   single-winner policy, because a critical constraint is rarely the
#   top-ranked candidate.
#
# NO MARGIN, and that is the actual defect this policy repairs. Measured
# on the same corpus, `SEMANTIC_POLICY[ENTITY_MEMORY].absolute_floor`
# never rejected a single invariant -- the rank #1 candidate always
# cleared 0.40 -- so for this pool the single-winner policy WAS the margin
# and nothing else, and the margin alone left the CONSTRAINTS section
# empty in 27 of the 33 scenes that had a legitimately relevant
# invariant. `margin_floor` asks "is #1 separated enough from #2 to be THE
# answer", which is only meaningful when at most one candidate can be
# right; 15 of 44 scenes have two or more co-relevant invariants, and
# co-relevant constraints are by definition close to each other.
#
# What this policy does NOT do, measured and deliberately not tuned away:
# it does not make the compiler a safety checklist (15 of 24 critical
# constraints are still missed; `preflight` remains the unconditional,
# complete view), and it does not solve precision -- false positives go
# from 2 to 22 while true positives go from 5 to 24. It was accepted
# because abstention does not degrade at all, silent scenes fall from 27
# to 8, and no previously admitted true positive is lost.
INVARIANT_SEMANTIC_FLOOR = 0.35
INVARIANT_ADMISSION_LIMIT = 2


class SemanticUnavailable(Exception):
    """Raised internally when the semantic channel cannot be used right
    now (extra not installed, model not set up, index missing/stale).
    Never propagates out of `_workspace.py`'s public methods -- every
    caller there catches this and degrades to lexical/FTS-only, exactly
    as if A7.4 did not exist."""


# ---------------------------------------------------------------------------
# Effective model identity
# ---------------------------------------------------------------------------


def model_identity_for(artifact: str) -> str:
    """The effective identity persisted with an index built from
    `artifact`: repository, pinned revision, and the exact artifact. See
    the module docstring for why all three belong in one string."""
    return f"{SEMANTIC_MODEL_REPO}@{SEMANTIC_MODEL_REVISION}#{artifact}"


def preferred_artifact(machine: str | None = None) -> str:
    """The artifact Cortex will try FIRST on this machine.

    Deliberately keyed on the machine ARCHITECTURE only -- never on the
    operating system, and never on individual CPU feature flags. The
    quantized artifacts are named after the instruction set their
    quantization scheme was tuned for, not one they require: ONNX
    Runtime executes them on any CPU of the same architecture, more
    slowly where the tuned instructions are absent. Probing CPU flags
    would mean either a new dependency or a Linux-only `/proc` scrape,
    and A16.2.1's conclusion was explicit -- a false optimization is
    worse than a slightly heavier setup. Anything unrecognized falls
    back to the full-precision artifact, which has no architecture
    assumptions at all, and `load_model_for_setup` falls back to it a
    second time if the preferred artifact does not actually load here."""
    detected = platform.machine() if machine is None else machine
    normalized = detected.strip().lower()
    if normalized in {"x86_64", "amd64", "x64"}:
        return ARTIFACT_X86_64
    if normalized in {"arm64", "aarch64"}:
        return ARTIFACT_ARM64
    return ARTIFACT_PORTABLE


# The identity this build of Cortex would create a NEW index with. An
# EXISTING index is always read through `artifact_for_index` instead,
# which obeys whatever artifact that index actually recorded.
SEMANTIC_MODEL_ID = model_identity_for(preferred_artifact())


def artifact_for_index(meta: "SemanticMeta") -> str | None:
    """The ONNX artifact an existing index must be queried with, or None
    if the index is not compatible with this build of Cortex at all
    (different provider, different model repository, different upstream
    revision, different normalization, or an artifact this build does not
    know how to load).

    Returning the RECORDED artifact rather than this machine's preferred
    one is what makes the A16.2.1 mixing hazard structurally impossible:
    the query is always encoded with the same artifact the stored vectors
    were, or the index is refused and Cortex degrades to lexical/FTS.
    That also means an index whose build fell back to the portable
    artifact stays usable here, instead of being permanently rejected for
    disagreeing with a machine-derived preference."""
    model_id = meta.model_id or ""
    prefix = f"{SEMANTIC_MODEL_REPO}@{SEMANTIC_MODEL_REVISION}#"
    if not model_id.startswith(prefix):
        return None
    artifact = model_id[len(prefix) :]
    if artifact not in SUPPORTED_ARTIFACTS:
        return None
    if not meta.matches(
        provider=SEMANTIC_PROVIDER,
        model_id=model_identity_for(artifact),
        normalization=SEMANTIC_NORMALIZATION,
    ):
        return None
    return artifact


# ---------------------------------------------------------------------------
# The ONNX encoder
# ---------------------------------------------------------------------------


class _OnnxTextEncoder:
    """The whole ONNX surface of Cortex, kept behind one `encode()`.

    Reproduces the model's published SentenceTransformer recipe: its own
    tokenizer, truncation at `SEMANTIC_MAX_SEQ_LENGTH`, padding with the
    tokenizer's real pad token, then mean pooling over the attention
    mask. Returns UNNORMALIZED vectors, exactly as the model itself does
    (its `modules.json` declares no `Normalize` module) -- `embed()`
    normalizes, so that contract is unchanged from A7.4."""

    def __init__(self, session: Any, tokenizer: Any, artifact: str) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {node.name for node in session.get_inputs()}
        self.identity = model_identity_for(artifact)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        chunks = [
            self._encode_batch(texts[start : start + SEMANTIC_ENCODE_BATCH_SIZE])
            for start in range(0, len(texts), SEMANTIC_ENCODE_BATCH_SIZE)
        ]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, np.ndarray] = {"input_ids": input_ids}
        # Which inputs a given artifact declares is read off the real
        # session rather than assumed -- not every export of the same
        # model exposes the same set.
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        last_hidden_state = self._session.run(None, feed)[0]
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (last_hidden_state * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return np.asarray(pooled, dtype=np.float32)


_model_cache: dict[str, _OnnxTextEncoder] = {}


def _artifact_path(artifact: str, *, local_files_only: bool) -> str:
    return huggingface_hub.hf_hub_download(
        SEMANTIC_MODEL_REPO,
        artifact,
        revision=SEMANTIC_MODEL_REVISION,
        local_files_only=local_files_only,
    )


def _build_encoder(artifact: str, *, local_files_only: bool) -> _OnnxTextEncoder:
    """Build (or return the process-cached) encoder for `artifact`.
    Loading an ONNX session is the single most expensive step in the
    whole semantic path -- seconds, against milliseconds for a query --
    so it is cached per artifact for the life of the process, exactly as
    the model2vec backend was before A16.3."""
    cached = _model_cache.get(artifact)
    if cached is not None:
        return cached
    onnx_path = _artifact_path(artifact, local_files_only=local_files_only)
    tokenizer_path = _artifact_path(TOKENIZER_FILENAME, local_files_only=local_files_only)
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=SEMANTIC_MAX_SEQ_LENGTH)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise SemanticUnavailable(
            f"Semantic tokenizer for {SEMANTIC_MODEL_REPO} has no '<pad>' token; "
            "the cached tokenizer file is not the expected one"
        )
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
    encoder = _OnnxTextEncoder(session, tokenizer, artifact)
    _model_cache[artifact] = encoder
    return encoder


def load_model_for_setup() -> _OnnxTextEncoder:
    """Load the semantic model, downloading the pinned artifacts if they
    are not already cached. Only ever called from `cortex semantic setup`
    -- the one entry point allowed to touch the network.

    Falls back ONCE, from this machine's preferred artifact to the
    full-precision portable one, if the preferred artifact cannot
    actually be fetched or loaded here. There is deliberately no third
    attempt: the identity of whatever artifact succeeded is what gets
    recorded with the index, so a fallback is fully visible downstream
    and can never be silently mixed with a different artifact's vectors."""
    artifact = preferred_artifact()
    try:
        return _build_encoder(artifact, local_files_only=False)
    except Exception as exc:
        if artifact == ARTIFACT_PORTABLE:
            raise SemanticUnavailable(
                f"Failed to load semantic model artifact {artifact!r} from "
                f"{SEMANTIC_MODEL_REPO}: {exc}"
            ) from exc
        try:
            return _build_encoder(ARTIFACT_PORTABLE, local_files_only=False)
        except Exception as fallback_exc:
            raise SemanticUnavailable(
                f"Failed to load semantic model artifacts {artifact!r} and "
                f"{ARTIFACT_PORTABLE!r} from {SEMANTIC_MODEL_REPO}: {fallback_exc}"
            ) from fallback_exc


def load_model_for_index(meta: "SemanticMeta") -> _OnnxTextEncoder | None:
    """The model that must answer queries against `meta`'s index, or None
    if this build cannot read that index at all.

    This is the whole of what `_workspace.py` needs to know: it never
    sees an artifact filename, a cache path, a tokenizer or an ONNX
    session. Raises (like `load_model_for_retrieval`) if the recorded
    artifact is not in the local cache -- callers already treat that as
    `SemanticUnavailable` and degrade to lexical/FTS."""
    artifact = artifact_for_index(meta)
    if artifact is None:
        return None
    return load_model_for_retrieval(artifact)


def artifacts_available(meta: "SemanticMeta") -> bool:
    """[A27] Whether the artifacts an existing index needs are already in
    the local cache -- WITHOUT loading them.

    This resolves cache paths (`local_files_only=True`, so it can never
    reach the network) and stops there: no `InferenceSession`, no
    tokenizer, none of the seconds `_build_encoder` costs. That is what
    lets `cortex status` distinguish "this index is fine" from "this
    index cannot be queried on this machine" while staying an
    observation, and lets `preflight()` report the same condition without
    paying for a model it is not going to use.

    Returns False for an index this build cannot read at all
    (`artifact_for_index` says so), which keeps model COMPATIBILITY
    ahead of model AVAILABILITY in exactly one place."""
    artifact = artifact_for_index(meta)
    if artifact is None:
        return False
    try:
        _artifact_path(artifact, local_files_only=True)
        _artifact_path(TOKENIZER_FILENAME, local_files_only=True)
    except Exception:
        return False
    return True


def load_model_for_retrieval(artifact: str = ARTIFACT_PORTABLE) -> _OnnxTextEncoder:
    """Load `artifact` for a normal retrieval call, from local cache
    only. Never touches the network (see module docstring): raises if the
    artifact is not already cached, which callers must treat as
    `SemanticUnavailable` and degrade from, never as a crash, and never
    as a reason to go and fetch it."""
    return _build_encoder(artifact, local_files_only=True)


def model_identity(model: Any) -> str:
    """The effective identity of the artifact `model` was actually loaded
    from -- which is not necessarily this machine's preferred artifact,
    because `load_model_for_setup` may have fallen back."""
    return getattr(model, "identity", SEMANTIC_MODEL_ID)


def model_dimensions(model: Any) -> int:
    """Read the real output dimensionality off the loaded model instead
    of trusting a constant -- the one number an artifact swap could
    change without anything else noticing."""
    return int(model.encode(["_cortex_semantic_dimension_probe_"]).shape[1])


def resolve_local_revision() -> str:
    """The upstream revision the index was built from. Pinned (see
    `SEMANTIC_MODEL_REVISION`), so there is nothing to discover at
    runtime: every artifact is fetched at exactly this revision, which is
    what makes this value true rather than merely reported."""
    return SEMANTIC_MODEL_REVISION


def embed(model: Any, texts: Sequence[str]) -> np.ndarray:
    """Embed `texts` and L2-normalize each row so cosine similarity
    reduces to a plain dot product. The backend model does not normalize
    its own output (its `modules.json` declares no `Normalize` module),
    and normalization is applied explicitly here rather than assumed, so
    this module's correctness never depends on that being true of
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
    model: Any,
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
    model: Any,
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


def set_admitted_ids(
    ranked: list[tuple[str, float]], *, floor: float, limit: int = SET_ADMISSION_LIMIT
) -> frozenset[str]:
    """[A23.1/A23.2] SET admission: every candidate clearing `floor`,
    capped at `limit`, with NO margin check. The alternative policy to
    `semantic_admitted_id` below, for pools whose consumer wants "which
    candidates are relevant enough to include" rather than "which single
    candidate is the intended one".

    Takes its `floor` explicitly rather than reading
    `SEMANTIC_POLICY[entity_type]`: A23.2 calibrated a Lesson-specific
    floor, and having this function reach into the MEMORY pool's policy
    would silently couple two calibrations that answer different
    questions. What is shared with `semantic_admitted_id` is the ranking
    and the model -- nothing else. The difference in what a pool may
    return is a property of the CONSUMING category, not of the
    similarity engine (see `Cortex._preflight_lesson_semantic_admitted`
    for the one caller, and A23 for why Lesson is such a category).

    Why the margin is deliberately absent here rather than merely
    relaxed: `margin_floor` asks "is #1 separated enough from #2 to be
    trusted as THE answer", which is only a meaningful question when at
    most one candidate can be right. A23 reproduced, on the real model,
    two complementary verified lessons scoring 0.3782 and 0.3206 against
    the same task -- both far above the 0.20 floor, both genuinely useful,
    and rejected TOGETHER because they were 0.0576 apart. Requiring
    co-relevant candidates to defeat each other is what that policy does;
    for a set-valued category it is the wrong question, not a
    mis-tuned answer to the right one. The margin is untouched for every
    pool that still asks the single-winner question.

    `limit` bounds context growth and, together with `floor`, is what
    replaces the margin's incidental precision work -- both calibrated
    together in A23.2, because neither is defensible alone: a cap without
    a floor never abstains, and a floor high enough to abstain reliably
    sits above the median relevant lesson. A `limit` of 0 or less admits
    nothing.

    Determinism: the returned SET is a pure function of `ranked`, which
    the caller obtained from `rank_candidates` -- unchanged by A23.1.
    This function never re-sorts and never breaks ties itself, so which
    candidates fall inside the cap is exactly as deterministic as the
    ranking that produced them. The ORDER experience is reported in is
    not decided here at all: `build_preflight` filters its own
    event-log-ordered lists by id membership, so a preflight result's
    ordering is unaffected by this function's existence.
    """
    if limit <= 0:
        return frozenset()
    admitted = [entity_id for entity_id, score in ranked if score >= floor]
    return frozenset(admitted[:limit])


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
