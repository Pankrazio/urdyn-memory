"""Optional semantic retrieval channel: local ONNX sentence embeddings
plus brute-force cosine similarity, used to widen candidate recall in
`preflight()`/`guard()` the same way `_retrieval.py`'s FTS5/BM25 channel
already does.

This module is the ONLY place in Urdyn that imports `onnxruntime`,
`tokenizers`, `huggingface_hub` or `numpy`. It is never imported from
`__init__.py`, `_workspace.py` at module scope, or any other module
reachable from a plain `import urdyn` -- callers
(`_workspace.py`, `_cli.py`) import it lazily, inside functions, wrapped
in `try/except ImportError`, so a base install without the
`urdyn-memory[semantic]` extra never even attempts to load them and
behaves exactly as it did before A7.4. Nothing outside this module ever
sees an `InferenceSession`, a `Tokenizer`, an artifact filename or a
Hugging Face cache path: the rest of Urdyn only ever gets
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
Urdyn owns this small encoder instead of taking a heavier third-party
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
from the repository at runtime ON PURPOSE: it is part of Urdyn's
embedding contract, and a value silently changed upstream must not be
able to silently change the vector space of an already-indexed
workspace. The same reasoning pins the model revision (see
`SEMANTIC_MODEL_REVISION`). Values that CAN drift without a contract
change (the padding token id, the output dimensionality) are read from
the real artifacts at setup instead of trusted.

Two loading entry points exist on purpose:
  - `load_model_for_setup` allows a network download if the artifact is
    not yet cached (used only by `urdyn semantic setup`).
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
skill / source) -- see `SEMANTIC_POLICY` below (source's SET-
admission floor/cap live separately, next to `LESSON_SEMANTIC_FLOOR`,
and are explicitly NOT independently calibrated -- see their own
comment). A16.2.1 re-validated the MEMORY
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

import contextvars
import dataclasses
import functools
import importlib.util
import platform
import re
import threading
import time
from contextlib import contextmanager
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._semantic_store import SemanticMeta
    from .semantic_backend import SemanticBackend, SemanticBackendIdentity

SEMANTIC_PROVIDER = "onnxruntime"

# VERSIONED NORMALIZATION IDENTITY. Persisted with every index
# (`semantic_meta.normalization`) and compared by `artifact_for_index`
# (built-in ONNX encoder) and by the external-backend identity check.
# Urdyn <= 0.4.0 -- and earlier unreleased development builds -- wrote "l2" with a float32
# normalization that turned finite-but-large rows into zero vectors and let
# NaN/Inf/zero rows reach the index (see `normalize_rows`). Vectors produced
# under the new contract are not bit-identical to the old ones and an old
# index may hold degenerate rows, so the two are NOT declared compatible: an
# "l2" index is refused here (rebuilt explicitly by `urdyn semantic setup`,
# canonical memory untouched), and a 0.4.0 or earlier development client -- which
# requires the exact string "l2" -- refuses an index written with this
# identity. Defined in `_semantic_identity` (no third-party
# imports) so the public `SemanticBackendIdentity` default is the same object.
from ._semantic_identity import LEGACY_SEMANTIC_NORMALIZATION, SEMANTIC_NORMALIZATION  # noqa: E402
from ._encoding_plan import PreparedEncoder  # noqa: E402  
from . import _representation  # noqa: E402  
from ._representation import IndexReference, InvalidRepresentation, Representation  # noqa: E402
from ._semantic_store import DETAIL_REPRESENTATION_MISMATCH, DETAIL_REPRESENTATION_UNVERIFIED  # noqa: E402

# Accepted deviation of a normalized float32 row's Euclidean norm
# from 1. Derivation: the float64 result y is unit up to ~d * 2**-53
# (< 5e-14 for d <= 384); rounding each component to float32 perturbs it
# by |delta_i| <= 2**-24 |y_i|, hence ||delta|| <= 2**-24 ~ 5.96e-8 and
# | ||y32|| - 1 | <= ~6e-8. The check itself is computed in float64.
# 1e-6 leaves a ~16x margin for that bound while every degenerate row
# (zero, NaN, Inf, unnormalized) misses it by orders of magnitude.
UNIT_NORM_TOLERANCE = 1e-6
# A sanity bound on row width, far above any supported model (384).
MAX_SEMANTIC_DIMENSIONS = 65_536


def default_runtime_available() -> bool:
    """Check the Python ONNX extra without loading a session or network."""
    try:
        return all(importlib.util.find_spec(name) is not None for name in (
            "onnxruntime", "tokenizers", "huggingface_hub"
        ))
    except (ImportError, ValueError):
        return False

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

# Official artifacts published by the model repository itself. Urdyn
# picks one internally; there is deliberately no user-facing choice.
ARTIFACT_PORTABLE = "onnx/model.onnx"
ARTIFACT_X86_64 = "onnx/model_quint8_avx2.onnx"
ARTIFACT_ARM64 = "onnx/model_qint8_arm64.onnx"
ARTIFACT_ARM64_SHA256 = "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672"
TOKENIZER_SHA256 = "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"
SUPPORTED_ARTIFACTS = frozenset({ARTIFACT_PORTABLE, ARTIFACT_X86_64, ARTIFACT_ARM64})

ENTITY_ATTEMPT = "attempt"
ENTITY_MEMORY = "memory"
ENTITY_SKILL = "skill"
ENTITY_SOURCE = "source"


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
# property of the MODEL, not of Urdyn, so a backend change invalidates
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

# The set admission policy for `context()`'s PROJECT EVIDENCE pool
# (current-observation document Evidence from a seeded Source). Same
# shape as `LESSON_SEMANTIC_FLOOR`/`SET_ADMISSION_LIMIT` and
# `INVARIANT_SEMANTIC_FLOOR`/`INVARIANT_ADMISSION_LIMIT` above, and for
# the same reason: several seeded documents can each cover a distinct,
# co-relevant facet of one task (e.g. one doc on dependency direction,
# another on component orthogonality), so this is a set-valued pool, not
# a single-winner contest.
#
# NOT INDEPENDENTLY CALIBRATED. Every other floor/cap pair in this
# module is the output of a frozen, scored evaluation corpus (A16.2,
# A16.3, A23.1/A23.2, A31.1) -- no such corpus exists yet for seeded
# document text, whose length and register differ from every other pool
# (a Memory/Lesson is a short, single-claim sentence; a seeded document
# is arbitrary prose, often a whole file). Inventing a number here would
# be exactly the "designing on assumptions" this project's own
# calibration discipline exists to prevent. `LESSON_SEMANTIC_FLOOR`/
# `SET_ADMISSION_LIMIT` are reused verbatim instead, as the closest
# available analog (free-form prose, admitted as a bounded set, not a
# single winner) -- a conscious placeholder, not a measurement, and
# tracked as a MUST follow-up: this pool needs its own frozen,
# pre-scored corpus of seeded documents before its operating point can
# be trusted the way the others are.
EVIDENCE_SEMANTIC_FLOOR = LESSON_SEMANTIC_FLOOR
EVIDENCE_ADMISSION_LIMIT = SET_ADMISSION_LIMIT


class SemanticUnavailable(Exception):
    """Raised internally when the semantic channel cannot be used right
    now (extra not installed, model not set up, index missing/stale).
    Never propagates out of `_workspace.py`'s public methods -- every
    caller there catches this and degrades to lexical/FTS-only, exactly
    as if A7.4 did not exist."""


class InvalidSemanticVectors(SemanticUnavailable):
    """Encoder output or stored vectors violate the numeric
    contract (shape, cardinality, dimensions, dtype, finiteness, non-zero
    rows, unit norm). Never repaired: the semantic channel is refused for
    that operation, exactly like any other `SemanticUnavailable`."""


class InvalidStoredVectors(InvalidSemanticVectors):
    """The `InvalidSemanticVectors` raised for bytes READ BACK from
    the derived index (wrong blob length, non-finite or non-unit row), as
    opposed to encoder output (a query or a row about to be written). Lets
    the per-call outcome say "a stored vector failed validation" only when
    that is what happened -- the remedy (rebuild) differs."""


class RepresentationUnverified(SemanticUnavailable):
    """The index names no verifiable representation: built before
    representation references existed (no reference), or its reference is malformed or of an unknown
    schema or assurance level. Never relabelled: rebuilt."""


class RepresentationMismatch(SemanticUnavailable):
    """The encoder at hand does not produce the representation the
    index was built with (different fingerprint). Its vectors are never
    compared with, nor added to, that index."""


class EncoderRunExpired(SemanticUnavailable):
    """An encode with an experimental deadline did not finish in
    time; its output, if any, is discarded. Two separate facts:
    `terminate_requested` -- the termination flag reached an ONNX `Run`
    that was executing (the only case in which the session is retired);
    `run_aborted` -- ONNX Runtime actually aborted that `Run` (it raised
    after the flag). A request is not a termination: a Run may still
    return normally, or never. Internal: callers see the usual per-call semantic outcome."""

    def __init__(self, message: str, *, terminate_requested: bool, run_aborted: bool = False) -> None:
        super().__init__(message)
        self.terminate_requested = terminate_requested
        self.run_aborted = run_aborted


class EncoderRetired(SemanticUnavailable):
    """This encoder instance was retired after a cancelled `Run`
    and accepts no new encode, even through a handle obtained earlier."""


class EncoderUnavailable(SemanticUnavailable):
    """No encoder can be handed out for this artifact right now: a
    retired one still has encodes running, or another construction did not
    finish within the bounded wait."""


# ---------------------------------------------------------------------------
# Effective model identity
# ---------------------------------------------------------------------------


def model_identity_for(artifact: str) -> str:
    """The effective identity persisted with an index built from
    `artifact`: repository, pinned revision, and the exact artifact. See
    the module docstring for why all three belong in one string."""
    return f"{SEMANTIC_MODEL_REPO}@{SEMANTIC_MODEL_REVISION}#{artifact}"


def preferred_artifact(machine: str | None = None) -> str:
    """The artifact Urdyn will try FIRST on this machine.

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


# The identity this build of Urdyn would create a NEW index with. An
# EXISTING index is always read through `artifact_for_index` instead,
# which obeys whatever artifact that index actually recorded.
SEMANTIC_MODEL_ID = model_identity_for(preferred_artifact())


def validate_external_identity(identity: "SemanticBackendIdentity") -> None:
    """Keep the existing calibrated policy tied to its known vector space.

    This API accepts different inference providers, not arbitrary models.
    A new model needs independently calibrated admission thresholds.
    """
    from .semantic_backend import SemanticBackendIdentity

    if not isinstance(identity, SemanticBackendIdentity):
        raise TypeError("backend.identity must be a SemanticBackendIdentity")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", identity.provider) or identity.provider == SEMANTIC_PROVIDER:
        raise ValueError("external backend needs a distinct, stable provider identity")
    if identity.model_id not in {model_identity_for(artifact) for artifact in SUPPORTED_ARTIFACTS}:
        raise ValueError("external backend must use a calibrated MiniLM artifact identity")
    if identity.model_revision != SEMANTIC_MODEL_REVISION:
        raise ValueError("external backend revision does not match the calibrated model")
    if identity.normalization == LEGACY_SEMANTIC_NORMALIZATION:
        # Urdyn, not the backend, normalizes; the field names the
        # engine contract the index is built under. A client pinned to the
        # legacy float32 "l2" contract is refused rather than relabelled.
        raise ValueError(
            f"external backend declares the legacy {LEGACY_SEMANTIC_NORMALIZATION!r} normalization; "
            f"this engine normalizes under {SEMANTIC_NORMALIZATION!r} (use the default, then rebuild the index)"
        )
    if identity.normalization != SEMANTIC_NORMALIZATION or identity.dimensions != 384:
        raise ValueError(
            f"external MiniLM backend requires 384 dimensions and {SEMANTIC_NORMALIZATION!r} normalization"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", identity.artifact_sha256):
        raise ValueError("external backend needs an artifact SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", identity.tokenizer_sha256):
        raise ValueError("external backend needs a tokenizer SHA-256 digest")
    if identity.model_id == model_identity_for(ARTIFACT_ARM64) and identity.artifact_sha256 != ARTIFACT_ARM64_SHA256:
        raise ValueError("external ARM64 artifact SHA-256 does not match the pinned MiniLM artifact")
    if identity.tokenizer_sha256 != TOKENIZER_SHA256:
        raise ValueError("external tokenizer SHA-256 does not match the pinned MiniLM tokenizer")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", identity.encoding_policy):
        raise ValueError("external backend needs a stable encoding policy identity")


# ---------------------------------------------------------------------------
# Representation identity (see `_representation`)
# ---------------------------------------------------------------------------

# What the core feeds the encoder, as defined by `_workspace._semantic_pool_entries`
# (record texts) and `_chunk.chunk_evidence` (seeded-document chunks). A code
# constant, declared by the core: change it with that composition.
SEMANTIC_TEXT_POLICY = "urdyn-pool-text-v1"
ONNX_PROVIDERS = ("CPUExecutionProvider",)
ONNX_POOLING = "mean-attention-mask-f32-v1"
ONNX_TOKEN_TYPE_IDS = "zeros"
ONNX_BATCH_POLICY = "fixed-chunks-in-call-order"
# The experimental per-text recipe: ONE text per ONNX Run, so no
# companion and no padding position ever enters the Run's dynamic
# quantization. Off unless
# `_experimental_per_text_encoding()` is active in the current context.
ONNX_PER_TEXT_POLICY = "per-text"
_ENCODING_POLICY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "urdyn_experimental_encoding_policy", default=None
)


@contextmanager
def _experimental_per_text_encoding():
    """Private harness, not an API: every built-in encode of this
    context -- setup, refresh and query alike -- uses the per-text recipe,
    under its own representation identity. Outside it a per-text index is
    refused, and inside it a batch32 index is refused: the two are never
    compatible by sharing a model, a tokenizer or a width. External backends
    are not affected. A ContextVar: a new thread does not inherit it."""
    token = _ENCODING_POLICY.set(ONNX_PER_TEXT_POLICY)
    try:
        yield
    finally:
        _ENCODING_POLICY.reset(token)


def _batch32_section() -> dict:
    return {"policy": ONNX_BATCH_POLICY, "size": SEMANTIC_ENCODE_BATCH_SIZE, "composition_independent": False}


def _per_text_section() -> dict:
    """Per-text's `composition_independent` is STRUCTURAL -- the recipe gives
    a Run nothing but the one text, checked before every Run -- not a claim
    that results are numerically repeatable, which is observed evidence."""
    return {"policy": ONNX_PER_TEXT_POLICY, "size": 1, "composition_independent": True}


def _active_batch_section() -> dict:
    """The `recipe.batch` of the policy active in this context."""
    return _per_text_section() if _ENCODING_POLICY.get() == ONNX_PER_TEXT_POLICY else _batch32_section()
_EXPECTED_TRUNCATION = {"max_length": SEMANTIC_MAX_SEQ_LENGTH, "stride": 0, "strategy": "longest_first",
                        "direction": "right"}
_PAD_TOKEN = "<pad>"


def _core_section() -> dict:
    from ._chunk import DEFAULT_CHUNK_MAX_CHARS

    return _representation.core_section(text_policy=SEMANTIC_TEXT_POLICY, chunk_max_chars=DEFAULT_CHUNK_MAX_CHARS,
                                        normalization=SEMANTIC_NORMALIZATION)


@functools.lru_cache(maxsize=1)
def _observed_onnx_runtime() -> tuple:
    import onnxruntime
    import tokenizers

    build = getattr(onnxruntime, "get_build_info", None)
    return (onnxruntime.__version__, build() if callable(build) else _representation.UNOBSERVED,
            tokenizers.__version__)


def _onnx_runtime_section() -> dict:
    version, build, tokenizers_version = _observed_onnx_runtime()
    return {"name": SEMANTIC_PROVIDER, "version": version, "build": build, "providers": list(ONNX_PROVIDERS),
            "session_options": "default", "tokenizers": tokenizers_version}


def _onnx_encoder_section(artifact: str, *, model_sha256: str, tokenizer_sha256: str, inputs: list[str],
                          dimensions: int, batch: dict | None = None) -> dict:
    """The built-in encoder's recipe, as `_construct_encoder` applies (and
    reads back) and `_encode_batch` executes it. Batching is recorded as it
    is: fixed chunks of `SEMANTIC_ENCODE_BATCH_SIZE` with padding to the
    longest text of the chunk, which is NOT independent of the batch's
    composition . Identity here means the same recipe, not the same
    vector for the same text in another batch.

    `batch` defaults to the active policy's (`_active_batch_section`,
    what the static check expects in this context); the default policy's is
    byte-for-byte the original batch section, so existing batch32 fingerprints are
    unchanged. A constructed encoder is always described with batch32
    (`onnx_representation`): it is cached across contexts, and per-text is
    only ever a view derived from it (`_PerTextEncoder`)."""
    return {
        "kind": _representation.ONNX_ENCODER_KIND,
        "assets": {"repository": SEMANTIC_MODEL_REPO, "revision": SEMANTIC_MODEL_REVISION, "artifact": artifact,
                   "model_sha256": model_sha256, "tokenizer_sha256": tokenizer_sha256},
        "recipe": {
            "truncation": {"max_length": SEMANTIC_MAX_SEQ_LENGTH, "strategy": _EXPECTED_TRUNCATION["strategy"],
                           "direction": _EXPECTED_TRUNCATION["direction"]},
            "padding": {"strategy": "batch-longest", "token": _PAD_TOKEN, "direction": "right"},
            "batch": _active_batch_section() if batch is None else batch,
            "inputs": sorted(inputs),
            "token_type_ids": ONNX_TOKEN_TYPE_IDS,
            "pooling": ONNX_POOLING,
            "dimensions": dimensions,
        },
        "runtime": _onnx_runtime_section(),
    }


def onnx_representation(artifact: str, *, model_sha256: str, tokenizer_sha256: str, inputs: list[str],
                        dimensions: int) -> Representation:
    return _representation.describe(
        _representation.CORE_VERIFIED,
        _onnx_encoder_section(artifact, model_sha256=model_sha256, tokenizer_sha256=tokenizer_sha256,
                              inputs=inputs, dimensions=dimensions, batch=_batch32_section()),
        _core_section(),
    )


def declared_representation(identity: "SemanticBackendIdentity") -> Representation:
    """An external backend's representation: its declaration, as declared.
    Built by the core from the validated identity -- the backend object
    supplies no descriptor of its own and cannot claim another level."""
    return _representation.describe(
        _representation.BACKEND_DECLARED,
        {"kind": _representation.EXTERNAL_ENCODER_KIND,
         "declared": {"provider": identity.provider, "model_id": identity.model_id,
                      "model_revision": identity.model_revision, "dimensions": identity.dimensions,
                      "artifact_sha256": identity.artifact_sha256, "tokenizer_sha256": identity.tokenizer_sha256,
                      "encoding_policy": identity.encoding_policy, "normalization": identity.normalization}},
        _core_section(),
    )


def representation_of(model: Any) -> Representation:
    """The representation `model` produces, as far as the core knows it.
    Only an encoder `_construct_encoder` built (and verified) is
    core-verified; an `ExternalEncoder` is backend-declared; anything else,
    whatever attributes it carries, is unattested."""
    if isinstance(model, _OnnxTextEncoder):
        verified = model.__dict__.get("_representation")
        if isinstance(verified, Representation):
            return verified
    if isinstance(model, ExternalEncoder):
        return model.representation
    if isinstance(model, _PerTextEncoder):
        return model.representation
    # Nothing about such an encoder is known -- not even what it claims to be
    # (attributes it carries are not evidence): one class, no identity.
    return _representation.describe(
        _representation.UNATTESTED, {"kind": _representation.UNATTESTED_ENCODER_KIND}, _core_section())


def index_representation(meta: "SemanticMeta") -> IndexReference:
    """The representation `meta`'s index names, or `RepresentationUnverified`.
    Never inferred from the other columns."""
    try:
        _, reference = _representation.read_provider(meta.provider or "")
    except InvalidRepresentation as exc:
        raise RepresentationUnverified(f"the index representation is not verifiable: {exc}") from None
    if reference is None:
        raise RepresentationUnverified("the index was built before representation identities existed")
    return reference


def bind_representation(model: Any, meta: "SemanticMeta") -> Any:
    """`model`, if it produces exactly the representation of `meta`'s index;
    otherwise `RepresentationMismatch` / `RepresentationUnverified`. Called
    before any of its vectors is compared with the index or written to it."""
    reference = index_representation(meta)
    actual = representation_of(model)
    if (actual.assurance, actual.fingerprint) != (reference.assurance, reference.fingerprint):
        raise RepresentationMismatch("the encoder does not produce the representation of this index")
    return model


def representation_problem(meta: "SemanticMeta", backend_identity: "SemanticBackendIdentity | None" = None) -> str | None:
    """What `semantic_state` can say about the index's representation
    without loading a model: None (compatible as far as can be told now), or
    the `DETAIL_*` that makes it unusable. For an external backend the whole
    (declared) representation is known, so this is the full check. For the
    built-in encoder the static fingerprint is compared here and the full one
    when the encoder is bound (`bind_representation`); an unattested index is
    decided there too."""
    try:
        reference = index_representation(meta)
    except RepresentationUnverified:
        return DETAIL_REPRESENTATION_UNVERIFIED
    if backend_identity is not None:
        expected = declared_representation(backend_identity)
        same = (reference.assurance, reference.fingerprint) == (expected.assurance, expected.fingerprint)
        return None if same else DETAIL_REPRESENTATION_MISMATCH
    if reference.assurance == _representation.UNATTESTED:
        return None
    if reference.assurance != _representation.CORE_VERIFIED:
        return DETAIL_REPRESENTATION_MISMATCH
    artifact = artifact_for_index(meta)
    if artifact is None:
        return DETAIL_REPRESENTATION_MISMATCH
    try:
        expected_static = _expected_static_fingerprint(artifact)
    except Exception:
        return DETAIL_REPRESENTATION_MISMATCH  # this process cannot even describe its runtime
    return None if expected_static == reference.static_fingerprint else DETAIL_REPRESENTATION_MISMATCH


_static_memo: dict[tuple, str] = {}


def _expected_static_fingerprint(artifact: str) -> str:
    """The static fingerprint this process would build `artifact`'s encoder
    with. Memoized on EVERY value it is derived from (a changed recipe
    constant, runtime or environment is a different key), because the
    retrieval path asks once per pool and per plan."""
    from ._chunk import DEFAULT_CHUNK_MAX_CHARS

    key = (artifact, SEMANTIC_MODEL_REPO, SEMANTIC_MODEL_REVISION, SEMANTIC_MAX_SEQ_LENGTH, SEMANTIC_ENCODE_BATCH_SIZE,
           SEMANTIC_NORMALIZATION, SEMANTIC_TEXT_POLICY, DEFAULT_CHUNK_MAX_CHARS, ONNX_PROVIDERS, ONNX_POOLING,
           ONNX_TOKEN_TYPE_IDS, ONNX_BATCH_POLICY, _PAD_TOKEN, tuple(sorted(_EXPECTED_TRUNCATION.items())),
           _observed_onnx_runtime(), _representation._observed_environment(),
           tuple(sorted(_active_batch_section().items())))
    cached = _static_memo.get(key)
    if cached is None:
        cached = _static_memo[key] = _representation.static_fingerprint_of({
            "schema": _representation.SCHEMA, "assurance": _representation.CORE_VERIFIED,
            "encoder": _onnx_encoder_section(artifact, model_sha256="", tokenizer_sha256="", inputs=[], dimensions=1),
            "core": _core_section(),
        })
    return cached


class ExternalEncoder:
    """Bind client inference to the identity it was configured with.

    The numeric checks this class used to run (float32 cast, shape,
    finiteness, zero rows) are now the single contract in `normalize_rows`,
    which `embed` applies to EVERY encoder: the raw client output is handed
    over unconverted, so its dtype is checked too (bool/complex/object are
    refused instead of being cast), and `output_dimensions` makes `embed`
    enforce the declared width on every call, the dimension probe included.
    Only the identity check stays here: it is not numeric."""

    def __init__(self, backend: "SemanticBackend", identity: "SemanticBackendIdentity") -> None:
        self._backend = backend
        self._identity = identity
        self.output_dimensions = identity.dimensions
        # Derived by the core from the configured identity.
        self.representation = declared_representation(identity)

    def encode(self, texts: Sequence[str]) -> Any:
        if self._backend.identity != self._identity:
            raise ValueError("semantic backend identity changed during use")
        return self._backend.encode(list(texts))


def artifact_for_index(meta: "SemanticMeta") -> str | None:
    """The ONNX artifact an existing index must be queried with, or None
    if the index is not compatible with this build of Urdyn at all
    (different provider, different model repository, different upstream
    revision, different normalization, or an artifact this build does not
    know how to load).

    Returning the RECORDED artifact rather than this machine's preferred
    one is what makes the A16.2.1 mixing hazard structurally impossible:
    the query is always encoded with the same artifact the stored vectors
    were, or the index is refused and Urdyn degrades to lexical/FTS.
    That also means an index whose build fell back to the portable
    artifact stays usable here, instead of being permanently rejected for
    disagreeing with a machine-derived preference."""
    # Artifact routing reads the base provider; whether the index's
    # representation can be produced here is `bind_representation`'s question.
    meta = dataclasses.replace(meta, provider=_representation.base_provider(meta.provider or ""))
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


# ---------------------------------------------------------------------------
# Experimental cooperative cancellation of ONNX `Run`
# ---------------------------------------------------------------------------
#
# Disabled unless `_experimental_encode_deadline(seconds)` is active in the
# CURRENT context (a ContextVar: a new thread does not inherit it). With it,
# one encode -- all of its chunks -- has one monotonic budget and one private
# `onnxruntime.RunOptions`; a temporary timer sets `terminate` at the
# deadline. ONNX Runtime aborts `Run` at its next check; the deadline does NOT
# bound tokenization, session creation, a node that ignores the flag, or the
# caller's total wait. An output that exists after the deadline is discarded.
#
# Request states: pending -> success | expired (one terminal transition,
# under the request's own lock). Run states: not started / active /
# termination requested / returned. Session states: usable / retired with
# encodes running / retired and drained. None of these imply another.
#
# Lock order (never held across Run, model load or timer join):
#   _model_cache_lock  ->  encoder lease lock          (retire)
#   encoder lease lock, released, then _model_cache_lock (last release)
#   request lock: alone, never while taking another lock.

_ENCODE_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "urdyn_experimental_encode_deadline", default=None
)
_ACTIVE_REQUEST: contextvars.ContextVar["_EncodeRequest | None"] = contextvars.ContextVar(
    "urdyn_active_encode_request", default=None
)
_monotonic = time.monotonic  # patched only by tests that drive the clock
ENCODER_BUILD_WAIT_SECONDS = 60.0


@contextmanager
def _experimental_encode_deadline(seconds: float):
    """Private harness: give every encode of this context a budget
    of `seconds` (immutable float). Not an API; off by default."""
    seconds = float(seconds)
    if not seconds > 0:
        raise ValueError("the experimental encode budget must be positive")
    token = _ENCODE_DEADLINE.set(seconds)
    try:
        yield
    finally:
        _ENCODE_DEADLINE.reset(token)


def _deadline_timer(delay: float, callback):
    """The temporary timer of ONE encode (tests may replace it)."""
    timer = threading.Timer(delay, callback)
    timer.daemon = True
    return timer


class _EncodeRequest:
    """State of one encode with a deadline. Every transition is taken under
    `_lock`; exactly one terminal state is ever reached."""

    def __init__(self, deadline: float, on_terminate=None) -> None:
        import onnxruntime

        self.deadline = deadline
        self._on_terminate = on_terminate  # retires the session; called WITHOUT this lock
        self.options = onnxruntime.RunOptions()  # private to this encode, never reset
        self._lock = threading.Lock()
        self.state = "pending"
        self._run_active = False
        self.terminate_reached_a_run = False

    def _expire_locked(self) -> None:
        if self.state == "pending":
            self.state = "expired"

    def on_deadline(self) -> None:
        """Timer callback: short; no wait, no model, no Run. Requests
        termination of the active Run, then retires the session logically
        (after releasing the request lock) -- also when that Run never
        returns, so no new encode can pile up on it."""
        with self._lock:
            if self.state != "pending":
                return  # a success already linearized: nothing to cancel
            self.state = "expired"
            if not self._run_active:
                return  # expired between Runs: the next check refuses; no Run was cancelled
            self.options.terminate = True
            self.terminate_reached_a_run = True
        if self._on_terminate is not None:
            self._on_terminate()

    def check(self, where: str) -> None:
        with self._lock:
            if self.state == "pending" and _monotonic() >= self.deadline:
                self.state = "expired"
            if self.state != "pending":
                raise EncoderRunExpired(f"encode deadline passed before {where}",
                                        terminate_requested=self.terminate_reached_a_run)

    def begin_run(self) -> None:
        with self._lock:
            if self.state == "pending" and _monotonic() >= self.deadline:
                self.state = "expired"
            if self.state != "pending":
                raise EncoderRunExpired("encode deadline passed before Run",
                                        terminate_requested=self.terminate_reached_a_run)
            self._run_active = True

    def end_run(self) -> None:
        with self._lock:
            self._run_active = False

    def finish(self) -> None:
        """Linearize success -- only if still pending AND within the deadline."""
        with self._lock:
            if self.state == "pending" and _monotonic() < self.deadline:
                self.state = "success"
                return
            self._expire_locked()
            cancelled = self.terminate_reached_a_run
        raise EncoderRunExpired("encode finished after its deadline; output discarded", terminate_requested=cancelled)


class _Lease:
    """Active encodes and retirement of ONE encoder instance."""

    __slots__ = ("lock", "active", "retired")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.retired = False


_model_cache_lock = threading.Lock()


class _OnnxTextEncoder:
    """The whole ONNX surface of Urdyn, kept behind one `encode()`.

    Reproduces the model's published SentenceTransformer recipe: its own
    tokenizer, truncation at `SEMANTIC_MAX_SEQ_LENGTH`, padding with the
    tokenizer's real pad token, then mean pooling over the attention
    mask. Returns UNNORMALIZED vectors, exactly as the model itself does
    (its `modules.json` declares no `Normalize` module) -- `embed()`
    normalizes, so that contract is unchanged from A7.4.

    Every encode holds a lease on this instance for its whole
    duration (all chunks). A retired instance accepts no new encode."""

    def __init__(self, session: Any, tokenizer: Any, artifact: str,
                 representation: Representation | None = None) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {node.name for node in session.get_inputs()}
        self.identity = model_identity_for(artifact)
        self._artifact = artifact
        self._lease = _Lease()
        # Set only by `_construct_encoder`, from the bytes it loaded.
        self._representation = representation

    def _lease_state(self) -> _Lease:
        # atomic under the GIL; covers subclasses that skip __init__
        return self.__dict__.setdefault("_lease", _Lease())

    @property
    def retired(self) -> bool:
        return self._lease_state().retired

    def _acquire(self) -> None:
        lease = self._lease_state()
        with lease.lock:
            if lease.retired:
                raise EncoderRetired("this encoder was retired after a cancelled Run")
            lease.active += 1

    def _release(self) -> None:
        lease = self._lease_state()
        with lease.lock:
            lease.active -= 1
            drained = lease.retired and lease.active == 0
        if drained:
            _forget_retired(self)

    def retire(self) -> None:
        """Logical retirement: no new lease; running encodes finish; the
        object is never closed or emptied here."""
        lease = self._lease_state()
        with _model_cache_lock:
            with lease.lock:
                lease.retired = True
                busy = lease.active > 0
            artifact = getattr(self, "_artifact", None)
            if _model_cache.get(artifact) is self:  # never evict a successor
                del _model_cache[artifact]
            if busy:
                _retired_in_use[artifact] = self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, per_text=False)

    def _encode(self, texts: Sequence[str], *, per_text: bool) -> np.ndarray:
        """The one encode path of both recipes: same lease, same
        deadline handling, same chunks loop; per-text is chunks of one."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        budget = _ENCODE_DEADLINE.get()
        self._acquire()
        try:
            if budget is None:
                return self._encode_chunks(texts, per_text)
            return self._encode_with_deadline(texts, budget, per_text)
        finally:
            self._release()

    def _encode_chunks(self, texts: list[str], per_text: bool = False) -> np.ndarray:
        if per_text:
            chunks = [self._encode_batch([text], one_sequence=True) for text in texts]
        else:  # the default path calls `_encode_batch` exactly as the batched recipe always has
            chunks = [
                self._encode_batch(texts[start : start + SEMANTIC_ENCODE_BATCH_SIZE])
                for start in range(0, len(texts), SEMANTIC_ENCODE_BATCH_SIZE)
            ]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def _encode_with_deadline(self, texts: list[str], budget: float, per_text: bool = False) -> np.ndarray:
        request = _EncodeRequest(_monotonic() + budget, on_terminate=self.retire)
        timer = _deadline_timer(budget, request.on_deadline)
        token = _ACTIVE_REQUEST.set(request)
        timer.start()
        try:
            try:
                result = self._encode_chunks(texts, per_text)
            except EncoderRunExpired:
                raise
            except Exception as exc:
                if request.terminate_reached_a_run:
                    raise EncoderRunExpired(f"Run aborted after the termination request: {exc}",
                                            terminate_requested=True, run_aborted=True) from exc
                raise  # an ONNX failure, not a timeout
            request.finish()  # after pooling: success only within the deadline
            return result
        except EncoderRunExpired as exc:
            if exc.terminate_requested:
                self.retire()  # idempotent: the callback has normally done it already
            raise
        finally:
            _ACTIVE_REQUEST.reset(token)
            timer.cancel()  # outside every lock
            timer.join()    # the callback, if it started, is over

    def _encode_batch(self, texts: list[str], *, one_sequence: bool = False) -> np.ndarray:
        request = _ACTIVE_REQUEST.get()
        if request is not None:
            request.check("tokenization")
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        if one_sequence and (input_ids.ndim != 2 or input_ids.shape[0] != 1 or not attention_mask.all()):
            # The per-text recipe's structural guarantee, checked
            # before the Run: exactly one sequence and not one padding position.
            raise SemanticUnavailable("the per-text recipe requires one unpadded sequence per Run")
        feed: dict[str, np.ndarray] = {"input_ids": input_ids}
        # Which inputs a given artifact declares is read off the real
        # session rather than assumed -- not every export of the same
        # model exposes the same set.
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        if request is None:
            last_hidden_state = self._session.run(None, feed)[0]
        else:
            request.begin_run()
            try:
                last_hidden_state = self._session.run(None, feed, request.options)[0]
            finally:
                request.end_run()
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (last_hidden_state * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return np.asarray(pooled, dtype=np.float32)


class _PerTextEncoder:
    """The built-in encoder under the experimental per-text recipe.
    A VIEW on the cached `_OnnxTextEncoder` -- same ONNX session, tokenizer,
    lease, retirement and cache (no second session, no second copy of the
    model) -- whose `encode` gives every Run exactly one text. Its
    representation is the base encoder's verified one with the per-text
    `recipe.batch`; an unattested base stays unattested."""

    def __init__(self, base: _OnnxTextEncoder) -> None:
        self._base = base
        self.identity = getattr(base, "identity", SEMANTIC_MODEL_ID)
        verified = base.__dict__.get("_representation")
        derived = base.__dict__.get("_per_text_representation")
        if derived is None:
            if isinstance(verified, Representation):
                descriptor = verified.descriptor
                descriptor["encoder"]["recipe"]["batch"] = _per_text_section()
                derived = Representation.build(descriptor)
            else:
                derived = representation_of(object())  # unattested, like its base
            base.__dict__["_per_text_representation"] = derived
        self.representation = derived

    @property
    def retired(self) -> bool:
        return self._base.retired

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._base._encode(texts, per_text=True)


def _with_active_policy(model: Any) -> Any:
    """The built-in `model` under the policy of the current context."""
    if _ENCODING_POLICY.get() == ONNX_PER_TEXT_POLICY and isinstance(model, _OnnxTextEncoder):
        return _PerTextEncoder(model)
    return model


_model_cache: dict[str, _OnnxTextEncoder] = {}
# Retired encoders that still have encodes running: no successor is
# built for their artifact until they drain (no unbounded session growth).
_retired_in_use: dict[str, _OnnxTextEncoder] = {}
_building: dict[str, threading.Event] = {}


def _forget_retired(encoder: _OnnxTextEncoder) -> None:
    with _model_cache_lock:
        artifact = getattr(encoder, "_artifact", None)
        if _retired_in_use.get(artifact) is encoder:
            del _retired_in_use[artifact]


def _artifact_path(artifact: str, *, local_files_only: bool) -> str:
    import huggingface_hub

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
    the model2vec backend was before A16.3.

    The cache is synchronized: one construction per artifact at a
    time, OUTSIDE the cache lock (others wait at most
    `ENCODER_BUILD_WAIT_SECONDS`, then get `EncoderUnavailable`); no
    successor while a retired encoder of the artifact still runs."""
    while True:
        with _model_cache_lock:
            cached = _model_cache.get(artifact)
            if cached is not None and not cached.retired:
                return cached
            if artifact in _retired_in_use:
                raise EncoderUnavailable(
                    f"the previous {artifact!r} session is retired and still in use; not rebuilt"
                )
            event = _building.get(artifact)
            builder = event is None
            if builder:
                event = _building[artifact] = threading.Event()
        if builder:
            break
        if not event.wait(ENCODER_BUILD_WAIT_SECONDS):
            raise EncoderUnavailable(f"another construction of {artifact!r} did not finish in time")
        with _model_cache_lock:
            cached = _model_cache.get(artifact)
            if cached is not None and not cached.retired:
                return cached
        raise EncoderUnavailable(f"the concurrent construction of {artifact!r} did not produce an encoder")
    try:
        encoder = _construct_encoder(artifact, local_files_only=local_files_only)
        with _model_cache_lock:
            _model_cache[artifact] = encoder
        return encoder
    finally:
        with _model_cache_lock:
            del _building[artifact]
        event.set()


def _construct_encoder(artifact: str, *, local_files_only: bool) -> _OnnxTextEncoder:
    """Load one ONNX session + tokenizer (no cache; seconds; not cancellable).

    The tokenizer file is read ONCE: its SHA-256 is taken over those
    bytes and the same bytes are what `tokenizers` parses. The model is hashed
    as a stream and loaded from its path, then RE-CHECKED: the file must still
    be the one hashed (device, inode, size, mtime), or the encoder is refused.
    Handing ONNX Runtime the hashed bytes instead would bind the digest
    exactly, but ORT then keeps ~113 MB more resident per session (measured);
    the re-check is the chosen alternative and its limit is
    stated: an in-place rewrite that restores size and mtime between hash and
    load is not detected. Truncation, padding, providers and the output width
    are then read back from the loaded objects; any disagreement with the
    recipe refuses the encoder instead of describing it wrongly."""
    import hashlib
    import os

    onnx_path = _artifact_path(artifact, local_files_only=local_files_only)
    tokenizer_path = _artifact_path(TOKENIZER_FILENAME, local_files_only=local_files_only)
    import onnxruntime
    from tokenizers import Tokenizer

    def identity(st: os.stat_result) -> tuple:
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    digest = hashlib.sha256()
    with open(onnx_path, "rb") as handle:
        hashed = identity(os.fstat(handle.fileno()))
        while block := handle.read(1 << 20):
            digest.update(block)
    model_sha256 = digest.hexdigest()
    with open(tokenizer_path, "rb") as handle:
        tokenizer_bytes = handle.read()
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    session = onnxruntime.InferenceSession(onnx_path, providers=list(ONNX_PROVIDERS))
    if identity(os.stat(onnx_path)) != hashed:
        raise SemanticUnavailable("the model file changed while it was being loaded; its identity is unknown")
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    del tokenizer_bytes
    tokenizer.enable_truncation(max_length=SEMANTIC_MAX_SEQ_LENGTH)
    pad_id = tokenizer.token_to_id(_PAD_TOKEN)
    if pad_id is None:
        raise SemanticUnavailable(
            f"Semantic tokenizer for {SEMANTIC_MODEL_REPO} has no '<pad>' token; "
            "the cached tokenizer file is not the expected one"
        )
    tokenizer.enable_padding(pad_id=pad_id, pad_token=_PAD_TOKEN)
    padding = tokenizer.padding or {}
    if (dict(tokenizer.truncation or {}) != _EXPECTED_TRUNCATION or padding.get("length") is not None
            or padding.get("pad_to_multiple_of") is not None or padding.get("pad_token") != _PAD_TOKEN
            or padding.get("direction") != "right"):
        raise SemanticUnavailable("the loaded tokenizer does not apply the expected truncation/padding recipe")
    if list(session.get_providers()) != list(ONNX_PROVIDERS):
        raise SemanticUnavailable("the ONNX session does not run on the expected execution provider")
    outputs = session.get_outputs()
    width = outputs[0].shape[-1] if outputs and outputs[0].shape else None
    if not (isinstance(width, int) and width > 0):
        raise SemanticUnavailable("the ONNX graph does not declare a fixed output width")
    inputs = sorted({node.name for node in session.get_inputs()} & {"input_ids", "attention_mask", "token_type_ids"})
    representation = onnx_representation(artifact, model_sha256=model_sha256, tokenizer_sha256=tokenizer_sha256,
                                         inputs=inputs, dimensions=width)
    return _OnnxTextEncoder(session, tokenizer, artifact, representation)


def load_model_for_setup() -> _OnnxTextEncoder:
    """Load the semantic model, downloading the pinned artifacts if they
    are not already cached. Only ever called from `urdyn semantic setup`
    -- the one entry point allowed to touch the network.

    Falls back ONCE, from this machine's preferred artifact to the
    full-precision portable one, if the preferred artifact cannot
    actually be fetched or loaded here. There is deliberately no third
    attempt: the identity of whatever artifact succeeded is what gets
    recorded with the index, so a fallback is fully visible downstream
    and can never be silently mixed with a different artifact's vectors."""
    artifact = preferred_artifact()
    try:
        return _with_active_policy(_build_encoder(artifact, local_files_only=False))
    except Exception as exc:
        if artifact == ARTIFACT_PORTABLE:
            raise SemanticUnavailable(
                f"Failed to load semantic model artifact {artifact!r} from "
                f"{SEMANTIC_MODEL_REPO}: {exc}"
            ) from exc
        try:
            return _with_active_policy(_build_encoder(ARTIFACT_PORTABLE, local_files_only=False))
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
    # The recorded artifact is necessary, not sufficient: the encoder
    # must produce the index's exact representation, or it is not used.
    return bind_representation(load_model_for_retrieval(artifact), meta)


def artifacts_available(meta: "SemanticMeta") -> bool:
    """[A27] Whether the artifacts an existing index needs are already in
    the local cache -- WITHOUT loading them.

    This resolves cache paths (`local_files_only=True`, so it can never
    reach the network) and stops there: no `InferenceSession`, no
    tokenizer, none of the seconds `_build_encoder` costs. That is what
    lets `urdyn status` distinguish "this index is fine" from "this
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
    return _with_active_policy(_build_encoder(artifact, local_files_only=True))


def model_identity(model: Any) -> str:
    """The effective identity of the artifact `model` was actually loaded
    from -- which is not necessarily this machine's preferred artifact,
    because `load_model_for_setup` may have fallen back."""
    return getattr(model, "identity", SEMANTIC_MODEL_ID)


def model_dimensions(model: Any) -> int:
    """Read the real output dimensionality off the loaded model instead
    of trusting a constant -- the one number an artifact swap could
    change without anything else noticing.

    The probe row goes through the same numeric contract as
    every other vector, so an encoder emitting degenerate output is
    refused here, before `semantic_setup` touches the existing index."""
    return int(embed(model, ["_urdyn_semantic_dimension_probe_"]).shape[1])


def resolve_local_revision() -> str:
    """The upstream revision the index was built from. Pinned (see
    `SEMANTIC_MODEL_REVISION`), so there is nothing to discover at
    runtime: every artifact is fetched at exactly this revision, which is
    what makes this value true rather than merely reported."""
    return SEMANTIC_MODEL_REVISION


def _invalid(message: str) -> InvalidSemanticVectors:
    return InvalidSemanticVectors(f"Semantic vectors rejected: {message}")


def require_unit_rows(vectors: np.ndarray, *, rows: int | None = None, dimensions: int | None = None) -> np.ndarray:
    """The numeric contract's acceptance check for vectors that
    are ALREADY normalized (normalizer output, stored index rows): a 2-D
    float32 array with the expected row count and width, every component
    finite, and every row of unit Euclidean norm within
    `UNIT_NORM_TOLERANCE` (computed in float64). Returns `vectors`
    unchanged or raises `InvalidSemanticVectors`; never repairs."""
    if not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32 or vectors.ndim != 2:
        raise _invalid("expected a 2-D float32 array")
    if rows is not None and vectors.shape[0] != rows:
        raise _invalid(f"expected {rows} rows, got {vectors.shape[0]}")
    if dimensions is not None and vectors.shape[1] != dimensions:
        raise _invalid(f"expected {dimensions} dimensions, got {vectors.shape[1]}")
    if vectors.shape[0] == 0:
        return vectors
    if not np.isfinite(vectors).all():
        raise _invalid("non-finite component")
    norms = np.sqrt(np.sum(np.square(vectors.astype(np.float64)), axis=1))
    bad = np.flatnonzero(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE)
    if bad.size:
        raise _invalid(f"row {int(bad[0])} has norm {float(norms[bad[0]])!r}, not 1 within {UNIT_NORM_TOLERANCE}")
    return vectors


def normalize_rows(raw: Any, *, rows: int, dimensions: int | None = None) -> np.ndarray:
    """The single numeric contract between an encoder and the
    index/query paths. Every vector Urdyn persists or compares passes
    through here (`embed`), or through `require_unit_rows` when read back.

    Acceptance: `raw` must be a real-valued (integer or floating, not
    bool/complex/object) 2-D array of exactly `rows` rows and
    1..`MAX_SEMANTIC_DIMENSIONS` columns (`dimensions`, when known). It is
    converted to float32 -- the encoder's declared output type -- and must
    then be entirely finite: a conversion that overflows to Inf is
    rejected, never clipped. Each row must be non-zero AFTER conversion,
    so a value that underflows to zero in the cast is rejected too. Only
    an explicitly empty request (`rows == 0`) may yield no rows.

    Normalization, per row x (float32 accepted, computed in float64):
        m = max_i |x_i| > 0,   z = x / m,   y = z / sqrt(sum_i z_i**2)
    For accepted x, m is a finite float32 magnitude in [2**-149, 2**128)
    and z_i is exact up to one rounding with max_i |z_i| = 1, so
    1 <= sum z_i**2 <= d: the norm can neither overflow nor underflow,
    whatever the input's scale (the old float32 `x / ||x||` overflowed
    for |x_i| ~ 1e20 and underflowed for |x_i| ~ 1e-20). The smallest
    non-zero |z_i| is >= 2**-149 / 2**128, far above float64's range
    limit, so no intermediate is subnormal in float64. y is then cast to
    float32 and checked by `require_unit_rows` -- the proof covers real
    arithmetic; the check covers the rounding.
    """
    try:
        array = np.asarray(raw)
    except Exception as exc:  # ragged nested sequences, foreign objects
        raise _invalid(f"encoder output is not an array ({exc})") from exc
    if array.dtype.kind not in "iuf":
        raise _invalid(f"unsupported dtype {array.dtype}")
    if rows == 0:
        if array.size != 0:
            raise _invalid(f"expected no rows for an empty request, got shape {array.shape}")
        return np.zeros((0, dimensions or 0), dtype=np.float32)
    if array.ndim != 2:
        raise _invalid(f"expected a 2-D array, got shape {array.shape}")
    if array.shape[0] != rows:
        raise _invalid(f"expected {rows} rows, got {array.shape[0]}")
    width = array.shape[1]
    if not 1 <= width <= MAX_SEMANTIC_DIMENSIONS:
        raise _invalid(f"unsupported width {width}")
    if dimensions is not None and width != dimensions:
        raise _invalid(f"expected {dimensions} dimensions, got {width}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        x32 = array.astype(np.float32)
    if not np.isfinite(x32).all():
        raise _invalid("non-finite component (NaN/Inf, or overflow in the float32 conversion)")
    x = x32.astype(np.float64)
    m = np.max(np.abs(x), axis=1)
    zero = np.flatnonzero(m == 0)
    if zero.size:
        raise _invalid(f"row {int(zero[0])} is zero (possibly after the float32 conversion)")
    z = x / m[:, None]
    y = z / np.sqrt(np.sum(z * z, axis=1))[:, None]
    return require_unit_rows(y.astype(np.float32), rows=rows, dimensions=width)


def embed(model: Any, texts: Sequence[str]) -> np.ndarray:
    """Embed `texts` and L2-normalize each row so cosine similarity
    reduces to a plain dot product. The backend model does not normalize
    its own output (its `modules.json` declares no `Normalize` module),
    and normalization is applied explicitly here rather than assumed, so
    this module's correctness never depends on that being true of
    whichever model is configured.

    Normalization and validation are `normalize_rows`: one row
    per text, or `InvalidSemanticVectors` -- never a silently repaired
    or degenerate vector.

    An encoder that declares `output_dimensions` (the external
    backend adapter) has that width enforced here as well, by the same
    contract; the built-in ONNX encoder declares none and is checked
    against the index width by its callers, exactly as before."""
    texts = list(texts)
    return normalize_rows(
        model.encode(texts), rows=len(texts), dimensions=getattr(model, "output_dimensions", None)
    )


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
    # Stored rows are re-checked against the numeric contract on
    # read (a corrupted blob raises instead of producing NaN scores), and
    # the query row must have the index's width.
    # Failures of the STORED rows are classified as such (see
    # `InvalidStoredVectors`); the computation is unchanged.
    try:
        matrix = require_unit_rows(
            np.stack([blob_to_vector(blob, dimensions) for _, blob in stored_vectors]),
            rows=len(ids),
            dimensions=dimensions,
        )
    except (InvalidSemanticVectors, ValueError) as exc:
        raise InvalidStoredVectors(f"Stored semantic vector rejected: {exc}") from exc
    # Inside a planned retrieval call the query was embedded outside
    # the lifecycle lock (`_encoding_plan`); the same rows are checked here.
    if isinstance(model, PreparedEncoder):
        query_rows = model.query_rows(query_text)
    else:
        query_rows = embed(model, [query_text])
    query_vector = require_unit_rows(query_rows, rows=1, dimensions=dimensions)[0]
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
    similarity engine (see `Urdyn._preflight_lesson_semantic_admitted`
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
