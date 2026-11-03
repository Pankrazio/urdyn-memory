"""The Urdyn workspace: identity, profile, and lifecycle."""

from __future__ import annotations

import contextvars
import dataclasses
import datetime as dt
import functools
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

from ._attempt import OUTCOME_FAILED, VALID_OUTCOMES, Attempt
from ._chunk import chunk_evidence, chunk_semantic_id, parse_chunk_semantic_id
from ._conflict import Conflict
from ._context import (
    DEFAULT_CONTEXT_BUDGET,
    SECTION_PROJECT_EVIDENCE,
    CompiledContext,
    compile_context,
)
from ._errors import (
    UrdynAlreadyInitializedError,
    UrdynManifestError,
    UrdynNotFoundError,
    UrdynSemanticUnavailableError,
    UrdynStorageError,
    UrdynStorageUnavailableError,
)
from ._event import (
    EVENT_KIND_ATTEMPT_RECORDED,
    EVENT_KIND_MEMORY_RECORDED,
    EVENT_KIND_MEMORY_SUPERSEDED,
    EVENT_KIND_SKILL_PROMOTED,
    Event,
)
from ._evidence import DEFAULT_EVIDENCE_KIND, VALID_EVIDENCE_KINDS, VERIFICATION_EVIDENCE_KINDS, Evidence
from ._gitignore import GitProtection, protect_workspace
from ._guard import GuardResult, build_guard_result
from ._manifest import (
    CANONICAL_PROFILES,
    LEGACY_WORKSPACE_ID_KEY,
    PROFILE_DEV,
    SCHEMA_VERSION,
    read_manifest,
    write_manifest,
)
from ._memory import (
    DEFAULT_KIND,
    EPISTEMIC_USER_ASSERTED,
    EPISTEMIC_VERIFIED,
    KIND_DECISION,
    KIND_INVALIDATION,
    KIND_INVARIANT,
    KIND_LESSON,
    KIND_PENDING,
    KIND_ROOT_CAUSE,
    VALID_EPISTEMIC_STATES,
    VALID_KINDS,
    Memory,
)
from ._preflight import (
    Preflight,
    PreflightConflict,
    ProjectEvidenceTrace,
    build_preflight,
    build_relevance_context,
    memory_is_relevant,
    minimum_sufficient_project_evidence,
    ordered_project_evidence,
    trace_project_evidence,
)
from ._relevance import attempt_search_text as _attempt_search_text
from ._relevance import is_relevant as _is_relevant
from ._relevance import memory_search_text as _memory_search_text
from ._relevance import skill_search_text as _skill_search_text
from ._relevance import tokens as _tokens
from ._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL, ENTITY_SOURCE
from ._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_EXTRA_MISSING,
    DETAIL_INDEX_UNREADABLE,
    DETAIL_MODEL_MISMATCH,
    DETAIL_MODEL_UNCACHED,
    DETAIL_NOT_SET_UP,
    DETAIL_QUERY_FAILED,
    DETAIL_QUERY_VECTORS_INVALID,
    DETAIL_REFRESH_FAILED,
    DETAIL_REPRESENTATION_MISMATCH,
    DETAIL_REPRESENTATION_UNVERIFIED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    STATUS_BUILDING,
    STATUS_READY,
    SemanticIndexStore,
    SemanticMeta,
    SemanticState,
    semantic_db_path_for,
)
from ._skill import Skill
from ._source import (
    SeedCandidateReport,
    SeedResult,
    Source,
    discover_candidate_paths,
    discover_seed_candidates,
    read_seed_candidate,
    resolve_seed_path,
)
from ._lifecycle import lifecycle, sync_directory, guarded, normalize_location, inference_slot
from . import _encoding_plan
from . import _representation
import os
import stat
from ._store import MemoryStore, SearchIndexState, db_path_for
from .semantic_backend import SemanticBackend, SemanticBackendIdentity

URDYN_DIRNAME = ".urdyn"
DEFAULT_RECALL_LIMIT = 20


def _no_store_evidence_lookup(evidence_id: str) -> Evidence:
    """`_ExperienceMaterial.empty()`'s `evidence_lookup`: unreachable in
    practice, since an empty candidate set never cites any Evidence id,
    but fails loudly rather than silently if that invariant is ever
    violated."""
    raise UrdynStorageError(f"Evidence {evidence_id!r} was requested but this workspace has no store")


@dataclasses.dataclass(frozen=True, slots=True)
class _ExperienceMaterial:
    """Everything `preflight()` and `context()` (A29.1) both need from
    one call to `Urdyn._gather_experience`: current-state-filtered
    candidate lists, already-admitted semantic id sets, and the
    retrieval-lifecycle state that call produced. Purely an internal
    hand-off between that method and its two callers -- never returned
    from a public API, never persisted."""

    attempts: list[Attempt]
    root_cause_memories: list[Memory]
    verified_lesson_memories: list[Memory]
    invariant_memories: list[Memory]
    invalidation_memories: list[Memory]
    pending_memories: list[Memory]
    decision_memories: list[Memory]
    open_conflict_list: list[Conflict]
    current_memory_by_id: dict[str, Memory]
    evidence_lookup: Callable[[str], Evidence]
    attempt_fts_candidates: list[tuple[str, str]]
    memory_fts_candidates: list[tuple[str, str]]
    attempt_semantic_admitted: frozenset[str]
    memory_semantic_admitted: frozenset[str]
    retrieval: "SemanticState"
    current_source_evidence: list[tuple[Source, Evidence]]
    source_fts_candidates: list[tuple[str, str]]
    search_index: str | None = None  # `missing` / `unchecked`, see MemoryStore.search_index_presence

    @classmethod
    def empty(cls, *, retrieval: "SemanticState") -> "_ExperienceMaterial":
        """The answer for a workspace with no canonical store yet (A27):
        every candidate list is empty, but `retrieval` still reports HOW
        the (empty) answer was produced, exactly like a populated one."""
        return cls(
            attempts=[],
            root_cause_memories=[],
            verified_lesson_memories=[],
            invariant_memories=[],
            invalidation_memories=[],
            pending_memories=[],
            decision_memories=[],
            open_conflict_list=[],
            current_memory_by_id={},
            evidence_lookup=_no_store_evidence_lookup,
            attempt_fts_candidates=[],
            memory_fts_candidates=[],
            attempt_semantic_admitted=frozenset(),
            memory_semantic_admitted=frozenset(),
            retrieval=retrieval,
            current_source_evidence=[],
            source_fts_candidates=[],
        )


def _open_conflicts_projection(conflicts: list[Conflict], current_ids: set[str]) -> list[Conflict]:
    """The A13 definition of "open", in exactly one place: a `Conflict`
    is open iff BOTH memories it names are current. Shared by
    `Urdyn.open_conflicts()` and `Urdyn.preflight()` so the rule is
    never reimplemented a second time for whichever caller happens to
    already have a store/`current_ids` open of its own."""
    return [
        conflict
        for conflict in conflicts
        if conflict.memory_ids[0] in current_ids and conflict.memory_ids[1] in current_ids
    ]


def _selected_project_evidence_keys(compiled: CompiledContext) -> frozenset[tuple[str, int]]:
    """`(source_path, chunk_index)` for every PROJECT EVIDENCE item the
    budget scan actually kept. `chunk_index` is `None` on a rendered item
    whose document was a single chunk (see `_context.compile_context`),
    which is index 0 by construction. Internal, used only to attribute
    selection back to `ProjectEvidenceTrace` rows."""
    return frozenset(
        (item.source_path or "", item.chunk_index or 0)
        for section in compiled.sections
        if section.heading == SECTION_PROJECT_EVIDENCE
        for item in section.items
    )


# [A54.3] Semantic-index-only entity type for derived, per-chunk
# representations of a seeded Source's current-observation Evidence --
# see `_semantic_pool_entries`'s "Strategy B'" paragraph below. Deliberately
# NOT `._retrieval.ENTITY_SOURCE`: that constant names the WHOLE-DOCUMENT
# pool the FTS/lexical widening channel still ranks (`_relevance.py`'s
# `search_candidates(..., ENTITY_SOURCE)`, unchanged by this), a completely
# separate store (`MemoryStore`'s FTS index) from the one this constant
# addresses (`SemanticIndexStore`'s vector table). Reusing one string for
# two different pools across two different stores would make "is this the
# lexical or the semantic Source pool" a fact a reader has to infer from
# context instead of from the name.
ENTITY_SOURCE_CHUNK = "source_chunk"


def _semantic_pool_entries(
    store: MemoryStore,
    *,
    memories: list[Memory] | None = None,
    attempts: list[Attempt] | None = None,
    skills: list[Skill] | None = None,
    source_evidence: list[tuple[Source, Evidence]] | None = None,
) -> tuple[tuple[str, list[tuple[str, str]]], ...]:
    """[A27] THE definition of what the semantic index is derived from:
    which canonical records feed it, and with which text.

    One definition, every consumer -- `semantic_setup()` (which builds
    the index), `Urdyn.semantic_state()` (which decides whether it is
    still current) and the incremental refresh (which repairs it), and
    now `Urdyn.context()`'s PROJECT EVIDENCE pool as well. Before A27
    this lived inline inside `semantic_setup()` and had exactly one
    reader, which was fine while nothing else needed to know what
    "semantically relevant" meant. It is extracted rather than copied
    because the freshness answer is only as true as its agreement with
    the build: a `status` command that judged coverage against a
    different set of records than `semantic_setup()` indexes would report
    a workspace permanently stale, or permanently current, and either way
    it would be a lie that no test could catch by inspecting one side
    alone.

    The Attempt/Memory/Skill pools and their texts are UNCHANGED from
    A7.4 -- the same representations `_relevance.py` already derives for
    FTS, no new canonical field.

    `ENTITY_SOURCE_CHUNK` is the fourth pool -- [A54.3, Strategy B',
    superseding the whole-document `ENTITY_SOURCE` embedding this pool
    used before]: every derived chunk (`_chunk.chunk_evidence`) of the
    current (latest-observation) Evidence of every seeded Source, keyed
    by `_chunk.chunk_semantic_id(evidence_id, chunk_index)` -- never a row
    id, never a path, always reconstructible from the same
    `(evidence_id, chunk_index)` pair the lexical chunk ranker already
    uses. A54.2/A54.2.1 measured that ONE embedding for an entire document
    (truncated at the model's own 128-token limit) cannot represent a
    passage buried past that window; embedding each chunk independently
    and aggregating a Source's semantic score as the MAXIMUM among its own
    current chunks (see `Urdyn._context_evidence_semantic_admitted`) lets
    one strongly relevant section make its Source admissible without a
    document scoring higher merely for having more chunks. Admission
    itself stays at SOURCE granularity, with the SAME
    `EVIDENCE_SEMANTIC_FLOOR`/`EVIDENCE_ADMISSION_LIMIT` cap as before --
    this is deliberately not a global top-K over hundreds of independent
    chunks, which would let one long document's many chunks crowd out
    every other seeded Source's only chance at a slot (A54.2.1 ruled this
    out explicitly).

    Evidence/Sources must be included because retrieval now consults
    them: superseding a Source makes this index stale for the new
    observation's Evidence, exactly like recording a new Memory does; the
    old observation's chunks are simply no longer in this pool -- they are
    not unindexed retroactively (the semantic vector store is append-only
    derived data, see `observe_source`), only excluded from THIS pool the
    same call after call, which is what "current" means here, and is
    re-enforced independently by `Urdyn._context_evidence_semantic_admitted`
    restricting ranking to chunks of the CURRENT observation before it
    ever ranks anything (A7.7's own discipline, unchanged). Ordinary
    canonical Conflicts remain absent: nothing has ever made them a
    retrieval unit of their own.

    Callers that have already read some of these lists pass them in; the
    parameters exist to avoid re-reading the canonical store inside a
    call that just materialized the very same rows, not to let a caller
    substitute a different population.
    """
    memories = store.timeline(None) if memories is None else memories
    attempts = store.list_attempts() if attempts is None else attempts
    skills = store.list_skills() if skills is None else skills
    source_evidence = store.list_current_source_evidence() if source_evidence is None else source_evidence
    chunk_entries = [
        (chunk_semantic_id(evidence.evidence_id, chunk.chunk_index), chunk.text)
        for _source, evidence in source_evidence
        for chunk in chunk_evidence(evidence)
    ]
    return (
        (ENTITY_ATTEMPT, [(a.attempt_id, _attempt_search_text(a.task, a.approach)) for a in attempts]),
        (ENTITY_MEMORY, [(m.memory_id, _memory_search_text(m.content)) for m in memories]),
        (ENTITY_SKILL, [(s.skill_id, _skill_search_text(s.name, s.purpose, s.conditions)) for s in skills]),
        (ENTITY_SOURCE_CHUNK, chunk_entries),
    )


def _load_semantic_module():
    """Lazily import the optional semantic channel (`onnxruntime`,
    `tokenizers`, `huggingface_hub`, `numpy`). Returns None -- never
    raises -- if the `urdyn-memory[semantic]` extra is not installed, so
    every caller in this module degrades to lexical/FTS-only exactly as
    if A7.4 did not exist. This is the ONLY place outside `_semantic.py`
    itself that imports it."""
    try:
        from . import _semantic
    except ImportError:
        return None
    return _semantic


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticSetupResult:
    """Report returned by `Urdyn.semantic_setup()`. Not a `Memory`, not
    canonical data -- purely a summary of what the (re)build just did."""

    provider: str
    model_id: str
    model_revision: str | None
    dimensions: int
    normalization: str
    attempt_count: int
    memory_count: int
    skill_count: int
    source_evidence_count: int = 0


# ---------------------------------------------------------------------------
# Per-call semantic outcome
# ---------------------------------------------------------------------------
#
# `semantic_state()` answers "is the derived INDEX usable" (built, compatible,
# covering canonical state). It does not, and must not, answer "did THIS call
# manage to use it": a stored row can fail the numeric contract on read, the
# index can be rebuilt concurrently, a backend can raise. Every semantic helper
# below degrades such a failure to "no semantic candidates", which -- without
# this -- was indistinguishable from the channel running and abstaining: the
# A26 failure mode, one level lower.
#
# One public retrieval call (`preflight`/`context`/`guard`) therefore records
# what its semantic reads actually did, in a ContextVar: per thread and per
# task, never shared between calls, never persisted, no cache. If the index was
# declared usable and any semantic read failed, the call is answered AGAIN with
# the semantic channel switched off, so the published answer is exactly the
# lexical/FTS answer and its `retrieval` says `unavailable` -- the existing
# state whose meaning is precisely "lexical only". No partial mixture of
# semantic pools that worked with pools that did not is ever published under a
# healthy label. The second pass reads canonical state afresh, performs no
# refresh and no semantic read; it only runs on the failure path.


@dataclasses.dataclass
class _SemanticCallOutcome:
    forced: SemanticState | None = None  # second pass: semantic channel off, report this state
    prepared: SemanticState | None = None  # what `_semantic_prepare` declared for this call
    failures: set[str] = dataclasses.field(default_factory=set)

    def degraded_state(self) -> SemanticState | None:
        if self.forced is not None or not self.failures or self.prepared is None:
            return None
        if not self.prepared.is_usable():
            return None  # already reported as not usable; nothing was hidden
        # A representation refusal names its own remedy (rebuild);
        # it wins over the generic outcomes, never over nothing.
        for detail in (DETAIL_REPRESENTATION_UNVERIFIED, DETAIL_REPRESENTATION_MISMATCH,
                       DETAIL_QUERY_VECTORS_INVALID, DETAIL_QUERY_FAILED):
            if detail in self.failures:
                break
        return SemanticState(
            status=SEMANTIC_UNAVAILABLE,
            detail=detail,
            missing=self.prepared.missing,
            indexed=self.prepared.indexed,
            refreshed=self.prepared.refreshed,
        )


_SEMANTIC_CALL: contextvars.ContextVar[_SemanticCallOutcome | None] = contextvars.ContextVar(
    "urdyn_semantic_call", default=None
)


def _semantic_suppressed() -> bool:
    outcome = _SEMANTIC_CALL.get()
    return outcome is not None and outcome.forced is not None


def _note_semantic_failure(exc: BaseException | None) -> None:
    """Record that a semantic read of the current call did not complete.
    `exc is None` means the channel vanished after preparation (e.g. a
    rebuild started mid-call)."""
    outcome = _SEMANTIC_CALL.get()
    if outcome is None:
        return
    semantic = _load_semantic_module() if exc is not None else None
    # Only bytes read back from the index count as "a stored vector
    # failed validation"; an invalid QUERY vector (encoder output) is a
    # failure of this query's semantic retrieval, not of the index.
    invalid = semantic is not None and isinstance(exc, semantic.InvalidStoredVectors)
    if semantic is not None and isinstance(exc, semantic.RepresentationUnverified):
        outcome.failures.add(DETAIL_REPRESENTATION_UNVERIFIED)
    elif semantic is not None and isinstance(exc, semantic.RepresentationMismatch):
        outcome.failures.add(DETAIL_REPRESENTATION_MISMATCH)
    else:
        outcome.failures.add(DETAIL_QUERY_VECTORS_INVALID if invalid else DETAIL_QUERY_FAILED)


def _semantic_outcome_guarded(method):
    """Wrap one public retrieval call; see the section comment above."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        trace = kwargs.get("_project_evidence_trace")
        trace_length = len(trace) if trace is not None else 0
        outcome = _SemanticCallOutcome()
        token = _SEMANTIC_CALL.set(outcome)
        try:
            result = method(self, *args, **kwargs)
        finally:
            _SEMANTIC_CALL.reset(token)
        degraded = outcome.degraded_state()
        if degraded is None:
            return result
        if trace is not None:
            del trace[trace_length:]  # the discarded pass must leave no instrumentation behind
        token = _SEMANTIC_CALL.set(_SemanticCallOutcome(forced=degraded))
        try:
            return method(self, *args, **kwargs)
        finally:
            _SEMANTIC_CALL.reset(token)

    return wrapper


# How long a call waits for this Brain's encoder while another call
# of the same process is using it, before answering without the semantic
# channel (reported in the call's outcome). A bounded WAIT, not a cancellation: the call that
# holds the encoder keeps running.
INFERENCE_SLOT_TIMEOUT_SECONDS = 30.0
# Plans per call: the first, plus one more if the storage was
# replaced (restore) or the index encoder changed before publication.
_INFERENCE_PLANS_PER_CALL = 2


class _SetupNotPublished(UrdynSemanticUnavailableError):
    """Semantic setup published nothing because the state it was
    prepared for changed, or the encoder stayed busy. Internal subclass:
    callers see `UrdynSemanticUnavailableError` with an explicit message."""


@dataclasses.dataclass(frozen=True)
class _SetupTicket:
    """What a rebuild was prepared for: the Brain (`brain_id`,
    re-checked by `@guarded`), its storage files (`incarnation`), the
    `building` marker it wrote, the backend it was captured with, and the
    canonical texts it embeds (`pools`, immutable tuples)."""

    brain_id: str
    incarnation: tuple
    marker: "SemanticMeta"
    backend: object
    backend_identity: object
    pools: tuple


def _encoder_slot(backend) -> object:
    """The inference slot an encoder is serialized under: one per
    client backend object, one for the built-in encoder."""
    return None if backend is None else ("external", id(backend))


def _inference_outside_lock(query_parameter: str):
    """Outermost decorator of a retrieval call whose encoder work
    can be planned from its query (`preflight`, `context`, `guard`): see
    `_encoding_plan` for the protocol and `Urdyn._run_with_prepared_encodings`
    for the steps. The decorated (guarded) method is unchanged."""

    def decorate(method):
        @functools.wraps(method)
        def call(self, *args, **kwargs):
            query = args[0] if args else kwargs.get(query_parameter)
            return self._run_with_prepared_encodings(query, method, self, *args, **kwargs)

        return call

    return decorate


def _raise_missing_brain(workspace: Path, *, creating_hint: bool = False) -> None:
    """`workspace` has no `.urdyn` entry. Say which of the three
    situations this is, never a manifest problem that does not exist."""
    stage = workspace / ".urdyn.creating"
    restore = [workspace.parent / ("." + workspace.name + suffix) for suffix in (".stage", ".rollback")]
    if os.path.lexists(stage):
        raise UrdynStorageUnavailableError(
            f"Brain creation at {workspace} was interrupted: {stage} exists and nothing was published. "
            "Inspect it and publish it explicitly with Urdyn.recover_creation(path); "
            "it is never recreated or removed automatically."
        )
    residues = [str(path) for path in restore if os.path.lexists(path)]
    if residues:
        raise UrdynStorageUnavailableError(
            f"A restore of {workspace} was interrupted ({', '.join(residues)}); "
            "run urdyn.backup.recover_interrupted_restore(path) before using this Brain."
        )
    if creating_hint:
        raise UrdynNotFoundError(
            f"No Urdyn Brain at {workspace}, and the directory is not empty: 'urdyn init' creates a Brain only in "
            "an empty or absent directory. To create one in this existing directory, run 'urdyn create [profile]' "
            "(Python: Urdyn.create(path, profile))."
        )
    raise UrdynNotFoundError(f"No Urdyn Brain at {workspace}")


class Urdyn:
    """A discovered or newly initialized Urdyn workspace."""

    def __init__(self, path: Path, profile: str, urdyn_id: str) -> None:
        self._path = path
        self._profile = profile
        self._urdyn_id = urdyn_id
        self._semantic_backend: SemanticBackend | None = None
        self._semantic_backend_identity: SemanticBackendIdentity | None = None
        # Set by create/recover_creation/init: what is known about the
        # Git protection of `.urdyn/` (reported by the CLI). None for `open`.
        self._git_protection: GitProtection | None = None

    @guarded
    def configure_semantic_backend(self, backend: SemanticBackend) -> None:
        """Attach a local encoder to this instance, retaining Urdyn's index rules.

        A reopened workspace must attach the backend again. With no
        backend attached, the existing Python ONNX behavior is unchanged.
        """
        semantic = _load_semantic_module()
        if semantic is None:
            raise UrdynSemanticUnavailableError("External semantic backends require NumPy")
        identity = backend.identity
        semantic.validate_external_identity(identity)
        if not callable(backend.available) or not callable(backend.encode):
            raise TypeError("semantic backend must provide available() and encode()")
        self._semantic_backend = backend
        self._semantic_backend_identity = identity

    def _external_semantic_ready(self) -> bool:
        return self._backend_ready(self._semantic_backend, self._semantic_backend_identity)

    @staticmethod
    def _backend_ready(backend, identity) -> bool:
        if backend is None or identity is None:
            return False
        try:
            return backend.identity == identity and bool(backend.available())
        except Exception:
            return False

    def _external_identity_matches(self, meta) -> bool:
        identity = self._semantic_backend_identity
        if identity is None:
            return False
        # The columns the identity has always been checked against,
        # on the base provider; the representation is checked separately.
        meta = dataclasses.replace(meta, provider=_representation.base_provider(meta.provider))
        return not (not meta.matches(
            provider=identity.index_provider, model_id=identity.model_id, normalization=identity.normalization
        ) or meta.dimensions != identity.dimensions or meta.model_revision != identity.model_revision)

    def _encoder_key(self, meta, semantic):
        """Which encoder answers `meta`'s index on this handle, as
        a value (never the encoder object), or None if none can. A prepared
        result is bound to it (`_encoding_plan`); the same conditions as
        `_semantic_model_for_index` below, without loading anything."""
        # The stored provider carries the index's representation
        # reference: a plan is bound to that representation, so an index
        # rebuilt under another one meanwhile never receives its rows. An
        # index whose representation this handle cannot produce has no key
        # (no model is loaded just to be refused).
        if semantic.representation_problem(meta, self._semantic_backend_identity) is not None:
            return None
        if self._semantic_backend is None:
            artifact = semantic.artifact_for_index(meta)
            return None if artifact is None else ("onnx", artifact, meta.dimensions, meta.provider)
        if not self._external_identity_matches(meta):
            return None
        return ("external", id(self._semantic_backend), self._semantic_backend_identity, meta.dimensions,
                meta.provider)

    def _semantic_model_for_index(self, meta, semantic):
        prepared = _encoding_plan.active()
        if prepared is not None:
            # Inside a planned call: the same checks, then an encoder
            # that only serves what was encoded outside the lifecycle lock.
            key = self._encoder_key(meta, semantic)
            if key is None:
                return None
            if self._semantic_backend is not None and not self._external_semantic_ready():
                return None
            return prepared.encoder_for(key)
        if self._semantic_backend is None:
            return semantic.load_model_for_index(meta)
        identity = self._semantic_backend_identity
        if not self._external_identity_matches(meta):
            return None
        if not self._external_semantic_ready():
            return None
        return semantic.bind_representation(semantic.ExternalEncoder(self._semantic_backend, identity), meta)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def urdyn_id(self) -> str:
        return self._urdyn_id

    def __repr__(self) -> str:
        return f"Urdyn(path={str(self._path)!r}, profile={self._profile!r})"

    @classmethod
    def create(cls, path: str | Path, profile: str = "general") -> "Urdyn":
        """Explicit creation; publish a complete store without overwriting data."""
        if profile not in CANONICAL_PROFILES:
            raise ValueError("Unknown profile")
        workspace = normalize_location(path)
        with lifecycle(workspace):
            if any((workspace.parent / ("." + workspace.name + suffix)).exists() for suffix in (".stage", ".rollback")):
                raise UrdynAlreadyInitializedError("Restore recovery required before creation")
            workspace.mkdir(parents=True, exist_ok=True)
            sync_directory(workspace.parent)
            target = workspace / URDYN_DIRNAME
            stage = workspace / ".urdyn.creating"
            if os.path.lexists(stage):
                raise UrdynAlreadyInitializedError(
                    f"Brain creation at {workspace} was interrupted ({stage} exists); "
                    "publish it explicitly with Urdyn.recover_creation(path)"
                )
            if os.path.lexists(target):
                raise UrdynAlreadyInitializedError(
                    f"{target} already exists (a Brain, or an entry that is not one); "
                    "create never replaces it -- open the Brain or inspect that entry"
                )
            # Before anything is staged: if `.urdyn/` cannot be put
            # out of Git's reach (entry not writable, files already tracked,
            # an explicit user negation), no Brain is created at all.
            protection = protect_workspace(workspace)
            try:
                stage.mkdir(mode=0o700)
            except OSError as exc:  # nothing exists yet: a typed, explicit failure
                raise UrdynStorageUnavailableError(f"Cannot create a Brain in {workspace}: {exc.strerror}") from exc
            identity = uuid.uuid4().hex
            write_manifest(stage, {"schema_version": SCHEMA_VERSION, "profile": profile,
                                   LEGACY_WORKSPACE_ID_KEY: identity})
            with MemoryStore.create_new(db_path_for(stage)):
                pass
            for name in ("manifest.json", "memory.db"):
                with (stage / name).open("rb") as stream: os.fsync(stream.fileno())
            sync_directory(stage)
            os.rename(stage, target)
            sync_directory(workspace)
            engine = cls.open(workspace)
            engine._git_protection = protection
            return engine

    @classmethod
    def recover_creation(cls, path: str | Path) -> "Urdyn":
        """Explicitly publish a complete interrupted create; preserve invalid staging."""
        workspace = normalize_location(path)
        with lifecycle(workspace):
            stage = workspace / ".urdyn.creating"
            target = workspace / URDYN_DIRNAME
            if target.exists():
                if stage.exists():
                    raise UrdynAlreadyInitializedError("Ambiguous creation state")
                return cls.open(workspace)
            # The staged store must be bound to the staged manifest:
            # a discordant stage is refused and preserved, never rebound.
            staged = read_manifest(stage)
            MemoryStore.verify_existing(db_path_for(stage), brain_id=staged[LEGACY_WORKSPACE_ID_KEY])
            protection = protect_workspace(workspace)  # same rule as create, before publishing
            for name in ("manifest.json", "memory.db"):
                with (stage / name).open("rb") as stream: os.fsync(stream.fileno())
            sync_directory(stage)
            os.rename(stage, target)
            sync_directory(workspace)
            engine = cls.open(workspace)
            engine._git_protection = protection
            return engine

    @classmethod
    def init(cls, path: str | Path = ".", profile: str = "general") -> "Urdyn":
        """Create absent roots; existing roots must contain a valid Brain.

        Use create explicitly to initialize an existing project directory.
        Manifest-only legacy archives are deliberately rejected.

        A non-empty directory holding no Brain is refused with
        `UrdynNotFoundError` naming the remedy (`create`): it cannot be told
        apart from a project whose Brain was lost, so `init` never creates
        there implicitly. Re-initializing an existing Brain (re)applies the
        Git protection of `create` (0.4.0 did the same); it never rewrites
        the Brain.
        """
        workspace = normalize_location(path)
        with lifecycle(workspace):
            if any((workspace.parent / ("." + workspace.name + suffix)).exists() for suffix in (".stage", ".rollback")) and not workspace.exists():
                raise UrdynAlreadyInitializedError("Restore recovery required")
            try:
                empty = not workspace.exists() or (workspace.is_dir() and not any(workspace.iterdir()))
            except OSError as exc:
                raise UrdynStorageUnavailableError(f"Brain location {workspace} cannot be inspected: {exc.strerror}") from exc
            if empty:
                return cls.create(workspace, profile)
            if not os.path.lexists(workspace / URDYN_DIRNAME):
                _raise_missing_brain(workspace, creating_hint=True)
            engine = cls.open(workspace)
            if engine.profile != profile:
                raise UrdynAlreadyInitializedError("Profile mismatch")
            engine._git_protection = protect_workspace(workspace)
            return engine

    @classmethod
    def open(cls, path: str | Path = ".") -> "Urdyn":
        workspace = normalize_location(path)
        with lifecycle(workspace):
            # Absent and present-but-invalid are different answers:
            # no `.urdyn` entry at all is "not found" (or an interrupted
            # create/restore, which is neither missing nor valid); anything
            # else is read and validated by `read_manifest`.
            try:
                entry = os.lstat(workspace / URDYN_DIRNAME)
            except FileNotFoundError:
                _raise_missing_brain(workspace)
            except NotADirectoryError:
                raise UrdynNotFoundError(f"No Urdyn Brain at {workspace}: it is not a directory") from None
            except OSError as exc:
                raise UrdynStorageUnavailableError(f"Brain location {workspace} cannot be inspected: {exc.strerror}") from exc
            if stat.S_ISREG(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISLNK(entry.st_mode)):
                raise UrdynManifestError(f"{workspace / URDYN_DIRNAME} exists but is not a directory")
            data = read_manifest(workspace / URDYN_DIRNAME)
            engine = cls(workspace, data["profile"], data[LEGACY_WORKSPACE_ID_KEY])
            engine.verify_storage()
            return engine

    def verify_storage(self) -> None:
        """Fail closed unless existing canonical storage passes read-only checks.

        Does not create, migrate, or repair storage. A legacy workspace without
        a database is ambiguous and is rejected even if it might be empty.
        This point-in-time check does not make later operations atomic.
        """
        MemoryStore.verify_existing(self._db_path, brain_id=self._urdyn_id)  # the handle's identity

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "Urdyn":
        """Locate the nearest Urdyn workspace, walking upward from `start`."""
        current = Path(start if start is not None else Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            urdyn_dir = candidate / URDYN_DIRNAME
            if urdyn_dir.is_dir():
                data = read_manifest(urdyn_dir)
                return cls.open(candidate)
        raise UrdynNotFoundError(
            f"No Urdyn workspace found in {current} or any parent directory. "
            "Create one with 'urdyn create' (or 'urdyn init' in an empty directory)."
        )

    @guarded
    def remember(
        self,
        content: str,
        *,
        kind: str = DEFAULT_KIND,
        epistemic_state: str = EPISTEMIC_USER_ASSERTED,
        supersedes: str | None = None,
        evidence: Sequence[Evidence] = (),
        supporting_evidence: Sequence[Evidence] = (),
    ) -> Memory:
        """Persist a new canonical memory and return it.

        `content` is recorded verbatim; Urdyn does not interpret,
        summarize, or verify it. Rejects empty or whitespace-only input.

        (A17) Recording the same canonical memory twice is idempotent: if
        Urdyn already holds a CURRENT memory exactly equivalent to this
        one -- same `content` (byte-for-byte), `kind`, `epistemic_state`,
        `supersedes`, and same provenance (`evidence`/
        `supporting_evidence`) -- that existing memory is returned
        unchanged, with its original `memory_id` and `recorded_at`, and
        nothing new is written. A repeated call is a retry of one
        operation, not a second thing to believe, and two current records
        of the same claim are not a richer history: they are a canonical
        integrity defect that also makes retrieval treat one claim as two
        competing candidates. This is exact equivalence only: differently
        worded memories that mean the same thing are NOT deduplicated,
        and re-asserting something that has since been superseded is a
        real new fact and gets its own memory.

        If `supersedes` is given, it must be the memory_id of an existing
        memory; that memory is preserved as history, not deleted or
        modified, and stops being "current". `evidence` records why this
        memory exists (its generic provenance) -- Evidence merely
        related to or contextual for this memory, not necessarily proof
        of it.

        `supporting_evidence` (A12.1) is a second, narrower pool: the
        Evidence the caller explicitly designates as SUPPORTING this
        specific memory, as opposed to generic provenance. Any item in
        `supporting_evidence` is automatically folded into the memory's
        `evidence_ids` as well (supporting implies related) -- the same
        Evidence never needs to be listed in both `evidence` and
        `supporting_evidence` to appear in both places. `evidence_ids`
        preserves `evidence`'s own order first, then appends whatever
        `supporting_evidence` contributes that wasn't already present,
        deduplicated.

        `epistemic_state` defaults to `user_asserted`. As of A12.1, a
        memory may only be marked `verified` if `supporting_evidence`
        (not generic `evidence`) includes at least one item whose kind
        actually represents a check (a test result, a command or tool
        output, an explicit user confirmation) -- an opinion
        (`user_statement`) or a bare file reference is not enough, and
        neither is a qualifying-kind Evidence cited only as generic
        `evidence` without being explicitly designated supporting.
        Urdyn refuses to accept a verified claim resting on nothing,
        refuses one resting only on an unchecked assertion, and refuses
        one whose only qualifying Evidence was never explicitly asserted
        as support for THIS memory.

        This designation is a structural requirement, not a semantic
        guarantee: Urdyn does not judge whether the designated Evidence
        actually is relevant to `content`, nor its direction (e.g. a
        FAILED test explicitly designated as supporting is still
        accepted) -- only that the caller made an explicit, auditable
        assertion instead of `verified` being inferred from Evidence
        kind alone. Memories recorded before A12.1 may be `verified`
        with an empty `supporting_evidence_ids`; that is "verified under
        the pre-A12.1 contract" and is never retroactively rewritten.
        """
        memory, _created = self._remember(
            content,
            kind=kind,
            epistemic_state=epistemic_state,
            supersedes=supersedes,
            evidence=evidence,
            supporting_evidence=supporting_evidence,
        )
        return memory

    def _remember(
        self,
        content: str,
        *,
        kind: str = DEFAULT_KIND,
        epistemic_state: str = EPISTEMIC_USER_ASSERTED,
        supersedes: str | None = None,
        evidence: Sequence[Evidence] = (),
        supporting_evidence: Sequence[Evidence] = (),
    ) -> tuple[Memory, bool]:
        """(A17) The whole implementation of `remember()`, plus the one
        piece of information its public return type deliberately does not
        carry: whether this call actually recorded a NEW memory (True) or
        collapsed onto an already-current exact equivalent (False).

        Internal to the `urdyn remember` CLI command, exactly like
        `_count_memories` is internal to `urdyn status`. "Was this call
        the one that recorded it?" is a property of the CALL, not of the
        canonical Memory -- a Memory reloaded tomorrow by `state()` or
        `timeline()` could not answer it, and would have to answer it
        identically for both calls if it tried. Putting it on the public
        model would therefore make the canonical record depend on how it
        was obtained; keeping it here lets the CLI say "Already
        remembered" without inventing a public concept for it.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content must not be empty or whitespace-only")
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind {kind!r}; expected one of {sorted(VALID_KINDS)}")
        if epistemic_state not in VALID_EPISTEMIC_STATES:
            raise ValueError(
                f"Unknown epistemic state {epistemic_state!r}; expected one of {sorted(VALID_EPISTEMIC_STATES)}"
            )

        memory_id = uuid.uuid4().hex
        if supersedes is not None and supersedes == memory_id:
            raise ValueError("A memory cannot supersede itself")

        evidence_ids = tuple(dict.fromkeys(item.evidence_id for item in evidence))
        supporting_ids_raw = tuple(dict.fromkeys(item.evidence_id for item in supporting_evidence))
        # Supporting implies related: fold any supporting-only ids into the
        # generic provenance trail, preserving `evidence`'s own order first.
        all_evidence_ids = evidence_ids + tuple(
            eid for eid in supporting_ids_raw if eid not in evidence_ids
        )
        # (A12.1.1) `evidence_ids` is the single master order (the only
        # order the storage layer's `position` column can reconstruct on
        # reload -- see `MemoryStore.add()`/`_all_memory_supporting_evidence_map`).
        # `supporting_evidence_ids` is re-derived from THAT order rather
        # than kept in the caller's own `supporting_evidence` order, so
        # the object returned here is byte-identical, ordering included,
        # to what a fresh `state()`/`recall()`/`timeline()` reload of the
        # same row produces. Keeping the caller's raw order instead would
        # silently diverge from the reloaded object the moment a
        # supporting id was ALSO present in `evidence` at a different
        # position -- an algorithm-dependent inconsistency, not a
        # deliberate contract.
        supporting_set = frozenset(supporting_ids_raw)
        supporting_evidence_ids = tuple(eid for eid in all_evidence_ids if eid in supporting_set)

        if epistemic_state == EPISTEMIC_VERIFIED:
            if not supporting_ids_raw:
                raise ValueError(
                    "A memory cannot be marked verified without at least one explicitly designated "
                    "supporting Evidence (pass it via `supporting_evidence`, not just `evidence`)"
                )
            if not any(item.kind in VERIFICATION_EVIDENCE_KINDS for item in supporting_evidence):
                raise ValueError(
                    "A memory can only be marked verified with supporting evidence strong enough to "
                    f"justify it (one of {sorted(VERIFICATION_EVIDENCE_KINDS)}), not an unchecked "
                    "assertion or reference"
                )

        recorded_at = dt.datetime.now(dt.timezone.utc)
        memory = Memory(
            memory_id=memory_id,
            content=content,
            kind=kind,
            epistemic_state=epistemic_state,
            recorded_at=recorded_at,
            supersedes=supersedes,
            evidence_ids=all_evidence_ids,
            supporting_evidence_ids=supporting_evidence_ids,
        )

        events = [
            Event(
                event_id=uuid.uuid4().hex,
                kind=EVENT_KIND_MEMORY_RECORDED,
                subject_id=memory_id,
                occurred_at=recorded_at,
            )
        ]
        if supersedes is not None:
            events.append(
                Event(
                    event_id=uuid.uuid4().hex,
                    kind=EVENT_KIND_MEMORY_SUPERSEDED,
                    subject_id=supersedes,
                    occurred_at=recorded_at,
                )
            )

        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            persisted = store.add(memory, events)

        # `store.add` returns the pre-existing memory instead of writing
        # anything when this call duplicates a current one (see its
        # docstring). The freshly generated `memory_id` above is what
        # distinguishes the two outcomes: it exists nowhere else by
        # construction, so getting it back means this call is the one
        # that recorded it.
        return persisted, persisted.memory_id == memory.memory_id

    @guarded
    def recall(
        self, query: str, *, limit: int = DEFAULT_RECALL_LIMIT, include_superseded: bool = False
    ) -> list[Memory]:
        """Search persisted memories with deterministic lexical matching.

        By default only current memories are searched: superseded history
        is not what "what do we currently believe about X" should surface.
        Pass `include_superseded=True` to search the full history instead.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Recall query must not be empty or whitespace-only")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return store.search(query, limit, current_only=not include_superseded)

    @guarded
    def timeline(self, *, kind: str | None = None) -> list[Memory]:
        """Return the full recorded history, oldest first, optionally
        filtered by `kind`. Includes both current and superseded memories:
        superseding never removes anything from the timeline."""
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind {kind!r}; expected one of {sorted(VALID_KINDS)}")

        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return store.timeline(kind)

    @guarded
    def state(self, *, kind: str | None = None) -> list[Memory]:
        """Return only the currently-valid memories, oldest first, optionally
        filtered by `kind`. This is the history projected onto "what Urdyn
        currently considers true": superseded memories are excluded."""
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind {kind!r}; expected one of {sorted(VALID_KINDS)}")

        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            current_ids = store.current_ids()
            return [memory for memory in store.timeline(kind) if memory.memory_id in current_ids]

    @guarded
    def record_conflict(self, memory_a: Memory, memory_b: Memory) -> Conflict:
        """Explicitly declare that two Memories cannot both be treated as a
        coherent description of the same state, and return the canonical
        `Conflict`.

        This is a structural assertion, not a semantic judgment (A13):
        Urdyn does not evaluate whether `memory_a` and `memory_b` are
        actually incompatible, only records that the caller says so. It
        never mutates either Memory, never changes an `epistemic_state`,
        and never implies invalidation or supersession -- both memories
        keep whatever current/verified status they already had. Kinds
        need not match: any two Memories may conflict.

        Only `memory_id` is trusted from `memory_a`/`memory_b`; both ids
        are validated against the canonically persisted store (exactly
        like `promote()` trusts only `lesson.memory_id`), so an object
        that merely shares a real Memory's id but disagrees with it
        cannot smuggle in a nonexistent relation. Raises `ValueError` if
        either id does not name an existing memory, or if the two ids are
        the same (a memory cannot conflict with itself). Neither memory
        is required to be current -- see `open_conflicts()` for the
        current-state projection.

        The relation is symmetric and idempotent: `record_conflict(a, b)`,
        a repeat of the same call, and `record_conflict(b, a)` all
        resolve to the same persisted relation and never create a second
        one or change its original `recorded_at`.
        """
        recorded_at = dt.datetime.now(dt.timezone.utc)
        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            return store.add_conflict(memory_a.memory_id, memory_b.memory_id, recorded_at)

    @guarded
    def conflicts(self) -> list[Conflict]:
        """Return every declared conflict relation, oldest first: the full
        canonical history, including conflicts no longer open because one
        or both participants stopped being current. Never rewritten or
        removed by a later supersession/invalidation -- see
        `open_conflicts()` for the current-state projection."""
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return store.list_conflicts()

    @guarded
    def open_conflicts(self) -> list[Conflict]:
        """Return only the conflicts that are currently operative: a
        `Conflict` is included here if and only if BOTH memories it names
        are still current. This is a derived projection, not stored
        state -- a supersession or invalidation of either participant
        removes a conflict from here automatically, the same way it
        already disappears from `state()`, with no separate resolution
        step or status to update.

        (A14.1) `preflight()` needs this exact same "open" projection
        while it already holds a store connection and a `current_ids`
        set of its own open -- opening a second connection here just to
        recompute both would be wasteful, not more correct. The filter
        itself is factored into `_open_conflicts_projection` below so
        both call sites share ONE definition of "open" rather than two
        copies that could drift apart.
        """
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return _open_conflicts_projection(store.list_conflicts(), store.current_ids())

    @guarded
    def add_evidence(self, content: str, *, kind: str = DEFAULT_EVIDENCE_KIND) -> Evidence:
        """Persist a new piece of evidence and return it.

        Evidence is support for a belief, not the belief itself: recording
        it here does not create or alter any memory.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Evidence content must not be empty or whitespace-only")
        if kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"Unknown evidence kind {kind!r}; expected one of {sorted(VALID_EVIDENCE_KINDS)}")

        evidence = Evidence(
            evidence_id=uuid.uuid4().hex,
            content=content,
            kind=kind,
            recorded_at=dt.datetime.now(dt.timezone.utc),
        )

        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            store.add_evidence(evidence)

        return evidence

    @guarded
    def get_evidence(self, evidence_id: str) -> Evidence:
        """Resolve an evidence_id (e.g. from `Memory.evidence_ids`) to its
        persisted `Evidence`. Raises `ValueError` if unknown."""
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        evidence = None
        if store is not None:
            with store:
                evidence = store.get_evidence(evidence_id)
        if evidence is None:
            raise ValueError(f"Unknown evidence {evidence_id!r}")
        return evidence

    @guarded
    def seed(self, paths: Sequence[str | Path]) -> list[SeedResult]:
        """Record one observation of each given project file and return
        what happened to each, in the order given.

        Seeding records PROVENANCE, not knowledge: each observation
        creates one `document_observation` Evidence and links it to the
        file's stable Source identity. No Memory is created, nothing
        becomes `verified` (a `document_observation` is not a qualifying
        kind -- reading a document does not check that its claims are
        true, see `_evidence.py`), and Urdyn never interprets what the
        file says. Turning a document into a belief stays an explicit
        act: `remember(..., evidence=[result.evidence])`.

        The Evidence holds the document's TEXT VERBATIM, so a later
        reader can still see what Urdyn observed after the file has
        changed or is gone; the SHA-256 digest, the size on disk and the
        moment of observation are structured columns alongside it (see
        `_source.py`). `.urdyn/` therefore keeps a local copy of every
        document it was asked to observe.

        Per file, `SeedResult.status` is `added` (a Source Urdyn did not
        track yet), `unchanged` (the digest still matches the last
        observation -- nothing is written at all), or `changed` (a new
        observation is appended, and the previous ones are kept). Each
        file's observation is its own atomic transaction.

        Every path is validated and read BEFORE anything is persisted, so
        one unacceptable path in the list cannot leave the others
        half-recorded. Raises `UrdynSourceError` for a path that escapes
        the workspace, is not a regular UTF-8 text file, exceeds the size
        limit, or matches the credential denylist -- see
        `resolve_seed_path` for the full policy, which applies to
        explicitly named files exactly as it does to discovered ones.
        """
        if isinstance(paths, (str, Path)):
            raise TypeError(
                "seed() takes a sequence of paths, not a single path; pass [path] to seed one file"
            )

        # Validate and hash everything first: a typo in the third path
        # must not leave the first two recorded.
        candidates = [
            read_seed_candidate(self._path, resolve_seed_path(self._path, URDYN_DIRNAME, raw))
            for raw in paths
        ]
        if not candidates:
            return []

        results = []
        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            for candidate in candidates:
                status, source, evidence = store.observe_source(
                    path=candidate.path,
                    digest=candidate.digest,
                    size_bytes=candidate.size_bytes,
                    observed_at=dt.datetime.now(dt.timezone.utc),
                    candidate_source_id=uuid.uuid4().hex,
                    candidate_evidence_id=uuid.uuid4().hex,
                    evidence_content=candidate.text,
                )
                results.append(SeedResult(status=status, source=source, evidence=evidence))
        return results

    @guarded
    def sources(self) -> list[Source]:
        """Every project file Urdyn tracks, ordered by path, each with its
        full observation history (oldest first).

        Reads the recorded observations only: it does not re-hash the
        files on disk, so it reports what Urdyn saw, not what is there
        now. Deciding whether a tracked file has since changed or
        disappeared is deliberately left to a caller (or a future
        watcher) that re-seeds -- `seed()` answering `unchanged` or
        `changed` IS that check, performed explicitly rather than as a
        side effect of listing.
        """
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return store.list_sources()

    def _require_dev_discovery(self) -> None:
        if self._profile != PROFILE_DEV:
            raise ValueError(
                f"Project context discovery is only available in the {PROFILE_DEV!r} profile; "
                f"this workspace is {self._profile!r}. Seed explicit paths instead."
            )

    @guarded
    def seed_candidates(self) -> list[str]:
        """Project files a `dev` workspace could seed, as workspace-relative
        POSIX paths, sorted.

        Pure suggestion: writes nothing and creates no store. Discovery is
        a BOUNDED recursive walk (A53) -- root manifest files plus
        documentation-like files at any depth -- pruned by `.gitignore`,
        `.git/info/exclude` and a mandatory directory exclusion list, with
        every survivor passing the same eligibility checks an explicit
        seed does. See `_source.discover_seed_candidates`.

        Raises `ValueError` outside the `dev` profile: automatic project
        discovery is this profile's behaviour, and silently returning
        nothing elsewhere would look like "this project has no
        documentation" rather than "Urdyn did not look".
        """
        self._require_dev_discovery()
        return discover_candidate_paths(self._path, URDYN_DIRNAME)

    @guarded
    def seed_candidate_report(self) -> SeedCandidateReport:
        """`seed_candidates()`, split into what seeding each candidate
        WOULD do: never-seeded, tracked-and-changed, tracked-and-identical
        (see `SeedCandidateReport`).

        Writes nothing and creates no store -- `sources()` opens the
        database only if it already exists, so this stays safe to call in
        a workspace that has never recorded anything.
        """
        self._require_dev_discovery()
        candidates = discover_seed_candidates(self._path, URDYN_DIRNAME)
        latest = {source.path: source.latest_observation.digest for source in self.sources()}

        new: list[str] = []
        changed: list[str] = []
        unchanged: list[str] = []
        for candidate in candidates:
            known = latest.get(candidate.path)
            if known is None:
                new.append(candidate.path)
            elif known == candidate.digest:
                unchanged.append(candidate.path)
            else:
                changed.append(candidate.path)
        return SeedCandidateReport(
            new=tuple(new), changed=tuple(changed), unchanged=tuple(unchanged)
        )

    @guarded
    def tracked_scope(self) -> frozenset[str]:
        """The CHEAP half of `watcher_scope()`: paths already tracked as a
        `Source`. One indexed read of canonical data, no filesystem walk
        at all -- this is what the watcher may safely recompute on every
        poll tick (see `_watcher.py`'s scope-cadence note)."""
        return frozenset(source.path for source in self.sources())

    @guarded
    def discovered_scope(self) -> frozenset[str]:
        """The EXPENSIVE half of `watcher_scope()`: the bounded recursive
        discovery walk. Separated from `tracked_scope()` so a caller that
        must run often (the watcher) can run this one rarely.

        Not profile-gated, unlike `seed_candidates()`: this is the
        watcher's own building block, and the watcher is already
        `dev`-only by construction. Gating it here would change
        `watcher_scope()` from "returns a set" to "raises" for every
        non-dev caller, which no caller expects."""
        return frozenset(discover_candidate_paths(self._path, URDYN_DIRNAME))

    @guarded
    def watcher_scope(self) -> frozenset[str]:
        """Workspace-relative paths the Dev watcher may observe:
        every path already tracked as a `Source`, unioned with everything
        discovery currently proposes.

        Bounded, but no longer free: since A53 the discovery half walks
        the tree (pruned, capped, and ignore-filtered -- see
        `_source.discover_seed_candidates`). Nothing is cached HERE, so
        this stays the always-correct answer; a caller on a hot path
        should combine `tracked_scope()` with a less frequently refreshed
        `discovered_scope()` instead, which is exactly what the watcher
        does.
        """
        return frozenset(self.tracked_scope() | self.discovered_scope())

    @guarded
    def learn(
        self,
        content: str,
        *,
        evidence: Sequence[Evidence] = (),
        supporting_evidence: Sequence[Evidence] = (),
        verified: bool = False,
        supersedes: str | None = None,
    ) -> Memory:
        """Persist a `Lesson`: a reusable conclusion drawn from experience.

        A lesson is a `Memory` of kind `lesson`. By default it is recorded
        as a candidate (`user_asserted`); pass `verified=True` together
        with confirming `supporting_evidence` (e.g. a test result
        explicitly designated as supporting this lesson -- see
        `remember()`'s docstring for the exact A12.1 contract) to record
        it as a verified lesson instead. `evidence` alone, even of a
        qualifying kind, is not enough for `verified=True` since A12.1.
        A candidate can later be superseded by a verified version of the
        same lesson via `supersedes`.
        """
        epistemic_state = EPISTEMIC_VERIFIED if verified else EPISTEMIC_USER_ASSERTED
        return self.remember(
            content,
            kind=KIND_LESSON,
            epistemic_state=epistemic_state,
            supersedes=supersedes,
            evidence=evidence,
            supporting_evidence=supporting_evidence,
        )

    @guarded
    def record_attempt(
        self,
        *,
        task: str,
        approach: str,
        outcome: str,
        evidence: Sequence[Evidence] = (),
    ) -> Attempt:
        """Persist a record of trying to do something and return it.

        Attempts are append-only: recording a later successful attempt
        never rewrites or removes an earlier failed one.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Attempt task must not be empty or whitespace-only")
        if not isinstance(approach, str) or not approach.strip():
            raise ValueError("Attempt approach must not be empty or whitespace-only")
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Unknown attempt outcome {outcome!r}; expected one of {sorted(VALID_OUTCOMES)}")

        recorded_at = dt.datetime.now(dt.timezone.utc)
        attempt = Attempt(
            attempt_id=uuid.uuid4().hex,
            task=task,
            approach=approach,
            outcome=outcome,
            recorded_at=recorded_at,
            evidence_ids=tuple(dict.fromkeys(item.evidence_id for item in evidence)),
        )
        event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_ATTEMPT_RECORDED,
            subject_id=attempt.attempt_id,
            occurred_at=recorded_at,
        )

        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            store.add_attempt(attempt, event)

        return attempt

    def _gather_experience(self, task: str) -> "_ExperienceMaterial":
        """Single read of canonical and derived-semantic state for
        `task`, shared by `preflight()` and `context()` (A29.1) so both
        answer from the SAME store snapshot, the SAME `timeline(None)`
        materialization, and exactly ONE semantic-lifecycle pass
        (`_semantic_prepare` -- A27's consumer-boundary contract): a
        second call here would mean a second incremental-refresh attempt
        and a second freshness classification for what is, from the
        caller's perspective, one answer to one task.

        This method only gathers current-state-filtered candidates and
        admits them through the existing lexical/FTS/semantic/provenance
        channels -- identical to what `preflight()` alone did before
        A29.1. It decides nothing about composition (which categories a
        caller shows, in what shape, under what budget): that policy
        lives in the caller (`preflight()`'s call to `build_preflight`,
        `context()`'s call to `compile_context`), never here.
        """
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            # (A27) Even a workspace with nothing recorded reports HOW it
            # looked: "nothing found" and "nothing could be found" are
            # different answers and must not print identically.
            return _ExperienceMaterial.empty(retrieval=self.semantic_state())
        with store:
            current_ids = store.current_ids()
            # [A14.1] ONE `timeline(None)` read instead of four separate
            # `timeline(kind)` calls: `timeline()` already reads and
            # materializes every Memory internally regardless of `kind`
            # (the parameter only filters the result), so the four
            # earlier calls were re-reading the same underlying data four
            # times. Partitioning this single, current-filtered list by
            # kind in Python is exactly equivalent -- filtering by
            # `current_ids` and filtering by `kind` are independent,
            # order-preserving passes, so partitioning after the fact
            # yields the identical ids in the identical (event-log)
            # order as calling `timeline(kind)` per kind would have (see
            # `TestTimelinePartitioningEquivalence` in
            # `test_preflight_conflicts.py`). `current_memory_by_id` is a
            # byproduct of the same single read: it is ALSO the
            # participant map open conflicts need below, so no second
            # materialization and no per-kind lookup is required to
            # build it.
            all_memories = store.timeline(None)
            current_memories = [m for m in all_memories if m.memory_id in current_ids]
            current_memory_by_id = {m.memory_id: m for m in current_memories}

            root_cause_memories = [m for m in current_memories if m.kind == KIND_ROOT_CAUSE]
            lesson_memories = [m for m in current_memories if m.kind == KIND_LESSON]
            verified_lesson_memories = [m for m in lesson_memories if m.epistemic_state == EPISTEMIC_VERIFIED]
            # [A9.1] Every CURRENT project-wide invariant, unfiltered by
            # task relevance -- see `Preflight.invariants`'s docstring for
            # why `preflight()` bypasses the lexical/FTS/semantic channels
            # for this list. `context()` (A29.1) does NOT inherit that
            # bypass: it filters this same list for task relevance itself
            # (see `Urdyn.context`), which is the whole point of keeping
            # it as one materialized list here rather than pre-deciding
            # its treatment inside this method.
            invariant_memories = [m for m in current_memories if m.kind == KIND_INVARIANT]
            # [A11.3] Every CURRENT invalidation -- UNLIKE invariants, this
            # goes through the same relevance channels as root causes/
            # lessons below (see `Preflight.open_invalidations`'s docstring
            # for why it does not inherit the invariants' "always include"
            # rule: an invalidation is not project-wide by default).
            invalidation_memories = [m for m in current_memories if m.kind == KIND_INVALIDATION]
            # [A22.1] Every CURRENT pending -- like invalidations, and
            # UNLIKE invariants, this goes through the same relevance
            # channels as root causes/lessons (see `Preflight.pending`).
            # Deliberately a pool of its own rather than a `kind` filter
            # applied to the shared memory pool: everything below that
            # ranks or restricts candidates must keep treating pending as
            # a separate population (see the semantic pool further down).
            pending_memories = [m for m in current_memories if m.kind == KIND_PENDING]
            # [A29.1] Every CURRENT decision. `preflight()` never reads
            # this list (see `_preflight.py`'s module docstring for why a
            # Decision is not "prior experience"); `context()` filters it
            # for task relevance exactly like pending/invalidations. Kept
            # here, materialized unconditionally, because it costs one
            # more list comprehension over data already in memory and lets
            # both consumers share this single gathering pass regardless
            # of which one a given call turns out to be.
            decision_memories = [m for m in current_memories if m.kind == KIND_DECISION]
            # [A14.1] Every OPEN conflict (see `_open_conflicts_projection`
            # -- the SAME definition `Urdyn.open_conflicts()` uses, over
            # the SAME `current_ids` already computed above). Relevance
            # filtering happens inside `build_preflight`, not here.
            open_conflict_list = _open_conflicts_projection(store.list_conflicts(), current_ids)
            attempts = store.list_attempts()

            # [A29.1] `build_preflight`'s `evidence_lookup` is called with
            # ids that only ever come from `verified_lesson_memories`/
            # `attempts` (its `candidate_evidence_ids` loop draws
            # exclusively from `verified_lessons`/`relevant_successes`,
            # both subsets of these two lists -- see `_preflight.py`).
            # This method's `with store:` block closes before its
            # RETURN VALUE is used by its callers, so a lookup closure
            # bound to `store` would try to query a closed connection the
            # moment `preflight()`/`context()` call `build_preflight`.
            # Materializing this bounded superset here, while `store` is
            # still open, and handing back a plain dict lookup instead,
            # is what keeps `evidence_lookup` usable after this method
            # returns without reopening or holding a second connection.
            evidence_ids_to_resolve: set[str] = set()
            for memory in verified_lesson_memories:
                evidence_ids_to_resolve.update(memory.evidence_ids)
            for attempt in attempts:
                evidence_ids_to_resolve.update(attempt.evidence_ids)
            evidence_by_id: dict[str, Evidence] = {}
            for evidence_id in evidence_ids_to_resolve:
                evidence = store.get_evidence(evidence_id)
                if evidence is None:
                    raise UrdynStorageError(
                        f"Evidence {evidence_id!r} is referenced by recorded experience but "
                        "missing from the store"
                    )
                evidence_by_id[evidence_id] = evidence

            def _must_get_evidence(evidence_id: str) -> Evidence:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    raise UrdynStorageError(
                        f"Evidence {evidence_id!r} is referenced by recorded experience but "
                        "missing from the store"
                    )
                return evidence

            query_tokens = frozenset(_tokens(task))
            attempt_fts_candidates = store.search_candidates(query_tokens, ENTITY_ATTEMPT)
            memory_fts_candidates = store.search_candidates(query_tokens, ENTITY_MEMORY)
            # The current (latest-observation) Evidence of every
            # seeded Source -- read once here, alongside the other
            # single-read materializations this method already does, and
            # handed to both `_semantic_pool_entries` (below) and
            # `Urdyn.context()`'s PROJECT EVIDENCE pool (via
            # `_ExperienceMaterial`) so neither re-reads the store.
            current_source_evidence = store.list_current_source_evidence()
            source_fts_candidates = store.search_candidates(query_tokens, ENTITY_SOURCE)

            # [A7.7] Consumer-specific eligibility, computed here (this is
            # exactly the same current+verified filtering already applied
            # two lines above to build root_cause_memories/
            # verified_lesson_memories -- reused, not duplicated) and
            # passed down as a plain id set: the semantic ranking pool for
            # MEMORY is restricted to these ids BEFORE ranking, so an
            # ineligible memory (not current, or an unverified lesson) can
            # no longer win the pool's single admission slot and starve a
            # genuinely usable candidate of consideration. Attempts have no
            # eligibility concept in preflight() (a succeeded attempt still
            # legitimately contributes recommended_validation evidence), so
            # that pool is intentionally left unrestricted.
            memory_eligible_ids = frozenset(m.memory_id for m in (*root_cause_memories, *verified_lesson_memories))

            # [A27] Bring the derived index up to date with what the
            # canonical store now holds, BEFORE any pool is ranked --
            # once per call, not once per pool. This is the step whose
            # absence produced A26: experience recorded after the last
            # index build was invisible to every semantic pool below, and
            # nothing said so. `retrieval` carries the outcome (current,
            # just-refreshed, degraded, or never enabled) all the way out
            # to the caller.
            retrieval = self._semantic_prepare(
                _semantic_pool_entries(
                    store, memories=all_memories, attempts=attempts, source_evidence=current_source_evidence
                )
            )

            # [A27] `eligible_ids` here is NOT a new admission policy: it
            # is the set of attempts that canonically EXIST, and it is a
            # no-op whenever the index holds nothing else (the normal
            # case, since canonical storage is append-only). It matters
            # when the index holds vectors for ids the canonical store no
            # longer has -- reachable by restoring/replacing `memory.db`
            # under a retained `semantic_index.db`. This pool is
            # winner-take-all, so such a vector could take its single
            # admission slot, or collapse the runner-up margin, and
            # starve a real attempt of consideration: the A7.7 starvation
            # bug, arriving from the derived side instead of the
            # eligibility side. Filtering the pool to canonical ids costs
            # nothing and removes it, which is why A27 does NOT need the
            # index to be garbage-collected down to exact equality with
            # canonical state.
            attempt_semantic_admitted = self._semantic_widen(
                task, ENTITY_ATTEMPT, eligible_ids=frozenset(a.attempt_id for a in attempts)
            )
            memory_semantic_admitted = self._preflight_memory_semantic_widen(
                task,
                root_cause_memories=root_cause_memories,
                verified_lesson_memories=verified_lesson_memories,
                memory_eligible_ids=memory_eligible_ids,
            ) or self._preflight_corroboration_admitted(
                task,
                root_cause_memories=root_cause_memories,
                verified_lesson_memories=verified_lesson_memories,
                attempts=attempts,
                memory_eligible_ids=memory_eligible_ids,
            )

            # [A11.3] A DISJOINT, independently-ranked semantic pool,
            # restricted to invalidation ids only. Deliberately NOT folded
            # into `memory_eligible_ids` above: that pool feeds a
            # winner-take-all race (`_preflight_memory_semantic_widen`/
            # `_preflight_corroboration_admitted`), and an unrelated but
            # strongly-scoring invalidation entering the SAME race could
            # win the pool's single admission slot away from a genuine
            # root cause/lesson, or collapse its margin -- a real
            # regression, verified directly against the unmodified A7.8
            # machinery in `test_preflight_invalidations.py`'s semantic
            # competition gate. Running this as a separate call over a
            # separate, invalidation-only `eligible_ids` restriction means
            # it can never compete against root causes/lessons for the
            # same slot, by construction. The two independently-computed
            # results are combined by plain set union below: since neither
            # race's outcome depended on the other, unioning cannot
            # reintroduce the competition this separation avoids.
            invalidation_eligible_ids = frozenset(m.memory_id for m in invalidation_memories)
            invalidation_semantic_admitted = self._semantic_widen(
                task, ENTITY_MEMORY, eligible_ids=invalidation_eligible_ids
            )

            # [A22.1] A third DISJOINT, independently-ranked semantic pool,
            # restricted to pending ids only -- the A11.3 pattern above,
            # applied for exactly the same reason and no other. Folding
            # pending into `memory_eligible_ids` instead would put it into
            # the winner-take-all race that already decides root causes/
            # lessons, where an unrelated-but-strongly-scoring pending
            # could take that pool's single admission slot or collapse its
            # margin -- the regression A11.3 documents and measures. Run
            # as its own call over its own restriction, that is impossible
            # by construction, in both directions: pending cannot steal
            # admission from root causes/lessons/invalidations, and none
            # of them can steal it from pending. The three independently
            # computed results are combined by plain set union below;
            # since no race's outcome depended on any other, and the three
            # restrictions are disjoint by `kind`, unioning cannot
            # reintroduce the competition this separation avoids.
            pending_eligible_ids = frozenset(m.memory_id for m in pending_memories)
            pending_semantic_admitted = self._semantic_widen(
                task, ENTITY_MEMORY, eligible_ids=pending_eligible_ids
            )

            # [A23.1] A fourth DISJOINT, independently-ranked semantic
            # pool, restricted to the verified lessons already computed
            # above -- the A11.3/A22.1 pattern, but for a different
            # reason and with a different ADMISSION POLICY. A11.3 and
            # A22.1 separated POOLS (who competes with whom) so one
            # category could not steal another's single slot; they left
            # the POLICY (how candidates are admitted inside a pool)
            # identical everywhere: winner-take-all plus margin. A23
            # measured that this policy is itself wrong for lessons, and
            # that pool separation alone cannot fix it, because the
            # competition is between lessons and each other: two
            # complementary verified lessons scoring 0.3782 and 0.3206
            # against the same task were rejected TOGETHER for being
            # 0.0576 apart, and a realistic four-lesson workspace
            # returned zero or one of four. `set_admitted_ids` reuses the
            # SAME absolute floor and the SAME ranking, drops only the
            # margin, and is bounded by an internal cap.
            #
            # Deliberately ADDITIVE rather than a move: verified lessons
            # stay in `memory_eligible_ids` above as well. Removing them
            # from that pool would look tidier, but it would destroy the
            # A7.8 shared-Evidence cluster rescue, under which a lesson
            # whose OWN score is below the floor is admitted because its
            # sibling root cause (same experience, same Evidence) cleared
            # it -- measured by
            # `test_preflight_admits_low_scoring_sibling_via_cluster_membership`.
            # A lesson can therefore be admitted by either channel; since
            # both only ever contribute ids to the same union below,
            # running both cannot re-create competition, and no memory
            # candidate can consume this pool's capacity because this
            # pool contains lessons only.
            #
            # [A23.4] What that other channel may admit a lesson FOR is
            # now bounded, though: on its own score it must clear this
            # pool's floor, not MEMORY's lower one. Otherwise the floor
            # and cap calibrated here would hold inside this pool only,
            # which A23.2 measured happening on a real query.
            lesson_eligible_ids = frozenset(m.memory_id for m in verified_lesson_memories)
            lesson_semantic_admitted = self._preflight_lesson_semantic_admitted(
                task, lesson_eligible_ids=lesson_eligible_ids
            )

            return _ExperienceMaterial(
                attempts=attempts,
                root_cause_memories=root_cause_memories,
                verified_lesson_memories=verified_lesson_memories,
                invariant_memories=invariant_memories,
                invalidation_memories=invalidation_memories,
                pending_memories=pending_memories,
                decision_memories=decision_memories,
                open_conflict_list=open_conflict_list,
                current_memory_by_id=current_memory_by_id,
                evidence_lookup=_must_get_evidence,
                attempt_fts_candidates=attempt_fts_candidates,
                memory_fts_candidates=memory_fts_candidates,
                attempt_semantic_admitted=attempt_semantic_admitted,
                memory_semantic_admitted=(
                    memory_semantic_admitted
                    | invalidation_semantic_admitted
                    | pending_semantic_admitted
                    | lesson_semantic_admitted
                ),
                retrieval=retrieval,
                current_source_evidence=current_source_evidence,
                source_fts_candidates=source_fts_candidates,
                search_index=store.search_index_presence(),
            )

    @_inference_outside_lock("task")  # encoder work planned, then run without the lock
    @guarded  # both passes of the call run under ONE lifecycle hold
    @_semantic_outcome_guarded
    def preflight(self, task: str) -> Preflight:
        """Select prior experience relevant to `task`, before starting it.

        Answers "what should an agent know before attempting this?" by
        surfacing known failures (matching failed attempts), root causes,
        verified lessons, still-open pending work, and any test/command
        evidence recommended as validation — each only if Urdyn has
        something relevant on record. This is lexical and deterministic,
        not a search engine: it will not return everything, and it will
        not return nothing just because the wording differs slightly.

        Unbounded and unbudgeted by design: every relevant item is
        returned. For a compact, budgeted answer to a narrower question
        ("what must this task respect right now"), see `context()`.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Preflight task must not be empty or whitespace-only")

        material = self._gather_experience(task)
        return build_preflight(
            task,
            attempts=material.attempts,
            root_cause_memories=material.root_cause_memories,
            verified_lesson_memories=material.verified_lesson_memories,
            evidence_lookup=material.evidence_lookup,
            attempt_fts_candidates=material.attempt_fts_candidates,
            memory_fts_candidates=material.memory_fts_candidates,
            attempt_semantic_admitted=material.attempt_semantic_admitted,
            memory_semantic_admitted=material.memory_semantic_admitted,
            invariant_memories=material.invariant_memories,
            invalidation_memories=material.invalidation_memories,
            pending_memories=material.pending_memories,
            open_conflicts=material.open_conflict_list,
            conflict_participants=material.current_memory_by_id,
            retrieval=material.retrieval,
            search_index=material.search_index,
        )

    @_inference_outside_lock("task")  # encoder work planned, then run without the lock
    @guarded  # both passes of the call run under ONE lifecycle hold
    @_semantic_outcome_guarded
    def context(
        self,
        task: str,
        *,
        budget: int = DEFAULT_CONTEXT_BUDGET,
        _project_evidence_trace: list[ProjectEvidenceTrace] | None = None,
    ) -> CompiledContext:
        """Compile the smallest budgeted working context relevant to
        `task` -- not "what does Urdyn know" (`preflight()`), but "what
        must an agent respect right now to start this task safely".

        Shares its ENTIRE retrieval pipeline with `preflight()`
        (`_gather_experience`, the same `_semantic_prepare` consumer-
        boundary lifecycle from A27, the same lexical/FTS/semantic/
        provenance admission from `_preflight.py`), so the two can never
        silently disagree about what is relevant to `task`. What differs
        is composition, decided entirely in `_context.compile_context`:

        - a real character `budget` (see `DEFAULT_CONTEXT_BUDGET`), with
          candidates admitted in one fixed cross-category priority order
          until the next one no longer fits;
        - current, project-wide Invariants are filtered for task
          relevance instead of included unconditionally (A9.1's
          `Preflight.invariants` bypass is deliberately NOT reused here);
        - current Decisions are admitted, a Memory kind `preflight()`
          never reads at all;
        - an Attempt sharing Evidence with an already-relevant RootCause
          is cited on that RootCause's line instead of costing a line of
          its own;
        - a seeded Source's CURRENT observation surfaces, task-
          relevance permitting, as PROJECT EVIDENCE -- raw, unverified
          document text, never a Memory (`Source != Evidence != Memory`
          holds all the way to this method's output; see
          `_context.compile_context`'s docstring). `preflight()` never
          reads Source/Evidence at all. A document too large to
          admit whole is offered as its own relevance-ranked PARAGRAPHS
          instead (see `_preflight.ordered_project_evidence`/`_chunk.py`),
          a purely derived, never-persisted view of the SAME current
          observation -- never a second, competing representation of it.

        Never mutates canonical state and is never itself persisted:
        `CompiledContext` is derived and reconstructible from canonical
        state plus `task` and `budget` alone, exactly like `Preflight`.

        `_project_evidence_trace` is INTERNAL instrumentation and not
        part of this method's contract: pass a list and it is filled with
        one `ProjectEvidenceTrace` per PROJECT EVIDENCE candidate
        considered (admission channels, lexical shared-token count,
        semantic score, final position, selected or omitted-for-budget).
        It is an out-parameter rather than a second return value
        precisely so `CompiledContext`'s shape -- a stable, rendered,
        CLI-visible contract -- is unchanged, and it is deliberately
        unreachable from the CLI. Leave it as `None` (the default) and
        nothing about this call differs.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Context task must not be empty or whitespace-only")
        if budget <= 0:
            raise ValueError("Context budget must be a positive integer")

        material = self._gather_experience(task)
        preflight_view = build_preflight(
            task,
            attempts=material.attempts,
            root_cause_memories=material.root_cause_memories,
            verified_lesson_memories=material.verified_lesson_memories,
            evidence_lookup=material.evidence_lookup,
            attempt_fts_candidates=material.attempt_fts_candidates,
            memory_fts_candidates=material.memory_fts_candidates,
            attempt_semantic_admitted=material.attempt_semantic_admitted,
            memory_semantic_admitted=material.memory_semantic_admitted,
            invariant_memories=material.invariant_memories,
            invalidation_memories=material.invalidation_memories,
            pending_memories=material.pending_memories,
            open_conflicts=material.open_conflict_list,
            conflict_participants=material.current_memory_by_id,
            retrieval=material.retrieval,
            search_index=material.search_index,
        )

        # [A29.1] Reuses `build_preflight`'s OWN selection for four of
        # six categories (`root_causes`, `verified_lessons`, `pending`,
        # `known_failures`/`recommended_validation`) rather than
        # re-deriving them: `preflight_view` already IS the relevant-
        # candidate answer `compile_context` needs for those, computed by
        # the exact same admission policy `preflight()` itself returns.
        relevance = build_relevance_context(
            task,
            attempts=material.attempts,
            attempt_fts_candidates=material.attempt_fts_candidates,
            memory_fts_candidates=material.memory_fts_candidates,
            attempt_semantic_admitted=material.attempt_semantic_admitted,
            source_evidence_fts_candidates=material.source_fts_candidates,
        )

        # [A29.1] Two more DISJOINT semantic pools, the A11.3/A22.1
        # pattern: `invariant`/`decision` candidates never compete with
        # root causes/lessons/pending/invalidations for a shared
        # admission slot. Cheap relative to the model load already paid
        # for by `_gather_experience`'s `_semantic_prepare` call above --
        # each is a brute-force rank over already-cached vectors, no
        # second model load (see `_semantic._model_cache`).
        # [A31.2] The two pools no longer share an admission POLICY:
        # invariants are a set-valued category and get their own, while
        # decisions keep the single-winner one.
        invariant_eligible_ids = frozenset(m.memory_id for m in material.invariant_memories)
        invariant_semantic_admitted = self._context_invariant_semantic_admitted(
            task, invariant_eligible_ids=invariant_eligible_ids
        )
        decision_eligible_ids = frozenset(m.memory_id for m in material.decision_memories)
        decision_semantic_admitted = self._semantic_widen(task, ENTITY_MEMORY, eligible_ids=decision_eligible_ids)

        # A fifth disjoint pool: the current-observation Evidence of
        # every seeded Source. [A54.3] `current_source_evidence` is passed
        # through directly (the same list `ordered_project_evidence` below
        # already receives) rather than a precomputed id set, because
        # Source-level semantic admission now needs to derive each
        # Source's OWN current chunk ids -- see
        # `_context_evidence_semantic_admitted`. A superseded observation
        # still can never win a slot here: it is simply absent from
        # `material.current_source_evidence` in the first place (A7.7,
        # unchanged).
        evidence_semantic_admitted, evidence_semantic_scores = self._context_evidence_semantic_admitted(
            task, current_source_evidence=material.current_source_evidence
        )
        # [A53.1] Ordered by RELEVANCE, not by workspace-relative path.
        # `list_current_source_evidence` returns Sources alphabetically
        # and `compile_context` admits by a prefix scan that stops at the
        # first candidate that does not fit, so consuming that order
        # directly handed the whole budget to whichever admitted document
        # happened to sort first. See
        # `_preflight.ordered_project_evidence` for the ordering key and
        # for why this is a permutation of the same candidate set rather
        # than a new admission rule.
        project_evidence_candidates = ordered_project_evidence(
            relevance.query_tokens,
            material.current_source_evidence,
            fts_admitted_ids=relevance.source_evidence_fts_admitted,
            semantic_admitted_ids=evidence_semantic_admitted,
            semantic_scores=evidence_semantic_scores,
        )
        # [A54] Minimum sufficient context: drop candidates that add no
        # NEW query-token coverage over higher-ranked ones already kept,
        # independent of `budget` -- see
        # `_preflight.minimum_sufficient_project_evidence`. Runs strictly
        # before `compile_context`'s own budget prefix-scan, which is
        # unchanged.
        sufficient_project_evidence = minimum_sufficient_project_evidence(
            relevance.query_tokens, project_evidence_candidates
        )
        relevant_project_evidence = tuple(
            (candidate.evidence, candidate.source_path, candidate.chunk)
            for candidate in sufficient_project_evidence
        )

        def _relevant(memory: Memory, semantic_admitted: frozenset[str]) -> bool:
            return memory_is_relevant(
                relevance.query_tokens,
                memory,
                fts_admitted_ids=relevance.memory_fts_admitted,
                semantic_admitted_ids=semantic_admitted,
                relevant_attempt_evidence_ids=relevance.relevant_attempt_evidence_ids,
            )

        # [A29.1] THE discriminant from `preflight()`'s own `invariants`
        # field: current invariants are filtered for task relevance here,
        # never included unconditionally. `invariants_excluded` makes the
        # exclusion itself visible rather than silent.
        relevant_invariants = tuple(m for m in material.invariant_memories if _relevant(m, invariant_semantic_admitted))
        relevant_decisions = tuple(m for m in material.decision_memories if _relevant(m, decision_semantic_admitted))

        compiled = compile_context(
            task=task,
            budget=budget,
            invariants=relevant_invariants,
            invariants_excluded=len(material.invariant_memories) - len(relevant_invariants),
            pending=preflight_view.pending,
            lessons=preflight_view.verified_lessons,
            decisions=relevant_decisions,
            root_causes=preflight_view.root_causes,
            known_failures=preflight_view.known_failures,
            recommended_validation_candidates=preflight_view.recommended_validation,
            open_conflicts=material.open_conflict_list,
            retrieval=material.retrieval,
            project_evidence=relevant_project_evidence,
            search_index=material.search_index,
        )
        if _project_evidence_trace is not None:
            sufficient_keys = frozenset(
                (candidate.source_path, candidate.chunk.chunk_index) for candidate in sufficient_project_evidence
            )
            _project_evidence_trace.extend(
                trace_project_evidence(
                    project_evidence_candidates,
                    _selected_project_evidence_keys(compiled),
                    sufficient_keys=sufficient_keys,
                )
            )
        return compiled

    @guarded
    def promote(
        self,
        lesson: Memory,
        *,
        name: str,
        purpose: str,
        steps: Sequence[str],
        conditions: Sequence[str] = (),
    ) -> Skill:
        """Explicitly turn a Lesson into a reusable Skill: a named,
        ordered procedure, not just a restated conclusion.

        Promotion is never automatic and never implicit: it always names
        the Lesson it comes from (`lesson`, whose `memory_id` must belong
        to an existing lesson memory Urdyn actually has on record) and
        always requires the caller to write the procedure out (`steps`)
        rather than reusing the lesson's own sentence as-is.

        `lesson` only supplies which memory_id to promote from. Nothing
        else about the object the caller passed in is trusted: the
        resulting Skill's `verification_state` and `evidence_ids` are
        derived from the CANONICAL Lesson Urdyn has actually persisted
        under that id, not from `lesson.epistemic_state` or
        `lesson.evidence_ids` as given here. A `Memory` object that
        happens to share a real Lesson's `memory_id` but disagrees with
        it (a stale copy, or a forged one) cannot elevate a Skill to
        `verified` or redirect its provenance — it is `verified` only if
        the *persisted* Lesson is itself `verified`, and `candidate`
        otherwise. This is the same epistemic honesty `remember()`
        already enforces, extended to Skill instead of duplicated for it.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Skill name must not be empty or whitespace-only")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Skill purpose must not be empty or whitespace-only")

        if isinstance(steps, (str, bytes)):
            raise ValueError("Skill steps must be a sequence of strings, not a single str or bytes")
        steps = tuple(steps)
        if not steps:
            raise ValueError("Skill steps must not be empty")
        for step in steps:
            if not isinstance(step, str) or not step.strip():
                raise ValueError("Skill steps must not contain empty or whitespace-only entries")

        if isinstance(conditions, (str, bytes)):
            raise ValueError("Skill conditions must be a sequence of strings, not a single str or bytes")
        conditions = tuple(conditions)
        for condition in conditions:
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError("Skill conditions must not contain empty or whitespace-only entries")

        skill_id = uuid.uuid4().hex
        recorded_at = dt.datetime.now(dt.timezone.utc)
        event = Event(
            event_id=uuid.uuid4().hex,
            kind=EVENT_KIND_SKILL_PROMOTED,
            subject_id=skill_id,
            occurred_at=recorded_at,
        )

        with MemoryStore.open_existing(self._db_path, brain_id=self._urdyn_id) as store:
            skill = store.add_skill(
                skill_id,
                name=name,
                purpose=purpose,
                steps=steps,
                conditions=conditions,
                source_lesson_id=lesson.memory_id,
                recorded_at=recorded_at,
                event=event,
            )

        return skill

    @guarded
    def get_skill(self, skill_id: str) -> Skill:
        """Resolve a skill_id to its persisted `Skill`. Raises `ValueError`
        if unknown."""
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        skill = None
        if store is not None:
            with store:
                skill = store.get_skill(skill_id)
        if skill is None:
            raise ValueError(f"Unknown skill {skill_id!r}")
        return skill

    @guarded
    def skills(self) -> list[Skill]:
        """Return every recorded Skill, oldest first."""
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return []
        with store:
            return store.list_skills()

    @_inference_outside_lock("action")  # encoder work planned, then run without the lock
    @guarded  # both passes of the call run under ONE lifecycle hold
    @_semantic_outcome_guarded
    def guard(self, action: str) -> GuardResult:
        """Check whether prior experience directly bears on an action
        about to be taken.

        Answers a narrower question than `preflight()`: not "what is
        worth knowing before this task" but "is there a known risk or an
        applicable Skill for this specific action". A known failure is
        only reported here if it shares Evidence with a Skill that
        matches `action`; lexical relevance to a failed attempt alone is
        not enough to produce a guard warning the way it is for
        `preflight()`. This is advisory only: it never blocks, mutates,
        or executes anything, only reports what Urdyn found.
        """
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Guard action must not be empty or whitespace-only")

        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return GuardResult(
                action=action,
                known_failures=(),
                applicable_skills=(),
                recommended_validation=(),
                retrieval=self.semantic_state(),
            )
        with store:
            skills = store.list_skills()
            attempts = store.list_attempts()

            def _must_get_evidence(evidence_id: str) -> Evidence:
                evidence = store.get_evidence(evidence_id)
                if evidence is None:
                    raise UrdynStorageError(
                        f"Evidence {evidence_id!r} is referenced by a recorded skill but "
                        "missing from the store"
                    )
                return evidence

            query_tokens = frozenset(_tokens(action))
            skill_fts_candidates = store.search_candidates(query_tokens, ENTITY_SKILL)
            attempt_fts_candidates = store.search_candidates(query_tokens, ENTITY_ATTEMPT)

            # [A7.7] Consumer-specific eligibility for guard()'s attempt
            # pool: guard()'s known_failures can only ever be built from a
            # FAILED attempt (see _guard.py's own generator condition,
            # reused here rather than re-decided) -- a succeeded attempt
            # can never appear in guard()'s output no matter how relevant,
            # so it must not be allowed to occupy the semantic pool's
            # single admission slot either. Skills have no eligibility
            # concept in guard() (verification_state is reported, never
            # gated on), so that pool is intentionally left unrestricted.
            failed_attempt_ids = frozenset(a.attempt_id for a in attempts if a.outcome == OUTCOME_FAILED)

            # [A27] `guard()` consumes the semantic channel directly (both
            # pools below), so it gets the same lifecycle as `preflight()`
            # through the same helper -- not for symmetry, but because an
            # index that predates the Skill an action is about to violate
            # fails here in the identical silent way. Same refresh, same
            # retrieval reporting, one implementation.
            retrieval = self._semantic_prepare(
                _semantic_pool_entries(store, attempts=attempts, skills=skills)
            )

            # [A27] Canonical-existence filter, exactly as in
            # `preflight()`'s attempt pool and for the same reason (see
            # the comment there): a no-op unless the index holds vectors
            # for skills the canonical store no longer has, in which case
            # it stops a phantom from taking this winner-take-all pool's
            # single slot.
            skill_semantic_admitted = self._semantic_widen(
                action, ENTITY_SKILL, eligible_ids=frozenset(s.skill_id for s in skills)
            )
            attempt_semantic_admitted = self._semantic_widen(action, ENTITY_ATTEMPT, eligible_ids=failed_attempt_ids)

            return build_guard_result(
                action,
                skills=skills,
                attempts=attempts,
                evidence_lookup=_must_get_evidence,
                skill_fts_candidates=skill_fts_candidates,
                attempt_fts_candidates=attempt_fts_candidates,
                skill_semantic_admitted=skill_semantic_admitted,
                attempt_semantic_admitted=attempt_semantic_admitted,
                retrieval=retrieval,
                search_index=store.search_index_presence(),
            )

    def _count_memories(self) -> int:
        """Return the number of persisted memories, or 0 if none exist yet.

        Internal to the `urdyn status` CLI command. Not part of the public
        API: the future semantics of "count" (current vs. superseded vs.
        invalidated memories) are not yet stable enough to commit to.
        """
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return 0
        with store:
            return store.count()

    @property
    def _db_path(self) -> Path:
        return db_path_for(self._path / URDYN_DIRNAME)

    @property
    def _semantic_db_path(self) -> Path:
        return semantic_db_path_for(self._path / URDYN_DIRNAME)

    # -- semantic retrieval (A7.4, optional) -------------------------------

    def semantic_setup(self) -> SemanticSetupResult:
        """(Re)build the derived semantic index for this workspace from
        canonical data: attempts (task+approach), all memories (content),
        skills (name+purpose+conditions), and the current
        observation of every seeded Source (its Evidence content) -- see
        `_semantic_pool_entries` for the exact definition. Always safe to
        call again: fully rebuilds from scratch every time (idempotent),
        which is also how a stale or model-mismatched index gets fixed.

        Raises `UrdynSemanticUnavailableError` if the `[semantic]` extra
        is not installed, or if the semantic model itself cannot be
        acquired or loaded -- this is the one semantic entry point allowed
        to fail loudly and to touch the network (to download the model if
        it is not already cached); `preflight()`/`guard()` never do
        either.

        A failure here leaves any previously built index exactly as it
        was: the model is loaded BEFORE `begin_rebuild()` clears anything,
        and a failure part-way through the rebuild itself leaves the index
        at `status='building'`, which `_semantic_context` refuses to use.
        Either way the outcome is a workspace that falls back to
        lexical/FTS, never one that reads a half-built index.

        The same states, without holding the lifecycle lock while
        the model loads or embeds: (1) checks under the lock; (2) model load
        and dimension probe, no lock -- a failure here still leaves the
        existing index untouched; (3) under the lock, the `building` marker
        and a snapshot of the canonical texts; (4) embedding, no lock -- a
        failure leaves the marker, as before; (5) under the lock, the
        rebuild is published in one transaction (`publish_rebuild`), only if
        the Brain, its storage files, the handle's backend, every snapshot
        record and the marker itself are unchanged. Records written meanwhile
        are not in it: the index is then `stale` and the next retrieval call
        refreshes it. Anything else changed: nothing is published and
        `UrdynSemanticUnavailableError` says so; there is no automatic
        retry. Other operations on the Brain proceed while it embeds; while
        the marker is there they read the index as unavailable, exactly as
        a rebuild in progress has always been reported.
        """
        semantic, backend, identity = self._setup_capture()
        model, provider, model_id, revision, normalization = self._setup_load_model(semantic, backend, identity)
        # The representation this rebuild produces, from the encoder
        # just loaded (core-verified, backend-declared or unattested); the
        # index records it, and only it, with every vector.
        representation = semantic.representation_of(model)
        with self._setup_slot(backend):
            try:
                dimensions = semantic.model_dimensions(model)
                if backend is not None and dimensions != identity.dimensions:
                    raise ValueError("external semantic backend dimensions changed")
                declared_width = representation.descriptor["encoder"].get("recipe", {}).get("dimensions")  # verified only
                if declared_width is not None and declared_width != dimensions:
                    raise ValueError("the encoder output width disagrees with its representation")
            except Exception as exc:  # onnxruntime/tokenizers/hub failures are not ours to leak
                if backend is not None:
                    raise UrdynSemanticUnavailableError("Could not load the external semantic model") from None
                raise UrdynSemanticUnavailableError(f"Could not load the semantic model: {exc}") from exc
        # NOT `SEMANTIC_MODEL_ID`: setup may have fallen back to a
        # different artifact than this machine prefers, and the index
        # must record the one it was actually built with (see
        # `_semantic.artifact_for_index`).
        marker = SemanticMeta(
            provider=_representation.tag_provider(provider, representation),
            model_id=model_id,
            model_revision=revision,
            dimensions=dimensions,
            normalization=normalization,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            status=STATUS_BUILDING,
        )
        ticket = self._setup_begin(marker, backend, identity)
        rows = self._setup_encode(semantic, model, ticket)
        self._setup_publish(ticket, rows)

        counts = {entity_type: len(entries) for entity_type, entries in ticket.pools}
        # [A54.3] `counts[ENTITY_SOURCE_CHUNK]` is a CHUNK count, not a
        # document count -- `SemanticSetupResult.source_evidence_count` is
        # public API and must keep meaning "how many seeded documents",
        # so it is derived separately as the number of DISTINCT parent
        # Evidence ids among the chunk pool's own ids, never the row count.
        source_document_ids = {
            parse_chunk_semantic_id(chunk_id)[0] for chunk_id, _ in dict(ticket.pools)[ENTITY_SOURCE_CHUNK]
        }
        return SemanticSetupResult(
            provider=provider,
            model_id=model_id,
            model_revision=revision,
            dimensions=dimensions,
            normalization=normalization,
            attempt_count=counts[ENTITY_ATTEMPT],
            memory_count=counts[ENTITY_MEMORY],
            skill_count=counts[ENTITY_SKILL],
            source_evidence_count=len(source_document_ids),
        )

    # -- semantic setup outside the lifecycle lock -----------------

    @guarded
    def _setup_capture(self):
        """Step 1 (lifecycle hold): the checks that always ran first, and the
        backend this rebuild will use, captured so a reconfiguration of the
        handle meanwhile is detected instead of mixed in."""
        semantic = _load_semantic_module()
        if semantic is None:
            raise UrdynSemanticUnavailableError(
                "Semantic retrieval requires the 'semantic' optional dependency. "
                "Install it with: pip install 'urdyn-memory[semantic]'"
            )
        if self._semantic_backend is None and not semantic.default_runtime_available():
            raise UrdynSemanticUnavailableError(
                "Semantic retrieval requires the 'semantic' optional dependency. "
                "Install it with: pip install 'urdyn-memory[semantic]'"
            )
        return semantic, self._semantic_backend, self._semantic_backend_identity

    def _setup_load_model(self, semantic, backend, identity):
        """Step 2 (no lock): load the model -- the one place allowed to
        download -- from what step 1 captured. Fails before any index is
        touched, with the same errors as before."""
        try:
            if backend is None:
                model = semantic.load_model_for_setup()
                return (model, semantic.SEMANTIC_PROVIDER, semantic.model_identity(model),
                        semantic.resolve_local_revision(), semantic.SEMANTIC_NORMALIZATION)
            if not self._backend_ready(backend, identity):
                raise ValueError("external semantic backend is unavailable")
            if identity is None:
                raise ValueError("external semantic backend has no identity")
            return (semantic.ExternalEncoder(backend, identity), identity.index_provider, identity.model_id,
                    identity.model_revision, identity.normalization)
        except Exception as exc:  # onnxruntime/tokenizers/hub failures are not ours to leak
            if backend is not None:
                raise UrdynSemanticUnavailableError("Could not load the external semantic model") from None
            raise UrdynSemanticUnavailableError(f"Could not load the semantic model: {exc}") from exc

    @contextmanager
    def _setup_slot(self, backend):
        """This Brain's inference slot for `backend` (bounded wait, see
        `INFERENCE_SLOT_TIMEOUT_SECONDS`); busy means not run, never queued."""
        with inference_slot(self._path, INFERENCE_SLOT_TIMEOUT_SECONDS, _encoder_slot(backend)) as acquired:
            if not acquired:
                raise _SetupNotPublished(
                    "This Brain's semantic encoder stayed busy with another call; semantic setup did not run "
                    "to completion and published nothing. Run it again."
                )
            yield

    @contextmanager
    def _rebuild_errors(self, backend):
        """The error mapping the rebuild steps always had: a client
        backend's own errors are not leaked."""
        try:
            yield
        except _SetupNotPublished:
            raise
        except Exception:
            if backend is not None:
                raise UrdynSemanticUnavailableError("External semantic index rebuild failed") from None
            raise

    @guarded
    def _setup_begin(self, marker, backend, identity) -> "_SetupTicket":
        """Step 3 (lifecycle hold): snapshot of the canonical texts and the
        `building` marker, in the same hold. From here on the index reads
        as unavailable until step 5 publishes, as a rebuild always has."""
        reason = self._setup_backend_changed(backend, identity)
        if reason:
            raise _SetupNotPublished(f"Semantic setup was not started: {reason}.")
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            pools = ((ENTITY_ATTEMPT, ()), (ENTITY_MEMORY, ()), (ENTITY_SKILL, ()), (ENTITY_SOURCE_CHUNK, ()))
        else:
            with store:
                pools = tuple((entity_type, tuple(entries)) for entity_type, entries in _semantic_pool_entries(store))
        with self._rebuild_errors(backend):
            with SemanticIndexStore.create_or_open(self._semantic_db_path) as semantic_store:
                semantic_store.begin_rebuild(
                    provider=marker.provider,
                    model_id=marker.model_id,
                    model_revision=marker.model_revision,
                    dimensions=marker.dimensions,
                    normalization=marker.normalization,
                    created_at=marker.created_at,
                )
        return _SetupTicket(
            brain_id=self._urdyn_id,
            incarnation=self._storage_incarnation(),
            marker=marker,
            backend=backend,
            backend_identity=identity,
            pools=pools,
        )

    def _setup_encode(self, semantic, model, ticket) -> list[tuple[str, str, bytes]]:
        """Step 4 (no lock, no SQLite connection): one `embed` per pool, the
        batches of the previous implementation, each under the inference
        slot. Rows become bytes at once: nothing mutable survives."""
        rows: list[tuple[str, str, bytes]] = []
        for entity_type, entries in ticket.pools:
            if not entries:
                continue
            ids = [entity_id for entity_id, _ in entries]
            texts = [text for _, text in entries]
            with self._setup_slot(ticket.backend), self._rebuild_errors(ticket.backend):
                try:
                    # One validated, unit-norm row per text, of the
                    # probed width -- or no index. A contract violation here
                    # leaves the index at status='building' (never published
                    # as ready) and canonical memory untouched.
                    vectors = semantic.require_unit_rows(
                        semantic.embed(model, texts), rows=len(texts), dimensions=ticket.marker.dimensions
                    )
                except semantic.InvalidSemanticVectors as exc:
                    raise UrdynSemanticUnavailableError(
                        f"The semantic encoder produced invalid vectors; the index was not built: {exc}"
                    ) from exc
            rows.extend(
                (entity_type, entity_id, semantic.vector_to_blob(vector))
                for entity_id, vector in zip(ids, vectors, strict=True)
            )
        return rows

    def _setup_backend_changed(self, backend, identity) -> str | None:
        if self._semantic_backend is not backend or self._semantic_backend_identity != identity:
            return "the semantic backend of this handle changed"
        return None

    def _setup_superseded_by(self, ticket) -> str | None:
        """Why the prepared rebuild no longer describes this Brain, or None.
        The guarded call has already re-checked the Brain's identity; this
        adds what `cortex_id` alone cannot prove."""
        reason = self._setup_backend_changed(ticket.backend, ticket.backend_identity)
        if reason:
            return reason
        if self._storage_incarnation() != ticket.incarnation:
            return "its storage files were replaced"
        with MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id) as store:
            current = {entity_type: dict(entries) for entity_type, entries in _semantic_pool_entries(store)}
        for entity_type, entries in ticket.pools:
            pool = current.get(entity_type, {})
            if any(pool.get(entity_id) != text for entity_id, text in entries):
                return "a record it covers changed or disappeared"
        return None

    @guarded
    def _setup_publish(self, ticket, rows) -> None:
        """Step 5 (lifecycle hold): verify, then publish in one transaction."""
        reason = self._setup_superseded_by(ticket)
        if reason:
            raise _SetupNotPublished(
                f"The Brain changed while its semantic index was being rebuilt ({reason}); "
                "nothing was published. Run semantic setup again."
            )
        with self._rebuild_errors(ticket.backend):
            with SemanticIndexStore.create_or_open(self._semantic_db_path) as semantic_store:
                published = semantic_store.publish_rebuild(ticket.marker, rows)
        if not published:
            raise _SetupNotPublished(
                "Another semantic setup of this Brain started or finished while this one was running; "
                "this rebuild was not published and the other one stands."
            )

    # -- derived search index --------------------------------------

    def search_index_state(self) -> SearchIndexState:
        """Whether this SQLite runtime has FTS5, and whether the
        derived search index is `missing`, `usable`, `invalid` or
        `unchecked` relative to canonical data, for every indexed entity
        type (memory, attempt, skill, source). Observational: never writes,
        never repairs (see `rebuild_search_index`). Cost grows with the
        number of records: it is a diagnostic, not a per-query check.

        Runs through `urdyn.maintenance` (lifecycle lock,
        this handle's identity and profile checked against the manifest,
        then the ordinary `verify_existing`), so it can still DIAGNOSE a
        Brain that ordinary operations refuse only because its FTS5 index
        is damaged; any other refusal is raised unchanged."""
        from . import maintenance

        return maintenance.inspect_search_index(self._path, _expected=(self._urdyn_id, self._profile))

    def rebuild_search_index(self) -> SearchIndexState:
        """Explicitly (re)build the derived search index from
        canonical data, recovering it from any state, including total
        absence. Atomic: one SQLite transaction under the Brain's lifecycle
        lock, rolled back entirely on failure. Canonical data is never
        modified. The result is validated INSIDE that transaction and
        committed only if `usable`; otherwise everything is rolled back and
        `UrdynStorageError` is raised. Raises `UrdynStorageUnavailableError`
        (nothing written) when this runtime has no FTS5. Returns the state
        validated before the commit.

        Same entry as `urdyn.maintenance.recover_search_index`
        (bound to this handle's identity): a Brain refused only because its
        FTS5 projection is damaged is recovered through the exceptional,
        backed-up path; the safety copy's location is reported by
        `recover_search_index` and kept in `<parent>/.urdyn-maintenance/`."""
        from . import maintenance

        return maintenance.recover_search_index(self._path, _expected=(self._urdyn_id, self._profile)).state

    # -- semantic lifecycle (A27) ------------------------------------------

    @guarded
    def semantic_state(self) -> SemanticState:
        """[A27] Whether the derived semantic index is currently a usable,
        up-to-date view of canonical state -- the question `preflight()`
        could not answer before A27, and silently answered "yes" to.

        OBSERVATIONAL AND CHEAP, by contract: it never loads the
        embedding model, never embeds anything, never touches the
        network, and never writes -- not to canonical storage and not to
        the derived index. `urdyn status` calls exactly this, so those
        properties are what let a state line exist at all. The most
        expensive thing it does is resolve two cached file paths (see
        `_semantic.artifacts_available`), and only in a workspace that
        has an index to ask about; a workspace that never enabled
        semantic retrieval answers from a single `Path.exists()` without
        importing the semantic runtime at all.

        The answer is COMPUTED from state, never remembered (see
        `_semantic_store.py`'s module docstring for why a stored dirty
        flag cannot be trusted across two databases with no atomic
        cross-store commit).
        """
        if not self._semantic_db_path.exists():
            return SemanticState(status=SEMANTIC_DISABLED, detail=DETAIL_NOT_SET_UP)
        store = MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id)
        if store is None:
            return self._semantic_state_for(())
        with store:
            return self._semantic_state_for(_semantic_pool_entries(store))

    def _semantic_state_for(self, pools: tuple[tuple[str, list[tuple[str, str]]], ...]) -> SemanticState:
        """Classify the index against `pools`, the canonical records it is
        supposed to cover (see `_semantic_pool_entries`).

        ORDER IS THE CONTRACT HERE, and it is not arbitrary:
        COMPATIBILITY IS DECIDED BEFORE FRESHNESS. An index built by a
        different model is not "missing some vectors" -- topping it up
        would write vectors from this build's model beside vectors from
        another one, which is precisely the mixing hazard A16.2.1
        measured (per-text cosine 0.9947 between two artifacts of the
        SAME model, enough to change the top-ranked candidate for ~1
        query in 14). So an incompatible or unloadable index reports
        UNAVAILABLE and is rebuilt explicitly, never refreshed
        incrementally. A16's `artifact_for_index` stays the sole
        authority on that question; A27 only asks it earlier.

        `pools` empty (no canonical store yet) is a legitimate READY: an
        index that covers nothing, when there is nothing to cover, is
        current.
        """
        try:
            index = SemanticIndexStore.open_if_exists(self._semantic_db_path)
        except UrdynStorageError:
            return SemanticState(status=SEMANTIC_UNAVAILABLE, detail=DETAIL_INDEX_UNREADABLE)
        if index is None:
            return SemanticState(status=SEMANTIC_DISABLED, detail=DETAIL_NOT_SET_UP)
        try:
            with index:
                meta = index.meta()
                indexed = {
                    entity_type: index.indexed_ids(entity_type)
                    for entity_type in (ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL, ENTITY_SOURCE_CHUNK)
                }
        except UrdynStorageError:
            return SemanticState(status=SEMANTIC_UNAVAILABLE, detail=DETAIL_INDEX_UNREADABLE)

        indexed_count = sum(len(ids) for ids in indexed.values())
        if meta is None or meta.status != STATUS_READY:
            # A7.4's own publication rule, reported rather than reasoned
            # about again: an interrupted rebuild is incomplete, not stale.
            return SemanticState(
                status=SEMANTIC_UNAVAILABLE, detail=DETAIL_BUILD_INCOMPLETE, indexed=indexed_count
            )

        semantic = _load_semantic_module()
        if semantic is None:
            return SemanticState(
                status=SEMANTIC_UNAVAILABLE, detail=DETAIL_EXTRA_MISSING, indexed=indexed_count
            )
        if self._semantic_backend is None:
            if not semantic.default_runtime_available():
                return SemanticState(
                    status=SEMANTIC_UNAVAILABLE, detail=DETAIL_EXTRA_MISSING, indexed=indexed_count
                )
            if semantic.artifact_for_index(meta) is None:
                return SemanticState(
                    status=SEMANTIC_UNAVAILABLE, detail=DETAIL_MODEL_MISMATCH, indexed=indexed_count
                )
            problem = semantic.representation_problem(meta)
            if problem is not None:
                return SemanticState(status=SEMANTIC_UNAVAILABLE, detail=problem, indexed=indexed_count)
            if not semantic.artifacts_available(meta):
                return SemanticState(
                    status=SEMANTIC_UNAVAILABLE, detail=DETAIL_MODEL_UNCACHED, indexed=indexed_count
                )
        else:
            if not self._external_identity_matches(meta):
                return SemanticState(
                    status=SEMANTIC_UNAVAILABLE, detail=DETAIL_MODEL_MISMATCH, indexed=indexed_count
                )
            problem = semantic.representation_problem(meta, self._semantic_backend_identity)
            if problem is not None:
                return SemanticState(status=SEMANTIC_UNAVAILABLE, detail=problem, indexed=indexed_count)
            if not self._external_semantic_ready():
                return SemanticState(
                    status=SEMANTIC_UNAVAILABLE, detail=DETAIL_MODEL_UNCACHED, indexed=indexed_count
                )

        missing = sum(
            1 for entity_type, entries in pools for entity_id, _ in entries if entity_id not in indexed[entity_type]
        )
        if missing:
            return SemanticState(status=SEMANTIC_STALE, missing=missing, indexed=indexed_count)
        return SemanticState(status=SEMANTIC_READY, indexed=indexed_count)

    def _semantic_prepare(self, pools: tuple[tuple[str, list[tuple[str, str]]], ...]) -> SemanticState:
        """Records the declared state for the current call
        (see `_SemanticCallOutcome`); on the lexical-only second pass it
        returns the degraded state without classifying or refreshing again."""
        outcome = _SEMANTIC_CALL.get()
        if outcome is not None and outcome.forced is not None:
            return outcome.forced
        state = self._semantic_prepare_index(pools)
        if outcome is not None:
            outcome.prepared = state
        return state

    def _semantic_prepare_index(self, pools: tuple[tuple[str, list[tuple[str, str]]], ...]) -> SemanticState:
        """[A27] The lifecycle step `preflight()`/`guard()` run before
        consulting the semantic channel: classify, and repair if the only
        thing wrong is that canonical work has happened since the last
        build.

        This is the whole of the A26 fix. Everything else A27 adds is
        about being honest when this cannot succeed.
        """
        state = self._semantic_state_for(pools)
        if state.status != SEMANTIC_STALE:
            return state
        refreshed = self._semantic_refresh(pools)
        if refreshed is None:
            return dataclasses.replace(state, detail=DETAIL_REFRESH_FAILED)
        # RECOMPUTED, not assumed: the refresh's own success is not
        # evidence that the index is now current (a canonical write may
        # have landed while it ran, and a partially-applied refresh must
        # still read as stale). Freshness is re-derived from storage
        # before anything is allowed to call itself ready -- the same
        # function, so there is exactly one freshness authority.
        return dataclasses.replace(self._semantic_state_for(pools), refreshed=refreshed)

    def _semantic_refresh(self, pools: tuple[tuple[str, list[tuple[str, str]]], ...]) -> int | None:
        """Embed and persist ONLY the records the index does not cover,
        returning how many were added, or None if the refresh could not
        run to completion.

        Four properties this must have, all load-bearing:

        OFFLINE. The model is obtained through `load_model_for_index`,
        which is local-cache-only by construction (A7.4's
        `local_files_only=True` path). A `preflight()` can therefore
        never trigger a download -- that stays the exclusive privilege of
        `urdyn semantic setup`. This is structural, not a convention the
        tests happen to respect.

        THE INDEX'S OWN MODEL, not this machine's preferred one. The
        artifact is whatever `meta` recorded, so a top-up cannot mix
        vector spaces (see `_semantic_state_for` for the measurement
        behind that).

        INCREMENTAL. The work list is the missing ids, re-read here from
        the index rather than inherited from the classification, so two
        consumers racing to refresh the same workspace narrow each
        other's work instead of duplicating it wholesale. A pool with
        nothing missing costs nothing, and a fully-covered index never
        reaches this method at all -- which is what keeps the model out
        of the common path.

        NON-DESTRUCTIVE. It never calls `begin_rebuild()`/
        `finish_rebuild()`: existing vectors are preserved and the A7.4
        `building`->`ready` publication rule keeps its single owner. A
        refresh interrupted half way leaves MORE covered records than it
        found and is still classified stale afterwards -- correct at
        every intermediate point, because coverage is recomputed rather
        than announced. There is no partial state that could be published
        as current, because publication is not an act here.
        """
        semantic = _load_semantic_module()
        if semantic is None:
            return None
        if self._semantic_backend is None and not semantic.default_runtime_available():
            return None
        try:
            index = SemanticIndexStore.open_if_exists(self._semantic_db_path)
            if index is None:
                return None
            with index:
                meta = index.meta()
                if meta is None or meta.status != STATUS_READY:
                    return None
                model = self._semantic_model_for_index(meta, semantic)
                if model is None:
                    return None
                index.expect(meta)  # write only into the index read here
                refreshed = 0
                for entity_type, entries in pools:
                    pending = self._pending_entries(index, entity_type, entries)
                    if not pending:
                        continue
                    if isinstance(model, _encoding_plan.PreparedEncoder):
                        # Only records still pending with the text
                        # encoded outside the lock; later writes stay pending.
                        ids, raw = model.refresh_rows(entity_type, pending)
                        if not ids:
                            continue
                    else:
                        ids, raw = [entity_id for entity_id, _ in pending], semantic.embed(
                            model, [text for _, text in pending]
                        )
                    # Same numeric contract as setup, against the
                    # width the index recorded; a violation aborts the refresh
                    # (caught below) before anything of this pool is written.
                    vectors = semantic.require_unit_rows(raw, rows=len(ids), dimensions=meta.dimensions)
                    # Written only into the index this refresh read
                    # (same marker, same representation), in one transaction.
                    if index.add_vectors(
                        entity_type,
                        [(entity_id, semantic.vector_to_blob(vector)) for entity_id, vector in zip(ids, vectors, strict=True)],
                    ) is False:
                        return None
                    refreshed += len(ids)
                return refreshed
        except Exception:
            # Every failure mode of a DERIVED, OPTIONAL channel -- model
            # files gone, index locked by a concurrent writer, storage
            # error, a backend raising something of its own -- degrades
            # to "could not refresh". Canonical memory is untouched and
            # the caller still gets its full lexical answer, plus a
            # retrieval line saying the semantic view is incomplete.
            return None

    # -- inference outside the lifecycle lock ----------------------

    @staticmethod
    def _pending_entries(index, entity_type: str, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """The records of one pool the index does not cover yet, in pool
        order: the refresh's work list, shared by planning and refresh."""
        known = index.indexed_ids(entity_type)
        return [(entity_id, text) for entity_id, text in entries if entity_id not in known]

    def _storage_incarnation(self) -> tuple:
        """`(memory.db, semantic_index.db)` file identities `(st_dev,
        st_ino)`, or None for a missing file. A replacement signal under the
        cooperating lifecycle (a restore publishes new files while the old
        ones still exist); never an identity, and never sufficient alone.
        `os.stat` only: opening and closing a database file here could drop
        SQLite's POSIX locks held by another connection of this process."""
        def one(path: Path):
            try:
                st = os.stat(path)
            except FileNotFoundError:
                return None
            return (st.st_dev, st.st_ino)

        return (one(self._db_path), one(self._semantic_db_path))

    def _current_encoder_key(self, semantic):
        """The encoder key a call would plan for right now, or None when the
        semantic channel would not be consulted (no index, index not ready,
        incompatible, external backend not ready). Read-only."""
        if semantic is None or (self._semantic_backend is None and not semantic.default_runtime_available()):
            return None
        if not self._semantic_db_path.exists():
            return None
        index = SemanticIndexStore.open_if_exists(self._semantic_db_path)
        if index is None:
            return None
        with index:
            meta = index.meta()
        if meta is None or meta.status != STATUS_READY:
            return None
        key = self._encoder_key(meta, semantic)
        if key is None or (self._semantic_backend is not None and not self._external_semantic_ready()):
            return None
        return key

    def _plan_inference(self, query) -> "_encoding_plan.EncodingTicket | None":
        """Step 1 (lifecycle hold, read-only): what this call will ask the
        encoder for. Never raises and never authorizes anything -- the
        guarded call re-checks everything; on any doubt there is no plan."""
        try:
            semantic = _load_semantic_module()
            if semantic is None or not self._semantic_db_path.exists():
                return None
            with lifecycle(self._path):
                manifest = read_manifest(self._path / URDYN_DIRNAME)
                if (manifest[LEGACY_WORKSPACE_ID_KEY], manifest["profile"]) != (self._urdyn_id, self._profile):
                    return None  # the guarded call reports it
                key = self._current_encoder_key(semantic)
                if key is None:
                    return None
                with MemoryStore.open_if_exists(self._db_path, brain_id=self._urdyn_id) as store:
                    pools = _semantic_pool_entries(store)
                stale = self._semantic_state_for(pools).status == SEMANTIC_STALE
                with SemanticIndexStore.open_if_exists(self._semantic_db_path) as index:
                    meta = index.meta()
                    # exactly the work `_semantic_prepare_index` would do now:
                    # refresh only a stale index, pool by pool, in pool order
                    refresh = tuple(
                        (entity_type, tuple(pending))
                        for entity_type, entries in pools
                        if stale and (pending := self._pending_entries(index, entity_type, entries))
                    )
                    stored = index.vector_count()
                if self._encoder_key(meta, semantic) != key:
                    return None
                return _encoding_plan.EncodingTicket(
                    brain_id=self._urdyn_id,
                    incarnation=self._storage_incarnation(),
                    encoder_key=key,
                    meta=meta,
                    backend=self._semantic_backend,
                    backend_identity=self._semantic_backend_identity,
                    query=query if stored or refresh else None,  # an empty index is never queried
                    refresh=refresh,
                )
        except Exception:
            return None

    def _infer_outside_lock(self, ticket) -> "_encoding_plan.PreparedEncodings":
        """Step 2: no lifecycle hold and no SQLite connection. The encoder is
        built from what the ticket captured, not from the handle, and runs
        under this Brain's in-process inference slot."""
        if ticket is None or not ticket.has_work:
            return _encoding_plan.PreparedEncodings(ticket)
        semantic = _load_semantic_module()
        if ticket.backend is None:
            def build_model():
                return semantic.load_model_for_index(ticket.meta)
        else:
            def build_model():
                return semantic.bind_representation(
                    semantic.ExternalEncoder(ticket.backend, ticket.backend_identity), ticket.meta)

        with inference_slot(self._path, INFERENCE_SLOT_TIMEOUT_SECONDS, _encoder_slot(ticket.backend)) as acquired:
            if not acquired:
                return _encoding_plan.PreparedEncodings(ticket, failure=_encoding_plan.EncoderBusy(
                    "this Brain's encoder is still busy with another call"))
            return _encoding_plan.infer(
                ticket,
                build_model=build_model,
                embed=lambda model, texts: semantic.embed(model, texts),
                validate_refresh=lambda rows, count: semantic.require_unit_rows(
                    rows, rows=count, dimensions=ticket.meta.dimensions),
            )

    def _plan_still_usable(self, ticket) -> bool:
        """Under the lifecycle lock, just before publication: whether the
        plan was made for the encoder and the storage files that exist now.
        Content is re-checked anyway, row by row; this only decides whether
        one fresh plan is worth making (restore, index rebuilt by another
        model, semantic channel appearing). Errors: keep the plan, the
        guarded call reports what is wrong."""
        try:
            key = self._current_encoder_key(_load_semantic_module())
            if ticket is None:
                return key is None
            return key == ticket.encoder_key and self._storage_incarnation() == ticket.incarnation
        except Exception:
            return True

    def _run_with_prepared_encodings(self, query, method, *args, **kwargs):
        """Plan (lifecycle hold), infer (no hold), publish (lifecycle
        hold, the unchanged guarded `method`). At most
        `_INFERENCE_PLANS_PER_CALL` plans; the last one is always used, its
        rows still served only for the same encoder and texts."""
        if not (isinstance(query, str) and query.strip()):
            # rejected by the method itself, before any semantic work: no plan
            with _encoding_plan.activated(_encoding_plan.PreparedEncodings(None)):
                return method(*args, **kwargs)
        for plan in range(_INFERENCE_PLANS_PER_CALL):
            ticket = self._plan_inference(query)
            prepared = self._infer_outside_lock(ticket)
            with lifecycle(self._path):
                if plan + 1 < _INFERENCE_PLANS_PER_CALL and not self._plan_still_usable(ticket):
                    continue
                with _encoding_plan.activated(prepared):
                    return method(*args, **kwargs)
        raise AssertionError("unreachable")  # pragma: no cover

    def _semantic_context(self):
        """Shared degraded-condition handling for every semantic entry
        point below: returns `None` for every condition that means "the
        semantic channel cannot be used right now" (extra not installed,
        index never built, mid-rebuild, model-config mismatch) -- never
        raises. Returns `(semantic_module, model, meta)` otherwise.

        [A16.3] The index itself decides which model answers its queries:
        `load_model_for_index` returns None for an index this build
        cannot read at all (different provider, model, upstream revision
        or normalization), and otherwise loads the exact model build that
        index was created with. That is what makes it impossible to score
        stored vectors against a query embedded by a different one -- a
        hazard measured in A16.2.1, not a theoretical one. Which model
        that is stays entirely inside `_semantic.py`; nothing here ever
        sees it.
        """
        semantic = _load_semantic_module()
        if semantic is None:
            return None
        if self._semantic_backend is None and not semantic.default_runtime_available():
            return None
        semantic_store = SemanticIndexStore.open_if_exists(self._semantic_db_path)
        if semantic_store is None:
            return None
        with semantic_store:
            meta = semantic_store.meta()
            if meta is None or meta.status != "ready":
                return None
        model = self._semantic_model_for_index(meta, semantic)
        if model is None:
            return None
        return semantic, model, meta

    def _semantic_vectors(self, entity_type: str) -> list[tuple[str, bytes]]:
        semantic_store = SemanticIndexStore.open_if_exists(self._semantic_db_path)
        if semantic_store is None:
            return []
        with semantic_store:
            return semantic_store.all_vectors(entity_type)

    def _semantic_widen(
        self, query_text: str, entity_type: str, *, eligible_ids: frozenset[str] | None = None
    ) -> frozenset[str]:
        """Best-effort semantic candidate widening for one entity-type
        pool. Returns an empty set -- never raises -- for every degraded
        condition this is expected to handle gracefully: extra not
        installed, index never built, index mid-rebuild
        (`status='building'`), index built for a different model
        configuration, model files missing from the local cache (e.g.
        after copying a workspace without its Hugging Face cache), or a
        corrupted stored vector. `preflight()`/`guard()` must never crash
        or hang because of this method.

        `eligible_ids`, if given, is a plain id filter applied BEFORE
        ranking (see `_semantic.py`'s [A7.7] module docstring note): this
        method still has no idea what "verified" or "current" mean, it
        only narrows the candidate pool to ids the caller already decided
        are usable. `None` means "no restriction" (attempts in
        preflight(), skills in guard() -- see call sites).
        """
        if _semantic_suppressed():
            return frozenset()
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset()
            semantic, model, meta = context
            stored_vectors = self._semantic_vectors(entity_type)
            if not stored_vectors:
                return frozenset()
            return semantic.semantic_admitted_ids(
                query_text,
                entity_type,
                model=model,
                stored_vectors=stored_vectors,
                dimensions=meta.dimensions,
                eligible_ids=eligible_ids,
            )
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset()

    def _preflight_lesson_semantic_admitted(
        self, task: str, *, lesson_eligible_ids: frozenset[str]
    ) -> frozenset[str]:
        """[A23.1] Semantic admission for preflight()'s LESSON pool: a
        bounded SET, not a single winner.

        Ranks the SAME MEMORY vectors with the SAME model against the
        SAME query as every other semantic pool, restricted (before
        ranking, per A7.7) to `lesson_eligible_ids`, then applies
        `_semantic.set_admitted_ids`: the Lesson-specific floor
        calibrated in A23.2, no margin, capped at `SET_ADMISSION_LIMIT`.
        No lesson-specific SCORE, no boost, and specifically no "return
        every verified lesson" shortcut -- a lesson below the floor is
        rejected here as firmly as anywhere else. The MEMORY pool's own
        floor and margin are untouched and unread by this path.

        `lesson_eligible_ids` is the caller's ALREADY-computed
        current+verified lesson set (see `preflight`), reused rather than
        re-derived: this method has no idea what "verified" or "current"
        mean, exactly like `_semantic_widen`. Relevance never grants
        authority -- an unverified, superseded or non-current lesson is
        not in that set, so no score can put it in this result.

        The cap applies to THIS channel only. `build_preflight` still
        unions these ids with whatever the lexical and FTS channels
        admitted, and those channels remain unbounded, so a task whose
        wording genuinely matches five lessons still surfaces five.

        Falls back to empty -- never raises -- on any degraded
        condition, exactly like `_semantic_widen`.
        """
        if _semantic_suppressed():
            return frozenset()
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset()
            semantic, model, meta = context
            memory_vectors = self._semantic_vectors(ENTITY_MEMORY)
            if not memory_vectors:
                return frozenset()
            ranked = semantic.semantic_rank_eligible(
                task,
                model=model,
                stored_vectors=memory_vectors,
                dimensions=meta.dimensions,
                eligible_ids=lesson_eligible_ids,
            )
            return semantic.set_admitted_ids(ranked, floor=semantic.LESSON_SEMANTIC_FLOOR)
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset()

    def _context_invariant_semantic_admitted(
        self, task: str, *, invariant_eligible_ids: frozenset[str]
    ) -> frozenset[str]:
        """[A31.2] Semantic admission for `context()`'s INVARIANT pool: a
        bounded SET, not a single winner.

        Same shape as `_preflight_lesson_semantic_admitted` above and for
        the same reason -- the project-wide constraints that bind one task
        are a set, not a contest. A31.1 measured what the single-winner
        policy costs this pool: the MEMORY floor rejected nothing at all
        (rank #1 always cleared it), so admission was decided by the
        margin alone, and near-tied CO-RELEVANT constraints were rejected
        TOGETHER -- an empty CONSTRAINTS section on 27 of the 33 corpus
        scenes that had a genuinely applicable invariant. The invariant
        floor and cap calibrated in A31.1 replace it; the MEMORY pool's
        own floor and margin are untouched and unread by this path, and
        every other pool still asks the single-winner question.

        `invariant_eligible_ids` is the caller's ALREADY-computed CURRENT
        invariant set, reused rather than re-derived, and restricting the
        pool BEFORE ranking (A7.7). A superseded or invalidated invariant
        is not in that set, so no score can bring it back.

        This channel is one of four, and the only one that changed: the
        lexical, FTS and provenance channels are unchanged and still
        unioned in `memory_is_relevant`. Admission is also not inclusion
        -- an admitted invariant competes for `budget` in CONSTRAINTS like
        any other candidate.

        Deliberately NOT reused by `preflight()`, whose invariants stay
        unconditional (A9.1): that view is the complete checklist, this
        one is the task-relevant selection.

        Falls back to empty -- never raises -- on any degraded condition,
        exactly like `_semantic_widen`.
        """
        if _semantic_suppressed():
            return frozenset()
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset()
            semantic, model, meta = context
            memory_vectors = self._semantic_vectors(ENTITY_MEMORY)
            if not memory_vectors:
                return frozenset()
            ranked = semantic.semantic_rank_eligible(
                task,
                model=model,
                stored_vectors=memory_vectors,
                dimensions=meta.dimensions,
                eligible_ids=invariant_eligible_ids,
            )
            return semantic.set_admitted_ids(
                ranked,
                floor=semantic.INVARIANT_SEMANTIC_FLOOR,
                limit=semantic.INVARIANT_ADMISSION_LIMIT,
            )
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset()

    def _context_evidence_semantic_admitted(
        self, task: str, *, current_source_evidence: Sequence[tuple[Source, Evidence]]
    ) -> tuple[frozenset[str], dict[str, float]]:
        """Semantic admission for `context()`'s PROJECT EVIDENCE
        pool: a bounded SET, not a single winner -- the same shape as
        `_context_invariant_semantic_admitted` and for the same reason:
        several seeded documents can each cover a distinct, co-relevant
        facet of one task, so this is a set-valued pool, not a contest
        with one winner.

        [A54.3, Strategy B'] Admission is at SOURCE granularity, exactly
        as before, but the score behind it is no longer one whole-document
        embedding: it ranks `ENTITY_SOURCE_CHUNK` vectors (every derived
        chunk of the current-observation Evidence of every seeded Source,
        see `_semantic_pool_entries`) against `task`, restricted (before
        ranking, per A7.7) to the CURRENT chunk ids derived fresh from
        `current_source_evidence` -- an append-only index can hold a
        superseded observation's chunk vectors, but they are never even
        candidates here, the same guarantee the old whole-document
        admission gave. Each Source's semantic score is then the MAXIMUM
        cosine among its own eligible chunks: one strongly relevant
        section is enough to make its Source admissible, and a document
        does not score higher merely for having more chunks (a straight
        sum or average would do exactly that, and was rejected -- see the
        A54.2.1 architectural review). `set_admitted_ids` then runs over
        these per-SOURCE scores with the UNCHANGED floor/limit, so at
        most `EVIDENCE_ADMISSION_LIMIT` distinct Sources are ever
        admitted here, never a flood of one long document's own chunks --
        this is what keeps admission itself immune to same-source
        flooding by construction, without a new per-source cap.

        Admission uses `semantic.EVIDENCE_SEMANTIC_FLOOR`/
        `EVIDENCE_ADMISSION_LIMIT`, which are NOT independently
        calibrated -- see that module's docstring. This is a documented
        placeholder operating point, not a measurement, and this change
        does not touch either number (see the A54.3 report's CALIBRATION
        FOLLOW-UP REQUIRED note: max-over-chunks scores run measurably
        higher than the old whole-document scores did, which makes that
        pre-existing calibration debt more visible, not something this
        change was scoped to pay down).

        Returns the admitted id SET (of `evidence_id`s, exactly as
        before -- callers downstream, `ordered_project_evidence` included,
        need no changes) together with the per-Source score the
        aggregation above produced. The scores are not a second admission
        gate and are never compared to any threshold: `ordered_project_evidence`
        uses them only to break ties between chunks with an identical
        lexical shared-token count, so a workspace without the semantic
        extra (where this returns an empty set and an empty mapping)
        orders exactly as it would have.

        Falls back to empty -- never raises -- on any degraded
        condition, exactly like `_semantic_widen`.
        """
        if _semantic_suppressed():
            return frozenset(), {}
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset(), {}
            semantic, model, meta = context
            chunk_vectors = self._semantic_vectors(ENTITY_SOURCE_CHUNK)
            if not chunk_vectors:
                return frozenset(), {}
            # chunk_semantic_id -> owning evidence_id, restricted to
            # CURRENT chunks only (A7.7): built fresh from the caller's
            # current-observation list, never from the (possibly stale,
            # append-only) index contents.
            owner_by_chunk_id: dict[str, str] = {
                chunk_semantic_id(evidence.evidence_id, chunk.chunk_index): evidence.evidence_id
                for _source, evidence in current_source_evidence
                for chunk in chunk_evidence(evidence)
            }
            if not owner_by_chunk_id:
                return frozenset(), {}
            ranked_chunks = semantic.semantic_rank_eligible(
                task,
                model=model,
                stored_vectors=chunk_vectors,
                dimensions=meta.dimensions,
                eligible_ids=frozenset(owner_by_chunk_id),
            )
            if not ranked_chunks:
                return frozenset(), {}
            # `ranked_chunks` is already best-first (`rank_candidates`),
            # so the first chunk seen for a given Source IS that Source's
            # maximum -- no separate max() pass needed.
            source_scores: dict[str, float] = {}
            for chunk_id, score in ranked_chunks:
                evidence_id = owner_by_chunk_id[chunk_id]
                if evidence_id not in source_scores:
                    source_scores[evidence_id] = score
            ranked_sources = sorted(source_scores.items(), key=lambda pair: -pair[1])
            admitted = semantic.set_admitted_ids(
                ranked_sources,
                floor=semantic.EVIDENCE_SEMANTIC_FLOOR,
                limit=semantic.EVIDENCE_ADMISSION_LIMIT,
            )
            return admitted, source_scores
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset(), {}

    def _preflight_memory_semantic_widen(
        self,
        task: str,
        *,
        root_cause_memories: list[Memory],
        verified_lesson_memories: list[Memory],
        memory_eligible_ids: frozenset[str],
    ) -> frozenset[str]:
        """[A7.8] Semantic admission for preflight()'s MEMORY pool,
        aware of shared provenance: a root cause and the verified
        lesson drawn from it are the SAME underlying experience
        described from two angles, not two candidates competing for
        the pool's single admission slot.

        Diagnosed in A7.8 from a real acceptance-testing miss: plain
        `_semantic_widen`'s single-winner-plus-margin admission
        (`_semantic.semantic_admitted_ids`) treats a root-cause/lesson
        pair as competitors. A query relevant to the underlying
        incident can score both comfortably above the pool's absolute
        floor while their near-identical similarity to EACH OTHER
        (rather than to any genuine competitor) collapses the margin
        between them, rejecting a target that would otherwise be
        admitted for a reason that has nothing to do with its
        relevance. `build_preflight` already trusts shared Evidence as
        proof of "same experience" for attempt-to-memory rescue (see
        `_preflight.py`'s module docstring); this extends the identical
        trust to memory-to-memory, since nothing about that reasoning
        is specific to attempts.

        Candidates are clustered by shared `evidence_ids`, restricted
        to the already-eligible `root_cause_memories`/
        `verified_lesson_memories` the caller computed (this method has
        no independent idea of what "eligible" means, exactly like
        `_semantic_widen`). The cluster containing the top-ranked
        candidate is treated as a single admission unit: its
        representative score is the top candidate's own score (nothing
        in the pool can outscore the global top by construction), and
        it competes on margin against the best-scoring candidate
        OUTSIDE its own cluster -- never against its own sibling. If
        the pool has no other cluster to compete against (every
        eligible candidate shares the same experience), margin is
        skipped entirely, generalizing the existing
        single-candidate-pool exemption
        (`_semantic.semantic_admitted_id`) to a single-CLUSTER pool.
        When the winning cluster is admitted, EVERY eligible member of
        it is admitted, not just the top-scoring one -- a rescued root
        cause without its own verified lesson (or vice versa) is
        exactly the partial, structurally-arbitrary result this method
        exists to avoid.

        [A23.4] Two category-boundary conditions, and no others: a
        representative that is a verified lesson must clear the LESSON
        floor rather than this pool's, and a cluster of lessons only is
        not admitted here at all. Both exist because verified lessons
        alone are eligible in two pools, and this one's floor is the
        lower of the two. Neither touches the cross-category rescue
        above: a score-borne representative obeys its own category's
        policy, a provenance-borne sibling does not have to.

        Falls back to empty -- never raises -- on any degraded
        condition, exactly like `_semantic_widen`.
        """
        if _semantic_suppressed():
            return frozenset()
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset()
            semantic, model, meta = context
            memory_vectors = self._semantic_vectors(ENTITY_MEMORY)
            if not memory_vectors:
                return frozenset()

            ranked = semantic.semantic_rank_eligible(
                task,
                model=model,
                stored_vectors=memory_vectors,
                dimensions=meta.dimensions,
                eligible_ids=memory_eligible_ids,
            )
            if not ranked:
                return frozenset()

            policy = semantic.SEMANTIC_POLICY[ENTITY_MEMORY]
            top_id, top_score = ranked[0]
            if top_score < policy.absolute_floor:
                return frozenset()

            # [A23.4] The representative establishes the experience's
            # relevance with its OWN score, so that score must satisfy its
            # OWN category's admission policy. A verified lesson is
            # eligible here as well as in the Lesson pool, and this floor
            # is MEMORY's 0.20 -- so without this check a lesson the
            # calibrated Lesson floor (0.30) had rejected could establish
            # relevance anyway, for itself and for whatever shares its
            # Evidence.
            lesson_ids = frozenset(m.memory_id for m in verified_lesson_memories)
            if top_id in lesson_ids and top_score < semantic.LESSON_SEMANTIC_FLOOR:
                return frozenset()

            evidence_by_id = {
                m.memory_id: frozenset(m.evidence_ids)
                for m in (*root_cause_memories, *verified_lesson_memories)
            }

            def _cluster(start_id: str, ranked_ids: list[str]) -> frozenset[str]:
                cluster = {start_id}
                changed = True
                while changed:
                    changed = False
                    cluster_evidence: frozenset[str] = frozenset().union(
                        *(evidence_by_id.get(member, frozenset()) for member in cluster)
                    )
                    for candidate_id in ranked_ids:
                        if candidate_id in cluster:
                            continue
                        if evidence_by_id.get(candidate_id, frozenset()) & cluster_evidence:
                            cluster.add(candidate_id)
                            changed = True
                return frozenset(cluster)

            ranked_ids = [entity_id for entity_id, _ in ranked]
            top_cluster = _cluster(top_id, ranked_ids)

            competitor_score = next(
                (score for entity_id, score in ranked if entity_id not in top_cluster), None
            )
            if competitor_score is not None:
                margin = top_score - competitor_score
                if margin < policy.margin_floor:
                    return frozenset()

            # [A23.4] A cluster of lessons only has no sibling of another
            # category to reconstruct, so nothing this method exists for is
            # at stake -- what is left is a pool with a lower floor and no
            # cap deciding a question the Lesson pool owns. Those lessons
            # reach preflight() through their own channel if its floor and
            # its cap admit them. Cross-category clusters are unaffected:
            # once a valid representative has established relevance,
            # provenance still rescues every member, at any score.
            if top_cluster <= lesson_ids:
                return frozenset()

            return top_cluster
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset()

    def _preflight_corroboration_admitted(
        self,
        task: str,
        *,
        root_cause_memories: list[Memory],
        verified_lesson_memories: list[Memory],
        attempts: list[Attempt],
        memory_eligible_ids: frozenset[str],
    ) -> frozenset[str]:
        """[A7.7] Rigorous query-conditioned structural corroboration --
        `preflight()` ONLY (see `_guard.py` for why `guard()`
        deliberately does not get this: A7.6 measured that the same
        mechanism, applied to the skill/attempt pools, could not be told
        apart from the payment-guard-clause false positive on any
        available signal).

        Fires only as a FALLBACK when normal semantic admission
        (`_semantic_widen`) found nothing. The top-ranked ELIGIBLE memory
        candidate is admitted anyway if -- and only if -- a REAL,
        canonically-linked Attempt (sharing at least one Evidence id with
        that memory; the only relationship `_relevance`/`_store` already
        expose) is INDEPENDENTLY relevant to the same task on its own
        terms: lexically relevant on its own text, OR itself fully
        admitted by the normal semantic policy within the attempt pool.
        "The related entity merely exists" or "scores above some floor"
        is deliberately NOT enough -- A7.6 found and rejected that
        weaker version because it let the payment-guard-clause false
        positive back in (a coincidentally-adjacent related memory
        cleared a low floor without being genuinely, independently
        relevant). Also refuses to corroborate a candidate whose own
        score is not even within a wide sanity band of its pool's floor
        -- this is not a rescue path for arbitrarily weak matches.
        """
        if _semantic_suppressed():
            return frozenset()
        try:
            context = self._semantic_context()
            if context is None:
                _note_semantic_failure(None)
                return frozenset()
            semantic, model, meta = context
            memory_vectors = self._semantic_vectors(ENTITY_MEMORY)
            if not memory_vectors:
                return frozenset()

            ranked = semantic.semantic_rank_eligible(
                task,
                model=model,
                stored_vectors=memory_vectors,
                dimensions=meta.dimensions,
                eligible_ids=memory_eligible_ids,
            )
            if not ranked:
                return frozenset()
            top_id, top_score = ranked[0]
            policy = semantic.SEMANTIC_POLICY[ENTITY_MEMORY]
            if top_score < policy.absolute_floor - 0.20:
                return frozenset()

            candidate = next(
                (m for m in (*root_cause_memories, *verified_lesson_memories) if m.memory_id == top_id), None
            )
            if candidate is None or not candidate.evidence_ids:
                return frozenset()

            evidence_ids = frozenset(candidate.evidence_ids)
            related_attempt_ids = frozenset(
                a.attempt_id for a in attempts if not evidence_ids.isdisjoint(a.evidence_ids)
            )
            if not related_attempt_ids:
                return frozenset()

            query_tokens = frozenset(_tokens(task))
            lexically_independent = any(
                _is_relevant(query_tokens, _attempt_search_text(a.task, a.approach))
                for a in attempts
                if a.attempt_id in related_attempt_ids
            )
            if lexically_independent:
                return frozenset({top_id})

            attempt_vectors = self._semantic_vectors(ENTITY_ATTEMPT)
            if not attempt_vectors:
                return frozenset()
            semantically_independent = semantic.semantic_admitted_ids(
                task,
                ENTITY_ATTEMPT,
                model=model,
                stored_vectors=attempt_vectors,
                dimensions=meta.dimensions,
                eligible_ids=related_attempt_ids,
            )
            if semantically_independent:
                return frozenset({top_id})
            return frozenset()
        except Exception as exc:
            _note_semantic_failure(exc)
            return frozenset()
