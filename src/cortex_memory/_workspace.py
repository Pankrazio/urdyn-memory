"""The Cortex workspace: identity, profile, and lifecycle."""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from collections.abc import Sequence
from pathlib import Path

from ._attempt import OUTCOME_FAILED, VALID_OUTCOMES, Attempt
from ._errors import (
    CortexAlreadyInitializedError,
    CortexManifestError,
    CortexNotFoundError,
    CortexSemanticUnavailableError,
    CortexStorageError,
)
from ._event import (
    EVENT_KIND_ATTEMPT_RECORDED,
    EVENT_KIND_MEMORY_RECORDED,
    EVENT_KIND_MEMORY_SUPERSEDED,
    EVENT_KIND_SKILL_PROMOTED,
    Event,
)
from ._evidence import DEFAULT_EVIDENCE_KIND, VALID_EVIDENCE_KINDS, VERIFICATION_EVIDENCE_KINDS, Evidence
from ._gitignore import ensure_gitignore_entry
from ._guard import GuardResult, build_guard_result
from ._manifest import CANONICAL_PROFILES, SCHEMA_VERSION, read_manifest, write_manifest
from ._memory import (
    DEFAULT_KIND,
    EPISTEMIC_USER_ASSERTED,
    EPISTEMIC_VERIFIED,
    KIND_LESSON,
    KIND_ROOT_CAUSE,
    VALID_EPISTEMIC_STATES,
    VALID_KINDS,
    Memory,
)
from ._preflight import Preflight, build_preflight
from ._relevance import attempt_search_text as _attempt_search_text
from ._relevance import is_relevant as _is_relevant
from ._relevance import memory_search_text as _memory_search_text
from ._relevance import skill_search_text as _skill_search_text
from ._relevance import tokens as _tokens
from ._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL
from ._semantic_store import SemanticIndexStore, semantic_db_path_for
from ._skill import Skill
from ._store import MemoryStore, db_path_for

CORTEX_DIRNAME = ".cortex"
DEFAULT_RECALL_LIMIT = 20


def _load_semantic_module():
    """Lazily import the optional semantic channel (`model2vec`/`numpy`).
    Returns None -- never raises -- if the `cortex-memory[semantic]`
    extra is not installed, so every caller in this module degrades to
    lexical/FTS-only exactly as if A7.4 did not exist. This is the ONLY
    place outside `_semantic.py` itself that imports it."""
    try:
        from . import _semantic
    except ImportError:
        return None
    return _semantic


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticSetupResult:
    """Report returned by `Cortex.semantic_setup()`. Not a `Memory`, not
    canonical data -- purely a summary of what the (re)build just did."""

    provider: str
    model_id: str
    model_revision: str | None
    dimensions: int
    normalization: str
    attempt_count: int
    memory_count: int
    skill_count: int


class Cortex:
    """A discovered or newly initialized Cortex workspace."""

    def __init__(self, path: Path, profile: str, cortex_id: str) -> None:
        self._path = path
        self._profile = profile
        self._cortex_id = cortex_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def cortex_id(self) -> str:
        return self._cortex_id

    def __repr__(self) -> str:
        return f"Cortex(path={str(self._path)!r}, profile={self._profile!r})"

    @classmethod
    def init(cls, path: str | Path = ".", profile: str = "general") -> "Cortex":
        """Initialize (or safely re-open) a Cortex workspace at `path`."""
        if profile not in CANONICAL_PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; expected one of {sorted(CANONICAL_PROFILES)}")

        workspace = Path(path).resolve()
        cortex_dir = workspace / CORTEX_DIRNAME

        if cortex_dir.exists() and not cortex_dir.is_dir():
            raise CortexManifestError(f"{cortex_dir} exists but is not a directory")

        if cortex_dir.is_dir():
            data = read_manifest(cortex_dir)
            if data["profile"] != profile:
                raise CortexAlreadyInitializedError(
                    f"Cortex workspace at {workspace} is already initialized with profile "
                    f"{data['profile']!r}; refusing to switch to {profile!r}. "
                    f"Remove {cortex_dir} to reinitialize."
                )
            ensure_gitignore_entry(workspace)
            return cls(workspace, data["profile"], data["cortex_id"])

        cortex_dir.mkdir(parents=True)
        cortex_id = uuid.uuid4().hex
        data = {"schema_version": SCHEMA_VERSION, "cortex_id": cortex_id, "profile": profile}
        write_manifest(cortex_dir, data)
        ensure_gitignore_entry(workspace)
        return cls(workspace, profile, cortex_id)

    @classmethod
    def open(cls, path: str | Path = ".") -> "Cortex":
        """Open a Cortex workspace whose root is exactly `path`."""
        workspace = Path(path).resolve()
        cortex_dir = workspace / CORTEX_DIRNAME
        if not cortex_dir.is_dir():
            raise CortexNotFoundError(f"No Cortex workspace found at {workspace}")
        data = read_manifest(cortex_dir)
        return cls(workspace, data["profile"], data["cortex_id"])

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "Cortex":
        """Locate the nearest Cortex workspace, walking upward from `start`."""
        current = Path(start if start is not None else Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            cortex_dir = candidate / CORTEX_DIRNAME
            if cortex_dir.is_dir():
                data = read_manifest(cortex_dir)
                return cls(candidate, data["profile"], data["cortex_id"])
        raise CortexNotFoundError(
            f"No Cortex workspace found in {current} or any parent directory. "
            "Run 'cortex init' to create one."
        )

    def remember(
        self,
        content: str,
        *,
        kind: str = DEFAULT_KIND,
        epistemic_state: str = EPISTEMIC_USER_ASSERTED,
        supersedes: str | None = None,
        evidence: Sequence[Evidence] = (),
    ) -> Memory:
        """Persist a new canonical memory and return it.

        `content` is recorded verbatim; Cortex does not interpret,
        summarize, or verify it. Rejects empty or whitespace-only input.

        If `supersedes` is given, it must be the memory_id of an existing
        memory; that memory is preserved as history, not deleted or
        modified, and stops being "current". `evidence` records why this
        memory exists (its provenance).

        `epistemic_state` defaults to `user_asserted`. Recording evidence
        does not by itself imply verification: a memory may only be
        marked `verified` if `evidence` includes at least one item whose
        kind actually represents a check (a test result, a command or
        tool output, an explicit user confirmation) — an opinion
        (`user_statement`) or a bare file reference is not enough. Cortex
        refuses to accept a verified claim resting on nothing, and
        refuses one resting only on an unchecked assertion.
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
        if epistemic_state == EPISTEMIC_VERIFIED:
            if not evidence_ids:
                raise ValueError("A memory cannot be marked verified without at least one piece of evidence")
            if not any(item.kind in VERIFICATION_EVIDENCE_KINDS for item in evidence):
                raise ValueError(
                    "A memory can only be marked verified with evidence strong enough to justify it "
                    f"(one of {sorted(VERIFICATION_EVIDENCE_KINDS)}), not an unchecked assertion or reference"
                )

        recorded_at = dt.datetime.now(dt.timezone.utc)
        memory = Memory(
            memory_id=memory_id,
            content=content,
            kind=kind,
            epistemic_state=epistemic_state,
            recorded_at=recorded_at,
            supersedes=supersedes,
            evidence_ids=evidence_ids,
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

        with MemoryStore.create_or_open(self._db_path) as store:
            store.add(memory, events)

        return memory

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

        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return []
        with store:
            return store.search(query, limit, current_only=not include_superseded)

    def timeline(self, *, kind: str | None = None) -> list[Memory]:
        """Return the full recorded history, oldest first, optionally
        filtered by `kind`. Includes both current and superseded memories:
        superseding never removes anything from the timeline."""
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind {kind!r}; expected one of {sorted(VALID_KINDS)}")

        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return []
        with store:
            return store.timeline(kind)

    def state(self, *, kind: str | None = None) -> list[Memory]:
        """Return only the currently-valid memories, oldest first, optionally
        filtered by `kind`. This is the history projected onto "what Cortex
        currently considers true": superseded memories are excluded."""
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind {kind!r}; expected one of {sorted(VALID_KINDS)}")

        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return []
        with store:
            current_ids = store.current_ids()
            return [memory for memory in store.timeline(kind) if memory.memory_id in current_ids]

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

        with MemoryStore.create_or_open(self._db_path) as store:
            store.add_evidence(evidence)

        return evidence

    def get_evidence(self, evidence_id: str) -> Evidence:
        """Resolve an evidence_id (e.g. from `Memory.evidence_ids`) to its
        persisted `Evidence`. Raises `ValueError` if unknown."""
        store = MemoryStore.open_if_exists(self._db_path)
        evidence = None
        if store is not None:
            with store:
                evidence = store.get_evidence(evidence_id)
        if evidence is None:
            raise ValueError(f"Unknown evidence {evidence_id!r}")
        return evidence

    def learn(
        self,
        content: str,
        *,
        evidence: Sequence[Evidence] = (),
        verified: bool = False,
        supersedes: str | None = None,
    ) -> Memory:
        """Persist a `Lesson`: a reusable conclusion drawn from experience.

        A lesson is a `Memory` of kind `lesson`. By default it is recorded
        as a candidate (`user_asserted`); pass `verified=True` together
        with confirming `evidence` (e.g. a test result) to record it as a
        verified lesson instead. A candidate can later be superseded by a
        verified version of the same lesson via `supersedes`.
        """
        epistemic_state = EPISTEMIC_VERIFIED if verified else EPISTEMIC_USER_ASSERTED
        return self.remember(
            content,
            kind=KIND_LESSON,
            epistemic_state=epistemic_state,
            supersedes=supersedes,
            evidence=evidence,
        )

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

        with MemoryStore.create_or_open(self._db_path) as store:
            store.add_attempt(attempt, event)

        return attempt

    def preflight(self, task: str) -> Preflight:
        """Select prior experience relevant to `task`, before starting it.

        Answers "what should an agent know before attempting this?" by
        surfacing known failures (matching failed attempts), root causes,
        verified lessons, and any test/command evidence recommended as
        validation — each only if Cortex has something relevant on
        record. This is lexical and deterministic, not a search engine:
        it will not return everything, and it will not return nothing
        just because the wording differs slightly.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Preflight task must not be empty or whitespace-only")

        empty = Preflight(
            task=task, known_failures=(), root_causes=(), verified_lessons=(), recommended_validation=()
        )
        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return empty
        with store:
            current_ids = store.current_ids()
            root_cause_memories = [m for m in store.timeline(KIND_ROOT_CAUSE) if m.memory_id in current_ids]
            lesson_memories = [m for m in store.timeline(KIND_LESSON) if m.memory_id in current_ids]
            verified_lesson_memories = [m for m in lesson_memories if m.epistemic_state == EPISTEMIC_VERIFIED]
            attempts = store.list_attempts()

            def _must_get_evidence(evidence_id: str) -> Evidence:
                evidence = store.get_evidence(evidence_id)
                if evidence is None:
                    raise CortexStorageError(
                        f"Evidence {evidence_id!r} is referenced by recorded experience but "
                        "missing from the store"
                    )
                return evidence

            query_tokens = frozenset(_tokens(task))
            attempt_fts_candidates = store.search_candidates(query_tokens, ENTITY_ATTEMPT)
            memory_fts_candidates = store.search_candidates(query_tokens, ENTITY_MEMORY)

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
            attempt_semantic_admitted = self._semantic_widen(task, ENTITY_ATTEMPT)
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

            return build_preflight(
                task,
                attempts=attempts,
                root_cause_memories=root_cause_memories,
                verified_lesson_memories=verified_lesson_memories,
                evidence_lookup=_must_get_evidence,
                attempt_fts_candidates=attempt_fts_candidates,
                memory_fts_candidates=memory_fts_candidates,
                attempt_semantic_admitted=attempt_semantic_admitted,
                memory_semantic_admitted=memory_semantic_admitted,
            )

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
        to an existing lesson memory Cortex actually has on record) and
        always requires the caller to write the procedure out (`steps`)
        rather than reusing the lesson's own sentence as-is.

        `lesson` only supplies which memory_id to promote from. Nothing
        else about the object the caller passed in is trusted: the
        resulting Skill's `verification_state` and `evidence_ids` are
        derived from the CANONICAL Lesson Cortex has actually persisted
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

        with MemoryStore.create_or_open(self._db_path) as store:
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

    def get_skill(self, skill_id: str) -> Skill:
        """Resolve a skill_id to its persisted `Skill`. Raises `ValueError`
        if unknown."""
        store = MemoryStore.open_if_exists(self._db_path)
        skill = None
        if store is not None:
            with store:
                skill = store.get_skill(skill_id)
        if skill is None:
            raise ValueError(f"Unknown skill {skill_id!r}")
        return skill

    def skills(self) -> list[Skill]:
        """Return every recorded Skill, oldest first."""
        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return []
        with store:
            return store.list_skills()

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
        or executes anything, only reports what Cortex found.
        """
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Guard action must not be empty or whitespace-only")

        empty = GuardResult(action=action, known_failures=(), applicable_skills=(), recommended_validation=())
        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return empty
        with store:
            skills = store.list_skills()
            attempts = store.list_attempts()

            def _must_get_evidence(evidence_id: str) -> Evidence:
                evidence = store.get_evidence(evidence_id)
                if evidence is None:
                    raise CortexStorageError(
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
            skill_semantic_admitted = self._semantic_widen(action, ENTITY_SKILL)
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
            )

    def _count_memories(self) -> int:
        """Return the number of persisted memories, or 0 if none exist yet.

        Internal to the `cortex status` CLI command. Not part of the public
        API: the future semantics of "count" (current vs. superseded vs.
        invalidated memories) are not yet stable enough to commit to.
        """
        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            return 0
        with store:
            return store.count()

    @property
    def _db_path(self) -> Path:
        return db_path_for(self._path / CORTEX_DIRNAME)

    @property
    def _semantic_db_path(self) -> Path:
        return semantic_db_path_for(self._path / CORTEX_DIRNAME)

    # -- semantic retrieval (A7.4, optional) -------------------------------

    def semantic_setup(self) -> SemanticSetupResult:
        """(Re)build the derived semantic index for this workspace from
        canonical data: attempts (task+approach), all memories (content),
        and skills (name+purpose+conditions) -- the same three
        representations `_relevance.py` already derives for FTS, no new
        canonical field. Always safe to call again: fully rebuilds from
        scratch every time (idempotent), which is also how a stale or
        model-mismatched index gets fixed.

        Raises `CortexSemanticUnavailableError` if the `[semantic]` extra
        is not installed -- this is the one semantic entry point allowed
        to fail loudly and to touch the network (to download the model if
        it is not already cached); `preflight()`/`guard()` never do
        either.
        """
        semantic = _load_semantic_module()
        if semantic is None:
            raise CortexSemanticUnavailableError(
                "Semantic retrieval requires the 'semantic' optional dependency. "
                "Install it with: pip install 'cortex-memory[semantic]'"
            )

        model = semantic.load_model_for_setup()
        dimensions = semantic.model_dimensions(model)
        revision = semantic.resolve_local_revision()
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()

        store = MemoryStore.open_if_exists(self._db_path)
        if store is None:
            memories, attempts, skills = [], [], []
        else:
            with store:
                memories = store.timeline(None)
                attempts = store.list_attempts()
                skills = store.list_skills()

        pools = (
            (ENTITY_ATTEMPT, [(a.attempt_id, _attempt_search_text(a.task, a.approach)) for a in attempts]),
            (ENTITY_MEMORY, [(m.memory_id, _memory_search_text(m.content)) for m in memories]),
            (ENTITY_SKILL, [(s.skill_id, _skill_search_text(s.name, s.purpose, s.conditions)) for s in skills]),
        )

        with SemanticIndexStore.create_or_open(self._semantic_db_path) as semantic_store:
            semantic_store.begin_rebuild(
                provider=semantic.SEMANTIC_PROVIDER,
                model_id=semantic.SEMANTIC_MODEL_ID,
                model_revision=revision,
                dimensions=dimensions,
                normalization=semantic.SEMANTIC_NORMALIZATION,
                created_at=created_at,
            )
            for entity_type, entries in pools:
                if not entries:
                    continue
                ids = [entity_id for entity_id, _ in entries]
                texts = [text for _, text in entries]
                vectors = semantic.embed(model, texts)
                rows = [(entity_id, semantic.vector_to_blob(vec)) for entity_id, vec in zip(ids, vectors)]
                semantic_store.add_vectors(entity_type, rows)
            semantic_store.finish_rebuild()

        return SemanticSetupResult(
            provider=semantic.SEMANTIC_PROVIDER,
            model_id=semantic.SEMANTIC_MODEL_ID,
            model_revision=revision,
            dimensions=dimensions,
            normalization=semantic.SEMANTIC_NORMALIZATION,
            attempt_count=len(attempts),
            memory_count=len(memories),
            skill_count=len(skills),
        )

    def _semantic_context(self):
        """Shared degraded-condition handling for every semantic entry
        point below: returns `None` for every condition that means "the
        semantic channel cannot be used right now" (extra not installed,
        index never built, mid-rebuild, model-config mismatch) -- never
        raises. Returns `(semantic_module, model, meta)` otherwise. Model
        loading is cached process-wide inside `_semantic.py`, so calling
        this more than once per `preflight()`/`guard()` call is cheap.
        """
        semantic = _load_semantic_module()
        if semantic is None:
            return None
        semantic_store = SemanticIndexStore.open_if_exists(self._semantic_db_path)
        if semantic_store is None:
            return None
        with semantic_store:
            meta = semantic_store.meta()
            if meta is None or meta.status != "ready":
                return None
            if not meta.matches(
                provider=semantic.SEMANTIC_PROVIDER,
                model_id=semantic.SEMANTIC_MODEL_ID,
                normalization=semantic.SEMANTIC_NORMALIZATION,
            ):
                return None
        model = semantic.load_model_for_retrieval()
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
        try:
            context = self._semantic_context()
            if context is None:
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
        except Exception:
            return frozenset()

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

        Diagnosed in A7.8 from a real Human Acceptance miss: plain
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

        Falls back to empty -- never raises -- on any degraded
        condition, exactly like `_semantic_widen`.
        """
        try:
            context = self._semantic_context()
            if context is None:
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

            return top_cluster
        except Exception:
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
        `preflight()` ONLY (see `_guard.py`/A7.7 report for why `guard()`
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
        try:
            context = self._semantic_context()
            if context is None:
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
        except Exception:
            return frozenset()
