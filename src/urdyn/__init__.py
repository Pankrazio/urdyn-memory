"""Urdyn Memory Engine.

A standalone, local-first, persistent, model-independent memory engine.
"""

from ._errors import (
    UrdynAlreadyInitializedError,
    UrdynError,
    UrdynManifestError,
    UrdynNotFoundError,
    UrdynSemanticUnavailableError,
    UrdynSourceError,
    UrdynStorageError,
    UrdynStorageUnavailableError,
    UrdynStorageCorruptError,
    UrdynStorageSchemaError,
)
from ._attempt import Attempt
from ._conflict import Conflict
from ._context import DEFAULT_CONTEXT_BUDGET, CompiledContext, ContextItem, ContextSection
from ._evidence import Evidence
from ._guard import GuardResult
from ._memory import Memory
from ._preflight import Preflight, PreflightConflict
from ._semantic_store import SemanticState
from ._skill import Skill
from ._source import SeedResult, Source, SourceObservation
from ._workspace import Urdyn, SemanticSetupResult
from .semantic_backend import SemanticBackend, SemanticBackendIdentity
from ._store import SearchIndexState

__all__ = [
    "Urdyn",
    "Memory",
    "Evidence",
    "Attempt",
    "Preflight",
    "PreflightConflict",
    "CompiledContext",
    "ContextSection",
    "ContextItem",
    "DEFAULT_CONTEXT_BUDGET",
    "Skill",
    "GuardResult",
    "SemanticSetupResult",
    "SemanticState",
    "SearchIndexState",
    "SemanticBackend",
    "SemanticBackendIdentity",
    "Conflict",
    "Source",
    "SourceObservation",
    "SeedResult",
    "UrdynError",
    "UrdynNotFoundError",
    "UrdynAlreadyInitializedError",
    "UrdynManifestError",
    "UrdynStorageError",
    "UrdynStorageUnavailableError",
    "UrdynStorageCorruptError",
    "UrdynStorageSchemaError",
    "UrdynSourceError",
    "UrdynSemanticUnavailableError",
]
