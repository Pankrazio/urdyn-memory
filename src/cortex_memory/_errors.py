"""Public exceptions raised by the Cortex Memory Engine."""


class CortexError(Exception):
    """Base class for all Cortex-specific errors."""


class CortexNotFoundError(CortexError):
    """Raised when no Cortex workspace can be located."""


class CortexAlreadyInitializedError(CortexError):
    """Raised when re-initializing a workspace with a conflicting profile."""


class CortexManifestError(CortexError):
    """Raised when a persisted Cortex manifest is missing or malformed."""


class CortexStorageError(CortexError):
    """Raised when the persisted memory store is missing, corrupted, or an
    unsupported schema version."""


class CortexSourceError(CortexError):
    """Raised when a path cannot be seeded as a project Source: it escapes
    the workspace, is not an ordinary text file, is too large, or matches
    the credential denylist.

    Deliberately its own class rather than the `ValueError` the rest of
    the write API raises for bad input. Seeding is the one operation that
    routinely runs over MANY caller-supplied paths at once (and a future
    automatic writer would run over more), where one rejected file is an
    expected per-item outcome rather than a caller bug. Catching
    `ValueError` to skip it would also swallow genuine programming errors;
    this class lets a batch caller skip exactly what Cortex refused."""


class CortexSemanticUnavailableError(CortexError):
    """Raised only by explicit semantic setup/maintenance calls (e.g. the
    `cortex semantic setup` CLI command) when the `cortex-memory[semantic]`
    optional dependency is not installed. Never raised by `preflight()` or
    `guard()`: those degrade silently to lexical/FTS-only instead, per
    A7.4's "no hidden download, no crash" requirement."""
