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


class CortexSemanticUnavailableError(CortexError):
    """Raised only by explicit semantic setup/maintenance calls (e.g. the
    `cortex semantic setup` CLI command) when the `cortex-memory[semantic]`
    optional dependency is not installed. Never raised by `preflight()` or
    `guard()`: those degrade silently to lexical/FTS-only instead, per
    A7.4's "no hidden download, no crash" requirement."""
