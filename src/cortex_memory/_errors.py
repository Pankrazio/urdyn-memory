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
