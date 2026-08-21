"""Public exceptions raised by the Urdyn Memory Engine."""


class UrdynError(Exception):
    """Base class for all Urdyn-specific errors."""


class UrdynNotFoundError(UrdynError):
    """Raised when no Urdyn workspace can be located."""


class UrdynAlreadyInitializedError(UrdynError):
    """Raised when re-initializing a workspace with a conflicting profile."""


class UrdynManifestError(UrdynError):
    """Raised when a persisted Urdyn manifest is missing or malformed."""


class UrdynStorageError(UrdynError):
    """Raised when the persisted memory store is missing, corrupted, or an
    unsupported schema version."""


class UrdynSourceError(UrdynError):
    """Raised when a path cannot be seeded as a project Source: it escapes
    the workspace, is not an ordinary text file, is too large, or matches
    the credential denylist.

    Deliberately its own class rather than the `ValueError` the rest of
    the write API raises for bad input. Seeding is the one operation that
    routinely runs over MANY caller-supplied paths at once (and a future
    automatic writer would run over more), where one rejected file is an
    expected per-item outcome rather than a caller bug. Catching
    `ValueError` to skip it would also swallow genuine programming errors;
    this class lets a batch caller skip exactly what Urdyn refused."""


class UrdynSemanticUnavailableError(UrdynError):
    """Raised only by explicit semantic setup/maintenance calls (e.g. the
    `urdyn semantic setup` CLI command) when the `urdyn-memory[semantic]`
    optional dependency is not installed. Never raised by `preflight()` or
    `guard()`: those degrade silently to lexical/FTS-only instead, per
    A7.4's "no hidden download, no crash" requirement."""
