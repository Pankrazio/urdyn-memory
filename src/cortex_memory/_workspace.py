"""The Cortex workspace: identity, profile, and lifecycle."""

from __future__ import annotations

import uuid
from pathlib import Path

from ._errors import CortexAlreadyInitializedError, CortexManifestError, CortexNotFoundError
from ._gitignore import ensure_gitignore_entry
from ._manifest import CANONICAL_PROFILES, SCHEMA_VERSION, read_manifest, write_manifest

CORTEX_DIRNAME = ".cortex"


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
