"""Persistence for the Urdyn workspace manifest.

The manifest is small identity/configuration metadata (workspace profile
and a stable identifier). It is not the Urdyn memory store itself.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ._errors import UrdynManifestError

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
# Named because A19.1 gives `dev` its first real behaviour: automatic
# project-context discovery (`urdyn seed` with no arguments) is offered
# only in this profile. The Source primitive underneath is profile-neutral
# -- what `dev` selects is the allowlist, not a different kind of data.
PROFILE_DEV = "dev"
CANONICAL_PROFILES = frozenset({"general", PROFILE_DEV, "lab"})
# The schema-v1 key predates the product rename. Keeping it unchanged lets a
# workspace remain readable after its directory is deliberately moved from
# the former product directory to `.urdyn/`, without an unversioned migration.
LEGACY_WORKSPACE_ID_KEY = "cortex_id"
_WORKSPACE_ID_RE = re.compile(r"[0-9a-f]{32}")


def write_manifest(urdyn_dir: Path, data: dict) -> None:
    """Write the manifest atomically to avoid partial-write corruption."""
    manifest_path = urdyn_dir / MANIFEST_FILENAME
    tmp_path = urdyn_dir / f".{MANIFEST_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def read_manifest(urdyn_dir: Path) -> dict:
    """Read and validate the manifest, failing explicitly on malformed data."""
    manifest_path = urdyn_dir / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UrdynManifestError(
            f"Urdyn directory {urdyn_dir} exists but has no manifest at {manifest_path}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UrdynManifestError(f"Malformed Urdyn manifest at {manifest_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise UrdynManifestError(f"Malformed Urdyn manifest at {manifest_path}: expected an object")

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise UrdynManifestError(
            f"Unsupported Urdyn manifest schema_version {schema_version!r} at {manifest_path} "
            f"(expected {SCHEMA_VERSION})"
        )

    profile = data.get("profile")
    if profile not in CANONICAL_PROFILES:
        raise UrdynManifestError(f"Invalid or missing profile {profile!r} in manifest at {manifest_path}")

    workspace_id = data.get(LEGACY_WORKSPACE_ID_KEY)
    if not isinstance(workspace_id, str) or not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise UrdynManifestError(f"Invalid or missing workspace identity in manifest at {manifest_path}")

    return data
