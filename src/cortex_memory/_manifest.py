"""Persistence for the Cortex workspace manifest.

The manifest is small identity/configuration metadata (workspace profile
and a stable identifier). It is not the Cortex memory store itself.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ._errors import CortexManifestError

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CANONICAL_PROFILES = frozenset({"general", "dev", "lab"})
_CORTEX_ID_RE = re.compile(r"[0-9a-f]{32}")


def write_manifest(cortex_dir: Path, data: dict) -> None:
    """Write the manifest atomically to avoid partial-write corruption."""
    manifest_path = cortex_dir / MANIFEST_FILENAME
    tmp_path = cortex_dir / f".{MANIFEST_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def read_manifest(cortex_dir: Path) -> dict:
    """Read and validate the manifest, failing explicitly on malformed data."""
    manifest_path = cortex_dir / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CortexManifestError(
            f"Cortex directory {cortex_dir} exists but has no manifest at {manifest_path}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CortexManifestError(f"Malformed Cortex manifest at {manifest_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise CortexManifestError(f"Malformed Cortex manifest at {manifest_path}: expected an object")

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise CortexManifestError(
            f"Unsupported Cortex manifest schema_version {schema_version!r} at {manifest_path} "
            f"(expected {SCHEMA_VERSION})"
        )

    profile = data.get("profile")
    if profile not in CANONICAL_PROFILES:
        raise CortexManifestError(f"Invalid or missing profile {profile!r} in manifest at {manifest_path}")

    cortex_id = data.get("cortex_id")
    if not isinstance(cortex_id, str) or not _CORTEX_ID_RE.fullmatch(cortex_id):
        raise CortexManifestError(f"Invalid or missing cortex_id in manifest at {manifest_path}")

    return data
