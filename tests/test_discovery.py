"""Tests for Urdyn.discover(), Urdyn.open(), and persistence round-trips."""

import json

import pytest

from urdyn import Urdyn, UrdynManifestError, UrdynNotFoundError
from urdyn._manifest import LEGACY_WORKSPACE_ID_KEY


def test_discover_from_workspace_root(tmp_path):
    Urdyn.init(tmp_path, "dev")

    cx = Urdyn.discover(tmp_path)

    assert cx.profile == "dev"
    assert cx.path == tmp_path


def test_discover_from_nested_directory(tmp_path):
    Urdyn.init(tmp_path, "dev")
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    cx = Urdyn.discover(nested)

    assert cx.profile == "dev"
    assert cx.path == tmp_path


def test_discover_nearest_workspace_wins(tmp_path):
    Urdyn.init(tmp_path, "general")
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    Urdyn.init(nested_root, "lab")
    deeper = nested_root / "src"
    deeper.mkdir()

    cx = Urdyn.discover(deeper)

    assert cx.path == nested_root
    assert cx.profile == "lab"


def test_discover_raises_when_not_found(tmp_path):
    with pytest.raises(UrdynNotFoundError):
        Urdyn.discover(tmp_path)


def test_open_requires_exact_workspace_root(tmp_path):
    Urdyn.init(tmp_path, "dev")
    nested = tmp_path / "src"
    nested.mkdir()

    with pytest.raises(UrdynNotFoundError):
        Urdyn.open(nested)


def test_profile_survives_reopening(tmp_path):
    Urdyn.init(tmp_path, "lab")

    reopened = Urdyn.open(tmp_path)

    assert reopened.profile == "lab"


def test_urdyn_id_survives_reopening(tmp_path):
    original = Urdyn.init(tmp_path, "dev")

    reopened = Urdyn.discover(tmp_path)

    assert reopened.urdyn_id == original.urdyn_id


def test_malformed_manifest_raises_explicit_error(tmp_path):
    urdyn_dir = tmp_path / ".urdyn"
    urdyn_dir.mkdir()
    (urdyn_dir / "manifest.json").write_text("{not valid json")

    with pytest.raises(UrdynManifestError):
        Urdyn.discover(tmp_path)


def test_unsupported_schema_version_is_rejected(tmp_path):
    urdyn_dir = tmp_path / ".urdyn"
    urdyn_dir.mkdir()
    (urdyn_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 99, "profile": "dev", LEGACY_WORKSPACE_ID_KEY: "a" * 32})
    )

    with pytest.raises(UrdynManifestError):
        Urdyn.discover(tmp_path)


def test_invalid_workspace_id_format_is_rejected(tmp_path):
    urdyn_dir = tmp_path / ".urdyn"
    urdyn_dir.mkdir()
    (urdyn_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "profile": "dev", LEGACY_WORKSPACE_ID_KEY: "not-a-valid-id"})
    )

    with pytest.raises(UrdynManifestError):
        Urdyn.discover(tmp_path)


def test_discover_does_not_skip_corrupted_nearby_workspace(tmp_path):
    """A corrupted .urdyn/ closer to `start` must raise, not be silently
    bypassed in favor of a valid parent Urdyn workspace."""
    Urdyn.init(tmp_path, "general")
    nested_root = tmp_path / "nested"
    nested_urdyn_dir = nested_root / ".urdyn"
    nested_urdyn_dir.mkdir(parents=True)
    (nested_urdyn_dir / "manifest.json").write_text("{not valid json")
    deeper = nested_root / "src"
    deeper.mkdir()

    with pytest.raises(UrdynManifestError):
        Urdyn.discover(deeper)
