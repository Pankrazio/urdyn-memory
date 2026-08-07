"""Tests for Cortex.discover(), Cortex.open(), and persistence round-trips."""

import pytest

from cortex_memory import Cortex, CortexManifestError, CortexNotFoundError


def test_discover_from_workspace_root(tmp_path):
    Cortex.init(tmp_path, "dev")

    cx = Cortex.discover(tmp_path)

    assert cx.profile == "dev"
    assert cx.path == tmp_path


def test_discover_from_nested_directory(tmp_path):
    Cortex.init(tmp_path, "dev")
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    cx = Cortex.discover(nested)

    assert cx.profile == "dev"
    assert cx.path == tmp_path


def test_discover_nearest_workspace_wins(tmp_path):
    Cortex.init(tmp_path, "general")
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    Cortex.init(nested_root, "lab")
    deeper = nested_root / "src"
    deeper.mkdir()

    cx = Cortex.discover(deeper)

    assert cx.path == nested_root
    assert cx.profile == "lab"


def test_discover_raises_when_not_found(tmp_path):
    with pytest.raises(CortexNotFoundError):
        Cortex.discover(tmp_path)


def test_open_requires_exact_workspace_root(tmp_path):
    Cortex.init(tmp_path, "dev")
    nested = tmp_path / "src"
    nested.mkdir()

    with pytest.raises(CortexNotFoundError):
        Cortex.open(nested)


def test_profile_survives_reopening(tmp_path):
    Cortex.init(tmp_path, "lab")

    reopened = Cortex.open(tmp_path)

    assert reopened.profile == "lab"


def test_cortex_id_survives_reopening(tmp_path):
    original = Cortex.init(tmp_path, "dev")

    reopened = Cortex.discover(tmp_path)

    assert reopened.cortex_id == original.cortex_id


def test_malformed_manifest_raises_explicit_error(tmp_path):
    cortex_dir = tmp_path / ".cortex"
    cortex_dir.mkdir()
    (cortex_dir / "manifest.json").write_text("{not valid json")

    with pytest.raises(CortexManifestError):
        Cortex.discover(tmp_path)


def test_unsupported_schema_version_is_rejected(tmp_path):
    cortex_dir = tmp_path / ".cortex"
    cortex_dir.mkdir()
    (cortex_dir / "manifest.json").write_text(
        '{"schema_version": 99, "profile": "dev", "cortex_id": "' + "a" * 32 + '"}'
    )

    with pytest.raises(CortexManifestError):
        Cortex.discover(tmp_path)


def test_invalid_cortex_id_format_is_rejected(tmp_path):
    cortex_dir = tmp_path / ".cortex"
    cortex_dir.mkdir()
    (cortex_dir / "manifest.json").write_text(
        '{"schema_version": 1, "profile": "dev", "cortex_id": "not-a-valid-id"}'
    )

    with pytest.raises(CortexManifestError):
        Cortex.discover(tmp_path)


def test_discover_does_not_skip_corrupted_nearby_workspace(tmp_path):
    """A corrupted .cortex/ closer to `start` must raise, not be silently
    bypassed in favor of a valid parent Cortex workspace."""
    Cortex.init(tmp_path, "general")
    nested_root = tmp_path / "nested"
    nested_cortex_dir = nested_root / ".cortex"
    nested_cortex_dir.mkdir(parents=True)
    (nested_cortex_dir / "manifest.json").write_text("{not valid json")
    deeper = nested_root / "src"
    deeper.mkdir()

    with pytest.raises(CortexManifestError):
        Cortex.discover(deeper)
