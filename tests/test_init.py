"""Tests for Cortex workspace initialization."""

import json

import pytest

from cortex_memory import Cortex, CortexAlreadyInitializedError, CortexManifestError
from cortex_memory._manifest import MANIFEST_FILENAME, SCHEMA_VERSION


@pytest.mark.parametrize("profile", ["general", "dev", "lab"])
def test_init_creates_workspace_with_profile(tmp_path, profile):
    cx = Cortex.init(tmp_path, profile)

    assert cx.profile == profile
    assert cx.path == tmp_path
    assert (tmp_path / ".cortex").is_dir()


def test_init_default_profile_is_general(tmp_path):
    cx = Cortex.init(tmp_path)

    assert cx.profile == "general"


def test_init_writes_valid_manifest(tmp_path):
    cx = Cortex.init(tmp_path, "dev")

    manifest_path = tmp_path / ".cortex" / MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text())

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["profile"] == "dev"
    assert data["cortex_id"] == cx.cortex_id
    assert data["cortex_id"]


def test_repeated_init_same_profile_is_idempotent(tmp_path):
    first = Cortex.init(tmp_path, "dev")
    second = Cortex.init(tmp_path, "dev")

    assert first.cortex_id == second.cortex_id
    assert first.profile == second.profile == "dev"


def test_repeated_init_different_profile_raises(tmp_path):
    Cortex.init(tmp_path, "dev")

    with pytest.raises(CortexAlreadyInitializedError):
        Cortex.init(tmp_path, "lab")

    # the original workspace must be left untouched
    cx = Cortex.open(tmp_path)
    assert cx.profile == "dev"


def test_repeated_init_preserves_cortex_id(tmp_path):
    first = Cortex.init(tmp_path, "general")
    second = Cortex.init(tmp_path, "general")

    assert first.cortex_id == second.cortex_id


def test_init_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError):
        Cortex.init(tmp_path, "not-a-profile")


def test_init_on_corrupted_workspace_raises_and_preserves_data(tmp_path):
    cortex_dir = tmp_path / ".cortex"
    cortex_dir.mkdir()
    manifest_path = cortex_dir / MANIFEST_FILENAME
    manifest_path.write_text("{not valid json")

    with pytest.raises(CortexManifestError):
        Cortex.init(tmp_path, "dev")

    # the corrupted manifest must not be silently replaced
    assert manifest_path.read_text() == "{not valid json"


def test_init_rejects_cortex_path_that_is_a_file(tmp_path):
    cortex_path = tmp_path / ".cortex"
    cortex_path.write_text("not a directory")

    with pytest.raises(CortexManifestError):
        Cortex.init(tmp_path, "dev")

    assert cortex_path.read_text() == "not a directory"
