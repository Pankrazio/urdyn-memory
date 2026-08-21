"""Tests for Urdyn workspace initialization."""

import json

import pytest

from urdyn import Urdyn, UrdynAlreadyInitializedError, UrdynManifestError
from urdyn._manifest import LEGACY_WORKSPACE_ID_KEY, MANIFEST_FILENAME, SCHEMA_VERSION


@pytest.mark.parametrize("profile", ["general", "dev", "lab"])
def test_init_creates_workspace_with_profile(tmp_path, profile):
    cx = Urdyn.init(tmp_path, profile)

    assert cx.profile == profile
    assert cx.path == tmp_path
    assert (tmp_path / ".urdyn").is_dir()


def test_init_default_profile_is_general(tmp_path):
    cx = Urdyn.init(tmp_path)

    assert cx.profile == "general"


def test_init_writes_valid_manifest(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")

    manifest_path = tmp_path / ".urdyn" / MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text())

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["profile"] == "dev"
    assert data[LEGACY_WORKSPACE_ID_KEY] == cx.urdyn_id
    assert data[LEGACY_WORKSPACE_ID_KEY]


def test_repeated_init_same_profile_is_idempotent(tmp_path):
    first = Urdyn.init(tmp_path, "dev")
    second = Urdyn.init(tmp_path, "dev")

    assert first.urdyn_id == second.urdyn_id
    assert first.profile == second.profile == "dev"


def test_repeated_init_different_profile_raises(tmp_path):
    Urdyn.init(tmp_path, "dev")

    with pytest.raises(UrdynAlreadyInitializedError):
        Urdyn.init(tmp_path, "lab")

    # the original workspace must be left untouched
    cx = Urdyn.open(tmp_path)
    assert cx.profile == "dev"


def test_repeated_init_preserves_urdyn_id(tmp_path):
    first = Urdyn.init(tmp_path, "general")
    second = Urdyn.init(tmp_path, "general")

    assert first.urdyn_id == second.urdyn_id


def test_init_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError):
        Urdyn.init(tmp_path, "not-a-profile")


def test_init_on_corrupted_workspace_raises_and_preserves_data(tmp_path):
    urdyn_dir = tmp_path / ".urdyn"
    urdyn_dir.mkdir()
    manifest_path = urdyn_dir / MANIFEST_FILENAME
    manifest_path.write_text("{not valid json")

    with pytest.raises(UrdynManifestError):
        Urdyn.init(tmp_path, "dev")

    # the corrupted manifest must not be silently replaced
    assert manifest_path.read_text() == "{not valid json"


def test_init_rejects_urdyn_path_that_is_a_file(tmp_path):
    urdyn_path = tmp_path / ".urdyn"
    urdyn_path.write_text("not a directory")

    with pytest.raises(UrdynManifestError):
        Urdyn.init(tmp_path, "dev")

    assert urdyn_path.read_text() == "not a directory"
