"""Exercise the public encoder contract with the real Urdyn workspace.

The deterministic encoder isolates lifecycle rules; it does not claim
numerical equivalence to MiniLM or Android inference.
"""

from __future__ import annotations

import pytest

from urdyn import SemanticBackendIdentity, Urdyn, UrdynSemanticUnavailableError
from urdyn._semantic import ARTIFACT_ARM64, ARTIFACT_ARM64_SHA256, TOKENIZER_SHA256, model_identity_for
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_EXTRA_MISSING,
    DETAIL_MODEL_MISMATCH,
    DETAIL_MODEL_UNCACHED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
)


class LocalEncoder:
    def __init__(self, provider="test-encoder-v1"):
        self.identity = SemanticBackendIdentity(
            provider=provider,
            model_id=model_identity_for(ARTIFACT_ARM64),
            model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            dimensions=384,
            artifact_sha256=ARTIFACT_ARM64_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            encoding_policy="deterministic-v1",
        )
        self.is_available = True
        self.calls = 0
        self.fail_after = None
        self.invalid = False

    def available(self):
        return self.is_available

    def encode(self, texts):
        self.calls += 1
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("inference interrupted")
        if self.invalid:
            return [[float("nan")] * 384 for _ in texts]
        rows = []
        for text in texts:
            vector = [0.0] * 384
            vector[0 if "migrations" in text.lower() or "schema" in text.lower() else 1] = 1.0
            rows.append(vector)
        return rows


def test_external_backend_rebuild_refresh_and_lexical_fallback(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder()
    workspace.configure_semantic_backend(backend)
    first = workspace.remember("Migrations require a dry run", kind="note")

    setup = workspace.semantic_setup()
    assert setup.provider == backend.identity.index_provider
    assert setup.dimensions == 384
    assert workspace.semantic_state().status == SEMANTIC_READY
    assert first.memory_id in {memory.memory_id for memory in workspace.recall("Migrations")}

    second = workspace.remember("Schema changes need review", kind="note")
    assert workspace.semantic_state().status == SEMANTIC_STALE
    workspace.preflight("schema")
    assert workspace.semantic_state().status == SEMANTIC_READY
    assert second.memory_id in {memory.memory_id for memory in workspace.recall("Schema")}

    backend.is_available = False
    state = workspace.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_MODEL_UNCACHED
    assert first.memory_id in {memory.memory_id for memory in workspace.recall("Migrations")}
    workspace.preflight("Migrations")  # lexical path remains usable


def test_index_identity_is_scoped_to_provider_and_instance(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    workspace.remember("Migrations require a dry run", kind="note")
    workspace.configure_semantic_backend(LocalEncoder())
    workspace.semantic_setup()

    other = Urdyn.open(tmp_path)
    other.configure_semantic_backend(LocalEncoder("test-encoder-v2"))
    assert other.semantic_state().detail == DETAIL_MODEL_MISMATCH
    assert other.recall("Migrations")

    different_batch = LocalEncoder()
    different_batch.identity = SemanticBackendIdentity(
        provider=different_batch.identity.provider,
        model_id=different_batch.identity.model_id,
        model_revision=different_batch.identity.model_revision,
        dimensions=384,
        artifact_sha256=ARTIFACT_ARM64_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        encoding_policy="batch8-v1",
    )
    other.configure_semantic_backend(different_batch)
    assert other.semantic_state().detail == DETAIL_MODEL_MISMATCH


def test_separate_workspaces_keep_backend_and_index_state_isolated(tmp_path):
    first = Urdyn.init(tmp_path / "first", "dev")
    second = Urdyn.init(tmp_path / "second", "dev")
    first_backend = LocalEncoder("first-client")
    second_backend = LocalEncoder("second-client")
    first.configure_semantic_backend(first_backend)
    second.configure_semantic_backend(second_backend)
    first.remember("Migrations require a dry run", kind="note")
    second.remember("Schema changes need review", kind="note")

    first.semantic_setup()
    second.semantic_setup()
    assert first.semantic_state().status == SEMANTIC_READY
    assert second.semantic_state().status == SEMANTIC_READY
    assert first.recall("Migrations")
    assert not second.recall("Migrations")

    first_backend.is_available = False
    assert first.semantic_state().detail == DETAIL_MODEL_UNCACHED
    assert second.semantic_state().status == SEMANTIC_READY
    assert second.recall("Schema")


def test_default_without_python_extra_keeps_legacy_status(tmp_path, monkeypatch):
    import urdyn._semantic as semantic

    workspace = Urdyn.init(tmp_path, "dev")
    workspace.configure_semantic_backend(LocalEncoder())
    workspace.semantic_setup()
    monkeypatch.setattr(semantic, "default_runtime_available", lambda: False)

    plain = Urdyn.open(tmp_path)
    assert plain.semantic_state().detail == DETAIL_EXTRA_MISSING
    assert plain.recall("Migrations") == []


def test_interrupted_rebuild_never_publishes_partial_index(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    workspace.remember("Migrations require a dry run", kind="note")
    backend = LocalEncoder()
    workspace.configure_semantic_backend(backend)
    backend.fail_after = 2  # dimension probe succeeds; the first pool fails

    with pytest.raises(UrdynSemanticUnavailableError, match="rebuild failed"):
        workspace.semantic_setup()
    state = workspace.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE
    assert state.detail == DETAIL_BUILD_INCOMPLETE
    assert workspace.recall("Migrations")


def test_invalid_vectors_are_rejected_before_rebuild(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder()
    backend.invalid = True
    workspace.configure_semantic_backend(backend)
    with pytest.raises(UrdynSemanticUnavailableError):
        workspace.semantic_setup()
    assert workspace.semantic_state().status != SEMANTIC_READY


def test_external_setup_and_retrieval_do_not_resolve_hub_artifacts(tmp_path, monkeypatch):
    import urdyn._semantic as semantic

    def no_hub(*args, **kwargs):
        pytest.fail("external backend must not access the model hub")

    monkeypatch.setattr(semantic, "_artifact_path", no_hub)
    workspace = Urdyn.init(tmp_path, "dev")
    workspace.configure_semantic_backend(LocalEncoder())
    workspace.remember("Migrations require a dry run", kind="note")
    workspace.semantic_setup()
    assert workspace.semantic_state().status == SEMANTIC_READY
    workspace.preflight("schema")


def test_changed_backend_identity_disables_existing_index(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder()
    workspace.configure_semantic_backend(backend)
    workspace.semantic_setup()
    backend.identity = SemanticBackendIdentity(
        provider="test-encoder-v2",
        model_id=backend.identity.model_id,
        model_revision=backend.identity.model_revision,
        dimensions=384,
        artifact_sha256=ARTIFACT_ARM64_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        encoding_policy="deterministic-v1",
    )
    assert workspace.semantic_state().detail == DETAIL_MODEL_UNCACHED


def test_uncalibrated_model_identity_is_rejected(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder()
    backend.identity = SemanticBackendIdentity(
        provider="different-model",
        model_id="unrelated/model@revision#artifact",
        model_revision="revision",
        dimensions=384,
        artifact_sha256=ARTIFACT_ARM64_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        encoding_policy="deterministic-v1",
    )
    with pytest.raises(ValueError, match="calibrated"):
        workspace.configure_semantic_backend(backend)


def test_wrong_asset_digest_is_rejected(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder()
    backend.identity = SemanticBackendIdentity(
        provider="test-encoder-v1",
        model_id=model_identity_for(ARTIFACT_ARM64),
        model_revision=backend.identity.model_revision,
        dimensions=384,
        artifact_sha256="0" * 64,
        tokenizer_sha256=TOKENIZER_SHA256,
        encoding_policy="deterministic-v1",
    )
    with pytest.raises(ValueError, match="artifact SHA-256"):
        workspace.configure_semantic_backend(backend)


def test_provider_identity_cannot_inject_index_fields(tmp_path):
    workspace = Urdyn.init(tmp_path, "dev")
    backend = LocalEncoder("backend;artifact=other")
    with pytest.raises(ValueError, match="provider identity"):
        workspace.configure_semantic_backend(backend)
