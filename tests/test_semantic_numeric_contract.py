"""The numeric contract for embeddings: one normalizer, explicit
rejection of degenerate encoder output, validated reads, and a versioned
normalization identity that neither this build nor Urdyn 0.4.0 mistakes
for the other's.

No real model is loaded: encoders are synthetic, workspaces are temporary.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import numpy as np
import pytest

from urdyn import Urdyn, UrdynSemanticUnavailableError
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_MODEL_MISMATCH,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SemanticIndexStore,
)

from test_semantic import _FakeStaticModel, fake_semantic  # noqa: F401  (pytest fixture)

semantic = pytest.importorskip("urdyn._semantic")

REPO = Path(__file__).resolve().parents[1]
PUBLIC_CHECKPOINT = "301006c6592564280e2cf1960f7f3a084ac0baf7"
D = 384


class _ArrayModel:
    """Returns whatever `producer(n)` builds for a request of n texts."""

    identity = semantic.model_identity_for(semantic.ARTIFACT_PORTABLE)

    def __init__(self, producer):
        self.producer = producer

    def encode(self, texts):
        return self.producer(len(list(texts)))


def _legacy_embed(raw):
    """Urdyn 0.4.0's normalization, verbatim, as a reference only."""
    vectors = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _norm_error(vectors):
    return float(np.max(np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1.0)))


# ---------------------------------------------------------------------------
# The normalizer itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-44, 1e-40, 1e-30, 1e-20, 1e-3, 1.0, 1e3, 1e20, 1e30, 3e37])
def test_rows_of_any_finite_scale_normalize_to_the_true_direction(scale):
    rng = np.random.default_rng(3303)
    direction = rng.normal(size=(8, D))
    raw = (direction / np.max(np.abs(direction), axis=1, keepdims=True) * scale).astype(np.float32)
    assert np.all(np.max(np.abs(raw), axis=1) > 0), "fixture must not underflow to zero"
    out = semantic.normalize_rows(raw, rows=8)
    assert out.dtype == np.float32 and out.shape == (8, D)
    assert _norm_error(out) <= semantic.UNIT_NORM_TOLERANCE
    reference = raw.astype(np.float64)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)  # float64 cannot overflow at these scales
    cosine = np.sum(out.astype(np.float64) * reference, axis=1)
    assert np.all(cosine >= 1 - 1e-6)


def test_extreme_single_components_do_not_overflow():
    raw = np.zeros((2, D), dtype=np.float32)
    raw[0, :2] = np.finfo(np.float32).max
    raw[1, 7] = np.finfo(np.float32).smallest_subnormal
    out = semantic.normalize_rows(raw, rows=2)
    assert np.allclose(out[0, :2], 2 ** -0.5, atol=1e-7) and out[1, 7] == 1.0
    assert _norm_error(out) <= semantic.UNIT_NORM_TOLERANCE


def test_ordinary_rows_match_the_legacy_normalizer_to_float32_rounding():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(500, D)).astype(np.float32)
    new, old = semantic.normalize_rows(raw, rows=500), _legacy_embed(raw)
    assert float(np.max(np.abs(new - old))) <= 2e-7
    assert _norm_error(new) <= semantic.UNIT_NORM_TOLERANCE
    # Ranking stability on ordinary data: the same order for every query.
    candidates, queries = new[:400], new[400:]
    legacy_candidates, legacy_queries = old[:400], old[400:]
    for q_new, q_old in zip(queries, legacy_queries):
        assert np.array_equal(np.argsort(-(candidates @ q_new))[:10], np.argsort(-(legacy_candidates @ q_old))[:10])


def test_normalization_is_deterministic_and_row_independent():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5, D)).astype(np.float32)
    batch = semantic.normalize_rows(raw, rows=5)
    assert np.array_equal(batch, semantic.normalize_rows(raw.copy(), rows=5))
    for i in range(5):
        assert np.array_equal(batch[i], semantic.normalize_rows(raw[i : i + 1], rows=1)[0])


def _with(value, dtype=np.float32):
    x = np.full((1, D), 0.25, dtype=dtype)
    x[0, 3] = value
    return x


@pytest.mark.parametrize(
    "raw, rows",
    [
        (np.zeros((1, D), np.float32), 1),  # zero row
        (_with(np.nan), 1),
        (_with(np.inf), 1),
        (_with(-np.inf), 1),
        (_with(1e39, np.float64), 1),  # finite float64, overflows the float32 conversion
        (np.full((1, D), 1e-50, np.float64), 1),  # collapses to zero in the conversion
        (np.ones((2, D), np.float32), 1),  # extra row
        (np.ones((0, D), np.float32), 1),  # missing row
        (np.ones(D, np.float32), 1),  # 1-D
        (np.ones((1, 2, D), np.float32), 1),  # 3-D
        (np.ones((1, D), bool), 1),
        (np.ones((1, D), np.complex64), 1),
        (np.array([["a"] * D]), 1),
        (np.ones((1, 0), np.float32), 1),  # zero width
        (np.ones((1, D), np.float32), 0),  # rows for an empty request
    ],
)
def test_degenerate_encoder_output_is_rejected_not_repaired(raw, rows):
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.normalize_rows(raw, rows=rows)


def test_ragged_output_is_rejected():
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.normalize_rows([[1.0, 2.0], [1.0]], rows=2)


def test_wrong_width_is_rejected_when_the_width_is_known():
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.normalize_rows(np.ones((1, D - 1), np.float32), rows=1, dimensions=D)


def test_explicitly_empty_request_yields_no_rows():
    assert semantic.normalize_rows(np.zeros((0, 0), np.float32), rows=0).shape == (0, 0)
    assert semantic.embed(_ArrayModel(lambda n: np.zeros((0, 0), np.float32)), []).shape[0] == 0


def test_contract_error_is_a_semantic_unavailability():
    assert issubclass(semantic.InvalidSemanticVectors, semantic.SemanticUnavailable)


# ---------------------------------------------------------------------------
# Query path: stored rows are re-validated, NaN cannot be admitted
# ---------------------------------------------------------------------------


def _unit(seed=0):
    v = np.random.default_rng(seed).normal(size=D).astype(np.float32)
    return semantic.normalize_rows(v[None, :], rows=1)[0]


def test_nan_query_can_no_longer_be_admitted_in_a_single_candidate_pool():
    stored = [("m1", semantic.vector_to_blob(_unit()))]
    model = _ArrayModel(lambda n: np.full((n, D), np.nan, np.float32))
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.semantic_admitted_ids("q", ENTITY_MEMORY, model=model, stored_vectors=stored, dimensions=D)


@pytest.mark.parametrize(
    "blob",
    [
        np.full(D, np.nan, np.float32).tobytes(),
        np.zeros(D, np.float32).tobytes(),
        np.full(D, 0.5, np.float32).tobytes(),
    ],
)
def test_corrupted_stored_rows_are_rejected_on_read(blob):
    model = _ArrayModel(lambda n: np.repeat(_unit()[None, :], n, axis=0))
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.semantic_rank_eligible("q", model=model, stored_vectors=[("m1", blob)], dimensions=D)


def test_query_width_must_match_the_index():
    model = _ArrayModel(lambda n: np.ones((n, D - 1), np.float32))
    with pytest.raises(semantic.InvalidSemanticVectors):
        semantic.semantic_rank_eligible(
            "q", model=model, stored_vectors=[("m1", semantic.vector_to_blob(_unit()))], dimensions=D
        )


# ---------------------------------------------------------------------------
# Workspace: no partial index published, canonical memory untouched
# ---------------------------------------------------------------------------


def _canonical(cx):
    return [(m.memory_id, m.content, m.kind, m.supersedes) for m in cx.timeline()]


def _vector_rows(cx):
    with sqlite3.connect(cx._semantic_db_path) as con:
        return con.execute("SELECT entity_type, entity_id, vector FROM semantic_vectors").fetchall()


def test_setup_with_degenerate_encoder_publishes_nothing(tmp_path, fake_semantic, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha note")
    cx.remember("beta note")
    before = _canonical(cx)

    class _NanAfterProbe(_FakeStaticModel):
        def encode(self, texts):
            vectors = super().encode(texts)
            if list(texts) != ["_urdyn_semantic_dimension_probe_"]:
                vectors[:] = np.nan
            return vectors

    monkeypatch.setattr(semantic, "load_model_for_setup", lambda model_id=None: _NanAfterProbe())
    with pytest.raises(UrdynSemanticUnavailableError):
        cx.semantic_setup()
    state = cx.semantic_state()
    assert state.status != SEMANTIC_READY and state.detail == DETAIL_BUILD_INCOMPLETE
    assert _vector_rows(cx) == []
    assert _canonical(cx) == before


def test_degenerate_dimension_probe_fails_before_touching_an_existing_index(tmp_path, fake_semantic, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha note")
    cx.semantic_setup()
    rows_before = _vector_rows(cx)

    class _AllNan(_FakeStaticModel):
        def encode(self, texts):
            return np.full((len(list(texts)), 6), np.nan, np.float32)

    monkeypatch.setattr(semantic, "load_model_for_setup", lambda model_id=None: _AllNan())
    with pytest.raises(UrdynSemanticUnavailableError):
        cx.semantic_setup()
    assert _vector_rows(cx) == rows_before
    assert cx.semantic_state().status == SEMANTIC_READY


def test_refresh_with_degenerate_encoder_writes_nothing_and_stays_stale(tmp_path, fake_semantic, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha note")
    cx.semantic_setup()
    rows_before = _vector_rows(cx)
    cx.remember("beta note")
    before = _canonical(cx)
    monkeypatch.setattr(semantic, "load_model_for_retrieval",
                        lambda model_id=None: _ArrayModel(lambda n: np.zeros((n, 6), np.float32)))
    result = cx.preflight("beta")
    assert result.retrieval.status == SEMANTIC_STALE
    assert _vector_rows(cx) == rows_before
    assert _canonical(cx) == before


def test_corrupted_stored_vector_degrades_preflight_without_crashing(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha root cause", kind="root_cause")
    cx.semantic_setup()
    before = _canonical(cx)
    with sqlite3.connect(cx._semantic_db_path) as con:
        con.execute("UPDATE semantic_vectors SET vector = ?", (np.full(6, np.nan, np.float32).tobytes(),))
    result = cx.preflight("alpha")  # must not raise
    assert result is not None
    assert _canonical(cx) == before


# ---------------------------------------------------------------------------
# Compatibility: versioned identity, both directions
# ---------------------------------------------------------------------------


def test_identity_is_versioned():
    assert semantic.SEMANTIC_NORMALIZATION != semantic.LEGACY_SEMANTIC_NORMALIZATION == "l2"


def _write_legacy_index(cx, dims=6):
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(provider=semantic.SEMANTIC_PROVIDER,
                            model_id=semantic.model_identity_for(semantic.ARTIFACT_PORTABLE),
                            model_revision="fake-revision", dimensions=dims,
                            normalization=semantic.LEGACY_SEMANTIC_NORMALIZATION,
                            created_at="2026-08-24T00:00:00+00:00")
        store.add_vectors(ENTITY_MEMORY, [(m.memory_id, np.full(dims, np.nan, np.float32).tobytes()) for m in cx.state()])
        store.finish_rebuild()


def test_a_0_4_0_index_is_refused_and_rebuilt_explicitly(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha note")
    _write_legacy_index(cx)  # even one holding NaN rows, as 0.4.0 could write
    before = _canonical(cx)
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        assert semantic.artifact_for_index(store.meta()) is None
    state = cx.semantic_state()
    assert state.status != SEMANTIC_READY and state.detail == DETAIL_MODEL_MISMATCH
    assert cx.preflight("alpha").retrieval.status != SEMANTIC_READY  # never read, never topped up
    result = cx.semantic_setup()  # the documented remedy: rebuild the derived index
    assert result.normalization == semantic.SEMANTIC_NORMALIZATION
    assert cx.semantic_state().status == SEMANTIC_READY
    assert _canonical(cx) == before


_OLD_CLIENT = r"""
import json, sys
import numpy as np
import urdyn, urdyn._semantic as semantic
from urdyn import Urdyn

class Fake:
    identity = semantic.model_identity_for(semantic.ARTIFACT_PORTABLE)
    def encode(self, texts):
        return np.ones((len(list(texts)), 6), dtype=np.float32)
    def __call__(self, *a, **k):
        return self

fake = Fake()
semantic.load_model_for_setup = fake
semantic.load_model_for_retrieval = fake
semantic.resolve_local_revision = lambda *a, **k: "fake-revision"
semantic.artifacts_available = lambda meta: semantic.artifact_for_index(meta) is not None
embeds = []
real_embed = semantic.embed
semantic.embed = lambda *a, **k: (embeds.append(1), real_embed(*a, **k))[1]

cx = Urdyn.open(sys.argv[1])
from urdyn._semantic_store import SemanticIndexStore
with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
    meta = store.meta()
state = cx.semantic_state()
pre = cx.preflight("alpha")
print(json.dumps({"urdyn_file": urdyn.__file__, "old_normalization": semantic.SEMANTIC_NORMALIZATION,
                  "index_normalization": meta.normalization, "artifact": semantic.artifact_for_index(meta),
                  "status": state.status, "detail": state.detail, "preflight_status": pre.retrieval.status,
                  "embed_calls": len(embeds)}))
"""


def _public_checkpoint_src(tmp_path):
    try:
        archive = subprocess.run(["git", "archive", "--format=tar", PUBLIC_CHECKPOINT, "src/urdyn"],
                                 cwd=REPO, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"public checkpoint {PUBLIC_CHECKPOINT[:7]} not available in this repository")
    target = tmp_path / "urdyn-0.4.0"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return target / "src"


def test_a_0_4_0_client_refuses_an_index_written_by_this_build(tmp_path, fake_semantic):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cx = Urdyn.init(workspace, "dev")
    cx.remember("alpha note")
    cx.semantic_setup()
    assert cx.semantic_state().status == SEMANTIC_READY
    index_digest = hashlib.sha256(Path(cx._semantic_db_path).read_bytes()).hexdigest()
    canonical_before = _canonical(cx)

    old_src = _public_checkpoint_src(tmp_path)
    run = subprocess.run([sys.executable, "-B", "-c", _OLD_CLIENT, str(workspace)],
                         env={"PYTHONPATH": str(old_src), "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout.strip().splitlines()[-1])
    assert Path(seen["urdyn_file"]).resolve().is_relative_to(old_src.resolve())  # really the 0.4.0 code
    assert seen["old_normalization"] == "l2"
    assert seen["index_normalization"] == semantic.SEMANTIC_NORMALIZATION
    assert seen["artifact"] is None
    assert seen["status"] != "ready" and seen["detail"] == DETAIL_MODEL_MISMATCH
    assert seen["preflight_status"] != "ready"
    assert seen["embed_calls"] == 0  # the old client never encoded against, or topped up, the new index
    assert hashlib.sha256(Path(cx._semantic_db_path).read_bytes()).hexdigest() == index_digest
    assert _canonical(cx) == canonical_before


# ---------------------------------------------------------------------------
# Public API is unchanged
# ---------------------------------------------------------------------------


EXPECTED_PUBLIC_NAMES = [
    "Urdyn",
    "Memory",
    "Evidence",
    "Attempt",
    "Preflight",
    "PreflightConflict",
    "CompiledContext",
    "ContextSection",
    "ContextItem",
    "DEFAULT_CONTEXT_BUDGET",
    "Skill",
    "GuardResult",
    "SemanticSetupResult",
    "SemanticState",
    "SearchIndexState",
    "SemanticBackend",
    "SemanticBackendIdentity",
    "Conflict",
    "Source",
    "SourceObservation",
    "SeedResult",
    "UrdynError",
    "UrdynNotFoundError",
    "UrdynAlreadyInitializedError",
    "UrdynManifestError",
    "UrdynStorageError",
    "UrdynStorageUnavailableError",
    "UrdynStorageCorruptError",
    "UrdynStorageSchemaError",
    "UrdynSourceError",
    "UrdynSemanticUnavailableError",
]


def _published_names(revision):
    try:
        source = subprocess.run(["git", "show", f"{revision}:src/urdyn/__init__.py"],
                                cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"checkpoint {revision[:7]} not available")
    tree = ast.parse(source)
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets))


def test_public_api_names_are_exactly_the_documented_ones():
    """The public name list is pinned here, in order: adding, dropping or
    reordering a public name is a deliberate API change that must update this
    list (and the documentation) in the same change."""
    import urdyn

    assert list(urdyn.__all__) == EXPECTED_PUBLIC_NAMES


def test_every_0_4_0_public_name_is_still_exported():
    """Backward compatibility with the last release: every name 0.4.0
    exported is still exported (read from the release commit when the Git
    history is available)."""
    import urdyn

    assert set(_published_names(PUBLIC_CHECKPOINT)) <= set(urdyn.__all__)
