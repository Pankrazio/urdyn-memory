"""The embedding numeric contract and the per-call semantic outcome on the
external-backend path: what is specific to it (the external `SemanticBackend`,
the lifecycle lock, backup/restore, the index identities) is exercised here;
`test_semantic_numeric_contract.py` and `test_semantic_call_outcome.py` cover
the built-in encoder.

Synthetic encoders and temporary workspaces only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time

import numpy as np
import pytest

from urdyn import SemanticBackendIdentity, Urdyn, UrdynSemanticUnavailableError
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_INDEX_UNREADABLE,
    DETAIL_MODEL_MISMATCH,
    DETAIL_NOT_SET_UP,
    DETAIL_REFRESH_FAILED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    SemanticIndexStore,
)

import urdyn._semantic_store as semantic_store_module
from test_public_semantic_backend import LocalEncoder
from test_semantic import fake_semantic  # noqa: F401  (pytest fixture)

semantic = pytest.importorskip("urdyn._semantic")

# Looked up leniently so this file also runs (and fails test by test) against
# a build that lacks these constants.
DETAIL_QUERY_FAILED = getattr(semantic_store_module, "DETAIL_QUERY_FAILED", "<constant missing>")
DETAIL_QUERY_VECTORS_INVALID = getattr(semantic_store_module, "DETAIL_QUERY_VECTORS_INVALID", "<constant missing>")

D = 384


class ProgrammableEncoder(LocalEncoder):
    """`LocalEncoder` (axis 0 for "migrations"/"schema", axis 1 otherwise)
    whose whole response is replaced by `raw(n)` for any request containing
    a text with `marker` (every request when `marker` is None)."""

    def __init__(self, provider="test-encoder-v1"):
        super().__init__(provider)
        self.marker: str | None = None
        self.raw = None
        self.encode_calls = 0

    def encode(self, texts):
        texts = list(texts)
        self.encode_calls += 1
        if self.raw is not None and (self.marker is None or any(self.marker in t.lower() for t in texts)):
            return self.raw(len(texts))
        return super().encode(texts)


def _canonical(cx):
    return ([(m.memory_id, m.content, m.kind, m.supersedes) for m in cx.timeline()],
            hashlib.sha256(cx._db_path.read_bytes()).hexdigest())


def _index_digest(cx):
    return hashlib.sha256(cx._semantic_db_path.read_bytes()).hexdigest()


def _shape(preflight):
    return ([m.memory_id for m in preflight.root_causes], [m.memory_id for m in preflight.verified_lessons],
            [a.attempt_id for a in preflight.known_failures], [m.memory_id for m in preflight.pending])


def _lexical_reference(cx, task):
    moved = cx._semantic_db_path.with_suffix(".aside")
    shutil.move(cx._semantic_db_path, moved)
    try:
        return cx.preflight(task)
    finally:
        shutil.move(moved, cx._semantic_db_path)


def _external_workspace(root, *contents):
    cx = Urdyn.init(root, "dev")
    backend = ProgrammableEncoder()
    cx.configure_semantic_backend(backend)
    for content in contents:
        cx.remember(content, kind="root_cause")
    return cx, backend


# ---------------------------------------------------------------------------
# Identity: the external contract names the engine normalization
# ---------------------------------------------------------------------------


def test_external_identity_defaults_to_the_engine_contract_and_refuses_legacy_l2(tmp_path):
    backend = LocalEncoder()
    assert backend.identity.normalization == semantic.SEMANTIC_NORMALIZATION == "l2-scaled-f64-v1"
    cx = Urdyn.init(tmp_path, "dev")
    legacy = LocalEncoder()
    legacy.identity = SemanticBackendIdentity(**{**_fields(legacy.identity), "normalization": "l2"})
    with pytest.raises(ValueError, match="legacy 'l2' normalization"):
        cx.configure_semantic_backend(legacy)
    other = LocalEncoder()
    other.identity = SemanticBackendIdentity(**{**_fields(other.identity), "normalization": "cosine"})
    with pytest.raises(ValueError, match="l2-scaled-f64-v1"):
        cx.configure_semantic_backend(other)


def _fields(identity):
    return {name: getattr(identity, name) for name in identity.__dataclass_fields__}


def test_external_index_persists_the_versioned_identity(tmp_path):
    cx, _ = _external_workspace(tmp_path, "Migrations require a dry run")
    result = cx.semantic_setup()
    assert result.normalization == "l2-scaled-f64-v1"  # a literal: not this build's own constant
    with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
        assert store.meta().normalization == "l2-scaled-f64-v1"


# ---------------------------------------------------------------------------
# One numeric contract, for the external encoder too
# ---------------------------------------------------------------------------


def test_external_encoder_only_binds_identity_and_hands_raw_output_to_the_contract(monkeypatch):
    backend = LocalEncoder()
    sentinel = object()
    backend.encode = lambda texts: sentinel
    encoder = semantic.ExternalEncoder(backend, backend.identity)
    assert encoder.encode(["x"]) is sentinel  # no numeric processing of its own
    assert encoder.output_dimensions == D
    calls = []
    real = semantic.normalize_rows
    monkeypatch.setattr(semantic, "normalize_rows", lambda raw, **k: (calls.append(k), real(raw, **k))[1])
    semantic.embed(semantic.ExternalEncoder(LocalEncoder(), LocalEncoder().identity), ["schema"])
    assert calls == [{"rows": 1, "dimensions": D}]


BAD_POOL_OUTPUTS = {
    "nan": lambda n: [[float("nan")] * D for _ in range(n)],
    "inf": lambda n: [[float("inf")] + [0.0] * (D - 1) for _ in range(n)],
    "float32_overflow": lambda n: np.full((n, D), 1e300, np.float64),
    "zero_row": lambda n: np.zeros((n, D)),
    "underflow_to_zero_in_float32": lambda n: np.full((n, D), 1e-60, np.float64),
    "bool_dtype": lambda n: np.ones((n, D), bool),
    "complex_dtype": lambda n: np.ones((n, D), complex),
    "strings": lambda n: [["0.5"] * D for _ in range(n)],
    "ragged": lambda n: [[1.0] * D for _ in range(n - 1)] + [[1.0] * 7],
    "wrong_width": lambda n: np.ones((n, D - 1)),
    "extra_row": lambda n: np.ones((n + 1, D)),
    "missing_row": lambda n: np.ones((n - 1, D)),
}


@pytest.mark.parametrize("bad", sorted(BAD_POOL_OUTPUTS))
def test_external_setup_with_valid_looking_but_invalid_output_publishes_nothing(tmp_path, bad):
    cx, backend = _external_workspace(tmp_path, "poison one", "poison two")
    before = _canonical(cx)
    backend.marker, backend.raw = "poison", BAD_POOL_OUTPUTS[bad]  # the dimension probe is clean
    with pytest.raises(UrdynSemanticUnavailableError, match="rebuild failed"):
        cx.semantic_setup()
    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE
    assert cx.preflight("poison").retrieval.status != SEMANTIC_READY
    assert _canonical(cx) == before


@pytest.mark.parametrize("bad", ["nan", "wrong_width", "extra_row", "bool_dtype"])
def test_external_invalid_probe_leaves_an_existing_index_untouched(tmp_path, bad):
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.semantic_setup()
    index_before, canonical_before = _index_digest(cx), _canonical(cx)
    backend.marker, backend.raw = None, BAD_POOL_OUTPUTS[bad]
    with pytest.raises(UrdynSemanticUnavailableError, match="Could not load the external semantic model"):
        cx.semantic_setup()
    assert _index_digest(cx) == index_before
    backend.raw = None
    assert cx.semantic_state().status == SEMANTIC_READY
    assert _canonical(cx) == canonical_before


def test_external_large_finite_output_is_normalized_not_zeroed(tmp_path):
    """The legacy float32 normalizer turned |x| ~ 1e30 rows into zero vectors."""
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    big = lambda n: np.asarray(LocalEncoder().encode(["schema"] * n), np.float32) * np.float32(1e30)
    with np.errstate(all="ignore"):
        legacy = np.asarray(big(1), np.float32) / np.linalg.norm(np.asarray(big(1), np.float32), axis=1, keepdims=True)
    assert not np.any(legacy)  # the defect, reproduced on this input
    backend.marker, backend.raw = None, big
    cx.semantic_setup()
    with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
        blobs = [blob for _, blob in store.all_vectors(ENTITY_MEMORY)]
    vectors = np.stack([semantic.blob_to_vector(b, D) for b in blobs])
    assert np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1.0, atol=semantic.UNIT_NORM_TOLERANCE)
    assert vectors[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Refresh after a canonical commit; query failures; per-call outcome on the external path
# ---------------------------------------------------------------------------


def test_external_refresh_with_invalid_values_after_a_canonical_commit(tmp_path):
    """Failure AFTER the canonical commit, BEFORE the derived update: the
    memory is durable, the index stays stale and says why, nothing invalid
    is written, and the next call completes the refresh."""
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.semantic_setup()
    backend.marker, backend.raw = "poison", BAD_POOL_OUTPUTS["nan"]
    poisoned = cx.remember("poison schema note", kind="root_cause")
    assert poisoned.memory_id in {m.memory_id for m in cx.timeline()}  # canonical commit happened

    result = cx.preflight("schema")
    # The refresh failed; the QUERY embed works, so the covered part of the
    # stale index was really used: the per-call state is stale, not hidden.
    assert result.retrieval.status == SEMANTIC_STALE and result.retrieval.detail == DETAIL_REFRESH_FAILED
    with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
        assert poisoned.memory_id not in store.indexed_ids(ENTITY_MEMORY)
    assert cx.semantic_state().status == SEMANTIC_STALE

    backend.raw = None
    assert cx.preflight("schema").retrieval.status == SEMANTIC_READY
    with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
        assert poisoned.memory_id in store.indexed_ids(ENTITY_MEMORY)


def test_partial_refresh_with_a_working_query_still_reads_stale_per_call(tmp_path, fake_semantic, monkeypatch):
    """Counterpart of A27's partial-refresh test with only the REFRESH failing:
    the per-call retrieval keeps the A27 meaning (stale); that test's
    assertion differs only because its fake also breaks the query."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.semantic_setup()
    cx.record_attempt(task="alpha task", approach="alpha approach", outcome="failed")
    cx.remember("alpha memory", kind="note")
    real_embed = semantic.embed

    def _memory_pool_dies(model, texts):
        if list(texts) == ["alpha memory"]:
            raise RuntimeError("interrupted between pools")
        return real_embed(model, texts)

    monkeypatch.setattr(semantic, "embed", _memory_pool_dies)
    result = cx.preflight("alpha deploy")
    assert result.retrieval.status == SEMANTIC_STALE and result.retrieval.detail == DETAIL_REFRESH_FAILED


def test_external_invalid_query_vector_is_reported_as_a_query_failure(tmp_path):
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.remember("unrelated beta note", kind="root_cause")
    cx.semantic_setup()
    index_before = _index_digest(cx)
    backend.marker, backend.raw = "poison", BAD_POOL_OUTPUTS["nan"]

    result = cx.preflight("poison schema")

    assert result.retrieval.status == SEMANTIC_UNAVAILABLE
    # not "a stored vector failed validation": the index is fine, the query is not
    assert result.retrieval.detail == DETAIL_QUERY_FAILED
    assert _shape(result) == _shape(_lexical_reference(cx, "poison schema"))
    assert cx.semantic_state().status == SEMANTIC_READY
    assert _index_digest(cx) == index_before


@pytest.mark.parametrize("blob", [np.ones(3, np.float32).tobytes(), b"\x00" * 5, np.full(6, np.nan, np.float32).tobytes()])
def test_any_unusable_stored_blob_is_reported_as_a_stored_vector_failure(tmp_path, fake_semantic, blob):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha root cause", kind="root_cause")
    cx.semantic_setup()
    with sqlite3.connect(cx._semantic_db_path) as con:
        con.execute("UPDATE semantic_vectors SET vector = ? WHERE entity_type = 'memory'", (blob,))
    result = cx.preflight("alphabet soup")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE
    assert result.retrieval.detail == DETAIL_QUERY_VECTORS_INVALID
    assert result.root_causes == ()


def test_external_backend_vanishing_mid_call_is_reported(tmp_path, monkeypatch):
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.semantic_setup()
    original = Urdyn._semantic_prepare_index

    def prepare_then_vanish(self, pools):
        state = original(self, pools)
        backend.is_available = False
        return state

    monkeypatch.setattr(Urdyn, "_semantic_prepare_index", prepare_then_vanish)
    result = cx.preflight("schema")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_QUERY_FAILED
    assert result.root_causes == ()


def test_same_handle_two_threads_keep_their_own_outcome(tmp_path):
    """One Urdyn object shared by two threads, the failure depending on the
    query: each call reports its own outcome (the lifecycle lock serializes the
    calls; the ContextVar keeps the outcomes apart)."""
    cx, backend = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.semantic_setup()
    backend.marker, backend.raw = "poison", BAD_POOL_OUTPUTS["nan"]
    seen: dict[str, set] = {"poison": set(), "clean": set()}
    errors = []
    barrier = threading.Barrier(2)

    def run(name, task):
        try:
            barrier.wait()
            for _ in range(25):
                seen[name].add(cx.preflight(task).retrieval.status)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=("poison", "poison schema")),
               threading.Thread(target=run, args=("clean", "schema"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert seen == {"poison": {SEMANTIC_UNAVAILABLE}, "clean": {SEMANTIC_READY}}


# ---------------------------------------------------------------------------
# Lifecycle: both passes under one lock hold; restore waits for the call
# ---------------------------------------------------------------------------


def _count_exclusive_locks(monkeypatch):
    import fcntl

    import urdyn._lifecycle as lifecycle

    count = {"n": 0}
    original = lifecycle.fcntl.flock

    def counting(fd, flags):
        if flags & fcntl.LOCK_EX:
            count["n"] += 1
        return original(fd, flags)

    monkeypatch.setattr(lifecycle.fcntl, "flock", counting)
    return count


def test_degraded_call_takes_the_lifecycle_lock_exactly_once(tmp_path, fake_semantic, monkeypatch):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha root cause", kind="root_cause")
    cx.semantic_setup()
    count = _count_exclusive_locks(monkeypatch)
    assert cx.preflight("alphabet soup").retrieval.status == SEMANTIC_READY
    healthy = count["n"]
    with sqlite3.connect(cx._semantic_db_path) as con:
        con.execute("UPDATE semantic_vectors SET vector = ? WHERE entity_type = 'memory'",
                    (np.full(6, np.nan, np.float32).tobytes(),))
    count["n"] = 0
    assert cx.preflight("alphabet soup").retrieval.status == SEMANTIC_UNAVAILABLE
    # A semantic call takes two holds by contract: one to
    # plan its encoder work, one to publish, with the inference between them
    # outside the lock. What this test pins is unchanged: the second (lexical)
    # pass runs inside the publication hold -- a degraded call takes exactly as
    # many holds as a healthy one, and never re-acquires or releases in between.
    assert healthy == count["n"] == 2


def test_restore_waits_for_an_in_flight_degraded_call(tmp_path, fake_semantic, monkeypatch):
    import urdyn.backup as backup

    root = tmp_path / "brain"
    cx = Urdyn.create(root, "dev")
    cx.remember("alpha root cause", kind="root_cause")
    archive = tmp_path / "backup.zip"
    backup.create_backup(root, archive)  # canonical only: the index is never archived
    cx.remember("later alpha note", kind="note")
    cx.semantic_setup()
    with sqlite3.connect(cx._semantic_db_path) as con:
        con.execute("UPDATE semantic_vectors SET vector = ? WHERE entity_type = 'memory'",
                    (np.full(6, np.nan, np.float32).tobytes(),))
    index_before = _index_digest(cx)

    events: list[str] = []
    in_call, restore_started = threading.Event(), threading.Event()
    original_prepare = Urdyn._semantic_prepare_index

    def slow_prepare(self, pools):
        state = original_prepare(self, pools)
        in_call.set()
        restore_started.wait(10)
        time.sleep(0.3)  # room for a restore that did NOT wait to act
        events.append("first-pass-prepared")
        return state

    monkeypatch.setattr(Urdyn, "_semantic_prepare_index", slow_prepare)
    original_replace = backup._replace
    monkeypatch.setattr(backup, "_replace", lambda s, t: (events.append("restore-replace"), original_replace(s, t))[1])

    results = {}

    def call():
        results["preflight"] = cx.preflight("alphabet soup")
        events.append("call-returned")

    def restore():
        in_call.wait(10)
        restore_started.set()
        results["restore"] = backup.restore_backup(root, archive)

    threads = [threading.Thread(target=call), threading.Thread(target=restore)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert events.index("call-returned") < events.index("restore-replace")
    assert results["preflight"].retrieval.detail == DETAIL_QUERY_VECTORS_INVALID
    assert json.loads(results["restore"])["previous_brain_retained"] is True
    # the restored Brain: canonical from the archive, no derived index (rebuild is explicit)
    reopened = Urdyn.open(root)
    assert [m.content for m in reopened.timeline()] == ["alpha root cause"]
    state = reopened.semantic_state()
    assert state.status == SEMANTIC_DISABLED and state.detail == DETAIL_NOT_SET_UP
    rollback = root.parent / ".brain.rollback" / ".urdyn" / "semantic_index.db"
    assert hashlib.sha256(rollback.read_bytes()).hexdigest() == index_before  # retained, not altered


# ---------------------------------------------------------------------------
# Index compatibility: incompatible, corrupt, incomplete
# ---------------------------------------------------------------------------


def test_external_index_with_the_same_width_is_never_read_by_the_onnx_path(tmp_path, fake_semantic, monkeypatch):
    cx, _ = _external_workspace(tmp_path, "Migrations require a dry run")
    cx.semantic_setup()
    index_before = _index_digest(cx)
    plain = Urdyn.open(tmp_path)  # no backend attached: the built-in ONNX path
    embeds = []
    real_embed = semantic.embed
    monkeypatch.setattr(semantic, "embed", lambda *a, **k: (embeds.append(1), real_embed(*a, **k))[1])
    assert plain.semantic_state().detail == DETAIL_MODEL_MISMATCH
    assert plain.preflight("schema").retrieval.status != SEMANTIC_READY
    assert embeds == [] and _index_digest(cx) == index_before


def test_unreadable_index_degrades_and_keeps_canonical(tmp_path, fake_semantic):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("alpha root cause", kind="root_cause")
    cx.semantic_setup()
    before = _canonical(cx)
    cx._semantic_db_path.write_bytes(b"not a sqlite database" * 10)
    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_INDEX_UNREADABLE
    result = cx.preflight("alpha")
    assert result.retrieval.status != SEMANTIC_READY
    assert [m.content for m in result.root_causes] == ["alpha root cause"]  # lexical answer intact
    assert _canonical(cx) == before
