"""The experimental per-text recipe of the built-in ONNX encoder.

The real loaders and the real `_OnnxTextEncoder` run here; ONNX Runtime and
`tokenizers` are replaced at their boundary (the `lab` fixture of
`test_representation_identity.py`). The fake
session COUPLES the rows of a Run the way the real graph's
`DynamicQuantizeLinear` does: one scalar computed over
the whole input tensor -- companions and padding included -- scales every
row. So a batch dependence is observable here, and its absence under
per-text is a discriminating result, not a property of a trivial fake. What
ONNX Runtime itself does needs the real artifact. Synthetic Brains only.
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

import urdyn._representation as rep
import urdyn._semantic as semantic
from test_representation_identity import (  # noqa: F401  (`lab` is a fixture)
    D,
    FACT,
    QUERY,
    FakeSession,
    _Backend,
    _identity,
    assert_explicit_refusal,
    brain,
    canonical,
    index_digest,
    lab,
    stored,
)
from urdyn import Urdyn, UrdynSemanticUnavailableError, backup
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_MODEL_MISMATCH,
    DETAIL_NOT_SET_UP,
    DETAIL_REPRESENTATION_MISMATCH,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
)

per_text = semantic._experimental_per_text_encoding
NOTES = ["Rollback scripts must be rehearsed", "Feature flags need an owner and an expiry date",
         "Cache keys must include the tenant", "Never deploy schema changes on a Friday evening"]


@pytest.fixture
def coupled(lab, monkeypatch):
    """The lab, with a session whose output for one row depends on the whole
    Run (like a per-tensor quantization scale), and a record of every Run."""
    runs: list[tuple[tuple, bool]] = []

    def run(self, output_names, feed, run_options=None):
        FakeSession.runs += 1
        if FakeSession.hold is not None and FakeSession.runs >= FakeSession.hold_from:
            FakeSession.entered.set()
            FakeSession.hold.wait(30)
        if "fail_at" in FakeSession.quirks and FakeSession.runs == FakeSession.quirks["fail_at"]:
            raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : synthetic failure")
        ids = feed["input_ids"].astype(np.float32)
        runs.append((ids.shape, bool(feed["attention_mask"].all())))
        scale = 1.0 + float(ids.max()) / 997.0 + 0.01 * ids.shape[1]  # whole-tensor statistic
        hidden = (ids[..., None] % 13 + 1) * (np.arange(D, dtype=np.float32) % 7 + 1) / 50.0
        hidden = hidden + scale * (np.arange(D, dtype=np.float32) % 3)
        return [hidden.astype(np.float32)]

    monkeypatch.setattr(FakeSession, "run", run)
    return runs


def vectors_of(cx):
    return dict(stored(cx)[1]["memory"])


def by_content(cx):
    ids = {m.memory_id: m.content for m in cx.timeline()}
    return {ids[k]: v for k, v in vectors_of(cx).items()}


def isolated(text):
    """What a query (one text, alone) gets: the reference representation."""
    with per_text():
        return semantic.embed(semantic.load_model_for_retrieval(semantic.preferred_artifact()), [text])[0]


# ---------------------------------------------------------------------------
# 1. The recipe: one unpadded sequence per Run, under its own identity
# ---------------------------------------------------------------------------


def test_per_text_gives_every_run_exactly_one_unpadded_sequence(coupled):
    texts = [f"note {i} " + "word " * (i % 7) for i in range(70)]
    with per_text():
        encoder = semantic.load_model_for_setup()
        assert semantic.embed(encoder, texts).shape == (70, D)
    assert len(coupled) == 70 and all(shape[0] == 1 and unpadded for shape, unpadded in coupled)
    coupled.clear()
    semantic.embed(semantic.load_model_for_setup(), texts)  # default: unchanged, batches of up to 32
    assert [shape[0] for shape, _ in coupled] == [32, 32, 6]


def test_a_padded_or_multi_sequence_run_is_refused_under_per_text(lab):
    encoder = semantic.load_model_for_setup()
    real = encoder._tokenizer

    class Padded:  # a tokenizer that pads the one text to a companion's length
        def encode_batch(self, texts):
            return real.encode_batch(texts + ["a much longer padding companion text"])[:1]

    encoder._tokenizer = Padded()
    with pytest.raises(semantic.SemanticUnavailable, match="one unpadded sequence"):
        encoder._encode(["short"], per_text=True)


def test_per_text_and_batch32_are_never_the_same_representation(lab):
    default = semantic.representation_of(semantic.load_model_for_setup())
    with per_text():
        view = semantic.load_model_for_setup()
        candidate = semantic.representation_of(view)
    assert isinstance(view, semantic._PerTextEncoder) and view._base is semantic._model_cache[view._base._artifact]
    same = default.descriptor
    other = candidate.descriptor
    assert same["encoder"]["assets"] == other["encoder"]["assets"]  # same model and tokenizer bytes
    assert same["encoder"]["recipe"]["dimensions"] == other["encoder"]["recipe"]["dimensions"] == D
    assert other["encoder"]["recipe"]["batch"] == {"policy": "per-text", "size": 1, "composition_independent": True}
    assert candidate.assurance == default.assurance == rep.CORE_VERIFIED
    assert candidate.fingerprint != default.fingerprint
    assert candidate.static_fingerprint != default.static_fingerprint  # refused before loading, both ways


def test_an_encoder_first_built_under_per_text_still_describes_batch32(lab, tmp_path):
    """The encoder is cached across contexts: its own identity must not
    depend on the context that happened to build it first."""
    with per_text():
        view = semantic.load_model_for_setup()  # the base is constructed HERE
        assert semantic.representation_of(view).descriptor["encoder"]["recipe"]["batch"]["policy"] == "per-text"
    base = semantic.load_model_for_setup()
    assert base is view._base
    assert semantic.representation_of(base).descriptor["encoder"]["recipe"]["batch"] == semantic._batch32_section()


def test_the_default_recipe_and_its_fingerprint_are_those_of_r03_7a(lab):
    assert semantic._active_batch_section() == {"policy": "fixed-chunks-in-call-order", "size": 32,
                                                "composition_independent": False}
    descriptor = semantic.representation_of(semantic.load_model_for_setup()).descriptor
    assert descriptor["encoder"]["recipe"]["batch"] == semantic._active_batch_section()


def test_external_backends_are_not_affected(lab, tmp_path):
    identity = _identity()
    expected = semantic.declared_representation(identity).fingerprint
    with per_text():
        assert semantic.declared_representation(identity).fingerprint == expected
        cx = brain(tmp_path)
        cx.configure_semantic_backend(_Backend(identity))
        cx.semantic_setup()
        assert rep.read_provider(stored(cx)[0].provider)[1].fingerprint == expected
        assert cx.semantic_state().status == SEMANTIC_READY


# ---------------------------------------------------------------------------
# 2. P1/P2: independence from the corpus, coherence of setup/refresh/query
# ---------------------------------------------------------------------------


def test_a_text_does_not_depend_on_its_companions_order_or_duplicates_under_per_text(coupled):
    target = "the schema migration failed on the second deploy"
    groups = [[target], ["x", target, "y" * 40], NOTES + [target] + NOTES, [target, target, "z"],
              list(reversed(NOTES + [target]))]
    with per_text():
        encoder = semantic.load_model_for_setup()
        rows = [semantic.embed(encoder, g)[g.index(target)] for g in groups]
    assert all(np.array_equal(rows[0], row) for row in rows[1:])
    batch = [semantic.embed(semantic.load_model_for_setup(), g)[g.index(target)] for g in groups]
    assert not all(np.array_equal(batch[0], row) for row in batch[1:])  # the fake does couple a batch


def test_setup_refresh_and_query_give_one_vector_per_text_under_per_text(coupled, tmp_path):
    with per_text():
        cx = brain(tmp_path, *NOTES[:2])
        cx.semantic_setup()
        for note in NOTES[2:]:
            cx.remember(note, kind="root_cause")
        assert cx.semantic_state().status == SEMANTIC_STALE
        refreshed = cx.preflight(QUERY)
        assert refreshed.retrieval.status == SEMANTIC_READY and refreshed.retrieval.refreshed == 2
        assert cx.preflight(QUERY).retrieval.status == SEMANTIC_READY
    stored_rows = by_content(cx)
    assert len(stored_rows) == 5
    for text, blob in stored_rows.items():  # setup rows and refresh rows alike
        assert blob == semantic.vector_to_blob(isolated(text)), text


def test_batch32_setup_rows_differ_from_the_query_representation(coupled, tmp_path):
    """The defect per-text removes, shown on the same fake: with the default
    recipe an indexed text is not stored as a query of it would encode it."""
    cx = brain(tmp_path, *NOTES)
    cx.semantic_setup()
    differing = [t for t, blob in by_content(cx).items() if blob != semantic.vector_to_blob(isolated(t))]
    assert differing


def test_a_rebuild_reproduces_the_same_per_text_index(coupled, tmp_path):
    with per_text():
        cx = brain(tmp_path, *NOTES)
        cx.semantic_setup()
        first = by_content(cx)
        semantic._model_cache.clear()  # a new session, as after a restart
        cx.remember("Feature flags need an owner and an expiry date", kind="note")  # a duplicate text
        cx.semantic_setup()
        second = by_content(cx)
    assert {t: first[t] for t in first} == {t: second[t] for t in first}


# ---------------------------------------------------------------------------
# 3. P3: identity -- each index is usable only under its own profile
# ---------------------------------------------------------------------------


def test_a_per_text_index_is_refused_outside_its_profile_and_reopens_inside_it(coupled, tmp_path):
    with per_text():
        cx = brain(tmp_path)
        cx.semantic_setup()
    before = (index_digest(cx), canonical(cx))
    semantic._model_cache.clear()
    FakeSession.loaded.clear()
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)  # default profile: refused
    assert FakeSession.loaded == []  # from the static fingerprint, no model loaded
    assert (index_digest(cx), canonical(cx)) == before
    with per_text():
        reopened = Urdyn.open(cx.path)
        assert reopened.semantic_state().status == SEMANTIC_READY
        assert reopened.preflight(QUERY).retrieval.status == SEMANTIC_READY


def test_a_batch32_index_is_refused_under_per_text_and_never_topped_up(coupled, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()  # default batch32 index
    cx.remember(NOTES[0], kind="root_cause")
    before = (index_digest(cx), stored(cx)[1])
    with per_text():
        assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)
    assert (index_digest(cx), stored(cx)[1]) == before


def test_an_index_of_another_profile_is_refused_under_per_text(coupled, tmp_path):
    cx = brain(tmp_path)
    cx.configure_semantic_backend(_Backend(_identity()))
    cx.semantic_setup()
    with per_text():
        assert_explicit_refusal(Urdyn.open(cx.path), DETAIL_MODEL_MISMATCH)
        legacy = brain(tmp_path / "legacy")
        legacy.semantic_setup()
    meta, vectors = stored(legacy)
    from test_representation_identity import write_index
    write_index(legacy, provider=rep.base_provider(meta.provider))  # shaped like an index written before representation references
    with per_text():
        assert legacy.semantic_state().detail == "the index has no verifiable representation identity"


# ---------------------------------------------------------------------------
# 4. P4/P5: canonical integrity and lifecycle under per-text
# ---------------------------------------------------------------------------


def test_an_inference_error_publishes_nothing_and_rollback_is_a_rebuild(coupled, tmp_path):
    cx = brain(tmp_path, *NOTES)
    before = canonical(cx)
    FakeSession.quirks["fail_at"] = 4  # probe, then the third text of the setup
    # unchanged setup contract: a built-in ONNX error propagates from setup as is
    with per_text(), pytest.raises(RuntimeError, match="synthetic failure"):
        cx.semantic_setup()
    assert (cx.semantic_state().status, cx.semantic_state().detail) == (SEMANTIC_UNAVAILABLE,
                                                                        DETAIL_BUILD_INCOMPLETE)
    assert canonical(cx) == before
    FakeSession.quirks.pop("fail_at")
    cx.semantic_setup()  # rollback to the default profile: a rebuild
    assert cx.semantic_state().status == SEMANTIC_READY, cx.semantic_state()
    with per_text():
        assert cx.semantic_state().detail == DETAIL_REPRESENTATION_MISMATCH
        cx.semantic_setup()  # and forward again
        assert cx.semantic_state().status == SEMANTIC_READY
    assert canonical(cx) == before


def test_restore_during_a_per_text_rebuild_publishes_nothing(coupled, tmp_path):
    cx = brain(tmp_path, *NOTES)
    archive = tmp_path / "backup.zip"
    backup.create_backup(cx.path, archive)
    cx.remember("written after the backup", kind="note")
    FakeSession.hold, FakeSession.hold_from = threading.Event(), 2
    box = {}

    def rebuild():
        with per_text():
            try:
                cx.semantic_setup()
                box["setup"] = "published"
            except UrdynSemanticUnavailableError as exc:
                box["setup"] = str(exc)

    thread = threading.Thread(target=rebuild)
    thread.start()
    try:
        assert FakeSession.entered.wait(10)
        backup.restore_backup(cx.path, archive)  # not blocked by the rebuild
    finally:
        FakeSession.hold.set()
        thread.join(30)
    assert "nothing was published" in box["setup"]
    restored = Urdyn.open(cx.path)
    assert (restored.semantic_state().status, restored.semantic_state().detail) == (SEMANTIC_DISABLED,
                                                                                    DETAIL_NOT_SET_UP)
    assert [m.content for m in restored.timeline()] == [FACT, *NOTES]


def test_late_per_text_rows_are_not_published_into_an_index_replaced_meanwhile(coupled, tmp_path):
    """A per-text call encodes its refresh outside the lock ; before
    it publishes, another rebuild -- another representation -- replaces the
    index. The late rows must not land in it."""
    with per_text():
        cx = brain(tmp_path)
        cx.semantic_setup()
    pending = cx.remember(NOTES[0], kind="root_cause")
    other = Urdyn.open(cx.path)
    other.configure_semantic_backend(_Backend(_identity()))
    FakeSession.hold, FakeSession.hold_from = threading.Event(), FakeSession.runs + 1
    box = {}

    def late():
        with per_text():
            box["result"] = Urdyn.open(cx.path).preflight(QUERY)

    thread = threading.Thread(target=late)
    thread.start()
    try:
        assert FakeSession.entered.wait(10)  # the per-text refresh is inside its Run
        other.semantic_setup()  # a declared index replaces the per-text one meanwhile
        declared = stored(cx)[1]
    finally:
        FakeSession.hold.set()
        thread.join(30)
    assert stored(cx)[1] == declared  # nothing of the late per-text rows was added
    assert pending.memory_id in dict(declared["memory"])  # (covered by the declared rebuild itself)
    result = box["result"]
    assert result.retrieval.status != SEMANTIC_READY and any(m.content == FACT for m in result.root_causes)


def test_the_experimental_deadline_still_bounds_one_budget_for_all_per_text_runs(coupled, monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(semantic, "_monotonic", lambda: clock["now"])
    monkeypatch.setattr(semantic, "_deadline_timer", lambda delay, cb: threading.Timer(3600, cb))

    def tick(self, *args, **kwargs):  # every Run consumes 1 s of the budget
        clock["now"] += 1.0
        return original(self, *args, **kwargs)

    original = FakeSession.run
    monkeypatch.setattr(FakeSession, "run", tick)
    with per_text(), semantic._experimental_encode_deadline(2.5):
        encoder = semantic.load_model_for_setup()
        with pytest.raises(semantic.EncoderRunExpired) as caught:
            encoder.encode(["one", "two", "three", "four"])
    assert caught.value.terminate_requested is False  # expired between Runs, no Run cancelled
    assert not encoder.retired


def test_per_text_vectors_are_valid_under_the_r03_3b_contract(coupled):
    with per_text():
        vectors = semantic.embed(semantic.load_model_for_setup(), NOTES)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1.0, atol=semantic.UNIT_NORM_TOLERANCE)
    assert hashlib.sha256(vectors.tobytes()).hexdigest()  # rows are plain float32 bytes, as before


# ---------------------------------------------------------------------------
# 5. TOCTOU on the model asset: what the re-check does and does not see
# ---------------------------------------------------------------------------


def _rewrite_during_load(monkeypatch, content, *, keep_stat):
    import os

    original = FakeSession.__init__

    def loading(self, model, providers=None):
        before = os.stat(model)
        with open(model, "r+b") as handle:  # in place: same inode
            handle.write(content)
        if keep_stat:
            os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))  # restore mtime
        original(self, model, providers)

    monkeypatch.setattr(FakeSession, "__init__", loading)


def test_an_in_place_rewrite_during_load_is_refused_when_it_changes_the_file_identity(lab, monkeypatch):
    """The realistic case: a cached asset written long before. (A rewrite
    inside the SAME coarse kernel clock tick as the file's last modification
    keeps its mtime -- a limit of the re-check, stated in the report.)"""
    import os
    import time

    path = semantic._artifact_path(semantic.preferred_artifact(), local_files_only=True)
    old = time.time() - 3600
    os.utime(path, (old, old))
    size = len(open(path, "rb").read())
    _rewrite_during_load(monkeypatch, b"X" * size, keep_stat=False)
    with pytest.raises(semantic.SemanticUnavailable, match="changed while it was being loaded"):
        semantic.load_model_for_setup()


def test_limit_a_same_size_rewrite_that_restores_mtime_during_load_is_not_detected(lab, monkeypatch):
    """Documented limit, asserted so it stays visible: the digest describes
    the file as hashed; a writer that rewrites it in place with the same size
    and restores its mtime inside the load window is not seen. The guarantee
    assumes the asset is not rewritten adversarially while it loads."""
    path = semantic._artifact_path(semantic.preferred_artifact(), local_files_only=True)
    hashed = hashlib.sha256(open(path, "rb").read()).hexdigest()
    size = len(open(path, "rb").read())
    _rewrite_during_load(monkeypatch, b"Y" * size, keep_stat=True)
    encoder = semantic.load_model_for_setup()
    recorded = semantic.representation_of(encoder).descriptor["encoder"]["assets"]["model_sha256"]
    assert recorded == hashed != hashlib.sha256(open(path, "rb").read()).hexdigest()
