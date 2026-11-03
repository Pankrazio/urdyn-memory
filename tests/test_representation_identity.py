"""Verifiable identity of the semantic representation.

The REAL loaders (`load_model_for_setup` -> `_build_encoder` ->
`_construct_encoder`) run here; only the two native libraries are replaced at
their boundary (`onnxruntime.InferenceSession`, `tokenizers.Tokenizer`) and
the Hugging Face cache by synthetic asset files. So what is proven is the
core's contract -- which bytes are hashed, what is recorded, what is compared,
what is refused -- not ONNX Runtime numerics (which need the real artifact).
No personal Brain, no network, no real model.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import tokenizers

import urdyn._representation as rep
import urdyn._semantic as semantic
from urdyn import SemanticBackendIdentity, Urdyn, UrdynSemanticUnavailableError, backup
from urdyn._semantic_store import (
    DETAIL_MODEL_MISMATCH,
    DETAIL_NOT_SET_UP,
    DETAIL_REPRESENTATION_MISMATCH,
    DETAIL_REPRESENTATION_UNVERIFIED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    SemanticIndexStore,
)

D = 384
FACT = "Schema migrations failed without a dry run"
QUERY = "schema migrations dry run"


# ---------------------------------------------------------------------------
# Fakes at the native-library boundary
# ---------------------------------------------------------------------------


class _Encoding:
    def __init__(self, ids, mask):
        self.ids, self.attention_mask = ids, mask


class FakeTokenizer:
    """Stands in for `tokenizers.Tokenizer`; `quirks` injects a recipe the
    core must refuse to describe (read-back check)."""

    loaded: list[str] = []
    quirks: dict = {}

    def __init__(self, source: str):
        self.source = source
        self.truncation = None
        self.padding = None

    @classmethod
    def from_str(cls, text):
        cls.loaded.append(text)
        return cls(text)

    @classmethod
    def from_file(cls, path):  # pragma: no cover - the core must not use it any more
        raise AssertionError("the tokenizer must be loaded from the hashed bytes")

    def token_to_id(self, token):
        return 1 if token == "<pad>" else None

    def enable_truncation(self, max_length):
        self.truncation = {"max_length": self.quirks.get("max_length", max_length), "stride": 0,
                           "strategy": "longest_first", "direction": "right"}

    def enable_padding(self, pad_id, pad_token):
        self.padding = {"length": None, "pad_to_multiple_of": self.quirks.get("pad_to_multiple_of"),
                        "pad_id": pad_id, "pad_token": pad_token, "pad_type_id": 0, "direction": "right"}

    def encode_batch(self, texts):
        limit = self.truncation["max_length"]
        rows = [[2 + (int(hashlib.sha256(w.encode()).hexdigest(), 16) % 997) for w in t.split()][:limit - 1] + [2]
                for t in texts]
        width = max(len(r) for r in rows)
        return [_Encoding(r + [1] * (width - len(r)), [1] * len(r) + [0] * (width - len(r))) for r in rows]


class _Node:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class FakeSession:
    """Stands in for `onnxruntime.InferenceSession`. Records the model bytes it
    was given; `hold` blocks inside Run (concurrency tests)."""

    loaded: list = []
    quirks: dict = {}
    hold: threading.Event | None = None
    hold_from = 1  # the first Run that blocks (1: every Run; 2: after the dimension probe)
    runs = 0
    entered = threading.Event()

    def __init__(self, model, providers=None):
        FakeSession.loaded.append(model)
        if "replace_during_load" in self.quirks:  # another writer swaps the file mid-load
            replacement = Path(str(model) + ".new")
            replacement.write_bytes(self.quirks["replace_during_load"])
            replacement.replace(model)
        self.model = model
        self.providers = list(providers or [])

    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask"), _Node("token_type_ids")]

    def get_outputs(self):
        return [_Node("last_hidden_state", ["batch_size", "sequence_length", self.quirks.get("width", D)])]

    def get_providers(self):
        return self.quirks.get("providers", self.providers)

    def run(self, output_names, feed, run_options=None):
        FakeSession.runs += 1
        if FakeSession.hold is not None and FakeSession.runs >= FakeSession.hold_from:
            FakeSession.entered.set()
            FakeSession.hold.wait(30)
        ids = feed["input_ids"].astype(np.float32)
        hidden = (ids[..., None] % 13 + 1) * (np.arange(D, dtype=np.float32) % 7 + 1) / 50.0
        return [hidden.astype(np.float32)]


@dataclasses.dataclass
class Lab:
    assets: Path

    def write(self, name: str, content: bytes) -> None:
        """Replace an asset file, as a cache update or a tampered file would;
        the process-wide encoder cache is dropped, as a new process would."""
        path = self.assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        semantic._model_cache.clear()

    def digest(self, name: str) -> str:
        return hashlib.sha256((self.assets / name).read_bytes()).hexdigest()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setattr(semantic, "_model_cache", {})
    monkeypatch.setattr(semantic, "_retired_in_use", {})
    monkeypatch.setattr(semantic, "_building", {})
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(FakeTokenizer, "loaded", [])
    monkeypatch.setattr(FakeTokenizer, "quirks", {})
    monkeypatch.setattr(FakeSession, "loaded", [])
    monkeypatch.setattr(FakeSession, "quirks", {})
    monkeypatch.setattr(FakeSession, "hold", None)
    monkeypatch.setattr(FakeSession, "hold_from", 1)
    monkeypatch.setattr(FakeSession, "runs", 0)
    monkeypatch.setattr(FakeSession, "entered", threading.Event())

    def artifact_path(name, *, local_files_only):
        path = assets / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic asset {name} v1".encode())
        return str(path)

    monkeypatch.setattr(semantic, "_artifact_path", artifact_path)
    return Lab(assets)


def brain(tmp_path, *notes):
    cx = Urdyn.init(tmp_path / "brain", "dev")
    cx.remember(FACT, kind="root_cause")
    for note in notes:
        cx.remember(note, kind="root_cause")
    return cx


def canonical(cx):
    return ([(m.memory_id, m.content, m.kind) for m in cx.timeline()],
            hashlib.sha256(cx._db_path.read_bytes()).hexdigest())


def index_digest(cx):
    return hashlib.sha256(cx._semantic_db_path.read_bytes()).hexdigest()


def stored(cx):
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        return store.meta(), {t: store.all_vectors(t) for t in ("memory", "attempt", "skill")}


def write_index(cx, *, provider, rows=None, dimensions=D):
    """An index built elsewhere (another build, process or machine), written
    through the store's own API, with the vectors of the current index."""
    meta, vectors = stored(cx)
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as store:
        store.begin_rebuild(provider=provider, model_id=meta.model_id, model_revision=meta.model_revision,
                            dimensions=dimensions, normalization=meta.normalization, created_at=meta.created_at)
        for entity_type, entries in (rows or vectors).items():
            store.add_vectors(entity_type, entries)
        store.finish_rebuild()


def current_descriptor(cx):
    meta, _ = stored(cx)
    encoder = semantic.load_model_for_index(meta)
    return semantic.representation_of(encoder).descriptor


def foreign_provider(descriptor, mutate):
    """The provider column of an index built under `descriptor` after
    `mutate` -- a different, but complete and well-formed, representation."""
    changed = json.loads(json.dumps(descriptor))
    mutate(changed)
    return rep.tag_provider(semantic.SEMANTIC_PROVIDER, rep.Representation.build(changed))


def lexical_hit(result):
    return any(m.content == FACT for m in result.root_causes)


def assert_explicit_refusal(cx, detail, *, state_detail=None):
    """Refused, said so, remedy named -- and NOT read as "no memories": the
    lexical channel still answers with the canonical record."""
    state = cx.semantic_state()
    if state_detail is not False:
        assert (state.status, state.detail) == (SEMANTIC_UNAVAILABLE, state_detail or detail)
    result = cx.preflight(QUERY)
    assert (result.retrieval.status, result.retrieval.detail) == (SEMANTIC_UNAVAILABLE, detail)
    line = result.retrieval.retrieval_mode()
    assert "semantic retrieval is unavailable" in line and "urdyn semantic setup" in line
    assert lexical_hit(result)
    return result


# ---------------------------------------------------------------------------
# 1. What the core verifies when it builds the encoder
# ---------------------------------------------------------------------------


def test_the_digests_are_of_the_bytes_handed_to_the_runtime(lab):
    encoder = semantic.load_model_for_setup()
    descriptor = semantic.representation_of(encoder).descriptor
    assets = descriptor["encoder"]["assets"]
    (model_path,) = FakeSession.loaded
    (tokenizer_text,) = FakeTokenizer.loaded
    assert model_path == str(lab.assets / assets["artifact"])  # the file hashed, re-checked after loading
    assert assets["model_sha256"] == lab.digest(assets["artifact"])
    assert assets["tokenizer_sha256"] == hashlib.sha256(tokenizer_text.encode()).hexdigest() == lab.digest(
        semantic.TOKENIZER_FILENAME)
    assert descriptor["assurance"] == rep.CORE_VERIFIED
    recipe = descriptor["encoder"]["recipe"]
    assert recipe["dimensions"] == D and recipe["inputs"] == ["attention_mask", "input_ids", "token_type_ids"]
    assert recipe["batch"] == {"policy": semantic.ONNX_BATCH_POLICY, "size": semantic.SEMANTIC_ENCODE_BATCH_SIZE,
                               "composition_independent": False}
    assert descriptor["encoder"]["runtime"]["version"] == onnxruntime.__version__
    assert descriptor["core"]["normalization"] == semantic.SEMANTIC_NORMALIZATION


@pytest.mark.parametrize("quirk", [
    ("tokenizer", {"pad_to_multiple_of": 8}),    # same model, different padding
    ("tokenizer", {"max_length": 256}),          # same model, different truncation
    ("session", {"providers": ["OtherExecutionProvider"]}),  # silent provider fallback
    ("session", {"width": "hidden_size"}),       # width not fixed by the graph
    ("session", {"replace_during_load": b"swapped"}),  # file replaced between hash and load
])
def test_a_recipe_the_core_cannot_read_back_is_refused_not_described(lab, tmp_path, quirk):
    cx = brain(tmp_path)
    cx.semantic_setup()
    before = (index_digest(cx), canonical(cx))
    target, values = quirk
    (FakeTokenizer if target == "tokenizer" else FakeSession).quirks.update(values)
    semantic._model_cache.clear()
    with pytest.raises(UrdynSemanticUnavailableError):
        cx.semantic_setup()
    assert (index_digest(cx), canonical(cx)) == before  # the existing index and canonical memory untouched


def test_a_descriptor_names_every_field_or_is_refused():
    good = {"schema": rep.SCHEMA, "assurance": rep.UNATTESTED, "encoder": {"kind": rep.UNATTESTED_ENCODER_KIND},
            "core": semantic._core_section()}
    rep.Representation.build(good)
    broken = [
        lambda d: d.pop("core"),
        lambda d: d["core"]["environment"].pop("cpu_features"),
        lambda d: d["core"].update(extra="x"),
        lambda d: d.update(schema="urdyn-representation-v0"),
        lambda d: d.update(assurance="certified"),
        lambda d: d.update(assurance=rep.CORE_VERIFIED),  # kind/level mismatch, fields missing
        lambda d: d["core"]["text"].update(chunk_max_chars=0),
        lambda d: d["core"].update(normalization=""),
    ]
    for mutate in broken:
        candidate = json.loads(json.dumps(good))
        mutate(candidate)
        with pytest.raises(rep.InvalidRepresentation):
            rep.Representation.build(candidate)


def test_the_same_width_is_not_the_same_representation(lab):
    base = semantic.representation_of(semantic.load_model_for_setup()).descriptor
    variants = {
        "tokenizer": lambda d: d["encoder"]["assets"].update(tokenizer_sha256="1" * 64),
        "model bytes": lambda d: d["encoder"]["assets"].update(model_sha256="2" * 64),
        "padding": lambda d: d["encoder"]["recipe"]["padding"].update(strategy="fixed-128"),
        "batch": lambda d: d["encoder"]["recipe"]["batch"].update(size=1, composition_independent=True),
        "pooling": lambda d: d["encoder"]["recipe"].update(pooling="cls-token-v1"),
        "normalization": lambda d: d["core"].update(normalization="l2"),
        "runtime": lambda d: d["encoder"]["runtime"].update(version="1.23.2"),
        "environment": lambda d: d["core"]["environment"].update(machine="aarch64"),
        "text": lambda d: d["core"]["text"].update(policy="urdyn-pool-text-v2"),
    }
    reference = rep.Representation.build(base)
    for name, mutate in variants.items():
        other = json.loads(json.dumps(base))
        mutate(other)
        built = rep.Representation.build(other)
        assert built.descriptor["encoder"]["recipe"]["dimensions"] == D
        assert built.fingerprint != reference.fingerprint, name
        # only what the loaded bytes establish is invisible to the static check
        assert (built.static_fingerprint == reference.static_fingerprint) == (name in {"tokenizer", "model bytes"}), name


# ---------------------------------------------------------------------------
# 2. Setup, refresh and query share one representation
# ---------------------------------------------------------------------------


def test_setup_refresh_and_query_use_one_verified_representation(lab, tmp_path):
    cx = brain(tmp_path)
    result = cx.semantic_setup()
    meta, _ = stored(cx)
    base, reference = rep.read_provider(meta.provider)
    encoder = semantic.load_model_for_index(meta)
    actual = semantic.representation_of(encoder)
    assert result.provider == base == semantic.SEMANTIC_PROVIDER  # the public result is unchanged
    assert reference == rep.IndexReference(rep.CORE_VERIFIED, actual.static_fingerprint, actual.fingerprint)
    assert cx.semantic_state().status == SEMANTIC_READY
    pending = cx.remember("Rollback scripts must be rehearsed", kind="root_cause")
    assert cx.semantic_state().status == SEMANTIC_STALE
    refreshed = cx.preflight(QUERY)
    assert refreshed.retrieval.status == SEMANTIC_READY and refreshed.retrieval.refreshed == 1
    after, vectors = stored(cx)
    assert after.provider == meta.provider and pending.memory_id in dict(vectors["memory"])


def test_older_builds_refuse_the_new_index_by_their_own_comparison(lab, tmp_path):
    """0.4.0 and earlier development builds compare the provider column EXACTLY
    (the built-in path with "onnxruntime", the external path with
    `index_provider`); the tagged value never equals either."""
    cx = brain(tmp_path)
    cx.semantic_setup()
    meta, _ = stored(cx)
    legacy = dataclasses.replace(meta, provider=rep.base_provider(meta.provider))
    assert not meta.matches(provider=semantic.SEMANTIC_PROVIDER, model_id=meta.model_id,
                            normalization=semantic.SEMANTIC_NORMALIZATION)
    assert legacy.matches(provider=semantic.SEMANTIC_PROVIDER, model_id=meta.model_id,
                          normalization=semantic.SEMANTIC_NORMALIZATION)
    identity = _identity()
    other = Urdyn.init(tmp_path / "external", "dev")
    other.remember(FACT, kind="root_cause")
    other.configure_semantic_backend(_Backend(identity))
    other.semantic_setup()
    external, _ = stored(other)
    assert external.provider.startswith(identity.index_provider + rep.PROVIDER_TAG)
    assert not external.matches(provider=identity.index_provider, model_id=identity.model_id,
                                normalization=identity.normalization)


# ---------------------------------------------------------------------------
# 3. Legacy and incomplete identities
# ---------------------------------------------------------------------------


def test_a_pre_r03_7a_index_is_refused_not_relabelled_and_rebuilt_explicitly(lab, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()
    meta, _ = stored(cx)
    write_index(cx, provider=rep.base_provider(meta.provider))  # what builds before representation references write
    before = (index_digest(cx), canonical(cx))
    cx.remember("Rollback scripts must be rehearsed", kind="root_cause")  # stale: must not be topped up
    canonical_now = canonical(cx)
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_UNVERIFIED)
    cx.semantic_state()
    assert index_digest(cx) == before[0]  # never written to, not even its metadata
    assert canonical(cx) == canonical_now
    cx.semantic_setup()  # the documented remedy
    assert cx.semantic_state().status == SEMANTIC_READY
    assert rep.read_provider(stored(cx)[0].provider)[1].assurance == rep.CORE_VERIFIED


@pytest.mark.parametrize("tag", [
    "",
    "urdyn-representation-v9/core-verified/{h}/{h}",
    "urdyn-representation-v1/certified/{h}/{h}",
    "urdyn-representation-v1/core-verified/not-a-digest/{h}",
    "urdyn-representation-v1/core-verified/{h}",
    "urdyn-representation-v1/core-verified/{h}/{h}/extra",
])
def test_an_incomplete_or_unknown_reference_is_never_treated_as_verified(lab, tmp_path, tag):
    cx = brain(tmp_path)
    cx.semantic_setup()
    write_index(cx, provider=semantic.SEMANTIC_PROVIDER + rep.PROVIDER_TAG + tag.format(h="a" * 64))
    before = index_digest(cx)
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_UNVERIFIED)
    assert index_digest(cx) == before


# ---------------------------------------------------------------------------
# 4. Incompatible profiles: same model, something else differs
# ---------------------------------------------------------------------------


def test_same_model_different_tokenizer_bytes_is_refused_at_bind(lab, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()
    pending = cx.remember("Rollback scripts must be rehearsed", kind="root_cause")
    lab.write(semantic.TOKENIZER_FILENAME, b"synthetic tokenizer.json v2 (same model, other bytes)")
    before = (index_digest(cx), canonical(cx))
    # Honest limit: without loading, the state cannot see a byte change...
    assert cx.semantic_state().status == SEMANTIC_STALE
    # ...the bind does, before any vector is compared or written.
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH, state_detail=False)
    assert (index_digest(cx), canonical(cx)) == before
    assert pending.memory_id not in dict(stored(cx)[1]["memory"])  # not topped up with foreign vectors


def test_same_model_different_model_bytes_is_refused_at_bind(lab, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()
    artifact = semantic.artifact_for_index(stored(cx)[0])
    lab.write(artifact, b"synthetic model v2: re-exported, same name, same revision")
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH, state_detail=False)


@pytest.mark.parametrize("name, mutate", [
    ("padding", lambda d: d["encoder"]["recipe"]["padding"].update(strategy="fixed-128")),
    ("pooling", lambda d: d["encoder"]["recipe"].update(pooling="cls-token-v1")),
    ("normalization", lambda d: d["core"].update(normalization="l2-float32")),
    ("batch", lambda d: d["encoder"]["recipe"]["batch"].update(size=1, composition_independent=True)),
    ("runtime", lambda d: d["encoder"]["runtime"].update(version="1.23.2")),
    ("environment", lambda d: d["core"]["environment"].update(cpu_features=["AVX512F"])),
])
def test_an_index_built_under_another_profile_is_refused_before_loading(lab, tmp_path, name, mutate):
    cx = brain(tmp_path)
    cx.semantic_setup()
    write_index(cx, provider=foreign_provider(current_descriptor(cx), mutate))
    semantic._model_cache.clear()
    FakeSession.loaded.clear()
    before = (index_digest(cx), canonical(cx))
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)
    assert FakeSession.loaded == []  # refused from the static fingerprint: no model loaded to say no
    assert (index_digest(cx), canonical(cx)) == before


def test_a_changed_batch_policy_in_this_process_is_refused(lab, tmp_path, monkeypatch):
    cx = brain(tmp_path)
    cx.semantic_setup()
    monkeypatch.setattr(semantic, "SEMANTIC_ENCODE_BATCH_SIZE", 16)
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)


def test_another_runtime_or_environment_is_not_presumed_equivalent(lab, tmp_path, monkeypatch):
    cx = brain(tmp_path)
    cx.semantic_setup()
    version, build, tok = semantic._observed_onnx_runtime()
    monkeypatch.setattr(semantic, "_observed_onnx_runtime", lambda: ("1.23.2", build, tok))
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)
    monkeypatch.setattr(semantic, "_observed_onnx_runtime", lambda: (version, build, tok))
    assert cx.semantic_state().status == SEMANTIC_READY
    numpy_version, machine, system, features = rep._observed_environment()
    monkeypatch.setattr(rep, "_observed_environment", lambda: (numpy_version, machine, system, features + ("AMX",)))
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)


# ---------------------------------------------------------------------------
# 5. External backends: declared, never verified
# ---------------------------------------------------------------------------


def _identity(**changes):
    values = dict(provider="test-encoder-v1", model_id=semantic.model_identity_for(semantic.ARTIFACT_ARM64),
                  model_revision=semantic.SEMANTIC_MODEL_REVISION, dimensions=D,
                  artifact_sha256=semantic.ARTIFACT_ARM64_SHA256, tokenizer_sha256=semantic.TOKENIZER_SHA256,
                  encoding_policy="deterministic-v1")
    values.update(changes)
    return SemanticBackendIdentity(**values)


class _Backend:
    def __init__(self, identity, *, offset=0.0, hold=None):
        self.identity, self.offset, self.hold, self.entered = identity, offset, hold, threading.Event()
        # self-attributed certificate: must be ignored by the core
        self.representation = "core-verified"

    def available(self):
        return True

    def encode(self, texts):
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(30)
        return [[1.0 + self.offset + (len(t) % 5)] + [0.5] * (D - 1) for t in texts]


def test_an_external_backend_is_declared_and_its_update_is_refused(lab, tmp_path, monkeypatch):
    cx = brain(tmp_path)
    cx.configure_semantic_backend(_Backend(_identity()))
    cx.semantic_setup()
    reference = rep.read_provider(stored(cx)[0].provider)[1]
    assert reference.assurance == rep.BACKEND_DECLARED  # its own attribute claimed more; ignored
    assert reference.fingerprint == semantic.declared_representation(_identity()).fingerprint
    assert cx.semantic_state().status == SEMANTIC_READY
    before = (index_digest(cx), canonical(cx))
    cx.configure_semantic_backend(_Backend(_identity(encoding_policy="deterministic-v2")))  # backend update
    assert_explicit_refusal(cx, DETAIL_MODEL_MISMATCH)
    cx.configure_semantic_backend(_Backend(_identity()))
    numpy_version, machine, system, features = rep._observed_environment()
    monkeypatch.setattr(rep, "_observed_environment", lambda: (numpy_version, "aarch64", system, features))
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH)
    assert (index_digest(cx), canonical(cx)) == before


def test_limit_an_external_change_under_the_same_declaration_is_not_detectable(lab, tmp_path):
    """Documented limit, asserted so it stays visible: the core cannot see
    inside a backend; a new implementation that keeps every declared field
    is accepted. Only an implementation the core verifies could close this."""
    cx = brain(tmp_path)
    cx.configure_semantic_backend(_Backend(_identity()))
    cx.semantic_setup()
    cx.configure_semantic_backend(_Backend(_identity(), offset=3.0))  # different vectors, same declaration
    assert cx.preflight(QUERY).retrieval.status == SEMANTIC_READY


def test_declared_and_verified_are_never_the_same_representation(lab, tmp_path):
    """Same model id, same tokenizer digest: an external runtime is still not
    the built-in one, in either direction."""
    verified = brain(tmp_path)
    verified.semantic_setup()
    external = Urdyn.open(verified.path)
    external.configure_semantic_backend(_Backend(_identity()))
    assert external.semantic_state().detail == DETAIL_MODEL_MISMATCH
    declared = Urdyn.init(tmp_path / "declared", "dev")
    declared.remember(FACT, kind="root_cause")
    declared.configure_semantic_backend(_Backend(_identity()))
    declared.semantic_setup()
    assert Urdyn.open(declared.path).semantic_state().detail == DETAIL_MODEL_MISMATCH
    meta, _ = stored(declared)
    assert semantic.representation_problem(meta) == DETAIL_REPRESENTATION_MISMATCH


def test_unattested_encoders_never_match_a_verified_index_or_the_reverse(lab, tmp_path, monkeypatch):
    class Fake:  # any object the core did not build, whatever it claims
        identity = semantic.model_identity_for(semantic.ARTIFACT_X86_64)
        representation = "core-verified"

        def encode(self, texts):
            return np.ones((len(texts), D), np.float32)

    assert semantic.representation_of(Fake()).assurance == rep.UNATTESTED
    cx = brain(tmp_path)
    cx.semantic_setup()  # verified
    monkeypatch.setattr(semantic, "load_model_for_retrieval", lambda artifact=None: Fake())
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH, state_detail=False)
    monkeypatch.setattr(semantic, "load_model_for_setup", lambda: Fake())
    cx.semantic_setup()  # unattested
    assert rep.read_provider(stored(cx)[0].provider)[1].assurance == rep.UNATTESTED
    monkeypatch.undo()
    monkeypatch.setattr(semantic, "_artifact_path", lambda name, *, local_files_only: str(lab.assets / name))
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    semantic._model_cache.clear()
    # honest limit: the state cannot tell an unattested index from a usable one
    # without an encoder; the bind refuses it before any comparison
    assert_explicit_refusal(cx, DETAIL_REPRESENTATION_MISMATCH, state_detail=False)


# ---------------------------------------------------------------------------
# 6. Partial updates, restore, concurrent publications
# ---------------------------------------------------------------------------


def test_a_refresh_never_writes_into_an_index_replaced_after_it_read_it(lab, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()
    meta, _ = stored(cx)
    with SemanticIndexStore.create_or_open(cx._semantic_db_path) as reader:
        reader.expect(reader.meta())
        # another writer, outside the cooperating lock, rebuilds meanwhile
        write_index(cx, provider=rep.base_provider(meta.provider), rows={"memory": []})
        assert reader.add_vectors("memory", [("m-new", b"\0" * (4 * D))]) is False
    assert stored(cx)[1]["memory"] == []  # nothing of the old representation was added


def test_a_partial_refresh_with_another_representation_adds_nothing(lab, tmp_path):
    cx = brain(tmp_path)
    cx.semantic_setup()
    first = cx.remember("Rollback scripts must be rehearsed", kind="root_cause")
    rows_before = stored(cx)[1]
    lab.write(semantic.TOKENIZER_FILENAME, b"synthetic tokenizer.json v2")
    cx.preflight(QUERY)
    cx.remember("Feature flags need an owner", kind="root_cause")
    cx.preflight(QUERY)
    assert stored(cx)[1] == rows_before  # neither pending record got a foreign vector
    assert first.memory_id not in dict(rows_before["memory"])


def test_restore_leaves_no_index_and_a_copied_foreign_index_is_refused(lab, tmp_path, monkeypatch):
    cx = brain(tmp_path)
    cx.semantic_setup()
    archive = tmp_path / "backup.zip"
    backup.create_backup(cx.path, archive)
    elsewhere = tmp_path / "elsewhere.db"
    numpy_version, machine, system, features = rep._observed_environment()
    monkeypatch.setattr(rep, "_observed_environment", lambda: (numpy_version, machine, "Windows", features))
    semantic._model_cache.clear()  # an encoder's identity is fixed when it is built: a new "process"
    cx.semantic_setup()  # "another machine" builds its own index
    shutil.copy(cx._semantic_db_path, elsewhere)
    monkeypatch.undo()
    monkeypatch.setattr(semantic, "_artifact_path", lambda name, *, local_files_only: str(lab.assets / name))
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    cx.remember("written after the backup", kind="note")
    backup.restore_backup(cx.path, archive)
    restored = Urdyn.open(cx.path)
    assert (restored.semantic_state().status, restored.semantic_state().detail) == (SEMANTIC_DISABLED,
                                                                                    DETAIL_NOT_SET_UP)
    assert [m.content for m in restored.timeline()] == [FACT]
    shutil.copy(elsewhere, restored._semantic_db_path)  # the user copies the other machine's index in
    assert_explicit_refusal(restored, DETAIL_REPRESENTATION_MISMATCH)
    assert [m.content for m in restored.timeline()] == [FACT]


def test_concurrent_rebuilds_with_different_representations_publish_one(lab, tmp_path):
    cx = brain(tmp_path)
    FakeSession.hold, FakeSession.hold_from = threading.Event(), 2  # block the encode, after the marker
    box = {}

    def verified_setup():
        try:
            cx.semantic_setup()
            box["verified"] = "published"
        except UrdynSemanticUnavailableError as exc:
            box["verified"] = str(exc)

    thread = threading.Thread(target=verified_setup)
    thread.start()
    try:
        assert FakeSession.entered.wait(10)  # verified rebuild is encoding, its marker written
        other = Urdyn.open(cx.path)
        other.configure_semantic_backend(_Backend(_identity()))
        other.semantic_setup()  # a declared rebuild starts and publishes meanwhile
    finally:
        FakeSession.hold.set()
        thread.join(30)
    assert "not published" in box["verified"]
    meta, vectors = stored(cx)
    assert rep.read_provider(meta.provider)[1].fingerprint == semantic.declared_representation(_identity()).fingerprint
    declared = Urdyn.open(cx.path)
    declared.configure_semantic_backend(_Backend(_identity()))
    assert declared.semantic_state().status == SEMANTIC_READY
    assert len(vectors["memory"]) == 1
