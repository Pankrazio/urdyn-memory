"""Cooperative cancellation of ONNX `Run` inside `_OnnxTextEncoder`.

The REAL `_OnnxTextEncoder` / `_build_encoder` code runs here, over a fake
session and tokenizer: these tests prove the Python protocol (request
states, leases, retirement, synchronized cache), NOT ONNX Runtime behaviour
-- which needs the real artifact.
Interleavings are built with a manual timer and a controlled monotonic
clock, never ordered by sleeps. The non-cooperative Run lives in a
subprocess owned by the test.
"""
from __future__ import annotations

import contextvars
import hashlib
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import onnxruntime
import pytest

import urdyn
import urdyn._semantic as semantic
from urdyn import Urdyn, backup
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    DETAIL_QUERY_FAILED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_UNAVAILABLE,
)

SRC = str(Path(urdyn.__file__).resolve().parents[1])
D = 384
ART = semantic.ARTIFACT_PORTABLE


# ---------------------------------------------------------------------------
# Fakes: session + tokenizer under the production encoder
# ---------------------------------------------------------------------------


class _Encoding:
    def __init__(self, ids, mask):
        self.ids, self.attention_mask = ids, mask


class FakeTokenizer:
    def encode_batch(self, texts):
        width = max(len(t.split()) for t in texts) + 1
        out = []
        for t in texts:
            ids = [1 + (int(hashlib.sha256(w.encode()).hexdigest(), 16) % 997) for w in t.split()] + [2]
            mask = [1] * len(ids)
            out.append(_Encoding(ids + [0] * (width - len(ids)), mask + [0] * (width - len(mask))))
        return out


class _Node:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """`run` computes a deterministic hidden state. `hold` (an Event) makes
    it wait inside Run; `cooperative` decides whether it honours
    `run_options.terminate` while waiting."""

    def __init__(self, cooperative=True):
        self.cooperative = cooperative
        self.hold: threading.Event | None = None
        self.entered = threading.Event()
        self.on_return = None  # runs just before Run returns (clock tricks)
        self.runs = 0

    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask")]

    def run(self, output_names, feed, run_options=None):
        self.runs += 1
        self.entered.set()
        if self.hold is not None:
            while not self.hold.wait(0.005):
                if self.cooperative and run_options is not None and run_options.terminate:
                    raise RuntimeError("[ONNXRuntimeError] : 1 : FAIL : Exiting due to terminate flag being set to true.")
        ids = feed["input_ids"].astype(np.float32)
        hidden = np.zeros((*ids.shape, D), np.float32)
        hidden[..., :] = (ids[..., None] % 13 + 1) * (np.arange(D, dtype=np.float32) % 7 + 1) / 50.0
        if self.on_return is not None:
            self.on_return()
        return [hidden]


def encoder(cooperative=True):
    return semantic._OnnxTextEncoder(FakeSession(cooperative), FakeTokenizer(), ART)


class ManualTimer:
    """Replaces the per-encode timer: the TEST decides when it fires."""

    made: list["ManualTimer"] = []

    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        self.started = self.cancelled = self.joined = False
        ManualTimer.made.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def join(self):
        self.joined = True

    def fire(self):
        self.callback()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def lab(monkeypatch):
    """Fresh cache, manual timer, controlled clock; afterwards: no lease,
    no interdiction, no construction, no live timer thread left."""
    monkeypatch.setattr(semantic, "_model_cache", {})
    monkeypatch.setattr(semantic, "_retired_in_use", {})
    monkeypatch.setattr(semantic, "_building", {})
    ManualTimer.made = []
    monkeypatch.setattr(semantic, "_deadline_timer", ManualTimer)
    clock = Clock()
    monkeypatch.setattr(semantic, "_monotonic", clock)
    yield clock
    assert semantic._building == {}
    for timer in ManualTimer.made:
        assert timer.cancelled and timer.joined  # disarmed and concluded in `finally`


def run_in_thread(fn, context=None):
    box = {}

    def target():
        try:
            box["value"] = (context.run(fn) if context else fn())
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def leases(enc):
    return enc._lease_state().active


TEXTS = ["schema migration failed", "cache warmup step"]


# ---------------------------------------------------------------------------
# Default path and A-D: one request
# ---------------------------------------------------------------------------


def test_default_path_has_no_runoptions_no_timer_and_the_same_output(lab):
    enc = encoder()
    plain = enc.encode(TEXTS)
    assert ManualTimer.made == [] and semantic._ACTIVE_REQUEST.get() is None
    with semantic._experimental_encode_deadline(30):
        guarded = enc.encode(TEXTS)
    assert np.array_equal(plain, guarded)  # 1. a healthy request: identical output
    assert leases(enc) == 0 and not enc.retired


def test_A_success_before_the_deadline_is_linearized_and_the_timer_disarmed(lab):
    enc = encoder()
    with semantic._experimental_encode_deadline(5):
        out = enc.encode(TEXTS)
    timer = ManualTimer.made[0]
    assert out.shape == (2, D) and timer.started and timer.cancelled and timer.joined
    timer.fire()  # a callback arriving after success changes nothing
    assert not enc.retired and leases(enc) == 0


def test_B_expiry_during_run_cancels_it_retires_the_session_and_returns_no_result(lab):
    enc = encoder(cooperative=True)
    enc._session.hold = threading.Event()
    ctx = contextvars.copy_context()
    ctx.run(lambda: semantic._ENCODE_DEADLINE.set(5.0))
    thread, box = run_in_thread(lambda: enc.encode(TEXTS), ctx)
    assert enc._session.entered.wait(5)
    lab.now += 6
    ManualTimer.made[0].fire()  # deadline while Run is active
    assert enc.retired  # retired by the callback, while the Run still runs
    thread.join(5)
    assert isinstance(box["error"], semantic.EncoderRunExpired) and box["error"].terminate_requested
    assert box["error"].run_aborted  # the (fake) Run honoured the flag and raised
    assert "value" not in box and leases(enc) == 0
    assert semantic._retired_in_use == {}  # drained


def test_C_a_run_that_returns_just_after_the_deadline_is_discarded_without_retiring(lab):
    enc = encoder()
    enc._session.on_return = lambda: setattr(lab, "now", lab.now + 10)  # Run ends past the deadline
    with semantic._experimental_encode_deadline(5), pytest.raises(semantic.EncoderRunExpired) as raised:
        enc.encode(TEXTS)
    assert raised.value.terminate_requested is False  # no terminate ever reached a Run
    assert not enc.retired  # no native cancellation: nothing to rebuild
    assert enc.encode(TEXTS).shape == (2, D)


def test_C_expiry_before_any_run_does_not_retire(lab):
    enc = encoder()
    big = [f"text {i}" for i in range(semantic.SEMANTIC_ENCODE_BATCH_SIZE + 3)]  # two chunks
    calls = {"n": 0}
    original = enc._session.on_return

    def advance():
        calls["n"] += 1
        lab.now += 10  # after the first chunk the budget is gone

    enc._session.on_return = advance
    with semantic._experimental_encode_deadline(5), pytest.raises(semantic.EncoderRunExpired) as raised:
        enc.encode(big)
    assert calls["n"] == 1  # the second chunk's Run never started (one budget for the whole encode)
    assert raised.value.terminate_requested is False and not enc.retired
    enc._session.on_return = original


def test_D_callback_racing_completion_one_terminal_state(lab):
    # (i) success first: the late callback does nothing
    enc = encoder()
    with semantic._experimental_encode_deadline(5):
        enc.encode(TEXTS)
    ManualTimer.made[-1].fire()
    assert not enc.retired
    # (ii) callback first, during Run, but the Run still returns rows: discarded, retired
    enc2 = encoder(cooperative=False)
    enc2._session.on_return = lambda: ManualTimer.made[-1].fire()
    with semantic._experimental_encode_deadline(5), pytest.raises(semantic.EncoderRunExpired) as raised:
        enc2.encode(TEXTS)
    assert raised.value.terminate_requested is True and enc2.retired and leases(enc2) == 0
    assert raised.value.run_aborted is False  # requested, NOT terminated: the Run returned rows


def test_an_onnx_failure_is_not_reported_as_a_timeout(lab):
    enc = encoder()

    def boom(*a, **k):
        raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad input")

    enc._session.run = boom
    with semantic._experimental_encode_deadline(5), pytest.raises(RuntimeError) as raised:
        enc.encode(TEXTS)
    assert not isinstance(raised.value, semantic.SemanticUnavailable)  # ONNX failure, distinct
    assert not enc.retired and leases(enc) == 0


def test_a_numeric_contract_failure_stays_distinct_and_in_the_core(lab):
    enc = encoder()
    enc._session.on_return = None
    with semantic._experimental_encode_deadline(5):
        raw = enc.encode(TEXTS)
    raw[0, :] = np.nan
    with pytest.raises(semantic.InvalidSemanticVectors) as raised:
        semantic.normalize_rows(raw, rows=2, dimensions=D)
    assert not isinstance(raised.value, semantic.EncoderRunExpired)


def test_the_deadline_does_not_follow_into_a_new_thread_unless_the_context_is_copied(lab):
    enc = encoder()
    with semantic._experimental_encode_deadline(5):
        thread, box = run_in_thread(lambda: enc.encode(TEXTS))
        thread.join(5)
        assert ManualTimer.made == []  # a new thread has its own (empty) context
        thread, box = run_in_thread(lambda: enc.encode(TEXTS), contextvars.copy_context())
        thread.join(5)
    assert len(ManualTimer.made) == 1 and box["value"].shape == (2, D)


def test_the_budget_must_be_positive():
    with pytest.raises(ValueError):
        with semantic._experimental_encode_deadline(0):
            pass


# ---------------------------------------------------------------------------
# E-G: shared encoder, leases, retirement, cache
# ---------------------------------------------------------------------------


def _victim_and_bystander(lab, victim_finishes_first):
    enc = semantic._build_encoder(ART, local_files_only=True)
    enc._session.hold = threading.Event()
    victim_ctx = contextvars.copy_context()
    victim_ctx.run(lambda: semantic._ENCODE_DEADLINE.set(5.0))
    victim, vbox = run_in_thread(lambda: enc.encode(["victim text"]), victim_ctx)
    assert enc._session.entered.wait(5)
    bystander, bbox = run_in_thread(lambda: enc.encode(["bystander text"]))  # no deadline: another Brain
    while leases(enc) < 2:
        pass
    lab.now += 6
    ManualTimer.made[0].fire()  # only the victim's RunOptions is terminated
    assert enc.retired and ART in semantic._retired_in_use and ART not in semantic._model_cache
    with pytest.raises(semantic.EncoderUnavailable):  # no successor while the old one runs
        semantic._build_encoder(ART, local_files_only=True)
    with pytest.raises(semantic.EncoderRetired):  # the old handle accepts no new encode
        enc.encode(["late caller with an old handle"])
    if victim_finishes_first:
        victim.join(5)
        assert ART in semantic._retired_in_use  # bystander still running
        enc._session.hold.set()
        bystander.join(5)
    else:
        enc._session.hold.set()
        bystander.join(5)
        victim.join(5)
    return enc, vbox, bbox


@pytest.mark.parametrize("victim_first", [True, False])
def test_E_F_G_one_cancelled_one_healthy_on_a_shared_encoder(lab, monkeypatch, victim_first):
    built = []
    monkeypatch.setattr(semantic, "_construct_encoder",
                        lambda artifact, local_files_only: built.append(1) or encoder())
    enc, vbox, bbox = _victim_and_bystander(lab, victim_first)
    assert isinstance(vbox["error"], semantic.EncoderRunExpired) and vbox["error"].terminate_requested
    reference = encoder().encode(["bystander text"])
    assert np.array_equal(bbox["value"], reference)  # 2. the healthy request is untouched
    assert leases(enc) == 0 and semantic._retired_in_use == {}  # drained: no leaked lease
    successor = semantic._build_encoder(ART, local_files_only=True)  # rebuilt only after draining
    assert successor is not enc and not successor.retired and len(built) == 2
    enc.retire()  # a late retire of the OLD encoder cannot evict the successor
    assert semantic._model_cache[ART] is successor


def test_repeated_cancellations_do_not_grow_live_sessions(lab, monkeypatch):
    built = []
    monkeypatch.setattr(semantic, "_construct_encoder",
                        lambda artifact, local_files_only: built.append(encoder()) or built[-1])
    for round_ in range(5):
        enc = semantic._build_encoder(ART, local_files_only=True)
        enc._session.hold = threading.Event()
        ctx = contextvars.copy_context()
        ctx.run(lambda: semantic._ENCODE_DEADLINE.set(5.0))
        thread, box = run_in_thread(lambda: enc.encode(TEXTS), ctx)
        assert enc._session.entered.wait(5)
        ManualTimer.made[-1].fire()
        thread.join(5)
        assert box["error"].terminate_requested and leases(enc) == 0
    assert len(built) == 5 and sum(not e.retired for e in built) == 0
    assert semantic._retired_in_use == {} and ART not in semantic._model_cache


def test_one_construction_at_a_time_and_bounded_waiting(lab, monkeypatch):
    gate, started, built = threading.Event(), threading.Event(), []

    def slow_construct(artifact, local_files_only):
        started.set()
        assert gate.wait(5)
        built.append(1)
        return encoder()

    monkeypatch.setattr(semantic, "_construct_encoder", slow_construct)
    monkeypatch.setattr(semantic, "ENCODER_BUILD_WAIT_SECONDS", 0.2)
    first, fbox = run_in_thread(lambda: semantic._build_encoder(ART, local_files_only=True))
    assert started.wait(5)
    with pytest.raises(semantic.EncoderUnavailable):  # bounded, not indefinite
        semantic._build_encoder(ART, local_files_only=True)
    waiter, wbox = run_in_thread(lambda: semantic._build_encoder(ART, local_files_only=True))
    monkeypatch.setattr(semantic, "ENCODER_BUILD_WAIT_SECONDS", 5)
    gate.set()
    first.join(5)
    waiter.join(5)
    assert len(built) == 1  # a single construction
    assert wbox.get("value") is fbox["value"] or isinstance(wbox.get("error"), semantic.EncoderUnavailable)


def test_cache_lock_is_not_held_while_constructing(lab, monkeypatch):
    seen = {}

    def construct(artifact, local_files_only):
        seen["free"] = semantic._model_cache_lock.acquire(blocking=False)
        if seen["free"]:
            semantic._model_cache_lock.release()
        return encoder()

    monkeypatch.setattr(semantic, "_construct_encoder", construct)
    semantic._build_encoder(ART, local_files_only=True)
    assert seen["free"]


# ---------------------------------------------------------------------------
# H: a Run that ignores terminate (subprocess owned by the test)
# ---------------------------------------------------------------------------

H_HARNESS = textwrap.dedent('''
    import contextvars, json, os, sys, threading, time
    sys.path.insert(0, sys.argv[1])
    import test_cooperative_onnx as t
    import urdyn._semantic as semantic
    import urdyn._workspace as workspace
    from urdyn import Urdyn

    workspace.INFERENCE_SLOT_TIMEOUT_SECONDS = 0.5  # inference slot: the stuck call holds it
    built = []
    def construct(artifact, local_files_only):
        enc = t.encoder(cooperative=False)
        enc._session.hold = threading.Event()  # never set: the Run never returns
        built.append(enc)
        return enc
    semantic._construct_encoder = construct
    semantic.default_runtime_available = lambda: True
    semantic.artifacts_available = lambda meta: True
    semantic.resolve_local_revision = lambda *a, **k: semantic.SEMANTIC_MODEL_REVISION
    semantic.preferred_artifact = lambda machine=None: semantic.ARTIFACT_PORTABLE

    root = sys.argv[2]
    cx = Urdyn.create(root, "dev")
    cx.remember("Migrations require a dry run", kind="root_cause")
    first = construct(semantic.ARTIFACT_PORTABLE, True)       # set up with a working session...
    first._session.hold = None
    semantic._model_cache[semantic.ARTIFACT_PORTABLE] = first
    cx.semantic_setup()
    first._session.hold = threading.Event()                    # ...then every Run blocks, ignoring terminate
    ctx = contextvars.copy_context()
    ctx.run(lambda: semantic._ENCODE_DEADLINE.set(0.2))
    stuck = threading.Thread(target=lambda: ctx.run(cx.preflight, "migrations dry run"), daemon=True)
    stuck.start()
    while not first.retired:
        time.sleep(0.01)
    out = {"retired_by_the_timer": first.retired, "caller_still_blocked": stuck.is_alive()}
    second = Urdyn.open(root).preflight("migrations dry run")   # another call meanwhile
    third = Urdyn.open(root).preflight("migrations dry run")
    Urdyn.open(root).remember("canonical still writable", kind="note")
    try:
        semantic._build_encoder(semantic.ARTIFACT_PORTABLE, local_files_only=True)
        out["direct_build"] = "built"
    except semantic.EncoderUnavailable:
        out["direct_build"] = "refused"
    
    out.update(second=[second.retrieval.status, second.retrieval.detail],
               third_root_causes=[m.content for m in third.root_causes],
               constructions=len(built), live_interdiction=semantic.ARTIFACT_PORTABLE in semantic._retired_in_use,
               caller_still_blocked_at_end=stuck.is_alive(), memories=len(Urdyn.open(root).timeline()))
    print(json.dumps(out), flush=True)
    os._exit(0)  # the stuck thread cannot be stopped: the harness process ends instead
''')


def test_H_a_run_ignoring_terminate_blocks_its_caller_but_no_session_proliferates(tmp_path):
    harness = tmp_path / "h.py"
    harness.write_text(H_HARNESS)
    env = {"PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", "/tmp"), "GIT_CONFIG_GLOBAL": "/dev/null"}
    process = subprocess.Popen([sys.executable, str(harness), str(Path(__file__).parent), str(tmp_path / "brain")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()  # our own child
        out, err = process.communicate()
        pytest.fail(f"harness did not finish: {err[-1500:]}")
    import json

    assert process.returncode == 0, err[-2000:]
    result = json.loads(out.strip().splitlines()[-1])
    assert result["retired_by_the_timer"] and result["caller_still_blocked"]  # NOT a containment
    assert result["caller_still_blocked_at_end"]
    assert result["second"] == [SEMANTIC_UNAVAILABLE, DETAIL_QUERY_FAILED]  # degraded, declared
    assert result["third_root_causes"] == ["Migrations require a dry run"]  # lexical answer intact
    assert result["constructions"] == 1 and result["live_interdiction"]  # no replacement session
    assert result["direct_build"] == "refused"
    assert result["memories"] == 2  # canonical available and written


# ---------------------------------------------------------------------------
# I: under the real lifecycle (inference and setup outside the lock, unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture
def onnx_path_brain(lab, monkeypatch, tmp_path):
    """A Brain on the built-in ONNX path, whose session is the fake one."""
    built = []

    def construct(artifact, local_files_only):
        built.append(encoder())
        return built[-1]

    monkeypatch.setattr(semantic, "_construct_encoder", construct)
    monkeypatch.setattr(semantic, "default_runtime_available", lambda: True)
    monkeypatch.setattr(semantic, "artifacts_available", lambda meta: True)
    monkeypatch.setattr(semantic, "resolve_local_revision", lambda *a, **k: semantic.SEMANTIC_MODEL_REVISION)
    monkeypatch.setattr(semantic, "preferred_artifact", lambda machine=None: ART)
    root = tmp_path / "brain"
    cx = Urdyn.create(root, "dev")
    cx.remember("Migrations require a dry run", kind="root_cause")
    cx.semantic_setup()
    return root, cx, built


def digest(root):
    return hashlib.sha256((root / ".urdyn" / "memory.db").read_bytes()).hexdigest()


def test_I_a_cancelled_query_is_declared_and_the_brain_is_intact(onnx_path_brain, lab):
    root, cx, built = onnx_path_brain
    before = digest(root)
    enc = built[-1]
    enc._session.entered.clear()  # setup's Runs already set it
    enc._session.hold = threading.Event()
    ctx = contextvars.copy_context()
    ctx.run(lambda: semantic._ENCODE_DEADLINE.set(5.0))
    thread, box = run_in_thread(lambda: cx.preflight("migrations dry run"), ctx)
    assert enc._session.entered.wait(5)
    lab.now += 6
    ManualTimer.made[-1].fire()
    thread.join(10)
    result = box["value"]
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_QUERY_FAILED
    assert [m.content for m in result.root_causes] == ["Migrations require a dry run"]
    assert digest(root) == before and enc.retired and leases(enc) == 0
    assert cx.preflight("migrations dry run").retrieval.status == SEMANTIC_READY  # rebuilt after draining
    assert len(built) == 2


def test_I_restore_before_publication_publishes_nothing(onnx_path_brain, lab):
    root, cx, built = onnx_path_brain
    archive = root.parent / "backup.zip"
    backup.create_backup(root, archive)
    enc = built[-1]
    enc._session.entered.clear()  # setup's Runs already set it
    enc._session.hold = threading.Event()
    ctx = contextvars.copy_context()
    ctx.run(lambda: semantic._ENCODE_DEADLINE.set(30.0))
    thread, box = run_in_thread(lambda: cx.preflight("migrations dry run"), ctx)
    assert enc._session.entered.wait(5)
    backup.restore_backup(root, archive)
    enc._session.hold.set()  # a healthy result, in time -- for a Brain that was replaced
    thread.join(10)
    assert box["value"].retrieval.status == SEMANTIC_DISABLED and not cx._semantic_db_path.exists()
    assert not enc.retired


def test_I_setup_expiring_during_a_pool_keeps_the_marker_and_publishes_nothing(onnx_path_brain, lab):
    root, cx, built = onnx_path_brain
    enc = built[-1]
    ctx = contextvars.copy_context()
    ctx.run(lambda: semantic._ENCODE_DEADLINE.set(5.0))
    probe_done = {"n": 0}

    def hold_pools():
        probe_done["n"] += 1

    enc._session.on_return = hold_pools
    enc._session.hold = None

    def pool_hold(*a, **k):
        pass

    original_run = enc._session.run

    def run(output_names, feed, run_options=None):
        if probe_done["n"] >= 1:  # the dimension probe went through; now a pool
            enc._session.hold = threading.Event()
        return original_run(output_names, feed, run_options)

    enc._session.run = run
    thread, box = run_in_thread(cx.semantic_setup, ctx)
    while not enc._session.entered.is_set() or enc._session.hold is None:
        pass
    while leases(enc) == 0:
        pass
    lab.now += 6
    ManualTimer.made[-1].fire()
    thread.join(10)
    assert isinstance(box.get("error"), semantic.EncoderRunExpired)  # ONNX setup errors propagate as before
    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE  # 6B marker
    assert enc.retired and leases(enc) == 0
