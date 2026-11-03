"""Semantic inference outside the lifecycle lock.

Synthetic Brains only, in `tmp_path`. The encoder is deterministic (axis 0
for "schema"/"migrations", axis 1 otherwise, or a configured axis) and is
synchronized with the test through explicit events -- a gate the test opens
-- rather than through sleeps, except where a slow encoder IS the scenario.
It records each request and whether the calling thread held the lifecycle
lock at that moment.

What these tests falsify:
- an encoder that is held, slow or stuck blocks other operations on the Brain
  (writes, reads, restore, rebuild; other threads, other processes);
- a result computed for a state that no longer exists is published
  (other Brain, restored Brain, other encoder, changed text);
- a failed inference is repeated, or turned into a healthy answer;
- inference runs while the lifecycle lock is held.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import urdyn
import urdyn._lifecycle as lifecycle_module
import urdyn._workspace as workspace_module
from urdyn import SemanticBackendIdentity, Urdyn, UrdynStorageError
from urdyn import backup
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._semantic import ARTIFACT_ARM64, ARTIFACT_ARM64_SHA256, TOKENIZER_SHA256, model_identity_for
from urdyn._semantic_store import (
    DETAIL_MODEL_MISMATCH,
    DETAIL_MODEL_UNCACHED,
    DETAIL_QUERY_FAILED,
    DETAIL_REFRESH_FAILED,
    SEMANTIC_DISABLED,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    SemanticIndexStore,
)

SRC = str(Path(urdyn.__file__).resolve().parents[1])
GATE_SECONDS = 30  # upper bound only: every gate is opened explicitly by the test
PROMPT = 5.0  # an operation that is not blocked by the encoder finishes well within this


class GatedEncoder:
    def __init__(self, provider="test-encoder-v1", axis=None):
        self.identity = SemanticBackendIdentity(
            provider=provider,
            model_id=model_identity_for(ARTIFACT_ARM64),
            model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            dimensions=384,
            artifact_sha256=ARTIFACT_ARM64_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            encoding_policy="deterministic-v1",
        )
        self.axis = axis
        self.is_available = True
        self.requests: list[tuple[str, ...]] = []
        self.under_lock: list[bool] = []
        self.hold_when = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.delay = 0.0
        self.fail_when = None

    def available(self):
        return self.is_available

    def encode(self, texts):
        texts = tuple(texts)
        self.requests.append(texts)
        self.under_lock.append(bool(getattr(lifecycle_module._local, "held", set())))
        if self.hold_when is not None and self.hold_when(texts):
            self.entered.set()
            assert self.release.wait(GATE_SECONDS), "test never opened the encoder gate"
        if self.delay:
            time.sleep(self.delay)
        if self.fail_when is not None and self.fail_when(texts):
            raise RuntimeError("encoder failed")
        rows = []
        for text in texts:
            vector = [0.0] * 384
            if self.axis is not None:
                vector[self.axis] = 1.0
            else:
                vector[0 if "migrations" in text.lower() or "schema" in text.lower() else 1] = 1.0
            rows.append(vector)
        return rows


class Background:
    def __init__(self, fn):
        self.value = self.error = None
        self.started = time.monotonic()
        self.elapsed = None
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.thread.start()

    def _run(self, fn):
        try:
            self.value = fn()
        except BaseException as exc:  # noqa: BLE001 -- reported to the test
            self.error = exc
        finally:
            self.elapsed = time.monotonic() - self.started

    def join(self, timeout):
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def result(self):
        assert self.join(GATE_SECONDS), "background call did not finish"
        if self.error is not None:
            raise self.error
        return self.value


def within(seconds, fn):
    """Run `fn` in a thread; (finished, elapsed, value). Never hangs the suite."""
    task = Background(fn)
    finished = task.join(seconds)
    if finished and task.error is not None:
        raise task.error
    return finished, task.elapsed, task.value


def brain(tmp_path, *root_causes, name="brain"):
    root = tmp_path / name
    cx = Urdyn.create(root, "dev")
    backend = GatedEncoder()
    cx.configure_semantic_backend(backend)
    for content in root_causes:
        cx.remember(content, kind="root_cause")
    cx.semantic_setup()
    backend.requests.clear()
    backend.under_lock.clear()
    return root, cx, backend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(index_path, entity_type=ENTITY_MEMORY):
    with SemanticIndexStore.open_if_exists(Path(index_path)) as store:
        return store.indexed_ids(entity_type)


def stored_axes(index_path):
    with sqlite3.connect(Path(index_path).resolve().as_uri() + "?mode=ro", uri=True) as con:
        return {int(np.argmax(np.frombuffer(blob, np.float32))) for (blob,) in con.execute(
            "SELECT vector FROM semantic_vectors")}


def lexical_only(cx, task):
    aside = cx._semantic_db_path.with_suffix(".aside")
    os.replace(cx._semantic_db_path, aside)
    try:
        return cx.preflight(task)
    finally:
        os.replace(aside, cx._semantic_db_path)


def shape(result):
    return ([m.memory_id for m in result.root_causes], [m.memory_id for m in result.verified_lessons],
            [a.attempt_id for a in result.known_failures])


def subprocess_env():
    return {"PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": os.environ.get("HOME", "/tmp")}


RETRIEVAL_CALLS = {
    "preflight": lambda cx: cx.preflight("schema change"),
    "context": lambda cx: cx.context("schema change"),
    "guard": lambda cx: cx.guard("schema change"),
}


# ---------------------------------------------------------------------------
# 1. Availability: a held encoder blocks nothing but its own call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", sorted(RETRIEVAL_CALLS))
def test_writes_and_reads_proceed_while_the_encoder_is_held(tmp_path, call):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.hold_when = lambda texts: texts == ("schema change",)
    running = Background(lambda: RETRIEVAL_CALLS[call](cx))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        other = Urdyn.open(root)
        finished, elapsed, found = within(PROMPT, lambda: (
            other.remember("written while the encoder is busy", kind="note"),
            other.recall("encoder is busy"),
            other.timeline(),
        )[1:])
        assert finished, "a canonical write or read waited for the encoder"
        assert running.thread.is_alive()  # the encoder really was still busy
    finally:
        backend.release.set()
    running.result()
    assert [m.content for m in found[0]] == ["written while the encoder is busy"]
    assert not any(backend.under_lock)


def test_preflight_answer_after_a_concurrent_write_is_published_for_the_current_state(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.hold_when = lambda texts: texts == ("schema change",)
    running = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        written = Urdyn.open(root).remember("schema rollout note", kind="root_cause")
    finally:
        backend.release.set()
    result = running.result()
    # the write landed during inference: the answer is published for the state
    # that includes it, which the index honestly does not cover yet -- never
    # silently "ready", and no row was invented for it
    assert result.retrieval.status == SEMANTIC_STALE and result.retrieval.missing == 1
    assert written.memory_id in {m.memory_id for m in cx.timeline()}
    assert written.memory_id not in indexed(cx._semantic_db_path)
    # covered by the next call, encoded outside the lock as well
    assert cx.preflight("schema change").retrieval.status == SEMANTIC_READY
    assert written.memory_id in indexed(cx._semantic_db_path)
    assert not any(backend.under_lock)


def test_concurrent_recalls_proceed_while_the_encoder_is_held(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.hold_when = lambda texts: texts == ("schema change",)
    running = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        readers = [Background(lambda: Urdyn.open(root).recall("migrations")) for _ in range(8)]
        assert all(reader.join(PROMPT) for reader in readers), "a recall waited for the encoder"
        assert all(len(reader.result()) == 1 for reader in readers)
    finally:
        backend.release.set()
    running.result()


def test_another_process_proceeds_while_the_encoder_is_held(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.hold_when = lambda texts: texts == ("schema change",)
    running = Background(lambda: cx.preflight("schema change"))
    code = ("import sys; from urdyn import Urdyn; cx = Urdyn.open(sys.argv[1]); "
            "cx.remember('written by another process', kind='note'); print(len(cx.recall('another process')))")
    try:
        assert backend.entered.wait(GATE_SECONDS)
        started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, "-c", code, str(root)], env=subprocess_env(),
                                     capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("another process waited for this process's encoder")
        elapsed = time.monotonic() - started
        assert process.returncode == 0, process.stderr
        assert process.stdout.strip() == "1"
        assert running.thread.is_alive()
    finally:
        backend.release.set()
    running.result()
    assert elapsed < 15


def test_slow_encoder_three_seconds_does_not_hold_the_lock(tmp_path):
    """A slow encoder: 3 s per inference. A write waits for the lock,
    never for the encoder; the call itself still takes as long as it must."""
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    for index in range(200):
        cx.remember(f"pending note {index}", kind="note")  # a stale index: 200 pending texts
    backend.delay = 3.0
    running = Background(lambda: cx.preflight("schema change"))
    deadline = time.monotonic() + GATE_SECONDS
    while not backend.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert backend.requests, "the encoder was never called"
    other = Urdyn.open(root)
    waits = []
    for index in range(3):
        started = time.monotonic()
        other.remember(f"written during slow inference {index}", kind="note")
        waits.append(time.monotonic() - started)
    result = running.result()
    assert max(waits) < 1.0, waits  # each write took the lock in milliseconds
    assert running.elapsed >= 6.0  # one refresh batch + one query, 3 s each, outside the lock
    assert len(backend.requests) == 2 and len(backend.requests[0]) == 200
    assert not any(backend.under_lock)
    assert result.retrieval.status == SEMANTIC_STALE and result.retrieval.missing == 3


def test_restore_during_inference_is_not_blocked_and_nothing_planned_is_published(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    archive = tmp_path / "backup.zip"
    backup.create_backup(root, archive)  # canonical only: the index is never archived
    later = cx.remember("schema note written after the backup", kind="root_cause")
    backend.hold_when = lambda texts: any("after the backup" in text for text in texts)
    running = Background(lambda: cx.preflight("schema migrations dry run"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, restored = within(PROMPT * 2, lambda: backup.restore_backup(root, archive))
        assert finished, "the restore waited for the encoder"
        assert json.loads(restored)["previous_brain_retained"] is True
    finally:
        backend.release.set()
    result = running.result()
    # answered for the restored Brain: the later note is not there, and the
    # restored Brain has no index (rebuild is explicit) -- nothing was put in one
    assert later.memory_id not in [m.memory_id for m in result.root_causes]
    assert [m.content for m in result.root_causes] == ["Migrations require a dry run"]
    assert result.retrieval.status == SEMANTIC_DISABLED
    assert not cx._semantic_db_path.exists()
    # nor was the planned vector published into the Brain that was replaced
    rollback_index = root.parent / ".brain.rollback" / ".urdyn" / "semantic_index.db"
    assert later.memory_id not in indexed(rollback_index)


def test_rebuild_with_another_encoder_during_inference_is_not_mixed_in(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    cx.remember("schema pending note", kind="root_cause")  # pending: a refresh is planned
    backend.hold_when = lambda texts: texts == ("schema change",)
    rebuilder = Urdyn.open(root)
    rebuilder.configure_semantic_backend(GatedEncoder(provider="test-encoder-v2", axis=7))
    running = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, _ = within(PROMPT * 2, rebuilder.semantic_setup)
        assert finished, "the rebuild waited for the encoder"
    finally:
        backend.release.set()
    result = running.result()
    # the index now belongs to another encoder: this handle's rows (planned,
    # refresh included) were not published into it, and it is not consulted
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_MODEL_MISMATCH
    assert stored_axes(cx._semantic_db_path) == {7}


def test_a_foreign_brain_swapped_in_during_inference_receives_nothing(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    cx.remember("schema pending note", kind="root_cause")  # pending: a refresh is planned
    foreign_root, foreign, foreign_backend = brain(tmp_path, "unrelated foreign note", name="foreign")
    foreign.remember("foreign pending note", kind="note")
    foreign_db, foreign_index = digest(foreign._db_path), digest(foreign._semantic_db_path)
    backend.hold_when = lambda texts: texts == ("schema change",)
    running = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        root.rename(tmp_path / "brain-away")
        foreign_root.rename(root)  # an external swap, outside the lifecycle
    finally:
        backend.release.set()
    with pytest.raises(UrdynStorageError):
        running.result()
    assert digest(root / ".urdyn" / "memory.db") == foreign_db
    assert digest(root / ".urdyn" / "semantic_index.db") == foreign_index


# ---------------------------------------------------------------------------
# 2. Failure policy: observable, bounded, never repeated
# ---------------------------------------------------------------------------


def test_encoder_failing_after_the_snapshot_is_reported_once_and_not_retried(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    cx.remember("schema pending note", kind="root_cause")
    reference = shape(lexical_only(cx, "schema migrations"))
    backend.fail_when = lambda texts: True
    result = cx.preflight("schema migrations")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_QUERY_FAILED
    assert shape(result) == reference  # exactly the lexical answer, under a degraded label
    assert backend.requests == [("schema pending note",), ("schema migrations",)]  # each once
    assert cx.semantic_state().status == SEMANTIC_STALE


def test_refresh_failure_with_a_working_query_keeps_the_stale_meaning(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    cx.remember("schema pending note", kind="root_cause")
    backend.fail_when = lambda texts: texts == ("schema pending note",)
    result = cx.preflight("schema migrations")
    assert result.retrieval.status == SEMANTIC_STALE and result.retrieval.detail == DETAIL_REFRESH_FAILED
    assert len(backend.requests) == 2


def test_encoder_unavailable_before_the_snapshot_is_never_called(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.is_available = False
    result = cx.preflight("schema migrations")
    assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_MODEL_UNCACHED
    assert backend.requests == []


def test_backend_vanishing_between_snapshot_and_publication_is_reported(tmp_path, monkeypatch):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    original = workspace_module.Urdyn._infer_outside_lock

    def infer_then_vanish(self, ticket):
        prepared = original(self, ticket)
        backend.is_available = False
        return prepared

    monkeypatch.setattr(workspace_module.Urdyn, "_infer_outside_lock", infer_then_vanish)
    result = cx.preflight("schema change")
    assert result.retrieval.status != SEMANTIC_READY
    assert result.root_causes == ()  # no semantic admission from rows of an encoder now gone


def test_a_stuck_encoder_blocks_only_its_own_call(tmp_path, monkeypatch):
    """No cancellation is claimed: the stuck call keeps running. Others are
    not blocked; a second retrieval call of this process waits a bounded time
    for the busy encoder and then answers lexically, saying so."""
    monkeypatch.setattr(workspace_module, "INFERENCE_SLOT_TIMEOUT_SECONDS", 0.5, raising=False)
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.hold_when = lambda texts: texts == ("schema change",)
    stuck = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        archive = tmp_path / "backup.zip"
        finished, _, _ = within(PROMPT, lambda: (Urdyn.open(root).remember("still writable", kind="note"),
                                                 backup.create_backup(root, archive)))
        assert finished
        reference = shape(lexical_only(cx, "migrations dry run"))
        finished, elapsed, second = within(PROMPT, lambda: cx.preflight("migrations dry run"))
        assert finished and elapsed >= 0.5
        assert second.retrieval.status == SEMANTIC_UNAVAILABLE and second.retrieval.detail == DETAIL_QUERY_FAILED
        assert shape(second) == reference
        assert stuck.thread.is_alive()  # not interrupted: Python cannot cancel it
    finally:
        backend.release.set()
    stuck.result()


# ---------------------------------------------------------------------------
# 3. The protocol itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", sorted(RETRIEVAL_CALLS))
def test_inference_never_runs_under_the_lifecycle_lock(tmp_path, call):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    cx.remember("schema pending note", kind="root_cause")
    RETRIEVAL_CALLS[call](cx)
    assert backend.requests and not any(backend.under_lock)


def test_a_semantic_call_takes_two_holds_and_a_lexical_call_one(tmp_path, monkeypatch):
    import fcntl

    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    plain = Urdyn.create(tmp_path / "plain", "dev")
    plain.remember("Migrations require a dry run", kind="root_cause")
    count = {"n": 0}
    original = lifecycle_module.fcntl.flock

    def counting(fd, flags):
        if flags & fcntl.LOCK_EX:
            count["n"] += 1
        return original(fd, flags)

    monkeypatch.setattr(lifecycle_module.fcntl, "flock", counting)
    cx.preflight("schema change")
    assert count["n"] == 2  # plan, then publication
    count["n"] = 0
    plain.preflight("migrations")
    assert count["n"] == 1  # no index: nothing to plan, one hold as before


def test_prepared_rows_are_bound_to_encoder_and_content(tmp_path):
    from urdyn._encoding_plan import EncodingNotPrepared, EncodingTicket, PreparedEncodings, infer

    ticket = EncodingTicket(
        brain_id="0" * 32, incarnation=(None, None), encoder_key=("k",), meta=None, backend=None,
        backend_identity=None, query="q", refresh=((ENTITY_MEMORY, (("id1", "old text"),)),))
    source = np.array([[3.0, 4.0]], np.float32)
    prepared = infer(ticket, build_model=lambda: object(), embed=lambda model, texts: source.copy()
                     if texts != ["q"] else np.array([[1.0, 0.0]], np.float32),
                     validate_refresh=lambda rows, count: rows)
    encoder = prepared.encoder_for(("k",))
    assert encoder.refresh_rows(ENTITY_MEMORY, [("id1", "new text")]) == ([], None)  # same id, other text
    ids, rows = encoder.refresh_rows(ENTITY_MEMORY, [("id1", "old text"), ("id2", "written later")])
    assert ids == ["id1"] and rows.tolist() == [[3.0, 4.0]]
    rows[0, 0] = 99.0  # a caller's copy: the prepared result does not change
    assert encoder.refresh_rows(ENTITY_MEMORY, [("id1", "old text")])[1].tolist() == [[3.0, 4.0]]
    with pytest.raises(EncodingNotPrepared):
        prepared.encoder_for(("other",)).query_rows("q")
    with pytest.raises(EncodingNotPrepared):
        encoder.query_rows("another query")
    with pytest.raises(EncodingNotPrepared):
        encoder.encode(["q"])  # never infers
    with pytest.raises(EncodingNotPrepared):
        PreparedEncodings(None).encoder_for(("k",)).query_rows("q")


def test_an_encoder_mutating_its_output_later_cannot_change_the_prepared_rows():
    from urdyn._encoding_plan import EncodingTicket, infer

    shared = np.array([[1.0, 2.0]], np.float32)
    ticket = EncodingTicket(brain_id="0" * 32, incarnation=(None, None), encoder_key=("k",), meta=None,
                            backend=None, backend_identity=None, query="q", refresh=())
    prepared = infer(ticket, build_model=lambda: object(), embed=lambda model, texts: shared,
                     validate_refresh=lambda rows, count: rows)
    shared[0, 0] = -5.0  # the encoder reuses its buffer after the lock was released
    assert prepared.encoder_for(("k",)).query_rows("q").tolist() == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# 4. Mixed load, processes, interruptions
# ---------------------------------------------------------------------------


def test_mixed_load_has_no_deadlock_and_loses_nothing(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    backend.delay = 0.02
    errors, written = [], []
    lock = threading.Lock()

    def writer(tag):
        handle = Urdyn.open(root)
        for index in range(15):
            memory = handle.remember(f"{tag} schema note {index}", kind="note")
            with lock:
                written.append(memory.memory_id)

    def reader():
        for _ in range(10):
            cx.preflight("schema change")
            cx.guard("schema change")

    workers = [Background(lambda t=t: writer(t)) for t in ("a", "b")] + [Background(reader) for _ in range(2)]
    for worker in workers:
        assert worker.join(90), "deadlock or starvation"
        if worker.error is not None:
            errors.append(worker.error)
    assert errors == []
    canonical = {m.memory_id for m in Urdyn.open(root).timeline()}
    assert set(written) <= canonical and len(written) == 30
    assert indexed(cx._semantic_db_path) <= canonical  # nothing published for a record that does not exist
    assert cx.preflight("schema change").retrieval.status == SEMANTIC_READY
    assert not any(backend.under_lock)


HELPER = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    import urdyn._workspace as w
    from urdyn import Urdyn, SemanticBackendIdentity
    from urdyn._semantic import ARTIFACT_ARM64, ARTIFACT_ARM64_SHA256, TOKENIZER_SHA256, model_identity_for

    root, point, marker = sys.argv[1], sys.argv[2], Path(sys.argv[3])

    def stop_here():
        marker.write_text(point)
        time.sleep(120)

    class Encoder:
        identity = SemanticBackendIdentity(
            provider="test-encoder-v1", model_id=model_identity_for(ARTIFACT_ARM64),
            model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42", dimensions=384,
            artifact_sha256=ARTIFACT_ARM64_SHA256, tokenizer_sha256=TOKENIZER_SHA256,
            encoding_policy="deterministic-v1")
        def available(self):
            return True
        def encode(self, texts):
            if point == "inference":
                stop_here()
            return [[0.0] + [1.0] + [0.0] * 382 for _ in texts]

    if point == "planning":
        original = w._semantic_pool_entries
        def planning(*args, **kwargs):
            stop_here()
        w._semantic_pool_entries = planning
    if point == "publication":
        from urdyn._semantic_store import SemanticIndexStore
        original_add = SemanticIndexStore.add_vectors
        def add_then_stop(self, entity_type, rows):
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute("DELETE FROM semantic_vectors")  # uncommitted damage
            stop_here()
        SemanticIndexStore.add_vectors = add_then_stop
    cx = Urdyn.open(root)
    cx.configure_semantic_backend(Encoder())
    cx.preflight("schema change")
''')


@pytest.mark.parametrize("point", ["planning", "inference", "publication"])
def test_a_process_killed_at_each_step_leaves_a_usable_brain(tmp_path, point):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    pending = cx.remember("schema pending note", kind="root_cause")
    vectors_before = indexed(cx._semantic_db_path)
    canonical_before = [(m.memory_id, m.content) for m in cx.timeline()]
    helper, marker = tmp_path / "helper.py", tmp_path / "marker"
    helper.write_text(HELPER)
    process = subprocess.Popen([sys.executable, str(helper), str(root), point, str(marker)], env=subprocess_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + GATE_SECONDS
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), process.stderr.read().decode()[-2000:] if process.poll() is not None else "no marker"
    finally:
        process.send_signal(signal.SIGKILL)
        process.wait(10)
    # the lock died with the process; canonical intact; the index never lost
    # committed vectors and holds nothing for records that do not exist
    finished, _, _ = within(PROMPT, lambda: Urdyn.open(root).remember("after the crash", kind="note"))
    assert finished
    reopened = Urdyn.open(root)
    assert [(m.memory_id, m.content) for m in reopened.timeline()][:2] == canonical_before
    assert vectors_before <= indexed(cx._semantic_db_path) <= {m.memory_id for m in reopened.timeline()}
    assert cx.preflight("schema change").retrieval.status == SEMANTIC_READY
    assert pending.memory_id in indexed(cx._semantic_db_path)
