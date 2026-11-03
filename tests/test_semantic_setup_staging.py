"""`semantic_setup` outside the lifecycle lock, published atomically.

Synthetic Brains only. The deterministic gated encoder of `test_lifecycle_concurrency.py` is reused:
the test opens an explicit gate instead of sleeping, and every request
records whether the calling thread held the lifecycle lock.

What these tests falsify:
- a slow or held rebuild blocks writes, reads, restore or other processes;
- a rebuild prepared for a state that no longer exists is published
  (restore, foreign Brain, another rebuild, another backend on the handle);
- a partial index is ever readable as `ready` (crash at every step);
- the failure states change: probe/model failure leaves the existing index,
  an embedding failure leaves the `building` marker ("build incomplete");
- setups queue or retry without bound.
"""
from __future__ import annotations

import hashlib
import json
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import urdyn._lifecycle as lifecycle_module
import urdyn._workspace as workspace_module
from urdyn import Urdyn, UrdynSemanticUnavailableError, UrdynStorageError
from urdyn import backup
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._semantic_store import (
    DETAIL_BUILD_INCOMPLETE,
    SEMANTIC_READY,
    SEMANTIC_STALE,
    SEMANTIC_UNAVAILABLE,
    SemanticIndexStore,
)
from test_lifecycle_concurrency import (
    GATE_SECONDS,
    PROMPT,
    Background,
    GatedEncoder,
    digest,
    indexed,
    stored_axes,
    subprocess_env,
    within,
)

PROBE = ("_urdyn_semantic_dimension_probe_",)


def brain(tmp_path, *root_causes, name="brain", backend=None):
    root = tmp_path / name
    cx = Urdyn.create(root, "dev")
    backend = backend or GatedEncoder()
    cx.configure_semantic_backend(backend)
    for content in root_causes:
        cx.remember(content, kind="root_cause")
    cx.semantic_setup()
    backend.requests.clear()
    backend.under_lock.clear()
    return root, cx, backend


def holding_the_pools(backend):
    backend.hold_when = lambda texts: texts != PROBE
    return backend


def meta(cx):
    with SemanticIndexStore.open_if_exists(cx._semantic_db_path) as store:
        return store.meta(), store.vector_count()


def peer(root, backend=None):
    """Another handle on the same Brain, with its own backend object."""
    handle = Urdyn.open(root)
    handle.configure_semantic_backend(backend or GatedEncoder())
    return handle


# ---------------------------------------------------------------------------
# 1. Availability during a rebuild
# ---------------------------------------------------------------------------


def test_writes_reads_and_other_processes_proceed_while_setup_embeds(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run", "Schema changes need review")
    holding_the_pools(backend)
    running = Background(cx.semantic_setup)
    code = ("import sys; from urdyn import Urdyn; cx = Urdyn.open(sys.argv[1]); "
            "cx.remember('written by another process', kind='note'); print(len(cx.recall('another process')))")
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, found = within(PROMPT, lambda: (
            Urdyn.open(root).remember("written while setup embeds", kind="note"),
            Urdyn.open(root).recall("setup embeds"),
        )[1])
        assert finished, "a canonical write or read waited for the rebuild"
        assert len(found) == 1
        try:
            process = subprocess.run([sys.executable, "-c", code, str(root)], env=subprocess_env(),
                                     capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("another process waited for this process's rebuild")
        assert process.returncode == 0 and process.stdout.strip() == "1", process.stderr
        # while the marker is there the index reads as a rebuild in progress
        finished, _, state = within(PROMPT, lambda: peer(root).semantic_state())
        assert finished
        assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE
        assert running.thread.is_alive()
    finally:
        backend.release.set()
    result = running.result()
    assert result.memory_count == 2  # the snapshot: what was published
    # the two writes made during the rebuild are honestly not covered yet ...
    state = cx.semantic_state()
    assert state.status == SEMANTIC_STALE and state.missing == 2
    # ... and the next retrieval call covers them
    assert cx.preflight("schema change").retrieval.status == SEMANTIC_READY
    assert not any(backend.under_lock)


def test_setup_never_embeds_under_the_lifecycle_lock_and_takes_three_short_holds(tmp_path, monkeypatch):
    import fcntl

    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    count = {"n": 0}
    original = lifecycle_module.fcntl.flock

    def counting(fd, flags):
        if flags & fcntl.LOCK_EX:
            count["n"] += 1
        return original(fd, flags)

    monkeypatch.setattr(lifecycle_module.fcntl, "flock", counting)
    cx.semantic_setup()
    assert count["n"] == 3  # checks; marker + snapshot; publication
    assert backend.requests == [PROBE, ("Migrations require a dry run",)]
    assert not any(backend.under_lock)


def test_a_retrieval_call_during_the_rebuild_is_lexical_and_writes_nothing(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    reader = peer(root)
    reader.remember("schema pending note", kind="root_cause")  # pending: a refresh would run
    holding_the_pools(backend)
    running = Background(cx.semantic_setup)
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, result = within(PROMPT, lambda: reader.preflight("migrations dry run"))
        assert finished
        assert result.retrieval.status == SEMANTIC_UNAVAILABLE and result.retrieval.detail == DETAIL_BUILD_INCOMPLETE
        assert [m.content for m in result.root_causes] == ["Migrations require a dry run"]  # lexical answer
        assert meta(cx)[1] == 0  # no refresh row went into the marker
    finally:
        backend.release.set()
    running.result()
    assert meta(cx)[0].status == "ready" and meta(cx)[1] == 2


# ---------------------------------------------------------------------------
# 2. Nothing is published for a state that no longer exists
# ---------------------------------------------------------------------------


def test_restore_during_setup_is_not_blocked_and_nothing_is_published(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    archive = tmp_path / "backup.zip"
    backup.create_backup(root, archive)  # canonical only: the index is never archived
    cx.remember("schema note written after the backup", kind="root_cause")
    holding_the_pools(backend)
    running = Background(cx.semantic_setup)
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, restored = within(PROMPT * 2, lambda: backup.restore_backup(root, archive))
        assert finished, "the restore waited for the rebuild"
        assert json.loads(restored)["previous_brain_retained"] is True
    finally:
        backend.release.set()
    with pytest.raises(UrdynSemanticUnavailableError, match="storage files were replaced"):
        running.result()
    assert not cx._semantic_db_path.exists()  # the restored Brain got no index
    assert [m.content for m in Urdyn.open(root).timeline()] == ["Migrations require a dry run"]
    assert cx.semantic_setup().memory_count == 1  # an explicit rerun works
    assert cx.semantic_state().status == SEMANTIC_READY


def test_a_foreign_brain_swapped_in_during_setup_receives_nothing(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    foreign_root, foreign, _ = brain(tmp_path, "unrelated foreign note", name="foreign")
    foreign_db, foreign_index = digest(foreign._db_path), digest(foreign._semantic_db_path)
    holding_the_pools(backend)
    running = Background(cx.semantic_setup)
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


def test_two_setups_in_one_process_the_last_started_stands(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run", "Schema changes need review")
    holding_the_pools(backend)
    first = Background(cx.semantic_setup)
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, _ = within(PROMPT * 2, lambda: peer(root, GatedEncoder(provider="test-encoder-v2", axis=7)).semantic_setup())
        assert finished, "a second rebuild with another encoder waited for the first"
    finally:
        backend.release.set()
    with pytest.raises(UrdynSemanticUnavailableError, match="Another semantic setup"):
        first.result()
    stored, count = meta(cx)
    assert stored.provider.startswith("test-encoder-v2") and stored.status == "ready" and count == 2
    assert stored_axes(cx._semantic_db_path) == {7}  # nothing of the first rebuild was mixed in


def test_two_setups_with_the_same_backend_do_not_run_inference_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_module, "INFERENCE_SLOT_TIMEOUT_SECONDS", 0.5)
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    holding_the_pools(backend)
    first = Background(cx.semantic_setup)
    try:
        assert backend.entered.wait(GATE_SECONDS)
        def second():
            other = Urdyn.open(root)
            other.configure_semantic_backend(backend)  # the SAME backend object
            with pytest.raises(UrdynSemanticUnavailableError, match="stayed busy"):
                other.semantic_setup()

        finished, _, _ = within(PROMPT, second)
        assert finished  # bounded: refused, never queued
    finally:
        backend.release.set()
    first.result()
    assert cx.semantic_state().status == SEMANTIC_READY


def test_two_setups_from_distinct_processes(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run", "Schema changes need review")
    entered, release = tmp_path / "entered", tmp_path / "release"
    helper = tmp_path / "slow_setup.py"
    helper.write_text(textwrap.dedent('''
        import sys, time
        from pathlib import Path
        from urdyn import Urdyn, SemanticBackendIdentity, UrdynSemanticUnavailableError
        from urdyn._semantic import ARTIFACT_ARM64, ARTIFACT_ARM64_SHA256, TOKENIZER_SHA256, model_identity_for
        root, entered, release = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
        class Encoder:
            identity = SemanticBackendIdentity(
                provider="test-encoder-v1", model_id=model_identity_for(ARTIFACT_ARM64),
                model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42", dimensions=384,
                artifact_sha256=ARTIFACT_ARM64_SHA256, tokenizer_sha256=TOKENIZER_SHA256,
                encoding_policy="deterministic-v1")
            def available(self):
                return True
            def encode(self, texts):
                if list(texts) != ["_urdyn_semantic_dimension_probe_"]:
                    entered.write_text("in")
                    deadline = time.monotonic() + 60  # bounded even if the test cannot release it
                    while not release.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                return [[0.0, 1.0] + [0.0] * 382 for _ in texts]
        cx = Urdyn.open(root)
        cx.configure_semantic_backend(Encoder())
        try:
            cx.semantic_setup()
            print("published")
        except UrdynSemanticUnavailableError as exc:
            print("not published:", exc)
    '''))
    process = subprocess.Popen([sys.executable, str(helper), str(root), str(entered), str(release)],
                               env=subprocess_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + GATE_SECONDS
        while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered.exists(), process.stderr.read() if process.poll() is not None else "no marker"
        finished, _, _ = within(PROMPT * 2, lambda: peer(root, GatedEncoder(provider="test-encoder-v2", axis=7)).semantic_setup())
        assert finished, "a rebuild in this process waited for the other process's rebuild"
    finally:
        release.write_text("go")
        out, err = process.communicate(timeout=60)
    assert process.returncode == 0, err
    assert out.startswith("not published: Another semantic setup"), out
    assert stored_axes(cx._semantic_db_path) == {7} and meta(cx)[0].status == "ready"


def test_a_backend_reconfigured_during_setup_is_not_mixed_in(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    holding_the_pools(backend)
    running = Background(cx.semantic_setup)
    try:
        assert backend.entered.wait(GATE_SECONDS)
        finished, _, _ = within(PROMPT, lambda: cx.configure_semantic_backend(GatedEncoder(provider="test-encoder-v2")))
        assert finished
    finally:
        backend.release.set()
    with pytest.raises(UrdynSemanticUnavailableError, match="backend of this handle changed"):
        running.result()
    state = cx.semantic_state()  # the marker stays: the last build did not finish
    assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE
    assert meta(cx)[1] == 0


# ---------------------------------------------------------------------------
# 3. Failure states (unchanged contract)
# ---------------------------------------------------------------------------


def test_an_embedding_failure_over_a_ready_index_leaves_the_marker_as_before(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    canonical = digest(cx._db_path)
    backend.fail_when = lambda texts: texts != PROBE
    with pytest.raises(UrdynSemanticUnavailableError, match="rebuild failed"):
        cx.semantic_setup()
    state = cx.semantic_state()
    assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE
    assert meta(cx)[1] == 0  # never a partial index
    assert digest(cx._db_path) == canonical
    backend.fail_when = None
    cx.semantic_setup()
    assert cx.semantic_state().status == SEMANTIC_READY


def test_a_probe_failure_leaves_the_existing_index_untouched(tmp_path):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    before = digest(cx._semantic_db_path)
    backend.fail_when = lambda texts: texts == PROBE
    with pytest.raises(UrdynSemanticUnavailableError, match="Could not load"):
        cx.semantic_setup()
    assert digest(cx._semantic_db_path) == before and cx.semantic_state().status == SEMANTIC_READY


def test_a_busy_encoder_refuses_setup_before_touching_the_index(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_module, "INFERENCE_SLOT_TIMEOUT_SECONDS", 0.5)
    root, cx, backend = brain(tmp_path, "Migrations require a dry run")
    before = digest(cx._semantic_db_path)
    backend.hold_when = lambda texts: texts == ("schema change",)
    stuck = Background(lambda: cx.preflight("schema change"))
    try:
        assert backend.entered.wait(GATE_SECONDS)
        with pytest.raises(UrdynSemanticUnavailableError, match="stayed busy"):
            cx.semantic_setup()
        assert digest(cx._semantic_db_path) == before
        assert stuck.thread.is_alive()  # the stuck call is not interrupted
    finally:
        backend.release.set()
    stuck.result()


# ---------------------------------------------------------------------------
# 4. Interruptions and restart
# ---------------------------------------------------------------------------

KILL_HELPER = textwrap.dedent('''
    import sys, time
    from pathlib import Path
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
            probe = list(texts) == ["_urdyn_semantic_dimension_probe_"]
            if (point == "probe" and probe) or (point == "embedding" and not probe):
                stop_here()
            return [[0.0, 1.0] + [0.0] * 382 for _ in texts]

    if point == "publication":
        from urdyn._semantic_store import SemanticIndexStore
        def publish_then_stop(self, expected, rows):
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute("DELETE FROM semantic_vectors")
            self._connection.executemany(
                "INSERT INTO semantic_vectors (entity_type, entity_id, vector) VALUES (?, ?, ?)", rows[:1])
            self._connection.execute("UPDATE semantic_meta SET status = 'ready' WHERE id = 1")
            stop_here()  # uncommitted: a partial index marked ready
        SemanticIndexStore.publish_rebuild = publish_then_stop
    cx = Urdyn.open(root)
    cx.configure_semantic_backend(Encoder())
    cx.semantic_setup()
''')


@pytest.mark.parametrize("point, expected", [("probe", "ready-unchanged"), ("embedding", "marker"),
                                             ("publication", "marker")])
def test_a_process_killed_at_each_step_never_leaves_a_partial_ready_index(tmp_path, point, expected):
    root, cx, backend = brain(tmp_path, "Migrations require a dry run", "Schema changes need review")
    index_before = digest(cx._semantic_db_path)
    canonical_before = [(m.memory_id, m.content) for m in cx.timeline()]
    helper, marker = tmp_path / "helper.py", tmp_path / "marker"
    helper.write_text(KILL_HELPER)
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
    # the lock died with the process, canonical intact
    finished, _, _ = within(PROMPT, lambda: Urdyn.open(root).remember("after the crash", kind="note"))
    assert finished
    assert [(m.memory_id, m.content) for m in Urdyn.open(root).timeline()][:2] == canonical_before
    state = cx.semantic_state()
    if expected == "ready-unchanged":
        assert digest(cx._semantic_db_path) == index_before
    else:
        assert state.status == SEMANTIC_UNAVAILABLE and state.detail == DETAIL_BUILD_INCOMPLETE
        assert meta(cx)[1] == 0  # the uncommitted partial index was rolled back
    # restart: an explicit rerun completes deterministically
    assert cx.semantic_setup().memory_count == 3
    assert cx.semantic_state().status == SEMANTIC_READY
    with sqlite3.connect(cx._semantic_db_path) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert indexed(cx._semantic_db_path) == {m.memory_id for m in cx.timeline()}
