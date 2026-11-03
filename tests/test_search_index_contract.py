"""Contract of the derived FTS5 search index (`docs/search-index.md`).

`open` and every read stay free of writes and never repair the
index; a public diagnostic tells the runtime's FTS5 support apart from the
state of the index (`missing`, `usable`, `invalid`, `unchecked`); an
explicit, atomic rebuild recovers it, including from total absence; the
retrieval paths that use FTS report when the channel could not contribute,
without changing ranking, thresholds or admission.

All four indexed entity types are exercised: memory, attempt, skill and
source (observations of seeded Sources). Synthetic Brains in temporary
directories only.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3

import pytest

from urdyn import Urdyn, UrdynStorageError, UrdynStorageUnavailableError
from urdyn._store import SearchIndexState
from urdyn._cli import main
from urdyn._retrieval import ENTITY_ATTEMPT, ENTITY_MEMORY, ENTITY_SKILL, ENTITY_SOURCE
from urdyn._store import SEARCH_INDEX_TABLE, MemoryStore

ENTITY_TYPES = (ENTITY_MEMORY, ENTITY_ATTEMPT, ENTITY_SKILL, ENTITY_SOURCE)
# One distinctive term per entity type, present only in that entity's indexed text.
TERMS = {ENTITY_MEMORY: "zebrafact", ENTITY_ATTEMPT: "quokkatask", ENTITY_SKILL: "ibexskill",
         ENTITY_SOURCE: "narwhalnote"}
_PROC_COUNT = 6


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _brain(tmp_path):
    """A Brain holding at least one indexed row of every entity type."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember(f"The deploy {TERMS[ENTITY_MEMORY]} needs a migration first.", kind="note")
    cx.record_attempt(task=f"Run the {TERMS[ENTITY_ATTEMPT]} migration", approach="direct apply",
                      outcome="failed")
    lesson = cx.learn("Migrations must be applied before the deploy.")
    cx.promote(lesson, name=f"Apply {TERMS[ENTITY_SKILL]} migrations", purpose="Safe schema changes",
               steps=["apply", "verify"], conditions=["before deploy"])
    (tmp_path / "notes.md").write_text(f"Project note: the {TERMS[ENTITY_SOURCE]} service owns the schema.\n",
                                       encoding="utf-8")
    cx.seed(["notes.md"])
    return cx


def _sql(db_path, statement, params=()):
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            return connection.execute(statement, params).fetchall()
    finally:
        connection.close()


def _index_objects(db_path):
    return sorted(r[0] for r in _sql(db_path, "SELECT name FROM sqlite_master WHERE name LIKE ?",
                                     (SEARCH_INDEX_TABLE + "%",)))


def _drop_index(db_path):
    _sql(db_path, f"DROP TABLE {SEARCH_INDEX_TABLE}")
    assert _index_objects(db_path) == []


def _files(cx):
    urdyn_dir = cx.path / ".urdyn"
    return {p.name: p.read_bytes() for p in sorted(urdyn_dir.iterdir()) if p.is_file()}


def _canonical_dump(db_path):
    """Every canonical object and row, excluding the derived search index."""
    connection = sqlite3.connect(db_path)
    try:
        # iterdump() wraps virtual tables in PRAGMA writable_schema lines: not canonical either.
        return [line for line in connection.iterdump()
                if SEARCH_INDEX_TABLE not in line and not line.startswith("PRAGMA writable_schema")]
    finally:
        connection.close()


def _fts_hits(db_path, entity_type):
    with MemoryStore.open_existing(db_path) as store:
        return [entity_id for entity_id, _ in store.search_candidates(frozenset({TERMS[entity_type]}), entity_type)]


def _counts(state: SearchIndexState):
    return {entity_type: (expected, indexed) for entity_type, expected, indexed in state.entities}


# --------------------------------------------------------------------------
# 1. Diagnostic: runtime support vs index state, never a write
# --------------------------------------------------------------------------


def test_fresh_brain_index_is_usable_for_every_entity_type(tmp_path):
    cx = _brain(tmp_path)
    state = cx.search_index_state()
    assert state.runtime_fts5 is True
    assert state.status == "usable", state.detail
    counts = _counts(state)
    assert set(counts) == set(ENTITY_TYPES)
    for entity_type in ENTITY_TYPES:
        expected, indexed = counts[entity_type]
        assert expected >= 1 and indexed == expected, entity_type


def test_a_missing_index_is_reported_missing(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    state = Urdyn.open(tmp_path).search_index_state()
    assert state.runtime_fts5 is True
    assert state.status == "missing"


@pytest.mark.parametrize("entity_type", ENTITY_TYPES)
def test_a_row_missing_from_the_index_is_invalid(tmp_path, entity_type):
    cx = _brain(tmp_path)
    _sql(cx._db_path, f"DELETE FROM {SEARCH_INDEX_TABLE} WHERE entity_type = ?", (entity_type,))
    state = cx.search_index_state()
    assert state.status == "invalid"
    expected, indexed = _counts(state)[entity_type]
    assert indexed == 0 and expected >= 1


@pytest.mark.parametrize("entity_type", ENTITY_TYPES)
def test_a_stale_row_is_invalid(tmp_path, entity_type):
    cx = _brain(tmp_path)
    _sql(cx._db_path, f"UPDATE {SEARCH_INDEX_TABLE} SET text = 'stale text' WHERE entity_type = ?", (entity_type,))
    assert cx.search_index_state().status == "invalid"


@pytest.mark.parametrize("entity_type", [*ENTITY_TYPES, "unknown-kind"])
def test_an_extra_row_is_invalid(tmp_path, entity_type):
    cx = _brain(tmp_path)
    _sql(cx._db_path, f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
         (entity_type, "f" * 32, "a row no canonical record backs"))
    assert cx.search_index_state().status == "invalid"


def test_a_corrupted_inverted_index_is_invalid(tmp_path):
    """Content rows intact, posting lists damaged: MATCH would silently miss."""
    cx = _brain(tmp_path)
    _sql(cx._db_path, f"DELETE FROM {SEARCH_INDEX_TABLE}_data WHERE id > 10")
    assert cx.search_index_state().status == "invalid"


def test_an_index_with_another_definition_is_invalid(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    _sql(cx._db_path, f"CREATE VIRTUAL TABLE {SEARCH_INDEX_TABLE} USING fts5(entity_type, entity_id, text, extra)")
    assert cx.search_index_state().status == "invalid"


def test_without_fts5_in_the_runtime_a_present_index_is_unchecked(tmp_path, monkeypatch):
    import urdyn._store as store_module

    cx = _brain(tmp_path)
    monkeypatch.setattr(store_module, "fts5_runtime_available", lambda: False)
    state = cx.search_index_state()
    assert (state.runtime_fts5, state.status) == (False, "unchecked")
    _drop_index(cx._db_path)
    state = cx.search_index_state()
    assert (state.runtime_fts5, state.status) == (False, "missing")


@pytest.mark.parametrize("damage", ["none", "missing", "stale", "corrupted"])
def test_the_diagnostic_never_writes(tmp_path, damage):
    cx = _brain(tmp_path)
    if damage == "missing":
        _drop_index(cx._db_path)
    elif damage == "stale":
        _sql(cx._db_path, f"UPDATE {SEARCH_INDEX_TABLE} SET text = 'stale'")
    elif damage == "corrupted":
        _sql(cx._db_path, f"DELETE FROM {SEARCH_INDEX_TABLE}_data WHERE id > 10")
    before = _files(cx)
    cx.search_index_state()
    cx.search_index_state()
    assert _files(cx) == before


# --------------------------------------------------------------------------
# 2. open and reads never repair (no automatic recovery)
# --------------------------------------------------------------------------


def test_open_and_reads_leave_a_missing_index_missing_and_bytes_unchanged(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    before = _files(cx)
    reopened = Urdyn.open(tmp_path)
    reopened.state()
    reopened.recall("deploy")
    reopened.preflight("deploy the migration")
    reopened.guard("apply the migration")
    reopened.context("deploy the migration")
    assert _index_objects(cx._db_path) == []
    assert _files(cx) == before


# --------------------------------------------------------------------------
# 3. Explicit rebuild
# --------------------------------------------------------------------------


@pytest.mark.parametrize("damage", ["missing", "row_missing", "stale", "extra", "corrupted", "definition"])
def test_rebuild_recovers_every_damage_and_keeps_canonical_data(tmp_path, damage):
    cx = _brain(tmp_path)
    db = cx._db_path
    canonical = _canonical_dump(db)
    if damage == "missing":
        _drop_index(db)
    elif damage == "row_missing":
        _sql(db, f"DELETE FROM {SEARCH_INDEX_TABLE} WHERE entity_type = 'source'")
    elif damage == "stale":
        _sql(db, f"UPDATE {SEARCH_INDEX_TABLE} SET text = 'stale'")
    elif damage == "extra":
        _sql(db, f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES ('memory', ?, 'x')",
             ("e" * 32,))
    elif damage == "corrupted":
        _sql(db, f"DELETE FROM {SEARCH_INDEX_TABLE}_data WHERE id > 10")
    elif damage == "definition":
        _drop_index(db)
        _sql(db, f"CREATE VIRTUAL TABLE {SEARCH_INDEX_TABLE} USING fts5(entity_type, entity_id, text, extra)")
    assert cx.search_index_state().status != "usable"

    state = cx.rebuild_search_index()

    assert state.status == "usable", state.detail
    assert cx.search_index_state() == state
    assert _canonical_dump(db) == canonical
    for entity_type in ENTITY_TYPES:
        assert len(_fts_hits(db, entity_type)) == 1, entity_type


def test_writes_after_a_rebuild_are_indexed_again(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    cx.remember("written while the index was missing: wombatword", kind="note")
    assert cx.search_index_state().status == "missing"
    cx.rebuild_search_index()
    cx.remember("written after the rebuild: koalaword", kind="note")
    assert cx.search_index_state().status == "usable"
    with MemoryStore.open_existing(cx._db_path) as store:
        assert len(store.search_candidates(frozenset({"wombatword"}), ENTITY_MEMORY)) == 1
        assert len(store.search_candidates(frozenset({"koalaword"}), ENTITY_MEMORY)) == 1


@pytest.mark.parametrize("start", ["missing", "stale"])
def test_a_failed_rebuild_rolls_back_to_the_previous_state(tmp_path, monkeypatch, start):
    import urdyn._store as store_module

    cx = _brain(tmp_path)
    if start == "missing":
        _drop_index(cx._db_path)
    else:
        _sql(cx._db_path, f"UPDATE {SEARCH_INDEX_TABLE} SET text = 'stale'")
    objects = _index_objects(cx._db_path)
    rows = _sql(cx._db_path, f"SELECT * FROM {SEARCH_INDEX_TABLE}") if start != "missing" else None
    canonical = _canonical_dump(cx._db_path)
    status_before = cx.search_index_state().status

    def explode(connection):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store_module, "_rebuild_search_index", explode)
    with pytest.raises(UrdynStorageError) as info:
        cx.rebuild_search_index()
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    monkeypatch.undo()

    assert _index_objects(cx._db_path) == objects
    if rows is not None:
        assert _sql(cx._db_path, f"SELECT * FROM {SEARCH_INDEX_TABLE}") == rows
    assert _canonical_dump(cx._db_path) == canonical
    assert cx.search_index_state().status == status_before
    assert cx.rebuild_search_index().status == "usable"


def test_rebuild_refuses_without_fts5_and_writes_nothing(tmp_path, monkeypatch):
    import urdyn._store as store_module

    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    before = _files(cx)
    monkeypatch.setattr(store_module, "fts5_runtime_available", lambda: False)
    with pytest.raises(UrdynStorageUnavailableError):
        cx.rebuild_search_index()
    assert _files(cx) == before


def _worker(workspace, action, barrier, queue):
    try:
        barrier.wait(timeout=30)
        from urdyn import Urdyn as _Urdyn

        cx = _Urdyn.open(workspace)
        if action == "rebuild":
            queue.put(("ok", cx.rebuild_search_index().status))
        else:
            queue.put(("ok", cx.remember(action, kind="note").memory_id))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run(workspace, actions, timeout=90):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(len(actions)), ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(workspace, a, barrier, queue)) for a in actions]
    for p in processes:
        p.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for p in processes:
        p.join(timeout=timeout)
        assert p.exitcode == 0
    assert not [r for r in results if r[0] == "error"], results
    return results


def test_concurrent_rebuilds_from_a_missing_index_leave_exactly_one_complete_index(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    results = _run(tmp_path, ["rebuild"] * _PROC_COUNT)
    assert {status for _, status in results} == {"usable"}
    state = cx.search_index_state()
    assert state.status == "usable"
    assert _sql(cx._db_path, "SELECT COUNT(*) FROM sqlite_master WHERE name = ?", (SEARCH_INDEX_TABLE,)) == [(1,)]
    total_expected = sum(expected for _, expected, _ in state.entities)
    assert _sql(cx._db_path, f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE}") == [(total_expected,)]


def test_rebuilds_interleaved_with_writers_lose_no_row(tmp_path):
    cx = _brain(tmp_path)
    _drop_index(cx._db_path)
    writes = [f"concurrent write number {i} capybaraterm" for i in range(_PROC_COUNT)]
    _run(tmp_path, ["rebuild", "rebuild", *writes])
    state = cx.search_index_state()
    assert state.status == "usable", state.detail
    # Whatever the interleaving (all serialized by the lifecycle lock), a write
    # either finds the index present and indexes itself in its own transaction,
    # or precedes a rebuild that projects it: none may be lost.
    with MemoryStore.open_existing(cx._db_path) as store:
        assert len(store.search_candidates(frozenset({"capybaraterm"}), ENTITY_MEMORY)) == _PROC_COUNT


# --------------------------------------------------------------------------
# 4. Retrieval: degradation observable, results unchanged, errors not silenced
# --------------------------------------------------------------------------


def test_retrieval_reports_the_index_and_its_absence(tmp_path):
    cx = _brain(tmp_path)
    assert cx.preflight("deploy the migration").search_index == "unchecked"
    assert cx.guard("apply the migration").search_index == "unchecked"
    assert cx.context("deploy the migration").search_index == "unchecked"
    _drop_index(cx._db_path)
    assert cx.preflight("deploy the migration").search_index == "missing"
    assert cx.guard("apply the migration").search_index == "missing"
    assert cx.context("deploy the migration").search_index == "missing"


def test_rebuild_restores_exactly_the_retrieval_results_of_an_intact_index(tmp_path):
    cx = _brain(tmp_path)
    queries = [f"{TERMS[ENTITY_ATTEMPT]} migration", f"{TERMS[ENTITY_SKILL]} migrations",
               f"{TERMS[ENTITY_SOURCE]} schema", f"{TERMS[ENTITY_MEMORY]} deploy"]
    intact = [(cx.preflight(q), cx.guard(q), cx.context(q).render()) for q in queries]
    _drop_index(cx._db_path)
    cx.rebuild_search_index()
    assert [(cx.preflight(q), cx.guard(q), cx.context(q).render()) for q in queries] == intact


def test_a_broken_index_fails_retrieval_loudly_instead_of_answering_empty(tmp_path):
    cx = _brain(tmp_path)
    # Shadow storage removed under a declared virtual table: MATCH cannot run.
    _sql(cx._db_path, f"DROP TABLE {SEARCH_INDEX_TABLE}_data")
    with pytest.raises(UrdynStorageError):
        cx.preflight(f"{TERMS[ENTITY_ATTEMPT]} migration")
    assert cx.search_index_state().status == "invalid"


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------


def test_cli_reports_and_rebuilds(tmp_path, monkeypatch, capsys):
    cx = _brain(tmp_path)
    monkeypatch.chdir(tmp_path)
    _drop_index(cx._db_path)

    assert main(["status"]) == 0
    assert "Search index: missing" in capsys.readouterr().out
    assert main(["search-index", "status"]) == 0
    assert "missing" in capsys.readouterr().out
    assert main(["search-index", "rebuild"]) == 0
    assert "usable" in capsys.readouterr().out
    assert main(["search-index", "status"]) == 0
    assert "usable" in capsys.readouterr().out
