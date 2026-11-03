"""The derived FTS5 search index: lazy creation/backfill, rebuild,
atomicity with canonical writes, and graceful behavior when this
SQLite build has no FTS5 support.

See `_store.py`'s module docstring for why the index is deliberately
NOT part of `STORE_SCHEMA_VERSION`: it holds no canonical truth, only a
rebuildable projection of `memories`/`attempts`/`skills`.
"""

import datetime as dt
import sqlite3
import uuid

import pytest

import legacy_brains as legacy
from urdyn import Urdyn, UrdynStorageError
from urdyn._evidence import Evidence
from urdyn._retrieval import ENTITY_MEMORY
from urdyn._store import SEARCH_INDEX_TABLE, MemoryStore

_CREATE_MEMORIES_V2_SQL = """
    CREATE TABLE memories (
        memory_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        epistemic_state TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        supersedes TEXT
    )
"""
_CREATE_EVIDENCE_SQL = """
    CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY, content TEXT NOT NULL, kind TEXT NOT NULL, recorded_at TEXT NOT NULL
    )
"""
_CREATE_MEMORY_EVIDENCE_SQL = """
    CREATE TABLE memory_evidence (
        memory_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (memory_id, evidence_id)
    )
"""
_CREATE_EVENTS_SQL = """
    CREATE TABLE events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
        subject_id TEXT NOT NULL, occurred_at TEXT NOT NULL
    )
"""
_CREATE_ATTEMPTS_SQL = """
    CREATE TABLE attempts (
        attempt_id TEXT PRIMARY KEY, task TEXT NOT NULL, approach TEXT NOT NULL, outcome TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
"""
_CREATE_ATTEMPT_EVIDENCE_SQL = """
    CREATE TABLE attempt_evidence (
        attempt_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (attempt_id, evidence_id)
    )
"""
_CREATE_SKILLS_SQL = """
    CREATE TABLE skills (
        skill_id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL, verification_state TEXT NOT NULL,
        source_lesson_id TEXT NOT NULL, recorded_at TEXT NOT NULL
    )
"""
_CREATE_SKILL_STEPS_SQL = """
    CREATE TABLE skill_steps (
        skill_id TEXT NOT NULL, step TEXT NOT NULL, position INTEGER NOT NULL, PRIMARY KEY (skill_id, position)
    )
"""
_CREATE_SKILL_CONDITIONS_SQL = """
    CREATE TABLE skill_conditions (
        skill_id TEXT NOT NULL, condition TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, position)
    )
"""
_CREATE_SKILL_EVIDENCE_SQL = """
    CREATE TABLE skill_evidence (
        skill_id TEXT NOT NULL, evidence_id TEXT NOT NULL, position INTEGER NOT NULL,
        PRIMARY KEY (skill_id, evidence_id)
    )
"""


def _table_exists(connection, name):
    return (
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _build_standalone_v4_database(db_path, *, lesson_content):
    """Build a complete, hand-crafted v4-schema database from nothing --
    never touching `Urdyn`/`MemoryStore` -- with one pre-existing
    verified lesson. Mirrors `test_migration_v4.py`'s own style: the
    schema this module would have produced before A7 is trusted from
    its own SQL, not from current code, so opening it for the first
    time under A7 is a genuine "search index missing on an established
    store" recovery, not just fresh-database initialization.
    """
    recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
    memory_id = uuid.uuid4().hex
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(_CREATE_MEMORIES_V2_SQL)
            connection.execute(_CREATE_EVIDENCE_SQL)
            connection.execute(_CREATE_MEMORY_EVIDENCE_SQL)
            connection.execute(_CREATE_EVENTS_SQL)
            connection.execute(_CREATE_ATTEMPTS_SQL)
            connection.execute(_CREATE_ATTEMPT_EVIDENCE_SQL)
            connection.execute(_CREATE_SKILLS_SQL)
            connection.execute(_CREATE_SKILL_STEPS_SQL)
            connection.execute(_CREATE_SKILL_CONDITIONS_SQL)
            connection.execute(_CREATE_SKILL_EVIDENCE_SQL)
            connection.execute("PRAGMA user_version = 4")
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, 'lesson', 'verified', ?, NULL)",
                (memory_id, lesson_content, recorded_at),
            )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) "
                "VALUES (?, 'memory_recorded', ?, ?)",
                (uuid.uuid4().hex, memory_id, recorded_at),
            )
    finally:
        connection.close()
    return memory_id


def test_fresh_workspace_gets_a_search_index_created_and_backfilled(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Searchable canonical content.", kind="note")

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        assert store.fts_enabled is True
        candidates = store.search_candidates(frozenset({"searchable", "content"}), ENTITY_MEMORY)
    assert len(candidates) == 1


def test_search_index_survives_reopening_without_duplication(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Searchable canonical content.", kind="note")
    db_path = cx._db_path
    del cx

    Urdyn.open(tmp_path).recall("anything")  # forces a store open/close cycle
    Urdyn.open(tmp_path).recall("anything")  # and again

    connection = sqlite3.connect(db_path)
    try:
        (count,) = connection.execute(f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE}").fetchone()
    finally:
        connection.close()
    assert count == 1


@pytest.mark.parametrize("version", [1, 2, 3])
def test_a_cortex_store_without_index_is_refused_and_no_index_is_created(tmp_path, version):
    """Was `test_v4_workspace_search_index_is_backfilled_on_first_open`
    (a hand-built pre-A7 store got its index created and backfilled by the
    first open). Neither half survives the current contracts: legacy schemas
    are refused and no open creates an index. On the AUTHENTIC
    Cortex Brains that really have no index (v1-v3), the store layer, the open
    and the FTS maintenance entry all refuse, and no index object appears."""
    from urdyn import UrdynStorageSchemaError, maintenance

    root, _ = legacy.cortex_brain(tmp_path, version)
    db = legacy.db_path(root)
    assert SEARCH_INDEX_TABLE not in legacy.store_facts(db)["tables"]
    before = legacy.snapshot(root)

    with pytest.raises(UrdynStorageSchemaError):
        MemoryStore.open_existing(db)
    with pytest.raises(UrdynStorageSchemaError):
        Urdyn.open(root)
    with pytest.raises(UrdynStorageSchemaError):
        maintenance.recover_search_index(root)

    assert legacy.snapshot(root) == before
    assert not [t for t in legacy.store_facts(db)["tables"] if t.startswith(SEARCH_INDEX_TABLE)]


@pytest.mark.parametrize("release", legacy.RELEASES)
def test_a_released_brain_index_is_rebuilt_from_its_existing_data_only_on_request(tmp_path, release):
    """The part of the original that survives: recovery of the index
    from data that PREDATES it, not just fresh initialization -- on an AUTHENTIC
    Brain written by a released Urdyn. Opening and reading never recreate a
    removed index; the explicit rebuild backfills every pre-existing memory."""
    root, generated = legacy.release_brain(tmp_path, release)
    db = legacy.db_path(root)
    connection = sqlite3.connect(db)
    try:
        with connection:
            connection.execute(f"DROP TABLE {SEARCH_INDEX_TABLE}")
    finally:
        connection.close()

    cx = Urdyn.open(root)
    assert sorted(m.memory_id for m in cx.timeline()) == sorted(generated["written"])
    assert cx.search_index_state().status == "missing"
    assert SEARCH_INDEX_TABLE not in legacy.store_facts(db)["tables"]

    assert cx.rebuild_search_index().status == "usable"
    with MemoryStore.open_existing(db) as store:
        found = {entity_id for entity_id, _ in store.search_candidates(frozenset({"synthetic"}), ENTITY_MEMORY)}
    assert found == set(generated["written"])
    assert "brain_identity" not in legacy.store_facts(db)["tables"]  # still legacy-unbound


def test_rebuild_search_index_recovers_from_a_cleared_index(tmp_path):
    """`rebuild_search_index()` is the internal primitive A7 requires:
    dropping/clearing the derived index and rebuilding it must restore
    identical retrieval behavior, proving the index carries no
    information the canonical tables do not already have."""
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Searchable canonical content.", kind="note")

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        before = store.search_candidates(frozenset({"searchable"}), ENTITY_MEMORY)
        store._connection.execute(f"DELETE FROM {SEARCH_INDEX_TABLE}")
        store._connection.commit()
        emptied = store.search_candidates(frozenset({"searchable"}), ENTITY_MEMORY)

        store.rebuild_search_index()
        after = store.search_candidates(frozenset({"searchable"}), ENTITY_MEMORY)

    assert before != []
    assert emptied == []
    assert after == before


def test_fts5_unavailable_falls_back_to_lexical_channel_only(tmp_path, monkeypatch):
    """Simulates a SQLite build without the FTS5 module: no real build
    difference is available to test against, so `_try_create_search_index`
    (the one function that would see the module-missing
    `sqlite3.OperationalError`) is monkeypatched to report unavailability
    the same way it would in that case. The workspace must still open,
    and `preflight()`/`guard()` must still work correctly through the
    lexical channel that predates A7."""
    import urdyn._store as store_module

    monkeypatch.setattr(store_module, "_try_create_search_index", lambda connection: False)

    cx = Urdyn.init(tmp_path, "dev")
    cx.record_attempt(
        task="Fix database connection pool exhaustion under load",
        approach="Increased the pool size without addressing connection leaks",
        outcome="failed",
    )

    store = MemoryStore.open_if_exists(cx._db_path)
    with store:
        assert store.fts_enabled is False
        assert store.search_candidates(frozenset({"pool"}), ENTITY_MEMORY) == []

    # the lexical channel alone still finds a close paraphrase
    result = cx.preflight("Fix database connection pool exhaustion under load")
    assert len(result.known_failures) == 1

    # and a naturally diluted paraphrase -- the case FTS5 exists to
    # widen -- correctly stays a miss without FTS5, not a crash
    diluted = (
        "I was reviewing the deployment checklist and also wanted to check on how "
        "to fix that database connection pool exhaustion problem we keep running "
        "into under load, could you help me understand what has already been "
        "tried there"
    )
    diluted_result = cx.preflight(diluted)
    assert diluted_result.known_failures == ()


def test_a_genuine_schema_bug_in_the_create_statement_is_not_silently_swallowed(tmp_path, monkeypatch):
    """`_try_create_search_index` must only treat "no such module: fts5"
    (the genuine "this SQLite build lacks FTS5" condition) as expected
    and recoverable. `sqlite3.OperationalError` is also raised for a
    malformed `CREATE VIRTUAL TABLE` statement -- a schema-level bug,
    e.g. a duplicate column name -- and that must still propagate as a
    loud `UrdynStorageError`, not be silently reinterpreted as "this
    SQLite build has no FTS5" and leave the store open with
    `fts_enabled=False` and no visible sign anything went wrong.
    """
    import urdyn._store as store_module

    monkeypatch.setattr(
        store_module,
        "_CREATE_SEARCH_INDEX_SQL",
        f"CREATE VIRTUAL TABLE {SEARCH_INDEX_TABLE} "
        "USING fts5(entity_type UNINDEXED, entity_type UNINDEXED, text)",
    )

    with pytest.raises(UrdynStorageError):
        Urdyn.init(tmp_path, "dev").remember("Anything.", kind="note")


def test_canonical_write_rollback_leaves_no_orphaned_index_row(tmp_path):
    """A memory write that fails validation (unknown evidence reference)
    must roll back completely, including any search index row that
    would have accompanied it -- the canonical write and its index
    entry share one transaction (see `_store.py`'s `add()`), so there is
    no window where one exists without the other."""
    cx = Urdyn.init(tmp_path, "dev")
    fake_evidence = Evidence(
        evidence_id="0" * 32, content="fake", kind="user_statement", recorded_at=dt.datetime.now(dt.timezone.utc)
    )

    with pytest.raises(ValueError):
        cx.remember("Should never be persisted.", evidence=[fake_evidence])

    connection = sqlite3.connect(cx._db_path)
    try:
        (memory_count,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
        (index_count,) = connection.execute(
            f"SELECT COUNT(*) FROM {SEARCH_INDEX_TABLE} WHERE text = ?", ("Should never be persisted.",)
        ).fetchone()
    finally:
        connection.close()

    assert memory_count == 0
    assert index_count == 0


def test_search_index_is_not_part_of_the_canonical_schema_version(tmp_path):
    cx = Urdyn.init(tmp_path, "dev")
    cx.remember("Anything.", kind="note")

    connection = sqlite3.connect(cx._db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        assert _table_exists(connection, SEARCH_INDEX_TABLE)
    finally:
        connection.close()
    assert version == 7


def test_entity_type_isolates_a_deliberate_id_collision(tmp_path):
    """`memory_id`/`attempt_id`/`skill_id` are independent UUID4
    namespaces with no cross-type uniqueness guarantee at the database
    level. `search_candidates` must key on `(entity_type, entity_id)`
    in practice -- verified here by forcing a collision no real
    workspace would ever produce (UUID4 collision odds are negligible)
    to prove entity_type isolation holds structurally, not just by
    accident of always-distinct ids."""
    cx = Urdyn.init(tmp_path, "dev")
    same_id = "a" * 32
    store = MemoryStore.create_or_open(cx._db_path)
    with store:
        with store._connection:
            store._connection.execute(
                f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
                (ENTITY_MEMORY, same_id, "This is a memory about database backups."),
            )
            store._connection.execute(
                f"INSERT INTO {SEARCH_INDEX_TABLE} (entity_type, entity_id, text) VALUES (?, ?, ?)",
                ("attempt", same_id, "Fix login redirect bug in the auth flow."),
            )

        memory_hits = store.search_candidates(frozenset({"backups"}), ENTITY_MEMORY)
        cross_contamination = store.search_candidates(frozenset({"backups"}), "attempt")

    assert [eid for eid, _ in memory_hits] == [same_id]
    assert cross_contamination == []
