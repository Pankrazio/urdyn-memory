"""The explicit search-index rebuild validates BEFORE it commits.

Defect fixed: `recreate_search_index()` committed DROP/CREATE/populate first
and only then inspected the result in a separate transaction, so an index
that failed validation -- or a validation that failed, or a process that
died while validating -- was already published. Now creation, population,
FTS5 integrity check and the comparison with the canonical projection all
run inside the one `BEGIN IMMEDIATE` transaction; anything short of
`usable` rolls it back entirely.

Each test names the starting index state it must find intact afterwards.
Canonical data and the Brain identity are checked every time. Synthetic
Brains in temporary directories only.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

from urdyn import Urdyn, UrdynStorageError
from urdyn._store import SEARCH_INDEX_TABLE

STARTS = ("missing", "stale", "usable")


def _sql(db_path, statement, params=()):
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            return connection.execute(statement, params).fetchall()
    finally:
        connection.close()


def _brain(tmp_path, start):
    cx = Urdyn.init(tmp_path, "dev")
    for i in range(5):
        cx.remember(f"canonical fact {i}: the deploy needs a migration", kind="note")
    cx.record_attempt(task="run the migration", approach="direct", outcome="failed")
    if start == "missing":
        _sql(cx._db_path, f"DROP TABLE {SEARCH_INDEX_TABLE}")
    elif start == "stale":
        _sql(cx._db_path, f"UPDATE {SEARCH_INDEX_TABLE} SET text = 'stale text'")
    return cx


def _snapshot(cx):
    """Index objects + rows, canonical dump, identity row, manifest bytes."""
    connection = sqlite3.connect(cx._db_path)
    try:
        objects = sorted(connection.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name LIKE ?", (SEARCH_INDEX_TABLE + "%",)).fetchall())
        rows = (connection.execute(f"SELECT rowid, * FROM {SEARCH_INDEX_TABLE} ORDER BY rowid").fetchall()
                if any(name == SEARCH_INDEX_TABLE for _, name, _ in objects) else None)
        # iterdump() wraps virtual tables in PRAGMA writable_schema lines: not canonical either.
        canonical = [line for line in connection.iterdump()
                     if SEARCH_INDEX_TABLE not in line and not line.startswith("PRAGMA writable_schema")]
        identity = connection.execute("SELECT * FROM brain_identity").fetchall()
        (integrity,) = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    manifest = (cx.path / ".urdyn" / "manifest.json").read_bytes()
    return {"objects": objects, "rows": rows, "canonical": canonical, "identity": identity,
            "manifest": manifest, "integrity": integrity}


def _assert_unchanged(cx, before):
    after = _snapshot(cx)
    assert after == before
    assert after["integrity"] == "ok"
    assert len(after["identity"]) == 1


# --------------------------------------------------------------------------
# A build that does not validate is never published
# --------------------------------------------------------------------------


@pytest.mark.parametrize("start", STARTS)
def test_an_incomplete_build_is_rolled_back_not_committed(tmp_path, monkeypatch, start):
    import urdyn._store as store_module

    cx = _brain(tmp_path, start)
    before = _snapshot(cx)
    original = store_module._rebuild_search_index

    def incomplete(connection):
        original(connection)
        connection.execute(f"DELETE FROM {SEARCH_INDEX_TABLE} WHERE rowid = (SELECT MIN(rowid) FROM {SEARCH_INDEX_TABLE})")

    monkeypatch.setattr(store_module, "_rebuild_search_index", incomplete)
    with pytest.raises(UrdynStorageError):
        cx.rebuild_search_index()
    _assert_unchanged(cx, before)


@pytest.mark.parametrize("start", STARTS)
def test_a_failure_while_validating_rolls_the_build_back(tmp_path, monkeypatch, start):
    """The canonical projection is read twice: to populate, then to validate.
    A failure on the second read happens after the build, before commit."""
    import urdyn._store as store_module

    cx = _brain(tmp_path, start)
    before = _snapshot(cx)
    original = store_module._search_index_projection
    calls = []

    def failing_on_validation(connection):
        calls.append(1)
        if len(calls) >= 2:
            raise sqlite3.OperationalError("disk I/O error while validating")
        return original(connection)

    monkeypatch.setattr(store_module, "_search_index_projection", failing_on_validation)
    with pytest.raises(UrdynStorageError) as info:
        cx.rebuild_search_index()
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    assert len(calls) >= 2
    _assert_unchanged(cx, before)


@pytest.mark.parametrize("start", STARTS)
def test_a_failure_while_populating_rolls_the_build_back(tmp_path, monkeypatch, start):
    import urdyn._store as store_module

    cx = _brain(tmp_path, start)
    before = _snapshot(cx)
    original = store_module._search_index_projection

    def failing_midway(connection):
        for i, row in enumerate(original(connection)):
            if i == 3:
                raise sqlite3.OperationalError("disk I/O error while populating")
            yield row

    monkeypatch.setattr(store_module, "_search_index_projection", failing_midway)
    with pytest.raises(UrdynStorageError) as info:
        cx.rebuild_search_index()
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    _assert_unchanged(cx, before)


# --------------------------------------------------------------------------
# Process killed mid-rebuild: nothing published
# --------------------------------------------------------------------------

_CHILD = r"""
import os, sys
import urdyn._store as store_module
from urdyn import Urdyn
stage = sys.argv[2]
original = store_module._search_index_projection
calls = []
def dying(connection):
    calls.append(1)
    if stage == "populate":
        for i, row in enumerate(original(connection)):
            if i == 3:
                os._exit(17)
            yield row
    else:
        if len(calls) >= 2:   # first call populated, second validates
            os._exit(17)
        yield from original(connection)
store_module._search_index_projection = dying
Urdyn.open(sys.argv[1]).rebuild_search_index()
os._exit(0)
"""


@pytest.mark.parametrize("stage", ["populate", "validate"])
@pytest.mark.parametrize("start", STARTS)
def test_a_process_killed_mid_rebuild_publishes_nothing(tmp_path, start, stage):
    cx = _brain(tmp_path, start)
    before = _snapshot(cx)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), stage], env=env,
                           capture_output=True, text=True, timeout=60)
    assert child.returncode == 17, child.stderr
    # The hot journal left by the dead process is rolled back by SQLite on next access.
    _assert_unchanged(cx, before)
    expected_status = {"missing": "missing", "stale": "invalid", "usable": "usable"}[start]
    assert Urdyn.open(tmp_path).search_index_state().status == expected_status
    assert Urdyn.open(tmp_path).rebuild_search_index().status == "usable"


# --------------------------------------------------------------------------
# Success path: validated inside, committed once, identity and canonical intact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("start", STARTS)
def test_a_successful_rebuild_changes_only_the_index(tmp_path, start):
    cx = _brain(tmp_path, start)
    before = _snapshot(cx)
    assert cx.rebuild_search_index().status == "usable"
    after = _snapshot(cx)
    assert after["canonical"] == before["canonical"]
    assert after["identity"] == before["identity"]
    assert after["manifest"] == before["manifest"]
    assert after["integrity"] == "ok"
    assert cx.search_index_state().status == "usable"


def test_the_validator_runs_inside_the_open_rebuild_transaction(tmp_path, monkeypatch):
    """Structural pin of the fix: when validation runs, the build is still
    uncommitted -- another connection sees the PREVIOUS (missing) index."""
    import urdyn._store as store_module

    cx = _brain(tmp_path, "missing")
    seen = {}
    original = store_module._validate_search_index

    def spy(connection, *, runtime):
        seen["in_transaction"] = connection.in_transaction
        other = sqlite3.connect(f"file:{cx._db_path}?mode=ro", uri=True)
        try:
            seen["visible_elsewhere"] = other.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = ?", (SEARCH_INDEX_TABLE,)).fetchone()[0]
        finally:
            other.close()
        return original(connection, runtime=runtime)

    monkeypatch.setattr(store_module, "_validate_search_index", spy)
    assert cx.rebuild_search_index().status == "usable"
    assert seen == {"in_transaction": True, "visible_elsewhere": 0}
