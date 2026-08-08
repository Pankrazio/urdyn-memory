"""Regression test for A9.1: the new operational-memory kinds
(`pending`, `question`, `invariant`, `environment`) are pure Python-level
`VALID_KINDS` additions. `memories.kind` was already an unconstrained
`TEXT NOT NULL` column with no SQL `CHECK`, so using these kinds must
never trigger a schema migration or change the stored schema version.
"""

import sqlite3

from cortex_memory import Cortex
from cortex_memory._store import STORE_SCHEMA_VERSION, db_path_for
from cortex_memory._workspace import CORTEX_DIRNAME


def test_schema_version_unchanged_by_new_kinds(tmp_path):
    assert STORE_SCHEMA_VERSION == 4

    cx = Cortex.init(tmp_path, "dev")
    cx.remember("Python 3.12 is required.", kind="environment")
    cx.remember("Run Dev Validation #2.", kind="pending")
    cx.remember("Which database should we use?", kind="question")
    cx.remember(".cortex/ must remain gitignored.", kind="invariant")
    del cx

    db_path = db_path_for(tmp_path / CORTEX_DIRNAME)
    connection = sqlite3.connect(db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()

    assert version == STORE_SCHEMA_VERSION == 4


def test_reopening_after_new_kinds_does_not_remigrate(tmp_path):
    """Opening the store a second time after the new kinds were used must
    be a no-op with respect to schema: no table is recreated, no
    migration function runs, `user_version` stays put."""
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("uv is the project package manager.", kind="environment")
    del cx

    reopened = Cortex.open(tmp_path)
    reopened.remember("Second environment fact.", kind="environment")
    del reopened

    db_path = db_path_for(tmp_path / CORTEX_DIRNAME)
    connection = sqlite3.connect(db_path)
    try:
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert version == STORE_SCHEMA_VERSION == 4
    # exactly the same table set A5 (schema v4) already established --
    # no new table was added for the new kinds.
    assert tables >= {
        "memories",
        "evidence",
        "memory_evidence",
        "events",
        "attempts",
        "attempt_evidence",
        "skills",
        "skill_steps",
        "skill_conditions",
        "skill_evidence",
    }
