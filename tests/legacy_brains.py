"""Shared helpers for tests on AUTHENTIC legacy Brains.

* Cortex development Brains, schemas v1-v6 (tests/data/legacy_cortex_brains):
  refused fail-closed, never migrated.
* Released Urdyn Brains 0.1.0-0.4.0, schema v7 without `brain_identity`
  (tests/data/released_brains): opened as legacy-unbound, never bound,
  converted or rewritten by an ordinary open.

Every archive is checked against its SHA256SUMS before use. The Brain files in
each archive are byte-identical to what the historical code wrote; `generate.json`
records what was written (see tests/data/fixture_tools/README.md).
Not a test module (no `test_` prefix); imported by the legacy-compatibility tests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
CORTEX_FIXTURES = DATA / "legacy_cortex_brains"
RELEASE_FIXTURES = DATA / "released_brains"
CORTEX_VERSIONS = (1, 2, 3, 4, 5, 6)
RELEASES = ("0.1.0", "0.2.0", "0.3.0", "0.4.0")
LOCK_DIRNAME = ".urdyn-locks"


def _verified(archive: Path) -> Path:
    sums = dict(line.split()[::-1] for line in (archive.parent / "SHA256SUMS").read_text().splitlines() if line.strip())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == sums[archive.name], f"{archive.name} is not authentic"
    return archive


def _extract(archive: Path, target: Path):
    target.mkdir(parents=True)
    with tarfile.open(_verified(archive)) as opened:
        opened.extractall(target, filter="data")
    return target / "brain", json.loads((target / "generate.json").read_text())


def cortex_brain(tmp_path: Path, version: int, *, layout: str = "urdyn"):
    """Authentic Cortex Brain of schema `version`. `layout="urdyn"` moves the
    same bytes under `.urdyn/` (what renaming the directory would give);
    `layout="cortex"` keeps them exactly as written."""
    root, generated = _extract(CORTEX_FIXTURES / f"v{version}.tar.gz", tmp_path / f"cortex-v{version}-{layout}")
    if layout == "urdyn":
        (root / ".cortex").rename(root / ".urdyn")
    return root, generated


def release_brain(tmp_path: Path, release: str):
    """Authentic Brain written by the released Urdyn `release`."""
    return _extract(RELEASE_FIXTURES / f"rel-{release}.tar.gz", tmp_path / f"release-{release}")


def release_experience_brain(tmp_path: Path, release: str):
    """Authentic Brain written by the released Urdyn `release` holding two
    notes, a user-asserted lesson and a failed attempt (ids in the second
    return value: `notes`, `lesson`, `attempt`)."""
    return _extract(RELEASE_FIXTURES / f"rel-{release}-experience.tar.gz", tmp_path / f"release-{release}-experience")


def db_path(root: Path, dirname: str = ".urdyn") -> Path:
    return root / dirname / "memory.db"


def snapshot(root: Path):
    """Every entry under the Brain root (names, types, bytes) and the names in
    its parent. The lifecycle lock directory every coordinated entry creates
    next to a Brain (derived, EMPTY files) is left out of the
    parent listing only after checking it holds nothing else."""
    entries = {}
    for path in sorted(root.rglob("*")):
        entries[path.relative_to(root).as_posix()] = path.read_bytes() if path.is_file() else "dir"
    locks = root.parent / LOCK_DIRNAME
    if locks.exists():
        for lock in locks.iterdir():
            assert lock.name == ".gitignore" or (lock.name.startswith(".urdyn-lock-") and lock.stat().st_size == 0)
    return entries, sorted(p.name for p in root.parent.iterdir() if p.name != LOCK_DIRNAME)


def store_facts(path: Path) -> dict:
    """Read-only view of a store: version, tables, columns of the tables that
    legacy steps change, memories as rows, integrity."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = sorted(r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        columns = {
            table: [r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')]
            for table in ("memories", "memory_evidence")
            if table in tables
        }
        return {
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "tables": tables,
            "columns": columns,
            "memories": sorted(connection.execute("SELECT * FROM memories").fetchall()),
            "integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
        }
    finally:
        connection.close()


def sabotage(path: Path, statement: str) -> None:
    """Apply one statement to a fixture copy (to make it malformed)."""
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(statement)
    finally:
        connection.close()
