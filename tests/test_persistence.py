"""Restart-persistence and workspace-isolation tests.

These tests must exercise real persistence: a fresh `Cortex` object
opened via `discover()`/`open()`, never the in-memory instance that
performed the original `remember()`.
"""

from cortex_memory import Cortex


def test_memory_survives_reopening_the_workspace(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    original = cx.remember("SQLite was selected for the first storage implementation.")
    del cx

    reopened = Cortex.open(tmp_path)
    results = reopened.recall("SQLite")

    assert len(results) == 1
    assert results[0].memory_id == original.memory_id
    assert results[0].content == original.content


def test_memory_survives_discovery_from_fresh_instance(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("The public API must not expose raw SQL.")
    del cx

    rediscovered = Cortex.discover(tmp_path)
    results = rediscovered.recall("public API")

    assert len(results) == 1


def test_multiple_memories_survive_reopening(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("first memory")
    cx.remember("second memory")
    del cx

    reopened = Cortex.open(tmp_path)

    assert reopened._count_memories() == 2


def test_two_workspaces_do_not_share_memory(tmp_path):
    workspace_a = tmp_path / "a"
    workspace_b = tmp_path / "b"

    cx_a = Cortex.init(workspace_a, "dev")
    cx_b = Cortex.init(workspace_b, "dev")

    cx_a.remember("only in workspace A")
    cx_b.remember("only in workspace B")

    assert cx_a._count_memories() == 1
    assert cx_b._count_memories() == 1
    assert cx_a.recall("workspace B") == []
    assert cx_b.recall("workspace A") == []
