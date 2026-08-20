"""A38: `cortex export "<task>" [--for generic] [--budget N]`, the first
task-aware, portable export of Cortex.

`context()` already answers "what must an agent respect right now" under
a budget (A29.1); `export` does not re-derive that answer -- it reuses
`Cortex.context()` verbatim and swaps only the renderer, from
`CompiledContext.render()` (a local-terminal view, `Retrieval:` included)
to `CompiledContext.render_portable()` (a provider-independent payload:
`Task:` plus the same shared compiled body, no `Retrieval:`, no run-
specific metadata). These tests anchor that renderer boundary and the
CLI's stdout/stderr contract; they deliberately do not re-assert the
byte-exact `_render_item` forms A36 already freezes in
`test_a29_context_compiler.py`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cortex_memory import Cortex
from cortex_memory._cli import main
from test_a29_context_compiler import (
    _INVARIANT_RELEVANT,
    _TASK,
    _populate,
    _workspace,
)
from test_cli_output_safety import assert_output_terminal_safe


# ---------------------------------------------------------------------------
# Renderer boundary: render_portable() vs render()
# ---------------------------------------------------------------------------


def test_portable_payload_carries_task_and_omits_retrieval(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    compiled = cx.context(_TASK)
    portable = compiled.render_portable()

    assert portable.startswith(f"Task: {_TASK}\n")
    assert "Retrieval:" not in portable


def test_portable_body_matches_local_render_body(tmp_path):
    """The compiled body (sections, empty-context messages, footer) must
    be the SAME text in both renderers -- only the envelope around it
    (`Retrieval:` vs `Task:`) differs."""
    cx = _workspace(tmp_path)
    _populate(cx)

    compiled = cx.context(_TASK)
    local_body = compiled.render().split("\n\n", 1)[1]
    portable_body = compiled.render_portable().split("\n\n", 1)[1]

    assert local_body == portable_body


def test_render_unchanged_by_render_portable_existing(tmp_path):
    """Calling the new renderer must not perturb the old one (no shared
    mutable state, no caching side effect)."""
    cx = _workspace(tmp_path)
    _populate(cx)
    compiled = cx.context(_TASK)

    before = compiled.render()
    compiled.render_portable()
    after = compiled.render()

    assert before == after


def test_portable_payload_has_no_dynamic_metadata(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    portable = cx.context(_TASK).render_portable()

    for forbidden in ("Cortex ID", str(tmp_path), "Semantic:", "Export successful", "Done", "Progress"):
        assert forbidden not in portable


# ---------------------------------------------------------------------------
# Multi-section content: provenance and conflict markers survive the
# portable renderer unchanged
# ---------------------------------------------------------------------------


def test_portable_payload_covers_multiple_sections_and_provenance(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)

    portable = cx.context(_TASK, budget=100000).render_portable()

    assert "CONSTRAINTS" in portable
    assert "DECISIONS" in portable
    assert "HISTORY" in portable
    assert f"from attempt [{ids['absorbed_attempt'].attempt_id}]" in portable


def test_portable_payload_discloses_conflict_marker(tmp_path):
    cx = _workspace(tmp_path)
    ids = _populate(cx)
    cx.record_conflict(ids["lesson"], ids["decision"])

    portable = cx.context(_TASK, budget=100000).render_portable()

    assert f"CONFLICTS WITH [{ids['decision'].memory_id}]" in portable


# ---------------------------------------------------------------------------
# Empty context: unrelated vs. budget-excluded (A34 distinction preserved)
# ---------------------------------------------------------------------------


def test_unrelated_task_portable_payload_still_carries_task(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    portable = cx.context("Rotate the TLS certificate for the nginx reverse proxy", budget=100000).render_portable()

    assert portable.startswith("Task: Rotate the TLS certificate")
    assert "No compiled context for this task." in portable


def test_budget_excluded_portable_payload_distinct_from_unrelated(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(_INVARIANT_RELEVANT, kind="invariant")

    portable = cx.context(_TASK, budget=1).render_portable()

    assert "No compiled context for this task." not in portable
    assert "No compiled items fit within the budget." in portable


# ---------------------------------------------------------------------------
# Safety: Unicode and control characters
# ---------------------------------------------------------------------------


def test_portable_payload_is_terminal_safe_with_control_characters(tmp_path):
    cx = _workspace(tmp_path)
    cx.remember(f"{_INVARIANT_RELEVANT}\x1b[2Jinjected", kind="invariant")

    portable = cx.context(_TASK, budget=100000).render_portable()

    assert_output_terminal_safe(portable)


def test_portable_payload_preserves_italian_unicode_task(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)
    task = "Aggiungere gestione dei retry per i job in background falliti nella coda"

    portable = cx.context(task).render_portable()

    assert f"Task: {task}" in portable


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_portable_payload_is_byte_identical_across_calls(tmp_path):
    cx = _workspace(tmp_path)
    _populate(cx)

    first = cx.context(_TASK, budget=2000).render_portable()
    second = cx.context(_TASK, budget=2000).render_portable()

    assert first == second


# ---------------------------------------------------------------------------
# CLI: stdout purity, stderr separation, target/budget flags
# ---------------------------------------------------------------------------


def test_cli_export_default_target_matches_render_portable(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["export", _TASK])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == cx.context(_TASK).render_portable() + "\n"
    assert captured.err == ""
    assert "Retrieval:" not in captured.out


def test_cli_export_for_generic_explicit(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["export", _TASK, "--for", "generic"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("Task: ")


def test_cli_export_unsupported_target_rejected_by_argparse(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["export", _TASK, "--for", "claude-code"])

    assert excinfo.value.code == 2
    assert capsys.readouterr().out == ""


def test_cli_export_budget_propagates_to_compiled_context(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    main(["export", _TASK, "--budget", "100000"])
    large_out = capsys.readouterr().out

    main(["export", _TASK, "--budget", "80"])
    small_out = capsys.readouterr().out

    assert len(small_out) < len(large_out)


def test_cli_export_workspace_absent_rc1_stdout_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["export", _TASK])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err != ""


def test_cli_export_invalid_budget_rc1_consistent_with_context(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    export_exit = main(["export", _TASK, "--budget", "0"])
    export_err = capsys.readouterr().err
    context_exit = main(["context", _TASK, "--budget", "0"])
    context_err = capsys.readouterr().err

    assert export_exit == 1 == context_exit
    assert export_err == context_err


def test_cli_export_empty_export_rc0(tmp_path, monkeypatch, capsys):
    Cortex.init(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["export", "Do anything at all"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task: Do anything at all" in captured.out
    assert "No compiled context for this task." in captured.out


def test_cli_export_pipe_and_redirect_yield_clean_payload(tmp_path):
    cx = Cortex.init(tmp_path)
    cx.remember(_INVARIANT_RELEVANT, kind="invariant")

    script = f"from cortex_memory._cli import main; main(['export', {_TASK!r}])"
    piped = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert piped.stdout.startswith("Task: ")
    assert "Retrieval:" not in piped.stdout
    for noise in ("Export successful", "Done", "Progress", "Status:"):
        assert noise not in piped.stdout


# ---------------------------------------------------------------------------
# No canonical mutation
# ---------------------------------------------------------------------------


def test_cli_export_never_mutates_canonical_state(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    before_timeline = [(m.memory_id, m.content, m.supersedes) for m in cx.timeline()]
    before_count = cx._count_memories()

    main(["export", _TASK, "--budget", "80"])
    main(["export", _TASK, "--budget", "100000"])
    main(["export", "an entirely unrelated task about lemon cakes"])
    capsys.readouterr()

    after_timeline = [(m.memory_id, m.content, m.supersedes) for m in cx.timeline()]
    after_count = cx._count_memories()

    assert before_timeline == after_timeline
    assert before_count == after_count


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_cli_help_lists_export_command():
    script = "from cortex_memory._cli import main; main(['--help'])"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert "export" in result.stdout


# ---------------------------------------------------------------------------
# `cortex context` stays exactly as before (A36 golden tests own the byte
# contract; this only checks export did not perturb its sibling command)
# ---------------------------------------------------------------------------


def test_cortex_context_command_still_prints_retrieval(tmp_path, monkeypatch, capsys):
    cx = _workspace(tmp_path)
    _populate(cx)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["context", _TASK])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Retrieval:" in out
