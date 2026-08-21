"""Tests for the `urdyn` command-line interface."""

import pytest

from urdyn._cli import main


def test_cli_init_dev(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init", "dev"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dev" in captured.out
    assert (tmp_path / ".urdyn").is_dir()


def test_cli_status_reports_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "lab"])

    exit_code = main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "lab" in captured.out
    assert str(tmp_path) in captured.out


def test_cli_status_outside_workspace_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()


def test_cli_repeated_init_dev_is_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    first_exit = main(["init", "dev"])
    second_exit = main(["init", "dev"])

    assert first_exit == 0
    assert second_exit == 0


def test_cli_remember_reports_memory_id(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["remember", "SQLite was selected for the first storage implementation."])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Remembered" in captured.out


def test_cli_remember_rejects_empty_text(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["remember", "   "])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()


def test_cli_recall_finds_remembered_text(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "SQLite was selected for the first storage implementation."])
    main(["remember", "The public API must not expose raw SQL."])

    exit_code = main(["recall", "SQLite"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SQLite was selected" in captured.out
    assert "public API" not in captured.out


def test_cli_recall_second_query_finds_other_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "SQLite was selected for the first storage implementation."])
    main(["remember", "The public API must not expose raw SQL."])

    exit_code = main(["recall", "public API"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "public API" in captured.out


def test_cli_recall_with_no_matches_reports_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "something unrelated"])

    exit_code = main(["recall", "nonexistent-term-xyz"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No memories found" in captured.out


def test_cli_status_reports_memory_count(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "first"])
    main(["remember", "second"])

    exit_code = main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Memories: 2" in captured.out


def test_cli_operates_on_nearest_workspace_from_subdirectory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "SQLite was selected for the first storage implementation."])

    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    exit_code = main(["recall", "SQLite"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SQLite" in captured.out


def test_cli_memory_persists_across_separate_invocations(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", "SQLite was selected for the first storage implementation."])

    # simulate a brand-new process by calling `main` again with no shared state
    exit_code = main(["recall", "SQLite"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "SQLite" in captured.out


def test_cli_historical_workflow_end_to_end(tmp_path, monkeypatch, capsys):
    """The end-to-end scenario the A3 milestone must demonstrate: an old
    decision is superseded by a new one, both remain visible in the
    timeline, and recall surfaces only the current one by default."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    main(["remember", "PostgreSQL was selected.", "--kind", "decision"])
    capsys.readouterr()

    exit_code = main(["recall", "PostgreSQL"])
    old_id = capsys.readouterr().out.strip().split("[")[1].split("]")[0]
    assert exit_code == 0

    exit_code = main(
        ["remember", "SQLite was selected for V1.", "--kind", "decision", "--supersedes", old_id]
    )
    remember_out = capsys.readouterr().out
    assert exit_code == 0
    assert f"Supersedes [{old_id}]" in remember_out

    exit_code = main(["recall", "selected"])
    recall_out = capsys.readouterr().out
    assert exit_code == 0
    assert "SQLite was selected for V1." in recall_out
    assert "PostgreSQL was selected." not in recall_out

    exit_code = main(["timeline", "--kind", "decision"])
    timeline_out = capsys.readouterr().out
    assert exit_code == 0
    lines = timeline_out.strip().splitlines()
    assert len(lines) == 2
    assert "(superseded)" in lines[0] and "PostgreSQL was selected." in lines[0]
    assert "(current)" in lines[1] and "SQLite was selected for V1." in lines[1]


def test_cli_timeline_on_empty_workspace_reports_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["timeline"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No history found" in captured.out


def test_cli_remember_supersede_unknown_id_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["remember", "orphaned", "--kind", "decision", "--supersedes", "0" * 32])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "error" in captured.err.lower()


def test_cli_attempt_and_preflight_end_to_end(tmp_path, monkeypatch, capsys):
    """The A4 end-to-end workflow: record a failed attempt, then check
    that a fresh `preflight` invocation on a related task surfaces it."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(
        [
            "attempt",
            "--task",
            "Update authentication refresh logic.",
            "--approach",
            "Modify token refresh handling directly.",
            "--outcome",
            "failed",
        ]
    )
    attempt_out = capsys.readouterr().out
    assert exit_code == 0
    assert "Recorded attempt" in attempt_out
    assert "(failed)" in attempt_out

    exit_code = main(["preflight", "Modify authentication refresh logic"])
    preflight_out = capsys.readouterr().out
    assert exit_code == 0
    assert "KNOWN FAILURES" in preflight_out
    assert "Modify token refresh handling directly." in preflight_out


def test_cli_preflight_on_unrelated_task_reports_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(
        [
            "attempt",
            "--task",
            "Update authentication refresh logic.",
            "--approach",
            "Modify token refresh handling directly.",
            "--outcome",
            "failed",
        ]
    )
    capsys.readouterr()

    exit_code = main(["preflight", "Refactor CSS button styles"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No relevant experience found." in captured.out


def test_cli_attempt_rejects_unknown_outcome(tmp_path, monkeypatch, capsys):
    # argparse's own `choices=` validation raises SystemExit(2) directly,
    # the same as an invalid `--kind` would for `remember`/`recall`.
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    with pytest.raises(SystemExit) as exc_info:
        main(["attempt", "--task", "t", "--approach", "a", "--outcome", "not-an-outcome"])
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "invalid choice" in captured.err.lower()


def test_cli_skills_on_empty_workspace_reports_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["skills"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No skills recorded." in captured.out


def test_cli_guard_reports_no_warnings_on_empty_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["guard", "Modify refresh-token persistence logic"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No known Urdyn warnings for this action." in captured.out


def test_cli_guard_rejects_empty_action(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["guard", "   "])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "error" in captured.err.lower()


def test_cli_skills_and_guard_end_to_end(tmp_path, monkeypatch, capsys):
    """The A5 end-to-end workflow surfaced through the CLI: an experience
    is turned into a skill through the Python API (promotion stays
    Python-API-only), and `urdyn guard`/`urdyn skills` can see it from
    a fresh CLI invocation with no shared process state."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    from urdyn import Urdyn

    cx = Urdyn.discover()
    error_evidence = cx.add_evidence(
        "Refresh token was invalidated during rotation.", kind="error_observation"
    )
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Reuse the previous refresh token after rotation.",
        outcome="failed",
        evidence=[error_evidence],
    )
    validation = cx.add_evidence("Authentication tests passed.", kind="test_result")
    cx.record_attempt(
        task="Update authentication refresh logic.",
        approach="Persist and use only the newly issued refresh token.",
        outcome="succeeded",
        evidence=[validation],
    )
    lesson = cx.learn(
        "After token rotation, use only the newly issued refresh token.",
        evidence=[error_evidence],
        supporting_evidence=[validation],
        verified=True,
    )
    cx.promote(
        lesson,
        name="Safely modify refresh-token rotation",
        purpose="Modify token rotation without invalidating authentication.",
        steps=[
            "Inspect the refresh-token rotation flow.",
            "Persist only the newly issued refresh token.",
            "Do not reuse the previous token.",
            "Run authentication refresh tests.",
        ],
    )

    exit_code = main(["skills"])
    skills_out = capsys.readouterr().out
    assert exit_code == 0
    assert "(verified) Safely modify refresh-token rotation" in skills_out

    exit_code = main(["guard", "Modify refresh-token persistence logic"])
    guard_out = capsys.readouterr().out
    assert exit_code == 0
    assert "URDYN WARNING" in guard_out
    assert "Reuse the previous refresh token after rotation." in guard_out
    assert "Safely modify refresh-token rotation" in guard_out
    assert "Authentication tests passed." in guard_out

    exit_code = main(["guard", "Change CSS button color"])
    unrelated_out = capsys.readouterr().out
    assert exit_code == 0
    assert "No known Urdyn warnings for this action." in unrelated_out
