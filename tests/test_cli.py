"""Tests for the `cortex` command-line interface."""

from cortex_memory._cli import main


def test_cli_init_dev(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["init", "dev"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dev" in captured.out
    assert (tmp_path / ".cortex").is_dir()


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
