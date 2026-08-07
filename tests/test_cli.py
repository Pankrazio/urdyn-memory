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
