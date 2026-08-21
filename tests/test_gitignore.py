"""Tests for .gitignore safety around the .urdyn/ directory."""

from urdyn import Urdyn


def test_creates_gitignore_when_missing(tmp_path):
    Urdyn.init(tmp_path, "dev")

    gitignore = tmp_path / ".gitignore"
    assert gitignore.exists()
    assert ".urdyn/" in gitignore.read_text().splitlines()


def test_appends_to_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\nnode_modules/\n")

    Urdyn.init(tmp_path, "dev")

    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines[:2] == ["*.log", "node_modules/"]
    assert ".urdyn/" in lines


def test_does_not_duplicate_existing_entry(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n.urdyn/\n")

    Urdyn.init(tmp_path, "dev")

    content = (tmp_path / ".gitignore").read_text()
    assert content.count(".urdyn/") == 1


def test_handles_missing_trailing_newline(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log")

    Urdyn.init(tmp_path, "dev")

    content = (tmp_path / ".gitignore").read_text()
    lines = content.splitlines()
    assert lines == ["*.log", ".urdyn/"]
    assert "*.log.urdyn/" not in content


def test_handles_empty_gitignore_file(tmp_path):
    (tmp_path / ".gitignore").write_text("")

    Urdyn.init(tmp_path, "dev")

    assert (tmp_path / ".gitignore").read_text() == ".urdyn/\n"


def test_repeated_init_does_not_modify_gitignore_again(tmp_path):
    Urdyn.init(tmp_path, "dev")
    first_content = (tmp_path / ".gitignore").read_text()

    Urdyn.init(tmp_path, "dev")
    second_content = (tmp_path / ".gitignore").read_text()

    assert first_content == second_content
