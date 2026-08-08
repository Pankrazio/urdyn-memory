"""CLI tests for A9.1: `cortex status` current-kind counts and the
`cortex preflight` INVARIANTS section.

`cortex status` must stay "git status of memory": real current counts,
no generated narrative, no health score, no suggestions.
"""

import pytest

from cortex_memory._cli import main


def test_cli_status_reports_zero_counts_on_empty_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Invariants: 0" in captured.out
    assert "Pending: 0" in captured.out
    assert "Open questions: 0" in captured.out
    assert "Environment facts: 0" in captured.out


def test_cli_status_counts_only_current_operational_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    # 2 invariants, one superseded -> 1 current
    main(["remember", "old invariant", "--kind", "invariant"])
    old_invariant_id = _last_remembered_id(capsys)
    main(["remember", "new invariant", "--kind", "invariant", "--supersedes", old_invariant_id])

    # 2 environment facts, one superseded -> 1 current
    main(["remember", "Python 3.12 is required.", "--kind", "environment"])
    old_env_id = _last_remembered_id(capsys)
    main(["remember", "Python 3.13 is required.", "--kind", "environment", "--supersedes", old_env_id])

    # 1 pending open, 1 pending closed -> 1 current
    main(["remember", "open pending item", "--kind", "pending"])
    main(["remember", "closed pending item", "--kind", "pending"])
    closed_pending_id = _last_remembered_id(capsys)
    main(["remember", "closed pending item done", "--kind", "note", "--supersedes", closed_pending_id])

    # 1 question open, 1 question resolved -> 1 current
    main(["remember", "open question", "--kind", "question"])
    main(["remember", "resolved question", "--kind", "question"])
    resolved_question_id = _last_remembered_id(capsys)
    main(["remember", "the answer", "--kind", "decision", "--supersedes", resolved_question_id])

    exit_code = main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Invariants: 1" in captured.out
    assert "Environment facts: 1" in captured.out
    assert "Pending: 1" in captured.out
    assert "Open questions: 1" in captured.out


def _last_remembered_id(capsys) -> str:
    """Extract the memory_id of the MOST RECENT CLI `remember` call, e.g.
    `Remembered [abc123...] (kind)`. Reads (and drains) whatever output
    has accumulated in `capsys` since it was last read -- which may
    include earlier, already-consumed `remember`/`init` calls -- so it
    always takes the LAST matching line, never the first."""
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("Remembered [")]
    return lines[-1].split("[", 1)[1].split("]", 1)[0]


def test_cli_preflight_shows_invariants_section_when_unrelated_to_task(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    main(["remember", ".cortex/ must remain gitignored.", "--kind", "invariant"])

    exit_code = main(["preflight", "Optimize database connection pooling."])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "INVARIANTS" in captured.out
    assert ".cortex/ must remain gitignored." in captured.out


def test_cli_preflight_omits_invariants_section_when_none_recorded(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    exit_code = main(["preflight", "some task nobody has attempted"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "INVARIANTS" not in captured.out


def test_cli_preflight_never_shows_pending_question_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    task = "Optimize database connection pooling for the storage layer."
    main(["remember", task, "--kind", "pending"])
    main(["remember", task, "--kind", "question"])
    main(["remember", task, "--kind", "environment"])

    exit_code = main(["preflight", task])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PENDING" not in captured.out
    assert "QUESTION" not in captured.out
    assert "ENVIRONMENT" not in captured.out


def test_cli_remember_kind_choices_include_new_operational_kinds(tmp_path, monkeypatch, capsys):
    """Confirms `--kind` accepts the new kinds without any extra CLI
    branching (they come automatically from `VALID_KINDS`, A9.1 §25)."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    for kind in ("pending", "question", "invariant", "environment"):
        exit_code = main(["remember", f"a {kind} fact", "--kind", kind])
        assert exit_code == 0

    captured = capsys.readouterr()
    assert captured.out.count("Remembered") == 4


def test_cli_remember_rejects_kind_outside_valid_kinds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    with pytest.raises(SystemExit):
        main(["remember", "something", "--kind", "blocked"])
