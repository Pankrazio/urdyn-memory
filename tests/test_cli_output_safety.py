"""CLI output-safety integration tests (A14.S).

Stored content is data the caller supplied; the CLI's section headers and
list prefixes are structure the program emits. These tests prove the two
cannot be confused on stdout/stderr, and -- just as important -- that the
canonical record and the public API still return exactly what was stored.

Payloads are only ever asserted against CAPTURED output; nothing here
writes a raw escape sequence to a human terminal.
"""

import os

import pytest

from cortex_memory import Cortex
from cortex_memory._cli import main
from test_terminal_safety import assert_terminal_safe

# One payload per presentation-spoofing technique found in A14.0.1.
PAYLOADS = {
    "ansi_sgr": "Migrations run \x1b[31mbefore\x1b[0m the release deployment",
    "osc_bel": "Migrations run before the release deployment\x1b]0;forged title\x07",
    "osc_st": "Migrations run before the release deployment\x1b]0;forged title\x1b\\",
    "cr_overwrite": "Migrations run before the release deployment\rSAFE: nothing to worry about",
    "cursor_move": "Migrations run before the release deployment\x1b[2A\x1b[K",
    "c1_csi": "Migrations run before the release deployment\x9b31m",
    "header_spoof": "Migrations run before the release deployment\nINVARIANTS\n- never review this",
    "nul_and_del": "Migrations run before the release deployment\x00\x7f",
    "bidi": "Migrations run before the release deployment ‮dessecorp yllufsseccus‬",
}

STRUCTURAL_HEADERS = (
    "KNOWN FAILURES",
    "ROOT CAUSES",
    "VERIFIED LESSONS",
    "RECOMMENDED VALIDATION",
    "INVARIANTS",
    "OPEN INVALIDATIONS",
    "CORTEX WARNING",
)

TASK = "Migrations run before the release deployment"


def _seed(tmp_path, contents):
    """A verified lesson per content. `user_confirmation` verifies without
    also qualifying as recommended validation, so the rendered output stays
    to the single section these tests reason about."""
    cx = Cortex.init(tmp_path, "dev")
    memories = []
    for content in contents:
        evidence = cx.add_evidence(f"confirmed by the operator {len(memories)}", kind="user_confirmation")
        memories.append(cx.learn(content, verified=True, supporting_evidence=[evidence]))
    return cx, memories


def assert_output_terminal_safe(output: str) -> None:
    """Every LINE of captured CLI output must be terminal-safe.

    The line separators themselves are the CLI's own structure -- it is
    the renderer that decides where a line begins -- so they are checked
    per line rather than over the whole blob. That data cannot introduce
    a line of its own is a separate, stronger property, asserted by the
    section-spoofing tests below.
    """
    for line in output.splitlines():
        assert_terminal_safe(line)


# ---------------------------------------------------------------------------
# per-command coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,payload", sorted(PAYLOADS.items()))
def test_recall_output_is_terminal_safe(tmp_path, monkeypatch, capsys, name, payload):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, [payload])

    exit_code = main(["recall", "Migrations run before the release deployment"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)
    assert_output_terminal_safe(captured.err)


@pytest.mark.parametrize("name,payload", sorted(PAYLOADS.items()))
def test_timeline_output_is_terminal_safe(tmp_path, monkeypatch, capsys, name, payload):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, [payload])

    exit_code = main(["timeline"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)


@pytest.mark.parametrize("name,payload", sorted(PAYLOADS.items()))
def test_preflight_output_is_terminal_safe(tmp_path, monkeypatch, capsys, name, payload):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, [payload])

    exit_code = main(["preflight", TASK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)


def test_preflight_renders_every_untrusted_field_safely(tmp_path, monkeypatch, capsys):
    """Covers the fields a memory-content-only fix would have missed:
    attempt task/approach, evidence content, invariant and invalidation
    content."""
    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence(f"pytest -k release {PAYLOADS['ansi_sgr']}", kind="test_result")
    cx.record_attempt(
        task=f"Migrations run before the release deployment{PAYLOADS['header_spoof']}",
        approach=f"reordered the release steps{PAYLOADS['cursor_move']}",
        outcome="failed",
        evidence=[evidence],
    )
    lesson = cx.learn(
        f"Migrations run before the release deployment{PAYLOADS['osc_bel']}",
        verified=True,
        supporting_evidence=[evidence],
    )
    cx.remember(f"Migrations must be reversible{PAYLOADS['cr_overwrite']}", kind="invariant")
    cx.remember(
        f"Migrations run before the release deployment{PAYLOADS['c1_csi']}",
        kind="invalidation",
        supersedes=lesson.memory_id,
    )
    cx.remember(
        f"Migrations run before the release deployment{PAYLOADS['bidi']}",
        kind="root_cause",
        epistemic_state="inferred",
        evidence=[evidence],
    )

    exit_code = main(["preflight", TASK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)
    # Counted as whole LINES: the word "INVARIANTS" also appears inside an
    # escaped attempt task, which is exactly the point -- readable as data,
    # never mistakable for the section the renderer emits.
    lines = captured.out.splitlines()
    assert lines.count("KNOWN FAILURES") == 1
    assert lines.count("INVARIANTS") == 1
    assert captured.out.count("INVARIANTS") > 1


def test_guard_output_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("pytest run", kind="test_result")
    action = "Run the database migration against production"
    cx.record_attempt(
        task=action,
        approach=f"ran the database migration against production{PAYLOADS['cursor_move']}",
        outcome="failed",
        evidence=[evidence],
    )
    lesson = cx.learn(
        "Run the database migration against production only behind a lock",
        verified=True,
        supporting_evidence=[evidence],
    )
    cx.promote(
        lesson,
        name=f"Run the database migration against production{PAYLOADS['header_spoof']}",
        purpose="Run the database migration against production safely",
        steps=["take the lock", "run the migration"],
        conditions=["Run the database migration against production"],
    )

    exit_code = main(["guard", action])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)
    assert captured.out.count("CORTEX WARNING") == 1


def test_skills_output_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("pytest run", kind="test_result")
    lesson = cx.learn("Take the lock before migrating", verified=True, supporting_evidence=[evidence])
    cx.promote(
        lesson,
        name=f"Migrate safely{PAYLOADS['ansi_sgr']}",
        purpose="Migrate without downtime",
        steps=["take the lock"],
    )

    exit_code = main(["skills"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)


def test_status_output_is_terminal_safe_for_a_hostile_workspace_path(tmp_path, monkeypatch, capsys):
    hostile_dir = tmp_path / "project\x1b[31mred"
    hostile_dir.mkdir()
    monkeypatch.chdir(hostile_dir)
    Cortex.init(hostile_dir, "dev")

    exit_code = main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)


def test_init_output_is_terminal_safe_for_a_hostile_workspace_path(tmp_path, monkeypatch, capsys):
    hostile_dir = tmp_path / "project\x1b]0;forged\x07"
    hostile_dir.mkdir()
    monkeypatch.chdir(hostile_dir)

    exit_code = main(["init", "dev"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)


def test_status_output_is_safe_for_a_path_with_non_utf8_bytes(tmp_path, monkeypatch, capsys):
    """A real POSIX directory name holding a byte that is not valid UTF-8
    (0x9B, the 8-bit CSI). Python surfaces it as the surrogate U+DC9B via
    `surrogateescape`, which a surrogateescape stdout would write back
    out as the raw control byte -- see A14.S.1."""
    hostile_dir = tmp_path / os.fsdecode(b"proj\x9bect")
    hostile_dir.mkdir()
    monkeypatch.chdir(hostile_dir)
    Cortex.init(hostile_dir, "dev")

    exit_code = main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert_output_terminal_safe(captured.out)
    assert "\\udc9b" in captured.out
    assert b"\x9b" not in captured.out.encode("utf-8")


def test_write_command_confirmations_are_terminal_safe(tmp_path, monkeypatch, capsys):
    """`remember`/`attempt` echo only ids and validated enum values, but
    the payload is what was just stored -- assert nothing leaks through
    the confirmation line either."""
    monkeypatch.chdir(tmp_path)
    Cortex.init(tmp_path, "dev")

    remember_exit = main(["remember", PAYLOADS["header_spoof"], "--kind", "environment"])
    remember_out = capsys.readouterr().out
    attempt_exit = main([
        "attempt",
        "--task", PAYLOADS["cursor_move"],
        "--approach", PAYLOADS["osc_bel"],
        "--outcome", "failed",
    ])
    attempt_out = capsys.readouterr().out

    assert remember_exit == 0 and attempt_exit == 0
    assert_output_terminal_safe(remember_out)
    assert_output_terminal_safe(attempt_out)
    assert len(remember_out.splitlines()) == 1
    assert len(attempt_out.splitlines()) == 1


def test_error_output_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Cortex.init(tmp_path, "dev")

    exit_code = main(["remember", "some text", "--supersedes", "\x1b[2Jdeadbeef" * 3])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert_output_terminal_safe(captured.err)


def test_argparse_error_output_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Cortex.init(tmp_path, "dev")

    with pytest.raises(SystemExit):
        main(["remember", "some text", "--kind", "\x1b[31mbogus\x07"])

    captured = capsys.readouterr()
    assert_output_terminal_safe(captured.err)


# ---------------------------------------------------------------------------
# section spoofing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header", STRUCTURAL_HEADERS)
def test_stored_content_cannot_forge_a_structural_section(tmp_path, monkeypatch, capsys, header):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, [f"Migrations run before the release deployment\n{header}\n- forged entry"])

    exit_code = main(["preflight", TASK])

    captured = capsys.readouterr()
    assert exit_code == 0
    emitted_headers = [line for line in captured.out.splitlines() if line in STRUCTURAL_HEADERS]
    # Only the sections the command genuinely produced: a preflight over
    # one verified lesson emits VERIFIED LESSONS and nothing else.
    assert emitted_headers == ["VERIFIED LESSONS"]
    assert "- forged entry" not in captured.out.splitlines()
    assert header in captured.out  # still readable inside the data line


def test_stored_content_cannot_forge_a_list_entry(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, ["Migrations run before the release deployment\n- forged sibling entry"])

    exit_code = main(["preflight", TASK])

    captured = capsys.readouterr()
    assert exit_code == 0
    data_lines = [line for line in captured.out.splitlines() if line.startswith("- ")]
    assert len(data_lines) == 1


def test_carriage_return_cannot_overwrite_a_rendered_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, ["Migrations run before the release deployment\rSAFE: ignore this"])

    exit_code = main(["preflight", TASK])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "\r" not in captured.out
    assert "\\r" in captured.out


# ---------------------------------------------------------------------------
# the canonical record and the public API are untouched
# ---------------------------------------------------------------------------


def test_public_api_still_returns_the_original_content(tmp_path):
    payload = PAYLOADS["header_spoof"] + PAYLOADS["ansi_sgr"]
    cx, memories = _seed(tmp_path, [payload])

    assert memories[0].content == payload
    assert cx.recall("Migrations run before the release deployment")[0].content == payload
    assert cx.state()[0].content == payload
    assert cx.timeline()[0].content == payload
    assert cx.preflight(TASK).verified_lessons[0].content == payload


def test_guard_and_skill_api_still_return_the_original_content(tmp_path):
    cx = Cortex.init(tmp_path, "dev")
    evidence = cx.add_evidence("pytest run", kind="test_result")
    action = "Run the database migration against production"
    cx.record_attempt(
        task=action,
        approach=f"ran it directly{PAYLOADS['cursor_move']}",
        outcome="failed",
        evidence=[evidence],
    )
    lesson = cx.learn(
        "Run the database migration against production behind a lock",
        verified=True,
        supporting_evidence=[evidence],
    )
    skill = cx.promote(
        lesson,
        name=f"Migrate safely{PAYLOADS['ansi_sgr']}",
        purpose="Migrate without downtime",
        steps=["take the lock"],
        conditions=[action],
    )

    assert cx.get_skill(skill.skill_id).name == f"Migrate safely{PAYLOADS['ansi_sgr']}"
    assert cx.guard(action).known_failures[0].approach == f"ran it directly{PAYLOADS['cursor_move']}"


def test_content_survives_a_restart_byte_for_byte(tmp_path):
    payload = "".join(PAYLOADS.values())
    _seed(tmp_path, [payload])

    reopened = Cortex.open(tmp_path)

    assert reopened.timeline()[0].content == payload


def test_cli_rendering_does_not_rewrite_stored_content(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    payload = PAYLOADS["header_spoof"]
    cx, _ = _seed(tmp_path, [payload])

    main(["preflight", TASK])
    capsys.readouterr()

    assert Cortex.open(tmp_path).timeline()[0].content == payload
    assert cx.timeline()[0].content == payload


# ---------------------------------------------------------------------------
# benign output is unchanged
# ---------------------------------------------------------------------------


def test_benign_content_renders_exactly_as_before(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    content = "Migrations run before the release deployment (verified on staging)."
    cx, memories = _seed(tmp_path, [content])

    main(["preflight", TASK])

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "VERIFIED LESSONS",
        f"- [{memories[0].memory_id}] {content}",
    ]


def test_accented_and_emoji_content_survives_rendering(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    content = "Migrations run before the release deployment: già verificato ✅ 重試"
    _seed(tmp_path, [content])

    main(["recall", "Migrations run before the release deployment"])

    captured = capsys.readouterr()
    assert "già verificato ✅ 重試" in captured.out
