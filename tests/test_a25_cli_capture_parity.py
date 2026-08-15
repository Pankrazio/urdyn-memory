"""(A25.1) `cortex evidence add` / `cortex learn`: the CLI half of the two
capture primitives A24's real-world validation found missing.

A20 closed `remember --evidence <ID>`: provenance only, and only a Python
caller could ever produce that id (`Cortex.add_evidence`) or reach
`verified` (`Cortex.learn(..., verified=True)`). A24 measured exactly six
Python-only capture operations across two real dev sessions -- four
`add_evidence` calls and two `learn(verified=True)` calls -- and those six
are precisely the ones that later produced the highest-value preflight
output (`VERIFIED LESSONS`, `RECOMMENDED VALIDATION`).

This file freezes the contract that closes that gap:

    cortex evidence add "<content>" [--kind KIND]
    cortex learn "<lesson>" [--evidence ID]... [--supporting-evidence ID]... [--verified]

Both are thin adapters over the existing public API
(`Cortex.add_evidence`, `Cortex.learn`): every kind check, every A12.1
verification-gate decision and all A17 idempotency behaviour is the
Core's, unchanged and untested-again here except to prove the CLI reaches
it correctly. No new domain rule is introduced by this module or by the
CLI code it exercises.
"""

from __future__ import annotations

import pytest

from cortex_memory import Cortex
from cortex_memory._cli import main
from cortex_memory._evidence import DEFAULT_EVIDENCE_KIND, EVIDENCE_ID_PATTERN
from cortex_memory._memory import EPISTEMIC_USER_ASSERTED, EPISTEMIC_VERIFIED
from test_terminal_safety import assert_terminal_safe


def _memories(workspace):
    return Cortex.open(workspace).timeline()


def _evidence_id_from_output(output: str) -> str:
    """Pull the bracketed id out of an `Evidence [<id>] (...)` line,
    mirroring how a shell script or an AI agent would extract it."""
    start = output.index("[") + 1
    end = output.index("]", start)
    return output[start:end]


# ---------------------------------------------------------------------------
# A/B/C/D/E -- `cortex evidence add`
# ---------------------------------------------------------------------------


def test_evidence_add_creates_evidence_and_prints_its_id(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["evidence", "add", "pytest -q -> 1 failed, 11 passed", "--kind", "test_result"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evidence [" in captured.out
    assert "(test_result)" in captured.out

    evidence_id = _evidence_id_from_output(captured.out)
    assert EVIDENCE_ID_PATTERN.fullmatch(evidence_id)

    evidence = Cortex.open(tmp_path).get_evidence(evidence_id)
    assert evidence.content == "pytest -q -> 1 failed, 11 passed"
    assert evidence.kind == "test_result"


def test_evidence_add_defaults_to_the_public_api_default_kind(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["evidence", "add", "I think this endpoint is idempotent"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"({DEFAULT_EVIDENCE_KIND})" in captured.out

    evidence_id = _evidence_id_from_output(captured.out)
    evidence = Cortex.open(tmp_path).get_evidence(evidence_id)
    assert evidence.kind == DEFAULT_EVIDENCE_KIND


def test_evidence_add_rejects_unknown_kind_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(["evidence", "add", "content", "--kind", "not-a-real-kind"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    # Specifically the --kind argument, not merely "evidence" being an
    # unrecognized subcommand -- this is what makes the assertion
    # non-vacuous against a baseline where `evidence` does not exist yet.
    assert "argument --kind" in captured.err
    assert "invalid choice" in captured.err.lower()

    # No store side effect: nothing was ever persisted for the rejected call.
    assert Cortex.open(tmp_path).sources() == []
    assert Cortex.open(tmp_path).timeline() == []


def test_evidence_add_cli_matches_python_api_semantics(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    main(["evidence", "add", "the same content", "--kind", "command_output"])
    captured = capsys.readouterr()
    cli_id = _evidence_id_from_output(captured.out)
    cli_evidence = Cortex.open(tmp_path).get_evidence(cli_id)

    api_evidence = Cortex.open(tmp_path).add_evidence("the same content", kind="command_output")

    assert cli_evidence.content == api_evidence.content == "the same content"
    assert cli_evidence.kind == api_evidence.kind == "command_output"
    # Evidence is append-only observation history, not canonically
    # deduplicated (unlike Memory/A17): two calls, two distinct ids.
    assert cli_evidence.evidence_id != api_evidence.evidence_id


def test_evidence_add_is_not_idempotent(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    main(["evidence", "add", "1 failed", "--kind", "error_observation"])
    main(["evidence", "add", "1 failed", "--kind", "error_observation"])
    captured = capsys.readouterr()

    ids = [_evidence_id_from_output(line) for line in captured.out.splitlines() if line.startswith("Evidence [")]
    assert len(ids) == 2
    assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# F/G/H/I/J/K/L/M -- `cortex learn`
# ---------------------------------------------------------------------------


def test_learn_without_verified_creates_a_candidate_lesson(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    failure = cx.add_evidence("1 failed: bool coerced to int", kind="error_observation")
    capsys.readouterr()

    exit_code = main(["learn", "Reject booleans before int checks", "--evidence", failure.evidence_id])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Learned" in captured.out
    assert "(candidate)" in captured.out

    (memory,) = _memories(tmp_path)
    assert memory.kind == "lesson"
    assert memory.epistemic_state == EPISTEMIC_USER_ASSERTED
    assert memory.evidence_ids == (failure.evidence_id,)
    assert memory.supporting_evidence_ids == ()


def test_learn_verified_with_qualifying_support_succeeds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    failure = cx.add_evidence("1 failed: bool coerced to int", kind="error_observation")
    passing = cx.add_evidence("12 passed", kind="test_result")
    capsys.readouterr()

    exit_code = main(
        [
            "learn",
            "Reject booleans before int checks",
            "--evidence",
            failure.evidence_id,
            "--supporting-evidence",
            passing.evidence_id,
            "--verified",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "(verified)" in captured.out

    (memory,) = _memories(tmp_path)
    assert memory.kind == "lesson"
    assert memory.epistemic_state == EPISTEMIC_VERIFIED
    # provenance/support distinction (M): the failure id is provenance
    # only, the passing id is both provenance (folded in) and support.
    assert memory.evidence_ids == (failure.evidence_id, passing.evidence_id)
    assert memory.supporting_evidence_ids == (passing.evidence_id,)


def test_learn_verified_with_non_qualifying_support_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    doc = cx.add_evidence("README says timeout defaults to 900", kind="document_observation")
    capsys.readouterr()

    exit_code = main(
        [
            "learn",
            "The default timeout is 900 seconds",
            "--supporting-evidence",
            doc.evidence_id,
            "--verified",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "verified" in captured.err.lower()
    # Fail closed: no lesson at all, not a silent downgrade to candidate.
    assert _memories(tmp_path) == []


def test_learn_verified_without_any_supporting_evidence_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["learn", "Some conclusion", "--verified"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert _memories(tmp_path) == []


def test_learn_unknown_evidence_id_fails_before_write(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    passing = cx.add_evidence("12 passed", kind="test_result")
    capsys.readouterr()

    exit_code = main(
        [
            "learn",
            "Some lesson",
            "--evidence",
            "deadbeefdeadbeefdeadbeefdeadbeef",
            "--supporting-evidence",
            passing.evidence_id,
            "--verified",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unknown evidence" in captured.err.lower()
    assert _memories(tmp_path) == []


def test_learn_unknown_supporting_evidence_id_fails_before_write(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    failure = cx.add_evidence("1 failed", kind="error_observation")
    capsys.readouterr()

    exit_code = main(
        [
            "learn",
            "Some lesson",
            "--evidence",
            failure.evidence_id,
            "--supporting-evidence",
            "deadbeefdeadbeefdeadbeefdeadbeef",
            "--verified",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unknown evidence" in captured.err.lower()
    assert _memories(tmp_path) == []


def test_learn_repeated_evidence_flags_preserve_order(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    first = cx.add_evidence("first observation", kind="user_statement")
    second = cx.add_evidence("second observation", kind="command_output")
    capsys.readouterr()

    main(
        [
            "learn",
            "Some lesson",
            "--evidence",
            first.evidence_id,
            "--evidence",
            second.evidence_id,
        ]
    )

    (memory,) = _memories(tmp_path)
    assert memory.evidence_ids == (first.evidence_id, second.evidence_id)


def test_learn_repeated_supporting_evidence_flags_preserve_order(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    first = cx.add_evidence("first check", kind="test_result")
    second = cx.add_evidence("second check", kind="command_output")
    capsys.readouterr()

    main(
        [
            "learn",
            "Some lesson",
            "--supporting-evidence",
            first.evidence_id,
            "--supporting-evidence",
            second.evidence_id,
            "--verified",
        ]
    )

    (memory,) = _memories(tmp_path)
    assert memory.supporting_evidence_ids == (first.evidence_id, second.evidence_id)


def test_learn_does_not_promote_generic_evidence_to_supporting(tmp_path, monkeypatch, capsys):
    """A qualifying-kind Evidence cited only as `--evidence` (not
    `--supporting-evidence`) must not silently verify anything -- the
    exact A20 boundary `remember --evidence` already enforces, preserved
    here for `learn`."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    passing = cx.add_evidence("12 passed", kind="test_result")
    capsys.readouterr()

    exit_code = main(["learn", "Some lesson", "--evidence", passing.evidence_id, "--verified"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert _memories(tmp_path) == []


# ---------------------------------------------------------------------------
# N/O -- preflight surfaces what CLI-only capture recorded
# ---------------------------------------------------------------------------


def test_verified_lesson_from_cli_appears_in_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Cortex.open(tmp_path)
    failure = cx.add_evidence("1 failed: bool coerced to int", kind="error_observation")
    passing = cx.add_evidence("12 passed", kind="test_result")
    capsys.readouterr()

    main(
        [
            "learn",
            "Reject booleans before int checks",
            "--evidence",
            failure.evidence_id,
            "--supporting-evidence",
            passing.evidence_id,
            "--verified",
        ]
    )
    capsys.readouterr()

    exit_code = main(["preflight", "Reject booleans before int checks in a new config option"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "VERIFIED LESSONS" in captured.out
    assert "Reject booleans before int checks" in captured.out
    assert "RECOMMENDED VALIDATION" in captured.out
    assert "12 passed" in captured.out
    # The non-qualifying provenance item must not appear as recommended
    # validation: only test_result/command_output do (§13 of the design).
    assert "1 failed" not in captured.out


# ---------------------------------------------------------------------------
# P -- terminal safety
# ---------------------------------------------------------------------------


def test_evidence_add_output_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    payload = "safe\nEvidence [forged]\x1b[31m (test_result)\x07"
    main(["evidence", "add", payload, "--kind", "user_statement"])
    captured = capsys.readouterr()

    # The CLI does not echo Evidence content by default, so the payload
    # itself never reaches stdout -- but the canonical record must still
    # keep it verbatim (sanitize on output, never on storage).
    for line in captured.out.splitlines():
        assert_terminal_safe(line)

    evidence_id = _evidence_id_from_output(captured.out)
    evidence = Cortex.open(tmp_path).get_evidence(evidence_id)
    assert evidence.content == payload


def test_learn_error_output_with_unsafe_content_is_terminal_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    payload = "unsafe\nCORTEX WARNING\x1b[2K"
    exit_code = main(["learn", payload, "--verified"])
    captured = capsys.readouterr()

    assert exit_code == 1
    for line in captured.err.splitlines():
        assert_terminal_safe(line)


# ---------------------------------------------------------------------------
# Q -- CLI-only A24 mini journey (acceptance anchor)
# ---------------------------------------------------------------------------


def test_cli_only_a24_mini_journey(tmp_path, monkeypatch, capsys):
    """Reproduces the high-value slice of A24's Session 1 -> Session 2 loop
    with ZERO Python capture calls: only `cortex evidence add`, `cortex
    learn` and `cortex preflight` on the argv boundary."""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "dev"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "evidence",
                "add",
                "pytest -q tests/test_config.py -> 1 failed: "
                "TypeError: bool coerced silently to int for MAX_RETRIES",
                "--kind",
                "error_observation",
            ]
        )
        == 0
    )
    failure_id = _evidence_id_from_output(capsys.readouterr().out)

    assert (
        main(
            [
                "evidence",
                "add",
                "pytest -q tests/test_config.py -> 12 passed",
                "--kind",
                "test_result",
            ]
        )
        == 0
    )
    support_id = _evidence_id_from_output(capsys.readouterr().out)

    assert (
        main(
            [
                "learn",
                "Reject booleans explicitly before any isinstance(value, int) check: "
                "bool is a subclass of int, so a boolean silently passes an "
                "integer type check.",
                "--evidence",
                failure_id,
                "--supporting-evidence",
                support_id,
                "--verified",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "preflight",
            "Reject booleans explicitly before any isinstance check when configuring a new setting",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "VERIFIED LESSONS" in captured.out
    assert "Reject booleans explicitly" in captured.out
    assert "RECOMMENDED VALIDATION" in captured.out
    assert "12 passed" in captured.out

    # And the negative half of the same acceptance anchor: a non-qualifying
    # verified request still fails closed, purely through CLI + Core.
    assert (
        main(
            [
                "evidence",
                "add",
                "README says integers default to 900",
                "--kind",
                "document_observation",
            ]
        )
        == 0
    )
    doc_id = _evidence_id_from_output(capsys.readouterr().out)
    exit_code = main(
        [
            "learn",
            "Default timeout is 900",
            "--supporting-evidence",
            doc_id,
            "--verified",
        ]
    )
    assert exit_code == 1
