"""(A20) `urdyn remember --evidence EVIDENCE_ID`: the CLI half of the
Evidence -> Knowledge boundary.

A19.1 gave Urdyn the ability to observe a project document
(`Source` -> `SourceObservation` -> `document_observation` Evidence)
without believing a word of it. What it deliberately did NOT give was any
way, from the CLI, to say "I have read that observation and I now claim
this" -- `Urdyn.remember(evidence=[...])` had accepted provenance since
long before, but only a Python caller could reach it.

A20 closes exactly that gap and nothing else. The contract these tests
freeze is:

    Evidence supplies provenance. The caller supplies the claim.

so the flag attaches provenance and never anything more: it does not
designate support, does not change `epistemic_state`, does not read or
interpret the Evidence's content, and cannot make a memory `verified`
(there is no CLI path to `verified` at all -- see
`test_document_observation_evidence_does_not_produce_a_verified_memory`).

Seeding a file still creates no belief; `recall` still finds nothing
until someone explicitly states a claim. The tests below walk that whole
journey against the real CLI, then push on the two properties a future
watcher and Context Compiler will depend on: a memory's provenance is a
frozen `evidence_id`, not a live pointer to a file (re-seeding the
document does not rewrite it, and deleting the document does not break
it).
"""

from __future__ import annotations

import pytest

from urdyn import Urdyn
from urdyn._cli import main
from urdyn._evidence import EVIDENCE_KIND_DOCUMENT_OBSERVATION
from urdyn._memory import EPISTEMIC_USER_ASSERTED

ARCHITECTURE_DOC = """# Architecture

## Authentication

Access tokens expire after 15 minutes. Refresh tokens are rotated on use.
"""

CLAIM = "Authentication access tokens expire after 15 minutes."

# `recall` is deterministic lexical matching (a semantic channel exists but
# is opt-in and not installed by these tests), so the query below is one
# the claim's own words satisfy. `"token expiration"` -- the phrasing of
# the A20 design journey -- matches NO memory even after the claim is
# recorded, which is a property of retrieval, not of provenance: A20
# changes nothing about either -- see
# `test_seeding_alone_creates_no_memory_and_recall_stays_empty`, which
# asserts that phrasing explicitly.
RECALL_QUERY = "tokens expire"


def _seed_architecture_doc(workspace, text: str = ARCHITECTURE_DOC) -> str:
    """Write/overwrite `docs/architecture.md`, seed it through the CLI, and
    return the `evidence_id` of the observation that seed recorded.

    Reads the id back from the canonical record rather than from the seed
    output, so the tests below are not coupled to `urdyn seed`'s
    formatting -- what they care about is the id a user would copy out of
    `urdyn sources <path>`.
    """
    docs = workspace / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "architecture.md").write_text(text, encoding="utf-8")

    assert main(["seed", "docs/architecture.md"]) == 0

    (source,) = [s for s in Urdyn.open(workspace).sources() if s.path == "docs/architecture.md"]
    return source.latest_observation.evidence_id


def _memories(workspace):
    return Urdyn.open(workspace).timeline()


def test_remember_accepts_one_evidence_and_persists_it(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    exit_code = main(["remember", CLAIM, "--evidence", evidence_id])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Remembered" in captured.out
    assert evidence_id in captured.out

    (memory,) = _memories(tmp_path)
    assert memory.content == CLAIM
    # The exact id, not a re-resolution of the path: this is the whole
    # point of the flag.
    assert memory.evidence_ids == (evidence_id,)


def test_remember_accepts_repeated_evidence_in_order(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    cx = Urdyn.open(tmp_path)
    first = cx.add_evidence("pytest -q -> 1011 passed", kind="test_result")
    second = cx.add_evidence("The user confirmed the 15-minute expiry.", kind="user_confirmation")
    capsys.readouterr()

    exit_code = main(
        ["remember", CLAIM, "--evidence", first.evidence_id, "--evidence", second.evidence_id]
    )

    assert exit_code == 0
    (memory,) = _memories(tmp_path)
    # `remember()` preserves the caller's order; the CLI adds no ordering
    # of its own.
    assert memory.evidence_ids == (first.evidence_id, second.evidence_id)
    # Qualifying KINDS cited as generic provenance still do not verify
    # anything: only `supporting_evidence` can, and the CLI cannot pass it.
    assert memory.epistemic_state == EPISTEMIC_USER_ASSERTED
    assert memory.supporting_evidence_ids == ()


def test_repeated_identical_evidence_flag_is_deduplicated_by_the_core(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    exit_code = main(
        ["remember", CLAIM, "--evidence", evidence_id, "--evidence", evidence_id]
    )

    assert exit_code == 0
    (memory,) = _memories(tmp_path)
    # No CLI-invented semantics: `_remember`'s existing `dict.fromkeys`
    # normalization is what collapses the repeat.
    assert memory.evidence_ids == (evidence_id,)


def test_evidence_provenance_leaves_the_memory_user_asserted(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    main(["remember", CLAIM, "--evidence", evidence_id])

    (memory,) = _memories(tmp_path)
    assert memory.epistemic_state == EPISTEMIC_USER_ASSERTED
    assert memory.kind == "note"


def test_document_observation_evidence_does_not_produce_a_verified_memory(
    tmp_path, monkeypatch, capsys
):
    """Reading a document does not check that its claims are true.

    The seeded text below asserts something checkable; citing the
    observation of it must still produce a stated belief, never a verified
    one. The CLI cannot express `verified` at all -- there is no
    `--epistemic-state` and no `--supporting-evidence` -- so this holds by
    construction rather than by a gate the flag could argue with.
    """
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path, "# Support\n\nTests pass on Windows.\n")
    capsys.readouterr()

    exit_code = main(["remember", "Tests pass on Windows.", "--evidence", evidence_id])

    assert exit_code == 0
    cx = Urdyn.open(tmp_path)
    (memory,) = cx.timeline()
    assert memory.epistemic_state == EPISTEMIC_USER_ASSERTED
    assert memory.epistemic_state != "verified"
    assert memory.supporting_evidence_ids == ()
    assert cx.get_evidence(evidence_id).kind == EVIDENCE_KIND_DOCUMENT_OBSERVATION


def test_unknown_evidence_fails_and_records_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["remember", CLAIM, "--evidence", "0" * 32])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()
    assert _memories(tmp_path) == []


def test_malformed_evidence_id_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["remember", CLAIM, "--evidence", "../../etc/passwd"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()
    assert _memories(tmp_path) == []


def test_one_unknown_evidence_among_valid_ones_records_nothing(tmp_path, monkeypatch, capsys):
    """Resolution happens before the write, so a bad id in second position
    cannot leave a half-provenanced memory behind."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    exit_code = main(
        ["remember", CLAIM, "--evidence", evidence_id, "--evidence", "f" * 32]
    )

    assert exit_code == 1
    assert _memories(tmp_path) == []


def test_evidence_id_from_another_workspace_is_unknown_here(tmp_path, monkeypatch, capsys):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    main(["init", "dev"])
    foreign = Urdyn.open(other).add_evidence("Recorded somewhere else entirely.")

    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["remember", CLAIM, "--evidence", foreign.evidence_id])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()
    assert _memories(here) == []


def test_error_for_a_hostile_evidence_id_is_terminal_safe(tmp_path, monkeypatch, capsys):
    """The rejected id is echoed back inside the error message, and it came
    from argv -- it goes through the same rendering boundary as every other
    piece of caller data (A14.S)."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    hostile = "\x1b[2Jurdyn: Remembered\x07"
    exit_code = main(["remember", CLAIM, "--evidence", hostile])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "\x1b" not in captured.err
    assert "\x07" not in captured.err


def test_identical_retry_preserves_a17_idempotency(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    assert main(["remember", CLAIM, "--evidence", evidence_id]) == 0
    first = capsys.readouterr().out
    assert main(["remember", CLAIM, "--evidence", evidence_id]) == 0
    second = capsys.readouterr().out

    assert "Remembered" in first and "Already remembered" not in first
    assert "Already remembered" in second

    memories = _memories(tmp_path)
    assert len(memories) == 1
    assert memories[0].memory_id in first
    assert memories[0].memory_id in second


def test_same_claim_with_different_evidence_is_a_distinct_memory(tmp_path, monkeypatch, capsys):
    """(A17, unchanged by A20) `_find_current_equivalent` compares the
    provenance tuples too, so the same sentence cited from a different
    observation is a different canonical memory -- not a duplicate to
    collapse. A20 freezes this behaviour; it does not introduce or alter
    it."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_a = _seed_architecture_doc(tmp_path)
    evidence_b = _seed_architecture_doc(
        tmp_path, ARCHITECTURE_DOC.replace("15 minutes", "15 minutes exactly")
    )
    assert evidence_a != evidence_b
    capsys.readouterr()

    assert main(["remember", CLAIM, "--evidence", evidence_a]) == 0
    assert main(["remember", CLAIM, "--evidence", evidence_b]) == 0

    memories = _memories(tmp_path)
    assert len(memories) == 2
    assert {m.evidence_ids for m in memories} == {(evidence_a,), (evidence_b,)}
    assert {m.content for m in memories} == {CLAIM}


def test_seeding_alone_creates_no_memory_and_recall_stays_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    # Neither the words of the document nor a topical phrasing of them
    # reaches anything: an observation is not a belief, and it is not on
    # the retrieval surface at all.
    for query in (RECALL_QUERY, "token expiration", "Access tokens"):
        assert main(["recall", query]) == 0
        assert "No memories found." in capsys.readouterr().out
    assert _memories(tmp_path) == []


def test_source_to_evidence_to_memory_journey(tmp_path, monkeypatch, capsys):
    """The full A20 journey, driven only through the CLI a user types."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)

    # The id the user copies is the one `urdyn sources <path>` prints.
    capsys.readouterr()
    assert main(["sources", "docs/architecture.md"]) == 0
    assert evidence_id in capsys.readouterr().out

    # Before the explicit claim, the document is evidence, not knowledge.
    assert main(["recall", RECALL_QUERY]) == 0
    assert "No memories found." in capsys.readouterr().out

    assert main(["remember", CLAIM, "--evidence", evidence_id]) == 0
    capsys.readouterr()

    assert main(["recall", RECALL_QUERY]) == 0
    recalled = capsys.readouterr().out
    assert CLAIM in recalled

    (memory,) = _memories(tmp_path)
    assert memory.memory_id in recalled
    assert memory.evidence_ids == (evidence_id,)
    # Provenance resolves to the observation of the document, and the
    # observation still holds the text verbatim.
    provenance = Urdyn.open(tmp_path).get_evidence(evidence_id)
    assert provenance.kind == EVIDENCE_KIND_DOCUMENT_OBSERVATION
    assert "Access tokens expire after 15 minutes." in provenance.content


def test_later_seed_does_not_rewrite_existing_provenance(tmp_path, monkeypatch, capsys):
    """A future watcher may one day report "the source supporting this
    memory has changed". It may not silently make it true retroactively."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_a = _seed_architecture_doc(tmp_path)
    capsys.readouterr()
    assert main(["remember", CLAIM, "--evidence", evidence_a]) == 0
    (before,) = _memories(tmp_path)

    evidence_b = _seed_architecture_doc(
        tmp_path, ARCHITECTURE_DOC.replace("15 minutes", "30 minutes")
    )
    assert evidence_b != evidence_a

    (after,) = _memories(tmp_path)
    assert after.memory_id == before.memory_id
    assert after.evidence_ids == (evidence_a,)
    assert after.supersedes is None
    cx = Urdyn.open(tmp_path)
    # No automatic supersession, invalidation or conflict was invented.
    assert [m.memory_id for m in cx.state()] == [before.memory_id]
    assert cx.conflicts() == []
    assert cx.timeline(kind="invalidation") == []
    # The observation cited is still the OLD one, and still says what it
    # said when it was cited.
    assert "15 minutes" in cx.get_evidence(evidence_a).content
    assert "30 minutes" in cx.get_evidence(evidence_b).content


def test_deleting_the_document_does_not_break_memory_or_provenance(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()
    assert main(["remember", CLAIM, "--evidence", evidence_id]) == 0
    capsys.readouterr()

    (tmp_path / "docs" / "architecture.md").unlink()

    assert main(["recall", RECALL_QUERY]) == 0
    assert CLAIM in capsys.readouterr().out

    (memory,) = _memories(tmp_path)
    assert memory.evidence_ids == (evidence_id,)
    # The canonical Evidence holds the text; nothing re-opens the file.
    evidence = Urdyn.open(tmp_path).get_evidence(evidence_id)
    assert "Access tokens expire after 15 minutes." in evidence.content

    # And the historical observation is still citable for a NEW claim,
    # even though the document is gone.
    assert main(["remember", "The architecture doc described token rotation.",
                 "--evidence", evidence_id]) == 0
    assert len(_memories(tmp_path)) == 2


def test_remember_without_evidence_is_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    capsys.readouterr()

    exit_code = main(["remember", "SQLite was selected for the first storage implementation."])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Remembered" in captured.out
    # No provenance line is printed when there is no provenance.
    assert "Evidence [" not in captured.out

    (memory,) = _memories(tmp_path)
    assert memory.evidence_ids == ()
    assert memory.epistemic_state == EPISTEMIC_USER_ASSERTED


def test_evidence_composes_with_kind_and_supersedes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])
    evidence_id = _seed_architecture_doc(tmp_path)
    capsys.readouterr()

    assert main(["remember", "Tokens expire after 30 minutes.", "--kind", "environment"]) == 0
    (original,) = _memories(tmp_path)

    exit_code = main(
        [
            "remember",
            CLAIM,
            "--kind",
            "environment",
            "--supersedes",
            original.memory_id,
            "--evidence",
            evidence_id,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert original.memory_id in captured.out

    cx = Urdyn.open(tmp_path)
    (current,) = cx.state(kind="environment")
    assert current.content == CLAIM
    assert current.kind == "environment"
    assert current.supersedes == original.memory_id
    assert current.evidence_ids == (evidence_id,)


@pytest.mark.parametrize("flag", ["--source", "--supporting-evidence", "--epistemic-state"])
def test_a20_does_not_introduce_the_flags_it_deliberately_refused(
    tmp_path, monkeypatch, capsys, flag
):
    """Not a spelling check: these three are the exact ways the CLI could
    acquire authority it must not have -- resolving provenance from a live
    path, designating support, or naming an epistemic state. Each must be
    rejected by the parser, which is what keeps `verified` unreachable from
    the command line."""
    monkeypatch.chdir(tmp_path)
    main(["init", "dev"])

    with pytest.raises(SystemExit) as excinfo:
        main(["remember", CLAIM, flag, "whatever"])

    assert excinfo.value.code == 2
    assert _memories(tmp_path) == []
