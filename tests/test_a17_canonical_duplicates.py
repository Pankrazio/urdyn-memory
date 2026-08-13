"""A17: exact canonical duplicate integrity.

Recording the same canonical memory twice must not leave two CURRENT
records of it. The defect this closes was found during A16's Human
Acceptance: the same `cortex remember` command run twice produced two
current memories with identical content, which semantic retrieval then
treated as two competing candidates -- identical scores, zero margin,
abstention -- suppressing a target that would otherwise have been
admitted (see `TestSemanticFalseAmbiguityRegression` at the bottom).

The scope is EXACT equivalence only. These tests are as much about what
must NOT collapse (a different kind, epistemic state, supersession,
provenance, or a single differing byte) as about what must.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata

import pytest

from cortex_memory import Cortex
from cortex_memory._cli import main
from test_cli_output_safety import assert_output_terminal_safe
from test_semantic import fake_semantic  # noqa: F401  (pytest fixture)

DUPLICATE_CONTENT = (
    "The service failed to start because the database configuration still "
    "pointed at a decommissioned endpoint."
)


def _events(cx, kind=None):
    """Read the append-only event log directly, bypassing every
    projection: `timeline()` alone could not distinguish "no second event
    was written" from "a second event was written but filtered out"."""
    connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
    try:
        if kind is None:
            rows = connection.execute("SELECT kind, subject_id FROM events ORDER BY sequence").fetchall()
        else:
            rows = connection.execute(
                "SELECT kind, subject_id FROM events WHERE kind = ? ORDER BY sequence", (kind,)
            ).fetchall()
    finally:
        connection.close()
    return rows


# ---------------------------------------------------------------------------
# Idempotency: the same canonical memory twice
# ---------------------------------------------------------------------------


class TestExactDuplicateIsIdempotent:
    def test_second_remember_returns_the_same_stable_id(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        second = cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert second.memory_id == first.memory_id

    def test_second_remember_returns_the_original_recording_time(self, tmp_path):
        """The duplicate call must not appear to re-date the memory: it is
        a retry, not a re-recording, exactly like `record_conflict`'s
        repeat declaration keeping its original `recorded_at`."""
        cx = Cortex.init(tmp_path, "dev")

        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        time.sleep(0.01)
        second = cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert second.recorded_at == first.recorded_at
        assert second == first

    def test_no_second_canonical_record_is_written(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert cx._count_memories() == 1
        assert len(cx.timeline()) == 1
        assert len(cx.state()) == 1

    def test_no_second_event_is_appended(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        memory = cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert _events(cx) == [("memory_recorded", memory.memory_id)]

    def test_no_second_search_index_row_is_written(self, tmp_path):
        """The derived FTS index is written inside the same transaction as
        the canonical row. A duplicate that wrote no memory must write no
        index row either, or the index would drift ahead of canonical
        data -- the precise inconsistency `_index_entity` exists to
        prevent."""
        cx = Cortex.init(tmp_path, "dev")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
        try:
            (rows,) = connection.execute(
                "SELECT COUNT(*) FROM search_index WHERE entity_type = 'memory'"
            ).fetchone()
        finally:
            connection.close()
        assert rows == 1

    def test_many_repeated_duplicates_stay_safe(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        ids = {cx.remember(DUPLICATE_CONTENT, kind="root_cause").memory_id for _ in range(10)}

        assert len(ids) == 1
        assert cx._count_memories() == 1

    def test_duplicate_is_recognized_after_reopening_the_workspace(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        reopened = Cortex.open(tmp_path)
        second = reopened.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert second.memory_id == first.memory_id
        assert reopened._count_memories() == 1

    def test_duplicate_is_recognized_after_discovery_from_a_subdirectory(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        nested = tmp_path / "src" / "deep"
        nested.mkdir(parents=True)

        discovered = Cortex.discover(nested)
        second = discovered.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert second.memory_id == first.memory_id

    def test_duplicate_is_recognized_from_a_separate_process(self, tmp_path):
        """Proof that recognition lives in the persisted store, not in any
        per-process or per-object cache: the second write happens in a
        fresh interpreter that has never seen the first one."""
        cx = Cortex.init(tmp_path, "dev")
        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        script = (
            "from cortex_memory import Cortex;"
            f"cx = Cortex.open({str(tmp_path)!r});"
            f"print(cx.remember({DUPLICATE_CONTENT!r}, kind='root_cause').memory_id)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )

        assert completed.stdout.strip() == first.memory_id
        assert cx._count_memories() == 1

    def test_learn_inherits_the_same_idempotency(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        first = cx.learn("Retries must reuse the original idempotency token.")
        second = cx.learn("Retries must reuse the original idempotency token.")

        assert second.memory_id == first.memory_id
        assert cx._count_memories() == 1


# ---------------------------------------------------------------------------
# What must NOT collapse: different canonical meaning
# ---------------------------------------------------------------------------


class TestDistinctCanonicalMemoriesAreNotCollapsed:
    def test_same_content_different_kind(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "Retries must reuse the original token"

        a = cx.remember(text, kind="root_cause")
        b = cx.remember(text, kind="lesson")

        assert a.memory_id != b.memory_id
        assert cx._count_memories() == 2

    def test_same_content_different_epistemic_state(self, tmp_path):
        """An epistemic upgrade is a change of authority, not a retry --
        and similarity must never be what confers authority."""
        cx = Cortex.init(tmp_path, "dev")
        text = "The retry storm was caused by an unbounded backoff."

        asserted = cx.remember(text, kind="root_cause", epistemic_state="user_asserted")
        inferred = cx.remember(text, kind="root_cause", epistemic_state="inferred")

        assert asserted.memory_id != inferred.memory_id
        assert {m.epistemic_state for m in cx.state()} == {"user_asserted", "inferred"}

    def test_same_content_verified_does_not_collapse_onto_user_asserted(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "The connection pool leak is fixed."
        asserted = cx.remember(text, kind="lesson")
        proof = cx.add_evidence("suite green", kind="test_result")

        verified = cx.remember(
            text, kind="lesson", epistemic_state="verified", supporting_evidence=[proof]
        )

        assert verified.memory_id != asserted.memory_id
        assert verified.epistemic_state == "verified"

    def test_same_content_different_supersedes_target(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        old_a = cx.remember("Python 3.11 is required.", kind="environment")
        old_b = cx.remember("Node 18 is required.", kind="environment")
        text = "The toolchain baseline was raised."

        first = cx.remember(text, kind="environment", supersedes=old_a.memory_id)
        second = cx.remember(text, kind="environment", supersedes=old_b.memory_id)

        assert first.memory_id != second.memory_id
        assert cx._count_memories() == 4
        # both supersession relations survive intact
        current = {m.memory_id for m in cx.state()}
        assert current == {first.memory_id, second.memory_id}

    def test_supersession_history_is_not_rewritten_by_a_later_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        old = cx.remember("Python 3.11 is required.", kind="environment")
        new = cx.remember("Python 3.12 is required.", kind="environment", supersedes=old.memory_id)

        again = cx.remember("Python 3.12 is required.", kind="environment", supersedes=old.memory_id)

        assert again.memory_id == new.memory_id
        assert [m.memory_id for m in cx.timeline()] == [old.memory_id, new.memory_id]
        assert [m.memory_id for m in cx.state()] == [new.memory_id]
        assert _events(cx, "memory_superseded") == [("memory_superseded", old.memory_id)]

    def test_a_conflicting_supersession_of_the_same_target_is_still_rejected(self, tmp_path):
        """Idempotency must absorb only an identical retry. A DIFFERENT
        memory superseding an already-superseded target is still the
        error it always was."""
        cx = Cortex.init(tmp_path, "dev")
        old = cx.remember("PostgreSQL was selected.", kind="decision")
        cx.remember("SQLite was selected for V1.", kind="decision", supersedes=old.memory_id)

        with pytest.raises(ValueError):
            cx.remember("MySQL was selected for V1.", kind="decision", supersedes=old.memory_id)

    def test_same_content_different_evidence(self, tmp_path):
        """New Evidence for the same conclusion is not a retry: Cortex has
        no evidence-merging concept, so collapsing here would silently
        DISCARD provenance the caller supplied."""
        cx = Cortex.init(tmp_path, "dev")
        text = "Database migration requires a backup."
        ev_a = cx.add_evidence("runbook section 4", kind="file_reference")
        ev_b = cx.add_evidence("incident 2026-03 postmortem", kind="file_reference")

        first = cx.remember(text, kind="lesson", evidence=[ev_a])
        second = cx.remember(text, kind="lesson", evidence=[ev_b])

        assert first.memory_id != second.memory_id
        assert first.evidence_ids == (ev_a.evidence_id,)
        assert second.evidence_ids == (ev_b.evidence_id,)

    def test_same_content_additional_evidence_is_not_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "Database migration requires a backup."
        ev_a = cx.add_evidence("runbook section 4", kind="file_reference")
        ev_b = cx.add_evidence("incident 2026-03 postmortem", kind="file_reference")

        first = cx.remember(text, kind="lesson", evidence=[ev_a])
        second = cx.remember(text, kind="lesson", evidence=[ev_a, ev_b])

        assert first.memory_id != second.memory_id

    def test_same_evidence_promoted_to_supporting_is_not_a_duplicate(self, tmp_path):
        """Same content, same Evidence ids, but one call explicitly
        designates that Evidence as SUPPORTING this memory. That
        designation is a canonical assertion (A12.1), not packaging."""
        cx = Cortex.init(tmp_path, "dev")
        text = "The nightly job now completes within the window."
        ev = cx.add_evidence("timing run", kind="command_output")

        related_only = cx.remember(text, kind="lesson", evidence=[ev])
        supporting = cx.remember(text, kind="lesson", supporting_evidence=[ev])

        assert related_only.memory_id != supporting.memory_id
        assert related_only.supporting_evidence_ids == ()
        assert supporting.supporting_evidence_ids == (ev.evidence_id,)

    def test_same_evidence_in_a_different_order_is_not_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "Two independent sources agree on the timeout value."
        ev_a = cx.add_evidence("source A", kind="file_reference")
        ev_b = cx.add_evidence("source B", kind="file_reference")

        first = cx.remember(text, kind="lesson", evidence=[ev_a, ev_b])
        second = cx.remember(text, kind="lesson", evidence=[ev_b, ev_a])

        assert first.memory_id != second.memory_id

    def test_paraphrase_is_not_a_duplicate(self, tmp_path):
        """A17 is exact-equivalence only: no fuzzy matching, no semantic
        deduplication. Two differently-worded memories coexist until some
        future, explicit mechanism says otherwise."""
        cx = Cortex.init(tmp_path, "dev")

        cx.remember("Retries must reuse the same token.", kind="lesson")
        cx.remember("Retry operations should preserve the original idempotency token.", kind="lesson")

        assert cx._count_memories() == 2

    def test_reasserting_a_superseded_memory_records_a_new_memory(self, tmp_path):
        """Only CURRENT equivalents collapse. Re-asserting something that
        was superseded is a genuine new claim about now, and gets its own
        record -- without ever resurrecting or rewriting the old one."""
        cx = Cortex.init(tmp_path, "dev")
        original = cx.remember("The staging queue runs a single consumer.", kind="environment")
        cx.remember("The staging queue runs three consumers.", kind="environment", supersedes=original.memory_id)

        again = cx.remember("The staging queue runs a single consumer.", kind="environment")

        assert again.memory_id != original.memory_id
        assert cx._count_memories() == 3
        assert original.memory_id not in {m.memory_id for m in cx.state()}
        # and the re-asserted memory is itself now the duplicate target
        assert cx.remember("The staging queue runs a single consumer.", kind="environment").memory_id == (
            again.memory_id
        )


# ---------------------------------------------------------------------------
# Exact text: edge cases (measured, never assumed equivalent)
# ---------------------------------------------------------------------------


class TestExactTextEdgeCases:
    """Cortex records `content` verbatim and normalizes nothing, anywhere.
    Duplicate detection inherits exactly that: these pairs differ by at
    least one byte, so they are two memories -- documented behaviour, not
    an oversight."""

    @pytest.mark.parametrize(
        "first,second",
        [
            ("hello", "hello "),  # trailing whitespace
            ("hello", "Hello"),  # case
            ("hello world", "hello  world"),  # internal whitespace
            ("line one\nline two", "line one line two"),  # newline vs space
            ("perché il servizio", "perche il servizio"),  # Italian accent
            (
                unicodedata.normalize("NFC", "città sicura"),
                unicodedata.normalize("NFD", "città sicura"),
            ),  # Unicode normalization forms
        ],
    )
    def test_byte_different_content_stays_distinct(self, tmp_path, first, second):
        cx = Cortex.init(tmp_path, "dev")

        a = cx.remember(first, kind="note")
        b = cx.remember(second, kind="note")

        assert a.memory_id != b.memory_id
        assert cx._count_memories() == 2

    def test_identical_unicode_content_is_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "Perché la migrazione è fallita: l'endpoint è già dismesso — 日本語 🎯"

        first = cx.remember(text, kind="root_cause")
        second = cx.remember(text, kind="root_cause")

        assert second.memory_id == first.memory_id
        assert cx.state()[0].content == text

    def test_identical_multiline_content_is_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "step one\nstep two\n\tindented\n"

        first = cx.remember(text, kind="note")
        second = cx.remember(text, kind="note")

        assert second.memory_id == first.memory_id

    def test_identical_very_long_content_is_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "the deployment pipeline stalls on the shared lock " * 2000

        first = cx.remember(text, kind="note")
        second = cx.remember(text, kind="note")

        assert second.memory_id == first.memory_id
        assert cx._count_memories() == 1

    def test_long_contents_differing_only_at_the_very_end_stay_distinct(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        base = "the deployment pipeline stalls on the shared lock " * 2000

        a = cx.remember(base + "A", kind="note")
        b = cx.remember(base + "B", kind="note")

        assert a.memory_id != b.memory_id

    def test_empty_content_is_still_rejected_not_deduplicated(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        with pytest.raises(ValueError):
            cx.remember("")
        with pytest.raises(ValueError):
            cx.remember("   \n\t ")

    def test_identical_control_character_content_is_a_duplicate(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        text = "Migrations run \x1b[31mbefore\x1b[0m the release\x00\x7f"

        first = cx.remember(text, kind="note")
        second = cx.remember(text, kind="note")

        assert second.memory_id == first.memory_id
        # stored verbatim: the duplicate path never rewrites content
        assert cx.state()[0].content == text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_second_remember_reports_already_remembered(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        main(["init", "dev"])
        capsys.readouterr()

        assert main(["remember", "--kind", "root_cause", DUPLICATE_CONTENT]) == 0
        first_out = capsys.readouterr().out
        assert main(["remember", "--kind", "root_cause", DUPLICATE_CONTENT]) == 0
        second_out = capsys.readouterr().out

        assert first_out.startswith("Remembered [")
        assert second_out.startswith("Already remembered [")
        memory_id = first_out.split("[")[1].split("]")[0]
        assert memory_id in second_out
        assert "(root_cause)" in second_out

    def test_status_and_timeline_stay_consistent_after_a_duplicate(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        main(["init", "dev"])
        main(["remember", "--kind", "root_cause", DUPLICATE_CONTENT])
        main(["remember", "--kind", "root_cause", DUPLICATE_CONTENT])
        capsys.readouterr()

        main(["status"])
        status_out = capsys.readouterr().out
        main(["timeline"])
        timeline_out = capsys.readouterr().out

        assert "Memories: 1" in status_out
        assert len(timeline_out.strip().splitlines()) == 1
        assert "(current)" in timeline_out

    def test_duplicate_output_is_terminal_safe_for_hostile_content(self, tmp_path, monkeypatch, capsys):
        """The duplicate branch prints ids and kinds Cortex validates, not
        caller text -- but the caller's text is what reached it, so the
        rendering boundary is asserted here too (A14.S)."""
        monkeypatch.chdir(tmp_path)
        main(["init", "dev"])
        hostile = "release is \x1b[31mblocked\x1b[0m\rSAFE: nothing to worry about"

        main(["remember", hostile])
        main(["remember", hostile])
        out = capsys.readouterr().out

        assert_output_terminal_safe(out)
        assert "Already remembered [" in out
        # the payload never reaches the duplicate line at all
        assert "SAFE: nothing to worry about" not in out


# ---------------------------------------------------------------------------
# Concurrency and atomicity
# ---------------------------------------------------------------------------


def test_concurrent_identical_remembers_produce_one_memory(tmp_path):
    """Check-then-insert would race if the duplicate lookup ran outside
    the write transaction: both callers would find nothing and both would
    insert. `add()` opens with `BEGIN IMMEDIATE` precisely so the second
    caller cannot read the store between the first one's check and its
    write. Threads here, but nothing about the mechanism is thread-local:
    each `remember()` opens its own connection, exactly as separate
    processes do.

    The store is materialized first, deliberately: concurrent FIRST-EVER
    creation of the schema file itself is a separate, PRE-EXISTING race
    in `_ensure_schema` (reproducible with `add_evidence` alone, with no
    memory write anywhere in sight), untouched by A17 and out of its
    scope. Seeding here keeps this test about the duplicate check rather
    than about that."""
    cx = Cortex.init(tmp_path, "dev")
    cx.remember("an unrelated pre-existing fact", kind="note")
    barrier = threading.Barrier(4)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        try:
            barrier.wait(timeout=10)
            memory = Cortex.open(tmp_path).remember(DUPLICATE_CONTENT, kind="root_cause")
            with lock:
                results.append(memory.memory_id)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(set(results)) == 1
    assert cx._count_memories() == 2  # the seed plus exactly one duplicate target
    assert len(_events(cx)) == 2


def test_a_rejected_duplicate_leaves_no_partial_state(tmp_path):
    """A recognized duplicate must be all-or-nothing in the other
    direction too: nothing written anywhere, not a memory without an
    event or an event without a memory."""
    cx = Cortex.init(tmp_path, "dev")
    ev = cx.add_evidence("suite green", kind="test_result")
    first = cx.remember(
        "The pool leak is fixed.", kind="lesson", epistemic_state="verified", supporting_evidence=[ev]
    )

    second = cx.remember(
        "The pool leak is fixed.", kind="lesson", epistemic_state="verified", supporting_evidence=[ev]
    )

    assert second == first
    connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
    try:
        (memories,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
        (events,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
        (links,) = connection.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()
    finally:
        connection.close()
    assert (memories, events, links) == (1, 1, 1)


def test_duplicate_lookup_does_not_scan_linearly_in_python(tmp_path):
    """Trend check, not a benchmark: a duplicate write against a store
    holding ~1k memories must not cost dramatically more than one against
    an almost empty store. A Python-side scan of every memory (each with
    its evidence links materialized) would not survive this bound."""
    small = Cortex.init(tmp_path / "small", "dev")
    small.remember(DUPLICATE_CONTENT, kind="root_cause")

    big = Cortex.init(tmp_path / "big", "dev")
    big.remember(DUPLICATE_CONTENT, kind="root_cause")
    for i in range(1000):
        big.remember(f"unrelated operational fact number {i}", kind="environment")

    start = time.perf_counter()
    small.remember(DUPLICATE_CONTENT, kind="root_cause")
    small_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    big.remember(DUPLICATE_CONTENT, kind="root_cause")
    big_elapsed = time.perf_counter() - start

    assert big._count_memories() == 1001
    assert big_elapsed < max(small_elapsed * 20, 0.25)


# ---------------------------------------------------------------------------
# The A16 Human Acceptance regression
# ---------------------------------------------------------------------------


class TestSemanticFalseAmbiguityRegression:
    """The defect exactly as it was hit by hand: two identical current
    memories score identically, the margin between them collapses to
    zero, and the semantic channel abstains -- suppressing a memory that
    is admitted the moment the canonical duplicate is gone.

    Fixed UPSTREAM, in canonical integrity: no semantic threshold, policy
    or ranking rule is touched here. The semantic channel simply receives
    correct canonical state.
    """

    def test_duplicate_no_longer_creates_a_false_ambiguity(self, tmp_path, fake_semantic):  # noqa: F811
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("alpha topic explained in the original wording", kind="root_cause")
        cx.remember("alpha topic explained in the original wording", kind="root_cause")
        cx.semantic_setup()

        result = cx.preflight("a completely different phrasing that happens to be about alpha")

        assert len(cx.state()) == 1
        assert len(result.root_causes) == 1

    def test_the_abstention_this_reproduces_is_real(self, tmp_path, fake_semantic):  # noqa: F811
        """Guards the test above against becoming vacuous: TWO genuinely
        distinct current memories that embed identically still collapse
        the margin and still cause abstention. That is the semantic
        policy working as designed -- the A17 fix is that identical
        CONTENT no longer produces that situation, not that the policy
        changed."""
        cx = Cortex.init(tmp_path, "dev")
        # same "alpha" concept for the fake encoder, different canonical text
        cx.remember("alpha topic explained in the original wording", kind="root_cause")
        cx.remember("alpha topic explained in slightly other wording", kind="root_cause")
        cx.semantic_setup()

        result = cx.preflight("a completely different phrasing that happens to be about alpha")

        assert len(cx.state()) == 2
        assert result.root_causes == ()

    def test_semantic_index_receives_one_vector_per_canonical_memory(self, tmp_path, fake_semantic):  # noqa: F811
        cx = Cortex.init(tmp_path, "dev")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        report = cx.semantic_setup()

        assert report.memory_count == 1


# ---------------------------------------------------------------------------
# A17.1: the duplicate lookup must never suppress write-boundary validation
# ---------------------------------------------------------------------------


class TestDuplicateLookupNeverBypassesWriteBoundaryValidation:
    """A17.R found that `MemoryStore.add()` ran the duplicate lookup
    BEFORE the A12.1.1 write-boundary check: a `verified` memory with no
    supporting Evidence -- rejected on every write since A12.1 -- was
    silently ACCEPTED (as "it's a duplicate, return the existing one")
    whenever a store already held a grandfathered legacy row (a
    `verified` memory persisted before A12.1 introduced the rule, which
    legitimately has an empty `supporting_evidence_ids`) with matching
    content/kind/epistemic_state/supersedes.

    Historical validity of an old row must never be confused with
    current write admissibility of a new request that merely resembles
    it. These tests inject exactly that kind of legacy row directly (the
    same technique `test_evidence_support.py`'s A12.1.1 write-boundary
    tests already use to bypass `Cortex.remember()`'s own gate and hit
    `MemoryStore.add()` head-on), then prove a matching request is still
    rejected by the SAME validation the baseline (pre-A17) codebase
    applies unconditionally."""

    @staticmethod
    def _inject_legacy_verified_memory(cx, *, content, kind="lesson", supporting_evidence_id=None):
        """Directly write a `memories` row shaped exactly like a pre-A12.1
        verified memory: `epistemic_state='verified'`,
        `supporting_evidence_ids=()` by default. `Cortex.remember()`
        itself refuses to construct this shape (that is the gate A12.1
        added) -- this bypasses it on purpose, the same way A12.1.1's own
        tests do, to simulate data that already existed before that gate
        existed.

        `supporting_evidence_id`, if given, links that (already-persisted)
        Evidence with `role='supporting'` -- for simulating the OTHER
        pre-A12.1-incompatible shape: a legacy row whose supporting
        Evidence exists but is of a non-qualifying kind (A12.1 only
        requires non-empty + qualifying kind going forward; a pre-A12.1
        row could have neither property enforced)."""
        import uuid

        from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED
        from cortex_memory._store import MemoryStore

        with MemoryStore.create_or_open(cx._db_path):
            pass  # materialize memory.db (and its schema) before hand-inserting

        legacy_id = uuid.uuid4().hex
        recorded_at = dt.datetime.now(dt.timezone.utc)
        connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
        try:
            connection.execute(
                "INSERT INTO memories (memory_id, content, kind, epistemic_state, recorded_at, supersedes) "
                "VALUES (?, ?, ?, 'verified', ?, NULL)",
                (legacy_id, content, kind, recorded_at.isoformat()),
            )
            if supporting_evidence_id is not None:
                connection.execute(
                    "INSERT INTO memory_evidence (memory_id, evidence_id, position, role) "
                    "VALUES (?, ?, 0, 'supporting')",
                    (legacy_id, supporting_evidence_id),
                )
            connection.execute(
                "INSERT INTO events (event_id, kind, subject_id, occurred_at) VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, EVENT_KIND_MEMORY_RECORDED, legacy_id, recorded_at.isoformat()),
            )
            connection.commit()
        finally:
            connection.close()
        return legacy_id, recorded_at

    def test_the_a17r_blocker_case_stays_rejected(self, tmp_path):
        """The exact reproduction from the A17.R report: a request that
        would fail on the baseline must still fail, even though a
        matching legacy row already exists and would otherwise be
        recognized as a current equivalent."""
        from cortex_memory._store import MemoryStore

        cx = Cortex.init(tmp_path, "dev")
        legacy_id, legacy_recorded_at = self._inject_legacy_verified_memory(
            cx, content="legacy verified claim", kind="lesson"
        )
        connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
        try:
            (memories_before,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
            (events_before,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
            (fts_before,) = connection.execute(
                "SELECT COUNT(*) FROM search_index WHERE entity_type = 'memory'"
            ).fetchone()
        finally:
            connection.close()

        from cortex_memory._memory import Memory
        from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED, Event

        forged = Memory(
            memory_id="f" * 32,
            content="legacy verified claim",
            kind="lesson",
            epistemic_state="verified",
            recorded_at=dt.datetime.now(dt.timezone.utc),
            evidence_ids=(),
            supporting_evidence_ids=(),
        )
        event = Event(
            event_id="e" * 32,
            kind=EVENT_KIND_MEMORY_RECORDED,
            subject_id=forged.memory_id,
            occurred_at=forged.recorded_at,
        )

        with pytest.raises(ValueError, match="supporting Evidence"):
            with MemoryStore.create_or_open(cx._db_path) as store:
                store.add(forged, [event])

        # zero side effects: the rejected write must not have touched
        # anything, and the legacy row itself must be exactly as it was
        connection = sqlite3.connect(cx.path / ".cortex" / "memory.db")
        try:
            (memories_after,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
            (events_after,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
            (fts_after,) = connection.execute(
                "SELECT COUNT(*) FROM search_index WHERE entity_type = 'memory'"
            ).fetchone()
            legacy_row = connection.execute(
                "SELECT content, kind, epistemic_state, recorded_at, supersedes FROM memories WHERE memory_id = ?",
                (legacy_id,),
            ).fetchone()
        finally:
            connection.close()

        assert memories_after == memories_before
        assert events_after == events_before
        assert fts_after == fts_before
        assert legacy_row == (
            "legacy verified claim",
            "lesson",
            "verified",
            legacy_recorded_at.isoformat(),
            None,
        )

    def test_verified_write_via_remember_stays_rejected_despite_a_legacy_lookalike(self, tmp_path):
        """Same property, exercised through the public API rather than
        `MemoryStore.add()` directly: `Cortex.remember()`'s own gate
        already rejects this (see `test_a12_1...` in
        `test_evidence_support.py`), but this proves the store-level fix
        did not accidentally make the outcome DEPEND on which gate runs
        first -- both must reject it, for the same reason."""
        cx = Cortex.init(tmp_path, "dev")
        self._inject_legacy_verified_memory(cx, content="another legacy claim", kind="lesson")

        with pytest.raises(ValueError, match="supporting Evidence"):
            cx.remember("another legacy claim", kind="lesson", epistemic_state="verified")

        assert cx._count_memories() == 1  # only the injected legacy row

    def test_non_qualifying_supporting_evidence_stays_rejected_despite_a_legacy_lookalike(self, tmp_path):
        """The second half of the A12.1.1 write boundary -- qualifying
        Evidence KIND, not just non-empty `supporting_evidence_ids` --
        must be equally immune. Without this, `_evidence_kind` would also
        be reachable with an unvalidated evidence_id if the ordering fix
        were only half-applied (see the docstring note on `add()`)."""
        from cortex_memory._store import MemoryStore
        from cortex_memory._memory import Memory
        from cortex_memory._event import EVENT_KIND_MEMORY_RECORDED, Event

        cx = Cortex.init(tmp_path, "dev")
        opinion = cx.add_evidence("I think this works.", kind="user_statement")
        # the legacy row is a genuine equivalence target: same content/kind/
        # epistemic_state AND same evidence_ids/supporting_evidence_ids as
        # `forged` below (both reference `opinion` as supporting) -- otherwise
        # `_find_current_equivalent` would never match them and this test
        # would pass regardless of validation order (a bug caught while
        # writing it: an EARLIER version left the legacy row's
        # `supporting_evidence_ids` empty, which made `forged` NOT a
        # duplicate of it, so the qualifying-kind check fired for reasons
        # unrelated to the fix being tested).
        legacy_id, _ = self._inject_legacy_verified_memory(
            cx, content="weakly supported claim", kind="lesson", supporting_evidence_id=opinion.evidence_id
        )

        forged = Memory(
            memory_id="a" * 32,
            content="weakly supported claim",
            kind="lesson",
            epistemic_state="verified",
            recorded_at=dt.datetime.now(dt.timezone.utc),
            evidence_ids=(opinion.evidence_id,),
            supporting_evidence_ids=(opinion.evidence_id,),
        )
        event = Event(
            event_id="b" * 32,
            kind=EVENT_KIND_MEMORY_RECORDED,
            subject_id=forged.memory_id,
            occurred_at=forged.recorded_at,
        )

        with pytest.raises(ValueError, match="qualifying kind"):
            with MemoryStore.create_or_open(cx._db_path) as store:
                store.add(forged, [event])

        assert cx._count_memories() == 1  # only the legacy row, id unchanged
        assert cx.state(kind="lesson")[0].memory_id == legacy_id


class TestValidDuplicateStillIdempotentAfterTheFix:
    """The control test the fix must not break: a VALID current memory,
    retried exactly, must still be idempotent. If A17.1 had "fixed" the
    blocker by simply disabling duplicate detection, this would fail."""

    def test_valid_verified_memory_retry_is_idempotent(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")
        proof = cx.add_evidence("suite green", kind="test_result")

        first = cx.remember(
            "The pool leak is fixed.", kind="lesson", epistemic_state="verified", supporting_evidence=[proof]
        )
        second = cx.remember(
            "The pool leak is fixed.", kind="lesson", epistemic_state="verified", supporting_evidence=[proof]
        )

        assert second == first
        assert cx._count_memories() == 1
        assert len(_events(cx)) == 1

    def test_valid_plain_memory_retry_is_still_idempotent(self, tmp_path):
        cx = Cortex.init(tmp_path, "dev")

        first = cx.remember(DUPLICATE_CONTENT, kind="root_cause")
        second = cx.remember(DUPLICATE_CONTENT, kind="root_cause")

        assert second.memory_id == first.memory_id
        assert second.recorded_at == first.recorded_at
        assert cx._count_memories() == 1

    def test_supersession_retry_is_still_idempotent_and_does_not_false_positive_has_superseder(self, tmp_path):
        """Direct guard against the fix regressing the exact property it
        had to preserve: `has_superseder` must stay evaluated AFTER the
        duplicate lookup, or this legitimate idempotent retry would be
        misreported as 'already superseded' (see the A17.1 docstring note
        on `MemoryStore.add`)."""
        cx = Cortex.init(tmp_path, "dev")
        old = cx.remember("Python 3.11 is required.", kind="environment")
        new = cx.remember("Python 3.12 is required.", kind="environment", supersedes=old.memory_id)

        again = cx.remember("Python 3.12 is required.", kind="environment", supersedes=old.memory_id)

        assert again.memory_id == new.memory_id
        assert _events(cx, "memory_superseded") == [("memory_superseded", old.memory_id)]

    def test_concurrent_valid_duplicate_retry_remains_atomically_idempotent(self, tmp_path):
        """Focused re-confirmation of A17.R's process-level concurrency
        proof, after the validation-order fix: `BEGIN IMMEDIATE` still
        wraps validation + duplicate lookup + insert as one unit, so
        reordering what happens INSIDE the transaction must not have
        reopened the check-then-insert race between processes."""
        cx = Cortex.init(tmp_path, "dev")
        cx.remember("concurrency seed", kind="note")
        barrier = threading.Barrier(4)
        results: list[str] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker():
            try:
                barrier.wait(timeout=10)
                memory = Cortex.open(tmp_path).remember(DUPLICATE_CONTENT, kind="root_cause")
                with lock:
                    results.append(memory.memory_id)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors, errors
        assert len(set(results)) == 1
        assert cx._count_memories() == 2  # seed + exactly one duplicate target
