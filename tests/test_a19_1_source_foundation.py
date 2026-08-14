"""A19.1 Source foundation & dev seed.

Cortex gains a `Source` primitive: the stable identity of a project file
it has observed, plus the append-only history of those observations. Each
observation records one `document_observation` Evidence holding the
document's text VERBATIM, alongside a SHA-256 digest, a size and a
timestamp kept as structured columns.

The invariant these tests exist to protect is the three-level separation:

    Source (identity) != Evidence (observation) != Memory (knowledge)

Seeding a file must never produce a Memory, and must never make one
`verified`.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import sqlite3
import uuid

import pytest

from cortex_memory import Cortex, CortexSourceError, CortexStorageError
from cortex_memory._cli import main as cli_main
from cortex_memory._evidence import (
    EVIDENCE_KIND_DOCUMENT_OBSERVATION,
    RECOMMENDED_VALIDATION_EVIDENCE_KINDS,
    VALID_EVIDENCE_KINDS,
    VERIFICATION_EVIDENCE_KINDS,
)
from cortex_memory._source import (
    MAX_SEED_FILE_BYTES,
    SEED_ADDED,
    SEED_CHANGED,
    SEED_UNCHANGED,
    compute_digest,
)
from cortex_memory._store import STORE_SCHEMA_VERSION, MemoryStore, db_path_for


def _dev_workspace(tmp_path, **files):
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return Cortex.init(tmp_path, "dev")


def _raw_counts(cx):
    with sqlite3.connect(cx._db_path) as connection:
        (sources,) = connection.execute("SELECT COUNT(*) FROM sources").fetchone()
        (observations,) = connection.execute("SELECT COUNT(*) FROM source_observations").fetchone()
        (evidence,) = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()
        (memories,) = connection.execute("SELECT COUNT(*) FROM memories").fetchone()
        (events,) = connection.execute("SELECT COUNT(*) FROM events").fetchone()
    return {
        "sources": sources,
        "observations": observations,
        "evidence": evidence,
        "memories": memories,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Domain contract
# ---------------------------------------------------------------------------


class TestSeedRecordsSourceNotKnowledge:
    def test_seed_creates_source_evidence_and_observation(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "hello project\n"})

        (result,) = cx.seed(["README.md"])

        assert result.status == SEED_ADDED
        assert result.source.path == "README.md"
        assert len(result.source.observations) == 1
        observation = result.source.latest_observation
        assert observation.evidence_id == result.evidence.evidence_id
        assert observation.digest == compute_digest(b"hello project\n")
        assert observation.size_bytes == len("hello project\n")
        assert result.evidence.kind == "document_observation"

    def test_seed_creates_no_memory_and_no_event(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "hello\n"})

        cx.seed(["README.md"])

        counts = _raw_counts(cx)
        assert counts["sources"] == 1
        assert counts["observations"] == 1
        assert counts["evidence"] == 1
        # The whole point of A19.1: a document is not knowledge.
        assert counts["memories"] == 0
        assert counts["events"] == 0
        assert cx.state() == []
        assert cx.timeline() == []

    def test_observed_text_is_persisted_verbatim(self, tmp_path):
        """The Evidence IS the observation: it holds what the document
        said, not a sentence describing that a document was read."""
        document = "# Architecture\r\n\tuse SQLite café \U0001f600\n"
        cx = _dev_workspace(tmp_path, **{"docs/architecture.md": document})

        (result,) = cx.seed(["docs/architecture.md"])

        # Byte-for-byte, including CR, TAB and astral characters: nothing
        # is normalized, stripped or escaped on the way in.
        assert result.evidence.content == document
        assert cx.get_evidence(result.evidence.evidence_id).content == document
        # And it really reached the canonical store, not just the return value.
        assert document.encode("utf-8") in db_path_for(tmp_path / ".cortex").read_bytes()

    def test_evidence_content_carries_no_structured_metadata(self, tmp_path):
        """Path/digest/size live in their own columns. Embedding them in
        the payload would force a future reader to parse them back out of
        a document."""
        cx = _dev_workspace(tmp_path, **{"docs/architecture.md": "just the text\n"})

        (result,) = cx.seed(["docs/architecture.md"])

        assert result.evidence.content == "just the text\n"
        assert "docs/architecture.md" not in result.evidence.content
        assert result.source.latest_observation.digest not in result.evidence.content

    def test_size_bytes_measures_the_file_not_the_decoded_text(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/a.md": "café\n"})

        (result,) = cx.seed(["docs/a.md"])

        assert result.source.latest_observation.size_bytes == len("café\n".encode("utf-8"))
        assert result.source.latest_observation.size_bytes != len("café\n")

    def test_source_id_is_canonical_and_path_is_relative(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/architecture.md": "# arch\n"})

        (result,) = cx.seed([tmp_path / "docs" / "architecture.md"])

        assert len(result.source.source_id) == 32
        assert int(result.source.source_id, 16) >= 0
        assert result.source.path == "docs/architecture.md"
        assert not result.source.path.startswith("/")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestSeedIdempotency:
    def test_unchanged_file_writes_nothing(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "stable\n"})
        (first,) = cx.seed(["README.md"])
        before = _raw_counts(cx)

        (second,) = cx.seed(["README.md"])

        assert second.status == SEED_UNCHANGED
        assert _raw_counts(cx) == before
        assert second.source.source_id == first.source.source_id
        # The pre-existing Evidence is returned, not a fresh one.
        assert second.evidence.evidence_id == first.evidence.evidence_id
        assert second.evidence.recorded_at == first.evidence.recorded_at
        # ...carrying the snapshot that was already stored.
        assert second.evidence.content == "stable\n"

    def test_changed_file_appends_without_destroying_history(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "version A\n"})
        (first,) = cx.seed(["README.md"])

        (tmp_path / "README.md").write_text("version B\n", encoding="utf-8")
        (second,) = cx.seed(["README.md"])

        assert second.status == SEED_CHANGED
        assert second.source.source_id == first.source.source_id
        assert len(second.source.observations) == 2
        assert second.source.observations[0].digest == compute_digest(b"version A\n")
        assert second.source.observations[1].digest == compute_digest(b"version B\n")
        assert second.evidence.content == "version B\n"
        # The first observation's Evidence still exists untouched, with
        # the text that version A actually had.
        assert cx.get_evidence(first.evidence.evidence_id).content == "version A\n"

    def test_a_then_b_then_a_produces_three_observations(self, tmp_path):
        """Returning to an earlier digest is a real third state, not a
        retry of the first: idempotency is judged against the LATEST
        observation only."""
        cx = _dev_workspace(tmp_path, **{"README.md": "A\n"})
        cx.seed(["README.md"])
        (tmp_path / "README.md").write_text("B\n", encoding="utf-8")
        cx.seed(["README.md"])
        (tmp_path / "README.md").write_text("A\n", encoding="utf-8")

        (third,) = cx.seed(["README.md"])

        assert third.status == SEED_CHANGED
        digests = [observation.digest for observation in third.source.observations]
        assert digests == [
            compute_digest(b"A\n"),
            compute_digest(b"B\n"),
            compute_digest(b"A\n"),
        ]
        assert len({observation.evidence_id for observation in third.source.observations}) == 3
        # Each observation keeps ITS OWN snapshot, in order: the third one
        # is a genuinely new observation that happens to have seen the
        # same text again, not a reuse of the first.
        snapshots = [
            cx.get_evidence(observation.evidence_id).content
            for observation in third.source.observations
        ]
        assert snapshots == ["A\n", "B\n", "A\n"]

    def test_first_observed_at_is_not_rewritten_by_a_later_change(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "A\n"})
        (first,) = cx.seed(["README.md"])

        (tmp_path / "README.md").write_text("B\n", encoding="utf-8")
        (second,) = cx.seed(["README.md"])

        assert second.source.first_observed_at == first.source.first_observed_at


# ---------------------------------------------------------------------------
# Path security (explicit paths go through the production path)
# ---------------------------------------------------------------------------


class TestSeedPathSecurity:
    def test_traversal_outside_workspace_is_refused(self, tmp_path):
        outside = tmp_path.parent / "outside.md"
        outside.write_text("not yours\n", encoding="utf-8")
        cx = _dev_workspace(tmp_path)

        with pytest.raises(CortexSourceError, match="outside"):
            cx.seed(["../outside.md"])

    def test_absolute_path_outside_workspace_is_refused(self, tmp_path):
        outside = tmp_path.parent / "outside.md"
        outside.write_text("not yours\n", encoding="utf-8")
        cx = _dev_workspace(tmp_path)

        with pytest.raises(CortexSourceError, match="outside"):
            cx.seed([str(outside)])

    def test_symlink_escaping_the_workspace_is_refused(self, tmp_path):
        outside = tmp_path.parent / "outside_target.md"
        outside.write_text("not yours\n", encoding="utf-8")
        cx = _dev_workspace(tmp_path)
        (tmp_path / "link.md").symlink_to(outside)

        with pytest.raises(CortexSourceError, match="outside"):
            cx.seed(["link.md"])

    def test_symlink_inside_workspace_is_recorded_under_its_resolved_path(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/real.md": "# real\n"})
        (tmp_path / "link.md").symlink_to(tmp_path / "docs" / "real.md")

        (result,) = cx.seed(["link.md"])

        # Identity follows the file actually read, so a symlink and its
        # target never become two Sources for one document.
        assert result.source.path == "docs/real.md"
        (again,) = cx.seed(["docs/real.md"])
        assert again.status == SEED_UNCHANGED
        assert again.source.source_id == result.source.source_id

    def test_secret_names_are_refused_even_when_named_explicitly(self, tmp_path):
        cx = _dev_workspace(
            tmp_path,
            **{
                ".env": "TOKEN=abc\n",
                "config/.env.local": "TOKEN=abc\n",
                "server.pem": "-----BEGIN-----\n",
                "id_rsa": "private\n",
                "app.key": "k\n",
            },
        )
        for path in (".env", "config/.env.local", "server.pem", "id_rsa", "app.key"):
            with pytest.raises(CortexSourceError, match="credential"):
                cx.seed([path])

        assert not db_path_for(tmp_path / ".cortex").exists()

    def test_cortex_own_directory_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n"})
        cx.seed(["README.md"])

        with pytest.raises(CortexSourceError, match="Cortex's own"):
            cx.seed([".cortex/manifest.json"])

    def test_directory_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/a.md": "x\n"})

        with pytest.raises(CortexSourceError, match="not a regular file"):
            cx.seed(["docs"])

    def test_missing_file_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path)

        with pytest.raises(CortexSourceError, match="Cannot read"):
            cx.seed(["nope.md"])

    def test_oversized_file_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path)
        (tmp_path / "big.md").write_text("x" * (MAX_SEED_FILE_BYTES + 1), encoding="utf-8")

        with pytest.raises(CortexSourceError, match="over the"):
            cx.seed(["big.md"])

    def test_binary_file_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path)
        (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x01\x02")

        with pytest.raises(CortexSourceError, match="binary"):
            cx.seed(["logo.png"])

    def test_empty_file_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"EMPTY.md": ""})

        with pytest.raises(CortexSourceError, match="empty"):
            cx.seed(["EMPTY.md"])

        assert not db_path_for(tmp_path / ".cortex").exists()

    def test_whitespace_only_file_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"BLANK.md": "  \n\t\n"})

        with pytest.raises(CortexSourceError, match="empty"):
            cx.seed(["BLANK.md"])

    def test_invalid_utf8_is_refused(self, tmp_path):
        cx = _dev_workspace(tmp_path)
        (tmp_path / "broken.md").write_bytes(b"caf\xe9 latin-1 only")

        with pytest.raises(CortexSourceError, match="UTF-8"):
            cx.seed(["broken.md"])

    def test_one_bad_path_records_none_of_the_batch(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "good\n"})

        with pytest.raises(CortexSourceError):
            cx.seed(["README.md", "missing.md"])

        assert cx.sources() == []

    def test_seed_rejects_a_bare_string(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n"})

        with pytest.raises(TypeError):
            cx.seed("README.md")


# ---------------------------------------------------------------------------
# Discovery (dev-only, read-only)
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovery_lists_allowlisted_files_only(self, tmp_path):
        cx = _dev_workspace(
            tmp_path,
            **{
                "README.md": "r\n",
                "LICENSE": "l\n",
                "pyproject.toml": "p\n",
                "AGENTS.md": "a\n",
                "CLAUDE.md": "c\n",
                "docs/architecture.md": "d\n",
                "docs/deep/nested.md": "n\n",
                "src/main.py": "s\n",
                "secrets.txt": "x\n",
                ".env": "TOKEN=1\n",
            },
        )

        candidates = cx.seed_candidates()

        assert candidates == [
            "AGENTS.md",
            "CLAUDE.md",
            "LICENSE",
            "README.md",
            "docs/architecture.md",
            "pyproject.toml",
        ]

    def test_discovery_writes_nothing(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "r\n"})

        cx.seed_candidates()

        assert not db_path_for(tmp_path / ".cortex").exists()
        assert cx.sources() == []

    def test_discovery_is_dev_only(self, tmp_path):
        cx = Cortex.init(tmp_path, "general")
        (tmp_path / "README.md").write_text("r\n", encoding="utf-8")

        with pytest.raises(ValueError, match="dev"):
            cx.seed_candidates()

    def test_explicit_seed_works_outside_dev(self, tmp_path):
        cx = Cortex.init(tmp_path, "general")
        (tmp_path / "README.md").write_text("r\n", encoding="utf-8")

        (result,) = cx.seed(["README.md"])

        assert result.status == SEED_ADDED


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class TestSources:
    def test_sources_are_ordered_by_path_with_history(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "r\n", "docs/a.md": "a\n"})
        cx.seed(["README.md", "docs/a.md"])
        (tmp_path / "README.md").write_text("r2\n", encoding="utf-8")
        cx.seed(["README.md"])

        items = cx.sources()

        assert [source.path for source in items] == ["README.md", "docs/a.md"]
        assert len(items[0].observations) == 2
        assert items[0].latest_observation.digest == compute_digest(b"r2\n")

    def test_sources_on_empty_workspace_creates_no_store(self, tmp_path):
        cx = _dev_workspace(tmp_path)

        assert cx.sources() == []
        assert not db_path_for(tmp_path / ".cortex").exists()


# ---------------------------------------------------------------------------
# Evidence / Memory integration
# ---------------------------------------------------------------------------


class TestProvenanceIntegration:
    def test_memory_can_cite_a_seeded_observation(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/decisions.md": "we chose SQLite\n"})
        (result,) = cx.seed(["docs/decisions.md"])

        memory = cx.remember(
            "The project stores canonical data in SQLite",
            kind="note",
            evidence=[result.evidence],
        )

        assert result.evidence.evidence_id in memory.evidence_ids
        assert memory.epistemic_state == "user_asserted"

    def test_document_observation_cannot_make_a_memory_verified(self, tmp_path):
        """Cortex genuinely read the document -- and that still verifies
        nothing about whether what it says is true."""
        cx = _dev_workspace(tmp_path, **{"README.md": "claims things\n"})
        (result,) = cx.seed(["README.md"])

        assert result.evidence.kind not in VERIFICATION_EVIDENCE_KINDS
        assert result.evidence.kind not in RECOMMENDED_VALIDATION_EVIDENCE_KINDS
        with pytest.raises(ValueError, match="strong enough to justify it"):
            cx.remember(
                "This is definitely true because the README says so",
                kind="note",
                epistemic_state="verified",
                supporting_evidence=[result.evidence],
            )


# ---------------------------------------------------------------------------
# Portability
# ---------------------------------------------------------------------------


class TestPortability:
    def test_workspace_copied_to_another_path_keeps_its_sources(self, tmp_path):
        origin = tmp_path / "origin"
        origin.mkdir()
        cx = _dev_workspace(origin, **{"README.md": "portable\n"})
        (first,) = cx.seed(["README.md"])

        import shutil

        moved = tmp_path / "moved-elsewhere"
        shutil.copytree(origin, moved)

        reopened = Cortex.open(moved)
        (source,) = reopened.sources()
        assert source.source_id == first.source.source_id
        assert source.path == "README.md"
        # And the unchanged file is still recognized at the new location.
        (again,) = reopened.seed(["README.md"])
        assert again.status == SEED_UNCHANGED

    def test_history_survives_the_original_file_being_deleted(self, tmp_path):
        """The point of keeping the payload: once the file is gone, the
        digest alone could only say THAT something was observed."""
        cx = _dev_workspace(tmp_path, **{"docs/decisions.md": "we chose SQLite\n"})
        (first,) = cx.seed(["docs/decisions.md"])
        (tmp_path / "docs" / "decisions.md").write_text("we chose Postgres\n", encoding="utf-8")
        (second,) = cx.seed(["docs/decisions.md"])

        (tmp_path / "docs" / "decisions.md").unlink()
        reopened = Cortex.open(tmp_path)

        (source,) = reopened.sources()
        assert [observation.digest for observation in source.observations] == [
            compute_digest(b"we chose SQLite\n"),
            compute_digest(b"we chose Postgres\n"),
        ]
        assert reopened.get_evidence(first.evidence.evidence_id).content == "we chose SQLite\n"
        assert reopened.get_evidence(second.evidence.evidence_id).content == "we chose Postgres\n"
        # Re-seeding a file that no longer exists fails plainly; nothing
        # marks the Source deleted (no deletion tracking in A19.1).
        with pytest.raises(CortexSourceError, match="Cannot read"):
            reopened.seed(["docs/decisions.md"])

    def test_workspace_copied_without_the_project_files_keeps_its_snapshots(self, tmp_path):
        import shutil

        origin = tmp_path / "origin"
        origin.mkdir()
        cx = _dev_workspace(origin, **{"README.md": "portable payload\n"})
        (first,) = cx.seed(["README.md"])

        # Only `.cortex/` travels: the project itself is left behind.
        archived = tmp_path / "archived"
        archived.mkdir()
        shutil.copytree(origin / ".cortex", archived / ".cortex")

        reopened = Cortex.open(archived)
        (source,) = reopened.sources()
        assert source.path == "README.md"
        assert reopened.get_evidence(first.evidence.evidence_id).content == "portable payload\n"

    def test_no_absolute_path_is_ever_persisted(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"docs/a.md": "a\n"})
        cx.seed([tmp_path / "docs" / "a.md"])

        with sqlite3.connect(cx._db_path) as connection:
            paths = [row[0] for row in connection.execute("SELECT path FROM sources")]
        assert paths == ["docs/a.md"]
        assert str(tmp_path).encode() not in db_path_for(tmp_path / ".cortex").read_bytes()


# ---------------------------------------------------------------------------
# Write-boundary validation and corruption (fail closed)
# ---------------------------------------------------------------------------


class TestWriteBoundary:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"candidate_source_id": "nothex"}, "Malformed source_id"),
            ({"candidate_evidence_id": "nothex"}, "Malformed evidence_id"),
            ({"digest": "abc"}, "Malformed source digest"),
            ({"size_bytes": -1}, "non-negative"),
            ({"path": "/etc/passwd"}, "workspace-relative"),
            ({"path": ""}, "workspace-relative"),
        ],
    )
    def test_store_rejects_malformed_observations(self, tmp_path, kwargs, message):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n"})
        base = {
            "path": "README.md",
            "digest": compute_digest(b"x\n"),
            "size_bytes": 2,
            "observed_at": dt.datetime.now(dt.timezone.utc),
            "candidate_source_id": uuid.uuid4().hex,
            "candidate_evidence_id": uuid.uuid4().hex,
            "evidence_content": "# README\nthe observed document text\n",
        }
        base.update(kwargs)

        with MemoryStore.create_or_open(cx._db_path) as store:
            with pytest.raises(ValueError, match=message):
                store.observe_source(**base)

        assert _raw_counts(cx)["sources"] == 0


class TestCorruptionFailsClosed:
    def _seeded(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n"})
        (result,) = cx.seed(["README.md"])
        return cx, result

    def test_malformed_source_id_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            # Rewritten on BOTH sides so the relation stays intact and the
            # id itself is what fails, rather than the orphan check that
            # `test_dangling_source_reference_is_rejected_on_read` covers.
            connection.execute("UPDATE sources SET source_id = 'not-a-canonical-id'")
            connection.execute("UPDATE source_observations SET source_id = 'not-a-canonical-id'")

        with pytest.raises(CortexStorageError, match="Corrupted source_id"):
            cx.sources()

    def test_absolute_persisted_path_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            # The table CHECK refuses this, which is itself the first line
            # of defence; the read path refuses it too.
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("UPDATE sources SET path = '/etc/passwd'")

    def test_malformed_digest_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            connection.execute("UPDATE source_observations SET digest = 'zz'")

        with pytest.raises(CortexStorageError, match="Corrupted digest"):
            cx.sources()

    def test_negative_size_is_refused_by_the_table(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("UPDATE source_observations SET size_bytes = -5")

    def test_malformed_observed_at_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            connection.execute("UPDATE source_observations SET observed_at = 'yesterday'")

        with pytest.raises(CortexStorageError, match="Corrupted observed_at"):
            cx.sources()

    def test_dangling_evidence_reference_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            connection.execute("DELETE FROM evidence")

        with pytest.raises(CortexStorageError, match="unknown evidence"):
            cx.sources()

    def test_dangling_source_reference_is_rejected_on_read(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            connection.execute("DELETE FROM sources")

        with pytest.raises(CortexStorageError, match="unknown source"):
            cx.sources()

    def test_source_without_observations_is_rejected_on_reseed(self, tmp_path):
        cx, _ = self._seeded(tmp_path)
        with sqlite3.connect(cx._db_path) as connection:
            connection.execute("DELETE FROM source_observations")

        with pytest.raises(CortexStorageError, match="no observations"):
            cx.seed(["README.md"])


# ---------------------------------------------------------------------------
# Failure injection: no partial canonical state
# ---------------------------------------------------------------------------


class _FailingConnection:
    """Minimal proxy that lets the REAL connection do everything except
    the one statement under test, so the rollback exercised is production
    machinery rather than a test double of it."""

    def __init__(self, connection, fail_on):
        self._connection = connection
        self._fail_on = fail_on

    def execute(self, sql, *args):
        if self._fail_on in sql:
            raise sqlite3.OperationalError(f"injected failure on {self._fail_on}")
        return self._connection.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __enter__(self):
        return self._connection.__enter__()

    def __exit__(self, *exc_info):
        return self._connection.__exit__(*exc_info)


class TestFailureInjection:
    @pytest.mark.parametrize(
        "fail_on",
        ["INSERT INTO sources", "INSERT INTO evidence", "INSERT INTO source_observations"],
    )
    def test_injected_failure_leaves_zero_partial_state(self, tmp_path, fail_on):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n", "other.md": "o\n"})
        cx.seed(["other.md"])
        before = _raw_counts(cx)

        store = MemoryStore.create_or_open(cx._db_path)
        store._connection = _FailingConnection(store._connection, fail_on)
        with pytest.raises(CortexStorageError):
            store.observe_source(
                path="README.md",
                digest=compute_digest(b"x\n"),
                size_bytes=2,
                observed_at=dt.datetime.now(dt.timezone.utc),
                candidate_source_id=uuid.uuid4().hex,
                candidate_evidence_id=uuid.uuid4().hex,
                evidence_content="# README\nthe observed document text\n",
            )
        store._connection._connection.close()

        assert _raw_counts(cx) == before
        # The store is still usable afterwards.
        (result,) = cx.seed(["README.md"])
        assert result.status == SEED_ADDED


# ---------------------------------------------------------------------------
# Concurrency: real processes, not threads
# ---------------------------------------------------------------------------

_PROC_COUNT = 6


def _seed_worker(workspace_dir, relative_path, barrier, queue):
    try:
        barrier.wait(timeout=30)
        from cortex_memory import Cortex as _Cortex

        (result,) = _Cortex.open(workspace_dir).seed([relative_path])
        queue.put(("ok", result.status, result.source.source_id, result.evidence.evidence_id))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, not swallowed
        queue.put(("error", f"{type(exc).__name__}: {exc}", None, None))


def _run_seed_workers(workspace_dir, relative_path, *, count=_PROC_COUNT, timeout=60):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(count)
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_seed_worker, args=(workspace_dir, relative_path, barrier, queue))
        for _ in range(count)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=timeout) for _ in processes]
    for process in processes:
        process.join(timeout=timeout)
        assert process.exitcode == 0, f"worker exited with {process.exitcode}"
    errors = [r for r in results if r[0] == "error"]
    assert not errors, errors
    return results


def _integrity(db_path):
    with sqlite3.connect(db_path) as connection:
        (integrity,) = connection.execute("PRAGMA integrity_check").fetchone()
        (user_version,) = connection.execute("PRAGMA user_version").fetchone()
    return integrity, user_version


class TestConcurrentSeed:
    def test_concurrent_first_seed_produces_one_source(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "shared\n"})
        assert not db_path_for(tmp_path / ".cortex").exists()

        results = _run_seed_workers(tmp_path, "README.md")

        assert len({r[2] for r in results}) == 1, "every process must resolve the same source_id"
        assert len({r[3] for r in results}) == 1, "every process must resolve the same observation"
        assert [r[1] for r in results].count(SEED_ADDED) == 1
        assert [r[1] for r in results].count(SEED_UNCHANGED) == _PROC_COUNT - 1

        counts = _raw_counts(cx)
        assert counts["sources"] == 1
        assert counts["observations"] == 1
        assert counts["evidence"] == 1
        assert _integrity(cx._db_path) == ("ok", STORE_SCHEMA_VERSION)

    def test_concurrent_unchanged_seed_writes_nothing(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "shared\n"})
        cx.seed(["README.md"])
        before = _raw_counts(cx)

        results = _run_seed_workers(tmp_path, "README.md")

        assert all(r[1] == SEED_UNCHANGED for r in results)
        assert _raw_counts(cx) == before
        assert _integrity(cx._db_path) == ("ok", STORE_SCHEMA_VERSION)

    def test_concurrent_changed_seed_appends_exactly_one_observation(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "A\n"})
        cx.seed(["README.md"])
        (tmp_path / "README.md").write_text("B\n", encoding="utf-8")

        results = _run_seed_workers(tmp_path, "README.md")

        assert [r[1] for r in results].count(SEED_CHANGED) == 1
        assert [r[1] for r in results].count(SEED_UNCHANGED) == _PROC_COUNT - 1
        counts = _raw_counts(cx)
        assert counts["sources"] == 1
        assert counts["observations"] == 2
        assert counts["evidence"] == 2
        assert _integrity(cx._db_path) == ("ok", STORE_SCHEMA_VERSION)
        # One winner wrote the new snapshot; the losers returned it rather
        # than writing a second copy of the same observation.
        (source,) = cx.sources()
        snapshots = [
            cx.get_evidence(observation.evidence_id).content
            for observation in source.observations
        ]
        assert snapshots == ["A\n", "B\n"]


# ---------------------------------------------------------------------------
# Evidence kind registry
# ---------------------------------------------------------------------------


class TestDocumentObservationKind:
    def test_kind_is_registered_but_never_verifying(self):
        assert EVIDENCE_KIND_DOCUMENT_OBSERVATION == "document_observation"
        assert EVIDENCE_KIND_DOCUMENT_OBSERVATION in VALID_EVIDENCE_KINDS
        # Reading a document does not check that its claims are true...
        assert EVIDENCE_KIND_DOCUMENT_OBSERVATION not in VERIFICATION_EVIDENCE_KINDS
        # ...and it is not something an agent can re-run to check itself.
        assert EVIDENCE_KIND_DOCUMENT_OBSERVATION not in RECOMMENDED_VALIDATION_EVIDENCE_KINDS

    def test_file_reference_keeps_its_previous_meaning(self):
        """A19.1 adds a kind, it does not repurpose one: `file_reference`
        still means "a pointer to a file", and callers who recorded it
        before A19.1 are unaffected."""
        assert "file_reference" in VALID_EVIDENCE_KINDS
        assert "file_reference" not in VERIFICATION_EVIDENCE_KINDS
        assert "file_reference" != EVIDENCE_KIND_DOCUMENT_OBSERVATION

    def test_a_seeded_observation_is_not_a_file_reference(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "x\n"})

        (result,) = cx.seed(["README.md"])

        assert result.evidence.kind == EVIDENCE_KIND_DOCUMENT_OBSERVATION


# ---------------------------------------------------------------------------
# Retrieval boundary: a Source is not a belief
# ---------------------------------------------------------------------------


class TestSeededDocumentsStayOutOfMemoryRetrieval:
    def test_recall_preflight_and_guard_ignore_seeded_documents(self, tmp_path):
        cx = _dev_workspace(
            tmp_path,
            **{
                "docs/architecture.md": (
                    "# Architecture\nThe authentication layer uses signed tokens\n"
                )
            },
        )
        cx.seed(["docs/architecture.md"])

        # The document's own vocabulary must not surface as knowledge...
        assert cx.recall("architecture") == []
        assert cx.recall("authentication layer signed tokens") == []
        assert cx.recall("architecture", include_superseded=True) == []
        # ...nor as experience or a warning.
        preflight = cx.preflight("Implement the authentication layer")
        assert preflight.is_empty()
        assert cx.guard("Change the authentication layer").is_empty()

    def test_cli_recall_finds_nothing_after_seeding(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "architecture.md").write_text("# arch\n", encoding="utf-8")
        main_ = cli_main
        main_(["init", "dev"])
        main_(["seed", "docs/architecture.md"])
        capsys.readouterr()

        exit_code = main_(["recall", "architecture"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "No memories found." in captured.out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_FORGERY_PAYLOADS = {
    "ansi_escape": "safe\x1b[2Aerased",
    "newline_forged_heading": "safe\nSOURCE\n  path: forged\n",
    "carriage_return": "safe\rforged",
    "bidi_override": "safe\u202Egnorw",
    "unicode_line_separator": "safe\u2028OBSERVATIONS (1, oldest first)",
    "c1_control": "safe\x9bforged",
}


class TestSourcesCli:
    def _seeded_workspace(self, tmp_path, monkeypatch, content="# arch\nline two\n"):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "architecture.md").write_text(content, encoding="utf-8")
        cli_main(["init", "dev"])
        cli_main(["seed", "docs/architecture.md"])

    def test_listing_stays_compact_and_dumps_no_snapshot(self, tmp_path, monkeypatch, capsys):
        self._seeded_workspace(tmp_path, monkeypatch)
        capsys.readouterr()

        exit_code = cli_main(["sources"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "docs/architecture.md" in captured.out
        assert "observation(s)" in captured.out
        assert "line two" not in captured.out

    def test_inspection_shows_history_and_snapshots(self, tmp_path, monkeypatch, capsys):
        self._seeded_workspace(tmp_path, monkeypatch)
        (tmp_path / "docs" / "architecture.md").write_text("# arch v2\n", encoding="utf-8")
        cli_main(["seed", "docs/architecture.md"])
        capsys.readouterr()

        exit_code = cli_main(["sources", "docs/architecture.md"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "SOURCE" in captured.out
        assert "OBSERVATIONS (2, oldest first)" in captured.out
        assert captured.out.count("DOCUMENT CONTENT") == 2
        # Both snapshots, oldest first, each rendered under its own observation.
        assert "     | # arch" in captured.out
        assert "     | line two" in captured.out
        assert "     | # arch v2" in captured.out
        assert captured.out.index("line two") < captured.out.index("# arch v2")

    def test_inspection_of_an_unknown_path_fails_cleanly(self, tmp_path, monkeypatch, capsys):
        self._seeded_workspace(tmp_path, monkeypatch)
        capsys.readouterr()

        exit_code = cli_main(["sources", "docs/nope.md"])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "No source recorded" in captured.out

    def test_seed_discloses_local_content_storage(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "internal-design.md").write_text("private plans\n", encoding="utf-8")
        cli_main(["init", "dev"])
        capsys.readouterr()

        cli_main(["seed", "internal-design.md"])

        captured = capsys.readouterr()
        assert "stored locally in .cortex/" in captured.out
        assert "not verified knowledge" in captured.out

    def test_unchanged_seed_claims_nothing_was_recorded(self, tmp_path, monkeypatch, capsys):
        self._seeded_workspace(tmp_path, monkeypatch)
        capsys.readouterr()

        cli_main(["seed", "docs/architecture.md"])

        captured = capsys.readouterr()
        assert "unchanged" in captured.out
        assert "stored locally" not in captured.out

    @pytest.mark.parametrize("name, payload", sorted(_FORGERY_PAYLOADS.items()))
    def test_stored_document_cannot_forge_cli_structure(
        self, tmp_path, monkeypatch, capsys, name, payload
    ):
        self._seeded_workspace(tmp_path, monkeypatch, content=f"{payload}\n")
        capsys.readouterr()

        exit_code = cli_main(["sources", "docs/architecture.md"])

        captured = capsys.readouterr()
        assert exit_code == 0
        lines = captured.out.splitlines()
        # Structure BEGINS exactly as many lines as the CLI emitted it on:
        # a document that spells out a heading cannot open a line with it.
        # (The same words appearing inside a `     | ` content line are
        # data, correctly rendered as data.)
        assert [line for line in lines if line.startswith("SOURCE")] == ["SOURCE"]
        assert len([line for line in lines if line.startswith("OBSERVATIONS (")]) == 1
        assert len([line for line in lines if line.startswith("     DOCUMENT CONTENT")]) == 1
        for line in lines:
            assert line == "" or line.startswith(("SOURCE", "  ", "OBSERVATIONS", "     "))
        for forbidden in ("\x1b", "\r", "\x9b", "\u2028", "\u202E"):
            assert forbidden not in captured.out
        # Every content line is introduced by structure the CLI emitted.
        content_lines = [
            line for line in captured.out.splitlines() if line.startswith("     | ")
        ]
        assert content_lines

    @pytest.mark.parametrize("name, payload", sorted(_FORGERY_PAYLOADS.items()))
    def test_canonical_snapshot_is_never_sanitized_on_storage(
        self, tmp_path, monkeypatch, capsys, name, payload
    ):
        """Sanitize on OUTPUT, never on STORAGE: the Python API keeps
        exactly the bytes the document held."""
        self._seeded_workspace(tmp_path, monkeypatch, content=f"{payload}\n")
        capsys.readouterr()

        cx = Cortex.open(tmp_path)
        (source,) = cx.sources()
        evidence = cx.get_evidence(source.latest_observation.evidence_id)

        assert evidence.content == f"{payload}\n"


# ---------------------------------------------------------------------------
# Migration v6 -> v7
# ---------------------------------------------------------------------------


def _downgrade_to_v6(db_path):
    """Turn a current store back into a v6-shaped one: A19.1's tables
    removed and the version stamp rolled back, so the real migration path
    runs on the next open."""
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE source_observations")
        connection.execute("DROP TABLE sources")
        connection.execute("PRAGMA user_version = 6")


class TestMigrationV6ToV7:
    def test_v6_store_migrates_and_keeps_its_existing_data(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "post-migration\n"})
        memory = cx.remember("a belief recorded before A19.1", kind="note")
        legacy = cx.add_evidence("see the README", kind="file_reference")
        _downgrade_to_v6(cx._db_path)

        reopened = Cortex.open(tmp_path)

        # Nothing is backfilled: a pre-v7 `file_reference` was never an
        # observation of a tracked Source, and guessing one out of its
        # free text is exactly what the structured columns exist to avoid.
        assert reopened.sources() == []
        assert reopened.get_evidence(legacy.evidence_id).kind == "file_reference"
        assert [item.memory_id for item in reopened.state()] == [memory.memory_id]
        assert _integrity(cx._db_path) == ("ok", STORE_SCHEMA_VERSION)

    def test_migrated_store_can_seed_normally(self, tmp_path):
        cx = _dev_workspace(tmp_path, **{"README.md": "post-migration\n"})
        cx.remember("a belief recorded before A19.1", kind="note")
        _downgrade_to_v6(cx._db_path)

        (result,) = Cortex.open(tmp_path).seed(["README.md"])

        assert result.status == SEED_ADDED
        assert result.evidence.content == "post-migration\n"
        assert result.evidence.kind == EVIDENCE_KIND_DOCUMENT_OBSERVATION
