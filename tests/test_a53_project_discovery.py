"""A53: bounded, safe, recursive project file discovery.

Non-vacuity: before this change `discover_candidate_paths` globbed
exactly two locations -- the workspace root and a FLAT `docs/*.md`. A
note written to `docs/research/foo.md` was invisible to `urdyn seed`
with no arguments and to the watcher, permanently and silently. Every
`test_recursive_*` test here fails on the pre-A53 codebase for that
reason.

The other half of this file is the price of that: once discovery walks
a tree, "what it will never walk into" stops being a property of a glob
and becomes a policy that has to be stated and tested -- the mandatory
directory exclusion list, `.gitignore`, and `.git/info/exclude`. The
privacy invariant (`test_privacy_*`) is the load-bearing one: a file the
project already told git to forget must never reach a Source, an
Observation, an Evidence, the search index, or compiled context.

Fixture naming: the private-directory cases use generic names
(`.private-notes/`, `.local-secrets/`). They stand for "whatever this
user chose to keep out of the repository", not for any particular
directory.
"""

from __future__ import annotations

import time

import pytest

from urdyn import Urdyn
from urdyn import _watcher
from urdyn._ignore import IgnoreRules, load_ignore_rules, parse_ignore_text
from urdyn._source import (
    MANDATORY_EXCLUDED_DIR_NAMES,
    MAX_SEED_FILE_BYTES,
    discover_candidate_paths,
)
from urdyn._store import db_path_for


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(root, relative: str, content: str = "content\n") -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _dev(tmp_path, **files) -> Urdyn:
    for name, content in files.items():
        _write(tmp_path, name, content)
    return Urdyn.init(tmp_path, "dev")


def _rules(text: str) -> IgnoreRules:
    return IgnoreRules(parse_ignore_text(text))


# ---------------------------------------------------------------------------
# 1. the ignore matcher, as a unit
# ---------------------------------------------------------------------------


class TestIgnoreMatcher:
    def test_comments_and_blank_lines_are_not_patterns(self):
        rules = _rules("# a comment\n\n   \nreal.md\n")

        assert not rules.is_ignored("a comment")
        assert not rules.is_ignored("#")
        assert rules.is_ignored("real.md")

    def test_unanchored_pattern_matches_at_any_depth(self):
        rules = _rules("notes.md\n")

        assert rules.is_ignored("notes.md")
        assert rules.is_ignored("a/b/notes.md")

    def test_leading_slash_anchors_to_the_workspace_root(self):
        rules = _rules("/notes.md\n")

        assert rules.is_ignored("notes.md")
        assert not rules.is_ignored("a/notes.md")

    def test_embedded_slash_also_anchors(self):
        rules = _rules("docs/notes.md\n")

        assert rules.is_ignored("docs/notes.md")
        assert not rules.is_ignored("a/docs/notes.md")

    def test_trailing_slash_is_directory_only(self):
        rules = _rules("build/\n")

        assert rules.is_ignored("build", is_dir=True)
        assert rules.is_ignored("build/x.md")  # everything beneath it, too
        assert not rules.is_ignored("build", is_dir=False)  # a FILE named build

    def test_wildcards(self):
        rules = _rules("*.log\nnote?.md\ndraft[0-9].md\n")

        assert rules.is_ignored("a/b/server.log")
        assert rules.is_ignored("note1.md")
        assert not rules.is_ignored("note12.md")
        assert rules.is_ignored("draft7.md")
        assert not rules.is_ignored("draftX.md")

    def test_double_star(self):
        rules = _rules("**/generated/**\n")

        assert rules.is_ignored("a/generated/x.md")
        assert rules.is_ignored("generated/deep/x.md")
        assert not rules.is_ignored("a/handwritten/x.md")

    def test_negation_reincludes_a_later_match(self):
        rules = _rules("*.md\n!keep.md\n")

        assert rules.is_ignored("a/drop.md")
        assert not rules.is_ignored("a/keep.md")

    def test_rules_are_cumulative_in_file_order(self):
        # The LAST matching rule wins, so re-excluding after a negation
        # sticks -- order, not specificity, decides.
        assert not _rules("x.md\n!x.md\n").is_ignored("x.md")
        assert _rules("!x.md\nx.md\n").is_ignored("x.md")

    def test_a_negation_cannot_resurrect_a_file_under_an_excluded_dir(self):
        rules = _rules("private/\n!private/public.md\n")

        assert rules.is_ignored("private/public.md")

    def test_empty_ruleset_ignores_nothing(self):
        assert not IgnoreRules().is_ignored("anything.md")

    def test_git_info_exclude_is_read_when_a_git_dir_exists(self, tmp_path):
        (tmp_path / ".git" / "info").mkdir(parents=True)
        (tmp_path / ".git" / "info" / "exclude").write_text(
            "# personal, never committed\n.private-notes/\n", encoding="utf-8"
        )

        rules = load_ignore_rules(tmp_path)

        assert rules.is_ignored(".private-notes", is_dir=True)
        assert rules.is_ignored(".private-notes/journal.md")
        assert not rules.is_ignored("docs/journal.md")

    def test_git_info_exclude_is_not_read_without_a_git_dir(self, tmp_path):
        # No `.git/` means this is not a repository; a stray file at that
        # path must not be interpreted as git configuration.
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "info").mkdir()
        (tmp_path / ".git" / "info" / "exclude").write_text("x.md\n", encoding="utf-8")
        assert load_ignore_rules(tmp_path).is_ignored("x.md")

        other = tmp_path / "not-a-repo"
        other.mkdir()
        assert not load_ignore_rules(other).is_ignored("x.md")

    def test_load_never_raises_on_a_workspace_with_no_ignore_files(self, tmp_path):
        assert not load_ignore_rules(tmp_path).is_ignored("README.md")


# ---------------------------------------------------------------------------
# 2. recursive discovery
# ---------------------------------------------------------------------------


class TestRecursiveDiscovery:
    def test_recursive_nested_markdown_and_text_are_discovered(self, tmp_path):
        cx = _dev(
            tmp_path,
            **{
                "README.md": "r\n",
                "docs/architecture.md": "a\n",
                "docs/research/2024/deep/note.md": "n\n",
                "notes/todo.txt": "t\n",
            },
        )

        assert cx.seed_candidates() == [
            "README.md",
            "docs/architecture.md",
            "docs/research/2024/deep/note.md",
            "notes/todo.txt",
        ]

    def test_source_code_is_never_discovered(self, tmp_path):
        cx = _dev(
            tmp_path,
            **{
                "README.md": "r\n",
                "src/main.py": "print(1)\n",
                "src/app.js": "1\n",
                "src/lib.rs": "fn main(){}\n",
            },
        )

        assert cx.seed_candidates() == ["README.md"]

    def test_manifest_globs_stay_root_only(self, tmp_path):
        cx = _dev(
            tmp_path,
            **{
                "pyproject.toml": "p\n",
                "vendor/other/pyproject.toml": "v\n",
            },
        )

        assert cx.seed_candidates() == ["pyproject.toml"]

    def test_repeated_discovery_is_idempotent_and_deduplicated(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "docs/a/b.md": "b\n"})

        first = cx.seed_candidates()
        second = cx.seed_candidates()

        assert first == second
        assert len(first) == len(set(first))
        assert first == sorted(first)

    def test_a_newly_created_nested_file_is_discovered(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        assert cx.seed_candidates() == ["README.md"]

        _write(tmp_path, "docs/research/new.md", "just written\n")

        assert cx.seed_candidates() == ["README.md", "docs/research/new.md"]

    def test_discovery_writes_nothing_even_when_recursive(self, tmp_path):
        cx = _dev(tmp_path, **{"docs/a/b/c.md": "deep\n"})

        cx.seed_candidates()

        assert not db_path_for(tmp_path / ".urdyn").exists()
        assert cx.sources() == []

    def test_workspace_without_git_still_discovers(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "docs/a/b.md": "b\n"})
        assert not (tmp_path / ".git").exists()

        assert cx.seed_candidates() == ["README.md", "docs/a/b.md"]

    @pytest.mark.parametrize("excluded", sorted(MANDATORY_EXCLUDED_DIR_NAMES))
    def test_mandatory_excluded_dirs_are_never_descended(self, tmp_path, excluded):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _write(tmp_path, f"{excluded}/notes.md", "hidden\n")
        _write(tmp_path, f"a/b/{excluded}/notes.md", "hidden deeper\n")

        candidates = cx.seed_candidates()

        assert candidates == ["README.md"]

    def test_egg_info_glob_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _write(tmp_path, "urdyn_memory.egg-info/PKG-INFO.txt", "meta\n")

        assert cx.seed_candidates() == ["README.md"]

    def test_urdyn_own_directory_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _write(tmp_path, ".urdyn/notes.md", "internal\n")

        assert cx.seed_candidates() == ["README.md"]


# ---------------------------------------------------------------------------
# 3. per-file eligibility, unchanged and still applied
# ---------------------------------------------------------------------------


class TestEligibilityGates:
    def test_binary_file_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "blob.md").write_bytes(b"\x00\x01\x02")

        assert cx.seed_candidates() == ["README.md"]

    def test_non_utf8_file_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "latin.md").write_bytes(b"caf\xe9 not utf-8")

        assert cx.seed_candidates() == ["README.md"]

    def test_blank_file_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "docs/blank.md": "   \n"})

        assert cx.seed_candidates() == ["README.md"]

    def test_oversized_file_is_excluded(self, tmp_path):
        cx = _dev(
            tmp_path,
            **{
                "README.md": "r\n",
                "docs/big.md": "x" * (MAX_SEED_FILE_BYTES + 1),
            },
        )

        assert cx.seed_candidates() == ["README.md"]

    def test_credential_named_file_is_excluded(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _write(tmp_path, "config/.env.txt", "TOKEN=1\n")

        assert cx.seed_candidates() == ["README.md"]

    def test_symlink_inside_the_workspace_is_followed_and_canonicalised(self, tmp_path):
        """A symlinked directory inside the workspace IS descended into,
        so a file only reachable through it is still found. It is not
        recorded twice, though: `resolve_seed_path` canonicalises every
        candidate, so the aliased spelling collapses onto the real path
        and one file stays one Source."""
        cx = _dev(tmp_path, **{"README.md": "r\n", "real/notes.md": "inside\n"})
        (tmp_path / "linked").symlink_to(tmp_path / "real")

        candidates = cx.seed_candidates()

        assert candidates == ["README.md", "real/notes.md"]

    def test_a_file_only_reachable_through_a_symlinked_dir_is_found(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        real = tmp_path / "vault"
        real.mkdir()
        (real / "notes.md").write_text("inside\n", encoding="utf-8")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "linked").symlink_to(real)

        # Reached via `docs/linked/notes.md`, canonicalised to the real path.
        assert "vault/notes.md" in cx.seed_candidates()

    def test_symlink_escaping_the_workspace_is_excluded(self, tmp_path):
        outside = tmp_path.parent / f"outside-{tmp_path.name}"
        outside.mkdir()
        (outside / "secret.md").write_text("elsewhere\n", encoding="utf-8")
        try:
            cx = _dev(tmp_path, **{"README.md": "r\n"})
            (tmp_path / "escape").symlink_to(outside)
            (tmp_path / "escape.md").symlink_to(outside / "secret.md")

            candidates = cx.seed_candidates()

            assert candidates == ["README.md"]
        finally:
            (outside / "secret.md").unlink(missing_ok=True)
            outside.rmdir()

    def test_symlink_cycle_terminates(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "loop/notes.md": "x\n"})
        (tmp_path / "loop" / "back").symlink_to(tmp_path / "loop")

        # The guarantee is that this RETURNS at all; a cycle-unaware walk
        # would recurse until the visit cap or a RecursionError.
        candidates = cx.seed_candidates()

        assert "README.md" in candidates
        assert "loop/notes.md" in candidates


# ---------------------------------------------------------------------------
# 4. the privacy invariant
# ---------------------------------------------------------------------------


class TestPrivacyInvariant:
    def test_gitignored_file_is_not_discovered(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "docs/keep.md": "k\n"})
        _write(tmp_path, "docs/drop.md", "d\n")
        (tmp_path / ".gitignore").write_text(".urdyn/\ndocs/drop.md\n", encoding="utf-8")

        assert cx.seed_candidates() == ["README.md", "docs/keep.md"]

    def test_gitignored_directory_is_not_descended(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _write(tmp_path, "scratch/a/b/notes.md", "n\n")
        (tmp_path / ".gitignore").write_text(".urdyn/\nscratch/\n", encoding="utf-8")

        assert cx.seed_candidates() == ["README.md"]

    def test_git_info_excluded_private_dir_never_surfaces_anywhere(self, tmp_path):
        """The full invariant for a directory excluded via
        `.git/info/exclude` (the private, uncommitted mechanism): it
        reaches no candidate list, no watcher scope, no Source, no
        Observation, no Evidence, no search index, and no compiled
        context."""
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        (tmp_path / ".git" / "info").mkdir(parents=True)
        (tmp_path / ".git" / "info" / "exclude").write_text(
            ".local-secrets/\n", encoding="utf-8"
        )
        _write(tmp_path, ".local-secrets/passwords.md", "hunter2 lives here\n")

        assert ".local-secrets/passwords.md" not in cx.seed_candidates()
        assert ".local-secrets/passwords.md" not in cx.watcher_scope()

        # Seed everything discovery DOES propose, then check the store.
        cx.seed(cx.seed_candidates())

        assert cx.sources()  # the run was non-vacuous: README.md WAS seeded
        assert all(not source.path.startswith(".local-secrets/") for source in cx.sources())
        for source in cx.sources():
            for observation in source.observations:
                assert "hunter2" not in cx.get_evidence(observation.evidence_id).content
        assert "hunter2" not in cx.context("passwords").render()

    def test_explicit_seed_of_an_ignored_file_still_works(self, tmp_path):
        """DELIBERATELY UNCHANGED. Ignore rules and the discovery
        allowlist govern what Urdyn proposes and watches on its own; they
        have never governed what a user may explicitly ask it to record.
        `urdyn seed <path>` remains a deliberate act, refused only by the
        path/content checks that also apply to it (outside the workspace,
        a credential name, oversized, binary)."""
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        (tmp_path / ".gitignore").write_text(".urdyn/\nprivate/\n", encoding="utf-8")
        _write(tmp_path, "private/chosen.md", "I asked for this\n")

        assert "private/chosen.md" not in cx.seed_candidates()

        (result,) = cx.seed(["private/chosen.md"])

        assert result.status == "added"
        assert result.source.path == "private/chosen.md"


# ---------------------------------------------------------------------------
# 5. watcher cadence: cheap half every tick, walk rarely
# ---------------------------------------------------------------------------


class TestWatcherScopeCadence:
    def test_tracked_scope_does_not_walk_and_discovered_scope_does(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n", "docs/a/b.md": "b\n"})
        cx.seed(["README.md"])

        assert cx.tracked_scope() == frozenset({"README.md"})
        assert cx.discovered_scope() == frozenset({"README.md", "docs/a/b.md"})
        assert cx.watcher_scope() == frozenset({"README.md", "docs/a/b.md"})

    def test_cache_reuses_the_discovery_result_until_the_interval_elapses(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        cache = _watcher._ScopeCache(cx.discovered_scope())

        _write(tmp_path, "docs/late.md", "created after the cache was primed\n")

        # Within the interval: the expensive half is NOT recomputed.
        assert "docs/late.md" not in cache.scope(cx)
        # Forced (what a fresh process does at baseline): it is.
        assert "docs/late.md" in cache.scope(cx, force=True)

    def test_cache_still_picks_up_a_newly_tracked_source_immediately(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        cache = _watcher._ScopeCache(cx.discovered_scope())

        _write(tmp_path, "src/module.py", "print(1)\n")
        cx.seed(["src/module.py"])

        # The cheap half is recomputed on EVERY call, cache or not: a
        # path seeded from another terminal must be watched at once.
        assert "src/module.py" in cache.scope(cx)

    def test_cache_refreshes_once_the_interval_has_elapsed(self, tmp_path, monkeypatch):
        # A fake-clock stand-in: shrinking the interval is equivalent to
        # advancing time and keeps the test off wall-clock waits.
        monkeypatch.setattr(_watcher, "_DISCOVERY_SCAN_INTERVAL", 0.0)
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        cache = _watcher._ScopeCache(cx.discovered_scope())

        _write(tmp_path, "docs/late.md", "created after the cache was primed\n")

        assert "docs/late.md" in cache.scope(cx)

    def test_cache_keeps_the_previous_result_when_a_scan_fails(self, tmp_path, monkeypatch):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        cache = _watcher._ScopeCache(cx.discovered_scope())
        monkeypatch.setattr(_watcher, "_DISCOVERY_SCAN_INTERVAL", 0.0)

        def _boom() -> frozenset[str]:
            raise OSError("filesystem went away")

        monkeypatch.setattr(cx, "discovered_scope", _boom)

        # A transient I/O failure must never silently un-watch everything.
        assert "README.md" in cache.scope(cx)

    def test_new_nested_file_is_discovered_by_a_live_watcher(self, tmp_path, monkeypatch):
        """End to end, with the discovery cadence shortened rather than
        waited out: a file created under a directory the pre-A53 glob
        never looked in becomes watched, and a later edit to it is
        observed."""
        monkeypatch.setattr(_watcher, "_DISCOVERY_SCAN_INTERVAL", 0.0)
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        cache = _watcher._ScopeCache(cx.discovered_scope())

        _write(tmp_path, "docs/research/deep/finding.md", "first draft\n")

        assert "docs/research/deep/finding.md" in cache.scope(cx)

        (result,) = cx.seed(["docs/research/deep/finding.md"])
        assert result.status == "added"
        _write(tmp_path, "docs/research/deep/finding.md", "second draft\n")
        (result,) = cx.seed(["docs/research/deep/finding.md"])
        assert result.status == "changed"

    def test_live_watcher_observes_a_nested_file_created_after_enable(self, tmp_path):
        """The whole point of A53, through the real detached watcher
        process: a note filed at `docs/research/deep/` -- somewhere the
        pre-A53 flat `docs/*.md` glob could never look -- is discovered
        and observed on its own, with no explicit `urdyn seed`.

        Bounded by `_DISCOVERY_SCAN_INTERVAL` rather than the 2s poll,
        which is exactly the cadence decision this change makes.
        """
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        _watcher.enable_and_start(cx)

        _write(tmp_path, "docs/research/deep/finding.md", "written after enable\n")

        deadline = time.monotonic() + _watcher._DISCOVERY_SCAN_INTERVAL + 15.0
        while time.monotonic() < deadline:
            if any(s.path == "docs/research/deep/finding.md" for s in cx.sources()):
                break
            time.sleep(0.2)

        source = [s for s in cx.sources() if s.path == "docs/research/deep/finding.md"]
        assert source, "the nested file was never discovered by the live watcher"
        assert len(source[0].observations) == 1
        assert (
            cx.get_evidence(source[0].latest_observation.evidence_id).content
            == "written after enable\n"
        )

    def test_live_watcher_never_observes_a_gitignored_new_file(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})
        (tmp_path / ".gitignore").write_text(".urdyn/\nscratch/\n", encoding="utf-8")
        _watcher.enable_and_start(cx)

        _write(tmp_path, "scratch/draft.md", "should stay invisible\n")

        time.sleep(_watcher._DISCOVERY_SCAN_INTERVAL + 6.0)
        assert all(s.path != "scratch/draft.md" for s in cx.sources())
        assert "scratch/draft.md" not in _watcher._scan_scope(cx)

    def test_repeated_scans_produce_no_duplicate_source(self, tmp_path):
        cx = _dev(tmp_path, **{"docs/a/b.md": "b\n"})
        cache = _watcher._ScopeCache()

        for _ in range(3):
            for path in cache.scope(cx, force=True):
                cx.seed([path])

        paths = [source.path for source in cx.sources()]
        assert paths == ["docs/a/b.md"]
        assert len(cx.sources()[0].observations) == 1


# ---------------------------------------------------------------------------
# 6. seed UX: new vs changed vs unchanged
# ---------------------------------------------------------------------------


class TestSeedCandidateReport:
    def test_report_splits_new_changed_and_unchanged(self, tmp_path):
        cx = _dev(
            tmp_path,
            **{
                "README.md": "r\n",
                "docs/tracked.md": "v1\n",
                "docs/stable.md": "s\n",
            },
        )
        cx.seed(["docs/tracked.md", "docs/stable.md"])
        _write(tmp_path, "docs/tracked.md", "v2\n")

        report = cx.seed_candidate_report()

        assert report.new == ("README.md",)
        assert report.changed == ("docs/tracked.md",)
        assert report.unchanged == ("docs/stable.md",)
        assert report.paths == cx.seed_candidates()

    def test_report_writes_nothing_and_creates_no_store(self, tmp_path):
        cx = _dev(tmp_path, **{"README.md": "r\n"})

        report = cx.seed_candidate_report()

        assert report.new == ("README.md",)
        assert not db_path_for(tmp_path / ".urdyn").exists()

    def test_report_is_dev_only(self, tmp_path):
        cx = Urdyn.init(tmp_path, "general")

        with pytest.raises(ValueError):
            cx.seed_candidate_report()


# ---------------------------------------------------------------------------
# 7. the CLI surface
# ---------------------------------------------------------------------------


def test_cli_seed_with_no_args_labels_new_and_changed(tmp_path, monkeypatch, capsys):
    from urdyn._cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    cx = _dev(tmp_path, **{"README.md": "r\n", "docs/nested/note.md": "v1\n"})
    cx.seed(["docs/nested/note.md"])
    _write(tmp_path, "docs/nested/note.md", "v2\n")

    assert cli_main(["seed"]) == 0

    out = capsys.readouterr().out
    assert "- README.md (new)" in out
    assert "- docs/nested/note.md (changed since last seed)" in out
    assert "Nothing was recorded." in out


def test_cli_seed_no_args_still_records_nothing(tmp_path, monkeypatch, capsys):
    from urdyn._cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    cx = _dev(tmp_path, **{"docs/a/b.md": "b\n"})

    assert cli_main(["seed"]) == 0

    assert cx.sources() == []


def test_visit_cap_is_documented_and_finite():
    from urdyn._source import MAX_DISCOVERY_VISITS

    assert isinstance(MAX_DISCOVERY_VISITS, int)
    assert 0 < MAX_DISCOVERY_VISITS <= 1_000_000


def test_discover_accepts_prebuilt_ignore_rules(tmp_path):
    _write(tmp_path, "README.md", "r\n")
    _write(tmp_path, "docs/a.md", "a\n")

    assert discover_candidate_paths(tmp_path, ".urdyn", ignore_rules=IgnoreRules()) == [
        "README.md",
        "docs/a.md",
    ]
    assert discover_candidate_paths(
        tmp_path, ".urdyn", ignore_rules=_rules("docs/\n")
    ) == ["README.md"]


def test_watcher_scope_stays_bounded_under_a_wide_tree(tmp_path):
    """A crude upper-bound check that discovery terminates promptly on a
    tree far wider than the flat `docs/` it used to look at."""
    cx = _dev(tmp_path, **{"README.md": "r\n"})
    for i in range(40):
        _write(tmp_path, f"docs/section{i}/sub/notes.md", f"note {i}\n")

    start = time.monotonic()
    candidates = cx.seed_candidates()
    elapsed = time.monotonic() - start

    assert len(candidates) == 41
    assert elapsed < 5.0
