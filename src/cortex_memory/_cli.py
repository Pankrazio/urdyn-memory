"""Command-line interface for the Cortex Memory Engine.

Every value printed here is either STRUCTURE the CLI itself emits
(section headers, `- ` prefixes, field labels, ids and enum values Cortex
validates on the way in) or DATA the caller stored (memory content,
attempt task/approach, skill names, evidence text, workspace paths and
manifest fields). Data is always passed through `terminal_safe_text`
before printing; structure never is. That split is the whole of A14.S --
see `_terminal.py` for why, and note that the sanitizing happens HERE, at
the rendering boundary, so the canonical record and the public API keep
returning exactly what the caller wrote.
"""

from __future__ import annotations

import argparse
import sys

from ._attempt import VALID_OUTCOMES
from ._context import DEFAULT_CONTEXT_BUDGET
from ._errors import CortexError
from ._evidence import DEFAULT_EVIDENCE_KIND, VALID_EVIDENCE_KINDS
from ._manifest import CANONICAL_PROFILES
from ._memory import DEFAULT_KIND, VALID_KINDS
from ._source import SEED_UNCHANGED
from ._terminal import terminal_safe_text as _safe
from ._workspace import DEFAULT_RECALL_LIMIT, Cortex


# (A27) Both of these are CLI STRUCTURE, not caller data: every
# component is a Cortex constant or an integer Cortex counted, so they
# are printed without `_safe` exactly like section headers are. They are
# also plain ASCII on purpose -- Cortex treats Windows consoles as a
# first-class target, and a lifecycle warning must never be the line that
# fails to encode.
_SEMANTIC_SETUP_HINT = (
    "Semantic retrieval is not enabled. Run `cortex semantic setup` to enable it "
    "(one-time model download; everything else stays offline)."
)


def _print_retrieval(state) -> None:
    """Print which retrieval substrate answered, in EVERY state and
    BEFORE the result itself -- including a healthy one, and including an
    empty result.

    A26's failure was not that Cortex printed something wrong; it was
    that `No relevant experience found.` is equally consistent with "the
    semantic channel ran and had nothing" and with "the semantic channel
    was never there". Printing this only when degraded would leave the
    healthy case communicated by omission, which is the same bug with a
    smaller blast radius. `None` (a result built without going through
    the workspace) prints nothing at all rather than guessing."""
    if state is None:
        return
    print(f"Retrieval: {state.retrieval_mode()}")


class _SafeArgumentParser(argparse.ArgumentParser):
    """`argparse.ArgumentParser` that renders its own error messages
    through the same terminal-safety boundary as everything else.

    argparse's messages quote offending values with `%r`, which already
    escapes control characters, so this is defence in depth rather than a
    known hole -- but the values in those messages come from the caller's
    argv, and "the caller's own input" is exactly the category this
    module refuses to print raw. Only `error()` is overridden: usage and
    help text are the CLI's own multi-line structure, and passing those
    through a renderer that escapes newlines would collapse them into one
    unreadable line.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        super().error(_safe(message))


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="cortex", description="Cortex Memory Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a Cortex workspace")
    init_parser.add_argument(
        "profile",
        nargs="?",
        default="general",
        choices=sorted(CANONICAL_PROFILES),
        help="Workspace profile (default: general)",
    )

    subparsers.add_parser("status", help="Show the current Cortex workspace status")

    remember_parser = subparsers.add_parser("remember", help="Record a new memory")
    remember_parser.add_argument("text", help="Memory content")
    remember_parser.add_argument(
        "--kind",
        default=DEFAULT_KIND,
        choices=sorted(VALID_KINDS),
        help=f"Memory kind (default: {DEFAULT_KIND})",
    )
    remember_parser.add_argument(
        "--supersedes",
        default=None,
        metavar="MEMORY_ID",
        help="Memory ID that this memory supersedes",
    )
    # (A20) Provenance only, repeatable. This is the CLI half of a
    # capability the library already had in full: `remember()` has taken
    # `evidence=` since long before A19.1 made `cortex seed` produce
    # citable `document_observation` Evidence, and without this flag the
    # only way to derive a belief from a seeded document was the Python
    # API. Deliberately NOT accompanied by `--supporting-evidence` or
    # `--epistemic-state`: the CLI records `user_asserted` memories and
    # nothing here changes that, so no CLI invocation can designate
    # support or reach `verified` at all -- the A12.1 gate is not merely
    # enforced against this path, it is unreachable from it. `--source
    # PATH` is likewise absent by decision: the caller cites an exact,
    # immutable evidence_id (read from `cortex sources <path>`), so the
    # provenance is never re-resolved against a file that has since
    # changed.
    remember_parser.add_argument(
        "--evidence",
        action="append",
        default=None,
        metavar="EVIDENCE_ID",
        help="Evidence ID this memory was derived from (repeatable). Provenance only: "
        "it never makes a memory verified.",
    )

    # (A25.1) Closes the CLI gap A24 measured: `remember --evidence` (A20)
    # could cite an Evidence id but the CLI had no way to PRODUCE one --
    # only `Cortex.add_evidence()` could. `evidence add` is a thin,
    # one-record-per-call adapter over exactly that method: no dedup, no
    # Event, no Source, no promotion to Memory (see `add_evidence()`'s own
    # docstring). Nested under `evidence` rather than a bare
    # `cortex evidence "<content>"` so a future `evidence show <id>` has a
    # positional slot to use without a breaking rename -- the same reason
    # `semantic setup` is nested instead of a bare `cortex semantic`.
    evidence_parser = subparsers.add_parser("evidence", help="Manage canonical Evidence")
    evidence_subparsers = evidence_parser.add_subparsers(dest="evidence_command", required=True)
    evidence_add_parser = evidence_subparsers.add_parser(
        "add", help="Record a new piece of Evidence and print its id"
    )
    evidence_add_parser.add_argument("content", help="Evidence content")
    evidence_add_parser.add_argument(
        "--kind",
        default=DEFAULT_EVIDENCE_KIND,
        choices=sorted(VALID_EVIDENCE_KINDS),
        help=f"Evidence kind (default: {DEFAULT_EVIDENCE_KIND})",
    )

    # (A25.1) The CLI half of `Cortex.learn()`. Deliberately its own
    # top-level command, not `remember --verified`: `remember` stays the
    # A20 `user_asserted`-only path, and `--epistemic-state` is not
    # exposed anywhere -- `verified` is reachable ONLY by requesting this
    # specific workflow, which the A12.1 gate inside `learn()`/`remember()`
    # still has the sole authority to grant or refuse. `--evidence` (generic
    # provenance) and `--supporting-evidence` (explicitly designated
    # support) are kept as two separate repeatable flags, exactly like the
    # public API, so the CLI cannot silently collapse the distinction A20
    # already established for `remember`.
    learn_parser = subparsers.add_parser("learn", help="Record a Lesson (candidate or verified)")
    learn_parser.add_argument("text", help="Lesson content")
    learn_parser.add_argument(
        "--evidence",
        action="append",
        default=None,
        metavar="EVIDENCE_ID",
        help="Evidence ID this lesson is derived from (repeatable). Provenance only.",
    )
    learn_parser.add_argument(
        "--supporting-evidence",
        action="append",
        default=None,
        metavar="EVIDENCE_ID",
        help="Evidence ID explicitly designated as supporting this lesson (repeatable). "
        "Required, and must include a qualifying kind, for --verified to succeed.",
    )
    learn_parser.add_argument(
        "--verified",
        action="store_true",
        help="Request the lesson be recorded as verified (subject to the existing verification gate)",
    )

    recall_parser = subparsers.add_parser("recall", help="Search recorded memories")
    recall_parser.add_argument("query", help="Search text")
    recall_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RECALL_LIMIT,
        help=f"Maximum number of results (default: {DEFAULT_RECALL_LIMIT})",
    )
    recall_parser.add_argument(
        "--include-superseded",
        action="store_true",
        help="Also search memories that have been superseded",
    )

    timeline_parser = subparsers.add_parser("timeline", help="Show the recorded history, oldest first")
    timeline_parser.add_argument(
        "--kind",
        default=None,
        choices=sorted(VALID_KINDS),
        help="Only show memories of this kind",
    )

    attempt_parser = subparsers.add_parser("attempt", help="Record an attempt at a task")
    attempt_parser.add_argument("--task", required=True, help="What was being worked on")
    attempt_parser.add_argument("--approach", required=True, help="What was tried")
    attempt_parser.add_argument("--outcome", required=True, choices=sorted(VALID_OUTCOMES))

    preflight_parser = subparsers.add_parser(
        "preflight", help="Show prior experience relevant to a task, before starting it"
    )
    preflight_parser.add_argument("task", help="Task description")

    context_parser = subparsers.add_parser(
        "context", help="Compile a budgeted working context relevant to a task, before starting it"
    )
    context_parser.add_argument("task", help="Task description")
    context_parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        help=f"Character budget for the compiled context (default: {DEFAULT_CONTEXT_BUDGET})",
    )

    # (A38) A portable sibling of `context`, not a new domain API: it
    # reuses `Cortex.context()` verbatim and only swaps the renderer
    # (`CompiledContext.render_portable()` instead of `render()`), so
    # stdout carries exactly the task-aware payload an external target
    # can consume -- no `Retrieval:` line, no success/progress noise.
    # `--for` is closed to `generic` on purpose: this tracer ships the
    # one renderer-boundary target A37 scoped it to, not a plugin point.
    export_parser = subparsers.add_parser(
        "export", help="Compile a portable, task-aware working context for an external target"
    )
    export_parser.add_argument("task", help="Task description")
    export_parser.add_argument(
        "--for",
        dest="target",
        default="generic",
        choices=["generic"],
        help="Export target (default: generic)",
    )
    export_parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        help=f"Character budget for the compiled context (default: {DEFAULT_CONTEXT_BUDGET})",
    )

    subparsers.add_parser("skills", help="List recorded skills")

    guard_parser = subparsers.add_parser(
        "guard", help="Check whether prior experience directly bears on an action about to be taken"
    )
    guard_parser.add_argument("action", help="Action about to be taken")

    seed_parser = subparsers.add_parser(
        "seed", help="Record project files as Cortex sources (with no paths: show candidates)"
    )
    seed_parser.add_argument(
        "paths",
        nargs="*",
        help="Project files to record. With none, list dev discovery candidates without recording.",
    )

    sources_parser = subparsers.add_parser("sources", help="List the project files Cortex tracks")
    sources_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Workspace-relative path of one source to inspect, with its full observation history",
    )

    semantic_parser = subparsers.add_parser(
        "semantic", help="Manage the optional semantic retrieval channel"
    )
    semantic_subparsers = semantic_parser.add_subparsers(dest="semantic_command", required=True)
    semantic_subparsers.add_parser(
        "setup",
        help="Download/prepare the semantic model and (re)build the semantic index for this workspace",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            cx = Cortex.init(".", args.profile)
            # The path is filesystem-derived, not Cortex-validated: a
            # directory name can hold anything, so it is data.
            print(f"Initialized Cortex workspace at {_safe(str(cx.path))}")
            print(f"Profile: {_safe(cx.profile)}")
            # (A27) `init` deliberately does NOT enable semantic
            # retrieval: doing so would download an embedding model, and
            # a first command in a new project must not reach the network.
            # It says so instead, because the alternative -- staying
            # silent -- is how a workspace ends up permanently
            # lexical-only without anyone deciding that (A26).
            print(_SEMANTIC_SETUP_HINT)
            return 0

        if args.command == "status":
            cx = Cortex.discover()
            # `profile`/`cortex_id` are read back from the on-disk
            # manifest, which a hand edit can change: data, not structure.
            print(f"Cortex workspace: {_safe(str(cx.path))}")
            print(f"Profile: {_safe(cx.profile)}")
            print(f"Cortex ID: {_safe(cx.cortex_id)}")
            print(f"Memories: {cx._count_memories()}")
            print(f"Invariants: {len(cx.state(kind='invariant'))}")
            print(f"Pending: {len(cx.state(kind='pending'))}")
            print(f"Open questions: {len(cx.state(kind='question'))}")
            print(f"Environment facts: {len(cx.state(kind='environment'))}")
            # (A27) Real, cheaply-computed derived state: no model is
            # loaded, nothing is embedded, no network is touched and
            # nothing is refreshed -- `status` observes, it never repairs.
            print(f"Semantic: {cx.semantic_state().describe()}")
            return 0

        if args.command == "remember":
            cx = Cortex.discover()
            # (A20) Every cited Evidence is resolved through the Core's own
            # lookup BEFORE anything is written, so an unknown id fails the
            # whole command with zero memories recorded rather than after a
            # partial write. The CLI never fabricates an `Evidence`: it
            # passes back the canonical objects the store returned, which is
            # what lets `remember()` apply its own rules (kind, existence,
            # ordering) to exactly the same values it would see from a
            # Python caller.
            evidence = [cx.get_evidence(evidence_id) for evidence_id in (args.evidence or ())]
            # `_remember` rather than `remember` for the same reason
            # `status` uses `_count_memories`: the CLI needs one internal
            # detail the public return type does not carry -- whether this
            # call actually recorded the memory or found it already
            # current (A17). Reporting an unchanged store as "Remembered"
            # would tell the user something happened when nothing did.
            memory, created = cx._remember(
                args.text, kind=args.kind, supersedes=args.supersedes, evidence=evidence
            )
            label = "Remembered" if created else "Already remembered"
            print(f"{label} [{memory.memory_id}] ({memory.kind})")
            if memory.supersedes:
                print(f"Supersedes [{memory.supersedes}]")
            # Printed from the PERSISTED memory, not from `args.evidence`:
            # what matters is the provenance Cortex actually holds, which
            # for an "Already remembered" collapse is the existing memory's
            # own trail, and which the Core may have deduplicated. Ids are
            # canonical Cortex identities (structure), like `memory_id`
            # above.
            for evidence_id in memory.evidence_ids:
                print(f"Evidence [{evidence_id}]")
            return 0

        if args.command == "evidence":
            if args.evidence_command == "add":
                cx = Cortex.discover()
                evidence = cx.add_evidence(args.content, kind=args.kind)
                print(f"Evidence [{evidence.evidence_id}] ({evidence.kind})")
                return 0
            parser.error(f"unknown evidence command {args.evidence_command!r}")
            return 2

        if args.command == "learn":
            cx = Cortex.discover()
            # (A25.1) Same A20 pattern as `remember`: every cited id --
            # provenance and supporting alike -- is resolved through the
            # Core's own lookup BEFORE `learn()` is called, so an unknown
            # id fails the whole command with zero Lesson recorded rather
            # than after a partial write. Kept as two separate lists so
            # the provenance/supporting distinction the public API makes
            # is never collapsed here.
            evidence = [cx.get_evidence(evidence_id) for evidence_id in (args.evidence or ())]
            supporting_evidence = [
                cx.get_evidence(evidence_id) for evidence_id in (args.supporting_evidence or ())
            ]
            memory = cx.learn(
                args.text,
                evidence=evidence,
                supporting_evidence=supporting_evidence,
                verified=args.verified,
            )
            state = "verified" if memory.epistemic_state == "verified" else "candidate"
            print(f"Learned [{memory.memory_id}] ({state})")
            # Printed from the PERSISTED memory, exactly like `remember`
            # above: what matters is the provenance Cortex actually holds.
            for evidence_id in memory.evidence_ids:
                print(f"Evidence [{evidence_id}]")
            for evidence_id in memory.supporting_evidence_ids:
                print(f"Supporting evidence [{evidence_id}]")
            return 0

        if args.command == "recall":
            cx = Cortex.discover()
            results = cx.recall(args.query, limit=args.limit, include_superseded=args.include_superseded)
            if not results:
                print("No memories found.")
                return 0
            for memory in results:
                print(f"[{memory.memory_id}] {_safe(memory.content)}")
            return 0

        if args.command == "timeline":
            cx = Cortex.discover()
            history = cx.timeline(kind=args.kind)
            if not history:
                print("No history found.")
                return 0
            current_ids = {memory.memory_id for memory in cx.state(kind=args.kind)}
            for memory in history:
                status = "current" if memory.memory_id in current_ids else "superseded"
                print(f"[{memory.memory_id}] ({status}) {_safe(memory.content)}")
            return 0

        if args.command == "attempt":
            cx = Cortex.discover()
            attempt = cx.record_attempt(task=args.task, approach=args.approach, outcome=args.outcome)
            print(f"Recorded attempt [{attempt.attempt_id}] ({attempt.outcome})")
            return 0

        if args.command == "preflight":
            cx = Cortex.discover()
            result = cx.preflight(args.task)
            _print_retrieval(result.retrieval)
            if result.is_empty():
                print("No relevant experience found.")
                return 0
            if result.known_failures:
                print("KNOWN FAILURES")
                for attempt in result.known_failures:
                    print(f"- [{attempt.attempt_id}] {_safe(attempt.task)} -- {_safe(attempt.approach)}")
            if result.root_causes:
                print("ROOT CAUSES")
                for memory in result.root_causes:
                    print(f"- [{memory.memory_id}] {_safe(memory.content)}")
            if result.verified_lessons:
                print("VERIFIED LESSONS")
                for memory in result.verified_lessons:
                    print(f"- [{memory.memory_id}] {_safe(memory.content)}")
            if result.recommended_validation:
                print("RECOMMENDED VALIDATION")
                for evidence in result.recommended_validation:
                    print(f"- [{evidence.evidence_id}] {_safe(evidence.content)}")
            if result.invariants:
                print("INVARIANTS")
                for memory in result.invariants:
                    print(f"- [{memory.memory_id}] {_safe(memory.content)}")
            if result.pending:
                print("PENDING")
                for memory in result.pending:
                    print(f"- [{memory.memory_id}] {_safe(memory.content)}")
            if result.open_invalidations:
                print("OPEN INVALIDATIONS")
                for memory in result.open_invalidations:
                    print(f"- [{memory.memory_id}] {_safe(memory.content)}")
            if result.open_conflicts:
                print("OPEN CONFLICTS")
                for view in result.open_conflicts:
                    # `PreflightConflict.memories` is already ordered like
                    # `conflict.memory_ids` (see `_preflight.py`) and needs
                    # no second lookup -- both Memory objects came back
                    # inside `result` itself. Both contents are untrusted
                    # data and go through the same `terminal_safe_text`
                    # boundary as every other field here (A14.S).
                    memory_a, memory_b = view.memories
                    print(f"- [{memory_a.memory_id}] {_safe(memory_a.content)}")
                    print(f"  <-> [{memory_b.memory_id}] {_safe(memory_b.content)}")
            return 0

        if args.command == "context":
            cx = Cortex.discover()
            compiled = cx.context(args.task, budget=args.budget)
            # `CompiledContext.render()` is itself the rendering boundary
            # (see `_context.py`): the text it returns is already
            # terminal-safe, unlike `Preflight`'s raw fields above, so no
            # second `_safe()` pass belongs here.
            print(compiled.render())
            return 0

        if args.command == "export":
            cx = Cortex.discover()
            compiled = cx.context(args.task, budget=args.budget)
            # `render_portable()` is itself the rendering boundary (see
            # `_context.py`): stdout gets ONLY the portable payload, with
            # no `Retrieval:` line and no CLI-added status text, so
            # `cortex export "<task>" > context.txt` and `| cat` both
            # capture exactly the payload an external target would read.
            print(compiled.render_portable())
            return 0

        if args.command == "skills":
            cx = Cortex.discover()
            items = cx.skills()
            if not items:
                print("No skills recorded.")
                return 0
            for skill in items:
                print(f"[{skill.skill_id}] ({skill.verification_state}) {_safe(skill.name)}")
            return 0

        if args.command == "guard":
            cx = Cortex.discover()
            result = cx.guard(args.action)
            _print_retrieval(result.retrieval)
            if result.is_empty():
                print("No known Cortex warnings for this action.")
                return 0
            print("CORTEX WARNING")
            if result.known_failures:
                print()
                print("Known failure:")
                for attempt in result.known_failures:
                    print(f"- {_safe(attempt.approach)}")
            if result.applicable_skills:
                print()
                for skill in result.applicable_skills:
                    label = "verified" if skill.verification_state == "verified" else "candidate"
                    print(f"Applicable skill ({label}):")
                    print(f"- {_safe(skill.name)}")
            if result.recommended_validation:
                print()
                print("Recommended validation:")
                for evidence in result.recommended_validation:
                    print(f"- {_safe(evidence.content)}")
            return 0

        if args.command == "seed":
            cx = Cortex.discover()
            if not args.paths:
                # Discovery only: this branch must never write. It opens
                # no store, creates no `memory.db`, and records nothing --
                # it reports what COULD be seeded and stops.
                candidates = cx.seed_candidates()
                if not candidates:
                    print("No project context candidates found.")
                    return 0
                print("Project context candidates:")
                for candidate in candidates:
                    print(f"- {_safe(candidate)}")
                print("Nothing was recorded. Run 'cortex seed <path>...' to record them.")
                return 0
            results = cx.seed(args.paths)
            for result in results:
                # Paths are filesystem-derived and are rendered as data;
                # the status word is Cortex's own vocabulary (structure).
                print(f"{result.status} {_safe(result.source.path)}")
            # Only claim something was recorded when something was: an
            # all-`unchanged` run writes nothing, and saying otherwise
            # would be the same lie `remember` avoids with "Already
            # remembered" (A17).
            if any(result.status != SEED_UNCHANGED for result in results):
                # Says plainly that a copy of the text now lives in
                # `.cortex/`: the user chose which files to observe, and
                # what Cortex keeps of them should not have to be inferred
                # from documentation.
                print("Recorded as project evidence, not verified knowledge.")
                print("Document content is stored locally in .cortex/.")
            return 0

        if args.command == "sources":
            cx = Cortex.discover()
            items = cx.sources()
            if args.path is None:
                if not items:
                    print("No project sources recorded.")
                    return 0
                # Compact listing: latest observation only, never the
                # stored documents. Dumping every snapshot here would make
                # the overview unusable on any real workspace.
                for source in items:
                    observation = source.latest_observation
                    print(f"[{source.source_id}] {_safe(source.path)}")
                    print(
                        f"  observed {observation.observed_at.isoformat()} | "
                        f"{observation.size_bytes} bytes | digest {observation.digest[:12]} | "
                        f"{len(source.observations)} observation(s)"
                    )
                return 0

            # Inspection of one Source. `sources()` is the only lookup the
            # public API offers at this scale (A19.1 adds no `get_source`),
            # so the match happens here, against the canonical
            # workspace-relative path the user can read in the listing.
            matches = [source for source in items if source.path == args.path]
            if not matches:
                print(f"No source recorded for {_safe(args.path)}.")
                return 1
            (source,) = matches
            print("SOURCE")
            print(f"  path: {_safe(source.path)}")
            print(f"  source_id: {source.source_id}")
            print(f"  first observed: {source.first_observed_at.isoformat()}")
            print(f"OBSERVATIONS ({len(source.observations)}, oldest first)")
            for position, observation in enumerate(source.observations, start=1):
                print(f"  {position}. observed {observation.observed_at.isoformat()}")
                print(f"     digest: {observation.digest}")
                print(f"     size: {observation.size_bytes} bytes")
                print(f"     evidence: {observation.evidence_id}")
                print("     DOCUMENT CONTENT")
                evidence = cx.get_evidence(observation.evidence_id)
                # The document is untrusted data and may hold newlines,
                # ESC sequences or bidi controls. It is split on its OWN
                # newlines and each resulting line is both prefixed by
                # structure the CLI emits and rendered through the terminal
                # boundary, so no line of output can be mistaken for
                # Cortex's own -- while the canonical Evidence keeps every
                # byte verbatim (see `_terminal.py`: sanitize on output,
                # never on storage).
                for line in evidence.content.split("\n"):
                    print(f"     | {_safe(line)}")
            return 0

        if args.command == "semantic":
            if args.semantic_command == "setup":
                cx = Cortex.discover()
                result = cx.semantic_setup()
                # [A16.3] `provider`/`model_id`/`model_revision` are all
                # derived from module constants now that the model repo,
                # its revision and the artifact are pinned (`model_id` is
                # the full `repo@revision#artifact` identity). `_safe()`
                # is kept on the revision anyway: this is output-rendering
                # code, and it should not be the place that has to be
                # right about where a value came from.
                print(f"Semantic model: {result.provider}/{result.model_id}")
                revision = _safe(result.model_revision) if result.model_revision else None
                print(f"Model revision: {revision or 'unknown'}")
                print(f"Dimensions: {result.dimensions} ({result.normalization})")
                print(
                    f"Indexed: {result.attempt_count} attempts, {result.memory_count} memories, "
                    f"{result.skill_count} skills"
                )
                print("Semantic index ready.")
                return 0
            parser.error(f"unknown semantic command {args.semantic_command!r}")
            return 2

        parser.error(f"unknown command {args.command!r}")
        return 2
    except (CortexError, ValueError) as exc:
        # Error text is structure the codebase writes, but it interpolates
        # ids, kinds and paths that came from outside -- rendered as data.
        print(f"cortex: error: {_safe(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
