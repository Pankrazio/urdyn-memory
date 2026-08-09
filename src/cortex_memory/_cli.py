"""Command-line interface for the Cortex Memory Engine."""

from __future__ import annotations

import argparse
import sys

from ._attempt import VALID_OUTCOMES
from ._errors import CortexError
from ._manifest import CANONICAL_PROFILES
from ._memory import DEFAULT_KIND, VALID_KINDS
from ._workspace import DEFAULT_RECALL_LIMIT, Cortex


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex Memory Engine")
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

    subparsers.add_parser("skills", help="List recorded skills")

    guard_parser = subparsers.add_parser(
        "guard", help="Check whether prior experience directly bears on an action about to be taken"
    )
    guard_parser.add_argument("action", help="Action about to be taken")

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
            print(f"Initialized Cortex workspace at {cx.path}")
            print(f"Profile: {cx.profile}")
            return 0

        if args.command == "status":
            cx = Cortex.discover()
            print(f"Cortex workspace: {cx.path}")
            print(f"Profile: {cx.profile}")
            print(f"Cortex ID: {cx.cortex_id}")
            print(f"Memories: {cx._count_memories()}")
            print(f"Invariants: {len(cx.state(kind='invariant'))}")
            print(f"Pending: {len(cx.state(kind='pending'))}")
            print(f"Open questions: {len(cx.state(kind='question'))}")
            print(f"Environment facts: {len(cx.state(kind='environment'))}")
            return 0

        if args.command == "remember":
            cx = Cortex.discover()
            memory = cx.remember(args.text, kind=args.kind, supersedes=args.supersedes)
            print(f"Remembered [{memory.memory_id}] ({memory.kind})")
            if memory.supersedes:
                print(f"Supersedes [{memory.supersedes}]")
            return 0

        if args.command == "recall":
            cx = Cortex.discover()
            results = cx.recall(args.query, limit=args.limit, include_superseded=args.include_superseded)
            if not results:
                print("No memories found.")
                return 0
            for memory in results:
                print(f"[{memory.memory_id}] {memory.content}")
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
                print(f"[{memory.memory_id}] ({status}) {memory.content}")
            return 0

        if args.command == "attempt":
            cx = Cortex.discover()
            attempt = cx.record_attempt(task=args.task, approach=args.approach, outcome=args.outcome)
            print(f"Recorded attempt [{attempt.attempt_id}] ({attempt.outcome})")
            return 0

        if args.command == "preflight":
            cx = Cortex.discover()
            result = cx.preflight(args.task)
            if result.is_empty():
                print("No relevant experience found.")
                return 0
            if result.known_failures:
                print("KNOWN FAILURES")
                for attempt in result.known_failures:
                    print(f"- [{attempt.attempt_id}] {attempt.task} -- {attempt.approach}")
            if result.root_causes:
                print("ROOT CAUSES")
                for memory in result.root_causes:
                    print(f"- [{memory.memory_id}] {memory.content}")
            if result.verified_lessons:
                print("VERIFIED LESSONS")
                for memory in result.verified_lessons:
                    print(f"- [{memory.memory_id}] {memory.content}")
            if result.recommended_validation:
                print("RECOMMENDED VALIDATION")
                for evidence in result.recommended_validation:
                    print(f"- [{evidence.evidence_id}] {evidence.content}")
            if result.invariants:
                print("INVARIANTS")
                for memory in result.invariants:
                    print(f"- [{memory.memory_id}] {memory.content}")
            if result.open_invalidations:
                print("OPEN INVALIDATIONS")
                for memory in result.open_invalidations:
                    print(f"- [{memory.memory_id}] {memory.content}")
            return 0

        if args.command == "skills":
            cx = Cortex.discover()
            items = cx.skills()
            if not items:
                print("No skills recorded.")
                return 0
            for skill in items:
                print(f"[{skill.skill_id}] ({skill.verification_state}) {skill.name}")
            return 0

        if args.command == "guard":
            cx = Cortex.discover()
            result = cx.guard(args.action)
            if result.is_empty():
                print("No known Cortex warnings for this action.")
                return 0
            print("CORTEX WARNING")
            if result.known_failures:
                print()
                print("Known failure:")
                for attempt in result.known_failures:
                    print(f"- {attempt.approach}")
            if result.applicable_skills:
                print()
                for skill in result.applicable_skills:
                    label = "verified" if skill.verification_state == "verified" else "candidate"
                    print(f"Applicable skill ({label}):")
                    print(f"- {skill.name}")
            if result.recommended_validation:
                print()
                print("Recommended validation:")
                for evidence in result.recommended_validation:
                    print(f"- {evidence.content}")
            return 0

        if args.command == "semantic":
            if args.semantic_command == "setup":
                cx = Cortex.discover()
                result = cx.semantic_setup()
                print(f"Semantic model: {result.provider}/{result.model_id}")
                print(f"Model revision: {result.model_revision or 'unknown (see A7.4 report)'}")
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
        print(f"cortex: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
