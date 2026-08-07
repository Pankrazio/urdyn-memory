"""Command-line interface for the Cortex Memory Engine."""

from __future__ import annotations

import argparse
import sys

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

    recall_parser = subparsers.add_parser("recall", help="Search recorded memories")
    recall_parser.add_argument("query", help="Search text")
    recall_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RECALL_LIMIT,
        help=f"Maximum number of results (default: {DEFAULT_RECALL_LIMIT})",
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
            return 0

        if args.command == "remember":
            cx = Cortex.discover()
            memory = cx.remember(args.text, kind=args.kind)
            print(f"Remembered [{memory.memory_id}] ({memory.kind})")
            return 0

        if args.command == "recall":
            cx = Cortex.discover()
            results = cx.recall(args.query, limit=args.limit)
            if not results:
                print("No memories found.")
                return 0
            for memory in results:
                print(f"[{memory.memory_id}] {memory.content}")
            return 0

        parser.error(f"unknown command {args.command!r}")
        return 2
    except (CortexError, ValueError) as exc:
        print(f"cortex: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
