"""Command-line interface for the Cortex Memory Engine."""

from __future__ import annotations

import argparse
import sys

from ._errors import CortexError
from ._manifest import CANONICAL_PROFILES
from ._workspace import Cortex


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
            return 0

        parser.error(f"unknown command {args.command!r}")
        return 2
    except CortexError as exc:
        print(f"cortex: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
