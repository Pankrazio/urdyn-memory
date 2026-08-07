"""Minimal .gitignore safety net for the Cortex workspace directory."""

from __future__ import annotations

from pathlib import Path

_ENTRY = ".cortex/"


def ensure_gitignore_entry(workspace: Path) -> None:
    """Ensure `.cortex/` is ignored, without disturbing existing content."""
    gitignore_path = workspace / ".gitignore"

    if not gitignore_path.exists():
        gitignore_path.write_text(_ENTRY + "\n", encoding="utf-8")
        return

    content = gitignore_path.read_text(encoding="utf-8")
    existing_lines = {line.strip() for line in content.splitlines()}
    if _ENTRY in existing_lines or _ENTRY.rstrip("/") in existing_lines:
        return

    new_content = content
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += _ENTRY + "\n"
    gitignore_path.write_text(new_content, encoding="utf-8")
