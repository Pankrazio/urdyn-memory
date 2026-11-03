# Security and privacy

## Local-first

All data stays in the project's `.urdyn/` directory. The engine makes no network requests, with one explicit
exception: `urdyn semantic setup` downloads the pinned embedding model from the Hugging Face Hub (once; afterwards
retrieval runs offline). There is no telemetry.

## What protects the data, and what does not

- **Git:** `.urdyn/` is kept out of version control by a `.gitignore` entry checked with Git before a Brain is created
  or restored ([storage-contract.md](storage-contract.md#git-protection)). This prevents accidental commits; it is
  not encryption or access control.
- **File permissions:** the Brain is protected by the operating system's permissions only. Data is not encrypted at
  rest. Anyone who can read the files as your user can read the memory.
- **Isolation between Brains:** every operation checks the Brain identity; this prevents mixing two Brains by
  mistake. It is not an authorization mechanism between users of the same account.
- **Safety copies:** recovering a damaged full-text index writes a full copy of `memory.db` to
  `<parent of root>/.urdyn-maintenance/` (mode 0600). Remove those copies when no longer needed.
- **Backups** contain the full canonical store; store them accordingly.

## Untrusted content

Memories, evidence and seeded documents may contain text written by others or by earlier agent runs. Output written
to a terminal is escaped (control sequences cannot be injected); text handed to a language model must still be
treated as untrusted data ([agent-integration.md](agent-integration.md)). Seeding refuses paths outside the workspace,
non-regular files, oversized files and common credential files.

## Reporting a vulnerability

See [SECURITY.md](../SECURITY.md).
