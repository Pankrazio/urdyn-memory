# Backup, restore and recovery

Backups are created and restored through the Python module `urdyn.backup`. There is no CLI command for them yet.

```python
from urdyn import backup

backup.create_backup("/path/to/project", "/backups/project.zip")
backup.inspect_backup("/backups/project.zip")             # JSON string: format, engine version, has_memories
backup.restore_backup("/path/to/project", "/backups/project.zip")
backup.recover_interrupted_restore("/path/to/project")
```

Failures raise `urdyn.backup.BackupError` (a `ValueError`); when a storage check failed, the original
`UrdynStorageError` is attached as `__cause__`.

## What a backup contains

A zip archive with `backup.json` (format version, engine version, size and SHA-256 of every member) and the canonical
files `.urdyn/manifest.json` and `.urdyn/memory.db`. The derived semantic index is not included; it is rebuilt with
`urdyn semantic setup`. `create_backup` refuses a Brain root that contains anything other than `.urdyn/` and
`.gitignore`, symlinks, and SQLite side files (an active writer is never copied). A manifest-only archive (from Brains
that never wrote a store) is accepted; an empty store is materialized in staging only during restore or inspection.

## Restore

Under the destination's lifecycle lock: the archive is verified (members, sizes, digests, engine validation),
extracted into an exclusive staging directory, hashed again and fsynced; the previous Brain is moved aside to
`.<name>.rollback`, the staging directory is activated, validated, and rolled back on error. The previous Brain is
kept.

- The restored Brain gets the same Git protection as `create` (the root's existing `.gitignore` is carried over byte
  for byte, `.urdyn/` is ensured and checked with Git) before any canonical byte is written to staging.
- Restore is refused if the destination holds a valid manifest with a **different** identity than the archive's.
  It is allowed onto a new destination, onto the same Brain, and onto a Brain whose manifest is missing or malformed
  (restore remains the repair path). An archive pairing one Brain's manifest with another Brain's store is refused.
- A crash after the previous Brain was moved aside but before activation is repaired by
  `recover_interrupted_restore`, which validates and restores the rollback; after activation it validates the active
  Brain. Leftover staging is preserved and blocks another restore until handled explicitly.

## Interrupted creation

If `create` is interrupted, `.urdyn.creating` remains and `open`/`init` refuse with `UrdynStorageUnavailableError`.
`Urdyn.recover_creation(path)` validates a complete staging area (including its identity binding) and publishes it;
an incomplete or ambiguous one is preserved and refused.

## Search index maintenance

A damaged full-text index is recovered with a safety copy first; see [search-index.md](search-index.md). Safety copies
live in `<parent of root>/.urdyn-maintenance/` and are not removed automatically.

## Durability limits

`fsync` is called at every publication point (files and directories). Tests simulate process termination; they are
not a proof of behaviour under power loss.
