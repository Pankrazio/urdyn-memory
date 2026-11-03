# Storage contract

This document specifies how Urdyn creates, opens, protects and coordinates a Brain on disk. It is normative for the
engine: behaviour described here is covered by the test suite.

## Layout

A Brain lives in a project directory (the *root*):

```
<root>/
  .gitignore              # gains a `.urdyn/` entry (see "Git protection")
  .urdyn/
    manifest.json         # schema_version, profile, cortex_id (the Brain identity, 32 hex digits)
    memory.db             # canonical SQLite store, user_version 7
    semantic_index.db     # optional, derived (see "Semantic retrieval" in the README)
<parent of root>/
  .urdyn-locks/           # lifecycle lock files (empty), self-ignoring for Git
```

`memory.db` is the only canonical data. Everything else under `.urdyn/` other than the manifest is derived and can
be rebuilt from it. The manifest key `cortex_id` keeps its historical name for compatibility.

## Creating, initializing and opening

| Operation | Behaviour |
|---|---|
| `Urdyn.create(path, profile)` / `urdyn create` | Explicitly creates a new Brain, also inside a non-empty project. Refuses if a Brain, a creation staging area or an interrupted restore is already there. Manifest and store are written in a staging directory (`.urdyn.creating`), synced, then published with a rename. Residues are never deleted automatically. |
| `Urdyn.init(path, profile)` / `urdyn init` | Creates a Brain only in an absent or empty directory; otherwise opens the existing Brain (the profile must match). A non-empty directory without a Brain is refused with `UrdynNotFoundError` naming the remedy (`create`): it cannot be told apart from a project whose Brain was lost. |
| `Urdyn.open(path)` / `Urdyn.discover()` | Requires a valid manifest and canonical store at the current schema. Never creates tables, identities or databases, and never migrates. |
| `Urdyn.recover_creation(path)` | Explicitly publishes a complete staging area left by an interrupted `create`. An incomplete or ambiguous staging area is preserved and refused. |
| `Urdyn.verify_storage()` | Point-in-time, read-only check (schema version, identity binding, `PRAGMA quick_check`, tables, columns). Every public operation runs it itself; it is a diagnostic, not a lasting authorization. |

A store is created only on a new, exclusively created file in unpublished staging; tables, identity binding and the
version stamp are written in one SQLite transaction. Reads never turn a missing store into an empty answer.

## Coordination

Each public operation takes, in this order: a re-entrant per-thread mutex, an exclusive `flock` on the Brain's lock
file, then SQLite connections and transactions. The lock file is
`<parent>/.urdyn-locks/.urdyn-lock-<sha256 of the normalized root path>`; it is never moved by a restore and never
deleted. A composite operation holds the lock until it returns. `preflight`, `context` and `guard` release it while
semantic inference runs (the inference result is published only if the Brain is still in the state it was computed
for). After `fork()` the child resets the registry and closes inherited lock descriptors.

Coordination is cooperative and local (POSIX, verified on Linux). It does not protect against arbitrary writes by
other programs, remote file systems or hardware faults.

## Brain identity

Every Brain created by this version stores, inside `memory.db`, a `brain_identity` table holding one immutable row:
the same identity as the manifest's `cortex_id`. The binding is additive (`user_version` stays 7) so that earlier
versions, 0.4.0 included, keep opening the store; its presence is the contract marker.

- `open`, every operation, `recover_creation`, backup, `inspect_backup` and restore compare the binding with the
  manifest (or with the identity of an open handle) before the integrity scan, without writing.
- A present but empty, ambiguous or malformed table raises `UrdynStorageCorruptError`; a valid binding to another
  identity raises a `UrdynStorageError` subclass reporting the mismatch. Nothing is repaired or rebound.
- Stores without the table (written by releases up to 0.4.0) are *legacy-unbound*: opened and used as before, never
  bound, converted or rewritten.
- Limits: the comparison is a consistency check, not proof of authenticity. A table removed by deliberate SQL makes
  the store indistinguishable from a legacy one, and a full copy of a Brain is a replica of the same identity.

## Git protection

`create`, `recover_creation`, `init` on an existing Brain and `restore_backup` make sure `.urdyn/` cannot be committed
by accident, before anything is staged:

- `.urdyn/` is appended to the root's `.gitignore` without altering existing bytes;
- a `.gitignore` that is a symlink or not a regular file, an explicit `!.urdyn` negation, `.urdyn` files already
  tracked by Git, or files Git would not ignore are refused, and no Brain is created;
- the Git index, configuration and work tree are never modified (`core.fsmonitor` is disabled for the queries);
- without Git the protection is reported as unverified. `open` never writes `.gitignore`.

Lock files live in a directory whose own `.gitignore` ignores everything in it. Lock files that Git already tracks,
and lock files left by earlier builds directly in the parent directory, are reported and never deleted.
`.gitignore` prevents accidental commits only; it is neither encryption nor access control.

## Errors

All errors derive from `UrdynError`. Storage errors derive from `UrdynStorageError`:

| Situation | Exception |
|---|---|
| No `.urdyn` entry | `UrdynNotFoundError` |
| `.urdyn` not a directory; manifest missing, malformed, not UTF-8, or of another schema | `UrdynManifestError` |
| Interrupted creation or restore; permissions; I/O; SQLite runtime problems | `UrdynStorageUnavailableError` |
| Canonical store missing or failing integrity checks | `UrdynStorageCorruptError` |
| Store at another schema version (legacy or future) | `UrdynStorageSchemaError` |
| Brain identity changed under an open handle | `UrdynStorageError` |

Public error messages do not include canonical content. `urdyn` prints errors as `urdyn: error: ...` and exits 1.

## Schema version

The canonical schema is version 7. Other versions are refused with `UrdynStorageSchemaError` and left byte for byte
as they are: there is no implicit migration. See [compatibility.md](compatibility.md).
