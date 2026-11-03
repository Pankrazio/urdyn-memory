# Compatibility

## Platforms

Python 3.12 or later. The lifecycle lock uses `fcntl.flock`: Linux is tested; other POSIX systems such as macOS are
expected to work but are not qualified; native Windows is not supported. The base package has no mandatory runtime
dependencies. The optional `semantic` extra adds ONNX Runtime, `tokenizers`, `huggingface-hub` and NumPy.

## Brains written by earlier versions

| Written by | Store | This version |
|---|---|---|
| Urdyn 0.1.0 – 0.4.0 | schema 7, no `brain_identity` table | **Opened and used as before** (*legacy-unbound*): never bound to an identity, converted or rewritten by an ordinary open; reads change no byte. |
| Development builds before 0.1.0 (named Cortex, directory `.cortex/`) | schemas 1 – 6 | **Not supported.** A `.cortex/` directory is not recognized (`UrdynNotFoundError`). The same files under `.urdyn/` are refused with `UrdynStorageSchemaError`. In both cases nothing is written. |
| A future version | schema > 7 | Refused with `UrdynStorageSchemaError`, untouched. |

There is no implicit schema migration. Schemas 1 – 6 were never produced by a release. The test suite pins this
behaviour on Brains written by the historical code itself (`tests/data/legacy_cortex_brains`,
`tests/data/released_brains`).

## Brains written by this version, opened by earlier versions

The identity binding is additive and `user_version` stays 7, so earlier versions (0.4.0 included) open, read and
write these Brains without altering the binding — they simply do not verify it. Rolling back to an earlier version is
therefore a matter of rebuilding derived data; canonical data never moves.

## Derived indexes

- **Full-text index:** stores written by 0.1.0 – 0.4.0 carry an index this version reports as `usable`.
- **Semantic index:** versions up to 0.4.0 wrote vectors under the normalization identity `"l2"`, which could turn
  finite-but-large rows into zero vectors. This version records a different normalization identity and a
  representation reference, and refuses an `"l2"` index (`urdyn semantic setup` rebuilds it; canonical memory is not
  touched). Conversely, 0.4.0 refuses an index written by this version. Indexes are never relabelled, only rebuilt.

## Running different versions side by side

Release 0.4.0 has no lifecycle lock, so a 0.4.0 process and a process of this version do not exclude each other:
stop running Urdyn processes on a Brain, the `dev` watcher included, before upgrading. Lock files are kept in
`<parent>/.urdyn-locks/`; files named `.urdyn-lock-*` left directly in the parent directory by development builds
are reported, never used or deleted.

## Backup format

Backups are new in this version (`urdyn.backup`, archive format version 1); 0.4.0 had no backup module.
