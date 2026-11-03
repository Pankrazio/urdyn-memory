# Changelog

## 0.4.1.dev1 — unreleased

Development version after 0.4.0. Alpha: APIs may still change before 0.4.1.

### Storage lifecycle

- Explicit Brain lifecycle: `Urdyn.create` / `urdyn create` creates a Brain, also in a non-empty project;
  `init` creates only in an absent or empty directory and otherwise re-opens; `open` never creates or migrates.
  A non-empty directory without a Brain is refused with `UrdynNotFoundError` naming the remedy.
- Creation and restore publish through staging and rename, with `fsync`; interrupted creation is recovered
  explicitly with `Urdyn.recover_creation`.
- Typed, truthful errors: `UrdynStorageUnavailableError`, `UrdynStorageCorruptError`, `UrdynStorageSchemaError`
  (subclasses of `UrdynStorageError`). SQLite and OS errors during store creation are reported as
  `UrdynStorageError` with the original exception as `__cause__`.
- Every operation runs a read-only storage verification (schema, identity, integrity) and is serialized by a
  per-Brain lifecycle lock; lock files live in a self-ignoring `.urdyn-locks/` directory next to the Brain. Paths
  that are not valid UTF-8 are supported.
- Canonical Brain identity: new stores are bound to the manifest's identity (additive, schema version unchanged);
  a store bound to another identity is refused. Stores written by 0.1.0–0.4.0 open unchanged (legacy-unbound).
- Git protection of `.urdyn/` is verified with Git before creating or restoring a Brain.

### Backup and restore

- New `urdyn.backup`: `create_backup`, `inspect_backup`, `restore_backup`, `recover_interrupted_restore` — verified
  archives, staged and atomic restore with rollback, identity checks.

### Full-text search index

- New diagnostic `Urdyn.search_index_state()` / `urdyn search-index status` (`missing`, `usable`, `invalid`,
  `unchecked`, plus FTS5 runtime support) and explicit, transactional, validated-before-commit rebuild
  `Urdyn.rebuild_search_index()` / `urdyn search-index rebuild`. `SearchIndexState` is exported.
- New `urdyn.maintenance`: recovers a Brain refused only because its FTS5 index is damaged, with a verified safety
  copy first.
- Opening and reading never repair the index; `Preflight`, `GuardResult` and `CompiledContext` report the index used
  in a new `search_index` field. `urdyn status` prints a `Search index:` line.

### Semantic retrieval

- Public `SemanticBackend` protocol and `SemanticBackendIdentity` for external embedding backends.
- Numeric contract for embeddings (finite values, expected width, robust normalization); indexes written by 0.4.0
  (`"l2"`) are refused and rebuilt, never mixed.
- Verifiable representation identity recorded with each index.
- Inference runs outside the lifecycle lock; refreshes and `semantic setup` publish atomically and only for the state
  they were computed from. Each call reports its own semantic outcome.

### Compatibility

- Development-era Cortex stores (schemas 1–6, never released) are refused with no change, instead of being migrated
  implicitly. See `docs/compatibility.md`.

## 0.4.0

Last release on PyPI.
