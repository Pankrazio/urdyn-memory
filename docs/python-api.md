# Python API

Distribution `urdyn-memory`, import package `urdyn`. Everything listed in `urdyn.__all__` is public; names starting
with an underscore are internal and may change. The list is pinned by the test suite.

## `Urdyn` — one Brain

| Group | Methods |
|---|---|
| Lifecycle | `Urdyn.create(path, profile)`, `Urdyn.init(path, profile)`, `Urdyn.open(path)`, `Urdyn.discover(start=None)`, `Urdyn.recover_creation(path)`, `verify_storage()`; properties `path`, `profile`, `urdyn_id` |
| Memories | `remember(content, kind=..., supersedes=..., evidence=...)`, `recall(query, limit=20, include_superseded=False)`, `timeline(kind=None)`, `state(kind=None)` |
| Evidence and lessons | `add_evidence(content, kind=...)`, `get_evidence(evidence_id)`, `learn(content, evidence=(), supporting_evidence=(), verified=False, supersedes=None)` |
| Experience | `record_attempt(task=, approach=, outcome=)`, `promote(lesson, name=, purpose=, steps=, conditions=())`, `get_skill(skill_id)`, `skills()` |
| Conflicts | `record_conflict(memory_a, memory_b)`, `conflicts()`, `open_conflicts()` |
| Project sources | `seed(paths)`, `sources()`, `seed_candidates()`, `seed_candidate_report()`, `tracked_scope()`, `discovered_scope()`, `watcher_scope()` |
| Retrieval | `preflight(task)` → `Preflight`, `context(task, budget=...)` → `CompiledContext`, `guard(action)` → `GuardResult` |
| Semantic channel | `semantic_setup()` → `SemanticSetupResult`, `semantic_state()` → `SemanticState`, `configure_semantic_backend(backend)` |
| Full-text index | `search_index_state()` → `SearchIndexState`, `rebuild_search_index()` → `SearchIndexState` |

Every public operation takes the Brain's lifecycle lock and checks that the manifest still names the handle's
identity and profile; a Brain replaced under an open handle by another identity raises `UrdynStorageError` and must
be reopened.

## Records and results

`Memory`, `Evidence`, `Attempt`, `Skill`, `Conflict`, `Source`, `SourceObservation`, `SeedResult` are immutable
records. `Preflight`, `PreflightConflict`, `CompiledContext` (`ContextSection`, `ContextItem`, `DEFAULT_CONTEXT_BUDGET`)
and `GuardResult` are derived, transient results; they are never stored. Their `retrieval` field reports the semantic
substrate used (`SemanticState`) and `search_index` the full-text index (`missing` / `unchecked`), so an empty answer
from a missing substrate is distinguishable from an empty answer from a healthy one.

## Optional semantic backend

`SemanticBackend` is the protocol for supplying your own embedding function (`available()` and `encode(texts)`),
with `SemanticBackendIdentity` describing the representation it produces. Vectors are checked against a numeric
contract (finite, expected width) and normalized by the engine; an index built by a different identity is refused,
never mixed.

## Maintenance (`urdyn.maintenance`)

`inspect_search_index(path)` and `recover_search_index(path)` → `SearchIndexRecovery(state, mode, backup_path)` work
even when `Urdyn.open` refuses a Brain whose only damage is in the full-text index. See
[search-index.md](search-index.md).

## Backups (`urdyn.backup`)

`create_backup(root, destination)`, `inspect_backup(source)`, `restore_backup(root, source)`,
`recover_interrupted_restore(root)`; errors are `BackupError`. See [backup-and-recovery.md](backup-and-recovery.md).

## Errors

`UrdynError` is the base class. `UrdynNotFoundError`, `UrdynAlreadyInitializedError`, `UrdynManifestError`,
`UrdynSourceError`, `UrdynSemanticUnavailableError`, and `UrdynStorageError` with `UrdynStorageUnavailableError`,
`UrdynStorageCorruptError`, `UrdynStorageSchemaError`. Invalid arguments raise `ValueError`. When to expect which:
[storage-contract.md](storage-contract.md#errors).
