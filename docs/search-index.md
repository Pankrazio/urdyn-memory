# Full-text search index

`search_index` is an SQLite FTS5 table inside `memory.db`. It is a **derived projection** of canonical data —
every memory (superseded ones included), attempt, skill and Source observation — used to widen candidates for
`preflight`, `context` and `guard`. It holds no canonical truth and is not part of the schema version.

## Rules

- **No automatic repair.** `open`, `init` on an existing Brain, `verify_storage` and every read never create,
  rebuild or modify the index. `create` builds it. An index that is missing stays missing until it is rebuilt
  explicitly; meanwhile new writes are not indexed, and the diagnostic reports it.
- **Retrieval reports the index it used.** `Preflight`, `GuardResult` and `CompiledContext` carry `search_index`:
  `missing` (the full-text channel could not contribute) or `unchecked` (present and used, not verified by that
  call). Ranking, thresholds and admission are unchanged; the field is not rendered in `context` output and does not
  consume budget. An index that is present but unusable makes the call fail with `UrdynStorageError`, never return
  an empty answer silently.

## Diagnostic

```python
state = Urdyn.open(".").search_index_state()     # SearchIndexState
state.runtime_fts5                               # does this SQLite runtime provide FTS5?
state.status                                     # "missing" | "usable" | "invalid" | "unchecked"
state.entities                                   # ((entity_type, expected_rows, indexed_rows_or_None), ...)
```

```console
$ urdyn search-index status
Search index: usable; FTS5 runtime: yes
  memory: 1 indexed / 1 expected
  attempt: 0 indexed / 0 expected
  skill: 0 indexed / 0 expected
  source: 0 indexed / 0 expected
```

| Status | Meaning |
|---|---|
| `missing` | No `search_index` object in the store. |
| `usable` | Expected definition, FTS5 `integrity-check` passed, rows identical (as a multiset) to the projection of canonical data, for each of the four entity types. |
| `invalid` | Present but not verifiably usable: other definition, failed integrity check, missing, extra, stale or unknown-type rows, or unreadable. |
| `unchecked` | Present but not verifiable because this SQLite runtime has no FTS5. |

The diagnostic never writes: it runs in a read transaction that is always rolled back. Its cost grows with the
number of records. A failure reading **canonical** data raises `UrdynStorageError`; a failure reading the **index**
only makes it `invalid`. `urdyn status` prints the same summary on its `Search index:` line.

## Rebuilding

```python
Urdyn.open(".").rebuild_search_index()           # returns the SearchIndexState validated before commit
```

```console
$ urdyn search-index rebuild
```

The rebuild runs under the Brain's lifecycle lock, in **one** `BEGIN IMMEDIATE` transaction: drop the existing
projection (only a recognized FTS5 projection — an ordinary table named `search_index`, orphan shadow tables or any
other `search_index*` object are refused and nothing is removed), create it with the expected definition, project
canonical data into it, then validate it (FTS5 `integrity-check` and comparison with the canonical projection). It
commits only if the result is `usable`; any failure or other verdict rolls everything back and raises
`UrdynStorageError` with the cause attached. A process killed mid-way leaves a journal that SQLite rolls back on the
next write access. Canonical tables are only read. With a runtime that has no FTS5 the rebuild is refused with
`UrdynStorageUnavailableError` before anything is written. Concurrent rebuilds are serialized and leave one
complete index.

## Recovering a damaged index

`PRAGMA quick_check` — run by `verify_storage` before every operation — includes FTS5 checks (SQLite ≥ 3.44). A
damaged index therefore makes ordinary operations, `Urdyn.open` included, refuse the Brain with
`UrdynStorageCorruptError`, even when canonical data is intact. That refusal is deliberate and unchanged; the one way
past it, for this cause only, is the maintenance entry:

```python
from urdyn import maintenance

maintenance.inspect_search_index("/path/to/project")      # SearchIndexState; works when open() refuses
result = maintenance.recover_search_index("/path/to/project")
result.mode          # "rebuilt" (ordinary rebuild) or "recovered" (damaged index recovered)
result.backup_path   # safety copy written before a recovery, else None
```

`urdyn search-index status|rebuild` use the same entry and locate the Brain without opening it. The handle methods
`search_index_state()` and `rebuild_search_index()` use it too, bound to the handle's identity and profile.

A recovery proceeds only if all of the following hold, and otherwise fails with nothing changed:

1. The Brain's lifecycle lock is held; manifest, schema version and identity binding check out as for any operation.
2. Every `quick_check` message is an FTS5 message about `search_index`, **and** every canonical table passes its own
   table-scoped `quick_check`. Any other result, or an SQLite error during the scan, is refused exactly as `open`
   refuses it.
3. The objects to remove are a recognized FTS5 projection (see above).
4. A safety copy of the whole database is written first, with SQLite's online backup API (committed WAL content
   included), to a new file in `<parent of root>/.urdyn-maintenance/` (directory 0700 with its own `.gitignore`,
   file 0600, exclusive creation, never overwritten), verified against the source and synced. It is outside the
   Brain root on purpose: an ordinary backup refuses unknown entries in the root. Copies are not deleted
   automatically and contain Brain data.
5. In one `BEGIN IMMEDIATE` transaction: the canonical fingerprint is compared with the copy, the projection is
   dropped, `quick_check` must pass, the index is rebuilt and fully validated, `quick_check` must pass again, and the
   canonical fingerprint, schema version and identity must be unchanged. Only then does it commit.
6. After the commit a fresh connection re-verifies the Brain through the ordinary checks. If the commit itself
   reports an error, success is reported only if that re-verification confirms it; otherwise the error says that
   the outcome could not be confirmed and names the safety copy.

Recognition of FTS5 `quick_check` messages is based on SQLite's wording (verified with SQLite 3.53). Another wording
leads to a refusal, never to an unsafe recovery.
