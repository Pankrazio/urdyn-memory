# Architecture

```
            CLI (urdyn)            Python API (urdyn.Urdyn, urdyn.backup, urdyn.maintenance)
                 \                           /
                  +------ Urdyn (one Brain) +
                  | lifecycle lock + identity check per operation
       +----------+------------+--------------------+------------------+
       | canonical store       | derived: FTS5      | derived: semantic | project sources
       | memory.db (SQLite v7) | search_index       | semantic_index.db | (seed, watcher)
       +-----------------------+--------------------+-------------------+
```

## Canonical data

`memory.db` holds the only canonical records: memories, evidence and their links (`related` or `supporting`), the
event log, attempts, skills (steps, conditions, evidence), conflicts, sources and their observations, and the Brain
identity binding. Records are append-only; supersession and invalidation add records instead of rewriting them, and
`state()` is a projection of the event history. Every canonical write and its full-text index row share one
transaction.

## Derived data

- **Full-text index** (`search_index`, FTS5, inside `memory.db`): a projection of canonical text used to widen
  candidates. Diagnosed and rebuilt explicitly, never repaired implicitly — [search-index.md](search-index.md).
- **Semantic index** (`semantic_index.db`, optional): embeddings of memories, attempts, skills and the current chunks
  of seeded documents, built by `urdyn semantic setup` with a pinned model (or an external `SemanticBackend`). Each
  index records the exact representation it was built with; a mismatch makes the channel report itself unavailable
  and the engine answers lexically. Refreshes and rebuilds publish atomically and only for the state they were
  computed from.

Derived data can always be deleted and rebuilt from canonical data; canonical data never depends on it.

## Retrieval

`preflight`, `context` and `guard` combine a structured lexical rule, the full-text channel and, when available, the
semantic channel, then apply per-consumer admission (current, relevant, verified where required). `context` adds a
character budget and de-duplication. Results carry the state of each substrate.

## Coordination and safety

A per-Brain `flock` serializes operations across threads and cooperating processes; semantic inference runs outside
it. Creation, restore and recovery publish by rename after `fsync`, keep residues for explicit handling, and refuse
rather than guess. Details: [storage-contract.md](storage-contract.md), [backup-and-recovery.md](backup-and-recovery.md).

## Project sources and the watcher

`seed` records project files as Sources; each change is a new observation (Evidence with provenance), never an
overwrite. The `dev` profile can run a local background watcher that records changes to tracked files; it never
sends data anywhere.
