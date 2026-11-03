# Getting started

## Install

```bash
pip install urdyn-memory                 # base engine: no runtime dependencies, works offline
pip install "urdyn-memory[semantic]"     # optional semantic retrieval (ONNX Runtime, tokenizers, ...)
```

Python 3.12 or later is required. See [compatibility.md](compatibility.md) for supported platforms.

## Create a Brain

A *Brain* is the `.urdyn/` directory that holds a project's memory.

```bash
cd my-project
urdyn create general      # explicit creation, also in a project that already has files
urdyn status
```

`urdyn init` is the alternative for an empty or new directory: it creates a Brain there, and otherwise re-opens the
existing one. In a non-empty directory without a Brain it refuses and points to `urdyn create`.

Profiles: `general`, `dev` and `lab`. They share the same store and retrieval rules; `dev` additionally enables
no-path seed discovery and starts the local project watcher (stop it with `urdyn watch stop`).

## Record and retrieve

```bash
urdyn remember "SQLite is the canonical project store." --kind decision
urdyn recall "SQLite"
urdyn attempt --task "migrate the schema" --approach "apply the migration in place" --outcome failed
urdyn learn "Run migrations in a transaction."
urdyn preflight "migrate the schema"
urdyn context "migrate the schema"
```

The same from Python:

```python
from urdyn import Urdyn

brain = Urdyn.create("my-project", "general")      # or Urdyn.open("my-project") / Urdyn.discover()
decision = brain.remember("SQLite is the canonical project store.", kind="decision")
for memory in brain.recall("SQLite"):
    print(memory.memory_id, memory.content)

brain.record_attempt(task="migrate the schema", approach="apply the migration in place", outcome="failed")
brain.learn("Run migrations in a transaction.")
print(brain.preflight("migrate the schema").known_failures)
print(brain.context("migrate the schema").render())
```

## Correct, don't overwrite

Memories are append-only. A new decision supersedes an old one; history is kept:

```python
newer = brain.remember("PostgreSQL is the canonical store.", kind="decision", supersedes=decision.memory_id)
[m.content for m in brain.state(kind="decision")]     # current: ["PostgreSQL is the canonical store."]
[m.content for m in brain.timeline(kind="decision")]  # history: both, oldest first
```

## Next steps

- [Python API](python-api.md) and [CLI reference](cli.md)
- [Integrating with AI agents](agent-integration.md)
- [Backup and recovery](backup-and-recovery.md)
- [Storage contract](storage-contract.md) and [architecture](architecture.md)
