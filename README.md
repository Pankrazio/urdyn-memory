# 🧠 Urdyn

> **Models are replaceable. Memory should remain portable.**

Urdyn is a local-first, private-by-default memory engine for projects, people, and AI agents. It keeps durable decisions, evidence, attempts, lessons, and project context in `.urdyn/`, independently of any model or provider.

## ⚡ Quick start

Install Urdyn from PyPI:

```bash
pip install urdyn-memory
```

Or install it from a local source checkout:

```bash
cd /path/to/urdyn-memory
python -m pip install .
```

Then create a Brain inside an existing project:

```bash
cd my-project
urdyn create dev
urdyn status
```

`urdyn init` creates a Brain only in an empty or not-yet-existing directory, and otherwise re-opens the Brain already there. In a non-empty directory without a Brain it refuses and points to `urdyn create`: such a directory cannot be told apart from a project whose Brain was lost, so a Brain is never created there implicitly. (Urdyn 0.4.0 created one; this version deliberately does not.)

The base engine requires Python 3.12+, declares zero mandatory runtime dependencies, and needs no model or API key.

## Why Urdyn?

Provider history belongs to a provider and a session. When the session ends, the model changes, or another tool takes over, the project's operational knowledge should not disappear with it.

**The model is replaceable. The memory is persistent.**

Urdyn is not a conversation archive. It stores explicit, structured records with provenance and current-state rules, then retrieves the material relevant to the task at hand. A new AI session can reconstruct useful project context without depending on a previous provider transcript.

## 🧠 What Urdyn remembers

Urdyn keeps different kinds of information separate instead of flattening everything into chat text:

| Concept | What it represents |
| --- | --- |
| **Memory** | Notes, decisions, root causes, pending work, questions, invariants, environment facts, and lessons |
| **Evidence** | User statements or confirmations, command/test/tool output, observed errors, file references, and document observations |
| **Attempt** | What was tried, how it was tried, and whether it succeeded, failed, or was partial |
| **Skill** | An ordered procedure deliberately promoted from a Lesson; never created automatically |
| **Source** | A project file identity with an append-only history of observations |
| **Current state** | The current projection of Memory after superseded records are excluded; full history remains available |

This preserves decisions, failures, provenance, and lessons as distinct records rather than pretending they all carry the same authority.

## 🔒 Local-first and private by default

- Canonical data lives inside `.urdyn/` in your workspace.
- No account, cloud service, or API key is required for the base engine.
- Base operation does not automatically upload data or download a model.
- `urdyn create`/`urdyn init` add `.urdyn/` to the directory's `.gitignore` (existing rules and line endings are kept, nothing is duplicated, a symlinked `.gitignore` is never written through). When Git is available they ask Git to confirm the Brain's files are ignored, and refuse to create a Brain if Git already tracks files under `.urdyn/` or a rule explicitly un-ignores it; the Git index is never modified. Without Git the entry is written and reported as not verified. A `.gitignore` entry only prevents accidental commits: it is not encryption, access control, or protection of data already pushed.
- Seeding a document stores the observed document content locally in `.urdyn/`; it is not sent elsewhere.

The optional semantic extra is the explicit exception to zero downloads: its setup fetches a pinned embedding model from Hugging Face.

## 🤖 Works with AI tools

Any AI tool or coding agent with shell access can use Urdyn through its public CLI. Tools without shell access can consume context that you export and provide to them manually.

```text
AI / tool
   │
   ▼
Urdyn public CLI / Python API
   │
   ▼
validation · provenance · memory rules
   │
   ▼
.urdyn/
```

This is a generic integration boundary, not an automatic provider integration. Urdyn 0.4.0 does not ship provider-specific adapters, MCP support, autonomous curation, or automatic invocation. AI tools should use the public CLI/API and never edit `.urdyn/` directly.

## Existing project files

Explicit file seeding works in every profile. In a `dev` workspace, `urdyn seed` with no paths lists discovery candidates and records nothing. Name regular UTF-8 text files explicitly to observe them:

```bash
urdyn seed                           # list candidates; record nothing
urdyn seed README.md pyproject.toml  # record specific files
```

Discovery in `dev` is a **bounded recursive walk**: root-level manifest files (`README*`, `LICENSE*`, `pyproject.toml`, ...) plus documentation-like files (`.md`, `.txt`) anywhere in the project tree, at any depth. It stays bounded and privacy-safe by construction, not by asking:

- it never descends into `.git/`, `.venv`/`venv/`, `node_modules/`, `dist/`/`build/`, caches, or Urdyn's own `.urdyn/` directory;
- it honors `.gitignore` and `.git/info/exclude` when present — anything a project already tells Git to forget, Urdyn never surfaces, indexes, or observes;
- it never follows a symlink outside the workspace, never reads a binary or oversized file, and applies a discovery-only sensitive-name filter (`*secret*`, `*password*`, `*token*`, ...) on top of the credential-name guard explicit seeding already has;
- Git is entirely optional: ignore files are read directly as plain text, never through a `git` subprocess, so discovery works identically with or without a repository.

Each seeded file becomes a Source with a document-observation Evidence record. Urdyn keeps the observed text, digest, size, and timestamp, but does not treat the document's claims as verified knowledge. Listing a candidate never records anything by itself — that stays an explicit `urdyn seed <path>...`.

## 👀 Project watcher

The `dev` profile can keep project-document observations current in the background:

```bash
urdyn watch status
urdyn watch start
urdyn watch stop
```

`urdyn init dev` enables and starts the watcher. It watches every already-tracked Source plus everything the same bounded, privacy-filtered discovery above currently proposes — including a newly created file nobody has seeded yet, not only files that already have a tracked history. Already-tracked files are checked on a fast, adaptive cadence (as often as every 2 seconds while active); noticing a brand-new, never-seen file uses a slower cadence (about every 10 seconds), since nothing about a file with no baseline can be lost by finding it a little later. Changes create Source/Observation/Evidence records, never automatic Memory or other canonical knowledge, and remain local. `urdyn watch stop` stops and persistently disables it.

The watcher is validated and supported on Linux in this release. Known 0.4.0 limits:

- Deletions and renames are not tracked. Existing history is retained, and a renamed file begins a new Source history.
- It is not a boot service. After a reboot, the next normal `urdyn` command restarts an enabled watcher and rechecks already tracked files.
- A file first created while the watcher is down is discovered only after it changes again, not retroactively at restart.

## Evidence ≠ Knowledge

**Evidence records what was observed. Memory records what the caller asks Urdyn to treat as knowledge, with an explicit epistemic state.**

Recording Evidence never creates a Memory, Lesson, or Skill automatically. A seeded README is faithful evidence of what that file said at that moment; it is not proof that the README is correct. A new Memory can be `user_asserted`, `inferred`, or `verified`, and `verified` requires explicitly designated supporting Evidence of a qualifying kind. Urdyn enforces that structural gate but does not claim to understand whether the evidence truly proves the conclusion.

## 🧩 Profiles

```bash
urdyn create [general|dev|lab]   # in an existing project
urdyn init [general|dev|lab]     # in an empty or new directory
```

| Profile | Implemented behavior |
| --- | --- |
| **`general`** | Core engine; explicit seed works, but no-path discovery and the watcher are unavailable |
| **`dev`** | Adds no-path project-file discovery and the Linux-validated background watcher |
| **`lab`** | Reserved canonical profile identifier; currently behaves like `general` |

All profiles share the same canonical store, retrieval, preflight, context, and export behavior. Today, the profile changes only no-path seed discovery and watcher availability.

## 📦 Python API

The distribution is `urdyn-memory`, the import package is `urdyn`, and the public workspace class is `Urdyn`:

```python
from urdyn import Urdyn

ud = Urdyn.discover()
ud.remember("SQLite is the canonical project store.", kind="decision")

for memory in ud.recall("SQLite is the canonical project store"):
    print(memory.content)
```

The Python API and the `urdyn` CLI share the core validation and persistence rules. This README shows only the essential entry points; use the public types exported from `urdyn` for library integration.

## Context compilation and export

Before starting work, ask Urdyn for relevant prior experience:

```bash
urdyn preflight "wrap a multi-step migration in one transaction"
urdyn context "wrap a multi-step migration in one transaction"
urdyn export "wrap a multi-step migration in one transaction"
```

`context` compiles a task-aware, character-budgeted working context. `export` renders the same kind of context as portable generic text suitable for redirection or piping:

```bash
urdyn export "<task description>" > context.txt
```

This export is task-scoped context, not a full backup or memory-archive export.

Seeded project documents can contribute task-relevant **Project Evidence** to compiled context through lexical retrieval and, when enabled, semantic retrieval. For large documents, Urdyn retrieves the portions relevant to the task instead of requiring the entire document to fit the context budget. The retrieved text remains document Evidence with its Source provenance: **Source != Evidence != Durable Memory**, and retrieving Evidence never promotes it to Memory.

## Semantic retrieval

The base engine works offline with lexical/full-text retrieval. Semantic retrieval is optional:

```bash
pip install "urdyn-memory[semantic]"
urdyn semantic setup
```

Setup downloads a pinned embedding model ([`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), at a fixed revision; see its model card for its license) and builds a derived local index next to the canonical store. The model is not redistributed with Urdyn. The index is rebuildable; when semantic retrieval is unavailable, canonical data remains intact and Urdyn falls back to lexical retrieval.

For seeded project documents, semantic relevance is computed at the granularity of derived text chunks rather than whole documents: a long document is split into paragraph-sized segments, each gets its own embedding, and a document becomes eligible for retrieval based on the strongest match among its own current chunks. This lets a long, multi-topic document surface on a query that only matches one of its sections, while admission still stays bounded to a small number of distinct documents per query, so one long document cannot crowd out every other seeded document. Chunks are always derived and rebuilt from the same canonical document text; they are never stored as their own kind of record and carry no authority independent of the document they came from.

## External semantic backend (experimental)

`SemanticBackend`, `SemanticBackendIdentity` and `Urdyn.configure_semantic_backend(backend)` let an application supply its own local embedding function. The backend is attached to one `Urdyn` instance and must be configured again after a new `open()` or `init()`. `available()` checks local availability without network access; `encode(texts)` returns one finite, non-zero vector per text. Urdyn validates type, count and width, converts to `float32` rejecting non-finite values and zero rows, normalizes under the versioned contract `l2-scaled-f64-v1`, and keeps managing the index, retrieval and fallback. An index built under the previous `l2` contract is neither read nor updated: rebuild it explicitly with `semantic_setup()`; canonical memory is not touched. If a call's semantic read fails, the answer is entirely lexical and says so (`unavailable`). The built-in ONNX backend remains the default.

Only identities of the MiniLM artifacts at the calibrated revision, with 384 dimensions, are accepted for now. The identity includes the SHA-256 of model and tokenizer and a stable identifier of the encoding policy (batching and padding included), all persisted with the index. Clients must verify the hashes of the actual files before reporting `available()`. The contract does not enable new models with uncalibrated admission thresholds. The `semantic-backend` extra installs only NumPy. This API is experimental and not yet part of a PyPI release.

```python
from urdyn import Urdyn

brain = Urdyn.open("/path/to/brain")
brain.configure_semantic_backend(backend)  # implements SemanticBackend
brain.semantic_setup()                     # rebuilds the derived index
```

## 🛠 Current scope and limitations

Urdyn is alpha software (latest release 0.4.0; this is the 0.4.1 development line). It does not currently include:

- cloud sync;
- a GUI or desktop application;
- native provider adapters or MCP integration;
- autonomous AI-driven memory curation;
- automatic conversation ingestion;
- full memory-archive import/export.

The CLI/API boundary is deliberate: Urdyn provides the memory engine and its rules, while a person or external tool decides what to record and when to consult it.

## 💾 Backup, recovery and compatibility

```python
from urdyn import backup

backup.create_backup("my-project", "my-project-backup.zip")
backup.restore_backup("my-project", "my-project-backup.zip")
```

Restore is staged and atomic, keeps the previous Brain, and refuses an archive of a different Brain. Brains written by Urdyn 0.1.0–0.4.0 open unchanged; development-era Cortex stores are refused, never migrated implicitly. The full-text index can be checked and rebuilt explicitly with `urdyn search-index status|rebuild`, which also recovers a damaged index with a safety copy. See [backup and recovery](docs/backup-and-recovery.md), [search index](docs/search-index.md) and [compatibility](docs/compatibility.md).

## 📚 Documentation

[Getting started](docs/getting-started.md) · [Python API](docs/python-api.md) · [CLI](docs/cli.md) · [AI agents](docs/agent-integration.md) · [Architecture](docs/architecture.md) · [Storage contract](docs/storage-contract.md) · [Security and privacy](docs/security-and-privacy.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security policy](SECURITY.md)

## Development

```bash
uv sync --frozen --all-extras
uv run pytest
HF_HUB_OFFLINE=1 uv run pytest -m real_model
uv build
```

The full test suite exercises the optional semantic backend, so development setup installs the semantic extra. The base package still has no mandatory runtime dependencies. Tests marked `real_model` are skipped unless the pinned model is already in the local cache. See [CONTRIBUTING.md](CONTRIBUTING.md).

Development requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## 🌍 Languages

This document is in English. See [README.it.md](README.it.md) for the Italian version.
