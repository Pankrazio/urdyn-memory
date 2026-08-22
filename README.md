# 🧠 Urdyn

> **Models are replaceable. Memory should remain portable.**

Urdyn is a local-first, private-by-default memory engine for projects, people, and AI agents. It keeps durable decisions, evidence, attempts, lessons, and project context in `.urdyn/`, independently of any model or provider.

## ⚡ Quick start

`urdyn-memory` is not published on PyPI yet. Once it is available, install the distribution with:

```bash
pip install urdyn-memory
```

Until then, install from an existing local source checkout:

```bash
cd /path/to/urdyn-memory
python -m pip install .
```

Then initialize Urdyn inside a project:

```bash
cd my-project
urdyn init dev
urdyn status
```

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
- `urdyn init` adds `.urdyn/` to the project's `.gitignore` automatically.
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

This is a generic integration boundary, not an automatic provider integration. Urdyn 0.1.0 does not ship provider-specific adapters, MCP support, autonomous curation, or automatic invocation. AI tools should use the public CLI/API and never edit `.urdyn/` directly.

## Existing project files

Explicit file seeding works in every profile. In a `dev` workspace, `urdyn seed` with no paths lists conservative discovery candidates and records nothing. Name regular UTF-8 text files explicitly to observe them:

```bash
urdyn seed                           # list candidates; record nothing
urdyn seed README.md pyproject.toml  # record specific files
```

Each seeded file becomes a Source with a document-observation Evidence record. Urdyn keeps the observed text, digest, size, and timestamp, but does not treat the document's claims as verified knowledge.

## 👀 Project watcher

The `dev` profile can keep project-document observations current in the background:

```bash
urdyn watch status
urdyn watch start
urdyn watch stop
```

`urdyn init dev` enables and starts the watcher. It watches only already tracked Sources plus the same conservative discovery allowlist used by `urdyn seed`; it never scans the whole project. Changes create Source/Observation/Evidence records, never automatic Memory or other canonical knowledge, and remain local. `urdyn watch stop` stops and persistently disables it.

The watcher is validated and supported on Linux in this release. Known 0.1.0 limits:

- Deletions and renames are not tracked. Existing history is retained, and a renamed file begins a new Source history.
- It is not a boot service. After a reboot, the next normal `urdyn` command restarts an enabled watcher and rechecks already tracked files.
- A file first created while the watcher is down is discovered only after it changes again, not retroactively at restart.

## Evidence ≠ Knowledge

**Evidence records what was observed. Memory records what the caller asks Urdyn to treat as knowledge, with an explicit epistemic state.**

Recording Evidence never creates a Memory, Lesson, or Skill automatically. A seeded README is faithful evidence of what that file said at that moment; it is not proof that the README is correct. A new Memory can be `user_asserted`, `inferred`, or `verified`, and `verified` requires explicitly designated supporting Evidence of a qualifying kind. Urdyn enforces that structural gate but does not claim to understand whether the evidence truly proves the conclusion.

## 🧩 Profiles

```bash
urdyn init [general|dev|lab]
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

## Semantic retrieval

The base engine works offline with lexical/full-text retrieval. Semantic retrieval is optional:

```bash
pip install "urdyn-memory[semantic]"
urdyn semantic setup
```

Setup downloads a pinned embedding model and builds a derived local index next to the canonical store. The index is rebuildable; when semantic retrieval is unavailable, canonical data remains intact and Urdyn falls back to lexical retrieval.

## 🛠 Current scope and limitations

Urdyn 0.1.0 is an alpha release. It does not currently include:

- cloud sync;
- a GUI or desktop application;
- native provider adapters or MCP integration;
- autonomous AI-driven memory curation;
- automatic conversation ingestion;
- full memory-archive import/export.

The CLI/API boundary is deliberate: Urdyn provides the memory engine and its rules, while a person or external tool decides what to record and when to consult it.

## Development

```bash
uv sync --extra semantic
uv run pytest
HF_HUB_OFFLINE=1 uv run pytest -m real_model
uv build
```

The full test suite exercises the optional semantic backend, so development setup installs the semantic extra. The base package still has no mandatory runtime dependencies.

Development requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## 🌍 Languages

This document is in English. See [README.it.md](README.it.md) for the Italian version.
