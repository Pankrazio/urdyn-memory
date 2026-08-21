# Urdyn Memory Engine

A local-first, persistent, structured, model-independent memory engine for humans, AI systems, and agents.

Models are replaceable. Memory should remain portable.

## Why Urdyn

AI sessions, models, and tools change constantly — a new session starts, a model gets swapped, a different agent picks up the work. What should not be lost every time that happens is the project's actual accumulated knowledge: decisions made and why, root causes found, lessons verified against real evidence, and the record of what was tried and failed. Urdyn is a small, independent store for that knowledge, so it outlives any single tool or model.

## Core principles

- **Local-first** — your memory lives in a workspace directory on disk, under your control.
- **Private-by-default** — nothing leaves the machine as part of normal operation; there is no account and no cloud dependency.
- **Model-independent** — the canonical store, retrieval, and context compilation all work without any AI model. Semantic retrieval is an optional add-on, not a requirement.
- **Evidence is not canonical truth** — Evidence records what was observed (a command's output, a user's confirmation, a file's content); Memory records what is believed. A memory only becomes `verified` when it names supporting Evidence of a kind strong enough to justify that (a deliberate gate, not a formality).
- **Canonical history is preserved** — memories can be superseded or invalidated, but the history of what was recorded, and when, is never silently rewritten.
- **No LLM required** — the base engine (recording, search, preflight, context compilation, export) runs on plain lexical/full-text retrieval with zero model downloads.

## Installation

From PyPI, once the package is published:

```bash
pip install urdyn-memory
```

From an existing local source checkout:

```bash
cd /path/to/urdyn-memory
python -m pip install .
```

Semantic retrieval is an optional extra (see [Semantic retrieval](#semantic-retrieval)):

```bash
pip install "urdyn-memory[semantic]"
```

For a local source checkout, use `python -m pip install ".[semantic]"` instead.

Requires Python 3.12+.

## Quick start

```bash
mkdir my-project && cd my-project
urdyn init dev
```

Record something worth remembering, backed by real evidence:

```bash
urdyn evidence add "Migration 042 failed halfway through on staging, leaving the schema partially updated." --kind error_observation

urdyn evidence add "Rerunning the migration inside a single transaction on staging: confirmed no partial schema state after a forced failure." --kind user_confirmation

urdyn learn "Always wrap multi-step schema migrations in a single transaction so a failure leaves the schema unchanged." \
  --supporting-evidence <evidence-id-from-the-user_confirmation-step> --verified
```

Before starting related work later, check what Urdyn already knows:

```bash
urdyn preflight "wrap a multi-step schema migration in a single transaction"
```

Compile a budgeted working context for the same task:

```bash
urdyn context "wrap a multi-step schema migration in a single transaction"
```

Export that same context in a portable form:

```bash
urdyn export "wrap a multi-step schema migration in a single transaction"
```

Run `urdyn --help` for the full command list (`remember`, `recall`, `timeline`, `attempt`, `skills`, `guard`, and more).

## Using Urdyn with your AI assistant

You don't need to memorize the `urdyn` command set yourself. Any coding agent or AI tool that has shell access to your project workspace can drive the `urdyn` CLI directly — this is a generic CLI integration path, not a provider-specific adapter. It works the same way with any agent capable of running shell commands, because it is just the same public CLI a human would type.

The agent should stick to the public CLI/API surface and never edit `.urdyn/` directly — `urdyn --help` and each subcommand's `--help` are enough for it to pick the right primitive (`remember`, `learn`, `evidence add`, `preflight`, `context`, `export`, and so on).

A simple instruction to your AI assistant is enough to establish this, for example:

> Use Urdyn as the persistent memory for this project. Use the `urdyn` CLI and never edit `.urdyn/` directly. Before significant work, consult the relevant Urdyn context. During work, record meaningful evidence, attempts, and durable project knowledge when appropriate. After verified outcomes, preserve reusable lessons. Use `urdyn --help` when needed.

If the model you're working with has no shell access, you can instead compile the context yourself and pass it along:

```bash
urdyn export "<task description>"
```

and give the resulting portable, compiled context to the model as part of your prompt.

This is a plain CLI integration boundary, not a native integration: Urdyn does not ship Claude/Codex/ChatGPT-specific adapters, MCP support, autonomous memory curation, or automatic invocation, and using an AI assistant to drive it does not imply the memory ends up organized any better than if a human had typed the same commands. The boundary is always:

```
AI / tool -> Urdyn public CLI/API -> Urdyn validation/policies -> .urdyn/
```

The model interacts only through the public CLI/API; it never manipulates the storage or internal files of `.urdyn/` directly.

## Existing projects

`urdyn seed` (available in the `dev` profile) lets Urdyn become aware of files already in your project:

```bash
urdyn seed                    # no paths: list discovery candidates, record nothing
urdyn seed README.md src/     # record specific paths
```

Seeded files become **Source / Evidence observations** — a record of what a file contained and when it was observed. They do not silently become canonical truth: seeding a file adds provenance Urdyn can later cite, it does not create a verified memory on its own.

## Project watcher (dev profile)

`urdyn init dev` also enables a local background process that keeps tracked project documents up to date automatically, so you do not have to remember to re-run `urdyn seed` after every edit:

```bash
urdyn watch status   # state, pid, last observation, tracked sources missing on disk
urdyn watch start    # enable + start (also what "init dev" does)
urdyn watch stop     # stop the process and disable it persistently
```

It only ever watches paths that are already a tracked Source, plus the same discovery allowlist `urdyn seed` uses — never a scan of the whole project. It never creates a Memory or any other canonical belief; it produces the same Source/Evidence observations `urdyn seed` does, and nothing leaves this machine. Three known V1 limits: file deletions and renames are not tracked (a deleted file's existing history is kept, and a renamed file starts a new one); the watcher does not restart on its own after a reboot — the next `urdyn` command in that workspace restarts it and re-checks every already-tracked file for changes it missed; and a file created while the watcher was not running is picked up only the next time it changes, not retroactively at restart. Validated on Linux; on other platforms `urdyn watch status` reports it as unavailable rather than claiming support that has not been tested there.

## Context compilation

```bash
urdyn context "<task description>"
```

Given a task description, Urdyn retrieves the memories, lessons, and evidence relevant to it and compiles them into a single working context under a character budget (`--budget`, default 4000), prioritizing the most relevant and canonical material first.

## Portable generic export

```bash
urdyn export "<task description>"
urdyn export "<task description>" > context.txt
urdyn export "<task description>" | some-other-tool
```

`export` compiles the same kind of task-aware working context as `context`, formatted as a portable, generic block of text (`--for generic`, currently the only export target) meant to be piped or redirected into another tool or prompt. It is a compiled, task-scoped context — not a full memory archive or database export.

## Semantic retrieval

Semantic (embedding-based) retrieval is **optional**. The base engine works fully offline with lexical/full-text search and requires no model download.

To enable it for a workspace:

```bash
pip install "urdyn-memory[semantic]"
urdyn semantic setup
```

This downloads and pins a specific sentence-transformers model on first use and builds a local semantic index next to your memory store. The index is derived, rebuildable, and safe to delete — Urdyn falls back to lexical-only retrieval if it is missing.

## Privacy

- Memory is stored locally, in a workspace directory (`.urdyn/`) on your machine.
- No account or sign-up is required.
- No cloud service is required for base operation (recording, search, preflight, context compilation, export).
- No LLM or AI model is required for base operation.
- `urdyn init` adds `.urdyn/` to the workspace's `.gitignore` automatically, so your memory store is not committed to the project's own repository by default.

Enabling the optional semantic extra downloads a model from Hugging Face on first setup; base operation does not.

## Profiles

```bash
urdyn init [general|dev|lab]
```

- **`dev`** — the profile with the most concrete behavior today: it enables `urdyn seed` project-file discovery and starts the [background project watcher](#project-watcher-dev-profile). It is also the profile most exercised by the test suite.
- **`general`** — the default profile for non-development use of Urdyn; behaves like the core engine without automatic project-file discovery.
- **`lab`** — a canonical profile identifier reserved for experimental or exploratory use; currently behaves the same as `general`.

All three profiles share the same canonical store, retrieval, preflight, context, and export behavior. The profile currently affects two things: whether `urdyn seed` (with no paths) can discover project files automatically, and whether the background project watcher runs.

## Current scope / limitations

Urdyn v1 does not include:

- MCP integration
- Cloud sync
- A GUI or desktop app
- Built-in AI provider adapters
- Autonomous AI-driven curation of memory
- Full memory-archive import/export (only the task-scoped `export` above)

## Development

```bash
uv sync
uv run pytest                          # full test suite
uv run pytest -m real_model            # tests that need the cached semantic model (skipped otherwise)
uv build                               # build wheel + sdist
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Italian documentation

See [README.it.md](README.it.md) for the Italian version of this document.
