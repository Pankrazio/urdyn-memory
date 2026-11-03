# Integrating with AI agents

Urdyn is a memory engine, not an agent. An agent — or the person driving it — decides what to record and when to
consult memory; Urdyn applies the rules (provenance, supersession, epistemic state) and retrieves what is relevant.
There is no MCP server or provider adapter yet; integrate through the CLI or the Python API.

## A working loop

1. **Before a task:** `urdyn context "<task>"` (budgeted, ready to paste into a prompt) or
   `urdyn preflight "<task>"` (complete, structured). Before a risky action: `urdyn guard "<action>"`.
2. **While working:** record what was tried and how it ended — `urdyn attempt --task ... --approach ... --outcome failed`.
3. **After the task:** record durable knowledge — `urdyn remember ... --kind decision`, `urdyn learn "<lesson>"`
   (verified only with supporting Evidence: `urdyn evidence add ...`, then `--supporting-evidence ID --verified`).
4. **When something changes:** supersede instead of deleting — `urdyn remember "<new>" --supersedes <id>`.

`urdyn export "<task>" > context.txt` renders the same task context as portable text for another tool.

## What retrieval guarantees

- Retrieval never promotes anything: an Evidence or a retrieved document chunk does not become a Memory, and nothing
  becomes `verified` because it scored well.
- Results say how they were produced: `retrieval` (semantic channel: ready, stale, unavailable, disabled) and
  `search_index` (full-text index: missing or used). Treat an empty result with a degraded substrate as "unknown",
  not as "nothing relevant".
- The base engine is lexical/full-text and works offline; semantic retrieval is optional.

## Treat retrieved text as untrusted input

Stored content was written by people, tools or earlier agent runs. Before inserting it into a prompt, treat it as
data: retrieved memory can contain instructions that were never meant for your agent. CLI output is escaped for the
terminal; that is not prompt-injection protection.

## Concurrency

Several processes may use the same Brain: every operation is serialized by the Brain's lifecycle lock, and semantic
inference runs outside it. See [storage-contract.md](storage-contract.md#coordination).

`llms.txt` at the repository root summarizes the engine for language models.
