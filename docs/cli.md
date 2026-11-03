# CLI reference

`urdyn` operates on the nearest Brain found walking up from the current directory. `urdyn <command> --help` shows
every option. Errors are printed as `urdyn: error: ...` with exit status 1; output that echoes stored content is
escaped so it is safe to print to a terminal.

| Command | Purpose |
|---|---|
| `urdyn create [general\|dev\|lab]` | Explicitly create a Brain in the current project directory. |
| `urdyn init [profile]` | Create a Brain in an empty or new directory, or re-open the existing one. |
| `urdyn status` | Workspace, profile, identity, counts, semantic state, search-index state, watcher (dev). |
| `urdyn remember TEXT [--kind K] [--supersedes ID] [--evidence ID ...]` | Record a memory. |
| `urdyn evidence add CONTENT [--kind K]` | Record a piece of Evidence and print its id. |
| `urdyn learn TEXT [--evidence ID ...] [--supporting-evidence ID ...] [--verified]` | Record a lesson (candidate, or verified with supporting Evidence). |
| `urdyn recall QUERY [--limit N] [--include-superseded]` | Lexical search over memories. |
| `urdyn timeline [--kind K]` | Recorded history, oldest first. |
| `urdyn attempt --task T --approach A --outcome {succeeded,failed,partial}` | Record an attempt. |
| `urdyn preflight TASK` | Prior experience relevant to a task. |
| `urdyn context TASK [--budget N]` | Budgeted working context for a task. |
| `urdyn export TASK [--for TARGET] [--budget N]` | Portable task context for an external tool. |
| `urdyn guard ACTION` | Known risks and applicable skills for an action. |
| `urdyn skills` | Recorded skills. |
| `urdyn seed [PATH ...]` | Record project files as Sources; without paths, show candidates (`dev`). |
| `urdyn sources [PATH]` | Tracked project files, or one Source's observation history. |
| `urdyn search-index status` | Check the derived full-text index against canonical data (no writes). |
| `urdyn search-index rebuild` | Rebuild it; recovers a damaged index with a safety copy. See [search-index.md](search-index.md). |
| `urdyn semantic setup` | Download the pinned embedding model (one time) and build the semantic index. |
| `urdyn watch status\|start\|stop` | Control the `dev` project watcher. |

`search-index status|rebuild` locate the Brain without opening it, so they also work when a damaged index makes other
commands refuse the Brain. Backups are available from Python only ([backup-and-recovery.md](backup-and-recovery.md)).
