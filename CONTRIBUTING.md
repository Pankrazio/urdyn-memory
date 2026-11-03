# Contributing

Issues and pull requests are welcome.

## Development setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --all-extras      # reproducible environment from uv.lock
uv run pytest                      # full suite
uv build                           # wheel and sdist
```

Tests marked `real_model` need the pinned embedding model in the local Hugging Face cache and are skipped
otherwise; populate the cache once with `urdyn semantic setup` in a scratch workspace, then run
`HF_HUB_OFFLINE=1 uv run pytest -m real_model`. The suite never downloads anything by itself. Some tests compare with
release `0.4.0` through `git show`; they are skipped when the Git history is not available.

## Guidelines

- Keep the storage guarantees in [docs/storage-contract.md](docs/storage-contract.md): no implicit migration, no
  silent repair, typed errors, canonical data never rewritten by a read.
- Behaviour changes come with tests that fail without the change. Tests use synthetic data in temporary directories
  only; never a real Brain.
- Compatibility fixtures in `tests/data/` are written by the historical code itself
  ([tests/data/fixture_tools/README.md](tests/data/fixture_tools/README.md)); do not hand-edit them.
- Changes to `urdyn.__all__` are API changes: update the pinned list in the tests and the documentation together.
- Comments and docstrings explain the code and its invariants for a reader without project history.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
