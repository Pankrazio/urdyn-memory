# Legacy Brain fixtures

The compatibility tests run on Brains written by the historical code itself, never on schemas rebuilt by hand
from current code. Each archive in `../legacy_cortex_brains/` and `../released_brains/` contains:

- `brain/` — the Brain exactly as the historical code wrote it (`.cortex/` or `.urdyn/` with `manifest.json` and
  `memory.db`, plus `.gitignore`);
- `generate.json` — what the generator recorded (ids written, current ids).

`SHA256SUMS` in each directory pins the archives; `tests/legacy_brains.py` verifies it before every use.

## Regenerating a fixture

Data is synthetic. Regeneration produces new ids, so new archives (and new `SHA256SUMS`) — the tests read the ids
from `generate.json` and do not depend on specific values.

```sh
# 1. export the historical source (a commit or a release tag of this repository)
git archive <commit-or-tag> src | tar -x -C /tmp/historical
# 2. run the generator with ONLY that source on the path (Python 3.12, no third-party packages needed)
mkdir -p /tmp/fixture && cd /tmp/fixture
PYTHONPATH=/tmp/historical/src python /path/to/tests/data/fixture_tools/generate_legacy_brain.py <package> brain > generate.json
# 3. package and pin
tar -czf v<N>.tar.gz --sort=name --owner=0 --group=0 brain generate.json
sha256sum *.tar.gz > SHA256SUMS
```

| Directory | Archive | Source | `<package>` | Store schema |
|---|---|---|---|---|
| `legacy_cortex_brains` | `v1.tar.gz` … `v6.tar.gz` | commits `39fccff`, `35f3523`, `877b815`, `cb4f354`, `cb0b3c8`, `99a5f4e` | `cortex_memory` | 1 … 6 |
| `released_brains` | `rel-<version>.tar.gz` | tags `v0.1.0` … `v0.4.0` | `urdyn` | 7 |
| `released_brains` | `rel-<version>-experience.tar.gz` | tags `v0.1.0` … `v0.4.0`, `generate_release_experience.py` | — | 7 |

Schemas 1–6 were produced only by development builds that predate 0.1.0 (named Cortex, directory `.cortex/`); no
release wrote them. Released Brains have no `brain_identity` table (they are *legacy-unbound*).
