"""Write one synthetic Brain with HISTORICAL Urdyn/Cortex code.

Run with the historical package on `sys.path` (see README.md in this
directory), never with the current one:

    python generate_legacy_brain.py <package> <brain-directory> > generate.json

`<package>` is `cortex_memory` (pre-0.1.0 development builds) or `urdyn`
(released versions). The Brain receives two or three notes, one superseding
another when that build supports supersession. The JSON written to stdout
lists the ids written and, when the build has `state()`, the current ids.
"""

import importlib
import inspect
import json
import sys
from pathlib import Path

package, root = sys.argv[1], Path(sys.argv[2])
module = importlib.import_module(package)
Engine = getattr(module, "Cortex", None) or getattr(module, "Urdyn")
cx = Engine.init(root, "general")
record = {"engine": Engine.__name__, "package": package, "written": []}
a = cx.remember("Synthetic fact alpha: the build uses make.")
b = cx.remember("Synthetic fact beta: tests run with pytest.")
record["written"] += [a.memory_id, b.memory_id]
if "supersedes" in inspect.signature(Engine.remember).parameters:
    c = cx.remember("Synthetic fact alpha v2: the build uses ninja.", supersedes=a.memory_id)
    record["written"].append(c.memory_id)
    record["supersession"] = True
record["state"] = [m.memory_id for m in cx.state()] if hasattr(cx, "state") else "n/a (no state() in this build)"
print(json.dumps(record, sort_keys=True))
