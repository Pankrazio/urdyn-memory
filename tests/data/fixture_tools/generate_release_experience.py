"""Write one synthetic Brain with a RELEASED Urdyn version: two notes, a
user-asserted lesson and a failed attempt, through that release's public API.

    python generate_release_experience.py <brain-directory> > generate.json
"""

import json
import sys
from pathlib import Path

from urdyn import Urdyn

cx = Urdyn.init(Path(sys.argv[1]), "general")
a = cx.remember("Release-era note about the refresh token flow.", kind="note")
b = cx.remember("Release-era note about token storage.", kind="note")
lesson = cx.learn("Use only the newly issued refresh token.")
attempt = cx.record_attempt(task="rotate the refresh token", approach="reuse the old token", outcome="failed")
print(json.dumps({"notes": [a.memory_id, b.memory_id], "lesson": lesson.memory_id,
                  "lesson_state": lesson.epistemic_state, "attempt": attempt.attempt_id}, sort_keys=True))
