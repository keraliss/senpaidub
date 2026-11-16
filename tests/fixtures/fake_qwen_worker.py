"""Stand-in for tts/qwen_worker.py in tests: same protocol, no model.

Each line becomes a tone of 0.2 s + 20 ms per character, so tests can check that every clip
went back to the right line. A line containing "STUCK" fills its whole cap and is reported as
capped. Every request is appended to $FAKE_QWEN_LOG as JSON.
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 24000
print(json.dumps({"ready": True, "sr": SR, "device": "fake", "model": "fake"}), flush=True)
for raw in sys.stdin:
    req = json.loads(raw)
    with open(os.environ["FAKE_QWEN_LOG"], "a") as log:
        log.write(json.dumps(req["items"]) + "\n")
    capped = []
    for it in req["items"]:
        stuck = "STUCK" in it["text"]
        dur = it["max_s"] if stuck else 0.2 + 0.02 * len(it["text"])
        t = np.arange(int(dur * SR)) / SR
        sf.write(it["out"], (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
        capped.append(stuck)
    print(json.dumps({"id": req["id"], "ok": True, "capped": capped}), flush=True)
