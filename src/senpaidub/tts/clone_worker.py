"""Chatterbox Turbo worker. Runs inside its own Python environment (it needs a different
PyTorch than Demucs), so this file must not import anything from senpaidub.

Protocol, one JSON object per line:
  stdout on start:  {"ready": true, "sr": 24000}
  stdin request:    {"id": 1, "text": "...", "ref": "/path/ref.wav", "out": "/path/out.wav"}
  stdout response:  {"id": 1, "ok": true} | {"id": 1, "ok": false, "error": "..."}
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main() -> None:
    # Keep stdout clean for the protocol: libraries print progress bars and notices.
    proto = sys.stdout
    sys.stdout = sys.stderr

    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    # On Apple Silicon the CPU beats the GPU here: Turbo generates in many tiny steps, and on
    # MPS it was 2x slower and grew to 7-12 GB, versus a flat ~4 GB on the CPU.
    device = os.environ.get("SENPAIDUB_CLONE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    # MPS has no float64; the loudness normaliser returns float64 audio.
    _norm = model.norm_loudness
    model.norm_loudness = lambda wav, sr, *a, **k: np.asarray(_norm(wav, sr, *a, **k), dtype=np.float32)

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    reply({"ready": True, "sr": model.sr, "device": device})
    for raw in sys.stdin:
        if not raw.strip():
            continue
        req = json.loads(raw)
        try:
            with torch.no_grad():
                wav = model.generate(req["text"], audio_prompt_path=req["ref"])
            sf.write(req["out"], wav.squeeze(0).detach().cpu().numpy().astype(np.float32), model.sr)
            del wav
            reply({"id": req["id"], "ok": True})
        except Exception as e:  # noqa: BLE001 — report every failure to the parent
            traceback.print_exc()
            reply({"id": req["id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            # Without this the MPS/CUDA allocator keeps every line's buffers: the process
            # grew to ~40 GB over one episode and pushed the machine into heavy swapping.
            model.conds = None
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
