"""Qwen3-TTS worker. Runs inside its own Python environment (it needs a different PyTorch and
transformers than Demucs), so this file must not import anything from senpaidub.

Protocol, one JSON object per line:
  stdout on start:  {"ready": true, "sr": 24000, "device": "mps", "model": "..."}
  stdin request:    {"id": 1, "items": [{"text": "...", "ref": "/ref.wav", "out": "/out.wav",
                                         "max_s": 6.5}, ...]}
  stdout response:  {"id": 1, "ok": true, "capped": [false, ...], "mem_gb": 4.6}
                    | {"id": 1, "ok": false, "error": "..."}

All items of a request are generated as one batch, which keeps the GPU far busier than one line
at a time. `max_s` caps each line: the model sometimes fails to stop and talks (or hums) until
its token limit, and a batch only finishes when its slowest line does. `capped` reports lines
that hit their cap so the caller can retry them. `mem_gb` is the GPU memory this process still
holds after the batch, so the caller can restart the worker when it creeps up.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

FRAMES_PER_S = 12.5  # the 12Hz tokenizer emits 12.5 codec frames per second of audio
# Lower randomness than the model's defaults (0.9): the words stay much closer to the text.
GENERATION = dict(temperature=0.5, top_k=20, top_p=0.9, subtalker_temperature=0.6, subtalker_top_k=30)


def pick_device(torch) -> str:
    forced = os.environ.get("SENPAIDUB_QWEN_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_memory_gb(torch, device: str) -> float | None:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device == "cuda":
        return torch.cuda.memory_reserved() / 2**30
    return None


def main() -> None:
    nice = int(os.environ.get("SENPAIDUB_QWEN_NICE", "0"))
    if nice and hasattr(os, "nice"):
        os.nice(nice)  # CPU only; the GPU side is handled by the caller's pauses
    # Keep stdout clean for the protocol: libraries print progress bars and notices.
    proto = sys.stdout
    sys.stdout = sys.stderr

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = os.environ.get("SENPAIDUB_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    device = pick_device(torch)
    # Half precision on a GPU: on Apple Silicon bf16 was 1.4x faster than fp32 with the same words.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
    cap_gb = os.environ.get("SENPAIDUB_QWEN_MAX_GB")
    if device == "mps" and cap_gb:
        # Optional hard cap on GPU memory. It is a hard limit, not a target: when the live
        # tensors exceed it, generation fails (seen at 7.5 GB, 2026-11-15), so it's opt-in.
        # Batches that hit it are split in half below before giving up.
        rec = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(1.0, float(cap_gb) * 2**30 / rec))
        print(f"MPS memory capped at {cap_gb} GB", file=sys.stderr, flush=True)

    def generate(items: list) -> list:
        """Generate a batch; on out-of-memory, free the cache and do it in two halves."""
        try:
            with torch.inference_mode():
                wavs, _ = model.generate_voice_clone(
                    text=[it["text"] for it in items],
                    language=["English"] * len(items),
                    ref_audio=[it["ref"] for it in items],
                    x_vector_only_mode=[True] * len(items),
                    max_new_tokens=int(FRAMES_PER_S * max(it["max_s"] for it in items)) + 2,
                    **GENERATION,
                )
            return list(wavs)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(items) == 1:
                raise
            print(f"out of memory with {len(items)} lines; splitting", file=sys.stderr, flush=True)
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            half = len(items) // 2
            return generate(items[:half]) + generate(items[half:])

    def free() -> None:
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    reply({"ready": True, "sr": 24000, "device": device, "model": model_id})
    for raw in sys.stdin:
        if not raw.strip():
            continue
        req = json.loads(raw)
        items = req["items"]
        try:
            wavs = generate(items)
            sr = 24000
            capped = []
            for it, w in zip(items, wavs):
                w = np.asarray(w, dtype=np.float32).reshape(-1)
                limit = int(it["max_s"] * sr)
                capped.append(w.size >= limit - int(0.25 * sr))
                sf.write(it["out"], w[:limit], sr)
            del wavs
            free()
            reply({"id": req["id"], "ok": True, "capped": capped, "mem_gb": gpu_memory_gb(torch, device)})
        except Exception as e:  # noqa: BLE001 — report every failure to the parent
            traceback.print_exc()
            free()
            reply({"id": req["id"], "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
