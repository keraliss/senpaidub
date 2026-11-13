"""Background audio: Demucs vocal removal (chunked) or ducking of the original mix."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ffmpeg
from .errors import ExternalToolError, UsageError

log = logging.getLogger("senpaidub")

MODEL_NAME = "htdemucs"
CHUNK_S = 60.0  # seconds of audio fed to Demucs at once (bounds memory on long episodes)
CROSSFADE_S = 2.0
FADE_S = 0.05  # duck fade in/out
MERGE_GAP_S = 0.3  # speech gaps shorter than this stay ducked (avoids pumping)


# --------------------------------------------------------------------------- extraction


def extract_audio(video: Path, stream_index: int, out: Path, rate: int) -> Path:
    ffmpeg.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video, "-map", f"0:{stream_index}",
         "-ac", "2", "-ar", str(rate), "-c:a", "pcm_f32le", out],
        what="Extracting original audio",
    )  # fmt: skip
    return out


# --------------------------------------------------------------------------- demucs


def pick_device(requested: str = "auto") -> str:
    import torch

    requested = requested.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise UsageError("--device cuda requested but CUDA is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise UsageError("--device mps requested but Apple MPS is not available.")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def separate_stems(
    src_wav: Path,
    out_no_vocals: Path,
    out_vocals: Path,
    *,
    device: str = "auto",
    on_progress: Callable[[float, float], None] | None = None,
) -> tuple[Path, Path]:
    """Write the Demucs `no_vocals` and `vocals` stems of src_wav (at the model's 44.1 kHz).

    The vocals stem is kept as a per-line loudness reference (and voice reference for cloning).
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:
        raise UsageError(
            "Demucs is not installed (pip install demucs), or use --no-separate to duck instead."
        ) from e

    dev = pick_device(device)
    if dev == "mps":
        # Ops MPS lacks fall back to CPU instead of failing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    log.info("Demucs device: %s", dev)
    try:
        model = get_model(MODEL_NAME)
    except Exception as e:  # noqa: BLE001
        raise ExternalToolError(f"Could not load the Demucs model '{MODEL_NAME}': {e}") from e
    model.eval()
    sr = model.samplerate
    vocals_idx = model.sources.index("vocals")

    info = sf.info(str(src_wav))
    if info.samplerate != sr:
        raise ValueError(f"expected {sr} Hz input, got {info.samplerate}")
    n = info.frames

    # Global normalisation, as the Demucs CLI does, computed in one streaming pass.
    total = 0.0
    total_sq = 0.0
    for block in sf.blocks(str(src_wav), blocksize=sr * 30, dtype="float32", always_2d=True):
        mono = block.mean(axis=1, dtype=np.float64)
        total += mono.sum()
        total_sq += np.square(mono).sum()
    mean = total / max(n, 1)
    std = float(np.sqrt(max(total_sq / max(n, 1) - mean**2, 1e-12)))

    step = int(CHUNK_S * sr)
    xf = int(CROSSFADE_S * sr)
    tails: list[np.ndarray] | None = None

    with sf.SoundFile(str(src_wav)) as reader, sf.SoundFile(
        str(out_no_vocals), "w", samplerate=sr, channels=2, subtype="FLOAT"
    ) as w_bg, sf.SoundFile(str(out_vocals), "w", samplerate=sr, channels=2, subtype="FLOAT") as w_vo:
        for start in range(0, n, step):
            reader.seek(start)
            x = reader.read(min(step + xf, n - start), dtype="float32", always_2d=True)
            mix = torch.from_numpy(((x.T - mean) / std).astype(np.float32))[None]
            with torch.no_grad():
                try:
                    est = apply_model(model, mix, shifts=1, split=True, overlap=0.25,
                                      device=dev, progress=False)  # fmt: skip
                except RuntimeError as e:
                    if dev == "cpu":
                        raise ExternalToolError(f"Demucs failed: {e}") from e
                    log.warning("Demucs on %s failed (%s); retrying on CPU", dev, e)
                    dev = "cpu"
                    est = apply_model(model, mix, shifts=1, split=True, overlap=0.25,
                                      device="cpu", progress=False)  # fmt: skip
            est = est[0].cpu().numpy()  # (sources, 2, L)
            bg = (np.delete(est, vocals_idx, axis=0).sum(axis=0) * std + mean).T  # (L, 2)
            vo = (est[vocals_idx] * std).T
            ys = [bg, vo]

            if tails is not None:
                for y, tail in zip(ys, tails):
                    ol = min(tail.shape[0], y.shape[0])
                    ramp = np.linspace(0.0, 1.0, ol, dtype=np.float32)[:, None]
                    y[:ol] = tail[:ol] * (1 - ramp) + y[:ol] * ramp
            last = start + step >= n
            for y, w in zip(ys, (w_bg, w_vo)):
                w.write(y if last else y[:step])
            tails = None if last else [y[step:] for y in ys]
            if on_progress:
                on_progress(min(start + step, n) / sr, n / sr)
    return out_no_vocals, out_vocals


# --------------------------------------------------------------------------- ducking


def speech_intervals(
    spans: Sequence[tuple[int, int]], rate: int, merge_gap_s: float = MERGE_GAP_S
) -> list[tuple[int, int]]:
    """Merge (start, end) sample spans that overlap or sit closer than merge_gap_s."""
    gap = int(merge_gap_s * rate)
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def duck_gain(
    intervals: Sequence[tuple[int, int]],
    start: int,
    length: int,
    duck_db: float,
    rate: int,
    fade_s: float = FADE_S,
) -> np.ndarray:
    """Per-sample gain for [start, start+length): 1 outside speech, duck_db inside, faded."""
    floor = 10 ** (min(duck_db, 0.0) / 20)
    fade = max(1, int(fade_s * rate))
    t = np.arange(start, start + length, dtype=np.float64)
    depth = np.zeros(length, dtype=np.float64)  # 0 = untouched, 1 = fully ducked
    end = start + length
    for a, b in intervals:
        if b + fade <= start or a - fade >= end:
            continue
        # Trapezoid: ramps up over [a-fade, a], flat over [a, b], down over [b, b+fade].
        d = np.clip(np.minimum((t - (a - fade)) / fade, ((b + fade) - t) / fade), 0.0, 1.0)
        np.maximum(depth, d, out=depth)
    return (1.0 - depth * (1.0 - floor)).astype(np.float32)
