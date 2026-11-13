"""Mix background + speech, normalise loudness, encode."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ffmpeg
from .separate import duck_gain
from .timing import RATE, PlacedClip, render_speech

BLOCK_S = 10
RESTORE_FADE_S = 0.08
TARGET_LUFS = -16.0
TRUE_PEAK = -1.5
CODECS = {
    # name: (ffmpeg encoder args)
    "aac": ["-c:a", "aac", "-b:a", "192k"],
    "opus": ["-c:a", "libopus", "-b:a", "160k"],
}


def mix(
    background: Path | None,
    clips: Sequence[PlacedClip],
    n_samples: int,
    out: Path,
    *,
    duck: tuple[Sequence[tuple[int, int]], float] | None = None,
    restore: tuple[Path, Sequence[tuple[int, int]]] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Write background (+ optional ducking) + speech as a 48 kHz stereo float WAV.

    With background=None the result is speech only. `restore=(vocals, intervals)` adds the
    original vocals stem back everywhere *except* the given intervals (where English speech
    replaces it), so screams, laughs, songs and anything Demucs misfiled as "vocals" survive.

    Float output can't clip; loudness/peak control happens in `encode`.
    Works block-by-block so a full episode never sits in memory at once.
    """
    block = BLOCK_S * RATE
    bg = sf.SoundFile(str(background)) if background is not None else None
    vo = sf.SoundFile(str(restore[0])) if restore is not None else None
    try:
        if bg is not None and bg.samplerate != RATE:
            raise ValueError(f"background must be {RATE} Hz, got {bg.samplerate}")
        with sf.SoundFile(str(out), "w", samplerate=RATE, channels=2, subtype="FLOAT") as w:
            for start in range(0, n_samples, block):
                length = min(block, n_samples - start)
                b = _read_block(bg, length)
                if duck is not None:
                    intervals, duck_db = duck
                    b *= duck_gain(intervals, start, length, duck_db, RATE)[:, None]
                if vo is not None and restore is not None:
                    keep = duck_gain(restore[1], start, length, -120.0, RATE, fade_s=RESTORE_FADE_S)
                    b += _read_block(vo, length) * keep[:, None]
                b += render_speech(clips, start, length)[:, None]
                w.write(b)
                if on_progress:
                    on_progress((start + length) / RATE)
    finally:
        for f in (bg, vo):
            if f is not None:
                f.close()
    return out


def _read_block(bg: sf.SoundFile | None, length: int) -> np.ndarray:
    if bg is None:
        return np.zeros((length, 2), np.float32)
    b = bg.read(length, dtype="float32", always_2d=True)
    if b.shape[1] == 1:
        b = np.repeat(b, 2, axis=1)
    b = np.ascontiguousarray(b[:, :2])
    if b.shape[0] < length:
        b = np.vstack([b, np.zeros((length - b.shape[0], 2), np.float32)])
    return b


def measure_lufs(src: Path, stream: str | None = None) -> float | None:
    """Integrated loudness (LUFS) of a file or one of its streams; None for silence."""
    cmd: list[str | Path] = ["ffmpeg", "-v", "info", "-i", src]
    if stream:
        cmd += ["-map", stream]
    cmd += ["-af", "loudnorm=print_format=json", "-f", "null", "-"]
    text = ffmpeg.run(cmd, what="Measuring loudness").stderr.decode(errors="replace")
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        value = float(json.loads(m[-1])["input_i"])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) and value > -70 else None


def encode(wav: Path, out: Path, codec: str = "aac", target_lufs: float | None = None) -> Path:
    """Apply one fixed gain so the track's loudness matches target_lufs (by default the
    original audio's, else TARGET_LUFS), limit peaks, and encode.

    A single gain keeps the mix's internal balance and dynamics intact (unlike dynamic
    loudness normalisation, which lifts quiet scenes).
    """
    target = TARGET_LUFS if target_lufs is None else target_lufs
    measured = measure_lufs(wav)
    gain_db = 0.0 if measured is None else target - measured
    filters = [f"volume={gain_db:.2f}dB", f"alimiter=limit={10 ** (TRUE_PEAK / 20):.3f}:level=false",
               f"aresample={RATE}"]  # fmt: skip
    ffmpeg.run(
        ["ffmpeg", "-y", "-v", "error", "-i", wav, "-af", ",".join(filters), *CODECS[codec],
         "-ar", str(RATE), "-ac", "2", out],
        what=f"Encoding the dub track ({codec})",
    )  # fmt: skip
    return out
