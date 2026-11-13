"""Slot calculation, speed-up and placement of TTS clips on the episode timeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import ffmpeg
from .subtitles import DialogueLine

RATE = 48000
SPEEDUP_EPSILON = 1.02  # don't bother time-stretching for less than 2%
SILENCE_DB = -45.0
EDGE_MARGIN_S = 0.01


def slot_ends(lines: Sequence[DialogueLine]) -> list[int]:
    """End of each line's available slot (ms).

    A line may run past its own end into silence, up to the next spoken line's start.
    `lines` must be sorted by start.
    """
    ends = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1].start if i + 1 < len(lines) else None
        ends.append(nxt if nxt is not None and nxt > line.end else line.end)
    return ends


def speed_factor(clip_s: float, slot_s: float, max_speedup: float) -> float:
    """Tempo factor needed to fit the clip in its slot, capped at max_speedup (≥ 1)."""
    if slot_s <= 0 or clip_s <= slot_s:
        return 1.0
    needed = clip_s / slot_s
    if needed < SPEEDUP_EPSILON:
        return 1.0
    return min(needed, max(1.0, max_speedup))


def atempo_chain(factor: float) -> str:
    """ffmpeg filter string for a tempo change; each atempo instance kept within [0.5, 2]."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    parts: list[float] = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def decode(data_path: str) -> np.ndarray:
    """Decode any audio file to 48 kHz mono float32."""
    proc = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-i", data_path, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"],
        what="Decoding TTS clip",
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def stretch(samples: np.ndarray, tempo: float) -> np.ndarray:
    proc = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
         "-filter:a", atempo_chain(tempo), "-f", "f32le", "pipe:1"],
        input=samples.astype(np.float32).tobytes(),
        what="Time-stretching TTS clip",
    )  # fmt: skip
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def trim_silence(samples: np.ndarray, threshold_db: float = SILENCE_DB) -> np.ndarray:
    """Remove leading/trailing silence so speech starts right at the subtitle time."""
    if samples.size == 0:
        return samples
    thresh = 10 ** (threshold_db / 20)
    loud = np.flatnonzero(np.abs(samples) > thresh)
    if loud.size == 0:
        return samples[:0]
    margin = int(EDGE_MARGIN_S * RATE)
    a = max(0, loud[0] - margin)
    b = min(samples.size, loud[-1] + margin + 1)
    return samples[a:b]


@dataclass
class PlacedClip:
    line: DialogueLine
    start_sample: int
    samples: np.ndarray
    tempo: float
    overflow_s: float  # how far the clip runs past its slot (0 if it fits)

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.samples.size


def fit_clip(
    line: DialogueLine,
    slot_end_ms: int,
    clip_path: str,
    max_speedup: float,
    target_rms: float | None = None,
) -> PlacedClip:
    samples = trim_silence(decode(clip_path))
    if target_rms is not None:
        from .levels import active_rms

        current = active_rms(samples, RATE)
        if current:
            samples = samples * np.float32(target_rms / current)
    slot_s = (slot_end_ms - line.start) / 1000
    clip_s = samples.size / RATE
    tempo = speed_factor(clip_s, slot_s, max_speedup)
    if tempo > 1.0:
        samples = stretch(samples, tempo)
    # Tiny fades so clips never click.
    fade = min(samples.size // 2, int(0.005 * RATE))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    overflow = max(0.0, samples.size / RATE - slot_s)
    if overflow < 0.02:
        overflow = 0.0
    return PlacedClip(line, int(round(line.start * RATE / 1000)), samples, tempo, overflow)


def render_speech(clips: Sequence[PlacedClip], start: int, length: int) -> np.ndarray:
    """Mono speech for samples [start, start+length) — used in blocks to keep memory low."""
    out = np.zeros(length, dtype=np.float32)
    end = start + length
    for c in clips:
        if c.end_sample <= start or c.start_sample >= end:
            continue
        a = max(c.start_sample, start)
        b = min(c.end_sample, end)
        out[a - start : b - start] += c.samples[a - c.start_sample : b - c.start_sample]
    return out


def total_samples(duration_s: float, clips: Sequence[PlacedClip]) -> int:
    n = int(math.ceil(duration_s * RATE))
    return max([n, *(c.end_sample for c in clips)])
