"""Per-line speech levels: make each English line as loud as the original Japanese line.

Matching against the isolated vocals keeps the original dialogue-to-music balance and the
performance's dynamics (shouts stay loud, whispers stay quiet).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_S = 0.05
GATE_DB = -55.0  # frames quieter than this are silence
TOP_DB = 20.0  # only frames within this range of the loudest frame count as "speech"
MAX_BELOW_MEDIAN_DB = 12.0
MAX_ABOVE_MEDIAN_DB = 8.0


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def active_rms(x: np.ndarray, rate: int) -> float | None:
    """RMS over the loud part of a signal (gated), or None if it's silent."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = max(1, int(FRAME_S * rate))
    frames = x[: x.size - x.size % n].reshape(-1, n) if x.size >= n else x.reshape(1, -1)
    if frames.size == 0:
        return None
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    loud = rms[rms > 10 ** (GATE_DB / 20)]
    if loud.size == 0:
        return None
    loud = loud[loud > loud.max() * 10 ** (-TOP_DB / 20)]
    return float(np.sqrt((loud**2).mean()))


def read_segment(sf_file: sf.SoundFile, start_ms: int, end_ms: int) -> np.ndarray:
    rate = sf_file.samplerate
    a = max(0, int(start_ms * rate / 1000))
    b = min(sf_file.frames, int(end_ms * rate / 1000))
    if b <= a:
        return np.zeros(0, np.float32)
    sf_file.seek(a)
    return sf_file.read(b - a, dtype="float32", always_2d=True).mean(axis=1)


def reference_levels(reference: Path, spans_ms: Sequence[tuple[int, int]]) -> list[float]:
    """Target RMS for each (start, end) span, measured from the reference audio.

    Silent or implausible measurements (bleed, off-screen mumbling) are pulled towards the
    episode median so one bad span can't make a line inaudible or deafening.
    """
    with sf.SoundFile(str(reference)) as f:
        raw = [active_rms(read_segment(f, a, b), f.samplerate) for a, b in spans_ms]
    measured = [r for r in raw if r is not None]
    if not measured:
        return [10 ** (-20 / 20)] * len(spans_ms)
    median = float(np.median(measured))
    lo = median * 10 ** (-MAX_BELOW_MEDIAN_DB / 20)
    hi = median * 10 ** (MAX_ABOVE_MEDIAN_DB / 20)
    return [median if r is None else float(np.clip(r, lo, hi)) for r in raw]


def pitch(x: np.ndarray, rate: int) -> float | None:
    """Median fundamental frequency (Hz) of the voiced parts, or None if unvoiced/silent.

    A plain autocorrelation estimator: good enough to tell voices apart by register.
    """
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = int(0.04 * rate)
    lo, hi = int(rate / 500), int(rate / 70)
    if x.size < n or hi >= n:
        return None
    values = []
    for i in range(0, x.size - n, n // 2):
        frame = x[i : i + n] - x[i : i + n].mean()
        if np.sqrt((frame**2).mean()) < 0.01:
            continue
        spec = np.fft.rfft(frame, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5:
            values.append(rate / k)
    return float(np.median(values)) if len(values) >= 3 else None


SAME_VOICE_RATIO = 1.25  # neighbours within this pitch ratio count as "probably the same voice"
NEIGHBOUR_WINDOW_MS = 60_000
GAP_S = 0.12


def write_references(
    vocals: Path, spans_ms: Sequence[tuple[int, int]], out_dir: Path, min_s: float
) -> list[Path]:
    """Build a voice reference of at least `min_s` seconds for each line.

    The line's own audio always comes first. Cloning models need several seconds, so short
    lines are padded with *other lines that sound like the same speaker* (similar pitch,
    nearest in time first), not with whatever audio surrounds them, which would often be the
    other character in the conversation and flip the voice (e.g. male ↔ female). If no
    similar line is close by, the line's own audio is repeated.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(vocals)) as f:
        rate = f.samplerate
        own = [read_segment(f, a - 150, b + 150) for a, b in spans_ms]
    pitches = [pitch(x, rate) for x in own]
    need = int(min_s * rate)
    gap = np.zeros(int(GAP_S * rate), np.float32)
    paths = []
    for i, (a, _b) in enumerate(spans_ms):
        parts = [own[i]]
        length = own[i].size
        if length < need and pitches[i] is not None:
            p = pitches[i]
            candidates = sorted(
                (j for j in range(len(spans_ms))
                 if j != i and pitches[j] is not None and own[j].size
                 and abs(spans_ms[j][0] - a) <= NEIGHBOUR_WINDOW_MS
                 and 1 / SAME_VOICE_RATIO <= pitches[j] / p <= SAME_VOICE_RATIO),
                key=lambda j: abs(spans_ms[j][0] - a),
            )  # fmt: skip
            for j in candidates:
                if length >= need:
                    break
                parts += [gap, own[j]]
                length += gap.size + own[j].size
        while length < need and own[i].size:  # repeat the line itself
            parts += [gap, own[i]]
            length += gap.size + own[i].size
        ref = np.concatenate(parts) if length else np.zeros(need, np.float32)
        p_out = out_dir / f"ref{i:04d}.wav"
        sf.write(str(p_out), ref, rate, subtype="PCM_16")
        paths.append(p_out)
    return paths
