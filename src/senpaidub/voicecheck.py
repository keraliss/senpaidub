"""Catch cloned lines that come out in the wrong register (e.g. a woman's line in a man's
voice) and regenerate them. Cloning models sample randomly, so a retry usually lands right.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from . import levels
from .subtitles import DialogueLine
from .tts.base import TTSBackend, TTSCache

log = logging.getLogger("senpaidub")

BOUNDARY_HZ = 175.0  # rough split between lower (male) and higher (female) voices
ORIGINAL_MARGIN_HZ = 25.0  # only act when the original line is clearly on one side
CLIP_MARGIN_HZ = 15.0
RETRIES = 2


def _clip_pitch(data: bytes) -> float | None:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 — undecodable clip: nothing to compare
        return None
    return levels.pitch(x.mean(axis=1), sr)


def is_mismatch(original_hz: float | None, clip_hz: float | None) -> bool:
    if original_hz is None or clip_hz is None:
        return False
    return (
        (original_hz > BOUNDARY_HZ) != (clip_hz > BOUNDARY_HZ)
        and abs(original_hz - BOUNDARY_HZ) > ORIGINAL_MARGIN_HZ
        and abs(clip_hz - BOUNDARY_HZ) > CLIP_MARGIN_HZ
    )


def fix_mismatches(
    backend: TTSBackend,
    lines: Sequence[DialogueLine],
    voices: Sequence[str],
    refs: Sequence[Path],
    vocals: Path,
    cache: TTSCache,
    on_done: Callable[[], None] | None = None,
) -> tuple[int, int]:
    """Returns (mismatched lines found, lines fixed). Each line is checked once (cached)."""
    marks = cache.dir.parent / "voicecheck"
    found = fixed = 0
    with sf.SoundFile(str(vocals)) as f:
        for line, voice, ref in zip(lines, voices, refs):
            key = cache.key(backend.name, voice, line.text, ref)
            mark = marks / key[:2] / key
            if mark.exists():
                if on_done:
                    on_done()
                continue
            original = levels.pitch(levels.read_segment(f, line.start, line.end), f.samplerate)
            data = cache.get(key)
            current = _clip_pitch(data) if data else None
            if is_mismatch(original, current):
                found += 1
                assert original is not None and current is not None and data is not None
                best, best_err = data, abs(np.log(current / original))
                for _ in range(RETRIES):
                    try:
                        cand = asyncio.run(backend.synthesize(line.text, voice, ref))
                    except Exception as e:  # noqa: BLE001 — keep the original clip
                        log.debug("retry failed: %s", e)
                        break
                    p = _clip_pitch(cand)
                    if p is None:
                        continue
                    err = abs(np.log(p / original))
                    if err < best_err:
                        best, best_err = cand, err
                    if not is_mismatch(original, p):
                        break
                if best is not data:
                    cache.put(key, best)
                    if not is_mismatch(original, _clip_pitch(best)):
                        fixed += 1
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.touch()
            if on_done:
                on_done()
    return found, fixed


# Stock voices for backends that can't clone: one per register.
REGISTER_VOICES = {
    "edge": ("en-US-AndrewNeural", "en-US-AvaNeural"),
    "google": ("en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Aoede"),  # Google's most natural voices
}


def voices_by_register(
    lines: Sequence[DialogueLine], vocals: Path, male: str, female: str
) -> tuple[list[str], int, int]:
    """Pick a male or female stock voice per line from the pitch of the original line.

    Lines whose pitch can't be measured (whispers, overlapping sound) take the voice of the
    nearest measured line before them, which is usually the same speaker continuing.
    Returns (voices, lines measured as male, lines measured as female).
    """
    out: list[str] = []
    males = females = 0
    last = female
    with sf.SoundFile(str(vocals)) as f:
        for line in lines:
            hz = levels.pitch(levels.read_segment(f, line.start, line.end), f.samplerate)
            if hz is not None:
                if hz > BOUNDARY_HZ:
                    last, females = female, females + 1
                else:
                    last, males = male, males + 1
            out.append(last)
    return out, males, females
