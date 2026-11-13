"""Offline TTS stand-in: a sine beep whose length scales with the text. For tests/dev."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import soundfile as sf

RATE = 24000
SECONDS_PER_CHAR = 0.065


class MockTTS:
    name = "mock"
    default_voice = "mock-default"

    async def synthesize(self, text: str, voice: str, reference: object = None) -> bytes:
        dur = max(0.3, len(text) * SECONDS_PER_CHAR)
        # Pitch derived from the voice name so different voices sound different.
        freq = 300 + int(hashlib.md5(voice.encode()).hexdigest(), 16) % 500
        t = np.arange(int(dur * RATE)) / RATE
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        fade = min(len(t) // 2, int(0.01 * RATE))
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]
        buf = io.BytesIO()
        sf.write(buf, tone.astype(np.float32), RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()
