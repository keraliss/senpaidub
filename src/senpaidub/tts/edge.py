"""Microsoft Edge online TTS (free, no API key). Returns MP3 bytes."""

from __future__ import annotations

import logging

log = logging.getLogger("senpaidub")

# The service occasionally returns no audio for a particular sentence in a particular voice
# (it's consistent, so retrying the same voice doesn't help). Fall back to a similar voice.
FALLBACKS = {
    "en-US-AndrewNeural": ["en-US-AndrewMultilingualNeural", "en-US-BrianNeural", "en-US-GuyNeural"],
    "en-US-AvaNeural": ["en-US-AvaMultilingualNeural", "en-US-EmmaNeural", "en-US-AriaNeural"],
}


class EdgeTTS:
    name = "edge"
    default_voice = "en-US-AriaNeural"

    async def _speak(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
        return b"".join(chunks)

    async def synthesize(self, text: str, voice: str, reference: object = None) -> bytes:
        from edge_tts.exceptions import NoAudioReceived

        candidates = [voice, *FALLBACKS.get(voice, [])]
        for i, v in enumerate(candidates):
            try:
                return await self._speak(text, v)
            except NoAudioReceived:
                if i == len(candidates) - 1:
                    raise
                log.debug("Edge returned no audio for %r in %s; trying %s", text, v, candidates[i + 1])
        raise AssertionError("unreachable")
