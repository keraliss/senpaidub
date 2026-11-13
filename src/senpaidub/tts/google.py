"""Google Cloud Text-to-Speech. Optional: pip install 'senpaidub[google]'.

Credentials, first found wins:
  1. GOOGLE_API_KEY environment variable (an API key from the Cloud console),
  2. a file named `google-api-key` next to config.toml holding that key,
  3. GOOGLE_APPLICATION_CREDENTIALS (a service-account JSON file).
"""

from __future__ import annotations

import asyncio
import os

from ..config import config_path
from ..errors import UsageError

SETUP_HINT = """\
Google TTS needs credentials. Easiest: in the Google Cloud console, enable the
"Cloud Text-to-Speech API", create an API key (APIs & Services → Credentials), then save it:
    echo 'YOUR_KEY' > "{key_file}"
(or set GOOGLE_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS for a service-account file)."""


def api_key() -> str | None:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    f = config_path().with_name("google-api-key")
    if f.is_file():
        return f.read_text().strip() or None
    return None


class GoogleTTS:
    name = "google"
    default_voice = "en-US-Chirp3-HD-Aoede"

    def __init__(self) -> None:
        try:
            from google.cloud import texttospeech  # noqa: F401
        except ImportError as e:
            raise UsageError("Google TTS needs an extra package: pip install 'senpaidub[google]'") from e
        self._key = api_key()
        if not self._key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise UsageError(SETUP_HINT.format(key_file=config_path().with_name("google-api-key")))
        self._client = None

    def _sync(self, text: str, voice: str) -> bytes:
        from google.cloud import texttospeech as tts

        if self._client is None:
            opts = {"api_key": self._key} if self._key else None
            self._client = tts.TextToSpeechClient(client_options=opts)
        lang = "-".join(voice.split("-")[:2]) or "en-US"
        resp = self._client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(language_code=lang, name=voice),
            audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=48000),
        )
        return resp.audio_content  # WAV (LINEAR16 includes a RIFF header)

    async def synthesize(self, text: str, voice: str, reference: object = None) -> bytes:
        return await asyncio.to_thread(self._sync, text, voice)
