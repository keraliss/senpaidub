import asyncio
from pathlib import Path

import numpy as np
import pytest

from senpaidub.batch import natural_key
from senpaidub.config import DEFAULTS, load_config, resolve
from senpaidub.download import is_url, name_from_url, sanitize_filename
from senpaidub.errors import NoSubtitlesError, UsageError
from senpaidub.probe import MediaInfo, Stream, choose_audio_stream, choose_subtitle_stream
from senpaidub.separate import duck_gain, speech_intervals
from senpaidub.tts.base import TTSCache, synthesize_all
from senpaidub.tts.mock import MockTTS
from senpaidub.voices import VoiceMap

# --------------------------------------------------------------------------- voices


def test_voice_map_lookup(tmp_path):
    p = tmp_path / "voices.yaml"
    p.write_text("default: en-US-AriaNeural\nspeakers:\n  Rimuru: en-US-GuyNeural\n  Shion: en-US-JennyNeural\n")
    vm = VoiceMap.load(p, "fallback")
    assert vm.default == "en-US-AriaNeural"
    assert vm.voice_for("Rimuru") == "en-US-GuyNeural"
    assert vm.voice_for(" rimuru ") == "en-US-GuyNeural"  # case/space-insensitive
    assert vm.voice_for("Gobta") == "en-US-AriaNeural"
    assert vm.voice_for("") == "en-US-AriaNeural"


def test_voice_map_default_falls_back(tmp_path):
    p = tmp_path / "voices.yaml"
    p.write_text("speakers:\n  A: v1\n")
    assert VoiceMap.load(p, "fallback").voice_for("B") == "fallback"
    assert VoiceMap.load(None, "fallback").voice_for("A") == "fallback"


@pytest.mark.parametrize("content", ["- a\n- b\n", "speakers: [1, 2]\n", "speakers:\n  A: [x]\n", "a: : :"])
def test_voice_map_rejects_bad_files(tmp_path, content):
    p = tmp_path / "voices.yaml"
    p.write_text(content)
    with pytest.raises(UsageError):
        VoiceMap.load(p, "fallback")


# --------------------------------------------------------------------------- probe/choice


def sub(i, codec="ass", lang="eng", title="", forced=False):
    return Stream(i, "subtitle", codec, lang, title, forced=forced)


def info(*streams):
    return MediaInfo("x.mkv", list(streams))


def test_choose_prefers_full_english_text_track():
    i = info(sub(2, "hdmv_pgs_subtitle"), sub(3, lang="jpn"), sub(4, title="Signs & Songs"), sub(5, title="Full"))
    assert choose_subtitle_stream(i, None).index == 5


def test_choose_falls_back_to_first_text_track():
    assert choose_subtitle_stream(info(sub(3, lang="spa"), sub(4, lang="por")), None).index == 3


def test_choose_errors():
    with pytest.raises(NoSubtitlesError, match="--sub-file"):
        choose_subtitle_stream(info(), None)
    with pytest.raises(NoSubtitlesError, match="OCR"):
        choose_subtitle_stream(info(sub(2, "hdmv_pgs_subtitle")), None)
    with pytest.raises(NoSubtitlesError, match="OCR"):
        choose_subtitle_stream(info(sub(2, "hdmv_pgs_subtitle")), 2)
    with pytest.raises(UsageError):
        choose_subtitle_stream(info(sub(2)), 9)


def test_choose_audio_prefers_japanese():
    a = [Stream(1, "audio", "aac", "eng", default=True), Stream(2, "audio", "flac", "jpn")]
    assert choose_audio_stream(info(*a)).index == 2
    b = [Stream(1, "audio", "aac", "eng"), Stream(2, "audio", "aac", "spa", default=True)]
    assert choose_audio_stream(info(*b)).index == 2


# --------------------------------------------------------------------------- ducking


def test_speech_intervals_merge_close_gaps():
    rate = 1000
    assert speech_intervals([(0, 100), (350, 400), (1000, 1100)], rate, 0.3) == [(0, 400), (1000, 1100)]


def test_duck_gain_shape_and_block_consistency():
    rate = 1000
    iv = [(200, 400)]
    g = duck_gain(iv, 0, 600, -12, rate, fade_s=0.05)
    floor = 10 ** (-12 / 20)
    assert g[0] == pytest.approx(1.0)
    assert g[149] == pytest.approx(1.0)
    assert g[175] == pytest.approx(1 - 0.5 * (1 - floor), abs=0.02)  # mid fade-in
    assert g[300] == pytest.approx(floor)
    assert g[499] == pytest.approx(1.0)
    pieces = np.concatenate([duck_gain(iv, s, 100, -12, rate, 0.05) for s in range(0, 600, 100)])
    np.testing.assert_allclose(g, pieces)


# --------------------------------------------------------------------------- TTS cache


def test_tts_cache_and_dedupe(tmp_path):
    calls = []

    class Counting(MockTTS):
        async def synthesize(self, text, voice, reference=None):
            calls.append((text, voice))
            return await super().synthesize(text, voice)

    cache = TTSCache(tmp_path)
    reqs = [("hi", "a", None), ("hi", "a", None), ("hi", "b", None), ("yo", "a", None)]
    paths, stats = asyncio.run(synthesize_all(Counting(), reqs, cache))
    assert len(calls) == 3 and stats.synthesized == 3
    assert paths[0] == paths[1] != paths[2]
    calls.clear()
    _, stats = asyncio.run(synthesize_all(Counting(), reqs, cache))
    assert calls == [] and stats.cached == 3


def test_tts_failure_is_clear(tmp_path):
    from senpaidub.errors import ExternalToolError

    class Broken(MockTTS):
        async def synthesize(self, text, voice, reference=None):
            raise ConnectionError("nope")

    with pytest.raises(ExternalToolError, match="failed after 2 attempts"):
        asyncio.run(synthesize_all(Broken(), [("hi", "a", None)], TTSCache(tmp_path), retries=2))


# --------------------------------------------------------------------------- config


def test_config_precedence(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('tts = "google"\nmax_speedup = 1.6\nset_default = true\n')
    cfg = resolve({"tts": None, "max_speedup": 1.2, "voice": None}, load_config(p))
    assert cfg["tts"] == "google"  # config beats default
    assert cfg["max_speedup"] == 1.2  # CLI beats config
    assert cfg["set_default"] is True
    assert cfg["duck_db"] == DEFAULTS["duck_db"]


@pytest.mark.parametrize("content", ['bogus = 1\n', 'max_speedup = "fast"\n', "set_default = 1\n", "tts = \n"])
def test_config_rejects_bad_values(tmp_path, content):
    p = tmp_path / "config.toml"
    p.write_text(content)
    with pytest.raises(UsageError):
        load_config(p)


# --------------------------------------------------------------------------- URLs / naming / batch


def test_is_url():
    assert is_url("http://127.0.0.1:11470/abc/0") and is_url("HTTPS://x/y.mkv")
    assert not is_url("Episode 01.mkv") and not is_url("C:\\anime\\ep.mkv")


def test_sanitize_filename():
    assert sanitize_filename('a/b\\c:d*e?"f<g>h|i') == "a b c d e f g h i"
    assert sanitize_filename("  ..  ") == "stream"
    assert sanitize_filename("CON") == "_CON"
    assert len(sanitize_filename("x" * 400)) == 150


def test_name_from_url():
    blank = MediaInfo("u")
    titled = MediaInfo("u", title="Show - 01")
    url = "http://127.0.0.1:11470/6b1f/%5BGroup%5D%20Show%20-%2001.mkv"
    assert name_from_url(url, blank, None) == "[Group] Show - 01"
    assert name_from_url(url, titled, None) == "Show - 01"
    assert name_from_url(url, titled, "My: Name") == "My Name"
    assert name_from_url("http://h/", blank, None) == "stream"


def test_natural_sort():
    names = ["Episode 10.mkv", "Episode 2.mkv", "episode 1.mkv", "Episode 02v2.mkv"]
    ordered = [p.name for p in sorted(map(Path, names), key=natural_key)]
    assert ordered == ["episode 1.mkv", "Episode 2.mkv", "Episode 02v2.mkv", "Episode 10.mkv"]


# --------------------------------------------------------------------------- stremio series mode

from senpaidub.stremio import StremioFile, episode_number, matches, output_name, sort_key  # noqa: E402


@pytest.mark.parametrize(
    "name, expected",
    [
        ("[SubsPlease] Dark Summoner to Dekiteiru - 01 (1080p) [00C3F7E4].mkv", (None, 1)),
        ("[Erai-raws] Show - 12v2 [1080p][Multiple Subtitle].mkv", (None, 12)),
        ("Show.Name.S02E05.1080p.WEB.mkv", (2, 5)),
        ("Show Name Episode 7.mp4", (None, 7)),
        ("[Group] 86 - Eighty Six - 03 (BD 1080p).mkv", (None, 3)),
        ("Movie (2019).mkv", (None, None)),
    ],
)
def test_episode_number(name, expected):
    assert episode_number(name) == expected


def test_series_matching_and_naming():
    n = "[SubsPlease] Dark Summoner to Dekiteiru - 02 (1080p) [5500169C].mkv"
    assert matches(n, "dark summoner") and matches(n, "Dekiteiru, Dark")
    assert not matches(n, "dark summoners") and not matches(n, "frieren")
    assert output_name(n, "I'm Dating a Dark Summoner") == "I'm Dating a Dark Summoner - 02"
    assert output_name("Show.S01E03.mkv", "Show") == "Show - S01E03"
    assert output_name(n, None) == "Dark Summoner to Dekiteiru - 02"


def test_series_sort_order():
    files = [StremioFile("h", 0, f"[G] Show - {e:02d} (1080p).mkv") for e in (10, 2, 1)]
    assert [episode_number(f.name)[1] for f in sorted(files, key=sort_key)] == [1, 2, 10]


# --------------------------------------------------------------------------- voice references

from senpaidub.levels import pitch, write_references  # noqa: E402


def _voice(f0, seconds, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))).astype(np.float32)


def test_pitch_estimate():
    assert pitch(_voice(130, 1.0), 16000) == pytest.approx(130, rel=0.05)
    assert pitch(_voice(260, 1.0), 16000) == pytest.approx(260, rel=0.05)
    assert pitch(np.zeros(16000, np.float32), 16000) is None


def test_short_line_reference_uses_same_voice_not_neighbour(tmp_path):
    import soundfile as sf

    rate = 16000
    # Man (130 Hz) 0-1 s, woman (260 Hz) 1.5-2.5 s, man again 10-13 s.
    audio = np.zeros(15 * rate, np.float32)
    audio[0:rate] = _voice(130, 1.0)
    audio[int(1.5 * rate):int(2.5 * rate)] = _voice(260, 1.0)
    audio[10 * rate:13 * rate] = _voice(130, 3.0)
    vocals = tmp_path / "vocals.wav"
    sf.write(vocals, audio, rate)
    spans = [(0, 1000), (1500, 2500), (10_000, 13_000)]
    refs = write_references(vocals, spans, tmp_path / "refs", min_s=5.4)
    for ref, expected in zip(refs, (130, 260, 130)):
        x, sr = sf.read(ref)
        assert x.size / sr >= 5.4
        assert pitch(x.astype(np.float32), sr) == pytest.approx(expected, rel=0.08)


def test_voice_mismatch_rule():
    from senpaidub.voicecheck import is_mismatch

    assert is_mismatch(250, 130)  # woman's line, man's voice
    assert is_mismatch(120, 240)
    assert not is_mismatch(250, 230)  # both high
    assert not is_mismatch(190, 140)  # original too close to the boundary to judge
    assert not is_mismatch(250, 170)  # clip too close to the boundary to judge
    assert not is_mismatch(None, 130) and not is_mismatch(250, None)


def test_config_turbo_points_to_qwen(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('tts = "turbo"\n')
    from senpaidub.config import validate_config

    with pytest.raises(UsageError, match="replaced by 'qwen'"):
        validate_config(resolve({"tts": None}, load_config(p)))
    assert DEFAULTS["tts"] == "qwen"
