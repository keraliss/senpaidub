from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DURATION = 12


def _ffmpeg(*args: str | Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path_factory, monkeypatch):
    """Never touch the real user cache/config during tests."""
    base = tmp_path_factory.mktemp("appdirs")
    monkeypatch.setenv("SENPAIDUB_CACHE", str(base / "cache"))
    monkeypatch.setenv("SENPAIDUB_CONFIG", str(base / "config.toml"))
    return base


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def sample_mkv(tmp_path_factory) -> Path:
    """12 s synthetic episode: test pattern + 440 Hz tone + the handcrafted .ass embedded."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("media")
    out = d / "Episode 01.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=duration={DURATION}:size=320x240:rate=24",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}:sample_rate=48000",
        "-i", FIXTURES / "sample.ass",
        "-map", "0", "-map", "1", "-map", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "ass",
        "-metadata:s:a:0", "language=jpn", "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=Full Subtitles",
        out,
    )  # fmt: skip
    return out


@pytest.fixture(scope="session")
def sample_mp4(tmp_path_factory) -> Path:
    """Same idea as an .mp4 with mov_text subtitles (exercises the mp4 → mkv path)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("media_mp4")
    out = d / "Episode 02.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=6:sample_rate=48000",
        "-i", FIXTURES / "sample.srt",
        "-map", "0", "-map", "1", "-map", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        out,
    )  # fmt: skip
    return out


@pytest.fixture(scope="session")
def no_subs_mkv(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media_nosubs") / "nosubs.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=6:sample_rate=48000",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", out,
    )  # fmt: skip
    return out
