"""End-to-end runs of the CLI on synthetic fixtures (MockTTS, no Demucs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import DURATION

pytestmark = pytest.mark.integration


def run_cli(*args: str | Path, env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "COLUMNS": "200", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "senpaidub.cli", *map(str, args)],
        capture_output=True, text=True, env=env,
    )  # fmt: skip


def streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def stream_hash(path: Path, spec: str) -> str:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", spec, "-c", "copy", "-f", "streamhash",
         "-hash", "md5", "-"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return out.stdout.split("=", 1)[-1].strip()


def test_list_tracks(sample_mkv):
    r = run_cli(sample_mkv, "--list-tracks")
    assert r.returncode == 0, r.stderr
    assert "subtitle" in r.stdout and "ass" in r.stdout and "jpn" in r.stdout


def test_dry_run_lists_dialogue_only(sample_mkv):
    r = run_cli(sample_mkv, "--dry-run", "--tts", "mock")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "Hello there." in out and "Gobta" in out
    assert "Tempest Village" not in out
    assert "5 spoken lines" in out


def test_full_pipeline_mock_no_separate(sample_mkv, tmp_path, isolated_dirs):
    out = tmp_path / "out.mkv"
    r = run_cli(sample_mkv, "--tts", "mock", "--no-separate", "-o", out)
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert not list(tmp_path.glob(".*partial*"))

    before, after = streams(sample_mkv), streams(out)
    a_before = [s for s in before["streams"] if s["codec_type"] == "audio"]
    a_after = [s for s in after["streams"] if s["codec_type"] == "audio"]
    assert len(a_after) == len(a_before) + 1
    new = a_after[-1]
    assert new["tags"]["language"] == "eng"
    assert new["tags"]["title"] == "English (TTS)"
    assert new["codec_name"] == "aac" and new["sample_rate"] == "48000"
    assert new["disposition"]["default"] == 0
    assert [s["codec_type"] for s in after["streams"]].count("subtitle") == 1
    assert float(after["format"]["duration"]) == pytest.approx(DURATION, abs=0.3)

    # Original video and audio are bit-for-bit copies.
    assert stream_hash(sample_mkv, "0:v:0") == stream_hash(out, "0:v:0")
    assert stream_hash(sample_mkv, "0:a:0") == stream_hash(out, "0:a:0")

    # Second run: output exists → skipped; with --force it re-uses the TTS cache.
    r2 = run_cli(sample_mkv, "--tts", "mock", "--no-separate", "-o", out)
    assert r2.returncode == 0 and "Skipped" in r2.stderr
    r3 = run_cli(sample_mkv, "--tts", "mock", "--no-separate", "-o", out, "--force")
    assert r3.returncode == 0, r3.stderr
    assert "TTS: 5 new, 0 cached" in r3.stderr  # caches are deleted after each episode by default
    # --keep-cache keeps them, so the next re-run makes no TTS calls.
    run_cli(sample_mkv, "--tts", "mock", "--no-separate", "-o", out, "--force", "--keep-cache")
    r4 = run_cli(sample_mkv, "--tts", "mock", "--no-separate", "-o", out, "--force")
    assert r4.returncode == 0, r4.stderr
    assert "TTS: 0 new, 5 cached" in r4.stderr


def test_speech_alignment(sample_mkv, tmp_path):
    """Speech starts within ~100 ms of each subtitle start."""
    import numpy as np
    import soundfile as sf

    from senpaidub.subtitles import filter_dialogue, load

    out = tmp_path / "speech.wav"
    r = run_cli(sample_mkv, "--tts", "mock", "--speech-only", "-o", out)
    assert r.returncode == 0, r.stderr
    audio, rate = sf.read(out)
    mono = np.abs(audio).mean(axis=1)
    lines = filter_dialogue(load(Path(__file__).parent / "fixtures" / "sample.ass")).lines
    tenth = rate // 10
    for line in lines:
        a = int(line.start / 1000 * rate)
        # Speech is present within 100 ms after the subtitle starts...
        first = np.flatnonzero(mono[a : a + tenth] > 0.01)
        assert first.size and first[0] <= tenth, f"no speech within 100 ms of {line.start} ms"
        # ...and doesn't start early (unless the previous line overflowed into this one).
        before = mono[a - tenth : a - rate // 200]
        if not (before > 0.01).any():
            continue
        assert line.text == "The end.", f"speech before {line.start} ms: {line.text}"
    assert "1 sped up, 1 overflowing" in r.stderr  # the long Benimaru line


def test_set_default(sample_mkv, tmp_path):
    out = tmp_path / "default.mkv"
    r = run_cli(sample_mkv, "--tts", "mock", "--no-separate", "--set-default", "-o", out)
    assert r.returncode == 0, r.stderr
    audio = [s for s in streams(out)["streams"] if s["codec_type"] == "audio"]
    assert [a["disposition"]["default"] for a in audio] == [0, 1]


def test_mp4_input_becomes_mkv(sample_mp4, tmp_path):
    r = run_cli(sample_mp4, "--tts", "mock", "--no-separate", "-o", tmp_path)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "Episode 02 [EN dub].mkv"
    info = streams(out)
    assert "matroska" in info["format"]["format_name"]
    assert [s["codec_type"] for s in info["streams"]] == ["video", "audio", "subtitle", "audio"]


def test_external_sub_file(no_subs_mkv, tmp_path):
    srt = Path(__file__).parent / "fixtures" / "sample.srt"
    r = run_cli(no_subs_mkv, "--tts", "mock", "--no-separate", "--sub-file", srt, "-o", tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "nosubs [EN dub].mkv").is_file()


def test_no_subtitles_exit_code(no_subs_mkv):
    r = run_cli(no_subs_mkv, "--tts", "mock", "--dry-run")
    assert r.returncode == 2
    assert "--sub-file" in r.stderr


def test_missing_input_exit_code(tmp_path):
    assert run_cli(tmp_path / "missing.mkv").returncode == 1


def test_batch_folder(sample_mkv, sample_mp4, tmp_path):
    src = tmp_path / "season"
    (src / "sub").mkdir(parents=True)
    os.symlink(sample_mkv, src / "Episode 10.mkv")
    os.symlink(sample_mp4, src / "sub" / "Episode 2.mp4")
    (src / "Episode 3.mkv").write_bytes(b"not a video")
    outdir = tmp_path / "out"
    r = run_cli(src, "--tts", "mock", "--no-separate", "-o", outdir)
    assert r.returncode == 3, r.stderr  # one broken file → ffprobe failure, batch continues
    assert (outdir / "Episode 10 [EN dub].mkv").is_file()
    assert (outdir / "sub" / "Episode 2 [EN dub].mkv").is_file()
    assert "2 dubbed, 0 skipped, 1 failed" in r.stderr
    assert r.stderr.index("Episode 3") < r.stderr.index("Episode 10")  # natural order


def test_stream_cache_key_is_stable_across_remuxes(sample_mkv, tmp_path):
    """Re-downloading a stream gives a different file but must reuse the same caches."""
    from senpaidub.download import audio_content_key

    copies = []
    for i in range(2):
        out = tmp_path / f"copy{i}.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(sample_mkv), "-map", "0", "-c", "copy", str(out)],
                       check=True)  # fmt: skip
        copies.append(out)
    assert copies[0].read_bytes() != copies[1].read_bytes()  # container metadata differs
    assert audio_content_key(copies[0]) == audio_content_key(copies[1])


def test_series_mode_files_originals_and_cleans_cache(sample_mkv, tmp_path, isolated_dirs):
    import shutil

    src = tmp_path / "downloads" / "Episode 01.mkv"
    src.parent.mkdir()
    shutil.copy(sample_mkv, src)
    lib = tmp_path / "anime"
    (isolated_dirs / "config.toml").write_text(f'library = "{lib.as_posix()}"\n')
    r = run_cli(src, "--tts", "mock", "--no-separate", "--series-name", "Test Show")
    assert r.returncode == 0, r.stderr
    show = lib / "Test Show"
    assert (show / "Test Show - 01 [EN dub].mkv").is_file()
    assert (show / "original" / "Episode 01.mkv").is_file() and not src.exists()  # moved, not copied
    assert (show / "original" / "subs" / "Test Show - 01.ass").read_text(encoding="utf-8-sig").count("Dialogue:") > 5
    cache = isolated_dirs / "cache"
    assert not list(cache.rglob("*.bin")) and not [p for p in cache.rglob("*") if p.is_file()]
