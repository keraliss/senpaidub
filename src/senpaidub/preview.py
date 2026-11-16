"""Inspect a video for the web UI's setup screen. Never runs OCR, Demucs or TTS."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from . import ffmpeg, hardsubs, subtitles
from .errors import SenpaiDubError
from .probe import MediaInfo, Stream, choose_subtitle_stream, probe


def _resolution(video: Path) -> str | None:
    try:
        proc = ffmpeg.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "json", str(video)],
            what="Reading video size",
        )  # fmt: skip
        s = (json.loads(proc.stdout).get("streams") or [{}])[0]
        if s.get("width") and s.get("height"):
            return f"{int(s['width'])}x{int(s['height'])}"
    except (SenpaiDubError, ValueError, TypeError, IndexError, AttributeError):
        pass
    return None


def _count(subs, info: MediaInfo, include_styles, exclude_styles) -> tuple[int, int]:
    result = subtitles.filter_dialogue(
        subs, include_styles, exclude_styles,
        skip_songs=True,
        extra_song_spans=subtitles.chapter_song_spans(info.chapters),
    )  # fmt: skip
    return len(result.lines), len(result.skipped)


def _track(video: Path, info: MediaInfo, s: Stream, chosen: int | None, inc, exc) -> dict:
    d = {
        "index": s.index, "codec": s.codec, "language": s.language or None,
        "title": s.title or None, "is_text": s.is_text_sub, "chosen": s.index == chosen,
        "lines": None, "skipped": None, "error": None,
    }  # fmt: skip
    if not s.is_text_sub:
        d["error"] = "image subtitles aren't supported"
        return d
    try:
        with tempfile.TemporaryDirectory(prefix="senpaidub-preview-") as tmp:
            subs = subtitles.load(subtitles.extract(video, s, Path(tmp)))
        d["lines"], d["skipped"] = _count(subs, info, inc, exc)
    except Exception as e:  # per-track problems must not break the whole preview
        d["error"] = re.sub(r"\s+", " ", str(e)).strip()[:300] or e.__class__.__name__
    return d


def preview(video: Path, sub_file: Path | None = None,
            include_styles: str | None = None, exclude_styles: str | None = None) -> dict:
    video = Path(video)
    info = probe(video)
    try:
        chosen = choose_subtitle_stream(info, None).index
    except SenpaiDubError:
        chosen = None

    sf = None
    if sub_file is not None:
        sub_file = Path(sub_file)
        sf = {"name": sub_file.name, "lines": None, "skipped": None, "error": None}
        try:
            sf["lines"], sf["skipped"] = _count(subtitles.load(sub_file), info, include_styles, exclude_styles)
        except Exception as e:
            sf["error"] = re.sub(r"\s+", " ", str(e)).strip()[:300] or e.__class__.__name__

    tracks = [_track(video, info, s, chosen, include_styles, exclude_styles) for s in info.subtitles]
    has_text = any(s.is_text_sub for s in info.subtitles)
    ocr = hardsubs.available()
    if sf is not None:
        source = "sub_file"
    elif has_text:
        source = "track"
    elif not info.subtitles and ocr:
        source = "ocr"
    else:
        source = "none"
    try:
        size = video.stat().st_size
    except OSError:
        size = 0
    return {
        "path": str(video.resolve()), "name": video.name, "duration": info.duration, "size": size,
        "resolution": _resolution(video), "encrypted": info.encrypted,
        "audio": [
            {"index": a.index, "codec": a.codec, "language": a.language or None,
             "title": a.title or None, "channels": a.channels}
            for a in info.audio
        ],
        "subtitles": tracks, "sub_file": sf,
        "ocr_available": ocr, "needs_ocr": not has_text and sf is None, "auto_source": source,
    }  # fmt: skip
