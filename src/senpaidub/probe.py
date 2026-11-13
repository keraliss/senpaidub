"""ffprobe wrapper and stream classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg
from .errors import ExternalToolError, NoSubtitlesError, UsageError

TEXT_SUB_CODECS = {"ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"}
IMAGE_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "dvb_teletext"}
ENGLISH = {"eng", "en", "en-us", "en-gb"}
JAPANESE = {"jpn", "ja"}
# Subtitle tracks that only carry signs/songs; not the dialogue track we want.
_SIGNS_TRACK_RE = re.compile(r"\b(signs?|songs?|forced)\b", re.IGNORECASE)


@dataclass
class Stream:
    index: int
    type: str  # "video" | "audio" | "subtitle" | "attachment" | "data"
    codec: str
    language: str = ""
    title: str = ""
    default: bool = False
    forced: bool = False
    channels: int | None = None
    encrypted: bool = False

    @property
    def is_text_sub(self) -> bool:
        return self.type == "subtitle" and self.codec in TEXT_SUB_CODECS

    @property
    def is_image_sub(self) -> bool:
        return self.type == "subtitle" and not self.is_text_sub


@dataclass
class MediaInfo:
    path: str
    streams: list[Stream] = field(default_factory=list)
    duration: float = 0.0
    title: str = ""
    format_name: str = ""
    chapters: list[dict] = field(default_factory=list)

    def of_type(self, t: str) -> list[Stream]:
        return [s for s in self.streams if s.type == t]

    @property
    def audio(self) -> list[Stream]:
        return self.of_type("audio")

    @property
    def subtitles(self) -> list[Stream]:
        return self.of_type("subtitle")

    @property
    def encrypted(self) -> bool:
        return any(s.encrypted for s in self.streams)


def probe(target: str | Path) -> MediaInfo:
    proc = ffmpeg.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters", "-of", "json",
         str(target)],
        what=f"Reading stream info from {target}",
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ExternalToolError(f"ffprobe returned unreadable output for {target}") from e
    return parse_probe(data, str(target))


def parse_probe(data: dict, path: str = "") -> MediaInfo:
    fmt = data.get("format", {}) or {}
    info = MediaInfo(
        path=path,
        duration=_float(fmt.get("duration")),
        title=(fmt.get("tags") or {}).get("title", "") or "",
        format_name=fmt.get("format_name", ""),
    )
    info.chapters = list(data.get("chapters") or [])
    for s in data.get("streams", []):
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        tag = (s.get("codec_tag_string") or "").lower()
        side = json.dumps(s.get("side_data_list", [])).lower()
        info.streams.append(
            Stream(
                index=int(s["index"]),
                type=s.get("codec_type", "") or "",
                codec=s.get("codec_name", "") or "",
                language=(tags.get("language") or "").lower(),
                title=tags.get("title", "") or "",
                default=bool(disp.get("default")),
                forced=bool(disp.get("forced")),
                channels=s.get("channels"),
                encrypted=tag in {"encv", "enca", "encs"} or "encryption" in side,
            )
        )
    if not info.duration:
        info.duration = max((_float(s.get("duration")) for s in data.get("streams", [])), default=0.0)
    return info


def _float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def choose_subtitle_stream(info: MediaInfo, sub_track: int | None) -> Stream:
    """Pick the subtitle stream to voice. `sub_track` is the ffprobe stream index."""
    subs = info.subtitles
    if sub_track is not None:
        match = [s for s in info.streams if s.index == sub_track]
        if not match:
            raise UsageError(f"There is no stream #{sub_track}. Use --list-tracks to see them.")
        s = match[0]
        if s.type != "subtitle":
            raise UsageError(f"Stream #{sub_track} is {s.type}, not a subtitle track.")
        if not s.is_text_sub:
            raise NoSubtitlesError(
                f"Stream #{sub_track} is an image-based subtitle ({s.codec}); "
                "OCR isn't supported yet. Pick a text track or pass --sub-file."
            )
        return s

    if not subs:
        raise NoSubtitlesError(
            "This file has no subtitle tracks. Supply English subtitles with --sub-file PATH (.srt or .ass)."
        )
    text = [s for s in subs if s.is_text_sub]
    if not text:
        codecs = ", ".join(sorted({s.codec for s in subs}))
        raise NoSubtitlesError(
            f"Only image-based subtitles found ({codecs}); OCR isn't supported yet. "
            "Supply a text subtitle file with --sub-file PATH."
        )

    def is_full_dialogue(s: Stream) -> bool:
        return not s.forced and not _SIGNS_TRACK_RE.search(s.title)

    english = [s for s in text if s.language in ENGLISH]
    for pool in (
        [s for s in english if is_full_dialogue(s)],
        english,
        [s for s in text if is_full_dialogue(s)],
        text,
    ):
        if pool:
            return pool[0]
    raise AssertionError("unreachable")


def choose_audio_stream(info: MediaInfo) -> Stream:
    audio = info.audio
    if not audio:
        raise UsageError(f"{info.path} has no audio stream.")
    for pool in (
        [s for s in audio if s.language in JAPANESE],
        [s for s in audio if s.default],
        audio,
    ):
        if pool:
            if len(pool) > 1:
                defaults = [s for s in pool if s.default]
                if defaults:
                    return defaults[0]
            return pool[0]
    raise AssertionError("unreachable")
