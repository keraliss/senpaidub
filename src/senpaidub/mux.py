"""Final mux: copy every original stream, append the dub track."""

from __future__ import annotations

import os
from pathlib import Path

from . import ffmpeg
from .probe import MediaInfo

TRACK_TITLE = "English (TTS)"
# Subtitle codecs Matroska can't hold as-is; converted to SRT (text only, timing unchanged).
_MKV_INCOMPATIBLE_SUBS = {"mov_text"}


def mux(
    video: Path, info: MediaInfo, dub_audio: Path, out: Path, *, set_default: bool = False
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.stem}.partial.mkv")
    new_idx = len(info.audio)  # index of the new track among output audio streams

    cmd: list[str | Path] = [
        "ffmpeg", "-y", "-v", "error", "-i", video, "-i", dub_audio,
        "-map", "0", "-map", "-0:d?", "-map", "1:a:0",
        "-c", "copy",
    ]  # fmt: skip
    for i, s in enumerate(info.subtitles):
        if s.codec in _MKV_INCOMPATIBLE_SUBS:
            cmd += [f"-c:s:{i}", "srt"]
    cmd += [
        f"-metadata:s:a:{new_idx}", "language=eng",
        f"-metadata:s:a:{new_idx}", f"title={TRACK_TITLE}",
    ]  # fmt: skip
    if set_default:
        for i in range(new_idx):
            cmd += [f"-disposition:a:{i}", "0"]
        cmd += [f"-disposition:a:{new_idx}", "default"]
    else:
        cmd += [f"-disposition:a:{new_idx}", "0"]
    cmd += ["-max_interleave_delta", "0", "-f", "matroska", partial]

    try:
        ffmpeg.run(cmd, what="Muxing the output file")
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    return out
