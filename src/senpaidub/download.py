"""Stream URL inputs (e.g. from Stremio): download once to a local file, then process it."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import ffmpeg
from .errors import ExternalToolError, UsageError
from .probe import MediaInfo, probe

log = logging.getLogger("senpaidub")

RECONNECT = [
    "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1",
    "-reconnect_delay_max", "30",
]  # fmt: skip
ATTEMPTS = 3


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def probe_url(url: str) -> MediaInfo:
    try:
        info = probe(url)
    except ExternalToolError as e:
        raise ExternalToolError(
            f"Couldn't read the stream at {url}.\n"
            "If it comes from Stremio, make sure Stremio is still running and the stream is playing "
            "or was recently opened, then copy a fresh URL.\n\n" + str(e)
        ) from e
    if info.encrypted:
        raise UsageError("This stream is encrypted (DRM). DRM-protected streams aren't supported.")
    return info


def download(
    url: str, info: MediaInfo, dest: Path, on_progress: Callable[[float], None] | None = None
) -> Path:
    """Copy every stream (no re-encode) into a local .mkv, retrying on dropped connections."""
    cmd: list[str | Path] = ["ffmpeg", "-y", *RECONNECT, "-i", url, "-map", "0", "-map", "-0:d?", "-c", "copy"]
    for i, s in enumerate(info.subtitles):
        if s.codec == "mov_text":
            cmd += [f"-c:s:{i}", "srt"]
    cmd += ["-f", "matroska", dest]
    last: ExternalToolError | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            ffmpeg.run_with_progress(cmd, on_progress or (lambda _s: None), what="Downloading stream")
            return dest
        except ExternalToolError as e:
            last = e
            log.warning("Download attempt %d/%d failed; retrying", attempt, ATTEMPTS)
    dest.unlink(missing_ok=True)
    raise ExternalToolError(
        f"Downloading {url} failed after {ATTEMPTS} attempts. Is the stream still available?\n{last}"
    )


def audio_content_key(path: Path) -> str:
    """Stable key for a downloaded stream: a hash of its audio packets.

    Hashing the whole file doesn't work — re-muxing the same stream writes different
    container metadata each time — but the audio packets themselves are identical.
    """
    proc = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a", "-c", "copy",
         "-f", "streamhash", "-hash", "sha256", "-"],
        what="Hashing the downloaded audio",
    )  # fmt: skip
    return hashlib.sha256(proc.stdout).hexdigest()


def sanitize_filename(name: str, max_len: int = 150) -> str:
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    # Reserved device names on Windows.
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\..*)?", name):
        name = f"_{name}"
    return name[:max_len].rstrip(" .") or "stream"


def name_from_url(url: str, info: MediaInfo, explicit: str | None) -> str:
    if explicit:
        base = explicit
    elif info.title.strip():
        base = info.title
    else:
        segs = [s for s in urlparse(url).path.split("/") if s]
        base = unquote(segs[-1]) if segs else "stream"
    base = re.sub(r"\.(mkv|mp4|m4v|webm|avi|ts|m3u8)$", "", base, flags=re.IGNORECASE)
    return sanitize_filename(base)
