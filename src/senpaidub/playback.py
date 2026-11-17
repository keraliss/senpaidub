"""Pause GPU work while the user watches a video (gentle mode).

Video players share the GPU with Demucs and Qwen, and macOS doesn't favour the player: a long
batch of GPU work makes playback stutter (seen with Stremio). `nice` only covers the CPU. So in
gentle mode the dub waits between chunks while a video is playing.

Detection is macOS-only: players (Stremio, browsers, IINA, QuickTime, VLC...) hold a
PreventUserIdleDisplaySleep power assertion while a video plays, listed by `pmset -g assertions`.
Elsewhere nothing is detected and the dub never waits.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable

log = logging.getLogger("senpaidub")

CHECK_EVERY_S = 5.0  # while paused, how often to look again
CACHE_S = 2.0  # don't run pmset more often than this
# Holders that aren't video: our own keep-awake, and macOS itself.
IGNORED = {"caffeinate", "powerd", "WindowServer", "coreaudiod", "loginwindow"}
_LINE_RE = re.compile(r"pid (\d+)\(([^)]*)\):.*\bPreventUserIdleDisplaySleep\b")

_cache: tuple[float, str | None] = (0.0, None)


def parse_assertions(text: str, own_pid: int | None = None) -> str | None:
    """Name of a process holding a display-sleep assertion (i.e. playing video), or None."""
    for line in text.splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        pid, name = int(m.group(1)), m.group(2)
        if name in IGNORED or pid == own_pid:
            continue
        return name
    return None


def video_playing() -> str | None:
    """The app playing a video right now, if any (macOS only; cached for a moment)."""
    global _cache
    if sys.platform != "darwin" or os.environ.get("SENPAIDUB_PAUSE_FOR_VIDEO") == "0":
        return None
    now = time.monotonic()
    if now - _cache[0] < CACHE_S:
        return _cache[1]
    try:
        out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    _cache = (now, parse_assertions(out, os.getpid()))
    return _cache[1]


def wait_while_video_plays(
    playing: Callable[[], str | None] = video_playing,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Block while a video is playing. Returns the seconds spent waiting."""
    app = playing()
    if app is None:
        return 0.0
    log.info("Paused while %s plays a video; continuing when it stops.", app)
    waited = 0.0
    while playing() is not None:
        sleep(CHECK_EVERY_S)
        waited += CHECK_EVERY_S
    log.info("Video stopped; continuing (paused %.0f s).", waited)
    return waited
