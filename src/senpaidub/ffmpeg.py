"""Single entry point for every ffmpeg/ffprobe invocation."""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from .errors import ExternalToolError

log = logging.getLogger("senpaidub")

INSTALL_HINT = """\
ffmpeg and ffprobe must be installed and on your PATH.
  macOS:   brew install ffmpeg
  Windows: winget install Gyan.FFmpeg   (or: choco install ffmpeg / scoop install ffmpeg)
  Linux:   sudo apt install ffmpeg      (or your distribution's package manager)"""


def check_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise ExternalToolError(f"Missing: {', '.join(missing)}\n{INSTALL_HINT}")


def _fmt(cmd: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run(
    cmd: Sequence[str | Path],
    *,
    input: bytes | None = None,
    what: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run ffmpeg/ffprobe, returning the completed process (stdout/stderr as bytes).

    Raises ExternalToolError with the tail of stderr on failure.
    """
    args = [str(c) for c in cmd]
    if args[0] == "ffmpeg":
        args[1:1] = ["-hide_banner", "-nostdin"] if input is None else ["-hide_banner"]
    log.debug("$ %s", _fmt(args))
    try:
        proc = subprocess.run(args, input=input, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise ExternalToolError(f"{args[0]} not found.\n{INSTALL_HINT}") from e
    if proc.returncode != 0:
        raise ExternalToolError(_failure_message(args, proc.stderr, what))
    return proc


def run_with_progress(
    cmd: Sequence[str | Path],
    on_progress: Callable[[float], None],
    *,
    what: str | None = None,
) -> None:
    """Run an ffmpeg command, calling on_progress(seconds_processed) as it goes."""
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-progress", "pipe:1", "-nostats"]
    args += [str(c) for c in cmd[1:]]
    log.debug("$ %s", _fmt(args))
    stderr_tail: list[str] = []
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace"
        )
    except FileNotFoundError as e:
        raise ExternalToolError(f"ffmpeg not found.\n{INSTALL_HINT}") from e

    # Drain stderr on a thread so the pipe can never fill up and block ffmpeg.
    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line)
            del stderr_tail[:-60]

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.isdigit():
            on_progress(int(value) / 1_000_000)
    proc.wait()
    t.join(timeout=5)
    if proc.returncode != 0:
        raise ExternalToolError(_failure_message(args, "".join(stderr_tail).encode(), what))


def _failure_message(args: list[str], stderr: bytes, what: str | None) -> str:
    tail = stderr.decode(errors="replace").strip().splitlines()[-15:]
    head = f"{what} failed" if what else f"{args[0]} failed"
    return f"{head}.\nCommand: {_fmt(args)}\n" + "\n".join(tail)
