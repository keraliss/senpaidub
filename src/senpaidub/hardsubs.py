"""Read burned-in ("hardsubbed") English subtitles off the picture, macOS only.

1. Decode the bottom strip of the video at a few frames per second.
2. Find subtitle text: thin white strokes with a dark outline on both sides. Line art rarely
   looks like that, and Japanese credits are dropped later by the text filter.
3. Split the frames into runs with the same text mask and OCR the middle frame of each run with
   Vision (via a small Swift helper compiled on first use).
4. Merge neighbouring runs with the same text into timed subtitle events.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysubs2

from .config import cache_dir
from .errors import ExternalToolError, UsageError

log = logging.getLogger("senpaidub")

FPS = 6
WIDTH = 1280
STRIP = 200  # px of the (1280-wide) frame bottom that holds 1–3 subtitle lines
MIN_TEXT_PX = 120  # fewer text-like pixels than this → no subtitle on screen
SAME_MASK = 0.6  # intersection/union of coarse masks above this → same subtitle
MAX_RUN = 2 * FPS  # re-read at least this often, in case a change of line went unnoticed
MERGE_RATIO = 0.85  # neighbouring runs whose texts are this similar are one subtitle
MERGE_GAP_S = 0.4
MIN_EVENT_S = 0.5  # credits flash by; real subtitles stay up longer
OCR_PROCS = 3

SWIFT_SRC = Path(__file__).with_name("ocr_vision.swift")


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("swiftc") is not None


def _helper() -> Path:
    digest = hashlib.sha256(SWIFT_SRC.read_bytes()).hexdigest()[:12]
    exe = cache_dir() / "bin" / f"ocr-vision-{digest}"
    if exe.is_file():
        return exe
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_suffix(".tmp")
    r = subprocess.run(["swiftc", "-O", str(SWIFT_SRC), "-o", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ExternalToolError(f"Couldn't build the OCR helper:\n{r.stderr[-2000:]}")
    tmp.rename(exe)
    return exe


# --------------------------------------------------------------------------- text masks


def _shift_or(a: np.ndarray, axis: int, steps: range) -> np.ndarray:
    """OR of `a` shifted by each step along `axis` (positive = towards higher indices)."""
    out = np.zeros_like(a)
    n = a.shape[axis]
    for s in steps:
        if s > 0:
            dst = [slice(None)] * 2
            src = [slice(None)] * 2
            dst[axis], src[axis] = slice(s, n), slice(0, n - s)
        else:
            dst = [slice(None)] * 2
            src = [slice(None)] * 2
            dst[axis], src[axis] = slice(0, n + s), slice(-s, n)
        out[tuple(dst)] |= a[tuple(src)]
    return out


def text_mask(rgb: np.ndarray) -> np.ndarray:
    """White pixels with dark pixels within 5 px on both sides (horizontally or vertically)."""
    white = rgb.min(axis=2) >= 190
    dark = rgb.max(axis=2) <= 80
    near = range(1, 6)
    h = _shift_or(dark, 1, near) & _shift_or(dark, 1, range(-1, -6, -1))
    v = _shift_or(dark, 0, near) & _shift_or(dark, 0, range(-1, -6, -1))
    m = white & (h | v)
    w = m.shape[1]
    m[:, : w // 10] = False  # subtitles are centred; ignore the outer edges
    m[:, w - w // 10 :] = False
    return m


def _coarse(m: np.ndarray, k: int = 4) -> np.ndarray:
    h, w = m.shape[0] // k * k, m.shape[1] // k * k
    return m[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3))


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    union = np.count_nonzero(a | b)
    return union == 0 or np.count_nonzero(a & b) / union >= SAME_MASK


def write_bmp(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    row = (w * 3 + 3) & ~3
    data = np.zeros((h, row), np.uint8)
    data[:, : w * 3] = rgb[::-1, :, ::-1].reshape(h, w * 3)
    header = b"BM" + (54 + data.size).to_bytes(4, "little") + b"\0\0\0\0" + (54).to_bytes(4, "little")
    info = (
        (40).to_bytes(4, "little") + w.to_bytes(4, "little") + h.to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (24).to_bytes(2, "little") + bytes(4)
        + data.size.to_bytes(4, "little") + bytes(16)
    )  # fmt: skip
    path.write_bytes(header + info + data.tobytes())


# --------------------------------------------------------------------------- runs


@dataclass
class Run:
    start: int  # frame index
    end: int  # exclusive
    image: Path | None = None
    text: str = ""


def find_runs(
    video: Path, work: Path, duration: float | None, on_progress: Callable[[float], None] | None = None
) -> list[Run]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-an", "-sn",
        # Undo anamorphic storage (e.g. 960x720 shown as 16:9) so text has its on-screen shape.
        "-vf", f"fps={FPS},scale=iw*sar:ih,setsar=1,scale={WIDTH}:-2,crop={WIDTH}:{STRIP}:0:ih-{STRIP}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]  # fmt: skip
    size = WIDTH * STRIP * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    runs: list[Run] = []
    frames: list[np.ndarray] = []  # frames of the current run (bounded below)
    prev: np.ndarray | None = None
    start = 0
    i = 0

    def close(end: int) -> None:
        if not frames:
            return
        img = work / f"run{len(runs):05d}.bmp"
        write_bmp(img, frames[len(frames) // 2])
        runs.append(Run(start, end, img))
        frames.clear()

    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(STRIP, WIDTH, 3)
        m = text_mask(rgb)
        present = np.count_nonzero(m) >= MIN_TEXT_PX
        c = _coarse(m) if present else None
        if c is not None and prev is not None and _same(c, prev) and i - start < MAX_RUN:
            frames.append(rgb)
        else:
            close(i)
            if c is not None:
                start = i
                frames.append(rgb)
        prev = c
        i += 1
        if on_progress and i % FPS == 0:
            on_progress(i / FPS)
    close(i)
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise ExternalToolError(f"ffmpeg failed reading video frames:\n{err[-2000:]}")
    log.debug("Hardsub scan: %d frames, %d candidate runs", i, len(runs))
    return runs


# --------------------------------------------------------------------------- OCR


def _keep_line(line: dict) -> bool:
    text = line["text"].strip()
    letters = sum(c.isascii() and c.isalpha() for c in text)
    centre = line["x"] + line["w"] / 2
    return (
        line["conf"] >= 0.4
        and letters >= 2
        and letters >= 0.6 * sum(not c.isspace() for c in text)
        and 0.38 <= centre <= 0.62  # subtitle lines are centred
        and 0.14 <= line["h"] <= 0.32  # ...and set in one font size (signs and credits differ)
    )


def clean(text: str) -> str:
    text = re.sub(r"\*{2,}", "", text)  # the release censors some words as ******
    text = re.sub(r"^[^\w\"'(\[¿¡-]+", "", text, flags=re.M)  # stray bullets/specks before a line
    text = re.sub(r"^-(?=\w)", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=\w)\. (?=[a-z])", ", ", text)  # Vision often reads a comma as a period
    text = re.sub(r"(?m)^1(?= [a-z])", "I", text)  # "1 suppose so"
    text = text.strip()
    if re.fullmatch(r'"[^"]+"', text, flags=re.S):  # a quoted caption (place, date, episode title)
        return ""
    return text


def ocr_runs(runs: list[Run]) -> None:
    if not runs:
        return
    exe = _helper()
    procs = []
    for k in range(OCR_PROCS):  # Vision runs mostly on one core per process
        chunk = runs[k::OCR_PROCS]
        if not chunk:
            continue
        f = tempfile.TemporaryFile("w+")
        f.write("\n".join(str(r.image) for r in chunk) + "\n")
        f.seek(0)
        procs.append(subprocess.Popen([str(exe)], stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    by_path = {}
    for proc in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise ExternalToolError(f"OCR helper failed:\n{err[-2000:]}")
        for line in out.splitlines():
            obj = json.loads(line)
            by_path[obj["path"]] = obj["lines"]
    for run in runs:
        lines = [l for l in by_path.get(str(run.image), []) if _keep_line(l)]
        lines.sort(key=lambda l: -l["y"])  # top line first
        run.text = clean("\n".join(l["text"].strip() for l in lines))


def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= MERGE_RATIO


def to_events(runs: list[Run]) -> pysubs2.SSAFile:
    """Merge runs with the same text (OCR jitter, short flickers) into subtitle events."""
    groups: list[list[Run]] = []
    for run in runs:
        if not run.text:
            continue
        last = groups[-1][-1] if groups else None
        if last and (run.start - last.end) / FPS <= MERGE_GAP_S and _similar(run.text, last.text):
            groups[-1].append(run)
        else:
            groups.append([run])
    subs = pysubs2.SSAFile()
    for g in groups:
        start, end = g[0].start / FPS, g[-1].end / FPS
        if end - start < MIN_EVENT_S:
            continue
        text = max(g, key=lambda r: r.end - r.start).text  # the longest-lived reading
        subs.append(pysubs2.SSAEvent(start=int(start * 1000), end=int(end * 1000), text=text.replace("\n", "\\N")))
    return subs


def extract(
    video: Path, out: Path, duration: float | None, on_progress: Callable[[float], None] | None = None
) -> Path:
    """OCR the burned-in subtitles of `video` into an .srt at `out`."""
    if not available():
        raise UsageError("Reading burned-in subtitles needs macOS (Vision OCR) with Xcode command-line tools.")
    with tempfile.TemporaryDirectory(prefix="senpaidub-ocr-") as tmp:
        runs = find_runs(video, Path(tmp), duration, on_progress)
        ocr_runs(runs)
    subs = to_events(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(out), format_="srt")
    return out
