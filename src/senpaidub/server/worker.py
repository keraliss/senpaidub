"""Job worker: runs one dub in its own process and reports progress as plain dicts on a queue.

`job_main(spec, q)` is the `multiprocessing` target; it calls `run_job(spec, q)`. Messages (all JSON-able dicts):
  {"t": "stage", "key", "label", "total", "unit", "group"}
  {"t": "progress", "key", "completed", "total"}
  {"t": "stage_end", "key"}
  {"t": "info", "msg"} / {"t": "warn", "msg"}
and finally {"t": "result", "result": {...}} or {"t": "error", "message", "exit_code"}.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console

from ..config import cache_dir, load_config, resolve, validate_config
from ..download import sanitize_filename
from ..errors import SenpaiDubError, UsageError
from ..pipeline import Options, Result, default_output_name, resolve_output
from ..report import STAGE_GROUPS, Reporter

PROGRESS_INTERVAL = 0.25  # seconds between progress messages per stage (~4/s)
UNSORTED = "Unsorted"

# Option keys a job spec may override (same names as config.toml); anything else is ignored.
CONFIG_KEYS = (
    "voice", "voice_map", "separate", "duck_db", "max_speedup", "include_styles", "exclude_styles",
    "set_default", "codec", "device", "concurrency", "speech_gain", "keep_cache", "qwen_mode",
)  # fmt: skip


class _QueueStage:
    def __init__(self, reporter: QueueReporter, key: str, total: float | None) -> None:
        self._r, self.key = reporter, key
        self.completed: float = 0.0
        self.total = total
        self._lock = threading.Lock()
        self._last_sent = 0.0
        self._sent: tuple[float, float | None] | None = None

    def advance(self, n: float = 1) -> None:
        with self._lock:
            self.completed += n
            self._maybe_send()

    def update(self, completed: float | None = None, total: float | None = None) -> None:
        with self._lock:
            if completed is not None:
                self.completed = completed
            if total is not None:
                self.total = total
            self._maybe_send()

    def _maybe_send(self) -> None:
        now = time.monotonic()
        final = self.total is not None and self.completed >= self.total
        if final or now - self._last_sent >= PROGRESS_INTERVAL:
            self._send(now)

    def _send(self, now: float) -> None:
        if self._sent == (self.completed, self.total):
            return
        self._last_sent = now
        self._sent = (self.completed, self.total)
        self._r.put({"t": "progress", "key": self.key, "completed": self.completed, "total": self.total})

    def flush(self) -> None:
        with self._lock:
            if self._sent is not None or self.completed:
                self._send(time.monotonic())


class QueueReporter(Reporter):
    """Reporter that turns pipeline events into dicts on a (multiprocessing) queue. Thread-safe."""

    def __init__(self, q: Any) -> None:
        self.q = q
        self._lock = threading.Lock()
        self._logged: deque[str] = deque(maxlen=20)  # warnings already forwarded from logging

    def put(self, msg: dict) -> None:
        with self._lock:
            self.q.put(msg)

    @contextmanager
    def _stage(self, key: str, label: str, total: float | None, unit: str) -> Iterator[_QueueStage]:
        st = _QueueStage(self, key, total)
        self.put({"t": "stage", "key": key, "label": label, "total": total, "unit": unit,
                  "group": STAGE_GROUPS.get(key)})  # fmt: skip
        try:
            yield st
        finally:
            st.flush()
            self.put({"t": "stage_end", "key": key})

    def stage(
        self, key: str, label: str, total: float | None = None, unit: str = "items"
    ) -> AbstractContextManager[_QueueStage]:
        return self._stage(key, label, total, unit)

    def info(self, msg: str, style: str | None = None) -> None:
        self.put({"t": "info", "msg": msg})

    def warn(self, msg: str) -> None:
        if msg in self._logged:  # the same warning already came through the logging handler
            self._logged.remove(msg)
            return
        self.put({"t": "warn", "msg": msg})

    def log_warning(self, msg: str) -> None:
        self._logged.append(msg)
        self.put({"t": "warn", "msg": msg})


class _LogForwarder(logging.Handler):
    """Forward the app logger's warnings (e.g. Demucs CPU fallback) as warn messages."""

    def __init__(self, reporter: QueueReporter) -> None:
        super().__init__(logging.WARNING)
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.reporter.log_warning(record.getMessage())
        except Exception:  # never let logging break a dub
            pass


def _is_upload(path: Path) -> bool:
    try:
        path.resolve().relative_to((cache_dir() / "uploads").resolve())
        return True
    except ValueError:
        return False


def build_options(spec: dict) -> Options:
    """Options for one job, built the way cli.py builds them (config file < spec values).

    spec keys: input, tts, sub_file, sub_track, use_ocr, series_name, force, library (optional;
    the configured library otherwise), output (optional), options (optional dict of config.toml keys).
    """
    return _plan(spec)[0]


def _plan(spec: dict) -> tuple[Options, bool]:
    """(options, output_is_folder). The flag is True when opts.output is a folder this module
    chose (series or Unsorted) and run_job must create; a name like "Dr. Stone" has a suffix,
    so the folder can't be told from a file path by looking at it."""
    overrides = {k: v for k, v in (spec.get("options") or {}).items() if k in CONFIG_KEYS}
    cfg = resolve({**overrides, "tts": spec.get("tts")}, load_config())
    validate_config(cfg)
    sub_file = Path(spec["sub_file"]) if spec.get("sub_file") else None
    if sub_file is not None and not sub_file.is_file():
        raise UsageError(f"Subtitle file not found: {sub_file}")
    series = spec.get("series_name") or None
    opts = Options(
        sub_track=spec.get("sub_track"),
        sub_file=sub_file,
        tts=cfg["tts"],
        voice=cfg["voice"],
        voice_map=Path(cfg["voice_map"]).expanduser() if cfg["voice_map"] else None,
        separate=bool(cfg["separate"]),
        duck_db=float(cfg["duck_db"]),
        max_speedup=float(cfg["max_speedup"]),
        include_styles=cfg["include_styles"] or None,
        exclude_styles=cfg["exclude_styles"] or None,
        set_default=bool(cfg["set_default"]),
        output=Path(spec["output"]).expanduser() if spec.get("output") else None,
        force=bool(spec.get("force")),
        codec=cfg["codec"],
        device=cfg["device"],
        concurrency=int(cfg["concurrency"]),
        qwen_mode=cfg["qwen_mode"],
        speech_gain_db=float(cfg["speech_gain"]),
        series_name=series,
        collect_originals=bool(series),
        clean_cache=not cfg["keep_cache"],
        force_ocr=bool(spec.get("use_ocr")),
    )
    library = Path(spec.get("library") or cfg["library"]).expanduser()
    folder = False
    if opts.output is None:
        if series:
            opts = replace(opts, output=library / sanitize_filename(series))
            folder = True
        elif _is_upload(Path(spec["input"])):
            # A staged upload has no home of its own; don't write the dub into the cache.
            opts = replace(opts, output=library / UNSORTED)
            folder = True
    return opts, folder


def expected_output(spec: dict) -> Path | None:
    """Where the dub will be written (None if the options can't be built)."""
    from ..pipeline import _series_stem

    try:
        opts, folder = _plan(spec)
    except Exception:
        return None
    path = Path(spec["input"])
    stem = _series_stem(path.name, opts) or path.stem
    if folder and opts.output is not None:  # a folder run_job will create
        return opts.output / default_output_name(stem, opts.speech_only)
    return resolve_output(stem, path.parent, opts)


def partial_output(spec: dict) -> Path | None:
    """The temp file mux writes before renaming (left behind if a job is killed)."""
    out = expected_output(spec)
    return out.with_name(f".{out.stem}.partial.mkv") if out is not None else None


def result_dict(res: Result) -> dict:
    return {
        "voiced": res.voiced,
        "sped_up": res.sped_up,
        "overflowing": res.overflowing,
        "voice_mismatches": res.voice_mismatches,
        "voice_fixed": res.voice_fixed,
        "tts_cached": res.tts_cached,
        "tts_new": res.tts_new,
        "skipped_reason": res.skipped_reason,
        "output": str(res.output) if res.output else None,
    }


def _detach_from_terminal() -> None:
    """Keep Ctrl+C on `senpaidub serve` away from this worker and its ffmpeg/qwen children.

    The server stops workers itself (and puts their jobs back to waiting); a SIGINT reaching them
    first would make them report a failure. A new session leaves the terminal's process group;
    an ignored SIGINT is inherited by the children as well.
    """
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # not the main thread (tests)
        pass


def run_job(spec: dict, q: Any) -> None:
    """Dub spec["input"], sending progress and the outcome on `q`."""
    reporter = QueueReporter(q)
    log = logging.getLogger("senpaidub")
    log.handlers[:] = [_LogForwarder(reporter)]
    log.setLevel(logging.INFO)
    log.propagate = False
    console = Console(file=io.StringIO(), stderr=False, quiet=True)
    try:
        from ..ffmpeg import check_tools
        from ..pipeline import process_file

        check_tools()
        opts, folder = _plan(spec)
        if folder and opts.output is not None:
            opts.output.mkdir(parents=True, exist_ok=True)
        res = process_file(Path(spec["input"]), opts, console, reporter=reporter)
        reporter.put({"t": "result", "result": result_dict(res)})
    except SenpaiDubError as e:
        reporter.put({"t": "error", "message": str(e), "exit_code": e.exit_code})
    except BaseException as e:  # noqa: BLE001 - report everything, including KeyboardInterrupt
        msg = str(e) or type(e).__name__
        reporter.put({"t": "error", "message": f"{type(e).__name__}: {msg}" if str(e) else msg,
                      "exit_code": 3})  # fmt: skip


def job_main(spec: dict, q: Any) -> None:
    """The worker process's target: detach from the terminal's Ctrl+C, then run the job."""
    _detach_from_terminal()
    run_job(spec, q)
