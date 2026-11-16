"""Job queue for the web UI: batches of dub jobs, each run in its own worker process.

State lives in memory and is written through to SQLite (one JSON row per batch and per job), so
the queue survives a restart. A scheduler thread starts waiting jobs; one reader thread per running
job drains its worker's message queue. Subscribers (the SSE endpoint) get events on an asyncio
queue fed thread-safely from those threads.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..config import DEFAULTS, data_dir, load_config, resolve
from ..download import sanitize_filename
from ..report import GROUPS
from . import files, worker

GROUP_WEIGHTS = {"subtitles": 0.05, "separate": 0.20, "voice": 0.55, "mix": 0.15, "mux": 0.05}
# Each stage's share of its group, in the order the stages run (stages that don't run are jumped).
STAGE_WEIGHTS: dict[str, tuple[tuple[str, float], ...]] = {
    "subtitles": (("download", 0.5), ("ocr", 0.4), ("subtitles", 0.1)),
    "separate": (("extract", 0.15), ("separate", 0.75), ("cache_stems", 0.10)),
    "voice": (("voices", 0.02), ("tts", 0.85), ("voicecheck", 0.13)),
    "mix": (("fit", 0.2), ("mix", 0.5), ("encode", 0.3)),
    "mux": (("mux", 1.0),),
}
STATUSES = ("waiting", "running", "done", "failed", "skipped", "cancelled")
TERMINAL = frozenset({"done", "failed", "skipped", "cancelled"})
ENGINES = ("qwen", "edge", "google", "mock")
LOG_SIZE = 50
EMIT_INTERVAL = 0.25  # ≤4 progress events per second per job
TICK = 0.25
SUBSCRIBER_QUEUE = 500

JOB_KEYS = (
    "id", "batch_id", "position", "name", "input", "status", "group", "stage_key", "stage_label",
    "stage_done", "stage_total", "stage_unit", "progress", "started_at", "finished_at", "output",
    "result", "error", "exit_code", "tts", "sub_file", "sub_track", "use_ocr",
)  # fmt: skip


class JobStateError(ValueError):
    """The job or batch is in the wrong state for that action (HTTP 409)."""


class Handle(Protocol):
    """A running job as the manager sees it (a worker process, or a fake in tests)."""

    q: Any  # .get(timeout=...) raising queue.Empty

    def is_alive(self) -> bool: ...
    def kill(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...

    @property
    def exitcode(self) -> int | None: ...


Runner = Callable[[dict], Handle]


class ProcessHandle:
    """worker.run_job in a spawned process; kill() takes down its whole process tree."""

    def __init__(self, spec: dict) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.q = ctx.Queue()
        self.proc = ctx.Process(target=worker.job_main, args=(spec, self.q), name="senpaidub-job")
        self.proc.start()

    def is_alive(self) -> bool:
        return self.proc.is_alive()

    def kill(self) -> None:
        import psutil

        try:
            parent = psutil.Process(self.proc.pid)
            children = parent.children(recursive=True)  # the qwen backend, ffmpeg, ...
        except psutil.Error:
            children = []
        for c in children:
            try:
                c.kill()
            except psutil.Error:
                pass
        try:
            self.proc.kill()
        except (OSError, ValueError):
            pass
        psutil.wait_procs(children, timeout=5)

    def join(self, timeout: float | None = None) -> None:
        self.proc.join(timeout)

    @property
    def exitcode(self) -> int | None:
        return self.proc.exitcode


def process_runner(spec: dict) -> Handle:
    return ProcessHandle(spec)


def _group_progress(
    group: str | None, done: float | None, total: float | None, ended: bool, key: str | None = None
) -> float:
    """Overall progress (0-1) for a job in stage `key` of `group`.

    Without a known `key` the stage spans its whole group.
    """
    if group not in GROUP_WEIGHTS:
        return 0.0
    i = GROUPS.index(group)
    base = sum(GROUP_WEIGHTS[g] for g in GROUPS[:i])
    if ended:
        frac = 1.0
    elif total:
        frac = max(0.0, min(1.0, (done or 0) / total))
    else:
        frac = 0.0
    offset, width = 0.0, 1.0
    for k, w in STAGE_WEIGHTS.get(group, ()):
        if k == key:
            width = w
            break
        offset += w
    else:
        offset = 0.0
    return base + GROUP_WEIGHTS[group] * (offset + width * frac)


def _offer(q: asyncio.Queue, event: dict) -> None:
    """Runs on the subscriber's loop: drop the oldest event rather than block."""
    while True:
        try:
            q.put_nowait(event)
            return
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return


class JobManager:
    def __init__(self, db_path: Path | None = None, library: Path | None = None, runner: Runner | None = None):
        self.db_path = Path(db_path) if db_path else data_dir() / "web.db"
        self.library = Path(library).expanduser() if library else None
        self._runner: Runner = runner or process_runner
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._batches: dict[str, dict] = {}
        self._jobs: dict[str, dict] = {}
        self._handles: dict[str, Handle] = {}
        self._readers: dict[str, threading.Thread] = {}
        self._cancelling: set[str] = set()
        self._dirty: set[str] = set()
        self._last_emit: dict[str, float] = {}
        self._subs: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self._subs_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, data TEXT NOT NULL)")
        self._db.commit()
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        with self._lock:
            for (data,) in self._db.execute("SELECT data FROM batches"):
                b = json.loads(data)
                self._batches[b["id"]] = b
            for (data,) in self._db.execute("SELECT data FROM jobs"):
                j = json.loads(data)
                if j["batch_id"] not in self._batches:
                    continue
                self._jobs[j["id"]] = j
                if j["status"] == "running":  # interrupted by a restart: start over
                    self._reset(j)
                    self._save_job(j)
            self._db.commit()

    def _save_job(self, j: dict, commit: bool = False) -> None:
        self._db.execute("INSERT OR REPLACE INTO jobs (id, batch_id, data) VALUES (?, ?, ?)",
                         (j["id"], j["batch_id"], json.dumps(j)))  # fmt: skip
        if commit:
            self._db.commit()

    def _save_batch(self, b: dict, commit: bool = False) -> None:
        self._db.execute("INSERT OR REPLACE INTO batches (id, data) VALUES (?, ?)", (b["id"], json.dumps(b)))
        if commit:
            self._db.commit()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._scheduler, name="senpaidub-scheduler", daemon=True)
            self._thread.start()
        self._sweep_uploads()

    def stop(self) -> None:
        """Stop scheduling and kill running workers; their jobs go back to waiting."""
        with self._lock:
            self._stopping = True
            handles = list(self._handles.values())
            thread, self._thread = self._thread, None
        self._wake.set()
        for h in handles:
            h.kill()
        for t in list(self._readers.values()):
            t.join(timeout=10)
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            self._db.commit()

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._db.close()

    def _scheduler(self) -> None:
        while True:
            self._wake.wait(TICK)
            self._wake.clear()
            with self._lock:
                if self._stopping:
                    return
                self._schedule()
                self._flush_dirty()

    def _schedule(self) -> None:
        running = sum(1 for j in self._jobs.values() if j["status"] == "running")
        for b in sorted(self._batches.values(), key=lambda b: b["created_at"]):
            if b["paused"]:
                continue
            waiting = sorted((j for j in self._jobs.values() if j["batch_id"] == b["id"] and j["status"] == "waiting"),
                             key=lambda j: j["position"])  # fmt: skip
            for j in waiting:
                if running >= max(1, int(b["parallel"])):
                    break
                self._launch(j, b)
                running += 1

    def _launch(self, j: dict, b: dict) -> None:
        self._reset(j)
        j["status"] = "running"
        j["started_at"] = time.time()
        spec = self._spec(j, b)
        try:
            handle = self._runner(spec)
        except Exception as e:  # noqa: BLE001
            self._fail(j, f"Couldn't start the dub process: {e}", 3)
            return
        self._handles[j["id"]] = handle
        t = threading.Thread(target=self._reader, args=(j["id"], handle, spec), name=f"senpaidub-job-{j['id'][:8]}",
                             daemon=True)  # fmt: skip
        self._readers[j["id"]] = t
        self._changed(j, status=True)
        t.start()

    def _spec(self, j: dict, b: dict) -> dict:
        return {
            "job_id": j["id"],
            "input": j["input"],
            "sub_file": j.get("sub_file"),
            "sub_track": j.get("sub_track"),
            "use_ocr": bool(j.get("use_ocr")),
            "tts": b["tts"],
            "series_name": b.get("series_name"),
            "force": bool(b.get("force")),
            "library": str(self.library) if self.library else None,
            "options": b.get("options") or {},
        }

    # ------------------------------------------------------------------ worker messages

    def _reader(self, job_id: str, handle: Handle, spec: dict) -> None:
        final = False
        while True:
            try:
                msg = handle.q.get(timeout=0.2)
            except queue.Empty:
                if handle.is_alive():
                    continue
                try:  # the process is gone; take anything still in the pipe
                    msg = handle.q.get(timeout=0.2)
                except Exception:  # noqa: BLE001
                    break
            except Exception:  # noqa: BLE001 - a killed writer can leave a broken message
                if handle.is_alive():
                    continue
                break
            self._on_message(job_id, msg)
            if msg.get("t") in ("result", "error"):
                final = True
                break
        handle.join(timeout=10)
        self._finished(job_id, handle, spec, final)

    def _on_message(self, job_id: str, msg: dict) -> None:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None or job_id in self._cancelling or j["status"] != "running":
                return
            t = msg.get("t")
            if t == "stage":
                j["group"] = msg.get("group")
                j["stage_key"] = msg.get("key")
                j["stage_label"] = msg.get("label")
                j["stage_total"] = msg.get("total")
                j["stage_unit"] = msg.get("unit")
                j["stage_done"] = 0
                self._bump(j, _group_progress(j["group"], 0, j["stage_total"], False, j["stage_key"]))
                self._changed(j)
            elif t == "progress":
                if msg.get("key") == j["stage_key"]:
                    j["stage_done"] = msg.get("completed")
                    if msg.get("total") is not None:
                        j["stage_total"] = msg.get("total")
                    self._bump(j, _group_progress(j["group"], j["stage_done"], j["stage_total"], False,
                                                 j["stage_key"]))  # fmt: skip
                    self._changed(j)
            elif t == "stage_end":
                if msg.get("key") == j["stage_key"]:
                    self._bump(j, _group_progress(j["group"], None, None, True, j["stage_key"]))
                    self._changed(j)
            elif t in ("info", "warn"):
                j["log"].append({"level": t, "msg": str(msg.get("msg", ""))})
                del j["log"][:-LOG_SIZE]
                self._changed(j)
            elif t == "result":
                res = dict(msg.get("result") or {})
                j["output"] = res.pop("output", None)
                j["result"] = res
                j["status"] = "skipped" if res.get("skipped_reason") else "done"
                j["progress"] = 1.0
                j["finished_at"] = time.time()
                self._changed(j, status=True)
            elif t == "error":
                if self._stopping:  # killed by the server's own shutdown: _finished re-queues it
                    return
                self._fail(j, str(msg.get("message") or "The dub failed."), msg.get("exit_code") or 3)

    def _bump(self, j: dict, p: float) -> None:
        j["progress"] = round(max(j["progress"] or 0.0, min(p, 0.99)), 4)

    def _fail(self, j: dict, message: str, exit_code: int) -> None:
        j["status"] = "failed"
        j["error"] = message
        j["exit_code"] = int(exit_code)
        j["finished_at"] = time.time()
        self._changed(j, status=True)

    def _finished(self, job_id: str, handle: Handle, spec: dict, final: bool) -> None:
        cancelled = False
        dead_uploads: list[str] = []
        with self._lock:
            self._handles.pop(job_id, None)
            self._readers.pop(job_id, None)
            j = self._jobs.get(job_id)
            if job_id in self._cancelling:
                self._cancelling.discard(job_id)
                cancelled = True
            elif j is not None and j["status"] == "running":
                if self._stopping:
                    self._reset(j)
                    self._save_job(j, commit=True)
                elif not final:
                    code = handle.exitcode
                    self._fail(j, f"The dub process stopped unexpectedly (exit code {code}).", 3)
            if j is not None and j["status"] in ("done", "skipped"):
                dead_uploads = self._unused_uploads(j)
            self._wake.set()
        for upload_id in dead_uploads:  # the dub is in the library now; drop the staged copy
            files.delete_upload(upload_id)
        if cancelled:
            partial = worker.partial_output(spec)
            if partial is not None:
                partial.unlink(missing_ok=True)

    # ------------------------------------------------------------------ staged uploads

    def _upload_refs(self, exclude: str | None = None) -> set[str]:
        """Upload ids that a job may still need (waiting, running, or failed/cancelled for Retry)."""
        refs: set[str] = set()
        for j in self._jobs.values():
            if j["id"] == exclude or j["status"] in ("done", "skipped"):
                continue
            for p in (j.get("input"), j.get("sub_file")):
                uid = files.upload_id_of(p) if p else None
                if uid:
                    refs.add(uid)
        return refs

    def _unused_uploads(self, j: dict) -> list[str]:
        ids = {files.upload_id_of(p) for p in (j.get("input"), j.get("sub_file")) if p}
        ids.discard(None)
        return sorted(ids - self._upload_refs(exclude=j["id"]))  # type: ignore[operator]

    def _sweep_uploads(self, max_age: float = 24 * 3600) -> None:
        """Delete day-old uploads no unfinished job needs (abandoned tabs, older finished dubs)."""
        with self._lock:
            keep = self._upload_refs()
        for upload_id in files.stale_uploads(keep, max_age):
            files.delete_upload(upload_id)

    # ------------------------------------------------------------------ change tracking / events

    def _changed(self, j: dict, status: bool = False) -> None:
        """Record a change. Status changes are saved and sent now; the rest at most every 0.25 s."""
        now = time.monotonic()
        if status or now - self._last_emit.get(j["id"], 0.0) >= EMIT_INTERVAL:
            self._dirty.discard(j["id"])
            self._last_emit[j["id"]] = now
            self._save_job(j, commit=status)
            self._publish({"type": "job", "job": self._job_view(j)})
            b = self._batches.get(j["batch_id"])
            if b is not None:
                self._publish({"type": "batch", "batch": self._batch_view(b, with_jobs=False)})
        else:
            self._dirty.add(j["id"])
        if status:
            self._wake.set()

    def _flush_dirty(self) -> None:
        now = time.monotonic()
        for job_id in list(self._dirty):
            j = self._jobs.get(job_id)
            if j is None:
                self._dirty.discard(job_id)
            elif now - self._last_emit.get(job_id, 0.0) >= EMIT_INTERVAL:
                self._changed(j)
        self._db.commit()

    def _publish(self, event: dict) -> None:
        with self._subs_lock:
            subs = list(self._subs.items())
        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(_offer, q, event)
            except RuntimeError:  # loop closed
                self.unsubscribe(q)

    def subscribe(self) -> asyncio.Queue[dict]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        with self._subs_lock:
            self._subs[q] = loop
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._subs_lock:
            self._subs.pop(q, None)

    # ------------------------------------------------------------------ views

    @staticmethod
    def _reset(j: dict) -> None:
        j.update(status="waiting", group=None, stage_key=None, stage_label=None, stage_done=None,
                 stage_total=None, stage_unit=None, progress=0.0, started_at=None, finished_at=None,
                 output=None, result=None, error=None, exit_code=None)  # fmt: skip
        j["log"] = []

    def _job_view(self, j: dict) -> dict:
        out = {k: j.get(k) for k in JOB_KEYS}
        out["log"] = [dict(e) for e in j.get("log", [])]
        if out["result"] is not None:
            out["result"] = dict(out["result"])
        started, finished = j.get("started_at"), j.get("finished_at")
        elapsed = ((finished or time.time()) - started) if started else None
        out["elapsed_s"] = round(elapsed, 1) if elapsed is not None else None
        p = j.get("progress") or 0.0
        out["eta_s"] = (round(elapsed * (1 - p) / p, 1)
                        if j["status"] == "running" and elapsed is not None and p > 0.05 else None)  # fmt: skip
        return out

    def _batch_jobs(self, batch_id: str) -> list[dict]:
        return sorted((j for j in self._jobs.values() if j["batch_id"] == batch_id), key=lambda j: j["position"])

    def _batch_view(self, b: dict, with_jobs: bool = True) -> dict:
        jobs = self._batch_jobs(b["id"])
        counts = dict.fromkeys(STATUSES, 0)
        for j in jobs:
            counts[j["status"]] += 1
        if not counts["waiting"] and not counts["running"]:
            status = "done"
        elif b["paused"]:
            status = "paused"
        elif counts["running"]:
            status = "running"
        else:
            status = "queued"
        progress = (sum(1.0 if j["status"] in TERMINAL else (j["progress"] or 0.0) for j in jobs) / len(jobs)
                    if jobs else 1.0)  # fmt: skip
        out = {
            "id": b["id"], "created_at": b["created_at"], "status": status, "tts": b["tts"],
            "series_name": b.get("series_name"), "parallel": b["parallel"], "force": b.get("force", False),
            "output_dir": b.get("output_dir"), "counts": counts, "progress": round(progress, 4),
        }  # fmt: skip
        if with_jobs:
            out["jobs"] = [self._job_view(j) for j in jobs]
        return out

    # ------------------------------------------------------------------ public API

    def _library(self) -> Path:
        if self.library is not None:
            return self.library
        try:
            lib = resolve({}, load_config())["library"]
        except Exception:  # noqa: BLE001 - a broken config is reported when the job runs
            lib = DEFAULTS["library"]
        return Path(lib).expanduser()

    def create_batch(
        self,
        items: list[dict],
        tts: str,
        series_name: str | None = None,
        parallel: int = 1,
        force: bool = False,
        options: dict | None = None,
    ) -> dict:
        """items: [{"input", "sub_file", "sub_track", "use_ocr", "name"}], in episode order.

        `options` (extra to the contract) overrides config.toml keys for these jobs, e.g.
        {"separate": False}; see worker.CONFIG_KEYS.
        """
        if not items:
            raise ValueError("Add at least one video.")
        if tts not in ENGINES:
            raise ValueError(f"Unknown voice engine {tts!r}; pick qwen, edge or google.")
        if any(not item.get("input") for item in items):
            raise ValueError("Every item needs an input path.")
        series_name = (series_name or "").strip() or None
        library = self._library()
        uploads = [worker._is_upload(Path(item["input"])) for item in items]
        if series_name:
            output_dir: str | None = str(library / sanitize_filename(series_name))
        elif any(uploads):
            output_dir = str(library / worker.UNSORTED)
        else:
            output_dir = None  # dubs go next to their videos
        now = time.time()
        b = {
            "id": uuid.uuid4().hex, "created_at": now, "tts": tts, "series_name": series_name,
            "parallel": max(1, int(parallel or 1)), "force": bool(force), "paused": False,
            "output_dir": output_dir, "options": dict(options or {}),
        }  # fmt: skip
        with self._lock:
            self._batches[b["id"]] = b
            self._save_batch(b)
            for i, item in enumerate(items):
                path = str(item["input"])
                j: dict[str, Any] = {
                    "id": uuid.uuid4().hex, "batch_id": b["id"], "position": i,
                    "name": item.get("name") or Path(path).name, "input": path, "tts": tts,
                    "sub_file": str(item["sub_file"]) if item.get("sub_file") else None,
                    "sub_track": item.get("sub_track"), "use_ocr": bool(item.get("use_ocr")),
                }  # fmt: skip
                self._reset(j)
                self._jobs[j["id"]] = j
                self._save_job(j)
            self._db.commit()
            view = self._batch_view(b)
            self._publish({"type": "batch", "batch": self._batch_view(b, with_jobs=False)})
        self._wake.set()
        return view

    def list_batches(self, limit: int = 50) -> list[dict]:
        with self._lock:
            bs = sorted(self._batches.values(), key=lambda b: b["created_at"], reverse=True)[:limit]
            return [self._batch_view(b, with_jobs=False) for b in bs]

    def get_batch(self, batch_id: str) -> dict:
        with self._lock:
            return self._batch_view(self._batches[batch_id])

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            return self._job_view(self._jobs[job_id])

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs[job_id]
            if j["status"] not in ("waiting", "running"):
                raise JobStateError(f"Can't cancel a job that is {j['status']}.")
            handle = self._handles.get(job_id) if j["status"] == "running" else None
            if handle is not None:
                self._cancelling.add(job_id)
            j["status"] = "cancelled"
            j["finished_at"] = time.time()
            self._changed(j, status=True)
            view = self._job_view(j)
        if handle is not None:
            handle.kill()
        return view

    def retry_job(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs[job_id]
            if j["status"] not in ("failed", "cancelled") or job_id in self._cancelling:
                raise JobStateError(f"Can't retry a job that is {j['status']}.")
            self._reset(j)
            self._changed(j, status=True)
            return self._job_view(j)

    def remove_job(self, job_id: str) -> None:
        with self._lock:
            j = self._jobs[job_id]
            if j["status"] != "waiting":
                raise JobStateError("Only a waiting job can be removed.")
            dead_uploads = self._unused_uploads(j)
            del self._jobs[job_id]
            self._dirty.discard(job_id)
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._db.commit()
            b = self._batches.get(j["batch_id"])
            if b is not None:
                self._publish({"type": "batch", "batch": self._batch_view(b, with_jobs=False)})
        for upload_id in dead_uploads:
            files.delete_upload(upload_id)

    def _set_paused(self, batch_id: str, paused: bool) -> dict:
        with self._lock:
            b = self._batches[batch_id]
            b["paused"] = paused
            self._save_batch(b, commit=True)
            self._publish({"type": "batch", "batch": self._batch_view(b, with_jobs=False)})
            view = self._batch_view(b)
        self._wake.set()
        return view

    def pause_batch(self, batch_id: str) -> dict:
        return self._set_paused(batch_id, True)

    def resume_batch(self, batch_id: str) -> dict:
        return self._set_paused(batch_id, False)

    def history(self, limit: int = 12) -> list[dict]:
        with self._lock:
            done = [j for j in self._jobs.values() if j["status"] == "done"]
            done.sort(key=lambda j: j.get("finished_at") or 0, reverse=True)
            return [self._job_view(j) for j in done[:limit]]

    def known_paths(self) -> set[str]:
        """Inputs and outputs of every known job (for the API's open/play check)."""
        with self._lock:
            return {p for j in self._jobs.values() for p in (j.get("input"), j.get("output")) if p}
