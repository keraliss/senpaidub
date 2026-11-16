"""Voice-cloning TTS (Qwen3-TTS): each line is spoken in the voice of the original Japanese
line, taken from the Demucs vocals stem.

The model runs in a separate Python environment via qwen_worker.py, on the GPU when there is one.
Lines are voiced in batches: the pipeline keeps many lines in flight, and a dispatcher thread
hands them to the worker `batch_size` at a time. One line at a time left most of the GPU idle;
batches of 8 keep the worker around 6-7 GB on a 16 GB Mac; 16 was ~25% faster but pushed it past 8 GB and slowed the whole machine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ExternalToolError, UsageError
from ..setup_tts import find_qwen_python, is_installed, qwen_model

log = logging.getLogger("senpaidub")

SETUP_HINT = """\
The 'qwen' voice-cloning backend needs its own Python environment (Qwen3-TTS needs a different
PyTorch than Demucs). Set it up with one of:
    senpaidub setup qwen
    the "Install" button in the web UI (senpaidub serve)
Or point SENPAIDUB_QWEN_PYTHON at a Python that has qwen-tts installed."""

WORKER_SCRIPT = Path(__file__).with_name("qwen_worker.py")
MIN_REF_S = 5.4  # references shorter than this are padded with the same speaker's other lines
GATHER_S = 0.05  # how long the dispatcher waits for more lines before sending a short batch


def cap_seconds(text: str, slot_s: float | None) -> float:
    """Longest clip worth generating for a line.

    The model occasionally fails to stop, so every line gets a cap. From the text alone a line
    never needs more than ~3 s + 1 s per 5 characters. When the subtitle slot is known the cap
    is tighter: anything past twice the slot can't be fitted (speed-up tops out well below 2x),
    but a natural reading of the text is always allowed.
    """
    by_text = 3.0 + len(text) / 5
    if slot_s is None or slot_s <= 0:
        return by_text
    natural = 0.5 + len(text) / 14
    return round(min(by_text, max(2.0 * slot_s, 1.3 * natural)), 2)


def find_python() -> Path:
    python = find_qwen_python()
    if python is None or not is_installed():  # missing, or a broken env that can't import qwen_tts
        raise UsageError(SETUP_HINT)
    return python


class _Worker:
    """One model process speaking the JSON-lines protocol of qwen_worker.py."""

    def __init__(self, python: Path) -> None:
        self.ready_info: dict = {}
        self.stderr: list[str] = []
        env = {**os.environ, "SENPAIDUB_QWEN_MODEL": qwen_model()}
        self.proc = subprocess.Popen(
            # -I: don't put this folder on sys.path (our google.py would shadow the real package)
            [str(python), "-I", "-u", str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )  # fmt: skip
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())
            del self.stderr[:-80]
            log.debug("[qwen] %s", line.rstrip())

    def tail(self) -> str:
        return "\n".join(self.stderr[-20:])

    def read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise ExternalToolError(f"Qwen worker exited unexpectedly:\n{self.tail()}")
        return json.loads(line)

    def wait_ready(self) -> _Worker:
        hello = self.read()
        if not hello.get("ready"):
            raise ExternalToolError(f"Qwen worker failed to start:\n{self.tail()}")
        self.ready_info = hello
        log.info("Qwen voices: %s on %s", hello.get("model"), hello.get("device"))
        return self

    def alive(self) -> bool:
        return self.proc.poll() is None

    def request(self, obj: dict) -> dict:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise ExternalToolError(f"Qwen worker stopped:\n{self.tail()}") from e
        return self.read()

    def close(self) -> None:
        if self.alive():
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()


@dataclass
class _Item:
    text: str
    ref: Path
    max_s: float
    future: concurrent.futures.Future[bytes] = field(default_factory=concurrent.futures.Future)
    retried: bool = False


_STOP = object()


class QwenTTS:
    name = "qwen"
    default_voice = "clone"
    needs_reference = True
    takes_slot = True  # synthesize() accepts slot_s, the room the line has on the timeline
    min_ref_s = MIN_REF_S

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = max(1, batch_size or int(os.environ.get("SENPAIDUB_QWEN_BATCH", "8")))
        # Keep a second batch queued so the GPU never waits for the next one to be gathered.
        self.concurrency = self.batch_size * 2
        self._python = find_python()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._starting: concurrent.futures.Future[_Worker] | None = None
        self._worker: _Worker | None = None
        self._queue: queue.Queue[object] = queue.Queue()
        self._dispatcher: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.batches: list[int] = []  # sizes of the batches sent, for logs and tests

    # ---- worker management

    def prewarm(self) -> None:
        """Start loading the model in the background (it takes ~15-30 s)."""
        with self._lock:
            if self._starting is None and self._worker is None:
                self._starting = self._pool.submit(self._launch)

    def _launch(self) -> _Worker:
        log.debug("Starting Qwen worker")
        return _Worker(self._python).wait_ready()

    def _current(self) -> _Worker:
        with self._lock:
            starting = self._starting
        if starting is not None:
            self._starting = None
            self._worker = starting.result()
        if self._worker is None or not self._worker.alive():
            if self._worker is not None:
                log.warning("Qwen worker stopped; restarting it.\n%s", self._worker.tail())
            self._worker = self._launch()
        return self._worker

    def close(self) -> None:
        if self._dispatcher is not None:
            self._queue.put(_STOP)
            self._dispatcher.join(timeout=30)
            self._dispatcher = None
        if self._starting is not None:
            try:
                self._worker = self._starting.result()
            except Exception:  # noqa: BLE001 — it failed to start; nothing to close
                pass
            self._starting = None
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._pool.shutdown(wait=False)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ---- batching

    def _ensure_dispatcher(self) -> None:
        with self._lock:
            if self._dispatcher is None:
                self._dispatcher = threading.Thread(target=self._loop, name="qwen-batches", daemon=True)
                self._dispatcher.start()

    def _loop(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            batch = [first]
            deadline = time.monotonic() + GATHER_S
            while len(batch) < self.batch_size:
                try:
                    item = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is _STOP:
                    self._queue.put(_STOP)  # finish this batch, then stop
                    break
                batch.append(item)
            self._run(batch)  # type: ignore[arg-type]

    def _run(self, batch: list[_Item]) -> None:
        outs: list[str] = []
        try:
            worker = self._current()
            self._next_id += 1
            for _ in batch:
                fd, out = tempfile.mkstemp(suffix=".wav", prefix="qwen-")
                os.close(fd)
                outs.append(out)
            items = [{"text": it.text, "ref": str(it.ref), "out": out, "max_s": it.max_s}
                     for it, out in zip(batch, outs)]  # fmt: skip
            self.batches.append(len(batch))
            resp = worker.request({"id": self._next_id, "items": items})
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "unknown error"))
            capped = resp.get("capped") or [False] * len(batch)
            for it, out, hit in zip(batch, outs, capped):
                if hit and not it.retried:
                    # Usually a line that didn't stop talking; sampling is random, so one retry
                    # almost always comes back normal. Keep it if it doesn't.
                    log.debug("qwen: retrying a line that hit its %.1fs cap: %s", it.max_s, it.text)
                    it.retried = True
                    self._queue.put(it)
                    continue
                it.future.set_result(Path(out).read_bytes())
        except Exception as e:  # noqa: BLE001 — hand the failure to every line in the batch
            for it in batch:
                if not it.future.done():
                    it.future.set_exception(e)
        finally:
            for out in outs:
                Path(out).unlink(missing_ok=True)

    # ---- synthesis

    async def synthesize(
        self, text: str, voice: str, reference: Path | None = None, slot_s: float | None = None
    ) -> bytes:
        if reference is None:
            raise UsageError("The qwen backend needs a reference voice clip for every line.")
        self._ensure_dispatcher()
        item = _Item(text, reference, cap_seconds(text, slot_s))
        self._queue.put(item)
        return await asyncio.wrap_future(item.future)
