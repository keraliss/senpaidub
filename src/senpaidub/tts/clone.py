"""Voice-cloning TTS (Chatterbox Turbo): each line is spoken in the voice and emotion of the
original Japanese line, taken from the Demucs vocals stem.

The model runs in a separate Python environment via clone_worker.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from ..errors import ExternalToolError, UsageError

log = logging.getLogger("senpaidub")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETUP_HINT = """\
The 'turbo' voice-cloning backend needs its own Python environment (Chatterbox needs a
different PyTorch than Demucs). From the project folder:
    python3.13 -m venv .venv-tts
    .venv-tts/bin/pip install chatterbox-tts "setuptools<80" soundfile
Or point SENPAIDUB_CLONE_PYTHON at a Python that has chatterbox-tts installed."""

MIN_REF_S = 5.4  # Turbo refuses prompts of 5 s or less


def find_python() -> Path:
    env = os.environ.get("SENPAIDUB_CLONE_PYTHON")
    candidates = [Path(env)] if env else []
    for venv in (".venv-tts",):
        candidates += [PROJECT_ROOT / venv / "bin" / "python", PROJECT_ROOT / venv / "Scripts" / "python.exe"]
    for c in candidates:
        if c.is_file():
            return c
    raise UsageError(SETUP_HINT)


class _Worker:
    """One Chatterbox process speaking the JSON-lines protocol of clone_worker.py."""

    def __init__(self, python: Path) -> None:
        worker = Path(__file__).with_name("clone_worker.py")
        self.stderr: list[str] = []
        self.proc = subprocess.Popen(
            # -I: don't put this folder on sys.path (our google.py would shadow the real package)
            [str(python), "-I", "-u", str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )  # fmt: skip
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())
            del self.stderr[:-80]
            log.debug("[turbo] %s", line.rstrip())

    def tail(self) -> str:
        return "\n".join(self.stderr[-20:])

    def read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise ExternalToolError(f"Chatterbox worker exited unexpectedly:\n{self.tail()}")
        return json.loads(line)

    def wait_ready(self) -> _Worker:
        hello = self.read()
        if not hello.get("ready"):
            raise ExternalToolError(f"Chatterbox worker failed to start:\n{self.tail()}")
        log.debug("Chatterbox Turbo ready on %s", hello.get("device"))
        return self

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.alive():
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()


class CloneTTS:
    name = "turbo"
    default_voice = "clone"
    needs_reference = True
    concurrency = 1  # one model on one GPU
    # On a GPU (MPS) the worker's memory grew with every line, so it was restarted regularly.
    # On the CPU (the default now) memory stays flat, so restarts are just a safety net. The
    # replacement starts loading `prewarm` lines early, so a restart costs almost no time.
    restart_every = int(os.environ.get("SENPAIDUB_CLONE_RESTART_EVERY", "250"))
    prewarm = 2

    def __init__(self) -> None:
        self._python = find_python()
        self._worker: _Worker | None = None
        self._spare: concurrent.futures.Future[_Worker] | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._next_id = 0
        self._served = 0

    # ---- worker management

    def _launch(self) -> _Worker:
        log.debug("Starting Chatterbox Turbo worker")
        return _Worker(self._python).wait_ready()

    def _current(self) -> _Worker:
        if self._worker is not None and self._served >= self.restart_every:
            self._worker.close()
            self._worker = None
        if self._worker is None or not self._worker.alive():
            spare, self._spare = self._spare, None
            self._worker = spare.result() if spare is not None else self._launch()
            self._served = 0
        if self._spare is None and self._served >= self.restart_every - self.prewarm:
            self._spare = self._pool.submit(self._launch)
        return self._worker

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._spare is not None:
            try:
                self._spare.result().close()
            except Exception:  # noqa: BLE001 — it failed to start; nothing to close
                pass
            self._spare = None
        self._pool.shutdown(wait=False)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ---- synthesis

    def _sync(self, text: str, reference: Path) -> bytes:
        with self._lock:
            worker = self._current()
            self._served += 1
            self._next_id += 1
            fd, out = tempfile.mkstemp(suffix=".wav", prefix="turbo-")
            os.close(fd)
            try:
                assert worker.proc.stdin is not None
                worker.proc.stdin.write(json.dumps(
                    {"id": self._next_id, "text": text, "ref": str(reference), "out": out}
                ) + "\n")  # fmt: skip
                worker.proc.stdin.flush()
                resp = worker.read()
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error", "unknown error"))
                return Path(out).read_bytes()
            finally:
                Path(out).unlink(missing_ok=True)

    async def synthesize(self, text: str, voice: str, reference: Path | None = None) -> bytes:
        if reference is None:
            raise UsageError("The turbo backend needs a reference voice clip for every line.")
        return await asyncio.to_thread(self._sync, text, reference)
