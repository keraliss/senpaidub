"""TTS backend protocol, on-disk clip cache, and concurrent synthesis with retries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..errors import SenpaiDubError, ExternalToolError

log = logging.getLogger("senpaidub")


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    default_voice: str

    async def synthesize(self, text: str, voice: str, reference: Path | None = None) -> bytes:
        """Return encoded audio bytes (any format ffmpeg can decode).

        `reference` is the original line's audio, used by voice-cloning backends.
        """
        ...


class TTSCache:
    """Clips stored as <dir>/<sha256[:2]>/<sha256>.bin, keyed on backend + voice + text."""

    def __init__(self, directory: Path):
        self.dir = directory

    @staticmethod
    def key(backend: str, voice: str, text: str, reference: Path | None = None) -> str:
        h = hashlib.sha256(f"{backend}\0{voice}\0{text}".encode())
        if reference is not None:
            h.update(b"\0ref\0" + hashlib.sha256(reference.read_bytes()).digest())
        return h.hexdigest()

    def path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.bin"

    def get(self, key: str) -> bytes | None:
        p = self.path(key)
        return p.read_bytes() if p.is_file() else None

    def put(self, key: str, data: bytes) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return p


class SynthStats:
    def __init__(self) -> None:
        self.cached = 0
        self.synthesized = 0


async def synthesize_all(
    backend: TTSBackend,
    requests: Sequence[tuple[str, str, Path | None]],  # (text, voice, reference)
    cache: TTSCache,
    *,
    concurrency: int = 4,
    retries: int = 3,
    on_done: Callable[[], None] | None = None,
    slots: Sequence[float | None] | None = None,
) -> tuple[list[Path], SynthStats]:
    """Synthesize every (text, voice) pair, returning the cached clip path for each.

    `slots` (seconds of room per line) is passed on as slot_s to backends with `takes_slot`.
    Backends that batch (`batch_size` > 1) get the lines shortest-first, so each batch holds
    lines of similar length and doesn't wait on one long line.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    stats = SynthStats()
    inflight: dict[str, asyncio.Task[Path]] = {}

    takes_slot = getattr(backend, "takes_slot", False)

    async def call(text: str, voice: str, ref: Path | None, slot: float | None) -> bytes:
        if takes_slot:
            return await backend.synthesize(text, voice, ref, slot_s=slot)  # type: ignore[call-arg]
        if ref is not None:
            return await backend.synthesize(text, voice, ref)
        return await backend.synthesize(text, voice)

    async def one(text: str, voice: str, ref: Path | None, slot: float | None) -> Path:
        key = cache.key(backend.name, voice, text, ref)
        if cache.get(key) is not None:
            stats.cached += 1
            return cache.path(key)
        async with sem:
            delay = 1.0
            for attempt in range(1, retries + 1):
                try:
                    data = await call(text, voice, ref, slot)
                    if not data:
                        raise RuntimeError("backend returned no audio")
                    stats.synthesized += 1
                    return cache.put(key, data)
                except SenpaiDubError:
                    raise  # setup problems / a dead worker: retrying won't help
                except Exception as e:  # noqa: BLE001 — every backend error is retryable here
                    if attempt == retries:
                        raise ExternalToolError(
                            f"TTS ({backend.name}, voice {voice}) failed after {retries} attempts "
                            f"for line: {text!r}\nLast error: {type(e).__name__}: {e}"
                        ) from e
                    log.debug("TTS attempt %d failed (%s); retrying in %.0fs", attempt, e, delay)
                    await asyncio.sleep(delay)
                    delay *= 2
        raise AssertionError("unreachable")

    async def tracked(text: str, voice: str, ref: Path | None, slot: float | None) -> Path:
        # Identical requests share one synthesis.
        k = f"{voice}\0{text}\0{ref}"
        if k not in inflight:
            inflight[k] = asyncio.ensure_future(one(text, voice, ref, slot))
        try:
            return await inflight[k]
        finally:
            if on_done:
                on_done()

    order = list(range(len(requests)))
    if getattr(backend, "batch_size", 1) > 1:
        order.sort(key=lambda i: len(requests[i][0]))
    started = {
        i: asyncio.ensure_future(tracked(*requests[i], slots[i] if slots is not None else None))
        for i in order
    }
    tasks = [started[i] for i in range(len(requests))]
    try:
        paths = await asyncio.gather(*tasks)
    except BaseException:
        for t in [*tasks, *inflight.values()]:
            t.cancel()
        raise
    return list(paths), stats


def make_backend(name: str, qwen_mode: str | None = None) -> TTSBackend:
    name = name.lower()
    if name == "edge":
        from .edge import EdgeTTS

        return EdgeTTS()
    if name == "google":
        from .google import GoogleTTS

        return GoogleTTS()
    if name == "qwen":
        from .qwen import QwenTTS

        return QwenTTS(mode=qwen_mode)
    if name == "mock":
        from .mock import MockTTS

        return MockTTS()
    from ..errors import UsageError

    raise UsageError(f"Unknown TTS backend {name!r}. Choose qwen, edge, google or mock.")
