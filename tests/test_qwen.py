"""The qwen backend's batching, caps and retries, against a fake worker (no model, no GPU)."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

import pytest
import soundfile as sf

from senpaidub.tts import qwen
from senpaidub.tts.base import TTSCache, synthesize_all

FAKE = Path(__file__).parent / "fixtures" / "fake_qwen_worker.py"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_QWEN_LOG", str(log))
    monkeypatch.setattr(qwen, "WORKER_SCRIPT", FAKE)
    monkeypatch.setattr(qwen, "find_python", lambda: Path(sys.executable))
    b = qwen.QwenTTS(batch_size=4, mode="fast")
    b.log = log  # type: ignore[attr-defined]
    yield b
    b.close()


def _requests(log: Path) -> list[list[dict]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def _duration(data: bytes) -> float:
    x, sr = sf.read(io.BytesIO(data))
    return len(x) / sr


def test_cap_seconds():
    text = "Why would you say such a thing, Amona?!"  # 39 characters
    assert qwen.cap_seconds(text, None) == pytest.approx(3 + 39 / 5)
    # A runaway can't go past twice its slot...
    assert qwen.cap_seconds(text, 3.4) == pytest.approx(6.8)
    # ...but a natural reading of the text is always allowed, even in a tiny slot.
    assert qwen.cap_seconds(text, 0.5) == pytest.approx(1.3 * (0.5 + 39 / 14), abs=0.01)
    # And the text-based cap still applies to huge slots.
    assert qwen.cap_seconds("Hi.", 30) == pytest.approx(3 + 3 / 5)


def test_lines_are_batched_and_mapped_back(backend, tmp_path):
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    texts = [f"line {'x' * i}" for i in range(10)]
    reqs = [(t, "clone", ref) for t in texts]
    paths, stats = asyncio.run(
        synthesize_all(backend, reqs, TTSCache(tmp_path / "cache"), concurrency=backend.concurrency,
                       slots=[2.0] * len(texts))  # fmt: skip
    )
    assert stats.synthesized == 10
    for t, p in zip(texts, paths):
        assert _duration(p.read_bytes()) == pytest.approx(0.2 + 0.02 * len(t), abs=0.01)
    sizes = [len(r) for r in _requests(backend.log)]
    assert sum(sizes) == 10 and max(sizes) <= 4 and len(sizes) < 10
    caps = {it["text"]: it["max_s"] for r in _requests(backend.log) for it in r}
    assert caps[texts[3]] == qwen.cap_seconds(texts[3], 2.0)  # the slot reached the worker


def test_capped_line_is_retried_once(backend, tmp_path):
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    data = asyncio.run(backend.synthesize("I am STUCK", "clone", ref, slot_s=1.0))
    sent = [it["text"] for r in _requests(backend.log) for it in r]
    assert sent.count("I am STUCK") == 2  # first try hit the cap, the retry did too: kept
    assert _duration(data) == pytest.approx(qwen.cap_seconds("I am STUCK", 1.0), abs=0.01)


def test_needs_reference(backend):
    from senpaidub.errors import UsageError

    with pytest.raises(UsageError):
        asyncio.run(backend.synthesize("hi", "clone"))


def test_prewarm_starts_the_worker(backend, tmp_path):
    backend.prewarm()
    backend.prewarm()  # idempotent
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    asyncio.run(backend.synthesize("hello", "clone", ref))
    assert backend._worker is not None and backend._worker.ready_info["device"] == "fake"


def _starts(log: Path) -> list[str]:
    return Path(str(log) + ".starts").read_text().splitlines()


def test_gentle_is_the_default_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen, "find_python", lambda: Path(sys.executable))
    monkeypatch.delenv("SENPAIDUB_QWEN_MODE", raising=False)
    monkeypatch.delenv("SENPAIDUB_QWEN_BATCH", raising=False)
    gentle = qwen.QwenTTS()
    fast = qwen.QwenTTS(mode="fast")
    assert gentle.mode == "gentle" and gentle.batch_size < fast.batch_size
    assert gentle.pause > 0 and gentle.restart_gb is not None and gentle.nice > 0
    assert fast.pause == 0 and fast.restart_gb is None and fast.nice == 0
    from senpaidub.errors import UsageError

    with pytest.raises(UsageError):
        qwen.QwenTTS(mode="turbo")


def test_worker_restarts_when_its_memory_creeps_up(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_QWEN_MEM_GB", "7.0")
    monkeypatch.setattr(qwen, "MIN_RESTART_INTERVAL_S", 0.0)
    backend.restart_gb = 5.5
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    for text in ("one", "two"):
        assert asyncio.run(backend.synthesize(text, "clone", ref))
    backend.close()  # waits for the dispatcher, which checks memory after handing back results
    assert backend.restarts == 2 and len(_starts(backend.log)) == 2  # the 2nd restart launches nothing


def test_no_restart_below_the_limit_or_too_soon(backend, tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    backend.restart_gb = 5.5
    monkeypatch.setenv("FAKE_QWEN_MEM_GB", "4.0")
    asyncio.run(backend.synthesize("low", "clone", ref))
    assert backend.restarts == 0
    backend.close()
    monkeypatch.setenv("FAKE_QWEN_MEM_GB", "7.0")  # over the limit, but the worker is brand new
    asyncio.run(backend.synthesize("young", "clone", ref))
    assert backend.restarts == 0


def test_gentle_pauses_after_each_batch(backend, tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(qwen.time, "sleep", slept.append)
    backend.pause = 0.5
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    asyncio.run(backend.synthesize("hello", "clone", ref))
    assert len(slept) == 1 and slept[0] >= 0


def test_gentle_worker_runs_at_low_cpu_priority(backend, tmp_path):
    backend.nice = 10
    ref = tmp_path / "ref.wav"
    sf.write(ref, [0.0] * 2400, 24000)
    asyncio.run(backend.synthesize("hello", "clone", ref))
    assert _starts(backend.log) == ["10"]
