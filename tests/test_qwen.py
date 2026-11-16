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
    b = qwen.QwenTTS(batch_size=4)
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
