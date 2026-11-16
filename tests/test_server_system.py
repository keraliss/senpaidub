from __future__ import annotations

import json
from pathlib import Path

from senpaidub.config import data_dir
from senpaidub.server import system


def info(**kw):
    base = {
        "os": "Linux", "os_version": "6", "arch": "x86_64", "cpu": "x", "cores": 8,
        "apple_silicon": False, "ram_gb": 16.0,
        "gpu": {"kind": "none", "name": None, "vram_gb": None},
        "disk_free_gb": {"library": 100.0, "cache": 100.0},
        "qwen_installed": True, "internet": True, "ocr_available": False,
    }
    base.update(kw)
    return base


def cuda(vram):
    return {"kind": "cuda", "name": "RTX", "vram_gb": vram}


def test_cuda_big():
    r = system.recommend(info(gpu=cuda(24.0)))
    assert (r["engine"], r["level"]) == ("qwen", "great")
    assert r["parallel"] >= 2 and r["minutes_per_episode"]["qwen"] == 10.0
    assert r["warning"] is None


def test_cuda_8gb():
    r = system.recommend(info(gpu=cuda(8.0)))
    assert r["engine"] == "qwen" and r["parallel"] == 2


def test_cuda_small_falls_to_cpu_rules():
    r = system.recommend(info(gpu=cuda(4.0), cores=8, ram_gb=16))
    assert r["engine"] == "edge" and r["level"] == "ok"


def test_apple_16gb():
    r = system.recommend(info(apple_silicon=True, os="macOS", ram_gb=16, cores=10,
                              gpu={"kind": "mps", "name": "Apple GPU", "vram_gb": None}))
    assert (r["engine"], r["level"], r["parallel"]) == ("qwen", "great", 1)
    assert r["minutes_per_episode"]["qwen"] == system.QWEN_APPLE_MIN


def test_apple_8gb():
    r = system.recommend(info(apple_silicon=True, ram_gb=8, cores=8))
    assert (r["engine"], r["level"], r["parallel"]) == ("edge", "ok", 1)
    assert r["minutes_per_episode"]["qwen"] > system.QWEN_APPLE_MIN
    assert any("memory-tight" in n for n in r["notes"])


def test_cpu_strong():
    r = system.recommend(info(cores=16, ram_gb=32))
    assert (r["engine"], r["level"]) == ("edge", "ok")
    assert r["parallel"] == 2
    assert r["minutes_per_episode"]["qwen"] == round(55 * 10 / 16)


def test_cpu_scaling_slower_with_fewer_cores():
    a = system.recommend(info(cores=8, ram_gb=16))["minutes_per_episode"]["qwen"]
    b = system.recommend(info(cores=16, ram_gb=16))["minutes_per_episode"]["qwen"]
    assert a > b


def test_weak_low_ram():
    r = system.recommend(info(ram_gb=4))
    assert r["level"] == "weak" and r["engine"] == "edge"
    assert r["minutes_per_episode"]["qwen"] is None
    assert "overheat" in r["warning"]


def test_weak_few_cores():
    r = system.recommend(info(cores=4, ram_gb=16))
    assert r["level"] == "weak" and r["parallel"] == 2  # edge is network-bound; 16 GB


def test_mid_cpu_is_weak():
    r = system.recommend(info(cores=6, ram_gb=8))
    assert r["level"] == "weak"


def test_offline_weak_forces_qwen():
    r = system.recommend(info(ram_gb=4, internet=False, qwen_installed=False))
    assert r["engine"] == "qwen" and r["minutes_per_episode"]["qwen"] is not None
    assert "offline" in r["warning"].lower()
    assert any("internet" in n for n in r["notes"])


def test_offline_apple():
    r = system.recommend(info(apple_silicon=True, ram_gb=8, internet=False))
    assert r["engine"] == "qwen"


def test_offline_strong_cpu():
    r = system.recommend(info(cores=16, ram_gb=32, internet=False))
    assert r["engine"] == "qwen"


def test_qwen_not_installed_note_and_low_disk():
    r = system.recommend(info(apple_silicon=True, ram_gb=16, qwen_installed=False,
                              disk_free_gb={"library": 3.0, "cache": 50.0}))
    assert any("install" in n.lower() for n in r["notes"])
    assert any("3 GB" in n for n in r["notes"])


def test_recommend_shape_and_pure():
    s = info()
    before = json.dumps(s, sort_keys=True)
    r = system.recommend(s)
    assert json.dumps(s, sort_keys=True) == before
    assert set(r) == {"engine", "parallel", "level", "headline", "warning", "minutes_per_episode", "notes"}
    assert set(r["minutes_per_episode"]) == {"qwen", "edge", "google"}


def test_collect_real(monkeypatch, tmp_path):
    monkeypatch.setattr(system, "_internet", lambda: False)
    s = system.collect(tmp_path)
    assert s["os"] in ("macOS", "Linux", "Windows")
    assert s["cores"] >= 1 and s["ram_gb"] > 0
    assert s["gpu"]["kind"] in ("cuda", "mps", "none")
    assert set(s["disk_free_gb"]) == {"library", "cache"}
    assert s["internet"] is False
    assert isinstance(s["qwen_installed"], bool)
    json.dumps(s)


def test_get_caches(monkeypatch, tmp_path):
    calls = []

    def fake(lib):
        calls.append(lib)
        return info(cores=len(calls) + 8)

    monkeypatch.setattr(system, "collect", fake)
    monkeypatch.setattr(system, "_qwen_installed", lambda: True)
    a = system.get(tmp_path)
    assert (data_dir() / "system.json").is_file()
    b = system.get(tmp_path)
    assert a == b and len(calls) == 1
    c = system.get(tmp_path, refresh=True)
    assert len(calls) == 2 and c["system"]["cores"] == 10
    assert system.get(tmp_path)["system"]["cores"] == 10


def test_get_overlays_live_qwen_and_internet(monkeypatch, tmp_path):
    monkeypatch.setattr(system, "collect", lambda lib: info(qwen_installed=False, internet=False))
    monkeypatch.setattr(system, "_qwen_installed", lambda: False)
    monkeypatch.setattr(system, "_internet", lambda: False)
    first = system.get(tmp_path)
    assert not first["system"]["qwen_installed"] and not first["system"]["internet"]
    # Qwen installed from the CLI and back online: the cached check must not hide it.
    monkeypatch.setattr(system, "_qwen_installed", lambda: True)
    monkeypatch.setattr(system, "_internet", lambda: True)
    later = system.get(tmp_path)
    assert later["system"]["qwen_installed"] and later["system"]["internet"]
    assert later["recommendation"] == system.recommend(later["system"])


def test_get_ignores_corrupt_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(system, "collect", lambda lib: info())
    f = data_dir() / "system.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{nope")
    assert "recommendation" in system.get(Path(tmp_path))
