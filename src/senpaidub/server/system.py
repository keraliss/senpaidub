"""Machine check: what this computer can do, and which voice engine suits it."""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from senpaidub import hardsubs
from senpaidub.config import cache_dir, data_dir

QWEN_APPLE_MIN = 12.0  # qwen on an Apple Silicon GPU, per 24-minute episode (10 measured on an M5)
QWEN_CUDA_MIN = 10.0  # qwen on an NVIDIA GPU
QWEN_CPU_MIN = 55.0  # qwen on a CPU-only machine (about 10 cores); very slow
BASE_CORES = 10
EDGE_MIN = 4.0
GOOGLE_MIN = 4.0
INTERNET_HOST = ("speech.platform.bing.com", 443)


def _os_name() -> str:
    return {"Darwin": "macOS", "Windows": "Windows"}.get(platform.system(), "Linux")


def _cpu_name(apple_silicon: bool) -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _gpu() -> dict[str, Any]:
    none = {"kind": "none", "name": None, "vram_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "kind": "cuda",
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(props.total_memory / 1024**3, 1),
            }
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return {"kind": "mps", "name": "Apple GPU", "vram_gb": None}
    except Exception:
        pass
    return none


def _free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1024**3, 1)
    except OSError:
        return 0.0


def _internet() -> bool:
    try:
        with socket.create_connection(INTERNET_HOST, timeout=2):
            return True
    except Exception:
        return False


def _qwen_installed() -> bool:
    try:
        from senpaidub import setup_tts

        return bool(setup_tts.is_installed())
    except Exception:
        return False


def collect(library: Path) -> dict:
    import psutil

    machine = platform.machine().lower()
    apple = sys.platform == "darwin" and machine in ("arm64", "aarch64")
    return {
        "os": _os_name(),
        "os_version": platform.mac_ver()[0] if sys.platform == "darwin" else platform.release(),
        "arch": platform.machine(),
        "cpu": _cpu_name(apple),
        "cores": os.cpu_count() or 1,
        "apple_silicon": apple,
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": _gpu(),
        "disk_free_gb": {"library": _free_gb(library), "cache": _free_gb(cache_dir())},
        "qwen_installed": _qwen_installed(),
        "internet": _internet(),
        "ocr_available": hardsubs.available(),
    }


def _cpu_qwen_minutes(cores: int) -> float:
    return round(QWEN_CPU_MIN * BASE_CORES / max(cores, 1), 0)


def recommend(sysinfo: dict) -> dict:
    ram = float(sysinfo.get("ram_gb") or 0)
    cores = int(sysinfo.get("cores") or 1)
    apple = bool(sysinfo.get("apple_silicon"))
    gpu = sysinfo.get("gpu") or {}
    internet = bool(sysinfo.get("internet", True))
    free = (sysinfo.get("disk_free_gb") or {}).get("library")
    notes: list[str] = []
    warning: str | None = None
    engine = "edge"
    parallel = 1
    qwen_min: float | None = None

    cuda = gpu.get("kind") == "cuda" and (gpu.get("vram_gb") or 0) >= 8
    net_parallel = 2 if ram >= 16 else 1

    if cuda:
        level, engine = "great", "qwen"
        vram = float(gpu["vram_gb"])
        parallel = 2 if vram < 16 else min(4, int(vram // 8))
        qwen_min = QWEN_CUDA_MIN
        headline = f"Fast GPU ({gpu.get('name') or 'NVIDIA'}): qwen voices run on the GPU."
        notes.append(f"About {int(QWEN_CUDA_MIN)} minutes per episode with qwen.")
    elif ram < 8 or (not apple and cores < 6):
        level, parallel = "weak", 1
        headline = "This computer is on the light side. Edge voices are the best fit."
        warning = (
            "Local voices would be very slow and could overheat this laptop. "
            "Use edge or google instead."
        )
        if not internet:
            engine = "qwen"
            qwen_min = _cpu_qwen_minutes(cores) * 1.5
            warning = (
                "You're offline, so only local voices work, and they will be very slow "
                "and could overheat this laptop. Connect to the internet to use edge or google."
            )
            headline = "Offline on a light machine: local voices only, and slowly."
        else:
            parallel = net_parallel if ram >= 16 else 1
    elif apple and ram >= 16:
        level, engine, parallel = "great", "qwen", 1
        qwen_min = QWEN_APPLE_MIN
        headline = "Apple Silicon with plenty of memory: qwen voices run on its GPU."
        notes.append(f"About {int(QWEN_APPLE_MIN)} minutes per episode with qwen.")
    elif apple:
        level = "ok"
        qwen_min = round(QWEN_APPLE_MIN * 1.3, 0)
        notes.append("Qwen works but is slow and memory-tight on 8 GB.")
        if internet:
            engine = "edge"
            parallel = 1
            headline = "Edge voices are the smooth choice with 8 GB of memory."
        else:
            engine = "qwen"
            headline = "Offline: local qwen voices only, and they are slow on 8 GB."
            warning = "Close other apps while dubbing; memory is tight."
    elif ram >= 16 and cores >= 8:
        level, parallel = "ok", 1
        qwen_min = _cpu_qwen_minutes(cores)
        notes.append(f"Qwen without a GPU takes roughly {int(qwen_min)} minutes per episode.")
        if internet:
            engine = "edge"
            parallel = net_parallel
            headline = "No fast GPU. Edge is much quicker; qwen sounds better but needs about an hour per episode here."
        else:
            engine = "qwen"
            headline = "Offline: local qwen voices only."
    else:
        level, parallel = "weak", 1
        qwen_min = None
        headline = "This computer is on the light side. Edge voices are the best fit."
        warning = (
            "Local voices would be very slow and could overheat this laptop. "
            "Use edge or google instead."
        )
        if not internet:
            engine = "qwen"
            qwen_min = _cpu_qwen_minutes(cores) * 1.5
            headline = "Offline on a light machine: local voices only, and slowly."
            warning = (
                "You're offline, so only local voices work, and they will be very slow "
                "and could overheat this laptop. Connect to the internet to use edge or google."
            )

    if not internet:
        notes.append("Edge and google voices need internet.")
        if not sysinfo.get("qwen_installed"):
            notes.append("Install qwen voices while online to dub offline.")
    elif engine == "qwen" and not sysinfo.get("qwen_installed"):
        notes.append("Qwen voices need a one-time install (a few GB).")
    if free is not None and free < 10:
        notes.append(f"Only {free:g} GB free in your library folder; an episode needs about as much as its video.")

    return {
        "engine": engine,
        "parallel": max(1, int(parallel)),
        "level": level,
        "headline": headline,
        "warning": warning,
        "minutes_per_episode": {"qwen": qwen_min, "edge": EDGE_MIN, "google": GOOGLE_MIN},
        "notes": notes,
    }


def _cache_file() -> Path:
    return data_dir() / "system.json"


def _refresh_live(cached: dict) -> dict:
    """Overlay the values that change after the first check (qwen installed by the CLI or the
    Settings page; back online) and recompute the recommendation from them."""
    sysinfo = cached["system"]
    live = dict(sysinfo, qwen_installed=_qwen_installed())
    if not live.get("internet"):
        live["internet"] = _internet()
    if live != sysinfo:
        cached = {"system": live, "recommendation": recommend(live)}
    return cached


def get(library: Path, refresh: bool = False) -> dict:
    f = _cache_file()
    if not refresh and f.is_file():
        try:
            cached = json.loads(f.read_text())
            if "system" in cached and "recommendation" in cached:
                return _refresh_live(cached)
        except (OSError, ValueError):
            pass
    system = collect(library)
    result = {"system": system, "recommendation": recommend(system)}
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(result, indent=2))
    except OSError:
        pass
    return result
