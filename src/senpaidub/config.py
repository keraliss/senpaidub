"""Config file (user config dir) and cache locations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

from .errors import UsageError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

APP = "senpaidub"

# key -> expected type(s). Keys mirror the long CLI option names (dashes → underscores).
KNOWN_KEYS: dict[str, type | tuple[type, ...]] = {
    "tts": str,
    "voice": str,
    "voice_map": str,
    "separate": bool,
    "duck_db": (int, float),
    "max_speedup": (int, float),
    "include_styles": str,
    "exclude_styles": str,
    "set_default": bool,
    "codec": str,
    "device": str,
    "concurrency": int,
    "speech_gain": (int, float),
    "library": str,
    "keep_cache": bool,
    "qwen_mode": str,
}

DEFAULTS: dict[str, Any] = {
    "tts": "qwen",
    "voice": None,  # backend's own default
    "voice_map": None,
    "separate": True,
    "duck_db": -12.0,
    "max_speedup": 1.4,
    "include_styles": None,
    "exclude_styles": None,
    "set_default": False,
    "codec": "aac",
    "device": "auto",
    "concurrency": 4,
    "speech_gain": 0.0,
    "library": "~/Downloads/anime",  # where series folders go
    "keep_cache": False,  # True keeps per-episode caches (faster re-runs, uses disk)
    "qwen_mode": "gentle",  # gentle keeps the computer usable; fast is full speed
}


def config_path() -> Path:
    env = os.environ.get("SENPAIDUB_CONFIG")
    return Path(env) if env else Path(user_config_dir(APP)) / "config.toml"


def cache_dir() -> Path:
    env = os.environ.get("SENPAIDUB_CACHE")
    return Path(env) if env else Path(user_cache_dir(APP))


def data_dir() -> Path:
    """Persistent app data (web UI database, local qwen voice environment)."""
    env = os.environ.get("SENPAIDUB_DATA")
    return Path(env) if env else Path(user_data_dir(APP))


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or config_path()
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise UsageError(f"Config file {path} is not valid TOML: {e}") from e
    unknown = sorted(set(data) - set(KNOWN_KEYS))
    if unknown:
        raise UsageError(f"Unknown key(s) in {path}: {', '.join(unknown)}")
    for k, v in data.items():
        if isinstance(v, bool) and KNOWN_KEYS[k] is not bool:
            raise UsageError(f"Config key '{k}' in {path} has the wrong type")
        if not isinstance(v, KNOWN_KEYS[k]):
            raise UsageError(f"Config key '{k}' in {path} has the wrong type")
    return data


def resolve(cli: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """CLI value (when given, i.e. not None) > config file > built-in default."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in config.items() if v is not None and v != ""})
    out.update({k: v for k, v in cli.items() if v is not None})
    return out


def validate_config(cfg: dict[str, Any]) -> None:
    """Check resolved option values before any work starts (shared by the CLI and the web worker)."""
    from .mix import CODECS  # lazy: mix pulls in numpy/soundfile

    if cfg["tts"] == "turbo":
        raise UsageError("The 'turbo' voices were replaced by 'qwen' (better voices, faster on a GPU). "
                         "Use --tts qwen, or set tts = \"qwen\" in config.toml.")
    if cfg["tts"] not in ("qwen", "edge", "google", "mock"):
        raise UsageError(f"--tts must be qwen, edge or google (got {cfg['tts']!r}).")
    if cfg["qwen_mode"] not in ("gentle", "fast"):
        raise UsageError(f"qwen_mode must be gentle or fast (got {cfg['qwen_mode']!r}).")
    if cfg["codec"] not in CODECS:
        raise UsageError(f"--codec must be one of {', '.join(CODECS)} (got {cfg['codec']!r}).")
    if cfg["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise UsageError(f"--device must be auto, cpu, cuda or mps (got {cfg['device']!r}).")
    if float(cfg["max_speedup"]) < 1.0:
        raise UsageError("--max-speedup must be at least 1.0.")
    if int(cfg["concurrency"]) < 1:
        raise UsageError("--concurrency must be at least 1.")
