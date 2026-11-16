"""Cross-platform install and lookup of the 'qwen' voice-cloning environment (Qwen3-TTS).

Qwen3-TTS needs a different PyTorch/transformers than Demucs, so it lives in its own virtualenv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .config import data_dir
from .errors import SenpaiDubError

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["qwen-tts", "soundfile"]
MODULE = "qwen_tts"
REPO_ENV = ".venv-qwen"
OVERRIDE_VAR = "SENPAIDUB_QWEN_PYTHON"
SUPPORTED = ((3, 13), (3, 12), (3, 11), (3, 10))
_IS_WINDOWS = sys.platform == "win32"

_import_cache: dict[str, bool] = {}


def qwen_model() -> str:
    """The Hugging Face model the qwen engine loads (override with SENPAIDUB_QWEN_MODEL)."""
    return os.environ.get("SENPAIDUB_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


class SetupError(SenpaiDubError):
    exit_code = 1


def qwen_env_dir() -> Path:
    return data_dir() / "venv-qwen"


def _env_python(env: Path) -> Path:
    return env / "Scripts" / "python.exe" if _IS_WINDOWS else env / "bin" / "python"


def _candidates() -> list[Path]:
    out: list[Path] = []
    override = os.environ.get(OVERRIDE_VAR)
    if override:
        out.append(Path(override))
    for env in (REPO_ROOT / REPO_ENV, qwen_env_dir()):
        out += [env / "bin" / "python", env / "Scripts" / "python.exe"]
    return out


def find_qwen_python() -> Path | None:
    """The Python with qwen-tts: $SENPAIDUB_QWEN_PYTHON, then ./.venv-qwen, then the data dir.

    The first candidate that can import qwen_tts wins, so a stale or half-installed env earlier
    in the order doesn't hide a working one. When none can, the first that exists is returned
    (for error messages); None when there is none at all.
    """
    existing = [c for c in _candidates() if c.is_file()]
    for c in existing:
        if _can_import(c):
            return c
    return existing[0] if existing else None


def _can_import(python: Path) -> bool:
    key = str(python)
    if key not in _import_cache:
        try:
            r = subprocess.run(
                [key, "-I", "-c", f"import {MODULE}"],
                capture_output=True, timeout=120, check=False,
            )  # fmt: skip
            ok = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:  # only cache success, so a later install is noticed
            _import_cache[key] = True
        return ok
    return _import_cache[key]


def is_installed() -> bool:
    py = find_qwen_python()
    return py is not None and _can_import(py)


def _version_of(cmd: list[str]) -> tuple[int, int] | None:
    try:
        r = subprocess.run(
            [*cmd, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30, check=False,
        )  # fmt: skip
        major, minor = r.stdout.strip().split(".")
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _base_command() -> list[str] | None:
    names = [f"python{a}.{b}" for a, b in SUPPORTED] + ["python3", "python"]
    for name in names:
        found = shutil.which(name)
        if found and _version_of([found]) in SUPPORTED:
            return [found]
    if _IS_WINDOWS:
        launcher = shutil.which("py")
        if launcher:
            for a, b in SUPPORTED:
                if _version_of([launcher, f"-{a}.{b}"]) == (a, b):
                    return [launcher, f"-{a}.{b}"]
    return None


def find_base_python() -> Path | None:
    """A Python 3.10-3.13 on PATH (PyTorch has no wheels for newer versions)."""
    cmd = _base_command()
    if cmd is None:
        return None
    if len(cmd) == 1:
        return Path(cmd[0])
    try:  # Windows 'py -3.x': resolve to the real interpreter
        r = subprocess.run(
            [*cmd, "-c", "import sys;print(sys.executable)"],
            capture_output=True, text=True, timeout=30, check=False,
        )  # fmt: skip
        return Path(r.stdout.strip()) if r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _run(cmd: list[str], on_line: Callable[[str], None], what: str) -> None:
    on_line("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )  # fmt: skip
    except OSError as e:
        raise SetupError(f"{what} could not start: {e}") from e
    assert proc.stdout is not None
    for line in proc.stdout:
        on_line(line.rstrip())
    if proc.wait() != 0:
        raise SetupError(f"{what} failed (exit code {proc.returncode}). See the log above.")


def _preload() -> str:
    return (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({qwen_model()!r}); print('model ready')"
    )


def install(on_line: Callable[[str], None] = print, download_model: bool = True) -> Path:
    """Create the qwen environment in qwen_env_dir(); idempotent. Returns its python."""
    env = qwen_env_dir()
    python = _env_python(env)
    if not python.is_file():
        base = find_base_python()
        if base is None:
            raise SetupError(
                "Qwen voices need Python 3.10 to 3.13 (PyTorch has no wheels for newer versions). "
                "Install Python 3.13 from python.org and try again."
            )
        on_line(f"Creating the voice environment in {env}")
        env.parent.mkdir(parents=True, exist_ok=True)
        _run([str(base), "-m", "venv", str(env)], on_line, "Creating the environment")
        if not python.is_file():
            raise SetupError(f"The environment was created but {python} is missing.")
    else:
        on_line(f"Reusing the environment in {env}")
    on_line("Installing Qwen3-TTS (a few GB, this takes several minutes)")
    _run([str(python), "-m", "pip", "install", *PACKAGES], on_line, "Installing Qwen3-TTS")
    if download_model:
        on_line("Downloading the voice model (~4.5 GB)")
        _run([str(python), "-I", "-c", _preload()], on_line, "Downloading the voice model")
    _import_cache.pop(str(python), None)
    return python
