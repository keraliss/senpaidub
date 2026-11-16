"""File handling for the web UI: uploads, folder browsing, open/reveal, zip streaming."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ..batch import find_videos, natural_key
from ..config import cache_dir

if TYPE_CHECKING:
    from starlette.datastructures import UploadFile

VIDEO_EXTS = {".mkv", ".mp4"}
SUB_EXTS = {".srt", ".ass", ".ssa"}

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_MARGIN = 1 << 30  # keep 1 GB free on the cache disk while saving uploads
NO_SPACE = "Not enough disk space for this upload."
_BAD_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def uploads_dir() -> Path:
    return cache_dir() / "uploads"


def _sanitize_name(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]  # drop any directory part
    name = _BAD_CHARS.sub("_", name).strip().strip(".")
    return name[:200]


def _free_bytes(folder: Path) -> int:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 1 << 62  # unknown: don't block the upload


async def save_upload(upload: UploadFile, expected_size: int | None = None) -> dict:
    """Stream `upload` into its own folder. `expected_size` (e.g. the request's Content-Length) is
    checked against free space first; writing also stops once the disk is nearly full."""
    name = _sanitize_name(upload.filename or "")
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXTS:
        kind = "video"
    elif ext in SUB_EXTS:
        kind = "subtitle"
    else:
        raise ValueError("Unsupported file type. Use .mkv, .mp4, .srt, .ass or .ssa.")
    if not Path(name).stem:
        raise ValueError("Invalid file name.")
    budget = _free_bytes(uploads_dir()) - UPLOAD_MARGIN
    if expected_size is not None and expected_size > budget:
        raise ValueError(NO_SPACE)
    upload_id = uuid.uuid4().hex
    folder = uploads_dir() / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / name
    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > budget:
                    raise ValueError(NO_SPACE)
                f.write(chunk)
    except BaseException:
        delete_upload(upload_id)
        raise
    return {"id": upload_id, "name": name, "size": size, "kind": kind, "path": str(dest)}


def _upload_folder(upload_id: str) -> Path:
    if not isinstance(upload_id, str) or not _ID_RE.match(upload_id):
        raise KeyError(upload_id)
    return uploads_dir() / upload_id


def upload_path(upload_id: str) -> Path:
    folder = _upload_folder(upload_id)
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.is_file():
                return p
    raise KeyError(upload_id)


def upload_id_of(path: Path | str) -> str | None:
    """The upload id when `path` is a staged upload (uploads_dir()/<id>/<name>), else None."""
    try:
        rel = Path(path).resolve().relative_to(uploads_dir().resolve())
    except (ValueError, OSError):
        return None
    first = rel.parts[0] if rel.parts else ""
    return first if _ID_RE.match(first) else None


def stale_uploads(keep: set[str], max_age: float = 24 * 3600) -> list[str]:
    """Ids of upload folders older than `max_age` seconds that aren't in `keep`."""
    root = uploads_dir()
    if not root.is_dir():
        return []
    cutoff = time.time() - max_age
    out = []
    for d in root.iterdir():
        try:
            if d.is_dir() and _ID_RE.match(d.name) and d.name not in keep and d.stat().st_mtime < cutoff:
                out.append(d.name)
        except OSError:
            continue
    return out


def delete_upload(upload_id: str) -> None:
    try:
        folder = _upload_folder(upload_id)
    except KeyError:
        return
    shutil.rmtree(folder, ignore_errors=True)


def allowed_roots(library: Path) -> list[Path]:
    roots: list[Path] = []
    for r in (Path.home(), library):
        try:
            rr = r.resolve()
        except OSError:
            continue
        if rr not in roots:
            roots.append(rr)
    return roots


def is_allowed(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    return any(resolved == r or r in resolved.parents for r in roots)


def _root_entries(roots: list[Path]) -> list[dict]:
    return [{"name": r.name or str(r), "path": str(r)} for r in roots]


def list_dir(path: Path | None, roots: list[Path]) -> dict:
    target = Path(path).resolve() if path else roots[0]
    if not is_allowed(target, roots):
        raise PermissionError("That folder is outside the allowed locations.")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a folder: {target}")
    entries: list[dict] = []
    for p in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), natural_key(p))):
        if p.name.startswith("."):
            continue
        try:
            if p.is_dir():
                entries.append({"name": p.name, "path": str(p), "kind": "dir", "size": None})
            elif p.suffix.lower() in VIDEO_EXTS:
                entries.append({"name": p.name, "path": str(p), "kind": "video", "size": p.stat().st_size})
        except OSError:
            continue
    parent = target.parent
    parent_s = str(parent) if parent != target and is_allowed(parent, roots) else None
    return {"path": str(target), "parent": parent_s, "roots": _root_entries(roots), "entries": entries}


def scan_videos(folder: Path, roots: list[Path]) -> list[dict]:
    folder = Path(folder).resolve()
    if not is_allowed(folder, roots):
        raise PermissionError("That folder is outside the allowed locations.")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    out = []
    for v in find_videos(folder):
        if not is_allowed(v, roots):  # symlink escaping the roots
            continue
        out.append({"path": str(v.resolve()), "name": v.name, "size": v.stat().st_size})
    return out


def open_path(path: Path, action: str) -> None:
    if action not in ("reveal", "play"):
        raise ValueError(f"Unknown action: {action}")
    path = Path(path)
    if sys.platform == "darwin":
        argv = ["open", "-R", str(path)] if action == "reveal" else ["open", str(path)]
    elif sys.platform.startswith("win"):
        if action == "reveal":
            argv = ["explorer", f"/select,{path}"]
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
            return
    else:
        target = path.parent if action == "reveal" else path
        argv = ["xdg-open", str(target)]
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603


class _Sink:
    """Write-only, non-seekable buffer that zipfile streams into."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.pos = 0

    def write(self, data) -> int:
        self.buf += data
        self.pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        data = bytes(self.buf)
        self.buf.clear()
        return data


def zip_stream(paths: list[Path]) -> Iterator[bytes]:
    sink = _Sink()
    used: set[str] = set()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:  # type: ignore[arg-type]
        for p in paths:
            p = Path(p)
            name, n = p.name, 1
            while name in used:
                n += 1
                name = f"{p.stem} ({n}){p.suffix}"
            used.add(name)
            info = zipfile.ZipInfo.from_file(p, name)
            info.compress_type = zipfile.ZIP_STORED
            with open(p, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
                    data = sink.take()
                    if data:
                        yield data
            data = sink.take()
            if data:
                yield data
    data = sink.take()
    if data:
        yield data


def disk_check(inputs: list[Path], dest: Path) -> dict:
    needed = 1.2 * sum(Path(p).stat().st_size for p in inputs if Path(p).exists())
    probe = Path(dest)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return {"needed_gb": round(needed / 1e9, 2), "free_gb": round(free / 1e9, 2), "ok": free >= needed}
