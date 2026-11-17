"""FastAPI app for the local web UI (`senpaidub serve`).

`create_app(manager, token, library)` wires the job manager, file helpers, machine check and
qwen installer to the HTTP API in the build contract (section 7) and serves the static UI.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import setup_tts
from ..config import DEFAULTS, config_path, load_config, resolve
from ..download import sanitize_filename
from ..errors import SenpaiDubError
from . import files, system
from .jobs import JobManager, JobStateError
from .security import LocalOnlyMiddleware

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
PAGES = {"/": "index.html", "/setup": "setup.html", "/job": "job.html", "/batch": "batch.html",
         "/settings": "settings.html"}  # fmt: skip
PING_SECONDS = 15.0
ENGINES = ("qwen", "edge", "google")
SETTINGS_KEYS = ("tts", "library", "keep_cache", "qwen_mode", "duck_db", "max_speedup")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _err(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


# ---------------------------------------------------------------------------- settings helpers


def _key_file() -> Path:
    return config_path().with_name("google-api-key")


def _read_key() -> str:
    try:
        key = _key_file().read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    return key or os.environ.get("GOOGLE_API_KEY", "").strip()


def _write_key(key: str) -> None:
    """Write the key into a fresh 0600 temp file, then swap it in, so the secret never lands in a
    file with looser permissions (an existing hand-made key file may be 0644)."""
    f = _key_file()
    if not key:
        f.unlink(missing_ok=True)
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key + "\n")
        os.replace(tmp, f)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_config(data: dict) -> None:
    import tomli_w

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(tomli_w.dumps({k: v for k, v in data.items() if v is not None}), encoding="utf-8")
    tmp.replace(path)


def _settings_view(library: Path | None = None) -> dict:
    """`library` is the folder the server actually uses (it may come from `serve --library`)."""
    cfg = resolve({}, load_config())
    key = _read_key()
    return {
        "tts": cfg["tts"] if cfg["tts"] in ENGINES else DEFAULTS["tts"],
        "library": str(library if library is not None else Path(cfg["library"]).expanduser()),
        "keep_cache": bool(cfg["keep_cache"]),
        "qwen_mode": cfg["qwen_mode"] if cfg["qwen_mode"] in ("gentle", "fast") else DEFAULTS["qwen_mode"],
        "duck_db": float(cfg["duck_db"]),
        "max_speedup": float(cfg["max_speedup"]),
        "google_key_set": bool(key),
        "google_key_hint": f"…{key[-4:]}" if len(key) >= 8 else ("set" if key else None),
    }


def _num(body: dict, key: str, lo: float, hi: float) -> float:
    v = body[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(f"{key} must be a number.")
    if not lo <= float(v) <= hi:
        raise ApiError(f"{key} must be between {lo:g} and {hi:g}.")
    return float(v)


def _validated_settings(body: dict) -> dict:
    out: dict[str, Any] = {}
    if "tts" in body:
        if body["tts"] not in ENGINES:
            raise ApiError("tts must be qwen, edge or google.")
        out["tts"] = body["tts"]
    if "library" in body:
        lib = body["library"]
        if not isinstance(lib, str) or not lib.strip():
            raise ApiError("library must be a folder path.")
        p = Path(lib.strip()).expanduser()
        if not p.is_absolute():
            raise ApiError("library must be an absolute folder path.")
        if p.exists() and not p.is_dir():
            raise ApiError(f"{p} is a file, not a folder.")
        out["library"] = str(p)
    if "keep_cache" in body:
        if not isinstance(body["keep_cache"], bool):
            raise ApiError("keep_cache must be true or false.")
        out["keep_cache"] = body["keep_cache"]
    if "qwen_mode" in body:
        if body["qwen_mode"] not in ("gentle", "fast"):
            raise ApiError("qwen_mode must be gentle or fast.")
        out["qwen_mode"] = body["qwen_mode"]
    if "duck_db" in body:
        out["duck_db"] = _num(body, "duck_db", -40, 0)
    if "max_speedup" in body:
        out["max_speedup"] = _num(body, "max_speedup", 1.0, 2.0)
    return out


# ---------------------------------------------------------------------------- qwen installer


class QwenInstall:
    def __init__(self, on_installed: Callable[[], None] | None = None) -> None:
        self.on_installed = on_installed
        self.lock = threading.Lock()
        self.installing = False
        self.error: str | None = None
        self.log: deque[str] = deque(maxlen=200)

    def view(self) -> dict:
        with self.lock:
            installing, error, log = self.installing, self.error, list(self.log)
        installed = False if installing else _qwen_installed()
        return {"installed": installed, "installing": installing, "log": log, "error": error}

    def start(self) -> None:
        with self.lock:
            if self.installing:
                return
            self.installing = True
            self.error = None
            self.log.clear()
        threading.Thread(target=self._run, name="qwen-install", daemon=True).start()

    def _line(self, line: str) -> None:
        with self.lock:
            self.log.append(str(line).rstrip())

    def _run(self) -> None:
        try:
            setup_tts.install(on_line=self._line)
            self._line("Qwen voices are installed.")
            if self.on_installed is not None:
                try:
                    self.on_installed()
                except Exception:  # noqa: BLE001 - the machine check can be re-run by hand
                    pass
        except Exception as e:  # noqa: BLE001 - shown in the UI
            with self.lock:
                self.error = str(e) or type(e).__name__
        finally:
            with self.lock:
                self.installing = False


def _qwen_installed() -> bool:
    try:
        return setup_tts.is_installed()
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------- the app


def create_app(manager: JobManager, token: str, library: Path, port: int | None = None) -> FastAPI:
    app = FastAPI(title="Senpai Dub", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalOnlyMiddleware, token=token, port=port)
    app.state.library = Path(library).expanduser()
    def lib() -> Path:
        return app.state.library

    qwen = QwenInstall(on_installed=lambda: system.get(lib(), refresh=True))
    app.state.qwen = qwen

    def roots() -> list[Path]:
        return files.allowed_roots(lib())

    def disk_path(raw: Any) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ApiError("path must be a file path.")
        p = Path(raw).expanduser()
        if not files.is_allowed(p, roots()):
            raise ApiError("That location is outside your home folder and library.", 403)
        return p.resolve()

    def video_input(item: dict) -> Path:
        if item.get("upload_id"):
            p = files.upload_path(str(item["upload_id"]))
        elif item.get("path"):
            p = disk_path(item["path"])
        else:
            raise ApiError("Give an upload_id or a path.")
        if not p.is_file():
            raise ApiError(f"Video not found: {p}", 404)
        if p.suffix.lower() not in files.VIDEO_EXTS:
            raise ApiError(f"Not a video (.mkv or .mp4): {p.name}")
        return p

    def sub_input(item: dict) -> Path | None:
        sid = item.get("sub_upload_id")
        if not sid:
            return None
        p = files.upload_path(str(sid))
        if p.suffix.lower() not in files.SUB_EXTS:
            raise ApiError("The extra subtitle file must be .srt, .ass or .ssa.")
        return p

    async def body_of(request: Request) -> dict:
        try:
            data = await request.json()
        except Exception as e:  # noqa: BLE001
            raise ApiError("Expected a JSON body.") from e
        if not isinstance(data, dict):
            raise ApiError("Expected a JSON object.")
        return data

    # ------------------------------------------------------------------ errors

    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, e: ApiError):
        return _err(str(e), e.status)

    @app.exception_handler(JobStateError)
    async def _state_error(_req: Request, e: JobStateError):
        return _err(str(e), 409)

    @app.exception_handler(KeyError)
    async def _not_found(_req: Request, _e: KeyError):
        return _err("Not found.", 404)

    @app.exception_handler(PermissionError)
    async def _forbidden(_req: Request, e: PermissionError):
        return _err(str(e) or "Not allowed.", 403)

    @app.exception_handler(NotADirectoryError)
    async def _not_dir(_req: Request, e: NotADirectoryError):
        return _err(str(e) or "Not a folder.", 400)

    @app.exception_handler(FileNotFoundError)
    async def _no_file(_req: Request, e: FileNotFoundError):
        return _err(str(e) or "File not found.", 404)

    @app.exception_handler(ValueError)
    async def _bad_value(_req: Request, e: ValueError):
        return _err(str(e) or "Bad request.", 400)

    @app.exception_handler(SenpaiDubError)
    async def _senpai(_req: Request, e: SenpaiDubError):
        return _err(str(e), 400)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, e: RequestValidationError):
        errs = e.errors()
        msg = "; ".join(f"{'.'.join(str(x) for x in er.get('loc', ())[1:]) or 'body'}: {er.get('msg')}"
                        for er in errs[:3])  # fmt: skip
        return _err(msg or "Bad request.", 400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_req: Request, e: StarletteHTTPException):
        return _err(str(e.detail), e.status_code)

    @app.exception_handler(Exception)
    async def _crash(_req: Request, e: Exception):
        return _err(f"Internal error: {type(e).__name__}: {e}", 500)

    # ------------------------------------------------------------------ pages

    def page(name: str):
        async def handler():
            return FileResponse(UI_DIR / name, media_type="text/html", headers={"Cache-Control": "no-cache"})

        return handler

    for route, name in PAGES.items():
        app.get(route, include_in_schema=False)(page(name))

    app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")

    # ------------------------------------------------------------------ system

    @app.get("/api/system")
    def get_system():
        return system.get(lib())

    @app.post("/api/system/recheck")
    def recheck_system():
        return system.get(lib(), refresh=True)

    # ------------------------------------------------------------------ uploads and files

    @app.post("/api/uploads")
    async def upload(file: UploadFile, request: Request):
        try:
            expected = int(request.headers.get("content-length") or 0) or None
        except ValueError:
            expected = None
        info = await files.save_upload(file, expected)
        info.pop("path", None)
        return info

    @app.delete("/api/uploads/{upload_id}")
    def delete_upload(upload_id: str):
        files.delete_upload(upload_id)
        return {"ok": True}

    @app.get("/api/fs")
    def list_fs(path: str | None = None):
        p = Path(path).expanduser() if path else None
        if p is not None and not p.exists():
            raise ApiError(f"Folder not found: {p}", 404)
        return files.list_dir(p, roots())

    @app.post("/api/scan")
    async def scan(request: Request):
        body = await body_of(request)
        folder = disk_path(body.get("path"))
        if not folder.is_dir():
            raise ApiError(f"Not a folder: {folder}")
        videos = await asyncio.to_thread(files.scan_videos, folder, roots())
        return {"videos": videos}

    @app.post("/api/probe")
    async def probe(request: Request):
        from ..preview import preview

        body = await body_of(request)
        video = video_input(body)
        sub = sub_input(body)
        result = await asyncio.to_thread(preview, video, sub)
        result["input"] = ({"upload_id": body["upload_id"]} if body.get("upload_id") else {"path": str(video)})
        return result

    # ------------------------------------------------------------------ batches and jobs

    @app.post("/api/batches")
    async def create_batch(request: Request):
        body = await body_of(request)
        tts = body.get("tts")
        allowed = ENGINES + (("mock",) if os.environ.get("SENPAIDUB_ALLOW_MOCK") == "1" else ())
        if tts not in allowed:
            raise ApiError("Pick a voice engine: qwen, edge or google.")
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ApiError("Add at least one video.")
        items = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ApiError("Each item must be an object.")
            video = video_input(raw)
            sub = sub_input(raw)
            track = raw.get("sub_track")
            if track is not None and (isinstance(track, bool) or not isinstance(track, int)):
                raise ApiError("sub_track must be a stream number.")
            items.append({"input": str(video), "sub_file": str(sub) if sub else None, "sub_track": track,
                          "use_ocr": bool(raw.get("use_ocr")), "name": video.name})  # fmt: skip
        series = body.get("series_name")
        if series is not None and not isinstance(series, str):
            raise ApiError("series_name must be text.")
        series = (series or "").strip() or None
        try:
            parallel = max(1, min(8, int(body.get("parallel") or 1)))
        except (TypeError, ValueError) as e:
            raise ApiError("parallel must be a number.") from e
        inputs = [Path(i["input"]) for i in items]
        if series:
            dest = lib() / sanitize_filename(series)
        elif any(i.startswith(str(files.uploads_dir().resolve())) for i in (str(p) for p in inputs)):
            dest = lib()
        else:
            dest = inputs[0].parent
        check = files.disk_check(inputs, dest)
        if not check["ok"]:
            raise ApiError(f"Not enough disk space: about {check['needed_gb']:.1f} GB needed, "
                           f"{check['free_gb']:.1f} GB free.")  # fmt: skip
        return manager.create_batch(items, tts, series_name=series, parallel=parallel,
                                    force=bool(body.get("force")))  # fmt: skip

    @app.get("/api/batches")
    def list_batches():
        return {"batches": manager.list_batches()}

    @app.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str):
        return manager.get_batch(batch_id)

    @app.post("/api/batches/{batch_id}/pause")
    def pause_batch(batch_id: str):
        return manager.pause_batch(batch_id)

    @app.post("/api/batches/{batch_id}/resume")
    def resume_batch(batch_id: str):
        return manager.resume_batch(batch_id)

    @app.get("/api/batches/{batch_id}/zip")
    def zip_batch(batch_id: str):
        b = manager.get_batch(batch_id)
        outs = [Path(j["output"]) for j in b["jobs"] if j["status"] in ("done", "skipped") and j.get("output")]
        outs = [p for p in outs if p.is_file()]
        if not outs:
            raise ApiError("No finished dubs to download yet.", 404)
        name = sanitize_filename(b.get("series_name") or "") or "senpai-dub"
        ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
        disp = f"attachment; filename=\"{ascii_name}.zip\"; filename*=UTF-8''{quote(name + '.zip')}"
        return StreamingResponse(files.zip_stream(outs), media_type="application/zip",
                                 headers={"Content-Disposition": disp})  # fmt: skip

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return manager.get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return manager.cancel_job(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        return manager.retry_job(job_id)

    @app.delete("/api/jobs/{job_id}")
    def remove_job(job_id: str):
        manager.remove_job(job_id)
        return {"ok": True}

    @app.get("/api/history")
    def history():
        return {"jobs": manager.history()}

    @app.post("/api/open")
    async def open_path(request: Request):
        body = await body_of(request)
        action = body.get("action")
        if action not in ("reveal", "play"):
            raise ApiError("action must be reveal or play.")
        raw = body.get("path")
        if not isinstance(raw, str) or not raw:
            raise ApiError("path is required.")
        p = Path(raw)
        known = {str(Path(k).resolve()) for k in manager.known_paths()}
        if str(p.resolve()) not in known:
            raise ApiError("Only dubs and videos from your jobs can be opened.", 403)
        if not p.exists():
            raise ApiError(f"File not found (moved or deleted?): {p}", 404)
        files.open_path(p.resolve(), action)
        return {"ok": True}

    @app.get("/api/events")
    async def events(request: Request):
        q = manager.subscribe()
        return StreamingResponse(sse_stream(manager, q, request), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # fmt: skip

    # ------------------------------------------------------------------ settings

    @app.get("/api/settings")
    def get_settings():
        return _settings_view(lib())

    @app.put("/api/settings")
    async def put_settings(request: Request):
        body = await body_of(request)
        unknown = sorted(set(body) - set(SETTINGS_KEYS) - {"google_api_key"})
        if unknown:
            raise ApiError(f"Unknown setting(s): {', '.join(unknown)}")
        changes = _validated_settings(body)
        if "google_api_key" in body:
            key = body["google_api_key"]
            if not isinstance(key, str) or any(c.isspace() for c in key.strip()):
                raise ApiError("The Google API key must be a single line of text.")
            _write_key(key.strip())
        if changes:
            data = load_config()  # validates the existing file first
            data.update(changes)
            _write_config(data)
            if "library" in changes:
                app.state.library = Path(changes["library"])
                manager.library = Path(changes["library"])
        return _settings_view(lib())

    # ------------------------------------------------------------------ qwen setup

    @app.get("/api/setup/qwen")
    def qwen_status():
        return qwen.view()

    @app.post("/api/setup/qwen")
    def qwen_install():
        qwen.start()
        return qwen.view()

    return app


async def sse_stream(manager: JobManager, q: asyncio.Queue, request: Request, ping: float | None = None):
    """Server-sent events from the manager's subscriber queue, with a ping comment every 15 s."""
    interval = PING_SECONDS if ping is None else ping
    try:
        yield "retry: 3000\n: connected\n\n"
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=interval)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    break
                yield ": ping\n\n"
                continue
            kind = event.get("type", "message")
            yield f"event: {kind}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
    finally:
        manager.unsubscribe(q)
