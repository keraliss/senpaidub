"""HTTP API of the web UI (contract section 7), through FastAPI's TestClient."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from senpaidub import setup_tts
from senpaidub.config import config_path
from senpaidub.server import app as app_mod
from senpaidub.server import files
from senpaidub.server.app import create_app, sse_stream
from senpaidub.server.jobs import JobManager
from senpaidub.server.security import COOKIE, host_allowed

TOKEN = "test-token-123"
PORT = 7860
BASE = f"http://127.0.0.1:{PORT}"

FAKE_SYSTEM = {
    "system": {"os": "macOS", "os_version": "15", "arch": "arm64", "cpu": "Apple M5", "cores": 10,
               "apple_silicon": True, "ram_gb": 24.0, "gpu": {"kind": "mps", "name": None, "vram_gb": None},
               "disk_free_gb": {"library": 100.0, "cache": 100.0}, "qwen_installed": False,
               "internet": True, "ocr_available": True},
    "recommendation": {"engine": "qwen", "parallel": 1, "level": "great", "headline": "Great machine",
                       "warning": None, "minutes_per_episode": {"qwen": 22.0, "edge": 4.0, "google": 4.0},
                       "notes": []},
}  # fmt: skip


def wait_for(cond, timeout: float = 60.0, what: str = "condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def library(tmp_path) -> Path:
    lib = tmp_path / "library"
    lib.mkdir()
    return lib


@pytest.fixture
def manager(tmp_path, library, monkeypatch):
    monkeypatch.setenv("SENPAIDUB_ALLOW_MOCK", "1")
    # No Demucs in tests: the worker reads config.toml like the CLI does.
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'separate = false\nlibrary = "{library}"\n', encoding="utf-8")
    m = JobManager(db_path=tmp_path / "web.db", library=library)
    m.start()
    yield m
    m.close()


@pytest.fixture
def opened(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(files, "open_path", lambda p, action: calls.append((Path(p), action)))
    return calls


@pytest.fixture
def client(manager, library, monkeypatch, opened):
    monkeypatch.setattr(app_mod.system, "get", lambda lib, refresh=False: json.loads(json.dumps(FAKE_SYSTEM)))
    app = create_app(manager, TOKEN, library, port=PORT)
    with TestClient(app, base_url=BASE) as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303
        yield c


def upload(client, path: Path, name: str | None = None) -> dict:
    with open(path, "rb") as f:
        r = client.post("/api/uploads", files={"file": (name or path.name, f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------- security


def test_host_allowed():
    assert host_allowed("127.0.0.1:7860", 7860)
    assert host_allowed("localhost:7860", 7860)
    assert host_allowed("[::1]:7860", 7860)
    assert not host_allowed("127.0.0.1:7861", 7860)
    assert not host_allowed("evil.example:7860", 7860)
    assert not host_allowed("127.0.0.1.evil.example", None)
    assert not host_allowed(None, 7860)


def test_no_cookie_is_401(manager, library):
    app = create_app(manager, TOKEN, library, port=PORT)
    with TestClient(app, base_url=BASE) as c:
        r = c.get("/api/history")
        assert r.status_code == 401
        assert "senpaidub serve" in r.json()["error"]
        page = c.get("/settings")
        assert page.status_code == 401
        assert "senpaidub serve" in page.text
        # Static assets need no session (no secrets in them).
        assert c.get("/assets/style.css").status_code == 200
        # A wrong token doesn't set a cookie.
        bad = c.get("/?token=nope", follow_redirects=False)
        assert bad.status_code == 401
        assert COOKIE not in bad.headers.get("set-cookie", "")


def test_token_sets_cookie_and_redirects(manager, library):
    app = create_app(manager, TOKEN, library, port=PORT)
    with TestClient(app, base_url=BASE) as c:
        r = c.get(f"/setup?token={TOKEN}&uploads=abc", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/setup?uploads=abc"
        cookie = r.headers["set-cookie"]
        assert f"{COOKIE}={TOKEN}" in cookie
        assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
        assert c.get("/api/history").status_code == 200
        assert "<html" in c.get("/").text.lower()


def test_bad_host_is_400(manager, library):
    app = create_app(manager, TOKEN, library, port=PORT)
    with TestClient(app, base_url="http://evil.example:7860") as c:
        assert c.get(f"/?token={TOKEN}", follow_redirects=False).status_code == 400
    with TestClient(app, base_url="http://127.0.0.1:9999") as c:
        assert c.get("/api/history").status_code == 400


def test_cross_origin_refused(client):
    r = client.post("/api/system/recheck", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    ok = client.post("/api/system/recheck", headers={"Origin": BASE})
    assert ok.status_code == 200


def test_pages_and_assets(client):
    for route in ("/", "/setup", "/job", "/batch", "/settings"):
        r = client.get(route)
        assert r.status_code == 200, route
        assert "text/html" in r.headers["content-type"]
    for asset in ("style.css", "app.js", "flow.js", "run.js", "senpai-normal.svg"):
        assert client.get(f"/assets/{asset}").status_code == 200, asset
    assert client.get("/assets/nope.js").status_code == 404


# ---------------------------------------------------------------------------- system + setup


def test_system_routes(client):
    assert client.get("/api/system").json()["recommendation"]["engine"] == "qwen"
    assert client.post("/api/system/recheck").json()["system"]["cores"] == 10


def test_qwen_setup(client, monkeypatch):
    monkeypatch.setattr(setup_tts, "is_installed", lambda: False)

    def fake_install(on_line=print, download_model=True):
        on_line("step one")
        on_line("step two")
        return Path("/x/python")

    monkeypatch.setattr(setup_tts, "install", fake_install)
    st = client.get("/api/setup/qwen").json()
    assert st == {"installed": False, "installing": False, "log": [], "error": None}
    started = client.post("/api/setup/qwen").json()
    assert set(started) == {"installed", "installing", "log", "error"}
    done = wait_for(lambda: (lambda s: s if not s["installing"] else None)(client.get("/api/setup/qwen").json()),
                    timeout=5, what="install")  # fmt: skip
    assert "step two" in done["log"] and done["error"] is None

    def failing(on_line=print, download_model=True):
        raise setup_tts.SetupError("no python")

    monkeypatch.setattr(setup_tts, "install", failing)
    client.post("/api/setup/qwen")
    failed = wait_for(lambda: (lambda s: s if not s["installing"] else None)(client.get("/api/setup/qwen").json()),
                      timeout=5, what="failed install")  # fmt: skip
    assert failed["error"] == "no python"


# ---------------------------------------------------------------------------- settings


def test_settings_round_trip(client, library):
    s = client.get("/api/settings").json()
    assert s["library"] == str(library)
    assert s["google_key_set"] is False and s["google_key_hint"] is None
    new_lib = library.parent / "anime2"
    r = client.put("/api/settings", json={"tts": "edge", "library": str(new_lib), "keep_cache": True,
                                          "duck_db": -18, "max_speedup": 1.6})  # fmt: skip
    assert r.status_code == 200, r.text
    s = r.json()
    assert (s["tts"], s["library"], s["keep_cache"], s["duck_db"], s["max_speedup"]) == \
        ("edge", str(new_lib), True, -18.0, 1.6)  # fmt: skip
    text = config_path().read_text()
    assert 'tts = "edge"' in text and "separate = false" in text  # other keys kept

    r = client.put("/api/settings", json={"google_api_key": "  AIzaSyFAKEKEYabcd  "})
    assert r.json()["google_key_set"] is True
    assert r.json()["google_key_hint"] == "…abcd"
    key_file = config_path().with_name("google-api-key")
    assert key_file.read_text().strip() == "AIzaSyFAKEKEYabcd"
    if os.name == "posix":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert "AIzaSy" not in json.dumps(client.get("/api/settings").json())
    r = client.put("/api/settings", json={"google_api_key": ""})
    assert r.json()["google_key_set"] is False and not key_file.exists()


@pytest.mark.parametrize("body", [{"tts": "mock"}, {"duck_db": 5}, {"max_speedup": 0.5}, {"keep_cache": "yes"},
                                  {"library": ""}, {"library": "relative/path"}, {"colour": "red"}])  # fmt: skip
def test_settings_validation(client, body):
    r = client.put("/api/settings", json=body)
    assert r.status_code == 400
    assert r.json()["error"]


# ---------------------------------------------------------------------------- files


def test_fs_and_scan_path_rules(client, library):
    (library / "Show").mkdir()
    (library / "Show" / "ep02.mkv").write_bytes(b"x")
    (library / "Show" / "ep10.mkv").write_bytes(b"x")
    (library / ".hidden").mkdir()
    listing = client.get("/api/fs", params={"path": str(library)}).json()
    assert [e["name"] for e in listing["entries"]] == ["Show"]
    assert client.get("/api/fs").status_code == 200  # defaults to the first root
    outside = Path("/etc") if Path("/etc").is_dir() else Path(os.environ.get("SystemRoot", "C:/Windows"))
    assert client.get("/api/fs", params={"path": str(outside)}).status_code == 403
    assert client.get("/api/fs", params={"path": str(library / "missing")}).status_code == 404
    vids = client.post("/api/scan", json={"path": str(library / "Show")}).json()["videos"]
    assert [v["name"] for v in vids] == ["ep02.mkv", "ep10.mkv"]
    assert client.post("/api/scan", json={"path": str(outside)}).status_code == 403
    assert client.post("/api/scan", json={}).status_code == 400


def test_upload_rules(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with open(bad, "rb") as f:
        r = client.post("/api/uploads", files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 400
    sub = tmp_path / "x.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    info = upload(client, sub, name="../../evil.srt")
    assert info["kind"] == "subtitle" and info["name"] == "evil.srt" and "path" not in info
    assert files.upload_path(info["id"]).is_file()
    assert client.delete(f"/api/uploads/{info['id']}").json() == {"ok": True}
    with pytest.raises(KeyError):
        files.upload_path(info["id"])


def test_upload_refused_when_disk_full(client, tmp_path, monkeypatch):
    monkeypatch.setattr(files, "_free_bytes", lambda folder: files.UPLOAD_MARGIN + 10)
    sub = tmp_path / "x.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n" * 5)
    with open(sub, "rb") as f:
        r = client.post("/api/uploads", files={"file": ("x.srt", f, "text/plain")})
    assert r.status_code == 400 and "disk space" in r.json()["error"]


def test_settings_show_the_served_library(manager, library, tmp_path, monkeypatch):
    # `serve --library X` differs from config.toml's library: the UI must show X.
    monkeypatch.setattr(app_mod.system, "get", lambda lib, refresh=False: json.loads(json.dumps(FAKE_SYSTEM)))
    served = tmp_path / "served-lib"
    app = create_app(manager, TOKEN, served, port=PORT)
    with TestClient(app, base_url=BASE) as c:
        c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert c.get("/api/settings").json()["library"] == str(served)


def test_key_file_never_written_with_loose_permissions(client):
    if os.name != "posix":
        pytest.skip("POSIX permissions")
    key_file = config_path().with_name("google-api-key")
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text("OLDKEY\n")
    key_file.chmod(0o644)
    seen = []
    real_fdopen = os.fdopen

    def spy(fd, *a, **kw):
        seen.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return real_fdopen(fd, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_mod.os, "fdopen", spy)
        assert client.put("/api/settings", json={"google_api_key": "NEWKEY1234"}).status_code == 200
    assert seen == [0o600]
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert key_file.read_text().strip() == "NEWKEY1234"
    assert not list(key_file.parent.glob(".google-api-key*"))


def test_probe(client, sample_mkv, library):
    info = upload(client, sample_mkv)
    r = client.post("/api/probe", json={"upload_id": info["id"]})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["input"] == {"upload_id": info["id"]}
    assert p["auto_source"] == "track" and p["subtitles"][0]["lines"] > 0
    disk = shutil.copy(sample_mkv, library / "Episode 01.mkv")
    p2 = client.post("/api/probe", json={"path": str(disk)}).json()
    assert p2["input"] == {"path": str(Path(disk).resolve())}
    assert client.post("/api/probe", json={"path": "/etc/hosts"}).status_code in (403, 404)
    assert client.post("/api/probe", json={"upload_id": "0" * 32}).status_code == 404
    assert client.post("/api/probe", json={}).status_code == 400
    assert client.post("/api/probe", content=b"not json", headers={"content-type": "application/json"}).status_code == 400


# ---------------------------------------------------------------------------- batches + jobs


def test_batch_validation(client, sample_mkv):
    info = upload(client, sample_mkv)
    item = {"upload_id": info["id"]}
    assert client.post("/api/batches", json={"items": [item], "tts": "nope"}).status_code == 400
    assert client.post("/api/batches", json={"items": [], "tts": "edge"}).status_code == 400
    assert client.post("/api/batches", json={"items": [{"path": "/etc/hosts"}], "tts": "edge"}).status_code in (403, 404)
    assert client.post("/api/batches", json={"items": [{**item, "sub_track": "x"}], "tts": "edge"}).status_code == 400
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_mock_without_env_rejected(client, sample_mkv, monkeypatch):
    monkeypatch.delenv("SENPAIDUB_ALLOW_MOCK")
    info = upload(client, sample_mkv)
    r = client.post("/api/batches", json={"items": [{"upload_id": info["id"]}], "tts": "mock"})
    assert r.status_code == 400


def test_pause_resume_cancel_retry_remove(client, sample_mkv, manager):
    manager._runner = lambda spec: (_ for _ in ()).throw(AssertionError("must not start"))
    info = upload(client, sample_mkv)
    items = [{"upload_id": info["id"]}, {"upload_id": info["id"]}]
    manager.stop()  # nothing runs: the jobs stay waiting
    b = client.post("/api/batches", json={"items": items, "tts": "edge", "parallel": 2}).json()
    assert b["parallel"] == 2 and len(b["jobs"]) == 2
    assert client.post(f"/api/batches/{b['id']}/pause").json()["status"] == "paused"
    assert client.post(f"/api/batches/{b['id']}/resume").json()["status"] != "paused"
    j1, j2 = b["jobs"]
    assert client.post(f"/api/jobs/{j1['id']}/retry").status_code == 409
    assert client.post(f"/api/jobs/{j1['id']}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{j1['id']}/cancel").status_code == 409
    assert client.post(f"/api/jobs/{j1['id']}/retry").json()["status"] == "waiting"
    assert client.delete(f"/api/jobs/{j2['id']}").json() == {"ok": True}
    assert client.get(f"/api/jobs/{j2['id']}").status_code == 404
    assert len(client.get(f"/api/batches/{b['id']}").json()["jobs"]) == 1
    listed = client.get("/api/batches").json()["batches"]
    assert listed[0]["id"] == b["id"] and "jobs" not in listed[0]
    assert client.get(f"/api/batches/{b['id']}/zip").status_code == 404  # nothing done yet


def test_end_to_end_mock_dub(client, sample_mkv, library, opened):
    info = upload(client, sample_mkv)
    preview = client.post("/api/probe", json={"upload_id": info["id"]}).json()
    track = next(s["index"] for s in preview["subtitles"] if s["chosen"])
    r = client.post("/api/batches", json={"items": [{"upload_id": info["id"], "sub_track": track}],
                                          "tts": "mock", "series_name": "Test Show"})  # fmt: skip
    assert r.status_code == 200, r.text
    batch = r.json()
    job_id = batch["jobs"][0]["id"]
    assert batch["output_dir"] == str(library / "Test Show")

    def finished():
        j = client.get(f"/api/jobs/{job_id}").json()
        return j if j["status"] in ("done", "failed", "skipped", "cancelled") else None

    job = wait_for(finished, timeout=120, what="mock dub")
    assert job["status"] == "done", job
    assert job["progress"] == 1.0 and job["result"]["voiced"] > 0
    out = Path(job["output"])
    assert out.is_file() and out.parent == library / "Test Show"

    hist = client.get("/api/history").json()["jobs"]
    assert hist and hist[0]["id"] == job_id
    assert client.post("/api/open", json={"path": str(out), "action": "play"}).json() == {"ok": True}
    assert opened[-1] == (out.resolve(), "play")
    assert client.post("/api/open", json={"path": str(out), "action": "delete"}).status_code == 400
    assert client.post("/api/open", json={"path": str(library), "action": "reveal"}).status_code == 403

    z = client.get(f"/api/batches/{batch['id']}/zip")
    assert z.status_code == 200
    assert z.headers["content-type"] == "application/zip"
    assert 'filename="Test Show.zip"' in z.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert names == [out.name]


def test_sse_stream_formats_and_unsubscribes(manager):
    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def go():
        q = manager.subscribe()
        gen = sse_stream(manager, q, FakeRequest(), ping=0.05)
        chunks = [await gen.__anext__()]
        q.put_nowait({"type": "job", "job": {"id": "abc"}})
        chunks.append(await gen.__anext__())
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()  # times out, sees the disconnect, stops
        return q, chunks

    q, chunks = asyncio.run(go())
    assert chunks[0].startswith("retry:")
    assert chunks[1].startswith("event: job\ndata: ") and chunks[1].endswith("\n\n")
    assert json.loads(chunks[1].split("data: ", 1)[1]) == {"type": "job", "job": {"id": "abc"}}
    assert q not in manager._subs


# ---------------------------------------------------------------------------- CLI


def _cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", "from senpaidub.cli import main; main()", *args],
                          capture_output=True, text=True, cwd=cwd, timeout=120)  # fmt: skip


def test_cli_dub_still_works(sample_mkv, tmp_path):
    r = _cli(str(sample_mkv), "--list-tracks")
    assert r.returncode == 0, r.stderr
    assert "subtitle" in r.stdout
    r = _cli(str(sample_mkv), "--dry-run", "--tts", "mock")
    assert r.returncode == 0, r.stderr
    assert _cli("--version").stdout.strip().startswith("senpaidub ")
    assert _cli(str(tmp_path / "missing.mkv")).returncode == 1


def test_cli_subcommand_dispatch(tmp_path, monkeypatch):
    from senpaidub import cli

    assert cli._pick_app(["serve", "--port", "1"])[0] is cli.serve_app
    assert cli._pick_app(["setup", "qwen"])[1] == ["qwen"]
    assert cli._pick_app(["ep.mkv"])[0] is cli.app
    r = _cli("serve", "--help", cwd=tmp_path)
    assert r.returncode == 0 and "--no-browser" in r.stdout
    r = _cli("setup", "nothing", cwd=tmp_path)
    assert r.returncode == 1 and "setup qwen" in (r.stdout + r.stderr)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "serve").write_bytes(b"")  # a file literally named "serve" is still an input
    assert cli._pick_app(["serve"])[0] is cli.app


def test_free_port_skips_busy(tmp_path):
    import socket

    from senpaidub import cli

    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    try:
        sock, got = cli._free_port(port)
        sock.close()
        assert got > port
    finally:
        busy.close()
