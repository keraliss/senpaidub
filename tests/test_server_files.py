from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import zipfile

import pytest
from starlette.datastructures import UploadFile

from senpaidub.server import files


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=name)


def test_save_upload_and_lookup(isolated_dirs):
    data = b"x" * (2 * (1 << 20) + 5)
    r = asyncio.run(files.save_upload(_upload("ep 01.MKV", data)))
    assert r["kind"] == "video" and r["size"] == len(data)
    assert files.upload_path(r["id"]).read_bytes() == data
    files.delete_upload(r["id"])
    with pytest.raises(KeyError):
        files.upload_path(r["id"])


def test_upload_name_sanitized(isolated_dirs):
    r = asyncio.run(files.save_upload(_upload("../../etc/ev:il.srt", b"1")))
    assert r["kind"] == "subtitle"
    p = files.upload_path(r["id"])
    assert p.parent.parent == files.uploads_dir()
    assert "/" not in r["name"] and ":" not in r["name"]


def test_upload_rejects_extension(isolated_dirs):
    with pytest.raises(ValueError):
        asyncio.run(files.save_upload(_upload("a.exe", b"1")))
    with pytest.raises(ValueError):
        asyncio.run(files.save_upload(_upload("", b"1")))


@pytest.mark.parametrize("bad", ["../x", "abc", "", "A" * 32, "0" * 31, "g" * 32])
def test_upload_id_validation(isolated_dirs, bad):
    with pytest.raises(KeyError):
        files.upload_path(bad)
    files.delete_upload(bad)  # must not raise


def test_is_allowed_traversal_and_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    roots = [root.resolve()]
    assert files.is_allowed(root / "a", roots)
    assert not files.is_allowed(root / ".." / "outside", roots)
    assert not files.is_allowed(root / "link", roots)
    assert not files.is_allowed(tmp_path / "root2", roots)


def test_list_dir(tmp_path):
    root = tmp_path.resolve()
    (root / "ep10.mkv").write_bytes(b"1")
    (root / "ep2.mkv").write_bytes(b"1")
    (root / "notes.txt").write_text("x")
    (root / ".hidden").mkdir()
    (root / "zdir").mkdir()
    (root / "adir").mkdir()
    res = files.list_dir(root, [root])
    assert [e["name"] for e in res["entries"]] == ["adir", "zdir", "ep2.mkv", "ep10.mkv"]
    assert res["parent"] is None
    sub = files.list_dir(root / "adir", [root])
    assert sub["parent"] == str(root)
    assert files.list_dir(None, [root])["path"] == str(root)
    with pytest.raises(PermissionError):
        files.list_dir(root.parent, [root])


def test_scan_videos(tmp_path):
    root = tmp_path.resolve()
    (root / "s").mkdir()
    (root / "s" / "b.mkv").write_bytes(b"12")
    (root / "a.mp4").write_bytes(b"1")
    (root / "a [EN dub].mkv").write_bytes(b"1")
    res = files.scan_videos(root, [root])
    assert [v["name"] for v in res] == ["a.mp4", "b.mkv"]
    assert res[1]["size"] == 2
    with pytest.raises(PermissionError):
        files.scan_videos(root.parent, [root])


def test_zip_stream_roundtrip(tmp_path):
    a = tmp_path / "a.mkv"
    b = tmp_path / "sub" / "a.mkv"
    b.parent.mkdir()
    a.write_bytes(os.urandom(3 * (1 << 20) + 7))
    b.write_bytes(b"hello")
    chunks = list(files.zip_stream([a, b]))
    assert len(chunks) > 2
    with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as zf:
        assert zf.testzip() is None
        assert sorted(zf.namelist()) == ["a (2).mkv", "a.mkv"]
        assert zf.read("a.mkv") == a.read_bytes()
        assert zf.read("a (2).mkv") == b"hello"
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())


def test_zip_stream_empty():
    with zipfile.ZipFile(io.BytesIO(b"".join(files.zip_stream([])))) as zf:
        assert zf.namelist() == []


def _capture(monkeypatch):
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))

    monkeypatch.setattr(subprocess, "Popen", fake)
    return calls


def test_open_path_macos(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    files.open_path(tmp_path / "x.mkv", "reveal")
    files.open_path(tmp_path / "x.mkv", "play")
    assert calls[0][0] == ["open", "-R", str(tmp_path / "x.mkv")]
    assert calls[1][0] == ["open", str(tmp_path / "x.mkv")]
    assert not calls[0][1].get("shell")


def test_open_path_linux(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    files.open_path(tmp_path / "x.mkv", "reveal")
    files.open_path(tmp_path / "x.mkv", "play")
    assert calls[0][0] == ["xdg-open", str(tmp_path)]
    assert calls[1][0] == ["xdg-open", str(tmp_path / "x.mkv")]


def test_open_path_windows(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    started = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", lambda p: started.append(p), raising=False)
    files.open_path(tmp_path / "x.mkv", "reveal")
    files.open_path(tmp_path / "x.mkv", "play")
    assert calls[0][0] == ["explorer", f"/select,{tmp_path / 'x.mkv'}"]
    assert started == [str(tmp_path / "x.mkv")]


def test_open_path_bad_action(tmp_path):
    with pytest.raises(ValueError):
        files.open_path(tmp_path, "delete")


def test_disk_check(tmp_path):
    f = tmp_path / "a.mkv"
    f.write_bytes(b"x" * 1000)
    r = files.disk_check([f], tmp_path / "new" / "dir")
    assert r["ok"] is True and r["free_gb"] > 0


def test_upload_refused_without_disk_space(isolated_dirs, monkeypatch):
    monkeypatch.setattr(files, "_free_bytes", lambda folder: files.UPLOAD_MARGIN + 10)
    with pytest.raises(ValueError, match="disk space"):
        asyncio.run(files.save_upload(_upload("ep.mkv", b"x"), expected_size=100))
    with pytest.raises(ValueError, match="disk space"):  # no Content-Length: stops while writing
        asyncio.run(files.save_upload(_upload("ep.mkv", b"x" * 100)))
    assert not any(files.uploads_dir().iterdir())  # nothing half-written left behind
    r = asyncio.run(files.save_upload(_upload("ep.mkv", b"x" * 5), expected_size=5))
    assert r["size"] == 5


def test_upload_id_of_and_stale(isolated_dirs):
    r = asyncio.run(files.save_upload(_upload("ep.mkv", b"1")))
    assert files.upload_id_of(r["path"]) == r["id"]
    assert files.upload_id_of(isolated_dirs / "elsewhere.mkv") is None
    old = files.uploads_dir() / r["id"]
    os.utime(old, (0, 0))
    fresh = asyncio.run(files.save_upload(_upload("ep2.mkv", b"1")))
    assert files.stale_uploads(set()) == [r["id"]]
    assert files.stale_uploads({r["id"]}) == []
    assert fresh["id"] not in files.stale_uploads(set())
