from __future__ import annotations

import json

from senpaidub.preview import preview

from .conftest import FIXTURES


def test_mkv_with_ass(sample_mkv):
    p = preview(sample_mkv)
    json.dumps(p)
    assert p["name"] == "Episode 01.mkv"
    assert p["resolution"] == "320x240"
    assert p["duration"] > 10 and p["size"] > 0 and p["encrypted"] is False
    assert p["audio"][0]["language"] == "jpn"
    (t,) = p["subtitles"]
    assert t["is_text"] and t["chosen"] and t["error"] is None
    assert t["lines"] > 0 and t["skipped"] >= 0
    assert p["auto_source"] == "track" and p["needs_ocr"] is False and p["sub_file"] is None


def test_mp4_mov_text(sample_mp4):
    p = preview(sample_mp4)
    (t,) = p["subtitles"]
    assert t["codec"] == "mov_text" and t["lines"] >= 1 and t["skipped"] >= 1  # the song line


def test_sub_file(sample_mkv):
    p = preview(sample_mkv, sub_file=FIXTURES / "sample.srt")
    assert p["sub_file"]["name"] == "sample.srt"
    assert p["sub_file"]["lines"] >= 1 and p["sub_file"]["error"] is None
    assert p["auto_source"] == "sub_file"


def test_bad_sub_file(sample_mkv, tmp_path):
    p = preview(sample_mkv, sub_file=tmp_path / "missing.srt")
    assert p["sub_file"]["lines"] is None and p["sub_file"]["error"]


def test_no_subs(no_subs_mkv, monkeypatch):
    monkeypatch.setattr("senpaidub.preview.hardsubs.available", lambda: True)
    p = preview(no_subs_mkv)
    assert p["subtitles"] == [] and p["needs_ocr"] and p["auto_source"] == "ocr"
    monkeypatch.setattr("senpaidub.preview.hardsubs.available", lambda: False)
    p = preview(no_subs_mkv)
    assert p["auto_source"] == "none" and p["ocr_available"] is False


def test_track_error_is_captured(sample_mkv, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr("senpaidub.preview.subtitles.extract", boom)
    t = preview(sample_mkv)["subtitles"][0]
    assert t["lines"] is None and "kaput" in t["error"]


def test_exclude_styles_changes_counts(sample_mkv):
    base = preview(sample_mkv)["subtitles"][0]["lines"]
    assert preview(sample_mkv, exclude_styles=".*")["subtitles"][0]["lines"] < base
