"""The Reporter interface: stage events from a real pipeline run (MockTTS)."""

from __future__ import annotations

import io
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from senpaidub import separate
from senpaidub.pipeline import Options, process_file
from senpaidub.report import GROUPS, STAGE_GROUPS, NullReporter, Reporter, RichReporter

pytestmark = pytest.mark.integration


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.lock = threading.Lock()

    def _add(self, *ev: object) -> None:
        with self.lock:
            self.events.append(ev)

    @contextmanager
    def _stage(self, key, label, total, unit):
        rec = self

        class S:
            done = 0.0

            def advance(self, n: float = 1) -> None:
                with rec.lock:
                    self.done += n
                rec._add("advance", key, n)

            def update(self, completed=None, total=None) -> None:
                rec._add("update", key, completed, total)

        s = S()
        self._add("stage", key, label, total, unit)
        try:
            yield s
        finally:
            self._add("end", key, s.done)

    def stage(self, key, label, total=None, unit="items"):
        return self._stage(key, label, total, unit)

    def info(self, msg, style=None):
        self._add("info", msg, style)

    def warn(self, msg):
        self._add("warn", msg)

    def stages(self) -> list[str]:
        return [e[1] for e in self.events if e[0] == "stage"]

    def get(self, key: str) -> tuple:
        return next(e for e in self.events if e[0] == "stage" and e[1] == key)

    def advanced(self, key: str) -> float:
        return next(e[2] for e in self.events if e[0] == "end" and e[1] == key)


def test_groups_cover_every_stage():
    assert GROUPS == ["subtitles", "separate", "voice", "mix", "mux"]
    assert set(STAGE_GROUPS.values()) == set(GROUPS)
    assert set(STAGE_GROUPS) == {"download", "subtitles", "ocr", "extract", "separate", "cache_stems",
                                 "voices", "tts", "voicecheck", "fit", "mix", "encode", "mux"}  # fmt: skip


def test_null_reporter_is_silent():
    r = NullReporter()
    with r.stage("tts", "x", 3) as st:
        st.advance()
        st.update(completed=2, total=3)
    r.info("hi")
    r.warn("careful")


def test_rich_reporter_output():
    console = Console(record=True, width=120)
    r = RichReporter(console)
    with r.stage("fit", "Fitting", 2) as st:
        st.advance(2)
    with r.stage("mux", "Muxing", unit="none") as st:
        st.advance()
    r.info("Subtitles: stream #2 ([eng]) — 5 lines")
    r.info("Kept temp files", style="dim")
    r.warn("not printed here: the logging handler prints warnings")
    text = console.export_text()
    assert "Subtitles: stream #2 ([eng]) — 5 lines" in text  # no markup interpretation
    assert "Kept temp files" in text
    assert "not printed" not in text


def _run(sample_mkv: Path, tmp_path: Path, reporter, **kw) -> object:
    opts = Options(tts="mock", output=tmp_path / "out.mkv", **kw)
    return process_file(sample_mkv, opts, Console(file=io.StringIO()), reporter)


def _assert_well_formed(rec: RecordingReporter) -> None:
    open_: list[str] = []
    for e in rec.events:
        if e[0] == "stage":
            assert e[1] in STAGE_GROUPS
            assert e[4] in ("items", "seconds", "none")
            assert not open_, f"stage {e[1]} started inside {open_}"
            open_.append(e[1])
        elif e[0] == "end":
            assert open_ and open_.pop() == e[1]
        elif e[0] in ("advance", "update"):
            assert open_ and open_[-1] == e[1]
    assert not open_
    order = [GROUPS.index(STAGE_GROUPS[k]) for k in rec.stages()]
    assert order == sorted(order), rec.stages()


def test_process_file_reports_stages(sample_mkv, tmp_path, isolated_dirs):
    rec = RecordingReporter()
    res = _run(sample_mkv, tmp_path, rec, separate=False)
    assert res.output and res.output.is_file()
    assert rec.stages() == ["subtitles", "extract", "tts", "fit", "mix", "encode", "mux"]
    _assert_well_formed(rec)
    assert rec.get("tts")[3] == res.voiced == 5
    assert rec.advanced("tts") == res.voiced
    assert rec.advanced("fit") == res.voiced
    assert rec.get("mix")[4] == "seconds"
    assert any(e[0] == "update" and e[1] == "mix" for e in rec.events)
    assert any(e[0] == "info" and "5 spoken lines" in e[1] for e in rec.events)


def test_separation_stages(sample_mkv, tmp_path, isolated_dirs, monkeypatch):
    def fake_stems(src, out_no_vocals, out_vocals, *, device="auto", on_progress=None):
        shutil.copy(src, out_no_vocals)
        shutil.copy(src, out_vocals)
        if on_progress:
            on_progress(1.0, 2.0)
        return out_no_vocals, out_vocals

    monkeypatch.setattr(separate, "separate_stems", fake_stems)
    rec = RecordingReporter()
    res = _run(sample_mkv, tmp_path, rec, clean_cache=False)
    assert rec.stages()[:4] == ["subtitles", "extract", "separate", "cache_stems"]
    assert ("update", "separate", 1.0, 2.0) in rec.events
    _assert_well_formed(rec)
    assert res.separation_cached is False

    rec2 = RecordingReporter()
    _run(sample_mkv, tmp_path, rec2, force=True)
    assert "separate" not in rec2.stages()
    assert any(e[0] == "info" and "cached vocal separation" in e[1] for e in rec2.events)


def test_reporter_none_uses_console(sample_mkv, tmp_path, isolated_dirs):
    res = _run(sample_mkv, tmp_path, None, separate=False)
    assert res.output and res.output.is_file() and res.voiced == 5
