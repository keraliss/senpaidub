from __future__ import annotations

import asyncio
import queue
import shutil
import threading
import time
from pathlib import Path

import pytest

from senpaidub.server import jobs as jobs_mod
from senpaidub.server import worker
from senpaidub.server.jobs import JobManager, JobStateError, _group_progress

RESULT = {"voiced": 3, "sped_up": 1, "overflowing": 0, "voice_mismatches": 0, "voice_fixed": 0,
          "tts_cached": 0, "tts_new": 3, "skipped_reason": None}  # fmt: skip


def wait_for(cond, timeout: float = 5.0, what: str = "condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class FakeHandle:
    def __init__(self, spec: dict, script) -> None:
        self.spec = spec
        self.q: queue.Queue = queue.Queue()
        self.killed = threading.Event()
        self.release = threading.Event()
        self._t = threading.Thread(target=script, args=(self,), daemon=True)
        self._t.start()

    def is_alive(self) -> bool:
        return self._t.is_alive()

    def kill(self) -> None:
        self.killed.set()

    def join(self, timeout=None) -> None:
        self._t.join(timeout)

    @property
    def exitcode(self):
        return -9 if self.killed.is_set() else 0


class FakeRunner:
    """Runs a scripted 'worker' in a thread; `script` can be swapped between runs."""

    def __init__(self, script) -> None:
        self.script = script
        self.handles: list[FakeHandle] = []

    def __call__(self, spec: dict) -> FakeHandle:
        h = FakeHandle(spec, self.script)
        self.handles.append(h)
        return h


def ok_script(h: FakeHandle) -> None:
    out = str(Path(h.spec["input"]).with_name("x [EN dub].mkv"))
    h.q.put({"t": "stage", "key": "subtitles", "label": "Loading subtitles", "total": None, "unit": "none",
             "group": "subtitles"})  # fmt: skip
    h.q.put({"t": "stage_end", "key": "subtitles"})
    h.q.put({"t": "info", "msg": "Subtitles: 3 spoken lines"})
    h.q.put({"t": "stage", "key": "tts", "label": "Voicing", "total": 3, "unit": "items", "group": "voice"})
    h.q.put({"t": "progress", "key": "tts", "completed": 3, "total": 3})
    h.q.put({"t": "stage_end", "key": "tts"})
    h.q.put({"t": "warn", "msg": "0:01 overflows its slot"})
    h.q.put({"t": "result", "result": {**RESULT, "output": out}})


def blocking_script(h: FakeHandle) -> None:
    h.q.put({"t": "stage", "key": "tts", "label": "Voicing", "total": 10, "unit": "items", "group": "voice"})
    h.q.put({"t": "progress", "key": "tts", "completed": 5, "total": 10})
    while not h.killed.is_set():
        if h.release.wait(0.01):
            h.q.put({"t": "result", "result": {**RESULT, "output": "/tmp/out.mkv"}})
            return


def error_script(h: FakeHandle) -> None:
    h.q.put({"t": "error", "message": "No dialogue lines", "exit_code": 2})


def crash_script(h: FakeHandle) -> None:
    h.q.put({"t": "stage", "key": "extract", "label": "Extracting", "total": None, "unit": "none",
             "group": "separate"})  # fmt: skip


@pytest.fixture
def make_manager(tmp_path):
    made: list[JobManager] = []

    def make(script=ok_script, db: Path | None = None) -> tuple[JobManager, FakeRunner]:
        runner = FakeRunner(script)
        m = JobManager(db_path=db or tmp_path / "web.db", library=tmp_path / "lib", runner=runner)
        made.append(m)
        return m, runner

    yield make
    for m in made:
        m.stop()


def items(n: int, tmp_path: Path) -> list[dict]:
    return [{"input": str(tmp_path / f"ep{i:02d}.mkv"), "sub_file": None, "sub_track": None, "use_ocr": False,
             "name": None} for i in range(1, n + 1)]  # fmt: skip


def status(m: JobManager, job_id: str) -> str:
    return m.get_job(job_id)["status"]


def test_successful_job(make_manager, tmp_path):
    m, _ = make_manager()
    m.start()
    b = m.create_batch(items(1, tmp_path), "edge")
    assert b["status"] == "queued" and b["counts"]["waiting"] == 1
    jid = b["jobs"][0]["id"]
    wait_for(lambda: status(m, jid) == "done", what="done")
    j = m.get_job(jid)
    assert j["progress"] == 1.0 and j["eta_s"] is None and j["elapsed_s"] >= 0
    assert j["output"].endswith("x [EN dub].mkv")
    assert j["result"]["voiced"] == 3 and "output" not in j["result"]
    assert j["group"] == "voice" and j["stage_key"] == "tts" and j["stage_done"] == 3
    assert [e["level"] for e in j["log"]] == ["info", "warn"]
    assert j["name"] == "ep01.mkv" and j["tts"] == "edge"
    bb = m.get_batch(b["id"])
    assert bb["status"] == "done" and bb["counts"]["done"] == 1 and bb["progress"] == 1.0
    assert [h["id"] for h in m.history()] == [jid]
    assert m.list_batches()[0]["id"] == b["id"] and "jobs" not in m.list_batches()[0]
    assert str(tmp_path / "ep01.mkv") in m.known_paths()


def test_skipped_failed_and_retry(make_manager, tmp_path):
    m, runner = make_manager(error_script)
    m.start()
    jid = m.create_batch(items(1, tmp_path), "edge")["jobs"][0]["id"]
    wait_for(lambda: status(m, jid) == "failed")
    j = m.get_job(jid)
    assert j["error"] == "No dialogue lines" and j["exit_code"] == 2
    with pytest.raises(JobStateError):
        m.cancel_job(jid)
    runner.script = lambda h: h.q.put({"t": "result", "result": {**RESULT, "skipped_reason": "output exists",
                                                                  "output": "/x.mkv"}})  # fmt: skip
    assert m.retry_job(jid)["status"] in ("waiting", "running")
    wait_for(lambda: status(m, jid) == "skipped")
    assert m.get_job(jid)["error"] is None
    with pytest.raises(JobStateError):
        m.retry_job(jid)
    assert m.history() == []


def test_worker_crash_marks_failed(make_manager, tmp_path):
    m, _ = make_manager(crash_script)
    m.start()
    jid = m.create_batch(items(1, tmp_path), "edge")["jobs"][0]["id"]
    wait_for(lambda: status(m, jid) == "failed")
    assert "stopped unexpectedly" in m.get_job(jid)["error"]


def test_cancel_running_and_waiting(make_manager, tmp_path):
    m, runner = make_manager(blocking_script)
    m.start()
    b = m.create_batch(items(2, tmp_path), "edge")
    j1, j2 = (j["id"] for j in b["jobs"])
    wait_for(lambda: status(m, j1) == "running" and m.get_job(j1)["stage_done"] == 5)
    assert status(m, j2) == "waiting"
    running = m.get_job(j1)
    assert running["group"] == "voice" and 0.25 < running["progress"] < 0.8
    assert m.cancel_job(j2)["status"] == "cancelled"
    assert m.cancel_job(j1)["status"] == "cancelled"
    assert runner.handles[0].killed.is_set()
    wait_for(lambda: not m._handles, what="reader finished")
    assert status(m, j1) == "cancelled" and status(m, j2) == "cancelled"
    assert m.get_batch(b["id"])["status"] == "done"
    assert len(runner.handles) == 1  # the cancelled waiting job never started
    runner.script = ok_script
    m.retry_job(j1)
    wait_for(lambda: status(m, j1) == "done")


def test_parallel_two(make_manager, tmp_path):
    m, runner = make_manager(blocking_script)
    m.start()
    b = m.create_batch(items(3, tmp_path), "edge", parallel=2)
    ids = [j["id"] for j in b["jobs"]]
    wait_for(lambda: [status(m, i) for i in ids] == ["running", "running", "waiting"])
    time.sleep(3 * jobs_mod.TICK)
    assert status(m, ids[2]) == "waiting" and m.get_batch(b["id"])["status"] == "running"
    runner.handles[0].release.set()
    wait_for(lambda: status(m, ids[0]) == "done" and status(m, ids[2]) == "running")
    for h in runner.handles:
        h.release.set()
    wait_for(lambda: m.get_batch(b["id"])["counts"]["done"] == 3)


def test_pause_resume_and_remove(make_manager, tmp_path):
    m, runner = make_manager(blocking_script)
    m.start()
    b = m.create_batch(items(3, tmp_path), "edge")
    ids = [j["id"] for j in b["jobs"]]
    wait_for(lambda: status(m, ids[0]) == "running")
    assert m.pause_batch(b["id"])["status"] == "paused"
    with pytest.raises(JobStateError):
        m.remove_job(ids[0])
    m.remove_job(ids[2])
    with pytest.raises(KeyError):
        m.get_job(ids[2])
    runner.handles[0].release.set()
    wait_for(lambda: status(m, ids[0]) == "done")
    time.sleep(3 * jobs_mod.TICK)
    assert status(m, ids[1]) == "waiting" and m.get_batch(b["id"])["status"] == "paused"
    m.resume_batch(b["id"])
    wait_for(lambda: status(m, ids[1]) == "running")
    runner.handles[1].release.set()
    wait_for(lambda: m.get_batch(b["id"])["status"] == "done")
    assert len(m.get_batch(b["id"])["jobs"]) == 2


def test_restart_recovery(make_manager, tmp_path):
    db = tmp_path / "web.db"
    m1, r1 = make_manager(blocking_script, db=db)
    m1.start()
    b = m1.create_batch(items(1, tmp_path), "qwen", series_name="Show")
    jid = b["jobs"][0]["id"]
    wait_for(lambda: status(m1, jid) == "running")
    # A second manager on the same database sees the job as interrupted (as after a crash).
    m2, _ = make_manager(blocking_script, db=db)
    j = m2.get_job(jid)
    assert j["status"] == "waiting" and j["progress"] == 0.0 and j["started_at"] is None
    bb = m2.get_batch(b["id"])
    assert bb["series_name"] == "Show" and bb["output_dir"] == str(tmp_path / "lib" / "Show")
    # A clean stop also puts running jobs back to waiting.
    m1.stop()
    assert r1.handles[0].killed.is_set()
    assert status(m1, jid) == "waiting"
    m3, _ = make_manager(blocking_script, db=db)
    assert status(m3, jid) == "waiting"


def test_subscribe_events(make_manager, tmp_path):
    m, _ = make_manager()
    m.start()

    async def run() -> list[dict]:
        q = m.subscribe()
        await asyncio.to_thread(m.create_batch, items(1, tmp_path), "edge")
        seen = []
        while True:
            ev = await asyncio.wait_for(q.get(), 5)
            seen.append(ev)
            if ev["type"] == "job" and ev["job"]["status"] == "done":
                break
        m.unsubscribe(q)
        return seen

    seen = asyncio.run(run())
    assert seen[0]["type"] == "batch" and seen[0]["batch"]["status"] == "queued"
    assert "jobs" not in seen[0]["batch"]
    assert any(e["type"] == "job" and e["job"]["status"] == "running" for e in seen)
    assert seen[-1]["job"]["progress"] == 1.0


def test_slow_subscriber_drops_oldest():
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    for i in range(5):
        jobs_mod._offer(q, {"n": i})
    assert [q.get_nowait()["n"], q.get_nowait()["n"]] == [3, 4]


def test_create_batch_validation(make_manager, tmp_path):
    m, _ = make_manager()
    with pytest.raises(ValueError):
        m.create_batch([], "edge")
    with pytest.raises(ValueError):
        m.create_batch(items(1, tmp_path), "nope")
    assert m.create_batch(items(1, tmp_path), "edge")["output_dir"] is None
    with pytest.raises(KeyError):
        m.get_batch("missing")


def test_group_progress_math():
    assert _group_progress(None, 0, 0, False) == 0.0
    assert _group_progress("subtitles", None, None, True) == pytest.approx(0.05)
    assert _group_progress("voice", 5, 10, False) == pytest.approx(0.25 + 0.275)
    assert _group_progress("voice", 5, None, False) == pytest.approx(0.25)
    assert _group_progress("mux", None, None, True) == pytest.approx(1.0)


# ------------------------------------------------------------------ worker


def test_queue_reporter_throttles_and_closes():
    q: queue.Queue = queue.Queue()
    r = worker.QueueReporter(q)
    with r.stage("tts", "Voicing", 1000, "items") as st:
        for _ in range(1000):
            st.advance()
    with pytest.raises(RuntimeError), r.stage("separate", "Separating", None, "seconds") as st:
        st.update(10, 100)
        raise RuntimeError("boom")
    r.info("hello", style="dim")
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    tts = [m for m in msgs if m.get("key") == "tts"]
    assert tts[0] == {"t": "stage", "key": "tts", "label": "Voicing", "total": 1000, "unit": "items", "group": "voice"}
    progress = [m for m in tts if m["t"] == "progress"]
    assert 1 <= len(progress) <= 5 and progress[-1]["completed"] == 1000
    assert tts[-1] == {"t": "stage_end", "key": "tts"}
    sep = [m for m in msgs if m.get("key") == "separate"]
    assert [m["t"] for m in sep] == ["stage", "progress", "stage_end"] and sep[1]["total"] == 100
    assert msgs[-1] == {"t": "info", "msg": "hello"}


def test_queue_reporter_dedupes_logged_warnings():
    q: queue.Queue = queue.Queue()
    r = worker.QueueReporter(q)
    r.log_warning("same")
    r.warn("same")
    r.warn("other")
    assert [q.get_nowait()["msg"] for _ in range(q.qsize())] == ["same", "other"]


def test_build_options(tmp_path, isolated_dirs):
    lib = tmp_path / "lib"
    o = worker.build_options({"input": str(tmp_path / "a.mkv"), "tts": "edge", "series_name": "My Show",
                              "library": str(lib)})  # fmt: skip
    assert o.output == lib / "My Show" and o.collect_originals and o.clean_cache and o.tts == "edge"
    o = worker.build_options({"input": str(tmp_path / "a.mkv"), "tts": "mock", "library": str(lib),
                              "options": {"separate": False, "keep_cache": True, "bogus": 1}})  # fmt: skip
    assert o.output is None and not o.collect_originals and not o.separate and not o.clean_cache
    upload = isolated_dirs / "cache" / "uploads" / "abc" / "ep.mkv"
    o = worker.build_options({"input": str(upload), "tts": "edge", "library": str(lib), "force": True})
    assert o.output == lib / "Unsorted" and o.force
    assert worker.partial_output({"input": str(upload), "tts": "edge", "library": str(lib)}) == \
        lib / "Unsorted" / ".ep [EN dub].partial.mkv"


def test_group_progress_moves_within_a_group():
    after_tts = _group_progress("voice", None, None, True, "tts")
    mid_check = _group_progress("voice", 5, 10, False, "voicecheck")
    assert after_tts < mid_check < _group_progress("voice", None, None, True, "voicecheck")
    assert _group_progress("voice", None, None, True, "voicecheck") == pytest.approx(0.80)
    fit = _group_progress("mix", None, None, True, "fit")
    mix = _group_progress("mix", 1, 2, False, "mix")
    enc = _group_progress("mix", 0, None, False, "encode")
    assert fit < mix < enc < 0.95
    assert _group_progress("mix", None, None, True, "encode") == pytest.approx(0.95)


def test_error_during_shutdown_requeues(make_manager, tmp_path):
    def interrupted(h: FakeHandle) -> None:  # a worker hit by the terminal's Ctrl+C
        h.q.put({"t": "stage", "key": "tts", "label": "Voicing", "total": 10, "unit": "items", "group": "voice"})
        h.release.wait(5)
        h.q.put({"t": "error", "message": "KeyboardInterrupt", "exit_code": 3})

    m, runner = make_manager(interrupted)
    m.start()
    jid = m.create_batch(items(1, tmp_path), "edge")["jobs"][0]["id"]
    wait_for(lambda: m.get_job(jid)["stage_key"] == "tts", what="running")
    with m._lock:
        m._stopping = True
    runner.handles[0].release.set()
    wait_for(lambda: not m._readers, what="reader done")
    assert status(m, jid) == "waiting"


def _staged(isolated_dirs, name="ep.mkv") -> Path:
    import uuid

    p = isolated_dirs / "cache" / "uploads" / uuid.uuid4().hex / name
    p.parent.mkdir(parents=True)
    p.write_bytes(b"video")
    return p


def test_upload_deleted_after_done_kept_after_failure(make_manager, tmp_path, isolated_dirs):
    m, runner = make_manager(ok_script)
    m.start()
    done_in = _staged(isolated_dirs)
    j1 = m.create_batch([{"input": str(done_in)}], "edge")["jobs"][0]["id"]
    wait_for(lambda: status(m, j1) == "done", what="done")
    wait_for(lambda: not done_in.parent.exists(), what="upload deleted")

    runner.script = error_script
    failed_in = _staged(isolated_dirs)
    j2 = m.create_batch([{"input": str(failed_in)}], "edge")["jobs"][0]["id"]
    wait_for(lambda: status(m, j2) == "failed", what="failed")
    time.sleep(0.2)
    assert failed_in.is_file()  # kept for Retry


def test_upload_shared_by_waiting_job_is_kept(make_manager, tmp_path, isolated_dirs):
    m, runner = make_manager(ok_script)
    staged = _staged(isolated_dirs)
    m.create_batch([{"input": str(staged)}], "edge")
    b2 = m.create_batch([{"input": str(staged)}], "edge")
    m.pause_batch(b2["id"])
    m.start()
    wait_for(lambda: m.get_batch(b2["id"]) and sum(1 for h in runner.handles) == 1 and
             not m._readers, what="first done")  # fmt: skip
    time.sleep(0.2)
    assert staged.is_file()  # the paused batch still needs it
    m.remove_job(b2["jobs"][0]["id"])
    assert not staged.parent.exists()


def test_start_sweeps_stale_uploads(make_manager, tmp_path, isolated_dirs):
    import os

    abandoned = _staged(isolated_dirs)
    os.utime(abandoned.parent, (0, 0))
    m, _ = make_manager()
    m.start()
    assert not abandoned.parent.exists()


def test_dotted_series_name_is_a_folder(tmp_path, isolated_dirs):
    lib = tmp_path / "lib"
    spec = {"input": str(tmp_path / "Dr. Stone - 01.mkv"), "tts": "edge", "series_name": "Dr. Stone",
            "library": str(lib)}  # fmt: skip
    opts, folder = worker._plan(spec)
    assert folder and opts.output == lib / "Dr. Stone"
    out = worker.expected_output(spec)
    assert out is not None and out.parent == lib / "Dr. Stone" and out.name.endswith("[EN dub].mkv")
    assert worker.partial_output(spec).parent == lib / "Dr. Stone"


def test_build_options_validates_config_and_use_ocr(tmp_path, isolated_dirs):
    from senpaidub.errors import UsageError

    base = {"input": str(tmp_path / "a.mkv"), "tts": "edge"}
    with pytest.raises(UsageError, match="codec"):
        worker.build_options({**base, "options": {"codec": "mp3"}})
    with pytest.raises(UsageError, match="max-speedup"):
        worker.build_options({**base, "options": {"max_speedup": 0.5}})
    assert worker.build_options({**base, "use_ocr": True}).force_ocr
    assert not worker.build_options(base).force_ocr


def test_job_main_detaches_from_ctrl_c(monkeypatch):
    calls = []
    monkeypatch.setattr(worker.os, "setsid", lambda: calls.append("setsid"), raising=False)
    monkeypatch.setattr(worker.signal, "signal", lambda sig, h: calls.append((sig, h)))
    monkeypatch.setattr(worker, "run_job", lambda spec, q: calls.append("run"))
    worker.job_main({}, None)
    assert calls == ["setsid", (worker.signal.SIGINT, worker.signal.SIG_IGN), "run"]


def test_run_job_reports_usage_error(tmp_path):
    q: queue.Queue = queue.Queue()
    worker.run_job({"input": str(tmp_path / "missing.mkv"), "tts": "mock"}, q)
    msgs = [q.get_nowait() for _ in range(q.qsize())]
    assert msgs[-1]["t"] == "error" and msgs[-1]["exit_code"] == 1 and "not found" in msgs[-1]["message"]


def test_real_worker_end_to_end(sample_mkv, tmp_path):
    video = tmp_path / "Episode 01.mkv"
    shutil.copy(sample_mkv, video)
    m = JobManager(db_path=tmp_path / "web.db", library=tmp_path / "lib")
    m.start()
    try:
        b = m.create_batch([{"input": str(video)}], "mock", options={"separate": False})
        jid = b["jobs"][0]["id"]
        wait_for(lambda: m.get_job(jid)["status"] in jobs_mod.TERMINAL, timeout=120, what="real job")
        j = m.get_job(jid)
        assert j["status"] == "done", j["error"]
        assert Path(j["output"]).is_file() and j["output"].endswith("[EN dub].mkv")
        assert j["result"]["voiced"] > 0 and j["progress"] == 1.0
        assert any("spoken lines" in e["msg"] for e in j["log"])
    finally:
        m.stop()


def test_force_ocr_reads_burned_in_subs_despite_tracks(sample_mkv, tmp_path, monkeypatch):
    from senpaidub import pipeline
    from senpaidub.errors import UsageError
    from senpaidub.probe import probe
    from senpaidub.report import NullReporter

    srt = tmp_path / "ocr.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nRead off the picture.\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(pipeline, "_burned_in_subtitles", lambda *a: calls.append(a) or srt)
    monkeypatch.setattr(pipeline.hardsubs, "available", lambda: True)
    info = probe(sample_mkv)
    assert info.subtitles  # the video has text tracks, OCR still wins when asked for
    opts = pipeline.Options(tts="mock", force_ocr=True)
    result, source, ocr_file = pipeline.load_dialogue(sample_mkv, info, opts, tmp_path, reporter=NullReporter())
    assert calls and ocr_file == srt and "OCR" in source
    assert [line.text for line in result.lines] == ["Read off the picture."]

    _, source, _ = pipeline.load_dialogue(sample_mkv, info, pipeline.Options(tts="mock"), tmp_path,
                                          reporter=NullReporter())  # fmt: skip
    assert source.startswith("stream #") and len(calls) == 1

    monkeypatch.setattr(pipeline.hardsubs, "available", lambda: False)
    with pytest.raises(UsageError, match="OCR"):
        pipeline.load_dialogue(sample_mkv, info, opts, tmp_path, reporter=NullReporter())
