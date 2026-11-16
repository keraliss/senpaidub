from __future__ import annotations

import subprocess

import pytest

from senpaidub import setup_tts
from senpaidub.errors import SenpaiDubError


def _fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENPAIDUB_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("SENPAIDUB_QWEN_PYTHON", raising=False)
    monkeypatch.setattr(setup_tts, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(setup_tts, "_IS_WINDOWS", False)
    setup_tts._import_cache.clear()
    return tmp_path


def test_error_exit_code():
    assert issubclass(setup_tts.SetupError, SenpaiDubError)
    assert setup_tts.SetupError.exit_code == 1


def test_lookup_none(env):
    assert setup_tts.find_qwen_python() is None
    assert not setup_tts.is_installed()


def test_lookup_order(env, monkeypatch):
    data_py = _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    assert setup_tts.find_qwen_python() == data_py
    repo_py = _fake_python(env / "repo" / ".venv-qwen" / "bin" / "python")
    assert setup_tts.find_qwen_python() == repo_py
    custom = _fake_python(env / "custom" / "python")
    monkeypatch.setenv("SENPAIDUB_QWEN_PYTHON", str(custom))
    assert setup_tts.find_qwen_python() == custom


def test_lookup_windows_layout(env):
    py = _fake_python(env / "repo" / ".venv-qwen" / "Scripts" / "python.exe")
    assert setup_tts.find_qwen_python() == py


def test_missing_override_falls_through(env, monkeypatch):
    monkeypatch.setenv("SENPAIDUB_QWEN_PYTHON", str(env / "nope"))
    repo_py = _fake_python(env / "repo" / ".venv-qwen" / "bin" / "python")
    assert setup_tts.find_qwen_python() == repo_py


def test_is_installed_checks_import(env, monkeypatch):
    _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(setup_tts.subprocess, "run", fake_run)
    assert setup_tts.is_installed()
    assert setup_tts.is_installed()
    assert len(calls) == 1  # cached
    assert "import qwen_tts" in calls[0]


def test_is_installed_false_when_import_fails(env, monkeypatch):
    _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    monkeypatch.setattr(
        setup_tts.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1)
    )
    assert not setup_tts.is_installed()


def test_find_base_python(env, monkeypatch):
    paths = {"python3.12": "/usr/bin/python3.12", "python3": "/usr/bin/python3"}
    monkeypatch.setattr(setup_tts.shutil, "which", lambda n: paths.get(n))
    versions = {"/usr/bin/python3.12": (3, 12), "/usr/bin/python3": (3, 14)}
    monkeypatch.setattr(setup_tts, "_version_of", lambda cmd: versions.get(cmd[0]))
    assert str(setup_tts.find_base_python()) == "/usr/bin/python3.12"


def test_find_base_python_rejects_too_new(env, monkeypatch):
    monkeypatch.setattr(
        setup_tts.shutil, "which", lambda n: "/usr/bin/python3" if n == "python3" else None
    )
    monkeypatch.setattr(setup_tts, "_version_of", lambda cmd: (3, 14))
    assert setup_tts.find_base_python() is None


class _FakeProc:
    def __init__(self, lines, code=0):
        self.stdout = iter(lines)
        self.returncode = code

    def wait(self):
        return self.returncode


def test_install_runs_steps(env, monkeypatch):
    monkeypatch.setattr(setup_tts, "find_base_python", lambda: env / "base" / "python3")
    cmds = []

    def fake_popen(cmd, **kw):
        assert kw.get("shell") is None
        cmds.append(cmd)
        if "venv" in cmd:
            _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
        return _FakeProc(["line one\n", "line two\n"])

    monkeypatch.setattr(setup_tts.subprocess, "Popen", fake_popen)
    lines: list[str] = []
    py = setup_tts.install(lines.append)
    assert py == setup_tts.qwen_env_dir() / "bin" / "python"
    assert len(cmds) == 3  # venv, pip, model preload
    assert cmds[0][1:3] == ["-m", "venv"]
    assert "pip" in cmds[1] and "qwen-tts" in cmds[1]
    assert "snapshot_download" in cmds[2][-1] and setup_tts.qwen_model() in cmds[2][-1]
    assert "line one" in lines and "line two" in lines


def test_install_skip_model_and_reuse(env, monkeypatch):
    _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    cmds = []
    monkeypatch.setattr(
        setup_tts.subprocess, "Popen", lambda cmd, **kw: cmds.append(cmd) or _FakeProc([])
    )
    setup_tts.install(lambda s: None, download_model=False)
    assert len(cmds) == 1 and "pip" in cmds[0]


def test_install_no_python(env, monkeypatch):
    monkeypatch.setattr(setup_tts, "find_base_python", lambda: None)
    with pytest.raises(setup_tts.SetupError, match="3.10"):
        setup_tts.install(lambda s: None)


def test_install_failure(env, monkeypatch):
    _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    monkeypatch.setattr(setup_tts.subprocess, "Popen", lambda cmd, **kw: _FakeProc(["boom\n"], 1))
    with pytest.raises(setup_tts.SetupError, match="failed"):
        setup_tts.install(lambda s: None)


def test_qwen_find_python_uses_lookup(env, monkeypatch):
    from senpaidub.errors import UsageError
    from senpaidub.tts import qwen

    with pytest.raises(UsageError, match="senpaidub setup qwen"):
        qwen.find_python()
    py = _fake_python(env / "repo" / ".venv-qwen" / "bin" / "python")
    with pytest.raises(UsageError, match="senpaidub setup qwen"):  # exists but can't import qwen_tts
        qwen.find_python()
    monkeypatch.setattr(setup_tts, "_can_import", lambda p: True)
    assert qwen.find_python() == py


def test_model_override(monkeypatch):
    monkeypatch.delenv("SENPAIDUB_QWEN_MODEL", raising=False)
    assert setup_tts.qwen_model() == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    monkeypatch.setenv("SENPAIDUB_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    assert setup_tts.qwen_model() == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def test_lookup_prefers_a_working_env(env, monkeypatch):
    broken = _fake_python(env / "repo" / ".venv-qwen" / "bin" / "python")
    good = _fake_python(setup_tts.qwen_env_dir() / "bin" / "python")
    monkeypatch.setattr(setup_tts, "_can_import", lambda p: p == good)
    assert setup_tts.find_qwen_python() == good
    assert setup_tts.is_installed()
    monkeypatch.setattr(setup_tts, "_can_import", lambda p: False)
    assert setup_tts.find_qwen_python() == broken  # none works: the first one, for messages
    assert not setup_tts.is_installed()
