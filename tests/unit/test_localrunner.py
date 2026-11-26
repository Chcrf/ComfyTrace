import json
import subprocess

import pytest

from comfytrace import localrunner
from comfytrace.cli import parse_args
from comfytrace.exceptions import RunnerConnectionError


def runner_args(*options):
    return parse_args(["regenerate", "--case-dir", "case", "--all", *options])


def test_explicit_url_bypasses_comfy_cli(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Explicit URLs should not require comfy-cli")
    monkeypatch.setattr(localrunner.shutil, "which", unexpected)
    assert localrunner.resolve_runner(runner_args("--runner-url", "http://explicit:8188")) == "http://explicit:8188"


@pytest.mark.parametrize(("running", "installed", "install", "commands"), [
    (True, True, False, ["env"]),
    (False, True, False, ["env", "launch"]),
    (False, False, True, ["env", "install", "launch"]),
    (False, False, False, ["env"]),
])
def test_local_runner_lifecycle(monkeypatch, tmp_path, running, installed, install, commands):
    executable = tmp_path / "bin" / "comfy"
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "outer-venv"))
    monkeypatch.setattr(localrunner.shutil, "which", lambda *_args, **_kwargs: str(executable))
    calls = {}
    url = "http://127.0.0.1:8271"

    def comfy(argv, **kwargs):
        command = argv[4]
        calls[command] = argv[5:]
        assert argv[:3] == [str(executable), "--where", "local"]
        assert kwargs["env"]["PATH"].split(localrunner.os.pathsep)[0] == str(executable.parent)
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert "VIRTUAL_ENV" not in kwargs["env"]
        if command == "install":
            assert argv[3] == "--skip-prompt" and not kwargs["capture_output"]
            return subprocess.CompletedProcess(argv, 0)
        assert argv[3] == "--json"
        if command == "env":
            data = {"server": {"running": running, "url": url if running else None},
                    "workspace": {"exists": installed}, "hardware": {"gpu": {"vendor": "nvidia"}}}
        else:
            data = {"url": url}
        envelope = {"command": command, "ok": True, "data": data}
        return subprocess.CompletedProcess(argv, 0, "Informational text\n" + json.dumps(envelope), "")

    monkeypatch.setattr(localrunner.subprocess, "run", comfy)
    args = runner_args(*(["--install-comfyui"] if install else []))
    if running or installed or install:
        assert localrunner.resolve_runner(args) == url
    else:
        with pytest.raises(RunnerConnectionError, match="--install-comfyui"):
            localrunner.resolve_runner(args)
    assert list(calls) == commands
    if install:
        assert calls["install"] == ["--version", "latest", "--skip-manager", "--nvidia"]
    if "launch" in calls:
        assert calls["launch"] == ["--background", "--", "--listen", "127.0.0.1", "--disable-auto-launch"]


def test_comfy_launch_failure_is_reported(monkeypatch, tmp_path):
    def failure(argv, **kwargs):
        error = {"command": "launch", "ok": False, "error": {"message": "port is already in use"}}
        return subprocess.CompletedProcess(argv, 1, json.dumps(error), "")
    monkeypatch.setattr(localrunner.subprocess, "run", failure)
    with pytest.raises(RunnerConnectionError, match="port is already in use"):
        localrunner._run_comfy(str(tmp_path / "comfy"), "launch")
