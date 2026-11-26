"""Find or start the local ComfyUI runner through comfy-cli."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from comfytrace.exceptions import RunnerConnectionError

log = logging.getLogger(__name__)


def _run_comfy(executable: str, command: str, *options: str, timeout: int | None = 30) -> dict[str, Any]:
    structured = command != "install"
    flags = ["--json"] if structured else ["--skip-prompt"]
    env = os.environ.copy()
    # Let comfy-cli select the workspace Python even when ComfyTrace runs in an active environment.
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env["PATH"] = os.pathsep.join((str(Path(executable).parent), env.get("PATH", os.defpath)))
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [executable, "--where", "local", *flags, command, *options],
            env=env, stdin=subprocess.DEVNULL, capture_output=structured,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerConnectionError(f"comfy {command} failed: {exc}") from exc

    if not structured:
        if result.returncode:
            raise RunnerConnectionError(f"comfy install failed with exit code {result.returncode}")
        return {}

    for line in reversed(result.stdout.splitlines()):
        try:
            envelope = json.loads(line)
        except ValueError:
            continue
        if not isinstance(envelope, dict) or envelope.get("command") != command:
            continue
        data = envelope.get("data")
        if result.returncode == 0 and envelope.get("ok") is True and isinstance(data, dict):
            return data
        error = envelope.get("error")
        if isinstance(error, dict) and error.get("message"):
            raise RunnerConnectionError(f"comfy {command} failed: {error['message']}")
        break
    if "Bad git executable" in result.stderr:
        raise RunnerConnectionError("comfy-cli needs Git on PATH")
    detail = (result.stderr or result.stdout).strip()
    raise RunnerConnectionError(
        f"comfy {command} did not return a successful JSON result (exit {result.returncode}): "
        f"{detail[-1200:] or 'install comfy-cli 1.28 or newer'}"
    )


def _endpoint(value: Any) -> str:
    try:
        url = urlsplit(value) if isinstance(value, str) else None
        if url and url.scheme == "http" and url.hostname and url.port and not (url.username or url.password):
            return value.rstrip("/")
    except ValueError:
        pass
    raise RunnerConnectionError("comfy-cli did not return a usable server URL")


def resolve_runner(args: argparse.Namespace) -> str:
    """Use an explicit endpoint or let comfy-cli manage the local server."""
    runner_url = getattr(args, "runner_url", None)
    if runner_url is not None:
        return runner_url

    requested = getattr(args, "comfy_cli", None)
    if requested:
        executable = shutil.which(str(Path(requested).resolve()))
    else:
        name = "comfy.exe" if os.name == "nt" else "comfy"
        executable = None
        for directory in (sysconfig.get_path("scripts"), *os.get_exec_path()):
            candidate = Path(directory) / name
            if candidate.is_absolute() and (executable := shutil.which(str(candidate))):
                break
    if not executable:
        if requested:
            raise RunnerConnectionError(f"comfy-cli executable not found: {requested}")
        raise RunnerConnectionError(
            "Install comfytrace[runner], put comfy on PATH, or pass --comfy-cli or --runner-url"
        )

    executable = str(Path(executable).resolve())
    info = _run_comfy(executable, "env")
    server = info.get("server")
    workspace = info.get("workspace")
    if not isinstance(server, dict) or not isinstance(workspace, dict):
        raise RunnerConnectionError("comfy env did not report server and workspace status")

    if server.get("running") is True:
        endpoint = _endpoint(server.get("url"))
        log.info("Using ComfyUI at %s", endpoint)
        return endpoint

    gpu = (info.get("hardware") or {}).get("gpu") or {}
    backend = {"nvidia": "--nvidia", "amd": "--amd", "apple": "--m-series"}.get(gpu.get("vendor"), "--cpu")
    if workspace.get("exists") is not True:
        if not getattr(args, "install_comfyui", False):
            raise RunnerConnectionError("ComfyUI workspace not found. Pass --install-comfyui to install it locally.")
        log.info("Installing ComfyUI with comfy-cli (%s)", backend)
        _run_comfy(executable, "install", "--version", "latest", "--skip-manager", backend, timeout=None)

    log.info("Starting local ComfyUI with comfy-cli")
    extras = ["--listen", "127.0.0.1", "--disable-auto-launch"]
    if backend == "--cpu":
        extras.append("--cpu")
    elif backend == "--amd" and os.name == "nt":
        extras.append("--directml")
    launched = _run_comfy(executable, "launch", "--background", "--", *extras, timeout=600)
    return _endpoint(launched.get("url"))
