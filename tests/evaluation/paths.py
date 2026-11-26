"""Cross-platform path resolution for real-capture evaluation dumps, symbols and withheld ground truth.

Resolution order (first match wins):

1. Per-case environment variable:
   - ``COMFYTRACE_EVAL_MEMORY_<CASE_TOKEN>``
   - ``COMFYTRACE_EVAL_SYMBOLS_<CASE_TOKEN>``
   - ``COMFYTRACE_EVAL_GROUNDTRUTH_<CASE_TOKEN>``
2. Per-case entry in the JSON file named by ``COMFYTRACE_EVAL_CONFIG`` (``cases.<case_id>.*``).
3. Shared directory environment variable:
   - ``COMFYTRACE_EVAL_DUMPS_DIR`` + ``memory_filename`` (or basename of ``memory``)
   - ``COMFYTRACE_EVAL_SYMBOLS_DIR`` (+ optional ``symbols_subdir``)
   - ``COMFYTRACE_EVAL_GROUNDTRUTH_DIR`` + ``groundtruth_filename`` (or ``<case_id>.groundtruth.json`` /
     ``<case_id>.json``)
4. Shared directory in ``COMFYTRACE_EVAL_CONFIG`` (``dumps_dir``, ``symbols_dir``, ``groundtruth_dir``).
5. Platform default in the case registry (``platform_defaults.<windows|linux|darwin>.*``).
6. Legacy default fields in the case registry (``memory``, ``symbols_dir``).

Ground-truth paths are resolved strictly for post-extraction scoring and must never be supplied to extraction.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_CASES_FILE = Path(__file__).resolve().parent.parent / "fixtures" / "groundtruth_cases.json"

ENV_CASES_FILE = "COMFYTRACE_EVAL_CASES_FILE"
ENV_CONFIG = "COMFYTRACE_EVAL_CONFIG"
ENV_DUMPS_DIR = "COMFYTRACE_EVAL_DUMPS_DIR"
ENV_SYMBOLS_DIR = "COMFYTRACE_EVAL_SYMBOLS_DIR"
ENV_GROUNDTRUTH_DIR = "COMFYTRACE_EVAL_GROUNDTRUTH_DIR"


def case_env_token(case_id: str) -> str:
    """Normalizes a case id (e.g. ``gen-only-run-1``) into an env var suffix (``GEN_ONLY_RUN_1``)."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", case_id.strip()).strip("_").upper()
    if not token:
        raise ValueError(f"invalid case id for environment token: {case_id!r}")
    return token


def _platform_key(platform_name: str | None = None) -> str:
    name = (platform_name or sys.platform).lower()
    if name.startswith("win"):
        return "windows"
    if name.startswith("linux"):
        return "linux"
    if name.startswith("darwin"):
        return "darwin"
    return name


def _portable_basename(raw_path: str) -> str:
    """Extracts the final path segment regardless of whether ``raw_path`` uses POSIX or Windows separators."""
    cleaned = raw_path.strip().rstrip("/\\")
    if not cleaned:
        raise ValueError(f"empty path cannot yield a filename: {raw_path!r}")
    parts = [p for p in re.split(r"[/\\]+", cleaned) if p]
    if not parts:
        raise ValueError(f"cannot extract filename from {raw_path!r}")
    return parts[-1]


def _safe_relative_parts(relative_path: str) -> tuple[str, ...]:
    """Splits a portable relative path (``/`` or ``\\``) and rejects absolute or parent-traversal segments."""
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"relative path required, got {relative_path!r}")
    parts = PurePosixPath(normalized).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid relative path segment in {relative_path!r}")
    return parts


def _expand_path(raw: str, env: Mapping[str, str], base_dir: Path | None = None) -> Path:
    """Expands ``$VAR`` / ``%VAR%`` and ``~``; resolves relative paths against ``base_dir`` when supplied."""
    expanded = raw
    # Expand both POSIX ($VAR / ${VAR}) and Windows (%VAR%) syntax portably against the provided env mapping.
    expanded = re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([A-Za-z_][A-Za-z0-9_]*)%",
        lambda m: env.get(m.group(1) or m.group(2) or m.group(3), m.group(0)),
        expanded,
    )
    path = Path(expanded).expanduser()
    if not path.is_absolute() and base_dir is not None:
        return base_dir / path
    return path


def _load_config(env: Mapping[str, str], config_path: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    raw_cfg = str(config_path) if config_path is not None else env.get(ENV_CONFIG, "").strip()
    if not raw_cfg:
        return {}, None
    resolved = _expand_path(raw_cfg, env)
    if not resolved.is_file():
        raise FileNotFoundError(f"{ENV_CONFIG} points to a missing file: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"evaluation config must be a JSON object: {resolved}")
    return data, resolved.parent


@dataclass(frozen=True)
class ResolvedCasePaths:
    """Resolved local paths for one real-capture case and where each path came from."""

    case_id: str
    memory: Path | None
    memory_source: str
    memory_explicit: bool
    symbols_dir: Path | None
    symbols_source: str
    symbols_explicit: bool
    groundtruth: Path | None
    groundtruth_source: str
    groundtruth_explicit: bool
    requires_symbols: bool


def load_groundtruth_cases(
    cases_file: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Loads registered real-capture evaluation cases from ``cases_file`` or ``COMFYTRACE_EVAL_CASES_FILE``."""
    active_env = os.environ if env is None else env
    if cases_file is not None:
        path = cases_file
    elif active_env.get(ENV_CASES_FILE, "").strip():
        path = _expand_path(active_env[ENV_CASES_FILE].strip(), active_env)
        if not path.is_file():
            raise FileNotFoundError(f"{ENV_CASES_FILE} points to a missing file: {path}")
    else:
        path = DEFAULT_CASES_FILE
    if not path.exists():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"cases registry must contain a 'cases' list: {path}")
    return cases


def resolve_case_paths(
    case: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    config_path: Path | None = None,
) -> ResolvedCasePaths:
    """Resolves dump, Windows kernel symbols, and optional post-extraction ground-truth paths for ``case``."""
    active_env = os.environ if env is None else env
    config, config_dir = _load_config(active_env, config_path=config_path)
    case_id = str(case["id"])
    token = case_env_token(case_id)
    plat = _platform_key(platform_name)
    cases_section = config.get("cases") or {}
    if not isinstance(cases_section, dict):
        raise ValueError("config 'cases' must be a JSON object")
    case_cfg = cases_section.get(case_id) or {}
    if not isinstance(case_cfg, dict):
        raise ValueError(f"config cases.{case_id} must be a JSON object")
    raw_plat_defaults = case.get("platform_defaults") or {}
    if not isinstance(raw_plat_defaults, dict):
        raise ValueError(f"case {case_id!r} platform_defaults must be a JSON object")
    plat_defaults = raw_plat_defaults.get(plat) or {}

    memory_filename = case.get("memory_filename")
    if memory_filename is None and case.get("memory"):
        memory_filename = _portable_basename(str(case["memory"]))

    # 1. Memory dump path
    mem_env_key = f"COMFYTRACE_EVAL_MEMORY_{token}"
    memory: Path | None = None
    memory_source = "unconfigured"
    memory_explicit = False
    if active_env.get(mem_env_key, "").strip():
        memory = _expand_path(active_env[mem_env_key].strip(), active_env)
        memory_source = f"env:{mem_env_key}"
        memory_explicit = True
    elif case_cfg.get("memory"):
        memory = _expand_path(str(case_cfg["memory"]), active_env, base_dir=config_dir)
        memory_source = f"config:cases.{case_id}.memory"
        memory_explicit = True
    elif active_env.get(ENV_DUMPS_DIR, "").strip() and memory_filename:
        base = _expand_path(active_env[ENV_DUMPS_DIR].strip(), active_env)
        memory = base.joinpath(*_safe_relative_parts(str(memory_filename)))
        memory_source = f"env:{ENV_DUMPS_DIR}"
        memory_explicit = False
    elif config.get("dumps_dir") and memory_filename:
        base = _expand_path(str(config["dumps_dir"]), active_env, base_dir=config_dir)
        memory = base.joinpath(*_safe_relative_parts(str(memory_filename)))
        memory_source = "config:dumps_dir"
        memory_explicit = False
    elif plat_defaults.get("memory"):
        memory = _expand_path(str(plat_defaults["memory"]), active_env)
        memory_source = f"case:platform_defaults.{plat}.memory"
    elif case.get("memory"):
        memory = _expand_path(str(case["memory"]), active_env)
        memory_source = "case:memory"

    # 2. Windows kernel symbols directory path
    sym_env_key = f"COMFYTRACE_EVAL_SYMBOLS_{token}"
    symbols_subdir = case.get("symbols_subdir")
    symbols_dir: Path | None = None
    symbols_source = "unconfigured"
    symbols_explicit = False
    if active_env.get(sym_env_key, "").strip():
        symbols_dir = _expand_path(active_env[sym_env_key].strip(), active_env)
        symbols_source = f"env:{sym_env_key}"
        symbols_explicit = True
    elif case_cfg.get("symbols_dir"):
        symbols_dir = _expand_path(str(case_cfg["symbols_dir"]), active_env, base_dir=config_dir)
        symbols_source = f"config:cases.{case_id}.symbols_dir"
        symbols_explicit = True
    elif active_env.get(ENV_SYMBOLS_DIR, "").strip():
        base = _expand_path(active_env[ENV_SYMBOLS_DIR].strip(), active_env)
        symbols_dir = base.joinpath(*_safe_relative_parts(str(symbols_subdir))) if symbols_subdir else base
        symbols_source = f"env:{ENV_SYMBOLS_DIR}"
        symbols_explicit = True
    elif config.get("symbols_dir"):
        base = _expand_path(str(config["symbols_dir"]), active_env, base_dir=config_dir)
        symbols_dir = base.joinpath(*_safe_relative_parts(str(symbols_subdir))) if symbols_subdir else base
        symbols_source = "config:symbols_dir"
        symbols_explicit = True
    elif plat_defaults.get("symbols_dir"):
        symbols_dir = _expand_path(str(plat_defaults["symbols_dir"]), active_env)
        symbols_source = f"case:platform_defaults.{plat}.symbols_dir"
    elif case.get("symbols_dir"):
        symbols_dir = _expand_path(str(case["symbols_dir"]), active_env)
        symbols_source = "case:symbols_dir"

    # 3. Optional withheld ground-truth path (post-extraction scorer ONLY)
    gt_env_key = f"COMFYTRACE_EVAL_GROUNDTRUTH_{token}"
    gt_disabled = "groundtruth_filename" in case and case["groundtruth_filename"] is None
    gt_filename = None if gt_disabled else (case.get("groundtruth_filename") or f"{case_id}.groundtruth.json")
    groundtruth: Path | None = None
    groundtruth_source = "unconfigured"
    groundtruth_explicit = False
    if active_env.get(gt_env_key, "").strip():
        groundtruth = _expand_path(active_env[gt_env_key].strip(), active_env)
        groundtruth_source = f"env:{gt_env_key}"
        groundtruth_explicit = True
    elif case_cfg.get("groundtruth"):
        groundtruth = _expand_path(str(case_cfg["groundtruth"]), active_env, base_dir=config_dir)
        groundtruth_source = f"config:cases.{case_id}.groundtruth"
        groundtruth_explicit = True
    elif active_env.get(ENV_GROUNDTRUTH_DIR, "").strip():
        if gt_filename is None:
            groundtruth_source = "case:disabled"
        else:
            base = _expand_path(active_env[ENV_GROUNDTRUTH_DIR].strip(), active_env)
            candidate = base.joinpath(*_safe_relative_parts(str(gt_filename)))
            alt = base / f"{case_id}.json"
            groundtruth = candidate if candidate.is_file() or not alt.is_file() else alt
            groundtruth_source = f"env:{ENV_GROUNDTRUTH_DIR}"
            groundtruth_explicit = True
    elif config.get("groundtruth_dir"):
        if gt_filename is None:
            groundtruth_source = "case:disabled"
        else:
            base = _expand_path(str(config["groundtruth_dir"]), active_env, base_dir=config_dir)
            candidate = base.joinpath(*_safe_relative_parts(str(gt_filename)))
            alt = base / f"{case_id}.json"
            groundtruth = candidate if candidate.is_file() or not alt.is_file() else alt
            groundtruth_source = "config:groundtruth_dir"
            groundtruth_explicit = True

    requires_symbols = bool(case.get("requires_symbols", case.get("expected_exit_code") != 3))
    return ResolvedCasePaths(
        case_id=case_id,
        memory=memory,
        memory_source=memory_source,
        memory_explicit=memory_explicit,
        symbols_dir=symbols_dir,
        symbols_source=symbols_source,
        symbols_explicit=symbols_explicit,
        groundtruth=groundtruth,
        groundtruth_source=groundtruth_source,
        groundtruth_explicit=groundtruth_explicit,
        requires_symbols=requires_symbols,
    )
