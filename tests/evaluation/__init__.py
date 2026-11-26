"""Portable real-capture path resolution and post-extraction evaluation scoring."""

from __future__ import annotations

from typing import Any

from tests.evaluation.paths import (
    DEFAULT_CASES_FILE,
    ENV_CASES_FILE,
    ENV_CONFIG,
    ENV_DUMPS_DIR,
    ENV_GROUNDTRUTH_DIR,
    ENV_SYMBOLS_DIR,
    ResolvedCasePaths,
    case_env_token,
    load_groundtruth_cases,
    resolve_case_paths,
)

__all__ = [
    "DEFAULT_CASES_FILE",
    "ENV_CASES_FILE",
    "ENV_CONFIG",
    "ENV_DUMPS_DIR",
    "ENV_GROUNDTRUTH_DIR",
    "ENV_SYMBOLS_DIR",
    "ResolvedCasePaths",
    "SCORER_SCHEMA_VERSION",
    "case_env_token",
    "format_summary",
    "load_groundtruth_cases",
    "resolve_case_paths",
    "score_case",
]


def __getattr__(name: str) -> Any:
    if name in ("SCORER_SCHEMA_VERSION", "format_summary", "score_case"):
        from tests.evaluation import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
