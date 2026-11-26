"""The CPython layouts ComfyTrace can decode, discovered from ``profiles.vNNN`` modules.

Each profile module exports ``LAYOUTS``, a tuple of ``CPythonLayout``. Every ``(image_os, (major, minor),
build_flags)`` key of every layout is registered once; a key claimed by two modules is a packaging error and is
refused with both module names. Adding a runtime is adding a module: no consumer changes.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
import re
from collections.abc import Mapping
from types import MappingProxyType

from comfytrace.cpython.base import CPythonLayout, LayoutKey

PROFILES_PACKAGE = "comfytrace.cpython.profiles"
PROFILE_MODULE = re.compile(r"v[0-9]+")


def discover(package: str) -> dict[LayoutKey, CPythonLayout]:
    """Imports every ``v<digits>`` module of ``package`` and indexes its ``LAYOUTS`` by key (uncached).
    Raises: ValueError when two modules register the same key; TypeError when ``LAYOUTS`` is malformed."""
    root = importlib.import_module(package)
    index: dict[LayoutKey, CPythonLayout] = {}
    owners: dict[LayoutKey, str] = {}
    names = sorted(info.name for info in pkgutil.iter_modules(root.__path__) if PROFILE_MODULE.fullmatch(info.name))
    for name in names:
        module = importlib.import_module(f"{package}.{name}")
        layouts = getattr(module, "LAYOUTS", None)
        if not isinstance(layouts, tuple) or not all(isinstance(layout, CPythonLayout) for layout in layouts):
            raise TypeError(f"{module.__name__}.LAYOUTS must be a tuple of CPythonLayout")
        for layout in layouts:
            for key in layout.keys:
                if key in owners:
                    raise ValueError(f"CPython layout {key!r} is registered by both {owners[key]} and "
                                     f"{module.__name__}")
                owners[key] = module.__name__
                index[key] = layout
    return index


@functools.cache
def _index(package: str) -> Mapping[LayoutKey, CPythonLayout]:
    return MappingProxyType(discover(package))


def layouts(package: str | None = None) -> Mapping[LayoutKey, CPythonLayout]:
    """The cached key -> layout index of ``package`` (default: the bundled profiles)."""
    return _index(package or PROFILES_PACKAGE)


def clear_cache() -> None:
    _index.cache_clear()


def resolve_layout(image_os: str, version: tuple[int, int], build_flags: str = "",
                   package: str | None = None) -> CPythonLayout | None:
    """The layout registered for this runtime, or None when none is."""
    return layouts(package).get((image_os, tuple(version), build_flags))  # type: ignore[arg-type]


def supported_versions(image_os: str, build_flags: str = "", package: str | None = None) -> frozenset[tuple[int, int]]:
    return frozenset(v for os_, v, flags in layouts(package) if os_ == image_os and flags == build_flags)
