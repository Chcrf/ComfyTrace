"""Which CPython runtime image a process maps, its version and build flags, and why it is refused.

Windows: ``python3<minor>.dll``; the version is in the file name. Linux: a mapped ``libpython3.N<flags>.so*`` is
preferred, otherwise an ELF executable named ``python``, ``python3`` or ``python3.N``, which must export
``Py_Version``. The version comes from the module's exported ``Py_Version`` (3.11+) and, for libpython, from its
file name; the two must agree. Only a runtime with a layout registered in ``registry`` is decoded.

Rejections, in this order: ``UNSUPPORTED_PYTHON_BUILD:<flags>`` (a flagged libpython, before its export is
consulted), ``PYTHON_VERSION_UNVERIFIED`` (an executable without ``Py_Version``), ``PYTHON_VERSION_MISMATCH:
<named>!=<exported>``, ``UNSUPPORTED_PYTHON_VERSION:<major>.<minor>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from comfytrace.cpython import registry
from comfytrace.procmem import ProcessMemory, Unreadable

PYTHON_DLL = re.compile(r"python3\d+\.dll")
LIBPYTHON = re.compile(r"libpython3\.(\d+)([a-z]*)\.so(\.\d+)*")
ELF_PYTHON = re.compile(r"python(3(\.\d+)?)?")
UNSUPPORTED_PYTHON_VERSION = "UNSUPPORTED_PYTHON_VERSION"
UNSUPPORTED_PYTHON_BUILD = "UNSUPPORTED_PYTHON_BUILD"
PYTHON_VERSION_UNVERIFIED = "PYTHON_VERSION_UNVERIFIED"
PYTHON_VERSION_MISMATCH = "PYTHON_VERSION_MISMATCH"


@dataclass(frozen=True)
class Runtime:
    """The runtime image of one process. ``unsupported`` is None exactly when a layout is registered for
    ``(image_os, version, build_flags)``."""

    image_os: str
    image: Any = None
    version: tuple[int, int] | None = None
    build_flags: str = ""
    unsupported: str | None = None


def dotted(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def exported_version(mem: ProcessMemory, image: Any) -> tuple[int, int] | None:
    """(major, minor) from the module's ``Py_Version`` (PY_VERSION_HEX, exported since 3.11), or None."""
    address = mem.elf_symbol(image, "Py_Version")
    if address is None:
        return None
    try:
        value = mem.q(address)
    except Unreadable:
        return None
    if value >> 32 or value >> 24 != 3 or (value >> 4) & 0xF not in (0xA, 0xB, 0xC, 0xF):
        return None
    return 3, (value >> 16) & 0xFF


def _gate(image_os: str, image: Any, version: tuple[int, int], flags: str) -> Runtime:
    if version not in registry.supported_versions(image_os, flags):
        return Runtime(image_os, image, version, flags, f"{UNSUPPORTED_PYTHON_VERSION}:{dotted(version)}")
    return Runtime(image_os, image, version, flags)


def windows_runtime(mem: ProcessMemory) -> Runtime:
    image = mem.image(lambda name: bool(PYTHON_DLL.fullmatch(name)))
    if image is None:
        return Runtime("windows")
    return _gate("windows", image, (3, int(image.name[7:-4])), "")


def linux_runtime(mem: ProcessMemory) -> Runtime:
    lib = next((m for m in mem.images if LIBPYTHON.fullmatch(m.name)), None)
    image = lib or next((m for m in mem.images if getattr(m, "kind", "") == "elf" and ELF_PYTHON.fullmatch(m.name)),
                        None)
    if image is None:
        return Runtime("linux")
    exported = exported_version(mem, image)
    flags = ""
    if lib is not None:
        match = LIBPYTHON.fullmatch(lib.name)
        assert match is not None
        named: tuple[int, int] | None = (3, int(match[1]))
        flags = match[2]
        if flags and named not in registry.supported_versions("linux", flags):
            return Runtime("linux", image, named, flags, f"{UNSUPPORTED_PYTHON_BUILD}:{flags}")
    else:
        suffix = image.name.partition("python3.")[2]
        named = (3, int(suffix)) if suffix else None
        if exported is None:
            return Runtime("linux", image, named, flags, PYTHON_VERSION_UNVERIFIED)
    if named is not None and exported is not None and named != exported:
        return Runtime("linux", image, named, flags, f"{PYTHON_VERSION_MISMATCH}:{dotted(named)}!={dotted(exported)}")
    version = exported or named
    assert version is not None
    return _gate("linux", image, version, flags)


def detect(mem: ProcessMemory) -> Runtime:
    """The CPython runtime of ``mem``: no image (``image`` None), a refused one (``unsupported``) or one with a
    registered layout."""
    if getattr(mem, "image_os", "windows") == "linux":
        return linux_runtime(mem)
    return windows_runtime(mem)
