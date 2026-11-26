"""CPython layout profiles: registry discovery, runtime resolution, split-dict dispatch, and a test-only profile
whose shifted offsets must drive every queue, history, cache and pixel scan."""

from __future__ import annotations

import dataclasses
import importlib
import struct
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from comfytrace import execcache, pyheap
from comfytrace.cpython import registry
from comfytrace.cpython.base import CPythonLayout
from comfytrace.procmem import MappedImage, ProcessMemory
from comfytrace.torchmem import DTYPE_OFFSET, NDIM_OFFSET, NUMEL_OFFSET, SIZES_OFFSET
from tests.synthetic import api_prompt
from tests.unit.test_execcache import signature
from tests.unit.test_linux_runtime_modules import LIB, linux_memory, python_module
from tests.unit.test_memory_recovery import PROMPT, PROMPT_ID, WORKFLOW, FakePython, Heap, _memory, _smooth_image

MINUS_ONE = (1 << 64) - 1


# -- helpers ----------------------------------------------------------------------------------------------------


@pytest.fixture
def profile_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Writes an isolated profile package under ``tmp_path`` and returns (package name, writer)."""
    name = f"ct_profiles_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str, source: str) -> None:
        (root / f"{module}.py").write_text(textwrap.dedent(source))
        importlib.invalidate_caches()
        registry.clear_cache()

    yield name, write
    registry.clear_cache()
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[module]


def windows_memory(py: FakePython, dll_name: str, extra: dict[int, bytes] | None = None) -> ProcessMemory:
    blobs = {py.dll.base: bytes(py.dll.buf), py.heap.base: bytes(py.heap.buf), **(extra or {})}
    image = MappedImage(dll_name, f"C:/x/{dll_name}", py.dll.base, py.dll.base + len(py.dll.buf))
    return _memory(blobs, [image])


def spy_chunks(monkeypatch: pytest.MonkeyPatch, mem: ProcessMemory) -> list[tuple[int, int | None]]:
    calls: list[tuple[int, int | None]] = []
    original = mem.chunks

    def chunks(max_bytes: int = 16 << 20, lo: int = 0, hi: int | None = None):
        calls.append((lo, hi))
        return original(max_bytes, lo, hi)

    monkeypatch.setattr(mem, "chunks", chunks)
    return calls


# -- 1. registry ------------------------------------------------------------------------------------------------


def test_registry_discovers_profiles_and_rejects_duplicate_keys(profile_package):
    package, write = profile_package
    write("v399", """
        from comfytrace.cpython.base import CPythonLayout
        LAYOUTS = (CPythonLayout(keys=(("windows", (3, 99), ""), ("linux", (3, 99), "t")), split_values_offset=24),)
    """)
    # Neither a helper nor a non-``v<digits>`` name is a profile, even with a colliding LAYOUTS.
    write("helpers", """
        from comfytrace.cpython.base import CPythonLayout
        LAYOUTS = (CPythonLayout(keys=(("windows", (3, 99), ""),)),)
    """)
    write("v399_draft", """
        from comfytrace.cpython.base import CPythonLayout
        LAYOUTS = (CPythonLayout(keys=(("windows", (3, 99), ""),)),)
    """)

    layout = registry.resolve_layout("windows", (3, 99), package=package)
    assert layout is not None and layout.split_values_offset == 24
    assert registry.resolve_layout("linux", (3, 99), "t", package=package) is layout
    assert registry.resolve_layout("linux", (3, 99), package=package) is None, "build flags are part of the key"
    assert registry.supported_versions("linux", "t", package=package) == {(3, 99)}
    assert registry.supported_versions("linux", package=package) == frozenset()

    # The bundled profiles are unaffected, and pyheap's version sets are derived from them.
    assert set(registry.layouts()) == {("windows", (3, 12), ""), ("linux", (3, 12), ""), ("windows", (3, 13), "")}
    assert pyheap.WINDOWS_PYTHON_VERSIONS == registry.supported_versions("windows") == {(3, 12), (3, 13)}
    assert pyheap.LINUX_PYTHON_VERSIONS == registry.supported_versions("linux") == {(3, 12)}
    assert registry.resolve_layout("windows", (3, 13)).split_values_offset == 8

    write("v400", """
        from comfytrace.cpython.base import CPythonLayout
        LAYOUTS = (CPythonLayout(keys=(("windows", (3, 99), ""),)),)
    """)
    with pytest.raises(ValueError) as error:
        registry.layouts(package)
    assert f"{package}.v399" in str(error.value) and f"{package}.v400" in str(error.value)
    assert "('windows', (3, 99), '')" in str(error.value)


# -- 2. runtime resolution --------------------------------------------------------------------------------------

V312, V313, V311 = 0x030C03F0, 0x030D01F0, 0x030B04F0

RUNTIME_MATRIX = [
    # (image os, file, exported Py_Version, reason, reported version)
    ("windows", "python312.dll", None, None, "3.12"),
    ("windows", "python313.dll", None, None, "3.13"),
    ("linux", LIB, V312, None, "3.12"),
    ("windows", "python311.dll", None, "UNSUPPORTED_PYTHON_VERSION:3.11", "3.11"),
    ("linux", "/usr/lib/libpython3.13.so.1.0", V313, "UNSUPPORTED_PYTHON_VERSION:3.13", "3.13"),  # Windows-only
    ("linux", "/usr/lib/libpython3.12d.so.1.0", V313, "UNSUPPORTED_PYTHON_BUILD:d", "3.12"),  # flags win
    ("linux", LIB, V313, "PYTHON_VERSION_MISMATCH:3.12!=3.13", "3.12"),
    ("linux", LIB, None, None, "3.12"),  # a library is named by its file
    ("linux", "/usr/lib/libpython3.11.so.1.0", None, "UNSUPPORTED_PYTHON_VERSION:3.11", "3.11"),
    ("linux", "/usr/bin/python3.12", None, "PYTHON_VERSION_UNVERIFIED", "3.12"),
    ("linux", "/usr/bin/python3", None, "PYTHON_VERSION_UNVERIFIED", None),
    ("linux", "/usr/bin/python3", V311, "UNSUPPORTED_PYTHON_VERSION:3.11", "3.11"),
    ("linux", "/usr/bin/python3", V312, None, "3.12"),
]


@pytest.mark.parametrize("image_os,path,exported,reason,version", RUNTIME_MATRIX)
def test_runtime_resolution_matrix(monkeypatch, image_os, path, exported, reason, version):
    py = FakePython()
    py.obj((0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}}, ("9",)))
    if image_os == "linux":
        python_module(py, exported)
        mem = linux_memory(py, path)
    else:
        mem = windows_memory(py, path)
    calls = spy_chunks(monkeypatch, mem)
    heap = pyheap.CPythonHeap(mem)
    result = pyheap.extract(mem, scan_pixels=False)

    assert (result.reason, result.python_version) == (reason, version)
    assert (heap.unsupported, pyheap._dotted(heap.version) if heap.version else None) == (reason, version)
    dll = (heap.dll.base, heap.dll.end)
    type_page_reads = [c for c in calls if c[1] is not None and dll[0] <= c[0] < dll[1]]
    if reason is None:
        major_minor = tuple(int(p) for p in version.split("."))
        assert heap.layout is registry.resolve_layout(image_os, major_minor)
        assert result.status == "AVAILABLE" and [i.prompt for i in result.items] == [PROMPT]
        assert type_page_reads, "a supported runtime's type pages are read"
    else:
        assert result.status == "UNAVAILABLE" and result.items == [] and not heap.available
        assert heap.layout is None and heap.types == {} and heap.readable_dll_bytes == 0
        assert result.type_objects == {} and result.readable_dll_bytes == 0
        assert type_page_reads == [], "a refused runtime's type pages are never scanned"
    if image_os == "linux":
        assert pyheap.linux_python(mem)[1:] == (heap.version, reason)


# -- 3. split values --------------------------------------------------------------------------------------------


def _split_dict(py: FakePython, word0: int, word8: int) -> int:
    """{"seed": ?} as a split table, written by hand: PyDictValues holds ``word0`` at +0 and ``word8`` at +8."""
    h = py.heap
    key = py.obj("seed")
    keys = h.alloc(32 + 8 + 16)
    struct.pack_into("<BBB", h.buf, keys + 8 - h.base, 3, 3, 2)  # log2 size 3, 8 index bytes, DICT_KEYS_SPLIT
    h.q(keys + 24, 1)  # dk_nentries
    h.q(keys + 32 + 8, key)  # the only entry: (me_key, me_value unused)
    values = h.alloc(16)
    h.q(values, word0)
    h.q(values + 8, word8)
    obj = h.alloc(48)
    h.q(obj + 8, py.types["dict"])
    h.q(obj + 16, 1)
    h.q(obj + 32, keys)
    h.q(obj + 40, values)
    return obj


@pytest.mark.parametrize("image_os,name,offset,expected", [
    ("windows", "python312.dll", 0, 7),
    ("linux", LIB, 0, 7),
    ("windows", "python313.dll", 8, 20),
])
def test_split_values_dispatch(image_os, name, offset, expected):
    py = FakePython()
    obj = _split_dict(py, py.obj(7), py.obj(20))
    if image_os == "linux":
        python_module(py, V312)
        mem = linux_memory(py, name)
    else:
        mem = windows_memory(py, name)
    heap = pyheap.CPythonHeap(mem)
    assert heap.layout is not None and heap.layout.split_values_offset == heap.split_values_offset == offset
    assert heap.decode(obj) == {"seed": expected}


# -- 4. a shifted test-only profile -----------------------------------------------------------------------------

SHIFTED_PROFILE = """
    from comfytrace.cpython.base import CPythonLayout

    # Two link words before every object header (as a Py_TRACE_REFS build has), a key table with a wider header,
    # and dict/set entry orders unlike release CPython. Nothing but this profile describes these objects.
    LAYOUTS = (CPythonLayout(
        keys=(("windows", (3, 98), ""),),
        object_header_size=32, refcnt_offset=16, type_offset=24,
        tp_name_offset=40, tp_basicsize_offset=48, tp_itemsize_offset=56,
        str_length_offset=32, str_state_offset=48, str_ascii_data_offset=56, str_compact_data_offset=72,
        int_tag_offset=32, int_digits_offset=40, float_value_offset=32,
        tuple_size_offset=32, tuple_items_offset=40,
        list_size_offset=32, list_items_offset=40, list_allocated_offset=48,
        dict_used_offset=32, dict_keys_offset=48, dict_values_offset=56,
        dk_header_offset=16, dk_nentries_offset=32, dk_indices_offset=40,
        unicode_entry_key=8, unicode_entry_value=0,
        general_entry_hash=16, general_entry_key=0, general_entry_value=8,
        split_values_offset=16,
        set_fill_offset=32, set_used_offset=40, set_mask_offset=48, set_table_offset=56, set_entry_key=8,
        bytes_size_offset=32, bytes_hash_offset=40, bytes_data_offset=48,
    ),)
"""
LINK = 0x2000_0000_0010  # the debug-build link words: a pointer that is never an object here


class ShiftedPython:
    """Writes objects in the shifted profile's layout, independently of comfytrace: every offset is a literal."""

    DLL = 0x7FF0_0000_0000

    def __init__(self) -> None:
        self.dll = Heap(self.DLL, 1 << 16)
        self.heap = Heap(0x2000_0000_0000, 1 << 22)
        self.types: dict[str, int] = {}
        for name in ("type", "tuple", "dict", "str", "int", "float", "list", "bool", "bytes", "NoneType",
                     "frozenset"):
            name_va = self.dll.alloc(16)
            self.dll.raw(name_va, name.encode() + b"\0")
            self.dll.raw(name_va - 1, b"\0")
            self.types[name] = self.dll.alloc(96)
            self.dll.q(self.types[name] + 40, name_va)  # tp_name
        for t in self.types.values():
            self.dll.q(t + 24, self.types["type"])  # ob_type
        self.dll.q(self.types["NoneType"] + 48, 32)  # tp_basicsize: the bare 32-byte header
        self.none = self.dll.alloc(32)
        self.dll.q(self.none + 16, 1 << 30)
        self.dll.q(self.none + 24, self.types["NoneType"])
        self.heap_types = {name: self._heap_type(name) for name in ("CacheEntry", "Tensor")}

    def _heap_type(self, name: str) -> int:
        name_va = self.heap.alloc(len(name) + 1)
        self.heap.raw(name_va, name.encode() + b"\0")
        t = self.header(96, self.types["type"])
        self.heap.q(t + 40, name_va)
        return t

    def header(self, size: int, kind: int, refcnt: int = 1) -> int:
        va = self.heap.alloc(size)
        self.heap.q(va, LINK)
        self.heap.q(va + 8, LINK)
        self.heap.q(va + 16, refcnt)
        self.heap.q(va + 24, kind)
        return va

    def tuple_of(self, *items: int, refcnt: int = 1) -> int:
        va = self.header(40 + 8 * len(items), self.types["tuple"], refcnt)
        self.heap.q(va + 32, len(items))
        for i, item in enumerate(items):
            self.heap.q(va + 40 + 8 * i, item)
        return va

    def list_of(self, items: list[int]) -> int:
        vector = self.heap.alloc(8 * max(1, len(items)))
        for i, item in enumerate(items):
            self.heap.q(vector + 8 * i, item)
        va = self.header(56, self.types["list"])
        self.heap.q(va + 32, len(items))
        self.heap.q(va + 40, vector)
        self.heap.q(va + 48, max(1, len(items)))
        return va

    def _key_table(self, n: int, kind: int, width: int) -> tuple[int, int]:
        log2_size = 3
        while (1 << log2_size) * 2 // 3 < n:
            log2_size += 1
        log2_index = log2_size
        if not n:
            log2_size, log2_index = 0, 3
        keys = self.heap.alloc(40 + (1 << log2_index) + width * max(1, n))
        struct.pack_into("<BBB", self.heap.buf, keys + 16 - self.heap.base, log2_size, log2_index, kind)
        self.heap.q(keys + 32, n)
        return keys, keys + 40 + (1 << log2_index)

    def _dict(self, keys: int, n: int) -> int:
        va = self.header(64, self.types["dict"])
        self.heap.q(va + 32, n)
        self.heap.q(va + 48, keys)
        self.heap.q(va + 56, 0)
        return va

    def dict_of(self, pairs: list[tuple[int, int]]) -> int:
        """A combined unicode-key table; each 16-byte entry is (value, key)."""
        keys, entries = self._key_table(len(pairs), 1, 16)
        for i, (key, value) in enumerate(pairs):
            self.heap.q(entries + 16 * i, value)
            self.heap.q(entries + 16 * i + 8, key)
        return self._dict(keys, len(pairs))

    def general_dict(self, pairs: list[tuple[int, int]]) -> int:
        """A general-key table; each 24-byte entry is (key, value, hash)."""
        keys, entries = self._key_table(len(pairs), 0, 24)
        for i, (key, value) in enumerate(pairs):
            self.heap.q(entries + 24 * i, key)
            self.heap.q(entries + 24 * i + 8, value)
            self.heap.q(entries + 24 * i + 16, 0x5000 + i)
        return self._dict(keys, len(pairs))

    def frozenset(self, members: list[int]) -> int:
        """Set entries are (hash, key)."""
        slots = 8
        while slots * 3 // 5 < len(members):
            slots *= 2
        table = self.heap.alloc(16 * slots)
        for i, member in enumerate(members):
            self.heap.q(table + 16 * i, 0x1234 + i)
            self.heap.q(table + 16 * i + 8, member)
        va = self.header(72, self.types["frozenset"])
        for offset, value in ((32, len(members)), (40, len(members)), (48, slots - 1), (56, table)):
            self.heap.q(va + offset, value)
        return va

    def obj(self, value: Any) -> int:
        h = self.heap
        if value is None:
            return self.none
        if isinstance(value, str):
            data = value.encode("ascii")
            va = self.header(56 + len(data) + 1, self.types["str"])
            h.q(va + 32, len(data))
            struct.pack_into("<I", h.buf, va + 48 - h.base, (1 << 2) | (1 << 5) | (1 << 6))
            h.raw(va + 56, data)
            return va
        if isinstance(value, int):
            digits, v = [], abs(value)
            while v:
                digits.append(v & ((1 << 30) - 1))
                v >>= 30
            va = self.header(40 + 4 * len(digits), self.types["int"])
            h.q(va + 32, (len(digits) << 3) | (1 if value == 0 else 2 if value < 0 else 0))
            for i, d in enumerate(digits):
                struct.pack_into("<I", h.buf, va + 40 + 4 * i - h.base, d)
            return va
        if isinstance(value, float):
            va = self.header(40, self.types["float"])
            struct.pack_into("<d", h.buf, va + 32 - h.base, value)
            return va
        if isinstance(value, tuple):
            return self.tuple_of(*(self.obj(v) for v in value))
        if isinstance(value, list):
            return self.list_of([self.obj(v) for v in value])
        if isinstance(value, dict):
            return self.dict_of([(self.obj(k), self.obj(v)) for k, v in value.items()])
        raise TypeError(value)

    def hashable(self, value: Any) -> int:
        """ComfyUI's to_hashable, written as objects."""
        if value is None or isinstance(value, str | int | float):
            return self.obj(value)
        pairs = sorted(value.items()) if isinstance(value, dict) else list(enumerate(value))
        return self.frozenset([self.tuple_of(self.obj(k), self.hashable(v)) for k, v in pairs])

    def tensor(self, sizes: list[int]) -> tuple[int, int]:
        impl = self.heap.alloc(0xB0)
        self.heap.q(impl + NDIM_OFFSET, len(sizes))
        for i, size in enumerate(sizes):
            self.heap.q(impl + SIZES_OFFSET + 8 * i, size)
        self.heap.q(impl + NUMEL_OFFSET, int(np.prod(sizes)))
        self.heap.q(impl + DTYPE_OFFSET, 6)
        va = self.header(40, self.heap_types["Tensor"])
        self.heap.q(va + 32, impl)  # cdata right after the 32-byte header
        return va, impl

    def cache_entry(self, ui: Any, outputs: list[list[int]]) -> int:
        out_list = self.list_of([self.list_of(items) for items in outputs])
        va = self.header(56, self.heap_types["CacheEntry"])
        self.heap.q(va + 32, 2)
        self.heap.q(va + 40, self.obj(ui))
        self.heap.q(va + 48, out_list)
        return va

    def live_bytes(self, data: bytes) -> int:
        va = self.header(48 + len(data) + 1, self.types["bytes"])
        self.heap.q(va + 32, len(data))
        self.heap.q(va + 40, MINUS_ONE)
        self.heap.raw(va + 48, data)
        return va + 48

    def memory(self, dll_name: str, extra: dict[int, bytes] | None = None) -> ProcessMemory:
        blobs = {self.dll.base: bytes(self.dll.buf), self.heap.base: bytes(self.heap.buf), **(extra or {})}
        image = MappedImage(dll_name, f"C:/x/{dll_name}", self.dll.base, self.dll.base + len(self.dll.buf))
        return _memory(blobs, [image])


def test_profile_drives_queue_cache_and_pixel_scans(profile_package, monkeypatch):
    package, write = profile_package
    write("v398", SHIFTED_PROFILE)
    bundled = registry.PROFILES_PACKAGE
    monkeypatch.setattr(registry, "PROFILES_PACKAGE", package)
    registry.clear_cache()
    (shifted,) = registry.layouts().values()
    stock = CPythonLayout(keys=(("windows", (0, 0), ""),))
    same = {f.name for f in dataclasses.fields(CPythonLayout) if getattr(shifted, f.name) == getattr(stock, f.name)}
    assert same == {"unicode_entry_size", "general_entry_size", "set_entry_size"}, "every offset is moved"

    py = ShiftedPython()
    queue = py.tuple_of(*(py.obj(v) for v in (0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}},
                                                ("9",))), refcnt=3)
    py.dict_of([(py.obj("prompt"), queue), (py.obj("outputs"), py.obj({"9": {"images": []}})),
                (py.obj("status"), py.obj({"status_str": "success"}))])
    prompt = api_prompt(seed=222)
    key = py.hashable(signature(prompt, "8"))
    image, impl = py.tensor([1, 48, 64, 3])
    py.general_dict([(key, py.cache_entry(None, [[image]]))])
    py.dict_of([(py.obj("8"), key)])  # ComfyUI's key set: node id -> the frozenset keying the cache
    live_va = py.live_bytes(_smooth_image(256, 256, seed=1).tobytes())
    freed = _smooth_image(256, 256, seed=2).tobytes()
    freed_base = 0x3000_0000_0000
    freed_blob = struct.pack("<QQQQQQ", LINK, LINK, LINK + 0x40, LINK + 0x80, len(freed), MINUS_ONE) + freed

    mem = py.memory("python398.dll", {freed_base: freed_blob})
    heap = pyheap.CPythonHeap(mem)
    assert heap.layout is shifted and heap.split_values_offset == 16
    assert heap.layout.is_none(mem, py.none, None, heap.dll), "None by shape: a bare 32-byte header"

    result = pyheap.extract(mem)
    assert (result.status, result.python_version) == ("AVAILABLE", "3.98")
    (item,) = result.items
    assert (item.prompt_id, item.prompt, item.workflow, item.refcnt) == (PROMPT_ID, PROMPT, WORKFLOW, 3)
    assert item.history["status"] == {"status_str": "success"} and item.liveness == "LIVE"
    buffers = {b.address: (b.width, b.height, b.channels, b.state) for b in result.pixel_buffers}
    assert buffers == {live_va: (256, 256, 3, "LIVE"), freed_base + 48: (256, 256, 3, "FREED_OR_UNTYPED")}

    entries, summary = execcache.extract(mem)
    assert (summary["status"], summary["entries"], summary["tensors_held"]) == ("FOUND", 1, 1)
    (entry,) = entries
    assert (entry.class_type, entry.value_kind, entry.node_ids) == ("VAEDecode", "CacheEntry", ["8"])
    [[held]] = entry.outputs
    assert (held["type"], held["tensorimpl"], held["sizes"]) == ("Tensor", impl, [1, 48, 64, 3])
    assert execcache.matching_nodes(entry.graph, prompt) == ["8"]

    # The very same bytes under a release runtime are not CPython objects: no offset is assumed anywhere.
    monkeypatch.setattr(registry, "PROFILES_PACKAGE", bundled)
    registry.clear_cache()
    stock_mem = py.memory("python313.dll", {freed_base: freed_blob})
    stock_result = pyheap.extract(stock_mem)
    assert stock_result.status == "UNAVAILABLE" and stock_result.reason.startswith("TYPE_OBJECTS_UNRESOLVED")
    assert execcache.extract(stock_mem)[0] == []
