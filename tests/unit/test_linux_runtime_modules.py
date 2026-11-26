"""Linux ELF module discovery, CPython 3.12 version gating and libc10 tensors, on synthetic VMAs and bytes."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from comfytrace import pyheap, torchmem
from comfytrace.procmem import MappedImage, ProcessMemory, elf_modules
from tests.unit.test_memory_recovery import PROMPT, PROMPT_ID, WORKFLOW, FakePython, Heap

PAGE = 4096
DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_GNU_HASH = 4, 5, 6, 10, 0x6FFFFEF5
LIB = "/opt/conda/lib/libpython3.12.so.1.0"
EXE = "/usr/bin/python3.12"


def _gnu_hash(name: bytes) -> int:
    h = 5381
    for c in name:
        h = (h * 33 + c) & 0xFFFFFFFF
    return h


def write_elf(h: Heap, symbols: dict[str, int], *, relocated: bool = True, gnu: bool = True,
              machine: int = 62) -> None:
    """A 64-bit ET_DYN header at the start of ``h`` whose dynamic section exports ``symbols`` (name -> address).
    ``relocated`` writes the DT_* pointers as ld.so leaves them (absolute); otherwise as link-time offsets."""
    base, names = h.base, list(symbols)
    strtab = h.alloc(1 + sum(len(n) + 1 for n in names))
    offsets, cursor = [], 1
    for n in names:
        h.raw(strtab + cursor, n.encode() + b"\0")
        offsets.append(cursor)
        cursor += len(n) + 1
    symtab = h.alloc(24 * (len(names) + 1))
    for i, (n, off) in enumerate(zip(names, offsets, strict=True), 1):
        struct.pack_into("<IBBHQQ", h.buf, symtab + 24 * i - base, off, 0x11, 0, 1, symbols[n] - base, 8)
    if gnu:
        table = h.alloc(16 + 8 + 4 + 4 * len(names))
        struct.pack_into("<IIIIQI", h.buf, table - base, 1, 1, 1, 6, (1 << 64) - 1, 1)
        for i, n in enumerate(names):
            value = _gnu_hash(n.encode()) & ~1 | (1 if i == len(names) - 1 else 0)
            struct.pack_into("<I", h.buf, table + 28 + 4 * i - base, value)
        hash_tag = DT_GNU_HASH
    else:
        nchain = len(names) + 1
        table = h.alloc(8 + 4 + 4 * nchain)
        chain = [0] + [i + 1 if i < len(names) else 0 for i in range(1, nchain)]
        struct.pack_into(f"<III{nchain}I", h.buf, table - base, 1, nchain, 1, *chain)
        hash_tag = DT_HASH
    rel = 0 if relocated else base
    tags = [(DT_STRTAB, strtab - rel), (DT_SYMTAB, symtab - rel), (DT_STRSZ, cursor), (hash_tag, table - rel), (0, 0)]
    dynamic = h.alloc(16 * len(tags))
    for i, (tag, value) in enumerate(tags):
        struct.pack_into("<qQ", h.buf, dynamic + 16 * i - base, tag, value)
    ident = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    h.raw(base, ident + struct.pack("<HHIQQQIHHHHHH", 3, machine, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0, 0))
    struct.pack_into("<IIQQQQQQ", h.buf, 64, 1, 5, 0, 0, 0, len(h.buf), len(h.buf), PAGE)
    struct.pack_into("<IIQQQQQQ", h.buf, 64 + 56, 2, 6, dynamic - base, dynamic - base, dynamic - base,
                     16 * len(tags), 16 * len(tags), 8)


def python_module(py: FakePython, version_hex: int | None, **kwargs: object) -> None:
    """Turns the fake python DLL arena into an ELF module exporting ``Py_Version`` (when given)."""
    symbols: dict[str, int] = {"PyType_Type": py.types["type"]}
    if version_hex is not None:
        at = py.dll.alloc(8)
        py.dll.q(at, version_hex)
        symbols["Py_Version"] = at
    write_elf(py.dll, symbols, **kwargs)  # type: ignore[arg-type]


class FlatLayer:
    """A process layer over separate virtual blobs; ``mapping`` names the virtual address as its own offset."""

    def __init__(self, blobs: dict[int, bytes]) -> None:
        self.blobs = blobs

    def mapping(self, offset: int, length: int, ignore_errors: bool = False):
        for va, blob in sorted(self.blobs.items()):
            lo, hi = max(offset, va), min(offset + length, va + len(blob))
            if lo < hi:
                yield lo, hi - lo, lo, hi - lo, "memory_layer"

    def read(self, offset: int, length: int, pad: bool = False) -> bytes:
        for va, blob in self.blobs.items():
            if va <= offset and offset + length <= va + len(blob):
                return blob[offset - va : offset - va + length]
        raise ValueError(f"0x{offset:x} not mapped")


def linux_memory(py: FakePython, path: str, *, header: bool = True) -> ProcessMemory:
    """The python arena as a two-VMA file mapping followed by an anonymous .bss VMA, plus the heap."""
    dll = py.dll.base
    blobs = {dll: bytes(py.dll.buf), py.heap.base: bytes(py.heap.buf)}
    if not header:  # the ELF header bytes are not resident; the type objects after them are
        blobs = {dll + 64: bytes(py.dll.buf[64:]), py.heap.base: bytes(py.heap.buf)}
    half = len(py.dll.buf) // 2
    vmas = [(dll, dll + half - 1, path), (dll + half, dll + len(py.dll.buf) - 1, path),
            (dll + len(py.dll.buf), dll + len(py.dll.buf) + PAGE - 1, None),
            (py.heap.base, py.heap.base + len(py.heap.buf) - 1, "[heap]")]
    return ProcessMemory.from_layer(4242, FlatLayer(blobs), vmas, None, image_os="linux")


def queue(py: FakePython) -> None:
    py.obj((0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}}, ("9",)))


def test_split_vmas_of_one_file_form_one_module_and_other_maps_are_not_modules():
    vmas = [(0x1000_0000, 0x1000_0FFF, LIB), (0x1000_1000, 0x1000_1FFF, None),  # an anonymous gap
            (0x1000_2000, 0x1000_3FFF, LIB), (0x1000_4000, 0x1000_4FFF, "[heap]"),
            (0x1000_5000, 0x1000_5FFF, LIB + " (deleted)"),  # another path ends the run
            (0x2000_0000, 0x2FFF_FFFF, "/models/checkpoints/sd15.safetensors"),
            (0x3000_0000, 0x3000_0FFF, "/torch/lib/libc10.so"), (0x3000_1000, 0x3000_1FFF, "/torch/lib/libc10.so"),
            (0x4000_0000, 0x4000_0FFF, "/usr/bin/python3"), (0x5000_0000, 0x5000_0FFF, "/usr/bin/bash")]
    images = elf_modules(vmas)
    assert [(m.name, m.segments) for m in images] == [
        ("libpython3.12.so.1.0", ((0x1000_0000, 0x1000_1000), (0x1000_2000, 0x1000_4000))),
        ("libpython3.12.so.1.0", ((0x1000_5000, 0x1000_6000),)),
        ("libc10.so", ((0x3000_0000, 0x3000_1000), (0x3000_1000, 0x3000_2000))),
        ("python3", ((0x4000_0000, 0x4000_1000),))]
    lib = images[0]
    assert lib.kind == "elf" and (lib.base, lib.end) == (0x1000_0000, 0x1000_4000)
    assert lib.contains(0x1000_0010) and lib.contains(0x1000_3FF8)
    assert not lib.contains(0x1000_1800), "the anonymous gap is not part of the module"
    assert MappedImage("c10.dll", "c10.dll", 0x10, 0x20).contains(0x18), "Windows images are unchanged"


def test_linux_libpython_split_maps_decode_cpython_312_through_the_layer():
    py = FakePython()
    queue(py)
    python_module(py, 0x030C03F0)
    mem = linux_memory(py, LIB)
    assert [m.name for m in mem.images] == ["libpython3.12.so.1.0"]
    assert mem.elf_symbol(mem.images[0], "PyType_Type") == py.types["type"]
    assert mem.elf_symbol(mem.images[0], "Py_Missing") is None
    result = pyheap.extract(mem, scan_pixels=False)
    assert (result.status, result.python_version, result.python_dll) == ("AVAILABLE", "3.12", LIB)
    (item,) = result.items
    assert item.prompt == PROMPT and item.workflow == WORKFLOW


@pytest.mark.parametrize("relocated,gnu", [(True, True), (False, True), (True, False), (False, False)])
def test_a_statically_linked_python_executable_is_verified_by_its_exported_version(relocated, gnu):
    py = FakePython()
    queue(py)
    python_module(py, 0x030C07F0, relocated=relocated, gnu=gnu)
    result = pyheap.extract(linux_memory(py, EXE), scan_pixels=False)
    assert (result.status, result.reason, result.python_version) == ("AVAILABLE", None, "3.12")
    assert [i.prompt_id for i in result.items] == [PROMPT_ID]


@pytest.mark.parametrize("path,version_hex,reason", [
    (EXE, None, "PYTHON_VERSION_UNVERIFIED"),
    ("/usr/bin/python3", 0x030D01F0, "UNSUPPORTED_PYTHON_VERSION:3.13"),
    ("/usr/bin/python3.11", 0x030C01F0, "PYTHON_VERSION_MISMATCH:3.11!=3.12"),
    ("/usr/lib/libpython3.11.so.1.0", None, "UNSUPPORTED_PYTHON_VERSION:3.11"),
    ("/usr/lib/libpython3.13.so.1.0", 0x030D00F0, "UNSUPPORTED_PYTHON_VERSION:3.13"),
    (LIB, 0x030D00F0, "PYTHON_VERSION_MISMATCH:3.12!=3.13"),
    ("/usr/lib/libpython3.12d.so.1.0", 0x030C00F0, "UNSUPPORTED_PYTHON_BUILD:d"),
    ("/usr/lib/libpython3.13t.so.1.0", 0x030D00F0, "UNSUPPORTED_PYTHON_BUILD:t"),
])
def test_unsupported_or_unverified_linux_pythons_fail_closed(path, version_hex, reason):
    py = FakePython()
    queue(py)
    python_module(py, version_hex)
    mem = linux_memory(py, path)
    result = pyheap.extract(mem, scan_pixels=False)
    assert (result.status, result.reason, result.items) == ("UNAVAILABLE", reason, [])
    assert result.type_objects == {}, "no type object is claimed for an unsupported runtime"
    assert not pyheap.CPythonHeap(mem).available


def test_without_a_resident_elf_header_an_executable_is_not_a_module_but_a_named_library_is():
    py = FakePython()
    queue(py)
    python_module(py, 0x030C03F0)
    assert pyheap.extract(linux_memory(py, EXE, header=False)).reason == "NO_PYTHON_DLL_MAPPED"
    # libpython is named by its file; with no header to read, its version is the one in its name.
    lib = pyheap.extract(linux_memory(py, LIB, header=False), scan_pixels=False)
    assert (lib.status, lib.python_version, len(lib.items)) == ("AVAILABLE", "3.12", 1)


def test_a_readable_non_elf_or_foreign_machine_header_drops_the_module():
    py = FakePython()
    python_module(py, 0x030C03F0, machine=40)  # 32-bit ARM
    assert linux_memory(py, LIB).images == []
    py = FakePython()
    py.dll.raw(py.dll.base, b"MZ" + bytes(62))
    assert linux_memory(py, LIB).images == []


def test_cpython_312_split_dict_values_start_at_offset_zero_on_linux():
    py = FakePython()
    python_module(py, 0x030C03F0)
    h = py.heap
    k1, k2, v1, v2 = py.obj("seed"), py.obj("steps"), py.obj(7), py.obj(20)
    keys = h.alloc(32 + 8 + 16 * 2)
    struct.pack_into("<BBB", h.buf, keys + 8 - h.base, 3, 3, 2)  # DICT_KEYS_SPLIT
    h.q(keys + 24, 2)
    h.q(keys + 40, k1)
    h.q(keys + 56, k2)
    values = h.alloc(8 + 16)
    h.q(values, v1)  # 3.12: values[0] is the first word; 3.13 would put a header here
    h.q(values + 8, v2)
    obj = h.alloc(48)
    h.q(obj + 8, py.types["dict"])
    h.q(obj + 16, 2)
    h.q(obj + 32, keys)
    h.q(obj + 40, values)
    heap = pyheap.CPythonHeap(linux_memory(py, LIB))
    assert heap.version == (3, 12) and heap.decode(obj) == {"seed": 7, "steps": 20}


# -- libc10 tensors -------------------------------------------------------------------------------------------

C10 = "/venv/lib/python3.12/site-packages/torch/lib/libc10.so"
C10_TEXT, C10_GAP, C10_DATA = 0x500000, 0x501000, 0x502000


def _tensor_memory(deleter: int = C10_DATA + 0x100, impl_vptr: int = C10_TEXT + 0x200,
                   image_os: str = "linux") -> tuple[ProcessMemory, np.ndarray]:
    base = 0x100000
    data_va = base + 0x20000
    floats = np.random.default_rng(5).random((1, 64, 64, 3), dtype=np.float32)
    raw = floats.tobytes()
    physical = bytearray(0x20000 + len(raw))
    physical[0x20000:] = raw
    storage = base + 0x100
    struct.pack_into("<7Q", physical, 0x100, C10_TEXT + 0x100, 1, data_va, deleter, data_va, 0xFF00, len(raw))
    words = [0] * 22
    words[:3] = [impl_vptr, 1, storage]
    words[8] = 4
    words[9:13] = [1, 64, 64, 3]
    words[14:18] = [12288, 192, 3, 1]
    words[19:22] = [0, 12288, 6]
    struct.pack_into("<22Q", physical, 0x200, *words)
    vmas = [(C10_TEXT, C10_GAP - 1, C10), (C10_GAP, C10_DATA - 1, None), (C10_DATA, C10_DATA + PAGE - 1, C10),
            (0x600000, 0x600FFF, "/usr/lib/libfoo.so"), (0x700000, 0x700FFF, "/models/unet.safetensors")]
    images = elf_modules(vmas) if image_os == "linux" else []
    mem = ProcessMemory(1, [(base, len(physical), 0)], bytes(physical), images, vads=vmas, image_os=image_os)
    return mem, floats


def test_libc10_split_maps_yield_a_cpu_image_tensor():
    mem, floats = _tensor_memory()
    (tensor,) = torchmem.image_tensors(mem)
    assert tensor.sizes == [1, 64, 64, 3]
    assert np.array_equal(tensor.rgb, np.clip(255.0 * floats[0], 0, 255).astype(np.uint8))


@pytest.mark.parametrize("deleter,impl_vptr", [
    (C10_GAP + 0x100, C10_TEXT + 0x200),  # the deleter word lies in the anonymous gap, not in libc10
    (C10_DATA + 0x100, 0x600200),  # the TensorImpl vtable lies in an unrelated library
    (C10_DATA + 0x100, 0x700200),  # ... or in a mapped model file
])
def test_linux_tensors_need_libc10_deleter_and_torch_vtables(deleter, impl_vptr):
    mem, _ = _tensor_memory(deleter, impl_vptr)
    assert torchmem.image_tensors(mem) == []


def test_a_linux_process_without_libc10_has_no_tensors_and_linux_names_are_not_windows_names():
    mem, _ = _tensor_memory()
    mem.images = [m for m in mem.images if m.name != "libc10.so"]
    assert torchmem.image_tensors(mem) == []
    windows, _ = _tensor_memory(image_os="windows")
    windows.images = elf_modules(windows.vads)
    assert torchmem.c10_image(windows) is None, "a Windows process looks for c10.dll only"


def test_from_layer_defaults_to_windows_images():
    layer = FlatLayer({0x10000: bytes(PAGE)})
    mem = ProcessMemory.from_layer(1, layer, [(0x10000, 0x10FFF, r"\Windows\System32\c10.dll"),
                                              (0x20000, 0x20FFF, "/usr/lib/libc10.so")], None)
    assert mem.image_os == "windows"
    assert [(m.name, m.segments, m.kind) for m in mem.images] == [("c10.dll", (), "pe")]
    with pytest.raises(ValueError):
        ProcessMemory.from_layer(1, layer, [], None, image_os="macos")
