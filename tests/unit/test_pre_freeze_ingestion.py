"""Pre-freeze regressions for memory/OS ingestion and runtime/heap recovery fixes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from comfytrace import linux_memstore
from comfytrace.cpython.base import CorruptObject, CPythonLayout
from comfytrace.memstore import Corrupt

EOF = bytes([0x11, 0x00, 0x00])
LAYOUT = CPythonLayout(keys=(("windows", (3, 12), ""),))
DUMMY, MINUS_ONE = 0x7FF0_0000_1000, (1 << 64) - 1


def _zero_run(zeros: int, trailing: bytes = b"") -> bytes:
    assert 4 <= zeros <= 2047 + 4 and len(trailing) <= 3
    run = zeros - 4
    return bytes([0x18 | (run & 7), 0xFC | len(trailing), 0xFF, run >> 3]) + trailing


def _rle(literals: bytes, *runs: bytes) -> bytes:
    """An LZO-RLE (bitstream v1) stream: a literal run, then zero runs, then EOF."""
    assert 4 <= len(literals) <= 238
    return bytes([17, 1, 17 + len(literals)]) + literals + b"".join(runs) + EOF


def test_lzo_rle_zero_run_decodes_like_the_kernel():
    data = _rle(b"ABCD", _zero_run(100))
    out, used = linux_memstore.lzo1x_decompress(data, out_size=104)
    assert out == b"ABCD" + bytes(100) and used == len(data)


def test_lzo_rle_zero_run_with_trailing_literals_and_full_page():
    data = _rle(b"WXYZ", _zero_run(2051), _zero_run(4096 - 6 - 2051, b"QR"))
    status, page = linux_memstore.decompress_linux_page(data, "lzo-rle")
    assert status == "OK_LZO" and page == b"WXYZ" + bytes(4096 - 6) + b"QR"


def test_lzo_m4_zero_distance_other_than_eof_is_corrupt_not_zeros():
    # Formerly decoded as length+2 invented zero bytes on an RLE stream.
    with pytest.raises(Corrupt):
        linux_memstore.lzo1x_decompress(bytes([17, 1, 21]) + b"ABCD" + bytes([0x13, 0x00, 0x00]) + EOF, 64)


def test_plain_lzo_ignores_the_rle_marker_shape():
    # Without the RLE header the same opcode is an ordinary M4 match (distance far beyond the output).
    with pytest.raises(Corrupt):
        linux_memstore.lzo1x_decompress(bytes([21]) + b"ABCD" + bytes([0x18, 0xFC, 0xFF, 0x00]) + EOF, 64)


class _Mem:
    def __init__(self, words: dict[int, int], table: int, entries: list[tuple[int, int]]) -> None:
        self.words, self.table, self.entries = words, table, entries

    def q(self, va: int) -> int:
        return self.words[va]

    def read(self, va: int, length: int) -> bytes:
        assert va == self.table
        return np.array([w for entry in self.entries for w in entry], np.uint64).tobytes()[:length]


def _set(entries: list[tuple[int, int]], used: int, fill: int) -> tuple[_Mem, int]:
    obj, table = 0x1000, 0x2000
    words = {obj + 16: fill, obj + 24: used, obj + 32: len(entries) - 1, obj + 40: table}
    return _Mem(words, table, entries), obj


def test_set_with_discarded_members_skips_dummy_entries():
    entries = [(0xA000, 11), (DUMMY, MINUS_ONE), (0xB000, 12), (DUMMY, MINUS_ONE)] + [(0, 0)] * 4
    mem, obj = _set(entries, used=2, fill=4)
    assert LAYOUT.set_members(mem, obj, 1 << 10) == [0xA000, 0xB000]


def test_set_fill_must_count_dummies_too():
    entries = [(0xA000, 11), (DUMMY, MINUS_ONE)] + [(0, 0)] * 6
    mem, obj = _set(entries, used=1, fill=1)
    with pytest.raises(CorruptObject):
        LAYOUT.set_members(mem, obj, 1 << 10)


def test_store_owner_name_is_read_virtually_across_a_page_boundary(monkeypatch):
    from comfytrace import ingest, memstore

    name_offset, eprocess = 0x5A8, 0xFFFF_8000_0000_0A50  # ImageFileName starts 8 bytes before a page end
    virtual = {eprocess + name_offset + i: b for i, b in enumerate(b"MemCompression\0")}

    class Kernel:
        physical = bytes(1 << 16)

        def read_kernel_qword(self, va):
            return 0

        def translate(self, va):  # the second page is mapped far from the first
            return 0x1000 if va >> 12 == (eprocess + name_offset) >> 12 else 0x8000

        def read_virtual(self, va, size):
            return bytes(virtual[va + i] for i in range(size))

    seen = {}

    def fake_open(kernel, smglobals, pagefile, owner_name):
        seen["name"] = owner_name(eprocess)
        raise memstore.Corrupt("stop")

    member = SimpleNamespace(vol=SimpleNamespace(members={"VirtualPageFileNumber": (0, None)}, type_name="t"))
    space = SimpleNamespace(get_type=lambda name: (
        SimpleNamespace(vol=SimpleNamespace(members={"Store": (0, member)})) if name.endswith("_MI_PARTITION")
        else SimpleNamespace(vol=SimpleNamespace(members={"ImageFileName": (name_offset, None)}))
        if name.endswith("_EPROCESS") else member))
    module = SimpleNamespace(symbol_table_name="nt", offset=0xF800_0000_0000,
                             get_symbol=lambda name: SimpleNamespace(address=0x100))
    context = SimpleNamespace(modules={"kernel": module}, symbol_space=space)
    monkeypatch.setattr(memstore.StoreReader, "open", staticmethod(fake_open))
    reader, note = ingest._store_reader(context, "kernel", ingest.RecoverySupport(physical=Kernel.physical,
                                                                                  kernel=Kernel()))
    assert reader is None and note == "UNAVAILABLE: stop"
    assert seen["name"] == "MemCompression"

def test_linux_protnone_large_page_translation_and_walk():
    import struct

    from comfytrace.pagetable import LARGE, PAGE, PFN_MASK, PROTNONE, VALID, PageTables

    physical = bytearray(4 * PAGE)
    struct.pack_into("<Q", physical, PAGE, 2 * PAGE | VALID)
    struct.pack_into("<Q", physical, 2 * PAGE, 3 * PAGE | VALID)
    frame = 2 << 20
    struct.pack_into("<Q", physical, 3 * PAGE, ((~frame) & PFN_MASK) | LARGE | PROTNONE)
    walker = PageTables(physical, PAGE, os_mode="linux")
    assert walker.translate(PAGE + 5) == frame + PAGE + 5
    assert list(walker.resident_pages(PAGE + 5, 2 * PAGE)) == [(PAGE, frame + PAGE)]
