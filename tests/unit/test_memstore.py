"""Compression-store pages: codecs, layout checks, and store pages read as process memory."""

import hashlib
import struct

import pytest

from comfytrace import memstore
from comfytrace.pagetable import PageTables, canonical
from comfytrace.procmem import ProcessMemory

PAGE = 4096
# A real 45-byte record from T2_running.raw (PID 3336, VA 0xdfe559d000) that expands to one page.
REAL_LZ4 = bytes.fromhex("1f000100ff3221ae03460111010600020800080c000f0200" + "ff" * 14 + "92500000000000")
REAL_SHA256 = "679f8c86459d16496ce231373d032f3a1b1582f9ec379377b658fbe257691444"


def test_lz4_record_from_t2_expands_to_its_page_and_damage_is_refused():
    out, used = memstore.lz4_block(REAL_LZ4)
    assert len(out) == PAGE and used == len(REAL_LZ4) and hashlib.sha256(out).hexdigest() == REAL_SHA256
    for bad in (REAL_LZ4[:-1], REAL_LZ4[:20], b"\x00\x33\x00"):
        try:
            got, used = memstore.lz4_block(bad)
        except memstore.Corrupt:
            continue
        assert not (len(got) == PAGE and used == len(bad))


def test_xpress_output_beyond_its_size_is_refused_not_cut():
    # One literal, then a match of 5000 bytes: more than the page it claims to be.
    with pytest.raises(memstore.Corrupt):
        memstore.xpress_lz77(bytes.fromhex("000000404107000fff8513"), PAGE)


def test_non_canonical_addresses_do_not_alias_low_pages():
    phys = Phys()
    dtb, tables = phys.alloc(), {}
    page = phys.alloc()
    _map(phys, dtb, 0x10000, page | 1, tables)
    pt = PageTables(phys.data, dtb)
    assert pt.translate(0x10000) == page
    assert pt.translate(0x1_0000_0001_0000) is None


def test_a_kernel_address_masked_to_48_bits_reads_once_made_canonical():
    phys = Phys()
    dtb, tables = phys.alloc(), {}
    page = phys.alloc()
    _map(phys, dtb, 0xFFFF_F803_1200_0000, page | 1, tables)
    pt = PageTables(phys.data, dtb)
    assert pt.translate(0xF803_1200_0000) is None
    assert pt.translate(canonical(0xF803_1200_0000)) == page
    assert canonical(0x7FF6_0000_0000) == 0x7FF6_0000_0000


def test_xpress_examples_of_the_specification():
    assert memstore.xpress_lz77(bytes.fromhex("3f000000") + b"abcdefghijklmnopqrstuvwxyz", 26)[0] == \
        b"abcdefghijklmnopqrstuvwxyz"
    assert memstore.xpress_lz77(bytes.fromhex("ffffff1f61626317000fff2601"), 300)[0] == b"abc" * 100


class Phys:
    def __init__(self, pages: int = 32) -> None:
        self.data = bytearray(pages * PAGE)
        self.next = 1

    def alloc(self) -> int:
        self.next += 1
        return (self.next - 1) * PAGE

    def q(self, at: int, value: int) -> None:
        self.data[at : at + 8] = struct.pack("<Q", value)


def _map(phys: Phys, dtb: int, va: int, leaf: int, tables: dict) -> None:
    base = dtb
    for shift in (39, 30, 21):
        key = (base, (va >> shift) & 0x1FF)
        if key not in tables:
            tables[key] = phys.alloc()
            phys.q(base + 8 * key[1], tables[key] | 1)
        base = tables[key]
    phys.q(base + 8 * ((va >> 12) & 0x1FF), leaf)


class FakeStore:
    pagefile = 2

    def page(self, key):
        return ("OK_LZ4", bytes([key & 0xFF]) * PAGE) if key != 0x99 else ("RECORD_FREE", None)


def test_store_pages_read_as_process_memory_without_a_physical_offset():
    phys = Phys()
    dtb, tables = phys.alloc(), {}
    resident = phys.alloc()
    phys.data[resident : resident + PAGE] = b"R" * PAGE
    _map(phys, dtb, 0x10000, resident | 1, tables)
    _map(phys, dtb, 0x11000, (0x41 << 32) | (2 << 12) | 0x80, tables)  # store key 0x41
    _map(phys, dtb, 0x12000, (0x99 << 32) | (2 << 12) | 0x80, tables)  # key whose record is free
    _map(phys, dtb, 0x13000, (0x42 << 32) | (0 << 12) | 0x80, tables)  # pagefile 0: out of reach
    _map(phys, dtb, 0x14000, (0x43 << 32) | 0x1234_0000 | (2 << 12), tables)  # bits 16-31 set: not a PTE
    mem = ProcessMemory.from_page_tables(3, PageTables(phys.data, dtb), [(0x10000, 0x14FFF, None)], phys.data,
                                         store=FakeStore())
    assert mem.read(0x10FFE, 4) == b"RR\x41\x41"
    assert mem.physical_of(0x10000) == resident and mem.physical_of(0x11000) is None
    assert mem.from_store(0x11000) and not mem.from_store(0x10000)
    assert mem.try_read(0x12000, 8) is None and mem.try_read(0x13000, 8) is None and mem.try_read(0x14000, 8) is None
    assert mem.store_status == {"OK_LZ4": 1, "RECORD_FREE": 1}
    assert [va for va, _ in mem.chunks()] == [0x10000]


def test_a_kernel_without_the_measured_store_layout_is_refused():
    phys = Phys()
    kdtb, tables = phys.alloc(), {}
    globals_page, manager_page = phys.alloc(), phys.alloc()
    smglobals, manager_va = 0xFFFF_F800_0000_0000, 0xFFFF_D000_0000_1000
    _map(phys, kdtb, smglobals, globals_page | 1, tables)
    _map(phys, kdtb, manager_va - PAGE, phys.alloc() | 1, tables)
    _map(phys, kdtb, manager_va, manager_page | 1, tables)
    manager = manager_va + 0x40
    phys.q(globals_page + 0x10, manager + memstore.PROFILE["manager_link"])
    kernel = PageTables(phys.data, kdtb)
    with pytest.raises(memstore.Corrupt, match="SmPa"):
        memstore.StoreReader.open(kernel, smglobals, 2, lambda e: "MemCompression")
    phys.data[manager_page + 0x40 - 12 : manager_page + 0x40 - 8] = b"SmPa"
    with pytest.raises(memstore.Corrupt, match="MemCompression"):
        memstore.StoreReader.open(kernel, smglobals, 2, lambda e: "MemCompression")


def test_store_and_layer_reads_shorter_than_asked_are_unreadable():
    from comfytrace.procmem import STORE_BASE, Unreadable

    mem = ProcessMemory(3, [(0x10000, PAGE, STORE_BASE)], b"", store_data=b"A")
    with pytest.raises(Unreadable):
        mem.read(0x10000, 8)

    class ShortLayer:
        def read(self, va, length, pad=False):
            return b"\x7f"

    layered = ProcessMemory(3, [], None, layer=ShortLayer())
    with pytest.raises(Unreadable):
        layered.q(0x10000)
