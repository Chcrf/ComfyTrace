"""High (5-level) user addresses and PyTorch structures straddling a scan-chunk boundary."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from comfytrace import torchmem
from comfytrace.elfcore import HEADER, PROGRAM, CoreLayer
from comfytrace.procmem import PAGE, USER_LIMIT, MappedImage, ProcessMemory, Unreadable

HIGH_VA = 0x00FF_7F00_0000_0000  # a 5-level (LA57) user address, above 2^47 and below 2^56


def _high_memory(**kwargs) -> ProcessMemory:
    physical = bytearray(2 * PAGE)
    struct.pack_into("<Q", physical, 0x18, 0xC0FFEE)
    return ProcessMemory(1, [(HIGH_VA, len(physical), 0)], bytes(physical),
                         vads=[(HIGH_VA, HIGH_VA + len(physical) - 1, None)], **kwargs)


def test_a_five_level_process_reads_and_scans_high_user_addresses():
    mem = _high_memory(user_limit=1 << 56)
    assert mem.q(HIGH_VA + 0x18) == 0xC0FFEE
    assert [va for va, _ in mem.chunks()] == [HIGH_VA]
    hits = [addr for addr, _w, _p in mem.scan_words(lambda w: np.nonzero(w == np.uint64(0xC0FFEE))[0])]
    assert hits == [HIGH_VA + 0x18]


def test_a_four_level_process_still_refuses_addresses_above_its_limit():
    mem = _high_memory()
    assert mem.user_limit == USER_LIMIT
    with pytest.raises(Unreadable):
        mem.read(HIGH_VA, 8)
    assert list(mem.chunks()) == []


def test_five_level_page_tables_carry_the_high_limit():
    tables = SimpleNamespace(levels=5, resident_entries=lambda lo, hi: [
        (va, va - HIGH_VA, "VALID", None) for va in (HIGH_VA, HIGH_VA + PAGE) if lo <= va < hi])
    physical = bytearray(2 * PAGE)
    physical[PAGE:PAGE + 4] = b"LA57"
    mem = ProcessMemory.from_page_tables(1, tables, [(HIGH_VA, HIGH_VA + 2 * PAGE - 1, None)], bytes(physical),
                                         image_os="linux")
    assert mem.user_limit == 1 << 56
    assert mem.read(HIGH_VA + PAGE, 4) == b"LA57"


def _core(start: int, payload: bytes, pid: int = 3215) -> bytes:
    status = bytearray(136)
    struct.pack_into("<i", status, 24, pid)
    status[40:46] = b"python"
    note = struct.pack("<III", 5, len(status), 3) + b"CORE\0\0\0\0" + bytes(status)
    offset = HEADER.size + PROGRAM.size * 2
    header = HEADER.pack(b"\x7fELF\x02\x01\x01" + bytes(9), 4, 62, 1, 0, HEADER.size, 0, 0,
                         HEADER.size, PROGRAM.size, 2, 0, 0, 0)
    table = PROGRAM.pack(4, 0, offset, 0, 0, len(note), 0, 4) + PROGRAM.pack(
        1, 6, offset + len(note), start, 0, len(payload), len(payload), 4096)
    return header + table + note + payload


def test_an_elf_core_keeps_high_user_load_segments_and_reads_them():
    payload = b"high".ljust(PAGE, b"\0")
    core = CoreLayer(_core(HIGH_VA, payload))
    assert [(s.start, s.memory_end) for s in core.segments] == [(HIGH_VA, HIGH_VA + PAGE)]
    mem = ProcessMemory.from_layer(core.pid, core, core.regions, None, image_os="linux")
    assert mem.user_limit == 1 << 56
    assert mem.read(HIGH_VA, 4) == b"high"
    # A kernel-half segment (vsyscall) is still not user space: a core holding only that has no range table.
    with pytest.raises(ValueError, match="virtual range table"):
        CoreLayer(_core(0xFFFF_FFFF_FF60_0000, payload))


CHUNK = 16 << 20  # ProcessMemory.chunks' default window


def _straddling_tensor_memory() -> tuple[ProcessMemory, int]:
    """A CPU StorageImpl whose seven words cross the first 16 MiB chunk boundary: its vptr and refcount end the
    first window, its data pointer, deleter, ctx, device and size_bytes begin the second."""
    base = 0x100000
    storage = base + CHUNK - 16
    data_va = base + 0x20000
    raw = np.random.default_rng(5).random((1, 64, 64, 3), dtype=np.float32).tobytes()
    physical = bytearray(CHUNK + PAGE)
    physical[0x20000:0x20000 + len(raw)] = raw
    struct.pack_into("<7Q", physical, storage - base, 0x500100, 1, data_va, 0x500200, data_va, 0xFF00, len(raw))
    words = [0] * 22
    words[:3] = [0x600200, 1, storage]
    words[8] = 4
    words[9:13] = [1, 64, 64, 3]
    words[14:18] = [12288, 192, 3, 1]
    words[19:22] = [0, 12288, 6]
    struct.pack_into("<22Q", physical, 0x200, *words)
    images = [MappedImage("c10.dll", "c10.dll", 0x500000, 0x501000),
              MappedImage("torch.dll", "torch.dll", 0x600000, 0x601000)]
    return ProcessMemory(1, [(base, len(physical), 0)], bytes(physical), images=images), storage


def test_a_dataptr_straddling_a_chunk_boundary_is_found_once():
    mem, storage = _straddling_tensor_memory()
    assert [va for va, _ in mem.chunks()] == [0x100000, 0x100000 + CHUNK], "the boundary falls inside the DataPtr"
    candidates = torchmem._dataptr_candidates(mem)
    assert [addr for addr, _ in candidates] == [storage + 24]
    (tensor,) = torchmem.image_tensors(mem)
    assert tensor.storage == storage and tensor.sizes == [1, 64, 64, 3]


def test_overlapping_word_scans_yield_each_address_once():
    physical = np.arange(2 * CHUNK // 8, dtype=np.uint64).tobytes()
    mem = ProcessMemory(1, [(0x100000, len(physical), 0)], physical)
    every_word = list(mem.scan_words(lambda w: np.arange(len(w)), overlap=6))
    addresses = [addr for addr, _w, _p in every_word]
    assert len(addresses) == len(set(addresses)) == len(physical) // 8
    for addr, w, pos in every_word[CHUNK // 8 - 2: CHUNK // 8 + 8]:
        assert int(w[pos]) == (addr - 0x100000) // 8
