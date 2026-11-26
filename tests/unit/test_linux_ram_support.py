"""Unit tests for full Linux x86-64 whole-RAM support: LiME physical views, Linux page tables,
vmemmap PFN provenance & XArray page-cache ordering, private page-table ordering, and zswap recovery."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from comfytrace import bitmaps, lime, linux_memstore, linux_pfn, pageorder
from comfytrace.carve import PhysicalSource
from comfytrace.models import PageOrigin
from comfytrace.pagetable import PROTNONE, VALID, PageTables
from comfytrace.procmem import ProcessMemory

PAGE = 4096
VMEMMAP = 0xFFFF_EA00_0000_0000
POOL = 0xFFFF_8880_0000_0000
PAGE_SIZE_STRUCT = 64


# ---------------------------------------------------------------------------
# 1. LiME v1 physical view
# ---------------------------------------------------------------------------

def _make_lime(segments: list[tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for s_addr, payload in segments:
        e_addr = s_addr + len(payload) - 1
        out += lime.LIME_HEADER.pack(lime.LIME_MAGIC, 1, s_addr, e_addr, 0)
        out += payload
    return bytes(out)


def test_lime_physical_image_maps_segments_and_never_pads_holes(tmp_path):
    seg0 = b"A" * PAGE
    seg1 = b"B" * (2 * PAGE)
    path = tmp_path / "ram.lime"
    path.write_bytes(_make_lime([(0x1000, seg0), (0x10000, seg1)]))

    segs = lime.parse_lime_segments(path)
    assert segs is not None and len(segs) == 2
    assert segs[0].s_addr == 0x1000 and segs[0].e_addr == 0x1FFF
    assert segs[1].s_addr == 0x10000 and segs[1].e_addr == 0x11FFF

    import mmap
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        view = lime.LimePhysicalImage(mm, segs)
        assert len(view) == 0x12000
        assert view[0x1000:0x2000] == seg0
        assert view[0x10000:0x12000] == seg1
        # Inside PCI hole: empty, never fabricated zeros
        assert view[0x2000:0x3000] == b""
        # Crossing into a hole: genuine prefix only
        assert view[0x1F00:0x2100] == b"A" * 0x100
        # PhysicalSource delegates read_before across segments without crossing holes
        src = PhysicalSource(view)
        assert src.read_prefix(0x1000, 16) == (b"A" * 16, None)
        assert src.read_before(0x10010, 64) == b"B" * 16
        blocks = list(view.iter_blocks(fh, PAGE, 64))
        assert [addr for addr, _ in blocks] == [0x1000, 0x10000, 0x10FC0, 0x11F80]


def test_lime_parser_rejects_malformed_and_stub_files(tmp_path):
    stub = tmp_path / "stub.lime"
    stub.write_bytes(b"EMiL" + bytes(4092))
    assert lime.parse_lime_segments(stub) is None

    truncated = tmp_path / "trunc.lime"
    truncated.write_bytes(lime.LIME_HEADER.pack(lime.LIME_MAGIC, 1, 0x1000, 0x2FFF, 0) + b"X" * 100)
    assert lime.parse_lime_segments(truncated) is None


# ---------------------------------------------------------------------------
# 2. Linux x86-64 PageTables (4-level and 5-level)
# ---------------------------------------------------------------------------

class ToyLinuxMemory:
    """Synthetic physical RAM with a 4-level Linux kernel page table, vmemmap, and pool."""

    def __init__(self, pages: int = 128) -> None:
        self.data = bytearray(pages * PAGE)
        self.next_page = 1
        self.kdtb = self.alloc()
        self._pt_nodes: dict[tuple[int, int, int], int] = {}
        self.layout = linux_pfn.LinuxPageLayout(
            page_size=PAGE_SIZE_STRUCT,
            flags_offset=0,
            compound_head_offset=8,
            mapping_offset=24,
            index_offset=32,
            refcount_offset=52,
            as_host_offset=0,
            as_ipages_offset=8,
            xa_head_offset=8,
            xa_node_shift_offset=0,
            xa_node_slots_offset=40,
            inode_sb_offset=0,
            inode_mode_offset=8,
            inode_dentry_offset=16,
            sb_type_offset=0,
            fstype_name_offset=0,
            dentry_parent_offset=0,
            dentry_name_len_offset=8,
            dentry_name_ptr_offset=16,
            dentry_alias_offset=24,
        )
        # Map 2 pages of vmemmap (covers PFNs 0..127) and 8 pages of kernel pool
        for i in range(2):
            self.map_va(self.kdtb, VMEMMAP + i * PAGE, self.alloc() | VALID)
        for i in range(8):
            self.map_va(self.kdtb, POOL + i * PAGE, self.alloc() | VALID)
        self.pool_cursor = POOL

    def alloc(self) -> int:
        base = self.next_page * PAGE
        self.next_page += 1
        return base

    def alloc_pool(self, size: int, align: int = 16) -> int:
        self.pool_cursor = (self.pool_cursor + align - 1) & ~(align - 1)
        va = self.pool_cursor
        self.pool_cursor += size
        return va

    def map_va(self, dtb: int, va: int, leaf: int, levels: int = 4) -> None:
        shifts = (48, 39, 30, 21) if levels == 5 else (39, 30, 21)
        base = dtb
        for shift in shifts:
            idx = (va >> shift) & 0x1FF
            key = (base, shift, idx)
            if key not in self._pt_nodes:
                child = self.alloc()
                self._pt_nodes[key] = child
                struct.pack_into("<Q", self.data, base + idx * 8, child | VALID)
            base = self._pt_nodes[key]
        struct.pack_into("<Q", self.data, base + ((va >> 12) & 0x1FF) * 8, leaf)

    def kernel_tables(self) -> PageTables:
        return PageTables(self.data, self.kdtb, os_mode="linux", levels=4)

    def put_va(self, va: int, blob: bytes) -> None:
        kt = self.kernel_tables()
        for i, b in enumerate(blob):
            where = kt.translate(va + i)
            assert where is not None, hex(va + i)
            self.data[where] = b

    def q(self, va: int, val: int) -> None:
        self.put_va(va, struct.pack("<Q", val))

    def set_page(self, pfn: int, *, refcount: int = 1, mapping: int = 0, index: int = 0,
                 compound_head: int = 0) -> None:
        buf = bytearray(PAGE_SIZE_STRUCT)
        struct.pack_into("<Q", buf, self.layout.compound_head_offset, compound_head)
        struct.pack_into("<Q", buf, self.layout.mapping_offset, mapping)
        struct.pack_into("<Q", buf, self.layout.index_offset, index)
        struct.pack_into("<i", buf, self.layout.refcount_offset, refcount)
        self.put_va(VMEMMAP + pfn * PAGE_SIZE_STRUCT, bytes(buf))

    def make_address_space(self, fs_name: str, *, mode: int = 0o100644, path_parts: tuple[str, ...] = ()) -> int:
        as_va = self.alloc_pool(64)
        inode_va = self.alloc_pool(64)
        sb_va = self.alloc_pool(32)
        fstype_va = self.alloc_pool(32)
        fsname_va = self.alloc_pool(len(fs_name) + 1)
        self.put_va(fsname_va, fs_name.encode() + b"\0")
        self.q(fstype_va + self.layout.fstype_name_offset, fsname_va)
        self.q(sb_va + self.layout.sb_type_offset, fstype_va)
        self.q(inode_va + self.layout.inode_sb_offset, sb_va)
        self.put_va(inode_va + self.layout.inode_mode_offset, struct.pack("<H", mode))
        self.q(as_va + self.layout.as_host_offset, inode_va)

        if path_parts:
            root_dentry = self.alloc_pool(64)
            self.q(root_dentry + self.layout.dentry_parent_offset, root_dentry)
            parent = root_dentry
            for part in path_parts:
                d = self.alloc_pool(64)
                name_va = self.alloc_pool(len(part) + 1)
                self.put_va(name_va, part.encode() + b"\0")
                self.q(d + self.layout.dentry_parent_offset, parent)
                self.put_va(d + self.layout.dentry_name_len_offset, struct.pack("<I", len(part)))
                self.q(d + self.layout.dentry_name_ptr_offset, name_va)
                parent = d
            # inode->i_dentry.first points to d_alias inside the leaf dentry
            self.q(inode_va + self.layout.inode_dentry_offset, parent + self.layout.dentry_alias_offset)
        return as_va

    def attach_xarray(self, as_va: int, index_to_pfn: dict[int, int]) -> None:
        node = self.alloc_pool(600, align=64)
        self.put_va(node + self.layout.xa_node_shift_offset, bytes([0]))
        for idx, pfn in index_to_pfn.items():
            page_ptr = VMEMMAP + pfn * PAGE_SIZE_STRUCT
            self.q(node + self.layout.xa_node_slots_offset + idx * 8, page_ptr)
        # XArray internal node pointer has bit 1 set (entry = node | 2)
        self.q(as_va + self.layout.as_ipages_offset + self.layout.xa_head_offset, node | 2)


def test_linux_pagetables_present_protnone_and_ignores_windows_transition_bits():
    from comfytrace.pagetable import PFN_MASK

    toy = ToyLinuxMemory()
    pdtb = toy.alloc()
    f1, f2 = toy.alloc(), toy.alloc()
    toy.map_va(pdtb, 0x400000, f1 | VALID)
    # Linux 4.19+ L1TF mitigation stores non-present PROT_NONE PFNs bit-inverted.
    toy.map_va(pdtb, 0x401000, ((~f2) & PFN_MASK) | PROTNONE)
    # A Linux swap PTE with bits 10/11 set must NOT be treated as resident
    swap_pte = linux_memstore.encode_linux_swap_pte(1, 0x55)
    toy.map_va(pdtb, 0x402000, swap_pte)

    pt = PageTables(toy.data, pdtb, os_mode="linux", levels=4)
    assert pt.translate(0x400123) == f1 + 0x123
    assert pt.translate(0x401456) == f2 + 0x456
    assert pt.translate(0x402000) is None
    assert [(va, kind) for va, _fr, kind, _p in pt.resident_entries(0x400000, 0x403000)] == [
        (0x400000, "VALID"),
        (0x401000, "PROTNONE"),
    ]


def test_linux_xarray_two_level_single_root_and_folio_tail_verification():
    toy = ToyLinuxMemory(pages=160)
    # 1. Single-entry root: serves index 0 only, never index >= 1
    single_as = toy.make_address_space("ext4", path_parts=("tmp", "single.bin"))
    toy.set_page(40, refcount=1, mapping=single_as, index=0)
    toy.set_page(41, refcount=1, mapping=single_as, index=1)  # unrelated neighbor, NOT a folio tail of 40
    toy.q(single_as + toy.layout.as_ipages_offset + toy.layout.xa_head_offset, VMEMMAP + 40 * PAGE_SIZE_STRUCT)

    # 2. Two-level XArray (root shift=6 -> child shift=0) for index 100 (slot 1 -> slot 36)
    multi_as = toy.make_address_space("ext4", path_parts=("tmp", "large.bin"))
    toy.set_page(55, refcount=1, mapping=multi_as, index=100)
    toy.set_page(56, refcount=0, compound_head=(VMEMMAP + 55 * PAGE_SIZE_STRUCT) | 1)  # valid folio tail (index 101)
    root_node = toy.alloc_pool(600, align=64)
    child_node = toy.alloc_pool(600, align=64)
    toy.put_va(root_node + toy.layout.xa_node_shift_offset, bytes([6]))
    toy.put_va(child_node + toy.layout.xa_node_shift_offset, bytes([0]))
    toy.q(root_node + toy.layout.xa_node_slots_offset + 1 * 8, child_node | 2)
    toy.q(child_node + toy.layout.xa_node_slots_offset + 36 * 8, VMEMMAP + 55 * PAGE_SIZE_STRUCT)
    # Sibling entry at slot 37 pointing back to canonical slot 36 (36 << 2 | 2)
    toy.q(child_node + toy.layout.xa_node_slots_offset + 37 * 8, (36 << 2) | 2)
    toy.q(multi_as + toy.layout.as_ipages_offset + toy.layout.xa_head_offset, root_node | 2)

    # 3. Movable non-file page (mapping & 2 set) must be PRIVATE, never FILE_*
    toy.set_page(62, refcount=1, mapping=0xFFFF_8880_0000_2002)

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert support.xarray_lookup(single_as, 0) == 40
    assert support.xarray_lookup(single_as, 1) is None
    assert support.xarray_lookup(multi_as, 100) == 55
    assert support.xarray_lookup(multi_as, 101) == 56
    assert support.xarray_lookup(multi_as, 102) is None
    assert support.origin(62) == PageOrigin.PRIVATE


# ---------------------------------------------------------------------------
# 3. LinuxPfnSupport: origins, backing filenames, XArray page ordering, bitmaps
# ---------------------------------------------------------------------------

def test_linux_pfn_origins_backing_names_and_folio_tails():
    toy = ToyLinuxMemory()
    ext4_as = toy.make_address_space("ext4", mode=0o100644, path_parts=("home", "user", "out.png"))
    exec_as = toy.make_address_space("ext4", mode=0o100755, path_parts=("usr", "lib", "libfoo.so"))
    shm_as = toy.make_address_space("tmpfs", mode=0o100600)

    toy.set_page(20, refcount=0)                                    # FREED
    toy.set_page(21, refcount=1, mapping=0xFFFF_8880_0000_1231)     # PRIVATE (PAGE_MAPPING_ANON)
    toy.set_page(22, refcount=2, mapping=shm_as, index=0)           # SHARED_MEMORY
    toy.set_page(23, refcount=1, mapping=ext4_as, index=0)          # FILE_DATA (folio head)
    toy.set_page(24, refcount=0, compound_head=(VMEMMAP + 23 * PAGE_SIZE_STRUCT) | 1)  # folio tail -> FILE_DATA
    toy.set_page(25, refcount=1, mapping=exec_as, index=0)          # FILE_IMAGE

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert support.origin(20) == PageOrigin.FREED
    assert support.origin(21) == PageOrigin.PRIVATE
    assert support.origin(22) == PageOrigin.SHARED_MEMORY
    assert support.origin(23) == PageOrigin.FILE_DATA
    assert support.origin(24) == PageOrigin.FILE_DATA
    assert support.page_mapping_index(24) == (ext4_as, 1)
    assert support.origin(25) == PageOrigin.FILE_IMAGE
    assert support.mapping_problem(0x10000, 20 << 12, "VALID", None) == "PFN_FREED"
    assert support.mapping_problem(0x10000, 21 << 12, "VALID", None) is None

    co = support.copy_origin([23, 24], 8192)
    assert co == {"pages": {"FILE_DATA": 2}, "bytes": 8192, "backing_file_names": ["/home/user/out.png"]}


def test_linux_xarray_page_order_stream_reconstructs_scattered_file():
    toy = ToyLinuxMemory()
    file_as = toy.make_address_space("ext4", path_parts=("tmp", "scattered.png"))
    # Place logical pages 0, 1, 2 at non-contiguous physical PFNs 50, 30, 60
    p0, p1, p2 = 50, 30, 60
    toy.data[p0 * PAGE : (p0 + 1) * PAGE] = b"0" * PAGE
    toy.data[p1 * PAGE : (p1 + 1) * PAGE] = b"1" * PAGE
    toy.data[p2 * PAGE : (p2 + 1) * PAGE] = b"2" * PAGE
    for idx, pfn in enumerate((p0, p1, p2)):
        toy.set_page(pfn, refcount=1, mapping=file_as, index=idx)
    toy.attach_xarray(file_as, {0: p0, 1: p1, 2: p2})

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    stream = pageorder.page_order_stream(toy.data, support, p0 * PAGE + 128, 3 * PAGE)
    assert stream is not None
    raw, method, frames = stream
    assert method == "PAGE_CACHE_ORDER"
    assert frames == [p0, p1, p2]
    assert raw == b"0" * (PAGE - 128) + b"1" * PAGE + b"2" * PAGE


def test_linux_ordered_regions_orders_scattered_shmem_bitmap():
    toy = ToyLinuxMemory(pages=160)
    shm_as = toy.make_address_space("tmpfs")
    # 16 scattered pages of opaque BGRA pixels in reverse physical order
    pfns = list(range(120, 104, -1))
    index_map = {}
    for idx, pfn in enumerate(pfns):
        arr = np.full((PAGE // 4, 4), 255, dtype=np.uint8)
        arr[:, 0] = (idx * 15) & 0xFF
        toy.data[pfn * PAGE : (pfn + 1) * PAGE] = arr.tobytes()
        toy.set_page(pfn, refcount=1, mapping=shm_as, index=idx)
        index_map[idx] = pfn
    toy.attach_xarray(shm_as, index_map)

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    regions = list(bitmaps.ordered_regions(toy.data, pfns, support, min_pages=16))
    assert len(regions) == 1
    _name, data, frames, ordering, gaps = regions[0]
    assert ordering == "PAGE_CACHE_ORDER"
    assert frames == pfns
    assert not gaps and len(data) == 16 * PAGE


def _fill(toy: ToyLinuxMemory, pfn: int, payload: bytes) -> None:
    assert len(payload) == PAGE
    toy.data[pfn * PAGE : (pfn + 1) * PAGE] = payload


def _two_level_folio(toy: ToyLinuxMemory) -> int:
    """Two-level XArray: head PFN 55 at index 100 (slot 1 -> 36), sibling slot 37 for tail index 101 (PFN 56)."""
    file_as = toy.make_address_space("ext4", path_parts=("tmp", "folio.bin"))
    toy.set_page(55, refcount=1, mapping=file_as, index=100)
    toy.set_page(56, refcount=0, compound_head=(VMEMMAP + 55 * PAGE_SIZE_STRUCT) | 1)
    root_node = toy.alloc_pool(600, align=64)
    child_node = toy.alloc_pool(600, align=64)
    toy.put_va(root_node + toy.layout.xa_node_shift_offset, bytes([6]))
    toy.put_va(child_node + toy.layout.xa_node_shift_offset, bytes([0]))
    toy.q(root_node + toy.layout.xa_node_slots_offset + 1 * 8, child_node | 2)
    toy.q(child_node + toy.layout.xa_node_slots_offset + 36 * 8, VMEMMAP + 55 * PAGE_SIZE_STRUCT)
    toy.q(child_node + toy.layout.xa_node_slots_offset + 37 * 8, (36 << 2) | 2)
    toy.q(file_as + toy.layout.as_ipages_offset + toy.layout.xa_head_offset, root_node | 2)
    return file_as


@pytest.mark.parametrize("slot", ["absent", "other_live_pfn"])
def test_linux_stream_rejects_unverified_cache_seed(slot):
    toy = ToyLinuxMemory()
    file_as = toy.make_address_space("ext4", path_parts=("tmp", "stale.bin"))
    stale, live, nxt = 70, 71, 72
    for pfn, ch in ((stale, b"S"), (live, b"L"), (nxt, b"N")):
        _fill(toy, pfn, ch * PAGE)
    # The stale seed still claims (mapping, 5) and is referenced (refcount > 0).
    toy.set_page(stale, refcount=1, mapping=file_as, index=5)
    toy.set_page(live, refcount=1, mapping=file_as, index=5)
    toy.set_page(nxt, refcount=1, mapping=file_as, index=6)
    toy.attach_xarray(file_as, {6: nxt} if slot == "absent" else {5: live, 6: nxt})

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert support.page_mapping_index(stale) == (file_as, 5)
    assert pageorder.page_order_stream(toy.data, support, stale * PAGE + 128, 3 * PAGE) is None


def test_linux_stream_accepts_verified_folio_tail_seed():
    toy = ToyLinuxMemory()
    file_as = _two_level_folio(toy)
    payload = bytes(range(256)) * (PAGE // 256)
    _fill(toy, 55, b"H" * PAGE)
    _fill(toy, 56, payload)

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert support.xarray_lookup(file_as, 102) is None
    stream = pageorder.page_order_stream(toy.data, support, 56 * PAGE + 128, 3 * PAGE)
    assert stream is not None
    raw, method, frames = stream
    assert raw == payload[128:]
    assert frames == [56]
    assert method == "PAGE_CACHE_ORDER"

    # Re-point the tail at a different live head that places it at index 101 too: the XArray's folio (head 55)
    # no longer owns PFN 56, so the seed is unverified.
    toy.set_page(54, refcount=1, mapping=file_as, index=99)
    toy.set_page(56, refcount=0, compound_head=(VMEMMAP + 54 * PAGE_SIZE_STRUCT) | 1)
    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert support.page_mapping_index(56) == (file_as, 101)
    assert pageorder.page_order_stream(toy.data, support, 56 * PAGE + 128, 3 * PAGE) is None


def _cache_file(toy: ToyLinuxMemory, index_to_pfn: dict[int, int], payloads: dict[int, bytes]) -> int:
    file_as = toy.make_address_space("ext4", path_parts=("tmp", "cache.bin"))
    for idx, pfn in index_to_pfn.items():
        toy.set_page(pfn, refcount=1, mapping=file_as, index=idx)
    for pfn, payload in payloads.items():
        _fill(toy, pfn, payload)
    toy.attach_xarray(file_as, index_to_pfn)
    return file_as


def test_linux_regions_choose_canonical_duplicate_independent_of_input_order():
    toy = ToyLinuxMemory()
    file_as = _cache_file(toy, {0: 70, 1: 71, 2: 72},
                          {70: b"a" * PAGE, 71: b"b" * PAGE, 72: b"c" * PAGE, 80: b"X" * PAGE})
    toy.set_page(80, refcount=1, mapping=file_as, index=1)  # identical metadata, different bytes

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    expected = [(f"MAPPING-0x{file_as:X}+0", b"a" * PAGE + b"b" * PAGE + b"c" * PAGE, [70, 71, 72],
                 "PAGE_CACHE_ORDER", {})]
    for pages in ([70, 71, 80, 72], [70, 80, 71, 72], [72, 80, 80, 71, 70, 71]):
        assert list(support.ordered_regions(toy.data, pages, min_pages=3)) == expected


def test_linux_regions_stale_candidates_do_not_qualify_group():
    toy = ToyLinuxMemory()
    file_as = _cache_file(toy, {0: 70, 1: 71, 2: 72},
                          {70: b"a" * PAGE, 71: b"b" * PAGE, 72: b"c" * PAGE, 81: b"X" * PAGE})
    toy.set_page(81, refcount=1, mapping=file_as, index=1)  # stale; canonical 71 is never scanned

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    assert list(support.ordered_regions(toy.data, [70, 81, 72], min_pages=3)) == []


def test_linux_regions_fill_stale_interior_from_canonical_page():
    toy = ToyLinuxMemory()
    file_as = _cache_file(toy, {0: 70, 1: 71, 2: 72},
                          {70: b"a" * PAGE, 71: b"b" * PAGE, 72: b"c" * PAGE, 81: b"X" * PAGE})
    toy.set_page(81, refcount=1, mapping=file_as, index=1)

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    regions = list(support.ordered_regions(toy.data, [70, 81, 72], min_pages=2))
    assert regions == [(f"MAPPING-0x{file_as:X}+0", b"a" * PAGE + b"b" * PAGE + b"c" * PAGE, [70, 71, 72],
                        "PAGE_CACHE_ORDER", {})]


@pytest.mark.parametrize("hole", ["missing_slot", "truncated_payload"])
def test_linux_regions_split_on_unreadable_cache_page(hole):
    toy = ToyLinuxMemory()
    index_to_pfn = {0: 70, 1: 71, 3: 73, 4: 74}
    payloads = {pfn: bytes([0x41 + idx]) * PAGE for idx, pfn in index_to_pfn.items()}
    physical: object = toy.data
    if hole == "truncated_payload":
        index_to_pfn[2] = 90
        payloads[90] = b"T" * PAGE
    file_as = _cache_file(toy, index_to_pfn, payloads)
    if hole == "truncated_payload":
        physical = bytes(toy.data[: 90 * PAGE + 100])

    support = linux_pfn.LinuxPfnSupport(toy.kernel_tables(), VMEMMAP, toy.layout)
    regions = list(support.ordered_regions(physical, [70, 71, 73, 74], min_pages=2))
    assert regions == [
        (f"MAPPING-0x{file_as:X}+0", payloads[70] + payloads[71], [70, 71], "PAGE_CACHE_ORDER", {}),
        (f"MAPPING-0x{file_as:X}+3", payloads[73] + payloads[74], [73, 74], "PAGE_CACHE_ORDER", {}),
    ]


# ---------------------------------------------------------------------------
# 4. Linux compressed-memory swap (zswap) & codecs
# ---------------------------------------------------------------------------

def test_linux_swap_pte_encoding_and_lzo1x_codec():
    pte = linux_memstore.encode_linux_swap_pte(3, 0x12345)
    assert not (pte & VALID) and not (pte & PROTNONE)
    assert linux_memstore.decode_linux_swap_pte(pte) == (3, 0x12345)
    assert linux_memstore.decode_linux_swap_pte(0x5000 | VALID) is None
    assert linux_memstore.decode_linux_swap_pte(0x5000 | PROTNONE) is None

    # Minimal valid LZO1X stream: literal "ABCD" (t = 17 + 4 = 21) + M4 EOS (0x11, 0x00, 0x00)
    out, used = linux_memstore.lzo1x_decompress(bytes([21]) + b"ABCD" + bytes([0x11, 0x00, 0x00]), out_size=4)
    assert out == b"ABCD" and used == 8
    with pytest.raises(linux_memstore.Corrupt):
        linux_memstore.lzo1x_decompress(bytes([21]) + b"AB", out_size=4)


def test_linux_store_pages_read_as_process_memory():
    toy = ToyLinuxMemory()
    pdtb = toy.alloc()
    res_frame = toy.alloc()
    toy.data[res_frame : res_frame + PAGE] = b"R" * PAGE
    toy.map_va(pdtb, 0x10000, res_frame | VALID)
    toy.map_va(pdtb, 0x11000, linux_memstore.encode_linux_swap_pte(2, 7))
    toy.map_va(pdtb, 0x12000, linux_memstore.encode_linux_swap_pte(2, 8))

    class FakeLinuxStore:
        def iter_ptes(self, tables, lo, hi):
            for va, entry in tables.leaf_entries(lo, hi):
                dec = linux_memstore.decode_linux_swap_pte(entry)
                if dec is not None and dec[0] == 2:
                    yield va, dec

        def page(self, key):
            return ("OK_LZO", b"Z" * PAGE) if key == (2, 7) else ("KEY_NOT_IN_STORE", None)

    pt = PageTables(toy.data, pdtb, os_mode="linux", levels=4)
    mem = ProcessMemory.from_page_tables(42, pt, [(0x10000, 0x12FFF, None)], toy.data,
                                         store=FakeLinuxStore(), image_os="linux")
    assert mem.read(0x10FFE, 4) == b"RRZZ"
    assert mem.physical_of(0x10000) == res_frame and mem.physical_of(0x11000) is None
    assert mem.from_store(0x11000) and not mem.from_store(0x10000)
    assert mem.try_read(0x12000, 8) is None
    assert mem.store_status == {"OK_LZO": 1, "KEY_NOT_IN_STORE": 1}


def test_linux_physical_fallback_forwards_walker_for_surviving_process_page_tables(tmp_path, monkeypatch):
    from comfytrace import ingest

    dump = tmp_path / "terminated.raw"
    dump.write_bytes(bytes(2 * PAGE))
    seen: dict[str, object] = {}

    def fake_linux_support(context, kernel_module_name, recovery, walker=None):
        seen["walker"] = walker
        recovery.notes["bitmap_ordering"] = "PAGE_CACHE_ORDER"

    monkeypatch.setattr(ingest, "linux_kernel_support", fake_linux_support)
    sentinel_walker = object()
    result = ingest.physical_ingestion(
        dump, {"image_os": "linux", "status": "SUCCESS"}, "NO_COMFYUI_PROCESS",
        context=object(), kernel_module_name="kernel", walker=sentinel_walker,
    )
    try:
        assert seen["walker"] is sentinel_walker
        assert result.mode == ingest.MODE_PHYSICAL
    finally:
        result.close()
