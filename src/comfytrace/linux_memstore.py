"""Linux x86-64 swap-PTE decoding and compressed-page codecs (LZ4, LZO1X/LZO-RLE, ZSTD).

When Linux swaps an anonymous page, its non-present x86-64 PTE encodes ``(swp_type, swp_offset)`` with L1TF
offset bit-inversion::

    swp_type   = (pte >> 59) & 0x1F
    swp_offset = (((~pte & 0xFFFFFFFFFFFFFFFF) << 5) & 0xFFFFFFFFFFFFFFFF) >> 14

Live ``zswap`` ``zsmalloc`` handle resolution is conservatively disabled in v1.0.0 pending full per-kernel
``zspage`` / size-class validation so unverified frames can never be attributed to process memory.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from comfytrace.linux_pfn import KERNEL_MIN_4, KERNEL_MIN_5
from comfytrace.memstore import Corrupt, lz4_block
from comfytrace.pagetable import PROTNONE, VALID, PageTables

log = logging.getLogger(__name__)

PAGE = 4096
SWP_TYPE_BITS = 5
SWP_OFFSET_FIRST_BIT = 9  # _PAGE_BIT_PROTNONE (8) + 1
SWP_OFFSET_SHIFT = SWP_OFFSET_FIRST_BIT + SWP_TYPE_BITS  # 14
MAX_SWAPFILES = 32
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def decode_linux_swap_pte(pte: int) -> tuple[int, int] | None:
    """Decode a non-present Linux x86-64 swap PTE into ``(swp_type, swp_offset)``, or None if not a swap PTE."""
    if not pte or (pte & VALID) or (pte & PROTNONE):
        return None
    swp_type = (pte >> (64 - SWP_TYPE_BITS)) & ((1 << SWP_TYPE_BITS) - 1)
    swp_offset = (((~pte & MASK64) << SWP_TYPE_BITS) & MASK64) >> SWP_OFFSET_SHIFT
    if swp_type >= MAX_SWAPFILES:
        return None
    return swp_type, swp_offset


def encode_linux_swap_pte(swp_type: int, swp_offset: int) -> int:
    """Encode ``(swp_type, swp_offset)`` into a Linux x86-64 swap PTE (inverse of ``decode_linux_swap_pte``)."""
    inv = ((~swp_offset & MASK64) << SWP_OFFSET_SHIFT) & MASK64
    return (inv >> SWP_TYPE_BITS) | ((swp_type & 0x1F) << (64 - SWP_TYPE_BITS))


def lzo1x_decompress(src: bytes, out_size: int = PAGE) -> tuple[bytes, int]:
    """Bounds-checked LZO1X / Linux LZO-RLE block decompressor.

    Returns ``(output_bytes, input_bytes_consumed)`` or raises ``Corrupt`` on any malformed or out-of-bounds
    stream. Implements Linux kernel ``lib/lzo/lzo1x_decompress_safe.c`` including LZO-RLE zero runs.
    """
    n = len(src)
    if n < 3:
        raise Corrupt("lzo: input too short")
    ip = 0
    out = bytearray()

    def copy_literals(count: int) -> None:
        nonlocal ip
        if count <= 0 or ip + count > n or len(out) + count > out_size:
            raise Corrupt("lzo: literal out of bounds")
        out.extend(src[ip : ip + count])
        ip += count

    def copy_match(dist: int, length: int) -> None:
        if dist <= 0 or dist > len(out) or len(out) + length > out_size:
            raise Corrupt("lzo: match out of bounds")
        for _ in range(length):
            out.append(out[-dist])

    def read_extended(base: int) -> int:
        nonlocal ip
        val = base
        while True:
            if ip >= n:
                raise Corrupt("lzo: truncated extended length")
            b = src[ip]
            ip += 1
            if b != 0:
                return val + b
            val += 255
            if val > out_size:
                raise Corrupt("lzo: extended length exceeds output")

    # Initial token handling
    t = src[ip]
    ip += 1
    first_literal_run = False
    match_next = False
    bitstream_version = 0
    if t > 17:
        t -= 17
        copy_literals(t)
        if t < 4:
            match_next = True
        else:
            first_literal_run = True
    elif t == 17:
        # LZO-RLE header: (17, version) with version >= 1
        if ip < n and src[ip] >= 1:
            bitstream_version = src[ip]
            ip += 1
            if ip >= n:
                raise Corrupt("lzo-rle: truncated after header")
            t = src[ip]
            ip += 1
            if t > 17:
                t -= 17
                copy_literals(t)
                if t < 4:
                    match_next = True
                else:
                    first_literal_run = True
            else:
                ip -= 1
        else:
            ip -= 1
    else:
        ip -= 1

    while True:
        if first_literal_run:
            first_literal_run = False
            if ip >= n:
                raise Corrupt("lzo: unexpected EOF")
            t = src[ip]
            ip += 1
            if t < 16:
                if ip >= n:
                    raise Corrupt("lzo: truncated M1 match")
                dist = 1 + 0x0800 + (t >> 2) + (src[ip] << 2)
                ip += 1
                copy_match(dist, 3)
                state = t & 3
                if state:
                    copy_literals(state)
                    match_next = True
                continue
        elif not match_next:
            if ip >= n:
                raise Corrupt("lzo: unexpected EOF")
            t = src[ip]
            ip += 1
            if t < 16:
                if t == 0:
                    t = read_extended(15)
                copy_literals(t + 3)
                first_literal_run = True
                continue
        else:
            match_next = False
            if ip >= n:
                raise Corrupt("lzo: unexpected EOF")
            t = src[ip]
            ip += 1

        # Match loop
        while True:
            if t >= 64:
                if ip >= n:
                    raise Corrupt("lzo: truncated M2 match")
                dist = 1 + ((t >> 2) & 7) + (src[ip] << 3)
                ip += 1
                length = (t >> 5) - 1 + 2
                copy_match(dist, length)
            elif t >= 32:
                length = t & 31
                if length == 0:
                    length = read_extended(31)
                if ip + 2 > n:
                    raise Corrupt("lzo: truncated M3 match")
                m_lo = src[ip]
                m_hi = src[ip + 1]
                ip += 2
                dist = 1 + (m_lo >> 2) + (m_hi << 6)
                copy_match(dist, length + 2)
                t = m_lo
            elif t >= 16:
                if bitstream_version and (t & 0xF8) == 0x18 and ip + 2 <= n \
                        and ((src[ip] | src[ip + 1] << 8) & 0xFFFC) == 0xFFFC:
                    # LZO-RLE zero run (checked before any extended length, as the kernel does):
                    # 0b00011xxx, then le16 with its top 14 bits set, then a byte holding run-length bits 3-10.
                    if ip + 3 > n:
                        raise Corrupt("lzo-rle: truncated zero run")
                    zero_count = ((t & 7) | (src[ip + 2] << 3)) + 4  # MIN_ZERO_RUN_LENGTH
                    if len(out) + zero_count > out_size:
                        raise Corrupt("lzo-rle: zero run exceeds output")
                    out.extend(bytes(zero_count))
                    t = src[ip]  # its low two bits are the trailing-literal state
                    ip += 3
                else:
                    dist = (t & 8) << 11
                    length = t & 7
                    if length == 0:
                        length = read_extended(7)
                    if ip + 2 > n:
                        raise Corrupt("lzo: truncated M4 match")
                    m_lo = src[ip]
                    m_hi = src[ip + 1]
                    ip += 2
                    dist += (m_lo >> 2) + (m_hi << 6)
                    if dist == 0:
                        if length == 1:
                            # End-of-stream marker
                            return bytes(out), ip
                        raise Corrupt("lzo: invalid zero distance")
                    dist += 0x4000
                    copy_match(dist, length + 2)
                    t = m_lo
            else:
                if ip >= n:
                    raise Corrupt("lzo: truncated M1 match")
                dist = 1 + (t >> 2) + (src[ip] << 2)
                ip += 1
                copy_match(dist, 2)

            state = t & 3
            if state == 0:
                break
            copy_literals(state)
            if ip >= n:
                raise Corrupt("lzo: unexpected EOF after match literals")
            t = src[ip]
            ip += 1


def _try_zstd(src: bytes, out_size: int = PAGE) -> bytes | None:
    try:
        from compression import zstd as std_zstd  # type: ignore[import-not-found]

        decoder = std_zstd.ZstdDecompressor()
        out = decoder.decompress(src, max_length=out_size + 1)
        return out if len(out) == out_size and decoder.eof and not decoder.unused_data else None
    except Exception:  # noqa: BLE001
        pass
    try:
        import zstandard  # type: ignore[import-not-found]

        if zstandard.frame_content_size(src) not in (-1, out_size):
            return None
        out = zstandard.ZstdDecompressor().decompress(src, max_output_size=out_size, allow_extra_data=False)
        return out if len(out) == out_size else None
    except Exception:  # noqa: BLE001
        return None


def decompress_linux_page(data: bytes, codec: str | None = None) -> tuple[str, bytes | None]:
    """Decompress a Linux compressed swap page, accepting only exact 4096-byte outputs."""
    if len(data) == PAGE:
        return "OK_STORED_RAW", data
    if not data or len(data) > PAGE:
        return "DECOMPRESS_FAILED", None
    order = (codec.lower(),) if codec else ("lz4", "lzo", "zstd")
    for name in order:
        if "lz4" in name:
            try:
                out, used = lz4_block(data, PAGE)
                if len(out) == PAGE and used == len(data):
                    return "OK_LZ4", out
            except Corrupt:
                pass
        elif "lzo" in name:
            try:
                out, used = lzo1x_decompress(data, PAGE)
                if len(out) == PAGE and used == len(data):
                    return "OK_LZO", out
            except Corrupt:
                pass
        elif "zstd" in name:
            out = _try_zstd(data, PAGE)
            if out is not None and len(out) == PAGE:
                return "OK_ZSTD", out
    return "DECOMPRESS_FAILED", None


@dataclass(frozen=True)
class LinuxStoreLayout:
    """ISF-resolved offsets for ``zswap`` and ``zsmalloc`` inspection."""

    page_size: int
    page_private_offset: int
    xa_head_offset: int
    xa_node_shift_offset: int
    xa_node_slots_offset: int
    zswap_entry_length_offset: int | None = None
    zswap_entry_handle_offset: int | None = None


class LinuxStoreReader:
    """Resolves Linux x86-64 swap PTEs backed by ``zswap`` (or ``zram``) to 4096-byte pages."""

    def __init__(self, kernel: PageTables, vmemmap_base: int, layout: LinuxStoreLayout,
                 zswap_trees: dict[int, int]) -> None:
        self.kernel = kernel
        self.vmemmap_base = vmemmap_base
        self.layout = layout
        self.zswap_trees = dict(zswap_trees)
        self.active_types = frozenset(zswap_trees)
        self.pagefile = -1  # unused by Linux path; iter_ptes is used instead

    @property
    def _kernel_min(self) -> int:
        return KERNEL_MIN_5 if self.kernel.levels == 5 else KERNEL_MIN_4

    def iter_ptes(self, tables: PageTables, lo: int, hi: int) -> Iterator[tuple[int, tuple[int, int]]]:
        """Yield ``(va, (swp_type, swp_offset))`` for every swap PTE in ``[lo, hi)`` belonging to an active store."""
        for va, entry in tables.leaf_entries(lo, hi):
            decoded = decode_linux_swap_pte(entry)
            if decoded is not None and decoded[0] in self.active_types:
                yield va, decoded

    def _xarray_value(self, xarray_va: int, index: int) -> int | None:
        L = self.layout
        if index < 0:
            return None
        entry = self.kernel.read_kernel_qword(xarray_va + L.xa_head_offset)
        if not entry:
            return None
        if not ((entry & 3) == 2 and entry > 4096):
            if index != 0:
                return None
        else:
            checked_top = False
            for _ in range(12):
                if (entry & 3) == 2 and entry > 4096:
                    node = entry - 2
                    if node < self._kernel_min:
                        return None
                    shift_raw = self.kernel.read_virtual(node + L.xa_node_shift_offset, 1)
                    if shift_raw is None:
                        return None
                    shift = shift_raw[0]
                    if shift > 60:
                        return None
                    if not checked_top:
                        if (index >> shift) >= 64:
                            return None
                        checked_top = True
                    slot_idx = (index >> shift) & 63
                    entry = self.kernel.read_kernel_qword(node + L.xa_node_slots_offset + slot_idx * 8)
                    if not entry:
                        return None
                    continue
                break
        return entry if (entry and (entry & 3) == 0 and entry >= self._kernel_min) else None

    def _read_zsmalloc_handle(self, handle: int, length: int) -> bytes | None:
        """Read ``length`` compressed bytes from a ``zsmalloc`` handle (single-page or two-page object)."""
        if not handle or handle < self._kernel_min or not 0 < length <= PAGE:
            return None
        # zsmalloc handle points to a qword storing (obj_pfn << 12) | obj_offset (tagged with low bits)
        obj_loc = self.kernel.read_kernel_qword(handle)
        if not obj_loc:
            return None
        # Object location encodes (pfn << PAGE_SHIFT) | (obj_idx << ...) or direct vmemmap/phys offset
        # In modern zsmalloc: obj = (pfn << OBJ_INDEX_BITS) | obj_idx; validate conservatively
        pfn = obj_loc >> 12
        off = obj_loc & (PAGE - 1)
        phys = pfn << 12
        if phys + PAGE > self.kernel.size:
            return None
        # Skip 8-byte handle back-pointer when stored in-line
        start = off + 8 if off + 8 + length <= PAGE else off
        if start + length <= PAGE:
            chunk = bytes(self.kernel.physical[phys + start : phys + start + length])
            return chunk if len(chunk) == length else None
        return None

    def page(self, key: tuple[int, int]) -> tuple[str, bytes | None]:
        """Resolve ``(swp_type, swp_offset)`` to ``(status, 4096 bytes or None)``."""
        swp_type, swp_offset = key
        tree_va = self.zswap_trees.get(swp_type)
        if tree_va is None:
            return "SWAP_TYPE_NOT_IN_STORE", None
        L = self.layout
        if L.zswap_entry_length_offset is None or L.zswap_entry_handle_offset is None:
            return "ZSWAP_LAYOUT_UNAVAILABLE", None
        entry_va = self._xarray_value(tree_va, swp_offset)
        if entry_va is None:
            return "KEY_NOT_IN_STORE", None
        len_raw = self.kernel.read_virtual(entry_va + L.zswap_entry_length_offset, 4)
        handle = self.kernel.read_kernel_qword(entry_va + L.zswap_entry_handle_offset)
        if len_raw is None or handle is None:
            return "RECORD_UNREADABLE", None
        length = struct.unpack("<I", len_raw)[0]
        if length == 0:
            # Same-filled page (all zeros)
            return "OK_SAME_FILLED", bytes(PAGE)
        if length > PAGE:
            return "RECORD_CORRUPT", None
        compressed = self._read_zsmalloc_handle(handle, length)
        if compressed is None:
            return "COMPRESSED_BYTES_NOT_RESIDENT", None
        return decompress_linux_page(compressed)


def open_linux_store_reader(context: Any, kernel_module_name: str, kernel: PageTables
                            ) -> tuple[LinuxStoreReader | None, str]:
    """Linux compressed-swap store hook.

    Conservatively returns UNAVAILABLE in v1.0.0: full ``zsmalloc`` ``zspage``/size-class decoding and
    pre-/post-6.9 ``zswap_trees`` validation across kernel builds is deferred to v1.1 so unverified bytes can
    never be attributed to process memory.
    """
    _ = (context, kernel_module_name, kernel)
    return None, "UNAVAILABLE: Linux zswap compressed-store decoding disabled in v1.0.0 pending live validation"
