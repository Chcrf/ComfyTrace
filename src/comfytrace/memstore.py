"""Pages in the Windows memory-compression store, read from the raw image.

A trimmed private page of a process may be compressed into the store instead of being written to pagefile.sys.
Its PTE is then a pagefile-format PTE whose PageFileLow is the partition's virtual store pagefile
(``_MI_PARTITION.Store.VirtualPageFileNumber``) and whose PageFileHigh is the store key. The compressed bytes
live in the MemCompression process. This reader follows the key through the store manager's B+trees to the
compressed record and decompresses it (LZ4 on build 26100; plain XPRESS kept for older stores).

None of the store structures are in the public symbols. The offsets in PROFILE were measured on build 26100
(T2 and run 2, work/fable-answers.md M5-M7, independently rechecked on run 1, T95) and are used only when the
image passes the layout checks in ``StoreReader.open``: the manager's pool tag is ``SmPa``, store 0 is owned by
MemCompression, and its region mask and shift are 0x1FFF and 13. A page is accepted only when decompression
consumes exactly the recorded compressed size and yields exactly 4096 bytes. Records of an image acquired while
the store was changing can still resolve to stale data, so every page read here is labelled STORE_DECODED.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from typing import Any

from comfytrace.pagetable import PageTables

log = logging.getLogger(__name__)

PAGE = 4096
PROFILE = {
    "partition_list": 0x10,       # SmGlobals + 0x10: LIST_ENTRY of per-partition store managers
    "manager_link": 0xC30,        # manager = Flink - 0xC30
    "manager_stores": 0x000,      # 32 pointers, each to 32 metadata entries of 0x28 bytes; entry + 0 = store
    "manager_key_tree": 0x1C0,    # {root, count}; key = PageFileHigh, value & 0x3FF = store index
    "metadata_entry": 0x28,
    "store_page_tree": 0x50,      # {root, count}; value = chunk key
    "store_chunk_arrays": 0x110,  # 32 pointers; array i holds 2**i entries of 16 bytes
    "store_chunk_shift": 0x218,
    "store_chunk_mask": 0x21C,
    "store_record_size": 0x220,
    "store_chunk_header": 0x228,
    "store_region_mask": 0x378,
    "store_region_shift": 0x37C,
    "store_regions": 0x1B68,      # pointer to the user VAs of 128 KiB regions in MemCompression
    "store_owner": 0x1D00,        # EPROCESS of MemCompression
}


class Corrupt(Exception):
    """Bytes are readable but do not form the expected structure."""


def lz4_block(src: bytes, out_size: int = PAGE) -> tuple[bytes, int]:
    """LZ4 block format. Returns (output, input bytes consumed). Raises Corrupt on any malformed sequence."""
    out = bytearray()
    ip, n = 0, len(src)
    while ip < n:
        token = src[ip]
        ip += 1
        literals = token >> 4
        if literals == 15:
            while True:
                if ip >= n:
                    raise Corrupt("lz4: literal length runs past input")
                literals += src[ip]
                ip += 1
                if src[ip - 1] != 255:
                    break
        if ip + literals > n or len(out) + literals > out_size:
            raise Corrupt("lz4: literals run past input or output")
        out += src[ip : ip + literals]
        ip += literals
        if ip == n:
            break
        if ip + 2 > n:
            raise Corrupt("lz4: offset runs past input")
        offset = src[ip] | src[ip + 1] << 8
        ip += 2
        length = token & 15
        if length == 15:
            while True:
                if ip >= n:
                    raise Corrupt("lz4: match length runs past input")
                length += src[ip]
                ip += 1
                if src[ip - 1] != 255:
                    break
        length += 4
        if offset == 0 or offset > len(out) or len(out) + length > out_size:
            raise Corrupt("lz4: match outside output")
        for _ in range(length):
            out.append(out[-offset])
    return bytes(out), ip


def xpress_lz77(src: bytes, out_size: int = PAGE) -> tuple[bytes, int]:
    """[MS-XCA] 2.4 plain LZ77 (COMPRESSION_FORMAT_XPRESS)."""
    out = bytearray()
    ip, n, flags, nflags, half = 0, len(src), 0, 0, 0
    while len(out) < out_size:
        if nflags == 0:
            if ip + 4 > n:
                raise Corrupt("xpress: flags run past input")
            flags = int.from_bytes(src[ip : ip + 4], "little")
            ip += 4
            nflags = 32
        nflags -= 1
        if not flags >> nflags & 1:
            if ip >= n:
                raise Corrupt("xpress: literal runs past input")
            if len(out) >= out_size:
                raise Corrupt("xpress: literal past output size")
            out.append(src[ip])
            ip += 1
            continue
        if ip == n:
            break
        if ip + 2 > n:
            raise Corrupt("xpress: match runs past input")
        word = src[ip] | src[ip + 1] << 8
        ip += 2
        length, offset = word & 7, (word >> 3) + 1
        if length == 7:
            if half == 0:
                if ip >= n:
                    raise Corrupt("xpress: length runs past input")
                length, half = src[ip] & 15, ip
                ip += 1
            else:
                length, half = src[half] >> 4, 0
            if length == 15:
                if ip >= n:
                    raise Corrupt("xpress: length runs past input")
                length = src[ip]
                ip += 1
                if length == 255:
                    if ip + 2 > n:
                        raise Corrupt("xpress: length runs past input")
                    length = src[ip] | src[ip + 1] << 8
                    ip += 2
                    if length == 0:
                        if ip + 4 > n:
                            raise Corrupt("xpress: length runs past input")
                        length = int.from_bytes(src[ip : ip + 4], "little")
                        ip += 4
                    if length < 22:
                        raise Corrupt("xpress: long length too small")
                    length -= 22
                length += 15
            length += 7
        length += 3
        if offset > len(out):
            raise Corrupt("xpress: match before output start")
        if len(out) + length > out_size:
            raise Corrupt("xpress: match past output size")
        for _ in range(length):
            out.append(out[-offset])
    return bytes(out), ip


class StoreReader:
    """Resolves store keys of one partition's store manager to page contents."""

    def __init__(self, kernel: PageTables, manager: int, pagefile: int, profile: dict[str, int]) -> None:
        self.kernel, self.manager, self.pagefile, self.p = kernel, manager, pagefile, profile
        self.key_root = self._q(kernel, manager + profile["manager_key_tree"]) or 0
        self._stores: dict[int, dict[str, Any] | None] = {}

    @classmethod
    def open(cls, kernel: PageTables, smglobals: int, pagefile: int, owner_name: Any,
             profile: dict[str, int] = PROFILE) -> StoreReader:
        """Raises Corrupt when the image fails a layout check. ``owner_name(eprocess)`` returns the image name."""
        flink = kernel.read_kernel_qword(smglobals + profile["partition_list"])
        if not flink:
            raise Corrupt("SmGlobals partition list unreadable")
        manager = flink - profile["manager_link"]
        where = kernel.translate(manager - 12)
        if where is None or bytes(kernel.physical[where : where + 4]) != b"SmPa":
            raise Corrupt("store manager pool tag is not SmPa (unknown store layout)")
        reader = cls(kernel, manager, pagefile, profile)
        store = reader.store(0)
        if store is None or not str(owner_name(store["owner"]) or "").startswith("MemCompression"):
            raise Corrupt("store 0 is not owned by MemCompression")
        if (store["region_mask"], store["region_shift"]) != (0x1FFF, 13):
            raise Corrupt("store region mask or shift differs from the measured layout")
        return reader

    @staticmethod
    def _q(tables: PageTables, va: int) -> int | None:
        raw = _read(tables, va, 8)
        return None if raw is None else int.from_bytes(raw, "little")

    def _d(self, va: int) -> int | None:
        raw = _read(self.kernel, va, 4)
        return None if raw is None else int.from_bytes(raw, "little")

    def _find(self, root: int, key: int) -> int | None:
        """B+tree lookup. Node: {u16 count, u8 level, u8 leaf, pad, u64 leftmost child}, entries from +0x10;
        leaf entry {u32 key, u32 value}, interior entry {u32 key, u32 pad, u64 child}."""
        node = root
        for _ in range(8):
            raw = _read(self.kernel, node, PAGE)
            if raw is None:
                raise Corrupt("tree node not resident")
            count, _level, leaf = struct.unpack_from("<HBB", raw, 0)
            if leaf > 1 or count > (510 if leaf else 255):
                raise Corrupt("tree node malformed")
            if leaf:
                for i in range(count):
                    k, v = struct.unpack_from("<II", raw, 0x10 + 8 * i)
                    if k == key:
                        return int(v)
                return None
            node = struct.unpack_from("<Q", raw, 8)[0]
            for i in range(count):
                k, _pad, child = struct.unpack_from("<IIQ", raw, 0x10 + 16 * i)
                if key < k:
                    break
                node = child
        raise Corrupt("tree deeper than 8 levels")

    def store(self, index: int) -> dict[str, Any] | None:
        if index not in self._stores:
            p, k = self.p, self.kernel
            array = self._q(k, self.manager + p["manager_stores"] + 8 * (index >> 5))
            base = self._q(k, array + p["metadata_entry"] * (index & 31)) if array else None
            info = None
            if base:
                owner = self._q(k, base + p["store_owner"]) or 0
                dtb = self._q(k, owner + 0x28) if owner else None  # KPROCESS.DirectoryTableBase
                info = {"base": base, "page_root": self._q(k, base + p["store_page_tree"]) or 0,
                        "chunk_shift": (self._d(base + p["store_chunk_shift"]) or 0) & 0xFF,
                        "chunk_mask": self._d(base + p["store_chunk_mask"]) or 0,
                        "record_size": self._d(base + p["store_record_size"]) or 0,
                        "chunk_header": self._d(base + p["store_chunk_header"]) or 0,
                        "region_mask": self._d(base + p["store_region_mask"]) or 0,
                        "region_shift": (self._d(base + p["store_region_shift"]) or 0) & 0xFF,
                        "regions": self._q(k, base + p["store_regions"]) or 0, "owner": owner,
                        "owner_tables": PageTables(k.physical, dtb) if dtb else None}
            self._stores[index] = info
        return self._stores[index]

    def page(self, key: int) -> tuple[str, bytes | None]:
        """(status, 4096 bytes or None). Only statuses starting with OK carry bytes."""
        try:
            value = self._find(self.key_root, key)
        except Corrupt as exc:
            return f"KEY_TREE_UNREADABLE: {exc}", None
        if value is None:
            return "KEY_NOT_IN_STORE_MANAGER", None
        s = self.store(value & 0x3FF)
        if s is None or s["owner_tables"] is None:
            return "STORE_MISSING", None
        try:
            chunk_key = self._find(s["page_root"], key)
        except Corrupt as exc:
            return f"PAGE_TREE_UNREADABLE: {exc}", None
        if chunk_key is None:
            return "KEY_NOT_IN_STORE", None
        number = (chunk_key >> s["chunk_shift"]) + 1  # the chunk's header repeats this number
        array_index = number.bit_length() - 1
        array = self._q(self.kernel, s["base"] + self.p["store_chunk_arrays"] + 8 * array_index)
        chunk = self._q(self.kernel, array + 16 * (number ^ (1 << array_index))) if array else None
        if not chunk:
            return "CHUNK_MISSING", None
        header = _read(self.kernel, chunk, s["chunk_header"])
        if header is None or len(header) < 24 or struct.unpack_from("<Q", header, 16)[0] != number:
            return "CHUNK_HEADER_MISMATCH", None
        record = _read(self.kernel, chunk + s["chunk_header"] + (chunk_key & s["chunk_mask"]) * s["record_size"], 12)
        if record is None:
            return "RECORD_UNREADABLE", None
        region_key, size_flags, _word, _check = struct.unpack("<IHHI", record)
        if region_key == 0xFFFFFFFF:
            return "RECORD_FREE", None
        size = size_flags & 0xFFF or PAGE  # 0 means stored uncompressed
        region = self._q(self.kernel, s["regions"] + 8 * (region_key >> s["region_shift"]))
        if not region:
            return "REGION_POINTER_NULL", None
        data = _read(s["owner_tables"], region + ((region_key & s["region_mask"]) << 4), size)
        if data is None:
            return "COMPRESSED_BYTES_NOT_RESIDENT", None
        if size == PAGE:
            return "OK_STORED_RAW", data
        for name, decoder in (("LZ4", lz4_block), ("XPRESS", xpress_lz77)):
            try:
                out, used = decoder(data)
            except Corrupt:
                continue
            if len(out) == PAGE and used == size:
                return f"OK_{name}", out
        return "DECOMPRESS_FAILED", None


def _read(tables: PageTables, va: int, length: int) -> bytes | None:
    out = bytearray()
    while length:
        where = tables.translate(va)
        take = min(length, PAGE - (va & 0xFFF))
        if where is None or where + take > tables.size:
            return None
        out += tables.physical[where : where + take]
        va, length = va + take, length - take
    return bytes(out)


def store_ptes(tables: PageTables, lo: int, hi: int, pagefile: int) -> Iterator[tuple[int, int]]:
    """(virtual page, store key) for every PTE in [lo, hi) naming the virtual store pagefile. Entries with bits
    16-31 set are not PTEs (stale page-table bytes) and are skipped."""
    for va, entry in tables.leaf_entries(lo, hi):
        if (entry & 0xC01) == 0 and (entry >> 12) & 0xF == pagefile and entry >> 32 and not (entry >> 16) & 0xFFFF:
            yield va, entry >> 32
