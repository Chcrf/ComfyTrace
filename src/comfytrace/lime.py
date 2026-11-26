"""Zero-copy physical address view over LiME (Linux Memory Extractor) v1 captures.

A LiME v1 file consists of 32-byte headers (magic ``EMiL``, version 1, inclusive physical start and end
addresses, reserved) followed by raw RAM bytes for each physical range. Presenting those ranges through a
sequence-like view lets page-table walkers, PFN/page-cache walkers, carvers, and raw scans operate directly on
real physical addresses without copying the capture or fabricating zeros across PCI/MMIO holes.
"""

from __future__ import annotations

import bisect
import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIME_MAGIC = 0x4C694D45  # b"EMiL" little-endian
LIME_HEADER = struct.Struct("<IIQQQ")
LIME_HEADER_SIZE = LIME_HEADER.size  # 32 bytes
PAGE = 4096


@dataclass(frozen=True)
class LimeSegment:
    """One contiguous physical RAM range inside a LiME v1 capture."""

    s_addr: int
    e_addr: int
    file_offset: int

    @property
    def length(self) -> int:
        return self.e_addr - self.s_addr + 1

    @property
    def end_exclusive(self) -> int:
        return self.e_addr + 1


def parse_lime_segments(path: Path) -> tuple[LimeSegment, ...] | None:
    """Parse and validate all LiME v1 segment headers in ``path``.

    Returns None when ``path`` is not a valid LiME v1 capture (for example, a flat raw image, another container
    format, or a malformed/truncated ``EMiL`` stub).
    """
    try:
        file_size = path.stat().st_size
        if file_size < LIME_HEADER_SIZE:
            return None
        segments: list[LimeSegment] = []
        with path.open("rb") as handle:
            pos = 0
            while pos < file_size:
                if pos + LIME_HEADER_SIZE > file_size:
                    return None
                handle.seek(pos)
                magic, version, s_addr, e_addr, _reserved = LIME_HEADER.unpack(handle.read(LIME_HEADER_SIZE))
                if magic != LIME_MAGIC or version != 1 or e_addr < s_addr:
                    return None
                length = e_addr - s_addr + 1
                data_offset = pos + LIME_HEADER_SIZE
                if data_offset + length > file_size:
                    return None
                segments.append(LimeSegment(s_addr, e_addr, data_offset))
                pos = data_offset + length
    except OSError:
        return None
    if not segments:
        return None
    segments.sort(key=lambda seg: seg.s_addr)
    for prev, nxt in zip(segments, segments[1:], strict=False):
        if prev.end_exclusive > nxt.s_addr:
            return None
    return tuple(segments)


class LimePhysicalImage:
    """Physical-address-indexed read-only view backed by an mmap of a LiME v1 file.

    Slicing ``image[start:stop]`` returns only genuine bytes from ``start`` up to the first unmapped physical
    hole (or ``stop``). Reads starting inside a hole return ``b""``. No bytes are fabricated or padded.
    """

    def __init__(self, mapped: mmap.mmap, segments: tuple[LimeSegment, ...]) -> None:
        self._mmap = mapped
        self.segments = segments
        self._starts = [seg.s_addr for seg in segments]
        self._size = segments[-1].end_exclusive

    def __len__(self) -> int:
        return self._size

    def file_offset_of(self, physical_address: int) -> int | None:
        """File offset of a single physical address, or None if it lies in a hole."""
        idx = bisect.bisect_right(self._starts, physical_address) - 1
        if idx < 0:
            return None
        seg = self.segments[idx]
        if physical_address > seg.e_addr:
            return None
        return seg.file_offset + (physical_address - seg.s_addr)

    def __getitem__(self, key: slice | int) -> Any:
        if isinstance(key, int):
            if key < 0:
                key += self._size
            off = self.file_offset_of(key)
            if off is None:
                raise IndexError(f"physical address 0x{key:x} is not mapped in LiME capture")
            return self._mmap[off]
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("LimePhysicalImage supports only contiguous slices")
        start = 0 if key.start is None else (key.start + self._size if key.start < 0 else key.start)
        stop = self._size if key.stop is None else (key.stop + self._size if key.stop < 0 else key.stop)
        if start < 0:
            start = 0
        if stop > self._size:
            stop = self._size
        if stop <= start:
            return b""
        idx = bisect.bisect_right(self._starts, start) - 1
        if idx < 0:
            return b""
        seg = self.segments[idx]
        if start > seg.e_addr:
            return b""
        if stop <= seg.end_exclusive:
            rel = start - seg.s_addr
            return self._mmap[seg.file_offset + rel : seg.file_offset + rel + (stop - start)]
        out = bytearray()
        cursor = start
        while idx < len(self.segments) and cursor < stop:
            seg = self.segments[idx]
            if cursor < seg.s_addr or cursor > seg.e_addr:
                break
            take = min(stop, seg.end_exclusive) - cursor
            rel = cursor - seg.s_addr
            out += self._mmap[seg.file_offset + rel : seg.file_offset + rel + take]
            cursor += take
            idx += 1
        return bytes(out)

    def read_before(self, offset: int, length: int) -> bytes:
        """Longest genuinely readable suffix of ``[offset - length, offset)``."""
        if offset <= 0 or length <= 0:
            return b""
        floor = max(0, offset - length)
        idx = bisect.bisect_right(self._starts, offset - 1) - 1
        if idx < 0:
            return b""
        pieces: list[bytes] = []
        cursor = offset
        while idx >= 0 and cursor > floor:
            seg = self.segments[idx]
            if cursor > seg.end_exclusive or cursor <= seg.s_addr:
                break
            lo = max(floor, seg.s_addr)
            rel = lo - seg.s_addr
            pieces.append(self._mmap[seg.file_offset + rel : seg.file_offset + rel + (cursor - lo)])
            cursor = lo
            idx -= 1
        return b"".join(reversed(pieces))

    def iter_blocks(self, handle: Any, block_size: int, overlap: int) -> Iterator[tuple[int, bytes]]:
        """Stream genuine RAM ranges in sliding windows with real physical base addresses."""
        if overlap >= block_size:
            raise ValueError("overlap must be smaller than block size")
        for seg in self.segments:
            rel = 0
            while rel < seg.length:
                take = min(block_size, seg.length - rel)
                handle.seek(seg.file_offset + rel)
                data = handle.read(take)
                if not data:
                    break
                yield seg.s_addr + rel, data
                if rel + block_size >= seg.length:
                    break
                rel += block_size - overlap

    def iter_page_chunks(self, handle: Any, chunk_size: int) -> Iterator[tuple[int, bytes]]:
        """Stream page-aligned chunks of each segment as ``(physical_base, bytes)``."""
        step = max(PAGE, chunk_size // PAGE * PAGE)
        for seg in self.segments:
            aligned_start = (seg.s_addr + PAGE - 1) & ~(PAGE - 1)
            aligned_end = seg.end_exclusive & ~(PAGE - 1)
            if aligned_end <= aligned_start:
                continue
            cursor = aligned_start
            while cursor < aligned_end:
                take = min(step, aligned_end - cursor)
                handle.seek(seg.file_offset + (cursor - seg.s_addr))
                data = handle.read(take)
                if not data:
                    break
                yield cursor, data
                cursor += len(data)
