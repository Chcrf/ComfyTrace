"""Bytes in the order of the mapping that owns a page, wherever its frames lie in the image.

A file's cached pages and a process's private pages are scattered over physical memory. Each frame's PFN entry
names the PTE that maps it: a prototype PTE in the section's array (file cache, shared sections) or a slot of a
page table (private memory). The next entries name the next pages, so a torn PNG can be read on in file or
virtual order. Every frame is checked against its own PFN entry (it must still belong to that entry), and
reading stops at the first page that is not resident or was reused: nothing is guessed or padded.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comfytrace.bitmaps import PAGE, _canonical, _slot_frame, _table_frame
from comfytrace.models import PageOrigin
from comfytrace.pagetable import leaf_frame


def page_order_stream(physical: Any, support: Any, start: int, max_bytes: int) -> tuple[
        bytes, str, list[int]] | None:
    """(bytes from ``start`` on, ordering method, frame numbers read) or None when the page's mapping cannot be
    followed. At least the first page's bytes from ``start`` are returned."""
    if hasattr(support, "page_order_stream"):
        return support.page_order_stream(physical, start, max_bytes)
    pfn = start >> 12
    origin = support.origin(pfn)
    first = bytes(physical[start : (pfn + 1) << 12])
    frames = [pfn]
    data = bytearray(first)
    if origin == PageOrigin.PRIVATE:
        where = support.table_slot(pfn)
        parent = support.table_slot(where[0]) if where else None
        if where is None or parent is None:
            return None
        directory, index = parent[0], parent[1] * 512 + where[1]
        while len(data) < max_bytes:
            index += 1
            frame = _slot_frame(physical, directory, index)
            if frame is None or frame + PAGE > len(physical) or support.table_slot(frame >> 12) != (
                    _table_frame(physical, directory, index >> 9), index & 511):
                break
            data += physical[frame : frame + PAGE]
            frames.append(frame >> 12)
        return bytes(data), "PAGE_TABLE_ORDER", frames
    if origin == PageOrigin.FREED:
        return None
    pte = support.pte_address(pfn)
    if pte is None:
        return None
    pte = _canonical(pte & ~7)
    while len(data) < max_bytes:
        pte += 8
        entry = support.kernel.read_kernel_qword(pte)
        frame = leaf_frame(entry) if entry is not None else None
        owner = support.pte_address(frame >> 12) if frame is not None else None
        if frame is None or frame + PAGE > len(physical) or owner is None or _canonical(owner & ~7) != pte:
            break
        data += physical[frame : frame + PAGE]
        frames.append(frame >> 12)
    return bytes(data), "PROTOTYPE_PTE_ORDER", frames


TEXT_TYPES = (b"tEXt", b"zTXt", b"iTXt")
NEXT_TYPES = (b"tEXt", b"zTXt", b"iTXt", b"IDAT", b"IEND", b"tIME", b"pHYs", b"sRGB", b"gAMA")


@dataclass(frozen=True)
class TornText:
    """A PNG text chunk cut at the end of the bytes read: its type, declared length and the data read so far."""

    record_id: str
    ctype: bytes
    declared: int
    head: bytes

    @property
    def missing(self) -> int:
        return self.declared - len(self.head)


def torn_text_chunk(record_id: str, data: bytes) -> TornText | None:
    """The text chunk ``data`` (a PNG's genuine bytes from its signature) ends inside, if any, and if its rest fits
    one page with its CRC and the next chunk header."""
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        if pos + 12 + length <= len(data):
            pos += 12 + length
            continue
        head = data[pos + 8 :]
        if ctype in TEXT_TYPES and 0 < length - len(head) <= PAGE - 12:
            return TornText(record_id, ctype, length, head)
        return None
    return None


def stitch_text_chunks(image_path: Path, torn: list[TornText], chunk: int = 64 << 20) -> dict[str, list[bytes]]:
    """For each torn chunk, every page of the image that completes it: the page starts with the missing bytes, then
    the chunk's CRC-32 (computed over type, head and those bytes), then a plausible next chunk type. One pass over
    the file; a page is only checksummed when the next chunk type is where it must be. Returns, per record, the
    distinct continuations found (each the missing bytes only)."""
    if not torn:
        return {}
    found: dict[str, list[bytes]] = {t.record_id: [] for t in torn}
    by_missing: dict[int, list[TornText]] = {}
    for t in torn:
        by_missing.setdefault(t.missing, []).append(t)
    prefix_crc = {t.record_id: zlib.crc32(t.ctype + t.head) for t in torn}
    from comfytrace.lime import parse_lime_segments

    segments = parse_lime_segments(image_path)
    buf_size = max(PAGE, chunk // PAGE * PAGE)

    def _scan_pages(pages: np.ndarray) -> None:
        for missing, group in by_missing.items():
            kinds = pages[:, missing + 8 : missing + 12]
            hit = np.zeros(len(pages), bool)
            for kind in NEXT_TYPES:
                hit |= (kinds == np.frombuffer(kind, np.uint8)).all(1)
            for row in np.nonzero(hit)[0]:
                page = pages[row].tobytes()
                stored = int.from_bytes(page[missing : missing + 4], "big")
                for t in group:
                    if zlib.crc32(page[:missing], prefix_crc[t.record_id]) == stored:
                        tail = page[:missing]
                        if tail not in found[t.record_id]:
                            found[t.record_id].append(tail)

    with open(image_path, "rb", buffering=0) as handle:
        buffer = bytearray(buf_size)
        view = memoryview(buffer)
        if segments is not None:
            for seg in segments:
                aligned_start = (seg.s_addr + PAGE - 1) & ~(PAGE - 1)
                aligned_end = seg.end_exclusive & ~(PAGE - 1)
                cursor = aligned_start
                while cursor < aligned_end:
                    want = min(buf_size, aligned_end - cursor)
                    handle.seek(seg.file_offset + (cursor - seg.s_addr))
                    filled = 0
                    while filled < want and (count := handle.readinto(view[filled:want])):
                        filled += count
                    usable = filled // PAGE * PAGE
                    if not usable:
                        break
                    _scan_pages(np.frombuffer(view[:usable], np.uint8).reshape(-1, PAGE))
                    cursor += usable
                    if filled < want:
                        break
        else:
            while True:
                filled = 0
                while filled < buf_size and (count := handle.readinto(view[filled:buf_size])):
                    filled += count
                usable = filled // PAGE * PAGE
                if usable:
                    _scan_pages(np.frombuffer(view[:usable], np.uint8).reshape(-1, PAGE))
                if filled < buf_size:
                    break
    return found
