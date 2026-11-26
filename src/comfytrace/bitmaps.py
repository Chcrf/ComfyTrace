"""Stage 2 (raw memory scan): decoded 32-bit bitmaps such as the ComfyUI frontend's decoded preview images.

Chromium (Electron) decodes every <img> to opaque BGRA/RGBA pixels, typically into pagefile-backed shared
sections that no process maps privately. A physical page whose every fourth byte is 0xFF and whose colour
bytes vary smoothly is a candidate. Pages are then put in order:

* with kernel support data (Volatility symbols: MmPfnDatabase, MmPteBase, the _MMPFN PteAddress offset), each
  page's PFN entry names its prototype PTE; prototype PTEs of one section are consecutive 8-byte slots, so the
  section is rebuilt in order by walking that PTE array (valid and transition entries only);
* without it, physically contiguous runs are used as they are.

Each ordered region is cut into images by its row stride (the stride that minimises vertical differences),
the row phase (where horizontal discontinuities line up), and kept only when it looks like a photograph-sized
image. No bytes are invented: unreadable pages end a region.
"""

from __future__ import annotations

import collections
import logging
import mmap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from comfytrace.models import PageOrigin
from comfytrace.pagetable import PROTOTYPE, PageTables, leaf_frame
from comfytrace.procmem import touch_pages

log = logging.getLogger(__name__)

PAGE = 4096
SCAN_CHUNK = 64 << 20
WINDOW = 64 << 10
WIDTHS = tuple(range(64, 2049, 16))
MIN_WIDTH, MIN_ROWS = 256, 64
MAX_HOLE_PAGES = 8  # pages without opaque pixels (white areas, overwritten rows) inside one private run
MAX_GAP_PAGES = 8  # missing or reused pages kept as a gap inside one private run
MAX_TRAILING_PAGES = 64  # reused pages read on after a run's last verified page, for the uncertain layer only
COHERENT_ROW_SCORE = 12.0  # a stride measured on a window's opaque pages only is used when rows differ less
SELF_MAP_SPAN = 1 << 39
ACTIVE = 6
FREED = frozenset({0, 1, 5})  # Zeroed, Free, Bad


class Gaps(dict[int, str]):
    """Page index of a region -> why its bytes are missing. ``uncertain`` holds, for a STALE_MAPPING page, the
    frame number the page table still names and the bytes read there. Those bytes are kept apart: the frame's
    PFN entry names another PTE, so nothing shows the frame still held this page when it was copied."""

    def __init__(self) -> None:
        super().__init__()
        self.uncertain: dict[int, tuple[int, bytes]] = {}
        # The same for the STALE_MAPPING pages that directly follow the region's last verified page, in order.
        # They are no part of the region: nothing after them anchors that an image goes on that far.
        self.trailing: list[tuple[int, bytes]] = []


@dataclass
class DecodedBitmap:
    region: str
    width: int
    rows: int
    rgb: np.ndarray
    physical_pages: list[int]
    byte_offset: int
    ordering: str
    notes: dict[str, Any] = field(default_factory=dict)
    # Gap rows as they read with the bytes of the frames the page table still names: row -> its pixels. Never
    # part of ``rgb``.
    uncertain_rows: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class FileNameLayout:
    file_pointer: int
    reference_bits: int
    file_name: int
    length: int
    maximum_length: int
    buffer: int


@dataclass(frozen=True)
class SectionLayout:
    """Offsets, from symbols, that tell a file-backed section page from shared memory."""

    original_pte_offset: int  # _MMPFN.OriginalPte
    control_area_offset: int  # _SUBSECTION.ControlArea
    flags_offset: int  # _CONTROL_AREA.u.Flags
    file_bit: int
    image_bit: int
    file_name: FileNameLayout | None = None


@dataclass
class PfnSupport:
    """Kernel data needed to order section pages and to check page mappings. Values come from symbols."""

    kernel: PageTables
    pfn_database: int
    entry_size: int
    pte_address_offset: int
    pte_base: int
    location_offset: int | None = None
    prototype_flag: tuple[int, int] | None = None  # (byte offset of _MMPFN.u4, bit of PrototypePte in that qword)
    section: SectionLayout | None = None
    pte_frame: tuple[int, int, int] | None = None  # (byte offset of the qword holding PteFrame, first bit, bits)

    def pte_address(self, pfn: int) -> int | None:
        return self.kernel.read_kernel_qword(self.pfn_database + pfn * self.entry_size + self.pte_address_offset)

    def _entry(self, pfn: int) -> bytes | None:
        """The PFN entry's bytes; an entry may straddle two pages of the database."""
        return self._read_virtual(self.pfn_database + pfn * self.entry_size, self.entry_size)

    def _read_virtual(self, va: int, size: int) -> bytes | None:
        out = bytearray()
        while len(out) < size:
            address = va + len(out)
            where = self.kernel.translate(address)
            take = min(size - len(out), PAGE - (address & (PAGE - 1)))
            if where is None or where + take > self.kernel.size:
                return None
            out.extend(self.kernel.physical[where : where + take])
        return bytes(out)

    def mapping_problem(self, va: int, frame: int, kind: str, proto: int | None) -> str | None:
        """Why the PFN database disagrees that ``frame`` holds this mapping of ``va``, or None when it agrees or
        cannot be checked (large pages, unreadable entries, offsets not resolved from symbols).

        Acquisition takes minutes, so a page trimmed or faulted in between the page-table page and the PFN entry
        being copied is consistent as long as the frame still belongs to this mapping:

        * a freed frame (Zeroed, Free, Bad) no longer holds the page;
        * a private frame's PteAddress must be this PTE in the self-map, on any list;
        * a shared frame's PteAddress is the section's prototype PTE, which must name this frame. Behind a valid
          hardware PTE a shared frame is in use, so it must be Active (a Standby file-cache frame there is a page
          table copied before the frame was reused)."""
        if kind == "LARGE" or self.location_offset is None or self.prototype_flag is None:
            return None
        entry = self._entry(frame >> 12)
        if entry is None:
            return None
        location = entry[self.location_offset] & 7
        if location in FREED:
            return "PFN_FREED"
        # The high bits of PteAddress can carry flags (bit 63 clear on T2's prototype pages); compare canonical VAs.
        owner = _canonical(int.from_bytes(entry[self.pte_address_offset : self.pte_address_offset + 8], "little") & ~7)
        flags_offset, bit = self.prototype_flag
        shared = int.from_bytes(entry[flags_offset : flags_offset + 8], "little") >> bit & 1
        if not shared:
            return None if proto is None and owner == self.pte_base + (va >> 12) * 8 else "PFN_MAPS_ANOTHER_PTE"
        if proto is not None and owner != _canonical(proto):
            return "PFN_MAPS_ANOTHER_PTE"
        if proto is None and kind == "VALID" and location != ACTIVE:
            return "SHARED_FRAME_NOT_IN_USE"
        prototype = self.kernel.read_kernel_qword(owner)
        if prototype is None or leaf_frame(prototype) != frame:
            return "PROTOTYPE_PTE_MAPS_ANOTHER_FRAME"
        return None

    def origin(self, pfn: int) -> PageOrigin:
        """What physical page ``pfn`` belonged to at capture. A frame without PrototypePte sits behind a hardware
        PTE. A section frame's OriginalPte is a subsection PTE only in a file-backed section, and the subsection's
        control area then says data file or image; in a pagefile-backed section it is a software PTE."""
        return self._origin(pfn)[0]

    def _origin(self, pfn: int) -> tuple[PageOrigin, int | None]:
        entry = self._entry(pfn)
        if entry is None or self.location_offset is None or self.prototype_flag is None:
            return PageOrigin.UNKNOWN, None
        if entry[self.location_offset] & 7 in FREED:
            return PageOrigin.FREED, None
        flags_offset, bit = self.prototype_flag
        if not int.from_bytes(entry[flags_offset : flags_offset + 8], "little") >> bit & 1:
            return PageOrigin.PRIVATE, None
        layout = self.section
        if layout is None:
            return PageOrigin.UNKNOWN, None
        original = int.from_bytes(entry[layout.original_pte_offset : layout.original_pte_offset + 8], "little")
        if not original & PROTOTYPE:
            return PageOrigin.SHARED_MEMORY, None
        control_area = self.kernel.read_kernel_qword(_canonical(original >> 16) + layout.control_area_offset)
        flags = None if control_area is None else self.kernel.read_kernel_qword(
            _canonical(control_area) + layout.flags_offset)
        if flags is not None and flags >> layout.image_bit & 1:
            return PageOrigin.FILE_IMAGE, None
        if flags is not None and flags >> layout.file_bit & 1:
            return PageOrigin.FILE_DATA, control_area
        return PageOrigin.FILE_UNCONFIRMED, None

    def _backing_file_name(self, control_area: int, layout: FileNameLayout) -> str | None:
        ref = self.kernel.read_kernel_qword(_canonical(control_area) + layout.file_pointer)
        if not ref:
            return None
        file_object = _canonical(ref & ~((1 << layout.reference_bits) - 1))
        if file_object < 0xFFFF_8000_0000_0000:
            return None
        descriptor = file_object + layout.file_name
        length = self._read_virtual(descriptor + layout.length, 2)
        maximum = self._read_virtual(descriptor + layout.maximum_length, 2)
        buffer = self.kernel.read_kernel_qword(descriptor + layout.buffer)
        size = int.from_bytes(length or b"", "little")
        if buffer is None or _canonical(buffer) < 0xFFFF_8000_0000_0000 \
                or not 0 < size <= min(int.from_bytes(maximum or b"", "little"), 4096) or size % 2:
            return None
        data = self._read_virtual(_canonical(buffer), size)
        try:
            name = data.decode("utf-16-le") if data is not None else None
        except UnicodeDecodeError:
            return None
        return name if name and name.isprintable() else None

    def copy_origin(self, pfns: Iterable[int], nbytes: int) -> dict[str, Any]:
        """One copy of a record: the origin of each page it occupies, counted, and the bytes it holds."""
        pages: collections.Counter[str] = collections.Counter()
        areas: set[int] = set()
        layout = self.section.file_name if self.section else None
        for pfn in dict.fromkeys(pfns):
            origin, area = self._origin(pfn)
            pages[str(origin)] += 1
            if area and layout and area not in areas:
                owner = self.pte_address(pfn)
                prototype = self.kernel.read_kernel_qword(_canonical(owner & ~7)) if owner else None
                if prototype is not None and leaf_frame(prototype) == pfn * PAGE:
                    areas.add(area)
        result: dict[str, Any] = {"pages": dict(sorted(pages.items())), "bytes": nbytes}
        if layout:
            names = {name for area in areas if (name := self._backing_file_name(area, layout)) is not None}
            if names:
                result["backing_file_names"] = sorted(names)
        return result

    def table_slot(self, pfn: int) -> tuple[int, int] | None:
        """(frame number of the page-table page, slot) of the PTE that maps ``pfn``: the PFN entry's PteFrame and the
        slot of its PteAddress. For a page-table page this names its parent table. None without the layout."""
        entry = self._entry(pfn) if self.pte_frame is not None else None
        if entry is None or self.pte_frame is None:
            return None
        offset, bit, bits = self.pte_frame
        frame = int.from_bytes(entry[offset : offset + 8], "little") >> bit & ((1 << bits) - 1)
        pte = int.from_bytes(entry[self.pte_address_offset : self.pte_address_offset + 8], "little")
        return frame, (pte & 0xFFF) >> 3

    def is_private(self, pte_address: int) -> bool:
        return self.pte_base <= pte_address < self.pte_base + SELF_MAP_SPAN


def _canonical(va: int) -> int:
    va &= 0x0000_FFFF_FFFF_FFFF
    return va | 0xFFFF_0000_0000_0000 if va & (1 << 47) else va


def opaque_pages(physical: Any, image_path: Path | None = None) -> list[int]:
    """Physical page numbers that look like opaque 32-bit pixels. With ``image_path``, the file that ``physical``
    maps, the pass reads the file into one buffer instead of slicing the mapping: the same bytes for about a
    third of the time."""
    found: list[int] = []
    if hasattr(physical, "iter_page_chunks") and image_path is not None:
        with open(image_path, "rb", buffering=0) as handle:
            for base, data in physical.iter_page_chunks(handle, SCAN_CHUNK):
                usable = len(data) // PAGE * PAGE
                if not usable:
                    continue
                a = np.frombuffer(data[:usable], np.uint8).reshape(-1, PAGE // 4, 4)
                idx = np.nonzero((a[:, :, 3] == 255).all(1))[0]
                for i in idx:
                    colour = a[i, :, :3].astype(np.int16)
                    if colour.std() > 3 and np.abs(np.diff(colour[:, 0])).mean() < 40:
                        found.append((base >> 12) + int(i))
        return found
    size = len(physical)
    dontneed = getattr(mmap, "MADV_DONTNEED", None)
    madvise = getattr(physical, "madvise", None) if isinstance(physical, mmap.mmap) else None
    handle = open(image_path, "rb", buffering=0) if image_path is not None else None  # noqa: SIM115 - closed below
    buffer = memoryview(bytearray(SCAN_CHUNK if handle is not None else 0))
    try:
        for base in range(0, size, SCAN_CHUNK):
            data: Any
            if handle is not None:
                handle.seek(base)
                filled, want = 0, min(size - base, SCAN_CHUNK)
                while filled < want and (count := handle.readinto(buffer[filled:want])):
                    filled += count
                data = buffer[:filled]
            else:
                data = physical[base : min(size, base + SCAN_CHUNK)]
                if dontneed is not None and madvise is not None:
                    # Pages of a whole-image mapping otherwise stay in this process's RSS.
                    madvise(dontneed, base, len(data))
            usable = len(data) // PAGE * PAGE
            if not usable:
                continue
            a = np.frombuffer(data[:usable], np.uint8).reshape(-1, PAGE // 4, 4)
            idx = np.nonzero((a[:, :, 3] == 255).all(1))[0]
            for i in idx:
                colour = a[i, :, :3].astype(np.int16)
                if colour.std() > 3 and np.abs(np.diff(colour[:, 0])).mean() < 40:
                    found.append((base >> 12) + int(i))
    finally:
        if handle is not None:
            handle.close()
    return found


def _group(keys: list[tuple[int, int]], step: int, max_gap: int) -> list[list[tuple[int, int]]]:
    """Groups (key, pfn) pairs whose keys advance by ``step`` with at most ``max_gap`` missing slots."""
    keys.sort()
    groups: list[list[tuple[int, int]]] = []
    for key, pfn in keys:
        if groups and 0 < key - groups[-1][-1][0] <= step * (max_gap + 1):
            groups[-1].append((key, pfn))
        else:
            groups.append([(key, pfn)])
    return groups


def _private_regions(physical: Any, pages: list[int], support: PfnSupport,
                     min_pages: int) -> Iterator[tuple[str, bytes, list[int], str, dict[int, str]]]:
    """Private pages in virtual order. A page's PFN entry names the page-table page that maps it (PteFrame) and
    its slot; the table page's own entry names its page directory and slot, so pages of one directory sort by
    virtual page number and an image continues across table pages. Each run is filled from the tables
    themselves, so pages that are not opaque (white areas, overwritten rows) keep their place. A slot whose
    frame is not resident (PAGE_NOT_PRESENT) or whose frame's PFN entry names another PTE (STALE_MAPPING, reused
    since) keeps its place as a gap when the run continues within MAX_GAP_PAGES; its bytes are zero and listed,
    never evidence. The bytes of a STALE_MAPPING frame are handed on separately (``Gaps.uncertain``) for the
    ownership-uncertain layer. More missing pages than that end the run."""
    parents: dict[int, tuple[int, int] | None] = {}
    keyed: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for pfn in pages:
        if support.origin(pfn) != PageOrigin.PRIVATE:
            continue
        where = support.table_slot(pfn)
        if where is None:
            continue
        table, slot = where
        if table not in parents:
            parents[table] = support.table_slot(table)
        parent = parents[table]
        if parent is not None:
            keyed[parent[0]].append((parent[1] * 512 + slot, pfn))
    for directory, keys in keyed.items():
        for group in _group(keys, 1, MAX_HOLE_PAGES):
            if len(group) < min_pages:
                continue
            frames: list[int] = []
            gaps = Gaps()
            missing: list[tuple[str, int | None]] = []
            data, start = bytearray(), group[0][0]
            for index in range(group[0][0], group[-1][0] + 1):
                frame = _slot_frame(physical, directory, index)
                page_bytes = bytes(physical[frame : frame + PAGE]) if frame is not None else b""
                if frame is None or len(page_bytes) != PAGE:
                    missing.append(("PAGE_NOT_PRESENT", None))
                elif support.table_slot(frame >> 12) != (_table_frame(physical, directory, index >> 9),
                                                         index & 511):
                    missing.append(("STALE_MAPPING", frame))
                else:
                    if frames:
                        for reason, stale in missing:  # a hole inside the run keeps its place
                            gaps[len(frames)] = reason
                            if stale is not None:
                                stale_bytes = bytes(physical[stale : stale + PAGE])
                                if len(stale_bytes) == PAGE:
                                    gaps.uncertain[len(frames)] = (stale >> 12, stale_bytes)
                            frames.append(-1)
                            data += bytes(PAGE)
                    else:
                        start = index
                    missing = []
                    data += page_bytes
                    frames.append(frame >> 12)
                    continue
                if len(missing) > MAX_GAP_PAGES:
                    if len(frames) - len(gaps) >= min_pages:
                        gaps.trailing = _stale_tail(physical, support, directory, missing, index + 1)
                        yield f"PT-0x{directory << 12:X}+{start:X}", bytes(data), frames, "PAGE_TABLE_ORDER", gaps
                    frames, gaps, missing, data = [], Gaps(), [], bytearray()
            if len(frames) - len(gaps) >= min_pages:
                gaps.trailing = _stale_tail(physical, support, directory, missing, group[-1][0] + 1)
                yield f"PT-0x{directory << 12:X}+{start:X}", bytes(data), frames, "PAGE_TABLE_ORDER", gaps


def _stale_tail(physical: Any, support: PfnSupport, directory: int, missing: list[tuple[str, int | None]],
                index: int) -> list[tuple[int, bytes]]:
    """(frame number, bytes) of the STALE_MAPPING pages that directly follow a run's last verified page: those
    already seen (``missing``), then the slots from ``index`` on. Stops at the first slot that is not present, at
    the first frame that still belongs to its slot (another run's page), and after MAX_TRAILING_PAGES."""
    tail: list[tuple[int, bytes]] = []
    for _reason, frame in missing:
        if frame is None:
            return tail
        tail.append((frame >> 12, bytes(physical[frame : frame + PAGE])))
    while len(tail) < MAX_TRAILING_PAGES:
        frame = _slot_frame(physical, directory, index)
        if frame is None or frame + PAGE > len(physical):
            break
        page_bytes = bytes(physical[frame : frame + PAGE])
        if len(page_bytes) != PAGE or support.table_slot(frame >> 12) == (
                _table_frame(physical, directory, index >> 9), index & 511):
            break
        tail.append((frame >> 12, page_bytes))
        index += 1
    return tail


def _table_frame(physical: Any, directory: int, slot: int) -> int | None:
    """Frame number of the page table that directory entry ``slot`` points to (a large page has none)."""
    if directory < 0 or not 0 <= slot < 512 or (directory + 1) << 12 > len(physical):
        return None
    entry = int.from_bytes(physical[(directory << 12) + slot * 8 : (directory << 12) + slot * 8 + 8], "little")
    return entry >> 12 & ((1 << 36) - 1) if entry & 1 and not entry & 0x80 else None


def _slot_frame(physical: Any, directory: int, index: int) -> int | None:
    table = _table_frame(physical, directory, index >> 9)
    if table is None or (table + 1) << 12 > len(physical):
        return None
    entry = int.from_bytes(physical[(table << 12) + (index & 511) * 8 : (table << 12) + (index & 511) * 8 + 8],
                           "little")
    return leaf_frame(entry)


def ordered_regions(physical: Any, pages: list[int], support: Any,
                    min_pages: int = 16) -> Iterator[tuple[str, bytes, list[int], str, dict[int, str]]]:
    """Yields (region id, bytes, physical page numbers, ordering method, gaps) for each ordered candidate region.
    ``gaps`` maps a page index of the region to why its bytes are missing; its frame number is -1."""
    if support is not None and hasattr(support, "ordered_regions"):
        yield from support.ordered_regions(physical, pages, min_pages)
        return
    if support is None:
        for group in _group([(p, p) for p in pages], 1, 1):
            if len(group) < min_pages:
                continue
            first, last = group[0][1], group[-1][1]
            yield (f"PHYS-0x{first << 12:X}", bytes(physical[first << 12 : (last + 1) << 12]),
                   list(range(first, last + 1)), "PHYSICAL_CONTIGUITY", {})
        return
    shared: list[tuple[int, int]] = []
    for pfn in pages:
        pte = support.pte_address(pfn)
        if pte:
            pte = _canonical(pte & ~7)
            if not support.is_private(pte):
                shared.append((pte, pfn))
    for group in _group(shared, 8, 8):
        if len(group) < min_pages:
            continue
        first, last = group[0][0], group[-1][0]
        data = bytearray()
        frames: list[int] = []
        start = first
        for slot in range(first, last + 8, 8):
            entry = support.kernel.read_kernel_qword(slot)
            frame = leaf_frame(entry) if entry is not None else None
            page_bytes = bytes(physical[frame : frame + PAGE]) if frame is not None else b""
            if frame is None or len(page_bytes) != PAGE:
                # A section page that is not resident ends the ordered run; never pad.
                if len(frames) >= min_pages:
                    yield f"SECTION-0x{start:X}", bytes(data), frames, "PROTOTYPE_PTE_ORDER", {}
                data, frames, start = bytearray(), [], slot + 8
                continue
            data += page_bytes
            frames.append(frame >> 12)
        if len(frames) >= min_pages:
            yield f"SECTION-0x{start:X}", bytes(data), frames, "PROTOTYPE_PTE_ORDER", {}
    yield from _private_regions(physical, pages, support, min_pages)


def _best_width(window: np.ndarray, keep: np.ndarray | None = None) -> tuple[int, float]:
    """The stride (in pixels) at which the window's rows differ least from the row above, and that difference.
    With ``keep`` (a flag per byte), only bytes it marks are compared."""
    best = (0, float("inf"))
    for w in WIDTHS:
        stride = w * 4
        if len(window) < 3 * stride:
            break
        score = (float(np.abs(window[stride:] - window[:-stride]).mean()) if keep is None
                 else _stride_score(window, w, keep))
        if score < best[1]:
            best = (w, score)
    return best


def _stride_score(window: np.ndarray, width: int, keep: np.ndarray) -> float:
    """Mean difference between bytes one row apart at ``width``, over the bytes ``keep`` marks."""
    stride = width * 4
    if len(window) < 3 * stride:
        return float("inf")
    pairs = keep[stride:] & keep[:-stride]
    return float(np.abs(window[stride:] - window[:-stride])[pairs].mean()) if pairs.any() else float("inf")


def _phase(segment: np.ndarray, width: int) -> int:
    """Pixel index where rows start: horizontal jumps peak at the row seam."""
    px = segment[: len(segment) // 4 * 4].reshape(-1, 4)[:, :3]
    jump = np.abs(px[1:] - px[:-1]).sum(1)
    scores = [float(jump[p::width].mean()) for p in range(width)]
    return (int(np.argmax(scores)) + 1) % width


def _photo_like(rgb: np.ndarray) -> bool:
    flat = (np.abs(np.diff(rgb.astype(np.int16), axis=1)).sum(2) == 0).mean()
    return bool(flat < 0.5 and rgb.std() > 10)


def cut_images(region: str, data: bytes, frames: list[int], ordering: str,
               gaps: dict[int, str] | None = None) -> list[DecodedBitmap]:
    """Cuts one ordered region into images by stride and row phase. A stride window shorter than half a window, or
    touching a gap, takes the width of the window before it, so the rows at a region's end or around a gap are
    kept. A window with no window before it that touches a gap, and any window holding a page that is not opaque
    pixels, is measured on its opaque pages only. Rows on a gap page are listed in
    ``notes["gap_rows"]`` as (first, end, reason); their pixels are zero.
    Where every gap page a row touches is a STALE_MAPPING page whose frame was read (``Gaps.uncertain``), the row
    as it reads with those bytes goes to ``uncertain_rows`` and ``notes["uncertain_rows"]``. So do the opaque rows
    that the reused pages directly after the region (``Gaps.trailing``) would add to its last image: their row
    numbers start at the image's height, and the image itself does not grow."""
    gaps = gaps if gaps is not None else {}  # a Gaps without interior gaps still carries its trailing pages
    uncertain: dict[int, tuple[int, bytes]] = dict(getattr(gaps, "uncertain", {}))
    filled = bytearray(data)
    for page, (_frame, content) in uncertain.items():
        filled[page * PAGE : (page + 1) * PAGE] = content
    for frame, content in getattr(gaps, "trailing", []):
        uncertain[len(filled) // PAGE] = (frame, content)
        filled += content
    with_uncertain = np.frombuffer(bytes(filled[: len(filled) // 4 * 4]), np.uint8)
    a = np.frombuffer(data[: len(data) // 4 * 4], np.uint8).astype(np.int16)
    # A page inside a run that is not opaque pixels (overwritten since, though its frame still belongs to the
    # mapping) spoils its window's stride measurement, as a gap's zeroed rows do. Such a window is measured on
    # its opaque pages only.
    whole = len(data) // PAGE * PAGE
    opaque_page = (np.frombuffer(data[:whole], np.uint8).reshape(-1, PAGE // 4, 4)[:, :, 3] == 255).all(1)
    keep = np.ones(len(a), bool)  # bytes past the last whole page are not judged
    keep[:whole] = np.repeat(opaque_page, PAGE)[: len(a)]
    widths: list[tuple[int, int]] = []
    for start in range(0, len(a), WINDOW):
        window = a[start : start + WINDOW]
        touches_gap = any(start // PAGE <= g < (start + len(window) + PAGE - 1) // PAGE for g in gaps)
        damaged = not opaque_page[start // PAGE : (start + len(window)) // PAGE].all()
        if widths and (len(window) < WINDOW // 2 or touches_gap):
            widths.append((start, widths[-1][1]))
        elif len(window) < WINDOW // 2:
            widths.append((start, 0))
        elif touches_gap or damaged:
            width, score = _best_width(window, keep[start : start + len(window)])
            widths.append((start, width if score <= COHERENT_ROW_SCORE else _best_width(window)[0]))
        else:
            widths.append((start, _best_width(window)[0]))
    out: list[DecodedBitmap] = []
    i = 0
    while i < len(widths):
        width = widths[i][1]
        j = i
        while j < len(widths) and widths[j][1] == width:
            j += 1
        lo, hi = widths[i][0], min(len(a), widths[j - 1][0] + WINDOW)
        i = j
        if width < MIN_WIDTH or width % 8:
            continue
        segment = a[lo:hi]
        phase = _phase(segment, width)
        start = lo + phase * 4
        rows = (hi - start) // (width * 4)
        if rows < MIN_ROWS:
            continue
        pixels = a[start : start + rows * width * 4].astype(np.uint8).reshape(rows, width, 4)
        opaque = (pixels[:, :, 3] == 255).all(1)
        rgb = np.ascontiguousarray(pixels[:, :, [2, 1, 0]])
        if not _photo_like(rgb):
            continue
        stride = width * 4
        gap_rows: list[list[Any]] = []
        for page in sorted(gaps):
            first = max(0, (page * PAGE - start) // stride)
            end = min(rows, ((page + 1) * PAGE - start + stride - 1) // stride)
            if first >= end:
                continue
            if gap_rows and gap_rows[-1][1] >= first and gap_rows[-1][2] == gaps[page]:
                gap_rows[-1][1] = max(gap_rows[-1][1], end)
            else:
                gap_rows.append([first, end, gaps[page]])
        pages = [f for f in frames[start // PAGE : (start + rows * stride - 1) // PAGE + 1] if f >= 0]
        notes: dict[str, Any] = {"non_opaque_rows": [int(r) for r in np.nonzero(~opaque)[0][:64]],
                                 "channel_order": "BGRA"}
        if gap_rows:
            notes["gap_rows"] = gap_rows
        readable: dict[int, np.ndarray] = {}
        for first, end, _reason in gap_rows if uncertain else []:
            for row in range(first, end):
                touched = range((start + row * stride) // PAGE, (start + (row + 1) * stride - 1) // PAGE + 1)
                if all(page in uncertain for page in touched if page in gaps):
                    row_pixels = with_uncertain[start + row * stride : start + (row + 1) * stride].reshape(width, 4)
                    readable[row] = np.ascontiguousarray(row_pixels[:, [2, 1, 0]])
        if hi == len(a) and len(with_uncertain) > len(a):
            # The region ends inside this image's rows, and reused pages follow it.
            for row in range(rows, (len(with_uncertain) - start) // stride):
                row_pixels = with_uncertain[start + row * stride : start + (row + 1) * stride].reshape(width, 4)
                if not (row_pixels[:, 3] == 255).all():
                    break  # the image's rows are opaque: the first row that is not ends what could be its rows
                readable[row] = np.ascontiguousarray(row_pixels[:, [2, 1, 0]])
        if readable:
            ranges: list[list[Any]] = []
            for row in sorted(readable):
                touched = range((start + row * stride) // PAGE, (start + (row + 1) * stride - 1) // PAGE + 1)
                frames_read = [f"0x{uncertain[page][0] << 12:X}" for page in touched if page in uncertain]
                if ranges and ranges[-1][1] == row:
                    ranges[-1][1] = row + 1
                    ranges[-1][2] += [f for f in frames_read if f not in ranges[-1][2]]
                else:
                    ranges.append([row, row + 1, frames_read])
            notes["uncertain_rows"] = ranges
        out.append(DecodedBitmap(region, width, rows, rgb, pages, start, ordering, notes, readable))
    return out


def scan(physical: Any, support: PfnSupport | None, image_path: Path | None = None,
         scattered: Any = None) -> list[DecodedBitmap]:
    """``scattered`` is the same image mapped without read-around. Ordering reads page tables and single pages
    from all over the image; through ``physical`` each of those that is not cached reads a whole read-ahead
    window. The opaque pages are read into the page cache first, in ascending order from several threads."""
    pages = opaque_pages(physical, image_path)
    log.info("decoded-bitmap scan: %d opaque 32-bit pages; ordering %s", len(pages),
             "by prototype PTE" if support else "by physical contiguity")
    found: list[DecodedBitmap] = []
    if support is not None and isinstance(scattered, mmap.mmap):
        touch_pages(scattered, np.array(pages, np.int64) << 12)
        physical = scattered
    for region, data, frames, ordering, gaps in ordered_regions(physical, pages, support):
        found += cut_images(region, data, frames, ordering, gaps)
    return found
