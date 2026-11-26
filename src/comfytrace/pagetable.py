"""x64 Windows page-table walking over a raw physical image, including transition and prototype PTEs.

Volatility's process layer treats prototype PTEs as invalid, so shared image pages (python3XY.dll ``.data`` with
CPython 3.12+ immortal type objects) and section-backed memory read as holes. This walker resolves:

* valid PTEs (bit 0): PFN in bits 12-47; large pages at PDPT/PD level;
* transition PTEs (bit 11 set, bit 10 clear): the page is still in RAM on the standby/modified list;
* prototype PTEs (bit 10 set): ``ProtoAddress = pte >> 16`` (sign-extended) is a kernel VA holding the shared
  prototype PTE, itself valid or transition when the page is resident.

Everything else (pagefile, demand-zero, VAD-described) is reported as not resident. The DTB of the process
and of the kernel come from Volatility (supporting data); all translation here is our own.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

PAGE = 4096
PFN_MASK = 0x0000_FFFF_FFFF_F000
VALID = 1
LARGE = 1 << 7
PROTNONE = 1 << 8
PROTOTYPE = 1 << 10
TRANSITION = 1 << 11
VAD_PROTO = 0xFFFF_FFFF_0000


def canonical(va: int, levels: int = 4) -> int:
    """``va`` sign-extended from bit 47 (4-level) or bit 56 (5-level). Volatility's Intel layers report
    kernel addresses masked to the virtual address width."""
    if levels == 5:
        return va | 0xFF00_0000_0000_0000 if va & (1 << 56) else va
    return va | 0xFFFF_0000_0000_0000 if va & (1 << 47) else va


def leaf_frame(pte: int) -> int | None:
    """Physical page base of a valid or transition (non-prototype) PTE, else None."""
    if pte & VALID or (pte & TRANSITION and not pte & PROTOTYPE):
        return pte & PFN_MASK
    return None


def linux_leaf_frame(pte: int) -> int | None:
    """Physical page base of a resident Linux x86-64 PTE (present, or L1TF-inverted PROT_NONE), else None."""
    if pte & VALID:
        return pte & PFN_MASK
    if pte & PROTNONE:
        return (~pte) & PFN_MASK
    return None


class PageTables:
    """Translation for one address space. ``kernel`` resolves Windows prototype PTE addresses (kernel VAs).
    Set ``os_mode="linux"`` (and ``levels=5`` for LA57) for Linux x86-64 page tables."""

    def __init__(self, physical: Any, dtb: int, kernel: PageTables | None = None, *,
                 os_mode: str = "windows", levels: int = 4) -> None:
        if os_mode not in ("windows", "linux"):
            raise ValueError(f"Unsupported page-table OS mode: {os_mode}")
        if levels not in (4, 5):
            raise ValueError(f"Unsupported paging levels: {levels}")
        self.physical = physical
        self.size = len(physical)
        self.dtb = dtb & PFN_MASK
        self.kernel = kernel
        self.os_mode = os_mode
        self.levels = levels
        self._top_shift = 48 if levels == 5 else 39
        self._upper_shifts = (48, 39, 30, 21) if levels == 5 else (39, 30, 21)
        self._tables: dict[int, np.ndarray] = {}

    def table(self, base: int) -> np.ndarray | None:
        if base < 0 or base + PAGE > self.size:
            return None
        cached = self._tables.get(base)
        if cached is None:
            raw = bytes(self.physical[base : base + PAGE])
            if len(raw) != PAGE:
                return None
            cached = np.frombuffer(raw, np.uint64)
            if len(self._tables) > 8192:
                self._tables.clear()
            self._tables[base] = cached
        return cached

    def _next(self, entry: int) -> int | None:
        if self.os_mode == "linux":
            return (entry & PFN_MASK) if entry & VALID else None
        # Windows: a paging-structure page may itself sit on the standby list (transition) and still be readable.
        return leaf_frame(entry)

    def _large_frame(self, entry: int, shift: int) -> int | None:
        if shift not in (21, 30) or not entry & LARGE:
            return None
        if entry & VALID:
            frame = entry & PFN_MASK
        elif self.os_mode == "linux" and entry & PROTNONE:
            frame = (~entry) & PFN_MASK
        else:
            return None
        return frame & ~((1 << shift) - 1)

    def _is_canonical(self, va: int) -> bool:
        if self.levels == 5:
            return 0 <= va < (1 << 56) or 0xFF00_0000_0000_0000 <= va < (1 << 64)
        return 0 <= va < (1 << 47) or 0xFFFF_8000_0000_0000 <= va < (1 << 64)

    def translate(self, va: int) -> int | None:
        """Physical address of ``va`` or None when the page is not resident or ``va`` is not canonical."""
        if not self._is_canonical(va):
            return None
        base = self.dtb
        for shift in self._upper_shifts:
            table = self.table(base)
            if table is None:
                return None
            entry = int(table[(va >> shift) & 0x1FF])
            if (frame := self._large_frame(entry, shift)) is not None:
                return frame + (va & ((1 << shift) - 1))
            nxt = self._next(entry)
            if nxt is None:
                return None
            base = nxt
        table = self.table(base)
        if table is None:
            return None
        frame = self.resolve_pte(int(table[(va >> 12) & 0x1FF]))
        return None if frame is None else frame + (va & 0xFFF)

    def resolve_pte(self, pte: int) -> int | None:
        resolved = self.resolve(pte)
        return None if resolved is None else resolved[0]

    def resolve(self, pte: int) -> tuple[int, str, int | None] | None:
        """(frame, kind, prototype PTE address) of a leaf PTE, or None when not resident."""
        if self.os_mode == "linux":
            if pte & VALID:
                return pte & PFN_MASK, "VALID", None
            if pte & PROTNONE:
                # Linux 4.19+ L1TF mitigation stores non-present PROT_NONE PFNs bit-inverted.
                return (~pte) & PFN_MASK, "PROTNONE", None
            return None
        frame = leaf_frame(pte)
        if frame is not None:
            return frame, "VALID" if pte & VALID else "TRANSITION", None
        if pte & PROTOTYPE and not pte & VALID and self.kernel is not None:
            proto = pte >> 16
            if proto == VAD_PROTO or proto == 0:
                return None
            address = canonical(proto)
            where = self.kernel.translate(address)
            if where is None or where + 8 > self.size:
                return None
            raw = bytes(self.physical[where : where + 8])
            if len(raw) != 8:
                return None
            entry = int.from_bytes(raw, "little")
            frame = leaf_frame(entry)
            if frame is None:
                return None
            return frame, "PROTOTYPE_VALID" if entry & VALID else "PROTOTYPE_TRANSITION", address
        return None

    def read_kernel_qword(self, va: int) -> int | None:
        raw = self.read_virtual(va, 8)
        return None if raw is None else int.from_bytes(raw, "little")

    def read_virtual(self, va: int, size: int) -> bytes | None:
        """Read ``size`` bytes starting at virtual address ``va`` across page boundaries, or None on fault."""
        out = bytearray()
        while len(out) < size:
            address = va + len(out)
            where = self.translate(address)
            take = min(size - len(out), PAGE - (address & (PAGE - 1)))
            if where is None or where + take > self.size:
                return None
            chunk = bytes(self.physical[where : where + take])
            if len(chunk) != take:
                return None
            out.extend(chunk)
        return bytes(out)

    def resident_pages(self, lo: int, hi: int) -> Iterator[tuple[int, int]]:
        """(virtual page, physical page base) for every resident page in [lo, hi), in address order."""
        for va, frame, _kind, _proto in self._walk(self.dtb, self._top_shift, 0, lo, hi):
            yield va, frame

    def resident_entries(self, lo: int, hi: int) -> Iterator[tuple[int, int, str, int | None]]:
        """As ``resident_pages``, with each page's kind (see ``resolve``, or LARGE) and prototype PTE address."""
        yield from self._walk(self.dtb, self._top_shift, 0, lo, hi)

    def leaf_entries(self, lo: int, hi: int) -> Iterator[tuple[int, int]]:
        """(virtual page, raw PTE) for every non-zero 4 KiB leaf entry in [lo, hi), resident or not."""
        def walk(base: int, shift: int, prefix: int) -> Iterator[tuple[int, int]]:
            table = self.table(base)
            if table is None:
                return
            first = max(0, (lo - prefix) >> shift) if lo > prefix else 0
            for index in range(first, min(511, (hi - 1 - prefix) >> shift) + 1):
                entry = int(table[index])
                if not entry:
                    continue
                va = prefix + (index << shift)
                if shift == 12:
                    yield va, entry
                elif not (entry & VALID and entry & LARGE) and (nxt := self._next(entry)) is not None:
                    yield from walk(nxt, shift - 9, va)

        yield from walk(self.dtb, self._top_shift, 0)

    def _walk(self, base: int, shift: int, prefix: int, lo: int, hi: int
              ) -> Iterator[tuple[int, int, str, int | None]]:
        table = self.table(base)
        if table is None:
            return
        span = 1 << shift
        first = max(0, (lo - prefix) >> shift) if lo > prefix else 0
        last = min(511, (hi - 1 - prefix) >> shift)
        for index in range(first, last + 1):
            entry = int(table[index])
            if not entry:
                continue
            va = prefix + index * span
            if shift == 12:
                resolved = self.resolve(entry)
                if resolved is not None:
                    yield va, *resolved
                continue
            if (frame0 := self._large_frame(entry, shift)) is not None:
                for off in range(max(0, lo - va) & ~(PAGE - 1), min(span, hi - va), PAGE):
                    yield va + off, frame0 + off, "LARGE", None
                continue
            nxt = self._next(entry)
            if nxt is not None:
                yield from self._walk(nxt, shift - 9, va, lo, hi)
