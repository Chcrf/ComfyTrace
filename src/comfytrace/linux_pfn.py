"""Linux ``struct page`` / ``vmemmap`` PFN provenance, XArray page-cache ordering, and page consistency.

This is the Linux x86-64 counterpart to Windows ``PfnSupport`` (``bitmaps.py``):

* **Page origin (``origin``, ``copy_origin``):** Each physical frame ``pfn`` has a ``struct page`` at
  ``vmemmap_base + pfn * sizeof(struct page)``. Tail pages of multi-page folios follow ``compound_head`` to
  their head page. A head ``_refcount <= 0`` is ``FREED``; ``mapping & 3`` (anon or movable) or
  ``mapping == 0`` is ``PRIVATE``; a canonical ``struct address_space *`` in ``mapping`` is resolved through
  ``inode -> i_sb -> s_type -> name`` to ``SHARED_MEMORY`` (``tmpfs``/``shmem``), ``FILE_IMAGE``, or
  ``FILE_DATA`` (with backing path walked from ``inode->i_dentry``).
* **File-cache & shmem page ordering (``PAGE_CACHE_ORDER``):** All pages of one file or ``shmem``/``tmpfs``
  buffer share ``page->mapping`` and carry their logical page offset in ``page->index`` (plus subpage offset
  for multi-page folios). Consecutive pages are looked up directly in ``mapping->i_pages`` (Linux XArray)
  and verified back against each frame's own ``struct page``.
* **Private page ordering (``PAGE_TABLE_ORDER``):** Anonymous pages owned by captured user processes are
  ordered by virtual address through their ``mm->pgd`` page tables, with ``Gaps`` for missing or freed frames.
* **Page consistency (``mapping_problem``):** Checks a mapped frame's ``struct page`` refcount so freed
  frames copied mid-acquisition are flagged ``PFN_FREED`` (``STALE_MAPPING``) just as on Windows.

Every struct size and member offset is resolved dynamically from the Volatility 3 Linux ISF symbol table.
"""

from __future__ import annotations

import collections
import logging
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from comfytrace.bitmaps import Gaps, _group
from comfytrace.models import PageOrigin
from comfytrace.pagetable import PageTables, canonical

log = logging.getLogger(__name__)

PAGE = 4096
MAX_HOLE_PAGES = 8
MAX_GAP_PAGES = 8
MAX_FOLIO_PAGES = 1024
SHMEM_FS_TYPES = frozenset({"tmpfs", "shmem", "shm", "hugetlbfs", "devtmpfs", "ramfs"})
KERNEL_MIN_4 = 0xFFFF_8000_0000_0000
KERNEL_MIN_5 = 0xFF00_0000_0000_0000
DEFAULT_VMEMMAP_BASE_4 = 0xFFFF_EA00_0000_0000
DEFAULT_VMEMMAP_BASE_5 = 0xFFD4_0000_0000_0000
DEFAULT_VMEMMAP_BASE = DEFAULT_VMEMMAP_BASE_4


def _resolve_member(symbol_space: Any, type_obj: Any, target: str, depth: int = 0) -> tuple[int, Any] | None:
    """Recursively locate a named member inside a struct/union (including anonymous nested structs/unions)."""
    if depth > 6:
        return None
    if hasattr(type_obj, "vol") and "members" not in type_obj.vol and hasattr(type_obj.vol, "type_name"):
        try:
            type_obj = symbol_space.get_type(type_obj.vol.type_name)
        except Exception:  # noqa: BLE001
            return None
    members = getattr(getattr(type_obj, "vol", None), "members", None)
    if not members:
        return None
    if target in members:
        off, child = members[target]
        return int(off), child
    for name, (off, child) in members.items():
        if name.startswith("unnamed_") or "unnamed" in name or name in ("", "u", "d_u"):
            sub = _resolve_member(symbol_space, child, target, depth + 1)
            if sub is not None:
                return int(off) + sub[0], sub[1]
    return None


def _member_offset(symbol_space: Any, table: str, struct_name: str, *candidates: str) -> int | None:
    try:
        st = symbol_space.get_type(f"{table}!{struct_name}")
    except Exception:  # noqa: BLE001
        return None
    for name in candidates:
        found = _resolve_member(symbol_space, st, name)
        if found is not None:
            return found[0]
    return None


@dataclass(frozen=True)
class LinuxPageLayout:
    """Offsets resolved from the captured kernel's Linux ISF symbol table."""

    page_size: int
    flags_offset: int
    compound_head_offset: int
    mapping_offset: int
    index_offset: int
    refcount_offset: int
    # address_space / XArray
    as_host_offset: int | None = None
    as_ipages_offset: int | None = None
    xa_head_offset: int | None = None
    xa_node_shift_offset: int | None = None
    xa_node_slots_offset: int | None = None
    # inode / super_block / dentry (optional for FILE_DATA vs SHARED_MEMORY & backing filenames)
    inode_sb_offset: int | None = None
    inode_mode_offset: int | None = None
    inode_dentry_offset: int | None = None
    sb_type_offset: int | None = None
    fstype_name_offset: int | None = None
    dentry_parent_offset: int | None = None
    dentry_name_len_offset: int | None = None
    dentry_name_ptr_offset: int | None = None
    dentry_alias_offset: int | None = None


def resolve_linux_page_layout(context: Any, table: str) -> LinuxPageLayout:
    """Resolve ``struct page``, ``address_space``, ``xarray``, and ``dentry`` offsets from the ISF."""
    ss = context.symbol_space
    page_type = ss.get_type(f"{table}!page")
    page_size = int(page_type.size)
    flags_off = _member_offset(ss, table, "page", "flags")
    head_off = _member_offset(ss, table, "page", "compound_head")
    mapping_off = _member_offset(ss, table, "page", "mapping")
    index_off = _member_offset(ss, table, "page", "__folio_index", "index")
    if index_off is None:
        index_off = _member_offset(ss, table, "folio", "index")
    refcount_off = _member_offset(ss, table, "page", "_refcount")
    if None in (flags_off, head_off, mapping_off, index_off, refcount_off):
        raise ValueError("required struct page members not found in Linux ISF")

    # Optional address_space / XArray offsets
    as_host = _member_offset(ss, table, "address_space", "host")
    as_ipages = _member_offset(ss, table, "address_space", "i_pages")
    xa_head = _member_offset(ss, table, "xarray", "xa_head")
    xa_shift = _member_offset(ss, table, "xa_node", "shift")
    xa_slots = _member_offset(ss, table, "xa_node", "slots")

    # Optional inode / super_block / dentry offsets
    inode_sb = _member_offset(ss, table, "inode", "i_sb")
    inode_mode = _member_offset(ss, table, "inode", "i_mode")
    inode_dentry = _member_offset(ss, table, "inode", "i_dentry")
    sb_type = _member_offset(ss, table, "super_block", "s_type")
    fstype_name = _member_offset(ss, table, "file_system_type", "name")
    dentry_parent = _member_offset(ss, table, "dentry", "d_parent")
    dentry_name_off = _member_offset(ss, table, "dentry", "d_name")
    qstr_len = _member_offset(ss, table, "qstr", "len")
    qstr_name = _member_offset(ss, table, "qstr", "name")
    dentry_name_len = (dentry_name_off + qstr_len) if (dentry_name_off is not None and qstr_len is not None) else None
    dentry_name_ptr = (dentry_name_off + qstr_name) if (dentry_name_off is not None and qstr_name is not None) else None
    dentry_alias = _member_offset(ss, table, "dentry", "d_alias")

    return LinuxPageLayout(
        page_size=page_size,
        flags_offset=flags_off,  # type: ignore[arg-type]
        compound_head_offset=head_off,  # type: ignore[arg-type]
        mapping_offset=mapping_off,  # type: ignore[arg-type]
        index_offset=index_off,  # type: ignore[arg-type]
        refcount_offset=refcount_off,  # type: ignore[arg-type]
        as_host_offset=as_host,
        as_ipages_offset=as_ipages,
        xa_head_offset=xa_head,
        xa_node_shift_offset=xa_shift,
        xa_node_slots_offset=xa_slots,
        inode_sb_offset=inode_sb,
        inode_mode_offset=inode_mode,
        inode_dentry_offset=inode_dentry,
        sb_type_offset=sb_type,
        fstype_name_offset=fstype_name,
        dentry_parent_offset=dentry_parent,
        dentry_name_len_offset=dentry_name_len,
        dentry_name_ptr_offset=dentry_name_ptr,
        dentry_alias_offset=dentry_alias,
    )


@dataclass
class LinuxPfnSupport:
    """Linux kernel ``vmemmap`` PFN support: page origin, consistency checks, and XArray/page-table ordering."""

    kernel: PageTables
    vmemmap_base: int
    layout: LinuxPageLayout
    process_tables: list[tuple[int, PageTables, list[tuple[int, int]]]] = field(default_factory=list)
    _process_tables_loader: Any = field(default=None, repr=False)
    _as_cache: dict[int, tuple[PageOrigin, int | None, str | None]] = field(default_factory=dict, repr=False)
    _pfn_map_cache: dict[int, tuple[int, int, PageTables]] | None = field(default=None, repr=False)

    @property
    def _kernel_min(self) -> int:
        return KERNEL_MIN_5 if self.kernel.levels == 5 else KERNEL_MIN_4

    def _is_kernel_ptr(self, value: int) -> bool:
        return value >= self._kernel_min

    def _read_page_raw(self, pfn: int) -> bytes | None:
        if pfn < 0:
            return None
        return self.kernel.read_virtual(self.vmemmap_base + pfn * self.layout.page_size, self.layout.page_size)

    def _head_info(self, pfn: int) -> tuple[int, bytes, int] | None:
        """Return ``(head_pfn, head_raw, subpage_offset)`` for ``pfn``, following ``compound_head`` on tail pages."""
        raw = self._read_page_raw(pfn)
        if raw is None:
            return None
        L = self.layout
        chead = struct.unpack_from("<Q", raw, L.compound_head_offset)[0]
        if chead & 1:
            head_va = chead & ~1
            if not self._is_kernel_ptr(head_va) or head_va < self.vmemmap_base:
                return None
            delta = head_va - self.vmemmap_base
            if delta % L.page_size:
                return None
            head_pfn = delta // L.page_size
            subpage = pfn - head_pfn
            if not 0 <= subpage < MAX_FOLIO_PAGES:
                return None
            head_raw = self._read_page_raw(head_pfn)
            if head_raw is None:
                return None
            return head_pfn, head_raw, subpage
        return pfn, raw, 0

    def page_mapping_index(self, pfn: int) -> tuple[int, int] | None:
        """Return ``(address_space_va, logical_page_index)`` if ``pfn`` is a live file/shmem page, else None."""
        info = self._head_info(pfn)
        if info is None:
            return None
        _head_pfn, head_raw, subpage = info
        L = self.layout
        refcount = struct.unpack_from("<i", head_raw, L.refcount_offset)[0]
        if refcount <= 0:
            return None
        mapping = struct.unpack_from("<Q", head_raw, L.mapping_offset)[0]
        if not mapping or (mapping & 3) or not self._is_kernel_ptr(mapping):
            return None
        index = struct.unpack_from("<Q", head_raw, L.index_offset)[0] + subpage
        return mapping, index

    def _read_cstring(self, va: int, max_len: int = 256) -> str | None:
        if not self._is_kernel_ptr(va):
            return None
        data = self.kernel.read_virtual(va, max_len)
        if data is None:
            # Try shorter read up to page boundary
            rem = PAGE - (va & (PAGE - 1))
            data = self.kernel.read_virtual(va, min(max_len, rem))
            if data is None:
                return None
        nul = data.find(0)
        raw = data[:nul] if nul >= 0 else data
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return text if text and text.isprintable() else None

    def _dentry_path(self, inode_va: int) -> str | None:
        L = self.layout
        if None in (L.inode_dentry_offset, L.dentry_alias_offset, L.dentry_parent_offset,
                    L.dentry_name_len_offset, L.dentry_name_ptr_offset):
            return None
        first = self.kernel.read_kernel_qword(inode_va + L.inode_dentry_offset)  # type: ignore[operator]
        if not first or not self._is_kernel_ptr(first):
            return None
        dentry = first - L.dentry_alias_offset  # type: ignore[operator]
        parts: list[str] = []
        seen: set[int] = set()
        for _ in range(32):
            if not self._is_kernel_ptr(dentry) or dentry in seen:
                break
            seen.add(dentry)
            parent = self.kernel.read_kernel_qword(dentry + L.dentry_parent_offset)  # type: ignore[operator]
            len_raw = self.kernel.read_virtual(dentry + L.dentry_name_len_offset, 4)  # type: ignore[operator]
            ptr = self.kernel.read_kernel_qword(dentry + L.dentry_name_ptr_offset)  # type: ignore[operator]
            if parent is None or len_raw is None or not ptr or not self._is_kernel_ptr(ptr):
                break
            name_len = struct.unpack("<I", len_raw)[0]
            if parent == dentry:
                break
            if 0 < name_len <= 255:
                name_bytes = self.kernel.read_virtual(ptr, name_len)
                if name_bytes is None:
                    break
                try:
                    part = name_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    break
                if part and part != "/" and part.isprintable():
                    parts.append(part)
            dentry = parent
        if not parts:
            return None
        return "/" + "/".join(reversed(parts))

    def _classify_address_space(self, mapping: int) -> tuple[PageOrigin, int | None, str | None]:
        cached = self._as_cache.get(mapping)
        if cached is not None:
            return cached
        L = self.layout
        result: tuple[PageOrigin, int | None, str | None] = (PageOrigin.FILE_UNCONFIRMED, None, None)
        if L.as_host_offset is not None and L.inode_sb_offset is not None \
                and L.sb_type_offset is not None and L.fstype_name_offset is not None:
            inode = self.kernel.read_kernel_qword(mapping + L.as_host_offset)
            if inode and self._is_kernel_ptr(inode):
                sb = self.kernel.read_kernel_qword(inode + L.inode_sb_offset)
                fs_type = (self.kernel.read_kernel_qword(sb + L.sb_type_offset)
                           if sb and self._is_kernel_ptr(sb) else None)
                name_ptr = (self.kernel.read_kernel_qword(fs_type + L.fstype_name_offset)
                            if fs_type and self._is_kernel_ptr(fs_type) else None)
                fs_name = self._read_cstring(name_ptr, 64) if name_ptr else None
                if fs_name is not None:
                    if fs_name.lower() in SHMEM_FS_TYPES:
                        result = (PageOrigin.SHARED_MEMORY, inode, None)
                    else:
                        mode_raw = (self.kernel.read_virtual(inode + L.inode_mode_offset, 2)
                                    if L.inode_mode_offset is not None else None)
                        mode = struct.unpack("<H", mode_raw)[0] if mode_raw is not None else 0
                        fname = self._dentry_path(inode)
                        is_exec = bool(mode & 0o111) or (
                            fname is not None and (".so" in fname or fname.endswith(".ko"))
                        )
                        origin = PageOrigin.FILE_IMAGE if is_exec else PageOrigin.FILE_DATA
                        result = (origin, inode, fname if origin == PageOrigin.FILE_DATA else None)
        self._as_cache[mapping] = result
        return result

    def origin(self, pfn: int) -> PageOrigin:
        return self._origin(pfn)[0]

    def _origin(self, pfn: int) -> tuple[PageOrigin, str | None]:
        info = self._head_info(pfn)
        if info is None:
            return PageOrigin.UNKNOWN, None
        _head_pfn, head_raw, _subpage = info
        L = self.layout
        refcount = struct.unpack_from("<i", head_raw, L.refcount_offset)[0]
        if refcount <= 0:
            return PageOrigin.FREED, None
        mapping = struct.unpack_from("<Q", head_raw, L.mapping_offset)[0]
        if not mapping or (mapping & 3):
            return PageOrigin.PRIVATE, None
        if not self._is_kernel_ptr(mapping):
            return PageOrigin.UNKNOWN, None
        origin, _inode, fname = self._classify_address_space(mapping)
        return origin, fname

    def copy_origin(self, pfns: Iterable[int], nbytes: int) -> dict[str, Any]:
        """One copy of a record: count pages by origin and collect backing file names for ``FILE_DATA`` pages."""
        pages: collections.Counter[str] = collections.Counter()
        names: set[str] = set()
        for pfn in dict.fromkeys(pfns):
            origin, fname = self._origin(pfn)
            pages[str(origin)] += 1
            if fname:
                names.add(fname)
        result: dict[str, Any] = {"pages": dict(sorted(pages.items())), "bytes": nbytes}
        if names:
            result["backing_file_names"] = sorted(names)
        return result

    def mapping_problem(self, va: int, frame: int, kind: str, proto: int | None) -> str | None:
        """Check whether ``frame`` is still a live page in ``vmemmap`` (not freed during acquisition)."""
        if kind == "LARGE":
            return None
        info = self._head_info(frame >> 12)
        if info is None:
            return None
        _head_pfn, head_raw, _subpage = info
        refcount = struct.unpack_from("<i", head_raw, self.layout.refcount_offset)[0]
        if refcount <= 0:
            return "PFN_FREED"
        return None

    def xarray_lookup(self, address_space_va: int, index: int) -> int | None:
        """Look up logical page ``index`` in ``address_space->i_pages`` XArray; return physical PFN or None."""
        L = self.layout
        if index < 0 or None in (L.as_ipages_offset, L.xa_head_offset, L.xa_node_shift_offset, L.xa_node_slots_offset):
            return None
        entry = self.kernel.read_kernel_qword(address_space_va + L.as_ipages_offset + L.xa_head_offset)  # type: ignore[operator]
        if not entry:
            return None
        # A non-node root entry only ever represents index 0.
        if not ((entry & 3) == 2 and entry > 4096):
            if index != 0:
                return None
        else:
            # Walk internal xa_node pointers (low 2 bits == 2 and value > 4096).
            checked_top = False
            for _ in range(12):
                if (entry & 3) == 2 and entry > 4096:
                    node = entry - 2
                    if not self._is_kernel_ptr(node):
                        return None
                    shift_raw = self.kernel.read_virtual(node + L.xa_node_shift_offset, 1)  # type: ignore[operator]
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
                    entry = self.kernel.read_kernel_qword(node + L.xa_node_slots_offset + slot_idx * 8)  # type: ignore[operator]
                    if not entry:
                        return None
                    # Follow sibling entry (low 2 bits == 2, <= 4096) to its canonical slot in the same node.
                    if (entry & 3) == 2 and entry <= 4096:
                        sib = entry >> 2
                        if sib >= 64:
                            return None
                        entry = self.kernel.read_kernel_qword(node + L.xa_node_slots_offset + sib * 8)  # type: ignore[operator]
                        if not entry:
                            return None
                    continue
                break
        # Leaf must be a canonical struct folio/page pointer (aligned, low 2 bits == 0).
        if not entry or (entry & 3) != 0 or not self._is_kernel_ptr(entry) or entry < self.vmemmap_base:
            return None
        delta = entry - self.vmemmap_base
        if delta % L.page_size:
            return None
        head_pfn = delta // L.page_size
        head_raw = self._read_page_raw(head_pfn)
        if head_raw is None:
            return None
        head_ref = struct.unpack_from("<i", head_raw, L.refcount_offset)[0]
        if head_ref <= 0:
            return None
        head_map = struct.unpack_from("<Q", head_raw, L.mapping_offset)[0]
        if head_map != address_space_va:
            return None
        head_index = struct.unpack_from("<Q", head_raw, L.index_offset)[0]
        subpage = index - head_index
        if not 0 <= subpage < MAX_FOLIO_PAGES:
            return None
        if subpage > 0:
            # Verify the candidate tail frame actually belongs to this multi-page folio.
            tail_raw = self._read_page_raw(head_pfn + subpage)
            if tail_raw is None:
                return None
            tail_chead = struct.unpack_from("<Q", tail_raw, L.compound_head_offset)[0]
            if tail_chead != (entry | 1):
                return None
        return head_pfn + subpage

    def _ensure_pfn_index(self) -> dict[int, tuple[int, int, PageTables]]:
        """Lazily map resident user physical PFNs -> ``(dtb, va, PageTables)`` for private page ordering."""
        if self._pfn_map_cache is None:
            if not self.process_tables and self._process_tables_loader is not None:
                self.process_tables = list(self._process_tables_loader())
            index: dict[int, tuple[int, int, PageTables]] = {}
            for dtb, pt, vmas in self.process_tables:
                for lo, hi in vmas:
                    for va, frame in pt.resident_pages(lo & ~(PAGE - 1), hi + 1):
                        pfn = frame >> 12
                        if pfn not in index:
                            index[pfn] = (dtb, va, pt)
            self._pfn_map_cache = index
        return self._pfn_map_cache

    def page_order_stream(self, physical: Any, start: int, max_bytes: int) -> tuple[bytes, str, list[int]] | None:
        """Follow a torn file/shmem or private page in logical order, returning ``(bytes, method, frames)``."""
        pfn = start >> 12
        first = bytes(physical[start : (pfn + 1) << 12])
        if not first:
            return None
        frames = [pfn]
        data = bytearray(first)

        # 1. File-cache or shmem/tmpfs page: follow mapping->i_pages XArray in logical index order
        mi = self.page_mapping_index(pfn)
        if mi is not None:
            mapping, idx = mi
            # The seed itself must be the page the XArray names for its own (mapping, index); a stale alias is
            # rejected outright rather than substituted or reordered as private memory.
            if self.xarray_lookup(mapping, idx) != pfn:
                return None
            while len(data) < max_bytes:
                idx += 1
                next_pfn = self.xarray_lookup(mapping, idx)
                if next_pfn is None:
                    break
                frame = next_pfn << 12
                page_bytes = bytes(physical[frame : frame + PAGE])
                if len(page_bytes) != PAGE:
                    break
                data += page_bytes
                frames.append(next_pfn)
            return bytes(data), "PAGE_CACHE_ORDER", frames

        # 2. Private anonymous page: follow virtual pages in owning process page tables
        origin = self.origin(pfn)
        if origin != PageOrigin.PRIVATE or (not self.process_tables and self._process_tables_loader is None):
            return None
        owner = self._ensure_pfn_index().get(pfn)
        if owner is None:
            return None
        _dtb, va, pt = owner
        while len(data) < max_bytes:
            va += PAGE
            frame = pt.translate(va)
            if frame is None:
                break
            next_pfn = frame >> 12
            if self.origin(next_pfn) == PageOrigin.FREED:
                break
            page_bytes = bytes(physical[frame : frame + PAGE])
            if len(page_bytes) != PAGE:
                break
            data += page_bytes
            frames.append(next_pfn)
        return bytes(data), "PAGE_TABLE_ORDER", frames

    def ordered_regions(self, physical: Any, pages: list[int],
                        min_pages: int = 16) -> Iterator[tuple[str, bytes, list[int], str, dict[int, str]]]:
        """Order opaque 32-bit pixel pages by Linux ``address_space`` XArray and process page tables."""
        by_mapping: dict[int, dict[int, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
        private_pages: list[int] = []
        for pfn in pages:
            mi = self.page_mapping_index(pfn)
            if mi is not None:
                by_mapping[mi[0]][mi[1]].add(pfn)
            elif self.origin(pfn) == PageOrigin.PRIVATE:
                private_pages.append(pfn)

        # 1. File / shmem / tmpfs regions in exact XArray logical page order. Only scanned pages that the XArray
        # names for their own index anchor a run; interior holes are then filled from the XArray itself.
        for mapping, candidates in by_mapping.items():
            lookups: dict[int, int | None] = {}

            def canonical_pfn(idx: int, mapping: int = mapping, lookups: dict[int, int | None] = lookups) -> int | None:
                if idx not in lookups:
                    lookups[idx] = self.xarray_lookup(mapping, idx)
                return lookups[idx]

            anchors: list[tuple[int, int]] = []
            for idx, cand in candidates.items():
                canon = canonical_pfn(idx)
                if canon is not None and canon in cand:
                    anchors.append((idx, canon))
            for group in _group(anchors, 1, MAX_HOLE_PAGES):
                if len(group) < min_pages:
                    continue
                first_idx, last_idx = group[0][0], group[-1][0]
                data = bytearray()
                frames: list[int] = []
                start = first_idx
                for idx in range(first_idx, last_idx + 1):
                    pfn = canonical_pfn(idx)
                    page_bytes = bytes(physical[pfn << 12 : (pfn + 1) << 12]) if pfn is not None else b""
                    if pfn is None or len(page_bytes) != PAGE:
                        if len(frames) >= min_pages:
                            yield f"MAPPING-0x{mapping:X}+{start:X}", bytes(data), frames, "PAGE_CACHE_ORDER", {}
                        data, frames, start = bytearray(), [], idx + 1
                        continue
                    data += page_bytes
                    frames.append(pfn)
                if len(frames) >= min_pages:
                    yield f"MAPPING-0x{mapping:X}+{start:X}", bytes(data), frames, "PAGE_CACHE_ORDER", {}

        # 2. Private anonymous regions in virtual page-table order
        if not private_pages or (not self.process_tables and self._process_tables_loader is None):
            return
        pfn_index = self._ensure_pfn_index()
        by_proc: dict[int, tuple[PageTables, list[tuple[int, int]]]] = {}
        for pfn in private_pages:
            hit = pfn_index.get(pfn)
            if hit is None:
                continue
            dtb, va, pt = hit
            if dtb not in by_proc:
                by_proc[dtb] = (pt, [])
            by_proc[dtb][1].append((va >> 12, pfn))

        for dtb, (pt, keys) in by_proc.items():
            for group in _group(keys, 1, MAX_HOLE_PAGES):
                if len(group) < min_pages:
                    continue
                frames = []
                gaps = Gaps()
                missing: list[tuple[str, int | None]] = []
                data, start = bytearray(), group[0][0]
                for vpn in range(group[0][0], group[-1][0] + 1):
                    frame = pt.translate(vpn << 12)
                    page_bytes = bytes(physical[frame : frame + PAGE]) if frame is not None else b""
                    if frame is None or len(page_bytes) != PAGE:
                        missing.append(("PAGE_NOT_PRESENT", None))
                    elif self.origin(frame >> 12) == PageOrigin.FREED:
                        missing.append(("STALE_MAPPING", frame))
                    else:
                        if frames:
                            for reason, stale in missing:
                                gaps[len(frames)] = reason
                                if stale is not None:
                                    gaps.uncertain[len(frames)] = (stale >> 12, bytes(physical[stale : stale + PAGE]))
                                frames.append(-1)
                                data += bytes(PAGE)
                        else:
                            start = vpn
                        missing = []
                        data += page_bytes
                        frames.append(frame >> 12)
                        continue
                    if len(missing) > MAX_GAP_PAGES:
                        if len(frames) - len(gaps) >= min_pages:
                            yield f"PT-0x{dtb:X}+{start:X}", bytes(data), frames, "PAGE_TABLE_ORDER", gaps
                        frames, gaps, missing, data = [], Gaps(), [], bytearray()
                if len(frames) - len(gaps) >= min_pages:
                    yield f"PT-0x{dtb:X}+{start:X}", bytes(data), frames, "PAGE_TABLE_ORDER", gaps


def open_linux_pfn_support(context: Any, kernel_module_name: str, kernel: PageTables,
                           process_tables: list[tuple[int, PageTables, list[tuple[int, int]]]] | None = None,
                           process_tables_loader: Any = None,
                           ) -> LinuxPfnSupport:
    """Construct ``LinuxPfnSupport`` from Volatility's resolved Linux kernel module and ISF."""
    kmod = context.modules[kernel_module_name]
    table = kmod.symbol_table_name
    layout = resolve_linux_page_layout(context, table)
    vmemmap_base = DEFAULT_VMEMMAP_BASE_5 if kernel.levels == 5 else DEFAULT_VMEMMAP_BASE_4
    try:
        sym_va = canonical(int(kmod.offset) + int(kmod.get_symbol("vmemmap_base").address), levels=kernel.levels)
        read_val = kernel.read_kernel_qword(sym_va)
        if read_val is not None and read_val >= (KERNEL_MIN_5 if kernel.levels == 5 else KERNEL_MIN_4):
            vmemmap_base = read_val
    except Exception:  # noqa: BLE001 - non-KASLR kernels may omit vmemmap_base symbol
        pass
    return LinuxPfnSupport(kernel, vmemmap_base, layout, list(process_tables or ()),
                           _process_tables_loader=process_tables_loader)
