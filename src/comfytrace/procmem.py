"""Fast random access to one process's resident memory.

Volatility supplies only supporting data here: the process layer's page-table translation and the VAD list
(start, end, mapped file name). Bytes are then read straight from the raw image through the translated
physical runs, so a structure walk over gigabytes of heap stays fast. Nothing is padded: a read that touches
an unmapped page raises ``Unreadable``.

Linux (``image_os="linux"``): the regions are VMAs ``(start, inclusive end, path or None)``. ld.so maps one ELF
module as several VMAs of the same file (header, text, rodata, relro, data), so consecutive VMAs of one path form
one ``MappedImage`` whose ``segments`` are those VMAs; anonymous VMAs between them (.bss, alignment gaps) do not
belong to it. Only shared objects (``*.so``, ``*.so.N``) and executables named ``python``, ``python3`` or
``python3.N`` become images, never other file-backed maps such as mmapped model weights. A module whose first page
is readable must carry a 64-bit little-endian x86-64 or AArch64 ELF header; an executable must show that header.
"""

from __future__ import annotations

import bisect
import logging
import mmap
import re
import struct
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PAGE = 4096
USER_LIMIT = 1 << 47
FIVE_LEVEL_USER_LIMIT = 1 << 56
STORE_BASE = 1 << 62  # run offsets at or above this index ProcessMemory.store_data, not the image
PREFETCH_THREADS = 8
PREFETCH_BATCH = 4096  # pages touched per numpy gather
PE_EXPORT_PATH = "<PE export name>"  # MappedImage.path of an image named by its own export directory
IMAGE_OSES = ("windows", "linux")
ELF_LIBRARY = re.compile(r"[^/]+\.so(\.\d+)*")
ELF_PYTHON_EXECUTABLE = re.compile(r"python(3(\.\d+)?)?")
ELF_MACHINES = {62: "x86_64", 183: "aarch64"}  # LP64 little-endian: the CPython and c10 layouts read here
DELETED_SUFFIX = " (deleted)"
DT_NULL, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_GNU_HASH = 0, 4, 5, 6, 10, 0x6FFFFEF5
SYMBOL_CHAIN_LIMIT = 1 << 16  # hash-chain steps per lookup; a smeared table must not loop for long


class Unreadable(Exception):
    """The requested bytes are not resident (or the address is not a user-mode address)."""


class StaleMapping(Unreadable):
    """The page table names a frame beyond the end of the image: a mapping copied at another moment of the
    acquisition, with no bytes to read."""


def touch_pages(physical: mmap.mmap, offsets: np.ndarray) -> None:
    """Reads one byte at each of ``offsets`` (ascending int64 offsets into the mapping) from several threads.
    numpy releases the GIL while it gathers, so the threads' page faults overlap; when the call returns every
    page is in the page cache and mapped."""
    view = np.frombuffer(physical, np.uint8)
    batches = [offsets[i : i + PREFETCH_BATCH] for i in range(0, len(offsets), PREFETCH_BATCH)]

    def touch(worker: int) -> None:
        for batch in batches[worker::PREFETCH_THREADS]:
            view[batch]

    workers = [threading.Thread(target=touch, args=(k,)) for k in range(PREFETCH_THREADS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


@dataclass(frozen=True)
class MappedImage:
    """A VAD backed by an executable image (DLL, PYD, EXE), or on Linux the VMAs of one ELF module.

    ``segments`` (half-open ``(start, end)`` ranges) is empty for a Windows image, which is one VAD; for an ELF
    module it lists the module's own VMAs, and ``contains`` is true only inside them."""

    name: str
    path: str
    base: int
    end: int
    segments: tuple[tuple[int, int], ...] = ()
    kind: str = "pe"

    @property
    def size(self) -> int:
        return self.end - self.base

    def contains(self, address: int) -> bool:
        if not self.segments:
            return self.base <= address < self.end
        return any(start <= address < end for start, end in self.segments)

    def ranges(self) -> tuple[tuple[int, int], ...]:
        """Half-open address ranges that belong to the image."""
        return self.segments or ((self.base, self.end),)


def _elf_name(path: str) -> str:
    """Lower-case file name of a Linux map path, without the kernel's `` (deleted)`` marker."""
    if path.endswith(DELETED_SUFFIX):
        path = path[: -len(DELETED_SUFFIX)]
    return path.rsplit("/", 1)[-1].lower()


def elf_modules(vmas: list[tuple[int, int, str | None]]) -> list[MappedImage]:
    """Candidate ELF modules: runs of consecutive VMAs backed by the same file, uninterrupted by another file.
    Anonymous and pseudo (``[heap]``, ``[vdso]``) VMAs inside a run neither end it nor join it."""
    groups: list[tuple[str, list[tuple[int, int]]]] = []
    for start, end, path in sorted(vmas):
        if not path or not path.startswith("/"):
            continue
        if groups and groups[-1][0] == path:
            groups[-1][1].append((start, end + 1))
        else:
            groups.append((path, [(start, end + 1)]))
    images = []
    for path, segments in groups:
        name = _elf_name(path)
        if ELF_LIBRARY.fullmatch(name) or ELF_PYTHON_EXECUTABLE.fullmatch(name):
            images.append(MappedImage(name, path, segments[0][0], segments[-1][1], tuple(segments), "elf"))
    return images


@dataclass
class ProcessMemory:
    """Resident virtual runs of one process, each as (va, length, physical offset), sorted by va."""

    pid: int
    runs: list[tuple[int, int, int]]
    physical: Any
    images: list[MappedImage] = field(default_factory=list)
    layer: Any = None
    vads: list[tuple[int, int, str | None]] = field(default_factory=list)
    stale: dict[int, str] = field(default_factory=dict)
    store_data: bytes = b""
    store_status: dict[str, int] = field(default_factory=dict)
    image_os: str = "windows"
    user_limit: int = USER_LIMIT  # 1 << 56 for a Linux process under 5-level paging
    _prefetched: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.image_os not in IMAGE_OSES:
            raise ValueError(f"Unsupported image OS: {self.image_os}")
        self.runs.sort()
        self._starts = [r[0] for r in self.runs]
        self._disjoint = all(a[0] + a[1] <= b[0] for a, b in zip(self.runs, self.runs[1:], strict=False))
        self.vads = sorted(self.vads)
        self._vad_starts = [v[0] for v in self.vads]

    @classmethod
    def from_layer(cls, pid: int, layer: Any, vads: list[tuple[int, int, str | None]], physical: Any,
                   image_os: str = "windows") -> ProcessMemory:
        """Builds the run table with ``layer.mapping``. ``physical`` is the raw image (bytes or mmap) when the
        layer beneath is the image file; otherwise (None) reads go through ``layer.read``. ``vads`` are Windows
        VADs or, with ``image_os="linux"``, Linux VMAs; both are ``(start, inclusive end, file name or None)``."""
        if image_os not in IMAGE_OSES:
            raise ValueError(f"Unsupported image OS: {image_os}")
        runs: list[tuple[int, int, int]] = []
        images = []
        for start, end, name in vads:
            if image_os == "windows" and name and name.lower().endswith((".dll", ".pyd", ".exe")):
                base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
                images.append(MappedImage(base, name, start, end + 1))
            try:
                for va, _length, phys, mlen, _layer in layer.mapping(start, end - start + 1, ignore_errors=True):
                    if mlen <= 0:
                        continue
                    if runs and runs[-1][0] + runs[-1][1] == va and runs[-1][2] + runs[-1][1] == phys:
                        runs[-1] = (runs[-1][0], runs[-1][1] + mlen, runs[-1][2])
                    else:
                        runs.append((int(va), int(mlen), int(phys)))
            except Exception:  # noqa: BLE001 - a corrupt VAD only loses its own pages
                log.debug("mapping failed for PID %s VAD 0x%x-0x%x", pid, start, end)
        # A mapping above the 4-level limit exists only under 5-level paging (LA57), whose user limit is 2^56.
        user_limit = FIVE_LEVEL_USER_LIMIT if any(end >= USER_LIMIT for _start, end, _name in vads) else USER_LIMIT
        mem = cls(pid, runs, physical, images, layer, list(vads), image_os=image_os, user_limit=user_limit)
        if image_os == "linux":
            mem.images = [m for m in elf_modules(mem.vads) if mem._elf_module_ok(m)]
        return mem

    @classmethod
    def from_page_tables(cls, pid: int, tables: Any, vads: list[tuple[int, int, str | None]],
                         physical: Any, pfn: Any = None, store: Any = None, *,
                         image_os: str = "windows") -> ProcessMemory:
        """Builds the run table with our own page-table walker (valid, transition and prototype PTEs on Windows;
        present and PROT_NONE PTEs on Linux). A page whose frame lies beyond the image is left out. With ``pfn``
        (PfnSupport / LinuxPfnSupport), a page whose PFN entry disagrees with the mapping is kept but listed in
        ``stale`` with the reason. With ``store`` (StoreReader / LinuxStoreReader), compressed swap/store pages
        are decompressed and readable like resident pages, with no physical offset."""
        if image_os not in IMAGE_OSES:
            raise ValueError(f"Unsupported image OS: {image_os}")
        runs: list[tuple[int, int, int]] = []
        images = []
        stale: dict[int, str] = {}
        size = len(physical)
        user_limit = FIVE_LEVEL_USER_LIMIT if getattr(tables, "levels", 4) == 5 else USER_LIMIT
        for start, end, name in sorted(vads):
            if image_os == "windows" and name and name.lower().endswith((".dll", ".pyd", ".exe")):
                base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
                images.append(MappedImage(base, name, start, end + 1))
            if start >= user_limit:
                continue
            for va, phys, kind, proto in tables.resident_entries(start & ~(PAGE - 1), min(end + 1, user_limit)):
                problem = "FRAME_BEYOND_IMAGE" if phys + PAGE > size else (
                    pfn.mapping_problem(va, phys, kind, proto) if pfn is not None else None)
                if problem:
                    stale[va] = problem
                if problem == "FRAME_BEYOND_IMAGE":
                    continue
                if runs and runs[-1][0] + runs[-1][1] == va and runs[-1][2] + runs[-1][1] == phys:
                    runs[-1] = (runs[-1][0], runs[-1][1] + PAGE, runs[-1][2])
                else:
                    runs.append((va, PAGE, phys))
        store_data = bytearray()
        status: dict[str, int] = {}
        if store is not None:
            if hasattr(store, "iter_ptes"):
                iter_ptes = store.iter_ptes
            else:
                from comfytrace.memstore import store_ptes

                def iter_ptes(tbl: Any, lo: int, hi: int) -> Any:
                    return store_ptes(tbl, lo, hi, store.pagefile)

            # Every VAD: copy-on-write pages of image and file views are private and can be compressed too.
            for start, end, _name in sorted(vads):
                if start >= user_limit:
                    continue
                for va, key in iter_ptes(tables, start & ~(PAGE - 1), min(end + 1, user_limit)):
                    outcome, data = store.page(key)
                    label = outcome.split(":")[0]
                    status[label] = status.get(label, 0) + 1
                    if data is not None:
                        runs.append((va, PAGE, STORE_BASE + len(store_data)))
                        store_data += data
        mem = cls(pid, runs, physical, images, vads=list(vads), stale=stale, store_data=bytes(store_data),
                  store_status=status, image_os=image_os, user_limit=user_limit)
        if image_os == "linux":
            mem.images = [m for m in elf_modules(mem.vads) if mem._elf_module_ok(m)]
        return mem

    def vad(self, va: int) -> tuple[int, int, str | None] | None:
        """(start, inclusive end, backing file name or None) of the VAD holding ``va``."""
        i = bisect.bisect_right(self._vad_starts, va) - 1
        return self.vads[i] if i >= 0 and va <= self.vads[i][1] else None

    def vad_name(self, va: int) -> tuple[bool, str | None]:
        """(inside a VAD, its backing file name or None for private memory)."""
        v = self.vad(va)
        return (True, v[2]) if v else (False, None)

    @property
    def resident_bytes(self) -> int:
        return sum(r[1] for r in self.runs)

    def image(self, predicate: Any) -> MappedImage | None:
        """The first mapped image whose lower-case file name satisfies ``predicate``. The VAD names the file
        through kernel structures that can be paged out while the image itself stays resident, so unnamed VADs
        are also identified by the name in their own PE export directory (path ``<PE export name>``)."""
        named = next((m for m in self.images if predicate(m.name)), None)
        if named is not None or self.image_os != "windows":
            return named
        return next((m for m in self.pe_images() if predicate(m.name)), None)

    def _elf_header(self, base: int) -> tuple[int, int, int, int] | None:
        """(e_type, e_machine, e_phoff, e_phnum) of a supported 64-bit little-endian ELF header at ``base``, or
        None when the bytes there are not one. Raises ``Unreadable`` when the header is not resident."""
        head = self.read(base, 64)
        if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return None
        e_type, e_machine = struct.unpack_from("<HH", head, 16)
        e_phoff = struct.unpack_from("<Q", head, 32)[0]
        e_phentsize, e_phnum = struct.unpack_from("<HH", head, 54)
        if e_type not in (2, 3) or e_machine not in ELF_MACHINES or e_phentsize != 56 or not 0 < e_phnum <= 64:
            return None
        return e_type, e_machine, e_phoff, e_phnum

    def _elf_module_ok(self, image: MappedImage) -> bool:
        """A shared object by name is kept unless its readable first page is not a supported ELF header; an
        executable by name is kept only when that header is resident and valid."""
        try:
            return self._elf_header(image.base) is not None
        except Unreadable:
            return not ELF_PYTHON_EXECUTABLE.fullmatch(image.name)

    def elf_symbol(self, image: MappedImage, symbol: str) -> int | None:
        """Address of a defined dynamic symbol of an ELF module, from its own resident program headers, dynamic
        section and GNU or SysV hash table. None when any of those is not resident or not consistent, or the
        symbol is not exported. Every address that is used must lie inside the module."""
        if image.kind != "elf":
            return None
        try:
            return self._elf_symbol(image, symbol.encode())
        except (Unreadable, struct.error, ValueError):
            return None

    def _elf_symbol(self, image: MappedImage, symbol: bytes) -> int | None:
        header = self._elf_header(image.base)
        if header is None:
            return None
        e_type, _machine, e_phoff, e_phnum = header
        phdrs = self.read(image.base + e_phoff, 56 * e_phnum)
        loads, dynamic = [], None
        for i in range(e_phnum):
            p_type, _flags, p_offset, p_vaddr, _paddr, _filesz, p_memsz = struct.unpack_from("<IIQQQQQ", phdrs, 56 * i)
            if p_type == 1:
                loads.append((p_offset, p_vaddr))
            elif p_type == 2:
                dynamic = (p_vaddr, p_memsz)
        if not loads or dynamic is None:
            return None
        offset, vaddr = min(loads, key=lambda load: load[1])
        if offset != 0:
            return None  # the mapping begins with the file header only when the first load maps offset 0
        bias = image.base - (vaddr & ~(PAGE - 1))
        if e_type == 2 and bias != 0:
            return None  # a non-PIE executable is mapped at its link address

        def inside(address: int, length: int = 1) -> int:
            if not (image.contains(address) and image.contains(address + length - 1)):
                raise ValueError("outside the module")
            return address

        def rd(address: int, length: int) -> bytes:
            return self.read(inside(address, length), length)

        def pointer(value: int) -> int:
            # ld.so relocates DT_* pointers in place on most targets; an unrelocated one is an offset from bias.
            return value if image.contains(value) else inside(bias + value)

        dyn_va = inside(bias + dynamic[0], min(dynamic[1], 16))
        tags: dict[int, int] = {}
        for i in range(min(dynamic[1] // 16, 512)):
            tag, value = struct.unpack("<qQ", rd(dyn_va + 16 * i, 16))
            if tag == DT_NULL:
                break
            tags.setdefault(tag, value)
        if DT_STRTAB not in tags or DT_SYMTAB not in tags:
            return None
        strtab, symtab = pointer(tags[DT_STRTAB]), pointer(tags[DT_SYMTAB])
        strsz = tags.get(DT_STRSZ, 1 << 24)

        def symbol_at(index: int) -> int | None:
            st_name, _info, _other, st_shndx, st_value, _size = struct.unpack(
                "<IBBHQQ", rd(symtab + 24 * index, 24))
            if st_name + len(symbol) >= strsz or st_shndx == 0 or not st_value:
                return None
            if rd(strtab + st_name, len(symbol) + 1) != symbol + b"\0":
                return None
            return inside(st_value if e_type == 2 else bias + st_value)

        if DT_GNU_HASH in tags:
            table = pointer(tags[DT_GNU_HASH])
            nbuckets, symoffset, bloom_size, _shift = struct.unpack("<IIII", rd(table, 16))
            if not nbuckets or nbuckets > 1 << 24 or bloom_size > 1 << 24:
                return None
            digest = 5381
            for c in symbol:
                digest = (digest * 33 + c) & 0xFFFFFFFF
            buckets = table + 16 + 8 * bloom_size
            index = struct.unpack("<I", rd(buckets + 4 * (digest % nbuckets), 4))[0]
            if index < symoffset:
                return None
            chain = buckets + 4 * nbuckets
            for _ in range(SYMBOL_CHAIN_LIMIT):
                value = struct.unpack("<I", rd(chain + 4 * (index - symoffset), 4))[0]
                if value | 1 == digest | 1 and (found := symbol_at(index)) is not None:
                    return found
                if value & 1:
                    return None
                index += 1
            return None
        if DT_HASH in tags:
            table = pointer(tags[DT_HASH])
            nbucket, nchain = struct.unpack("<II", rd(table, 8))
            if not nbucket or nbucket > 1 << 24 or nchain > 1 << 24:
                return None
            digest = 0
            for c in symbol:
                digest = ((digest << 4) + c) & 0xFFFFFFFF
                digest = (digest ^ (digest >> 24 & 0xF0)) & 0x0FFFFFFF
            index = struct.unpack("<I", rd(table + 8 + 4 * (digest % nbucket), 4))[0]
            for _ in range(min(nchain, SYMBOL_CHAIN_LIMIT)):
                if not index or index >= nchain:
                    return None
                if (found := symbol_at(index)) is not None:
                    return found
                index = struct.unpack("<I", rd(table + 8 + 4 * (nbucket + index), 4))[0]
        return None

    def pe_images(self) -> list[MappedImage]:
        """Unnamed VADs that begin with a readable PE image, named by its export directory (cached)."""
        if getattr(self, "_pe_images", None) is None:
            found = []
            for start, end, name in self.vads:
                if name is None and (exported := self._pe_export_name(start)):
                    found.append(MappedImage(exported.lower(), PE_EXPORT_PATH, start, end + 1))
            self._pe_images = found
        return self._pe_images

    def _pe_export_name(self, base: int) -> str | None:
        try:
            head = self.read(base, PAGE)
            if head[:2] != b"MZ":
                return None
            pe = int.from_bytes(head[0x3C:0x40], "little")
            if not 0x40 <= pe <= PAGE - 0x108 or head[pe:pe + 4] != b"PE\0\0":
                return None
            optional = pe + 24
            magic = int.from_bytes(head[optional:optional + 2], "little")
            directories = optional + {0x20B: 112, 0x10B: 96}.get(magic, -optional)
            if directories < optional:
                return None
            export_rva = int.from_bytes(head[directories:directories + 4], "little")
            if not export_rva:
                return None
            name_rva = int.from_bytes(self.read(base + export_rva + 12, 4), "little")
            name = self.cstr(base + name_rva, 96) if name_rva else ""
        except Unreadable:
            return None
        return name if name and all(c.isalnum() or c in "._-" for c in name) else None

    def _read_physical(self, phys: int, length: int) -> bytes:
        if phys >= STORE_BASE:
            data = self.store_data[phys - STORE_BASE : phys - STORE_BASE + length]
            if len(data) != length:
                raise Unreadable(f"store offset 0x{phys - STORE_BASE:x} is beyond the decoded pages")
            return data
        if self.physical is not None:
            data = bytes(self.physical[phys : phys + length])
            if len(data) != length:
                # A page-table entry can name a frame beyond the end of the image (a stale or smeared mapping).
                raise Unreadable(f"physical 0x{phys:x} is beyond the image")
            return data
        raise Unreadable("no physical source")

    def physical_of(self, va: int) -> int | None:
        """Offset in the image, or None when ``va`` is not resident or was decompressed from the store."""
        if self.image_os == "linux" and self.physical is None:
            return None  # Layer offsets for segmented images and process cores are not flat physical addresses.
        i = bisect.bisect_right(self._starts, va) - 1
        if i >= 0:
            start, length, phys = self.runs[i]
            if va < start + length and phys < STORE_BASE:
                return phys + va - start
        return None

    def from_store(self, va: int) -> bool:
        i = bisect.bisect_right(self._starts, va) - 1
        return i >= 0 and va < self.runs[i][0] + self.runs[i][1] and self.runs[i][2] >= STORE_BASE

    def read(self, va: int, length: int) -> bytes:
        """Exactly ``length`` genuine bytes at ``va`` or ``Unreadable``."""
        if not 0x10000 <= va < self.user_limit or length < 0:
            raise Unreadable(hex(va))
        if self.physical is None:
            return self._read_layer(va, length)
        out = bytearray()
        cursor, end = va, va + length
        i = bisect.bisect_right(self._starts, cursor) - 1
        while cursor < end:
            if i < 0 or i >= len(self.runs) or not self.runs[i][0] <= cursor < self.runs[i][0] + self.runs[i][1]:
                reason = self.stale.get(cursor & ~(PAGE - 1))
                raise StaleMapping(f"{hex(cursor)}: {reason}") if reason else Unreadable(hex(cursor))
            start, run_len, phys = self.runs[i]
            take = min(end, start + run_len) - cursor
            out += self._read_physical(phys + cursor - start, take)
            cursor += take
            i += 1
        return bytes(out)

    def _read_layer(self, va: int, length: int) -> bytes:
        try:
            data = bytes(self.layer.read(va, length, pad=False))
        except Exception as exc:  # noqa: BLE001 - any fault is "not resident"
            raise Unreadable(hex(va)) from exc
        if len(data) != length:
            raise Unreadable(f"{hex(va)}: short read")
        return data

    def stale_pages(self, va: int, length: int) -> list[int]:
        """Pages in [va, va + length) whose PFN entry disagrees with their mapping."""
        first = va & ~(PAGE - 1)
        return [p for p in range(first, va + length, PAGE) if p in self.stale]

    def prefetch(self) -> None:
        """Reads this process's resident pages of the image into the page cache, once, before the scans.

        The scans visit the runs in virtual order, which is scattered order in the image, several times; a page
        that is not cached then costs one disk read at a time. Here one byte of every page is read in ascending
        file order from several threads, which is several times faster on a cold cache and leaves the pages
        cached and mapped for every scan. No byte that is read changes, and the pages are page cache, not private
        memory. Does nothing when the image is not a mapping."""
        physical = self.physical
        if self._prefetched or not isinstance(physical, mmap.mmap):
            return
        self._prefetched = True
        runs = np.array([(phys, length) for _va, length, phys in self.runs if phys < STORE_BASE], np.int64)
        if not len(runs):
            return
        first = runs[:, 0] >> 12
        count = ((runs[:, 0] + runs[:, 1] + PAGE - 1) >> 12) - first
        # Every page number of every run; a shared page can be mapped twice.
        pages = np.unique(np.repeat(first - np.cumsum(count) + count, count) + np.arange(int(count.sum())))
        touch_pages(physical, pages[pages < len(physical) >> 12] << 12)

    def try_read(self, va: int, length: int) -> bytes | None:
        try:
            return self.read(va, length)
        except Unreadable:
            return None

    def q(self, va: int) -> int:
        return int.from_bytes(self.read(va, 8), "little")

    def cstr(self, va: int, limit: int = 64) -> str:
        return self.read(va, limit).split(b"\0", 1)[0].decode("latin-1")

    def chunks(self, max_bytes: int = 16 << 20, lo: int = 0,
               hi: int | None = None) -> Iterator[tuple[int, bytes]]:
        """Virtually contiguous resident bytes in windows of at most ``max_bytes`` (runs are merged by va)."""
        # A piece lies inside one run, so when runs do not overlap it is sliced from that run directly and
        # copied once into its window; read() would look the run up again and copy the bytes twice more.
        hi = self.user_limit if hi is None else min(hi, self.user_limit)
        direct = self.physical is not None and self._disjoint
        pending_va: int | None = None
        parts: list[bytes] = []
        size = 0
        for start, length, phys in self.runs:
            if start + length <= lo or start >= hi:
                continue
            s, e = max(start, lo), min(start + length, hi)
            if pending_va is not None and pending_va + size != s:
                yield pending_va, b"".join(parts)
                pending_va, parts, size = None, [], 0
            cursor = s
            while cursor < e:
                if pending_va is None:
                    pending_va = cursor
                take = min(e - cursor, max_bytes - size)
                try:
                    if direct and 0x10000 <= cursor < self.user_limit:
                        parts.append(self._read_physical(phys + cursor - start, take))
                    else:
                        parts.append(self.read(cursor, take))
                except Unreadable:
                    break
                size += take
                cursor += take
                if size >= max_bytes:
                    yield pending_va, b"".join(parts)
                    pending_va, parts, size = None, [], 0
        if pending_va is not None and size:
            yield pending_va, b"".join(parts)

    def scan_words(self, test: Any, lo: int = 0, hi: int | None = None,
                   overlap: int = 0) -> Iterator[tuple[int, np.ndarray, int]]:
        """Yields (address of word, the window's uint64 array, index) for every index ``test(words)`` returns.

        With ``overlap``, a window that continues the previous one virtually begins with that window's last
        ``overlap`` words, so a structure of up to ``overlap + 1`` words straddling a chunk boundary is seen
        whole. Each address is yielded once."""
        tail = np.zeros(0, np.uint64)
        tail_end: int | None = None
        recent: set[int] = set()
        for va, data in self.chunks(lo=lo, hi=hi):
            words = np.frombuffer(data[: len(data) // 8 * 8], np.uint64)
            if len(words) == 0:
                tail, tail_end, recent = np.zeros(0, np.uint64), None, set()
                continue
            base = va
            if overlap and tail_end == va and len(tail):
                words = np.concatenate([tail, words])
                base = va - 8 * len(tail)
            else:
                recent = set()
            emitted: set[int] = set()
            keep_from = base + 8 * (len(words) - overlap)
            for pos in test(words):
                address = base + int(pos) * 8
                if address in recent:
                    continue
                if overlap and address >= keep_from:
                    emitted.add(address)
                yield address, words, int(pos)
            if overlap:
                tail = words[-overlap:].copy()
                tail_end = base + 8 * len(words) if len(data) % 8 == 0 else None
                recent = emitted
