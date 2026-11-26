"""Stage 1 (continued): ComfyUI IMAGE tensors held by PyTorch on the CPU.

A c10 ``DataPtr`` for the default CPU allocator is laid out ``(data, deleter, ctx == data, device)`` followed by
``size_bytes`` inside a StorageImpl that begins with an intrusive_ptr header ``(vptr, refcount|weakcount)``.
The deleter is a function inside c10.dll; it is found by structure (the most common c10.dll word in that
position with ``ctx == data`` and a CPU device), never from a fixed offset.

A TensorImpl (measured MSVC x64 torch 2.x layout, the same as ComfyTrace's earlier evidence-bundle profile):
+0x00 vptr, +0x08 refcounts, +0x10 StorageImpl*, +0x40 ndim, +0x48 sizes[5], +0x70 strides[5],
+0x98 storage_offset, +0xa0 numel, +0xa8 dtype (6 = float32). ComfyUI IMAGE tensors are float32
[B, H, W, 3] in [0, 1]; they are rendered as ComfyUI saves them: ``clip(255 * x).astype(uint8)``.

Linux: c10 is ``libc10.so`` (the VMAs of that ELF module, see procmem). The DataPtr order above also holds for
the Itanium C++ ABI with libstdc++ (``std::unique_ptr<void, DeleterFnPtr>`` stores the deleter before the
pointer), and the TensorImpl members up to ``dtype`` are plain fields after a single vptr, so the same offsets are
read; a build whose layout differs (libc++, another torch release) fails the structural tests and yields nothing.
Linux adds two checks the Windows path does not make: the StorageImpl vptr must lie inside libc10 and the
TensorImpl vptr inside a ``libc10``/``libtorch*`` module, so a pointer into any other mapped library or file
never vouches for a tensor.
"""

from __future__ import annotations

import collections
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import as_strided

from comfytrace.procmem import PAGE, MappedImage, ProcessMemory, Unreadable

log = logging.getLogger(__name__)

CPU_DEVICE = 0xFF00  # c10::Device{DeviceType::CPU (0), index -1} in the low 16 bits
FLOAT32 = 6
NDIM_OFFSET, SIZES_OFFSET, NUMEL_OFFSET, DTYPE_OFFSET = 0x40, 0x48, 0xA0, 0xA8
DATAPTR_WORDS = 7  # StorageImpl vptr, refcounts, data, deleter, ctx, device, size_bytes
TENSORIMPL_WORDS = 22  # TensorImpl words read from its vptr through dtype
LIBC10 = re.compile(r"libc10\.so(\.\d+)*")
LIBTORCH = re.compile(r"lib(c10|torch[a-z_]*)\.so(\.\d+)*")


def _linux(mem: ProcessMemory) -> bool:
    return getattr(mem, "image_os", "windows") == "linux"


def c10_image(mem: ProcessMemory) -> MappedImage | None:
    if _linux(mem):
        return next((m for m in mem.images if m.kind == "elf" and LIBC10.fullmatch(m.name)), None)
    return mem.image(lambda name: name == "c10.dll")


def _in_ranges(w: np.ndarray, ranges: tuple[tuple[int, int], ...]) -> np.ndarray:
    inside = np.zeros(len(w), bool)
    for lo, hi in ranges:
        inside |= (w >= np.uint64(lo)) & (w < np.uint64(hi))
    return inside


@dataclass
class ImageTensor:
    process_id: int
    tensorimpl: int
    storage: int
    data_address: int
    sizes: list[int]
    strides: list[int]
    batch_index: int
    rgb: np.ndarray
    live: bool
    stale_rows: frozenset[int] = frozenset()
    byte_range: tuple[int, int] = (0, 0)  # (virtual address, length) of this batch item's floats


def tensor_shape(mem: ProcessMemory, impl: int) -> dict[str, Any] | None:
    """``sizes`` and ``dtype`` of the TensorImpl at ``impl`` (up to five dimensions, stored inline), or None when
    the words there are not a consistent TensorImpl."""
    try:
        ndim = mem.q(impl + NDIM_OFFSET)
        if ndim > 5:
            return None
        sizes = [mem.q(impl + SIZES_OFFSET + 8 * i) for i in range(ndim)]
        numel, dtype = mem.q(impl + NUMEL_OFFSET), mem.q(impl + DTYPE_OFFSET) & 0xFF
    except Unreadable:
        return None
    if any(size >= 1 << 32 for size in sizes) or math.prod(sizes) != numel:
        return None
    return {"sizes": sizes, "dtype": dtype}


def _dataptr_candidates(mem: ProcessMemory) -> list[tuple[int, tuple[int, ...]]]:
    """One pass for both the deleter vote and the storages: (address, the seven words from the StorageImpl vptr to
    size_bytes) of every c10.dll word placed like a CPU DataPtr's deleter (``ctx == data``, CPU device)."""
    c10 = c10_image(mem)
    if c10 is None:
        return []
    ranges = c10.ranges()

    def test(w: np.ndarray) -> np.ndarray:
        pos = np.nonzero(_in_ranges(w, ranges))[0]
        pos = pos[(pos >= 3) & (pos + 3 < len(w))]
        keep = (w[pos - 1] == w[pos + 1]) & (w[pos - 1] != 0) & ((w[pos + 2] & np.uint64(0xFFFF)) == CPU_DEVICE)
        return pos[keep]

    # Seven words around the deleter: carry six across each chunk boundary so a straddling DataPtr is seen whole.
    return [(addr, tuple(int(x) for x in w[pos - 3 : pos + 4]))
            for addr, w, pos in mem.scan_words(test, overlap=DATAPTR_WORDS - 1)]


def _storages(candidates: list[tuple[int, tuple[int, ...]]], deleter: int) -> dict[int, tuple[int, int, int, int]]:
    """StorageImpl address -> (vptr, refcount, data, nbytes)."""
    return {addr - 24: (m[0], m[1] & 0xFFFFFFFF, m[2], m[6]) for addr, m in candidates if m[3] == deleter}


def find_deleter(mem: ProcessMemory, candidates: list[tuple[int, tuple[int, ...]]]) -> int | None:
    counts: collections.Counter[int] = collections.Counter()
    if _linux(mem):
        c10 = c10_image(mem)
        vouching = [c10] if c10 is not None else []
    else:
        vouching = mem.images
    for _addr, m in candidates:
        # The StorageImpl header before the DataPtr is (vptr, refcount|weakcount): an aligned vtable pointer.
        if m[0] & 7 == 0 and any(image.contains(m[0]) for image in vouching):
            counts[m[3]] += 1
    return counts.most_common(1)[0][0] if counts else None


def image_tensors(mem: ProcessMemory) -> list[ImageTensor]:
    candidates = _dataptr_candidates(mem)
    deleter = find_deleter(mem, candidates)
    if deleter is None:
        return []
    storages = _storages(candidates, deleter)
    linux = _linux(mem)
    if linux:
        c10 = c10_image(mem)
        storages = {k: v for k, v in storages.items() if c10 is not None and c10.contains(v[0])}
    live_vptr = collections.Counter(v for v, ref, _, _ in storages.values() if ref).most_common(1)
    if not live_vptr:
        return []
    live = {k: v for k, v in storages.items() if v[0] == live_vptr[0][0] and v[1] > 0}
    keys = np.array(sorted(live), np.uint64)
    impls = []

    def holds_storage(w: np.ndarray) -> np.ndarray:
        # Same positions as isin over the whole window, looked up only where the word lies in the keys' range.
        pos = np.nonzero((w >= keys[0]) & (w <= keys[-1]))[0]
        return pos[np.isin(w[pos], keys)]

    for addr, w, pos in mem.scan_words(holds_storage, overlap=TENSORIMPL_WORDS - 1):
        if pos >= 2 and pos + 19 < len(w):
            impls.append((addr - 16, w[pos - 2 : pos + 20].copy()))
    if not impls:
        return []
    if linux:
        # Only a vtable inside libc10/libtorch can be a TensorImpl's; a word that merely equals a storage address
        # elsewhere (another object, a pointer into some other module) is not counted.
        torch = [m for m in mem.images if m.kind == "elf" and LIBTORCH.fullmatch(m.name)]
        impls = [(a, m) for a, m in impls if any(t.contains(int(m[0])) for t in torch)]
        if not impls:
            log.info("PID %s: no TensorImpl vtable inside libc10/libtorch; no tensors read", mem.pid)
            return []
    impl_votes = collections.Counter(int(m[0]) for _, m in impls if int(m[1]) & 0xFFFFFFFF).most_common(1)
    if not impl_votes:
        return []
    impl_vptr = impl_votes[0][0]
    out: list[ImageTensor] = []
    for addr, m in impls:
        if int(m[0]) != impl_vptr or not int(m[1]) & 0xFFFFFFFF:
            continue
        ndim = int(m[8])
        if ndim != 4:
            continue
        sizes, strides = [int(x) for x in m[9:13]], [int(x) for x in m[14:18]]
        offset, numel, dtype = int(m[19]), int(m[20]), int(m[21]) & 0xFF
        if dtype != FLOAT32 or sizes[3] != 3 or int(np.prod(sizes)) != numel or not 64 <= sizes[1] <= 8192:
            continue
        _vptr, _ref, data_va, nbytes = live[int(m[2])]
        extent = offset + sum((s - 1) * st for s, st in zip(sizes, strides, strict=True)) + 1
        if extent * 4 > nbytes:
            continue
        try:
            raw = mem.read(data_va, nbytes)
        except Unreadable:
            log.info("IMAGE tensor at 0x%x: storage not fully resident", addr)
            continue
        t = as_strided(np.frombuffer(raw, np.float32)[offset:], shape=sizes, strides=[4 * s for s in strides])
        if not (np.isfinite(t).all() and float(np.min(t)) >= 0.0 and float(np.max(t)) <= 1.0):
            continue
        # Elements on a page whose PFN entry disagrees with its mapping, viewed with the tensor's own strides.
        flagged = np.zeros(nbytes // 4, bool)
        for page in mem.stale_pages(data_va, nbytes):
            flagged[max(0, (page - data_va) // 4) : max(0, (page + PAGE - data_va + 3) // 4)] = True
        marks = as_strided(flagged[offset:], shape=sizes, strides=list(strides))
        for b in range(sizes[0]):
            rgb = np.clip(255.0 * t[b], 0, 255).astype(np.uint8)
            stale = frozenset(int(r) for r in np.nonzero(marks[b].any(axis=(1, 2)))[0])
            first = offset + b * strides[0]
            span = sum((s - 1) * st for s, st in zip(sizes[1:], strides[1:], strict=True)) + 1
            out.append(ImageTensor(mem.pid, addr, int(m[2]), data_va, sizes, strides, b, rgb, True, stale,
                                   (data_va + 4 * first, 4 * span)))
    return out
