"""Stage 1: ComfyUI evidence from live and freed CPython 3.x objects in a ComfyUI process.

ComfyUI's server queues ``(number, prompt_id, prompt, extra_data, outputs_to_execute[, sensitive])`` tuples and
its PromptQueue keeps history dicts ``{"prompt": <that tuple>, "outputs": ..., "status": ...}``. Type objects
are located inside python3XY.dll by their tp_name; queue tuples are found by header shape and decoded
structurally (dict key tables, compact str of every kind, multi-digit int, float, list, tuple). No JSON text
and no file is read.

Linux: the type objects are searched in ``libpython3.N.so*`` or, when no libpython is mapped, in a statically
linked executable named ``python``, ``python3`` or ``python3.N``. The version comes from the module's exported
``Py_Version`` (3.11+), read through its own ELF dynamic symbol table, and for libpython also from its file name;
the two must agree, and an executable must show ``Py_Version``. Anything else (a version or build without a
registered layout, an unresolvable version) leaves the stage UNAVAILABLE with the reason (``cpython.runtime``).

Every struct offset comes from the ``CPythonLayout`` the registry holds for the runtime (``cpython.profiles``);
this module and ``execcache`` hold none. The layout is resolved before any type page is scanned.

Pixel buffers: CPython ``bytes`` objects that hold raw RGB/RGBA pixels (``Image.tobytes`` / numpy round trips).
A live one starts ``(refcnt, &PyBytes_Type, ob_size, ob_shash)``; a freed one keeps ``ob_size`` and
``ob_shash == -1`` after the heap overwrites the first two words with free-list links. Candidates are kept only
when the size is an image size and the bytes look like pixels.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comfytrace.cpython import registry, runtime
from comfytrace.cpython.base import MINUS_ONE, CorruptObject, CPythonLayout
from comfytrace.cpython.runtime import (
    ELF_PYTHON,
    LIBPYTHON,
    PYTHON_DLL,
    PYTHON_VERSION_MISMATCH,
    PYTHON_VERSION_UNVERIFIED,
    UNSUPPORTED_PYTHON_BUILD,
    UNSUPPORTED_PYTHON_VERSION,
)
from comfytrace.procmem import ProcessMemory, StaleMapping, Unreadable

log = logging.getLogger(__name__)

__all__ = ["CorruptObject", "CPythonHeap", "CPythonLayout", "ELF_PYTHON", "LIBPYTHON", "LINUX_PYTHON_VERSIONS",
           "MINUS_ONE", "PYTHON_DLL", "PYTHON_VERSION_MISMATCH", "PYTHON_VERSION_UNVERIFIED",
           "UNSUPPORTED_PYTHON_BUILD", "UNSUPPORTED_PYTHON_VERSION", "WINDOWS_PYTHON_VERSIONS"]

LINUX_PYTHON_VERSIONS = registry.supported_versions("linux")
WINDOWS_PYTHON_VERSIONS = registry.supported_versions("windows")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_MISSING = object()
TYPE_NAMES = ("type", "tuple", "dict", "str", "int", "float", "list", "bool", "NoneType", "bytes", "frozenset")
REQUIRED_TYPES = ("type", "tuple", "dict", "str", "int")
DECODE_BUDGET = 2_000_000
UNREADABLE = "UNREADABLE"
CORRUPT_OBJECT = "CORRUPT_OBJECT"
STALE_MAPPING = "STALE_MAPPING"


def json_pointer(path: list[Any]) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path)


def hole(path: list[Any], exc: Exception, stale: dict[int, str] | None = None) -> dict[str, str]:
    """A hole's reason. An invalid object on a page whose PFN entry disagrees with its mapping (``stale``) is
    STALE_MAPPING: its bytes come from another moment of the acquisition."""
    detail = str(exc)
    if isinstance(exc, StaleMapping):
        reason = STALE_MAPPING
    elif isinstance(exc, CorruptObject):
        label = (stale or {}).get((exc.address or 0) & ~0xFFF) if exc.address is not None else None
        reason = STALE_MAPPING if label else CORRUPT_OBJECT
        if label:
            detail = f"{detail} at 0x{exc.address:x}: {label}"
    else:
        reason = UNREADABLE
    return {"json_pointer": json_pointer(path), "reason": reason, "detail": detail[:120]}


@dataclass
class QueueItem:
    process_id: int
    prompt_id: str
    number: Any
    address: int
    refcnt: int
    length: int
    prompt: dict[str, Any] | None = None
    extra_data: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    history: dict[str, Any] | None = None
    prompt_holes: list[dict[str, str]] = field(default_factory=list)
    extra_holes: list[dict[str, str]] = field(default_factory=list)
    owner: str | None = None

    @property
    def workflow(self) -> dict[str, Any] | None:
        pnginfo = (self.extra_data or {}).get("extra_pnginfo")
        workflow = pnginfo.get("workflow") if isinstance(pnginfo, dict) else None
        return workflow if isinstance(workflow, dict) else None

    @property
    def workflow_holes(self) -> list[dict[str, str]]:
        prefix = "/extra_pnginfo/workflow"
        return [{**h, "json_pointer": h["json_pointer"][len(prefix):] or "/"} for h in self.extra_holes
                if h["json_pointer"] == prefix or h["json_pointer"].startswith(prefix + "/")]

    @property
    def liveness(self) -> str:
        """LIVE only when the tuple is held by a history record; a tuple found by shape alone is UNKNOWN."""
        return "LIVE" if self.owner is not None and self.refcnt > 0 else "UNKNOWN"


@dataclass
class PixelBuffer:
    process_id: int
    address: int
    width: int
    height: int
    channels: int
    data: bytes
    state: str
    row_score: float
    stale_rows: frozenset[int] = frozenset()


@dataclass
class PythonExtraction:
    process_id: int
    status: str
    reason: str | None = None
    python_dll: str | None = None
    type_objects: dict[str, str] = field(default_factory=dict)
    items: list[QueueItem] = field(default_factory=list)
    pixel_buffers: list[PixelBuffer] = field(default_factory=list)
    readable_dll_bytes: int = 0
    python_version: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "status": self.status,
            "reason": self.reason,
            "python_dll": self.python_dll,
            "python_version": self.python_version,
            "readable_dll_bytes": self.readable_dll_bytes,
            "type_objects": self.type_objects,
            "queue_items": len(self.items),
            "prompt_ids": sorted({i.prompt_id for i in self.items}),
            "pixel_buffers": len(self.pixel_buffers),
        }


class CPythonHeap:
    """Structural decoder for the CPython objects of one process, through the layout registered for its runtime
    (``cpython.registry``). Every byte offset comes from ``self.layout``."""

    def __init__(self, mem: ProcessMemory) -> None:
        self.mem = mem
        self.types: dict[str, int] = {}
        self.readable_dll_bytes = 0
        found = runtime.detect(mem)
        self.runtime = found
        self.dll = found.image
        self.version: tuple[int, int] | None = found.version
        self.build_flags = found.build_flags
        self.unsupported: str | None = found.unsupported
        # The layout is chosen before any type page is read: a refused runtime is never scanned.
        self.layout: CPythonLayout | None = None
        if self.dll is not None and self.unsupported is None and self.version is not None:
            self.layout = registry.resolve_layout(found.image_os, self.version, self.build_flags)
            if self.layout is None:  # supported_versions and the index disagree only if a profile is malformed
                self.unsupported = f"{UNSUPPORTED_PYTHON_VERSION}:{_dotted(self.version)}"
        self.split_values_offset = (self.layout.split_values_offset if self.layout is not None
                                    else 0 if self.version == (3, 12) else 8)
        if self.layout is not None:
            chunks = [c for lo, hi in self.dll.ranges() for c in self.mem.chunks(lo=lo, hi=hi)]
            self.readable_dll_bytes = sum(len(data) for _, data in chunks)
            self._find_types(chunks)
        self.names = {v: k for k, v in self.types.items()}

    @property
    def available(self) -> bool:
        return (self.unsupported is None and self.layout is not None
                and all(name in self.types for name in REQUIRED_TYPES))

    def _find_types(self, chunks: list[tuple[int, bytes]]) -> None:
        assert self.dll is not None and self.layout is not None
        layout = self.layout
        name_at: dict[int, str] = {}
        for va, data in chunks:
            for name in TYPE_NAMES:
                needle = b"\0" + name.encode() + b"\0"
                start = 0
                while (i := data.find(needle, start)) >= 0:
                    start = i + 1
                    name_at[va + i + 1] = name
        if not name_at:
            return
        wanted = np.array(sorted(name_at), np.uint64)
        # tp_name sits at a fixed offset of a static type object; its ob_type is PyType_Type, named "type".
        for va, data in chunks:
            words = np.frombuffer(data[: len(data) // 8 * 8], np.uint64)
            for slot in np.nonzero(np.isin(words, wanted))[0]:
                name = name_at[int(words[slot])]
                if name in self.types:
                    continue
                candidate = va + int(slot) * 8 - layout.tp_name_offset
                try:
                    if layout.type_name(self.mem, layout.type_address(self.mem, candidate)) == "type":
                        self.types[name] = candidate
                except Unreadable:
                    continue

    def _layout(self) -> CPythonLayout:
        if self.layout is None:
            raise Unreadable(self.unsupported or "no CPython runtime mapped")
        return self.layout

    def in_dll(self, va: int) -> bool:
        return self.dll is not None and self.dll.contains(va)

    def kind_of(self, obj: int) -> str:
        layout = self._layout()
        t = layout.type_address(self.mem, obj)
        if t in self.names:
            return self.names[t]
        return "<" + layout.type_name(self.mem, t) + ">"

    def is_none(self, obj: int) -> bool:
        """None is a header-only static singleton inside the runtime image. Its tp_name page may be paged out, so
        without a NoneType it is recognised by shape (``CPythonLayout.is_none``)."""
        return self._layout().is_none(self.mem, obj, self.types.get("NoneType"), self.dll)

    def decode_str(self, obj: int) -> str:
        return self._layout().decode_str(self.mem, obj)

    def decode_int(self, obj: int) -> int:
        return self._layout().decode_int(self.mem, obj)

    def dict_items(self, obj: int) -> list[tuple[int, int]]:
        return self._layout().dict_items(self.mem, obj)

    def decode(self, obj: int, depth: int = 0, budget: list[int] | None = None,
               path: list[Any] | None = None, holes: list[dict[str, str]] | None = None) -> Any:
        """Decodes a JSON-shaped object. With ``holes`` given, a child that cannot be decoded is recorded as a
        hole (JSON pointer and reason) instead of failing the whole graph: a dict entry is dropped, a list item
        keeps its position as None. Missing data is never invented."""
        budget = budget if budget is not None else [DECODE_BUDGET]
        path = path if path is not None else []
        budget[0] -= 1
        if depth > 60 or budget[0] < 0:
            raise Unreadable("depth/budget")
        try:
            kind = self.kind_of(obj)
        except Unreadable:
            if self.is_none(obj):
                return None
            raise
        if kind == "NoneType" or (kind not in self.types and self.is_none(obj)):
            return None
        if kind == "str":
            return self.decode_str(obj)
        if kind == "int":
            return self.decode_int(obj)
        if kind == "bool":
            return bool(self.decode_int(obj))
        layout = self._layout()
        if kind == "float":
            return layout.decode_float(self.mem, obj)
        if kind in ("list", "tuple"):
            n = layout.sequence_size(self.mem, obj, kind)
            if n > (1 << 20 if kind == "list" else 1 << 16) or (kind == "list" and n > layout.list_allocated(self.mem,
                                                                                                             obj)):
                raise CorruptObject(f"{kind} size", obj)
            base = layout.sequence_base(self.mem, obj, kind)
            out_list: list[Any] = []
            for i in range(n):
                out_list.append(self._child(lambda i=i: layout.sequence_item(self.mem, base, i), depth, budget,
                                            [*path, i], holes))
            return out_list
        if kind == "dict":
            out: dict[str, Any] = {}
            for index, (key, value) in enumerate(self.dict_items(obj)):
                try:
                    k = self.decode(key, depth + 1, budget, [*path, f"?key{index}"], None)
                except Unreadable as exc:
                    if holes is None:
                        raise
                    holes.append(hole([*path, f"?key{index}"], exc, self.mem.stale))
                    continue
                name = k if isinstance(k, str) else json.dumps(k)
                if name in out:
                    # A dict never holds a key twice; a repeated key means the entries are not one dict's.
                    raise CorruptObject(f"duplicate dict key {name!r}", obj)
                decoded = self._child(lambda value=value: value, depth, budget, [*path, name], holes, missing=_MISSING)
                if decoded is not _MISSING:
                    out[name] = decoded
            return out
        raise CorruptObject(f"unsupported type {kind}", obj)

    def _child(self, address: Any, depth: int, budget: list[int], path: list[Any],
               holes: list[dict[str, str]] | None, missing: Any = None) -> Any:
        try:
            return self.decode(address(), depth + 1, budget, path, holes)
        except Unreadable as exc:
            if holes is None:
                raise
            holes.append(hole(path, exc, self.mem.stale))
            return missing

    # -- ComfyUI queue and history -------------------------------------------------------------------------

    def queue_items(self) -> list[QueueItem]:
        tuple_t, str_t, dict_t = self.types["tuple"], self.types["str"], self.types["dict"]
        layout, mem = self._layout(), self.mem
        candidates = list(layout.iter_tuple_candidates(mem, tuple_t, (5, 6)))
        items: list[QueueItem] = []
        for obj in candidates:
            try:
                n = layout.sequence_size(mem, obj, "tuple")
                base = layout.sequence_base(mem, obj, "tuple")
                elems = [layout.sequence_item(mem, base, i) for i in range(n)]
                if layout.type_address(mem, elems[1]) != str_t or layout.string_length(mem, elems[1]) != 36:
                    continue
                if layout.type_address(mem, elems[2]) != dict_t or layout.type_address(mem, elems[3]) != dict_t:
                    continue
                prompt_id = self.decode_str(elems[1])
                if not UUID.fullmatch(prompt_id):
                    continue
                try:
                    number = self.decode(elems[0])
                except Unreadable:
                    number = None
                item = QueueItem(mem.pid, prompt_id, number, obj, layout.refcount(mem, obj), n)
                try:
                    prompt = self.decode(elems[2], holes=item.prompt_holes)
                    item.prompt = prompt if _is_prompt(prompt) else None
                    if item.prompt is None:
                        item.errors.append("prompt is not an API graph")
                except Unreadable as exc:
                    item.errors.append(f"prompt: {exc}")
                try:
                    item.extra_data = self.decode(elems[3], holes=item.extra_holes)
                except Unreadable as exc:
                    item.errors.append(f"extra_data: {exc}")
                items.append(item)
            except (Unreadable, UnicodeDecodeError, struct.error, ValueError, IndexError):
                continue
        self._attach_history(items)
        return items

    def _attach_history(self, items: list[QueueItem]) -> None:
        """History dicts whose "prompt" value is one of the queue tuples carry outputs and status."""
        if not items or "dict" not in self.types:
            return
        dict_t, str_t = self.types["dict"], self.types["str"]
        by_addr = {i.address: i for i in items}
        wanted = np.array(sorted(by_addr), np.uint64)
        referrers: set[int] = set()

        def refers(w: np.ndarray) -> np.ndarray:
            # Same positions as isin over the whole window, looked up only where the word lies in the range.
            pos = np.nonzero((w >= wanted[0]) & (w <= wanted[-1]))[0]
            return pos[np.isin(w[pos], wanted)]

        for slot, _w, _p in self.mem.scan_words(refers):
            referrers.add(slot)
        if not referrers:
            return
        slots = np.array(sorted(referrers), np.uint64)
        layout = self._layout()
        for obj, _used, keys, _values in layout.iter_dict_candidates(self.mem, dict_t):
            i = int(np.searchsorted(slots, keys))
            if i == len(slots) or int(slots[i]) - keys > 1 << 16:
                continue
            try:
                fields: dict[str, int] = {}
                for key, value in self.dict_items(obj):
                    if layout.type_address(self.mem, key) == str_t:
                        fields[self.decode_str(key)] = value
                item = by_addr.get(fields.get("prompt", -1))
                if item is None or not {"outputs", "status"} <= fields.keys():
                    continue
                record: dict[str, Any] = {"record_address": f"0x{obj:X}"}
                for name in ("outputs", "status", "meta"):
                    if name in fields:
                        try:
                            record[name] = self.decode(fields[name])
                        except Unreadable as exc:
                            record[f"{name}_error"] = str(exc)
                item.history = record
                item.owner = f"history record 0x{obj:X}"
            except (Unreadable, UnicodeDecodeError, struct.error, ValueError):
                continue

    # -- raw pixel buffers -----------------------------------------------------------------------------------

    def pixel_buffers(self, sizes: dict[int, list[tuple[int, int, int]]]) -> list[PixelBuffer]:
        found: list[PixelBuffer] = []
        for _obj, size, data_va, state in self._layout().iter_bytes_candidates(self.mem, sizes,
                                                                                self.types.get("bytes")):
            data = self.mem.try_read(data_va, size)
            if data is None:
                continue
            shape = best_shape(data, sizes[size])
            if shape is None:
                continue
            (width, height, channels), score = shape
            stride = width * channels
            rows = frozenset(r for page in self.mem.stale_pages(data_va, size)
                             for r in range(max(0, page - data_va) // stride,
                                            (min(size, page + 4096 - data_va) - 1) // stride + 1))
            found.append(PixelBuffer(self.mem.pid, data_va, width, height, channels, data, state, score, rows))
        return found


def _is_prompt(obj: Any) -> bool:
    return (isinstance(obj, dict) and bool(obj)
            and all(isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs"), dict)
                    for v in obj.values()))


def image_sizes(min_side: int = 256, max_side: int = 2048, step: int = 64) -> dict[int, list[tuple[int, int, int]]]:
    """Byte length -> candidate (width, height, channels) for ComfyUI-like 8-bit RGB/RGBA images."""
    sizes: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for w in range(min_side, max_side + 1, step):
        for h in range(min_side, max_side + 1, step):
            for c in (3, 4):
                sizes[w * h * c].append((w, h, c))
    return dict(sizes)


def best_shape(data: bytes, shapes: list[tuple[int, int, int]], max_zero_fraction: float = 0.2,
               max_row_score: float = 30.0) -> tuple[tuple[int, int, int], float] | None:
    """Chooses the shape whose rows are most continuous. Rejects mostly-zero or row-incoherent data."""
    a = np.frombuffer(data, np.uint8)
    if (a == 0).mean() > max_zero_fraction or a.astype(np.float32).std() < 8:
        return None
    best: tuple[tuple[int, int, int], float] | None = None
    sample = a[: min(len(a), 1 << 20)].astype(np.int16)
    opaque = bool((a[3::4] == 255).mean() > 0.98)
    for w, h, c in shapes:
        stride = w * c
        if len(sample) <= 2 * stride or (c == 4 and not opaque):
            continue
        score = float(np.abs(sample[stride:] - sample[:-stride]).mean())
        if best is None or score < best[1]:
            best = ((w, h, c), score)
    if best is None or best[1] > max_row_score:
        return None
    return best


_exported_version = runtime.exported_version
_dotted = runtime.dotted


def linux_python(mem: ProcessMemory) -> tuple[Any, tuple[int, int] | None, str | None]:
    """(module holding the CPython runtime, its version, why it is not decoded or None) for a Linux process.
    A mapped libpython is preferred; otherwise a python executable, which must export ``Py_Version``."""
    found = runtime.linux_runtime(mem)
    return found.image, found.version, found.unsupported


def extract(mem: ProcessMemory, scan_pixels: bool = True) -> PythonExtraction:
    """Detects whether the process can be read as CPython objects and, if so, extracts ComfyUI queue/history
    items and raw pixel buffers."""
    heap = CPythonHeap(mem)
    result = PythonExtraction(mem.pid, "UNAVAILABLE",
                              python_dll=heap.dll.path if heap.dll is not None else None,
                              type_objects={k: f"0x{v:X}" for k, v in heap.types.items()},
                              readable_dll_bytes=heap.readable_dll_bytes,
                              python_version=_dotted(heap.version) if heap.version else None)
    if heap.dll is None:
        result.reason = "NO_PYTHON_DLL_MAPPED"
        return result
    if heap.unsupported is not None:
        result.reason = heap.unsupported
        return result
    if not heap.readable_dll_bytes:
        result.reason = "PYTHON_DLL_PAGES_UNAVAILABLE"
        return result
    if not heap.available:
        missing = [n for n in REQUIRED_TYPES if n not in heap.types]
        result.reason = "TYPE_OBJECTS_UNRESOLVED:" + ",".join(missing)
        return result
    result.status = "AVAILABLE"
    result.items = heap.queue_items()
    if scan_pixels:
        result.pixel_buffers = heap.pixel_buffers(image_sizes())
    log.info("PID %d: CPython objects readable (%s); %d queue items, %d pixel buffers", mem.pid,
             heap.dll.name if heap.dll else "?", len(result.items), len(result.pixel_buffers))
    return result
