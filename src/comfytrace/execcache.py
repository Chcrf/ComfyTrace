"""Stage 1 (continued): ComfyUI's execution cache, read from CPython objects.

ComfyUI keeps the outputs of the nodes of its latest prompt, so that an unchanged node is not run again. Each
cache is a dict ``{input signature: value}``. The signature is ComfyUI's own key (``CacheKeySetInputSignature``):
the node's class and inputs followed by those of every ancestor, made hashable, so every list is a frozenset of
``(index, item)`` pairs and every mapping a frozenset of ``(key, value)`` pairs:

    [[class_type, is_changed, (name, value) | (name, ("ANCESTOR", ancestor index, output socket)), ...],
     <the same for ancestor 0>, <ancestor 1>, ...]

The value is ``CacheEntry(ui, outputs)`` (a named tuple), or on older versions the outputs list alone, with the UI
data in a second cache. ``outputs`` holds one list per output socket.

A cached IMAGE output is a ``torch.Tensor`` whose TensorImpl is one of the IMAGE tensors ``torchmem`` already
finds. Its cache key then names, in ComfyUI's own record, the node that produced the picture and every setting
upstream of it (seed, steps, prompts, model, size). No file, no embedded text and no regeneration is involved.

A signature is compared with a recovered API prompt structurally (``matching_nodes``): the same classes, the same
input names and literal values, and links that lead to matching nodes on the same output sockets. ``is_changed``
is not compared: it is computed by the node at run time and is not part of a prompt.

Only what a live dict still holds is read. Entries of earlier prompts are deleted by ComfyUI's clean-up, and their
freed one-item lists lose their only pointer to the allocator's free-list link.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from comfytrace.carve import canonical_json
from comfytrace.cpython.base import CPythonLayout
from comfytrace.memrecover import job_timeline
from comfytrace.procmem import Unreadable
from comfytrace.pyheap import CorruptObject, CPythonHeap, hole
from comfytrace.torchmem import tensor_shape

log = logging.getLogger(__name__)

ANCESTOR = "ANCESTOR"
MAX_ENTRIES = 1 << 16
MAX_SET_SLOTS = 1 << 20
DECODE_BUDGET = 200_000
TENSOR_TYPES = ("Tensor", "Parameter")
BOOKKEEPING = "bookkeeping"
UNHASHABLE = "<unhashable>"
NAN = "<NaN>"


@dataclass
class CacheEntry:
    """One ``signature -> value`` entry of one cache dict."""

    process_id: int
    cache_dict: int
    key_address: int
    value_address: int
    value_kind: str
    signature: list[Any]
    graph: dict[str, dict[str, Any]]
    ui: Any = None
    outputs: list[list[dict[str, Any]]] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    holes: list[dict[str, str]] = field(default_factory=list)
    bookkeeping: dict[str, Any] = field(default_factory=dict)

    @property
    def class_type(self) -> str:
        return str(self.graph["0"]["class_type"])

    @property
    def signature_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.signature).encode("utf-8")).hexdigest()

    @property
    def ui_filenames(self) -> list[str]:
        return sorted({str(item["filename"]) for item in _walk(self.ui)
                       if isinstance(item, dict) and isinstance(item.get("filename"), str)})

    def tensors(self) -> list[tuple[int, int, dict[str, Any]]]:
        """(output socket, index in that socket's list, description) of every tensor held directly."""
        return [(slot, index, item) for slot, items in enumerate(self.outputs) for index, item in enumerate(items)
                if item.get("tensorimpl") is not None]

    def to_dict(self) -> dict[str, Any]:
        return {"process_id": self.process_id, "cache_dict": f"0x{self.cache_dict:X}",
                "key_address": f"0x{self.key_address:X}", "value_address": f"0x{self.value_address:X}",
                "value_kind": self.value_kind, "class_type": self.class_type,
                "node_ids_in_key_set": self.node_ids, "signature_sha256": self.signature_sha256,
                "bookkeeping": self.bookkeeping, "graph": self.graph, "ui": self.ui,
                "ui_filenames": self.ui_filenames,
                "outputs": [[{k: (f"0x{v:X}" if k in ("address", "tensorimpl") and isinstance(v, int) else v)
                              for k, v in item.items()} for item in items] for items in self.outputs],
                "holes": self.holes}


def _walk(value: Any) -> Any:
    stack = [value]
    while stack:
        item = stack.pop()
        yield item
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


# -- signatures -------------------------------------------------------------------------------------------------


def _layout(heap: CPythonHeap) -> CPythonLayout:
    if heap.layout is None:
        raise Unreadable(heap.unsupported or "no CPython runtime mapped")
    return heap.layout


def set_members(heap: CPythonHeap, obj: int) -> list[int]:
    """Member addresses of a set or frozenset (``CPythonLayout.set_members``)."""
    return _layout(heap).set_members(heap.mem, obj, MAX_SET_SLOTS)


def _pair(heap: CPythonHeap, layout: CPythonLayout, obj: int) -> tuple[int, int] | None:
    """The two item addresses of an exact 2-tuple, or None for anything else."""
    mem = heap.mem
    if layout.type_address(mem, obj) != heap.types["tuple"] or layout.sequence_size(mem, obj, "tuple") != 2:
        return None
    base = layout.sequence_base(mem, obj, "tuple")
    return layout.sequence_item(mem, base, 0), layout.sequence_item(mem, base, 1)


def decode_hashable(heap: CPythonHeap, obj: int, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Undoes ComfyUI's ``to_hashable``: a frozenset of ``(0..n-1, item)`` pairs is a list, one of ``(str, value)``
    pairs a dict, scalars are themselves. An object ComfyUI could not hash is ``{"<unhashable>": type name}``."""
    budget = budget if budget is not None else [DECODE_BUDGET]
    budget[0] -= 1
    if depth > 40 or budget[0] < 0:
        raise Unreadable("depth/budget")
    kind = heap.kind_of(obj)
    # A header-only object can also be Ellipsis or NotImplemented. A cache key requires a known type.
    if kind in ("NoneType", "<NoneType>"):
        return None
    if kind in ("str", "int", "bool"):
        return heap.decode(obj)
    if kind == "float":
        value = heap.decode(obj)
        return value if math.isfinite(value) else {NAN: "float"}
    if kind != "frozenset":
        if kind == "<Unhashable>":
            return {UNHASHABLE: kind[1:-1]}
        raise CorruptObject(f"unsupported type {kind} in a signature", obj)
    layout = _layout(heap)
    pairs: list[tuple[Any, Any]] = []
    for member in set_members(heap, obj):
        pair = _pair(heap, layout, member)
        if pair is None:
            raise CorruptObject("frozenset member is not a pair", member)
        pairs.append((decode_hashable(heap, pair[0], depth + 1, budget),
                      decode_hashable(heap, pair[1], depth + 1, budget)))
    names = [k for k, _ in pairs]
    if all(isinstance(k, int) and not isinstance(k, bool) for k in names) and sorted(names) == list(range(len(pairs))):
        return [v for _, v in sorted(pairs, key=lambda kv: kv[0])]
    if all(isinstance(k, str) for k in names) and len(set(names)) == len(names):
        return dict(sorted(pairs))
    raise CorruptObject("frozenset is neither a sequence nor a mapping", obj)


def _is_ancestor(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 3 and value[0] == ANCESTOR
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value[1:]) and value[1] >= 0)


def signature_graph(signature: Any) -> dict[str, dict[str, Any]]:
    """The API-shaped graph a signature describes. Node "0" is the cached node and node ``i + 1`` its ancestor
    ``i``; ids are positions in the signature, since a signature holds no node id. Each node has ``class_type``,
    ``inputs`` (a link is ``[position, socket]``), ``links`` (the names of the inputs that are links),
    ``is_changed`` and, where ComfyUI put the node's id into its signature, ``node_id``.
    Raises: ValueError when the value is not a signature."""
    if not isinstance(signature, list) or not signature:
        raise ValueError("not a signature list")
    graph: dict[str, dict[str, Any]] = {}
    for position, node in enumerate(signature):
        if not isinstance(node, list) or len(node) < 2 or not isinstance(node[0], str) or not node[0]:
            raise ValueError(f"signature item {position} is not a node")
        entry: dict[str, Any] = {"class_type": node[0], "inputs": {}, "links": [], "is_changed": node[1]}
        for item in node[2:]:
            if isinstance(item, str) and "node_id" not in entry and not entry["inputs"]:
                entry["node_id"] = item
            elif isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) \
                    and item[0] not in entry["inputs"]:
                name, value = item
                if _is_ancestor(value):
                    if value[1] + 1 >= len(signature):
                        raise ValueError(f"signature item {position} links past the last ancestor")
                    entry["inputs"][name] = [str(value[1] + 1), value[2]]
                    entry["links"].append(name)
                else:
                    entry["inputs"][name] = value
            else:
                raise ValueError(f"signature item {position} has an input that is not (name, value)")
        graph[str(position)] = entry
    return graph


def _is_link(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str | int)
            and not isinstance(value[0], bool) and isinstance(value[1], int) and not isinstance(value[1], bool))


def _has_unknown(value: Any) -> bool:
    """True when a recovered value holds a placeholder for something that could not be read as a value: an object
    ComfyUI could not hash, or a non-finite float."""
    return any(isinstance(item, dict) and (UNHASHABLE in item or NAN in item) for item in _walk(value))


def _same_literal(cached: Any, recovered: Any) -> bool:
    if _has_unknown(cached):
        return False  # a required value that is not known never strictly matches, not even an equal-looking marker
    if cached in ([], {}) and recovered in ([], {}):
        return True  # to_hashable turns both into an empty frozenset
    try:
        return canonical_json(cached) == canonical_json(recovered)
    except (TypeError, ValueError, RecursionError):
        return False


def matching_nodes(graph: dict[str, dict[str, Any]], prompt: Any) -> list[str]:
    """Ids of the nodes of an API prompt that have exactly the cached node's signature: the same class, the same
    input names, equal literal values, and links leading on the same sockets to nodes that match in turn, all the
    way up. Recorded runtime input fingerprints must also agree."""
    if not isinstance(prompt, dict):
        return []
    memo: dict[tuple[str, str], bool] = {}

    def same(position: str, node_id: str) -> bool:
        key = (position, node_id)
        if key in memo:
            return memo[key]
        memo[key] = False  # a cycle never matches
        cached, node = graph.get(position), prompt.get(node_id)
        if cached is None or not isinstance(node, dict):
            return False
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or node.get("class_type") != cached["class_type"] \
                or set(inputs) != set(cached["inputs"]) or cached.get("node_id", node_id) != node_id:
            return False
        if ("is_changed" in node and "is_changed" in cached
                and not _same_literal(cached["is_changed"], node["is_changed"])):
            return False
        for name, expected in cached["inputs"].items():
            actual = inputs[name]
            if name in cached["links"]:
                if not _is_link(actual) or actual[1] != expected[1] or not same(expected[0], str(actual[0])):
                    return False
            elif _is_link(actual) or not _same_literal(expected, actual):
                return False
        memo[key] = True
        return True

    return [str(node_id) for node_id in prompt if same("0", str(node_id))]


def consumers(prompt: dict[str, Any], node_id: str, socket: int | None = None) -> list[str]:
    """Ids of the nodes of ``prompt`` that take an input directly from ``node_id`` (from output ``socket`` only,
    when one is given: a node fed by another output of a multi-output node never received this tensor)."""
    return [str(other) for other, node in prompt.items() if isinstance(node, dict)
            and any(_is_link(v) and str(v[0]) == node_id and (socket is None or v[1] == socket)
                    for v in (node.get("inputs") or {}).values())]


# -- cache dicts ------------------------------------------------------------------------------------------------


def _describe(heap: CPythonHeap, obj: int, depth: int = 0) -> dict[str, Any]:
    """What one cached output object is: its type, and for a tensor (or a dict of tensors, as a LATENT is) the
    TensorImpl and shape."""
    mem = heap.mem
    try:
        kind = heap.kind_of(obj)
    except Unreadable:
        return {"type": "NoneType" if _safe_none(heap, obj) else "UNREADABLE", "address": obj}
    name = kind[1:-1] if kind.startswith("<") else kind
    out: dict[str, Any] = {"type": name, "address": obj}
    try:
        layout = _layout(heap)
        if name in TENSOR_TYPES:
            impl = mem.q(obj + layout.object_header_size)  # THPVariable: PyObject_HEAD, then the TensorImpl
            shape = tensor_shape(mem, impl)
            if shape is not None:
                out.update(tensorimpl=impl, **shape)
        elif kind == "dict" and depth == 0:
            members = {}
            for key, value in heap.dict_items(obj)[:16]:
                if layout.type_address(mem, key) == heap.types["str"]:
                    members[heap.decode_str(key)] = _describe(heap, value, depth + 1)
            out["members"] = {k: {a: (f"0x{b:X}" if a in ("address", "tensorimpl") else b) for a, b in v.items()}
                              for k, v in members.items()}
    except (Unreadable, UnicodeDecodeError, ValueError):
        out["unreadable"] = True
    return out


def _safe_none(heap: CPythonHeap, obj: int) -> bool:
    try:
        return heap.is_none(obj)
    except Unreadable:
        return False


def _sequence(heap: CPythonHeap, obj: int, limit: int = 256) -> list[int] | None:
    """Item addresses of a list or tuple (exact or subclass layout is the caller's concern), or None."""
    mem = heap.mem
    try:
        layout = _layout(heap)
        kind = heap.kind_of(obj)
        if kind not in ("list", "tuple"):
            return None
        n = layout.sequence_size(mem, obj, kind)
        if n > limit:
            return None
        base = layout.sequence_base(mem, obj, kind)
        return [layout.sequence_item(mem, base, i) for i in range(n)]
    except Unreadable:
        return None


def _value(heap: CPythonHeap, value: int, entry: CacheEntry) -> None:
    """Reads a cache value into ``entry``: ``CacheEntry(ui, outputs)``, an outputs list, or a UI dict."""
    mem, layout = heap.mem, _layout(heap)
    kind = heap.kind_of(value)
    ui = outputs = None
    # CacheEntry is a named tuple: tuple storage under its own heap type.
    if kind == "<CacheEntry>" and layout.sequence_size(mem, value, kind) == 2:
        base = layout.sequence_base(mem, value, kind)
        entry.value_kind, ui, outputs = ("CacheEntry", layout.sequence_item(mem, base, 0),
                                         layout.sequence_item(mem, base, 1))
    elif kind == "list":
        entry.value_kind, outputs = "outputs list", value
    elif kind == "dict":
        entry.value_kind, ui = "ui dict", value
    elif kind in ("int", "float"):
        # The LRU and RAM-pressure caches keep a generation number and a time.time() per key in dicts of their own.
        entry.value_kind = BOOKKEEPING
        number = heap.decode(value)
        if kind == "float" and 10**9 < number < 10**11:
            entry.bookkeeping["timestamp_utc"] = datetime.fromtimestamp(number, timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z")
        else:
            entry.bookkeeping["generation" if kind == "int" else "number"] = number
        return
    else:
        entry.value_kind = f"unsupported {kind}"
        return
    if ui is not None and not _safe_none(heap, ui):
        try:
            entry.ui = heap.decode(ui, holes=entry.holes, path=["ui"])
        except Unreadable as exc:
            entry.holes.append(hole(["ui"], exc, mem.stale))
    slots = _sequence(heap, outputs) if outputs is not None else None
    for slot in slots or []:
        items = _sequence(heap, slot)
        entry.outputs.append([_describe(heap, item) for item in items] if items is not None
                             else [_describe(heap, slot)])


def _general_entries(heap: CPythonHeap, keys: int) -> np.ndarray | None:
    """(hash, key, value) rows of a dict key table with general keys, or None for any other table."""
    return _layout(heap).general_entries(heap.mem, keys, MAX_ENTRIES)


def find_entries(heap: CPythonHeap) -> tuple[list[CacheEntry], dict[str, int]]:
    """Every entry of every dict whose keys are ComfyUI input signatures, and counts of what was looked at."""
    counts = {"cache_dicts": 0, "entries": 0, "undecodable_keys": 0}
    frozenset_t = heap.types.get("frozenset")
    if frozenset_t is None:
        return [], counts
    mem, layout = heap.mem, _layout(heap)
    found: list[CacheEntry] = []
    seen: set[int] = set()
    for obj, used, keys, values in layout.iter_dict_candidates(mem, heap.types["dict"]):
        if obj in seen or values:  # a split table holds instance attributes, never a cache
            continue
        try:
            rows = _general_entries(heap, keys)
            if rows is None:
                continue
            live = rows[(rows[:, 1] != 0) & (rows[:, 2] != 0)]
            if len(live) != used or not len(live) or layout.type_address(mem, int(live[0, 1])) != frozenset_t:
                continue
        except Unreadable:
            continue
        seen.add(obj)
        entries = []
        undecodable = 0
        for _hash, key, value in live:
            try:
                signature = decode_hashable(heap, int(key))
                graph = signature_graph(signature)
            except (Unreadable, ValueError, UnicodeDecodeError):
                undecodable += 1
                continue
            entry = CacheEntry(mem.pid, obj, int(key), int(value), "", signature, graph)
            try:
                _value(heap, int(value), entry)
            except (Unreadable, UnicodeDecodeError, ValueError) as exc:
                entry.value_kind = entry.value_kind or "UNREADABLE"
                entry.holes.append(hole(["value"], exc, mem.stale))
            entries.append(entry)
        if entries:  # other Python dicts are keyed by frozensets too; only a dict holding a signature counts
            counts["cache_dicts"] += 1
            counts["undecodable_keys"] += undecodable
            found += entries
    found = _merge_bookkeeping(found)
    counts["entries"] = len(found)
    _key_set_node_ids(heap, found)
    return found, counts


def _merge_bookkeeping(entries: list[CacheEntry]) -> list[CacheEntry]:
    """A generation number or timestamp kept under the same key object joins that key's cache entry."""
    by_key: dict[int, CacheEntry] = {e.key_address: e for e in entries if e.value_kind != BOOKKEEPING}
    out = [e for e in entries if e.value_kind != BOOKKEEPING]
    for entry in entries:
        if entry.value_kind != BOOKKEEPING:
            continue
        if entry.key_address in by_key:
            by_key[entry.key_address].bookkeeping.update(entry.bookkeeping)
        else:
            by_key[entry.key_address] = entry
            out.append(entry)
    return out


def _key_set_node_ids(heap: CPythonHeap, entries: list[CacheEntry]) -> None:
    """ComfyUI's key set maps each node id of the latest prompt to the very frozenset that keys the cache. A
    str-keyed dict entry whose value is that frozenset gives the node id."""
    if not entries:
        return
    mem, layout, str_t = heap.mem, _layout(heap), heap.types["str"]
    by_key: dict[int, list[CacheEntry]] = {}
    for entry in entries:
        by_key.setdefault(entry.key_address, []).append(entry)
    wanted = np.array(sorted(by_key), np.uint64)

    for name, value in layout.iter_value_key_candidates(mem, wanted):
        try:
            if layout.type_address(mem, name) != str_t or layout.string_length(mem, name) > 64:
                continue
            node_id = heap.decode_str(name)
        except (Unreadable, UnicodeDecodeError, ValueError):
            continue
        for entry in by_key[value]:
            if node_id not in entry.node_ids:
                entry.node_ids.append(node_id)


def extract(mem: Any) -> tuple[list[CacheEntry], dict[str, Any]]:
    """The execution-cache entries of one ComfyUI Python process and a summary for ``recovery_stages``."""
    heap = CPythonHeap(mem)
    if not heap.available:
        return [], {"status": "UNAVAILABLE: process not readable as CPython objects"}
    if "frozenset" not in heap.types:
        return [], {"status": "UNAVAILABLE: frozenset type object not resolved"}
    entries, counts = find_entries(heap)
    held = sum(1 for e in entries for _ in e.tensors())
    log.info("PID %d: execution cache: %d dicts, %d entries, %d tensors held", mem.pid, counts["cache_dicts"],
             counts["entries"], held)
    return entries, {"status": "FOUND" if entries else "NOT_FOUND", **counts, "tensors_held": held,
                     "node_classes": sorted({e.class_type for e in entries})}


def image_ties(entries: list[CacheEntry]) -> dict[tuple[int, int], dict[str, Any]]:
    """(process id, TensorImpl address) -> the cache entry that holds that tensor as an output, as the evidence
    an image record carries. A tensor held by several entries keeps the first and lists the others."""
    ties: dict[tuple[int, int], dict[str, Any]] = {}
    # A node that passes its input on (PreviewImage) holds the same tensor as the node that made it; the maker
    # has the shorter signature, since the other's signature contains it.
    for entry in sorted(entries, key=lambda e: len(e.signature)):
        for slot, index, item in entry.tensors():
            key = (entry.process_id, int(item["tensorimpl"]))
            tie = {"holder": "ComfyUI execution cache", "cache_dict": f"0x{entry.cache_dict:X}",
                   "key_address": f"0x{entry.key_address:X}", "value_kind": entry.value_kind,
                   "class_type": entry.class_type, "node_ids_in_key_set": entry.node_ids,
                   "output_socket": slot, "list_index": index, "signature_sha256": entry.signature_sha256,
                   "bookkeeping": entry.bookkeeping, "graph": entry.graph}
            if key in ties:
                ties[key].setdefault("also_held_by", []).append(
                    {k: tie[k] for k in ("cache_dict", "key_address", "class_type", "signature_sha256")})
            else:
                ties[key] = tie
    return ties


def job_matches(graph: dict[str, dict[str, Any]], items: Any, socket: int | None = None) -> list[dict[str, Any]]:
    """The queue and history items whose prompt holds a node with this signature: one row per job, with the
    matching node ids and the nodes that take their input directly from output ``socket`` of them.

    ``reused`` is true when the job's own history lists every matching node as ``execution_cached``: the job
    returned a tensor an earlier job computed, so it is not the tensor's producer."""
    rows: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.prompt is None or item.prompt_id in rows:
            continue
        nodes = matching_nodes(graph, item.prompt)
        if nodes:
            cached = set(job_timeline(item).get("cached_nodes") or [])
            rows[item.prompt_id] = {"prompt_id": item.prompt_id, "node_ids": nodes,
                                    "feeds": sorted({c for n in nodes for c in consumers(item.prompt, n, socket)}),
                                    "reused": set(nodes) <= cached}
    return list(rows.values())
