"""Pure-memory recovery stages on synthetic memory: page tables, CPython objects, pixel buffers, bitmaps."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import struct

import numpy as np
import pytest

from comfytrace import bitmaps, evidence, memrecover, pyheap
from comfytrace.carve import IdAllocator, WorkflowCarver
from comfytrace.models import CarveType, Completeness
from comfytrace.pagetable import PageTables
from comfytrace.procmem import MappedImage, ProcessMemory, Unreadable
from tests.synthetic import api_prompt

PAGE = 4096
VALID, TRANSITION, PROTOTYPE = 1, 1 << 11, 1 << 10


class Phys:
    """A growable fake physical image with page allocation."""

    def __init__(self, pages: int = 64) -> None:
        self.data = bytearray(pages * PAGE)
        self.next = 1

    def alloc(self) -> int:
        base = self.next * PAGE
        self.next += 1
        return base

    def put(self, at: int, data: bytes) -> None:
        self.data[at : at + len(data)] = data

    def q(self, at: int, value: int) -> None:
        self.put(at, struct.pack("<Q", value))


def _map(phys: Phys, dtb: int, va: int, leaf: int, tables: dict[tuple[int, int], int]) -> None:
    base = dtb
    for shift in (39, 30, 21):
        index = (va >> shift) & 0x1FF
        key = (base, index)
        if key not in tables:
            tables[key] = phys.alloc()
            phys.q(base + 8 * index, tables[key] | VALID)
        base = tables[key]
    phys.q(base + 8 * ((va >> 12) & 0x1FF), leaf)


def test_page_tables_resolve_valid_transition_and_prototype_ptes():
    phys = Phys(128)
    kdtb, udtb = phys.alloc(), phys.alloc()
    ktables: dict[tuple[int, int], int] = {}
    utables: dict[tuple[int, int], int] = {}
    valid, standby, shared, paged = phys.alloc(), phys.alloc(), phys.alloc(), phys.alloc()
    proto_page = phys.alloc()
    proto_va = 0xFFFF_E585_0000_0000
    _map(phys, kdtb, proto_va, proto_page | VALID, ktables)
    phys.q(proto_page + 8, shared | TRANSITION)  # the shared page itself is on the standby list
    _map(phys, udtb, 0x10000, valid | VALID, utables)
    _map(phys, udtb, 0x11000, standby | TRANSITION, utables)
    _map(phys, udtb, 0x12000, (((proto_va + 8) & 0xFFFF_FFFF_FFFF) << 16) | PROTOTYPE, utables)
    _map(phys, udtb, 0x13000, (0x1234 << 32) | 0x80, utables)  # pagefile-format PTE: not resident
    kernel = PageTables(phys.data, kdtb)
    user = PageTables(phys.data, udtb, kernel=kernel)
    assert user.translate(0x10010) == valid + 0x10
    assert user.translate(0x11000) == standby
    assert user.translate(0x12008) == shared + 8
    assert user.translate(0x13000) is None
    assert list(user.resident_pages(0x10000, 0x14000)) == [(0x10000, valid), (0x11000, standby), (0x12000, shared)]
    assert PageTables(phys.data, udtb).translate(0x12000) is None  # no kernel support, no prototype resolution
    del paged


def _memory(blobs: dict[int, bytes], images: list[MappedImage] | None = None) -> ProcessMemory:
    """A process whose virtual pages map one-to-one onto a fake physical buffer."""
    phys = bytearray()
    runs = []
    for va, blob in sorted(blobs.items()):
        size = (len(blob) + PAGE - 1) // PAGE * PAGE
        runs.append((va, size, len(phys)))
        phys += blob.ljust(size, b"\0")
    return ProcessMemory(4242, runs, bytes(phys), images or [])


def test_process_memory_reads_never_pad():
    mem = _memory({0x10000: b"A" * PAGE, 0x12000: b"B" * PAGE})
    assert mem.read(0x10FFE, 2) == b"AA"
    with pytest.raises(Unreadable):
        mem.read(0x10FFE, 4)  # 0x11000 is not resident
    assert [va for va, _ in mem.chunks()] == [0x10000, 0x12000]


class Heap:
    """Writes CPython 3.13 x64 objects into a byte arena at a fixed virtual base."""

    def __init__(self, base: int, size: int = 1 << 20) -> None:
        self.base = base
        self.buf = bytearray(size)
        self.top = 0x100

    def alloc(self, size: int) -> int:
        at = self.top
        self.top += (size + 15) // 16 * 16
        return self.base + at

    def q(self, va: int, value: int) -> None:
        struct.pack_into("<Q", self.buf, va - self.base, value)

    def raw(self, va: int, data: bytes) -> None:
        self.buf[va - self.base : va - self.base + len(data)] = data

    def q_at(self, va: int) -> int:
        return int(struct.unpack_from("<Q", self.buf, va - self.base)[0])


CORRUPT = object()
REPEATED_KEY = object()  # a dict whose second entry reuses the first entry's key object
OVERFULL_LIST = object()  # a list claiming more items than its vector holds


class FakePython:
    """A python313.dll image with static type objects plus a heap of objects built from Python values."""

    DLL = 0x7FF0_0000_0000

    def __init__(self) -> None:
        self.dll = Heap(self.DLL, 1 << 16)
        self.heap = Heap(0x2000_0000_0000, 1 << 21)
        self.types: dict[str, int] = {}
        for name in ("type", "tuple", "dict", "str", "int", "float", "list", "bool", "bytes", "NoneType"):
            name_va = self.dll.alloc(16)
            self.dll.raw(name_va, name.encode() + b"\0")
            self.dll.raw(name_va - 1, b"\0")
            self.types[name] = self.dll.alloc(64)
            self.dll.q(self.types[name] + 24, name_va)
        for t in self.types.values():
            self.dll.q(t + 8, self.types["type"])
        self.none = self.dll.alloc(16)
        self.dll.q(self.none + 8, self.types["NoneType"])

    def obj(self, value: object) -> int:
        h = self.heap
        if value is None:
            return self.none
        if value is CORRUPT:
            # A dict whose key table pointer lands on unrelated bytes, as T2 job 0's nodes[1].inputs[0].
            junk = h.alloc(64)
            h.raw(junk, bytes(range(7, 71)))
            va = h.alloc(48)
            h.q(va + 8, self.types["dict"])
            h.q(va + 16, 2)
            h.q(va + 32, junk)
            return va
        if value is REPEATED_KEY:
            va = self.obj({"id": 1, "type": "KSampler"})
            keys = h.q_at(va + 32)
            entries = keys + 32 + (1 << 3)
            h.q(entries + 16, h.q_at(entries))
            return va
        if value is OVERFULL_LIST:
            va = self.obj([1, 2])
            h.q(va + 32, 1)
            return va
        if isinstance(value, str):
            data = value.encode("ascii")
            va = h.alloc(40 + len(data) + 1)
            h.q(va + 8, self.types["str"])
            h.q(va + 16, len(data))
            struct.pack_into("<I", h.buf, va + 32 - h.base, (1 << 2) | (1 << 5) | (1 << 6))
            h.raw(va + 40, data)
            return va
        if isinstance(value, int):
            digits = []
            v = abs(value)
            while v:
                digits.append(v & ((1 << 30) - 1))
                v >>= 30
            va = h.alloc(24 + 4 * len(digits))
            h.q(va + 8, self.types["int"])
            sign = 1 if value == 0 else (2 if value < 0 else 0)
            h.q(va + 16, (len(digits) << 3) | sign)
            for i, d in enumerate(digits):
                struct.pack_into("<I", h.buf, va + 24 + 4 * i - h.base, d)
            return va
        if isinstance(value, tuple):
            items = [self.obj(v) for v in value]
            va = h.alloc(24 + 8 * len(items))
            h.q(va, 1)
            h.q(va + 8, self.types["tuple"])
            h.q(va + 16, len(items))
            for i, item in enumerate(items):
                h.q(va + 24 + 8 * i, item)
            return va
        if isinstance(value, float):
            va = h.alloc(24)
            h.q(va + 8, self.types["float"])
            struct.pack_into("<d", h.buf, va + 16 - h.base, value)
            return va
        if isinstance(value, list):
            items =[self.obj(v) for v in value]
            vector = h.alloc(8 * max(1, len(items)))
            for i, item in enumerate(items):
                h.q(vector + 8 * i, item)
            va = h.alloc(40)
            h.q(va + 8, self.types["list"])
            h.q(va + 16, len(items))
            h.q(va + 24, vector)
            h.q(va + 32, max(1, len(items)))
            return va
        if isinstance(value, dict):
            pairs =[(self.obj(k), self.obj(v)) for k, v in value.items()]
            log2_size = 3
            while (1 << log2_size) * 2 // 3 < len(pairs):
                log2_size += 1
            log2_index = log2_size  # one index byte per slot below 256 slots
            if not pairs:
                log2_size, log2_index = 0, 3  # CPython's shared empty key table
            keys = h.alloc(32 + (1 << log2_index) + 16 * max(1, len(pairs)))
            struct.pack_into("<BBB", h.buf, keys + 8 - h.base, log2_size, log2_index, 1)
            h.q(keys + 24, len(pairs))
            entries = keys + 32 + (1 << log2_index)
            for i, (k, v) in enumerate(pairs):
                h.q(entries + 16 * i, k)
                h.q(entries + 16 * i + 8, v)
            va = h.alloc(48)
            h.q(va + 8, self.types["dict"])
            h.q(va + 16, len(pairs))
            h.q(va + 32, keys)
            return va
        raise TypeError(value)

    def memory(self, extra: dict[int, bytes] | None = None) -> ProcessMemory:
        blobs = {self.dll.base: bytes(self.dll.buf), self.heap.base: bytes(self.heap.buf), **(extra or {})}
        image = MappedImage("python313.dll", "C:/x/python313.dll", self.dll.base, self.dll.base + len(self.dll.buf))
        return _memory(blobs, [image])


PROMPT = {"3": {"class_type": "KSampler", "inputs": {"seed": 558648719999037, "steps": 20, "cfg": 8,
                                                     "sampler_name": "euler", "scheduler": "normal",
                                                     "positive": ["6", 0], "negative": ["7", 0],
                                                     "model": ["4", 0], "latent_image": ["5", 0]}},
          "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "cyberpunk street", "clip": ["4", 1]}}}
WORKFLOW = {"last_node_id": 9, "nodes": [{"id": 3, "type": "KSampler"}], "links": None, "version": 0.4}
PROMPT_ID = "74c67026-6e2a-4011-8e1a-d2c518fb0137"


def test_queue_item_is_decoded_from_cpython_objects():
    py = FakePython()
    py.obj((0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}}, ("9",)))
    result = pyheap.extract(py.memory(), scan_pixels=False)
    assert result.status == "AVAILABLE"
    (item,) = result.items
    assert item.prompt_id == PROMPT_ID and item.number == 0
    assert item.prompt == PROMPT
    assert item.workflow == WORKFLOW  # links: None decodes to None


def test_python_stage_reports_why_it_is_unavailable():
    mem = _memory({0x10000: b"\0" * PAGE})
    result = pyheap.extract(mem)
    assert result.status == "UNAVAILABLE" and result.reason == "NO_PYTHON_DLL_MAPPED"


def test_python_stage_distinguishes_a_mapped_dll_with_no_readable_pages():
    mem = _memory({0x10000: b"\0" * PAGE})
    mem.images.append(MappedImage("python312.dll", "python312.dll", 0x20000, 0x23000))

    result = pyheap.extract(mem)

    assert result.status == "UNAVAILABLE" and result.reason == "PYTHON_DLL_PAGES_UNAVAILABLE"
    assert result.summary()["readable_dll_bytes"] == 0


def test_python_stage_refuses_an_unsupported_windows_tuple_layout():
    py = FakePython()
    py.obj((0, PROMPT_ID, PROMPT, {}, ("9",)))
    mem = py.memory()
    original = mem.images[0]
    mem.images[0] = MappedImage("python314.dll", "python314.dll", original.base, original.end)
    result = pyheap.extract(mem, scan_pixels=False)
    assert result.status == "UNAVAILABLE" and result.reason == "UNSUPPORTED_PYTHON_VERSION:3.14"
    assert not result.items


def _smooth_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h)) + 60], axis=-1)
    return np.clip(base + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)


def test_freed_bytes_object_with_image_size_is_a_pixel_buffer():
    py = FakePython()
    rgb = _smooth_image(256, 256)
    header = struct.pack("<QQQq", 0x2000_0000_1000, 0x2000_0000_2000, rgb.nbytes, -1)  # free-list links, size, -1
    blob = header + rgb.tobytes()
    mem = py.memory({0x3000_0000_0000: blob})
    (buf,) = pyheap.extract(mem).pixel_buffers
    assert (buf.width, buf.height, buf.channels, buf.state) == (256, 256, 3, "FREED_OR_UNTYPED")
    assert buf.address == 0x3000_0000_0000 + 32


def test_a_stale_page_inside_a_pixel_buffer_labels_its_rows_instead_of_losing_the_image():
    py = FakePython()
    rgb = _smooth_image(256, 256)
    base = 0x3000_0000_0000
    mem = py.memory({base: struct.pack("<QQQq", 0, 0, rgb.nbytes, -1) + rgb.tobytes()})
    stale_va = base + 16 * PAGE
    mem.stale[stale_va] = "PFN_MAPS_ANOTHER_PTE"
    (buf,) = pyheap.extract(mem).pixel_buffers
    first, last = (stale_va - base - 32) // 768, (stale_va + PAGE - 1 - base - 32) // 768
    assert buf.stale_rows == frozenset(range(first, last + 1)) and buf.data == rgb.tobytes()
    (record,) = memrecover.image_records([memrecover.from_pixel_buffer(buf, mem.physical_of)])
    reasons = {h["row_range"][0]: h["reason"] for h in record.evidence["holes"]}
    assert reasons == dict.fromkeys(range(first, last + 1), "STALE_MAPPING")


def test_mostly_zero_buffer_is_not_a_pixel_buffer():
    assert pyheap.best_shape(bytes(256 * 256 * 3), pyheap.image_sizes()[256 * 256 * 3]) is None


def test_decoded_bitmap_is_cut_by_stride_and_phase():
    rgb = _smooth_image(300, 512, seed=1)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((300, 512, 1), 255, np.uint8)], axis=2)
    lead = 4 * 37  # the bitmap starts part-way into the region
    data = b"\x10" * lead + bgra.tobytes() + b"\0" * (PAGE * 4)
    found = bitmaps.cut_images("TEST", data, list(range(len(data) // PAGE + 1)), "PHYSICAL_CONTIGUITY")
    assert found, "no bitmap cut"
    best = max(found, key=lambda b: b.rows)
    assert best.width == 512
    # rows that were recovered equal the source rows exactly
    first = next(r for r in range(300) if np.array_equal(rgb[r], best.rgb[0]))
    assert np.array_equal(best.rgb[: min(best.rows, 300 - first)], rgb[first : first + best.rows])


def test_thumbnail_corroborates_full_image_and_flags_damaged_rows():
    # Texture that varies every 2 rows, so a misaligned comparison cannot pass by accident.
    rng = np.random.default_rng(2)
    full = np.repeat(np.repeat(rng.integers(0, 256, (256, 256, 3)), 2, axis=0), 2, axis=1).astype(np.uint8)
    # The decoded thumbnail lost its first row, as on the T2 dump: rows are offset by one.
    thumb = full.astype(np.float32).reshape(256, 2, 256, 2, 3).mean((1, 3)).round().astype(np.uint8)[1:]
    damaged = full.copy()
    damaged[244:248] = 0
    a = memrecover.ImageCandidate(CarveType.PIXEL_BUFFER, damaged, damaged.tobytes(), 1, 0x1000, [], {}, True)
    b = memrecover.ImageCandidate(CarveType.DECODED_BITMAP, thumb, thumb.tobytes(), None, None, [], {}, False)
    lone = memrecover.ImageCandidate(CarveType.DECODED_BITMAP, _smooth_image(300, 640, 3),
                                     b"", None, None, [], {}, False)
    memrecover.corroborate([a, b, lone])
    assert a.corroborations and b.corroborations and not lone.corroborations
    assert a.suspect_rows == {244, 245, 246, 247}
    records = memrecover.image_records([a, b, lone])
    assert [r.pixel_metadata["confidence"] for r in records] == ["CORROBORATED", "CORROBORATED", "SINGLE_COPY"]
    # The full-size buffer's extent is whole, but its damaged rows make it partial.
    assert records[0].completeness == Completeness.PARTIAL and records[1].completeness == Completeness.PARTIAL
    assert [h["row_range"][0] for h in records[0].evidence["holes"]] == [244, 245, 246, 247]
    assert records[0].pixel_metadata["corroborations"][0]["copy"] == records[1].artifact_id


def test_a_third_copy_clears_rows_only_the_damaged_copy_lost():
    rng = np.random.default_rng(3)
    full = np.repeat(rng.integers(0, 256, (150, 256, 3)), 2, axis=0).astype(np.uint8)
    damaged = full[40:].copy()
    damaged[:8] = 0
    copies = [memrecover.ImageCandidate(CarveType.DECODED_BITMAP, rgb, rgb.tobytes(), None, None, [], {}, False)
              for rgb in (full, damaged, full[10:290].copy())]
    memrecover.corroborate(copies)
    assert copies[1].suspect_rows == set(range(8))
    assert not copies[0].suspect_rows and not copies[2].suspect_rows


def test_aliases_of_shared_pages_are_not_independent_raster_corroboration():
    rgb = np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    first = memrecover.ImageCandidate(CarveType.DECODED_BITMAP, rgb, rgb.tobytes(), 5, 0x10000,
                                      [], {}, True, pages=[3])
    alias = memrecover.ImageCandidate(CarveType.DECODED_BITMAP, rgb, rgb.tobytes(), 6, 0x20000,
                                      [], {}, True, pages=[3])
    memrecover.corroborate([first, alias])
    assert first.corroborations == alias.corroborations == []

    alias.pages = [4]
    memrecover.corroborate([first, alias])
    assert first.corroborations and alias.corroborations


def test_workflow_records_merge_copies_and_carry_history():
    item = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}})
    item.history = {"status": {"status_str": "success", "completed": True},
                    "outputs": {"9": {"images": [{"filename": "ComfyUI_temp_00001_.png"}]}}}
    copy = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x2000, 2, 6, PROMPT, None)
    records = memrecover.workflow_records([item, copy], IdAllocator("WFPY"))
    assert [r.prompt_format for r in records] == ["API_GRAPH", "UI_WORKFLOW"]
    api = records[0]
    assert api.encoding_detected == "CPYTHON_OBJECTS"
    assert api.execution_parameters["seed"] == 558648719999037
    assert api.execution_parameters["output_filenames"] == ["ComfyUI_temp_00001_.png"]
    assert len(api.execution_parameters["source_locations"]) == 2


def test_complete_ui_object_stays_context_in_both_evidence_views():
    from comfytrace.correlate import correlate
    from comfytrace.evidence import memory_view

    item = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, None, {"extra_pnginfo": {"workflow": WORKFLOW}})
    records = memrecover.workflow_records([item], IdAllocator("WFPY"))
    [ui] = records
    assert ui.completeness == Completeness.COMPLETE and ui.prompt_format == "UI_WORKFLOW"
    for images, workflows in (([], records), memory_view([], records)):
        result = correlate(images, workflows)
        [finding] = result.findings
        assert finding.tier == "E6" and finding.artifact_ids == [ui.workflow_id]
        assert "UI workflow context" in finding.rationale
        assert not result.links and result.tier_summary["E5"] == 0


@pytest.mark.parametrize("variant", [
    "agreement", "different_id", "different_value", "conflicting_complete", "conflicting_partial",
])
def test_recovered_queue_graph_uses_partial_process_corroboration_without_inheriting_pid(variant):
    complete = api_prompt()
    partial = copy.deepcopy(complete)
    sampler_id, sampler = next((key, node) for key, node in partial.items() if node["class_type"] == "KSampler")
    sampler["inputs"]["latent_image"][0] = None
    prompt_id = "00000000-0000-0000-0000-000000000000" if variant == "different_id" else PROMPT_ID
    if variant == "different_value":
        sampler["inputs"]["seed"] += 1
    item = pyheap.QueueItem(4242, prompt_id, 0, 0x1000, 1, 5, partial, None)
    item.prompt_holes.append({"json_pointer": f"/{sampler_id}/inputs/latent_image/0", "reason": "UNREADABLE"})
    items = [item]
    if variant == "conflicting_partial":
        changed = copy.deepcopy(item)
        changed.address += 0x1000
        changed.prompt[sampler_id]["inputs"]["seed"] += 1
        items.append(changed)
    process_records = memrecover.workflow_records(items, IdAllocator("WFPY"))
    queue = [[0, PROMPT_ID, complete, {}, ["9"]]]
    if variant == "conflicting_complete":
        different = copy.deepcopy(complete)
        different[sampler_id]["inputs"]["seed"] += 1
        queue.append([1, PROMPT_ID, different, {}, ["9"]])
    serial_records = WorkflowCarver().carve_stream(json.dumps(queue).encode())

    rows = evidence.evidence_document([], [*process_records, *serial_records], {})["records"]

    assert all(w.source_process_id is None for w in serial_records)
    process_record = process_records[0]
    assert process_record.completeness == Completeness.PARTIAL
    if variant == "agreement":
        (support,) = rows[1]["queue_graph_corroboration"]
        assert support["workflow_id"] == process_record.workflow_id
        assert support["process_id"] == 4242
        assert support["matched_graph_values"] > len(partial)
    else:
        assert all("queue_graph_corroboration" not in row for row in rows)


def test_corrupt_nested_dict_becomes_a_hole_and_siblings_survive():
    py = FakePython()
    workflow = {"last_node_id": 9, "nodes": [{"id": 3, "flags": {}, "inputs": [CORRUPT, {"name": "clip"}]}],
                "version": 0.4}
    py.obj((0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": workflow}}, ("9",)))
    (item,) = pyheap.extract(py.memory(), scan_pixels=False).items
    assert item.workflow == {"last_node_id": 9, "nodes": [{"id": 3, "flags": {}, "inputs": [None, {"name": "clip"}]}],
                             "version": 0.4}
    assert [(h["json_pointer"], h["reason"]) for h in item.workflow_holes] == [("/nodes/0/inputs/0", "CORRUPT_OBJECT")]
    (api, ui) = memrecover.workflow_records([item], IdAllocator("WFPY"))
    assert ui.completeness == Completeness.PARTIAL and ui.evidence["holes"][0]["json_pointer"] == "/nodes/0/inputs/0"
    assert api.completeness != Completeness.PARTIAL or api.evidence["holes"] == []


def test_a_repeated_dict_key_or_an_overfull_list_is_a_hole_not_a_quiet_rewrite():
    py = FakePython()
    workflow = {"nodes": [REPEATED_KEY, {"id": 4}], "links": OVERFULL_LIST, "version": 0.4}
    py.obj((0, PROMPT_ID, PROMPT, {"extra_pnginfo": {"workflow": workflow}}, ("9",)))
    (item,) = pyheap.extract(py.memory(), scan_pixels=False).items
    assert item.workflow == {"nodes": [None, {"id": 4}], "version": 0.4}
    assert sorted((h["json_pointer"], h["reason"]) for h in item.workflow_holes) == [
        ("/links", "CORRUPT_OBJECT"), ("/nodes/0", "CORRUPT_OBJECT")]


def test_tensor_rows_on_a_flagged_page_are_stale_holes():
    from comfytrace import torchmem

    base = 0x100000
    data_va = base + 0x20000
    raw = np.random.default_rng(3).random((1, 64, 64, 3), dtype=np.float32).tobytes()
    physical = bytearray(0x20000 + len(raw))
    physical[0x20000:] = raw
    storage = base + 0x100
    struct.pack_into("<7Q", physical, 0x100, 0x600100, 1, data_va, 0x500100, data_va, 0xFF00, len(raw))
    words = [0] * 22
    words[:3] = [0x600200, 1, storage]
    words[8] = 4
    words[9:13] = [1, 64, 64, 3]
    words[14:18] = [12288, 192, 3, 1]
    words[19:22] = [0, 12288, 6]
    struct.pack_into("<22Q", physical, 0x200, *words)
    images = [MappedImage("c10.dll", "c10.dll", 0x500000, 0x501000),
              MappedImage("torch.dll", "torch.dll", 0x600000, 0x601000)]
    mem = ProcessMemory(1, [(base, len(physical), 0)], bytes(physical), images=images,
                        stale={data_va + PAGE: "PFN_FREED"})
    (tensor,) = torchmem.image_tensors(mem)
    # One row is 64 * 3 float32 = 768 bytes; the flagged page [4096, 8192) touches rows 5 to 10.
    assert tensor.stale_rows == frozenset(range(5, 11))
    (record,) = memrecover.image_records([memrecover.from_tensor(tensor, mem.physical_of)])
    assert record.completeness == Completeness.PARTIAL
    assert {h["reason"] for h in record.evidence["holes"]} == {"STALE_MAPPING"}
    assert [h["row_range"][0] for h in record.evidence["holes"]] == list(range(5, 11))


def test_scorecard_reports_per_output_status_and_liveness():
    from comfytrace import evidence

    live = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {"extra_pnginfo": {"workflow": WORKFLOW}})
    live.owner = "history record 0x1"
    live.history = {"outputs": {"9": {"images": [{"filename": "a.png"}, {"filename": "b.png"}]}}}
    api_only = pyheap.QueueItem(4242, "fb93593a-6cc0-4e0a-a7cc-dc727e31d4c2", 1, 0x2000, 1, 5, PROMPT, {})
    records = memrecover.workflow_records([live, api_only], IdAllocator("WFPY"))
    card = evidence.scorecard([live, api_only], records, "ab" * 32)
    rows = {(r["prompt_id"], r["batch_index"]): r for r in card["rows"]}
    assert rows[(PROMPT_ID, 0)]["output_filename"] == "a.png" and rows[(PROMPT_ID, 1)]["liveness"] == "LIVE"
    assert rows[("fb93593a-6cc0-4e0a-a7cc-dc727e31d4c2", None)]["ui_workflow"]["status"] == "NOT_PRESENT"
    assert rows[("fb93593a-6cc0-4e0a-a7cc-dc727e31d4c2", None)]["liveness"] == "UNKNOWN"
    assert card["summary"] == {"jobs_in_backend_holders": 2, "live_jobs": 1, "api_prompt_recovered": 2,
                               "ui_workflow_recovered": 1, "ui_workflow_not_present": 1, "outputs": 2,
                               "jobs_with_linked_image": 0,
                               "outputs_with_linked_image": 0, "outputs_with_linked_memory_image": 0,
                               "outputs_with_execution_cache_image": 0}


def test_identical_graphs_of_two_jobs_stay_two_records():
    a = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {})
    other = "fb93593a-6cc0-4e0a-a7cc-dc727e31d4c2"
    b = pyheap.QueueItem(4242, other, 1, 0x2000, 1, 5, PROMPT, {})
    records = memrecover.workflow_records([a, b], IdAllocator("WFPY"))
    assert sorted(r.evidence["prompt_id"] for r in records) == sorted([PROMPT_ID, other])


def test_pages_the_pfn_database_disagrees_with_are_left_out_with_their_reason():
    from comfytrace.procmem import StaleMapping

    phys = Phys(128)
    kdtb, udtb = phys.alloc(), phys.alloc()
    ktables: dict[tuple[int, int], int] = {}
    utables: dict[tuple[int, int], int] = {}
    pfn_db, pte_base, proto_va = 0xFFFF_C000_0000_0000, 0xFFFF_9080_0000_0000, 0xFFFF_E585_0000_0000
    db_pages = [phys.alloc(), phys.alloc()]
    for i, page in enumerate(db_pages):
        _map(phys, kdtb, pfn_db + i * PAGE, page | VALID, ktables)
    proto_page = phys.alloc()
    _map(phys, kdtb, proto_va, proto_page | VALID, ktables)

    def pfn_entry(frame: int, pte_address: int, location: int, shared: bool = False) -> None:
        at = pfn_db + (frame >> 12) * 48
        raw = bytearray(48)
        raw[8:16] = struct.pack("<Q", pte_address)
        raw[34] = location
        raw[40:48] = struct.pack("<Q", (1 << 63) if shared else 0)
        for k in range(48):
            va = at + k
            phys.data[db_pages[(va - pfn_db) // PAGE] + (va & 0xFFF)] = raw[k]

    good, moved, freed, shared, trimmed, cached = (phys.alloc() for _ in range(6))
    pfn_entry(good, pte_base + (0x10000 >> 12) * 8, 6)
    pfn_entry(moved, 0xFFFF_9080_1234_5678 & ~7, 6)  # the frame now belongs to another PTE
    pfn_entry(freed, pte_base + (0x12000 >> 12) * 8, 1)  # the frame went to the Free list
    phys.q(proto_page + 16, shared | VALID)
    pfn_entry(shared, (proto_va + 16) & 0x7FFF_FFFF_FFFF_FFFF, 6, shared=True)  # bit 63 clear, as on T2
    pfn_entry(trimmed, pte_base + (0x15000 >> 12) * 8, 2)  # trimmed to Standby during acquisition: still ours
    phys.q(proto_page + 24, cached | TRANSITION)
    pfn_entry(cached, proto_va + 24, 2, shared=True)  # a Standby file-cache frame behind a valid private PTE
    _map(phys, udtb, 0x10000, good | VALID, utables)
    _map(phys, udtb, 0x11000, moved | VALID, utables)
    _map(phys, udtb, 0x12000, freed | TRANSITION, utables)
    _map(phys, udtb, 0x13000, shared | VALID, utables)  # a valid hardware PTE of a shared section page
    _map(phys, udtb, 0x14000, (len(phys.data) + PAGE) | VALID, utables)
    _map(phys, udtb, 0x15000, trimmed | VALID, utables)
    _map(phys, udtb, 0x16000, cached | VALID, utables)
    kernel = PageTables(phys.data, kdtb)
    support = bitmaps.PfnSupport(kernel, pfn_db, 48, 8, pte_base, 34, (40, 63))
    mem = ProcessMemory.from_page_tables(7, PageTables(phys.data, udtb, kernel=kernel),
                                         [(0x10000, 0x16FFF, None)], phys.data, support)
    assert mem.stale == {0x11000: "PFN_MAPS_ANOTHER_PTE", 0x12000: "PFN_FREED", 0x14000: "FRAME_BEYOND_IMAGE",
                         0x16000: "SHARED_FRAME_NOT_IN_USE"}
    # Labelled pages stay readable: the disagreement does not say which copy is current.
    assert all(mem.try_read(va, PAGE) is not None for va in (0x10000, 0x11000, 0x12000, 0x13000, 0x15000, 0x16000))
    assert mem.stale_pages(0x10800, 0x2000) == [0x11000, 0x12000]
    with pytest.raises(StaleMapping):
        mem.read(0x14008, 8)  # the frame beyond the image has nothing to read
    try:
        mem.read(0x14008, 8)
    except Unreadable as exc:
        assert pyheap.hole(["nodes", 1], exc)["reason"] == pyheap.STALE_MAPPING
    corrupt = pyheap.CorruptObject("dict keys", 0x11040)
    assert pyheap.hole(["nodes", 2], corrupt, mem.stale)["reason"] == pyheap.STALE_MAPPING
    assert pyheap.hole(["nodes", 2], pyheap.CorruptObject("dict keys", 0x10040), mem.stale)["reason"] == \
        pyheap.CORRUPT_OBJECT
    unchecked = ProcessMemory.from_page_tables(7, PageTables(phys.data, udtb, kernel=kernel),
                                               [(0x10000, 0x16FFF, None)], phys.data)
    assert unchecked.stale == {0x14000: "FRAME_BEYOND_IMAGE"} and unchecked.try_read(0x11000, PAGE) is not None


def test_each_page_is_told_apart_as_private_shared_file_image_or_freed():
    from comfytrace.models import PageOrigin

    phys = Phys(32)
    kdtb, ktables = phys.alloc(), {}
    pfn_db, pte_base = 0xFFFF_C000_0000_0000, 0xFFFF_9080_0000_0000
    db_page, pool = phys.alloc(), phys.alloc()
    _map(phys, kdtb, pfn_db, db_page | VALID, ktables)
    pool_va = 0xFFFF_A000_0000_0000
    _map(phys, kdtb, pool_va, pool | VALID, ktables)
    # Pool: three subsections (ControlArea pointer at +0x10) and three control areas (flags word at +0x38).
    for n, flags in enumerate((1 << 7, 1 << 5, 0)):
        area = pool_va + 0x800 + 0x100 * n
        phys.q(pool + 0x100 * n + 0x10, area)
        phys.q(pool + 0x800 + 0x100 * n + 0x38, flags)

    def entry(pfn: int, location: int, shared: bool, original: int = 0) -> None:
        raw = bytearray(48)
        raw[16:24] = struct.pack("<Q", original)
        raw[34] = location
        raw[40:48] = struct.pack("<Q", (1 << 63) if shared else 0)
        phys.put(db_page + pfn * 48, bytes(raw))

    def subsection_pte(va: int) -> int:
        return (va & 0xFFFF_FFFF_FFFF) << 16 | PROTOTYPE

    entry(1, 6, False)  # behind a hardware PTE
    entry(2, 2, True, original=0x4_0000_0000 | 0x80)  # pagefile-backed section: a software OriginalPte
    entry(3, 2, True, original=subsection_pte(pool_va))  # control area flagged File
    entry(4, 6, True, original=subsection_pte(pool_va + 0x100))  # control area flagged Image
    entry(5, 2, True, original=subsection_pte(pool_va + 0x200))  # neither flag
    entry(6, 2, True, original=subsection_pte(0xFFFF_B000_0000_0000))  # subsection not readable
    entry(7, 1, True, original=subsection_pte(pool_va))  # on the Free list: the old owner is gone
    entry(8, 0, False)  # Zeroed
    layout = bitmaps.SectionLayout(original_pte_offset=16, control_area_offset=0x10, flags_offset=0x38,
                                   file_bit=7, image_bit=5)
    support = bitmaps.PfnSupport(PageTables(phys.data, kdtb), pfn_db, 48, 8, pte_base, 34, (40, 63), layout)
    assert [support.origin(p) for p in range(1, 9)] == [
        PageOrigin.PRIVATE, PageOrigin.SHARED_MEMORY, PageOrigin.FILE_DATA, PageOrigin.FILE_IMAGE,
        PageOrigin.FILE_UNCONFIRMED, PageOrigin.FILE_UNCONFIRMED, PageOrigin.FREED, PageOrigin.FREED]
    assert support.origin(1000) == PageOrigin.UNKNOWN  # beyond the readable PFN database
    assert support.copy_origin([1, 2, 1, 3], 9000) == {"pages": {"FILE_DATA": 1, "PRIVATE": 1, "SHARED_MEMORY": 1},
                                                       "bytes": 9000}
    unresolved = bitmaps.PfnSupport(PageTables(phys.data, kdtb), pfn_db, 48, 8, pte_base, 34, (40, 63))
    assert unresolved.origin(1) == PageOrigin.PRIVATE and unresolved.origin(3) == PageOrigin.UNKNOWN

    file_name_layout = bitmaps.FileNameLayout(
        file_pointer=0x48, reference_bits=4, file_name=0x40, length=0, maximum_length=2, buffer=8,
    )
    named_layout = bitmaps.SectionLayout(16, 0x10, 0x38, file_bit=7, image_bit=5, file_name=file_name_layout)
    named = bitmaps.PfnSupport(PageTables(phys.data, kdtb), pfn_db, 48, 8, pte_base, 34, (40, 63), named_layout)
    path = r"\ComfyUI\output\fixture.png"
    encoded = path.encode("utf-16-le")
    phys.q(pool + 0x848, (pool_va + 0x500) | 3)
    phys.put(pool + 0x540, struct.pack("<HH", len(encoded), len(encoded)))
    phys.q(pool + 0x548, pool_va + 0x600)
    phys.put(pool + 0x600, encoded)
    phys.q(db_page + 3 * 48 + 8, pool_va + 0x400)
    phys.q(pool + 0x400, 3 * PAGE | VALID)

    assert named.copy_origin([3, 3, 4, 7], 200)["backing_file_names"] == [path]
    assert "backing_file_names" not in named.copy_origin([4, 7], 200)
    phys.q(pool + 0x400, 4 * PAGE | VALID)
    assert named.copy_origin([3], 200) == {"pages": {"FILE_DATA": 1}, "bytes": 200}


@pytest.mark.parametrize(("copies", "verdict"), [
    ([], "NOT_CLASSIFIED"),
    ([{"pages": {}, "bytes": 0}], "NOT_CLASSIFIED"),
    ([{"pages": {"FILE_DATA": 3}, "bytes": 100}, {"pages": {"PRIVATE": 2, "SHARED_MEMORY": 1}, "bytes": 100}],
     "MEMORY"),
    ([{"pages": {"FILE_DATA": 3}, "bytes": 100}, {"pages": {"FREED": 1}, "bytes": 40}], "MEMORY_PARTIAL"),
    ([{"pages": {"FILE_DATA": 2}, "bytes": 100}, {"pages": {"FILE_UNCONFIRMED": 1}, "bytes": 100}], "FILE_BACKED"),
    ([{"pages": {"FILE_DATA": 1, "PRIVATE": 1}, "bytes": 100}], "MIXED"),
    ([{"pages": {"UNKNOWN": 2}, "bytes": 100}], "UNKNOWN"),
])
def test_a_record_stands_on_memory_only_when_a_whole_copy_avoids_file_pages(copies, verdict):
    from comfytrace import evidence

    assert evidence.memory_source(copies, 100) == verdict


def test_a_private_bitmap_is_ordered_through_its_page_tables_across_a_table_boundary():
    """A decoded image in private process memory: frames scattered, two page-table pages, one frame reused."""
    phys = Phys(420)
    kdtb, udtb = phys.alloc(), phys.alloc()
    ktables: dict[tuple[int, int], int] = {}
    utables: dict[tuple[int, int], int] = {}
    pfn_db, pte_base = 0xFFFF_C000_0000_0000, 0xFFFF_9080_0000_0000
    db_pages = [phys.alloc() for _ in range(5)]
    for i, page in enumerate(db_pages):
        _map(phys, kdtb, pfn_db + i * PAGE, page | VALID, ktables)

    def pfn_entry(frame: int, pte_address: int, pte_frame: int) -> None:
        raw = bytearray(48)
        raw[8:16] = struct.pack("<Q", pte_address)
        raw[34] = 6  # Active
        raw[40:48] = struct.pack("<Q", pte_frame >> 12)  # PteFrame in bits 0..39, PrototypePte (bit 63) clear
        at = (frame >> 12) * 48
        for k in range(48):
            phys.data[db_pages[(at + k) // PAGE] + (at + k) % PAGE] = raw[k]

    rgb = _smooth_image(240, 512, seed=3)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((240, 512, 1), 255, np.uint8)], axis=2).tobytes()
    va0 = 0x2000_0000 - 60 * PAGE  # 60 pages under one page-table page, 60 under the next
    frames = [phys.alloc() for _ in range(120)]
    np.random.default_rng(5).shuffle(frames)
    for i, frame in enumerate(frames):
        phys.put(frame, bgra[i * PAGE : (i + 1) * PAGE])
        _map(phys, udtb, va0 + i * PAGE, frame | VALID, utables)
    pdpt = utables[(udtb, (va0 >> 39) & 511)]
    pd = utables[(pdpt, (va0 >> 30) & 511)]
    for i, frame in enumerate(frames):
        va = va0 + i * PAGE
        table = utables[(pd, (va >> 21) & 511)]
        pfn_entry(frame, pte_base + (va >> 12) * 8, table)
        pfn_entry(table, pte_base + ((va >> 21) & 511) * 8, pd)
    pfn_entry(frames[70], pte_base + 8, udtb)  # reused since: its PFN entry names another table
    kernel = PageTables(phys.data, kdtb)
    support = bitmaps.PfnSupport(kernel, pfn_db, 48, 8, pte_base, 34, (40, 63), pte_frame=(40, 0, 40))
    [found] = [b for b in bitmaps.scan(phys.data, support) if b.ordering == "PAGE_TABLE_ORDER"]
    # One image across both table pages; page 70's frame was reused, so its rows (140, 141) are a gap, not pixels.
    assert (found.width, found.rows) == (512, 240) and found.notes["gap_rows"] == [[140, 142, "STALE_MAPPING"]]
    assert frames[70] >> 12 not in found.physical_pages and frames[0] >> 12 == found.physical_pages[0]
    genuine = np.r_[0:140, 142:240]
    assert np.array_equal(found.rgb[genuine], rgb[genuine]) and not found.rgb[140:142].any()
    # The frame the page table still names was read into a separate layer, never into the image.
    assert found.notes["uncertain_rows"] == [[140, 142, [f"0x{frames[70]:X}"]]]
    assert np.array_equal(np.stack([found.uncertain_rows[r] for r in (140, 141)]), rgb[140:142])
    [record] = memrecover.image_records([memrecover.from_bitmap(found)])
    assert record.completeness == "PARTIAL" and record.evidence["ownership_uncertain"]["rows_total"] == 2
    assert [(g.relative_offset, g.length_bytes, g.reason) for g in record.gaps] == [(140 * 1536, 2 * 1536,
                                                                                     "STALE_MAPPING")]
    assert b"".join(span for _, span in record.spans) == rgb[genuine].tobytes(), "raw holds genuine rows only"
    digest = hashlib.sha256(rgb[genuine].tobytes()).hexdigest()
    assert record.pixel_sha256 == digest
    assert record.evidence["sha256"] == {"raw": digest, "pixels": digest}
    unordered = bitmaps.PfnSupport(kernel, pfn_db, 48, 8, pte_base, 34, (40, 63))
    assert not [b for b in bitmaps.scan(phys.data, unordered) if b.ordering == "PAGE_TABLE_ORDER"]


@pytest.mark.parametrize("mapping", ["PAGE_TABLE_ORDER", "PROTOTYPE_PTE_ORDER"])
def test_a_scattered_png_is_read_on_in_the_order_of_its_mapping(mapping):
    from comfytrace.carve import parse_png
    from comfytrace.pageorder import page_order_stream
    from tests.synthetic import noise_image, png_bytes

    png = png_bytes(noise_image(120, 90, seed=11), {"prompt": "{}"})
    count = (len(png) + PAGE - 1) // PAGE
    assert count >= 3
    phys = Phys(160)
    kdtb, udtb = phys.alloc(), phys.alloc()
    ktables: dict[tuple[int, int], int] = {}
    utables: dict[tuple[int, int], int] = {}
    pfn_db, pte_base, proto_va = 0xFFFF_C000_0000_0000, 0xFFFF_9080_0000_0000, 0xFFFF_E585_0000_0000
    db_pages = [phys.alloc() for _ in range(2)]
    for i, page in enumerate(db_pages):
        _map(phys, kdtb, pfn_db + i * PAGE, page | VALID, ktables)
    proto_page = phys.alloc()
    _map(phys, kdtb, proto_va, proto_page | VALID, ktables)

    def pfn_entry(frame: int, pte_address: int, pte_frame: int, shared: bool) -> None:
        raw = bytearray(48)
        raw[8:16] = struct.pack("<Q", pte_address)
        raw[34] = 6 if not shared else 2
        raw[40:48] = struct.pack("<Q", pte_frame >> 12 | ((1 << 63) if shared else 0))
        at = (frame >> 12) * 48
        for k in range(48):
            phys.data[db_pages[(at + k) // PAGE] + (at + k) % PAGE] = raw[k]

    frames = [phys.alloc() for _ in range(count)]
    np.random.default_rng(2).shuffle(frames)
    va0 = 0x500000
    for i, frame in enumerate(frames):
        phys.put(frame, png[i * PAGE : (i + 1) * PAGE])
        if mapping == "PAGE_TABLE_ORDER":
            _map(phys, udtb, va0 + i * PAGE, frame | VALID, utables)
        else:
            phys.q(proto_page + 8 * i, frame | VALID)
    if mapping == "PAGE_TABLE_ORDER":
        pd = utables[(utables[(udtb, 0)], 0)]
        for i, frame in enumerate(frames):
            va = va0 + i * PAGE
            table = utables[(pd, (va >> 21) & 511)]
            pfn_entry(frame, pte_base + (va >> 12) * 8, table, False)
            pfn_entry(table, pte_base + ((va >> 21) & 511) * 8, pd, False)
    else:
        for i, frame in enumerate(frames):
            pfn_entry(frame, (proto_va + 8 * i) & 0x7FFF_FFFF_FFFF_FFFF, proto_page, True)
    support = bitmaps.PfnSupport(PageTables(phys.data, kdtb), pfn_db, 48, 8, pte_base, 34, (40, 63),
                                 pte_frame=(40, 0, 40))
    torn = parse_png(bytes(phys.data[frames[0] : frames[0] + PAGE]), True, None, frames[0])
    assert torn is not None and not torn.complete
    data, method, read = page_order_stream(phys.data, support, frames[0], len(png) + PAGE)
    assert method == mapping and read[:count] == [f >> 12 for f in frames]
    parsed = parse_png(data, False, None, frames[0])
    assert parsed.complete and parsed.spans[0][1] == png


def _history(prompt_id: str, *events: tuple[str, int], nodes: tuple[str, ...] = ()) -> dict:
    messages = [[name, {"prompt_id": prompt_id, "timestamp": stamp}] for name, stamp in events]
    messages.insert(1, ["execution_cached", {"nodes": list(nodes), "prompt_id": prompt_id, "timestamp": 1}])
    return {"status": {"status_str": "success", "completed": True, "messages": messages}, "outputs": {}}


def test_a_job_timeline_comes_from_comfyui_own_timestamps():
    item = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {"create_time": 1794065346900})
    item.history = _history(PROMPT_ID, ("execution_start", 1794065346982), ("execution_success", 1794065421254),
                            nodes=("1",))
    item.history["status"]["messages"].append(["execution_start", {"prompt_id": "another", "timestamp": 5}])
    [api] = memrecover.workflow_records([item], IdAllocator("WFPY"))
    assert api.execution_parameters["job_timeline"] == {
        "queued_utc": "2026-11-07T15:29:06.900Z", "started_utc": "2026-11-07T15:29:06.982Z",
        "finished_utc": "2026-11-07T15:30:21.254Z", "outcome": "success", "cached_nodes": ["1"],
        "run_seconds": 74.272}


def test_an_interrupted_job_and_a_copy_without_history_keep_what_was_recorded():
    bare = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {})
    held = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x2000, 1, 5, PROMPT, {})
    held.history = _history(PROMPT_ID, ("execution_interrupted", 1794065421254))
    [api] = memrecover.workflow_records([bare, held], IdAllocator("WFPY"))
    assert api.execution_parameters["job_timeline"] == {
        "finished_utc": "2026-11-07T15:30:21.254Z", "outcome": "interrupted", "cached_nodes": []}


def test_the_scorecard_lists_images_linked_to_the_job_prompt_and_then_the_regeneration_links():
    from comfytrace import evidence
    from comfytrace.models import CorrelationLink

    item = pyheap.QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, PROMPT, {})
    item.history = {"outputs": {"9": {"images": [{"filename": "a.png"}, {"filename": "b.png"}]}}}
    [api] = memrecover.workflow_records([item], IdAllocator("WFPY"))
    carved = dataclasses.replace(api, workflow_id="WF-001")
    other = dataclasses.replace(api, workflow_id="WF-002", canonical_json_sha256="0" * 64, evidence={})
    links = [CorrelationLink("LINK-001", "WF-001", "CARVE-PNG-001", "STRICT", ["CANONICAL_PROMPT_HASH"], ""),
             CorrelationLink("LINK-002", "WF-002", "CARVE-PNG-002", "STRICT", ["CANONICAL_PROMPT_HASH"], ""),
             CorrelationLink("LINK-003", "WF-001", "CARVE-PNG-003", "UNASSOCIATED", [], "")]
    card = evidence.scorecard([item], [api, carved, other], "ab" * 32, links)
    job_candidate = {"artifact_id": "CARVE-PNG-001", "basis": "CORRELATION_STRICT",
                     "matched_keys": ["CANONICAL_PROMPT_HASH"], "role": "JOB"}
    assert all(row["image"] == {"status": "JOB_LINKED", "candidates": [job_candidate]} for row in card["rows"])
    image_links = {"manifest_sha256": "cd" * 32, "jobs": [{"prompt_id": PROMPT_ID, "image_status": "OBSERVED_EXACT",
                   "observed": [{"artifact_id": "BITMAP-001", "kind": "DECODED_BITMAP", "agreement": "EXACT"}]}]}
    assert card["summary"]["outputs_with_linked_memory_image"] == 0, "the PNG's pages are unknown here"
    sources = {"BITMAP-001": "MEMORY"}
    once = evidence.apply_image_links(copy.deepcopy(card), image_links, sources)
    assert once == evidence.apply_image_links(copy.deepcopy(once), image_links, sources), "applying twice is a no-op"
    assert once["summary"]["outputs_with_linked_memory_image"] == 0
    assert [c["basis"] for c in once["rows"][0]["image"]["candidates"]] == [
        "CORRELATION_STRICT", "REGENERATION_COMPARISON"]
    assert once["rows"][0]["image"]["regeneration"]["image_status"] == "OBSERVED_EXACT"
    assert once["summary"]["outputs_with_linked_image"] == 0
    assert once["summary"]["jobs_with_linked_image"] == 1


def test_the_scorecard_does_not_copy_an_image_to_an_identical_graphs_other_job():
    from comfytrace import evidence
    from comfytrace.models import CorrelationLink

    other = "00000000-0000-0000-0000-000000000000"
    items = [pyheap.QueueItem(4242, pid, 0, address, 1, 5, PROMPT, {})
             for pid, address in ((PROMPT_ID, 0x1000), (other, 0x2000))]
    workflows = memrecover.workflow_records(items, IdAllocator("WFPY"))
    first = next(w for w in workflows if w.execution_parameters.get("prompt_id") == PROMPT_ID)
    links = [CorrelationLink("LINK-001", first.workflow_id, "CARVE-PNG-001", "STRICT", ["PROMPT_ID"], "")]

    rows = evidence.scorecard(items, workflows, "ab" * 32, links)["rows"]

    assert {row["prompt_id"]: len(row["image"]["candidates"]) for row in rows} == {PROMPT_ID: 1, other: 0}
