"""PFN file-cache backing names (``PfnSupport.copy_origin``) as contextual evidence only.

Everything here is synthetic: a toy kernel address space with a PFN database, prototype PTEs, subsections, control
areas and file objects with toy paths. A page earns a name only when its own PFN entry, its prototype PTE and its
control area agree; names never change the origin counts, the memory source, a link class or an output position.
"""

from __future__ import annotations

import copy
import json
import struct

import pytest

from comfytrace import bitmaps, evidence
from comfytrace.carve import IdAllocator, PngCarver, WorkflowCarver
from comfytrace.correlate import correlate
from comfytrace.models import PageOrigin
from comfytrace.pagetable import PROTOTYPE, PageTables
from comfytrace.pyheap import QueueItem
from tests.synthetic import api_prompt, noise_image, png_bytes, prompt_request_body

PAGE = 4096
VALID, TRANSITION = 1, 1 << 11
PFN_DB, PTE_BASE = 0xFFFF_C000_0000_0000, 0xFFFF_9080_0000_0000
POOL, PROTO = 0xFFFF_A000_0000_0000, 0xFFFF_E585_0000_0000
ENTRY = 48
ACTIVE, STANDBY, FREE = 6, 2, 1
FILE_BIT, IMAGE_BIT = 7, 5
PATH_A, PATH_B, PATH_C = r"\toy\cache\alpha.dat", r"\toy\cache\beta.dat", r"\toy\cache\gamma.dat"
LAYOUT = bitmaps.FileNameLayout(file_pointer=0x48, reference_bits=4, file_name=0x40, length=0, maximum_length=2,
                                buffer=8)
SECTION = bitmaps.SectionLayout(original_pte_offset=16, control_area_offset=0x10, flags_offset=0x38,
                                file_bit=FILE_BIT, image_bit=IMAGE_BIT, file_name=LAYOUT)
STALE_PTE = 0xFFFF_B000_0000_0000  # a kernel address nothing maps


class Toy:
    """A fake physical image whose kernel half holds a PFN database, a prototype PTE array and a toy pool."""

    def __init__(self) -> None:
        self.data = bytearray(96 * PAGE)
        self.next = 1
        self.kdtb = self._alloc()
        self.tables: dict[tuple[int, int], int] = {}
        for i in range(2):
            self.map(PFN_DB + i * PAGE)
        self.map(PROTO)
        for i in range(8):
            self.map(POOL + i * PAGE)
        self.support = bitmaps.PfnSupport(PageTables(self.data, self.kdtb), PFN_DB, ENTRY, 8, PTE_BASE, 34, (40, 63),
                                          SECTION)
        self.files = 0

    def _alloc(self) -> int:
        base = self.next * PAGE
        self.next += 1
        return base

    def map(self, va: int, leaf: int | None = None) -> int:
        """Maps ``va`` to a fresh page (or ``leaf``) in the kernel tables; returns the leaf entry."""
        base = self.kdtb
        for shift in (39, 30, 21):
            key = (base, (va >> shift) & 0x1FF)
            if key not in self.tables:
                self.tables[key] = self._alloc()
                self.put_phys(base + 8 * key[1], struct.pack("<Q", self.tables[key] | VALID))
            base = self.tables[key]
        leaf = (self._alloc() | VALID) if leaf is None else leaf
        self.put_phys(base + 8 * ((va >> 12) & 0x1FF), struct.pack("<Q", leaf))
        if hasattr(self, "support"):  # PageTables caches the table pages it has read
            self.support.kernel = PageTables(self.data, self.kdtb)
        return leaf

    def put_phys(self, at: int, blob: bytes) -> None:
        self.data[at : at + len(blob)] = blob

    def put(self, va: int, blob: bytes) -> None:
        """Writes through the kernel page tables, so a blob may cross a page boundary."""
        tables = PageTables(self.data, self.kdtb)
        for i, byte in enumerate(blob):
            where = tables.translate(va + i)
            assert where is not None, hex(va + i)
            self.data[where] = byte

    def q(self, va: int, value: int) -> None:
        self.put(va, struct.pack("<Q", value))

    # --- file objects, control areas, subsections -------------------------------------------------------------
    def file(self, path: str, *, flags: int = 1 << FILE_BIT) -> int:
        """A toy file with a subsection, control area, file object and name buffer. Returns its number."""
        n = self.files
        self.files += 1
        encoded = path.encode("utf-16-le")
        self.q(self.subsection(n) + 0x10, self.control_area(n))
        self.q(self.control_area(n) + 0x38, flags)
        self.q(self.control_area(n) + 0x48, self.file_object(n) | 3)  # EX_FAST_REF: reference count in low bits
        self.put(self.file_object(n) + 0x40, struct.pack("<HH", len(encoded), len(encoded) + 2))
        self.q(self.file_object(n) + 0x48, self.name_buffer(n))
        self.put(self.name_buffer(n), encoded)
        return n

    @staticmethod
    def subsection(n: int) -> int:
        return POOL + 0x100 * n

    @staticmethod
    def control_area(n: int) -> int:
        return POOL + 0x800 + 0x100 * n

    @staticmethod
    def file_object(n: int) -> int:
        return POOL + PAGE + 0x100 * n

    @staticmethod
    def name_buffer(n: int) -> int:
        return POOL + 2 * PAGE + 0x100 * n

    # --- PFN entries and prototype PTEs -----------------------------------------------------------------------
    def entry(self, pfn: int, *, location: int = STANDBY, shared: bool = True, original: int = 0,
              pte_address: int = 0) -> None:
        raw = bytearray(ENTRY)
        raw[8:16] = struct.pack("<Q", pte_address)
        raw[16:24] = struct.pack("<Q", original)
        raw[34] = location
        raw[40:48] = struct.pack("<Q", (1 << 63) if shared else 0)
        self.put(PFN_DB + pfn * ENTRY, bytes(raw))

    @staticmethod
    def subsection_pte(n: int) -> int:
        return (Toy.subsection(n) & 0xFFFF_FFFF_FFFF) << 16 | PROTOTYPE

    @staticmethod
    def slot(pfn: int) -> int:
        """Address of the prototype PTE of a toy section page (each frame has its own slot)."""
        return PROTO + pfn * 8

    def cached(self, pfn: int, n: int, *, location: int = STANDBY, proto: int | None = None) -> None:
        """Frame ``pfn`` as a data page of file ``n``: its PFN entry and a prototype PTE that names it back."""
        self.entry(pfn, location=location, original=self.subsection_pte(n), pte_address=self.slot(pfn))
        self.q(self.slot(pfn), (pfn * PAGE | VALID) if proto is None else proto)

    def names(self, pfns: list[int]) -> list[str] | None:
        return self.support.copy_origin(pfns, 100).get("backing_file_names")


@pytest.fixture
def toy() -> Toy:
    return Toy()


def test_a_page_whose_entry_prototype_and_control_area_agree_is_named(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    assert toy.support.copy_origin([3, 3], 4096) == {"pages": {"FILE_DATA": 1}, "bytes": 4096,
                                                     "backing_file_names": [PATH_A]}
    # A Standby frame (transition prototype PTE) and an Active one (valid) both count; low PteAddress flag bits and a
    # clear bit 63 (as on T2) do not matter.
    toy.cached(4, a, location=ACTIVE, proto=4 * PAGE | TRANSITION)
    toy.entry(4, location=ACTIVE, original=toy.subsection_pte(a), pte_address=(toy.slot(4) & ~(1 << 63)) | 5)
    assert toy.names([4]) == [PATH_A]


@pytest.mark.parametrize("break_it", [
    pytest.param(lambda t, pfn: t.entry(pfn, original=t.subsection_pte(0), pte_address=0), id="no-pte-address"),
    pytest.param(lambda t, pfn: t.entry(pfn, original=t.subsection_pte(0), pte_address=STALE_PTE),
                 id="pte-address-unreadable"),
    pytest.param(lambda t, pfn: t.q(t.slot(pfn), 9 * PAGE | VALID), id="prototype-names-a-reused-frame"),
    pytest.param(lambda t, pfn: t.q(t.slot(pfn), 0), id="prototype-empty"),
    pytest.param(lambda t, pfn: t.q(t.slot(pfn), (0x1234 << 32) | 0x80), id="prototype-paged-out"),
    pytest.param(lambda t, pfn: t.q(t.slot(pfn), t.subsection_pte(0)), id="prototype-not-yet-faulted"),
    pytest.param(lambda t, pfn: t.q(t.slot(pfn), (pfn * PAGE) | TRANSITION | PROTOTYPE),
                 id="prototype-transition-and-prototype-format"),
    pytest.param(lambda t, pfn: t.entry(pfn, original=t.subsection_pte(0), pte_address=t.slot(pfn + 1)),
                 id="pte-address-is-the-neighbours-slot"),
])
def test_a_stale_reused_or_missing_prototype_pointer_names_nothing(toy, break_it):
    toy.file(PATH_A)
    toy.cached(3, 0)
    toy.cached(4, 0)  # the neighbour's own slot is intact and names frame 4, never frame 3
    break_it(toy, 3)
    result = toy.support.copy_origin([3], 100)
    assert result == {"pages": {"FILE_DATA": 1}, "bytes": 100}
    assert toy.names([4]) == [PATH_A]  # the neighbour keeps its own name


def test_the_name_is_read_live_on_every_call_with_no_cache(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    assert toy.names([3]) == [PATH_A]
    renamed = r"\toy\cache\renamed.dat".encode("utf-16-le")
    toy.put(toy.file_object(a) + 0x40, struct.pack("<HH", len(renamed), len(renamed) + 2))
    toy.put(toy.name_buffer(a), renamed)
    assert toy.names([3]) == [r"\toy\cache\renamed.dat"]
    toy.q(toy.slot(3), 7 * PAGE | VALID)  # the frame is reused between two calls
    assert toy.names([3]) is None


@pytest.mark.parametrize(("setup", "origin"), [
    pytest.param(lambda t: t.cached(3, 0, location=FREE), PageOrigin.FREED, id="freed-frame-with-intact-prototype"),
    pytest.param(lambda t: (t.cached(3, 0), t.entry(3, shared=False, original=t.subsection_pte(0),
                                                    pte_address=t.slot(3))), PageOrigin.PRIVATE,
                 id="private-frame-with-a-leftover-subsection-pte"),
    pytest.param(lambda t: (t.cached(3, 0), t.entry(3, original=0x4_0000_0000 | 0x80, pte_address=t.slot(3))),
                 PageOrigin.SHARED_MEMORY, id="pagefile-section"),
    pytest.param(lambda t: (t.cached(3, 0), t.q(t.control_area(0) + 0x38, 1 << IMAGE_BIT)), PageOrigin.FILE_IMAGE,
                 id="image-section"),
    pytest.param(lambda t: (t.cached(3, 0), t.q(t.control_area(0) + 0x38, 0)), PageOrigin.FILE_UNCONFIRMED,
                 id="control-area-without-file-flag"),
    pytest.param(lambda t: (t.cached(3, 0), t.q(t.subsection(0) + 0x10, STALE_PTE)), PageOrigin.FILE_UNCONFIRMED,
                 id="control-area-unreadable"),
])
def test_only_data_file_pages_are_named(toy, setup, origin):
    toy.file(PATH_A)
    setup(toy)
    assert toy.support.origin(3) == origin
    assert toy.support.copy_origin([3], 10) == {"pages": {str(origin): 1}, "bytes": 10}


def _break(name: str):
    """Corrupts file 0's name chain in one way."""
    def apply(t: Toy) -> None:
        fo, ca, buf = t.file_object(0), t.control_area(0), t.name_buffer(0)
        encoded = PATH_A.encode("utf-16-le")
        if name == "no-file-pointer":
            t.q(ca + 0x48, 0)
        elif name == "file-object-in-user-half":
            t.map(0x7FF0_0000_0000)
            t.put(0x7FF0_0000_0040, struct.pack("<HH", len(encoded), len(encoded)))
            t.q(0x7FF0_0000_0048, buf)
            t.q(ca + 0x48, 0x7FF0_0000_0000)
        elif name == "file-object-unreadable":
            t.q(ca + 0x48, STALE_PTE)
        elif name == "empty-name":
            t.put(fo + 0x40, struct.pack("<HH", 0, 40))
        elif name == "odd-length":
            t.put(fo + 0x40, struct.pack("<HH", len(encoded) - 1, 200))
        elif name == "length-above-maximum":
            t.put(fo + 0x40, struct.pack("<HH", len(encoded), len(encoded) - 2))
        elif name == "length-above-4096":
            t.put(buf, "a".encode("utf-16-le") * 2049)  # printable, readable, and 2 bytes over the cap
            t.put(fo + 0x40, struct.pack("<HH", 4098, 8000))
        elif name == "null-buffer":
            t.q(fo + 0x48, 0)
        elif name == "buffer-in-user-half":
            t.map(0x7FF0_0001_0000)
            t.put(0x7FF0_0001_0000, encoded)
            t.q(fo + 0x48, 0x7FF0_0001_0000)
        elif name == "buffer-unreadable":
            t.q(fo + 0x48, STALE_PTE)
        elif name == "buffer-runs-into-an-unmapped-page":
            at = POOL + 7 * PAGE + PAGE - 10  # the last mapped pool page: the next one is not mapped
            t.put(at, encoded[:10])
            t.q(fo + 0x48, at)
        elif name == "lone-surrogate":
            t.put(buf, b"\x00\xd8" + encoded[2:])
        elif name == "embedded-nul":
            t.put(buf, encoded[:4] + b"\x00\x00" + encoded[6:])
        elif name == "control-character":
            t.put(buf, "\\toy\ncache".encode("utf-16-le"))
            t.put(fo + 0x40, struct.pack("<HH", 20, 40))
        else:
            raise AssertionError(name)
    return apply


@pytest.mark.parametrize("name", [
    "no-file-pointer", "file-object-in-user-half", "file-object-unreadable", "empty-name", "odd-length",
    "length-above-maximum", "length-above-4096", "null-buffer", "buffer-in-user-half", "buffer-unreadable",
    "buffer-runs-into-an-unmapped-page", "lone-surrogate", "embedded-nul", "control-character",
])
def test_a_name_chain_that_does_not_read_back_cleanly_names_nothing(toy, name):
    toy.file(PATH_A)
    toy.cached(3, 0)
    assert toy.names([3]) == [PATH_A]
    _break(name)(toy)
    assert toy.support.copy_origin([3], 100) == {"pages": {"FILE_DATA": 1}, "bytes": 100}


def test_name_buffer_across_a_page_boundary_and_fast_reference_bits_are_read_exactly(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    encoded = PATH_A.encode("utf-16-le")
    at = POOL + 3 * PAGE + PAGE - 10  # the buffer starts ten bytes before the end of a mapped page
    toy.put(at, encoded)  # the next pool page is mapped (to another frame), so the name is whole
    toy.q(toy.file_object(a) + 0x48, at)
    toy.q(toy.control_area(a) + 0x48, toy.file_object(a) | 0xF)  # all four reference bits set
    assert toy.names([3]) == [PATH_A]


def test_a_mixed_copy_names_only_the_pages_that_verify(toy):
    a, b, c = toy.file(PATH_B), toy.file(PATH_A), toy.file(PATH_C)
    toy.cached(3, a)
    toy.cached(4, b)
    toy.cached(5, c)
    toy.q(toy.slot(5), 40 * PAGE | VALID)  # frame 5's prototype names a reused frame: file C is not credited
    toy.entry(6, location=ACTIVE, shared=False)  # a private page next to them
    toy.entry(7, original=0x4_0000_0000 | 0x80)  # a shared-memory page
    result = toy.support.copy_origin([3, 4, 5, 6, 7], 20000)
    assert result == {"pages": {"FILE_DATA": 3, "PRIVATE": 1, "SHARED_MEMORY": 1}, "bytes": 20000,
                      "backing_file_names": [PATH_A, PATH_B]}  # sorted; C was never named
    assert toy.names([5]) is None
    assert toy.names([5, 6, 7]) is None


def test_a_page_cannot_borrow_the_name_of_another_pages_file(toy):
    a, b = toy.file(PATH_A), toy.file(PATH_B)
    toy.cached(3, a)
    toy.cached(4, b)
    # Frame 3's entry is torn: its PteAddress now names frame 4's prototype PTE, its OriginalPte still names file A.
    toy.entry(3, original=toy.subsection_pte(a), pte_address=toy.slot(4))
    assert toy.names([3]) is None
    assert toy.names([4]) == [PATH_B]
    assert toy.names([3, 4]) == [PATH_B]


def test_boundary_the_pfn_entry_alone_decides_which_control_area_a_verified_page_names(toy):
    """Documents a limit, not a guarantee: the prototype PTE is checked against the frame, not against the
    subsection. An entry that names file B in OriginalPte and a prototype PTE that really holds the frame is
    credited to B. No symbol layout here ties a prototype PTE address to a subsection's range."""
    a, b = toy.file(PATH_A), toy.file(PATH_B)
    toy.cached(3, a)
    toy.entry(3, original=toy.subsection_pte(b), pte_address=toy.slot(3))
    assert toy.names([3]) == [PATH_B]


def test_one_control_area_is_named_once_and_a_later_verified_page_can_still_name_it(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    toy.cached(4, a)
    toy.cached(5, a)
    toy.q(toy.slot(3), 40 * PAGE | VALID)
    assert toy.names([3, 4, 5]) == [PATH_A]
    toy.q(toy.slot(4), 41 * PAGE | VALID)
    assert toy.names([3, 4, 5]) == [PATH_A]
    toy.q(toy.slot(5), 42 * PAGE | VALID)
    assert toy.names([3, 4, 5]) is None


def test_each_copy_of_a_record_keeps_its_own_names(toy):
    a, b = toy.file(PATH_A), toy.file(PATH_B)
    toy.cached(3, a)
    toy.cached(4, b)
    first, second = toy.support.copy_origin([3], 10), toy.support.copy_origin([4, 6], 10)
    assert first["backing_file_names"] == [PATH_A] and second["backing_file_names"] == [PATH_B]
    assert toy.support.copy_origin([1000], 10) == {"pages": {"UNKNOWN": 1}, "bytes": 10}  # past the PFN database


def test_a_frame_with_no_symbols_for_names_reports_no_names(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    plain = bitmaps.SectionLayout(16, 0x10, 0x38, file_bit=FILE_BIT, image_bit=IMAGE_BIT)
    support = bitmaps.PfnSupport(toy.support.kernel, PFN_DB, ENTRY, 8, PTE_BASE, 34, (40, 63), plain)
    assert support.copy_origin([3], 10) == {"pages": {"FILE_DATA": 1}, "bytes": 10}
    no_section = bitmaps.PfnSupport(toy.support.kernel, PFN_DB, ENTRY, 8, PTE_BASE, 34, (40, 63))
    assert no_section.copy_origin([3], 10) == {"pages": {"UNKNOWN": 1}, "bytes": 10}


def test_names_never_change_the_memory_source_of_a_record(toy):
    a = toy.file(PATH_A)
    toy.cached(3, a)
    toy.cached(4, a)
    toy.entry(6, location=ACTIVE, shared=False)
    named_file = toy.support.copy_origin([3, 4], 8192)
    named_mixed = toy.support.copy_origin([3, 6], 8192)
    private = toy.support.copy_origin([6], 8192)
    for origin in (named_file, named_mixed):
        assert "backing_file_names" in origin
        bare = {key: value for key, value in origin.items() if key != "backing_file_names"}
        assert evidence.memory_source([origin], 8192) == evidence.memory_source([bare], 8192)
    assert evidence.memory_source([named_file], 8192) == "FILE_BACKED"
    assert evidence.memory_source([named_mixed], 8192) == "MIXED"
    assert evidence.memory_source([named_mixed, private], 8192) == "MEMORY"
    assert evidence.memory_source([named_file, private], 8192) == "MEMORY"


# --- the names never reach linking, output positions or the scorecard ---------------------------------------------

PROMPT_ID = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"
OUTPUTS = ["toy_00001_.png", "toy_00002_.png"]


def _linked_job(origins: list[dict]):
    """A carved PNG whose embedded prompt equals a recovered job's graph, and that job's queue item."""
    graph = api_prompt(batch_size=2)
    png = png_bytes(noise_image(64, 48, seed=3), {"prompt": json.dumps(graph)}, idat_chunk=512)
    [image] = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(png)
    [workflow] = WorkflowCarver().carve_stream(prompt_request_body(graph, prompt_id=PROMPT_ID),
                                               base_physical_offset=0x100000)
    workflow.evidence["prompt_id"] = PROMPT_ID
    image.page_origins = copy.deepcopy(origins)
    item = QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, graph)
    item.history = {"outputs": {"9": {"images": [{"filename": name} for name in OUTPUTS]}}}
    return image, workflow, item


def _origins(*names: str) -> list[dict]:
    origin: dict = {"pages": {"FILE_DATA": 3}, "bytes": 11000}
    if names:
        origin["backing_file_names"] = list(names)
    return [origin]


def test_a_backing_name_equal_to_an_output_filename_awards_no_output_position():
    named_origin = _origins(rf"\toy\output\{OUTPUTS[1]}", rf"\toy\output\{OUTPUTS[0]}")
    cards = {}
    for label, origins in (("named", named_origin), ("bare", _origins())):
        image, workflow, item = _linked_job(origins)
        result = correlate([image], [workflow])
        assert [link.link_class for link in result.links] == ["STRICT"]
        cards[label] = (evidence.scorecard([item], [workflow], "ab" * 32, result.links, [image]),
                        [link.to_dict() for link in result.links], [f.to_dict() for f in result.findings])
    assert cards["named"] == cards["bare"]  # names change nothing: links, findings, rows
    card = cards["named"][0]
    assert [(r["output_filename"], r["batch_index"]) for r in card["rows"]] == list(zip(OUTPUTS, (0, 1), strict=True))
    for row in card["rows"]:
        assert row["image"]["status"] == "JOB_LINKED"
        [candidate] = row["image"]["candidates"]
        assert candidate["role"] == "JOB" and candidate["memory_source"] == "FILE_BACKED"
        assert "batch_index" not in candidate
    assert card["summary"]["outputs_with_linked_image"] == 0
    assert "toy\\\\output" not in json.dumps(card) and "backing" not in json.dumps(card)


def test_a_backing_name_alone_does_not_link_an_image_to_a_job():
    image, workflow, item = _linked_job(_origins(rf"\toy\output\{OUTPUTS[0]}"))
    card = evidence.scorecard([item], [workflow], "ab" * 32, (), [image])  # no correlation link at all
    for row in card["rows"]:
        assert row["image"] == {"status": "NOT_LINKED", "candidates": []}


def test_the_evidence_document_only_carries_names_as_page_origin_context():
    named, workflow, _ = _linked_job(_origins(rf"\toy\output\{OUTPUTS[0]}"))
    bare = copy.deepcopy(named)
    bare.page_origins = _origins()
    tool = {"name": "toy"}
    with_names = evidence.evidence_document([named], [workflow], tool)
    without = evidence.evidence_document([bare], [workflow], tool)
    [record] = (r for r in with_names["records"] if r["kind"] == "image")
    assert record["page_origins"][0]["backing_file_names"] == [rf"\toy\output\{OUTPUTS[0]}"]
    assert record["memory_source"] == "FILE_BACKED"
    for entry in with_names["records"]:
        for origin in entry["page_origins"]:
            origin.pop("backing_file_names", None)
    assert with_names == without


def test_report_shows_backing_names_as_page_context_and_escapes_them():
    from comfytrace.workbench import render_workbench_html
    from tests.unit.test_report import _unit_rep

    image, workflow, _ = _linked_job(_origins(r"\toy\output\copy&variant.png"))
    result = correlate([image], [workflow])
    ev = evidence.evidence_document([image], [workflow], {"name": "toy"})
    rep, data, _ = _unit_rep(
        [f.to_dict() for f in result.findings], [workflow.to_dict()], [image.to_dict()],
        [link.to_dict() for link in result.links], evidence=ev,
    )
    assert rep.records[image.artifact_id]["page_origins"][0]["backing_file_names"] == [r"\toy\output\copy&variant.png"]
    [finding] = data["findings"]
    [shown_image] = finding["images"]
    assert shown_image["source"] == "only on file-backed pages (a file may stand behind it)"
    assert shown_image["link"] == "STRICT" and shown_image["file"] is None
    assert "copy&variant.png" not in json.dumps(finding)
    escaped_page = render_workbench_html({**data, "case": {**data["case"], "caseId": "case&<1>"}})
    assert "case&amp;&lt;1&gt;" in escaped_page.split("<body>")[0]
    assert "case&<1>" not in escaped_page.split("<body>")[0]
