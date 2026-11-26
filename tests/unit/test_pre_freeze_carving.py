"""Pre-freeze regression tests for confirmed bugs in carving, media, bitmaps,
fragments, outputs, pageorder, metadata, correlate, and evidence."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from comfytrace import bitmaps, evidence, media
from comfytrace.carve import (
    Ihdr,
    PngCarver,
    ProcessLayerSource,
    WorkflowCarver,
    _decode_text_chunk,
    execution_parameters,
    graph_is_complete,
    parse_png,
)
from comfytrace.correlate import CACHE_KEY, KeyMatcher, correlate, resolve_links
from comfytrace.fragments import _agrees, ui_parameters, ui_workflow_nodes
from comfytrace.metadata import resolve_workflow
from comfytrace.models import CorrelationLink, GapDescriptor, GapReason, LinkClass, PageOrigin, WorkflowRecord
from comfytrace.outputs import OutputFileCarver
from comfytrace.pageorder import page_order_stream, stitch_text_chunks, torn_text_chunk
from tests.synthetic import PAGE, FakeProcLayer, api_prompt, noise_image, png_bytes, prompt_request_body
from tests.unit.test_bitmaps import ACTIVE, Toy
from tests.unit.test_media import fixture as media_fixture
from tests.unit.test_metadata import api_graph as metadata_api_graph
from tests.unit.test_metadata import workflow as metadata_workflow
from tests.unit.test_pageorder import _png_with_workflow

# ---------------------------------------------------------------------------
# carve.py


class _PagedOutLayer(FakeProcLayer):
    """Fake process layer where specific virtual pages raise SwappedInvalidAddressException."""

    def __init__(self, physical: bytes, swapped_pages: set[int]) -> None:
        super().__init__(physical, {})
        self.swapped_pages = swapped_pages

    def read(self, offset: int, length: int, pad: bool = False) -> bytes:
        from volatility3.framework.exceptions import SwappedInvalidAddressException

        page = offset // PAGE
        end_page = (offset + length - 1) // PAGE
        for current in range(page, end_page + 1):
            if current in self.swapped_pages:
                raise SwappedInvalidAddressException(
                    "proc", current * PAGE, 0, current * PAGE, "swapped out"
                )
        return super().read(offset, length, pad=pad)


def test_parse_png_preserves_trailing_paged_out_gap_at_available_end():
    png = png_bytes(noise_image(100, 100, seed=1), {"prompt_id": "p1"}, idat_chunk=50000)
    count = (len(png) + PAGE - 1) // PAGE
    assert count >= 4
    base = 0x20000000
    swapped = {base // PAGE + count - 2, base // PAGE + count - 1}
    layer = _PagedOutLayer(png.ljust(count * PAGE, b"\x00"), swapped)
    layer.map_virtual(base, list(range(count)))

    [record] = PngCarver().carve_stream(
        png[:PAGE], virtual_address=base, process_id=42, source=ProcessLayerSource(layer)
    )
    assert record.completeness == "PARTIAL"
    assert record.validation["has_gaps"] is True
    assert len(record.gaps) == 1
    assert record.gaps[0].reason == GapReason.PAGED_OUT
    assert record.gaps[0].relative_offset == (count - 2) * PAGE
    assert record.gaps[0].length_bytes > 0


def test_parse_png_preserves_both_interior_hole_and_trailing_gap():
    png = png_bytes(noise_image(100, 100, seed=2), {}, idat_chunk=50000)
    spans = [(0, png[:PAGE]), (2 * PAGE, png[2 * PAGE : 3 * PAGE])]
    gaps = [
        GapDescriptor(PAGE, PAGE, GapReason.PAGE_NOT_PRESENT),
        GapDescriptor(3 * PAGE, PAGE, GapReason.PAGED_OUT),
    ]
    parsed = parse_png(b"", False, GapReason.PAGE_NOT_PRESENT, 0, spans=spans, gaps=gaps)
    assert parsed is not None and not parsed.complete
    assert [(g.relative_offset, str(g.reason)) for g in parsed.gaps] == [
        (PAGE, "PAGE_NOT_PRESENT"),
        (3 * PAGE, "PAGED_OUT"),
    ]
    assert parsed.gaps[1].length_bytes > 0


def test_ihdr_and_compressed_text_validation():
    bad_comp = struct.pack(">IIBBBBB", 16, 16, 8, 2, 1, 0, 0)
    bad_filt = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 1, 0)
    assert Ihdr.parse(bad_comp) is None
    assert Ihdr.parse(bad_filt) is None

    full_z = zlib.compress(b'{"3":{"class_type":"KSampler","inputs":{}}}')
    assert _decode_text_chunk(b"zTXt", b"prompt\x00\x00" + full_z) is not None
    # Truncated zlib stream (missing ADLER32 / end-of-stream) must fail.
    assert _decode_text_chunk(b"zTXt", b"prompt\x00\x00" + full_z[:-6]) is None
    # Empty compressed payload must fail.
    assert _decode_text_chunk(b"zTXt", b"prompt\x00\x00") is None
    # Non-zero compression method must fail.
    assert _decode_text_chunk(b"zTXt", b"prompt\x00\x01" + full_z) is None
    # Missing NUL separator must fail.
    assert _decode_text_chunk(b"tEXt", b"prompt_no_nul") is None
    assert _decode_text_chunk(b"iTXt", b"prompt\x00\x01\x00en\x00\x00" + full_z[:-6]) is None
    assert _decode_text_chunk(b"iTXt", b"prompt\x00\x01\x00en\x00\x00" + full_z) is not None


def test_json_string_draft_rejects_lone_surrogate_in_runtime_fields():
    body = prompt_request_body(api_prompt(seed=7), prompt_id="\ud800").decode("utf-8")
    escaped = json.dumps({"data": body}).encode("utf-8")
    assert WorkflowCarver().carve_stream(b"\x00" + escaped + b"\x00") == []


def test_bool_is_not_accepted_as_int_in_graph_parameters_or_links():
    graph = api_prompt()
    graph["5"]["inputs"]["width"] = True
    graph["5"]["inputs"]["height"] = False
    params = execution_parameters(graph, {})
    assert "width" not in params and "height" not in params
    # A list [node_id, True] is not a valid integer socket link (neither for _resolve_text nor graph_is_complete).
    graph_bool_text = api_prompt()
    graph_bool_text["3"]["inputs"]["positive"] = ["6", True]
    assert "positive_prompt" not in execution_parameters(graph_bool_text, {})
    graph_bool_extra = api_prompt()
    graph_bool_extra["3"]["inputs"]["extra_flag"] = ["999", True]
    assert graph_is_complete("API_GRAPH", graph_bool_extra, execution_parameters(graph_bool_extra, {})) is True


# ---------------------------------------------------------------------------
# media.py


def _find_flac_frame_boundaries(flac: bytes) -> list[int]:
    pos = 4
    while True:
        last = bool(flac[pos] & 0x80)
        length = int.from_bytes(flac[pos + 1 : pos + 4], "big")
        pos += 4 + length
        if last:
            break
    starts = [pos]
    variable = flac[pos + 1] & 1
    first = media._flac_frame_header(flac, pos, variable)
    assert first is not None
    at = flac.find(b"\xff" + bytes((0xF8 | variable,)), pos + first[0] + 2)
    while at != -1:
        if media._flac_frame_header(flac, at, variable) is not None and media._crc16(
            flac[pos : at - 2]
        ) == int.from_bytes(flac[at - 2 : at], "big"):
            starts.append(at)
            pos = at
        at = flac.find(b"\xff" + bytes((0xF8 | variable,)), at + 1)
    return starts


def test_parse_flac_non_final_truncated_frame_ignores_accidental_crc16_match():
    flac = media_fixture("audio.flac")
    starts = _find_flac_frame_boundaries(flac)
    assert len(starts) >= 3
    # Build a truncated prefix cutting inside frame 0 (non-final), and plant a matching CRC-16 inside it.
    frame0_start, frame1_start = starts[0], starts[1]
    assert frame1_start - frame0_start > 40
    cut = frame0_start + 32
    tampered = bytearray(flac[:cut])
    crc = media._crc16(bytes(tampered[frame0_start : cut - 2]))
    tampered[cut - 2] = crc >> 8
    tampered[cut - 1] = crc & 0xFF

    parsed = media.parse_flac(bytes(tampered))
    assert parsed is not None
    assert not parsed.complete
    assert parsed.halt == "FRAME_CRC_FAILURE"
    assert parsed.end == frame0_start
    assert parsed.more is True
    assert parsed.validation["frames_verified"] == 0


def test_media_record_page_crossing_at_exact_page_boundary():
    webp = media_fixture("still.webp")
    # Tear right after the VP8L chunk ends, aligned to an exact page edge (last verified byte at 2*PAGE - 1).
    exif = webp.index(b"EXIF")
    head = webp[:exif]
    mem = bytearray(4 * PAGE)
    start = 2 * PAGE - len(head)
    mem[start : 2 * PAGE] = head
    mem[2 * PAGE :] = b"\xaa" * (2 * PAGE)
    parsed = media.parse_webp(bytes(mem[start:]))
    assert parsed is not None and not parsed.complete and start + parsed.end == 2 * PAGE
    candidate = media.MediaCandidate(parsed, start, False, None, True, head, None)
    record = media.MediaCarver(physical_mode=True).record(candidate)
    assert record is not None and record.gaps[0].reason == GapReason.NONCONTIGUOUS_PHYSICAL

    # Conversely, a 0-byte missing gap (walked full container, failed pixel decode) at a page edge
    # must NOT be classified as NONCONTIGUOUS_PHYSICAL.
    broken = bytearray(webp)
    at = broken.index(b"VP8L") + 14
    broken[at : at + 24] = bytes(24)
    mem2 = bytearray(4 * PAGE)
    start2 = 2 * PAGE - len(broken)
    mem2[start2 : 2 * PAGE] = broken
    parsed2 = media.parse_webp(bytes(broken))
    assert parsed2 is not None and not parsed2.complete and parsed2.needed == parsed2.end
    cand2 = media.MediaCandidate(parsed2, start2, False, None, True, bytes(broken), None)
    rec2 = media.MediaCarver(physical_mode=True).record(cand2)
    assert rec2 is not None and rec2.gaps[0].reason == GapReason.TRUNCATED_STREAM


# ---------------------------------------------------------------------------
# bitmaps.py & pageorder.py


def test_table_frame_never_reads_past_page_directory_boundary():
    # Two frames: frame 0 is a page directory whose last slot (511) is valid,
    # frame 1 sits immediately after it and happens to have valid-looking qwords at slot 0.
    phys = bytearray(4 * PAGE)
    phys[511 * 8 : 512 * 8] = struct.pack("<Q", (2 * PAGE) | 1)
    phys[PAGE : PAGE + 8] = struct.pack("<Q", (3 * PAGE) | 1)  # frame 1 offset 0 (slot 512!)
    assert bitmaps._table_frame(phys, 0, 511) == 2
    assert bitmaps._table_frame(phys, 0, 512) is None
    assert bitmaps._table_frame(phys, 0, -1) is None
    assert bitmaps._table_frame(phys, 10, 0) is None
    assert bitmaps._slot_frame(phys, 0, 512 * 512) is None

    # page_order_stream starting at the very last slot of a page directory must stop cleanly.
    phys[2 * PAGE + 511 * 8 : 2 * PAGE + 512 * 8] = struct.pack("<Q", (3 * PAGE) | 1)
    support = SimpleNamespace(
        origin=lambda _pfn: PageOrigin.PRIVATE,
        table_slot=lambda pfn: (2, 511) if pfn == 3 else ((0, 511) if pfn == 2 else None),
    )
    res = page_order_stream(phys, support, 3 * PAGE, 4 * PAGE)
    assert res is not None
    data, method, frames = res
    assert method == "PAGE_TABLE_ORDER" and frames == [3] and len(data) == PAGE


def test_ordered_regions_canonicalizes_prototype_pte_addresses():
    toy = Toy()
    # Map 16 contiguous prototype PTEs for frames 20..35; clear bit 63 and set low flag bits on half of them.
    pfns = list(range(20, 36))
    for i, pfn in enumerate(pfns):
        slot = toy.slot(pfn)
        toy.q(slot, (pfn * PAGE) | 1)
        raw_pte = ((slot & ~(1 << 63)) | 5) if i % 2 else slot
        toy.entry(pfn, location=ACTIVE, original=0x4_0000_0000 | 0x80, pte_address=raw_pte)
        toy.data[pfn * PAGE : (pfn + 1) * PAGE] = bytes([i, 10, 20, 255]) * (PAGE // 4)
    regions = list(bitmaps.ordered_regions(toy.data, pfns, toy.support, min_pages=16))
    assert len(regions) == 1
    assert regions[0][2] == pfns and regions[0][3] == "PROTOTYPE_PTE_ORDER"


def test_cut_images_gap_at_window_start_does_not_taint_previous_window():
    # Window 0: non-photo noise (width 0 / different). Window 1 (pages 16..31): clean 256px photo.
    # Window 2 (pages 32..47): gap on page 32 (exact start of window 2), continuing the 256px photo.
    rng = np.random.default_rng(123)
    width = 256
    rows = 128  # 128 * 256 * 4 = 131072 = 32 pages (windows 1 and 2)
    img = np.zeros((rows, width, 4), dtype=np.uint8)
    row = rng.integers(20, 230, (width, 3), dtype=np.uint8)
    for r in range(rows):
        row = np.clip(row.astype(np.int16) + rng.integers(-2, 3, (width, 3)), 0, 255).astype(np.uint8)
        img[r, :, :3] = row
        img[r, :, 3] = 255
    # Window 0: 16 pages of 512px-stride smooth image so window 0 measures width=512 (not 256).
    w0_rows = 32  # 32 * 512 * 4 = 65536 = 16 pages
    w0 = np.zeros((w0_rows, 512, 4), dtype=np.uint8)
    row0 = rng.integers(20, 230, (512, 3), dtype=np.uint8)
    for r in range(w0_rows):
        row0 = np.clip(row0.astype(np.int16) + rng.integers(-2, 3, (512, 3)), 0, 255).astype(np.uint8)
        w0[r, :, :3] = row0
        w0[r, :, 3] = 255
    data = bytearray(w0.tobytes() + img.tobytes())
    # Put a gap at page 32 (the exact first page of window 2).
    data[32 * PAGE : 33 * PAGE] = bytes(PAGE)
    gaps = bitmaps.Gaps()
    gaps[32] = "PAGE_NOT_PRESENT"
    frames = [p if p != 32 else -1 for p in range(len(data) // PAGE)]
    found = bitmaps.cut_images("TEST", bytes(data), frames, "PAGE_TABLE_ORDER", gaps)
    assert any(bmp.width == 256 for bmp in found), [b.width for b in found]


@pytest.mark.parametrize("chunk", [PAGE + 123, PAGE // 2])
def test_stitch_text_chunks_handles_non_page_multiple_chunk_sizes_and_empty_input(tmp_path: Path, chunk: int):
    own = _png_with_workflow(111)
    cut = own.index(b"workflow\x00") + 900
    rest = own[cut:]
    dump = bytearray(10 * PAGE)
    start = 3 * PAGE - cut
    assert start >= 0
    dump[start : 3 * PAGE] = own[:cut]
    dump[6 * PAGE : 6 * PAGE + min(len(rest), PAGE)] = rest[:PAGE]
    image_path = tmp_path / "dump.raw"
    image_path.write_bytes(bytes(dump))

    assert stitch_text_chunks(image_path, [], chunk=chunk) == {}

    [record] = PngCarver(physical_mode=True).carve_stream(bytes(dump[start : 3 * PAGE]), base_physical_offset=start)
    torn = torn_text_chunk(record.artifact_id, record.spans[0][1])
    assert torn is not None
    found = stitch_text_chunks(image_path, [torn], chunk=chunk)
    assert found[record.artifact_id] == [rest[: torn.missing]]


# ---------------------------------------------------------------------------
# outputs.py


def test_outputs_handles_huge_etag_non_ascii_digits_and_virtual_no_etag():
    huge_etag_resp = (
        b'HTTP/1.1 200 OK\x00Content-Disposition: filename="a.png"\x00'
        b'Etag: "fffffffffffffffff-100"\x00Content-Length: \xb2\x00'
        b'Server: Python/3.13 aiohttp/3.14.3\x00\x00'
    )
    url_superscript = b"/view?filename=a.png&type=output&t=%C2%B2 "
    no_etag_1 = (
        b'HTTP/1.1 200 OK\x00Content-Disposition: filename="b.png"\x00'
        b'Content-Length: 100\x00Server: aiohttp/3.14.3\x00\x00'
    )
    no_etag_2 = (
        b'HTTP/1.1 200 OK\x00Content-Disposition: filename="b.png"\x00'
        b'Content-Length: 200\x00Server: aiohttp/3.14.3\x00\x00'
    )

    carver = OutputFileCarver()
    carver.carve_stream(huge_etag_resp + url_superscript, base_physical_offset=0x1000)
    carver.carve_stream(no_etag_1, virtual_address=0x10000, process_id=55)
    carver.carve_stream(no_etag_2, virtual_address=0x20000, process_id=55)

    a_entry = carver.files["a.png"].to_dict("OUT-001")
    assert a_entry["versions"][0]["size"] == 0x100
    assert a_entry["versions"][0]["content_length"] is None
    assert "file_mtime_utc" not in a_entry["versions"][0]

    b_entry = carver.files["b.png"].to_dict("OUT-002")
    assert sorted(v["content_length"] for v in b_entry["versions"]) == [100, 200]


# ---------------------------------------------------------------------------
# fragments.py


def test_fragments_agrees_rejects_int_vs_bool_and_handles_malformed_ui():
    assert _agrees(1, True) is False
    assert _agrees(0, False) is False
    assert _agrees(1.0, True) is False
    assert _agrees(True, True) is True
    assert _agrees(1, 1.0) is True

    # Non-dict widgets_values_named must not crash ui_parameters.
    assert ui_parameters([{"type": "CustomNode", "widgets_values_named": [1, 2, 3]}]) == {}

    # Deeply nested (>64) or NaN subgraph workflow must fall back cleanly instead of crashing.
    nested: dict = {"leaf": 1}
    for _ in range(70):
        nested = {"wrap": nested}
    ui = {
        "nodes": [{"id": 3, "type": "KSampler", "widgets_values": [42, "fixed", 20, 7, "euler", "normal", 1.0]}],
        "links": [],
        "definitions": {"subgraphs": [{"id": "sg1", "nodes": []}]},
        "extra": nested,
    }
    nodes = ui_workflow_nodes(ui)
    assert ui_parameters(nodes)["seed"] == 42


# ---------------------------------------------------------------------------
# metadata/resolver.py


def test_resolver_does_not_alias_model_name_and_loader_with_loras_on_broken_model():
    graph = metadata_api_graph()
    graph["3"]["inputs"]["model"] = ["42", 0]  # missing source node
    graph["4"]["inputs"]["clip"] = ["42", 1]  # also breaks CLIP LoRA walk
    record = resolve_workflow(metadata_workflow(json.dumps(graph).encode(), graph))
    assert record is not None
    fields = record["samplers"][0]["fields"]
    assert fields["model_name"] is not fields["loras"]
    assert fields["model_loader"] is not fields["loras"]
    assert "coverage" not in fields["model_name"]
    assert "coverage" not in fields["model_loader"]
    assert "unresolved_clip_branches" not in fields["model_name"]
    assert "unresolved_clip_branches" not in fields["model_loader"]
    assert fields["loras"].get("coverage") == "partial; upstream model unresolved"
    assert "unresolved_clip_branches" in fields["loras"]


# ---------------------------------------------------------------------------
# correlate.py


def test_correlate_conflicting_is_changed_cannot_earn_supported_without_prompt_id():
    wf_graph = api_prompt(seed=10)
    wf_graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "in.png"}, "is_changed": ["hashA"]}
    img_graph = api_prompt(seed=10)
    img_graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "in.png"}, "is_changed": ["hashB"]}

    [workflow] = WorkflowCarver().carve_stream(json.dumps(wf_graph).encode(), base_physical_offset=0x10000)
    [image] = PngCarver().carve_stream(png_bytes(noise_image(64, 48, seed=1), {"prompt": json.dumps(img_graph)}))
    link = KeyMatcher().evaluate_link(workflow, image)
    assert link.link_class == LinkClass.UNASSOCIATED
    assert "input revision contradicts" in link.confidence_notes


def test_resolve_links_eliminates_fully_cached_peer_before_deduping_submission_copies():
    graph_base = api_prompt(seed=55)
    graph_base["10"] = {"class_type": "LoadImage", "inputs": {"image": "in.png"}}
    graph_annotated = json.loads(json.dumps(graph_base))
    graph_annotated["10"]["is_changed"] = ["rev-1"]

    carver = WorkflowCarver()
    [p1_pre] = carver.carve_stream(prompt_request_body(graph_base, "P1"), base_physical_offset=0x10000)
    [p1_post] = carver.carve_stream(prompt_request_body(graph_annotated, "P1"), base_physical_offset=0x20000)
    [p2_cached] = carver.carve_stream(prompt_request_body(graph_annotated, "P2"), base_physical_offset=0x30000)
    p2_cached.execution_parameters["job_timeline"] = {"cached_nodes": list(graph_annotated.keys())}

    [image] = PngCarver().carve_stream(png_bytes(noise_image(64, 48, seed=9), {"prompt": json.dumps(graph_annotated)}))

    # Case 1: embedded prompt matches p1_post (CANONICAL_PROMPT_HASH), p1_pre (EXECUTION_GRAPH_MATCH),
    # and p2_cached (fully cached). After eliminating p2_cached, p1_post must win cleanly without AMBIGUOUS.
    result = correlate([image], [p1_pre, p1_post, p2_cached])
    assert [(link.workflow_id, link.link_class) for link in result.links] == [(p1_post.workflow_id, LinkClass.STRICT)]

    # Case 2: execution-cache links on p1_pre, p1_post, and p2_cached.
    # After eliminating p2_cached, _preferred_cache_copy must collapse p1_pre + p1_post to p1_post.
    candidates = [
        CorrelationLink("", wf.workflow_id, image.artifact_id, LinkClass.STRICT, [CACHE_KEY], "cache")
        for wf in (p1_pre, p1_post, p2_cached)
    ]
    resolved = resolve_links([image], candidates, [p1_pre, p1_post, p2_cached])
    assert [(link.workflow_id, link.link_class) for link in resolved] == [(p1_post.workflow_id, LinkClass.STRICT)]


# ---------------------------------------------------------------------------
# evidence.py


def test_evidence_document_does_not_mutate_record_observations():
    wf = WorkflowRecord(
        workflow_id="WFPY-001", completeness="COMPLETE", prompt_format="API_GRAPH",
        encoding_detected="CPYTHON_OBJECTS", source_process_id=10, virtual_address="0x1000",
        physical_offsets=[], canonical_json_sha256="aa" * 32, execution_parameters={},
        dependencies={"node_classes": [], "model_assets": []}, raw_file_path="",
        raw_bytes=b"{}", graph={},
        evidence={"origin": "OBSERVED", "observations": [{"holder_id": "python.queue_item"}]},
    )
    doc = evidence.evidence_document([], [wf], {"name": "test"})
    assert doc["records"][0]["observations"][0]["observation_id"] == "WFPY-001/obs-1"
    assert "observation_id" not in wf.evidence["observations"][0]
