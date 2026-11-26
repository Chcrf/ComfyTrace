import hashlib
import io
import json
import struct

import pytest
from PIL import Image

from comfytrace.carve import (
    UTF16_ANCHOR,
    GapTracker,
    Ihdr,
    PhysicalSource,
    PngCarver,
    ProcessLayerSource,
    StreamReader,
    WorkflowCarver,
    canonical_json,
)
from tests.synthetic import PAGE, FakeProcLayer, api_prompt, garbage, noise_image, png_bytes, prompt_request_body


def _ihdr(png: bytes) -> dict:
    width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", png[16:29])
    return Ihdr(width, height, depth, ctype, interlace).as_meta()


def test_png_carver_crc32_validation():
    img = noise_image(40, 30)
    png = png_bytes(img, {"prompt": json.dumps(api_prompt())})
    buffer = garbage(5000) + png + garbage(3000, seed=9)
    [record] = PngCarver().carve_stream(buffer, base_physical_offset=0x10000)
    assert record.completeness == "COMPLETE"
    assert record.container_sha256 == hashlib.sha256(png).hexdigest()
    assert record.pixel_sha256 == hashlib.sha256(img.tobytes()).hexdigest()
    assert record.pixel_metadata == {"mode": "RGB", "width": 40, "height": 30}
    assert record.physical_offsets == [f"0x{0x10000 + 5000:016X}"]
    assert record.validation == {"crc32_valid": True, "scanlines_valid": True, "has_gaps": False}
    assert json.loads(record.embedded_text["prompt"]) == api_prompt()
    assert record.spans == [(0, png)]

    corrupt = bytearray(png)
    first_idat = corrupt.index(b"IDAT")
    corrupt[first_idat + 40] ^= 0xFF
    [bad] = PngCarver().carve_stream(bytes(corrupt))
    assert bad.completeness == "PARTIAL"
    assert bad.validation["crc32_valid"] is False
    assert bad.gaps[0].reason == "CRC_FAILURE"
    assert bad.gaps[0].relative_offset == first_idat - 4
    assert bad.spans == [(0, bytes(corrupt[: first_idat - 4]))], "only verified bytes are kept, unpadded"


def test_png_magic_without_valid_ihdr_is_ignored():
    assert PngCarver().carve_stream(b"\x89PNG\r\n\x1a\n" + garbage(200)) == []


def test_png_partial_idat_decompressobj_defiltering():
    img = noise_image(64, 64, "RGBA", seed=4)
    png = png_bytes(img, idat_chunk=512)
    raw_pixels = img.tobytes()
    truncated = png[: len(png) // 2]
    [record] = PngCarver().carve_stream(truncated)
    meta = record.derived_preview_metadata
    assert record.completeness == "PARTIAL"
    assert record.gaps[0].reason == "CRC_FAILURE"
    assert 0 < meta["decoded_rows"] < 64 and meta["total_rows"] == 64
    stride = 64 * 4
    preview = Image.open(io.BytesIO(record.preview_png))
    assert preview.size == (64, meta["decoded_rows"])
    assert preview.tobytes() == raw_pixels[: meta["decoded_rows"] * stride], "defiltered rows equal the source"

    carver = PngCarver()
    idat = b"".join(
        png[i + 8 : i + 8 + struct.unpack(">I", png[i : i + 4])[0]]
        for i in range(8, len(png) - 12)
        if png[i + 4 : i + 8] == b"IDAT"
    )
    pixels, rows, halt = carver.decompress_partial_idat(idat[: len(idat) // 3], _ihdr(png))
    assert halt == "INCOMPLETE_ZLIB_STREAM"
    assert pixels == raw_pixels[: rows * stride]
    pixels, rows, halt = carver.decompress_partial_idat(idat, _ihdr(png))
    assert (rows, halt) == (64, None) and pixels == raw_pixels


@pytest.mark.parametrize("mode", ["L", "RGB", "LA", "P"])
def test_partial_preview_across_colour_types(mode):
    img = noise_image(33, 20, "L", seed=2).convert(mode) if mode != "L" else noise_image(33, 20, "L")
    png = png_bytes(img, idat_chunk=256)
    [record] = PngCarver().carve_stream(png[: int(len(png) * 0.7)])
    rows = record.derived_preview_metadata["decoded_rows"]
    assert rows > 0
    preview = Image.open(io.BytesIO(record.preview_png))
    assert preview.convert("RGBA").tobytes() == img.crop((0, 0, 33, rows)).convert("RGBA").tobytes()


def test_png_adam7_interlaced_partial_preview_unsupported():
    png = png_bytes(noise_image(16, 16), interlace=True)
    [record] = PngCarver().carve_stream(png[: len(png) - 30])
    assert record.completeness == "PARTIAL"
    assert record.preview_png is None
    assert record.derived_preview_metadata["scanline_halt_reason"] == "ADAM7_INTERLACED_PARTIAL_PREVIEW_UNSUPPORTED"


def test_workflow_carver_utf8_utf16le_dual_scanning():
    graph = api_prompt(seed=42, text="a lighthouse at dusk")
    utf8 = json.dumps(graph).encode("utf-8")
    utf16 = json.dumps(api_prompt(seed=43, text="neon city, rain")).encode("utf-16-le")
    assert UTF16_ANCHOR in utf16 and len(UTF16_ANCHOR) == 24
    head = garbage(3001) + utf8 + garbage(2000, seed=5)
    head += b"\x00" * (len(head) % 2)
    buffer = head + utf16 + garbage(500, seed=6)
    records = WorkflowCarver().carve_stream(buffer, base_physical_offset=0)
    by_enc = {r.encoding_detected: r for r in records}
    assert set(by_enc) == {"UTF-8", "UTF-16LE"}
    assert by_enc["UTF-8"].raw_bytes == utf8
    assert by_enc["UTF-8"].completeness == "COMPLETE"
    assert by_enc["UTF-8"].execution_parameters["seed"] == 42
    assert by_enc["UTF-8"].execution_parameters["positive_prompt"] == "a lighthouse at dusk"
    assert by_enc["UTF-8"].canonical_json_sha256 == hashlib.sha256(canonical_json(graph).encode()).hexdigest()
    assert by_enc["UTF-16LE"].raw_bytes == utf16
    assert by_enc["UTF-16LE"].execution_parameters["seed"] == 43


def test_workflow_carver_outermost_json_bounded_lookback():
    graph = api_prompt()
    body = prompt_request_body(graph)
    padding = b" " * (300 * 1024)
    wrapped = body[:1] + padding + body[1:]
    buffer = garbage(4096) + wrapped + garbage(4096, seed=8)
    [record] = WorkflowCarver().carve_stream(buffer)
    assert record.prompt_format == "API_GRAPH"
    assert record.graph == graph, "root graph recovered, not an inner node"
    assert record.execution_parameters["prompt_id"] == "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"
    assert record.raw_bytes == wrapped

    far = body[:1] + b" " * (1024 * 1024 + 10) + body[1:]
    [inner] = WorkflowCarver().carve_stream(garbage(100) + far)
    assert inner.graph == graph
    assert "prompt_id" not in inner.execution_parameters, "the wrapper beyond the 1 MB look-back is not reached"


def test_carves_identical_graphs_with_separate_ids_from_serialized_queue_items():
    graph = api_prompt()
    ids = ("d6c4a8f1-34e2-4f32-b7d1-08f23491ba02", "00000000-0000-0000-0000-000000000000")
    queue = [[index, prompt_id, graph, {}, ["9"]] for index, prompt_id in enumerate(ids)]
    response = json.dumps({"queue_running": [queue[0]], "queue_pending": [queue[1]]}).encode()

    records = WorkflowCarver().carve_stream(response)

    assert [w.execution_parameters.get("prompt_id") for w in records] == list(ids)
    assert all(w.completeness == "COMPLETE" for w in records)
    assert [json.loads(w.raw_bytes) for w in records] == queue


def test_carves_a_queue_item_stored_as_a_json_string():
    prompt_id = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"
    queue = [1, prompt_id, api_prompt(), {}, ["9"]]

    (record,) = WorkflowCarver().carve_stream(json.dumps(json.dumps(queue)).encode())

    assert record.completeness == "COMPLETE"
    assert record.execution_parameters["prompt_id"] == prompt_id
    assert record.execution_parameters["runtime_id_source"] == "SERIALIZED_QUEUE_ITEM"


def test_workflow_carver_recovers_sdxl_and_nested_conditioning_prompts():
    graph = api_prompt()
    graph["6"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {
        "clip": ["4", 1], "text_g": "still life", "text_l": "yellow kettle",
    }}
    graph["10"] = {"class_type": "ConditioningCombine", "inputs": {
        "conditioning_1": ["11", 0], "conditioning_2": ["12", 0],
    }}
    graph["11"] = {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["6", 0]}}
    graph["12"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "green cup", "clip": ["4", 1]}}
    graph["13"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["10", 0], "negative": ["7", 0],
    }}
    graph["3"]["inputs"].update(positive=["13", 0], negative=["13", 1])

    [workflow] = WorkflowCarver().carve_stream(prompt_request_body(graph))

    assert workflow.completeness == "COMPLETE"
    assert workflow.execution_parameters["positive_prompt"] == "still life\nyellow kettle\ngreen cup"
    assert workflow.execution_parameters["negative_prompt"] == "blurry, low quality, distorted"


def test_workflow_fragment_is_partial_and_inner_nodes_are_not_duplicated():
    node = {"class_type": "KSampler", "inputs": {"seed": 7, "steps": 30, "sampler_name": "dpmpp_2m"}}
    buffer = garbage(800) + json.dumps(node).encode() + garbage(800, seed=1)
    [fragment] = WorkflowCarver().carve_stream(buffer)
    assert fragment.completeness == "PARTIAL"
    assert fragment.prompt_format == "API_NODE_FRAGMENT"
    assert fragment.execution_parameters["seed"] == 7

    torn = json.dumps(api_prompt()).encode()
    records = WorkflowCarver().carve_stream(torn[: len(torn) - 40] + garbage(400))
    assert records and all(r.completeness == "PARTIAL" for r in records)


def test_identical_workflow_copies_merge_into_one_record():
    body = json.dumps(api_prompt()).encode()
    buffer = body + garbage(5000) + body
    records = WorkflowCarver().carve_stream(buffer, base_physical_offset=0x1000)
    assert len(records) == 1
    assert records[0].physical_offsets == ["0x0000000000001000", f"0x{0x1000 + len(body) + 5000:016X}"]


def test_workflow_spanning_window_boundary_is_read_through_source():
    body = json.dumps(api_prompt()).encode()
    data = garbage(10_000) + body + garbage(10_000, seed=4)
    split = 10_000 + len(body) // 2
    records = WorkflowCarver().carve_stream(data[split - 64 :], base_physical_offset=split - 64,
                                            source=PhysicalSource(data))
    assert [r.graph for r in records] == [api_prompt()]


def test_stream_reader_physical_mmap_fallback(tmp_path):
    img = noise_image(96, 96, seed=11)
    png = png_bytes(img, idat_chunk=900)
    assert len(png) > 3 * PAGE
    dump = bytearray(garbage(16 * PAGE))
    start = 2 * PAGE + 100
    dump[start : start + len(png)] = png
    tear = 4 * PAGE
    dump[tear : tear + PAGE] = garbage(PAGE, seed=99)
    path = tmp_path / "phys.raw"
    path.write_bytes(bytes(dump))

    windows = list(StreamReader(path).stream_physical(block_size=5 * PAGE, overlap=PAGE))
    assert windows[0][0] == 0 and windows[1][0] == 4 * PAGE
    assert b"".join(w[1][: 4 * PAGE] for w in windows[:-1]) + windows[-1][1] == bytes(dump)

    carver = PngCarver(physical_mode=True)
    records = []
    for offset, window in windows:
        records += carver.carve_stream(window, base_physical_offset=offset, source=PhysicalSource(bytes(dump)))
    [record] = records
    assert record.completeness == "PARTIAL"
    assert record.gaps[0].reason == "NONCONTIGUOUS_PHYSICAL"
    assert record.gaps[0].synthetic_padding_applied is False
    end = start + record.recovered_bytes
    assert end <= tear, "carving halts before the torn page; nothing is stitched"
    assert record.spans == [(0, bytes(dump[start:end]))]
    assert record.derived_preview_metadata["decoded_rows"] > 0


def test_png_split_by_unmapped_page_in_process_layer():
    img = noise_image(80, 80, seed=12)
    png = png_bytes(img, idat_chunk=700)
    frames = (len(png) + PAGE - 1) // PAGE
    physical = bytearray(garbage((frames + 4) * PAGE))
    order = list(range(frames + 3, 3, -1))[:frames]
    for i, frame in enumerate(order):
        physical[frame * PAGE : frame * PAGE + PAGE] = png[i * PAGE : (i + 1) * PAGE].ljust(PAGE, b"\x01")
    layer = FakeProcLayer(bytes(physical), {})
    base = 0x20000000
    layer.map_virtual(base, order)
    [complete] = PngCarver().carve_stream(layer.read(base, PAGE), virtual_address=base, process_id=1,
                                          source=ProcessLayerSource(layer))
    assert complete.completeness == "COMPLETE", "scattered physical frames are contiguous in virtual memory"

    del layer.page_table[base // PAGE + 2]
    [partial] = PngCarver().carve_stream(layer.read(base, PAGE), virtual_address=base, process_id=1,
                                         source=ProcessLayerSource(layer))
    assert partial.completeness == "PARTIAL"
    assert partial.gaps[0].reason == "PAGE_NOT_PRESENT"
    assert partial.derived_preview_metadata["scanline_halt_reason"] == "PAGE_GAP_ENCOUNTERED"
    assert all(pad is False for _, _, pad in layer.reads)


def test_gap_tracker_writes_unpadded_spans(tmp_path):
    spans = [(0, b"\x89PNG-genuine"), (8192, b"tail-bytes")]
    paths = GapTracker(tmp_path).export_raw_spans("CARVE-PNG-002", spans)
    assert paths == ["exports/raw/CARVE-PNG-002/span_000.bin", "exports/raw/CARVE-PNG-002/span_001.bin"]
    assert [(tmp_path / p).read_bytes() for p in paths] == [b"\x89PNG-genuine", b"tail-bytes"]


def test_workflow_at_vad_start_preceded_by_unmapped_memory():
    """Regression: extending a run leftwards must stop at the first unreadable page, not discard the run."""
    body = prompt_request_body(api_prompt())
    physical = bytearray(garbage(8 * PAGE))
    physical[3 * PAGE : 3 * PAGE + len(body)] = body
    layer = FakeProcLayer(bytes(physical), {})
    base = 0x213B8E20000
    layer.map_virtual(base, [3])
    source = ProcessLayerSource(layer)
    assert source.read_before(base, 1 << 20) == b""
    [record] = WorkflowCarver().carve_stream(layer.read(base, PAGE), virtual_address=base, process_id=1,
                                             source=source)
    assert record.graph == api_prompt()
    assert record.virtual_address == "0x00000213B8E20000"
