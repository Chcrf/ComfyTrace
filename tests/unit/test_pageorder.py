"""Supplementary layer: torn ComfyUI text chunks completed by a CRC-verified page."""

import json
import struct
import zlib

from comfytrace.carve import IdAllocator, PngCarver
from comfytrace.cli import _stitch_texts
from comfytrace.correlate import png_workflows
from comfytrace.pageorder import stitch_text_chunks, torn_text_chunk
from tests.synthetic import api_prompt, noise_image, png_bytes
from tests.unit.test_fragments import ui_workflow

PAGE = 4096


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _png_with_workflow(seed: int) -> bytes:
    """A ComfyUI-like PNG: prompt chunk, then a workflow chunk long enough to cross a page."""
    plain = png_bytes(noise_image(64, 48, seed=seed), {"prompt": json.dumps(api_prompt(seed=seed))})
    workflow = ui_workflow(seed)
    workflow["extra"]["padding"] = "x" * 2600  # same length for every seed: heads differ only in the seed
    iend = plain.rindex(b"IEND") - 4
    return plain[:iend] + _chunk(b"tEXt", b"workflow\x00" + json.dumps(workflow).encode()) + plain[iend:]


def test_a_workflow_chunk_cut_at_a_page_edge_is_completed_by_the_page_that_checksums(tmp_path):
    own, other = _png_with_workflow(111), _png_with_workflow(222)
    cut = own.index(b"workflow\x00") + 900  # the page edge falls inside the workflow chunk
    rest_own = own[cut:]
    # The dump: the torn copy ends at a page edge; its true continuation and another job's continuation (a
    # different seed, so a different chunk and CRC) lie on pages elsewhere.
    dump = bytearray(10 * PAGE)
    start = 3 * PAGE - cut
    dump[start : 3 * PAGE] = own[:cut]
    dump[6 * PAGE : 6 * PAGE + min(len(rest_own), PAGE)] = rest_own[:PAGE]
    other_cut = other.index(b"workflow\x00") + 900
    dump[8 * PAGE : 8 * PAGE + min(len(other) - other_cut, PAGE)] = other[other_cut : other_cut + PAGE]
    image_path = tmp_path / "dump.raw"
    image_path.write_bytes(bytes(dump))

    [record] = PngCarver(IdAllocator("CARVE-PNG"), physical_mode=True).carve_stream(
        bytes(dump[start : 3 * PAGE]), base_physical_offset=start)
    torn = torn_text_chunk(record.artifact_id, record.spans[0][1])
    assert torn is not None and torn.ctype == b"tEXt" and torn.missing > 0
    found = stitch_text_chunks(image_path, [torn], chunk=3 * PAGE)
    # The heads are identical (the seed lies past the cut), so the other job's tail checksums too: the seed decides.
    assert sorted(found[record.artifact_id]) == sorted([rest_own[: torn.missing], other[other_cut:][: torn.missing]])
    _stitch_texts([record], image_path, dump)
    assert json.loads(record.stitched_text["workflow"])["nodes"][1]["widgets_values"][0] == 111
    assert record.evidence["stitched_text"]["crc32_verified"] is True
    _, workflow = png_workflows([record], IdAllocator("WFPNG"))
    assert workflow.encoding_detected == "PNG_TEXT_STITCHED" and workflow.prompt_format == "UI_WORKFLOW"


def test_a_torn_chunk_is_stitched_when_the_carver_saw_the_whole_dump(tmp_path):
    """As on a real dump: the carver reads on into the next physical page, the chunk's CRC fails there, and the
    record keeps only the chunks before it. The torn chunk's head is read again from the same page."""
    own = _png_with_workflow(111)
    cut = own.index(b"workflow\x00") + 900
    dump = bytearray(10 * PAGE)
    start = 3 * PAGE - cut
    dump[start : 3 * PAGE] = own[:cut]
    dump[3 * PAGE : 4 * PAGE] = b"\x55" * PAGE  # another owner's page follows physically
    dump[6 * PAGE : 6 * PAGE + min(len(own) - cut, PAGE)] = own[cut : cut + PAGE]
    image_path = tmp_path / "dump.raw"
    image_path.write_bytes(bytes(dump))
    [record] = PngCarver(IdAllocator("CARVE-PNG"), physical_mode=True).carve_stream(bytes(dump), base_physical_offset=0)
    assert record.gaps[0].reason == "NONCONTIGUOUS_PHYSICAL"
    assert len(record.spans[0][1]) < own.index(b"workflow\x00") - 8 + 1, "only the chunks before the torn one are kept"
    assert torn_text_chunk(record.artifact_id, record.spans[0][1]) is None
    _stitch_texts([record], image_path, dump)
    assert json.loads(record.stitched_text["workflow"])["nodes"][1]["widgets_values"][0] == 111
    assert record.evidence["stitched_text"]["crc32_verified"] is True
    assert len(record.spans[0][1]) < cut, "the record's own bytes are unchanged"


def test_a_stitched_workflow_is_exported_as_derived_material_only(tmp_path):
    from comfytrace.report import EvidentiaryExporter

    record = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(_png_with_workflow(5))[0]
    record.stitched_text, record.embedded_text = {"workflow": record.embedded_text.pop("workflow")}, \
        dict(record.embedded_text)
    [_, workflow] = png_workflows([record], IdAllocator("WFPNG"))
    path = EvidentiaryExporter(tmp_path).export_workflow(workflow)
    assert path.startswith("exports/derived/DERIVED_stitched_")
    assert not list((tmp_path / "exports" / "raw").glob("*.json"))
