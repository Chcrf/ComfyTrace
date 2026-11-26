"""Output identity inside one job: complete containers differ by their whole bytes, never by a shared prefix.

All inputs are synthetic: tiny benign PNGs built here (stored zlib blocks so that two outputs can share a long
encoded prefix and differ later)."""

import json
import struct
import zlib

from comfytrace.carve import IdAllocator, PngCarver, WorkflowCarver
from comfytrace.correlate import _has_encoded_pixels, correlate
from tests.synthetic import api_prompt

WIDTH, HEIGHT = 64, 48
ROW = 1 + WIDTH * 3
SHARED_ROWS = 10  # 1930 raw bytes, past the first KiB of the encoded stream


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def stored_png(graph: dict, tail_value: int, *, pixel_tail: int | None = None, idat_chunk: int | None = None,
               animated: bool = False) -> bytes:
    """A valid RGB PNG with stored (level 0) zlib blocks: its first SHARED_ROWS rows are identical for every
    value of ``tail_value``; later rows differ. ``pixel_tail`` sets the pixels of the later rows apart from the
    bytes (a different container with the same pixels needs a different zlib framing instead)."""
    rows = bytearray()
    for y in range(HEIGHT):
        value = 7 if y < SHARED_ROWS else (tail_value if pixel_tail is None else pixel_tail)
        rows += b"\x00" + bytes([value]) * (WIDTH * 3)
    deflater = zlib.compressobj(0)
    stream = deflater.compress(bytes(rows)) + deflater.flush()
    pieces = [stream] if idat_chunk is None else [stream[i:i + idat_chunk] for i in range(0, len(stream), idat_chunk)]
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
    out += _chunk(b"tEXt", b"prompt\x00" + json.dumps(graph).encode())
    if animated:
        # A valid one-frame APNG: the default image is frame 0, so its fcTL precedes the IDAT.
        out += _chunk(b"acTL", struct.pack(">II", 1, 0))
        out += _chunk(b"fcTL", struct.pack(">IIIIIHHBB", 0, WIDTH, HEIGHT, 0, 0, 1, 1, 0, 0))
    for piece in pieces:
        out += _chunk(b"IDAT", piece)
    out += _chunk(b"IEND", b"")
    return bytes(out)


def carve(data: bytes, offset: int, carver: PngCarver):
    return carver.carve_stream(data, base_physical_offset=offset)


def workflow_for(graph: dict):
    [record] = WorkflowCarver().carve_stream(json.dumps(graph).encode(), base_physical_offset=0x900000)
    return record


def findings_of(result, *images):
    return [sorted(a for a in f.artifact_ids if a.startswith("CARVE-PNG")) for f in result.findings
            if any(i.artifact_id in f.artifact_ids for i in images)]


def test_two_complete_outputs_sharing_their_first_kib_are_two_findings():
    graph = api_prompt(seed=41, batch_size=2)
    first, second = stored_png(graph, 100), stored_png(graph, 200)
    assert first[:2200] == second[:2200] and first != second  # the premise: a shared encoded prefix
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(first, 0x10000, carver)
    [b] = carve(second, 0x40000, carver)
    assert a.completeness == b.completeness == "COMPLETE" and a.container_sha256 != b.container_sha256
    result = correlate([a, b], [workflow_for(graph)])
    assert [f.tier for f in result.findings] == ["E2", "E2"]
    assert sorted(findings_of(result, a, b)) == sorted([[a.artifact_id], [b.artifact_id]])


def test_identical_complete_copies_stay_one_output():
    graph = api_prompt(seed=42)
    data = stored_png(graph, 100)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(data, 0x10000, carver)
    [b] = carve(data, 0x40000, carver)
    result = correlate([a, b], [workflow_for(graph)])
    assert findings_of(result, a, b) == [[a.artifact_id, b.artifact_id]]


def test_a_partial_joins_the_one_complete_container_its_verified_bytes_agree_with():
    graph = api_prompt(seed=43, batch_size=2)
    first, second = stored_png(graph, 100, idat_chunk=1100), stored_png(graph, 200, idat_chunk=1100)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(first, 0x10000, carver)
    [b] = carve(second, 0x40000, carver)
    # Torn after the second IDAT chunk (past the point the outputs differ): the bytes agree with exactly one.
    cut = first.index(b"IDAT") - 4 + 2 * (12 + 1100) + 30
    [torn] = carve(second[:cut], 0x80000, carver)
    assert torn.completeness == "PARTIAL" and _has_encoded_pixels(torn)
    result = correlate([a, b, torn], [workflow_for(graph)])
    assert sorted(findings_of(result, a, b, torn)) == sorted([[a.artifact_id], [b.artifact_id, torn.artifact_id]])


def test_a_partial_whose_verified_bytes_agree_with_several_containers_stays_separate():
    graph = api_prompt(seed=44, batch_size=2)
    first, second = stored_png(graph, 100, idat_chunk=1100), stored_png(graph, 200, idat_chunk=1100)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(first, 0x10000, carver)
    [b] = carve(second, 0x40000, carver)
    cut = first.index(b"IDAT") - 4 + (12 + 1100) + 30  # inside the shared bytes: both containers agree
    [torn] = carve(first[:cut], 0x80000, carver)
    assert torn.completeness == "PARTIAL" and _has_encoded_pixels(torn)
    result = correlate([a, b, torn], [workflow_for(graph)])
    assert sorted(findings_of(result, a, b, torn)) == sorted(
        [[a.artifact_id], [b.artifact_id], [torn.artifact_id]])
    assert all(f.tier in ("E2", "E3") for f in result.findings)


def test_partials_without_a_complete_container_are_not_merged_by_a_shared_prefix():
    graph = api_prompt(seed=45, batch_size=2)
    first, second = stored_png(graph, 100, idat_chunk=1100), stored_png(graph, 200, idat_chunk=1100)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    cut = first.index(b"IDAT") - 4 + (12 + 1100) + 30
    [one] = carve(first[:cut], 0x10000, carver)
    [two] = carve(second[:cut], 0x40000, carver)
    assert _has_encoded_pixels(one) and _has_encoded_pixels(two)
    result = correlate([one, two], [workflow_for(graph)])
    assert sorted(findings_of(result, one, two)) == sorted([[one.artifact_id], [two.artifact_id]])


def test_decoded_holders_attach_only_by_exact_pixels_to_one_container():
    graph = api_prompt(seed=46, batch_size=3)
    first, second = stored_png(graph, 100), stored_png(graph, 200)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(first, 0x10000, carver)
    [b] = carve(second, 0x40000, carver)
    assert a.pixel_sha256 != b.pixel_sha256
    holder_a = _holder(a, "TENSOR_IMAGE", "CARVE-TEN-1")
    holder_b = _holder(b, "PIXEL_BUFFER", "CARVE-BUF-1")
    stranger = _holder(a, "DECODED_BITMAP", "CARVE-BMP-1")
    stranger.pixel_sha256 = "0" * 64
    result = correlate([a, b, holder_a, holder_b, stranger], [workflow_for(graph)])
    by_member = {m: f for f in result.findings for m in f.artifact_ids}
    assert by_member["CARVE-TEN-1"] is by_member[a.artifact_id]
    assert by_member["CARVE-BUF-1"] is by_member[b.artifact_id]
    assert by_member["CARVE-BMP-1"] is not by_member[a.artifact_id]


def test_a_holder_matching_the_pixels_of_several_containers_remains_separate():
    graph = api_prompt(seed=47, batch_size=2)
    # Same pixels, different bytes: zlib framing changes the container, not the decoded rows.
    plain = stored_png(graph, 100)
    chunked = stored_png(graph, 100, idat_chunk=700)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(plain, 0x10000, carver)
    [b] = carve(chunked, 0x40000, carver)
    assert a.pixel_sha256 == b.pixel_sha256 and a.container_sha256 != b.container_sha256
    holder = _holder(a, "TENSOR_IMAGE", "CARVE-TEN-2")
    result = correlate([a, b, holder], [workflow_for(graph)])
    by_member = {m: f for f in result.findings for m in f.artifact_ids}
    assert len({id(by_member[x]) for x in (a.artifact_id, b.artifact_id, holder.artifact_id)}) == 3


def test_animated_containers_with_no_pixel_digest_stay_separate_by_container():
    graph = api_prompt(seed=48, batch_size=2)
    first = stored_png(graph, 100, animated=True)
    second = stored_png(graph, 200, animated=True)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [a] = carve(first, 0x10000, carver)
    [b] = carve(second, 0x40000, carver)
    assert a.pixel_sha256 is None and b.pixel_sha256 is None and a.completeness == "COMPLETE"
    result = correlate([a, b], [workflow_for(graph)])
    assert sorted(findings_of(result, a, b)) == sorted([[a.artifact_id], [b.artifact_id]])


def _holder(source, carve_type: str, artifact_id: str):
    import dataclasses

    return dataclasses.replace(
        source, artifact_id=artifact_id, carve_type=carve_type, container_sha256=None, spans=[],
        embedded_text=dict(source.embedded_text), process_id=1234, physical_offsets=[], evidence={})
