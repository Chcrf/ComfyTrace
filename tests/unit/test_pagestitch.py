"""Synthetic tests for content-verified PNG page stitching."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from comfytrace.pagestitch import PAGE, PageIndex, stitch_png


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _stored_png(width: int = 256, height: int = 300, idat_size: int = 24000) -> tuple[bytes, bytes]:
    """A continuous random-ish RGB image encoded as stored DEFLATE blocks."""
    rng = np.random.default_rng(42)
    rows = []
    previous = rng.integers(0, 256, (width, 3), dtype=np.uint8)
    for _ in range(height):
        delta = rng.integers(-5, 6, (width, 3), dtype=np.int16)
        previous = np.clip(previous.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        rows.append(b"\x00" + previous.tobytes())
    raw = b"".join(rows)
    compressed = zlib.compress(raw, level=0)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = bytearray(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr))
    for start in range(0, len(compressed), idat_size):
        png += _chunk(b"IDAT", compressed[start : start + idat_size])
    png += _chunk(b"IEND", b"")
    return bytes(png), compressed


def _scattered(png: bytes, *, remove: int | None = None, decoy: bytes | None = None) -> tuple[bytes, bytes, list[int]]:
    cut = PAGE
    prefix = png[:cut]
    file_pages = [png[pos : pos + PAGE] for pos in range(cut, len(png), PAGE)]
    physical_pages = [bytes(np.random.default_rng(1000 + i).integers(0, 256, PAGE, dtype=np.uint8))
                      for i in range(len(file_pages) * 2 + 12)]
    frames = list(range(3, 3 + len(file_pages) * 2, 2))
    for number, (frame, page) in enumerate(zip(frames, file_pages, strict=True)):
        if number != remove:
            physical_pages[frame] = page.ljust(PAGE, b"\x00")
    if decoy is not None:
        physical_pages[1] = decoy
    return prefix, b"".join(physical_pages), [frame * PAGE for frame in frames]


def test_broad_stitch_skips_untagged_png_but_examiner_offset_can_select_it():
    from comfytrace.carve import PngCarver, parse_png
    from comfytrace.cli import _content_stitch_pngs

    png, _ = _stored_png(width=32, height=64)
    prefix, physical, _ = _scattered(png)
    parsed = parse_png(prefix, False, "PAGE_NOT_PRESENT")
    assert parsed is not None
    original = PngCarver().record_from(parsed, 0, False, None, "TORN")
    images = [original]
    capture = prefix + physical[len(prefix):]

    _content_stitch_pngs(images, capture, require_workflow=True)
    assert images == [original]

    _content_stitch_pngs(images, capture, offsets=frozenset({0}), seconds=10)
    [rebuilt] = images
    assert rebuilt.derived_reconstruction == png and rebuilt.is_derived


def _padding_bit_decoy(png: bytes) -> tuple[int, bytes]:
    """Change ignored stored-block padding bits: inflate still succeeds but the IDAT CRC must reject it."""
    targets = []
    compressed = _idat_data(png)
    cursor = 2  # skip the zlib header
    while cursor + 5 <= len(compressed):
        targets.append(cursor)
        final = compressed[cursor] & 1
        stored_length = int.from_bytes(compressed[cursor + 1 : cursor + 3], "little")
        cursor += 5 + stored_length
        if final:
            break
    locations: dict[int, int] = {}
    stream_offset = 0
    pos = 8
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos : pos + 8])
        if kind == b"IDAT":
            for target in targets:
                if stream_offset <= target < stream_offset + length:
                    locations[target] = pos + 8 + target - stream_offset
            stream_offset += length
        pos += 12 + length
    # Stored blocks ignore the upper five bits of their one-byte header.
    for absolute in locations.values():
        page_number = absolute // PAGE - 1
        within = absolute % PAGE
        if page_number >= 0:
            original = png[(page_number + 1) * PAGE : (page_number + 2) * PAGE].ljust(PAGE, b"\x00")
            changed = bytearray(original)
            changed[within] ^= 0xF8
            try:
                if zlib.decompress(_idat_data(png)) == zlib.decompress(_idat_data_with_page(png, page_number, changed)):
                    return page_number, bytes(changed)
            except zlib.error:
                pass
    raise AssertionError("could not construct a valid stored-DEFLATE decoy")


def _idat_data(png: bytes) -> bytes:
    out = bytearray()
    pos = 8
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos : pos + 8])
        if kind == b"IDAT":
            out += png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    return bytes(out)


def _idat_data_with_page(png: bytes, page_number: int, page: bytes) -> bytes:
    altered = bytearray(png)
    start = (page_number + 1) * PAGE
    altered[start : min(start + PAGE, len(altered))] = page[: max(0, len(altered) - start)]
    return _idat_data(bytes(altered))


def test_scattered_pages_rebuild_exactly_and_zero_padded_final_page_is_indexed():
    png, _ = _stored_png()
    prefix, physical, expected_frames = _scattered(png)
    index = PageIndex.build(physical, chunk_pages=16)
    result = stitch_png(prefix, 0, physical, index, beam=8)
    assert result is not None and result.complete
    assert result.data == png
    assert result.page_offsets == expected_frames
    assert result.candidate_counts and all(count >= 1 for count in result.candidate_counts)
    assert expected_frames[-1] in index.boundary_offsets


def test_missing_page_rolls_back_to_the_last_crc_boundary_without_unsealed_bytes():
    png, _ = _stored_png()
    prefix, full, _ = _scattered(png)
    complete = stitch_png(prefix, 0, full, PageIndex.build(full, chunk_pages=16), beam=8)
    assert complete is not None and complete.complete
    # Remove a page after at least one stitched IDAT CRC has been sealed.
    prefix, physical, _ = _scattered(png, remove=9)
    result = stitch_png(prefix, 0, physical, PageIndex.build(physical, chunk_pages=16), beam=8)
    assert result is not None and not result.complete
    assert png.startswith(result.data)
    assert len(result.data) > len(prefix)
    # The returned byte is exactly the end of a CRC-verified chunk, never the end of an accepted interior page.
    pos = 8
    ends = set()
    while pos + 12 <= len(png):
        length = int.from_bytes(png[pos : pos + 4], "big")
        pos += 12 + length
        ends.add(pos)
    assert len(result.data) in ends


def test_valid_deflate_decoy_with_wrong_idat_crc_is_never_kept():
    png, _ = _stored_png()
    page_number, decoy = _padding_bit_decoy(png)
    prefix, physical, expected_frames = _scattered(png, decoy=decoy)
    result = stitch_png(prefix, 0, physical, PageIndex.build(physical, chunk_pages=16), beam=8)
    assert result is not None and result.complete and result.data == png
    assert PAGE not in result.page_offsets
    assert expected_frames[page_number] in result.page_offsets


def test_two_checksum_valid_tails_do_not_prove_one_reconstruction():
    png, _ = _stored_png(width=32, height=64, idat_size=12000)
    first = png[:-12] + _chunk(b"tEXt", b"Description\x00A") + png[-12:]
    second = png[:-12] + _chunk(b"tEXt", b"Description\x00B") + png[-12:]
    assert first[:PAGE] == second[:PAGE] and len(first) < 2 * PAGE
    physical = first[PAGE:].ljust(PAGE, b"\x00") + second[PAGE:].ljust(PAGE, b"\x00")
    index = PageIndex.build(physical)

    assert stitch_png(first[:PAGE], 0, physical, index) is None


def test_two_divergent_sealed_prefixes_remain_unresolved_when_both_tails_are_missing():
    png, _ = _stored_png(width=32, height=64, idat_size=12000)
    longer_text = _chunk(b"tEXt", b"Comment\x00" + b"x" * PAGE)
    first = png[:-12] + _chunk(b"tEXt", b"Description\x00A") + longer_text + png[-12:]
    second = png[:-12] + _chunk(b"tEXt", b"Description\x00B") + longer_text + png[-12:]
    assert first[:PAGE] == second[:PAGE] and len(first) > 2 * PAGE
    physical = first[PAGE:2 * PAGE] + second[PAGE:2 * PAGE]
    index = PageIndex.build(physical)

    assert stitch_png(first[:PAGE], 0, physical, index) is None


@pytest.mark.parametrize("reason", ["PAGE_NOT_PRESENT", "NONCONTIGUOUS_PHYSICAL"])
def test_cli_wiring_replaces_the_record_but_exports_raw_prefix_and_derived_rebuild(tmp_path, reason):
    import json

    from comfytrace.carve import PngCarver, parse_png
    from comfytrace.cli import _content_stitch_pngs, _export

    png, _ = _stored_png()
    prefix, physical, expected_frames = _scattered(png)
    physical = bytearray(physical)
    physical[:PAGE] = prefix
    parsed = parse_png(prefix, reason == "NONCONTIGUOUS_PHYSICAL", reason, 0)
    assert parsed is not None
    original = PngCarver().record_from(parsed, 0, False, None, "CARVE-PNG-001")
    original.page_origins = [{"pages": ["PRIVATE"], "bytes": len(prefix)}]
    original_offsets = list(original.physical_offsets)
    original_spans = list(original.spans)
    assert original.gaps[0].reason == reason
    images = [original]

    _content_stitch_pngs(images, bytes(physical))
    [stitched] = images
    assert stitched is not original and stitched.completeness == "COMPLETE" and stitched.is_derived
    assert stitched.physical_offsets == original_offsets and stitched.spans == original_spans
    assert stitched.page_origins == original.page_origins
    assert stitched.derived_reconstruction == png
    assert stitched.evidence["assembly"] == {
        "method": "CONTENT_STITCHED", "derived": True, "pages": len(expected_frames),
        "page_offsets": [f"0x{offset:X}" for offset in expected_frames], "sealed_bytes": len(png),
        "prefix_page_offsets": ["0x0"],
        "prefix_bytes": len(prefix), "last_page_bytes": len(png) - len(expected_frames) * PAGE,
        "source_gaps": [gap.to_dict() for gap in original.gaps],
        "candidate_counts": stitched.evidence["assembly"]["candidate_counts"],
    }
    from comfytrace.evidence import evidence_document

    [proof] = evidence_document(images, [], {})["records"]
    assert proof["memory_source"] == "MEMORY_PARTIAL"
    assert proof["assembly"]["method"] == "CONTENT_STITCHED" and "derived" in proof["sha256"]

    _export(tmp_path, images, [])
    [raw] = [path for path in (tmp_path / "exports" / "raw").rglob("*") if path.is_file()]
    [derived] = [path for path in (tmp_path / "exports" / "derived").iterdir() if path.is_file()]
    assert raw.read_bytes() == prefix
    assert derived.name.startswith("DERIVED_content_stitched_") and derived.read_bytes() == png

    from tests.integration.test_evaluation_matrix import assert_no_padding_and_hashes
    from tests.unit.test_report import _finding, _unit_rep, _workflow

    artifact = stitched.to_dict()
    assert artifact["derived_preview_metadata"]["raw_recovered_bytes"] == len(prefix)
    (tmp_path / "recovery-evidence.json").write_text(json.dumps({"records": [proof]}), encoding="utf-8")
    assert_no_padding_and_hashes(tmp_path, {"findings": {"artifacts": [artifact], "workflows": []}})
    rep, data, page = _unit_rep(
        [_finding("FIND-1", "E3", [stitched.artifact_id, "WF-1"])],
        [_workflow("WF-1")], [artifact], case_dir=tmp_path,
        evidence={"records": [proof]},
    )
    [wb_finding] = data["findings"]
    [wb_image] = wb_finding["images"]
    assert wb_finding["tier"] == "E3"
    assert wb_image["derived"] is True and wb_image["derivedNote"] == "Reconstructed by content stitched"
    assert wb_image["source"] == "only partly on ordinary memory pages"
    assert wb_image["file"] == raw.relative_to(tmp_path).as_posix()
    assert artifact["derived_preview_metadata"]["rendering"] == (
        "CRC-sealed reconstruction from content-matched physical pages"
    )
    assert rep.records[stitched.artifact_id]["sha256"]["raw"] == proof["sha256"]["raw"]
    assert rep.records[stitched.artifact_id]["sha256"]["derived"] == proof["sha256"]["derived"]
    assert "Reconstructed by content stitched" in page


def test_stitched_metadata_and_independent_intact_copy_keep_separate_provenance(tmp_path):
    from comfytrace.carve import IdAllocator, PngCarver, canonical_json, parse_png
    from comfytrace.cli import _content_stitch_pngs, _export, _merge_duplicate_images
    from comfytrace.correlate import png_workflows
    from comfytrace.evidence import memory_view
    from tests.synthetic import api_prompt

    png, _ = _stored_png(width=32, height=64, idat_size=12000)
    png = png[:-12] + _chunk(b"tEXt", b"prompt\0" + canonical_json(api_prompt()).encode()) + png[-12:]
    prefix, physical, _ = _scattered(png)
    parsed = parse_png(prefix, False, "PAGE_NOT_PRESENT")
    assert parsed is not None
    original = PngCarver().record_from(parsed, 0, False, None, "TORN")
    original.page_origins = [{"pages": ["PRIVATE"], "bytes": len(prefix)}]
    images = [original]
    _content_stitch_pngs(images, prefix + physical[len(prefix):])
    [reconstructed] = images
    assert reconstructed.derived_reconstruction == png
    assert not reconstructed.embedded_text and "prompt" in reconstructed.stitched_text

    [intact] = PngCarver().carve_stream(png, base_physical_offset=PAGE * 100)
    restored = _merge_duplicate_images([reconstructed, intact])
    assert restored == [reconstructed, intact] and len(intact.physical_offsets) == 1
    assert _merge_duplicate_images([intact, reconstructed]) == [intact, reconstructed]
    workflows = png_workflows(restored, IdAllocator("WFPNG"))
    assert [w.encoding_detected for w in workflows] == ["PNG_TEXT_STITCHED", "PNG_TEXT"]
    assert [w.execution_parameters["source_locations"][0]["png"] for w in workflows] == [
        reconstructed.artifact_id, intact.artifact_id,
    ]

    _export(tmp_path, restored, workflows)
    assert workflows[0].raw_file_path.startswith("exports/derived/DERIVED_")
    assert workflows[1].raw_file_path.startswith("exports/raw/")
    assert memory_view([reconstructed], workflows[:1]) == ([], [])


def test_a_complete_content_stitched_image_is_capped_at_the_derived_e3_policy():
    import json

    from comfytrace.carve import IdAllocator, PngCarver
    from comfytrace.correlate import correlate, png_workflows
    from tests.synthetic import api_prompt, noise_image, png_bytes

    png = png_bytes(noise_image(32, 32, seed=9), {"prompt": json.dumps(api_prompt(width=32, height=32))})
    images = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(png)
    images[0].is_derived = True
    workflows = png_workflows(images, IdAllocator("WFPNG"))
    result = correlate(images, workflows)
    assert result.tier_summary["E3"] == 1 and result.tier_summary["E2"] == 0


def test_stitch_continues_a_prefix_already_assembled_from_file_cache_pages():
    from comfytrace.carve import PngCarver, parse_png
    from comfytrace.cli import _content_stitch_pngs
    from comfytrace.models import GapReason

    png, _ = _stored_png()
    _, physical, expected = _scattered(png)
    data = bytearray(physical)
    data[:PAGE] = png[:PAGE]
    prefix = png[:2 * PAGE]
    parsed = parse_png(prefix, False, GapReason.PAGE_NOT_PRESENT, 0)
    assert parsed is not None
    artifact = PngCarver().record_from(parsed, 0, False, None)
    artifact.evidence["assembly"] = {"method": "PROTOTYPE_PTE_ORDER", "page_offsets": ["0x0", f"0x{expected[0]:X}"]}
    images = [artifact]

    _content_stitch_pngs(images, bytes(data), offsets=frozenset({PAGE}))
    assert images == [artifact]
    _content_stitch_pngs(images, bytes(data), offsets=frozenset({0}))

    [recovered] = images
    assert recovered.derived_reconstruction == png
    assert recovered.evidence["assembly"]["prefix_page_offsets"] == ["0x0", f"0x{expected[0]:X}"]
    assert recovered.evidence["assembly"]["page_offsets"] == [f"0x{offset:X}" for offset in expected[1:]]


def test_deadline_already_expired_returns_none_without_reading_pages():
    import time

    class _CountingPhysical:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.reads = 0

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, item: slice) -> bytes:
            self.reads += 1
            return self._data[item]

    png, _ = _stored_png()
    prefix, physical, _ = _scattered(png)
    index = PageIndex.build(physical, chunk_pages=16)
    counting = _CountingPhysical(physical)

    assert stitch_png(prefix, 0, counting, index, deadline=time.perf_counter() - 1.0) is None
    assert counting.reads == 0


def test_deadline_non_expired_preserves_exact_reconstruction():
    import time

    png, _ = _stored_png()
    prefix, physical, expected_frames = _scattered(png)
    index = PageIndex.build(physical, chunk_pages=16)
    result = stitch_png(prefix, 0, physical, index, beam=8, deadline=time.perf_counter() + 60.0)
    assert result is not None and result.complete
    assert result.data == png
    assert result.page_offsets == expected_frames


def test_deadline_amortized_check_aborts_long_interior_loop_early(monkeypatch):
    from comfytrace import pagestitch

    png, _ = _stored_png(width=64, height=64, idat_size=12000)
    prefix = png[:PAGE]
    noise_pages = pagestitch._DEADLINE_CHECK_INTERVAL * 3 + 20
    rng = np.random.default_rng(77)
    noise = rng.integers(0, 256, (noise_pages, PAGE), dtype=np.uint8).tobytes()
    file_pages = [png[pos : pos + PAGE].ljust(PAGE, b"\x00") for pos in range(PAGE, len(png), PAGE)]
    physical = noise + b"".join(file_pages)
    index = PageIndex.build(physical, chunk_pages=64)

    class _CountingPhysical:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.reads = 0

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, item: slice) -> bytes:
            self.reads += 1
            return self._data[item]

    # Entry check + step 0 start + state 0 start = 3 calls before entering inner loop;
    # expire on the very first amortized inner-loop check (4th call).
    calls = 0

    def fake_perf_counter() -> float:
        nonlocal calls
        calls += 1
        return 100.0 if calls >= 4 else 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", fake_perf_counter)
    counting = _CountingPhysical(physical)
    result = stitch_png(prefix, 0, counting, index, beam=8, deadline=50.0)
    assert result is None
    assert calls == 4
    assert counting.reads < pagestitch._DEADLINE_CHECK_INTERVAL


def test_deadline_mid_step_never_accepts_unvetted_complete_tail(monkeypatch):
    from comfytrace import pagestitch

    png, _ = _stored_png(width=32, height=64, idat_size=12000)
    first = png[:-12] + _chunk(b"tEXt", b"Description\x00A") + png[-12:]
    second = png[:-12] + _chunk(b"tEXt", b"Description\x00B") + png[-12:]
    assert first[:PAGE] == second[:PAGE] and len(first) < 2 * PAGE

    # Place valid tail A before enough noise pages to trigger an amortized check, then conflicting tail B after.
    gap_pages = pagestitch._DEADLINE_CHECK_INTERVAL + 16
    rng = np.random.default_rng(91)
    noise = rng.integers(0, 256, (gap_pages, PAGE), dtype=np.uint8).tobytes()
    physical = first[PAGE:].ljust(PAGE, b"\x00") + noise + second[PAGE:].ljust(PAGE, b"\x00")
    index = PageIndex.build(physical, chunk_pages=64)

    # Without deadline, ambiguity rejects both tails.
    assert stitch_png(first[:PAGE], 0, physical, index) is None

    # Expire after tail A has been accepted into completed, before tail B is visited.
    calls = 0

    def fake_perf_counter() -> float:
        nonlocal calls
        calls += 1
        return 100.0 if calls >= 4 else 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", fake_perf_counter)
    assert stitch_png(first[:PAGE], 0, physical, index, deadline=50.0) is None
    assert calls == 4


def test_deadline_mid_step_never_accepts_unvetted_sealed_prefix(monkeypatch):
    from comfytrace import pagestitch

    png, _ = _stored_png(width=32, height=64, idat_size=12000)
    longer_text = _chunk(b"tEXt", b"Comment\x00" + b"x" * PAGE)
    first = png[:-12] + _chunk(b"tEXt", b"Description\x00A") + longer_text + png[-12:]
    second = png[:-12] + _chunk(b"tEXt", b"Description\x00B") + longer_text + png[-12:]
    assert first[:PAGE] == second[:PAGE] and len(first) > 2 * PAGE

    gap_pages = pagestitch._DEADLINE_CHECK_INTERVAL + 16
    rng = np.random.default_rng(92)
    noise = rng.integers(0, 256, (gap_pages, PAGE), dtype=np.uint8).tobytes()
    physical = first[PAGE : 2 * PAGE] + noise + second[PAGE : 2 * PAGE]
    index = PageIndex.build(physical, chunk_pages=64)

    assert stitch_png(first[:PAGE], 0, physical, index) is None

    calls = 0

    def fake_perf_counter() -> float:
        nonlocal calls
        calls += 1
        return 100.0 if calls >= 4 else 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", fake_perf_counter)
    assert stitch_png(first[:PAGE], 0, physical, index, deadline=50.0) is None
    assert calls == 4


def test_deadline_expiry_after_completed_seal_returns_only_sealed_bytes(monkeypatch):
    from comfytrace import pagestitch

    png, _ = _stored_png()
    prefix, physical, _ = _scattered(png)
    index = PageIndex.build(physical, chunk_pages=16)

    # First verify how many steps seal the first IDAT chunk (step 5 seals the 24000-byte IDAT).
    # Each step with 1 active state makes 2 perf_counter calls (step start + state start) when pages < interval,
    # plus 1 entry check = 1 + 2 * steps. Stop before step 8 so IDAT #1 is sealed cleanly, not the full PNG.
    calls = 0

    def fake_perf_counter() -> float:
        nonlocal calls
        calls += 1
        return 100.0 if calls >= 18 else 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", fake_perf_counter)
    result = stitch_png(prefix, 0, physical, index, beam=8, deadline=50.0)
    assert result is not None and not result.complete
    assert png.startswith(result.data)
    assert len(result.data) > len(prefix)

    pos = 8
    ends = set()
    while pos + 12 <= len(png):
        length = int.from_bytes(png[pos : pos + 4], "big")
        pos += 12 + length
        ends.add(pos)
    assert len(result.data) in ends


def test_mmap_view_is_released_and_page_index_cache_is_reusable(tmp_path):
    import mmap

    png, _ = _stored_png(width=64, height=64, idat_size=12000)
    prefix, physical, expected_frames = _scattered(png)
    dump_path = tmp_path / "phys.raw"
    dump_path.write_bytes(physical)
    with dump_path.open("rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            index = PageIndex.build(mm, chunk_pages=16)
            first = stitch_png(prefix, 0, mm, index, beam=8)
            assert first is not None and first.complete and first.data == png
            assert first.page_offsets == expected_frames

            # Reuse the same PageIndex with an excluded frame and verify cache isolation.
            excluded = stitch_png(prefix, 0, mm, index, beam=8, excluded_page_offsets=frozenset({expected_frames[0]}))
            assert excluded is None

            # Reuse again without exclusion -> still recovers identically.
            second = stitch_png(prefix, 0, mm, index, beam=8)
            assert second is not None and second.complete and second.data == png
            assert second.page_offsets == expected_frames
        finally:
            # Must not raise BufferError due to leaked memoryview references.
            mm.close()


def test_deadline_mid_step_rollback_preserves_prior_sealed_chunk_but_drops_incomplete_step(monkeypatch):
    from comfytrace import pagestitch

    png, _ = _stored_png(width=64, height=96, idat_size=12000)
    prefix = png[:PAGE]
    file_pages = [png[pos : pos + PAGE].ljust(PAGE, b"\x00") for pos in range(PAGE, len(png), PAGE)]
    gap_pages = pagestitch._DEADLINE_CHECK_INTERVAL + 16
    rng = np.random.default_rng(93)
    noise = rng.integers(0, 256, (gap_pages, PAGE), dtype=np.uint8).tobytes()
    # Put all true PNG pages BEFORE the noise block so on the final step the completing page is found first,
    # and the amortized deadline check fires in the trailing noise before the final step finishes!
    physical = b"".join(file_pages) + noise
    index = PageIndex.build(physical, chunk_pages=64)

    full = stitch_png(prefix, 0, physical, index, beam=8)
    assert full is not None and full.complete and full.data == png
    final_step = len(full.candidate_counts) - 1
    assert final_step >= 3

    # Count how many perf_counter calls happen up to the start of the final step when deadline is active.
    # Instead of guessing call counts, run once with a non-expiring deadline to record total calls, then expire
    # right inside the final step's candidate loop (after page 0..final_step has been tested, before loop ends).
    total_calls = 0

    def count_calls() -> float:
        nonlocal total_calls
        total_calls += 1
        return 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", count_calls)
    counted = stitch_png(prefix, 0, physical, index, beam=8, deadline=50.0)
    assert counted is not None and counted.complete and total_calls >= 4

    # Expire on the very last perf_counter call (inside the final step's trailing noise scan, AFTER the completing
    # IEND page at offset (final_step * PAGE) was already added to `completed`).
    seen = 0

    def expire_on_final_step_loop() -> float:
        nonlocal seen
        seen += 1
        return 100.0 if seen >= total_calls else 10.0

    monkeypatch.setattr(pagestitch.time, "perf_counter", expire_on_final_step_loop)
    partial = stitch_png(prefix, 0, physical, index, beam=8, deadline=50.0)
    assert partial is not None and not partial.complete
    assert png.startswith(partial.data)
    assert len(prefix) < len(partial.data) < len(png)
    assert len(partial.candidate_counts) == final_step


def test_straddling_crc_and_header_boundaries_and_low_entropy_fallback():
    # Craft an IDAT whose 4-byte CRC straddles the first page boundary (4094 bytes in page 1, 2 bytes in page 2),
    # and a second IDAT whose 8-byte header straddles a page boundary, plus a low-entropy final page only in nonzero.
    width, height = 64, 128
    raw = b"".join(b"\x00" + bytes([i % 256]) * (width * 3) for i in range(height))
    compressed = zlib.compress(raw, level=0)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    head = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
    # Prefix is 1 page; first IDAT starts at len(head) + 8. Make first IDAT end at 2 * PAGE - 2 (CRC splits 2 + 2).
    first_idat_len = 2 * PAGE - 2 - len(head) - 8
    assert 0 < first_idat_len < len(compressed)
    # Second IDAT ends at 4 * PAGE - 7 (4 bytes CRC -> ends at 4 * PAGE - 3 -> next header splits 3 + 5).
    second_idat_len = 4 * PAGE - 7 - (2 * PAGE + 10)
    assert first_idat_len + second_idat_len < len(compressed)
    rest_idat = compressed[first_idat_len + second_idat_len :]
    png = (
        head
        + _chunk(b"IDAT", compressed[:first_idat_len])
        + _chunk(b"IDAT", compressed[first_idat_len : first_idat_len + second_idat_len])
        + _chunk(b"IDAT", rest_idat)
        + _chunk(b"IEND", b"")
    )
    prefix, physical, expected_frames = _scattered(png)
    # Force an index with empty high_entropy_offsets so the lazy nonzero fallback path is exercised too.
    full_index = PageIndex.build(physical, chunk_pages=16)
    fallback_index = PageIndex(
        total_pages=full_index.total_pages,
        nonzero_offsets=full_index.nonzero_offsets,
        high_entropy_offsets=np.empty(0, dtype=np.uint64),
        boundary_offsets=full_index.boundary_offsets,
        entropy_threshold=full_index.entropy_threshold,
        build_seconds=full_index.build_seconds,
    )

    for idx in (full_index, fallback_index):
        result = stitch_png(prefix, 0, physical, idx, beam=8)
        assert result is not None and result.complete
        assert result.data == png
        assert result.page_offsets == expected_frames
