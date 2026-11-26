"""Tier rules added after the evaluation: a confirmed close match earns E3; an image too small to be a sampler's
output earns no tier; gap rows read from frames of uncertain ownership stay a separate layer."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from comfytrace import bitmaps, link, memrecover
from comfytrace.carve import IdAllocator, PngCarver
from comfytrace.correlate import MIN_OUTPUT_SIDE, OTHER_IMAGES, correlate, png_workflows
from comfytrace.models import CarveType, LinkClass
from comfytrace.report import EvidentiaryExporter, apply_image_links, memory_only_after_links, tier_counts
from tests.synthetic import api_prompt, noise_image, png_bytes
from tests.unit.test_link import _bind_regenerations, _case, _case_regeneration
from tests.unit.test_memory_recovery import _smooth_image

SHA = "ab" * 32


def finding(finding_id: str, tier: str, artifacts: list[str], rationale: str = "") -> dict[str, Any]:
    return {"finding_id": finding_id, "tier": tier, "base_tier": tier, "extraction_tier": tier,
            "artifact_ids": artifacts, "link_ids": [], "rationale": rationale, "regeneration": None,
            "image_link": None}


def close(prompt_id: str, images: list[str], prompts: list[str]) -> dict[str, Any]:
    return {"tier": "E3", "basis": link.CLOSE_MATCH, "prompt_id": prompt_id, "artifact_ids": images,
            "workflow_ids": prompts, "agreement": {}, "regenerated": "x.png"}


def test_close_matches_of_one_job_form_one_e3_finding_with_its_prompt():
    findings = [finding("FIND-001", "UNTIERED", ["BITMAP-001"]), finding("FIND-002", "UNTIERED", ["BITMAP-002"]),
                finding("FIND-003", "E5", ["WFPY-001", "WF-009"]), finding("FIND-004", "E5", ["WFPY-003"]),
                finding("FIND-005", "UNTIERED", ["BITMAP-003"])]
    links = {"manifest_sha256": SHA, "promotions": [close("job-a", ["BITMAP-001", "BITMAP-002"], ["WFPY-001"])]}
    notes = apply_image_links(findings, links, SHA)
    assert [(f["finding_id"], f["tier"], f["artifact_ids"]) for f in findings] == [
        ("FIND-001", "E3", ["BITMAP-001", "BITMAP-002", "WFPY-001", "WF-009"]),
        ("FIND-004", "E5", ["WFPY-003"]), ("FIND-005", "UNTIERED", ["BITMAP-003"])]
    assert findings[0]["image_link"] == link.CLOSE_MATCH and findings[0]["absorbed"] == ["FIND-003"]
    assert "closely matching" in findings[0]["rationale"]
    assert [(n["promotion"], n["applied_to"]) for n in notes] == [("BITMAP-001", "FIND-001"),
                                                                   ("BITMAP-002", "FIND-001")]
    assert tier_counts(findings)["E3"] == 1 and tier_counts(findings)["E2"] == 0


def test_close_copies_join_the_jobs_exact_finding_and_never_raise_its_tier():
    findings = [finding("FIND-001", "UNTIERED", ["PIXBUF-001"]), finding("FIND-002", "UNTIERED", ["BITMAP-001"]),
                finding("FIND-003", "E5", ["WFPY-001"])]
    exact = {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-a",
             "artifact_id": "PIXBUF-001", "workflow_ids": ["WFPY-001"], "agreement": {}, "regenerated": "x.png"}
    links = {"manifest_sha256": SHA, "promotions": [exact, close("job-a", ["BITMAP-001"], ["WFPY-001"])]}
    notes = apply_image_links(findings, links, SHA)
    [only] = findings
    assert (only["tier"], only["artifact_ids"]) == ("E2", ["PIXBUF-001", "WFPY-001", "BITMAP-001"])
    assert notes[-1]["note"] == f"{link.CLOSE_MATCH}: joined FIND-001 (E2)"


def test_a_close_match_takes_its_image_out_of_the_other_images_group_and_absorbs_a_lone_partial_prompt():
    findings = [finding("FIND-001", "UNTIERED", ["BITMAP-001", "BITMAP-002", "CARVE-PNG-007"],
                        f"{OTHER_IMAGES}: 3 images with no ComfyUI text."),
                finding("FIND-002", "E6", ["WFPY-001"]), finding("FIND-003", "E6", ["WF-020", "WF-021"])]
    links = {"manifest_sha256": SHA, "promotions": [close("job-a", ["BITMAP-002"], ["WFPY-001"])]}
    apply_image_links(findings, links, SHA)
    assert [(f["finding_id"], f["tier"], f["artifact_ids"]) for f in findings] == [
        ("FIND-001", "UNTIERED", ["BITMAP-001", "CARVE-PNG-007"]), ("FIND-003", "E6", ["WF-020", "WF-021"]),
        ("FIND-004", "E3", ["BITMAP-002", "WFPY-001"])]
    assert tier_counts(findings) == {"E1": 0, "E2": 0, "E3": 1, "E4": 0, "E5": 0, "E6": 1, "UNTIERED": 0,
                                     "Other images": 2}
    # An image that is not untiered in this view (absent from it) changes nothing.
    untouched = [finding("FIND-001", "E5", ["WFPY-001"])]
    notes = apply_image_links(untouched, links, SHA)
    assert untouched[0]["tier"] == "E5" and notes[0]["applied_to"] is None


def test_the_memory_only_view_gets_its_own_row_after_image_links():
    stored = [{"finding_id": "FIND-001", "tier": "UNTIERED", "artifact_ids": ["BITMAP-001"], "link_ids": [],
               "rationale": "Partial image with no corroborating workflow key (orphan)."},
              {"finding_id": "FIND-002", "tier": "E5", "artifact_ids": ["WFPY-001"], "link_ids": [],
               "rationale": "Complete prompt graph."}]
    links = {"manifest_sha256": SHA, "promotions": [close("job-a", ["BITMAP-001"], ["WFPY-001"])]}
    evidence = {"memory_only_findings": stored}
    assert memory_only_after_links(evidence, links, SHA)["E3"] == 1
    assert stored[0]["tier"] == "UNTIERED", "the stored findings are not changed"
    assert memory_only_after_links(evidence, links, "cd" * 32) is None, "links made against another manifest"
    assert memory_only_after_links({}, links, SHA) is None and memory_only_after_links(evidence, None, SHA) is None


def test_an_exact_image_whose_job_has_only_a_partial_prompt_is_e3(tmp_path):
    pixels = np.asarray(noise_image(32, 32, seed=102))
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", pixels)])
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["findings"]["workflows"] = [{
        "workflow_id": "WFPY-001", "prompt_format": "API_GRAPH", "completeness": "PARTIAL",
        "execution_parameters": {"prompt_id": "job-a"},
    }]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", pixels)

    [promotion] = link.link(case)["promotions"]
    assert (promotion["tier"], promotion["artifact_ids"], promotion["workflow_ids"]) == (
        "E3", ["PIXBUF-001"], ["WFPY-001"])
    manifest["findings"]["workflows"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _bind_regenerations(case)
    assert link.link(case)["promotions"] == [], "no recovered prompt, no finding"


def test_a_one_pixel_png_that_carries_a_prompt_earns_no_tier_but_its_prompt_does():
    def records(size: int) -> tuple[list, list]:
        png = png_bytes(noise_image(size, size, seed=4), {"prompt": json.dumps(api_prompt(seed=42))})
        images = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(png)
        return images, png_workflows(images, IdAllocator("WFPNG"))

    images, workflows = records(1)
    result = correlate(images, workflows)
    [image] = images
    assert image.link_status == LinkClass.STRICT and image.associated_workflow_id == "WFPNG-001"
    assert result.tier_summary["E2"] == result.tier_summary["E3"] == 0
    assert result.tier_summary["untiered_orphan_images"] == 1 and result.tier_summary["E5"] == 1
    untiered = next(f for f in result.findings if f.artifact_ids == [image.artifact_id])
    assert untiered.tier == "UNTIERED" and f"below {MIN_OUTPUT_SIDE} pixels" in untiered.rationale
    assert untiered.link_ids == [result.links[0].link_id]
    images, workflows = records(MIN_OUTPUT_SIDE)
    assert correlate(images, workflows).tier_summary["E2"] == 1, "the smallest picture a latent node makes counts"


def _bitmap_with_stale_page(true_bytes_in_frame: bool = True) -> tuple[bitmaps.DecodedBitmap, np.ndarray]:
    """A 512-pixel-wide decoded bitmap of 240 rows whose page 70 is a STALE_MAPPING gap. The frame the page table
    still names was read: it holds the picture's own bytes, or another owner's."""
    rgb = _smooth_image(240, 512, seed=3)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((240, 512, 1), 255, np.uint8)], axis=2).tobytes()
    data = bytearray(bgra)
    page = 70
    data[page * bitmaps.PAGE : (page + 1) * bitmaps.PAGE] = bytes(bitmaps.PAGE)
    gaps = bitmaps.Gaps()
    gaps[page] = "STALE_MAPPING"
    content = bgra[page * bitmaps.PAGE : (page + 1) * bitmaps.PAGE] if true_bytes_in_frame else b"\x7f" * bitmaps.PAGE
    gaps.uncertain[page] = (0x4321, content)
    frames = [-1 if i == page else 1000 + i for i in range(len(data) // bitmaps.PAGE)]
    [found] = bitmaps.cut_images("TEST", bytes(data), frames, "PAGE_TABLE_ORDER", gaps)
    return found, rgb


def test_rows_read_from_a_reused_frame_are_kept_apart_from_the_image():
    found, rgb = _bitmap_with_stale_page()
    assert found.notes["gap_rows"] == [[140, 142, "STALE_MAPPING"]] and not found.rgb[140:142].any()
    assert found.notes["uncertain_rows"] == [[140, 142, ["0x4321000"]]]
    assert sorted(found.uncertain_rows) == [140, 141]
    assert np.array_equal(np.stack([found.uncertain_rows[r] for r in (140, 141)]), rgb[140:142])
    without = bitmaps.Gaps()
    without[70] = "STALE_MAPPING"
    candidate = memrecover.from_bitmap(found)
    [record] = memrecover.image_records([candidate])
    genuine = np.r_[0:140, 142:240]
    assert b"".join(span for _, span in record.spans) == rgb[genuine].tobytes(), "raw holds genuine rows only"
    assert [(g.relative_offset, g.reason) for g in record.gaps] == [(140 * 1536, "STALE_MAPPING")]
    assert "uncertain_rows" not in record.pixel_metadata["source"]
    layer = record.evidence["ownership_uncertain"]
    assert layer["rows_total"] == 2 and layer["agreeing_copies"] == []
    [rows] = layer["rows"]
    assert rows["row_range"] == [140, 142] and rows["frames_read"] == ["0x4321000"] and rows["bytes"] == 2 * 1536
    assert rows["continuity"]["edge_row_error"] <= 3 * rows["continuity"]["typical_row_error"]
    assert record.uncertain_spans == [(140, 142, rgb[140:142].tobytes())]
    # The same record without the layer has the same hashes: the uncertain rows are in neither.
    plain = bitmaps.DecodedBitmap(found.region, found.width, found.rows, found.rgb, found.physical_pages,
                                  found.byte_offset, found.ordering,
                                  {k: v for k, v in found.notes.items() if k != "uncertain_rows"})
    [bare] = memrecover.image_records([memrecover.from_bitmap(plain)])
    assert (bare.pixel_sha256, bare.evidence["sha256"]) == (record.pixel_sha256, record.evidence["sha256"])
    assert "ownership_uncertain" not in bare.evidence


def test_a_gap_in_a_regions_first_stride_window_does_not_cost_the_rows_around_it():
    """T2's castle bitmap: a reused page among its first 16 pages spoiled the first window's stride, and the 32 rows
    of that window were dropped with it."""
    rgb = _smooth_image(240, 512, seed=3)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((240, 512, 1), 255, np.uint8)], axis=2).tobytes()
    data = bytearray(bgra)
    data[5 * bitmaps.PAGE : 6 * bitmaps.PAGE] = bytes(bitmaps.PAGE)
    gaps = bitmaps.Gaps()
    gaps[5] = "STALE_MAPPING"
    gaps.uncertain[5] = (0x77, bgra[5 * bitmaps.PAGE : 6 * bitmaps.PAGE])
    frames = [-1 if i == 5 else 1000 + i for i in range(len(data) // bitmaps.PAGE)]
    [found] = bitmaps.cut_images("TEST", bytes(data), frames, "PAGE_TABLE_ORDER", gaps)
    assert (found.width, found.rows, found.byte_offset) == (512, 240, 0), "the first window's rows are kept"
    assert found.notes["gap_rows"] == [[10, 12, "STALE_MAPPING"]]
    genuine = np.r_[0:10, 12:240]
    assert np.array_equal(found.rgb[genuine], rgb[genuine]) and sorted(found.uncertain_rows) == [10, 11]


def test_an_overwritten_page_in_the_first_window_does_not_cost_that_windows_rows():
    """T2's castle bitmap: page 11 of its buffer still belongs to the mapping but no longer holds pixels. It
    spoiled the first window's stride, and the window's 32 rows were dropped."""
    rgb = _smooth_image(240, 512, seed=3)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((240, 512, 1), 255, np.uint8)], axis=2).tobytes()
    data = bytearray(bgra)
    data[11 * bitmaps.PAGE : 12 * bitmaps.PAGE] = np.random.default_rng(1).integers(0, 255, bitmaps.PAGE,
                                                                                    np.uint8).tobytes()
    [found] = bitmaps.cut_images("TEST", bytes(data), list(range(len(data) // bitmaps.PAGE)), "PAGE_TABLE_ORDER")
    assert (found.width, found.rows, found.byte_offset) == (512, 240, 0)
    assert found.notes["non_opaque_rows"] == [22, 23], "the overwritten page's rows are listed, not hidden"
    genuine = np.r_[0:22, 24:240]
    assert np.array_equal(found.rgb[genuine], rgb[genuine])
    # A narrower image before it keeps its own width: at twice its stride its rows would also look continuous.
    other = _smooth_image(120, 256, seed=8)
    narrow = np.concatenate([other[:, :, ::-1], np.full((120, 256, 1), 255, np.uint8)], axis=2).tobytes()
    two = bytearray(narrow[: 30 * bitmaps.PAGE] + bgra)
    clean = bitmaps.cut_images("TEST", bytes(two), list(range(len(two) // bitmaps.PAGE)), "PAGE_TABLE_ORDER")
    two[4 * bitmaps.PAGE : 5 * bitmaps.PAGE] = bytes(bitmaps.PAGE)
    found = bitmaps.cut_images("TEST", bytes(two), list(range(len(two) // bitmaps.PAGE)), "PAGE_TABLE_ORDER")
    assert [(b.width, b.rows) for b in found] == [(b.width, b.rows) for b in clean] and found[0].width == 256


def test_reused_pages_right_after_a_picture_give_rows_beyond_its_height_and_never_grow_it():
    rgb = _smooth_image(240, 512, seed=3)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((240, 512, 1), 255, np.uint8)], axis=2).tobytes()
    kept = 100 * bitmaps.PAGE  # the mapping's verified pages end here: 200 whole rows
    gaps = bitmaps.Gaps()
    gaps.trailing = [(0x5000 + i, bgra[kept + i * bitmaps.PAGE : kept + (i + 1) * bitmaps.PAGE]) for i in range(20)]
    gaps.trailing.append((0x6000, b"\x10\x20\x30\x00" * 1024))  # not opaque pixels: the picture's rows stop before
    [found] = bitmaps.cut_images("TEST", bgra[:kept], list(range(100)), "PAGE_TABLE_ORDER", gaps)
    assert (found.rows, found.rgb.shape) == (200, (200, 512, 3)) and "gap_rows" not in found.notes
    assert sorted(found.uncertain_rows) == list(range(200, 240))
    assert np.array_equal(np.stack([found.uncertain_rows[r] for r in range(200, 240)]), rgb[200:])
    [[first, end, frames]] = found.notes["uncertain_rows"]
    assert (first, end, frames[0], len(frames)) == (200, 240, "0x5000000", 20)
    whole = memrecover.ImageCandidate(CarveType.PIXEL_BUFFER, rgb, rgb.tobytes(), 7, 0x1000, [], {}, True)
    bitmap = memrecover.from_bitmap(found)
    memrecover.corroborate([whole, bitmap])
    _, record = memrecover.image_records([whole, bitmap])
    assert record.dimensions == (512, 200) and record.gaps == [] and record.recovered_bytes == 200 * 1536
    layer = record.evidence["ownership_uncertain"]
    [rows] = layer["rows"]
    assert rows["position"] == "after the last recovered row" and layer["rows_total"] == 40
    assert layer["agreeing_copies"] == [{"scale": 1, "rows_compared": 40, "rows_agreeing": 40, "copy": "PIXBUF-001"}]
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(record.uncertain_preview_png)).size == (512, 240)
    assert Image.open(io.BytesIO(record.preview_png)).size == (512, 200), "the image's own preview is unchanged"


def test_an_independent_copy_says_whether_the_uncertain_rows_still_hold_the_picture(tmp_path):
    for genuine_frame, agreeing in ((True, 2), (False, 0)):
        found, rgb = _bitmap_with_stale_page(genuine_frame)
        whole = memrecover.ImageCandidate(CarveType.PIXEL_BUFFER, rgb, rgb.tobytes(), 7, 0x1000, [], {}, True)
        bitmap = memrecover.from_bitmap(found)
        memrecover.corroborate([whole, bitmap])
        _, record = memrecover.image_records([whole, bitmap])
        layer = record.evidence["ownership_uncertain"]
        assert layer["agreeing_copies"] == [{"scale": 1, "rows_compared": 2, "rows_agreeing": agreeing,
                                             "copy": "PIXBUF-001"}]
        assert record.completeness == "PARTIAL" and len(record.gaps) == 1, "agreement does not fill the gap"
    case = tmp_path / "case"
    exporter = EvidentiaryExporter(case)
    exporter.export_raw_artifact(record, record.spans)
    exporter.export_uncertain(record)
    [rows] = record.evidence["ownership_uncertain"]["rows"]
    assert rows["file"] == f"exports/uncertain/{record.artifact_id}/rows_0140-0142.rgb"
    assert (case / rows["file"]).read_bytes() == b"\x7f" * (2 * 1536)
    preview = record.evidence["ownership_uncertain"]["derived_preview"]
    assert preview.startswith("exports/derived/DERIVED_preview_") and (case / preview).is_file()
    assert not [p for p in (case / "exports" / "raw").rglob("*") if "rows_" in p.name]
