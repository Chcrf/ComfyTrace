"""E3 needs recovered pixels: a job linked only to partial PNGs with zero decoded rows is tiered by its prompt."""

from __future__ import annotations

import json

from comfytrace.carve import IdAllocator, PngCarver, WorkflowCarver
from comfytrace.correlate import correlate
from comfytrace.models import Completeness
from tests.synthetic import api_prompt, noise_image, png_bytes


def _job(seed: int):
    graph = api_prompt(seed=seed)
    png = png_bytes(noise_image(64, 48, seed=seed), {"prompt": json.dumps(graph)}, idat_chunk=512)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    [header] = carver.carve_stream(png[: png.index(b"IDAT") - 4], base_physical_offset=0x100000)
    [torn] = carver.carve_stream(png[: len(png) - 900], base_physical_offset=0x200000)
    [workflow] = WorkflowCarver().carve_stream(json.dumps(graph).encode(), base_physical_offset=0x300000)
    return header, torn, workflow


def test_a_complete_prompt_linked_only_to_a_zero_row_png_is_e5_keeping_its_container():
    header, _, workflow = _job(91)
    assert header.derived_preview_metadata["decoded_rows"] == 0
    result = correlate([header], [workflow])
    [link] = result.links
    [finding] = result.findings
    assert finding.tier == "E5"
    assert finding.artifact_ids == [header.artifact_id, workflow.workflow_id]
    assert finding.link_ids == [link.link_id]
    assert f"{header.artifact_id}: 0 of 48 rows decoded" in finding.rationale
    assert "Zero PNG pixel rows decoded." in finding.rationale
    assert (result.tier_summary["E5"], result.tier_summary["E3"]) == (1, 0)
    assert header.link_status == link.link_class and header.associated_workflow_id == workflow.workflow_id


def test_missing_row_metadata_is_unknown_and_keeps_e3():
    header, _, workflow = _job(92)
    header.derived_preview_metadata = None
    [finding] = correlate([header], [workflow]).findings
    assert finding.tier == "E3"
    header, _, workflow = _job(93)
    header.derived_preview_metadata = {k: v for k, v in header.derived_preview_metadata.items()
                                       if k != "decoded_rows"}
    [finding] = correlate([header], [workflow]).findings
    assert finding.tier == "E3"


def test_a_zero_row_copy_beside_a_decoded_partial_keeps_e3():
    header, torn, workflow = _job(94)
    assert torn.derived_preview_metadata["decoded_rows"] > 0
    [finding] = correlate([torn, header], [workflow]).findings
    assert finding.tier == "E3"
    assert finding.artifact_ids == [torn.artifact_id, header.artifact_id, workflow.workflow_id]
    assert "Container provenance only" not in finding.rationale


def test_a_partial_prompt_linked_only_to_a_zero_row_png_is_e6():
    header, _, workflow = _job(95)
    workflow.completeness = Completeness.PARTIAL
    result = correlate([header], [workflow])
    [finding] = result.findings
    assert finding.tier == "E6"
    assert finding.artifact_ids == [header.artifact_id, workflow.workflow_id]
    assert finding.link_ids == [result.links[0].link_id]
    assert "tiered E6 by its prompt" in finding.rationale
