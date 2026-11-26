import hashlib
import json

from PIL import Image

from comfytrace.carve import native_pixel_digest
from comfytrace.jobview import interval, union_size
from comfytrace.report import _Report, _verified_replay_positions
from comfytrace.workbench import workbench_data


def test_jobs_view_keeps_identical_graph_jobs_separate_and_unions_spans(tmp_path):
    output_path = "exports/derived/out.png"
    path = tmp_path / output_path
    path.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10), "green").save(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    pixel_hash, pixel_meta = native_pixel_digest(data)
    manifest = {"case_metadata": {"memory_image": {"sha256": "dump"}}, "findings": {
        "workflows": [
            {"workflow_id": identifier, "execution_parameters": {"prompt_id": prompt}, "prompt_format": "API_GRAPH",
             "canonical_json_sha256": "shared-graph", "completeness": "COMPLETE"}
            for identifier, prompt in (("WFPY-001", "first"), ("WF-002", "first"), ("WFPY-003", "second"))
        ],
        "artifacts": [{"artifact_id": "IMG-001", "carve_type": "ENCODED_PNG", "completeness": "COMPLETE",
                       "pixel_sha256": pixel_hash, "pixel_metadata": pixel_meta},
                      {"artifact_id": "IMG-002", "carve_type": "ENCODED_PNG", "completeness": "COMPLETE"}],
    }}
    findings = [{"finding_id": "FIND-001", "rationale": "", "tier": "E1", "base_tier": "E2",
                 "extraction_tier": "E2", "regeneration": "REGEN-001", "link_ids": [],
                 "artifact_ids": ["WFPY-001", "WF-002", "IMG-001", "IMG-002"]}]
    raster = {"file": output_path, "sha256": digest, "node_id": "7", "batch_index": 0}
    replay = {"handoff_id": "REGEN-001", "workflow_id": "WFPY-001", "target_manifest_sha256": "manifest",
              "status": "COMPLETED", "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
              "output_rasters": [raster], "derived_regenerated_image": output_path,
              "evaluation": {"selected_raster": raster, "target_artifact_id": "IMG-001",
                             "exact_pixel_match": True, "compared_image_count": 1,
                             "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
                             "evaluation_subtier": "EXACT_MATCH", "evaluated_tier": "E1",
                             "derived_regenerated_image_sha256": digest}}
    proof = {"handoff_id": "REGEN-001", "prompt_id": "first", "regenerated_image": output_path,
             "regenerated_sha256": digest, "length_bytes": 60, "matched_pixels": 20, "reference_pixels": 100,
             "scale": 1}
    scan = {
        "manifest_sha256": "manifest", "memory_sha256": "dump", "matches": [
            {**proof, "physical_offset": 100, "first_reference_pixel": 0},
            {**proof, "physical_offset": 130, "first_reference_pixel": 10},
        ],
    }
    (tmp_path / "replay-search.json").write_text(json.dumps(scan))
    notes = [{"handoff_id": "REGEN-001", "applied_to": "FIND-001"}]

    assert _verified_replay_positions(tmp_path, manifest["findings"]["artifacts"][0], replay) == {("7", 0)}
    physical_spans = [interval(m["physical_offset"], m["length_bytes"]) for m in scan["matches"]]
    pixel_spans = [interval(m["first_reference_pixel"], m["matched_pixels"]) for m in scan["matches"]]
    assert union_size(physical_spans) == 90
    assert sum(stop - start for start, stop in physical_spans) - union_size(physical_spans) == 30
    assert union_size(pixel_spans) == 30

    rep = _Report(tmp_path, manifest, "manifest", findings, [replay], notes, None)
    wb = workbench_data(rep)
    [group] = wb["groups"]
    assert group["submissions"]["promptIds"] == ["first", "second"]
    assert group["copies"] == ["WF-002", "WFPY-001", "WFPY-003"]
    assert group["leadWorkflowId"] == "WFPY-001"
    [shown] = wb["findings"]
    assert [im["id"] for im in shown["images"]] == ["IMG-001", "IMG-002"]
    assert shown["images"][0]["comparison"]["status"] == "EXACT_MATCH"
    assert shown["images"][1]["comparison"]["status"] == "NOT_COMPARED"
