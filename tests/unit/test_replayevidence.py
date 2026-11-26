import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from comfytrace.cli import parse_args, run_search_replay
from comfytrace.report import effective_findings, load_regenerations
from tests.unit.test_link import _case, _case_regeneration

pytest.importorskip("ahocorasick_rs")


@pytest.mark.parametrize("ambiguous", [False, True])
def test_whole_frame_search_earns_e1_only_for_unique_dump_bytes(tmp_path, ambiguous):
    reference = np.random.default_rng(5).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    offset = 89
    memory = tmp_path / "memory.raw"
    memory.write_bytes(bytes(offset) + reference.tobytes() + bytes(1024))
    case = _case(tmp_path, [])
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["case_metadata"]["memory_image"]["sha256"] = hashlib.sha256(memory.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", reference)
    if ambiguous:
        _case_regeneration(case, "REGEN-002", "WFPY-002", "job-b", reference)
    manifest = json.loads(manifest_path.read_bytes())
    workflows = manifest["findings"]["workflows"]
    for workflow in workflows:
        workflow.update(encoding_detected="CPYTHON_OBJECTS", prompt_format="API_GRAPH", completeness="COMPLETE")
    manifest["findings"]["findings"] = [
        {"finding_id": f"FIND-{i}", "tier": "E5", "artifact_ids": [w["workflow_id"]]}
        for i, w in enumerate(workflows)
    ]
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    records_dir = case / "regenerations"
    for record_path in records_dir.glob("*.json"):
        record = json.loads(record_path.read_bytes())
        record["target_manifest_sha256"] = manifest_sha
        record_path.write_text(json.dumps(record))

    args = parse_args(["search-replay", "--case-dir", str(case), "--memory", str(memory)])
    assert run_search_replay(args) == 0
    records = load_regenerations(records_dir)
    findings, _ = effective_findings(manifest, records, manifest_sha, case)
    if ambiguous:
        assert [f["tier"] for f in findings] == ["E5", "E5"]
    else:
        (finding,) = findings
        assert finding["tier"] == "E1"
        assert finding["extraction_tier"] == "E5" and finding["base_tier"] == "E2"
        evidence = finding["replay_evidence"]
        assert evidence["physical_offset"] == offset and evidence["subtier"] == "EXACT_MATCH"
        assert (case / Path(evidence["raw_file"])).read_bytes() == reference.tobytes()
        sidecar = case / "replay-search.json"
        scan = json.loads(sidecar.read_bytes())
        for hit in scan["matches"]:
            hit["physical_offset"] += 1
        sidecar.write_text(json.dumps(scan))
        altered, _ = effective_findings(manifest, records, manifest_sha, case)
        assert altered[0]["tier"] == "E5"
