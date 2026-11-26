import hashlib
import json

import numpy as np
import pytest

from comfytrace.cli import parse_args, run_search_replay
from tests.unit.test_link import _case, _case_regeneration

pytest.importorskip("ahocorasick_rs")


def test_search_verifies_raw_frames_and_discards_decoy_anchors(tmp_path):
    rng = np.random.default_rng(9)
    reference = rng.integers(30, 220, (48, 64, 3), dtype=np.uint8)
    observed = reference.copy()
    observed[12:] += 1
    # The physical dump contains a full copy, a fragment on a separate page, and an anchor alone.
    payload = bytearray(7 * 4096)
    frame_offset = 83
    frame = observed.tobytes()
    payload[frame_offset:frame_offset + len(frame)] = frame
    fragment_offset = 4 * 4096
    fragment = reference.tobytes()[3072:7167]
    payload[fragment_offset:fragment_offset + len(fragment)] = fragment
    payload[6 * 4096:6 * 4096 + 18] = reference.tobytes()[:18]
    memory = tmp_path / "dump.raw"
    memory.write_bytes(payload)
    case = _case(tmp_path, [])
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["case_metadata"]["memory_image"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", reference)
    before = manifest_path.read_bytes()

    args = parse_args(["search-replay", "--case-dir", str(case), "--memory", str(memory)])
    assert run_search_replay(args) == 0
    result = json.loads((case / "replay-search.json").read_bytes())
    assert manifest_path.read_bytes() == before
    assert result["manifest_sha256"] == hashlib.sha256(before).hexdigest()
    hits = [record for record in result["matches"] if record["scale"] == 1 and record["layout"] == "RGB"]
    assert [(h["physical_offset"], h["whole_frame"]) for h in hits] == [(frame_offset, True), (fragment_offset, False)]
    assert [h["length_bytes"] for h in hits] == [len(frame), len(fragment)]
    for hit in hits:
        start = hit["physical_offset"]
        assert hashlib.sha256(payload[start:start + hit["length_bytes"]]).hexdigest() == hit["raw_sha256"]

    memory.write_bytes(bytes(len(payload)))
    assert run_search_replay(args) == 1
    assert json.loads((case / "replay-search.json").read_bytes()) == result


def test_a_graph_without_a_prompt_id_locates_pixels_but_names_no_producer(tmp_path):
    from comfytrace.link import case_regenerations

    rng = np.random.default_rng(11)
    reference = rng.integers(30, 220, (48, 64, 3), dtype=np.uint8)
    payload = bytearray(4 * 4096)
    payload[200:200 + reference.nbytes] = reference.tobytes()
    memory = tmp_path / "dump.raw"
    memory.write_bytes(payload)
    case = _case(tmp_path, [])
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["case_metadata"]["memory_image"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["findings"]["workflows"] = [{
        "workflow_id": "WFPNG-001", "encoding_detected": "PNG_TEXT", "prompt_format": "API_GRAPH",
        "completeness": "COMPLETE", "canonical_json_sha256": "c" * 64, "execution_parameters": {}}]
    manifest_path.write_text(json.dumps(manifest))
    # A graph read from a PNG's text chunk: complete, replayable, and without an original prompt ID.
    _case_regeneration(case, "REGEN-001", "WFPNG-001", None, reference)

    assert case_regenerations(case) == [], "linking still needs a recovered prompt ID"
    assert [r["prompt_id"] for r in case_regenerations(case, unbound=True)] == [None]
    args = parse_args(["search-replay", "--case-dir", str(case), "--memory", str(memory)])
    assert run_search_replay(args) == 0
    result = json.loads((case / "replay-search.json").read_bytes())
    assert result["unbound_workflows"] == ["WFPNG-001"]
    whole = [h for h in result["matches"] if h["whole_frame"] and h["scale"] == 1 and h["layout"] == "RGB"]
    assert [(h["physical_offset"], h["prompt_id"], h["producer"]) for h in whole] == [(200, None, "UNRESOLVED")]

    from comfytrace.replayevidence import apply_replay_evidence
    from comfytrace.report import load_regenerations

    manifest_bytes = manifest_path.read_bytes()
    findings = [{"finding_id": "F-1", "artifact_ids": ["WFPNG-001"], "tier": "E3", "base_tier": "E3"}]
    notes = apply_replay_evidence(findings, json.loads(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest(),
                                  case, load_regenerations(case / "regenerations"))
    assert notes == [] and findings[0]["tier"] == "E3", "an unbound hit promotes nothing"


def test_replayable_prompts_add_unseen_complete_graphs_after_the_backend_prompts():
    from comfytrace.regenerate import recovered_prompts, replayable_prompts

    def wf(workflow_id, encoding, digest, completeness="COMPLETE", fmt="API_GRAPH"):
        return {"workflow_id": workflow_id, "encoding_detected": encoding, "prompt_format": fmt,
                "completeness": completeness, "canonical_json_sha256": digest,
                "raw_file_path": f"exports/raw/{workflow_id}.json"}

    manifest = {"findings": {"workflows": [
        wf("WFPY-001", "CPYTHON_OBJECTS", "a"), wf("WFPNG-001", "PNG_TEXT", "a"), wf("WFPNG-002", "PNG_TEXT", "b"),
        wf("WF-003", "UTF-8", "b"), wf("WF-004", "UTF-8", "c", "PARTIAL"),
        wf("WF-005", "UTF-8", "d", fmt="UI_WORKFLOW"),
    ]}}
    assert [w["workflow_id"] for w in recovered_prompts(manifest)] == ["WFPY-001"]
    assert [w["workflow_id"] for w in replayable_prompts(manifest)] == ["WFPY-001", "WFPNG-002"]
    # A copy naming its submission wins over an earlier anonymous copy of the same graph.
    manifest["findings"]["workflows"][3]["execution_parameters"] = {"prompt_id": "job-b"}
    assert [w["workflow_id"] for w in replayable_prompts(manifest)] == ["WFPY-001", "WF-003"]
    # A backend prompt without its exported graph cannot be replayed.
    manifest["findings"]["workflows"][0]["raw_file_path"] = ""
    assert [w["workflow_id"] for w in replayable_prompts(manifest)][0] != "WFPY-001"
