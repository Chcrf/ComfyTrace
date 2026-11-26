"""A job whose own history ends interrupted or in error with explicitly empty outputs earns no output credit.

Synthetic QueueItem histories, a noise raster and a throw-away case directory; nothing is recorded or remote.
"""

from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
from typing import Any

import numpy as np
import pytest
from PIL import Image

from comfytrace import evidence, link, memrecover
from comfytrace.carve import IdAllocator
from comfytrace.memrecover import (
    OUTPUT_CONTEXT_FIELD,
    TERMINAL_WITHOUT_RECORDED_OUTPUTS,
    terminal_without_outputs_ids,
    terminal_without_outputs_jobs,
)
from comfytrace.models import CorrelationLink
from comfytrace.pyheap import QueueItem
from tests.synthetic import api_prompt, noise_image

DONE, CUT, OTHER = ("5c1d6b2e-7a41-4c1b-8f0e-2a9d3c4b5e61", "e8a0b9f3-2d6c-4a7e-b1c5-9f4e3d2c1b0a",
                    "0b9b6a52-6d1c-4d63-8a53-3f2f7a8f4d10")
START, STOP = 1794065346982, 1794065421254
SHA = "ab" * 32
GRAPH = api_prompt()
_addresses = itertools.count(0x1000, 0x1000)
IMAGE = {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}


def _history(prompt_id: str, event: str, status_str: str, completed: bool | None, outputs: Any) -> dict[str, Any]:
    messages = [["execution_start", {"prompt_id": prompt_id, "timestamp": START}],
                [event, {"prompt_id": prompt_id, "timestamp": STOP}]]
    status: dict[str, Any] = {"status_str": status_str, "messages": messages}
    if completed is not None:
        status["completed"] = completed
    history: dict[str, Any] = {"status": status}
    if outputs is not None:
        history["outputs"] = outputs
    return history


def interrupted(prompt_id: str = CUT, outputs: Any = None) -> dict[str, Any]:
    return _history(prompt_id, "execution_interrupted", "error", False, {} if outputs is None else outputs)


def success(prompt_id: str = DONE) -> dict[str, Any]:
    return _history(prompt_id, "execution_success", "success", True, {"9": {"images": [IMAGE]}})


def item(prompt_id: str, history: dict[str, Any] | None, workflow: dict[str, Any] | None = None) -> QueueItem:
    extra = {"create_time": START - 80, **({"extra_pnginfo": {"workflow": workflow}} if workflow else {})}
    queue_item = QueueItem(4242, prompt_id, 0, next(_addresses), 1, 5, copy.deepcopy(GRAPH), extra)
    queue_item.history, queue_item.owner = history, "history" if history is not None else ""
    return queue_item


# ------------------------------------------------------------------------------------------- the shared helper


@pytest.mark.parametrize("histories, marked", [
    pytest.param([interrupted()], True, id="interrupted"),
    pytest.param([_history(CUT, "execution_error", "error", False, {})], True, id="execution error"),
    pytest.param([None, interrupted(), None], True, id="bare copies cannot contradict a reliable history"),
    pytest.param([None], False, id="bare copy alone is no evidence"),
    pytest.param([{}], False, id="empty history is no evidence"),
    pytest.param([_history(CUT, "execution_interrupted", "error", False, None)], False, id="outputs not stated"),
    pytest.param([_history(CUT, "execution_interrupted", "error", False, [])], False, id="outputs unreadable"),
    pytest.param([_history(CUT, "execution_interrupted", "error", None, {})], False, id="completed not stated"),
    pytest.param([_history(CUT, "execution_interrupted", "error", True, {})], False, id="completed is true"),
    pytest.param([_history(CUT, "execution_interrupted", "success", False, {})], False, id="success status"),
    pytest.param([_history(CUT, "execution_interrupted", "running", False, {})], False, id="other status"),
    pytest.param([_history(CUT, "execution_success", "error", False, {})], False, id="success message"),
    pytest.param([{"status": {"status_str": "error", "completed": False, "messages": []}, "outputs": {}}], False,
                 id="no terminal message"),
    pytest.param([{"outputs": {}}], False, id="no status"),
    pytest.param([interrupted(outputs={"9": {"images": [IMAGE]}})], False, id="cancelled with a recorded image"),
    pytest.param([interrupted(outputs={"9": {"text": ["x"]}})], False, id="cancelled with other recorded output"),
    pytest.param([interrupted(), interrupted(outputs={"9": {"images": [IMAGE]}})], False,
                 id="a second copy lists an output"),
    pytest.param([interrupted(), _history(CUT, "execution_success", "success", True, {})], False,
                 id="a second copy shows success"),
    pytest.param([interrupted(), {"outputs": {}}], True, id="an unreadable second copy is unknown"),
])
def test_only_a_readable_terminal_history_with_empty_outputs_marks_a_job(histories, marked):
    items = [item(CUT, history) for history in histories]
    assert terminal_without_outputs_jobs(items) == ({CUT} if marked else set())


def test_a_job_is_judged_by_its_own_copies_only():
    items = [item(CUT, interrupted()), item(DONE, success()), item(OTHER, None)]
    assert terminal_without_outputs_jobs(items) == {CUT}


def test_the_field_is_persisted_on_every_workflow_copy_of_the_job_and_read_back():
    ui = {"last_node_id": 9, "nodes": [], "links": None, "version": 0.4}
    items = [item(CUT, None, ui), item(CUT, interrupted(), ui), item(DONE, success(), ui), item(OTHER, None)]
    records = memrecover.workflow_records(items, IdAllocator("WFPY"))
    by_job: dict[str, list[Any]] = {}
    for record in records:
        by_job.setdefault(record.execution_parameters["prompt_id"], []).append(record)
    assert {r.prompt_format for r in by_job[CUT]} == {"API_GRAPH", "UI_WORKFLOW"}
    assert all(r.execution_parameters[OUTPUT_CONTEXT_FIELD] == TERMINAL_WITHOUT_RECORDED_OUTPUTS
               for r in by_job[CUT])
    # The graph of the completed and the ordinary job keeps its old parameters, byte for byte.
    assert all(OUTPUT_CONTEXT_FIELD not in r.execution_parameters for r in by_job[DONE] + by_job[OTHER])
    assert terminal_without_outputs_ids(r.execution_parameters for r in records) == {CUT}


def test_a_manifest_without_the_positive_field_keeps_the_old_behavior():
    assert terminal_without_outputs_ids([{"prompt_id": CUT}, {"prompt_id": CUT, "history_status": "error",
                                                              "history_completed": False}]) == set()
    marked = {"prompt_id": CUT, OUTPUT_CONTEXT_FIELD: TERMINAL_WITHOUT_RECORDED_OUTPUTS, "history_status": "error"}
    assert terminal_without_outputs_ids([marked, {"prompt_id": CUT}]) == {CUT}
    for conflicting in ({"output_filenames": ["a.png"]}, {"history_completed": True}, {"history_status": "success"},
                        {OUTPUT_CONTEXT_FIELD: "SOMETHING_ELSE"}, {"job_timeline": {"outcome": "success"}}):
        assert terminal_without_outputs_ids([marked, {"prompt_id": CUT, **conflicting}]) == set()
    assert terminal_without_outputs_ids([{**marked, "output_filenames": ["a.png"]}]) == set()


# ---------------------------------------------------------------------------------------------- the scorecard


def _links(*workflows_and_artifacts: tuple[str, str]) -> list[CorrelationLink]:
    return [CorrelationLink(f"LINK-{n:03d}", workflow_id, artifact_id, "STRICT", ["CANONICAL_PROMPT_HASH"], "")
            for n, (workflow_id, artifact_id) in enumerate(workflows_and_artifacts, 1)]


def _card(items: list[QueueItem], artifact_for: dict[str, str]) -> dict[str, Any]:
    workflows = memrecover.workflow_records(items, IdAllocator("WFPY"))
    api = {w.execution_parameters["prompt_id"]: w.workflow_id for w in workflows if w.prompt_format == "API_GRAPH"}
    return evidence.scorecard(items, workflows, SHA, _links(*((api[pid], art) for pid, art in artifact_for.items())))


def test_a_job_without_recorded_outputs_gets_no_image_credit_but_keeps_its_matches_as_context():
    card = _card([item(CUT, interrupted())], {CUT: "CARVE-PNG-001"})
    [row] = card["rows"]
    assert row[OUTPUT_CONTEXT_FIELD] == TERMINAL_WITHOUT_RECORDED_OUTPUTS and row["output_node_id"] is None
    assert row["image"]["status"] == "NOT_LINKED" and row["image"]["candidates"] == []
    [context] = row["image"]["contextual_observations"]
    assert (context["artifact_id"], context["role"], context["basis"]) == (
        "CARVE-PNG-001", "CONTEXT_NOT_OUTPUT", "CORRELATION_STRICT")
    assert row["api_prompt"]["status"] == "RECOVERED", "the graph stays recovered"
    assert card["summary"]["jobs_with_linked_image"] == 0 and card["summary"]["outputs_with_linked_image"] == 0
    assert card["summary"]["jobs_terminal_without_recorded_outputs"] == 1


def test_a_completed_alternative_keeps_its_credit_beside_an_interrupted_twin():
    card = _card([item(DONE, success()), item(CUT, interrupted())], {DONE: "CARVE-PNG-001", CUT: "CARVE-PNG-001"})
    rows = {row["prompt_id"]: row for row in card["rows"]}
    assert rows[DONE]["output_node_id"] == "9" and rows[DONE]["image"]["status"] == "JOB_LINKED"
    assert [c["artifact_id"] for c in rows[DONE]["image"]["candidates"]] == ["CARVE-PNG-001"]
    assert OUTPUT_CONTEXT_FIELD not in rows[DONE]
    assert rows[CUT]["image"]["status"] == "NOT_LINKED" and rows[CUT]["image"]["candidates"] == []
    assert card["summary"]["jobs_with_linked_image"] == 1 and card["summary"]["outputs"] == 1


def test_a_cancelled_job_with_a_recorded_output_stays_visible_and_credited():
    cancelled = item(CUT, interrupted(outputs={"9": {"images": [IMAGE]}}))
    [row] = _card([cancelled], {CUT: "CARVE-PNG-001"})["rows"]
    assert (row["output_node_id"], row["output_filename"]) == ("9", IMAGE["filename"])
    assert OUTPUT_CONTEXT_FIELD not in row
    assert row["image"]["status"] == "JOB_LINKED" and [c["artifact_id"] for c in row["image"]["candidates"]] == [
        "CARVE-PNG-001"]
    [record] = memrecover.workflow_records([cancelled], IdAllocator("WFPY"))
    assert OUTPUT_CONTEXT_FIELD not in record.execution_parameters
    assert record.execution_parameters["output_filenames"] == [IMAGE["filename"]]
    assert record.execution_parameters["job_timeline"]["outcome"] == "interrupted"


def test_ordinary_jobs_without_history_are_unchanged():
    [row] = _card([item(OTHER, None)], {OTHER: "CARVE-PNG-001"})["rows"]
    assert OUTPUT_CONTEXT_FIELD not in row and set(row["image"]) == {"status", "candidates"}
    assert row["image"]["status"] == "JOB_LINKED"
    card = _card([item(OTHER, None)], {})
    assert "jobs_terminal_without_recorded_outputs" not in card["summary"]


def test_image_links_never_credit_a_job_without_recorded_outputs_and_apply_twice_alike():
    card = _card([item(DONE, success()), item(CUT, interrupted())], {DONE: "CARVE-PNG-001", CUT: "CARVE-PNG-001"})
    context = {"artifact_id": "BITMAP-001", "kind": "PIXEL_BUFFER", "agreement": "ROWS", "complete": True,
               "relation": "WHOLE_FRAME_GRAPH_MATCH", "output_credit": "NONE"}
    stale = {"prompt_id": CUT, "image_status": "OBSERVED_EXACT",  # a links file made before the defense
             "observed": [{"artifact_id": "BITMAP-002", "kind": "PIXEL_BUFFER", "agreement": "ROWS"}]}
    links = {"manifest_sha256": "cd" * 32, "jobs": [
        {**stale, "contextual_observed": [context]}, {"prompt_id": DONE, "image_status": "REGENERATED_ONLY",
                                                      "observed": []}]}
    sources = {"BITMAP-001": "MEMORY"}
    once = evidence.apply_image_links(copy.deepcopy(card), links, sources)
    assert once == evidence.apply_image_links(copy.deepcopy(once), links, sources)
    rows = {row["prompt_id"]: row for row in once["rows"]}
    image = rows[CUT]["image"]
    assert image["status"] == "NOT_LINKED" and image["candidates"] == []
    assert {c["artifact_id"] for c in image["contextual_observations"]} == {"CARVE-PNG-001", "BITMAP-001"}
    assert image["regeneration"]["image_status"] == "OBSERVED_EXACT"
    assert once["summary"]["jobs_with_linked_image"] == 1, "only the completed job"
    assert once["summary"]["outputs_with_linked_memory_image"] == 0


# --------------------------------------------------------------------------------------------- the link layer


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _case(root, workflows: dict[str, dict[str, Any]], memory: dict[str, np.ndarray],
          regens: list[tuple[str, np.ndarray]]):
    """``workflows`` maps workflow id to its execution parameters, as ``workflow_records`` wrote them."""
    (root / "raw").mkdir(parents=True)
    (root / "regenerations").mkdir()
    (root / "exports" / "derived").mkdir(parents=True)
    artifacts, records = [], []
    for artifact_id, rgb in memory.items():
        raw = rgb.tobytes()
        (root / "raw" / f"{artifact_id}.bin").write_bytes(raw)
        artifacts.append({"artifact_id": artifact_id, "carve_type": "PIXEL_BUFFER", "completeness": "COMPLETE",
                          "raw_file_paths": [f"raw/{artifact_id}.bin"], "gaps": [], "link_status": "UNASSOCIATED",
                          "pixel_sha256": _sha(raw), "container_sha256": None,
                          "pixel_metadata": {"width": rgb.shape[1], "height": rgb.shape[0],
                                             "source": {"channels": 3}}})
        records.append({"record_id": artifact_id, "sha256": {"raw": _sha(raw)}})
    manifest = json.dumps({"findings": {"artifacts": artifacts, "workflows": [
        {"workflow_id": wid, "prompt_format": "API_GRAPH", "completeness": "COMPLETE", "execution_parameters": params}
        for wid, params in workflows.items()]}}).encode()
    (root / "manifest.json").write_bytes(manifest)
    (root / "recovery-evidence.json").write_text(json.dumps({"records": records}))
    for number, (workflow_id, rgb) in enumerate(regens, 1):
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        data = buffer.getvalue()
        name = f"exports/derived/REGEN-{number:03d}.png"
        (root / name).write_bytes(data)
        raster = {"node_id": "9", "batch_index": 0, "file": name, "sha256": _sha(data)}
        (root / "regenerations" / f"REGEN-{number:03d}.json").write_text(json.dumps({
            "handoff_id": f"HANDOFF-{number:03d}", "workflow_id": workflow_id, "status": "COMPLETED",
            "target_manifest_sha256": _sha(manifest), "derived_regenerated_image": name,
            "evaluation": {"derived_regenerated_image_sha256": _sha(data), "compared_image": raster},
            "output_rasters": [raster],
            "preflight_check": {"status": "PASSED", "runner_object_info_verified": True,
                                "model_match_level": "SHA256", "model_sha256": "cd" * 32}}))
    return root


def _params(items: list[QueueItem]) -> dict[str, dict[str, Any]]:
    """Recorded parameters of each job's API workflow, named WF-001... in the order given."""
    records = [r for r in memrecover.workflow_records(items, IdAllocator("WFPY")) if r.prompt_format == "API_GRAPH"]
    return {f"WF-{n:03d}": r.execution_parameters for n, r in enumerate(records, 1)}


FRAME = np.asarray(noise_image(64, 48, seed=1))
PREFIX = np.ascontiguousarray(FRAME[:24])
UNRELATED = np.asarray(noise_image(64, 48, seed=2))


def _linked(tmp_path, items, memory, regens):
    params = _params(items)
    return link.link(_case(tmp_path / "case", params, memory, regens))


def _attribution(document, artifact_id="BITMAP-001"):
    return next(c for c in document["memory_images"] if c["artifact_id"] == artifact_id)


def test_exact_replay_of_a_job_without_recorded_outputs_credits_nothing(tmp_path):
    document = _linked(tmp_path, [item(CUT, interrupted())], {"BITMAP-001": FRAME}, [("WF-001", FRAME)])
    candidate = _attribution(document)
    assert (candidate["attribution"], candidate["prompt_id"], candidate["agreeing_jobs"]) == (
        "UNATTRIBUTED", None, [])
    assert candidate["output_attribution"] == "UNATTRIBUTED" and candidate["selected_raster"] is None
    [match] = candidate["contextual_matches"]
    assert (match["relation"], match["output_credit"], match["prompt_id"]) == (
        "WHOLE_FRAME_GRAPH_MATCH", "NONE", CUT)
    [job] = document["jobs"]
    assert (job["image_status"], job["observed"], job["decoded_latents"]) == ("REGENERATED_ONLY", [], [])
    assert job[OUTPUT_CONTEXT_FIELD] == TERMINAL_WITHOUT_RECORDED_OUTPUTS
    assert [c["artifact_id"] for c in job["contextual_observed"]] == ["BITMAP-001"]
    assert document["promotions"] == []
    [regenerated] = document["regenerated"]
    assert regenerated["residue_check"] == "CONTEXT_ONLY" and regenerated["agreeing_memory_images"] == []
    assert document["terminal_without_recorded_outputs"]["prompt_ids"] == [CUT]


def test_partial_replay_of_a_job_without_recorded_outputs_credits_nothing(tmp_path):
    document = _linked(tmp_path, [item(CUT, interrupted())], {"BITMAP-001": PREFIX}, [("WF-001", FRAME)])
    candidate = _attribution(document)
    assert candidate["attribution"] == "UNATTRIBUTED" and candidate["prompt_id"] is None
    assert [m["relation"] for m in candidate["contextual_matches"]] == ["PARTIAL_OR_SCALED_GRAPH_MATCH"]
    assert document["promotions"] == [] and document["jobs"][0]["observed"] == []


def test_an_unrelated_image_leaves_a_job_without_outputs_with_no_context_at_all(tmp_path):
    document = _linked(tmp_path, [item(CUT, interrupted())], {"BITMAP-001": UNRELATED}, [("WF-001", FRAME)])
    assert "contextual_matches" not in _attribution(document)
    assert document["jobs"][0]["contextual_observed"] == []
    assert document["regenerated"][0]["residue_check"] == "UNCHECKED"


def test_a_completed_alternative_keeps_its_credit_and_its_promotion(tmp_path):
    items = [item(DONE, success()), item(CUT, interrupted())]
    both = _linked(tmp_path, items, {"BITMAP-001": FRAME}, [("WF-001", FRAME), ("WF-002", FRAME)])
    candidate = _attribution(both)
    assert (candidate["attribution"], candidate["prompt_id"], candidate["agreeing_jobs"]) == (
        "SUPPORTED", DONE, [DONE])
    jobs = {job["prompt_id"]: job for job in both["jobs"]}
    assert jobs[DONE]["image_status"] == "OBSERVED_EXACT" and jobs[CUT]["observed"] == []
    assert [p["prompt_id"] for p in both["promotions"]] == [DONE]
    assert [m["prompt_id"] for m in candidate["contextual_matches"]] == [CUT]


def test_a_manifest_without_the_field_keeps_the_old_ambiguity_and_credit(tmp_path):
    bare_params = {wid: {"prompt_id": params["prompt_id"]} for wid, params in
                   _params([item(DONE, success()), item(CUT, interrupted())]).items()}
    twins = link.link(_case(tmp_path / "twins", bare_params, {"BITMAP-001": FRAME},
                            [("WF-001", FRAME), ("WF-002", FRAME)]))
    assert _attribution(twins)["attribution"] == "AMBIGUOUS" and twins["promotions"] == []
    alone = link.link(_case(tmp_path / "alone", {"WF-002": bare_params["WF-002"]}, {"BITMAP-001": FRAME},
                            [("WF-002", FRAME)]))
    assert _attribution(alone)["prompt_id"] == CUT and [p["tier"] for p in alone["promotions"]] == ["E2"]
    assert "terminal_without_recorded_outputs" not in alone


def test_a_cancelled_job_with_a_recorded_output_is_still_linked_and_promoted(tmp_path):
    cancelled = item(CUT, interrupted(outputs={"9": {"images": [IMAGE]}}))
    document = _linked(tmp_path, [cancelled], {"BITMAP-001": FRAME}, [("WF-001", FRAME)])
    candidate = _attribution(document)
    assert (candidate["attribution"], candidate["prompt_id"]) == ("SUPPORTED", CUT)
    assert document["jobs"][0]["image_status"] == "OBSERVED_EXACT"
    assert [p["tier"] for p in document["promotions"]] == ["E2"]
    assert "contextual_matches" not in candidate and "terminal_without_recorded_outputs" not in document


def test_a_workflow_of_a_job_without_outputs_is_not_required_to_be_replayed(tmp_path):
    params = _params([item(DONE, success()), item(CUT, interrupted())])
    case = _case(tmp_path / "case", params, {"BITMAP-001": FRAME}, [("WF-001", FRAME)])
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["findings"]["artifacts"][0].update({"link_status": "AMBIGUOUS",
                                                 "ambiguous_candidate_workflows": ["WF-001", "WF-002"]})
    (case / "manifest.json").write_text(json.dumps(manifest))
    for record in (case / "regenerations").glob("REGEN-*.json"):
        data = json.loads(record.read_text())
        data["target_manifest_sha256"] = _sha((case / "manifest.json").read_bytes())
        record.write_text(json.dumps(data))
    candidate = _attribution(link.link(case))
    assert candidate["unreplayed_workflow_ids"] == [] and candidate["prompt_id"] == DONE
