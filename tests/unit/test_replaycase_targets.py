"""replay-case picks each workflow's comparison target over all of its findings, not finding by finding."""

from __future__ import annotations

import pytest

from comfytrace.replaycase import _targets


def _partial(aid: str, rows: int | None, width: int = 64) -> dict:
    metadata = {} if rows is None else {"decoded_rows": rows}
    return {"artifact_id": aid, "carve_type": "PARTIAL_PNG", "completeness": "PARTIAL", "is_derived": False,
            "raw_file_paths": [f"exports/raw/{aid}/span_000.bin"], "derived_preview_path": f"exports/derived/{aid}.png",
            "derived_preview_metadata": metadata, "pixel_metadata": {"width": width, "height": 64}}


def _complete(aid: str, **extra) -> dict:
    return {"artifact_id": aid, "carve_type": "ENCODED_PNG", "completeness": "COMPLETE", "is_derived": False,
            "raw_file_paths": [f"exports/raw/{aid}.png"], "derived_preview_path": None,
            "derived_preview_metadata": None, "pixel_metadata": {"width": 64, "height": 64}, **extra}


def _workflow(wid: str, prompt_id: str | None = "job-a") -> dict:
    return {"workflow_id": wid, "execution_parameters": {"prompt_id": prompt_id} if prompt_id else {}}


def _finding(fid: str, tier: str, ids: list[str], link_ids: list[str] = ()) -> dict:
    return {"finding_id": fid, "tier": tier, "artifact_ids": ids, "link_ids": list(link_ids)}


def _manifest(artifacts, findings, workflows=None, links=()) -> dict:
    return {"findings": {"artifacts": artifacts, "findings": findings,
                         "workflows": workflows or [_workflow("WF-001")], "correlation_links": list(links)}}


@pytest.mark.parametrize("reverse", [False, True])
def test_more_decoded_rows_win_whatever_the_finding_order(reverse):
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-001", "WF-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-002", "WF-001"])]
    manifest = _manifest([_partial("CARVE-PNG-001", 1), _partial("CARVE-PNG-002", 4)],
                         findings[::-1] if reverse else findings)
    selection = _targets(manifest)["WF-001"]
    assert selection["selected_artifact_id"] == "CARVE-PNG-002"
    assert selection["path"] == "exports/derived/CARVE-PNG-002.png"
    assert selection["selection_reason"] == "PARTIAL_MOST_DECODED_ROWS"
    assert selection["candidate_artifact_ids"] == ["CARVE-PNG-002", "CARVE-PNG-001"]
    assert sorted(selection["finding_ids"]) == ["FIND-001", "FIND-002"]


@pytest.mark.parametrize("reverse", [False, True])
def test_an_intact_e2_png_wins_over_a_later_e3_partial(reverse):
    findings = [_finding("FIND-001", "E2", ["CARVE-PNG-009", "WF-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-001", "WF-001"])]
    manifest = _manifest([_complete("CARVE-PNG-009"), _partial("CARVE-PNG-001", 60)],
                         findings[::-1] if reverse else findings)
    selection = _targets(manifest)["WF-001"]
    assert selection["selected_artifact_id"] == "CARVE-PNG-009"
    assert selection["path"] == "exports/raw/CARVE-PNG-009.png"
    assert selection["selection_reason"] == "COMPLETE_NATIVE"


def test_a_direct_png_ranks_before_a_derived_one_and_an_animation_after_partials():
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-001", "CARVE-PNG-002", "CARVE-PNG-003", "WF-001"])]
    artifacts = [_complete("CARVE-PNG-001", is_derived=True), _complete("CARVE-PNG-002"),
                 _complete("CARVE-PNG-003", pixel_metadata={"width": 64, "height": 64, "animated": True})]
    assert _targets(_manifest(artifacts, findings))["WF-001"]["candidate_artifact_ids"] == [
        "CARVE-PNG-002", "CARVE-PNG-001", "CARVE-PNG-003"]
    artifacts[1] = _partial("CARVE-PNG-002", 2)
    assert _targets(_manifest(artifacts, findings))["WF-001"]["candidate_artifact_ids"] == [
        "CARVE-PNG-001", "CARVE-PNG-002", "CARVE-PNG-003"]


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_rows_break_the_tie_by_lowest_artifact_id(reverse):
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-005", "WF-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-003", "WF-001"])]
    manifest = _manifest([_partial("CARVE-PNG-005", 8), _partial("CARVE-PNG-003", 8)],
                         findings[::-1] if reverse else findings)
    assert _targets(manifest)["WF-001"]["selected_artifact_id"] == "CARVE-PNG-003"


def test_zero_row_and_unknown_row_partials_are_never_targets():
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-001", "CARVE-PNG-002", "WF-001"])]
    selection = _targets(_manifest([_partial("CARVE-PNG-001", 0), _partial("CARVE-PNG-002", None)],
                                   findings))["WF-001"]
    assert selection["selected_artifact_id"] is None and selection["path"] is None
    assert selection["selection_reason"] == "NO_COMPARABLE_TARGET"
    assert selection["candidate_artifact_ids"] == []
    assert selection["excluded_candidates"] == [
        {"artifact_id": "CARVE-PNG-001", "reason": "ZERO_DECODED_ROWS"},
        {"artifact_id": "CARVE-PNG-002", "reason": "DECODED_ROWS_UNKNOWN"}]
    findings.append(_finding("FIND-002", "E3", ["CARVE-PNG-003", "WF-001"]))
    selection = _targets(_manifest([_partial("CARVE-PNG-001", 0), _partial("CARVE-PNG-002", None),
                                    _partial("CARVE-PNG-003", 3)], findings))["WF-001"]
    assert selection["selected_artifact_id"] == "CARVE-PNG-003"


def test_findings_outside_e2_and_e3_supply_no_target():
    manifest = _manifest([_complete("CARVE-PNG-001")], [_finding("FIND-001", "E5", ["CARVE-PNG-001", "WF-001"])])
    selection = _targets(manifest)["WF-001"]
    assert selection["selected_artifact_id"] is None and selection["selection_reason"] == "NO_COMPARABLE_TARGET"
    assert selection["excluded_candidates"] == [{"artifact_id": "CARVE-PNG-001", "reason": "PROMPT_TIER_FINDING"}]
    prompt_only = _manifest([], [_finding("FIND-001", "E5", ["WF-001"]),
                                 _finding("FIND-002", "UNTIERED", ["CARVE-PNG-001"])])
    assert _targets(prompt_only) == {}, "a finding with no image has nothing to audit"


def test_zero_rows_are_reported_even_without_a_preview_path():
    header = {**_partial("CARVE-PNG-1439", 0), "derived_preview_path": None}
    no_preview = {**_partial("CARVE-PNG-1440", 5), "derived_preview_path": None}
    manifest = _manifest([header, no_preview],
                         [_finding("FIND-001", "E3", ["CARVE-PNG-1439", "CARVE-PNG-1440", "WF-001"])])
    assert _targets(manifest)["WF-001"]["excluded_candidates"] == [
        {"artifact_id": "CARVE-PNG-1439", "reason": "ZERO_DECODED_ROWS"},
        {"artifact_id": "CARVE-PNG-1440", "reason": "NO_COMPARABLE_RASTER"}]


@pytest.mark.parametrize("reverse", [False, True])
def test_an_image_audited_in_an_e5_finding_is_still_a_target_through_its_e3_finding(reverse):
    findings = [_finding("FIND-001", "E5", ["CARVE-PNG-002", "WF-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-002", "WF-001"])]
    selection = _targets(_manifest([_partial("CARVE-PNG-002", 6)], findings[::-1] if reverse else findings))["WF-001"]
    assert selection["selected_artifact_id"] == "CARVE-PNG-002"
    assert selection["excluded_candidates"] == []
    assert selection["finding_ids"] == ["FIND-001", "FIND-002"]


def test_a_freshly_extracted_zero_row_e5_finding_gets_a_target_selection_audit():
    import json

    from comfytrace.carve import IdAllocator, PngCarver, WorkflowCarver
    from comfytrace.correlate import correlate
    from tests.synthetic import api_prompt, noise_image, png_bytes

    graph = api_prompt(seed=96)
    png = png_bytes(noise_image(64, 48, seed=96), {"prompt": json.dumps(graph)}, idat_chunk=512)
    [header] = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(png[: png.index(b"IDAT") - 4])
    [workflow] = WorkflowCarver().carve_stream(json.dumps(graph).encode())
    result = correlate([header], [workflow])
    [finding] = result.findings
    assert finding.tier == "E5"
    manifest = _manifest([header.to_dict()], [finding.to_dict()], [workflow.to_dict()],
                         [link.to_dict() for link in result.links])
    selection = _targets(manifest)[workflow.workflow_id]
    assert selection["selected_artifact_id"] is None and selection["path"] is None
    assert selection["selection_reason"] == "NO_COMPARABLE_TARGET"
    assert selection["excluded_candidates"] == [{"artifact_id": header.artifact_id, "reason": "ZERO_DECODED_ROWS"}]
    assert selection["finding_ids"] == [finding.finding_id]


def test_repeated_submissions_of_one_graph_keep_their_own_targets():
    workflows = [_workflow("WFPY-001", "job-a"), _workflow("WFPY-002", "job-b")]
    links = [{"link_id": "LINK-001", "workflow_id": "WFPY-001", "artifact_id": "CARVE-PNG-001",
              "matched_keys": ["PROMPT_ID"]},
             {"link_id": "LINK-002", "workflow_id": "WFPY-002", "artifact_id": "CARVE-PNG-002",
              "matched_keys": ["PROMPT_ID"]}]
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-001", "WFPY-001"], ["LINK-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-002", "WFPY-002"], ["LINK-002"])]
    targets = _targets(_manifest([_partial("CARVE-PNG-001", 2), _partial("CARVE-PNG-002", 40)], findings,
                                 workflows, links))
    assert targets["WFPY-001"]["selected_artifact_id"] == "CARVE-PNG-001"
    assert targets["WFPY-002"]["selected_artifact_id"] == "CARVE-PNG-002"


def test_an_image_linked_to_another_submission_is_excluded():
    workflows = [_workflow("WFPY-001", "job-a"), _workflow("WFPY-002", "job-b"), _workflow("WFPNG-001", None)]
    links = [{"link_id": "LINK-001", "workflow_id": "WFPY-002", "artifact_id": "CARVE-PNG-001",
              "matched_keys": ["PROMPT_ID"]},
             {"link_id": "LINK-002", "workflow_id": "WFPNG-001", "artifact_id": "CARVE-PNG-002",
              "matched_keys": ["EXECUTION_GRAPH_MATCH", "INPUT_REVISION_UNVERIFIED"]}]
    findings = [_finding("FIND-001", "E3", ["CARVE-PNG-001", "WFPY-002", "WFPY-001"], ["LINK-001"]),
                _finding("FIND-002", "E3", ["CARVE-PNG-002", "WFPNG-001", "WFPY-001"], ["LINK-002"])]
    targets = _targets(_manifest([_partial("CARVE-PNG-001", 30), _partial("CARVE-PNG-002", 20)], findings,
                                 workflows, links))
    selection = targets["WFPY-001"]
    assert selection["selected_artifact_id"] is None
    assert selection["excluded_candidates"] == [
        {"artifact_id": "CARVE-PNG-001", "reason": "PROMPT_ID_CONFLICT"},
        {"artifact_id": "CARVE-PNG-002", "reason": "INPUT_REVISION_UNVERIFIED"}]
    assert targets["WFPY-002"]["selected_artifact_id"] == "CARVE-PNG-001"
    assert targets["WFPNG-001"]["selected_artifact_id"] == "CARVE-PNG-002"
