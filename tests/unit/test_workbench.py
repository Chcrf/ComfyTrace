"""The interactive workbench is ComfyTrace's single offline HTML report, built from effective findings."""

from __future__ import annotations

import json
import re

import pytest

from comfytrace.cli import parse_args, run_report
from comfytrace.report import tier_counts
from comfytrace.workbench import render_workbench_html
from tests.unit.test_report import _case


def _data(page: str) -> dict:
    match = re.search(r'<script id="ct-data" type="application/json">(.*?)</script>', page, re.S)
    assert match, "the case data block is embedded"
    return json.loads(match.group(1))


def test_extract_writes_a_self_contained_workbench_beside_the_report(tmp_path, mock_memory_image, fake_ingestor,
                                                                     symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir, "T1")
    assert not (case / "report.html").exists(), "report.html generation is eradicated"
    page = (case / "workbench.html").read_text(encoding="utf-8")
    assert "__COMFYTRACE_" not in page, "every template mark is filled"
    assert not re.search(r'(?:src|href)="https?://', page), "nothing is fetched from the network"
    data = _data(page)
    manifest = json.loads((case / "manifest.json").read_text())
    counted = {t: n for t, n in tier_counts(manifest["findings"]["findings"]).items() if t.startswith("E")}
    assert {t: v["count"] for t, v in data["tiers"].items() if t != "UT"} == {
        t: counted.get(t, 0) for t in ("E1", "E2", "E3", "E4", "E5", "E6")}
    assert len(data["findings"]) == sum(v["count"] for t, v in data["tiers"].items() if t != "UT")
    assert data["case"]["dumpSha256"] == manifest["case_metadata"]["memory_image"]["sha256"]
    tiers = [f["tier"] for f in data["findings"]]
    assert tiers == sorted(tiers), "strongest findings first"


def test_rerendering_the_report_refreshes_the_workbench(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir, "T1")
    (case / "workbench.html").unlink()
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    assert (case / "workbench.html").is_file()


def test_case_text_cannot_end_the_data_block_early():
    data = {"case": {"caseId": "<b>case</b>", "dumpName": "x.raw"}, "tiers": {},
            "findings": [{"rationale": "</script><script>alert(1)</script>"}], "untiered": []}
    page = render_workbench_html(data)
    assert "</script><script>alert(1)" not in page
    assert _data(page)["findings"][0]["rationale"] == "</script><script>alert(1)</script>"
    assert "<b>case</b>" not in page.split("<body>")[0], "the title is escaped"


def test_e3_finding_shows_executable_reproduction_without_downgrading(tmp_path):
    from io import BytesIO

    from PIL import Image

    from comfytrace.report import sha256_file
    from comfytrace.workbench import write_workbench

    derived = tmp_path / "exports" / "derived"
    derived.mkdir(parents=True)
    buf = BytesIO()
    Image.new("RGB", (16, 16), (40, 80, 120)).save(buf, format="PNG")
    regen_png = derived / "DERIVED_regen.png"
    regen_png.write_bytes(buf.getvalue())
    regen_sha = sha256_file(regen_png)

    manifest = {
        "schema_version": "1.0.0",
        "case_metadata": {"case_id": "CASE-E3", "comfytrace_version": "1.0.0", "carving_mode": "PHYSICAL_FALLBACK",
                          "memory_image": {"path": "dump.raw", "size_bytes": 1024, "sha256": "a" * 64}},
        "findings": {
            "artifacts": [{"artifact_id": "IMG-001", "carve_type": "PARTIAL_PNG", "completeness": "PARTIAL",
                           "raw_file_paths": [], "gaps": []}],
            "workflows": [{"workflow_id": "WF-001", "prompt_format": "API_GRAPH", "completeness": "COMPLETE",
                           "execution_parameters": {"seed": 42}}],
            "correlation_links": [],
            "findings": [{"finding_id": "FIND-001", "tier": "E3", "artifact_ids": ["IMG-001", "WF-001"],
                          "link_ids": [], "rationale": "Partial image linked to complete workflow"}],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)

    regens = tmp_path / "regenerations"
    regens.mkdir()
    regen_doc = {
        "handoff_id": "REGEN-001",
        "workflow_id": "WF-001",
        "target_manifest_sha256": manifest_sha,
        "status": "COMPLETED",
        "derived_regenerated_image": "exports/derived/DERIVED_regen.png",
        "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
        "security_audit": {"model_format": "SAFETENSORS"},
        "evaluation": {
            "promoted_finding_tier": None,
            "evaluated_tier": "E4",
            "evaluation_mode": "NO_MEMORY_TARGET",
            "evaluation_subtier": "EXECUTABLE_REPRODUCTION",
            "target_artifact_id": None,
            "derived_regenerated_image_sha256": regen_sha,
        },
    }
    (regens / "REGEN-001.json").write_text(json.dumps(regen_doc), encoding="utf-8")

    wb = write_workbench(tmp_path)
    data = _data(wb.read_text(encoding="utf-8"))
    f = data["findings"][0]
    assert f["tier"] == "E3", "E3 finding must never downgrade to E4"
    assert f["replay"] is not None
    assert f["replay"]["id"] == "REGEN-001"
    assert f["replay"]["result"] == "EXECUTABLE_REPRODUCTION"
    assert f["replay"]["uri"].startswith("data:image/png;base64,")
    assert "could not be pixel-compared" in f["replay"]["summary"]


def test_findings_of_one_workflow_graph_form_a_group_with_its_replay_once_and_each_child_scoped(tmp_path):
    from comfytrace.workbench import write_workbench
    from tests.unit.test_report import INCONCLUSIVE_ROWS, _replay_case

    case, _, regen_sha = _replay_case(tmp_path, INCONCLUSIVE_ROWS)
    page = write_workbench(case, output_html=tmp_path / "wb.html").read_text(encoding="utf-8")
    data = _data(page)
    by_id = {f["id"]: f for f in data["findings"]}
    one, two, fragment = by_id["FIND-1"], by_id["FIND-2"], by_id["FIND-3"]

    # Row coverage is shown as rows: 1 row and 0 rows of 512 both round to 0%.
    assert one["images"][0]["completeness"] == "PARTIAL \u00b7 1 of 512 rows"
    assert two["images"][0]["completeness"] == "PARTIAL \u00b7 0 of 512 rows"
    assert one["images"][0]["rows"] == "1 of 512 rows"

    # The replay's hash is the derived PNG file's SHA-256, labelled as such.
    assert one["replay"]["derivedFileSha256"] == regen_sha and "pixelSha256" not in one["replay"]
    assert one["replay"]["derivedPixelSha256"] is not None and len(one["replay"]["derivedPixelSha256"]) == 64
    assert one["replay"]["derivedPixelSha256"] != regen_sha
    assert "Derived file SHA-256" in page

    # The tested finding shows its inconclusive comparison; the sibling shows the run but no comparison.
    assert (one["replay"]["scope"], one["replay"]["comparisonStatus"], one["replay"]["match"]) == (
        "TESTED", "INCONCLUSIVE", False)
    assert "no match is claimed" in one["replay"]["summary"]
    assert two["replay"]["scope"] == "UNTESTED_SIBLING" and two["replay"]["ssim"] is None
    assert two["replay"]["summary"] == "this artifact was not compared; REGEN-001 compared IMG-1 (FIND-1)"
    assert two["images"][0]["comparison"]["status"] == "NOT_COMPARED"
    assert "REGEN-001 re-ran WF-1" in two["replay"]["execution"]

    [group] = [g for g in data["groups"] if g["id"] == one["groupId"]]
    assert two["groupId"] == group["id"] and fragment["groupId"] != group["id"]
    assert [c["id"] for c in group["children"]] == ["FIND-1", "FIND-2"]
    assert group["params"]["seed"] == "777" and group["leadWorkflowId"] == "WF-1"
    [replay] = group["replays"]
    assert (replay["id"], replay["testedArtifact"], replay["testedFinding"], replay["derivedFileSha256"]) == (
        "REGEN-001", "IMG-1", "FIND-1", regen_sha)
    comparisons = {c["id"]: c["comparisons"] for c in group["children"]}
    assert "inconclusive" in comparisons["FIND-1"]["IMG-1"]
    assert comparisons["FIND-2"]["IMG-2"].startswith("this artifact was not compared")
    assert group["submissions"]["promptIds"] == [] and group["submissions"]["unboundWorkflowIds"] == ["WF-1"]
    assert "unbound copies" in group["submissions"]["note"]

    # The seed-only fragment is cross-referenced on the group and stays its own E6 finding.
    [xref] = group["seedXrefs"]
    assert (xref["findingId"], xref["tier"], xref["seed"], xref["workflowIds"]) == ("FIND-3", "E6", "777", ["WF-F"])
    assert fragment["tier"] == "E6" and fragment["replay"] is None
    assert not re.search(r'(?:src|href)="https?://', page)


def test_a_prompt_id_names_a_known_submission_without_claiming_the_run_proves_it(tmp_path):
    from comfytrace.workbench import write_workbench
    from tests.unit.test_report import INCONCLUSIVE_ROWS, _artifact, _replay_case, _workflow

    rows = {"decoded_rows": 1, "total_rows": 512}
    workflows = [{**_workflow("WF-1", seed=777, prompt_id="job-7"), "canonical_json_sha256": "ab" * 32}]
    case, _, _ = _replay_case(tmp_path, INCONCLUSIVE_ROWS, workflows=workflows, evidence={"records": []},
                              artifacts=[_artifact("IMG-1", "PARTIAL", "PARTIAL_PNG", derived_preview_metadata=rows)],
                              findings=[{"finding_id": "FIND-1", "tier": "E3", "artifact_ids": ["IMG-1", "WF-1"],
                                         "link_ids": [], "rationale": "r"}])
    [group] = _data(write_workbench(case, output_html=tmp_path / "wb.html").read_text(encoding="utf-8"))["groups"]
    assert group["submissions"]["promptIds"] == ["job-7"] and group["submissions"]["unboundWorkflowIds"] == []
    assert "does not show which submission produced each output" in group["submissions"]["note"]


@pytest.mark.parametrize(("evaluation", "status", "phrase"), [
    ({"evaluated_tier": "INCONCLUSIVE", "evaluation_subtier": "INSUFFICIENT_DIMENSIONS",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "ssim_score": None, "ssim_unavailable_reason": "window_too_small",
      "mean_abs_diff": 1.5, "max_abs_diff": 9, "fraction_above_tolerance": 0.25}, "INCONCLUSIVE",
     "too thin for a 2D SSIM"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "ANIMATED_COMPARISON_UNAVAILABLE",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "exact_pixel_match": False},
     "ANIMATED_COMPARISON_UNAVAILABLE", "not a pixel contradiction"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "BELOW_THRESHOLDS",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "ssim_score": 0.41, "phash_hamming_distance": 20},
     "NON_MATCHING", "and differed: SSIM 0.410, pHash distance 20"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "DIMENSION_MISMATCH",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "dimension_mismatch": {"regenerated": [16, 16], "target": [8, 8]}},
     "NON_MATCHING", "the dimensions do not agree"),
])
def test_inconclusive_animated_and_non_matching_replays_stay_visible_and_claim_no_match(tmp_path, evaluation,
                                                                                         status, phrase):
    from comfytrace.workbench import write_workbench
    from tests.unit.test_report import _artifact, _replay_case

    case, _, _ = _replay_case(tmp_path, evaluation, artifacts=[_artifact("IMG-1"), _artifact("IMG-2")])
    data = _data(write_workbench(case, output_html=tmp_path / "wb.html").read_text(encoding="utf-8"))
    one = next(f for f in data["findings"] if f["id"] == "FIND-1")
    assert one["tier"] == "E3", "no comparison outcome here changes the tier"
    assert one["replay"] is not None, "a compared replay is never shown as 'not replayed'"
    assert (one["replay"]["comparisonStatus"], one["replay"]["match"]) == (status, False)
    assert phrase in one["replay"]["summary"] and one["images"][0]["comparison"]["status"] == status
    assert "not applied" in one["replay"]["promotion"]


def test_untiered_findings_and_the_other_images_bucket_are_counted_apart_and_labelled_by_their_reason(tmp_path):
    from comfytrace.workbench import write_workbench
    from tests.unit.test_report import _artifact, _replay_case, _workflow

    rationale = ("Image of 1 x 1 pixels STRICT-linked to WF-1 via EMBEDDED_PROMPT: below 16 pixels a side, smaller "
                 "than any picture ComfyUI's latent nodes produce, so it is not counted as a job's output.")
    link = {"link_id": "LINK-002", "workflow_id": "WF-1", "artifact_id": "IMG-1", "link_class": "STRICT",
            "matched_keys": ["EMBEDDED_PROMPT"], "confidence_notes": "n"}
    findings = [{"finding_id": "FIND-2", "tier": "UNTIERED", "artifact_ids": ["IMG-1"], "link_ids": ["LINK-002"],
                 "rationale": rationale},
                {"finding_id": "FIND-5", "tier": "UNTIERED", "artifact_ids": ["IMG-5"], "link_ids": [],
                 "rationale": "Partial image with no corroborating workflow key (orphan)."},
                {"finding_id": "FIND-18", "tier": "UNTIERED", "artifact_ids": ["OTH-1", "OTH-2", "OTH-3"],
                 "link_ids": [], "rationale": "Other images in memory: 3 images with no ComfyUI text."},
                {"finding_id": "FIND-28", "tier": "E6", "artifact_ids": ["WF-1"], "link_ids": [], "rationale": "r"}]
    artifacts = [_artifact(i) for i in ("IMG-1", "IMG-5", "OTH-1", "OTH-2", "OTH-3")]
    case, _, _ = _replay_case(tmp_path, None, artifacts=artifacts, findings=findings, links=[link],
                              workflows=[_workflow("WF-1", complete=False)], evidence={"records": []})
    data = _data(write_workbench(case, output_html=tmp_path / "wb.html").read_text(encoding="utf-8"))
    ut = data["tiers"]["UT"]
    assert (ut["count"], ut["otherImagesFindings"], ut["otherImagesArtifacts"]) == (2, 1, 3)
    assert (data["counts"]["untiered_findings"], data["counts"]["other_images_artifacts"]) == (2, 3)
    labels = {f["id"]: (f["tierLabel"], f["tierDesc"]) for f in data["untiered"]}
    assert labels["FIND-2"][0] == "Linked image excluded by size"
    assert "below 16 pixels a side" in labels["FIND-2"][1] and "LINK-002 (STRICT, EMBEDDED_PROMPT) to WF-1" in labels[
        "FIND-2"][1]
    assert labels["FIND-5"][0] == "Image not tied to a job"
    assert "FIND-18" not in labels, "the bucket is not listed as one image"


def test_the_replay_payload_carries_the_target_selection(tmp_path):
    from comfytrace.workbench import write_workbench
    from tests.unit.test_report import INCONCLUSIVE_ROWS, _replay_case

    selection = {"selection_reason": "MOST_DECODED_ROWS", "candidate_artifact_ids": ["IMG-1"],
                 "excluded_candidates": [{"artifact_id": "IMG-2", "reason": "ZERO_DECODED_ROWS"}],
                 "finding_ids": ["FIND-1", "FIND-2"]}
    case, _, _ = _replay_case(tmp_path, {**INCONCLUSIVE_ROWS, "target_selection": selection})
    data = _data(write_workbench(case, output_html=tmp_path / "wb.html").read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in data["findings"]}
    expected = {k: selection[k] for k in ("selection_reason", "candidate_artifact_ids", "excluded_candidates")}
    assert by_id["FIND-1"]["replay"]["targetSelection"] == expected
    assert by_id["FIND-2"]["replay"]["targetSelection"] == expected, "the sibling sees why it was not compared"


def test_the_replay_footer_never_prints_another_artifacts_result_beside_an_untested_copy():
    from importlib import resources

    template = resources.files("comfytrace").joinpath("templates/workbench.html").read_text(encoding="utf-8")
    assert "${esc(r.result)}" not in template, "the raw record result is not printed beside an image"
    assert "const scopedResult = !r ? \"\" : testedHere ? r.comparisonStatus" in template
    assert "${esc(scopedResult)}" in template


def test_the_raw_frame_pane_and_group_rows_say_what_each_observation_is():
    from importlib import resources

    template = resources.files("comfytrace").joinpath("templates/workbench.html").read_text(encoding="utf-8")
    assert '${isRaw ? "Recovered frame (no preview)" : "Recovered from RAM"}' in template
    assert "This observation applied to ${esc(rp.appliedTo)}." in template
    assert "Applied to ${esc(rp.appliedTo" not in template, "no run-wide application note beside one observation"
    assert "(recorded verdict ${esc(r.recordedVerdict)})" in template


def _template() -> str:
    from importlib import resources

    return resources.files("comfytrace").joinpath("templates/workbench.html").read_text(encoding="utf-8")


def test_a_promoted_finding_carries_the_tier_extraction_gave_it(tmp_path, mock_memory_image, fake_ingestor,
                                                                symbols_dir):
    from comfytrace.report import sha256_file
    from tests.unit.test_report import _regen

    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    _regen(case, "E1", sha256_file(case / "manifest.json"))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    [finding] = _data((case / "workbench.html").read_text(encoding="utf-8"))["findings"]
    assert (finding["tier"], finding["baseTier"]) == ("E1", "E2"), "the page names the E2 -> E1 promotion"
    assert finding["replay"]["promoted"] is True


def test_an_unpromoted_finding_has_its_own_tier_as_base_tier():
    from tests.unit.test_report import _artifact, _finding, _unit_rep, _workflow

    _, data, _ = _unit_rep([_finding("FIND-1", "E5", ["WF-1"]), _finding("FIND-2", "UNTIERED", ["IMG-9"], "Orphan")],
                           [_workflow("WF-1")], [_artifact("IMG-9")])
    assert [(f["tier"], f["baseTier"]) for f in data["findings"] + data["untiered"]] == [("E5", "E5"), ("UT", "UT")]


@pytest.mark.parametrize(("encoding", "text"), [
    ("UTF-8_JSON_STRING", "decoded from a Python UTF-8 JSON string"),
    ("UTF-16LE_JSON_STRING", "decoded from a Python UTF-16 JSON string"),
    ("SERIALIZED_QUEUE_ITEM", "decoded from ComfyUI's job queue"),
    ("RESERIALIZED_PYTHON_OBJECTS", "decoded from ComfyUI's Python objects"),
    ("DECOMPRESSED_OR_DECODED_PNG_TEXT", "read from a PNG's text chunk"),
    ("CONTAINER_METADATA_TEXT", "read from media container metadata"),
    ("PNG_TEXT_STITCHED", "reconstructed from a PNG's torn text chunk"),
    ("WEBP_EMBEDDED_TEXT", "read from a WEBP file's embedded metadata"),
    ("SOME_FUTURE_ENCODING", "SOME_FUTURE_ENCODING"),
])
def test_a_workflow_records_origin_is_named_in_plain_words(encoding, text):
    from tests.unit.test_report import _finding, _unit_rep, _workflow

    lead = {**_workflow("WF-1"), "encoding_detected": encoding}
    copy = {**_workflow("WF-2"), "encoding_detected": encoding}
    _, data, _ = _unit_rep([_finding("FIND-1", "E5", ["WF-1", "WF-2"])], [lead, copy])
    wf = data["findings"][0]["workflow"]
    assert wf["source"] == text
    assert [c["source"] for c in wf["copies"]] == [text]


def test_a_workflow_record_without_a_recorded_encoding_says_so():
    from comfytrace.workbench import _encoding_text

    assert _encoding_text(None) == _encoding_text("") == "\u2014"


def test_the_page_shows_raw_tokens_in_plain_english_and_keeps_them_in_tooltips():
    template = _template()
    for table in ("HUMAN_STATUS", "HUMAN_LINK", "HUMAN_KEY", "HUMAN_MODE", "HUMAN_SEL_REASON"):
        assert f"const {table} = {{" in template
    for token, label in [("ROW_SLICE_PERCEPTUAL", "Close match (recovered rows)"),
                         ("NO_MEMORY_TARGET", "Workflow ran in lab (no memory pixels to compare)"),
                         ("INSUFFICIENT_ROWS", "Inconclusive (too few rows to score)"),
                         ("STRICT", "Verified link"), ("UNASSOCIATED", "No metadata link"),
                         ("CANONICAL_PROMPT_HASH", "Exact workflow graph hash"),
                         ("PHYSICAL_FALLBACK", "Physical scan (degraded"),
                         ("PARTIAL_MOST_DECODED_ROWS", "Partial image with most decoded rows"),
                         ("ZERO_DECODED_ROWS", "0 decoded rows (header only)")]:
        assert re.search(rf'\b{token}: "{re.escape(label)}', template), token
    assert '<span class="term" title="${esc(raw)}">${esc(label)}</span>' in template, "the raw token is the tooltip"


def test_the_provenance_chain_verifies_only_a_metadata_key_link_and_states_missing_values():
    template = _template()
    assert '!["AMBIGUOUS", "UNASSOCIATED", "NONE", DASH].includes(activeCopy.link)' in template
    assert 'activeCopy.link !== "AMBIGUOUS" ? "verified"' not in template
    assert '"Virtual heap object"' not in template and '"Process virtual memory"' not in template
    assert '"N/A"' not in template, "missing values are the builder's dash, never N/A"
    assert '"Partial stream (no whole-image pixel hash)"' in template
    assert "activeCopy.pixelSha256 && `Pixel SHA-256: ${activeCopy.pixelSha256}`" in template, "no 'null' is copied"
    assert "r.derivedPixelSha256 && `Pixel SHA-256: ${r.derivedPixelSha256}`" in template
    assert "`Derived file SHA-256: ${r.derivedFileSha256}`" in template


def test_a_partial_image_is_drawn_rows_first_with_the_hatch_only_below_them():
    template = _template()
    assert 'style="top:${pct}%"' not in template, "the hatch no longer overlays the image from a rounded percentage"
    assert "const isPartial = Boolean(rows) || pct < 100;" in template, "1 row of 512 (0%) is still partial"
    assert '<image class="art-rows" href="${src}" x="0" y="0" width="${W}" height="${visH}"' in template
    assert '<rect class="art-hatch" x="0" y="${visH}" width="${W}" height="${rest}"' in template


def test_repeated_and_empty_elements_are_cut_and_the_workflow_card_is_kept():
    template = _template()
    assert "tier-explainer" not in template and "meta-grid" not in template and "no-image-box" not in template
    assert "g.children.length > 1 || g.seedXrefs.length > 0" in template, "no one-finding group wrappers"
    assert "[job.promptId, job.queued, job.duration, job.status].some(present)" in template, "no all-dash job strip"
    assert "imgs.length > 1 ? `" in template, "no one-tab copy strip"
    assert "visualImages(f)" in template and "im => !im.container" in template, "0-row containers are not drawn"
    assert "<span>Workflow</span>" in template and "Provenance Chain" in template
    assert "Pixel SHA-256" in template and "Derived file SHA-256" in template


def test_the_page_meets_contrast_print_viewport_and_keyboard_requirements():
    template = _template()
    assert "--ink-muted: #556070;" in template and template.count("--ink-muted: #8b98ad;") == 2
    assert "--ut-fg: #475569; --ut-bg: #f1f5f9;" in template
    assert ':root, :root[data-theme="dark"], :root[data-theme="light"], :root:not([data-theme="light"])' in template
    assert template.count("height: calc(100vh - var(--topbar-h));") >= 2
    layout = ".workflow-layout.has-graph { grid-template-columns: minmax(250px, 0.95fr) minmax(270px, 1.05fr); }"
    assert layout in template, "graph and parameters side by side on wide screens"
    assert ".wf-node:focus-visible rect.node-box" in template and 'button:focus-visible' in template
    assert 'if (e.key !== "Enter" && e.key !== " ") return;' in template
