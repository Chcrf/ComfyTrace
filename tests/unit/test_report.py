import json
import os
import re
from pathlib import Path

import pytest

from comfytrace.cli import parse_args, run_extract, run_report
from comfytrace.models import ForensicManifest, RawArtifactRecord, RegenerationRecord, WorkflowRecord
from comfytrace.report import EvidentiaryExporter, ManifestWriter, save_regeneration_record, sha256_file

SCHEMAS = Path(__file__).parent.parent / "fixtures" / "schemas"


def _case(tmp_path: Path, mock_memory_image, fake_ingestor, symbols_dir, state="T1") -> Path:
    case = tmp_path / f"case_{state}"
    args = parse_args(["extract", "--memory", str(mock_memory_image(state)), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case), "--case-id", "CASE-2026-0042", "--investigator", "Examiner"])
    run_extract(args, ingestor=fake_ingestor)
    return case


def test_evidentiary_split_raw_vs_derived(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir, "T3")
    raw = sorted(p.relative_to(case).as_posix() for p in (case / "exports" / "raw").rglob("*") if p.is_file())
    derived = sorted(p.name for p in (case / "exports" / "derived").iterdir())
    assert raw == ["exports/raw/CARVE-PNG-001/span_000.bin"]
    assert derived == ["DERIVED_preview_CARVE-PNG-001.png"]
    manifest = json.loads((case / "manifest.json").read_text())
    [artifact] = manifest["findings"]["artifacts"]
    assert artifact["is_derived"] is False
    assert not any("derived" in p for p in artifact["raw_file_paths"])
    assert (case / raw[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_exporter_never_overwrites_evidence(tmp_path):
    exporter = EvidentiaryExporter(tmp_path)
    data = json.loads((SCHEMAS / "raw_artifact_intact.json").read_text())
    record = RawArtifactRecord.from_dict(data)
    exporter.export_raw_artifact(record, b"\x89PNG-bytes")
    with pytest.raises(FileExistsError):
        exporter.export_raw_artifact(record, b"\x89PNG-other")
    assert exporter.export_derived_preview("preview_X.png", b"p") == "exports/derived/DERIVED_preview_X.png"
    wf = WorkflowRecord.from_dict(json.loads((SCHEMAS / "workflow.json").read_text()))
    wf.raw_bytes = '{"a": 1}'.encode("utf-16-le")
    assert exporter.export_workflow(wf) == "exports/raw/WF-001.json"
    assert (tmp_path / "exports/raw/WF-001.json").read_bytes() == wf.raw_bytes, "original encoding kept verbatim"


OPEN_DICTS = {"execution_parameters"}


def _assert_same_shape(actual, expected, path="$"):
    if isinstance(expected, dict) and expected:
        assert isinstance(actual, dict), path
        keys = list(actual)
        if path.rsplit(".", 1)[-1] in OPEN_DICTS:
            # Parameters a record does not carry are omitted, never guessed: compare the keys it has.
            keys = [k for k in keys if k in expected]
            expected = {k: v for k, v in expected.items() if k in actual}
        assert keys == list(expected), f"{path}: keys {list(actual)} != {list(expected)}"
        for key in expected:
            _assert_same_shape(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list) and expected and isinstance(actual, list):
        for i, item in enumerate(actual):
            _assert_same_shape(item, expected[0], f"{path}[{i}]")


def test_manifest_immutability_and_schema_conformance(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = json.loads((SCHEMAS / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0.0"
    meta_keys = list(manifest["case_metadata"])
    assert meta_keys[: len(expected["case_metadata"])] == list(expected["case_metadata"])
    assert meta_keys[len(expected["case_metadata"]):] == ["carving_mode", "degradation_reason", "recovery_stages"]
    for section in ("memory_image", "symbol_resolution"):
        _assert_same_shape(manifest["case_metadata"][section], expected["case_metadata"][section])
    _assert_same_shape(manifest["process_context"], expected["process_context"])
    _assert_same_shape(manifest["findings"]["findings"], expected["findings"]["findings"])
    _assert_same_shape(manifest["findings"]["correlation_links"], expected["findings"]["correlation_links"])
    assert list(manifest["findings"]) == list(expected["findings"])
    intact = json.loads((SCHEMAS / "raw_artifact_intact.json").read_text())
    _assert_same_shape(manifest["findings"]["artifacts"], [intact])
    _assert_same_shape(manifest["findings"]["workflows"], [json.loads((SCHEMAS / "workflow.json").read_text())])

    assert not os.access(manifest_path, os.W_OK) or os.geteuid() == 0
    before = sha256_file(manifest_path)
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    with pytest.raises(FileExistsError):
        ManifestWriter(case).write_manifest(ForensicManifest.from_dict(manifest))
    assert sha256_file(manifest_path) == before


def _regen(case: Path, promoted: str | None, sha: str, handoff: str = "REGEN-001",
           evaluated="E1", workflow="WF-001") -> Path:
    record = RegenerationRecord.from_dict(json.loads((SCHEMAS / "regeneration.json").read_text()))
    record.handoff_id = handoff
    record.workflow_id = workflow
    record.target_manifest_sha256 = sha
    record.evaluation = {**record.evaluation, "promoted_finding_tier": promoted, "evaluated_tier": evaluated}
    record.evaluation.update(ssim_threshold=0.95, phash_threshold=4)
    derived = case / record.derived_regenerated_image
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_bytes((case / "exports/raw/CARVE-PNG-001.png").read_bytes())
    record.evaluation["derived_regenerated_image_sha256"] = sha256_file(derived)
    return save_regeneration_record(case, record)


def _wb_data(page: str) -> dict:
    match = re.search(r'<script id="ct-data" type="application/json">(.*?)</script>', page, re.S)
    assert match, "the case data block is embedded"
    return json.loads(match.group(1))


def _case_rep(case: Path):
    from comfytrace.report import (
        EVIDENCE_NAME,
        LINKS_NAME,
        _Report,
        effective_findings,
        load_regenerations,
    )

    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = sha256_file(manifest_path)
    regens = load_regenerations(case / "regenerations")
    sidecars = {name: json.loads((case / name).read_text(encoding="utf-8"))
                for name in (EVIDENCE_NAME, LINKS_NAME) if (case / name).is_file()}
    findings, notes = effective_findings(manifest, regens, sha, case, sidecars.get(LINKS_NAME))
    return _Report(case, manifest, sha, findings, regens, notes, sidecars.get(EVIDENCE_NAME))


def _render_workbench(case: Path) -> tuple[dict, str]:
    from comfytrace.workbench import render_workbench_html, workbench_data

    rep = _case_rep(case)
    data = workbench_data(rep)
    return data, render_workbench_html(data)


def test_html_reporter_offline_rerender(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    sha = sha256_file(case / "manifest.json")
    assert not (case / "report.html").exists()
    first = (case / "workbench.html").read_text(encoding="utf-8")
    assert '"id":"FIND-001","tier":"E2"' in first
    assert not re.search(r'(?:src|href)="https?://', first)

    _regen(case, "E1", sha)
    _regen(case, "E1", "0" * 64, handoff="REGEN-002")
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    assert not (case / "report.html").exists()
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    assert '"id":"FIND-001","tier":"E1"' in wb
    assert "REGEN-001" in wb
    assert json.loads((case / "manifest.json").read_text())["findings"]["findings"][0]["tier"] == "E2"


def test_report_on_missing_manifest_fails(tmp_path):
    assert run_report(parse_args(["report", "--case-dir", str(tmp_path)])) == 1


def test_an_image_link_promotion_merges_the_orphan_and_the_jobs_prompt_into_one_e2_finding():
    from comfytrace.report import effective_findings

    def finding(fid, tier, ids):
        return {"finding_id": fid, "tier": tier, "artifact_ids": ids, "link_ids": [], "rationale": ""}

    manifest = {"findings": {"artifacts": [], "findings": [
        finding("FIND-1", "UNTIERED", ["TENSOR-001"]), finding("FIND-2", "E5", ["WFPY-001"]),
        finding("FIND-3", "E5", ["WFPY-002"]), finding("FIND-4", "E5", ["WFPY-009"])]}}
    links = {"manifest_sha256": "ab" * 32, "promotions": [
        {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p", "artifact_id": "TENSOR-001",
         "workflow_ids": ["WFPY-001", "WFPY-002"]}]}
    findings, notes = effective_findings(manifest, [], "ab" * 32, None, links)
    by_id = {f["finding_id"]: f for f in findings}
    assert set(by_id) == {"FIND-1", "FIND-4"}, "the prompt's E5 copies are absorbed; other workflows stay"
    assert (by_id["FIND-1"]["tier"], by_id["FIND-1"]["extraction_tier"]) == ("E2", "UNTIERED")
    assert by_id["FIND-1"]["artifact_ids"] == ["TENSOR-001", "WFPY-001", "WFPY-002"]
    assert by_id["FIND-1"]["base_tier"] == "E2", "a matching regeneration can promote it to E1"
    assert notes == [{"promotion": "TENSOR-001", "applied_to": "FIND-1", "note": "REGENERATION_WHOLE_FRAME_MATCH"}]
    findings, notes = effective_findings(manifest, [], "cd" * 32, None, links)
    assert [f["tier"] for f in findings] == ["UNTIERED", "E5", "E5", "E5"] and notes[0]["applied_to"] is None


def _unit_rep(findings, workflows=(), artifacts=(), links=(), case_dir=None, evidence=None,
              regens=(), regen_links=None):
    """Builds _Report, workbench_data, and rendered workbench HTML from a hand-made manifest."""
    from comfytrace.report import _Report, effective_findings
    from comfytrace.workbench import render_workbench_html, workbench_data

    manifest = {"schema_version": "1.0.0",
                "case_metadata": {"case_id": "C-1", "investigator": "I",
                                  "execution_timestamp_utc": "2026-11-07T00:00:00Z",
                                  "comfytrace_version": "1.0.0", "command_line": "comfytrace extract",
                                  "memory_image": {"path": "C:\\dumps\\mem.raw", "size_bytes": 2048,
                                                   "sha256": "ab" * 32},
                                  "symbol_resolution": {"status": "SUCCESS", "mode": "OFFLINE_CACHE"},
                                  "carving_mode": "PROCESS_VIRTUAL", "degradation_reason": None,
                                  "recovery_stages": {"python_objects": {"status": "AVAILABLE"}}},
                "process_context": [],
                "findings": {"tier_summary": {}, "findings": list(findings), "artifacts": list(artifacts),
                             "workflows": list(workflows), "correlation_links": list(links)}}
    effective, notes = effective_findings(manifest, list(regens), "cd" * 32, case_dir, regen_links)
    rep = _Report(case_dir or Path("."), manifest, "cd" * 32, effective, list(regens), notes, evidence)
    data = workbench_data(rep)
    return rep, data, render_workbench_html(data)


def _finding(fid, tier, ids, rationale="r", links=()):
    return {"finding_id": fid, "tier": tier, "artifact_ids": list(ids), "link_ids": list(links), "rationale": rationale}


def _workflow(wid, complete=True, prompt="a cat", fmt="API_GRAPH", **params):
    return {"workflow_id": wid, "completeness": "COMPLETE" if complete else "PARTIAL", "prompt_format": fmt,
            "encoding_detected": "UTF-8", "source_process_id": 1, "virtual_address": "0x10",
            "physical_offsets": ["0x20"],
            "canonical_json_sha256": "ef" * 32, "execution_parameters": {"positive_prompt": prompt, **params},
            "dependencies": {"node_classes": [], "model_assets": []}, "raw_file_path": f"exports/raw/{wid}.json"}


def test_workbench_opens_with_case_details_and_ranked_findings(tmp_path, mock_memory_image, fake_ingestor,
                                                                          symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    page = (case / "workbench.html").read_text(encoding="utf-8")
    data = _wb_data(page)
    assert data["case"]["caseId"] == "CASE-2026-0042" and data["case"]["dumpName"] == "memdump_T1.raw"
    assert data["case"]["mode"] == "PROCESS_VIRTUAL"
    [finding] = data["findings"]
    assert (finding["id"], finding["tier"]) == ("FIND-001", "E2")
    assert finding["tierLabel"] == "Image and prompt recovered together"
    assert finding["workflow"]["id"] == "WF-001" and finding["replay"] is None
    assert [im["id"] for im in finding["images"]] == ["CARVE-PNG-001"]
    payload = re.search(r'<script id="ct-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert payload.isascii(), "embedded JSON payload is pure ASCII"


def test_the_report_embeds_a_bounded_preview_and_stays_readable_when_moved_alone(tmp_path, mock_memory_image,
                                                                                 fake_ingestor, symbols_dir):
    import base64

    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    page = (case / "workbench.html").read_text(encoding="utf-8")
    data = _wb_data(page)
    png = case / "exports" / "raw" / "CARVE-PNG-001.png"
    [image] = data["findings"][0]["images"]
    expected_uri = f"data:image/png;base64,{base64.b64encode(png.read_bytes()).decode('ascii')}"
    assert image["uri"] == expected_uri
    assert image["containerSha256"] == sha256_file(png)
    alone = tmp_path / "elsewhere"
    alone.mkdir()
    (alone / "moved.html").write_text(page, encoding="utf-8")
    moved = (alone / "moved.html").read_text(encoding="utf-8")
    assert re.findall(r'(?:src|href)="(?!#|data:|\$\{src\})[^"]*"', moved) == [], "no link leads outside the page"
    assert _wb_data(moved)["findings"][0]["images"][0]["uri"] == expected_uri
    assert re.findall(r"http://(?!www\.w3\.org/2000/svg)", moved) == [] and "https://" not in moved
    assert "<link" not in moved and "@import" not in moved


def test_a_torn_image_is_labelled_partial_and_derived_and_the_degraded_run_is_stated(tmp_path, mock_memory_image,
                                                                                     fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir, "T3")
    data, _ = _render_workbench(case)
    assert data["case"]["mode"] == "PHYSICAL_FALLBACK"
    assert data["findings"] == []
    [ut] = data["untiered"]
    assert ut["tier"] == "UT" and ut["tierLabel"] == "Image not tied to a job"
    assert "no strong key ties to a recovered prompt" in ut["tierDesc"]
    [img] = ut["images"]
    assert img["completeness"].startswith("PARTIAL \u00b7 ") and " of " in img["rows"]
    assert 0 < img["partialPct"] < 100
    assert img["derived"] is False and img["file"] == "exports/raw/CARVE-PNG-001/span_000.bin"
    assert img["uri"].startswith("data:image/png;base64,")


def test_an_applied_regeneration_is_labelled_regenerated_and_an_unapplied_one_is_explained(tmp_path, mock_memory_image,
                                                                                           fake_ingestor, symbols_dir):
    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    sha = sha256_file(case / "manifest.json")
    _regen(case, "E1", sha)
    _regen(case, "E1", "0" * 64, handoff="REGEN-002")
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    rep = _case_rep(case)
    data = _wb_data((case / "workbench.html").read_text(encoding="utf-8"))
    [finding] = data["findings"]
    assert finding["tier"] == "E1" and finding["tierLabel"] == "Reproduced"
    replay = finding["replay"]
    assert replay["id"] == "REGEN-001" and replay["applied"] is True and replay["promoted"] is True
    assert replay["comparisonStatus"] == "PERCEPTUAL_MATCH"
    assert "close perceptual match" in replay["summary"] and replay["uri"].startswith("data:image/png;base64,")
    assert replay["others"] == []
    unapplied = next(r for r in rep.regenerations if r["handoff_id"] == "REGEN-002")
    assert rep.replay_problem(unapplied) == "made against another manifest"
    assert rep.regen_note["REGEN-002"]["applied_to"] is None
    assert "target_manifest_sha256 does not match this manifest" in rep.regen_note["REGEN-002"]["note"]


def test_findings_are_ranked_by_usefulness_and_prompt_only_findings_stay_compact():
    findings = [_finding("FIND-1", "E5", ["WF-5"]), _finding("FIND-2", "E3", ["WF-3"]),
                _finding("FIND-3", "E2", ["WF-2"]), _finding("FIND-4", "UNTIERED", ["IMG-9"], "Orphan"),
                _finding("FIND-5", "UNTIERED", ["IMG-1", "IMG-2"], "Other images in memory: 2 images")]
    workflows = [_workflow("WF-5", prompt="five"), _workflow("WF-3", prompt="three", complete=False),
                 _workflow("WF-2", prompt="two")]
    rep, data, _ = _unit_rep(findings, workflows)
    assert [f["id"] for f in data["findings"]] == ["FIND-3", "FIND-2", "FIND-1"]
    assert [f["id"] for f in data["untiered"]] == ["FIND-4"]
    assert data["tiers"]["UT"]["otherImagesFindings"] == 1
    assert data["tiers"]["UT"]["otherImagesArtifacts"] == 2
    assert [f["finding_id"] for f in rep.ordered()] == ["FIND-3", "FIND-2", "FIND-1", "FIND-4"]


def test_workbench_preserves_long_prompts_and_escapes_case_text():
    from comfytrace.workbench import render_workbench_html

    evil = '<script>alert(1)</script> "quoted" & ' + "x" * 600
    _, data, page = _unit_rep([_finding("FIND-1", "E5", ["WF-1"])], [_workflow("WF-1", prompt=evil, seed=7)])
    assert "</script><script>alert(1)" not in page
    assert _wb_data(page)["findings"][0]["workflow"]["params"]["positive"] == evil
    escaped = render_workbench_html({**data, "case": {**data["case"], "caseId": evil}})
    assert "<script>alert(1)</script>" not in escaped.split("<body>")[0]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in escaped.split("<body>")[0]


def test_uncertain_evidence_is_named_as_such():
    fragment = _workflow("WF-F", complete=False, fmt="API_FRAGMENT", prompt="part")
    evidence = {"records": [{"record_id": "WF-F", "kind": "workflow", "memory_source": "FILE_BACKED",
                             "fragment": {"attribution": "AMBIGUOUS", "cut_before": True, "cut_after": False,
                                          "agreeing_seeds": [1, 2]}}]}
    rep, data, _ = _unit_rep([_finding("FIND-1", "E6", ["WF-F"])], [fragment], evidence=evidence)
    [finding] = data["findings"]
    assert finding["tier"] == "E6" and finding["tierLabel"] == "Partial prompt, no image"
    assert rep.source("WF-F") == "FILE_BACKED"
    assert rep.records["WF-F"]["fragment"]["attribution"] == "AMBIGUOUS"
    assert data["groups"][0]["seedXrefs"] == []


def test_previews_are_bounded_scaled_and_confined_to_the_case_folder(tmp_path):
    import io

    from PIL import Image

    from comfytrace.report import _Previews

    case = tmp_path / "case"
    case.mkdir()
    big = io.BytesIO()
    Image.effect_noise((900, 900), 120).convert("RGB").save(big, format="PNG")
    (case / "big.png").write_bytes(big.getvalue())
    (case / "note.png").write_bytes(b"not an image")
    (tmp_path / "outside.png").write_bytes(big.getvalue())
    previews = _Previews(case)
    assert len(big.getvalue()) > previews.PER_IMAGE
    scaled = previews.load("big.png")
    assert scaled["uri"].startswith("data:image/png;base64,") and scaled["scaled"][1] == (900, 900)
    assert max(scaled["scaled"][0]) <= 640 and len(scaled["uri"]) < previews.PER_IMAGE * 4 // 3 + 100
    assert "not a PNG or WebP" in previews.load("note.png")["omitted"]
    assert "not in the case folder" in previews.load("../outside.png")["omitted"]
    assert "not in the case folder" in previews.load("missing.png")["omitted"]
    tight = _Previews(case)
    tight.TOTAL = 10
    assert "budget" in tight.load("big.png")["omitted"]


def _png(color, size=(8, 8)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _artifact(artifact_id, completeness="COMPLETE", carve_type="ENCODED_PNG", **fields):
    return {"artifact_id": artifact_id, "completeness": completeness, "carve_type": carve_type,
            "link_status": "STRICT", "raw_file_paths": [], "container_sha256": None, "pixel_sha256": None,
            "pixel_metadata": None, "derived_preview_path": None, "derived_preview_metadata": None, "gaps": [],
            "physical_offsets": [], "source_layer": "physical", **fields}


def test_a_preview_whose_bytes_differ_from_the_recorded_hash_is_withheld_and_called_out(tmp_path, mock_memory_image,
                                                                                         fake_ingestor, symbols_dir):
    import base64

    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    before_data, _ = _render_workbench(case)
    assert before_data["findings"][0]["images"][0]["uri"].startswith("data:image/png;base64,")
    assert before_data["findings"][0]["images"][0]["omitted"] is None

    swapped = _png("red")
    (case / "exports" / "raw" / "CARVE-PNG-001.png").write_bytes(swapped)  # a swapped file, still a valid PNG
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    data = _wb_data(wb)
    [image] = data["findings"][0]["images"]
    assert image["uri"] is None, "bytes that do not match the manifest's hash are never drawn as the recovered image"
    assert "does not match the hash recorded for it" in image["omitted"]
    assert base64.b64encode(swapped).decode("ascii") not in wb
    rep = _case_rep(case)
    loaded = rep.previews.load("exports/raw/CARVE-PNG-001.png", image["containerSha256"])
    assert loaded["check"] == "mismatch" and "uri" not in loaded


def test_a_derived_rendering_is_checked_against_the_recorded_pixel_hash_or_labelled_unverified(tmp_path):
    import hashlib
    import io

    from PIL import Image

    from comfytrace.report import _Previews

    data = _png((10, 20, 30))
    with Image.open(io.BytesIO(data)) as img:
        pixel_sha = hashlib.sha256(img.tobytes()).hexdigest()
    (tmp_path / "p.png").write_bytes(data)
    previews = _Previews(tmp_path)
    assert previews.load("p.png", pixel_sha256=pixel_sha)["check"] == "pixels"
    assert previews.load("p.png", pixel_sha256="0" * 64)["check"] == "none"
    assert previews.load("p.png", expected=hashlib.sha256(data).hexdigest())["check"] == "match"
    assert previews.load("p.png", expected="0" * 64)["check"] == "mismatch"
    assert "uri" not in previews.load("p.png", expected="0" * 64)
    assert "no SHA-256 was recorded" in previews.load("p.png", require_hash=True)["omitted"]


def test_the_lead_finding_follows_the_finding_numbers_not_their_text_order(tmp_path):
    case = tmp_path
    for name, color in (("a.png", "red"), ("b.png", "blue")):
        (case / name).write_bytes(_png(color))
    import hashlib

    def digest(color):
        return hashlib.sha256(_png(color)).hexdigest()

    artifacts = [_artifact("IMG-9", raw_file_paths=["a.png"], container_sha256=digest("red")),
                 _artifact("IMG-10", raw_file_paths=["b.png"], container_sha256=digest("blue"))]
    findings = [_finding("FIND-10", "E2", ["IMG-10", "WF-10"]), _finding("FIND-9", "E2", ["IMG-9", "WF-9"])]
    workflows = [_workflow("WF-9", prompt="nine"), _workflow("WF-10", prompt="ten")]
    rep, data, _ = _unit_rep(findings, workflows, artifacts, case_dir=case)
    assert [f["id"] for f in data["findings"]] == ["FIND-9", "FIND-10"]
    assert data["findings"][0]["workflow"]["params"]["positive"] == "nine"
    assert [f["finding_id"] for f in rep.ordered()] == ["FIND-9", "FIND-10"]


def test_the_overview_counts_only_the_findings_left_after_links_and_names_a_lead_that_exists():
    from comfytrace.report import _Report, effective_findings
    from comfytrace.workbench import workbench_data

    manifest = {"schema_version": "1.0.0", "case_metadata": {
        "case_id": "C", "memory_image": {}, "symbol_resolution": {}, "carving_mode": "PROCESS_VIRTUAL",
        "recovery_stages": {}},
        "process_context": [],
        "findings": {"tier_summary": {}, "artifacts": [_artifact("TENSOR-001", carve_type="TENSOR_IMAGE")],
                     "workflows": [_workflow("WFPY-001", prompt="promoted"), _workflow("WFPY-002", prompt="absorbed")],
                     "correlation_links": [],
                     "findings": [_finding("FIND-1", "UNTIERED", ["TENSOR-001"]),
                                  _finding("FIND-2", "E5", ["WFPY-001"]), _finding("FIND-3", "E5", ["WFPY-002"])]}}
    links = {"manifest_sha256": "ab" * 32, "promotions": [
        {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p", "artifact_id": "TENSOR-001",
         "workflow_ids": ["WFPY-001", "WFPY-002"]}]}
    findings, notes = effective_findings(manifest, [], "ab" * 32, None, links)
    rep = _Report(Path("."), manifest, "ab" * 32, findings, [], notes, None)
    data = workbench_data(rep)
    assert data["counts"] == {
        "findings_by_tier": {"E1": 0, "E2": 1, "E3": 0, "E4": 0, "E5": 0, "E6": 0, "UNTIERED": 0},
        "linked_findings": 1, "linked_workflow_graphs": 1, "linked_image_artifacts": 1,
        "prompt_only_findings": 0, "workflow_graphs": 1, "untiered_findings": 0,
        "other_images_findings": 0, "other_images_artifacts": 0,
    }
    [lead] = data["findings"]
    assert lead["id"] == "FIND-1" and lead["tier"] == "E2"
    assert lead["workflowIds"] == ["WFPY-001", "WFPY-002"]
    assert data["untiered"] == []


def test_an_empty_report_does_not_claim_the_memory_held_no_comfyui():
    _, data, page = _unit_rep([])
    assert data["findings"] == [] and data["untiered"] == [] and data["groups"] == []
    assert all(v == 0 for v in data["counts"]["findings_by_tier"].values())
    assert "no ComfyUI evidence" not in page


def test_a_recovered_video_is_not_labelled_an_image():
    video = _artifact("VID-1", carve_type="ENCODED_MP4", raw_file_paths=["exports/raw/VID-1.mp4"])
    _, data, _ = _unit_rep([_finding("FIND-1", "E5", ["VID-1", "WF-1"])], [_workflow("WF-1")], [video])
    [media] = data["findings"][0]["images"]
    assert media["kind"] == "MP4 video"
    assert media["uri"] is None and media["omitted"] == "no previewable rendering was exported for this artifact"


def test_omissions_in_a_findings_provenance_and_prompt_are_stated():
    gaps = [{"relative_offset": n, "length_bytes": 1, "reason": "TRUNCATED_STREAM", "synthetic_padding_applied": False}
            for n in range(14)]
    torn = _artifact("IMG-1", "PARTIAL", "PARTIAL_PNG", gaps=gaps)
    link_records = [{"link_id": f"LINK-{n}", "workflow_id": "WF-1", "artifact_id": "IMG-1", "link_class": "STRICT",
                     "matched_keys": ["PROMPT_ID"], "confidence_notes": "n"} for n in range(10)]
    rep, data, _ = _unit_rep([_finding("FIND-1", "E3", ["IMG-1", "WF-1"], links=[f"LINK-{n}" for n in range(10)])],
                             [_workflow("WF-1", prompt="word " * 200)], [torn], link_records)
    [finding] = data["findings"]
    assert len(finding["links"]) == 10
    assert finding["workflow"]["params"]["positive"] == "word " * 200
    assert rep.workflows["WF-1"]["raw_file_path"] == "exports/raw/WF-1.json"
    assert len(rep.artifacts["IMG-1"]["gaps"]) == 14


def test_an_applied_regeneration_shows_the_selected_raster_and_old_records_still_work(tmp_path, mock_memory_image,
                                                                                    fake_ingestor, symbols_dir):
    import base64
    import hashlib

    case = _case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir)
    sha = sha256_file(case / "manifest.json")
    path = _regen(case, "E1", sha)  # an old-style record: only derived_regenerated_image and its hash
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    old = _wb_data((case / "workbench.html").read_text(encoding="utf-8"))["findings"][0]["replay"]
    assert old["derivedPath"] == "exports/derived/DERIVED_regen_WF-001.png"
    assert old["uri"].startswith("data:image/png;base64,")

    other = _png("green", (16, 16))
    (case / "exports" / "derived" / "DERIVED_regen_second.png").write_bytes(other)
    record = json.loads(path.read_text())
    first = record["derived_regenerated_image"]
    second = "exports/derived/DERIVED_regen_second.png"
    outputs = [{"file": first, "sha256": record["evaluation"]["derived_regenerated_image_sha256"],
                "node_id": "9", "batch_index": 0},
               {"file": second, "sha256": hashlib.sha256(other).hexdigest(), "node_id": "9", "batch_index": 1}]

    # The shape ``comfytrace regenerate`` writes: every output on the record, the selected one in the evaluation.
    record["output_rasters"] = outputs
    record["evaluation"]["selected_raster"] = outputs[1]
    path.write_text(json.dumps(record))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    new = _wb_data((case / "workbench.html").read_text(encoding="utf-8"))["findings"][0]["replay"]
    assert new["derivedPath"] == second and new["derivedFileSha256"] == outputs[1]["sha256"]
    assert (new["outputNode"], new["batchIndex"]) == ("9", 1)
    assert base64.b64encode(other).decode() in new["uri"]

    # The same fields on the record itself, the selected one named by its position.
    del record["output_rasters"], record["evaluation"]["selected_raster"]
    record["matching_rasters"], record["selected_raster"] = outputs, 1
    path.write_text(json.dumps(record))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    matched = _wb_data((case / "workbench.html").read_text(encoding="utf-8"))["findings"][0]["replay"]
    assert matched["derivedPath"] == second and base64.b64encode(other).decode() in matched["uri"]

    del record["matching_rasters"]
    record["selected_raster"] = {"file": second}  # no hash recorded anywhere for this one: it cannot be verified
    path.write_text(json.dumps(record))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    unverified = _wb_data(wb)["findings"][0]["replay"]
    assert unverified["uri"] is None
    assert "no SHA-256 was recorded for this image" in unverified["omitted"]
    assert base64.b64encode(other).decode() not in wb


def test_exact_replay_positions_require_verified_rasters_and_declared_output_coverage(tmp_path):
    from comfytrace.carve import native_pixel_digest
    from comfytrace.report import _verified_replay_positions

    files = tmp_path / "exports" / "derived"
    files.mkdir(parents=True)
    target = _png("green", (16, 16))
    pixels, meta = native_pixel_digest(target)
    artifact = {"pixel_sha256": pixels, "pixel_metadata": meta}
    rasters = []
    for i, png in enumerate((target, _png("purple", (16, 16)))):
        path = files / f"DERIVED_{i}.png"
        path.write_bytes(png)
        rasters.append({"file": path.relative_to(tmp_path).as_posix(), "sha256": sha256_file(path),
                        "node_id": "9", "batch_index": i})
    replay = {"output_rasters": rasters, "preflight_check": {"compared_output_nodes": ["9"]}, "evaluation": {
        "exact_pixel_match": True, "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "compared_image_count": 2,
    }}
    assert _verified_replay_positions(tmp_path, artifact, replay) == {("9", 0)}

    second = tmp_path / rasters[1]["file"]
    second.write_bytes(target)
    assert _verified_replay_positions(tmp_path, artifact, replay) == set()
    rasters[1]["sha256"] = sha256_file(second)
    assert _verified_replay_positions(tmp_path, artifact, replay) == {("9", 0), ("9", 1)}
    replay["preflight_check"]["compared_output_nodes"].append("10")
    assert _verified_replay_positions(tmp_path, artifact, replay) == set()


def test_an_image_link_shows_the_regenerated_output_it_matched_and_says_when_the_output_is_ambiguous(tmp_path):
    import base64
    import hashlib

    from comfytrace.report import _Report, effective_findings
    from comfytrace.workbench import workbench_data

    derived = tmp_path / "exports" / "derived"
    derived.mkdir(parents=True)
    data = _png("purple", (12, 12))
    (derived / "DERIVED_r.png").write_bytes(data)
    selected = {"prompt_id": "p", "file": "exports/derived/DERIVED_r.png", "sha256": hashlib.sha256(data).hexdigest(),
                "handoff_id": "REGEN-001", "node_id": "4", "batch_index": 0}
    other = {**selected, "file": "exports/derived/DERIVED_r2.png", "batch_index": 1}

    def run(**extra):
        manifest = {"schema_version": "1.0.0", "case_metadata": {
            "case_id": "C", "memory_image": {}, "symbol_resolution": {}, "carving_mode": "PROCESS_VIRTUAL",
            "recovery_stages": {}}, "process_context": [],
            "findings": {"tier_summary": {}, "artifacts": [_artifact("TENSOR-001", carve_type="TENSOR_IMAGE")],
                         "workflows": [_workflow("WFPY-001", prompt_id="p")], "correlation_links": [],
                         "findings": [_finding("FIND-1", "UNTIERED", ["TENSOR-001"]),
                                      _finding("FIND-2", "E5", ["WFPY-001"])]}}
        links = {"manifest_sha256": "ab" * 32, "promotions": [
            {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p", "artifact_id": "TENSOR-001",
             "workflow_ids": ["WFPY-001"], **extra}]}
        regenerations = [{
            "handoff_id": "REGEN-001", "workflow_id": "WFPY-001", "status": "COMPLETED",
            "target_manifest_sha256": "ab" * 32,
            "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
            "output_rasters": [selected, other], "derived_regenerated_image": selected["file"],
            "evaluation": {"derived_regenerated_image_sha256": selected["sha256"]},
        }]
        findings, notes = effective_findings(manifest, regenerations, "ab" * 32, tmp_path, links)
        rep = _Report(tmp_path, manifest, "ab" * 32, findings, regenerations, notes, None)
        return findings, notes, rep, workbench_data(rep)

    findings, _, rep, wb = run(selected_raster=selected, matching_rasters=[selected, other],
                               output_attribution="AMBIGUOUS")
    [promoted] = findings
    assert promoted["tier"] == "E2" and promoted["matched_output"] is None
    assert [f["id"] for f in wb["findings"]] == ["FIND-1"]
    assert base64.b64encode(data).decode() in wb["findings"][0]["replay"]["uri"]

    wrong_findings, wrong_notes, _, wrong_wb = run(selected_raster={**selected, "sha256": "0" * 64})
    assert [f["tier"] for f in wrong_findings] == ["UNTIERED", "E5"]
    assert any("verified case-bound raster or candidate replay coverage is missing" in n["note"]
               for n in wrong_notes)
    assert [f["id"] for f in wrong_wb["untiered"]] == ["FIND-1"]

    legacy_findings, legacy_notes, _, _ = run()
    assert [f["tier"] for f in legacy_findings] == ["UNTIERED", "E5"]
    assert legacy_notes[0]["applied_to"] is None


def test_the_job_timeline_orders_jobs_by_start_beside_their_file_write_times(tmp_path):
    from comfytrace.report import _Report
    from comfytrace.reportview import _timeline_jobs
    from comfytrace.workbench import workbench_data

    def job(workflow_id, prompt_id, started, filename):
        return {**_workflow(workflow_id, prompt_id=prompt_id, seed=1, output_filenames=[filename],
                            source_locations=[{"queue_number": int(workflow_id[-1])}]),
                "execution_parameters": {
                    "prompt_id": prompt_id, "seed": 1, "output_filenames": [filename],
                    "source_locations": [{"queue_number": int(workflow_id[-1])}],
                    "job_timeline": {"queued_utc": started, "started_utc": started,
                                     "finished_utc": started, "outcome": "success"}}}

    workflows = [job("WFPY-2", "p-late", "2026-11-07T15:40:00.000Z", "b.png"),
                 job("WFPY-1", "p-early", "2026-11-07T15:29:06.982Z", "a.png"),
                 _workflow("WF-9", complete=False, fmt="UI_WORKFLOW")]
    timelines = _timeline_jobs(workflows)
    assert sorted(timelines, key=lambda pid: timelines[pid]["execution_parameters"]["job_timeline"]["started_utc"]) == [
        "p-early", "p-late"]
    manifest = {"schema_version": "1.0.0", "case_metadata": {"case_id": "C", "memory_image": {}},
                "findings": {"tier_summary": {}, "artifacts": [], "workflows": workflows, "correlation_links": [],
                             "findings": [_finding("FIND-1", "E5", ["WFPY-1"]),
                                          _finding("FIND-2", "E5", ["WFPY-2"])]}}
    rep = _Report(tmp_path, manifest, "ab" * 32,
                  [{**f, "base_tier": f["tier"], "extraction_tier": f["tier"], "regeneration": None, "image_link": None}
                   for f in manifest["findings"]["findings"]],
                  [], [], None)
    wb = workbench_data(rep)
    by_id = {f["id"]: f["job"] for f in wb["findings"]}
    assert by_id["FIND-1"] == {"promptId": "p-early", "queued": "2026-11-07T15:29:06.982Z", "duration": "—",
                               "status": "success"}
    assert by_id["FIND-2"] == {"promptId": "p-late", "queued": "2026-11-07T15:40:00.000Z", "duration": "—",
                               "status": "success"}


# ---- replay facts, workflow groups, counts and untiered labels -------------------------------------------------

def _replay_case(tmp_path: Path, evaluation: dict | None = None, *, artifacts=None, findings=None, workflows=None,
                 evidence=None, links=None, regen_workflow="WF-1") -> tuple[Path, str, str]:
    """A case folder from a hand-made manifest with one verified regeneration record (REGEN-001) whose evaluation
    is ``evaluation``. Default: two E3 outputs of one workflow graph (IMG-1 with 1 of 512 rows, IMG-2 with 0 of
    512) and an E6 fragment agreeing only with that graph's seed. Returns (case, manifest SHA-256, derived SHA-256)."""
    derived = tmp_path / "exports" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "DERIVED_regen.png").write_bytes(_png("teal", (16, 16)))
    regen_sha = sha256_file(derived / "DERIVED_regen.png")
    rows = {"decoded_rows": 1, "total_rows": 512}
    artifacts = artifacts if artifacts is not None else [
        _artifact("IMG-1", "PARTIAL", "PARTIAL_PNG", derived_preview_metadata=rows),
        _artifact("IMG-2", "PARTIAL", "PARTIAL_PNG", derived_preview_metadata={**rows, "decoded_rows": 0})]
    workflows = workflows if workflows is not None else [
        {**_workflow("WF-1", seed=777), "canonical_json_sha256": "ab" * 32},
        {**_workflow("WF-F", complete=False, fmt="UI_FRAGMENT", seed=777), "canonical_json_sha256": "cd" * 32}]
    findings = findings if findings is not None else [
        _finding("FIND-1", "E3", ["IMG-1", "WF-1"], "1 copy of one output (partial)"),
        _finding("FIND-2", "E3", ["IMG-2", "WF-1"], "1 copy of another output (partial)"),
        _finding("FIND-3", "E6", ["WF-F"], "3 page fragments agreeing only with seed 777")]
    manifest = {"schema_version": "1.0.0",
                "case_metadata": {"case_id": "C-R", "comfytrace_version": "1.0.0", "carving_mode": "PROCESS_VIRTUAL",
                                  "memory_image": {"path": "mem.raw", "size_bytes": 1, "sha256": "ee" * 32},
                                  "symbol_resolution": {}, "recovery_stages": {}},
                "process_context": [],
                "findings": {"tier_summary": {}, "artifacts": artifacts, "workflows": workflows,
                             "correlation_links": list(links or []), "findings": findings}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sha = sha256_file(tmp_path / "manifest.json")
    if evidence is None:
        evidence = {"records": [{"record_id": "WF-F", "fragment": {
            "attribution": "ONE_SEED", "agreeing_seeds": [777], "cut_before": True, "cut_after": False}}]}
    (tmp_path / "recovery-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    if evaluation is not None:
        raster = {"file": "exports/derived/DERIVED_regen.png", "sha256": regen_sha, "handoff_id": "REGEN-001",
                  "node_id": "9", "batch_index": 0}
        record = {"handoff_id": "REGEN-001", "workflow_id": regen_workflow, "target_manifest_sha256": sha,
                  "status": "COMPLETED", "derived_regenerated_image": raster["file"],
                  "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
                  "security_audit": {"model_format": "SAFETENSORS"},
                  "evaluation": {"promoted_finding_tier": None, "target_artifact_id": "IMG-1",
                                 "ssim_threshold": 0.95, "phash_threshold": 4, "selected_raster": raster,
                                 "derived_regenerated_image_sha256": regen_sha, **evaluation}}
        (tmp_path / "regenerations").mkdir(exist_ok=True)
        (tmp_path / "regenerations" / "REGEN-001.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path, sha, regen_sha


def _replay_report(case: Path):
    from comfytrace.report import _Report, effective_findings, load_regenerations

    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    sha = sha256_file(case / "manifest.json")
    regens = load_regenerations(case / "regenerations")
    evidence = json.loads((case / "recovery-evidence.json").read_text(encoding="utf-8"))
    findings, notes = effective_findings(manifest, regens, sha, case)
    return _Report(case, manifest, sha, findings, regens, notes, evidence)


INCONCLUSIVE_ROWS = {"evaluated_tier": "INCONCLUSIVE", "evaluation_subtier": "INSUFFICIENT_ROWS",
                     "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 1, "ssim_score": None,
                     "ssim_unavailable_reason": "window_too_small", "phash_hamming_distance": 0,
                     "mean_abs_diff": 3.25, "max_abs_diff": 41, "fraction_above_tolerance": 0.125,
                     "exact_pixel_match": False, "row_slice_pixel_match": False}


def test_a_replay_is_three_separate_facts_and_its_comparison_never_reaches_a_sibling_output(tmp_path):
    case, _, regen_sha = _replay_case(tmp_path, INCONCLUSIVE_ROWS)
    rep = _replay_report(case)
    by_id = {f["finding_id"]: f for f in rep.findings}
    [tested] = rep.replay_summaries(by_id["FIND-1"])
    assert tested["scope"] == "TESTED"
    assert tested["execution"]["derived_file_sha256"] == regen_sha
    assert (tested["execution"]["output_node"], tested["execution"]["batch_index"]) == ("9", 0)
    assert tested["comparison"]["status"] == "INCONCLUSIVE" and tested["comparison"]["match"] is False
    label = tested["comparison"]["label"]
    assert "1 of 512 rows are too few for a 2D SSIM" in label and "no match is claimed" in label
    assert "mean absolute difference 3.25" in label and "max difference 41" in label
    assert "match with" not in label and "close" not in label
    assert tested["promotion"] == {"applied": False, "from": "E3", "to": "E3", "changed": False,
                                   "label": "not applied: this comparison raises no tier; the tier stays E3"}

    [sibling] = rep.replay_summaries(by_id["FIND-2"])
    assert sibling["scope"] == "UNTESTED_SIBLING"
    assert sibling["comparison"]["status"] == "NOT_COMPARED"
    assert sibling["comparison"]["label"] == "this artifact was not compared; REGEN-001 compared IMG-1 (FIND-1)"
    assert "ssim" not in sibling["comparison"] and "mean_abs_diff" not in sibling["comparison"]
    assert sibling["artifacts"]["IMG-2"]["status"] == "NOT_COMPARED"
    assert sibling["execution"]["derived_file_sha256"] == regen_sha, "the run itself is the group's fact"
    assert rep.replay_summaries(by_id["FIND-3"]) == [], "the E6 fragment is another graph"

    from comfytrace.workbench import workbench_data

    data = workbench_data(rep)
    by_wb = {f["id"]: f for f in data["findings"]}
    one = by_wb["FIND-1"]
    assert one["replay"]["execution"] == tested["execution"]["label"]
    assert one["replay"]["comparisonStatus"] == "INCONCLUSIVE"
    assert one["replay"]["promotion"] == tested["promotion"]["label"]
    two = by_wb["FIND-2"]
    assert two["replay"]["summary"] == "this artifact was not compared; REGEN-001 compared IMG-1 (FIND-1)"
    assert two["images"][0]["comparison"]["status"] == "NOT_COMPARED"


def test_an_unverified_replay_record_is_not_shown_as_a_lab_run(tmp_path):
    case, _, _ = _replay_case(tmp_path, INCONCLUSIVE_ROWS)
    (case / "exports" / "derived" / "DERIVED_regen.png").write_bytes(_png("red", (16, 16)))
    rep = _replay_report(case)
    assert all(rep.replay_summaries(f) == [] for f in rep.findings)


@pytest.mark.parametrize(("evaluation", "status", "match", "phrase"), [
    ({"evaluated_tier": "E1", "evaluation_subtier": "EXACT_MATCH", "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
      "exact_pixel_match": True}, "EXACT_MATCH", True, "exact pixel match with IMG-1"),
    ({"evaluated_tier": "E1", "evaluation_subtier": "PERCEPTUAL_MATCH", "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
      "ssim_score": 0.97, "phash_hamming_distance": 2}, "PERCEPTUAL_MATCH", True,
     "close perceptual match with IMG-1 (SSIM 0.970, pHash distance 2)"),
    ({"evaluated_tier": "E1", "evaluation_subtier": "PERCEPTUAL_MATCH", "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
      "ssim_score": 0.5, "phash_hamming_distance": 2}, "UNSUPPORTED_MATCH", False, "not treated as a match"),
    ({"evaluated_tier": "PARTIAL_CORROBORATION", "evaluation_subtier": "ROW_SLICE_EXACT",
      "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 28, "row_slice_pixel_match": True},
     "ROW_SLICE_EXACT", True, "exact match of IMG-1 over its 28 of 512 rows recovered"),
    ({"evaluated_tier": "PARTIAL_CORROBORATION", "evaluation_subtier": "ROW_SLICE_PERCEPTUAL",
      "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 10, "ssim_score": 0.997,
      "phash_hamming_distance": 4}, "ROW_SLICE_PERCEPTUAL", True,
     "close perceptual match of IMG-1 over its 10 of 512 rows recovered (SSIM 0.997, pHash distance 4)"),
    (INCONCLUSIVE_ROWS, "INCONCLUSIVE", False, "comparison of IMG-1 inconclusive"),
    ({"evaluated_tier": "INCONCLUSIVE", "evaluation_subtier": "INSUFFICIENT_DIMENSIONS",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "ssim_score": None, "mean_abs_diff": 0.5},
     "INCONCLUSIVE", False, "the image is too thin for a 2D SSIM; no match is claimed (mean absolute difference 0.50)"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "ANIMATED_COMPARISON_UNAVAILABLE",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION"}, "ANIMATED_COMPARISON_UNAVAILABLE", False,
     "sequence comparison unavailable for IMG-1"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "BELOW_THRESHOLDS",
      "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 1, "ssim_score": 0.0,
      "phash_hamming_distance": 0}, "NON_MATCHING", False,
     "compared IMG-1 (1 of 512 rows) and differed: SSIM 0.000, pHash distance 0"),
    ({"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "DIMENSION_MISMATCH",
      "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "dimension_mismatch": {"regenerated": [8, 8], "target": [9, 9]}},
     "NON_MATCHING", False, "differed: the dimensions do not agree (regenerated [8, 8], target [9, 9])"),
    ({"evaluated_tier": "E4", "evaluation_subtier": "EXECUTABLE_REPRODUCTION", "evaluation_mode": "NO_MEMORY_TARGET",
      "target_artifact_id": None}, "NO_MEMORY_TARGET", False, "workflow ran in the lab; no comparable memory pixels"),
    ({"evaluated_tier": "SOMETHING_NEW"}, "UNRECOGNISED", False, "not treated as a match"),
])
def test_every_recorded_comparison_outcome_reads_honestly(evaluation, status, match, phrase):
    from comfytrace.report import comparison_fact

    evaluation = {"target_artifact_id": "IMG-1", "ssim_threshold": 0.95, "phash_threshold": 4, **evaluation}
    artifact = _artifact("IMG-1", "PARTIAL", "PARTIAL_PNG", derived_preview_metadata={"total_rows": 512})
    fact = comparison_fact(evaluation, artifact)
    assert (fact["status"], fact["match"]) == (status, match)
    assert phrase in fact["label"]
    assert "Not replayed" not in fact["label"]


def test_a_non_matching_replay_on_the_tested_finding_is_shown_not_hidden(tmp_path):
    case, _, _ = _replay_case(tmp_path, {"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "BELOW_THRESHOLDS",
                                         "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 1,
                                         "ssim_score": 0.0, "phash_hamming_distance": 0})
    rep = _replay_report(case)
    finding = next(f for f in rep.findings if f["finding_id"] == "FIND-1")
    assert finding["regeneration"] is None and finding["tier"] == "E3", "promotion gates are unchanged"
    [summary] = rep.replay_summaries(finding)
    assert summary["comparison"]["status"] == "NON_MATCHING"
    assert "a 2D SSIM over 1 row carries little weight" in summary["comparison"]["label"]


def test_the_summary_counts_findings_workflow_graphs_and_artifacts_separately(tmp_path):
    from comfytrace.workbench import workbench_data

    case, _, _ = _replay_case(tmp_path)
    counts = workbench_data(_replay_report(case))["counts"]
    assert counts["linked_findings"] == 2
    assert counts["linked_image_artifacts"] == 2
    assert counts["linked_workflow_graphs"] == 1
    assert counts["prompt_only_findings"] == 1


def test_a_close_match_of_a_prompt_with_several_tiered_outputs_stays_at_workflow_level():
    from comfytrace.report import apply_image_links

    def finding(fid, tier, ids):
        return {"finding_id": fid, "tier": tier, "base_tier": tier, "extraction_tier": tier, "artifact_ids": ids,
                "link_ids": [], "rationale": "", "regeneration": None, "image_link": None}

    findings = [finding("FIND-1", "E2", ["IMG-A", "WF-1"]), finding("FIND-2", "E2", ["IMG-B", "WF-1"]),
                finding("FIND-3", "UNTIERED", ["BITMAP-1"])]
    links = {"manifest_sha256": "ab" * 32, "promotions": [
        {"tier": "E3", "basis": "REGENERATION_CLOSE_MATCH", "prompt_id": "job-a", "artifact_ids": ["BITMAP-1"],
         "workflow_ids": ["WF-1"], "agreement": {}, "regenerated": "x.png"}]}
    notes = apply_image_links(findings, links, "ab" * 32)
    by_id = {f["finding_id"]: f for f in findings}
    assert by_id["FIND-1"]["artifact_ids"] == ["IMG-A", "WF-1"] and by_id["FIND-1"]["image_link"] is None
    assert by_id["FIND-2"]["artifact_ids"] == ["IMG-B", "WF-1"] and by_id["FIND-2"]["image_link"] is None
    scoped = by_id["FIND-3"]
    assert (scoped["tier"], scoped["artifact_ids"]) == ("E3", ["BITMAP-1", "WF-1"])
    assert scoped["close_match_scope"] == "WORKFLOW" and scoped["output_candidates"] == ["FIND-1", "FIND-2"]
    assert "not shown to be that of FIND-1, FIND-2" in scoped["rationale"]
    assert "join none of them" in scoped["rationale"]
    assert notes == [{"promotion": "BITMAP-1", "applied_to": "FIND-3", "note": "REGENERATION_CLOSE_MATCH: "
                      "output not shown to be that of FIND-1, FIND-2; kept at workflow level"}]


def test_a_size_excluded_linked_image_names_its_reason_and_its_link():
    rationale = ("Image of 1 x 1 pixels STRICT-linked to WF-1 via EMBEDDED_PROMPT: below 16 pixels a side, smaller "
                 "than any picture ComfyUI's latent nodes produce, so it is not counted as a job's output.")
    link = {"link_id": "LINK-002", "workflow_id": "WF-1", "artifact_id": "IMG-1", "link_class": "STRICT",
            "matched_keys": ["EMBEDDED_PROMPT"], "confidence_notes": "n"}
    _, data, _ = _unit_rep(
        [_finding("FIND-2", "UNTIERED", ["IMG-1"], rationale, links=["LINK-002"]),
         _finding("FIND-9", "UNTIERED", ["IMG-9"], "Partial image with no corroborating workflow key")],
        [_workflow("WF-1")], [_artifact("IMG-1"), _artifact("IMG-9")], [link],
    )
    by_id = {f["id"]: f for f in data["untiered"]}
    assert by_id["FIND-2"]["tierLabel"] == "Linked image excluded by size"
    assert "below 16 pixels a side" in by_id["FIND-2"]["tierDesc"]
    assert "LINK-002 (STRICT, EMBEDDED_PROMPT) to WF-1" in by_id["FIND-2"]["tierDesc"]
    assert by_id["FIND-9"]["tierLabel"] == "Image not tied to a job"


# ---- supervisory review fixes: container provenance, output identity, raw whole frames, legacy SSIM, E4 -----------

def _zero_row_manifest(promotion_tier="E2"):
    """An untiered memory image, and an E5 whose linked PNG container decoded zero rows, for one job."""
    container = _artifact("CARVE-PNG-303", "PARTIAL", "PARTIAL_PNG", container_sha256="12" * 32,
                          derived_preview_metadata={"decoded_rows": 0, "total_rows": 1024})
    link = {"link_id": "LINK-004", "workflow_id": "WF-5", "artifact_id": "CARVE-PNG-303", "link_class": "STRICT",
            "matched_keys": ["EMBEDDED_PROMPT"], "confidence_notes": "n"}
    manifest = {"schema_version": "1.0.0",
                "case_metadata": {"case_id": "C-Z", "memory_image": {}, "symbol_resolution": {},
                                  "carving_mode": "PROCESS_VIRTUAL", "recovery_stages": {}},
                "process_context": [],
                "findings": {"tier_summary": {}, "correlation_links": [link],
                             "artifacts": [_artifact("TENSOR-001", carve_type="TENSOR_IMAGE"), container],
                             "workflows": [_workflow("WF-5", prompt_id="job-5")],
                             "findings": [_finding("FIND-1", "UNTIERED", ["TENSOR-001"]),
                                          _finding("FIND-2", "E5", ["CARVE-PNG-303", "WF-5"],
                                                   "1 copy of one output (partial). Container provenance only.",
                                                   links=["LINK-004"])]}}
    if promotion_tier == "E2":
        promo = {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-5",
                 "artifact_id": "TENSOR-001", "workflow_ids": ["WF-5"]}
    else:
        promo = {"tier": "E3", "basis": "REGENERATION_CLOSE_MATCH", "prompt_id": "job-5",
                 "artifact_ids": ["TENSOR-001"], "workflow_ids": ["WF-5"], "agreement": {}}
    return manifest, {"manifest_sha256": "ab" * 32, "promotions": [promo]}


@pytest.mark.parametrize("promotion_tier", ["E2", "E3"])
def test_an_absorbed_zero_row_container_is_kept_as_container_provenance_with_its_link(tmp_path, promotion_tier):
    from comfytrace.report import _Report, effective_findings
    from comfytrace.workbench import workbench_data

    manifest, links = _zero_row_manifest(promotion_tier)
    findings, notes = effective_findings(manifest, [], "ab" * 32, None, links)
    [promoted] = findings
    assert promoted["tier"] == promotion_tier and promoted["absorbed"] == ["FIND-2"]
    assert "CARVE-PNG-303" not in promoted["artifact_ids"], "a zero-row header is not an output image"
    assert promoted["container_provenance"] == [{"finding_id": "FIND-2", "tier": "E5",
                                                 "artifact_ids": ["CARVE-PNG-303"], "link_ids": ["LINK-004"]}]
    assert promoted["container_provenance_artifacts"] == ["CARVE-PNG-303"]
    assert promoted["container_provenance_links"] == ["LINK-004"]

    rep = _Report(tmp_path, manifest, "ab" * 32, findings, [], notes, None)
    assert rep.containers(promoted, []) == [{"artifact_id": "CARVE-PNG-303", "from_finding": "FIND-2",
                                             "link_ids": ["LINK-004"], "rows_text": "0 of 1024 rows"}]
    shown = workbench_data(rep)
    [wb] = shown["findings"]
    assert [i["id"] for i in wb["images"]] == ["TENSOR-001"]
    assert wb["containerProvenance"] == [{"artifactId": "CARVE-PNG-303", "rows": "0 of 1024 rows",
                                          "fromFinding": "FIND-2", "linkIds": ["LINK-004"]}]


def _batch_findings(home_output):
    def finding(fid, tier, ids, **extra):
        return {"finding_id": fid, "tier": tier, "base_tier": tier, "extraction_tier": tier, "artifact_ids": ids,
                "link_ids": [], "rationale": "", "regeneration": None, "image_link": None, **extra}

    return [finding("FIND-1", "E2", ["PIXBUF-A", "WF-1"], matched_output=home_output),
            finding("FIND-2", "UNTIERED", ["BITMAP-B"])]


def _close(node, batch, attribution="SUPPORTED", jobs=None):
    raster = {"prompt_id": "job-a", "file": f"exports/derived/r{batch}.png", "sha256": "0" * 64,
              "handoff_id": "REGEN-1", "node_id": node, "batch_index": batch, "verified": True}
    evidence = {"selected_raster": raster, "matching_rasters": [raster], "output_attribution": attribution}
    links = {"manifest_sha256": "ab" * 32, "promotions": [
        {"tier": "E3", "basis": "REGENERATION_CLOSE_MATCH", "prompt_id": "job-a", "artifact_ids": ["BITMAP-B"],
         "workflow_ids": ["WF-1"], "agreement": {}, "selected_raster": raster,
         "raster_evidence_by_artifact": {"BITMAP-B": evidence}}]}
    if jobs is not None:
        links["jobs"] = [{"prompt_id": "job-a", "rasters": jobs}]
    return links


def _verified_run(handoff, *positions, count=None, workflow="WF-1"):
    """A verified regeneration record of ``workflow`` whose complete output inventory is ``positions``."""
    rasters = [{"file": f"exports/derived/{handoff}_{n}_{b}.png", "sha256": "0" * 64, "node_id": n, "batch_index": b}
               for n, b in positions]
    return {"handoff_id": handoff, "workflow_id": workflow, "target_manifest_sha256": "ab" * 32,
            "status": "COMPLETED", "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
            "security_audit": {"model_format": "SAFETENSORS"}, "output_rasters": rasters,
            "evaluation": {"compared_image_count": len(rasters) if count is None else count}}


@pytest.mark.parametrize(("home_output", "links", "regens", "joins"), [
    (["9", 0], _close("9", 0), None, True),  # both shown to be output (9, 0)
    (["9", 0], _close("9", 1), None, False),  # the copy is output B, the finding output A
    (None, _close("9", 1), None, False),  # the finding's output is not known, and no inventory
    (["9", 1], _close("9", 1, "AMBIGUOUS"), None, False),  # the copy's output is not shown
    # Two runs of the job, one output position between them: one output, so the copy is the finding's output.
    (None, _close("9", 0), [_verified_run("REGEN-1", ("9", 0)), _verified_run("REGEN-2", ("9", 0))], True),
    (None, _close("9", 0), [_verified_run("REGEN-1", ("9", 0), ("9", 1))], False),  # a batch of two
    # A record whose compared_image_count disagrees with its rasters does not establish one output.
    (None, _close("9", 0), [_verified_run("REGEN-1", ("9", 0), count=2)], False),
    # A run of another manifest is not this job's inventory.
    (None, _close("9", 0), [{**_verified_run("REGEN-1", ("9", 0)), "target_manifest_sha256": "cd" * 32}], False),
    # The links file's job summary lists one raster; the job summary alone never establishes one output.
    (None, _close("9", 0, jobs=[{"node_id": "9", "batch_index": 0}]), None, False),
])
def test_a_close_match_joins_an_existing_finding_only_when_both_are_the_same_output(home_output, links, regens,
                                                                                    joins):
    from comfytrace.report import apply_image_links

    findings = _batch_findings(home_output)
    [note] = apply_image_links(findings, links, "ab" * 32, None, regens)
    by_id = {f["finding_id"]: f for f in findings}
    if joins:
        assert set(by_id) == {"FIND-1"} and by_id["FIND-1"]["artifact_ids"] == ["PIXBUF-A", "WF-1", "BITMAP-B"]
        assert by_id["FIND-1"]["matched_output"] == ["9", 0]
        assert note["note"] == "REGENERATION_CLOSE_MATCH: joined FIND-1 (E2)"
    else:
        assert by_id["FIND-1"]["artifact_ids"] == ["PIXBUF-A", "WF-1"], "output A keeps only its own copy"
        scoped = by_id["FIND-2"]
        assert (scoped["tier"], scoped["close_match_scope"], scoped["output_candidates"]) == (
            "E3", "WORKFLOW", ["FIND-1"])
        assert note == {"promotion": "BITMAP-B", "applied_to": "FIND-2", "note": "REGENERATION_CLOSE_MATCH: output "
                        "not shown to be that of FIND-1; kept at workflow level"}


@pytest.mark.parametrize("regens", [
    None,
    [_verified_run("REGEN-1", ("9", 1))],  # even a one-position inventory cannot override the conflict
    [_verified_run("REGEN-1", ("9", 0), ("9", 1))],
])
def test_an_explicit_output_conflict_always_vetoes_the_close_match_join(regens):
    """Finding A is output (9, 0); copy B is shown to be output (9, 1); the job summary lists only B's raster
    (another output was excluded from still-image linkage). B must not join A, and A keeps its own position."""
    from comfytrace.report import apply_image_links

    findings = _batch_findings(["9", 0])
    links = _close("9", 1, jobs=[{"node_id": "9", "batch_index": 1, "handoff_id": "REGEN-1"}])
    [note] = apply_image_links(findings, links, "ab" * 32, None, regens)
    by_id = {f["finding_id"]: f for f in findings}
    assert by_id["FIND-1"]["artifact_ids"] == ["PIXBUF-A", "WF-1"] and by_id["FIND-1"]["matched_output"] == ["9", 0]
    assert by_id["FIND-2"]["close_match_scope"] == "WORKFLOW" and by_id["FIND-2"]["matched_output"] == ["9", 1]
    assert note["applied_to"] == "FIND-2" and "kept at workflow level" in note["note"]


def test_a_single_output_inventory_never_joins_a_finding_shown_to_be_another_output():
    from comfytrace.report import apply_image_links

    findings = _batch_findings(["9", 3])
    [note] = apply_image_links(findings, _close("9", 0, "AMBIGUOUS"), "ab" * 32, None,
                               [_verified_run("REGEN-1", ("9", 0))])
    assert "kept at workflow level" in note["note"]


def test_a_whole_frame_image_link_records_the_output_it_matched():
    from comfytrace.report import apply_image_links

    findings = _batch_findings(None)[1:] + [{"finding_id": "FIND-3", "tier": "UNTIERED", "base_tier": "UNTIERED",
                                             "extraction_tier": "UNTIERED", "artifact_ids": ["PIXBUF-A"],
                                             "link_ids": [], "rationale": "", "regeneration": None,
                                             "image_link": None}]
    raster = {"node_id": "9", "batch_index": 0}
    links = {"manifest_sha256": "ab" * 32, "promotions": [
        {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-a", "artifact_id": "PIXBUF-A",
         "workflow_ids": ["WF-1"], "selected_raster": raster, "output_attribution": "SUPPORTED"},
        *_close("9", 1)["promotions"]]}
    apply_image_links(findings, links, "ab" * 32)
    by_id = {f["finding_id"]: f for f in findings}
    assert by_id["FIND-3"]["matched_output"] == ["9", 0]
    assert by_id["FIND-2"]["close_match_scope"] == "WORKFLOW", "batch output B does not join output A"


def _raw_replay_case(tmp_path):
    """FIND-1 promoted to E1 by a raw whole-frame match of REGEN-001's batch output 1; REGEN-001 itself was recorded
    as NO_MEMORY_TARGET with batch output 0 selected."""
    no_target = {"evaluated_tier": "E4", "evaluation_subtier": "EXECUTABLE_REPRODUCTION",
                 "evaluation_mode": "NO_MEMORY_TARGET", "target_artifact_id": None}
    case, _, first_sha = _replay_case(tmp_path, no_target)
    second = case / "exports" / "derived" / "DERIVED_regen_b1.png"
    second.write_bytes(_png("orange", (16, 16)))
    second_sha = sha256_file(second)
    path = case / "regenerations" / "REGEN-001.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["output_rasters"] = [
        {"file": "exports/derived/DERIVED_regen.png", "sha256": first_sha, "node_id": "9", "batch_index": 0},
        {"file": "exports/derived/DERIVED_regen_b1.png", "sha256": second_sha, "node_id": "9", "batch_index": 1}]
    path.write_text(json.dumps(record), encoding="utf-8")
    rep = _replay_report(case)
    finding = next(f for f in rep.findings if f["finding_id"] == "FIND-1")
    assert finding["regeneration"] == "REGEN-001", "the E3 executable-reproduction rule attached the run"
    finding.update(tier="E1", base_tier="E2", image_link="RAW_REPLAY_WHOLE_FRAME", replay_evidence={
        "handoff_id": "REGEN-001", "regenerated_image": "exports/derived/DERIVED_regen_b1.png",
        "regenerated_sha256": second_sha, "physical_offset": 0x4000, "length_bytes": 768,
        "raw_sha256": "77" * 32, "raw_file": f"exports/raw/replay-search/{'77' * 32}.bin", "subtier": "EXACT_MATCH",
        "metrics": {"exact_pixel_match": True, "ssim_score": 1.0, "phash_hamming_distance": 0},
        "pixel_sha256": "88" * 32})
    return case, rep, finding, second


def test_a_raw_whole_frame_e1_is_its_own_observation_with_the_matched_output(tmp_path):
    import base64

    case, rep, finding, second = _raw_replay_case(tmp_path)
    raw, original = rep.replay_summaries(finding)
    assert (raw["observation"], raw["scope"]) == ("RAW_WHOLE_FRAME", "RAW_WHOLE_FRAME")
    assert raw["execution"]["derived_path"] == "exports/derived/DERIVED_regen_b1.png"
    assert (raw["execution"]["output_node"], raw["execution"]["batch_index"]) == ("9", 1)
    assert raw["comparison"]["status"] == "EXACT_MATCH" and raw["comparison"]["match"] is True
    assert "physical offset 0x4000" in raw["comparison"]["label"]
    assert raw["promotion"]["applied"] is True and "raised this finding to E1" in raw["promotion"]["label"]
    assert original["observation"] == "ORIGINAL_EVALUATION" and original["promotion"]["applied"] is False
    assert original["comparison"]["label"].startswith("original lab comparison: ")
    assert "raised no tier" in original["promotion"]["label"]
    assert original["execution"]["derived_path"] == "exports/derived/DERIVED_regen.png"

    from comfytrace.workbench import workbench_data

    one = next(f for f in workbench_data(rep)["findings"] if f["id"] == "FIND-1")
    assert one["replay"]["observation"] == "RAW_WHOLE_FRAME"
    assert one["replay"]["derivedPath"] == "exports/derived/DERIVED_regen_b1.png"
    assert base64.b64encode(second.read_bytes()).decode() in one["replay"]["uri"], "the figure is the matched output"


def test_the_workbench_leads_with_the_raw_whole_frame_observation(tmp_path):
    from comfytrace.workbench import workbench_data

    _, rep, _, _ = _raw_replay_case(tmp_path)
    one = next(f for f in workbench_data(rep)["findings"] if f["id"] == "FIND-1")
    replay = one["replay"]
    assert (replay["observation"], replay["result"], replay["match"]) == ("RAW_WHOLE_FRAME", "EXACT_MATCH", True)
    assert replay["derivedPath"] == "exports/derived/DERIVED_regen_b1.png" and replay["batchIndex"] == 1
    assert replay["rawFrame"]["physical_offset"] == 0x4000
    [other] = replay["others"]
    assert other["observation"] == "ORIGINAL_EVALUATION" and other["summary"].startswith("original lab comparison")


def test_a_legacy_strip_ssim_of_zero_is_not_shown_as_a_measurement(tmp_path):
    from comfytrace.report import comparison_fact
    from comfytrace.workbench import render_workbench_html, workbench_data

    legacy = {"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "BELOW_THRESHOLDS",
              "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 1, "ssim_score": 0.0,
              "phash_hamming_distance": 0, "row_slice_pixel_match": False}
    artifact = _artifact("IMG-1", "PARTIAL", "PARTIAL_PNG", derived_preview_metadata={"total_rows": 512})
    fact = comparison_fact({"target_artifact_id": "IMG-1", "ssim_threshold": 0.95, "phash_threshold": 4, **legacy},
                           artifact)
    assert (fact["status"], fact["match"], fact["ssim"], fact["legacy_ssim_placeholder"]) == (
        "INCONCLUSIVE", False, None, True)
    assert "SSIM 0.000" not in fact["label"] and "outside the thresholds" not in fact["label"]
    assert "the recovered strip is not pixel-identical to the replay's rows" in fact["label"]
    assert "2D SSIM is unavailable for fewer than 3 rows" in fact["label"]

    case, _, _ = _replay_case(tmp_path, legacy)
    data = workbench_data(_replay_report(case))
    one = next(f for f in data["findings"] if f["id"] == "FIND-1")
    assert one["replay"]["ssim"] is None
    assert "SSIM 0.000" not in one["replay"]["summary"]
    assert "SSIM 0.000" not in render_workbench_html(data)
    assert json.loads((case / "regenerations" / "REGEN-001.json").read_text())["evaluation"]["ssim_score"] == 0.0


def _e4_case(tmp_path, **evaluation):
    findings = [_finding("FIND-1", "E5", ["WF-1"]), _finding("FIND-2", "E5", ["WF-1"]),
                _finding("FIND-3", "E2", ["TENSOR-1", "WF-1"])]
    return _replay_case(tmp_path, {"promoted_finding_tier": "E4", "evaluated_tier": "E4", "target_artifact_id": None,
                                   "evaluation_subtier": "EXECUTABLE_REPRODUCTION",
                                   "evaluation_mode": "NO_MEMORY_TARGET", **evaluation},
                        artifacts=[_artifact("TENSOR-1", carve_type="TENSOR_IMAGE")], findings=findings,
                        workflows=[_workflow("WF-1")], evidence={"records": []})[0]


def test_an_e4_run_applies_to_the_findings_it_names_and_never_by_list_order(tmp_path):
    from comfytrace.report import effective_findings, load_regenerations

    def applied(case):
        manifest = json.loads((case / "manifest.json").read_text())
        findings, notes = effective_findings(manifest, load_regenerations(case / "regenerations"),
                                             sha256_file(case / "manifest.json"), case)
        return {f["finding_id"]: (f["tier"], f["regeneration"]) for f in findings}, notes

    named, notes = applied(_e4_case(tmp_path / "named", eligible_e4_finding_ids=["FIND-2"]))
    assert named == {"FIND-1": ("E5", None), "FIND-2": ("E4", "REGEN-001"), "FIND-3": ("E2", None)}
    assert [n["applied_to"] for n in notes if n.get("handoff_id") == "REGEN-001"] == ["FIND-2"]
    legacy, notes = applied(_e4_case(tmp_path / "legacy"))
    assert legacy == {"FIND-1": ("E4", "REGEN-001"), "FIND-2": ("E4", "REGEN-001"), "FIND-3": ("E2", None)}
    assert [n["applied_to_all"] for n in notes if n.get("handoff_id") == "REGEN-001"] == [["FIND-1", "FIND-2"]]


def test_a_linked_zero_row_prompt_finding_names_its_container():
    container = _artifact("CARVE-PNG-303", "PARTIAL", "PARTIAL_PNG",
                          derived_preview_metadata={"decoded_rows": 0, "total_rows": 1024})
    link = {"link_id": "LINK-004", "workflow_id": "WF-5", "artifact_id": "CARVE-PNG-303", "link_class": "STRICT",
            "matched_keys": ["EMBEDDED_PROMPT"], "confidence_notes": "n"}
    rep, data, _ = _unit_rep(
        [_finding("FIND-2", "E5", ["CARVE-PNG-303", "WF-5"], links=["LINK-004"]),
         _finding("FIND-3", "E6", ["CARVE-PNG-303", "WF-6"], links=["LINK-004"])],
        [_workflow("WF-5"), _workflow("WF-6", complete=False, fmt="API_FRAGMENT")], [container], [link],
    )
    by_id = {f["id"]: f for f in data["findings"]}
    assert by_id["FIND-2"]["images"][0]["container"] is True
    assert by_id["FIND-2"]["images"][0]["rows"] == "0 of 1024 rows"
    assert [lnk["id"] for lnk in by_id["FIND-2"]["links"]] == ["LINK-004"]
    assert by_id["FIND-3"]["images"][0]["container"] is True
    assert by_id["FIND-3"]["images"][0]["rows"] == "0 of 1024 rows"
    assert rep.containers(rep.findings[0], rep.parts(rep.findings[0])[0]) == [
        {"artifact_id": "CARVE-PNG-303", "from_finding": None, "link_ids": ["LINK-004"], "rows_text": "0 of 1024 rows"}
    ]


# ---- round 3: E3 attachment ignores the E4 filter, legacy strips read INCONCLUSIVE, per-observation application -----

def test_an_empty_e4_eligibility_list_does_not_block_attaching_a_run_to_an_e3_finding(tmp_path):
    from comfytrace.report import effective_findings, load_regenerations

    def attached(case):
        manifest = json.loads((case / "manifest.json").read_text())
        findings, _ = effective_findings(manifest, load_regenerations(case / "regenerations"),
                                         sha256_file(case / "manifest.json"), case)
        return {f["finding_id"]: (f["tier"], f["regeneration"]) for f in findings}

    no_target = {"evaluated_tier": "E4", "evaluation_subtier": "EXECUTABLE_REPRODUCTION",
                 "evaluation_mode": "NO_MEMORY_TARGET", "target_artifact_id": None}
    new = attached(_replay_case(tmp_path / "new", {**no_target, "eligible_e4_finding_ids": []})[0])
    legacy = attached(_replay_case(tmp_path / "legacy", no_target)[0])
    assert new == legacy == {"FIND-1": ("E3", "REGEN-001"), "FIND-2": ("E3", "REGEN-001"), "FIND-3": ("E6", None)}


def test_an_empty_e4_eligibility_list_still_raises_no_e5(tmp_path):
    from comfytrace.report import effective_findings, load_regenerations

    case = _e4_case(tmp_path, eligible_e4_finding_ids=[])
    manifest = json.loads((case / "manifest.json").read_text())
    findings, _ = effective_findings(manifest, load_regenerations(case / "regenerations"),
                                     sha256_file(case / "manifest.json"), case)
    assert {f["finding_id"]: f["tier"] for f in findings} == {"FIND-1": "E5", "FIND-2": "E5", "FIND-3": "E2"}


def test_a_legacy_thin_strip_reads_inconclusive_and_keeps_its_recorded_verdict(tmp_path):
    from comfytrace.workbench import workbench_data

    legacy = {"evaluated_tier": "NON_MATCHING", "evaluation_subtier": "BELOW_THRESHOLDS",
              "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 1, "ssim_score": 0.0,
              "phash_hamming_distance": 0, "row_slice_pixel_match": False}
    case, _, _ = _replay_case(tmp_path, legacy)
    rep = _replay_report(case)
    [summary] = rep.replay_summaries(next(f for f in rep.findings if f["finding_id"] == "FIND-1"))
    fact = summary["comparison"]
    assert (fact["status"], fact["subtier"], fact["recorded_verdict"]) == ("INCONCLUSIVE", "BELOW_THRESHOLDS",
                                                                          "BELOW_THRESHOLDS")
    assert "inconclusive" in fact["label"] and "recorded verdict BELOW_THRESHOLDS" in fact["label"]
    data = workbench_data(_replay_report(case))
    one = next(f for f in data["findings"] if f["id"] == "FIND-1")
    assert (one["replay"]["comparisonStatus"], one["replay"]["recordedVerdict"]) == ("INCONCLUSIVE",
                                                                                     "BELOW_THRESHOLDS")
    assert one["replay"]["result"] == "BELOW_THRESHOLDS", "the raw record's verdict is still carried"
    assert one["images"][0]["comparison"]["status"] == "INCONCLUSIVE"
    [group] = [g for g in data["groups"] if g["id"] == one["groupId"]]
    assert (group["replays"][0]["comparisonStatus"], group["replays"][0]["recordedVerdict"]) == (
        "INCONCLUSIVE", "BELOW_THRESHOLDS")
    # A record that never said the strip was unequal is not rewritten.
    newer = {**legacy}
    del newer["row_slice_pixel_match"]
    from comfytrace.report import comparison_fact

    assert comparison_fact({"target_artifact_id": "IMG-1", **newer})["status"] == "NON_MATCHING"


def test_each_group_observation_says_only_what_it_applied_to(tmp_path):
    from comfytrace.workbench import workbench_data

    _, rep, _, _ = _raw_replay_case(tmp_path)
    data = workbench_data(rep)
    one = next(f for f in data["findings"] if f["id"] == "FIND-1")
    [group] = [g for g in data["groups"] if g["id"] == one["groupId"]]
    rows = {r["observation"]: r for r in group["replays"]}
    assert rows["RAW_WHOLE_FRAME"]["appliedTo"] == "FIND-1"
    assert rows["ORIGINAL_EVALUATION"]["comparisonStatus"] == "NO_MEMORY_TARGET"
    assert rows["ORIGINAL_EVALUATION"]["appliedTo"] == "FIND-2", \
        "the original no-target run stays attached to the sibling E3 only, never to the E1 finding"


def test_incomplete_or_duplicate_regen_inventory_does_not_merge_close_match():
    from comfytrace.report import _regen_positions, apply_image_links

    sha = "a" * 64
    base_regen = {"workflow_id": "WF-1", "target_manifest_sha256": sha, "status": "COMPLETED",
                  "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
                  "security_audit": {"model_format": "SAFETENSORS"}}
    # 1) selected_raster only (no output_rasters, no compared_image_count)
    selected_only = {**base_regen, "evaluation": {"selected_raster": {"node_id": "9", "batch_index": 1}}}
    assert _regen_positions(selected_only) is None
    # 2) compared_image_count=2 with duplicate (node_id, batch_index) entries
    dup_positions = {**base_regen,
                     "output_rasters": [{"node_id": "9", "batch_index": 1}, {"node_id": "9", "batch_index": 1}],
                     "evaluation": {"compared_image_count": 2}}
    assert _regen_positions(dup_positions) is None

    for regen in (selected_only, dup_positions):
        findings = [
            {"finding_id": "FIND-A", "tier": "E2", "base_tier": "E2", "extraction_tier": "E2",
             "artifact_ids": ["IMG-A", "WF-1"], "link_ids": ["LINK-A"], "rationale": "E2 unknown output"},
            {"finding_id": "FIND-B", "tier": "UNTIERED", "base_tier": "UNTIERED", "extraction_tier": "UNTIERED",
             "artifact_ids": ["IMG-B"], "link_ids": [], "rationale": "Orphan"},
        ]
        links = {
            "manifest_sha256": sha,
            "promotions": [{
                "tier": "E3",
                "basis": "REPLAY_IMAGE_MATCH_CLOSE",
                "prompt_id": "job-1",
                "workflow_ids": ["WF-1"],
                "artifact_ids": ["IMG-B"],
                "selected_raster": {"node_id": "9", "batch_index": 1},
                "raster_evidence_by_artifact": {
                    "IMG-B": {"status": "SUPPORTED", "selected_raster": {"node_id": "9", "batch_index": 1}},
                },
            }],
        }
        apply_image_links(findings, links, sha, regenerations=[regen])
        assert "IMG-B" not in findings[0]["artifact_ids"]
        wf_findings = [f for f in findings if f.get("close_match_scope") == "WORKFLOW"]
        assert len(wf_findings) == 1 and "IMG-B" in wf_findings[0]["artifact_ids"]


def test_candidate_evaluations_give_each_tested_sibling_its_own_comparison(tmp_path):
    from comfytrace.workbench import workbench_data

    primary = {
        "evaluated_tier": "PARTIAL_CORROBORATION", "evaluation_subtier": "ROW_SLICE_PERCEPTUAL",
        "evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": 16, "ssim_score": 0.998,
        "phash_hamming_distance": 0, "row_slice_pixel_match": False,
        "candidate_evaluations": {
            "IMG-2": {
                "target_artifact_id": "IMG-2", "evaluated_tier": "PARTIAL_CORROBORATION",
                "evaluation_subtier": "ROW_SLICE_PERCEPTUAL", "evaluation_mode": "PARTIAL_ROW_BOUNDED",
                "decoded_rows_evaluated": 10, "ssim_score": 0.991, "phash_hamming_distance": 2,
                "ssim_threshold": 0.95, "phash_threshold": 4, "row_slice_pixel_match": False,
            },
        },
    }
    case, _, _ = _replay_case(tmp_path, primary)
    data = workbench_data(_replay_report(case))
    by_id = {f["id"]: f for f in data["findings"]}
    assert by_id["FIND-1"]["replay"]["scope"] == "TESTED"
    assert by_id["FIND-1"]["replay"]["ssim"] == "0.998"
    assert by_id["FIND-2"]["replay"]["scope"] == "TESTED"
    assert by_id["FIND-2"]["replay"]["testedArtifact"] == "IMG-2"
    assert by_id["FIND-2"]["replay"]["ssim"] == "0.991"
    assert by_id["FIND-2"]["images"][0]["comparison"]["status"] == "ROW_SLICE_PERCEPTUAL"
