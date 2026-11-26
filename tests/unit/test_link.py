"""Linking memory images to jobs through regenerated images."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from comfytrace import latents, link
from comfytrace.report import verify_image_link_promotions

MEMORY_SHA = "ab" * 32


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.repeat(np.repeat(rng.integers(0, 256, (256, 256, 3)), 2, 0), 2, 1).astype(np.uint8)


def _case(tmp_path, memory, partial=()):
    case = tmp_path / "case"
    (case / "exports" / "raw").mkdir(parents=True)
    artifacts, records = [], []
    for artifact_id, kind, rgb in memory:
        raw = rgb.tobytes()
        path = f"exports/raw/{artifact_id}.bin"
        (case / path).write_bytes(raw)
        artifacts.append({"artifact_id": artifact_id, "carve_type": kind, "raw_file_paths": [path],
                          "completeness": "COMPLETE" if kind == "PIXEL_BUFFER" and artifact_id not in partial
                          else "PARTIAL",
                          "pixel_metadata": {"width": rgb.shape[1], "height": rgb.shape[0], "source": {"channels": 3}}})
        records.append({"record_id": artifact_id, "sha256": {"raw": hashlib.sha256(raw).hexdigest()}})
    (case / "manifest.json").write_text(json.dumps({"case_metadata": {"memory_image": {"sha256": MEMORY_SHA}},
                                                    "findings": {"artifacts": artifacts}}))
    (case / "recovery-evidence.json").write_text(json.dumps({"records": records}))
    return case


def _regen(tmp_path, jobs):
    regen = tmp_path / "regen"
    regen.mkdir()
    for prompt_id, rgb in jobs:
        Image.fromarray(rgb).save(regen / f"{prompt_id}.png")
        data = (regen / f"{prompt_id}.png").read_bytes()
        (regen / f"{prompt_id}.json").write_text(json.dumps({"prompt_id": prompt_id,
                                                             "output_sha256": hashlib.sha256(data).hexdigest()}))
    return regen


def test_exact_copy_and_offset_thumbnail_link_to_their_job(tmp_path):
    job_a, job_b = _texture(1), _texture(2)
    damaged = job_a.copy()
    damaged[244:248] = 0
    thumb = job_a.astype(np.float32).reshape(256, 2, 256, 2, 3).mean((1, 3)).round().astype(np.uint8)[1:]
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", damaged), ("BITMAP-001", "DECODED_BITMAP", thumb),
                            ("BITMAP-002", "DECODED_BITMAP", _texture(9)[:300]), ("PIXBUF-002", "PIXEL_BUFFER", job_b)],
                 partial={"PIXBUF-001"})
    result = link.link(case, _regen(tmp_path, [("job-a", job_a), ("job-b", job_b), ("job-c", _texture(3))]))
    by_id = {c["artifact_id"]: c for c in result["memory_images"]}
    assert by_id["PIXBUF-001"]["attribution"] == "SUPPORTED" and by_id["PIXBUF-001"]["prompt_id"] == "job-a"
    assert by_id["BITMAP-001"]["prompt_id"] == "job-a" and by_id["BITMAP-001"]["best"]["scale"] == 2
    assert by_id["BITMAP-002"]["attribution"] == "UNATTRIBUTED"
    jobs = {j["prompt_id"]: j for j in result["jobs"]}
    # A full-size copy with damaged rows is partial evidence; only a whole copy makes a job exact.
    assert jobs["job-a"]["image_status"] == "OBSERVED_PARTIAL_OR_SCALED"
    assert jobs["job-b"]["image_status"] == "OBSERVED_EXACT"
    assert jobs["job-c"]["image_status"] == "REGENERATED_ONLY"
    assert jobs["job-c"]["regenerated"]["residue_check"] == "UNCHECKED"


def test_two_jobs_with_the_same_image_leave_a_memory_image_ambiguous(tmp_path):
    same = _texture(3)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", same)])
    result = link.link(case, _regen(tmp_path, [("job-a", same), ("job-b", same)]))
    (only,) = result["memory_images"]
    assert only["attribution"] == "AMBIGUOUS" and only["prompt_id"] is None
    assert sorted(only["agreeing_jobs"]) == ["job-a", "job-b"]


def test_tampered_raw_bytes_are_refused(tmp_path):
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", _texture(4))])
    (case / "exports/raw/PIXBUF-001.bin").write_bytes(b"\0" * (512 * 512 * 3))
    try:
        link.link(case, _regen(tmp_path, [("job-a", _texture(4))]))
    except ValueError as exc:
        assert "raw bytes" in str(exc)
    else:
        raise AssertionError("tampered evidence was accepted")


def test_partial_bitmap_deep_inside_the_image_links(tmp_path):
    job_a, job_b = _texture(5), _texture(6)
    half = job_a.astype(np.float32).reshape(256, 2, 256, 2, 3).mean((1, 3)).round().astype(np.uint8)
    case = _case(tmp_path, [("BITMAP-001", "DECODED_BITMAP", job_a[150:450]),
                            ("BITMAP-002", "DECODED_BITMAP", half[90:250])])
    result = link.link(case, _regen(tmp_path, [("job-a", job_a), ("job-b", job_b)]))
    by_id = {c["artifact_id"]: c for c in result["memory_images"]}
    assert by_id["BITMAP-001"]["prompt_id"] == "job-a" and by_id["BITMAP-001"]["best"]["row_offset"] == 150
    assert by_id["BITMAP-002"]["prompt_id"] == "job-a" and by_id["BITMAP-002"]["best"]["row_offset"] == 90


def _case_regeneration(case, handoff, workflow_id, prompt_id, rgb, status="COMPLETED"):
    manifest = json.loads((case / "manifest.json").read_text())
    workflows = manifest["findings"].setdefault("workflows", [])
    if not any(w["workflow_id"] == workflow_id for w in workflows):
        workflows.append({"workflow_id": workflow_id, "execution_parameters": {"prompt_id": prompt_id}})
    (case / "manifest.json").write_text(json.dumps(manifest))
    (case / "regenerations").mkdir(exist_ok=True)
    _bind_regenerations(case)
    (case / "exports" / "derived").mkdir(parents=True, exist_ok=True)
    path = f"exports/derived/DERIVED_regen_{workflow_id}.png"
    Image.fromarray(rgb).save(case / path)
    digest = hashlib.sha256((case / path).read_bytes()).hexdigest()
    (case / "regenerations" / f"{handoff}.json").write_text(json.dumps({
        "handoff_id": handoff, "workflow_id": workflow_id, "status": status, "derived_regenerated_image": path,
        "evaluation": {"derived_regenerated_image_sha256": digest},
        "preflight_check": {"status": "PASSED", "runner_object_info_verified": True,
                            "model_match_level": "OBSERVED_IN_MEMORY"}}))
    _bind_regenerations(case)


def _bind_regenerations(case):
    """Points every regeneration record at the case's current manifest, as regenerate does."""
    digest = hashlib.sha256((case / "manifest.json").read_bytes()).hexdigest()
    for path in (case / "regenerations").glob("REGEN-*.json"):
        record = json.loads(path.read_text())
        record["target_manifest_sha256"] = digest
        path.write_text(json.dumps(record))


def test_case_regeneration_records_are_linked_without_a_separate_folder(tmp_path):
    job_a, job_b = _texture(11), _texture(12)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a)])
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", job_a)
    _case_regeneration(case, "REGEN-002", "WFPY-002", "job-b", job_b)
    _case_regeneration(case, "REGEN-003", "WFPY-003", "job-c", job_a, status="SECURITY_REJECTED")
    result = link.link(case)
    (only,) = result["memory_images"]
    assert only["attribution"] == "SUPPORTED" and only["prompt_id"] == "job-a"
    assert only["output_attribution"] == "UNKNOWN"
    assert [r["prompt_id"] for r in result["regenerated"]] == ["job-a", "job-b"]
    assert result["regenerated"][0]["record"]["model_match_level"] == "OBSERVED_IN_MEMORY"
    Image.fromarray(job_b).save(case / "exports/derived/DERIVED_regen_WFPY-001.png")
    try:
        link.link(case)
    except ValueError as exc:
        assert "REGEN-001" in str(exc)
    else:
        raise AssertionError("a regenerated image that no longer matches its record was accepted")


def test_replay_of_one_shared_graph_cannot_pick_a_producer(tmp_path):
    rgb = _texture(11)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", rgb)])
    _case_regeneration(case, "REGEN-001", "WF-001", "job-a", rgb)
    location = case / "manifest.json"
    manifest = json.loads(location.read_bytes())
    workflows = manifest["findings"]["workflows"]
    identity = {"prompt_format": "API_GRAPH", "completeness": "COMPLETE", "canonical_json_sha256": "cd" * 32}
    workflows[0].update(identity)
    location.write_text(json.dumps(manifest))
    _bind_regenerations(case)
    single = link.link(case)
    assert single["promotions"]

    workflows.append({**identity, "workflow_id": "WF-002", "execution_parameters": {"prompt_id": "job-b"}})
    location.write_text(json.dumps(manifest))
    _bind_regenerations(case)
    result = link.link(case)
    [memory] = result["memory_images"]
    assert memory["agreeing_jobs"] == ["job-a", "job-b"]
    assert memory["attribution"] == memory["output_attribution"] == "AMBIGUOUS"
    assert [r["prompt_id"] for r in memory["matching_rasters"]] == ["job-a"]
    assert result["promotions"] == []

    single["manifest_sha256"] = result["manifest_sha256"]
    record = json.loads((case / "regenerations/REGEN-001.json").read_bytes())
    verified, notes = verify_image_link_promotions(single, manifest, [record], result["manifest_sha256"], case)
    assert verified["promotions"] == [] and "same recovered API graph" in notes[0]["note"]

    _case_regeneration(case, "REGEN-002", "WF-002", "job-b", rgb)
    result = link.link(case)
    assert result["memory_images"][0]["attribution"] == "AMBIGUOUS"
    assert result["promotions"] == []


def test_partial_edit_pixels_that_may_come_from_the_input_are_left_ambiguous(tmp_path):
    edited = _texture(11)
    case = _case(tmp_path, [("BITMAP-001", "DECODED_BITMAP", edited[:100]),
                            ("BITMAP-002", "DECODED_BITMAP", edited[200:320])])
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", edited)
    path = case / "regenerations" / "REGEN-001.json"
    record = json.loads(path.read_bytes())
    input_hashes = {"source.png": "ab" * 32}
    record["preflight_check"]["input_images"] = input_hashes
    record["evaluation"].update(target_artifact_id="BITMAP-002",
                                input_comparison={"status": "DISTINCT", "verified_inputs": input_hashes})
    path.write_text(json.dumps(record))

    result = link.link(case)
    unchanged, edited_region = result["memory_images"]

    assert unchanged["attribution"] == "AMBIGUOUS"
    assert unchanged["matching_rasters"][0]["metrics"]["input_ambiguity"]
    assert unchanged["output_attribution"] == "AMBIGUOUS"
    assert edited_region["attribution"] == "SUPPORTED"

    # The report checks the input comparison even when a links file contains a claimed promotion.
    result["promotions"] = [
        {"tier": "E3", "artifact_ids": [candidate["artifact_id"]], "prompt_id": "job-a",
         "selected_raster": edited_region["selected_raster"]}
        for candidate in (unchanged, edited_region)
    ]
    manifest = json.loads((case / "manifest.json").read_bytes())
    verified, notes = verify_image_link_promotions(result, manifest, [record], result["manifest_sha256"], case)
    assert [entry["artifact_ids"] for entry in verified["promotions"]] == [["BITMAP-002"]]
    assert [entry["promotion"] for entry in notes] == ["BITMAP-001"]


def _latent_decode(case, decode_id, rgb, status="COMPLETED", latent_sha=None):
    if not (case / latents.LATENTS_NAME).exists():
        raw = b"latent bytes"
        (case / "exports/raw/LATENT-001.f32").write_bytes(raw)
        (case / latents.LATENTS_NAME).write_text(json.dumps({"memory_sha256": MEMORY_SHA, "latents": [
            {"latent_id": "LATENT-001", "raw_file_path": "exports/raw/LATENT-001.f32",
             "sha256": hashlib.sha256(raw).hexdigest()}]}))
    own = json.loads((case / latents.LATENTS_NAME).read_text())["latents"][0]["sha256"]
    (case / "latent-decodes").mkdir(exist_ok=True)
    path = f"exports/derived/DERIVED_latent_{decode_id}.png"
    Image.fromarray(rgb).save(case / path)
    digest = hashlib.sha256((case / path).read_bytes()).hexdigest()
    (case / "latent-decodes" / f"{decode_id}.json").write_text(json.dumps({
        "decode_id": decode_id, "latent_id": "LATENT-001", "latent_sha256": latent_sha or own, "status": status,
        "derived_image": path,
        "derived_image_sha256": digest,
        "checkpoint": {"name": "m.safetensors", "model_match_level": "HASHED"}}))


def test_decoded_latent_gives_a_job_its_own_status_without_counting_as_observed(tmp_path):
    job_a, job_b, job_c = _texture(21), _texture(22), _texture(23)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a)])
    for i, (pid, rgb) in enumerate([("job-a", job_a), ("job-b", job_b), ("job-c", job_c)], 1):
        _case_regeneration(case, f"REGEN-00{i}", f"WFPY-00{i}", pid, rgb)
    near_b = np.clip(job_b.astype(int) + np.random.default_rng(1).integers(-6, 7, job_b.shape), 0, 255)
    _latent_decode(case, "LATDEC-001", near_b.astype(np.uint8))
    _latent_decode(case, "LATDEC-002", job_c, status="FAILED")
    result = link.link(case)
    (decoded,) = result["decoded_latents"]
    assert decoded["attribution"] == "SUPPORTED" and decoded["prompt_id"] == "job-b"
    assert decoded["psnr_db_by_job"]["job-b"] > 30 > link.LATENT_AGREE_DB > decoded["psnr_db_by_job"]["job-a"]
    status = {j["prompt_id"]: j["image_status"] for j in result["jobs"]}
    assert status == {"job-a": "OBSERVED_EXACT", "job-b": "DECODED_FROM_MEMORY_LATENT", "job-c": "REGENERATED_ONLY"}
    assert result["regenerated"][1]["residue_check"] == "UNCHECKED", "a decoded latent is not a memory image"


def test_a_complete_copy_of_part_of_the_frame_is_not_exact(tmp_path):
    job_a = _texture(31)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a[:256])])
    result = link.link(case, _regen(tmp_path, [("job-a", job_a), ("job-b", _texture(32))]))
    (only,) = result["memory_images"]
    assert only["prompt_id"] == "job-a" and only["whole_frame"] is False
    assert {j["prompt_id"]: j["image_status"] for j in result["jobs"]}["job-a"] == "OBSERVED_PARTIAL_OR_SCALED"


def test_agreement_is_counted_per_job_not_per_regeneration(tmp_path):
    job_a, job_b = _texture(41), _texture(42)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a)])
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", job_a)
    _case_regeneration(case, "REGEN-002", "WFPY-002", "job-a", _texture(43))
    _case_regeneration(case, "REGEN-003", "WFPY-003", "job-b", job_b)
    near = np.clip(job_a.astype(int) + np.random.default_rng(2).integers(-6, 7, job_a.shape), 0, 255).astype(np.uint8)
    _latent_decode(case, "LATDEC-001", near)
    result = link.link(case)
    (memory,) = result["memory_images"]
    assert memory["attribution"] == "SUPPORTED" and memory["agreeing_jobs"] == ["job-a"]
    (decoded,) = result["decoded_latents"]
    # The second, unrelated regeneration of job-a must not hide the first one's agreement.
    assert decoded["attribution"] == "SUPPORTED" and decoded["prompt_id"] == "job-a"


def test_regenerations_and_decodes_of_another_case_are_refused(tmp_path):
    job_a = _texture(51)
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a)])
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", job_a)
    record_path = case / "regenerations" / "REGEN-001.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, "target_manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another manifest"):
        link.link(case)
    _bind_regenerations(case)
    _latent_decode(case, "LATDEC-001", job_a, latent_sha="1" * 64)
    with pytest.raises(ValueError, match="not this case's"):
        link.link(case)


def _photo(seed: int) -> np.ndarray:
    """Smooth content with fine detail, as a generated picture has: what a lossy preview keeps closely."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:512, 0:512] / 512
    a, b, c = rng.uniform(2, 9, 3)
    base = np.stack([np.sin(a * x + b * y), np.cos(b * x - c * y), np.sin(c * (x + y))], -1) * 100 + 128
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)


def test_a_lossy_preview_of_a_job_links_to_it_but_never_as_exact(tmp_path):
    import io

    job_a, job_b = _photo(1), _photo(2)
    buffer = io.BytesIO()
    Image.fromarray(job_a).save(buffer, "JPEG", quality=90)
    preview = np.asarray(Image.open(buffer).convert("RGB"))[20:500]
    case = _case(tmp_path, [("BITMAP-001", "DECODED_BITMAP", preview), ("PIXBUF-001", "PIXEL_BUFFER", preview)])
    result = link.link(case, _regen(tmp_path, [("job-a", job_a), ("job-b", job_b)]))
    by_id = {c["artifact_id"]: c for c in result["memory_images"]}
    for artifact in ("BITMAP-001", "PIXBUF-001"):
        assert by_id[artifact]["prompt_id"] == "job-a" and by_id[artifact]["agreement"] == "LOSSY"
        assert by_id[artifact]["best"]["identical_pixel_fraction"] < 0.5
    jobs = {j["prompt_id"]: j for j in result["jobs"]}
    assert jobs["job-a"]["image_status"] == "OBSERVED_PARTIAL_OR_SCALED", "a lossy copy is not exact"
    assert jobs["job-b"]["image_status"] == "REGENERATED_ONLY"


def test_an_exact_whole_frame_copy_gives_an_e2_promotion_with_the_jobs_complete_prompt(tmp_path):
    import io

    job_a, job_b = _photo(1), _photo(2)
    buffer = io.BytesIO()
    Image.fromarray(job_b).save(buffer, "JPEG", quality=90)
    lossy_b = np.asarray(Image.open(buffer).convert("RGB"))
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", job_a), ("PIXBUF-002", "PIXEL_BUFFER", lossy_b)])
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["findings"]["workflows"] = [
        {"workflow_id": wid, "prompt_format": fmt, "completeness": done, "execution_parameters": {"prompt_id": pid}}
        for wid, fmt, done, pid in (("WFPY-001", "API_GRAPH", "COMPLETE", "job-a"),
                                    ("WFPY-002", "UI_WORKFLOW", "COMPLETE", "job-a"),
                                    ("WF-003", "API_GRAPH", "PARTIAL", "job-a"),
                                    ("WFPY-004", "API_GRAPH", "COMPLETE", "job-b"))]
    (case / "manifest.json").write_text(json.dumps(manifest))
    _case_regeneration(case, "REGEN-001", "WFPY-001", "job-a", job_a)
    _case_regeneration(case, "REGEN-002", "WFPY-004", "job-b", job_b)
    result = link.link(case)
    assert result["manifest_sha256"] == hashlib.sha256((case / "manifest.json").read_bytes()).hexdigest()
    promotion, close = result["promotions"]
    assert (promotion["artifact_id"], promotion["workflow_ids"]) == ("PIXBUF-001", ["WFPY-001"])
    assert promotion["tier"] == "E2"
    # job-b's copy is lossy: a confirmed close match earns E3, never E2
    assert (close["tier"], close["basis"]) == ("E3", "REGENERATION_CLOSE_MATCH")
    assert (close["artifact_ids"], close["workflow_ids"]) == (["PIXBUF-002"], ["WFPY-004"])
    assert close["agreement"]["PIXBUF-002"]["agreement"] == "LOSSY"


def test_a_bitmap_with_a_gap_is_rebuilt_from_its_genuine_rows_and_still_links(tmp_path):
    job = _photo(5)
    damaged = job.copy()
    damaged[200:203] = 0  # the gap's rows: no genuine pixels, zeros in memory
    case = _case(tmp_path, [("BITMAP-001", "DECODED_BITMAP", damaged)])
    genuine = np.r_[0:200, 203:512]
    raw = job[genuine].tobytes()
    manifest = json.loads((case / "manifest.json").read_text())
    [art] = manifest["findings"]["artifacts"]
    (case / art["raw_file_paths"][0]).write_bytes(raw)
    art["gaps"] = [{"relative_offset": 200 * 512 * 3, "length_bytes": 3 * 512 * 3, "reason": "STALE_MAPPING",
                    "synthetic_padding_applied": False}]
    (case / "manifest.json").write_text(json.dumps(manifest))
    (case / "recovery-evidence.json").write_text(json.dumps(
        {"records": [{"record_id": "BITMAP-001", "sha256": {"raw": hashlib.sha256(raw).hexdigest()}}]}))
    [memory] = link.memory_images(case)
    assert memory["gap_rows"] == [200, 201, 202] and np.array_equal(memory["rgb"][genuine], job[genuine])
    result = link.link(case, _regen(tmp_path, [("job", job), ("other", _photo(6))]))
    [image] = result["memory_images"]
    assert image["prompt_id"] == "job" and image["best"]["rows_compared"] == 509, "gap rows are not compared"


def test_one_genuine_row_cannot_support_attribution():
    rgb = _texture(61)[:32, :8]
    assert link.compare(rgb, rgb, range(1, 32)) is None


def test_only_verified_case_regeneration_can_promote(tmp_path):
    rgb = _texture(81)[:32, :8]
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", rgb)])
    _case_regeneration(case, "REGEN-001", "WF-001", "job", rgb)
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["findings"]["workflows"][0].update(prompt_format="API_GRAPH", completeness="COMPLETE")
    manifest_path.write_text(json.dumps(manifest))
    _bind_regenerations(case)
    assert len(link.link(case)["promotions"]) == 1
    assert link.link(case, _regen(tmp_path, [("job", rgb)]))["promotions"] == []


def test_every_ambiguous_candidate_needs_a_verified_replay(tmp_path):
    rgb = _texture(82)[:32, :8]
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", rgb)])
    _case_regeneration(case, "REGEN-001", "WF-001", "job-a", rgb)
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    findings = manifest["findings"]
    findings["artifacts"][0].update(link_status="AMBIGUOUS", ambiguous_candidate_workflows=["WF-001", "WF-002"])
    findings["workflows"][0].update(prompt_format="API_GRAPH", completeness="COMPLETE")
    manifest_path.write_text(json.dumps(manifest))
    _bind_regenerations(case)
    result = link.link(case)
    assert result["promotions"] == []
    assert result["memory_images"][0]["unreplayed_workflow_ids"] == ["WF-002"]
    _case_regeneration(case, "REGEN-002", "WF-002", "job-b", 255 - rgb)
    assert len(link.link(case)["promotions"]) == 1


def test_close_matches_keep_unreplayed_jobs_unresolved_even_without_a_prior_tie(tmp_path, monkeypatch):
    rgb = _texture(83)[:32, :32]
    case = _case(tmp_path, [("BITMAP-001", "DECODED_BITMAP", rgb)])
    _case_regeneration(case, "REGEN-001", "WF-001", "job-a", rgb)
    path = case / "manifest.json"
    manifest = json.loads(path.read_bytes())
    workflows = manifest["findings"]["workflows"]
    workflows[0].update(prompt_format="API_GRAPH", completeness="COMPLETE")
    workflows.append({"workflow_id": "WF-002", "prompt_format": "API_GRAPH", "completeness": "COMPLETE",
                      "execution_parameters": {"prompt_id": "job-b"}})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _bind_regenerations(case)
    monkeypatch.setattr(link, "latent_decodes", lambda _: [
        {"decode_id": "LATDEC-001", "latent_id": "LAT-001", "file": "derived.png", "rgb": rgb},
    ])

    result = link.link(case)
    for image in (result["memory_images"][0], result["decoded_latents"][0]):
        assert image["attribution"] == image["output_attribution"] == "AMBIGUOUS"
        assert image["prompt_id"] is None and image["unreplayed_workflow_ids"] == ["WF-002"]
    assert result["promotions"] == []

    _case_regeneration(case, "REGEN-002", "WF-002", "job-b", 255 - rgb)
    result = link.link(case)
    assert result["memory_images"][0]["attribution"] == result["decoded_latents"][0]["attribution"] == "SUPPORTED"
    assert [promotion["tier"] for promotion in result["promotions"]] == ["E3"]
    only_first = json.loads((case / "regenerations/REGEN-001.json").read_bytes())
    verified, notes = verify_image_link_promotions(
        result, manifest, [only_first], result["manifest_sha256"], case,
    )
    assert verified["promotions"] == [] and notes


def test_promotion_cites_only_matching_raster(tmp_path):
    rgb = _texture(62)[:32, :8]
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", rgb)])
    for i, raster in enumerate([255 - rgb, rgb], 1):
        _case_regeneration(case, f"REGEN-00{i}", f"WF-00{i}", "job", raster)
    manifest = json.loads((case / "manifest.json").read_text())
    for workflow in manifest["findings"]["workflows"]:
        workflow.update(prompt_format="API_GRAPH", completeness="COMPLETE")
    (case / "manifest.json").write_text(json.dumps(manifest))
    _bind_regenerations(case)
    result = link.link(case)
    assert [p["regenerated"] for p in result["promotions"]] == [result["regenerated"][1]["file"]]
    assert [r["residue_check"] for r in result["regenerated"]] == ["UNCHECKED", "MATCHED"]
    assert result["promotions"][0]["selected_raster"]["handoff_id"] == "REGEN-002"


def test_gap_fill_cannot_choose_the_alignment():
    rgb = np.random.default_rng(63).integers(80, 200, (32, 8, 3), dtype=np.uint8)
    memory = rgb.copy()
    memory[24:] = 0
    decoy = memory.copy()
    decoy[10:24] += 8
    metrics = link.compare(np.concatenate([decoy, rgb]), memory, range(24, 32))
    assert metrics["row_offset"] == 32 and metrics["rows_compared"] == 24
    assert metrics["rows_agreeing"] == 24


def test_grouped_close_matches_preserve_each_artifacts_raster(tmp_path):
    a, b = (_texture(i)[:32, :8] for i in (71, 72))
    case = _case(tmp_path, [("BITMAP-1", "DECODED_BITMAP", a), ("BITMAP-2", "DECODED_BITMAP", b)])
    for i, rgb in enumerate((a, b), 1):
        _case_regeneration(case, f"REGEN-00{i}", f"WF-00{i}", "batch-job", rgb)
    manifest = json.loads((case / "manifest.json").read_text())
    for row in manifest["findings"]["workflows"]:
        row.update(prompt_format="API_GRAPH", completeness="COMPLETE")
    (case / "manifest.json").write_text(json.dumps(manifest))
    _bind_regenerations(case)
    [promotion] = link.link(case)["promotions"]
    assert promotion["tier"] == "E3" and promotion["artifact_ids"] == ["BITMAP-1", "BITMAP-2"]
    assert promotion["selected_raster"] is None and promotion["regenerated"] is None
    for i, artifact in enumerate(promotion["artifact_ids"], 1):
        evidence = promotion["raster_evidence_by_artifact"][artifact]
        assert evidence["selected_raster"]["handoff_id"] == f"REGEN-00{i}"
        assert evidence["matching_rasters"][0]["metrics"] == promotion["agreement"][artifact]


def test_identical_flat_frame_prefers_full_overlap():
    rgb = np.zeros((32, 8, 3), dtype=np.uint8)
    metrics = link.compare(rgb, rgb)
    assert metrics["row_offset"] == 0 and metrics["rows_compared"] == 32
