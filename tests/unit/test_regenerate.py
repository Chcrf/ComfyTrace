import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from comfytrace import localrunner, regenerate
from comfytrace.carve import PngCarver
from comfytrace.cli import parse_args, run_extract, run_regenerate, run_report
from comfytrace.exceptions import MissingDependencyError
from comfytrace.regenerate import TieredEvaluator
from comfytrace.report import sha256_file
from tests.synthetic import DumpBuilder, api_prompt, noise_image, png_bytes


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_loads_a_recovered_serialized_queue_item_for_regeneration(tmp_path):
    graph = api_prompt()
    prompt_id = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"
    exported = tmp_path / "WF-001.json"
    exported.write_text(json.dumps(json.dumps([5, prompt_id, graph, {}, ["9"]])), encoding="utf-8")

    assert regenerate.load_workflow_file(exported) == ("API_GRAPH", graph, {"prompt_id": prompt_id})


@pytest.fixture
def t1_case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir) -> Path:
    case = tmp_path / "case01"
    args = parse_args(["extract", "--memory", str(mock_memory_image("T1")), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case)])
    assert run_extract(args, ingestor=fake_ingestor) == 0
    return case


@pytest.fixture
def e5_case(tmp_path, symbols_dir) -> Path:
    dump = DumpBuilder(frames=64)
    dump.put(20, json.dumps(api_prompt(seed=4242, text="unlinked workflow")).encode())
    memory = dump.write(tmp_path / "e5.raw")
    case = tmp_path / "case_e5"
    args = parse_args(["extract", "--memory", str(memory), "--symbols-dir", str(symbols_dir), "--outdir", str(case)])
    assert run_extract(args) == 3
    return case


def _regenerate(case: Path, url: str | None, *extra: str, workflow: str = "WF-001", compare: bool = True) -> int:
    argv = ["regenerate", "--case-dir", str(case), "--workflow", str(case / "exports/raw" / f"{workflow}.json"),
            *extra]
    if url is not None:
        argv += ["--runner-url", url]
    if compare:
        argv += ["--compare-to", str(case / "exports/raw/CARVE-PNG-001.png")]
    return run_regenerate(parse_args(argv))


def _record(case: Path, handoff: str = "REGEN-001") -> dict:
    return json.loads((case / "regenerations" / f"{handoff}.json").read_text())


def test_preflight_security_rejected_unvetted_custom_node(t1_case, mock_runner_server, tmp_path):
    url, state = mock_runner_server
    graph = api_prompt()
    graph["10"] = {"class_type": "SuspiciousPythonExec", "inputs": {"code": "import os"}}
    workflow = tmp_path / "custom.json"
    workflow.write_text(json.dumps(graph))
    manifest = json.loads((t1_case / "manifest.json").read_text())
    with pytest.raises(regenerate.RegenerationError):
        regenerate.run_regeneration(parse_args(["regenerate", "--case-dir", str(t1_case), "--workflow", str(workflow),
                                                "--runner-url", url]))
    assert manifest["findings"]["workflows"][0]["canonical_json_sha256"]

    wf = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    wf.graph, wf.prompt_format = graph, "API_GRAPH"
    passed, status, audit = regenerate.PreflightChecker(url, [], False, None).validate(wf)
    assert (passed, status) == (False, "SECURITY_REJECTED")
    assert audit["preflight_check"]["unvetted_node_classes"] == ["SuspiciousPythonExec"]
    state.object_info["SuspiciousPythonExec"] = {"input": {}}
    passed, status, audit = regenerate.PreflightChecker(url, ["SuspiciousPythonExec"], False, None).validate(wf)
    assert (passed, status) == (True, "PASSED")
    assert audit["security_audit"]["allowed_custom_nodes"] == ["SuspiciousPythonExec"]
    assert state.submitted == [], "preflight never submits"


def test_preflight_recognizes_builtin_sdxl_and_image_transform_nodes(t1_case, mock_runner_server):
    url, state = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    workflow = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    graph = api_prompt()
    graph["6"]["class_type"] = "CLIPTextEncodeSDXL"
    graph["10"] = {"class_type": "ImageCrop", "inputs": {"image": ["8", 0], "width": 64, "height": 48}}
    graph["9"]["inputs"]["images"] = ["10", 0]
    workflow.graph = graph
    state.object_info.update({"CLIPTextEncodeSDXL": {"input": {}}, "ImageCrop": {"input": {}}})

    passed, status, audit = regenerate.PreflightChecker(url, [], False, None).validate(workflow)

    assert (passed, status) == (True, "PASSED")
    assert audit["preflight_check"]["compared_output_nodes"] == ["9"]
    assert state.submitted == []


@pytest.mark.parametrize(("blend_class", "first", "second"), [
    ("ImageCompositeMasked", "source", "destination"),
    ("ImageBlend", "image1", "image2"),
])
def test_mixed_source_composite_is_compared_as_a_reference_without_admitting_input_previews(
    t1_case, mock_runner_server, tmp_path, blend_class, first, second,
):
    url, state = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    workflow = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "source.png"}}
    graph["11"] = {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}}
    graph["12"] = {"class_type": blend_class, "inputs": {first: ["8", 0], second: ["10", 0]}}
    graph["13"] = {"class_type": "ImageBatch", "inputs": {"image1": ["8", 0], "image2": ["10", 0]}}
    graph["14"] = {"class_type": "PreviewImage", "inputs": {"images": ["13", 0]}}
    graph["15"] = {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
    graph["9"]["inputs"]["images"] = ["12", 0]
    workflow.graph = graph
    for name in ("LoadImage", "PreviewImage", blend_class, "ImageBatch"):
        state.object_info[name] = {"input": {}}
    (tmp_path / "source.png").write_bytes(_png(noise_image(64, 48, seed=8)))

    passed, status, audit = regenerate.PreflightChecker(url, [], False, None, tmp_path).validate(workflow)

    assert (passed, status) == (True, "PASSED")
    assert audit["preflight_check"]["compared_output_nodes"] == ["15", "9"]
    assert audit["preflight_check"]["comparison_only_output_nodes"] == ["9"]
    assert state.submitted == []


def test_modular_sampler_compares_both_outputs_only_with_a_real_schedule(t1_case, mock_runner_server):
    url, state = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    workflow = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    graph = api_prompt()
    graph["3"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0],
        "latent_image": ["5", 0],
    }}
    graph["10"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 43}}
    graph["11"] = {"class_type": "CFGGuider", "inputs": {
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "cfg": 7.0,
    }}
    graph["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    graph["13"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["4", 0], "scheduler": "normal", "steps": 12, "denoise": 0.8,
    }}
    graph["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 1], "vae": ["4", 2]}}
    graph["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "denoised"}}
    workflow.graph = graph
    for name in ("SamplerCustomAdvanced", "RandomNoise", "CFGGuider", "KSamplerSelect", "BasicScheduler"):
        state.object_info[name] = {"input": {}}
    checker = regenerate.PreflightChecker(url, [], False, None)

    passed, status, audit = checker.validate(workflow)

    assert (passed, status) == (True, "PASSED")
    assert audit["preflight_check"]["compared_output_nodes"] == ["15", "9"]
    assert audit["preflight_check"]["comparison_only_output_nodes"] == []
    graph["13"]["inputs"]["denoise"] = 0.0
    passed, status, _ = checker.validate(workflow)
    assert (passed, status) == (False, "SECURITY_REJECTED")
    assert state.submitted == []


def test_preflight_security_rejected_pickle_checkpoint(t1_case, mock_runner_server):
    url, _ = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    wf = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    wf.graph, wf.prompt_format = api_prompt(ckpt="legacy.ckpt"), "API_GRAPH"
    wf.dependencies = {"node_classes": [], "model_assets": ["legacy.ckpt"]}
    passed, status, audit = regenerate.PreflightChecker(url, [], False, None).validate(wf)
    assert (passed, status) == (False, "SECURITY_REJECTED")
    passed, status, audit = regenerate.PreflightChecker(url, [], True, None).validate(wf)
    assert (passed, status) == (True, "PASSED")
    assert audit["security_audit"]["model_format"] == "PICKLE"
    assert "forensic_warning" in audit["preflight_check"]


def test_preflight_missing_dependency_and_model_hashes(t1_case, mock_runner_server, tmp_path):
    url, state = mock_runner_server
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "v1-5-pruned-emaonly.safetensors").write_bytes(b"weights")
    del state.object_info["VAEDecode"]
    assert _regenerate(t1_case, url, "--models-dir", str(tmp_path / "models")) == 1
    record = _record(t1_case)
    assert record["status"] == "MISSING_DEPENDENCY"
    assert record["preflight_check"]["missing_node_classes"] == ["VAEDecode"]
    assert record["preflight_check"]["runner_object_info_verified"] is True

    state.object_info["VAEDecode"] = {"input": {}}
    assert _regenerate(t1_case, url, "--models-dir", str(tmp_path / "models")) == 0
    check = _record(t1_case, "REGEN-002")["preflight_check"]
    assert check["model_match_level"] == "HASHED"
    assert check["model_sha256"] == {"v1-5-pruned-emaonly.safetensors": hashlib.sha256(b"weights").hexdigest()}


def test_unreachable_runner_is_operational_failure(t1_case):
    assert _regenerate(t1_case, "http://127.0.0.1:9") == 1


@pytest.mark.parametrize("explicit", [True, False])
def test_tiered_evaluator_exact_pixel_match(t1_case, mock_runner_server, explicit, monkeypatch):
    url, state = mock_runner_server
    if not explicit:
        monkeypatch.setattr(localrunner.shutil, "which", lambda *_args, **_kwargs: "comfy")
        monkeypatch.setattr(localrunner, "_run_comfy", lambda *_args, **_kwargs: {
            "server": {"running": True, "url": url}, "workspace": {"exists": True},
        })
    manifest_sha = sha256_file(t1_case / "manifest.json")
    assert _regenerate(t1_case, url if explicit else None) == 0
    record = _record(t1_case)
    assert record["target_manifest_sha256"] == manifest_sha
    assert record["status"] == "COMPLETED"
    evaluation = record["evaluation"]
    assert (evaluation["evaluated_tier"], evaluation["evaluation_subtier"]) == ("E1", "EXACT_MATCH")
    assert evaluation["exact_pixel_match"] is True and evaluation["promoted_finding_tier"] == "E1"
    assert record["derived_regenerated_image"] == "exports/derived/DERIVED_regen_WF-001.png"
    assert record["runner_environment"]["comfyui_version"] == "0.35.0"
    assert record["runner_environment"]["cuda_driver_version"] == "UNREPORTED"
    assert state.submitted[0]["prompt"] == api_prompt()
    assert sha256_file(t1_case / "manifest.json") == manifest_sha
    assert run_report(parse_args(["report", "--case-dir", str(t1_case)])) == 0
    assert not (t1_case / "report.html").exists()
    wb = (t1_case / "workbench.html").read_text(encoding="utf-8")
    assert '"id":"FIND-001","tier":"E1"' in wb


def test_tiered_evaluator_perceptual_ssim_phash_match(t1_case, mock_runner_server):
    url, state = mock_runner_server
    img = noise_image(64, 48, seed=5)
    px = img.load()
    for x in range(0, 64, 9):
        r, g, b = px[x, 10]
        px[x, 10] = ((r + 3) % 256, g, b)
    state.image = _png(img)
    assert _regenerate(t1_case, url) == 0
    evaluation = _record(t1_case)["evaluation"]
    assert (evaluation["evaluated_tier"], evaluation["evaluation_subtier"]) == ("E1", "PERCEPTUAL_MATCH")
    assert evaluation["exact_pixel_match"] is False
    assert evaluation["ssim_score"] >= 0.95 and evaluation["phash_hamming_distance"] <= 4


def test_non_matching_regeneration_keeps_e2(t1_case, mock_runner_server):
    url, state = mock_runner_server
    state.image = _png(noise_image(64, 48, seed=99).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    assert _regenerate(t1_case, url) == 2
    record = _record(t1_case)
    evaluation = record["evaluation"]
    assert evaluation["evaluated_tier"] == "NON_MATCHING"
    assert evaluation["promoted_finding_tier"] is None and evaluation["target_finding_tier_retained"] == "E2"
    assert evaluation["ssim_score"] is not None, "scores are recorded even below threshold"
    assert record["derived_regenerated_image"] == "exports/derived/DERIVED_regen_NON_MATCHING_WF-001.png"


def test_resized_regeneration_fails_without_silent_resize(t1_case, mock_runner_server):
    url, state = mock_runner_server
    state.image = _png(noise_image(64, 48, seed=5).resize((128, 96)))
    assert _regenerate(t1_case, url) == 2
    assert _record(t1_case)["evaluation"]["evaluation_subtier"] == "DIMENSION_MISMATCH"


def test_tiered_evaluator_e3_partial_row_slice_corroboration():
    full = noise_image(64, 64, "RGB", seed=6)
    png = png_bytes(full, idat_chunk=400)
    [partial] = PngCarver().carve_stream(png[: len(png) // 2])
    rows = partial.derived_preview_metadata["decoded_rows"]
    tier, subtier, metrics = TieredEvaluator().evaluate_reproduction(_png(full), partial, partial.preview_png)
    assert (tier, subtier) == ("PARTIAL_CORROBORATION", "ROW_SLICE_EXACT")
    assert metrics["evaluation_mode"] == "PARTIAL_ROW_BOUNDED"
    assert metrics["cropped_dimensions"] == {"width": 64, "height": rows}
    assert metrics["row_slice_pixel_match"] is True

    other = noise_image(64, 64, "RGB", seed=7)
    tier, _, metrics = TieredEvaluator().evaluate_reproduction(_png(other), partial, partial.preview_png)
    assert tier == "NON_MATCHING" and metrics["row_slice_pixel_match"] is False


@pytest.mark.parametrize(("edit_top", "status"), [(False, "MATCHES_INPUT"), (True, "DISTINCT")])
def test_partial_replay_rows_must_distinguish_an_edited_image_from_its_input(tmp_path, edit_top, status):
    source = noise_image(64, 64, "RGB", seed=6)
    path = tmp_path / "source.png"
    path.write_bytes(_png(source))
    expected = {path.name: sha256_file(path)}
    edited = source.copy()
    edited.paste((235, 30, 160), (0, 0 if edit_top else 32, 48, 32 if edit_top else 64))
    encoded = png_bytes(edited, idat_chunk=200)
    [partial] = PngCarver().carve_stream(encoded[:len(encoded) // 3])
    rows = partial.derived_preview_metadata["decoded_rows"]
    assert 0 < rows < 64
    if not edit_top:
        assert rows <= 32
    evaluator = TieredEvaluator()

    check = evaluator.compare_partial_inputs(partial, partial.preview_png, tmp_path, expected, exact_only=True)

    assert check["status"] == status
    assert check["verified_inputs"] == expected


def test_unlinked_e5_workflow_reproduction_is_e4(e5_case, mock_runner_server):
    url, _ = mock_runner_server
    manifest = json.loads((e5_case / "manifest.json").read_text())
    assert [f["tier"] for f in manifest["findings"]["findings"]] == ["E5"]
    assert _regenerate(e5_case, url, compare=False) == 0
    evaluation = _record(e5_case)["evaluation"]
    assert (evaluation["evaluated_tier"], evaluation["promoted_finding_tier"]) == ("E4", "E4")
    assert "not that an image existed in memory" in evaluation["note"]
    assert run_report(parse_args(["report", "--case-dir", str(e5_case)])) == 0
    assert not (e5_case / "report.html").exists()
    wb = (e5_case / "workbench.html").read_text(encoding="utf-8")
    assert '"id":"FIND-001","tier":"E4"' in wb


def test_ui_workflow_is_unavailable(t1_case, mock_runner_server):
    url, _ = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    wf = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    wf.prompt_format = "UI_WORKFLOW"
    passed, status, audit = regenerate.PreflightChecker(url, [], False, None).validate(wf)
    assert (passed, status) == (False, "UNAVAILABLE")
    assert "UI-to-API compiler" in audit["preflight_check"]["reason"]


def test_missing_regen_extras_only_block_perceptual_comparison(monkeypatch):
    def unavailable():
        raise MissingDependencyError("needs comfytrace[regen]")

    monkeypatch.setattr(regenerate, "_perceptual_backends", unavailable)
    img = _png(noise_image(16, 16))
    [record] = PngCarver().carve_stream(img)
    tier, subtier, metrics = TieredEvaluator().evaluate_reproduction(img, record, img)
    assert (tier, subtier) == ("E1", "EXACT_MATCH"), "exact matching needs only core dependencies"
    assert metrics["perceptual_scores"].startswith("UNAVAILABLE")
    with pytest.raises(MissingDependencyError):
        TieredEvaluator().evaluate_reproduction(_png(noise_image(16, 16, seed=2)), record, img)


def test_every_regeneration_record_embeds_manifest_hash(t1_case, mock_runner_server):
    url, state = mock_runner_server
    _regenerate(t1_case, url)
    _regenerate(t1_case, url)
    state.image = _png(noise_image(64, 48, seed=1))
    _regenerate(t1_case, url)
    manifest_sha = sha256_file(t1_case / "manifest.json")
    records = sorted((t1_case / "regenerations").glob("REGEN-*.json"))
    assert [p.stem for p in records] == ["REGEN-001", "REGEN-002", "REGEN-003"]
    assert all(json.loads(p.read_text())["target_manifest_sha256"] == manifest_sha for p in records)


def test_preflight_rejects_input_names_escaping_inputs_dir(t1_case, mock_runner_server, tmp_path):
    url, state = mock_runner_server
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (tmp_path / "secret.png").write_bytes(b"not for the runner")
    manifest = json.loads((t1_case / "manifest.json").read_text())
    wf = regenerate.WorkflowRecord.from_dict(manifest["findings"]["workflows"][0])
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "../secret.png"}}
    wf.graph, wf.prompt_format = graph, "API_GRAPH"
    state.object_info["LoadImage"] = {"input": {}}
    passed, status, audit = regenerate.PreflightChecker(url, [], False, None, inputs).validate(wf)
    assert (passed, status) == (False, "SECURITY_REJECTED")
    assert "escape" in audit["preflight_check"]["reason"]
    graph["10"]["inputs"]["image"] = "source.png"
    passed, status, _ = regenerate.PreflightChecker(url, [], False, None, inputs).validate(wf)
    assert (passed, status) == (False, "UNAVAILABLE")
    (inputs / "source.png").write_bytes(b"png")
    passed, status, audit = regenerate.PreflightChecker(url, [], False, None, inputs).validate(wf)
    assert (passed, status) == (True, "PASSED")


def _models(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "v1-5-pruned-emaonly.safetensors").write_bytes(b"weights")
    return models, hashlib.sha256(b"weights").hexdigest()


def _identity(case, verdict, sha256, memory_sha256=None):
    manifest = json.loads((case / "manifest.json").read_text())
    document = {"memory_sha256": memory_sha256 or manifest["case_metadata"]["memory_image"]["sha256"],
                "models": [{"name": "v1-5-pruned-emaonly.safetensors", "verdict": verdict, "sha256": sha256}]}
    (case / "model-identity.json").write_text(json.dumps(document))


def test_weights_observed_in_this_dump_raise_the_model_match_level(t1_case, mock_runner_server, tmp_path):
    url, _ = mock_runner_server
    models, sha = _models(tmp_path)
    flags = ("--models-dir", str(models))
    _identity(t1_case, "OBSERVED", sha)
    assert _regenerate(t1_case, url, *flags) == 0
    check = _record(t1_case)["preflight_check"]
    assert check["model_match_level"] == "OBSERVED_IN_MEMORY"
    assert check["model_memory_identity"] == {"v1-5-pruned-emaonly.safetensors": "OBSERVED"}


def test_identity_of_another_file_dump_or_verdict_stays_hashed(t1_case, mock_runner_server, tmp_path):
    url, _ = mock_runner_server
    models, sha = _models(tmp_path)
    flags = ("--models-dir", str(models))
    for n, (verdict, digest, dump) in enumerate([("OBSERVED", "0" * 64, None), ("OBSERVED", sha, "f" * 64),
                                                 ("SHARED_COMPONENTS_ONLY", sha, None)], 1):
        _identity(t1_case, verdict, digest, dump)
        assert _regenerate(t1_case, url, *flags) == 0
        assert _record(t1_case, f"REGEN-{n:03d}")["preflight_check"]["model_match_level"] == "HASHED"


def test_batch_rasters_round_trip_and_exact_output_wins(tmp_path, mock_runner_server, monkeypatch):
    from comfytrace import link
    from comfytrace.carve import WorkflowCarver
    from comfytrace.report import EvidentiaryExporter

    t1_case = tmp_path
    url, state = mock_runner_server
    exporter = EvidentiaryExporter(t1_case)
    exact = state.image
    [target] = PngCarver().carve_stream(exact)
    exporter.export_raw_artifact(target, exact)
    [workflow] = WorkflowCarver().carve_stream(json.dumps(api_prompt()).encode())
    workflow.execution_parameters["prompt_id"] = "batch-job"
    exporter.export_workflow(workflow)
    manifest_path = t1_case / "manifest.json"
    manifest = {"findings": {"workflows": [workflow.to_dict()], "artifacts": [target.to_dict()],
                             "findings": [{"tier": "E2", "artifact_ids": [target.artifact_id, workflow.workflow_id]}]}}
    manifest_path.write_text(json.dumps(manifest))
    image = Image.open(io.BytesIO(exact)).convert("RGB")
    near = image.copy()
    near.putpixel((0, 0), tuple(255 - value for value in near.getpixel((0, 0))))
    monkeypatch.setattr(TieredEvaluator, "_perceptual", lambda *args: (0.99, 0))
    outputs = [({"node_id": "9", "batch_index": i}, data) for i, data in enumerate((_png(near), exact, exact))]
    monkeypatch.setattr(regenerate.RunnerClient, "retrieve_output_rasters", lambda *args: outputs)
    assert _regenerate(t1_case, url) == 0
    record = _record(t1_case)
    assert record["evaluation"]["evaluation_subtier"] == "EXACT_MATCH"
    assert record["evaluation"]["selected_raster"] == record["output_rasters"][1]
    recovered = link.case_regenerations(t1_case)
    assert [r["batch_index"] for r in recovered] == [0, 1, 2]
    for r, (_, data) in zip(recovered, outputs, strict=True):
        assert r["node_id"] == "9" and r["handoff_id"] == "REGEN-001"
        assert (t1_case / r["file"]).read_bytes() == data and r["sha256"] == hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(link, "memory_images", lambda case: [
        {"artifact_id": "PIXBUF-001", "kind": "PIXEL_BUFFER", "complete": True, "rgb": recovered[1]["rgb"]}])
    [memory] = link.link(t1_case)["memory_images"]
    assert memory["attribution"] == "SUPPORTED" and memory["output_attribution"] == "AMBIGUOUS"
    assert sorted(r["batch_index"] for r in memory["matching_rasters"]) == [0, 1, 2]
    assert memory["selected_raster"]["batch_index"] == 1
    (t1_case / recovered[2]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="REGEN-001"):
        link.case_regenerations(t1_case)
