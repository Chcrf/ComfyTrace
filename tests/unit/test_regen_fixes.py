import hashlib
import io
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import structural_similarity

from comfytrace import regenerate as r
from comfytrace.carve import PngCarver, WorkflowCarver
from comfytrace.cli import parse_args, run_regenerate, run_report
from comfytrace.exceptions import ExecutionTimeoutError, PromptSubmissionError, RegenerationError, RunnerConnectionError
from comfytrace.models import ForensicManifest, RegenerationRecord
from comfytrace.report import ManifestWriter, effective_findings, save_regeneration_record, sha256_file
from tests.synthetic import api_prompt, noise_image, png_bytes


def encoded(image, fmt="PNG"):
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


def _symlink(path, target, *, target_is_directory=False):
    try:
        path.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows does not grant this process permission to create symbolic links")
        raise


def workflow(graph):
    [record] = WorkflowCarver().carve_stream(json.dumps(graph).encode())
    return record


def info(graph, choices=()):
    return {node["class_type"]: {"input": {"required": {"model": [list(choices)]}}}
            for node in graph.values()}


SAVER_SCHEMA = {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}, "output_node": True}


def preflight(monkeypatch, graph, models=None, inputs=None, choices=(), allow_pickle=False, allowed=(), samplers=(),
              schemas=None):
    monkeypatch.setattr(r.RunnerClient, "object_info", lambda self: {**info(graph, choices), **(schemas or {})})
    return r.PreflightChecker("http://stub.invalid", list(allowed), allow_pickle, models, inputs,
                              samplers).validate(workflow(graph))


class Response:
    status_code = 200
    text = ""

    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content

    def json(self):
        return self.payload


@pytest.fixture
def case_factory(tmp_path, monkeypatch):
    def make(tier="E5", target=False, graph=None, copies=False):
        case = tmp_path / "case"
        graph = api_prompt() if graph is None else graph
        wf = workflow(graph)
        wf.raw_file_path = "exports/raw/WF-001.json"
        (case / "exports/raw").mkdir(parents=True)
        (case / wf.raw_file_path).write_bytes(wf.raw_bytes)
        image = png_bytes(noise_image(64, 48, seed=5))
        artifacts = []
        ids = [wf.workflow_id]
        if target:
            payload = image[:len(image) // 2] if tier == "E3" else image
            [artifact] = PngCarver().carve_stream(payload)
            artifact.raw_file_paths = ["exports/raw/CARVE-PNG-001/span_000.bin" if tier == "E3"
                                       else "exports/raw/CARVE-PNG-001.png"]
            if tier == "E3":
                artifact.derived_preview_metadata["raw_span_sha256"] = [hashlib.sha256(payload).hexdigest()]
            path = case / artifact.raw_file_paths[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            artifacts = [artifact.to_dict()]
            ids.insert(0, artifact.artifact_id)
        workflows = [wf.to_dict()]
        if copies:
            copy = deepcopy(wf)
            copy.workflow_id = "WF-002"
            copy.raw_file_path = "exports/raw/WF-002.json"
            (case / copy.raw_file_path).write_text(json.dumps(graph, indent=2))
            workflows.append(copy.to_dict())
            ids.append(copy.workflow_id)
        manifest = ForensicManifest("1.0.0", {}, [], {
            "findings": [{"finding_id": "FIND-001", "tier": tier, "artifact_ids": ids,
                          "link_ids": [], "rationale": "regression fixture"}],
            "artifacts": artifacts, "workflows": workflows, "correlation_links": [],
        })
        ManifestWriter(case).write_manifest(manifest)
        state = SimpleNamespace(image=image, status={"completed": True, "status_str": "success"}, calls=[])

        def request(self, method, path, **kwargs):
            state.calls.append((path, kwargs))
            if path == "/object_info":
                return Response(info(graph, ["v1-5-pruned-emaonly.safetensors"]))
            if path == "/system_stats":
                return Response({})
            if path == "/upload/image":
                return Response({"name": getattr(state, "upload_name", kwargs["files"]["image"][0])})
            if path == "/prompt":
                return Response({"prompt_id": "p"})
            if path == "/history/p":
                return Response({"p": {"status": state.status, "outputs": {"9": {"images": [
                    {"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
            if path == "/view":
                return Response(content=state.image)
            raise AssertionError(path)

        monkeypatch.setattr(r.RunnerClient, "_request", request)
        argv = ["regenerate", "--case-dir", str(case), "--workflow", str(case / wf.raw_file_path),
                "--runner-url", "http://stub.invalid"]
        if target:
            argv += ["--compare-to", str(case / artifacts[0]["raw_file_paths"][0])]
        return case, parse_args(argv), state
    return make


def saved(case):
    return json.loads((case / "regenerations/REGEN-001.json").read_text())


def test_f10_requested_relative_model_is_hashed_without_claiming_verification(tmp_path, monkeypatch):
    for directory, payload in [("a", b"WRONG"), ("b", b"REQUESTED")]:
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "same.safetensors").write_bytes(payload)
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "b/same.safetensors"}}}
    passed, _, audit = preflight(monkeypatch, graph, models=tmp_path, choices=["b/same.safetensors"])
    assert passed
    assert audit["preflight_check"]["model_match_level"] == "HASHED"
    assert audit["preflight_check"]["model_sha256"] == {
        "b/same.safetensors": hashlib.sha256(b"REQUESTED").hexdigest()}
    (tmp_path / "b/same.safetensors").unlink()
    passed, status, audit = preflight(monkeypatch, graph, models=tmp_path, choices=["b/same.safetensors"])
    assert not passed and status == "MISSING_DEPENDENCY"
    assert audit["preflight_check"]["model_match_level"] != "HASH_VERIFIED"


@pytest.mark.parametrize("name", ["../outside.safetensors", "link.safetensors", r"..\outside.safetensors"])
def test_f10_model_escape_is_rejected(tmp_path, monkeypatch, name):
    models = tmp_path / "models"
    models.mkdir()
    if name == "link.safetensors":
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"outside")
        _symlink(models / name, outside)
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}}}
    assert preflight(monkeypatch, graph, models=models, choices=[name])[:2] == (False, "SECURITY_REJECTED")


@pytest.mark.parametrize("name", [r"..\escape\source.png", "../source.png", "sub/source.png",
                                      "sub\\source.png", "/source.png", "C:source.png"])
def test_f11_input_separators_rejected_before_upload(tmp_path, monkeypatch, name):
    graph = {"1": {"class_type": "LoadImage", "inputs": {"image": name}}}
    assert preflight(monkeypatch, graph, inputs=tmp_path)[:2] == (False, "SECURITY_REJECTED")
    monkeypatch.setattr(r.RunnerClient, "_request", lambda *a, **kw: pytest.fail("unsafe upload sent"))
    with pytest.raises(RegenerationError):
        r.RunnerClient("stub").upload_input(tmp_path / "unused.png", name)


def test_f11_upload_validates_the_returned_name_and_subfolder(tmp_path, monkeypatch):
    local = tmp_path / "source.png"
    local.write_bytes(b"input")
    seen = []

    def request(self, method, path, **kwargs):
        seen.append((kwargs["files"]["image"][0], kwargs["data"]))
        return Response({"name": "source.png", "subfolder": "../escape"})

    monkeypatch.setattr(r.RunnerClient, "_request", request)
    with pytest.raises(RegenerationError):
        r.RunnerClient("stub").upload_input(local, "source.png")
    assert seen == [("source.png", {"overwrite": "true", "subfolder": "", "type": "input"})]


@pytest.mark.parametrize("node_class,key", [("CLIPTextEncode", "text"), ("SaveImage", "filename_prefix")])
def test_f12_embedding_pickle_policy_covers_any_text_input(tmp_path, monkeypatch, node_class, key):
    graph = {"1": {"class_type": node_class, "inputs": {key: "portrait (embedding:legacy.pt:1.2)"}}}
    if node_class == "SaveImage":
        graph["2"] = {"class_type": "KSampler", "inputs": {"steps": 20, "denoise": 1.0}}
        graph["1"]["inputs"]["images"] = ["2", 0]
    (tmp_path / "legacy.pt").write_bytes(b"pickle")
    assert preflight(monkeypatch, graph, models=tmp_path)[:2] == (False, "SECURITY_REJECTED")
    assert preflight(monkeypatch, graph, models=tmp_path, allow_pickle=True)[:2] == (True, "PASSED")


def test_f12_embedding_needs_models_dir_and_is_not_a_runner_dropdown_choice(tmp_path, monkeypatch):
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style.safetensors portrait"}}}
    passed, status, audit = preflight(monkeypatch, graph)
    assert (passed, status) == (False, "SECURITY_REJECTED") and "--models-dir" in audit["preflight_check"]["reason"]
    assert preflight(monkeypatch, graph, models=tmp_path)[:2] == (False, "SECURITY_REJECTED"), "file absent"
    (tmp_path / "style.safetensors").write_bytes(b"embedding weights")
    passed, _, audit = preflight(monkeypatch, graph, models=tmp_path)
    assert passed and audit["preflight_check"]["missing_model_assets"] == []
    assert audit["preflight_check"]["model_sha256"] == {
        "style.safetensors": hashlib.sha256(b"embedding weights").hexdigest()}


@pytest.mark.parametrize("status", [{"completed": True, "status_str": "error"},
                                    {"completed": True, "status_str": "failed"},
                                    {"completed": True, "status_str": "errored"},
                                    {"completed": True, "messages": [["execution_error", {}]]}])
def test_f13_failed_history_never_completes_or_promotes(case_factory, status):
    case, args, state = case_factory()
    state.status = status
    assert run_regenerate(args) == 1
    record = saved(case)
    assert record["status"] == "FAILED"
    assert record["evaluation"].get("promoted_finding_tier") is None
    assert "/view" not in [path for path, _ in state.calls]


@pytest.mark.parametrize("image", [
    {"filename": "in.png", "subfolder": "", "type": "input"},
    {"filename": "../tmp.png", "subfolder": "", "type": "temp"},
    {"filename": "../out.png", "subfolder": "", "type": "output"},
    {"filename": "out.png", "subfolder": r"..\private", "type": "output"},
    {"filename": "out.png", "subfolder": "../private", "type": "output"},
])
def test_f13_only_safe_output_images_reach_view(monkeypatch, image):
    def request(self, method, path, **kwargs):
        assert path.startswith("/history/"), "unsafe view requested"
        return Response({"p": {"status": {"completed": True}, "outputs": {"1": {"images": [image]}}}})
    monkeypatch.setattr(r.RunnerClient, "_request", request)
    with pytest.raises(PromptSubmissionError):
        r.RunnerClient("stub", timeout=1).retrieve_output_rasters("p")


def test_preview_image_temp_output_is_downloaded_but_input_is_not(monkeypatch):
    def request(self, method, path, **kwargs):
        if path == "/view":
            assert kwargs["params"] == {"filename": "ComfyUI_temp_00001_.png", "subfolder": "", "type": "temp"}
            return Response(content=b"preview")
        return Response({"p": {"status": {"completed": True}, "outputs": {
            "9": {"images": [{"filename": "cond.png", "subfolder": "", "type": "input"}]},
            "73": {"images": [{"filename": "ComfyUI_temp_00001_.png", "subfolder": "", "type": "temp"}]}}}})
    monkeypatch.setattr(r.RunnerClient, "_request", request)
    assert r.RunnerClient("stub", timeout=1).retrieve_output_rasters("p")[0][1] == b"preview"


def test_f13_incomplete_history_does_not_download_an_image(monkeypatch):
    monkeypatch.setattr(r.time, "sleep", lambda interval: None)
    polls = []

    def request(self, method, path, **kwargs):
        if path == "/view":
            assert len(polls) == 2
            assert kwargs["params"] == {"filename": "out.png", "subfolder": "safe/sub", "type": "output"}
            return Response(content=b"output")
        polls.append(path)
        return Response({"p": {"status": {"completed": len(polls) == 2}, "outputs": {"1": {"images": [
            {"filename": "out.png", "subfolder": "safe/sub", "type": "output"}]}}}})
    monkeypatch.setattr(r.RunnerClient, "_request", request)
    assert r.RunnerClient("stub").retrieve_output_rasters("p")[0][1] == b"output"


@pytest.mark.parametrize("promotion,base,target", [("E1", "E2", True), ("E4", "E5", False)])
@pytest.mark.parametrize("invalid", ["status", "preflight", "registration", "manifest",
                                     "evaluation", "mode", "subtier"])
def test_f14_report_applies_only_consistent_successful_records(case_factory, promotion, base, target, invalid):
    case, args, _ = case_factory(base, target)
    assert run_regenerate(args) == 0
    record = saved(case)
    manifest = json.loads((case / "manifest.json").read_text())
    sha = sha256_file(case / "manifest.json")
    findings, _ = effective_findings(manifest, [record], sha, case)
    assert findings[0]["tier"] == promotion
    bad = deepcopy(record)
    if invalid == "status":
        bad["status"] = "FAILED"
    elif invalid == "preflight":
        bad["preflight_check"]["status"] = "SECURITY_REJECTED"
    elif invalid == "registration":
        bad["preflight_check"]["runner_object_info_verified"] = False
    elif invalid == "manifest":
        bad["target_manifest_sha256"] = "0" * 64
    elif invalid == "evaluation":
        bad["evaluation"]["evaluated_tier"] = "NON_MATCHING"
    elif invalid == "subtier":
        bad["evaluation"]["evaluation_subtier"] = "BELOW_THRESHOLDS"
    else:
        bad["evaluation"]["evaluation_mode"] = "PARTIAL_ROW_BOUNDED"
    findings, notes = effective_findings(manifest, [bad], sha, case)
    assert findings[0]["tier"] == base
    assert notes[0]["applied_to"] is None


def test_f1_ssim_threshold_uses_unrounded_score():
    target = noise_image(16, 16, seed=81)
    regen = target.copy()
    regen.putpixel((7, 7), (1, 2, 3))
    raw = float(structural_similarity(np.asarray(target.convert("RGBA")), np.asarray(regen.convert("RGBA")),
                                      channel_axis=2, data_range=255))
    threshold = (raw + round(raw, 6)) / 2
    assert raw < threshold < round(raw, 6)
    [record] = PngCarver().carve_stream(encoded(target))
    tier, _, metrics = r.TieredEvaluator(threshold, 64).evaluate_reproduction(encoded(regen), record, encoded(target))
    assert tier == "NON_MATCHING"
    assert metrics["ssim_score"] == raw


def test_f2_non_png_runner_output_is_failed_and_never_saved_as_png(case_factory):
    case, args, state = case_factory()
    state.image = encoded(Image.new("RGB", (8, 8)), "GIF")
    assert run_regenerate(args) == 1
    assert saved(case)["status"] == "FAILED"
    assert not list((case / "exports/derived").glob("*.png"))


@pytest.mark.parametrize("late_stage", ["history", "view"])
def test_f3_deadline_applies_after_history_and_download(monkeypatch, late_stage):
    now = [0.0]
    budgets = []
    monkeypatch.setattr(r.time, "monotonic", lambda: now[0])

    def request(self, method, path, **kwargs):
        budgets.append(kwargs.get("timeout", 30))
        if late_stage in path:
            now[0] = 2.0
        if path == "/view":
            return Response(content=b"late output")
        return Response({"p": {"status": {"completed": True}, "outputs": {"1": {"images": [
            {"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
    monkeypatch.setattr(r.RunnerClient, "_request", request)
    with pytest.raises(ExecutionTimeoutError):
        r.RunnerClient("stub", timeout=1).retrieve_output_rasters("p")
    assert budgets and all(0 < budget <= 1 for budget in budgets)


def test_f4_exact_partial_rows_work_without_perceptual_extras(monkeypatch):
    full = noise_image(64, 64, seed=7)
    png = png_bytes(full, idat_chunk=400)
    [partial] = PngCarver().carve_stream(png[:len(png) // 2])

    def unavailable():
        raise r.MissingDependencyError("optional extras absent")
    monkeypatch.setattr(r, "_perceptual_backends", unavailable)
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(encoded(full), partial, partial.preview_png)
    assert (tier, subtier) == ("PARTIAL_CORROBORATION", "ROW_SLICE_EXACT")
    assert metrics["row_slice_pixel_match"] is True


def test_f5_sft_is_not_authorized_as_safetensors(monkeypatch):
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "weights.sft"}}}
    assert preflight(monkeypatch, graph, choices=["weights.sft"])[:2] == (False, "SECURITY_REJECTED")


@pytest.mark.parametrize("node_class", ["UpscaleModelLoader"])
def test_f6_core_loaders_do_not_require_custom_node_authorization(monkeypatch, node_class):
    graph = {"1": {"class_type": node_class, "inputs": {}}}
    assert preflight(monkeypatch, graph)[:2] == (True, "PASSED")


def test_f7_html_exposes_escaped_security_preflight_and_environment(case_factory):
    case, args, _ = case_factory()
    assert run_regenerate(args) == 0
    record = saved(case)
    record["handoff_id"] = "REGEN-002"
    record["security_audit"]["allowed_custom_nodes"] = ["<AuthorizedNode>"]
    record["preflight_check"]["forensic_warning"] = "pickle warning"
    record["runner_environment"]["comfyui_version"] = "inspected-runner-version"
    save_regeneration_record(case, RegenerationRecord.from_dict(record))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    assert not (case / "report.html").exists()
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    assert "<AuthorizedNode>" not in wb
    assert (case / "workbench.html").is_file()


@pytest.mark.parametrize("base", ["E2", "E3"])
def test_f8_execution_without_an_earned_e1_or_e4_has_nonzero_exit(case_factory, base):
    case, args, _ = case_factory(base, True)
    if base == "E2":
        args.compare_to = None
    assert run_regenerate(args) == 2
    assert saved(case)["evaluation"]["promoted_finding_tier"] is None
    if base == "E3":
        assert saved(case)["evaluation"]["evaluated_tier"] == "PARTIAL_CORROBORATION"
        assert saved(case)["evaluation"]["target_finding_tier_retained"] == "E3"


@pytest.mark.parametrize("unreachable", [False, True])
def test_f9_rejections_and_unreachable_runner_keep_available_schema_fields(case_factory, monkeypatch, unreachable):
    graph = api_prompt()
    if not unreachable:
        graph["9"]["class_type"] = "UnapprovedNode"
    case, args, _ = case_factory("E2", True, graph=graph)
    before = sha256_file(case / "manifest.json")
    if unreachable:
        def unavailable(self):
            raise RunnerConnectionError("runner unreachable")
        monkeypatch.setattr(r.RunnerClient, "object_info", unavailable)
    assert run_regenerate(args) == 1
    record = saved(case)
    assert record["status"] == ("FAILED" if unreachable else "SECURITY_REJECTED")
    assert record["target_manifest_sha256"] == before == sha256_file(case / "manifest.json")
    assert record["derived_regenerated_image"] is None
    assert record["preflight_check"]["runner_object_info_verified"] is False
    for field in ["evaluation_mode", "exact_pixel_match", "ssim_score", "phash_hamming_distance", "evaluated_tier",
                  "evaluation_subtier", "promoted_finding_tier", "target_finding_tier_retained"]:
        assert field in record["evaluation"]
    assert record["evaluation"]["target_finding_tier_retained"] == "E2"
    assert record["runner_environment"]["python_version"] == "UNREPORTED"


@pytest.mark.parametrize("node_class", ["LoadLatent", "SaveLatent"])
def test_r1_latent_nodes_require_explicit_authorization(monkeypatch, node_class):
    graph = {"1": {"class_type": node_class, "inputs": {"latent": "source.latent"}}}
    assert preflight(monkeypatch, graph)[:2] == (False, "SECURITY_REJECTED")
    assert preflight(monkeypatch, graph, allowed=[node_class])[:2] == (True, "PASSED")


@pytest.mark.parametrize("name", ["../../srv/secret.latent", "/srv/x.latent", r"..\secret.latent",
                                  "C:secret.latent", "prior.latent [output]", "prior.latent [temp]",
                                  "prior.latent [input]", "prior.latent[output]", "prior.latent[temp]",
                                  "prior.latent[input]", "", ".", "..", "source.latent "])
def test_r1_authorized_latent_loader_rejects_unsafe_names(case_factory, name):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadLatent", "inputs": {"latent": name}}
    case, args, state = case_factory(graph=graph)
    args.allow_node = ["LoadLatent"]
    assert run_regenerate(args) == 1
    assert saved(case)["status"] == "SECURITY_REJECTED"
    assert "/prompt" not in [path for path, _ in state.calls]


@pytest.mark.parametrize("loader", ["LoadImage", "LoadImageMask"])
@pytest.mark.parametrize("annotation", [" [output]", " [temp]", " [input]", "[output]", "[temp]", "[input]"])
def test_r2_annotated_images_rejected_before_upload(case_factory, tmp_path, loader, annotation):
    name = f"cond.png{annotation}"
    graph = api_prompt()
    graph["10"] = {"class_type": loader, "inputs": {"image": name}}
    case, args, state = case_factory(graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / name).write_bytes(state.image)
    assert run_regenerate(args) == 1
    assert saved(case)["status"] == "SECURITY_REJECTED"
    assert not {"/upload/image", "/prompt"}.intersection(path for path, _ in state.calls)


@pytest.mark.parametrize("reply", ["cond.png [output]", "cond.png [temp]", "cond.png [input]",
                                 "cond.png[output]", "cond.png[temp]", "cond.png[input]", "other.png"])
def test_r2_upload_reply_must_match_the_hashed_input(case_factory, tmp_path, reply):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    case, args, state = case_factory(graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / "cond.png").write_bytes(state.image)
    state.upload_name = reply
    assert run_regenerate(args) == 1
    assert saved(case)["status"] == "FAILED"
    assert "/upload/image" in [path for path, _ in state.calls]
    assert "/prompt" not in [path for path, _ in state.calls]


@pytest.mark.parametrize("output", ["SaveImage", "PreviewImage"])
@pytest.mark.parametrize("disconnected_sampler", [False, True])
def test_r3_only_sampled_outputs_are_compared(case_factory, tmp_path, output, disconnected_sampler):
    graph = api_prompt() if disconnected_sampler else {}
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["11"] = {"class_type": output, "inputs": {"images": ["10", 0]}}
    case, args, state = case_factory("E2", True, graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / "cond.png").write_bytes(state.image)
    code = run_regenerate(args)
    record = saved(case)
    assert record["evaluation"]["promoted_finding_tier"] is None
    if not disconnected_sampler:
        assert code == 1 and record["status"] == "SECURITY_REJECTED"
        assert "sampler ancestor" in record["preflight_check"]["reason"]
        assert "/prompt" not in [path for path, _ in state.calls]
    else:
        assert "11" not in record["preflight_check"]["compared_output_nodes"], "a preview of an input is ignored"


def test_a_sampler_free_image_edit_is_complete_and_replays_as_a_reference(case_factory, tmp_path):
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "width": 64, "height": 48,
                                                     "upscale_method": "nearest-exact", "crop": "disabled"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    recovered = workflow(graph)
    assert recovered.completeness == "COMPLETE"
    assert recovered.execution_parameters["workflow_operation"] == "IMAGE_EDIT"
    assert "seed" not in recovered.execution_parameters
    assert workflow({n: v for n, v in graph.items() if n != "1"}).completeness == "PARTIAL"

    case, args, _ = case_factory("E2", True, graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / "source.png").write_bytes(encoded(noise_image(32, 24, seed=6)))
    assert run_regenerate(args) == 2
    record = saved(case)
    assert record["status"] == "COMPLETED"
    assert record["preflight_check"]["comparison_only_output_nodes"] == ["9"]
    assert record["evaluation"]["evaluated_tier"] == "REFERENCE_COMPARISON"
    assert record["evaluation"]["exact_pixel_match"] is True
    assert record["evaluation"]["promoted_finding_tier"] is None


@pytest.mark.parametrize("kind", ["cycle", "dangling"])
def test_r3_invalid_ancestry_is_rejected_without_hanging(monkeypatch, kind):
    graph = {"1": {"class_type": "ImageInvert", "inputs": {"image": ["1" if kind == "cycle" else "99", 0]}},
             "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
    assert preflight(monkeypatch, graph)[:2] == (False, "SECURITY_REJECTED")


@pytest.mark.parametrize("target", [False, True])
def test_r3_unsampled_complete_workflow_cannot_earn_e4_or_e1(case_factory, tmp_path, target):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["9"]["inputs"]["images"] = ["10", 0]
    assert workflow(graph).completeness == "COMPLETE"
    case, args, state = case_factory("E2" if target else "E5", target, graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / "cond.png").write_bytes(state.image)
    assert run_regenerate(args) == 1
    assert saved(case)["evaluation"]["promoted_finding_tier"] is None
    assert "/prompt" not in [path for path, _ in state.calls]


@pytest.mark.parametrize("reencode", [False, True])
def test_r3_target_supplied_as_input_withholds_e1_even_with_a_sampler(case_factory, tmp_path, reencode):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
    graph["3"]["inputs"]["latent_image"] = ["11", 0]
    case, args, state = case_factory("E2", True, graph=graph)
    args.inputs_dir = tmp_path
    payload = encoded(Image.open(io.BytesIO(state.image))) if reencode else state.image
    if reencode:
        assert hashlib.sha256(payload).hexdigest() != hashlib.sha256(state.image).hexdigest()
    (tmp_path / "cond.png").write_bytes(payload)
    assert run_regenerate(args) == 2
    record = saved(case)
    assert record["status"] == "COMPLETED"
    assert record["evaluation"]["evaluated_tier"] == "E1"
    assert record["evaluation"]["promoted_finding_tier"] is None
    assert record["evaluation"]["target_finding_tier_retained"] == "E2"
    assert "input" in record["evaluation"]["note"]
    assert "/prompt" in [path for path, _ in state.calls]


def test_r3_distinct_conditioning_image_preserves_earned_e1(case_factory, tmp_path):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    case, args, _ = case_factory("E2", True, graph=graph)
    args.inputs_dir = tmp_path
    (tmp_path / "cond.png").write_bytes(encoded(noise_image(64, 48, seed=99)))
    assert run_regenerate(args) == 0
    assert saved(case)["evaluation"]["promoted_finding_tier"] == "E1"


@pytest.mark.parametrize("flag,value", [("--ssim-threshold", "-0.01"), ("--ssim-threshold", "1.01"),
                                      ("--ssim-threshold", "nan"), ("--ssim-threshold", "inf"),
                                      ("--phash-threshold", "-1"), ("--phash-threshold", "65"),
                                      ("--phash-threshold", "1.5")])
def test_r4_cli_rejects_out_of_bounds_thresholds(flag, value):
    with pytest.raises(SystemExit) as exc:
        parse_args(["regenerate", "--case-dir", "case", "--workflow", "workflow.json", "--runner-url", "stub",
                    flag, value])
    assert exc.value.code == 1


@pytest.mark.parametrize("ssim,phash", [(0, 0), (1, 64)])
def test_r4_cli_accepts_threshold_bounds(ssim, phash):
    args = parse_args(["regenerate", "--case-dir", "case", "--workflow", "workflow.json", "--runner-url", "stub",
                       "--ssim-threshold", str(ssim), "--phash-threshold", str(phash)])
    assert (args.ssim_threshold, args.phash_threshold) == (ssim, phash)


def test_r4_thresholds_are_recorded_for_success(case_factory):
    case, args, _ = case_factory()
    args.ssim_threshold, args.phash_threshold = 0.87, 9
    assert run_regenerate(args) == 0
    evaluation = saved(case)["evaluation"]
    assert (evaluation["ssim_threshold"], evaluation["phash_threshold"]) == (0.87, 9)


def report_tier(case, record, tier):
    (case / "regenerations/REGEN-001.json").write_text(json.dumps(record))
    assert run_report(parse_args(["report", "--case-dir", str(case)])) == 0
    assert not (case / "report.html").exists()
    assert f'"id":"FIND-001","tier":"{tier}"' in (case / "workbench.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("invalid", ["low_ssim", "high_phash", "missing_ssim", "missing_phash", "text_ssim",
                                    "bool_phash", "nan", "inf", "missing_threshold", "bad_threshold"])
def test_r5_report_requires_perceptual_scores_within_recorded_thresholds(case_factory, invalid):
    case, args, _ = case_factory("E2", True)
    assert run_regenerate(args) == 0
    record = saved(case)
    evaluation = record["evaluation"]
    evaluation.update(evaluation_subtier="PERCEPTUAL_MATCH", exact_pixel_match=False, ssim_score=0.97,
                      phash_hamming_distance=3, ssim_threshold=0.96, phash_threshold=3)
    report_tier(case, record, "E1")
    field, value = {
        "low_ssim": ("ssim_score", 0.959), "high_phash": ("phash_hamming_distance", 4),
        "missing_ssim": ("ssim_score", None), "missing_phash": ("phash_hamming_distance", None),
        "text_ssim": ("ssim_score", "0.99"), "bool_phash": ("phash_hamming_distance", False),
        "nan": ("ssim_score", float("nan")), "inf": ("ssim_score", float("inf")),
        "missing_threshold": ("ssim_threshold", None), "bad_threshold": ("phash_threshold", 65),
    }[invalid]
    evaluation[field] = value
    report_tier(case, record, "E2")


@pytest.mark.parametrize("base,target", [("E2", True), ("E5", False)])
@pytest.mark.parametrize("invalid", ["missing", "modified", "no_hash", "wrong_hash", "escape"])
def test_r5_report_requires_the_hash_bound_derived_file(case_factory, tmp_path, base, target, invalid):
    case, args, _ = case_factory(base, target)
    assert run_regenerate(args) == 0
    record = saved(case)
    derived = case / record["derived_regenerated_image"]
    assert record["evaluation"]["derived_regenerated_image_sha256"] == sha256_file(derived)
    report_tier(case, record, "E1" if target else "E4")
    if invalid == "missing":
        derived.unlink()
    elif invalid == "modified":
        derived.write_bytes(b"replaced")
    elif invalid == "no_hash":
        del record["evaluation"]["derived_regenerated_image_sha256"]
    elif invalid == "wrong_hash":
        record["evaluation"]["derived_regenerated_image_sha256"] = "0" * 64
    else:
        outside = tmp_path / "outside.png"
        outside.write_bytes(derived.read_bytes())
        record["derived_regenerated_image"] = "../outside.png"
    report_tier(case, record, base)


@pytest.mark.parametrize("model_format", ["PICKLE", "MIXED"])
def test_r5_report_rejects_unauthorized_pickle_audit(case_factory, model_format):
    case, args, _ = case_factory()
    assert run_regenerate(args) == 0
    record = saved(case)
    record["security_audit"].update(model_format=model_format, pickle_models_permitted=False)
    report_tier(case, record, "E5")
    record["security_audit"]["pickle_models_permitted"] = True
    report_tier(case, record, "E4")


@pytest.mark.parametrize("digest", ["container_sha256", "pixel_sha256"])
def test_r3_report_does_not_reapply_an_input_as_reproduction(case_factory, digest):
    case, args, _ = case_factory("E2", True)
    assert run_regenerate(args) == 0
    record = saved(case)
    artifact = json.loads((case / "manifest.json").read_text())["findings"]["artifacts"][0]
    record["preflight_check"]["input_images"] = {"cond.png": artifact[digest]}
    report_tier(case, record, "E2")


@pytest.mark.parametrize("text", ["embedding:style.safetensors:evil", "(embedding:style.safetensors:evil)",
                                 "embedding:style.safetensors:1.2", "(embedding:style.safetensors:evil:1.2)",
                                 "embedding:style.safetensors:evil\tportrait", "embedding:",
                                 "embedding:style[x].safetensors"])
def test_r6_full_embedding_word_is_checked(monkeypatch, text):
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}
    assert preflight(monkeypatch, graph, choices=["style.safetensors"])[:2] == (False, "SECURITY_REJECTED")


@pytest.mark.parametrize("text", ["embedding:style.safetensors", "embedding:style.safetensors\nportrait",
                                 "(embedding:style.safetensors:1.2)", "(embedding:style.safetensors:-0.25)",
                                 "(embedding:style.safetensors)", "embedding:style.safetensors\tx"])
def test_r6_plain_and_parenthesized_weighted_embeddings_are_hashed(tmp_path, monkeypatch, text):
    (tmp_path / "style.safetensors").write_bytes(b"embedding")
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}
    passed, _, audit = preflight(monkeypatch, graph, models=tmp_path, choices=["style.safetensors"])
    assert passed
    assert audit["preflight_check"]["model_sha256"] == {"style.safetensors": hashlib.sha256(b"embedding").hexdigest()}


@pytest.mark.parametrize("target", [False, True])
def test_workflow_copies_in_a_finding_preserve_earned_promotion(case_factory, target):
    case, args, _ = case_factory("E2" if target else "E5", target, copies=True)
    assert run_regenerate(args) == 0
    report_tier(case, saved(case), "E1" if target else "E4")


def test_previewing_an_input_beside_a_sampled_output_passes_preflight(monkeypatch, tmp_path):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["13"] = {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}}
    (tmp_path / "cond.png").write_bytes(b"png")
    passed, status, audit = preflight(monkeypatch, graph, inputs=tmp_path,
                                      choices=["v1-5-pruned-emaonly.safetensors", "cond.png"])
    assert (passed, status) == (True, "PASSED")
    assert audit["preflight_check"]["compared_output_nodes"] == ["9"]


def test_history_outputs_of_uncompared_nodes_are_ignored(monkeypatch):
    def request(self, method, path, **kwargs):
        if path == "/view":
            assert kwargs["params"]["filename"] == "sampled.png"
            return Response(content=b"sampled")
        return Response({"p": {"status": {"completed": True}, "outputs": {
            "13": {"images": [{"filename": "preview_of_input.png", "subfolder": "", "type": "output"}]},
            "9": {"images": [{"filename": "sampled.png", "subfolder": "", "type": "output"}]}}}})
    monkeypatch.setattr(r.RunnerClient, "_request", request)
    assert r.RunnerClient("stub", timeout=1).retrieve_output_rasters("p", ["9"])[0][1] == b"sampled"


@pytest.mark.parametrize("text,name", [("embedding:EasyNegative.safetensors, bad hands", "EasyNegative.safetensors"),
                                       ("(worst quality, embedding:bad.safetensors)", "bad.safetensors"),
                                       ("((embedding:bad.safetensors:1.1))", "bad.safetensors"),
                                       ("(embedding:bad.safetensors:1.1),", "bad.safetensors"),
                                       ("embedding:bad", "bad.safetensors")])
def test_ordinary_prompt_punctuation_around_embeddings(tmp_path, monkeypatch, text, name):
    (tmp_path / name).write_bytes(b"embedding")
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}
    passed, _, audit = preflight(monkeypatch, graph, models=tmp_path, choices=[name])
    assert passed and list(audit["preflight_check"]["model_sha256"]) == [name]


def test_embedding_audit_follows_comfyui_file_resolution(tmp_path, monkeypatch):
    picked = "style.safetensors).pt"
    (tmp_path / "style.safetensors").write_bytes(b"safe")
    (tmp_path / picked).write_bytes(b"pickle")
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style.safetensors)"}}}
    assert preflight(monkeypatch, graph, models=tmp_path, choices=[picked])[:2] == (False, "SECURITY_REJECTED")
    passed, _, audit = preflight(monkeypatch, graph, models=tmp_path, choices=[picked], allow_pickle=True)
    assert passed and audit["preflight_check"]["model_sha256"] == {picked: hashlib.sha256(b"pickle").hexdigest()}


def test_comma_extension_file_beside_the_stripped_name_is_ambiguous(tmp_path, monkeypatch):
    (tmp_path / "style.safetensors").write_bytes(b"safe")
    (tmp_path / "style.safetensors,.pt").write_bytes(b"pickle")
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style.safetensors, portrait"}}}
    assert preflight(monkeypatch, graph, models=tmp_path, allow_pickle=True,
                     choices=["style.safetensors", "style.safetensors,.pt"])[:2] == (False, "SECURITY_REJECTED")


def test_comma_fallback_needs_the_models_dir_to_resolve(monkeypatch):
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style.safetensors, portrait"}}}
    passed, status, audit = preflight(monkeypatch, graph, choices=["style.safetensors"])
    assert (passed, status) == (False, "SECURITY_REJECTED") and "--models-dir" in audit["preflight_check"]["reason"]


def test_embedding_found_in_two_subdirectories_is_ambiguous(tmp_path, monkeypatch):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "style.safetensors").write_bytes(sub.encode())
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style.safetensors"}}}
    assert preflight(monkeypatch, graph, models=tmp_path, choices=["a/style.safetensors"])[:2] == (
        False, "SECURITY_REJECTED")


def test_authorized_custom_output_node_is_compared(monkeypatch):
    graph = api_prompt()
    saver = next(k for k, n in graph.items() if n["class_type"] == "SaveImage")
    graph[saver]["class_type"] = "AuthorizedSaveImage"
    passed, _, audit = preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"],
                                 allowed=["AuthorizedSaveImage"], schemas={"AuthorizedSaveImage": SAVER_SCHEMA})
    assert passed and audit["preflight_check"]["compared_output_nodes"] == [saver]


def test_comma_chain_with_two_loadable_files_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "style,.safetensors").write_bytes(b"maybe corrupt")
    (tmp_path / "style.pt").write_bytes(b"pickle")
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "embedding:style, portrait"}}}
    passed, status, audit = preflight(monkeypatch, graph, models=tmp_path, allow_pickle=True,
                                      choices=["style,.safetensors", "style.pt"])
    assert (passed, status) == (False, "SECURITY_REJECTED") and "style.pt" in audit["preflight_check"]["reason"]


def test_sampler_named_output_is_not_its_own_sampler(monkeypatch, tmp_path):
    graph = {"10": {"class_type": "LoadImage", "inputs": {"image": "cond.png"}},
             "9": {"class_type": "SamplerPreviewImage", "inputs": {"images": ["10", 0]}}}
    (tmp_path / "cond.png").write_bytes(b"png")
    _, _, audit = preflight(monkeypatch, graph, inputs=tmp_path, choices=["cond.png"],
                            allowed=["SamplerPreviewImage"])
    assert audit["preflight_check"]["compared_output_nodes"] == []


def test_custom_sampler_counts_only_when_vetted(monkeypatch):
    graph = api_prompt()
    sampler = next(k for k, n in graph.items() if n["class_type"] == "KSampler")
    saver = next(k for k, n in graph.items() if n["class_type"] == "SaveImage")
    graph[sampler]["class_type"] = "KSamplerEfficient"
    choices = ["v1-5-pruned-emaonly.safetensors"]
    assert preflight(monkeypatch, graph, choices=choices, allowed=["KSamplerEfficient"])[:2] == (
        False, "SECURITY_REJECTED")
    passed, _, audit = preflight(monkeypatch, graph, choices=choices, samplers=["KSamplerEfficient"])
    assert passed and audit["preflight_check"]["compared_output_nodes"] == [saver]


def _embed(monkeypatch, root, text, choices, allow_pickle=False):
    graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}
    return preflight(monkeypatch, graph, models=root, choices=choices, allow_pickle=allow_pickle)


def test_embedding_loader_policy_follows_the_requested_name_not_a_symlink_target(tmp_path, monkeypatch):
    (tmp_path / "payload.safetensors").write_bytes(b"tensors")
    _symlink(tmp_path / "style.pt", "payload.safetensors")
    choices = ["style.pt", "payload.safetensors"]
    assert _embed(monkeypatch, tmp_path, "embedding:style.pt", choices)[:2] == (False, "SECURITY_REJECTED")
    (tmp_path / "payload.pt").write_bytes(b"pickle bytes")
    _symlink(tmp_path / "plain.safetensors", "payload.pt")
    passed, _, audit = _embed(monkeypatch, tmp_path, "embedding:plain.safetensors", ["plain.safetensors"])
    assert passed and audit["preflight_check"]["model_sha256"] == {
        "plain.safetensors": hashlib.sha256(b"pickle bytes").hexdigest()}


def test_dangling_symlink_falls_through_to_the_extension_ComfyUI_opens(tmp_path, monkeypatch):
    _symlink(tmp_path / "style", "other")
    (tmp_path / "style.pt").write_bytes(b"pickle")
    (tmp_path / "other.safetensors").write_bytes(b"tensors")
    choices = ["style.pt", "other.safetensors"]
    assert _embed(monkeypatch, tmp_path, "embedding:style", choices)[:2] == (False, "SECURITY_REJECTED")
    passed, _, audit = _embed(monkeypatch, tmp_path, "embedding:style", choices, allow_pickle=True)
    assert passed and audit["preflight_check"]["model_sha256"] == {"style.pt": hashlib.sha256(b"pickle").hexdigest()}


def test_embedding_symlinked_outside_the_models_dir_is_rejected(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (tmp_path / "elsewhere.safetensors").write_bytes(b"x")
    _symlink(models / "style.safetensors", tmp_path / "elsewhere.safetensors")
    passed, status, audit = _embed(monkeypatch, models, "embedding:style.safetensors", ["style.safetensors"])
    assert (passed, status) == (False, "SECURITY_REJECTED") and "outside" in audit["preflight_check"]["reason"]


def test_embedding_found_through_a_symlinked_subdirectory_is_audited(tmp_path, monkeypatch):
    models, shared = tmp_path / "models", tmp_path / "shared"
    models.mkdir()
    shared.mkdir()
    (shared / "style.safetensors").write_bytes(b"x")
    _symlink(models / "linked", shared, target_is_directory=True)
    passed, status, audit = _embed(monkeypatch, models, "embedding:style.safetensors", ["style.safetensors"])
    assert (passed, status) == (False, "SECURITY_REJECTED") and "outside" in audit["preflight_check"]["reason"]


def test_sampler_feeding_only_the_filename_does_not_prove_the_image(monkeypatch, tmp_path):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["12"] = {"class_type": "LatentFilename", "inputs": {"samples": ["3", 0]}}
    graph["9"]["inputs"] = {"images": ["10", 0], "filename_prefix": ["12", 0]}
    (tmp_path / "cond.png").write_bytes(b"png")
    schemas = {"SaveImage": SAVER_SCHEMA,
               "LatentFilename": {"input": {"required": {"samples": ["LATENT"]}}, "output": ["STRING"],
                                  "output_node": False}}
    passed, status, audit = preflight(monkeypatch, graph, inputs=tmp_path, allowed=["LatentFilename"],
                                      choices=["v1-5-pruned-emaonly.safetensors", "cond.png"], schemas=schemas)
    assert (passed, status) == (False, "SECURITY_REJECTED")
    assert "compared_output_nodes" not in audit["preflight_check"]


def test_authorized_intermediate_node_is_not_a_compared_output(monkeypatch):
    graph = api_prompt()
    graph["8"]["class_type"] = "CustomDecode"
    schemas = {"CustomDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                                "output": ["IMAGE"], "output_node": False}}
    passed, _, audit = preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"],
                                 allowed=["CustomDecode"], schemas=schemas)
    assert passed and audit["preflight_check"]["compared_output_nodes"] == ["9"]


@pytest.mark.parametrize("graph", [
    {"9": {"class_type": "SamplerSaver", "inputs": {"images": ["9", 0]}}},
    {"9": {"class_type": "SamplerSaver", "inputs": {"images": ["8", 0]}},
     "8": {"class_type": "ImageInvert", "inputs": {"image": ["9", 0]}}}])
def test_cyclic_graph_cannot_make_an_output_its_own_sampler(monkeypatch, graph):
    assert preflight(monkeypatch, graph, samplers=["SamplerSaver"], schemas={"SamplerSaver": SAVER_SCHEMA})[:2] == (
        False, "SECURITY_REJECTED")


def test_image_batch_with_an_unsampled_input_is_not_compared(monkeypatch, tmp_path):
    graph = api_prompt()
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "cond.png"}}
    graph["11"] = {"class_type": "ImageBatch", "inputs": {"image1": ["10", 0], "image2": ["8", 0]}}
    graph["9"]["inputs"]["images"] = ["11", 0]
    (tmp_path / "cond.png").write_bytes(b"png")
    assert preflight(monkeypatch, graph, inputs=tmp_path,
                     choices=["v1-5-pruned-emaonly.safetensors", "cond.png"])[:2] == (False, "SECURITY_REJECTED")


@pytest.mark.parametrize("text", ["embedding:sub\\style.safetensors", "embedding:sub\\style"])
def test_backslash_in_an_embedding_name_is_rejected(tmp_path, monkeypatch, text):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "style.safetensors").write_bytes(b"a")
    assert _embed(monkeypatch, tmp_path, text, [])[:2] == (False, "SECURITY_REJECTED")


def test_loader_value_mentioning_embedding_keeps_its_model_checks(tmp_path, monkeypatch):
    graph = api_prompt(ckpt="embedding:ok.safetensors legacy.ckpt")
    (tmp_path / "ok.safetensors").write_bytes(b"ok")
    assert preflight(monkeypatch, graph, models=tmp_path, choices=["embedding:ok.safetensors legacy.ckpt"])[:2] == (
        False, "SECURITY_REJECTED")


def test_prompt_text_computed_from_a_link_is_rejected(monkeypatch):
    graph = api_prompt()
    graph["12"] = {"class_type": "StringConcatenate", "inputs": {"a": "portrait, embed", "b": "ding:evil"}}
    graph["6"]["inputs"]["text"] = ["12", 0]
    passed, status, audit = preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"],
                                      allowed=["StringConcatenate"])
    assert (passed, status) == (False, "SECURITY_REJECTED") and "6.text" in audit["preflight_check"]["reason"]


def test_symlink_alias_of_the_same_embedding_is_one_dependency(tmp_path, monkeypatch):
    (tmp_path / "styles").mkdir()
    (tmp_path / "styles" / "style.safetensors").write_bytes(b"tensors")
    _symlink(tmp_path / "style.safetensors", tmp_path / "styles" / "style.safetensors")
    passed, _, audit = _embed(monkeypatch, tmp_path, "embedding:style", [])
    assert passed and list(audit["preflight_check"]["model_sha256"]) == ["style.safetensors"]


@pytest.mark.parametrize("slot", [1, 999])
def test_raster_edge_from_a_non_raster_or_missing_output_slot_is_unsampled(monkeypatch, slot):
    graph = api_prompt()
    graph["8"]["inputs"]["samples"] = ["3", slot]
    schemas = {"KSampler": {"input": {}, "output": ["LATENT", "STRING"]}}
    assert preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"], schemas=schemas)[:2] == (
        False, "SECURITY_REJECTED")


def test_checkpoint_named_like_an_embedding_keeps_its_loader_checks(tmp_path, monkeypatch):
    name = "v1-5-pruned-emaonly.safetensors"
    graph = api_prompt()
    graph["6"]["inputs"]["text"] = f"embedding:{name}"
    (tmp_path / name).write_bytes(b"both")
    assert preflight(monkeypatch, graph, models=tmp_path, choices=["different.safetensors"])[:2] == (
        False, "MISSING_DEPENDENCY")
    (tmp_path / name).unlink()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / name).write_bytes(b"embedding only")
    assert preflight(monkeypatch, graph, models=tmp_path, choices=[name])[:2] == (False, "MISSING_DEPENDENCY")


@pytest.mark.parametrize("reverse", [False, True])
def test_deep_graph_acceptance_does_not_depend_on_json_order(monkeypatch, reverse):
    graph = api_prompt()
    saver = graph.pop("9")
    previous = "8"
    for i in range(100, 700):
        graph[str(i)] = {"class_type": "ImageInvert", "inputs": {"image": [previous, 0]}}
        previous = str(i)
    graph["9"] = {**saver, "inputs": {**saver["inputs"], "images": [previous, 0]}}
    if reverse:
        graph = dict(reversed(list(graph.items())))
    passed, _, audit = preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"])
    assert passed and audit["preflight_check"]["compared_output_nodes"] == ["9"]


@pytest.mark.parametrize("cls,settings", [("KSampler", {"denoise": 0.0}), ("KSampler", {"steps": 0}),
                                          ("KSampler", {"denoise": ["12", 0]}),
                                          ("KSamplerAdvanced", {"start_at_step": 20, "end_at_step": 10000}),
                                          ("KSamplerAdvanced", {"start_at_step": 5, "end_at_step": 5})])
def test_sampler_settings_that_skip_sampling_do_not_prove_the_image(monkeypatch, cls, settings):
    graph = api_prompt()
    graph["3"]["class_type"] = cls
    graph["3"]["inputs"].update(settings)
    assert preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"])[:2] == (
        False, "SECURITY_REJECTED")


def test_ordinary_advanced_sampler_and_partial_denoise_pass(monkeypatch):
    graph = api_prompt()
    graph["3"]["inputs"]["denoise"] = 0.35
    assert preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"])[:2] == (True, "PASSED")
    graph["3"]["class_type"] = "KSamplerAdvanced"
    graph["3"]["inputs"].update(start_at_step=0, end_at_step=10000)
    assert preflight(monkeypatch, graph, choices=["v1-5-pruned-emaonly.safetensors"])[:2] == (True, "PASSED")


def test_every_compared_image_is_evaluated_and_the_used_one_recorded(case_factory, monkeypatch):
    case, args, _ = case_factory("E2", True)
    target = (case / "exports" / "raw").glob("*.png")
    target_png = next(iter(sorted(target))).read_bytes()
    other = png_bytes(noise_image(64, 48, seed=77))

    def rasters(self, prompt_id, nodes=None, limit=64):
        return [({"node_id": "9", "batch_index": 0, "filename": "a.png"}, other),
                ({"node_id": "9", "batch_index": 1, "filename": "b.png"}, target_png)]

    monkeypatch.setattr(r.RunnerClient, "retrieve_output_rasters", rasters)
    run_regenerate(args)
    record = saved(case)
    assert record["evaluation"]["compared_image"]["batch_index"] == 1
    assert record["evaluation"]["compared_image_count"] == 2
