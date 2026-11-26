"""Comparisons too small for an SSIM window are INCONCLUSIVE, never a fabricated SSIM of 0.0 or 1.0."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from comfytrace import regenerate as r
from comfytrace.carve import PngCarver, WorkflowCarver
from comfytrace.cli import parse_args, run_regenerate
from comfytrace.models import ForensicManifest
from comfytrace.report import ManifestWriter
from tests.synthetic import api_prompt, noise_image

pytest.importorskip("skimage")
pytest.importorskip("imagehash")


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _partial_target(rows: int, width: int = 64, height: int = 64, **metadata) -> SimpleNamespace:
    return SimpleNamespace(artifact_id="CARVE-PNG-007", completeness="PARTIAL", pixel_metadata=None,
                           derived_preview_metadata={"decoded_rows": rows, **metadata}, dimensions=(width, height))


def _strip(image: Image.Image, rows: int) -> bytes:
    return _png(image.crop((0, 0, image.width, rows)))


def _changed_top_row(image: Image.Image, pixels: int = 8, delta: int = 40) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    array[0, :pixels] = (array[0, :pixels].astype(int) + delta) % 256
    return Image.fromarray(array, "RGB")


@pytest.mark.parametrize("rows", [1, 2])
def test_an_exact_thin_strip_is_row_slice_exact(rows):
    full = noise_image(64, 64, "RGB", seed=11)
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(
        _png(full), _partial_target(rows), _strip(full, rows))
    assert (tier, subtier) == ("PARTIAL_CORROBORATION", "ROW_SLICE_EXACT")
    assert metrics["row_slice_pixel_match"] is True
    assert metrics["ssim_score"] is None, "no SSIM window fits, so none is reported"


@pytest.mark.parametrize("rows", [1, 2])
def test_an_unequal_thin_strip_is_inconclusive_with_difference_statistics(rows):
    full = noise_image(64, 64, "RGB", seed=12)
    regen = _changed_top_row(full, pixels=8, delta=40)
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(
        _png(regen), _partial_target(rows), _strip(full, rows))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_ROWS")
    assert metrics["ssim_score"] is None
    assert metrics["ssim_unavailable_reason"] == "window_too_small"
    assert metrics["row_slice_pixel_match"] is False
    assert metrics["decoded_rows_evaluated"] == rows
    assert metrics["cropped_dimensions"] == {"width": 64, "height": rows}
    assert isinstance(metrics["phash_hamming_distance"], int)
    # 8 pixels of the top row differ by 40 on R, G and B; alpha never differs.
    expected_mae = 8 * 3 * 40 / (64 * rows * 4)
    assert metrics["mean_abs_diff"] == pytest.approx(expected_mae)
    assert metrics["max_abs_diff"] == 40
    assert metrics["fraction_above_tolerance"] == pytest.approx(8 / (64 * rows))
    assert metrics["difference_tolerance"] == 2


def test_a_phash_of_zero_cannot_make_a_thin_strip_match(monkeypatch):
    full = noise_image(64, 64, "RGB", seed=13)
    regen = _changed_top_row(full, pixels=1, delta=3)
    monkeypatch.setattr(r.TieredEvaluator, "_perceptual", lambda self, a, b: (None, 0))
    tier, subtier, _ = r.TieredEvaluator().evaluate_reproduction(_png(regen), _partial_target(1), _strip(full, 1))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_ROWS")


def test_reduced_sample_precision_rows_are_inconclusive_not_exact():
    full = noise_image(64, 64, "RGB", seed=14)
    target = _partial_target(1, sample_precision_reduced=True)
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(_png(full), target, _strip(full, 1))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_ROWS")
    assert metrics["row_slice_pixel_match"] is False
    assert metrics["mean_abs_diff"] == 0.0 and metrics["ssim_score"] is None


def test_zero_decoded_rows_are_inconclusive_not_a_contradiction():
    full = noise_image(64, 64, "RGB", seed=15)
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(_png(full), _partial_target(0), _strip(full, 1))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_ROWS")
    assert metrics["ssim_unavailable_reason"] == "no_decoded_rows"


@pytest.mark.parametrize("size", [(64, 2), (1, 64), (2, 2)])
def test_a_thin_intact_image_is_inconclusive_unless_exact(size):
    target = noise_image(*size, "RGB", seed=16)
    [record] = PngCarver().carve_stream(_png(target))
    evaluator = r.TieredEvaluator()
    tier, subtier, metrics = evaluator.evaluate_reproduction(_png(target), record, _png(target))
    assert (tier, subtier) == ("E1", "EXACT_MATCH")
    assert metrics["ssim_score"] is None
    other = Image.fromarray(255 - np.asarray(target), "RGB")
    tier, subtier, metrics = evaluator.evaluate_reproduction(_png(other), record, _png(target))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_DIMENSIONS")
    assert metrics["ssim_score"] is None and metrics["ssim_unavailable_reason"] == "window_too_small"
    assert metrics["exact_pixel_match"] is False and metrics["max_abs_diff"] > 2


def test_thin_comparisons_need_no_perceptual_extras(monkeypatch):
    def unavailable():
        raise r.MissingDependencyError("optional extras absent")

    monkeypatch.setattr(r, "_perceptual_backends", unavailable)
    target = noise_image(64, 2, "RGB", seed=17)
    [record] = PngCarver().carve_stream(_png(target))
    other = Image.fromarray(255 - np.asarray(target), "RGB")
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(_png(other), record, _png(target))
    assert (tier, subtier) == ("INCONCLUSIVE", "INSUFFICIENT_DIMENSIONS")
    assert metrics["perceptual_scores"].startswith("UNAVAILABLE")
    with pytest.raises(r.MissingDependencyError):
        big = noise_image(16, 16, "RGB", seed=18)
        [record] = PngCarver().carve_stream(_png(big))
        r.TieredEvaluator().evaluate_reproduction(_png(noise_image(16, 16, "RGB", seed=19)), record, _png(big))


def test_an_animated_png_is_an_assessability_limit():
    frames = [noise_image(16, 16, "RGB", seed=s) for s in (20, 21)]
    out = io.BytesIO()
    frames[0].save(out, format="PNG", save_all=True, append_images=frames[1:])
    [record] = PngCarver().carve_stream(_png(frames[0]))
    tier, subtier, metrics = r.TieredEvaluator().evaluate_reproduction(out.getvalue(), record, _png(frames[0]))
    assert subtier == "ANIMATED_COMPARISON_UNAVAILABLE"
    assert metrics["comparison_unavailable_reason"] == "animated_sequence"
    assert "ssim_score" not in metrics, "nothing was compared, so no score is recorded"


def test_an_inconclusive_input_strip_is_unverified_never_distinct(tmp_path):
    source = noise_image(64, 64, "RGB", seed=22)
    path = tmp_path / "source.png"
    path.write_bytes(_png(_changed_top_row(source, pixels=4, delta=30)))
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    check = r.TieredEvaluator().compare_partial_inputs(
        _partial_target(1), _strip(source, 1), tmp_path, expected, exact_only=False)
    assert check["status"] == "UNVERIFIED"
    assert check["verified_inputs"] == expected
    assert check["inconclusive_inputs"] == [path.name]


def test_inconclusive_evaluations_serialize_and_keep_their_target():
    manifest = {"findings": {"findings": [
        {"finding_id": "FIND-001", "tier": "E3", "artifact_ids": ["CARVE-PNG-007", "WF-001"]}]}}
    metrics = {"evaluation_mode": "PARTIAL_ROW_BOUNDED", "exact_pixel_match": False, "ssim_score": None,
               "phash_hamming_distance": 3, "decoded_rows_evaluated": 1, "row_slice_pixel_match": False,
               "cropped_dimensions": {"width": 64, "height": 1}, "ssim_unavailable_reason": "window_too_small",
               "mean_abs_diff": 1.5, "max_abs_diff": 40, "fraction_above_tolerance": 0.125}
    block = r._evaluation_block(manifest, SimpleNamespace(workflow_id="WF-001"),
                                SimpleNamespace(artifact_id="CARVE-PNG-007"), "INCONCLUSIVE", "INSUFFICIENT_ROWS",
                                metrics, 0.95, 4)
    assert block["target_artifact_id"] == "CARVE-PNG-007"
    assert block["evaluation_mode"] == "PARTIAL_ROW_BOUNDED"
    assert (block["promoted_finding_tier"], block["target_finding_tier_retained"]) == (None, "E3")
    assert block["ssim_score"] is None and block["decoded_rows_evaluated"] == 1
    assert json.loads(json.dumps(block))["mean_abs_diff"] == 1.5


def test_a_pixelless_e5_finding_with_containers_is_still_promoted_by_an_executable_replay():
    manifest = {"findings": {"findings": [
        {"finding_id": "FIND-001", "tier": "E5", "artifact_ids": ["CARVE-PNG-003", "WF-001"]}]}}
    block = r._evaluation_block(manifest, SimpleNamespace(workflow_id="WF-001"), None, "E4",
                                "EXECUTABLE_REPRODUCTION", {"evaluation_mode": "NO_MEMORY_TARGET"}, 0.95, 4)
    assert block["promoted_finding_tier"] == "E4"


def _executability(findings: list[dict]) -> dict:
    return r._evaluation_block({"findings": {"findings": findings}}, SimpleNamespace(workflow_id="WF-001"), None,
                               "E4", "EXECUTABLE_REPRODUCTION", {"evaluation_mode": "NO_MEMORY_TARGET"}, 0.95, 4)


def test_e4_promotion_does_not_depend_on_finding_order():
    holder = {"finding_id": "FIND-001", "tier": "E2", "artifact_ids": ["TENSOR-001", "WF-001"]}
    zero_row = {"finding_id": "FIND-002", "tier": "E5", "artifact_ids": ["CARVE-PNG-003", "WF-001"]}
    forward, backward = _executability([holder, zero_row]), _executability([zero_row, holder])
    for block in (forward, backward):
        assert block["promoted_finding_tier"] == "E4"
        assert block["eligible_e4_finding_ids"] == ["FIND-002"]
        assert block["target_finding_id"] == "FIND-002"
    assert forward == backward


def test_e4_promotion_needs_an_e5_finding_and_names_every_eligible_one():
    holder = {"finding_id": "FIND-001", "tier": "E2", "artifact_ids": ["TENSOR-001", "WF-001"]}
    partial = {"finding_id": "FIND-002", "tier": "E3", "artifact_ids": ["CARVE-PNG-001", "WF-001"]}
    for order in ([holder, partial], [partial, holder]):
        block = _executability(order)
        assert block["promoted_finding_tier"] is None and block["eligible_e4_finding_ids"] == []
        assert block["target_finding_tier_retained"] == "E2" and "target_finding_id" not in block
    first = {"finding_id": "FIND-004", "tier": "E5", "artifact_ids": ["WF-001"]}
    second = {"finding_id": "FIND-003", "tier": "E5", "artifact_ids": ["CARVE-PNG-002", "WF-001"]}
    block = _executability([first, holder, second])
    assert block["promoted_finding_tier"] == "E4"
    assert block["eligible_e4_finding_ids"] == ["FIND-003", "FIND-004"]
    assert "target_finding_id" not in block, "several E5 findings: none is named the target"
    assert _executability([])["eligible_e4_finding_ids"] == []


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content

    def json(self):
        return self.payload


def test_a_thin_intact_replay_is_recorded_inconclusive_with_its_target(tmp_path, monkeypatch):
    case = tmp_path / "case"
    graph = api_prompt()
    [wf] = WorkflowCarver().carve_stream(json.dumps(graph).encode())
    wf.raw_file_path = "exports/raw/WF-001.json"
    (case / "exports/raw").mkdir(parents=True)
    (case / wf.raw_file_path).write_bytes(wf.raw_bytes)
    target = noise_image(64, 2, "RGB", seed=23)
    [artifact] = PngCarver().carve_stream(_png(target))
    artifact.raw_file_paths = ["exports/raw/CARVE-PNG-001.png"]
    (case / artifact.raw_file_paths[0]).write_bytes(_png(target))
    ManifestWriter(case).write_manifest(ForensicManifest("1.0.0", {}, [], {
        "findings": [{"finding_id": "FIND-001", "tier": "E2", "artifact_ids": [artifact.artifact_id, wf.workflow_id],
                      "link_ids": [], "rationale": "fixture"}],
        "artifacts": [artifact.to_dict()], "workflows": [wf.to_dict()], "correlation_links": [],
    }))
    regen = _png(Image.fromarray(255 - np.asarray(target), "RGB"))

    def request(self, method, path, **kwargs):
        if path == "/object_info":
            return _Response({node["class_type"]: {"input": {"required": {"model": [[
                "v1-5-pruned-emaonly.safetensors"]]}}} for node in graph.values()})
        if path == "/system_stats":
            return _Response({})
        if path == "/prompt":
            return _Response({"prompt_id": "p"})
        if path == "/history/p":
            return _Response({"p": {"status": {"completed": True}, "outputs": {"9": {"images": [
                {"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
        if path == "/view":
            return _Response(content=regen)
        raise AssertionError(path)

    monkeypatch.setattr(r.RunnerClient, "_request", request)
    args = parse_args(["regenerate", "--case-dir", str(case), "--workflow", str(case / wf.raw_file_path),
                       "--runner-url", "http://stub.invalid", "--compare-to", str(case / artifact.raw_file_paths[0])])
    args.target_selection = {"selected_artifact_id": artifact.artifact_id, "selection_reason": "COMPLETE_NATIVE",
                             "candidate_artifact_ids": [artifact.artifact_id], "excluded_candidates": [],
                             "finding_ids": ["FIND-001"], "path": artifact.raw_file_paths[0]}
    assert run_regenerate(args) == 2
    record = json.loads((case / "regenerations/REGEN-001.json").read_text())
    evaluation = record["evaluation"]
    assert record["status"] == "COMPLETED"
    assert (evaluation["evaluated_tier"], evaluation["evaluation_subtier"]) == ("INCONCLUSIVE",
                                                                               "INSUFFICIENT_DIMENSIONS")
    assert evaluation["target_artifact_id"] == artifact.artifact_id
    assert evaluation["evaluation_mode"] == "INTACT_NATIVE_RESOLUTION"
    assert evaluation["ssim_score"] is None and evaluation["promoted_finding_tier"] is None
    assert evaluation["target_finding_tier_retained"] == "E2"
    assert evaluation["max_abs_diff"] > 2
    assert evaluation["target_selection"]["selection_reason"] == "COMPLETE_NATIVE"
    assert record["derived_regenerated_image"] == "exports/derived/DERIVED_regen_INCONCLUSIVE_WF-001.png"
