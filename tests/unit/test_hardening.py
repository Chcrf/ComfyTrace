"""Regressions for defects found in the independent review of dc2455f."""

import io
import json
import time
import tracemalloc
import zlib

import pytest
from PIL import Image

from comfytrace import regenerate
from comfytrace.carve import PngCarver, WorkflowCarver, graph_dependencies
from comfytrace.cli import parse_args, run_extract, run_regenerate
from comfytrace.correlate import correlate
from comfytrace.regenerate import TieredEvaluator
from tests.synthetic import PAGE, DumpBuilder, api_prompt, garbage, noise_image, png_bytes, prompt_request_body


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_unrecognized_wrappers_do_not_hide_inner_graphs():
    graph = api_prompt()
    history = json.dumps({"run-id": {"prompt": [0, "run-id", graph, {}, []]}}).encode()
    [record] = WorkflowCarver().carve_stream(history)
    assert record.graph == graph and record.execution_parameters["prompt_id"] == "run-id"
    [inner] = WorkflowCarver().carve_stream(json.dumps({"payload": graph}).encode())
    assert inner.graph == graph


def test_identical_graphs_keep_every_execution_prompt_id():
    graph = api_prompt()
    blob = prompt_request_body(graph, "first") + b"\x00" + prompt_request_body(graph, "second")
    workflows = WorkflowCarver().carve_stream(blob)
    assert [workflow.execution_parameters["prompt_id"] for workflow in workflows] == ["first", "second"]
    assert [workflow.raw_bytes for workflow in workflows] == blob.split(b"\x00")
    [image] = PngCarver().carve_stream(png_bytes(noise_image(64, 48), {"prompt_id": "second"}))
    result = correlate([image], workflows)
    assert [link.link_class for link in result.links] == ["STRICT"]
    assert result.links[0].workflow_id == workflows[1].workflow_id


def test_truncated_single_idat_keeps_readable_bytes_and_rows():
    png = _png(noise_image(64, 64, seed=3))
    cut = len(png) // 2
    [record] = PngCarver().carve_stream(png[:cut])
    assert record.recovered_bytes == cut
    assert record.spans == [(0, png[:cut])]
    assert record.gaps[0].relative_offset == cut
    assert record.derived_preview_metadata["decoded_rows"] > 0


def test_partial_inflation_is_bounded_by_ihdr():
    meta = {"width": 1, "height": 1, "bit_depth": 8, "color_type": 0, "interlace": 0}
    payload = zlib.compress(b"\x00" * 50_000_000)
    tracemalloc.start()
    pixels, rows, halt = PngCarver().decompress_partial_idat(payload, meta)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert (pixels, rows, halt) == (b"\x00", 1, None)
    assert peak < 5_000_000


def test_compressed_text_chunk_is_bounded():
    from comfytrace.carve import _decode_text_chunk

    bomb = b"prompt\x00\x00" + zlib.compress(b"A" * (20 * 1024 * 1024))
    assert _decode_text_chunk(b"zTXt", bomb) is None


def test_huge_integer_candidate_does_not_abort_carving():
    graph = api_prompt()
    blob = b'{"x": ' + b"9" * 5000 + b', "p": ' + json.dumps(graph).encode() + b"}"
    [record] = WorkflowCarver().carve_stream(blob)
    assert record.graph == graph


def test_lookback_is_measured_in_bytes():
    graph = api_prompt()
    wrapper = json.dumps({"note": "é" * 600_000, "prompt": graph, "prompt_id": "far"}, ensure_ascii=False)
    [record] = WorkflowCarver().carve_stream(wrapper.encode())
    assert record.graph == graph
    assert "prompt_id" not in record.execution_parameters, "wrapper starts more than 1 MB (bytes) back"


def test_many_control_separated_candidates_scale_linearly():
    unit = json.dumps({"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}}).encode() + b"\x00"

    def cost(n):
        start = time.perf_counter()
        WorkflowCarver().carve_stream(unit * n)
        return time.perf_counter() - start

    small, large = cost(2000), cost(16000)
    assert large < small * 8 * 2.5, f"8x input took {large / small:.1f}x time"


def test_tear_reason_uses_absolute_dump_offsets():
    png = bytearray(png_bytes(noise_image(40, 40, seed=4), idat_chunk=1036))
    idat = png.index(b"IDAT") - 4
    png[idat + 50] ^= 0xFF
    origin = PAGE - idat - 100
    [record] = PngCarver(physical_mode=True).carve_stream(garbage(origin) + bytes(png))
    assert record.gaps[0].reason == "NONCONTIGUOUS_PHYSICAL"
    [inside] = PngCarver(physical_mode=True).carve_stream(bytes(png))
    assert inside.gaps[0].reason == "CRC_FAILURE"


def test_dual_clip_loader_models_are_subject_to_format_policy():
    graph = api_prompt()
    graph["11"] = {"class_type": "DualCLIPLoader",
                   "inputs": {"clip_name1": "evil.pt", "clip_name2": "other.ckpt", "type": "flux"}}
    graph["12"] = {"class_type": "UNETLoader", "inputs": {"weights": "hidden.pth"}}
    assets = graph_dependencies(graph)["model_assets"]
    assert {"evil.pt", "other.ckpt", "hidden.pth"} <= set(assets)


def test_palette_images_with_different_colours_are_not_exact_matches():
    pngs = []
    for colour in ((255, 0, 0), (0, 0, 255)):
        img = Image.new("P", (8, 8), 0)
        img.putpalette(list(colour) + [0] * 765)
        pngs.append(_png(img))
    [red] = PngCarver().carve_stream(pngs[0])
    tier, subtier, metrics = TieredEvaluator().evaluate_reproduction(pngs[1], red, pngs[0])
    assert tier == "NON_MATCHING"
    assert metrics["native_pixel_sha256_match"] is False and metrics["exact_pixel_match"] is False


@pytest.fixture
def partial_case(tmp_path, symbols_dir):
    full = noise_image(64, 64, seed=21)
    graph = api_prompt(width=64, height=64)
    png = png_bytes(full, {"prompt": json.dumps(graph)}, idat_chunk=600)
    dump = DumpBuilder(frames=64)
    dump.put(10, png[: len(png) // 2])
    dump.put(30, json.dumps(graph).encode())
    memory = dump.write(tmp_path / "partial.raw")
    case = tmp_path / "case_partial"
    args = parse_args(["extract", "--memory", str(memory), "--symbols-dir", str(symbols_dir), "--outdir", str(case)])
    assert run_extract(args) == 3
    return case, full


def test_partial_target_rows_come_from_raw_spans_not_the_preview_file(partial_case, mock_runner_server):
    case, full = partial_case
    url, state = mock_runner_server
    state.image = _png(full)
    state.object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0].append("x")
    manifest = json.loads((case / "manifest.json").read_text())
    [artifact] = manifest["findings"]["artifacts"]
    assert artifact["completeness"] == "PARTIAL" and artifact["link_status"] == "STRICT"
    preview = case / artifact["derived_preview_path"]
    preview.chmod(0o644)
    preview.write_bytes(_png(Image.new("RGB", (64, artifact["derived_preview_metadata"]["decoded_rows"]), "red")))

    argv = ["regenerate", "--case-dir", str(case), "--workflow", str(case / "exports/raw/WF-001.json"),
            "--runner-url", url,
            "--compare-to", str(case / artifact["raw_file_paths"][0])]
    assert run_regenerate(parse_args(argv)) == 2
    record = json.loads((case / "regenerations/REGEN-001.json").read_text())
    assert record["evaluation"]["evaluated_tier"] == "PARTIAL_CORROBORATION"
    assert record["evaluation"]["target_finding_tier_retained"] == "E3"

    span = case / artifact["raw_file_paths"][0]
    span.chmod(0o644)
    span.write_bytes(span.read_bytes()[:-10])
    assert run_regenerate(parse_args(argv)) == 1


def test_input_uploads_reject_runner_renames(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "source.png").write_bytes(_png(noise_image(8, 8)))
    sent = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"name": "source (1).png", "subfolder": "", "type": "input"}

    def fake_request(self, method, path, **kwargs):
        sent.update(kwargs.get("data", {}), file=kwargs["files"]["image"][0])
        return Response()

    monkeypatch.setattr(regenerate.RunnerClient, "_request", fake_request)
    with pytest.raises(regenerate.PromptSubmissionError):
        regenerate.RunnerClient("http://stub.invalid").upload_input(inputs / "source.png", "source.png")
    assert sent == {"overwrite": "true", "subfolder": "", "type": "input", "file": "source.png"}


def test_failed_runner_execution_still_writes_an_audit_record(tmp_path, mock_memory_image, fake_ingestor,
                                                             symbols_dir, mock_runner_server, monkeypatch):
    url, _ = mock_runner_server
    case = tmp_path / "case01"
    run_extract(parse_args(["extract", "--memory", str(mock_memory_image("T1")), "--symbols-dir", str(symbols_dir),
                            "--outdir", str(case)]), ingestor=fake_ingestor)

    def boom(self, prompt_id, nodes=None):
        raise regenerate.ExecutionTimeoutError("no output within 1s")

    monkeypatch.setattr(regenerate.RunnerClient, "retrieve_output_rasters", boom)
    argv = ["regenerate", "--case-dir", str(case), "--workflow", str(case / "exports/raw/WF-001.json"),
            "--runner-url", url]
    assert run_regenerate(parse_args(argv)) == 1
    record = json.loads((case / "regenerations/REGEN-001.json").read_text())
    assert record["status"] == "FAILED" and "ExecutionTimeoutError" in record["evaluation"]["error"]
