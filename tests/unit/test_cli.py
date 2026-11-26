import subprocess
import sysconfig
from pathlib import Path

import pytest

from comfytrace.cli import extract_exit_code, main, parse_args, run_extract
from comfytrace.models import empty_tier_summary


def test_cli_arg_parsing():
    args = parse_args(["extract", "--memory", "m.raw", "--symbols-dir", "/symbols", "--outdir", "case01"])
    assert (args.command, args.memory, args.symbols_dir, args.outdir) == (
        "extract", Path("m.raw"), Path("/symbols"), Path("case01"))
    assert args.allow_symbol_download is False
    assert args.content_stitch_png is False
    assert parse_args([
        "extract", "--memory", "m.raw", "--symbols-dir", "/symbols", "--outdir", "case01",
        "--content-stitch-png",
    ]).content_stitch_png is True

    reg = parse_args([
        "regenerate", "--case-dir", "case01", "--workflow", "case01/exports/raw/WF-001.json",
        "--runner-url", "http://127.0.0.1:8188",
        "--allow-node", "IPAdapterApply", "--allow-node", "ControlNetLoaderAdvanced",
    ])
    assert reg.allow_node == ["IPAdapterApply", "ControlNetLoaderAdvanced"]
    assert (reg.ssim_threshold, reg.phash_threshold) == (0.95, 4)
    assert reg.allow_pickle_models is False

    rep = parse_args(["report", "--case-dir", "case01"])
    assert rep.case_dir == Path("case01")
    with pytest.raises(SystemExit):
        parse_args(["extract", "--memory", "m.raw"])


def _summary(**counts):
    summary = empty_tier_summary()
    summary.update(counts)
    return summary


def test_cli_exit_code_precedence_contract():
    assert extract_exit_code(False, _summary(E2=1, E5=2), 3) == 0
    assert extract_exit_code(False, _summary(), 0) == 4
    assert extract_exit_code(False, _summary(E2=1, E6=1), 2) == 2
    assert extract_exit_code(False, _summary(untiered_orphan_images=1), 1) == 2
    assert extract_exit_code(False, _summary(untiered_ambiguous_images=1, E5=2), 3) == 2
    assert extract_exit_code(False, _summary(E3=1), 2) == 2
    assert extract_exit_code(True, _summary(E3=1), 2) == 3, "degraded outranks partial"
    assert extract_exit_code(True, _summary(), 0) == 3, "degraded outranks zero artifacts"


def test_main_maps_failures_to_exit_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "--memory", str(tmp_path / "missing.raw"), "--symbols-dir", str(tmp_path),
              "--outdir", str(tmp_path / "c")])
    assert exc.value.code == 1


def test_usage_error_exits_1_not_partial_findings_code(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "--memory", str(tmp_path / "m.raw"), "--outdir", str(tmp_path / "c")])
    assert exc.value.code == 1


def test_console_entry_point_runs():
    name = "comfytrace.exe" if sysconfig.get_platform().startswith("win") else "comfytrace"
    exe = Path(sysconfig.get_path("scripts")) / name
    out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "comfytrace 1.0.0"


def test_extraction_records_memory_only_correlations_independently(
    tmp_path, mock_memory_image, fake_ingestor, symbols_dir, monkeypatch
):
    import json

    from comfytrace.report import HtmlReporter

    monkeypatch.setattr(HtmlReporter, "render_report", lambda *_: None)
    case = tmp_path / "case"
    args = parse_args([
        "extract", "--memory", str(mock_memory_image("T1")), "--symbols-dir", str(symbols_dir),
        "--outdir", str(case),
    ])
    assert run_extract(args, ingestor=fake_ingestor) == 0
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    proof = json.loads((case / "recovery-evidence.json").read_text(encoding="utf-8"))
    assert manifest["findings"]["correlation_links"]
    assert proof["memory_only_correlation_links"] == []


@pytest.mark.parametrize("state", ["T1", "T3"])
def test_extraction_records_digests_of_exported_bytes(tmp_path, mock_memory_image, fake_ingestor, symbols_dir, state):
    import hashlib
    import json

    case = tmp_path / "case"
    args = parse_args([
        "extract", "--memory", str(mock_memory_image(state)), "--symbols-dir", str(symbols_dir), "--outdir", str(case),
    ])
    assert run_extract(args, ingestor=fake_ingestor) == (0 if state == "T1" else 3)
    findings = json.loads((case / "manifest.json").read_bytes())["findings"]
    proof = json.loads((case / "recovery-evidence.json").read_bytes())
    digests = {r["record_id"]: r["sha256"]["raw"] for r in proof["records"] if r["kind"] in ("image", "workflow")}
    for image in findings["artifacts"]:
        data = b"".join((case / path).read_bytes() for path in image["raw_file_paths"])
        assert digests[image["artifact_id"]] == hashlib.sha256(data).hexdigest()
    for workflow in findings["workflows"]:
        data = (case / workflow["raw_file_path"]).read_bytes()
        assert digests[workflow["workflow_id"]] == hashlib.sha256(data).hexdigest()


def test_physical_pass_skips_pages_owned_by_carved_processes():
    import json

    from comfytrace.carve import Block, PhysicalSource
    from comfytrace.cli import _carve
    from comfytrace.ingest import MODE_VIRTUAL, AddressSpace, IngestionResult
    from tests.synthetic import PAGE, api_prompt, noise_image, png_bytes

    owned_png = png_bytes(noise_image(8, 8, seed=1), {})
    free_png = png_bytes(noise_image(8, 8, seed=2), {})
    owned_wf = json.dumps(api_prompt()).encode()
    memory = bytearray(4 * PAGE)
    memory[16:16 + len(owned_png)] = owned_png
    memory[PAGE + 16:PAGE + 16 + len(owned_wf)] = owned_wf
    memory[2 * PAGE + 16:2 * PAGE + 16 + len(free_png)] = free_png
    space = AddressSpace(None, [Block(0, bytes(memory))], PhysicalSource(bytes(memory)), skip_pages=frozenset({0, 1}))
    images, workflows, _ = _carve(IngestionResult(MODE_VIRTUAL, {}, [], [space]))
    assert [i.physical_offsets for i in images] == [[f"0x{2 * PAGE + 16:016X}"]]
    assert workflows == []


def test_top_level_help_lists_only_extract_and_model_id(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "{extract,model-id,env-id}" in help_text
    for hidden in ("regenerate", "replay-case", "decode-latents", "search-replay"):
        assert hidden not in help_text
    # Hidden subcommands remain callable for scripting and tests.
    assert parse_args(["report", "--case-dir", "c"]).command == "report"
    mid = parse_args(["model-id", "--case-dir", "c", "--memory", "m", "--model", "x.safetensors"])
    assert mid.command == "model-id"


def test_extract_with_model_file_or_dir_triggers_auto_replay(tmp_path, mock_memory_image, fake_ingestor, symbols_dir,
                                                             monkeypatch):
    from comfytrace import replaycase

    ckpt = tmp_path / "models" / "v1-5.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")

    seen = []

    def fake_replay(rargs, ingestor=None):
        seen.append((rargs.models_dir, rargs.model_id, rargs.latents))
        return 0

    monkeypatch.setattr(replaycase, "run_replay_case", fake_replay)

    case_file = tmp_path / "case_file"
    args = parse_args(["extract", "--memory", str(mock_memory_image("T1")), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case_file), "--model", str(ckpt)])
    assert run_extract(args, ingestor=fake_ingestor) == 0
    assert seen == [(ckpt.parent, True, True)]

    seen.clear()
    case_dir = tmp_path / "case_dir"
    args = parse_args(["extract", "--memory", str(mock_memory_image("T3")), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case_dir), "--model", str(ckpt.parent)])
    assert run_extract(args, ingestor=fake_ingestor) == 3
    # T3 has no complete API workflow, so auto-replay is skipped cleanly.
    assert seen == []
