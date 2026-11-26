"""replay-case runs the post-extraction steps in data order, once each, and skips outputs that already exist.

Every stage is stubbed: this checks the orchestration only (no runner, no memory search, no ingestion)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ahocorasick_rs")

from comfytrace import latentdecode, latents, link, modelid, replaycase  # noqa: E402
from comfytrace.cli import parse_args  # noqa: E402


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    memory = tmp_path / "dump.raw"
    memory.write_bytes(b"memory image")
    case_dir = tmp_path / "case"
    (case_dir / "exports/raw").mkdir(parents=True)
    (case_dir / "exports/raw/WFPY-001.json").write_text("{}")
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_metadata": {"memory_image": {"sha256": hashlib.sha256(b"memory image").hexdigest()}},
        "findings": {"artifacts": [], "findings": [], "workflows": [{
            "workflow_id": "WFPY-001", "encoding_detected": "CPYTHON_OBJECTS", "prompt_format": "API_GRAPH",
            "completeness": "COMPLETE", "canonical_json_sha256": "a" * 64, "raw_file_path": "exports/raw/WFPY-001.json",
            "execution_parameters": {"prompt_id": "job-a"}}]}}))
    calls: list[str] = []

    def regenerate(args):
        calls.append(f"regenerate:{args.workflow_id}")
        (case_dir / "regenerations").mkdir(exist_ok=True)
        (case_dir / "regenerations/REGEN-001.json").write_text(json.dumps({"status": "COMPLETED"}))
        return 0

    def identity(case_dir_, processes, models_dir, memory_sha):
        calls.append("model-id")
        return {"models": []}

    def write_identity(case_dir_, document):
        (case_dir_ / modelid.IDENTITY_NAME).write_text("{}")

    found: list[dict] = [{"latent_id": "LAT-001"}]

    def write_latents(case_dir_, document):
        calls.append("latents")
        (case_dir_ / latents.LATENTS_NAME).write_text(json.dumps({"latents": found}))

    def decode(args):
        calls.append("decode-latents")
        (args.case_dir / latentdecode.DECODES_DIR).mkdir(exist_ok=True)
        (args.case_dir / latentdecode.DECODES_DIR / latentdecode.COMPLETE_NAME).write_text("{}")
        return 0

    def write_links(case_dir_):
        calls.append("link")
        (case_dir_ / link.LINKS_NAME).write_text("{}")

    monkeypatch.setattr(replaycase, "resolve_runner", lambda args: "http://runner")
    monkeypatch.setattr(replaycase, "run_regeneration", regenerate)
    monkeypatch.setattr(replaycase, "search", lambda *a, **k: calls.append("search-replay"))
    monkeypatch.setattr(replaycase, "HtmlReporter", lambda d: SimpleNamespace(
        render_report=lambda path: calls.append("report") or d / "workbench.html"))
    monkeypatch.setattr(modelid, "identity_document", identity)
    monkeypatch.setattr(modelid, "write_identity", write_identity)
    monkeypatch.setattr(latents, "latent_sides", lambda graphs: {})
    monkeypatch.setattr(latents, "find", lambda mem, sides: [])
    monkeypatch.setattr(latents, "document", lambda *a: {})
    monkeypatch.setattr(latents, "write", write_latents)
    monkeypatch.setattr(latentdecode, "run_decode", decode)
    monkeypatch.setattr(link, "write_links", write_links)

    def ingestor(memory_path, symbols):
        calls.append("ingest")
        return SimpleNamespace(recovery=SimpleNamespace(python_processes=[]), close=lambda: None)

    def run(*extra: str) -> int:
        args = parse_args(["replay-case", "--memory", str(memory), "--case-dir", str(case_dir),
                           "--runner-url", "http://runner", *extra])
        return replaycase.run_replay_case(args, ingestor)

    return SimpleNamespace(run=run, calls=calls, dir=case_dir, tmp=tmp_path, found=found)


@pytest.mark.parametrize("status, code", [("SECURITY_REJECTED", 0), ("MISSING_DEPENDENCY", 0), ("FAILED", 1)])
def test_a_preflight_refusal_is_not_a_failed_replay(case, monkeypatch, status, code):
    def refuse(args):
        (case.dir / "regenerations").mkdir(exist_ok=True)
        (case.dir / "regenerations/REGEN-001.json").write_text(json.dumps({"status": status}))
        return 1

    monkeypatch.setattr(replaycase, "run_regeneration", refuse)
    assert case.run() == code
    assert "search-replay" not in case.calls, "nothing completed, so there is nothing to search for"


def test_no_retained_latents_means_nothing_to_decode(case):
    case.found.clear()
    assert case.run("--latents", "--symbols-dir", str(case.tmp), "--models-dir", str(case.tmp)) == 0
    assert case.calls == ["ingest", "latents", "regenerate:WFPY-001", "search-replay", "link", "report"]


def test_the_full_pipeline_runs_in_data_order_with_one_ingestion(case):
    assert case.run("--model-id", "--latents", "--symbols-dir", str(case.tmp), "--models-dir", str(case.tmp)) == 0
    assert case.calls == ["ingest", "model-id", "latents", "regenerate:WFPY-001", "decode-latents",
                          "search-replay", "link", "report"]


def test_a_rerun_keeps_every_write_once_output_and_only_replays(case):
    options = ("--model-id", "--latents", "--symbols-dir", str(case.tmp), "--models-dir", str(case.tmp))
    assert case.run(*options) == 0
    case.calls.clear()
    assert case.run(*options) == 0
    assert case.calls == ["regenerate:WFPY-001", "report"], "nothing already written is redone or overwritten"


@pytest.mark.parametrize("partial", [False, True])
def test_an_interrupted_decode_directory_does_not_skip_decoding(case, partial):
    options = ("--model-id", "--latents", "--symbols-dir", str(case.tmp), "--models-dir", str(case.tmp))
    assert case.run(*options) == 0
    marker = case.dir / latentdecode.DECODES_DIR / latentdecode.COMPLETE_NAME
    marker.unlink()  # as if the run had stopped before its last pair
    if partial:
        (case.dir / latentdecode.DECODES_DIR / "LATDEC-001.json").write_text(json.dumps({"status": "COMPLETED"}))
    case.calls.clear()
    assert case.run(*options) == 0
    assert "decode-latents" in case.calls, "a directory without its completion marker is resumed, not kept"
    case.calls.clear()
    assert case.run(*options) == 0
    assert "decode-latents" not in case.calls


def test_without_process_options_no_memory_is_ingested(case):
    assert case.run() == 0
    assert case.calls == ["regenerate:WFPY-001", "search-replay", "link", "report"]


@pytest.mark.parametrize("rows, target", [(4, "exports/derived/CARVE-PNG-002.png"), (0, None)])
def test_the_globally_selected_target_and_its_reason_reach_the_replay(case, monkeypatch, rows, target):
    manifest = json.loads((case.dir / "manifest.json").read_text())

    def partial(aid, decoded):
        return {"artifact_id": aid, "carve_type": "PARTIAL_PNG", "completeness": "PARTIAL", "is_derived": False,
                "raw_file_paths": [f"exports/raw/{aid}/span_000.bin"],
                "derived_preview_path": f"exports/derived/{aid}.png",
                "derived_preview_metadata": {"decoded_rows": decoded}, "pixel_metadata": {"width": 64, "height": 64}}

    manifest["findings"]["artifacts"] = [partial("CARVE-PNG-001", min(rows, 1)), partial("CARVE-PNG-002", rows)]
    manifest["findings"]["findings"] = [
        {"finding_id": "FIND-001", "tier": "E3", "artifact_ids": ["CARVE-PNG-002", "WFPY-001"], "link_ids": []},
        {"finding_id": "FIND-002", "tier": "E3", "artifact_ids": ["CARVE-PNG-001", "WFPY-001"], "link_ids": []}]
    (case.dir / "manifest.json").write_text(json.dumps(manifest))
    seen = []

    def regenerate(args):
        seen.append((args.compare_to, args.target_selection))
        (case.dir / "regenerations").mkdir(exist_ok=True)
        (case.dir / "regenerations/REGEN-001.json").write_text(json.dumps({"status": "COMPLETED"}))
        return 0

    monkeypatch.setattr(replaycase, "run_regeneration", regenerate)
    assert case.run() == 0
    [(compare_to, selection)] = seen
    assert compare_to == (case.dir / target if target else None)
    assert selection["path"] == target
    assert selection["selection_reason"] == ("PARTIAL_MOST_DECODED_ROWS" if target else "NO_COMPARABLE_TARGET")


@pytest.mark.parametrize("extra", [("--model-id",), ("--latents", "--symbols-dir", ".")])
def test_process_evidence_needs_symbols_and_models(case, extra):
    with pytest.raises(ValueError, match="--symbols-dir|--models-dir"):
        case.run(*extra)
    assert case.calls == []
