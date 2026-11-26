import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from comfytrace.carve import canonical_json
from comfytrace.cli import parse_args, run_extract
from tests.evaluation.paths import (
    ENV_CONFIG,
    ENV_DUMPS_DIR,
    ENV_GROUNDTRUTH_DIR,
    ENV_SYMBOLS_DIR,
    case_env_token,
    load_groundtruth_cases,
    resolve_case_paths,
)
from tests.evaluation.scorer import score_case
from tests.synthetic import api_prompt, noise_image

GROUNDTRUTH = Path(__file__).parent.parent / "fixtures" / "groundtruth_cases.json"


def extract(memory: Path, outdir: Path, symbols: Path, ingestor) -> tuple[int, dict]:
    args = parse_args(["extract", "--memory", str(memory), "--symbols-dir", str(symbols), "--outdir", str(outdir)])
    code = run_extract(args, ingestor=ingestor)
    manifest = json.loads((outdir / "manifest.json").read_text()) if (outdir / "manifest.json").exists() else {}
    return code, manifest


def assert_no_padding_and_hashes(case: Path, manifest: dict) -> None:
    """Chain-of-custody invariant 1: every raw export equals genuine dump bytes and carries its gaps in metadata."""
    evidence = json.loads((case / "recovery-evidence.json").read_bytes())
    hashes = {record["record_id"]: record["sha256"]["raw"] for record in evidence["records"]
              if record["kind"] in ("image", "workflow")}
    for artifact in manifest["findings"]["artifacts"]:
        assert all(g["synthetic_padding_applied"] is False for g in artifact["gaps"])
        written = b"".join((case / p).read_bytes() for p in artifact["raw_file_paths"])
        digest = hashlib.sha256(written).hexdigest()
        assert digest == hashes[artifact["artifact_id"]], artifact["artifact_id"]
        meta = artifact.get("derived_preview_metadata") or {}
        container_digest = digest
        if "raw_recovered_bytes" in meta:
            assert artifact["is_derived"] and len(written) == meta["raw_recovered_bytes"]
            assert meta["raw_span_sha256"] == [
                hashlib.sha256((case / p).read_bytes()).hexdigest() for p in artifact["raw_file_paths"]
            ]
            rebuilt = (case / artifact["derived_preview_path"]).read_bytes()
            assert len(rebuilt) == artifact["recovered_bytes"]
            container_digest = hashlib.sha256(rebuilt).hexdigest()
            assert container_digest == meta["sha256"]
        else:
            assert len(written) == artifact["recovered_bytes"]
        if artifact["completeness"] == "COMPLETE" and artifact["container_sha256"] is not None:
            assert container_digest == artifact["container_sha256"], artifact["artifact_id"]
        if preview := artifact.get("derived_preview_path"):
            assert hashlib.sha256((case / preview).read_bytes()).hexdigest() == (
                artifact["derived_preview_metadata"]["preview_sha256"]
            )
    for workflow in manifest["findings"]["workflows"]:
        written = (case / workflow["raw_file_path"]).read_bytes()
        assert hashlib.sha256(written).hexdigest() == hashes[workflow["workflow_id"]], workflow["workflow_id"]
    for path in (case / "exports" / "derived").glob("*"):
        assert path.name.startswith("DERIVED_")


def test_t1_active_memory_e2_finding_recovery(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    memory = mock_memory_image("T1")
    case = tmp_path / "case01"
    code, manifest = extract(memory, case, symbols_dir, fake_ingestor)
    assert code == 0
    body = manifest["findings"]
    assert body["tier_summary"]["E2"] == 1 and body["tier_summary"]["E5"] == 0
    [finding] = body["findings"]
    assert (finding["finding_id"], finding["tier"]) == ("FIND-001", "E2")
    # The job's finding also holds the prompt and UI workflow the PNG carries in its own text chunks.
    assert finding["artifact_ids"] == ["CARVE-PNG-001", "WF-001", "WFPNG-001", "WFPNG-002"]
    [link] = body["correlation_links"]
    assert link["link_class"] == "STRICT" and "CANONICAL_PROMPT_HASH" in link["matched_keys"]
    [image] = body["artifacts"]
    assert image["raw_file_paths"] == ["exports/raw/CARVE-PNG-001.png"]
    assert image["process_id"] == 13484 and image["virtual_address"] == "0x00000213BC000000"
    assert image["physical_offsets"], "virtual address translated to a physical offset"
    workflow, *embedded = body["workflows"]
    assert [w["encoding_detected"] for w in embedded] == ["PNG_TEXT", "PNG_TEXT"]
    assert workflow["raw_file_path"] == "exports/raw/WF-001.json"
    assert workflow["execution_parameters"]["prompt_id"] == "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"
    roles = {p["pid"]: p["role"] for p in manifest["process_context"]}
    assert roles == {19556: "ROOT_LAUNCHER", 13484: "WORKER_CHILD"}
    assert manifest["case_metadata"]["memory_image"]["sha256"] == hashlib.sha256(memory.read_bytes()).hexdigest()
    assert manifest["case_metadata"]["exit_code"] == 0
    assert_no_padding_and_hashes(case, manifest)
    metadata = json.loads((case / "workflow-metadata.json").read_text())
    [sampler] = next(r for r in metadata["workflows"] if r["workflow_id"] == "WF-001")["samplers"]
    assert sampler["fields"]["seed"]["value"] == workflow["execution_parameters"]["seed"]
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    assert "FIND-001" in wb and not (case / "report.html").exists()


def test_t2_closed_residual_heap_finding_recovery(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = tmp_path / "case02"
    code, manifest = extract(mock_memory_image("T2"), case, symbols_dir, fake_ingestor)
    assert code == 2
    tiers = {tuple(f["artifact_ids"]): f["tier"] for f in manifest["findings"]["findings"]}
    assert tiers == {("CARVE-PNG-001",): "UNTIERED", ("WF-001",): "E6"}
    [image] = manifest["findings"]["artifacts"]
    assert image["link_status"] == "UNASSOCIATED" and image["completeness"] == "COMPLETE"
    assert manifest["findings"]["tier_summary"]["untiered_orphan_images"] == 1
    assert_no_padding_and_hashes(case, manifest)


def test_omit_unrelated_exports_keeps_their_hashes_and_linked_bytes(tmp_path, symbols_dir):
    import comfytrace.ingest as ingest
    from tests.synthetic import DumpBuilder, png_bytes

    linked = png_bytes(noise_image(64, 48), {"prompt": json.dumps(api_prompt())})
    unrelated = png_bytes(noise_image(31, 29))
    dump = DumpBuilder(frames=20)
    dump.put(3, linked)
    dump.put(12, unrelated)
    memory = dump.write(tmp_path / "mixed.raw")
    case = tmp_path / "scoped"
    args = parse_args(["extract", "--memory", str(memory), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case), "--omit-unrelated-exports"])
    assert run_extract(args, ingestor=ingest.ingest_memory) == 3
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_metadata"]["unrelated_image_exports_omitted"] == 1
    by_sha256 = {image["container_sha256"]: image for image in manifest["findings"]["artifacts"]}
    [path] = by_sha256[hashlib.sha256(linked).hexdigest()]["raw_file_paths"]
    assert (case / path).read_bytes() == linked
    assert by_sha256[hashlib.sha256(unrelated).hexdigest()]["raw_file_paths"] == []


def test_t3_terminated_physical_carving_fallback(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = tmp_path / "case03"
    code, manifest = extract(mock_memory_image("T3"), case, symbols_dir, fake_ingestor)
    assert code == 3
    meta = manifest["case_metadata"]
    assert meta["carving_mode"] == "PHYSICAL_FALLBACK"
    assert meta["symbol_resolution"]["status"] == "FAILED"
    assert manifest["process_context"] == []
    [finding] = manifest["findings"]["findings"]
    assert finding["tier"] == "UNTIERED"
    [image] = manifest["findings"]["artifacts"]
    assert image["process_id"] is None and image["source_layer"] == "physical_UNATTRIBUTED"
    assert image["completeness"] == "PARTIAL" and image["link_status"] == "UNASSOCIATED"
    assert image["gaps"][0]["reason"] in ("NONCONTIGUOUS_PHYSICAL", "CRC_FAILURE")
    assert image["raw_file_paths"] == ["exports/raw/CARVE-PNG-001/span_000.bin"]
    assert image["derived_preview_path"] == "exports/derived/DERIVED_preview_CARVE-PNG-001.png"
    assert 0 < image["derived_preview_metadata"]["decoded_rows"] < image["derived_preview_metadata"]["total_rows"]
    assert_no_padding_and_hashes(case, manifest)
    wb = (case / "workbench.html").read_text(encoding="utf-8")
    assert "PHYSICAL_FALLBACK" in wb and not (case / "report.html").exists()


def test_t3_resolved_kernel_without_comfyui_leaves_physical_evidence_unattributed(
    tmp_path, mock_memory_image, symbols_dir, mock_volatility_context, monkeypatch,
):
    from comfytrace import ingest

    def resolve_kernel(engine):
        engine.kernel_module_name = "kernel"
        return mock_volatility_context

    monkeypatch.setattr(ingest.VolatilityEngine, "initialize_context", resolve_kernel)
    monkeypatch.setattr(ingest.ProcessWalker, "enumerate_comfyui_hierarchy", lambda self: [])
    case = tmp_path / "terminated"
    code, manifest = extract(mock_memory_image("T3"), case, symbols_dir, ingest.ingest_memory)

    assert code == 3
    metadata = manifest["case_metadata"]
    assert metadata["carving_mode"] == "PHYSICAL_FALLBACK"
    assert metadata["degradation_reason"] == "NO_COMFYUI_PROCESS"
    assert metadata["symbol_resolution"]["status"] == "SUCCESS"
    assert metadata["recovery_stages"]["python_objects"]["status"] == "UNAVAILABLE: NO_COMFYUI_PROCESS"
    assert manifest["process_context"] == []
    [artifact] = manifest["findings"]["artifacts"]
    assert artifact["process_id"] is None
    assert artifact["link_status"] == "UNASSOCIATED"
    assert_no_padding_and_hashes(case, manifest)


def test_examiner_pid_failure_cannot_publish_an_unscoped_manifest(
    tmp_path, mock_memory_image, symbols_dir, monkeypatch,
):
    from comfytrace import ingest
    from comfytrace.exceptions import SymbolResolutionError

    def unresolved(_engine):
        raise SymbolResolutionError("no offline symbols")

    monkeypatch.setattr(ingest.VolatilityEngine, "initialize_context", unresolved)
    case = tmp_path / "examiner-pid"
    args = parse_args(["extract", "--memory", str(mock_memory_image("T3")), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case), "--comfyui-pid", "13484"])

    assert run_extract(args) == 1
    assert not (case / "manifest.json").exists()


def test_multi_workflow_negative_zero_false_strict(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    case = tmp_path / "case04"
    code, manifest = extract(mock_memory_image("MULTI_WORKFLOW_NEGATIVE"), case, symbols_dir, fake_ingestor)
    body = manifest["findings"]
    workflows = {w["workflow_id"]: w for w in body["workflows"]}

    def embedded_hash(artifact_id: str) -> str:
        with Image.open(case / "exports" / "raw" / f"{artifact_id}.png") as img:
            graph = json.loads(img.text["prompt"],
                               parse_float=lambda number: int(float(number)) if float(number).is_integer()
                               else float(number))
        return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    for link in body["correlation_links"]:
        if link["link_class"] == "STRICT":
            assert workflows[link["workflow_id"]]["canonical_json_sha256"] == embedded_hash(link["artifact_id"])
    classes = sorted(link["link_class"] for link in body["correlation_links"])
    assert classes == ["STRICT", "STRICT", "SUPPORTED"], "the drifted graph links only by parameters"
    assert code == 0
    assert body["tier_summary"]["E2"] == 3


def test_manifest_is_immutable_across_reruns(tmp_path, mock_memory_image, fake_ingestor, symbols_dir):
    memory = mock_memory_image("T1")
    case = tmp_path / "case05"
    assert extract(memory, case, symbols_dir, fake_ingestor)[0] == 0
    before = (case / "manifest.json").read_bytes()
    assert extract(memory, case, symbols_dir, fake_ingestor)[0] == 1
    assert (case / "manifest.json").read_bytes() == before


def test_missing_memory_image_is_operational_failure(tmp_path, symbols_dir, fake_ingestor):
    code, _ = extract(tmp_path / "absent.raw", tmp_path / "case", symbols_dir, fake_ingestor)
    assert code == 1


def test_empty_image_exits_4_only_when_not_degraded(tmp_path, symbols_dir, fake_ingestor):
    empty = tmp_path / "blank.raw"
    empty.write_bytes(b"\x00" * 65536)
    code, manifest = extract(empty, tmp_path / "case", symbols_dir, fake_ingestor)
    assert code == 3, "symbols unresolved: degraded outranks zero artifacts"
    assert manifest["findings"]["tier_summary"] == {k: 0 for k in manifest["findings"]["tier_summary"]}


def _pixel_sha(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def test_t1_post_extraction_scoring_with_withheld_groundtruth(
    tmp_path, mock_memory_image, fake_ingestor, symbols_dir
):
    """Ground truth is withheld during extraction and supplied only to the post-extraction scorer."""
    memory = mock_memory_image("T1")
    case = tmp_path / "case_t1_scored"
    code, manifest = extract(memory, case, symbols_dir, fake_ingestor)
    assert code == 0
    manifest_before = (case / "manifest.json").read_bytes()

    graph = api_prompt()
    withheld_gt = {
        "schema_version": 1,
        "case_id": "synthetic-t1",
        "dump_sha256": hashlib.sha256(memory.read_bytes()).hexdigest(),
        "jobs": [
            {
                "job_id": "t1-job",
                "prompt_id": "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02",
                "canonical_prompt_sha256": hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest(),
                "expected_tier": "E2",
                "outputs": [
                    {
                        "output_id": "t1-job:out-0",
                        "pixel_sha256": _pixel_sha(noise_image(64, 48, seed=5)),
                        "expected_tier": "E2",
                    }
                ],
            }
        ],
    }
    gt_path = tmp_path / "withheld_t1.groundtruth.json"
    gt_path.write_text(json.dumps(withheld_gt), encoding="utf-8")

    report = score_case(case, gt_path)
    assert (case / "manifest.json").read_bytes() == manifest_before, "scorer must never modify the case"
    assert report["metrics"]["job_recovery_rate"]["numerator"] == 1
    assert report["metrics"]["job_recovery_rate"]["denominator"] == 1
    assert report["metrics"]["output_recovery_rate"]["rate"] == 1.0
    assert report["metrics"]["output_correctly_linked_rate"]["rate"] == 1.0
    assert report["metrics"]["link_precision"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
        "definition": report["metrics"]["link_precision"]["definition"],
    }
    assert report["links"]["wrong"] == 0
    assert report["links"]["ambiguous"] == 0
    assert report["metrics"]["gt_tier_pass_rate"]["rate"] == 1.0
    assert report["tiers"]["over_tiered"] == 0
    assert report["metrics"]["finding_policy_valid_rate"]["rate"] == 1.0
    assert report["metrics"]["stable_gt_artifact_match_rate"]["rate"] == 1.0
    assert report["metrics"]["stable_gt_workflow_match_rate"]["rate"] == 1.0
    assert manifest["findings"]["tier_summary"]["E2"] == 1


def test_multi_workflow_negative_post_extraction_scoring(
    tmp_path, mock_memory_image, fake_ingestor, symbols_dir
):
    """All 3 jobs (2 STRICT + 1 SUPPORTED drifted graph) score with zero wrong or ambiguous links."""
    memory = mock_memory_image("MULTI_WORKFLOW_NEGATIVE")
    case = tmp_path / "case_multi_scored"
    code, _ = extract(memory, case, symbols_dir, fake_ingestor)
    assert code == 0

    graphs = [api_prompt(seed=1000 + k, text=f"subject {k}") for k in range(3)]
    withheld_gt = {
        "schema_version": 1,
        "case_id": "synthetic-multi-negative",
        "dump_sha256": hashlib.sha256(memory.read_bytes()).hexdigest(),
        "jobs": [
            {
                "job_id": f"job-{k}",
                "prompt_id": f"00000000-0000-0000-0000-00000000000{k}",
                "canonical_prompt_sha256": hashlib.sha256(canonical_json(graphs[k]).encode("utf-8")).hexdigest(),
                "expected_tier": "E2",
                "outputs": [
                    {
                        "output_id": f"job-{k}:out-0",
                        "pixel_sha256": _pixel_sha(noise_image(64, 48, seed=40 + k)),
                        "expected_tier": "E2",
                    }
                ],
            }
            for k in range(3)
        ],
    }

    report = score_case(case, withheld_gt)
    assert report["metrics"]["job_recovery_rate"]["numerator"] == 3
    assert report["metrics"]["job_recovery_rate"]["denominator"] == 3
    assert report["metrics"]["output_correctly_linked_rate"]["numerator"] == 3
    assert report["metrics"]["output_correctly_linked_rate"]["denominator"] == 3
    assert report["metrics"]["link_precision"]["numerator"] == 3
    assert report["metrics"]["link_precision"]["denominator"] == 3
    assert report["links"]["wrong"] == 0
    assert report["links"]["wrong_strict"] == 0
    assert report["links"]["ambiguous"] == 0
    assert report["metrics"]["gt_tier_pass_rate"]["numerator"] == 6
    assert report["metrics"]["gt_tier_pass_rate"]["denominator"] == 6
    assert report["metrics"]["stable_gt_artifact_match_rate"]["rate"] == 1.0
    assert report["metrics"]["stable_gt_workflow_match_rate"]["rate"] == 1.0


def test_t2_and_t3_post_extraction_scoring_orphans_and_partial_raw_identity(
    tmp_path, mock_memory_image, fake_ingestor, symbols_dir
):
    """Unassociated orphans and partial physical artifacts score cleanly without false wrong-link counts."""
    t2_memory = mock_memory_image("T2")
    t2_case = tmp_path / "case_t2_scored"
    assert extract(t2_memory, t2_case, symbols_dir, fake_ingestor)[0] == 2
    t2_gt = {
        "schema_version": 1,
        "case_id": "synthetic-t2",
        "dump_sha256": hashlib.sha256(t2_memory.read_bytes()).hexdigest(),
        "jobs": [
            {
                "job_id": "t2-job",
                "outputs": [
                    {
                        "output_id": "t2-job:out-0",
                        "pixel_sha256": _pixel_sha(noise_image(48, 48, seed=8)),
                        "expected_tier": "E2",
                    }
                ],
            }
        ],
    }
    t2_report = score_case(t2_case, t2_gt)
    assert t2_report["metrics"]["output_recovery_rate"]["rate"] == 1.0
    assert t2_report["metrics"]["output_correctly_linked_rate"]["rate"] == 0.0
    assert t2_report["links"]["wrong"] == 0
    assert t2_report["links"]["ambiguous"] == 0
    assert t2_report["links"]["unassociated_gt_outputs"] == 1
    assert t2_report["metrics"]["link_precision"]["denominator"] == 0
    assert t2_report["metrics"]["link_precision"]["rate"] is None
    assert t2_report["tiers"]["under_tiered"] == 1
    assert t2_report["tiers"]["over_tiered"] == 0

    t3_memory = mock_memory_image("T3")
    t3_case = tmp_path / "case_t3_scored"
    code, t3_manifest = extract(t3_memory, t3_case, symbols_dir, fake_ingestor)
    assert code == 3
    [partial_art] = t3_manifest["findings"]["artifacts"]
    raw_sha = hashlib.sha256(
        b"".join((t3_case / p).read_bytes() for p in partial_art["raw_file_paths"])
    ).hexdigest()
    assert partial_art["container_sha256"] is None and partial_art["pixel_sha256"] is None
    t3_gt = {
        "schema_version": 1,
        "case_id": "synthetic-t3",
        "dump_sha256": hashlib.sha256(t3_memory.read_bytes()).hexdigest(),
        "jobs": [
            {
                "job_id": "t3-job",
                "outputs": [
                    {
                        "output_id": "t3-job:out-0",
                        "raw_sha256s": [raw_sha],
                        "expected_tier": "UNTIERED",
                    }
                ],
            }
        ],
    }
    t3_report = score_case(t3_case, t3_gt)
    assert t3_report["metrics"]["output_recovery_rate"]["rate"] == 1.0
    assert t3_report["metrics"]["stable_gt_artifact_match_rate"]["rate"] == 1.0
    assert t3_report["identity_stability"]["artifact_matches_by_kind"] == {"raw_sha256": 1}
    assert t3_report["links"]["wrong"] == 0
    assert t3_report["metrics"]["gt_tier_pass_rate"]["rate"] == 1.0


def _groundtruth_cases() -> list[dict[str, Any]]:
    return load_groundtruth_cases(None)


def _groundtruth_params() -> list[dict[str, Any] | None]:
    cases = _groundtruth_cases()
    return list(cases) if cases else [None]


@pytest.mark.groundtruth
@pytest.mark.parametrize("case", _groundtruth_params(), ids=lambda c: c["id"] if c else "none")
def test_groundtruth_lab_dump(case: dict[str, Any] | None, tmp_path: Path):
    if case is None:
        pytest.skip("no ground-truth lab cases registered")
    resolved = resolve_case_paths(case)
    token = case_env_token(resolved.case_id)
    if resolved.memory is None or not resolved.memory.is_file():
        message = (
            f"ground-truth lab dump for {resolved.case_id!r} not available at {resolved.memory} "
            f"(source: {resolved.memory_source}; configure {ENV_DUMPS_DIR}, "
            f"COMFYTRACE_EVAL_MEMORY_{token}, or {ENV_CONFIG})"
        )
        if resolved.memory_explicit:
            pytest.fail(message)
        pytest.skip(message)

    symbols_dir = resolved.symbols_dir
    if resolved.symbols_explicit and (symbols_dir is None or not symbols_dir.is_dir()):
        pytest.fail(
            f"configured Windows kernel symbols directory for {resolved.case_id!r} not available at {symbols_dir} "
            f"(source: {resolved.symbols_source}; configure {ENV_SYMBOLS_DIR}, "
            f"COMFYTRACE_EVAL_SYMBOLS_{token}, or {ENV_CONFIG})"
        )
    if resolved.requires_symbols and (symbols_dir is None or not symbols_dir.is_dir()):
        pytest.skip(
            f"Windows kernel symbols directory for {resolved.case_id!r} not available at {symbols_dir} "
            f"(source: {resolved.symbols_source}; configure {ENV_SYMBOLS_DIR}, "
            f"COMFYTRACE_EVAL_SYMBOLS_{token}, or {ENV_CONFIG})"
        )
    if symbols_dir is None:
        symbols_dir = tmp_path / "empty_symbols"
        symbols_dir.mkdir()

    memory = resolved.memory
    digest = hashlib.sha256()
    with open(memory, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    assert digest.hexdigest() == case["sha256"], "dump does not match the registered acquisition"

    outdir = tmp_path / "gt"
    # Ground truth is never passed into extraction; only --memory, --symbols-dir and --outdir are supplied.
    args = parse_args(["extract", "--memory", str(memory), "--symbols-dir", str(symbols_dir), "--outdir", str(outdir)])
    code = run_extract(args)
    assert code == case["expected_exit_code"]
    if code == 1:
        return
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert_no_padding_and_hashes(outdir, manifest)
    for tier, minimum in case.get("expected_min_tiers", {}).items():
        assert manifest["findings"]["tier_summary"][tier] >= minimum
    kinds = Counter(a["carve_type"] for a in manifest["findings"]["artifacts"])
    for kind, minimum in case.get("expected_min_carve_types", {}).items():
        assert kinds[kind] >= minimum, f"{kind}: {kinds[kind]} < {minimum}"
    allowed = case.get("strict_allowed")
    if allowed is not None:
        links = manifest["findings"]["correlation_links"]
        strict = [link["artifact_id"] for link in links if link["link_class"] == "STRICT"]
        assert set(strict) <= set(allowed)

    # Optional post-extraction scoring against withheld ground truth (supplied ONLY after extraction completes).
    if resolved.groundtruth is not None:
        if not resolved.groundtruth.is_file():
            if resolved.groundtruth_explicit:
                pytest.fail(
                    f"configured withheld ground truth for {resolved.case_id!r} not found at "
                    f"{resolved.groundtruth} (source: {resolved.groundtruth_source})"
                )
            return
        report = score_case(outdir, resolved.groundtruth, view="effective", strict_dump_sha=True)
        limits = case.get("expected_score_limits", {})
        for scope, key in (("jobs", "job_recovery_rate"), ("outputs", "output_recovery_rate")):
            recovery = report["metrics"][key]
            if f"expected_{scope}" in limits:
                assert recovery["denominator"] == limits[f"expected_{scope}"]
            assert recovery["numerator"] >= limits.get(f"min_recovered_{scope}", int(recovery["denominator"] > 0))
        correctly_linked = report["metrics"]["output_correctly_linked_rate"]["numerator"]
        assert correctly_linked >= limits.get("min_correct_output_links", 0)
        assert report["links"]["wrong"] <= limits.get("max_wrong_links", 0)
        assert report["links"]["wrong_strict"] <= limits.get("max_wrong_strict_links", 0)
        assert report["output_identity"]["mismatched"] <= limits.get("max_output_identity_mismatches", 0)
        assert report["tiers"]["over_tiered"] <= limits.get("max_over_tiered", 0)
        assert report["tiers"]["policy_invalid_findings"] <= limits.get("max_policy_invalid_findings", 0)
        assert report["promotions"]["invalid"] <= limits.get("max_invalid_promotions", 0)


def test_extract_never_reads_withheld_groundtruth_env_or_files(
    tmp_path, mock_memory_image, fake_ingestor, symbols_dir, monkeypatch
):
    """Confirms extraction never accesses withheld ground truth or eval config even when configured in env."""
    memory = mock_memory_image("T1")
    gt_dir = tmp_path / "withheld_gt"
    gt_dir.mkdir()
    gt_file = gt_dir / "t1.groundtruth.json"
    gt_file.write_text('{"schema_version": 1, "jobs": []}', encoding="utf-8")
    cfg_file = tmp_path / "eval_cfg.json"
    cfg_file.write_text(json.dumps({"groundtruth_dir": str(gt_dir)}), encoding="utf-8")

    monkeypatch.setenv(ENV_GROUNDTRUTH_DIR, str(gt_dir))
    monkeypatch.setenv("COMFYTRACE_EVAL_GROUNDTRUTH_T1", str(gt_file))
    monkeypatch.setenv(ENV_CONFIG, str(cfg_file))

    forbidden = {gt_file.resolve(), cfg_file.resolve()}
    real_open = open
    real_path_open = Path.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() in forbidden:
            raise AssertionError(f"extraction accessed withheld file: {file}")
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        if self.resolve() in forbidden:
            raise AssertionError(f"extraction accessed withheld file: {self}")
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)

    case = tmp_path / "case_isolated"
    code, manifest = extract(memory, case, symbols_dir, fake_ingestor)
    assert code == 0
    assert manifest["findings"]["tier_summary"]["E2"] == 1


def test_groundtruth_runner_fails_on_configured_but_missing_explicit_symbols_or_groundtruth(
    tmp_path, mock_memory_image, monkeypatch
):
    """Configured-but-missing explicit symbols (even when requires_symbols=False) or ground truth must fail."""
    memory = mock_memory_image("T3")
    mem_sha = hashlib.sha256(memory.read_bytes()).hexdigest()
    t3_case = {
        "id": "synthetic-t3-case",
        "memory_filename": memory.name,
        "sha256": mem_sha,
        "requires_symbols": False,
        "expected_exit_code": 3,
    }

    # 1. Explicit symbols_dir points to a missing folder even though requires_symbols=False -> MUST fail.
    monkeypatch.setenv(ENV_DUMPS_DIR, str(memory.parent))
    monkeypatch.setenv(ENV_SYMBOLS_DIR, str(tmp_path / "missing_symbols_dir"))
    monkeypatch.delenv(ENV_GROUNDTRUTH_DIR, raising=False)
    monkeypatch.delenv(ENV_CONFIG, raising=False)
    with pytest.raises(pytest.fail.Exception, match="configured Windows kernel symbols directory"):
        test_groundtruth_lab_dump(t3_case, tmp_path / "run_missing_syms")

    # 2. Explicit COMFYTRACE_EVAL_GROUNDTRUTH_DIR configured with missing file -> MUST fail, never silently pass.
    valid_syms = tmp_path / "empty_syms"
    valid_syms.mkdir()
    missing_gt_dir = tmp_path / "missing_gt_dir"
    monkeypatch.setenv(ENV_SYMBOLS_DIR, str(valid_syms))
    monkeypatch.setenv(ENV_GROUNDTRUTH_DIR, str(missing_gt_dir))
    with pytest.raises(pytest.fail.Exception, match="configured withheld ground truth"):
        test_groundtruth_lab_dump(t3_case, tmp_path / "run_missing_gt")
