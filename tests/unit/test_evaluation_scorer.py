"""Synthetic unit tests for portable evaluation path resolution and post-extraction scoring rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.paths import (
    DEFAULT_CASES_FILE,
    ENV_CASES_FILE,
    ENV_CONFIG,
    ENV_DUMPS_DIR,
    ENV_GROUNDTRUTH_DIR,
    ENV_SYMBOLS_DIR,
    case_env_token,
    load_groundtruth_cases,
    resolve_case_paths,
)
from tests.evaluation.scorer import format_summary, main, score_case

DUMP_SHA = "a1" * 32


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _write_case(
    tmp_path: Path,
    *,
    name: str = "case",
    artifacts: list[dict[str, Any]],
    workflows: list[dict[str, Any]],
    links: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]] | None = None,
    memory_only_findings: list[dict[str, Any]] | None = None,
    image_links: dict[str, Any] | None = None,
    regenerations: list[dict[str, Any]] | None = None,
    derived_files: dict[str, bytes] | None = None,
    raw_files: dict[str, bytes] | None = None,
    dump_sha: str = DUMP_SHA,
    include_promotion_replays: bool = False,
) -> Path:
    case = tmp_path / name
    case.mkdir(parents=True, exist_ok=True)
    for rel, content in (raw_files or {}).items():
        p = case / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    for rel, content in (derived_files or {}).items():
        p = case / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    manifest = {
        "schema_version": "1.0.0",
        "case_metadata": {
            "case_id": f"CASE-{name}",
            "exit_code": 0,
            "carving_mode": "VIRTUAL_PROCESS",
            "memory_image": {"path": "synthetic.raw", "size_bytes": 4096, "sha256": dump_sha},
        },
        "process_context": [],
        "findings": {
            "tier_summary": {},
            "findings": findings,
            "artifacts": artifacts,
            "workflows": workflows,
            "correlation_links": links,
            "supporting_artifacts": [],
        },
    }
    manifest_path = case / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    if evidence_records is not None or memory_only_findings is not None:
        ev_doc: dict[str, Any] = {"schema_version": 1, "records": evidence_records or []}
        if memory_only_findings is not None:
            ev_doc["memory_only_findings"] = memory_only_findings
        (case / "recovery-evidence.json").write_text(json.dumps(ev_doc, indent=2), encoding="utf-8")

    if image_links is not None:
        il = dict(image_links)
        if il.get("manifest_sha256") == "AUTO":
            il["manifest_sha256"] = manifest_sha
        if include_promotion_replays:
            regenerations = list(regenerations or [])
            il["promotions"] = [dict(promotion) for promotion in il.get("promotions", [])]
            for index, promotion in enumerate(il["promotions"], 100):
                prompt_id = promotion["prompt_id"]
                workflow = next(w for w in workflows if w["execution_parameters"].get("prompt_id") == prompt_id)
                asset = f"exports/derived/DERIVED_test_{index}.png"
                payload = f"synthetic runner output {index}".encode()
                digest = hashlib.sha256(payload).hexdigest()
                (case / asset).parent.mkdir(parents=True, exist_ok=True)
                (case / asset).write_bytes(payload)
                handoff = f"REGEN-{index}"
                promotion["selected_raster"] = {"handoff_id": handoff, "file": asset, "sha256": digest}
                regenerations.append({
                    "handoff_id": handoff, "workflow_id": workflow["workflow_id"], "status": "COMPLETED",
                    "target_manifest_sha256": manifest_sha,
                    "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
                    "derived_regenerated_image": asset, "evaluation": {"derived_regenerated_image_sha256": digest},
                })
        (case / "image-links.json").write_text(json.dumps(il, indent=2), encoding="utf-8")

    if regenerations:
        reg_dir = case / "regenerations"
        reg_dir.mkdir(exist_ok=True)
        for rec in regenerations:
            r = dict(rec)
            if r.get("target_manifest_sha256") == "AUTO":
                r["target_manifest_sha256"] = manifest_sha
            (reg_dir / f"{r['handoff_id']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    return case


def _art(
    artifact_id: str,
    *,
    carve_type: str = "ENCODED_PNG",
    completeness: str = "COMPLETE",
    link_status: str = "UNASSOCIATED",
    associated_workflow_id: str | None = None,
    ambiguous_candidate_workflows: list[str] | None = None,
    container_sha256: str | None = None,
    pixel_sha256: str | None = None,
    physical_offsets: list[str] | None = None,
    width: int = 512,
    height: int = 512,
    raw_file_paths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "completeness": completeness,
        "link_status": link_status,
        "associated_workflow_id": associated_workflow_id,
        "ambiguous_candidate_workflows": ambiguous_candidate_workflows or [],
        "is_derived": False,
        "carve_type": carve_type,
        "source_layer": "process_1000",
        "process_id": 1000,
        "virtual_address": "0x10000",
        "physical_offsets": physical_offsets or [],
        "recovered_bytes": 1024,
        "container_sha256": container_sha256,
        "pixel_sha256": pixel_sha256,
        "pixel_metadata": {"width": width, "height": height},
        "validation": {},
        "gaps": [],
        "raw_file_paths": raw_file_paths or [],
        "derived_preview_path": None,
        "derived_preview_metadata": None,
    }


def _wf(
    workflow_id: str,
    *,
    prompt_id: str | None = None,
    additional_prompt_ids: list[str] | None = None,
    canonical_json_sha256: str | None = None,
    completeness: str = "COMPLETE",
    prompt_format: str = "API_GRAPH",
    encoding_detected: str = "CPYTHON_OBJECTS",
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if prompt_id is not None:
        params["prompt_id"] = prompt_id
    if additional_prompt_ids:
        params["additional_prompt_ids"] = additional_prompt_ids
    return {
        "workflow_id": workflow_id,
        "completeness": completeness,
        "prompt_format": prompt_format,
        "encoding_detected": encoding_detected,
        "source_process_id": 1000,
        "virtual_address": "0x20000",
        "physical_offsets": [],
        "canonical_json_sha256": canonical_json_sha256 or "",
        "execution_parameters": params,
        "dependencies": {"node_classes": [], "model_assets": []},
        "raw_file_path": "",
    }


# ---------------------------------------------------------------------------
# Portable real-capture path resolution (Windows & Linux)
# ---------------------------------------------------------------------------


def test_registered_groundtruth_cases_preserves_sha_and_structural_assertions():
    cases = load_groundtruth_cases(DEFAULT_CASES_FILE, env={})
    [run1] = [c for c in cases if c["id"] == "gen-only-run-1"]
    assert run1["sha256"] == "83b7c875b39e4998d105bc21b8fa7594076c0709d43731ce9bc6dd1fba92139a"
    assert run1["expected_exit_code"] == 2
    assert run1["expected_min_tiers"] == {"E2": 4, "E3": 7}
    assert run1["expected_min_carve_types"] == {"TENSOR_IMAGE": 1, "DECODED_BITMAP": 7}
    assert run1["memory_filename"] == "gen_only_run_1.raw"
    assert run1["requires_symbols"] is True


def test_portable_path_resolution_precedence_windows_and_linux(tmp_path):
    case = load_groundtruth_cases(DEFAULT_CASES_FILE, env={})[0]
    assert case_env_token(case["id"]) == "GEN_ONLY_RUN_1"

    # Shared directory env vars work identically across platforms
    dumps = tmp_path / "dumps"
    syms = tmp_path / "symbols"
    gt_dir = tmp_path / "gt"
    for folder in (dumps, syms, gt_dir):
        folder.mkdir()
    (gt_dir / "gen-only-run-1.groundtruth.json").write_text("{}", encoding="utf-8")

    env_shared = {
        ENV_DUMPS_DIR: str(dumps),
        ENV_SYMBOLS_DIR: str(syms),
        ENV_GROUNDTRUTH_DIR: str(gt_dir),
    }
    for plat in ("win32", "linux"):
        res = resolve_case_paths(case, env=env_shared, platform_name=plat)
        assert res.memory == dumps / "gen_only_run_1.raw"
        assert res.memory_source == f"env:{ENV_DUMPS_DIR}"
        assert res.memory_explicit is False
        assert res.symbols_dir == syms
        assert res.symbols_source == f"env:{ENV_SYMBOLS_DIR}"
        assert res.symbols_explicit is True
        assert res.groundtruth == gt_dir / "gen-only-run-1.groundtruth.json"

    # Config file with config-relative paths beats defaults, while per-case env vars beat config file
    cfg_dir = tmp_path / "cfg_bundle"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "eval-local.json"
    cfg_file.write_text(
        json.dumps({
            "dumps_dir": "local_dumps",
            "symbols_dir": "local_symbols",
            "cases": {
                "gen-only-run-1": {
                    "memory": "override_dumps/run1.raw",
                    "symbols_dir": "win11_symbols",
                    "groundtruth": "withheld/run1_gt.json",
                }
            },
        }),
        encoding="utf-8",
    )
    res_cfg = resolve_case_paths(case, env={ENV_CONFIG: str(cfg_file)}, platform_name="win32")
    assert res_cfg.memory == cfg_dir / "override_dumps/run1.raw"
    assert res_cfg.memory_explicit is True
    assert res_cfg.symbols_dir == cfg_dir / "win11_symbols"
    assert res_cfg.groundtruth == cfg_dir / "withheld/run1_gt.json"
    assert res_cfg.groundtruth_explicit is True

    # Per-case env vars win over everything else
    env_per_case = {
        ENV_CONFIG: str(cfg_file),
        "COMFYTRACE_EVAL_MEMORY_GEN_ONLY_RUN_1": str(tmp_path / "exact.raw"),
        "COMFYTRACE_EVAL_SYMBOLS_GEN_ONLY_RUN_1": str(tmp_path / "exact_syms"),
        "COMFYTRACE_EVAL_GROUNDTRUTH_GEN_ONLY_RUN_1": str(tmp_path / "exact_gt.json"),
    }
    res_env = resolve_case_paths(case, env=env_per_case, platform_name="linux")
    assert res_env.memory == tmp_path / "exact.raw"
    assert res_env.symbols_dir == tmp_path / "exact_syms"
    assert res_env.groundtruth == tmp_path / "exact_gt.json"


def test_path_resolution_handles_cross_platform_legacy_paths_and_rejects_traversal(tmp_path):
    win_case = {
        "id": "win-legacy",
        "memory": r"D:\captures\T2_running.raw",
        "symbols_dir": r"D:\symbols\win11",
        "expected_exit_code": 2,
    }
    res = resolve_case_paths(win_case, env={ENV_DUMPS_DIR: str(tmp_path)}, platform_name="linux")
    assert res.memory == tmp_path / "T2_running.raw"

    bad_case = {"id": "bad-rel", "memory_filename": "../escape.raw"}
    with pytest.raises(ValueError, match="invalid relative path"):
        resolve_case_paths(bad_case, env={ENV_DUMPS_DIR: str(tmp_path)})

    with pytest.raises(FileNotFoundError, match=ENV_CONFIG):
        resolve_case_paths(win_case, env={ENV_CONFIG: str(tmp_path / "missing.json")})

    with pytest.raises(FileNotFoundError, match=ENV_CASES_FILE):
        load_groundtruth_cases(env={ENV_CASES_FILE: str(tmp_path / "missing_cases.json")})


# ---------------------------------------------------------------------------
# Stable evidence identity vs transient IDs
# ---------------------------------------------------------------------------


def test_scores_are_invariant_when_all_transient_ids_are_renumbered(tmp_path):
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {
                "job_id": "job-1",
                "prompt_id": "pid-1",
                "canonical_prompt_sha256": _sha("prompt-1"),
                "expected_tier": "E2",
                "outputs": [
                    {
                        "output_id": "job-1:out-0",
                        "container_sha256": _sha("png-1"),
                        "pixel_sha256": _sha("pix-1"),
                        "expected_tier": "E2",
                    }
                ],
            },
            {
                "job_id": "job-2",
                "canonical_prompt_sha256": _sha("prompt-2"),
                "outputs": [
                    {
                        "output_id": "job-2:out-0",
                        "physical_offsets": ["0x000000000002A000"],
                        "carve_type": "DECODED_BITMAP",
                        "expected_tier": "E3",
                    }
                ],
            },
        ],
    }

    def build_variant(name: str, a1: str, a2: str, a3: str, w1: str, w2: str, f1: str, f2: str, l1: str, l2: str):
        return _write_case(
            tmp_path,
            name=name,
            artifacts=[
                _art(a1, link_status="STRICT", associated_workflow_id=w1,
                     container_sha256=_sha("png-1"), pixel_sha256=_sha("pix-1")),
                _art(a2, carve_type="TENSOR_IMAGE", link_status="STRICT", associated_workflow_id=w1,
                     pixel_sha256=_sha("pix-1")),
                _art(a3, carve_type="DECODED_BITMAP", completeness="PARTIAL", link_status="SUPPORTED",
                     associated_workflow_id=w2, physical_offsets=["0x2a000"]),
            ],
            workflows=[
                _wf(w1, prompt_id="pid-1", canonical_json_sha256=_sha("prompt-1")),
                _wf(w2, canonical_json_sha256=_sha("prompt-2"), encoding_detected="PNG_TEXT"),
            ],
            links=[
                {"link_id": l1, "workflow_id": w1, "artifact_id": a1, "link_class": "STRICT",
                 "matched_keys": ["PROMPT_ID"]},
                {"link_id": f"{l1}b", "workflow_id": w1, "artifact_id": a2, "link_class": "STRICT",
                 "matched_keys": ["EXECUTION_CACHE_SIGNATURE"]},
                {"link_id": l2, "workflow_id": w2, "artifact_id": a3, "link_class": "SUPPORTED",
                 "matched_keys": ["SEED"]},
            ],
            findings=[
                {"finding_id": f1, "tier": "E2", "artifact_ids": [a1, a2, w1], "link_ids": [l1, f"{l1}b"],
                 "rationale": "r1"},
                {"finding_id": f2, "tier": "E3", "artifact_ids": [a3, w2], "link_ids": [l2], "rationale": "r2"},
            ],
        )

    c1 = build_variant("run_a", "CARVE-PNG-001", "TENSOR-001", "BITMAP-001", "WFPY-001", "WFPNG-001",
                       "FIND-001", "FIND-002", "LINK-001", "LINK-002")
    c2 = build_variant("run_b", "CARVE-PNG-901", "TENSOR-777", "BITMAP-404", "WFPY-888", "WFPNG-555",
                       "FIND-999", "FIND-888", "LINK-333", "LINK-444")

    r1 = score_case(c1, gt)
    r2 = score_case(c2, gt)
    assert r1["metrics"] == r2["metrics"]
    assert r1["links"] == r2["links"]
    assert r1["tiers"] == r2["tiers"]
    assert r1["identity_stability"] == r2["identity_stability"]
    assert r1["metrics"]["stable_gt_artifact_match_rate"]["rate"] == 1.0
    assert r1["metrics"]["stable_gt_workflow_match_rate"]["rate"] == 1.0
    assert r1["identity_stability"]["artifact_matches_by_kind"] == {
        "container_sha256": 1,
        "physical_offset": 1,
        "pixel_sha256": 1,
    }


def test_transient_id_fallback_is_explicitly_disclosed_and_dump_sha_mismatch_raises(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[_art("CARVE-PNG-001", link_status="STRICT", associated_workflow_id="WF-001")],
        workflows=[_wf("WF-001")],
        links=[{"link_id": "LINK-001", "workflow_id": "WF-001", "artifact_id": "CARVE-PNG-001",
                "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]}],
        findings=[{"finding_id": "FIND-001", "tier": "E2", "artifact_ids": ["CARVE-PNG-001", "WF-001"],
                   "link_ids": ["LINK-001"], "rationale": "ok"}],
    )
    gt_transient = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{
            "job_id": "j1",
            "workflow_ids": ["WF-001"],
            "outputs": [{"output_id": "j1:0", "artifact_ids": ["CARVE-PNG-001"]}],
        }],
    }
    report = score_case(case, gt_transient)
    assert report["metrics"]["stable_gt_artifact_match_rate"]["numerator"] == 0
    assert report["metrics"]["stable_gt_artifact_match_rate"]["denominator"] == 1
    assert report["metrics"]["stable_gt_workflow_match_rate"]["numerator"] == 0
    assert report["metrics"]["stable_gt_workflow_match_rate"]["denominator"] == 1

    gt_wrong_dump = {**gt_transient, "dump_sha256": "ff" * 32}
    with pytest.raises(ValueError, match="dump SHA-256 mismatch"):
        score_case(case, gt_wrong_dump)


# ---------------------------------------------------------------------------
# Wrong vs Ambiguous links & Distractors
# ---------------------------------------------------------------------------


def test_wrong_links_vs_ambiguous_abstentions_and_distractors(tmp_path):
    """Proves AMBIGUOUS is never counted as WRONG, wrong STRICT links invalidate tiers,
    distractors wrongly linked count as WRONG, and uncatalogued links stay explicitly separate."""
    case = _write_case(
        tmp_path,
        artifacts=[
            # 1. Output of job-1 WRONGLY STRICT-linked to job-2 -> WRONG + invalid E2 finding!
            _art("A-WRONG", link_status="STRICT", associated_workflow_id="WF-2", container_sha256=_sha("out-1")),
            # 2. Output of job-2 marked AMBIGUOUS including true job-2 -> AMBIGUOUS (not WRONG!)
            _art("A-AMBIG-IN", link_status="AMBIGUOUS", ambiguous_candidate_workflows=["WF-1", "WF-2"],
                 container_sha256=_sha("out-2")),
            # 3. Output of job-3 marked AMBIGUOUS excluding true job-3 -> AMBIGUOUS (excludes truth, still not WRONG)
            _art("A-AMBIG-OUT", link_status="AMBIGUOUS", ambiguous_candidate_workflows=["WF-1", "WF-2"],
                 container_sha256=_sha("out-3")),
            # 4. Expected-ambiguous pixel buffer matching both job-4 and job-5 -> AMBIGUOUS_EXPECTED
            _art("A-AMBIG-EXP", carve_type="PIXEL_BUFFER", link_status="AMBIGUOUS",
                 ambiguous_candidate_workflows=["WF-4", "WF-5"], pixel_sha256=_sha("shared-pix")),
            # 5. Distractor wrongly SUPPORTED-linked to job-1 -> WRONG_DISTRACTOR_LINKED + OVER_TIERED
            _art("A-DIST-BAD", link_status="SUPPORTED", associated_workflow_id="WF-1",
                 container_sha256=_sha("dist-bad")),
            # 6. Distractor <16px test PNG allowed to keep link while UNTIERED -> NOT wrong, tier PASS
            _art("A-TINY-OK", link_status="STRICT", associated_workflow_id="WF-1",
                 container_sha256=_sha("tiny-ok"), width=1, height=1),
            # 7. Uncatalogued artifact linked to job-1 -> reported separately, not inflated into correct or wrong
            _art("A-UNCAT", link_status="STRICT", associated_workflow_id="WF-1", container_sha256=_sha("uncat")),
        ],
        workflows=[
            _wf("WF-1", prompt_id="p1", canonical_json_sha256=_sha("g1")),
            _wf("WF-2", prompt_id="p2", canonical_json_sha256=_sha("g2")),
            _wf("WF-3", prompt_id="p3", canonical_json_sha256=_sha("g3")),
            _wf("WF-4", prompt_id="p4", canonical_json_sha256=_sha("g4")),
            _wf("WF-5", prompt_id="p5", canonical_json_sha256=_sha("g5")),
        ],
        links=[
            {"link_id": "L1", "workflow_id": "WF-2", "artifact_id": "A-WRONG",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L2", "workflow_id": "WF-1", "artifact_id": "A-AMBIG-IN",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L3", "workflow_id": "WF-2", "artifact_id": "A-AMBIG-IN",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L4", "workflow_id": "WF-1", "artifact_id": "A-AMBIG-OUT",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L5", "workflow_id": "WF-2", "artifact_id": "A-AMBIG-OUT",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L6", "workflow_id": "WF-4", "artifact_id": "A-AMBIG-EXP",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L7", "workflow_id": "WF-5", "artifact_id": "A-AMBIG-EXP",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L8", "workflow_id": "WF-1", "artifact_id": "A-DIST-BAD",
             "link_class": "SUPPORTED", "matched_keys": ["SEED"]},
            {"link_id": "L9", "workflow_id": "WF-1", "artifact_id": "A-TINY-OK",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L10", "workflow_id": "WF-1", "artifact_id": "A-UNCAT",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
        ],
        findings=[
            {"finding_id": "F-WRONG-E2", "tier": "E2", "artifact_ids": ["A-WRONG", "WF-2"],
             "link_ids": ["L1"], "rationale": "bad"},
            {"finding_id": "F-AMBIG-1", "tier": "UNTIERED", "artifact_ids": ["A-AMBIG-IN"],
             "link_ids": ["L2", "L3"], "rationale": "ambig"},
            {"finding_id": "F-AMBIG-2", "tier": "UNTIERED", "artifact_ids": ["A-AMBIG-OUT"],
             "link_ids": ["L4", "L5"], "rationale": "ambig"},
            {"finding_id": "F-AMBIG-3", "tier": "UNTIERED", "artifact_ids": ["A-AMBIG-EXP"],
             "link_ids": ["L6", "L7"], "rationale": "ambig"},
            {"finding_id": "F-DIST-E2", "tier": "E2", "artifact_ids": ["A-DIST-BAD", "WF-1"],
             "link_ids": ["L8"], "rationale": "bad dist"},
            {"finding_id": "F-TINY", "tier": "UNTIERED", "artifact_ids": ["A-TINY-OK"],
             "link_ids": ["L9"], "rationale": "tiny"},
            {"finding_id": "F-UNCAT", "tier": "E2", "artifact_ids": ["A-UNCAT", "WF-1"],
             "link_ids": ["L10"], "rationale": "uncat"},
        ],
    )

    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {"job_id": "job-1", "prompt_id": "p1",
             "outputs": [{"output_id": "o1", "container_sha256": _sha("out-1"), "expected_tier": "E2"}]},
            {"job_id": "job-2", "prompt_id": "p2", "outputs": [{"output_id": "o2", "container_sha256": _sha("out-2")}]},
            {"job_id": "job-3", "prompt_id": "p3", "outputs": [{"output_id": "o3", "container_sha256": _sha("out-3")}]},
            {"job_id": "job-4", "prompt_id": "p4",
             "outputs": [{"output_id": "o4", "pixel_sha256": _sha("shared-pix"), "expected_ambiguous": True}]},
            {"job_id": "job-5", "prompt_id": "p5",
             "outputs": [{"output_id": "o5", "pixel_sha256": _sha("shared-pix"), "expected_ambiguous": True}]},
        ],
        "distractors": [
            {"distractor_id": "dist-bad", "container_sha256": _sha("dist-bad")},
            {"distractor_id": "tiny-ok", "container_sha256": _sha("tiny-ok"),
             "allow_link_without_tier": True, "expected_tier": "UNTIERED"},
        ],
    }

    report = score_case(case, gt)
    links = report["links"]
    assert links["correct"] == 0
    assert links["wrong"] == 2  # A-WRONG + A-DIST-BAD (NOT the ambiguous ones, NOT A-TINY-OK, NOT A-UNCAT!)
    assert links["wrong_strict"] == 1
    assert links["wrong_supported"] == 1
    assert links["wrong_breakdown"] == {
        "cross_job": 1,
        "same_job_output": 0,
        "overconfident_ambiguous": 0,
        "invalid_attribution_rule": 0,
        "distractor_linked": 1,
        "unmatched_workflow": 0,
        "conflicting_sources": 0,
    }
    assert links["ambiguous"] == 3
    assert links["ambiguous_breakdown"] == {
        "expected": 1,
        "includes_true_job": 1,
        "excludes_true_job": 1,
        "same_job_output": 0,
        "distractor": 0,
        "unmatched": 0,
    }
    assert links["allowed_untiered_distractor_links"] == 1
    assert links["unmatched_attributed_artifacts"] == 1

    # Explicit denominators exclude AMBIGUOUS from link_precision/wrong_link_rate,
    # exclude allow_link_without_tier from distractor_false_link_rate, and surface uncatalogued coverage.
    assert report["metrics"]["link_precision"]["numerator"] == 0
    assert report["metrics"]["link_precision"]["denominator"] == 2
    assert report["metrics"]["wrong_link_rate"]["numerator"] == 2
    assert report["metrics"]["wrong_link_rate"]["denominator"] == 2
    assert report["metrics"]["wrong_strict_link_rate"]["numerator"] == 1
    assert report["metrics"]["wrong_strict_link_rate"]["denominator"] == 1
    assert report["metrics"]["attributed_gt_coverage_rate"]["numerator"] == 2
    assert report["metrics"]["attributed_gt_coverage_rate"]["denominator"] == 3
    assert report["metrics"]["distractor_false_link_rate"]["numerator"] == 1
    assert report["metrics"]["distractor_false_link_rate"]["denominator"] == 1
    assert report["metrics"]["artifact_ambiguous_rate"]["numerator"] == 3
    assert report["metrics"]["artifact_ambiguous_rate"]["denominator"] == 4

    # Wrong-linked A-WRONG must NOT credit o1 with E2, and uncatalogued F-UNCAT must NOT credit job-1 with E2!
    o1 = next(o for o in report["outputs"] if o["output_id"] == "o1")
    assert o1["achieved_tier"] is None
    assert o1["wrongly_linked_findings"] == ["F-WRONG-E2"]
    assert o1["tier_check"]["status"] == "UNDER_TIERED"
    j1 = next(j for j in report["jobs"] if j["job_id"] == "job-1")
    assert j1["achieved_tier"] is None

    # Finding policy validator flags both F-WRONG-E2 and F-DIST-E2
    checks = {c["finding_id"]: c for c in report["finding_checks"]}
    assert "WRONG_LINK_TIER" in checks["F-WRONG-E2"]["issues"]
    assert "WRONG_LINK_TIER" in checks["F-DIST-E2"]["issues"]
    assert "DISTRACTOR_TIERED" in checks["F-DIST-E2"]["issues"]
    assert checks["F-TINY"]["valid"] is True

    # Distractor dist-bad is OVER_TIERED; tiny-ok PASSES
    dists = {d["distractor_id"]: d for d in report["distractors"]}
    assert dists["dist-bad"]["tier_check"]["status"] == "OVER_TIERED"
    assert dists["tiny-ok"]["tier_check"]["status"] == "PASS"


# ---------------------------------------------------------------------------
# Tier correctness: over-tiering, <16px rule, effective vs extraction vs memory_only
# ---------------------------------------------------------------------------


def test_over_tiering_partial_and_tiny_images_caught_by_gt_and_policy(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[
            _art("A-PARTIAL", completeness="PARTIAL", link_status="STRICT", associated_workflow_id="WF-1",
                 raw_file_paths=["exports/raw/p.bin"]),
            _art("A-TINY-BAD", completeness="COMPLETE", link_status="STRICT", associated_workflow_id="WF-2",
                 container_sha256=_sha("tiny-bad"), width=1, height=1),
        ],
        workflows=[
            _wf("WF-1", prompt_id="p1", completeness="COMPLETE"),
            _wf("WF-2", prompt_id="p2", completeness="COMPLETE"),
        ],
        links=[
            {"link_id": "L1", "workflow_id": "WF-1", "artifact_id": "A-PARTIAL",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L2", "workflow_id": "WF-2", "artifact_id": "A-TINY-BAD",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
        ],
        findings=[
            # Buggy E2 on a PARTIAL image!
            {"finding_id": "F-OVER-PARTIAL", "tier": "E2", "artifact_ids": ["A-PARTIAL", "WF-1"],
             "link_ids": ["L1"], "rationale": "bad"},
            # Buggy E2 on a 1x1 test PNG!
            {"finding_id": "F-OVER-TINY", "tier": "E2", "artifact_ids": ["A-TINY-BAD", "WF-2"],
             "link_ids": ["L2"], "rationale": "bad"},
        ],
        raw_files={"exports/raw/p.bin": b"partial-png-bytes"},
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {
                "job_id": "j1",
                "prompt_id": "p1",
                "outputs": [{
                    "output_id": "j1:0",
                    "raw_sha256": hashlib.sha256(b"partial-png-bytes").hexdigest(),
                    "max_allowed_tier": "E3",
                }],
            },
            {"job_id": "j2", "prompt_id": "p2", "outputs": []},
        ],
        "distractors": [
            {"distractor_id": "tiny-bad", "container_sha256": _sha("tiny-bad"), "allow_link_without_tier": True},
        ],
    }
    report = score_case(case, gt)
    assert report["tiers"]["over_tiered"] == 2
    assert report["metrics"]["gt_tier_over_tiered_rate"]["numerator"] == 2
    assert report["metrics"]["gt_tier_over_tiered_rate"]["denominator"] == 2
    checks = {c["finding_id"]: c for c in report["finding_checks"]}
    assert "E2_WITHOUT_COMPLETE_IMAGE" in checks["F-OVER-PARTIAL"]["issues"]
    assert "BELOW_MIN_OUTPUT_SIZE_TIERED" in checks["F-OVER-TINY"]["issues"]
    assert "DISTRACTOR_TIERED" in checks["F-OVER-TINY"]["issues"]


def test_effective_view_applies_image_links_and_regenerations_while_refusing_stale_sidecars(tmp_path):
    derived_bytes = b"fake-regenerated-png-bytes"
    derived_sha = hashlib.sha256(derived_bytes).hexdigest()
    case = _write_case(
        tmp_path,
        include_promotion_replays=True,
        artifacts=[
            _art("PIXBUF-001", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-a")),
            _art("PIXBUF-002", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-b")),
        ],
        workflows=[
            _wf("WFPY-001", prompt_id="job-a", completeness="COMPLETE"),
            _wf("WFPY-002", prompt_id="job-b", completeness="COMPLETE"),
            _wf("WFPY-003", prompt_id="job-c", completeness="COMPLETE"),
        ],
        links=[],
        findings=[
            {"finding_id": "FIND-001", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-001"],
             "link_ids": [], "rationale": "orphan"},
            {"finding_id": "FIND-002", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-002"],
             "link_ids": [], "rationale": "orphan"},
            {"finding_id": "FIND-003", "tier": "E5", "artifact_ids": ["WFPY-001"],
             "link_ids": [], "rationale": "prompt"},
            {"finding_id": "FIND-004", "tier": "E5", "artifact_ids": ["WFPY-002"],
             "link_ids": [], "rationale": "prompt"},
            {"finding_id": "FIND-005", "tier": "E5", "artifact_ids": ["WFPY-003"],
             "link_ids": [], "rationale": "prompt"},
        ],
        image_links={
            "schema_version": 1,
            "manifest_sha256": "AUTO",
            "memory_images": [
                {"artifact_id": "PIXBUF-001", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "job-a", "agreeing_jobs": ["job-a"]},
                {"artifact_id": "PIXBUF-002", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": False,
                 "agreement": "LOSSY", "attribution": "SUPPORTED", "prompt_id": "job-b", "agreeing_jobs": ["job-b"]},
            ],
            "promotions": [
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-a",
                 "artifact_id": "PIXBUF-001", "workflow_ids": ["WFPY-001"]},
                {"tier": "E3", "basis": "REGENERATION_CLOSE_MATCH", "prompt_id": "job-b",
                 "artifact_ids": ["PIXBUF-002"], "workflow_ids": ["WFPY-002"]},
            ],
        },
        regenerations=[
            {
                "handoff_id": "REGEN-001",
                "workflow_id": "WFPY-001",
                "target_manifest_sha256": "AUTO",
                "status": "COMPLETED",
                "security_audit": {"model_format": "SAFETENSORS"},
                "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
                "evaluation": {
                    "promoted_finding_tier": "E1",
                    "evaluated_tier": "E1",
                    "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
                    "evaluation_subtier": "EXACT_MATCH",
                    "exact_pixel_match": True,
                    "target_artifact_id": "PIXBUF-001",
                    "derived_regenerated_image_sha256": derived_sha,
                },
                "derived_regenerated_image": "exports/derived/DERIVED_regen_1.png",
            },
            {
                "handoff_id": "REGEN-002",
                "workflow_id": "WFPY-003",
                "target_manifest_sha256": "AUTO",
                "status": "COMPLETED",
                "security_audit": {"model_format": "SAFETENSORS"},
                "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
                "evaluation": {
                    "promoted_finding_tier": "E4",
                    "evaluated_tier": "E4",
                    "evaluation_mode": "NO_MEMORY_TARGET",
                    "evaluation_subtier": "EXECUTABLE_REPRODUCTION",
                    "target_artifact_id": None,
                    "derived_regenerated_image_sha256": derived_sha,
                },
                "derived_regenerated_image": "exports/derived/DERIVED_regen_1.png",
            },
        ],
        derived_files={"exports/derived/DERIVED_regen_1.png": derived_bytes},
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {"job_id": "ja", "prompt_id": "job-a", "expected_tier": "E1",
             "outputs": [{"output_id": "ja:0", "pixel_sha256": _sha("pix-a"), "expected_tier": "E1"}]},
            {"job_id": "jb", "prompt_id": "job-b", "expected_tier": "E3",
             "outputs": [{"output_id": "jb:0", "pixel_sha256": _sha("pix-b"),
                          "expected_tier": "E3", "max_allowed_tier": "E3"}]},
            {"job_id": "jc", "prompt_id": "job-c", "expected_tier": "E4", "outputs": []},
        ],
    }

    # Extraction view ignores image-links & regenerations
    ext = score_case(case, gt, view="extraction")
    assert ext["links"]["correct"] == 0
    assert ext["links"]["unassociated_gt_outputs"] == 2
    assert {o["output_id"]: o["achieved_tier"] for o in ext["outputs"]} == {"ja:0": "UNTIERED", "jb:0": "UNTIERED"}
    assert {j["job_id"]: j["achieved_tier"] for j in ext["jobs"]} == {"ja": "E5", "jb": "E5", "jc": "E5"}

    # Effective view sees E1, E3 close-match, and E4
    eff = score_case(case, gt, view="effective")
    assert eff["case"]["image_links_applied"] is True
    assert eff["links"]["correct"] == 2
    assert eff["links"]["wrong"] == 0
    assert {o["output_id"]: o["achieved_tier"] for o in eff["outputs"]} == {"ja:0": "E1", "jb:0": "E3"}
    assert {j["job_id"]: j["achieved_tier"] for j in eff["jobs"]} == {"ja": "E1", "jb": "E3", "jc": "E4"}
    assert eff["metrics"]["gt_tier_pass_rate"]["rate"] == 1.0
    assert eff["metrics"]["finding_policy_valid_rate"]["rate"] == 1.0

    # Stale image-links.json (mismatched manifest_sha256) is refused
    stale_links = json.loads((case / "image-links.json").read_text(encoding="utf-8"))
    stale_links["manifest_sha256"] = "0" * 64
    (case / "image-links.json").write_text(json.dumps(stale_links), encoding="utf-8")
    stale = score_case(case, gt, view="effective")
    assert stale["case"]["image_links_applied"] is False
    assert stale["case"]["image_links_ignored_mismatch"] is True
    assert stale["links"]["correct"] == 0


@pytest.mark.parametrize("content_stitched", [False, True])
def test_memory_only_view_excludes_file_backed_and_stitched_records_with_explicit_denominators(
    tmp_path, content_stitched,
):
    case = _write_case(
        tmp_path,
        artifacts=[
            _art("A-MEM", link_status="STRICT", associated_workflow_id="W-MEM", container_sha256=_sha("mem-img")),
            _art("A-FILE", link_status="STRICT", associated_workflow_id="W-FILE", container_sha256=_sha("file-img")),
        ],
        workflows=[
            _wf("W-MEM", prompt_id="p-mem", encoding_detected="CPYTHON_OBJECTS"),
            _wf("W-FILE", prompt_id="p-file",
                encoding_detected="PNG_TEXT_STITCHED" if content_stitched else "PNG_TEXT"),
        ],
        links=[
            {"link_id": "L1", "workflow_id": "W-MEM", "artifact_id": "A-MEM",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L2", "workflow_id": "W-FILE", "artifact_id": "A-FILE",
             "link_class": "STRICT", "matched_keys": ["EMBEDDED_PROMPT"]},
        ],
        findings=[
            {"finding_id": "F1", "tier": "E2", "artifact_ids": ["A-MEM", "W-MEM"],
             "link_ids": ["L1"], "rationale": "mem"},
            {"finding_id": "F2", "tier": "E2", "artifact_ids": ["A-FILE", "W-FILE"],
             "link_ids": ["L2"], "rationale": "file"},
        ],
        evidence_records=[
            {"record_id": "A-MEM", "kind": "image", "memory_source": "MEMORY"},
            {"record_id": "A-FILE", "kind": "image",
             **({"memory_source": "MEMORY_PARTIAL", "assembly": {"method": "CONTENT_STITCHED"}}
                if content_stitched else {"memory_source": "FILE_BACKED"})},
            {"record_id": "W-MEM", "kind": "workflow", "memory_source": "NOT_CLASSIFIED"},
            {"record_id": "W-FILE", "kind": "workflow",
             "memory_source": "MEMORY" if content_stitched else "FILE_BACKED"},
        ],
        memory_only_findings=[
            {"finding_id": "F1", "tier": "E2", "artifact_ids": ["A-MEM", "W-MEM"],
             "link_ids": ["L1"], "rationale": "mem"},
        ],
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {"job_id": "j-mem", "prompt_id": "p-mem",
             "outputs": [{"output_id": "o-mem", "container_sha256": _sha("mem-img"), "expected_tier": "E2"}]},
            {"job_id": "j-file", "prompt_id": "p-file",
             "outputs": [{"output_id": "o-file", "container_sha256": _sha("file-img"), "expected_tier": "E2"}]},
        ],
    }

    eff = score_case(case, gt, view="effective")
    mem = score_case(case, gt, view="memory_only")
    assert eff["metrics"]["output_correctly_linked_rate"]["numerator"] == 2
    assert eff["metrics"]["output_correctly_linked_rate"]["denominator"] == 2
    assert mem["metrics"]["output_correctly_linked_rate"]["numerator"] == 1
    assert mem["metrics"]["output_correctly_linked_rate"]["denominator"] == 2
    assert mem["metrics"]["job_recovery_rate"]["numerator"] == 1
    assert mem["metrics"]["job_recovery_rate"]["denominator"] == 2
    assert mem["tiers"]["under_tiered"] == 1


def test_zero_denominators_and_cli_smoke(tmp_path, capsys):
    case = _write_case(tmp_path, artifacts=[], workflows=[], links=[], findings=[])
    gt_path = tmp_path / "empty_gt.json"
    gt_path.write_text(json.dumps({"schema_version": 1, "dump_sha256": DUMP_SHA, "jobs": []}), encoding="utf-8")
    out_json = tmp_path / "scores" / "report.json"

    code = main(["--case-dir", str(case), "--groundtruth", str(gt_path), "--json-out", str(out_json)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "0/0 (n/a)" in captured
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    for metric in saved["metrics"].values():
        assert metric["denominator"] == 0
        assert metric["numerator"] == 0
        assert metric["rate"] is None
        assert metric["definition"]
    assert "ComfyTrace post-extraction evaluation" in format_summary(saved)


def test_scoring_requires_the_declared_acquisition_condition(tmp_path):
    case = _write_case(tmp_path, artifacts=[], workflows=[], links=[], findings=[])
    truth = {
        "schema_version": 1, "dump_sha256": DUMP_SHA, "jobs": [],
        "expected_carving_mode": "PROCESS_VIRTUAL", "expected_degradation_reason": None,
    }
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["case_metadata"]
    metadata["carving_mode"] = "PHYSICAL_FALLBACK"
    metadata["degradation_reason"] = "NO_COMFYUI_PROCESS"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="carving mode mismatch"):
        score_case(case, truth)

    truth["expected_carving_mode"] = "PHYSICAL_FALLBACK"
    with pytest.raises(ValueError, match="degradation reason mismatch"):
        score_case(case, truth)

    truth["expected_degradation_reason"] = "NO_COMFYUI_PROCESS"
    assert score_case(case, truth)["case"]["degradation_reason"] == "NO_COMFYUI_PROCESS"


# ---------------------------------------------------------------------------
# Regressions for audited false-pass risks
# ---------------------------------------------------------------------------


def test_configured_shared_groundtruth_dir_is_explicit_unless_case_opts_out(tmp_path):
    case = {"id": "case-a", "memory_filename": "a.raw", "groundtruth_filename": "case-a.groundtruth.json"}
    res_env = resolve_case_paths(case, env={ENV_GROUNDTRUTH_DIR: str(tmp_path / "missing_gt")})
    assert res_env.groundtruth_explicit is True
    assert res_env.groundtruth == tmp_path / "missing_gt" / "case-a.groundtruth.json"

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"groundtruth_dir": "gt_folder"}), encoding="utf-8")
    res_cfg = resolve_case_paths(case, env={ENV_CONFIG: str(cfg)})
    assert res_cfg.groundtruth_explicit is True
    assert res_cfg.groundtruth == tmp_path / "gt_folder" / "case-a.groundtruth.json"

    disabled_case = {"id": "case-no-gt", "memory_filename": "b.raw", "groundtruth_filename": None}
    res_disabled = resolve_case_paths(disabled_case, env={ENV_GROUNDTRUTH_DIR: str(tmp_path / "missing_gt")})
    assert res_disabled.groundtruth is None
    assert res_disabled.groundtruth_explicit is False


def test_a_shared_pixel_digest_cannot_credit_two_output_positions(tmp_path):
    digest = _sha("shared")
    case = _write_case(
        tmp_path, name="duplicate_batch_pixels",
        artifacts=[_art("A", pixel_sha256=digest, link_status="STRICT", associated_workflow_id="W")],
        workflows=[_wf("W", prompt_id="p")],
        links=[{"link_id": "L", "artifact_id": "A", "workflow_id": "W", "link_class": "STRICT"}],
        findings=[{"finding_id": "F", "tier": "E2", "artifact_ids": ["A", "W"], "link_ids": ["L"], "rationale": ""}],
    )
    ground_truth = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{"job_id": "j", "prompt_id": "p", "outputs": [
            {"output_id": f"b{index}", "output_node_id": "9", "batch_index": index, "pixel_sha256": digest}
            for index in (0, 1)
        ]}],
    }
    report = score_case(case, ground_truth)
    assert report["metrics"]["output_recovery_rate"]["numerator"] == 0
    assert report["metrics"]["output_correctly_linked_rate"]["numerator"] == 0
    assert [out["unresolved_artifact_ids"] for out in report["outputs"]] == [["A"], ["A"]]


def test_conflicting_output_nodes_do_not_earn_exact_attribution(tmp_path):
    digest = _sha("edited-output")
    case = _write_case(
        tmp_path,
        artifacts=[_art("A", link_status="STRICT", associated_workflow_id="W", pixel_sha256=digest)],
        workflows=[_wf("W", prompt_id="p")],
        links=[{"link_id": "L", "artifact_id": "A", "workflow_id": "W", "link_class": "STRICT"}],
        findings=[{"finding_id": "F", "tier": "E2", "artifact_ids": ["A", "W"], "link_ids": ["L"]}],
    )
    ground_truth = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{"job_id": "j", "prompt_id": "p", "outputs": [
            {"output_id": "edited", "output_node_id": "8", "batch_index": 0, "pixel_sha256": digest},
            {"output_id": "other", "output_node_id": "7", "batch_index": 0, "pixel_sha256": _sha("other-output")},
        ]}],
    }
    rows = [
        {"prompt_id": "p", "output_node_id": node, "batch_index": 0,
         "image": {"candidates": [{"artifact_id": "A", "role": "OUTPUT"}]}}
        for node in ("7", "8")
    ]
    (case / "scorecard.json").write_text(json.dumps({"dump_sha256": DUMP_SHA, "rows": rows}), encoding="utf-8")

    report = score_case(case, ground_truth)
    assert report["output_identity"]["matched"] == 0
    assert report["links"]["wrong_breakdown"]["same_job_output"] == 1


@pytest.mark.parametrize(("predicted_job", "expected_wrong"), [("edit", 0), ("unrelated", 1)])
def test_input_context_link_is_only_allowed_for_its_own_job(tmp_path, predicted_job, expected_wrong):
    digest = _sha("source-image")
    case = _write_case(
        tmp_path,
        artifacts=[_art("A", link_status="STRICT", associated_workflow_id="W", pixel_sha256=digest)],
        workflows=[_wf("W", prompt_id=predicted_job)],
        links=[{"link_id": "L", "artifact_id": "A", "workflow_id": "W", "link_class": "STRICT"}],
        findings=[{"finding_id": "F", "tier": "UNTIERED", "artifact_ids": ["A"], "link_ids": ["L"]}],
    )
    truth = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{"job_id": job, "prompt_id": job, "outputs": []} for job in ("edit", "unrelated")],
        "distractors": [{
            "distractor_id": "source",
            "pixel_sha256": digest,
            "allow_link_without_tier": True,
            "allowed_link_job_ids": ["edit"],
            "max_allowed_tier": "UNTIERED",
        }],
    }
    report = score_case(case, truth)
    assert report["links"]["wrong_breakdown"]["distractor_linked"] == expected_wrong
    assert report["links"]["allowed_untiered_distractor_links"] == 1 - expected_wrong


def test_same_job_different_outputs_identity_matching_mismatch_and_unknown_surfacing(tmp_path):
    """Scores known output_node_id/batch_index/per-raster digest, catches same-job wrong-output false passes,
    and explicitly surfaces unknown predicted/GT identities and output-level ambiguity."""
    case = _write_case(
        tmp_path,
        name="same_job_outputs",
        include_promotion_replays=True,
        artifacts=[
            # 1. Exact match on node 9, batch 0, raster r0 -> CORRECT + MATCHED
            _art("PIXBUF-001", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-0")),
            # 2. Same job p1, WRONGLY attributed to batch 0 / raster r0 instead of batch 1 / r1
            _art("PIXBUF-002", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-1")),
            # 3. Same job p1, job-only link (no node/batch/raster) -> CORRECT at job level, UNKNOWN_PREDICTED surfaced
            _art("CARVE-PNG-001", link_status="STRICT", associated_workflow_id="WFPY-001",
                 container_sha256=_sha("png-2"), pixel_sha256=_sha("pix-2")),
            # 4. Same job p1, attribution supplies node/batch/raster, GT omits them -> UNKNOWN_GROUNDTRUTH surfaced
            _art("PIXBUF-003", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-3")),
            # 5. Same job p1, output-level AMBIGUOUS across batch 0 and batch 1 -> AMBIGUOUS_SAME_JOB_OUTPUT
            _art("PIXBUF-004", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-4")),
        ],
        workflows=[_wf("WFPY-001", prompt_id="p1", canonical_json_sha256=_sha("g1"))],
        links=[
            {"link_id": "L1", "workflow_id": "WFPY-001", "artifact_id": "CARVE-PNG-001",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
        ],
        findings=[
            {"finding_id": "F1", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-001"], "link_ids": [], "rationale": "o"},
            {"finding_id": "F2", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-002"], "link_ids": [], "rationale": "o"},
            {"finding_id": "F3", "tier": "E2", "artifact_ids": ["CARVE-PNG-001", "WFPY-001"], "link_ids": ["L1"],
             "rationale": "ok"},
            {"finding_id": "F4", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-003"], "link_ids": [], "rationale": "o"},
            {"finding_id": "F5", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-004"], "link_ids": [], "rationale": "o"},
        ],
        image_links={
            "schema_version": 1,
            "manifest_sha256": "AUTO",
            "memory_images": [
                {"artifact_id": "PIXBUF-001", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p1", "agreeing_jobs": ["p1"],
                 "output_attribution": "SUPPORTED",
                 "selected_raster": {"node_id": "9", "batch_index": 0, "sha256": _sha("raster-0")}},
                {"artifact_id": "PIXBUF-002", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p1", "agreeing_jobs": ["p1"],
                 "output_node_id": "9", "batch_index": 0, "raster_sha256": _sha("raster-0")},
                {"artifact_id": "PIXBUF-003", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p1", "agreeing_jobs": ["p1"],
                 "output_node_id": "10", "batch_index": 0, "raster_sha256": _sha("raster-3")},
                {"artifact_id": "PIXBUF-004", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p1", "agreeing_jobs": ["p1"],
                 "output_attribution": "AMBIGUOUS",
                 "agreeing_outputs": [
                     {"output_node_id": "9", "batch_index": 0},
                     {"output_node_id": "9", "batch_index": 1},
                 ]},
            ],
            "promotions": [
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p1",
                 "artifact_id": "PIXBUF-001", "workflow_ids": ["WFPY-001"]},
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p1",
                 "artifact_id": "PIXBUF-002", "workflow_ids": ["WFPY-001"]},
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "p1",
                 "artifact_id": "PIXBUF-003", "workflow_ids": ["WFPY-001"]},
            ],
        },
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{
            "job_id": "job-1",
            "prompt_id": "p1",
            "outputs": [
                {"output_id": "job-1:b0", "output_node_id": "9", "batch_index": 0,
                 "pixel_sha256": _sha("pix-0"), "raster_sha256": _sha("raster-0"), "expected_tier": "E2"},
                {"output_id": "job-1:b1", "output_node_id": "9", "batch_index": 1,
                 "pixel_sha256": _sha("pix-1"), "raster_sha256": _sha("raster-1"), "expected_tier": "E2"},
                {"output_id": "job-1:b2", "output_node_id": "9", "batch_index": 2,
                 "container_sha256": _sha("png-2"), "expected_tier": "E2"},
                {"output_id": "job-1:b3", "pixel_sha256": _sha("pix-3")},
                {"output_id": "job-1:b4", "output_node_id": "9", "batch_index": 4,
                 "pixel_sha256": _sha("pix-4")},
            ],
        }],
    }

    report = score_case(case, gt)
    arts = {a["artifact_id"]: a for a in report["artifacts"]}
    assert arts["PIXBUF-001"]["link_verdict"] == "CORRECT"
    assert arts["PIXBUF-001"]["output_identity_status"] == "MATCHED"
    assert arts["PIXBUF-002"]["link_verdict"] == "WRONG"
    assert arts["PIXBUF-002"]["link_subreason"] == "WRONG_SAME_JOB_OUTPUT"
    assert arts["PIXBUF-002"]["output_identity_status"] == "MISMATCHED"
    assert arts["CARVE-PNG-001"]["link_verdict"] == "CORRECT"
    assert arts["CARVE-PNG-001"]["output_identity_status"] == "UNKNOWN_PREDICTED"
    assert arts["PIXBUF-003"]["link_verdict"] == "CORRECT"
    assert arts["PIXBUF-003"]["output_identity_status"] == "UNKNOWN_GROUNDTRUTH"
    assert arts["PIXBUF-004"]["link_verdict"] == "AMBIGUOUS"
    assert arts["PIXBUF-004"]["link_subreason"] == "AMBIGUOUS_SAME_JOB_OUTPUT"
    assert arts["PIXBUF-004"]["output_identity_status"] == "AMBIGUOUS"

    assert report["links"]["wrong"] == 1
    assert report["links"]["wrong_breakdown"]["same_job_output"] == 1
    assert report["links"]["ambiguous_breakdown"]["same_job_output"] == 1
    assert report["output_identity"] == {
        "job_matched_artifacts": 4,
        "matched": 1,
        "mismatched": 1,
        "ambiguous": 1,
        "unknown_predicted": 1,
        "unknown_groundtruth": 1,
    }
    assert report["metrics"]["output_identity_known_rate"]["numerator"] == 2
    assert report["metrics"]["output_identity_known_rate"]["denominator"] == 4
    assert report["metrics"]["output_identity_accuracy"]["numerator"] == 1
    assert report["metrics"]["output_identity_accuracy"]["denominator"] == 2

    # Same-job wrong-output artifact must NOT earn E2 for job-1:b1 and invalidates its finding.
    b1 = next(o for o in report["outputs"] if o["output_id"] == "job-1:b1")
    assert b1["achieved_tier"] is None
    assert b1["tier_check"]["status"] == "UNDER_TIERED"
    f2 = next(f for f in report["finding_checks"] if f["finding_id"] == "F2")
    assert "WRONG_LINK_TIER" in f2["issues"]


def test_scorecard_output_tie_mismatch_and_mixed_outputs_in_single_finding_caught(tmp_path):
    """Scorecard role=='OUTPUT' mismatch catches wrong-batch tensor, and conflating two GT outputs in one finding
    triggers MIXED_GT_OUTPUTS_IN_FINDING without letting the partial second output inherit E2."""
    case = _write_case(
        tmp_path,
        name="mixed_outputs",
        artifacts=[
            _art("CARVE-PNG-001", completeness="COMPLETE", link_status="STRICT", associated_workflow_id="WF-1",
                 container_sha256=_sha("out0")),
            _art("BITMAP-001", carve_type="DECODED_BITMAP", completeness="PARTIAL", link_status="SUPPORTED",
                 associated_workflow_id="WF-1", pixel_sha256=_sha("out1")),
            _art("TENSOR-001", carve_type="TENSOR_IMAGE", completeness="COMPLETE", link_status="STRICT",
                 associated_workflow_id="WF-1", pixel_sha256=_sha("out2")),
        ],
        workflows=[_wf("WF-1", prompt_id="p1")],
        links=[
            {"link_id": "L1", "workflow_id": "WF-1", "artifact_id": "CARVE-PNG-001",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L2", "workflow_id": "WF-1", "artifact_id": "BITMAP-001",
             "link_class": "SUPPORTED", "matched_keys": ["SEED"]},
            {"link_id": "L3", "workflow_id": "WF-1", "artifact_id": "TENSOR-001",
             "link_class": "STRICT", "matched_keys": ["EXECUTION_CACHE_SIGNATURE"]},
        ],
        findings=[
            # Buggy finding mixing two distinct GT outputs (out0 COMPLETE + out1 PARTIAL)!
            {"finding_id": "F-MIX", "tier": "E2", "artifact_ids": ["CARVE-PNG-001", "BITMAP-001", "WF-1"],
             "link_ids": ["L1", "L2"], "rationale": "mixed"},
            {"finding_id": "F-TENS", "tier": "E2", "artifact_ids": ["TENSOR-001", "WF-1"],
             "link_ids": ["L3"], "rationale": "tensor"},
        ],
    )
    # Scorecard attributes TENSOR-001 to node "9" batch 0, while GT says out2 is node "10" batch 1.
    (case / "scorecard.json").write_text(json.dumps({
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "rows": [{
            "prompt_id": "p1",
            "output_node_id": "9",
            "batch_index": 0,
            "image": {"status": "LINKED", "candidates": [
                {"artifact_id": "TENSOR-001", "basis": "CORRELATION_STRICT", "role": "OUTPUT", "batch_index": 0},
            ]},
        }],
    }), encoding="utf-8")
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{
            "job_id": "j1",
            "prompt_id": "p1",
            "outputs": [
                {"output_id": "j1:out0", "container_sha256": _sha("out0"), "expected_tier": "E2"},
                {"output_id": "j1:out1", "pixel_sha256": _sha("out1"), "max_allowed_tier": "E3"},
                {"output_id": "j1:out2", "output_node_id": "10", "batch_index": 1, "pixel_sha256": _sha("out2")},
            ],
        }],
    }
    report = score_case(case, gt)
    checks = {c["finding_id"]: c for c in report["finding_checks"]}
    assert "MIXED_GT_OUTPUTS_IN_FINDING" in checks["F-MIX"]["issues"]
    assert "WRONG_LINK_TIER" in checks["F-TENS"]["issues"]
    outs = {o["output_id"]: o for o in report["outputs"]}
    assert outs["j1:out1"]["achieved_tier"] is None
    assert outs["j1:out1"]["observed_tier"] == "E2"
    assert outs["j1:out1"]["tier_check"]["status"] == "OVER_TIERED"
    assert report["links"]["wrong_breakdown"]["same_job_output"] == 1


def test_ambiguous_overconfident_unmapped_and_disjoint_prompt_id_regressions(tmp_path):
    """Catches single-job expected_ambiguous false pass, disjoint prompt_id hash reuse, invalid SUPPORTED with
    >1 agreeing_jobs, conflicting mapped-vs-unmapped sources, and partial AMBIGUOUS coverage."""
    shared_graph = _sha("shared-prompt-graph")
    case = _write_case(
        tmp_path,
        name="link_edge_cases",
        artifacts=[
            # 1. Single-job output marked expected_ambiguous=True, over-confidently STRICT-linked -> WRONG
            _art("A-OVERCONF", link_status="STRICT", associated_workflow_id="WF-1",
                 container_sha256=_sha("o-overconf")),
            # 2. Linked to WF-FOREIGN (disjoint prompt_id "p-other", same canonical hash) -> WRONG_UNMATCHED_WORKFLOW
            _art("A-DISJOINT", link_status="STRICT", associated_workflow_id="WF-FOREIGN",
                 container_sha256=_sha("o-disjoint")),
            # 3. image-links SUPPORTED with >1 agreeing_jobs -> WRONG_INVALID_ATTRIBUTION_RULE
            _art("PIXBUF-BAD-RULE", carve_type="PIXEL_BUFFER", pixel_sha256=_sha("o-bad-rule")),
            # 4. Extraction STRICT->WF-1 (mapped), image-links SUPPORTED->p-unmapped -> WRONG_CONFLICTING_SOURCES
            _art("TENSOR-CONFLICT", carve_type="TENSOR_IMAGE", link_status="STRICT", associated_workflow_id="WF-1",
                 pixel_sha256=_sha("o-conflict")),
            # 5. Matches 2 GT jobs (j2, j3), AMBIGUOUS lists WF-2 + WF-FOREIGN (misses j3) -> NOT AMBIGUOUS_EXPECTED
            _art("A-PARTIAL-AMBIG", link_status="AMBIGUOUS", ambiguous_candidate_workflows=["WF-2", "WF-FOREIGN"],
                 pixel_sha256=_sha("o-shared")),
        ],
        workflows=[
            _wf("WF-1", prompt_id="p1", canonical_json_sha256=shared_graph),
            _wf("WF-2", prompt_id="p2", canonical_json_sha256=_sha("g2")),
            _wf("WF-FOREIGN", prompt_id="p-foreign", canonical_json_sha256=shared_graph),
        ],
        links=[
            {"link_id": "L1", "workflow_id": "WF-1", "artifact_id": "A-OVERCONF",
             "link_class": "STRICT", "matched_keys": ["CANONICAL_PROMPT_HASH"]},
            {"link_id": "L2", "workflow_id": "WF-FOREIGN", "artifact_id": "A-DISJOINT",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L3", "workflow_id": "WF-1", "artifact_id": "TENSOR-CONFLICT",
             "link_class": "STRICT", "matched_keys": ["EXECUTION_CACHE_SIGNATURE"]},
            {"link_id": "L4", "workflow_id": "WF-2", "artifact_id": "A-PARTIAL-AMBIG",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L5", "workflow_id": "WF-FOREIGN", "artifact_id": "A-PARTIAL-AMBIG",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
        ],
        findings=[
            {"finding_id": "F1", "tier": "E2", "artifact_ids": ["A-OVERCONF", "WF-1"],
             "link_ids": ["L1"], "rationale": "r"},
            {"finding_id": "F2", "tier": "E2", "artifact_ids": ["A-DISJOINT", "WF-FOREIGN"],
             "link_ids": ["L2"], "rationale": "r"},
            {"finding_id": "F3", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-BAD-RULE"],
             "link_ids": [], "rationale": "r"},
            {"finding_id": "F4", "tier": "E2", "artifact_ids": ["TENSOR-CONFLICT", "WF-1"],
             "link_ids": ["L3"], "rationale": "r"},
            {"finding_id": "F5", "tier": "UNTIERED", "artifact_ids": ["A-PARTIAL-AMBIG"],
             "link_ids": ["L4", "L5"], "rationale": "r"},
        ],
        image_links={
            "schema_version": 1,
            "manifest_sha256": "AUTO",
            "memory_images": [
                {"artifact_id": "PIXBUF-BAD-RULE", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p1", "agreeing_jobs": ["p1", "p2"]},
                {"artifact_id": "TENSOR-CONFLICT", "kind": "TENSOR_IMAGE", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "p-unmapped",
                 "agreeing_jobs": ["p-unmapped"]},
            ],
            "promotions": [],
        },
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {
                "job_id": "j1",
                "prompt_id": "p1",
                "canonical_prompt_sha256": shared_graph,
                "outputs": [
                    {"output_id": "j1:0", "container_sha256": _sha("o-overconf"), "expected_ambiguous": True},
                    {"output_id": "j1:1", "container_sha256": _sha("o-disjoint")},
                    {"output_id": "j1:2", "pixel_sha256": _sha("o-bad-rule")},
                    {"output_id": "j1:3", "pixel_sha256": _sha("o-conflict")},
                ],
            },
            {"job_id": "j2", "prompt_id": "p2", "outputs": [{"output_id": "j2:0", "pixel_sha256": _sha("o-shared")}]},
            {"job_id": "j3", "prompt_id": "p3", "outputs": [{"output_id": "j3:0", "pixel_sha256": _sha("o-shared")}]},
        ],
    }

    report = score_case(case, gt)
    arts = {a["artifact_id"]: a for a in report["artifacts"]}
    assert arts["A-OVERCONF"]["link_subreason"] == "WRONG_OVERCONFIDENT_AMBIGUOUS"
    assert arts["A-DISJOINT"]["link_subreason"] == "WRONG_UNMATCHED_WORKFLOW"
    assert arts["PIXBUF-BAD-RULE"]["link_subreason"] == "WRONG_INVALID_ATTRIBUTION_RULE"
    assert arts["TENSOR-CONFLICT"]["link_subreason"] == "WRONG_CONFLICTING_SOURCES"
    assert arts["A-PARTIAL-AMBIG"]["link_subreason"] == "AMBIGUOUS_INCLUDES_TRUE_JOB"
    assert arts["A-PARTIAL-AMBIG"]["unmapped_targets"] == ["WF-FOREIGN"]
    assert report["links"]["correct"] == 0
    assert report["links"]["wrong"] == 4


def test_promotion_correctness_catches_invalid_e2_e1_cross_job_and_policy_invalid_credit(tmp_path):
    """Catches invalid E2/E1 promotions (bitmap promoted to E2 then E1, cross-job promotion), prevents policy-invalid
    findings from passing expected_tier, catches over-tiered AMBIGUOUS, and enforces MEMORY_PARTIAL in memory_only."""
    derived_bytes = b"fake-regen"
    derived_sha = hashlib.sha256(derived_bytes).hexdigest()
    case = _write_case(
        tmp_path,
        name="bad_promotions",
        include_promotion_replays=True,
        artifacts=[
            # 1. DECODED_BITMAP illegally promoted to E2 (REGENERATION_WHOLE_FRAME_MATCH) then E1!
            _art("BITMAP-BAD-E2", carve_type="DECODED_BITMAP", completeness="COMPLETE", pixel_sha256=_sha("bmp-1")),
            # 2. PIXBUF of job-a illegally promoted with WFPY-B of job-b!
            _art("PIXBUF-CROSS", carve_type="PIXEL_BUFFER", completeness="COMPLETE", pixel_sha256=_sha("pix-cross")),
            # 3. PARTIAL image given E2 when GT expects E2 -> must NOT pass expected_tier!
            _art("A-PARTIAL-E2", completeness="PARTIAL", link_status="STRICT", associated_workflow_id="WFPY-C",
                 raw_file_paths=["exports/raw/partial.bin"]),
            # 4. AMBIGUOUS image wrongly placed in E2 when GT sets max_allowed_tier="UNTIERED" -> OVER_TIERED!
            _art("A-AMBIG-E2", completeness="COMPLETE", link_status="AMBIGUOUS",
                 ambiguous_candidate_workflows=["WFPY-A", "WFPY-B"], container_sha256=_sha("ambig-e2")),
        ],
        workflows=[
            _wf("WFPY-A", prompt_id="job-a", completeness="COMPLETE"),
            _wf("WFPY-B", prompt_id="job-b", completeness="COMPLETE"),
            _wf("WFPY-C", prompt_id="job-c", completeness="COMPLETE"),
        ],
        links=[
            {"link_id": "L1", "workflow_id": "WFPY-C", "artifact_id": "A-PARTIAL-E2",
             "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]},
            {"link_id": "L2", "workflow_id": "WFPY-A", "artifact_id": "A-AMBIG-E2",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
            {"link_id": "L3", "workflow_id": "WFPY-B", "artifact_id": "A-AMBIG-E2",
             "link_class": "AMBIGUOUS", "matched_keys": ["SEED"]},
        ],
        findings=[
            {"finding_id": "F-BMP", "tier": "UNTIERED", "artifact_ids": ["BITMAP-BAD-E2"],
             "link_ids": [], "rationale": "o"},
            {"finding_id": "F-CROSS", "tier": "UNTIERED", "artifact_ids": ["PIXBUF-CROSS"],
             "link_ids": [], "rationale": "o"},
            {"finding_id": "F-WFA", "tier": "E5", "artifact_ids": ["WFPY-A"], "link_ids": [], "rationale": "p"},
            {"finding_id": "F-WFB", "tier": "E5", "artifact_ids": ["WFPY-B"], "link_ids": [], "rationale": "p"},
            {"finding_id": "F-PARTIAL", "tier": "E2", "artifact_ids": ["A-PARTIAL-E2", "WFPY-C"], "link_ids": ["L1"],
             "rationale": "buggy e2 on partial"},
            {"finding_id": "F-AMBIG", "tier": "E2", "artifact_ids": ["A-AMBIG-E2", "WFPY-A"], "link_ids": ["L2", "L3"],
             "rationale": "buggy e2 on ambig"},
        ],
        evidence_records=[
            {"record_id": "A-PARTIAL-E2", "kind": "image", "memory_source": "MEMORY_PARTIAL"},
            {"record_id": "WFPY-C", "kind": "workflow", "memory_source": "MEMORY"},
        ],
        memory_only_findings=[
            {"finding_id": "F-PARTIAL", "tier": "E2", "artifact_ids": ["A-PARTIAL-E2", "WFPY-C"], "link_ids": ["L1"],
             "rationale": "buggy e2 on memory_partial"},
        ],
        image_links={
            "schema_version": 1,
            "manifest_sha256": "AUTO",
            "memory_images": [
                {"artifact_id": "BITMAP-BAD-E2", "kind": "DECODED_BITMAP", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "job-a", "agreeing_jobs": ["job-a"]},
                {"artifact_id": "PIXBUF-CROSS", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "job-a", "agreeing_jobs": ["job-a"]},
            ],
            "promotions": [
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-a",
                 "artifact_id": "BITMAP-BAD-E2", "workflow_ids": ["WFPY-A"]},
                {"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": "job-a",
                 "artifact_id": "PIXBUF-CROSS", "workflow_ids": ["WFPY-B"]},
            ],
        },
        regenerations=[{
            "handoff_id": "REGEN-001",
            "workflow_id": "WFPY-A",
            "target_manifest_sha256": "AUTO",
            "status": "COMPLETED",
            "security_audit": {"model_format": "SAFETENSORS"},
            "preflight_check": {"status": "PASSED", "runner_object_info_verified": True},
            "evaluation": {
                "promoted_finding_tier": "E1",
                "evaluated_tier": "E1",
                "evaluation_mode": "INTACT_NATIVE_RESOLUTION",
                "evaluation_subtier": "EXACT_MATCH",
                "exact_pixel_match": True,
                "target_artifact_id": "BITMAP-BAD-E2",
                "derived_regenerated_image_sha256": derived_sha,
            },
            "derived_regenerated_image": "exports/derived/DERIVED_regen_1.png",
        }],
        derived_files={"exports/derived/DERIVED_regen_1.png": derived_bytes},
        raw_files={"exports/raw/partial.bin": b"partial-bytes"},
    )
    gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [
            {"job_id": "ja", "prompt_id": "job-a", "outputs": [
                {"output_id": "ja:bmp", "pixel_sha256": _sha("bmp-1"), "expected_tier": "E1"},
                {"output_id": "ja:cross", "pixel_sha256": _sha("pix-cross"), "expected_tier": "E2"},
                {"output_id": "ja:ambig", "container_sha256": _sha("ambig-e2"), "max_allowed_tier": "UNTIERED"},
            ]},
            {"job_id": "jb", "prompt_id": "job-b", "outputs": []},
            {"job_id": "jc", "prompt_id": "job-c", "outputs": [
                {"output_id": "jc:partial", "raw_sha256": hashlib.sha256(b"partial-bytes").hexdigest(),
                 "expected_tier": "E2"},
            ]},
        ],
    }

    report = score_case(case, gt, view="effective")
    checks = {c["finding_id"]: c for c in report["finding_checks"]}
    assert "E2_CLOSE_OR_LOSSY_IMAGE_LINK" in checks["F-BMP"]["issues"]
    assert "E2_PROMOTION_PROMPT_MISMATCH" in checks["F-CROSS"]["issues"]
    assert "FINDING_JOB_MISMATCH" in checks["F-CROSS"]["issues"]
    assert "E2_WITHOUT_COMPLETE_IMAGE" in checks["F-PARTIAL"]["issues"]
    assert "AMBIGUOUS_LINK_TIER" in checks["F-AMBIG"]["issues"]
    assert report["promotions"]["total"] == 2
    assert report["promotions"]["invalid"] == 2
    assert report["metrics"]["promotion_valid_rate"]["rate"] == 0.0

    outs = {o["output_id"]: o for o in report["outputs"]}
    # Policy-invalid findings must NOT earn expected_tier credit!
    assert outs["ja:bmp"]["achieved_tier"] is None
    assert outs["ja:bmp"]["tier_check"]["status"] == "UNDER_TIERED"
    assert outs["jc:partial"]["achieved_tier"] is None
    assert outs["jc:partial"]["tier_check"]["status"] == "UNDER_TIERED"
    # Over-tiered AMBIGUOUS finding MUST trigger OVER_TIERED!
    assert outs["ja:ambig"]["observed_tier"] == "E2"
    assert outs["ja:ambig"]["tier_check"]["status"] == "OVER_TIERED"

    # CLI --fail-on-over-tier exits 2 on invalid findings/promotions.
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    assert main(["--case-dir", str(case), "--groundtruth", str(gt_path), "--fail-on-over-tier"]) == 2

    # memory_only view downgrades MEMORY_PARTIAL completeness to PARTIAL.
    mem_case = _write_case(
        tmp_path,
        name="mem_partial_downgrade",
        artifacts=[
            _art("A-MEM-PART", completeness="COMPLETE", link_status="STRICT", associated_workflow_id="WF-M",
                 container_sha256=_sha("mem-part")),
        ],
        workflows=[_wf("WF-M", prompt_id="pm", completeness="COMPLETE")],
        links=[{"link_id": "LM", "workflow_id": "WF-M", "artifact_id": "A-MEM-PART",
                "link_class": "STRICT", "matched_keys": ["PROMPT_ID"]}],
        findings=[{"finding_id": "FM", "tier": "E2", "artifact_ids": ["A-MEM-PART", "WF-M"],
                   "link_ids": ["LM"], "rationale": "full"}],
        evidence_records=[
            {"record_id": "A-MEM-PART", "kind": "image", "memory_source": "MEMORY_PARTIAL"},
            {"record_id": "WF-M", "kind": "workflow", "memory_source": "MEMORY"},
        ],
        memory_only_findings=[
            {"finding_id": "FM", "tier": "E2", "artifact_ids": ["A-MEM-PART", "WF-M"],
             "link_ids": ["LM"], "rationale": "invalid e2 on memory_partial"},
        ],
    )
    mem_gt = {
        "schema_version": 1,
        "dump_sha256": DUMP_SHA,
        "jobs": [{"job_id": "jm", "prompt_id": "pm",
                  "outputs": [{"output_id": "jm:0", "container_sha256": _sha("mem-part"), "max_allowed_tier": "E3"}]}],
    }
    mem_report = score_case(mem_case, mem_gt, view="memory_only")
    assert "E2_WITHOUT_COMPLETE_IMAGE" in mem_report["finding_checks"][0]["issues"]
    assert mem_report["tiers"]["over_tiered"] == 1


# ---------------------------------------------------------------------------
# Independent extraction corroboration (self-described PNG metadata earns no credit)
# ---------------------------------------------------------------------------


def _strict_link(link_id: str, workflow_id: str, artifact_id: str, keys: list[str] | None) -> dict[str, Any]:
    link: dict[str, Any] = {
        "link_id": link_id, "workflow_id": workflow_id, "artifact_id": artifact_id, "link_class": "STRICT",
    }
    if keys is not None:
        link["matched_keys"] = keys
    return link


def _own_png_wf(workflow_id: str, prompt_id: str, artifact_id: str) -> dict[str, Any]:
    wf = _wf(workflow_id, prompt_id=prompt_id, encoding_detected="PNG_TEXT")
    wf["execution_parameters"]["source_locations"] = [{"png": artifact_id}]
    return wf


def _one_output_job(tag: str) -> dict[str, Any]:
    return {"job_id": f"j-{tag}", "prompt_id": f"p-{tag}",
            "outputs": [{"output_id": f"o-{tag}", "container_sha256": _sha(f"img-{tag}")}]}


def test_self_described_png_is_legacy_correct_but_earns_no_independent_extraction_credit(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[
            _art("A-SELF", link_status="STRICT", associated_workflow_id="W-SELF", container_sha256=_sha("img-self")),
            _art("A-OWNID", link_status="STRICT", associated_workflow_id="W-OWNID", container_sha256=_sha("img-ownid")),
            _art("A-IND", link_status="STRICT", associated_workflow_id="W-IND", container_sha256=_sha("img-ind")),
            _art("A-UNV", link_status="STRICT", associated_workflow_id="W-UNV", container_sha256=_sha("img-unv")),
            _art("A-NOKEYS", link_status="STRICT", associated_workflow_id="W-NOKEYS",
                 container_sha256=_sha("img-nokeys")),
            _art("A-BOTH", link_status="STRICT", associated_workflow_id="W-BOTH-MEM",
                 container_sha256=_sha("img-both")),
            _art("A-AMB", link_status="AMBIGUOUS", ambiguous_candidate_workflows=["W-AMB"],
                 container_sha256=_sha("img-amb")),
        ],
        workflows=[
            _own_png_wf("W-SELF", "p-self", "A-SELF"),
            _own_png_wf("W-OWNID", "p-ownid", "A-OWNID"),
            _wf("W-IND", prompt_id="p-ind"),
            _wf("W-UNV", prompt_id="p-unv"),
            _wf("W-NOKEYS", prompt_id="p-nokeys"),
            _own_png_wf("W-BOTH-PNG", "p-both", "A-BOTH"),
            _wf("W-BOTH-MEM", prompt_id="p-both"),
            _wf("W-AMB", prompt_id="p-amb"),
        ],
        links=[
            _strict_link("L1", "W-SELF", "A-SELF", ["EMBEDDED_PROMPT"]),
            _strict_link("L2", "W-OWNID", "A-OWNID", ["PROMPT_ID"]),
            _strict_link("L3", "W-IND", "A-IND", ["PROMPT_ID"]),
            _strict_link("L4", "W-UNV", "A-UNV", ["EXECUTION_GRAPH_MATCH", "INPUT_REVISION_UNVERIFIED"]),
            _strict_link("L5", "W-NOKEYS", "A-NOKEYS", None),
            _strict_link("L6", "W-BOTH-PNG", "A-BOTH", ["EMBEDDED_PROMPT"]),
            _strict_link("L7", "W-BOTH-MEM", "A-BOTH", ["CANONICAL_PROMPT_HASH"]),
            {"link_id": "L8", "workflow_id": "W-AMB", "artifact_id": "A-AMB", "link_class": "AMBIGUOUS",
             "matched_keys": ["PROMPT_ID"]},
        ],
        findings=[],
    )
    gt = {"schema_version": 1, "dump_sha256": DUMP_SHA,
          "jobs": [_one_output_job(t) for t in ("self", "ownid", "ind", "unv", "nokeys", "both", "amb")]}

    report = score_case(case, gt, view="extraction")
    metrics = report["metrics"]
    # Legacy metric: every strict, same-job link is CORRECT, including the self-described one.
    assert metrics["output_correctly_linked_rate"]["numerator"] == 6
    assert metrics["output_correctly_linked_rate"]["denominator"] == 7
    assert metrics["link_precision"]["rate"] == 1.0
    assert report["links"]["wrong"] == 0
    # New metric: only separately recovered strict evidence counts.
    independent = {o["output_id"]: o["independently_corroborated"] for o in report["outputs"]}
    assert independent == {
        "o-self": False, "o-ownid": False, "o-ind": True, "o-unv": False,
        "o-nokeys": False, "o-both": True, "o-amb": False,
    }
    rate = metrics["output_independently_corroborated_rate"]
    assert (rate["numerator"], rate["denominator"]) == (2, 7)
    given = metrics["output_independently_corroborated_given_correct_rate"]
    assert (given["numerator"], given["denominator"]) == (2, 6)
    reasons = {a["artifact_id"]: a["independent_extraction_reason"] for a in report["artifacts"]}
    assert reasons["A-SELF"] == "SELF_DESCRIBED_OR_UNVERIFIED_INPUT"
    assert reasons["A-OWNID"] == "OWN_EMBEDDED_METADATA"
    assert reasons["A-UNV"] == "SELF_DESCRIBED_OR_UNVERIFIED_INPUT"
    assert reasons["A-NOKEYS"] == "NO_INDEPENDENT_KEYS"
    assert reasons["A-IND"] == reasons["A-BOTH"] == "INDEPENDENT_STRICT_EXTRACTION"
    assert reasons["A-AMB"] == "NOT_CORRECT_GT_OUTPUT_LINK"


def test_a_png_linked_only_to_its_own_metadata_has_zero_independent_credit_in_every_view(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[_art("A-SELF", link_status="STRICT", associated_workflow_id="W-SELF",
                        container_sha256=_sha("img-self"))],
        workflows=[_own_png_wf("W-SELF", "p-self", "A-SELF")],
        links=[_strict_link("L1", "W-SELF", "A-SELF", ["EMBEDDED_PROMPT"])],
        findings=[],
        evidence_records=[{"record_id": "A-SELF", "kind": "image", "memory_source": "MEMORY"},
                          {"record_id": "W-SELF", "kind": "workflow", "memory_source": "MEMORY"}],
        memory_only_findings=[],
    )
    gt = {"schema_version": 1, "dump_sha256": DUMP_SHA, "jobs": [_one_output_job("self")]}
    for view in ("extraction", "effective", "memory_only"):
        report = score_case(case, gt, view=view)
        assert report["metrics"]["output_correctly_linked_rate"]["numerator"] == 1
        independent = report["metrics"]["output_independently_corroborated_rate"]
        assert (independent["numerator"], independent["denominator"]) == (0, 1)
        assert report["metrics"]["output_independently_corroborated_given_correct_rate"]["rate"] == 0.0


def test_regeneration_only_match_earns_no_independent_extraction_credit(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[_art("PIXBUF-001", carve_type="PIXEL_BUFFER", pixel_sha256=_sha("pix-a"))],
        workflows=[_wf("WFPY-001", prompt_id="job-a")],
        links=[],
        findings=[],
        image_links={
            "schema_version": 1, "manifest_sha256": "AUTO",
            "memory_images": [
                {"artifact_id": "PIXBUF-001", "kind": "PIXEL_BUFFER", "complete": True, "whole_frame": True,
                 "agreement": "ROWS", "attribution": "SUPPORTED", "prompt_id": "job-a", "agreeing_jobs": ["job-a"]},
            ],
            "promotions": [],
        },
    )
    gt = {"schema_version": 1, "dump_sha256": DUMP_SHA,
          "jobs": [{"job_id": "ja", "prompt_id": "job-a",
                    "outputs": [{"output_id": "ja:0", "pixel_sha256": _sha("pix-a")}]}]}
    report = score_case(case, gt, view="effective")
    assert report["metrics"]["output_correctly_linked_rate"]["numerator"] == 1
    assert report["metrics"]["output_independently_corroborated_rate"]["numerator"] == 0
    assert report["artifacts"][0]["independent_extraction_reason"] == "NOT_EXTRACTION_STRICT"


def test_independent_corroboration_respects_memory_only_view_and_keeps_full_denominator(tmp_path):
    case = _write_case(
        tmp_path,
        artifacts=[
            _art("A-MEM", link_status="STRICT", associated_workflow_id="W-MEM", container_sha256=_sha("img-mem")),
            _art("A-FILE", link_status="STRICT", associated_workflow_id="W-FILE", container_sha256=_sha("img-file")),
            _art("A-DROPPED", link_status="STRICT", associated_workflow_id="W-DROPPED",
                 container_sha256=_sha("img-dropped")),
        ],
        workflows=[_wf("W-MEM", prompt_id="p-mem"), _wf("W-FILE", prompt_id="p-file"),
                   _wf("W-DROPPED", prompt_id="p-dropped")],
        links=[
            _strict_link("L1", "W-MEM", "A-MEM", ["PROMPT_ID"]),
            _strict_link("L2", "W-FILE", "A-FILE", ["PROMPT_ID"]),
            _strict_link("L3", "W-DROPPED", "A-DROPPED", ["PROMPT_ID"]),
        ],
        findings=[],
        evidence_records=[
            {"record_id": "A-MEM", "kind": "image", "memory_source": "MEMORY"},
            {"record_id": "A-FILE", "kind": "image", "memory_source": "FILE_BACKED"},
            {"record_id": "A-DROPPED", "kind": "image", "memory_source": "MEMORY"},
        ],
        memory_only_findings=[],
    )
    evidence_path = case / "recovery-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # The memory-only correlation keeps only A-MEM's link: A-DROPPED's strict link rests on non-memory evidence.
    evidence["memory_only_correlation_links"] = [_strict_link("L1", "W-MEM", "A-MEM", ["PROMPT_ID"])]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    gt = {"schema_version": 1, "dump_sha256": DUMP_SHA,
          "jobs": [_one_output_job(t) for t in ("mem", "file", "dropped")]}

    effective = score_case(case, gt, view="effective")
    assert effective["metrics"]["output_independently_corroborated_rate"]["numerator"] == 3
    memory_only = score_case(case, gt, view="memory_only")
    rate = memory_only["metrics"]["output_independently_corroborated_rate"]
    assert (rate["numerator"], rate["denominator"]) == (1, 3)
    assert {o["output_id"] for o in memory_only["outputs"] if o["independently_corroborated"]} == {"o-mem"}
    assert "denominator" in rate["definition"] and "memory_only" in rate["definition"]
