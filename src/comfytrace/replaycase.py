"""Run the post-extraction pipeline for one case: process evidence, replay, memory search, linking, report.

Order follows the data: model identity is read by regeneration and latent decoding, latent decodes and
regenerations are read by linking, and the report reads everything. Every output is written once; a step whose
output already exists is skipped, so a rerun completes what an earlier run left out and changes nothing it made.
Model identity and latent recovery read ComfyUI's process memory and are opt-in (``--model-id``, ``--latents``).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from comfytrace.casefiles import MANIFEST_NAME, sha256_file
from comfytrace.localrunner import resolve_runner
from comfytrace.regenerate import (
    MISSING_DEPENDENCY,
    SECURITY_REJECTED,
    prompt_id_of,
    recovered_prompts,
    replayable_prompts,
    run_regeneration,
)
from comfytrace.replaysearch import search
from comfytrace.report import HtmlReporter

log = logging.getLogger(__name__)


TARGET_TIERS = ("E2", "E3")
AUDIT_TIERS = ("E5", "E6")
"""Findings tiered by their prompt; their images (zero-row containers) are examined and recorded, never compared."""
TARGET_CARVE_TYPES = ("ENCODED_PNG", "PARTIAL_PNG")
NO_COMPARABLE_TARGET = "NO_COMPARABLE_TARGET"
NO_COMPARABLE_RASTER = "NO_COMPARABLE_RASTER"
ZERO_DECODED_ROWS = "ZERO_DECODED_ROWS"
PROMPT_TIER_FINDING = "PROMPT_TIER_FINDING"
# The exclusion an artifact is recorded under when its findings give different ones: what the artifact itself
# lacks first (the same in every finding), then what a particular finding's link rules out.
_EXCLUSION_ORDER = (ZERO_DECODED_ROWS, "DECODED_ROWS_UNKNOWN", NO_COMPARABLE_RASTER, "PROMPT_ID_CONFLICT",
                    "INPUT_REVISION_UNVERIFIED", PROMPT_TIER_FINDING)


def _prompt_ids(workflow: dict[str, Any] | None) -> set[str]:
    params = (workflow or {}).get("execution_parameters") or {}
    return {pid for pid in (params.get("prompt_id"), *(params.get("additional_prompt_ids") or []))
            if isinstance(pid, str) and pid}


def _rank(artifact: dict[str, Any]) -> tuple[tuple[Any, ...] | None, str, str | None]:
    """(sort key, reason, comparison path) for one candidate, or (None, exclusion reason, None).

    Order: a directly recovered complete still PNG, a complete DERIVED reconstruction, the partial with the most
    decoded rows (then the widest), a complete animation (only its first frame decodes). Lowest artifact ID breaks
    ties. A partial with no decoded rows, or none recorded, has no pixels to compare and is never a target."""
    aid = artifact["artifact_id"]
    if artifact["completeness"] == "COMPLETE" and artifact.get("raw_file_paths"):
        path = artifact["raw_file_paths"][0]
        if (artifact.get("pixel_metadata") or {}).get("animated"):
            return (3, 0, 0, aid), "COMPLETE_ANIMATED", path
        if artifact.get("is_derived"):
            return (1, 0, 0, aid), "COMPLETE_DERIVED", path
        return (0, 0, 0, aid), "COMPLETE_NATIVE", path
    if artifact["completeness"] == "COMPLETE":
        return None, NO_COMPARABLE_RASTER, None
    rows = (artifact.get("derived_preview_metadata") or {}).get("decoded_rows")
    known = isinstance(rows, int) and not isinstance(rows, bool)
    # Zero decoded rows is checked first: a header-only container has no preview, and the audit should say why.
    if known and rows <= 0:
        return None, ZERO_DECODED_ROWS, None
    if not artifact.get("derived_preview_path"):
        return None, NO_COMPARABLE_RASTER, None
    if not known:
        return None, "DECODED_ROWS_UNKNOWN", None
    width = (artifact.get("pixel_metadata") or {}).get("width") or 0
    return (2, -rows, -width, aid), "PARTIAL_MOST_DECODED_ROWS", artifact["derived_preview_path"]


def _conflict(workflows: dict[str, dict[str, Any]], links: dict[str, dict[str, Any]], workflow: dict[str, Any],
              artifact_id: str, finding: dict[str, Any]) -> str | None:
    """Why a finding's image cannot be this workflow's output although both are in the finding: its link names
    another submission, or it was linked only to copies without a prompt ID because the input revision is
    unverified."""
    own = _prompt_ids(workflow)
    for link_id in finding.get("link_ids", []):
        link = links.get(link_id)
        if link is None or link.get("artifact_id") != artifact_id:
            continue
        linked = _prompt_ids(workflows.get(link.get("workflow_id")))
        if own and linked and not own & linked:
            return "PROMPT_ID_CONFLICT"
        if own and "INPUT_REVISION_UNVERIFIED" in (link.get("matched_keys") or []):
            return "INPUT_REVISION_UNVERIFIED"
    return None


def _targets(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The comparison target of each workflow, chosen over every E2/E3 finding the workflow belongs to.

    Candidates from all of a workflow's findings are ranked together (see ``_rank``), so neither the order of the
    findings nor of their artifacts changes the choice. The images of E5/E6 findings (zero-row containers, which
    correlation tiers by their prompt) are examined for the audit and always excluded: a replay of such a workflow
    runs without a target, which is what lets it show executability (E4). Each value records the selected artifact
    (``selected_artifact_id``, ``path``; None when nothing comparable remains), the ``selection_reason``, the ranked
    ``candidate_artifact_ids``, the ``excluded_candidates`` with their reasons and the ``finding_ids`` searched."""
    artifacts = {item["artifact_id"]: item for item in manifest["findings"]["artifacts"]}
    workflows = {w["workflow_id"]: w for w in manifest["findings"]["workflows"]}
    links = {link["link_id"]: link for link in manifest["findings"].get("correlation_links", [])}
    pools: dict[str, dict[str, Any]] = {}
    for finding in manifest["findings"]["findings"]:
        comparable = finding["tier"] in TARGET_TIERS
        if not comparable and finding["tier"] not in AUDIT_TIERS:
            continue
        images = [artifacts[key] for key in finding["artifact_ids"]
                  if key in artifacts and artifacts[key]["carve_type"] in TARGET_CARVE_TYPES]
        if not comparable and not images:
            continue
        for workflow_id in (key for key in finding["artifact_ids"] if key in workflows):
            pool = pools.setdefault(workflow_id, {"finding_ids": [], "ranked": {}, "excluded": {}})
            pool["finding_ids"].append(finding["finding_id"])
            for image in images:
                aid = image["artifact_id"]
                key, reason, path = _rank(image)
                if key is not None:
                    conflict = _conflict(workflows, links, workflows[workflow_id], aid, finding)
                    reason = conflict or (reason if comparable else PROMPT_TIER_FINDING)
                    if conflict is None and comparable:
                        previous = pool["ranked"].get(aid)
                        if previous is None or key < previous[0]:
                            pool["ranked"][aid] = (key, reason, path)
                        continue
                pool["excluded"].setdefault(aid, set()).add(reason)
    targets = {}
    for workflow_id, pool in pools.items():
        ranked = sorted(pool["ranked"].items(), key=lambda item: item[1][0])
        best = ranked[0] if ranked else None
        excluded = {aid: min(reasons, key=_EXCLUSION_ORDER.index)
                    for aid, reasons in pool["excluded"].items() if aid not in pool["ranked"]}
        targets[workflow_id] = {
            "selected_artifact_id": best[0] if best else None,
            "path": best[1][2] if best else None,
            "selection_reason": best[1][1] if best else NO_COMPARABLE_TARGET,
            "candidate_artifact_ids": [aid for aid, _ in ranked],
            "excluded_candidates": [{"artifact_id": aid, "reason": reason} for aid, reason in sorted(excluded.items())],
            "finding_ids": sorted(pool["finding_ids"]),
        }
    return targets


def _check_options(args: argparse.Namespace, *, image_format: str = "physical") -> None:
    want_model_id, want_latents = getattr(args, "model_id", False), getattr(args, "latents", False)
    if (want_model_id or want_latents) and image_format == "physical" and args.symbols_dir is None:
        raise ValueError("--model-id and --latents read process memory; pass --symbols-dir")
    if (want_model_id or want_latents) and args.models_dir is None:
        raise ValueError("--model-id and --latents need --models-dir (model identity and the VAE for decoding)")


def _process_evidence(args: argparse.Namespace, manifest: dict[str, Any], memory_sha: str,
                      ingestor: Any = None) -> list[str]:
    """Model identity and retained latents from one ingestion of the memory image. Returns the steps run."""
    from comfytrace import latents
    from comfytrace.cli import ingest_memory
    from comfytrace.ingest import ingest_case
    from comfytrace.modelid import IDENTITY_NAME, identity_document, write_identity

    case_dir: Path = args.case_dir
    run_model_id = getattr(args, "model_id", False) and not (case_dir / IDENTITY_NAME).exists()
    run_latents = getattr(args, "latents", False) and not (case_dir / latents.LATENTS_NAME).exists()
    for name, wanted, runs in ((IDENTITY_NAME, getattr(args, "model_id", False), run_model_id),
                               (latents.LATENTS_NAME, getattr(args, "latents", False), run_latents)):
        if wanted and not runs:
            log.info("%s exists; kept as written", name)
    if not (run_model_id or run_latents):
        return []
    ingestion = ingest_case(args.memory, args.symbols_dir, manifest["case_metadata"], ingestor or ingest_memory)
    done = []
    try:
        processes = ingestion.recovery.python_processes
        if run_model_id:
            document = identity_document(case_dir, processes, args.models_dir, memory_sha)
            write_identity(case_dir, document)
            for model in document["models"]:
                log.info("model identity: %s: %s", model["name"], model["verdict"])
            done.append("model-id")
        if run_latents:
            graphs = [json.loads((case_dir / w["raw_file_path"]).read_text(encoding="utf-8"))
                      for w in recovered_prompts(manifest) if w.get("raw_file_path")]
            sides = latents.latent_sides(graphs)
            blocks = [b for mem in processes for b in latents.find(mem, sides)]
            latents.write(case_dir, latents.document(case_dir, blocks, memory_sha, sides))
            log.info("%d retained latents recorded", len(blocks))
            done.append("latents")
    finally:
        ingestion.close()
    return done


def _decode_latents(args: argparse.Namespace) -> int | None:
    """Return the decode exit status, or None when the stage did not run."""
    from comfytrace import latents
    from comfytrace.latentdecode import DECODES_DIR, decodes_complete, run_decode

    path = args.case_dir / latents.LATENTS_NAME
    if not getattr(args, "latents", False) or not path.is_file():
        return None
    if decodes_complete(args.case_dir):
        log.info("%s is complete; kept as written", DECODES_DIR)
        return None
    if (args.case_dir / DECODES_DIR).exists():
        log.info("%s exists but its run did not finish; resuming the missing decodes", DECODES_DIR)
    if not json.loads(path.read_bytes()).get("latents"):
        log.info("no retained latents were found; nothing to decode")
        return None
    decode = argparse.Namespace(**{**vars(args), "timeout": max(args.timeout, 600)})
    try:
        return run_decode(decode)
    except Exception as exc:  # noqa: BLE001 - decoded latents are optional evidence; linking and the report go on
        log.error("latent decoding failed: %s", exc)
        return 1


def _link(case_dir: Path) -> bool:
    from comfytrace.link import LINKS_NAME, write_links

    if (case_dir / LINKS_NAME).exists():
        log.info("%s exists; kept as written", LINKS_NAME)
        return False
    write_links(case_dir)
    return True


def run_replay_case(args: argparse.Namespace, ingestor: Any = None) -> int:
    from comfytrace.ingest import supports_raw_search

    case_dir = args.case_dir
    manifest_path = case_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    can_search = supports_raw_search(manifest["case_metadata"], args.memory)
    if can_search and find_spec("ahocorasick_rs") is None:
        raise ValueError("replay-case requires the [search] extra for physical-memory search")
    source = manifest["case_metadata"].get("symbol_resolution") or {}
    _check_options(args, image_format=source.get("image_format", "physical"))
    workflows = replayable_prompts(manifest)
    if not workflows:
        raise ValueError("the case has no complete recovered API graphs")
    if unbound := sum(1 for w in workflows if not prompt_id_of(w)):
        log.info("%d graph(s) without a recovered prompt ID are replayed to locate pixels; their hits name no "
                 "producer", unbound)
    memory_sha = sha256_file(args.memory)
    if memory_sha != manifest["case_metadata"]["memory_image"]["sha256"]:
        raise ValueError(f"{args.memory} is not the image this case was extracted from")

    steps = _process_evidence(args, manifest, memory_sha, ingestor)

    targets = _targets(manifest)
    artifacts_by_id = {item["artifact_id"]: item for item in manifest["findings"]["artifacts"]}
    runner_url = resolve_runner(args)
    args.runner_url = runner_url
    records = case_dir / "regenerations"
    previous = set(records.glob("REGEN-*.json"))
    failed = 0
    for workflow in workflows:
        identifier = workflow["workflow_id"]
        selection = targets.get(identifier)
        target = selection["path"] if selection else None
        extra_paths = []
        if selection:
            for cid in selection["candidate_artifact_ids"][1:]:
                if cid in artifacts_by_id:
                    _, _, cpath = _rank(artifacts_by_id[cid])
                    if cpath:
                        extra_paths.append(case_dir / cpath)
        replay = argparse.Namespace(**vars(args))
        replay.workflow = case_dir / workflow["raw_file_path"]
        replay.workflow_id = identifier
        replay.compare_to = case_dir / target if target else None
        replay.extra_compare_to = extra_paths
        replay.target_selection = selection
        if selection:
            log.info("%s: target %s (%s) of candidates %s", identifier, selection["selected_artifact_id"],
                     selection["selection_reason"], selection["candidate_artifact_ids"])
        try:
            failed += run_regeneration(replay) == 1
        except Exception as exc:  # noqa: BLE001
            log.error("%s: %s", identifier, exc)
            failed += 1
    steps.append("regenerate")

    decode_status = _decode_latents(args)
    if decode_status is not None:
        if decode_status:
            log.warning("latent decoding did not complete; linking continues without it")
        steps.append("decode-latents (incomplete)" if decode_status else "decode-latents")

    new_records = set(records.glob("REGEN-*.json")) - previous
    statuses = {p.stem: json.loads(p.read_bytes())["status"] for p in new_records}
    completed = {stem for stem, status in statuses.items() if status == "COMPLETED"}
    # A preflight refusal (an unvetted node, an unsafe name, a model the runner lacks) is the audit working, not a
    # failed replay: graphs carved from stale or hostile PNGs are refused by design.
    refused = Counter(status for status in statuses.values() if status in (SECURITY_REJECTED, MISSING_DEPENDENCY))
    failed -= sum(refused.values())
    if refused:
        log.info("preflight refused %s", ", ".join(f"{n} {status}" for status, n in sorted(refused.items())))
    if completed and can_search:
        search(case_dir, args.memory, set(), handoffs=completed)
        steps.append("search-replay")
    elif completed:
        log.info("skipping raw replay search because the case has no flat physical capture")
    if _link(case_dir):
        steps.append("link")
    output = HtmlReporter(case_dir).render_report(manifest_path)
    log.info("replay-case ran %s; %d workflows: %d completed, %d refused by preflight, %d failed. Report written "
             "to %s", " -> ".join(steps), len(workflows), len(completed), sum(refused.values()), failed, output)
    return 1 if failed or decode_status else 0
