"""Evidence verification, replay comparisons and effective finding tiers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from comfytrace.carve import native_pixel_digest
from comfytrace.casefiles import derived_file, matches_derived_file
from comfytrace.correlate import OTHER_IMAGES, replay_graph_peers, unreplayed_complete_jobs
from comfytrace.memrecover import terminal_without_outputs_ids


def target_is_input(preflight: dict[str, Any], target: dict[str, Any]) -> bool:
    hashes = set((preflight.get("input_images") or {}).values())
    hashes.update((preflight.get("input_image_pixel_sha256") or {}).values())
    return any(target.get(key) in hashes for key in ("container_sha256", "pixel_sha256") if target.get(key))


def partial_rows_distinguish_inputs(regen: dict[str, Any], target_id: str | None) -> bool:
    expected = (regen.get("preflight_check") or {}).get("input_images") or {}
    if not expected:
        return True
    evaluation = regen.get("evaluation") or {}
    comparison = evaluation.get("input_comparison") or {}
    return (
        evaluation.get("target_artifact_id") == target_id
        and comparison.get("status") == "DISTINCT"
        and comparison.get("verified_inputs") == expected
    )


def comparison_only_output(regen: dict[str, Any], node_id: str | None) -> bool:
    reference_nodes = (regen.get("preflight_check") or {}).get("comparison_only_output_nodes") or []
    return node_id is not None and any(str(node) == str(node_id) for node in reference_nodes)


def _perceptual_match(evaluation: dict[str, Any]) -> bool:
    ssim, phash = evaluation.get("ssim_score"), evaluation.get("phash_hamming_distance")
    ssim_threshold, phash_threshold = evaluation.get("ssim_threshold"), evaluation.get("phash_threshold")
    return (
        isinstance(ssim, (int, float))
        and isinstance(phash, (int, float))
        and isinstance(ssim_threshold, (int, float))
        and not any(isinstance(value, bool) for value in (ssim, phash, ssim_threshold))
        and type(phash_threshold) is int
        and 0 <= ssim_threshold <= ssim <= 1
        and 0 <= phash <= phash_threshold <= 64
    )


def derived_image_matches(case_dir: Path | None, regen: dict[str, Any]) -> bool:
    name = regen.get("derived_regenerated_image")
    expected = (regen.get("evaluation") or {}).get("derived_regenerated_image_sha256")
    return matches_derived_file(case_dir, name, expected)


def regeneration_preflight_verified(regen: dict[str, Any]) -> bool:
    preflight = regen.get("preflight_check") or {}
    return (
        regen.get("status") == "COMPLETED"
        and preflight.get("status") == "PASSED"
        and preflight.get("runner_object_info_verified") is True
    )


def regeneration_permits_promotion(regen: dict[str, Any]) -> bool:
    security = regen.get("security_audit") or {}
    return regeneration_preflight_verified(regen) and (
        security.get("model_format") not in ("PICKLE", "MIXED") or security.get("pickle_models_permitted") is True
    )


def _verified_replay_positions(
    case_dir: Path, artifact: dict[str, Any], replay: dict[str, Any]
) -> set[tuple[str, int]]:
    evaluation = replay.get("evaluation") or {}
    pixel_sha = artifact.get("pixel_sha256")
    pixel_meta = artifact.get("pixel_metadata") or {}
    rasters = replay.get("output_rasters") or []
    expected_nodes = (replay.get("preflight_check") or {}).get("compared_output_nodes")
    if (
        not pixel_sha
        or evaluation.get("exact_pixel_match") is not True
        or evaluation.get("evaluation_mode") != "INTACT_NATIVE_RESOLUTION"
        or _regen_positions(replay) is None
        or (
            expected_nodes is not None
            and {str(node) for node in expected_nodes} != {str(raster.get("node_id")) for raster in rasters}
        )
    ):
        return set()
    matches: set[tuple[str, int]] = set()
    for raster in rasters:
        node, index = raster.get("node_id"), raster.get("batch_index")
        path, digest = raster.get("file"), raster.get("sha256")
        resolved = derived_file(case_dir, path)
        if resolved is None or not digest:
            return set()
        try:
            data = resolved.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                return set()
            pixels, meta = native_pixel_digest(data)
        except (OSError, ValueError, SyntaxError):
            return set()
        if pixels == pixel_sha and all(value == pixel_meta.get(key) for key, value in meta.items()):
            matches.add((str(node), index))
    return matches


def verify_image_link_promotions(
    links: dict[str, Any] | None,
    manifest: dict[str, Any],
    regenerations: list[dict[str, Any]],
    manifest_sha256: str,
    case_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not links or links.get("manifest_sha256") != manifest_sha256 or not links.get("promotions"):
        return links, []
    prompt_ids = {
        w["workflow_id"]: (w.get("execution_parameters") or {}).get("prompt_id")
        for w in manifest["findings"].get("workflows", [])
    }
    ambiguous = {
        a["artifact_id"]: set(a.get("ambiguous_candidate_workflows") or [])
        for a in manifest["findings"].get("artifacts", [])
        if a.get("link_status") == "AMBIGUOUS"
    }
    verified_files: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}
    artifacts = {a["artifact_id"]: a for a in manifest["findings"].get("artifacts", [])}
    verified_workflows: set[str] = set()
    # A workflow of a job that ended without recorded outputs cannot be the producer (as in link.py), so an
    # ambiguity it causes needs no replay of it.
    unproduced = terminal_without_outputs_ids(
        w.get("execution_parameters") or {} for w in manifest["findings"].get("workflows", [])
    )
    unproduced_workflows = {wid for wid, pid in prompt_ids.items() if pid in unproduced}
    peers = replay_graph_peers(manifest["findings"].get("workflows", []), unproduced)
    for regen in regenerations:
        if regen.get("target_manifest_sha256") != manifest_sha256 or not regeneration_permits_promotion(regen):
            continue
        workflow_id = regen["workflow_id"]
        if not (prompt_id := prompt_ids.get(workflow_id)):
            continue
        evaluation = regen.get("evaluation") or {}
        rasters = regen.get("output_rasters") or [
            {
                **(evaluation.get("compared_image") or {}),
                "file": regen.get("derived_regenerated_image"),
                "sha256": evaluation.get("derived_regenerated_image_sha256"),
            }
        ]
        for raster in rasters:
            name, digest = raster.get("file"), raster.get("sha256")
            if not comparison_only_output(regen, raster.get("node_id")) and matches_derived_file(
                case_dir, name, digest
            ):
                key: tuple[Any, ...] = (
                    regen["handoff_id"],
                    name,
                    digest,
                    raster.get("node_id"),
                    raster.get("batch_index"),
                )
                verified_files[key] = (prompt_id, regen)
                verified_workflows.add(workflow_id)

    untested_jobs = unreplayed_complete_jobs(
        manifest["findings"].get("workflows", []), {pid for pid, _ in verified_files.values()}, unproduced
    )
    accepted = []
    notes: list[dict[str, Any]] = []
    for promotion in links["promotions"]:
        members = [promotion["artifact_id"]] if "artifact_id" in promotion else promotion["artifact_ids"]
        if promotion.get("prompt_id") in peers:
            notes.extend(
                {
                    "promotion": artifact_id,
                    "applied_to": None,
                    "note": "replay cannot distinguish submissions with the same recovered API graph",
                }
                for artifact_id in members
            )
            continue
        per_artifact = promotion.get("raster_evidence_by_artifact") or {}
        supported = not (untested_jobs and promotion.get("tier") == "E3")
        for artifact_id in members:
            raster = per_artifact.get(artifact_id, promotion).get("selected_raster") or {}
            key = tuple(raster.get(field) for field in ("handoff_id", "file", "sha256", "node_id", "batch_index"))
            candidate = verified_files.get(key)
            if candidate is None or candidate[0] != promotion.get("prompt_id"):
                supported = False
            elif target_is_input(candidate[1].get("preflight_check") or {}, artifacts.get(artifact_id, {})) or (
                promotion.get("tier") == "E3" and not partial_rows_distinguish_inputs(candidate[1], artifact_id)
            ):
                supported = False
            if not ambiguous.get(artifact_id, set()) - unproduced_workflows <= verified_workflows:
                supported = False
        if supported:
            accepted.append(promotion)
        else:
            notes.extend(
                {
                    "promotion": image,
                    "applied_to": None,
                    "note": "verified case-bound raster or candidate replay coverage is missing",
                }
                for image in members
            )
    return {**links, "promotions": accepted}, notes


def _output_key(evidence: dict[str, Any] | None) -> list[Any] | None:
    """The job output (node, batch index) a memory copy is shown to be, or None when that is not supported."""
    raster = (evidence or {}).get("selected_raster") or {}
    if (
        (evidence or {}).get("output_attribution") != "SUPPORTED"
        or raster.get("node_id") is None
        or type(raster.get("batch_index")) is not int
    ):
        return None
    return [str(raster["node_id"]), raster["batch_index"]]


def _absorb(
    home: dict[str, Any],
    absorbed: list[dict[str, Any]],
    prompts: list[str],
    artifacts: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Moves what absorbed prompt findings held into ``home``; returns the workflow copies to list among its
    artifacts. A linked image an absorbed finding held (an E5/E6 tiered by its prompt because its PNG containers
    decoded zero rows) is not an output image of ``home``: it is kept with its link IDs as workflow-level container
    provenance. Without artifact records (``artifacts`` None) nothing can be told apart, so every ID is listed and
    the links are kept on ``home``."""
    copies: list[str] = []
    provenance = list(home.get("container_provenance") or [])
    links = list(home.get("link_ids") or [])
    for f in absorbed:
        own = [a for a in f["artifact_ids"] if a not in prompts]
        if artifacts is None:
            copies += own
            links += [link for link in f.get("link_ids", []) if link not in links]
            continue
        copies += [a for a in own if a not in artifacts]
        boxes = [a for a in own if a in artifacts]
        if boxes:
            provenance.append(
                {
                    "finding_id": f["finding_id"],
                    "tier": f["base_tier"],
                    "artifact_ids": boxes,
                    "link_ids": list(f.get("link_ids", [])),
                }
            )
    home["link_ids"] = links
    if provenance:
        home.update(
            container_provenance=provenance,
            container_provenance_artifacts=[a for p in provenance for a in p["artifact_ids"]],
            container_provenance_links=[link for p in provenance for link in p["link_ids"]],
        )
    return copies


def _regen_positions(regen: dict[str, Any]) -> set[tuple[str, int]] | None:
    """Every (node, batch index) a regeneration produced, from its complete output inventory, or None when the
    record does not establish it: requires an explicit ``output_rasters`` list, a positive integer
    ``compared_image_count`` matching its length, and a valid, distinct (node_id, batch_index) for every entry."""
    evaluation = regen.get("evaluation") or {}
    rasters = regen.get("output_rasters")
    count = evaluation.get("compared_image_count")
    if not isinstance(rasters, list) or not rasters or type(count) is not int or count <= 0 or count != len(rasters):
        return None
    positions: set[tuple[str, int]] = set()
    for raster in rasters:
        if not isinstance(raster, dict):
            return None
        node, index = raster.get("node_id"), raster.get("batch_index")
        if node is None or type(index) is not int or index < 0:
            return None
        pos = (str(node), index)
        if pos in positions:
            return None
        positions.add(pos)
    return positions


def _replay_inventory(
    regenerations: list[dict[str, Any]] | None, manifest_sha256: str
) -> dict[str, set[tuple[str, int]] | None]:
    """Per workflow, the output positions of its verified regenerations for this manifest (None: not established).
    Several regenerations of one workflow pool their positions; one unestablished record makes the whole unknown."""
    out: dict[str, set[tuple[str, int]] | None] = {}
    for regen in regenerations or []:
        if regen.get("target_manifest_sha256") != manifest_sha256 or not regeneration_permits_promotion(regen):
            continue
        workflow, positions = regen.get("workflow_id"), _regen_positions(regen)
        if workflow in out and out[workflow] is None:
            continue
        out[workflow] = None if positions is None else {*(out.get(workflow) or set()), *positions}
    return out


def apply_image_links(
    findings: list[dict[str, Any]],
    links: dict[str, Any] | None,
    manifest_sha256: str,
    artifacts: dict[str, dict[str, Any]] | None = None,
    regenerations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Applies image-links.json E2 promotions to the report's copy of the findings: the memory image's orphan
    finding becomes an E2 finding with the job's complete prompt, whose E5 findings it absorbs. A links file made
    against another manifest is reported but never applied. ``artifacts`` (the manifest's image records by ID)
    lets an absorbed finding's PNG containers be kept apart as container provenance; ``regenerations`` (the case's
    records) give each job's complete output inventory, which a close match may rely on."""
    if not links:
        return []
    if links.get("manifest_sha256") != manifest_sha256:
        return [
            {
                "promotion": None,
                "applied_to": None,
                "note": "image-links.json was made against another manifest; not applied",
            }
        ]
    outputs = _replay_inventory(regenerations, manifest_sha256)
    notes = []
    for promo in links.get("promotions", []):
        if promo.get("tier") == "E3":
            inventory = [outputs.get(w) for w in promo.get("workflow_ids", []) if w in outputs]
            notes += _apply_close_match(findings, promo, inventory, artifacts)
            continue
        image, prompts = promo["artifact_id"], promo["workflow_ids"]
        orphan = next((f for f in findings if f["base_tier"] == "UNTIERED" and f["artifact_ids"] == [image]), None)
        if orphan is None:
            notes.append({"promotion": image, "applied_to": None, "note": "the image is not an untiered orphan"})
            continue
        absorbed = [f for f in findings if f["base_tier"] == "E5" and set(f["artifact_ids"]) & set(prompts)]
        copies = _absorb(orphan, absorbed, prompts, artifacts)
        orphan.update(
            tier="E2",
            base_tier="E2",
            artifact_ids=[image, *prompts, *copies],
            image_link=promo["basis"],
            rationale=f"Complete memory image matches the regeneration of job {promo['prompt_id']} as a "
            f"whole frame ({promo['basis']}); complete prompt {', '.join(prompts)}.",
            absorbed=[f["finding_id"] for f in absorbed],
            matched_output=_output_key(promo),
        )
        for f in absorbed:
            findings.remove(f)
        notes.append({"promotion": image, "applied_to": orphan["finding_id"], "note": promo["basis"]})
    return notes


def _apply_close_match(
    findings: list[dict[str, Any]],
    promo: dict[str, Any],
    inventory: list[set[tuple[str, int]] | None] | None = None,
    artifacts: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Applies one E3 close-match promotion: the job's linked memory images leave their untiered findings (an
    orphan each, or the "Other images" group) and form one E3 finding with the job's prompt, absorbing the
    findings that held only that prompt.

    When the job's prompt already sits in a tiered image finding, the copies join it (its tier stays) only when the
    evidence ties both to the same output of the job: the copies' supported raster and the finding's matched output
    share a (node, batch index), or the complete output inventory of the job's verified regenerations (``inventory``,
    one position set per replayed workflow, not the links file's filtered job summary) has one position. An explicit
    position conflict between the finding and the copies, or with that one position, always vetoes the join. Without
    a join, as when several findings hold the prompt (one per batch output), the copies form their own E3 finding,
    scoped to the workflow, and join none of them. A links file made before the raster fields existed carries no
    output evidence at all; there a sole tiered finding is still joined, as before."""
    images, prompts, basis = promo["artifact_ids"], promo["workflow_ids"], promo["basis"]
    orphans = [
        f
        for f in findings
        if f["base_tier"] == "UNTIERED" and len(f["artifact_ids"]) == 1 and f["artifact_ids"][0] in images
    ]
    taken = [f["artifact_ids"][0] for f in orphans]
    for f in findings:
        if f["rationale"].startswith(OTHER_IMAGES) and set(f["artifact_ids"]) & set(images):
            taken += [a for a in f["artifact_ids"] if a in images]
            f["artifact_ids"] = [a for a in f["artifact_ids"] if a not in images]
    if not taken:
        return [
            {"promotion": image, "applied_to": None, "note": "the image is not an untiered image here"}
            for image in images
        ]
    taken.sort(key=images.index)
    per_artifact = promo.get("raster_evidence_by_artifact") or {}
    keys = [_output_key(per_artifact.get(a, promo)) for a in taken]
    copies_key = keys[0] if keys and keys[0] is not None and all(k == keys[0] for k in keys) else None
    legacy = "raster_evidence_by_artifact" not in promo and "selected_raster" not in promo
    pooled: set[tuple[str, int]] = set()
    established = bool(inventory) and all(p is not None for p in inventory or [])
    for p in inventory or []:
        pooled |= p or set()
    only = [list(next(iter(pooled)))] if established and len(pooled) == 1 else None
    eligible = [f for f in findings if f["tier"] in ("E1", "E2", "E3") and set(f["artifact_ids"]) & set(prompts)]
    home = None
    if len(eligible) == 1:
        known = eligible[0].get("matched_output")
        stated = [k for k in (known, *keys, *(only or [])) if k is not None]
        conflict = any(k != stated[0] for k in stated)
        if not conflict and (legacy or only is not None or (copies_key is not None and known == copies_key)):
            home = eligible[0]
            if home.get("matched_output") is None:
                home["matched_output"] = copies_key or (only[0] if only else None)
    candidates = [f["finding_id"] for f in eligible] if home is None else []
    for f in orphans[1 if home is None and orphans else 0 :]:
        findings.remove(f)
    if home is not None:
        home["artifact_ids"] += [a for a in taken if a not in home["artifact_ids"]]
        home["rationale"] += f" Close-match memory copies of job {promo['prompt_id']} ({basis}): {', '.join(taken)}."
        home["image_link"] = home.get("image_link") or basis
        note = f"{basis}: joined {home['finding_id']} ({home['tier']})"
    else:
        absorbed = (
            []
            if candidates
            else [
                f
                for f in findings
                if (f["base_tier"] == "E5" and set(f["artifact_ids"]) & set(prompts))
                or (f["base_tier"] == "E6" and set(f["artifact_ids"]) <= set(prompts))
            ]
        )
        if orphans:
            home = orphans[0]
        else:
            numbers = [
                int(f["finding_id"].rsplit("-", 1)[-1])
                for f in findings
                if f["finding_id"].rsplit("-", 1)[-1].isdigit()
            ]
            home = {
                "finding_id": f"FIND-{max(numbers, default=0) + 1:03d}",
                "link_ids": [],
                "extraction_tier": "UNTIERED",
                "regeneration": None,
            }
            findings.append(home)
        copies = _absorb(home, absorbed, prompts, artifacts)
        unresolved = (
            f" Which output of the job they are is not shown to be that of {', '.join(candidates)}, which "
            "already hold its prompt, so they are linked at workflow level and join none of them."
            if candidates
            else ""
        )
        home.update(
            tier="E3",
            base_tier="E3",
            artifact_ids=[*taken, *prompts, *copies],
            image_link=basis,
            rationale=f"{len(taken)} memory {'copy' if len(taken) == 1 else 'copies'} closely matching "
            f"the regeneration of job {promo['prompt_id']} ({basis}). Recovered prompt "
            f"{', '.join(prompts)}.{unresolved}",
            absorbed=[f["finding_id"] for f in absorbed],
            matched_output=copies_key,
        )
        if candidates:
            home.update(close_match_scope="WORKFLOW", output_candidates=candidates)
        for f in absorbed:
            findings.remove(f)
        note = (
            f"{basis}: output not shown to be that of {', '.join(candidates)}; kept at workflow level"
            if candidates
            else basis
        )
    return [{"promotion": image, "applied_to": home["finding_id"], "note": note} for image in taken]


def tier_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Findings per tier, with the "Other images" group counted by its images."""
    summary = dict.fromkeys(["E1", "E2", "E3", "E4", "E5", "E6", "UNTIERED", "Other images"], 0)
    for f in findings:
        if f["rationale"].startswith(OTHER_IMAGES):
            summary["Other images"] += len(f["artifact_ids"])
        else:
            summary[f["tier"]] = summary.get(f["tier"], 0) + 1
    return summary


def memory_only_after_links(
    evidence: dict[str, Any] | None, links: dict[str, Any] | None, manifest_sha256: str
) -> dict[str, int] | None:
    """Tier counts of the memory-only findings (recovery-evidence.json) with image-link promotions applied, or
    None when the case has no memory-only findings or no image links made against this manifest."""
    stored = (evidence or {}).get("memory_only_findings")
    if stored is None or not links or links.get("manifest_sha256") != manifest_sha256:
        return None
    findings = [
        dict(f, base_tier=f["tier"], extraction_tier=f["tier"], regeneration=None, image_link=None) for f in stored
    ]
    apply_image_links(findings, links, manifest_sha256)
    return tier_counts(findings)


def effective_findings(
    manifest: dict[str, Any],
    regenerations: list[dict[str, Any]],
    manifest_sha256: str,
    case_dir: Path | None = None,
    image_links: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Applies image-link promotions, then regeneration outcomes, to a copy of the manifest's findings. A record
    whose target_manifest_sha256 does not match this manifest is reported but never applied."""
    findings = [
        dict(f, base_tier=f["tier"], extraction_tier=f["tier"], regeneration=None, image_link=None)
        for f in manifest["findings"]["findings"]
    ]
    notes: list[dict[str, Any]] = []
    if case_dir is not None:
        image_links, notes = verify_image_link_promotions(
            image_links,
            manifest,
            regenerations,
            manifest_sha256,
            case_dir,
        )
    notes += apply_image_links(
        findings,
        image_links,
        manifest_sha256,
        {a["artifact_id"]: a for a in manifest["findings"].get("artifacts", [])},
        regenerations if case_dir is not None else None,
    )
    for regen in regenerations:
        applied = None
        if regen.get("target_manifest_sha256") != manifest_sha256:
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "target_manifest_sha256 does not match this manifest; not applied",
                }
            )
            continue
        evaluation = regen.get("evaluation") or {}
        if not regeneration_preflight_verified(regen):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "regeneration needs a completed run and verified preflight",
                }
            )
            continue
        promoted = evaluation.get("promoted_finding_tier")
        evaluated = evaluation.get("evaluated_tier")
        mode = evaluation.get("evaluation_mode")
        subtier = evaluation.get("evaluation_subtier")
        target = evaluation.get("target_artifact_id")
        workflow_id = regen.get("workflow_id")
        if not regeneration_permits_promotion(regen):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "pickle model execution was not permitted; not applied",
                }
            )
            continue
        if not derived_image_matches(case_dir, regen):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "derived regenerated image is missing or its SHA-256 does not match; not applied",
                }
            )
            continue
        if comparison_only_output(regen, (evaluation.get("selected_raster") or {}).get("node_id")):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "The selected raster is available for reference comparison.",
                }
            )
            continue
        artifact: dict[str, Any] = next(
            (a for a in manifest["findings"].get("artifacts", []) if a["artifact_id"] == target), {}
        )
        if promoted == "E1" and target_is_input(regen.get("preflight_check") or {}, artifact):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "target image was supplied as a conditioning input; not applied",
                }
            )
            continue
        if evaluated == "PARTIAL_CORROBORATION" and not partial_rows_distinguish_inputs(regen, target):
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": None,
                    "note": "The replay comparison did not distinguish the recovered rows from replay inputs.",
                }
            )
            continue
        # An E4 run raises every E5 finding it fits, or only those the record names (eligible_e4_finding_ids),
        # never whichever matching finding the list happens to hold last.
        eligible_ids = evaluation.get("eligible_e4_finding_ids")
        named = {str(i) for i in eligible_ids} if isinstance(eligible_ids, list) else None
        executable: list[dict[str, Any]] = []
        for finding in findings:
            ids = finding["artifact_ids"]
            if (
                promoted == evaluated == "E1"
                and mode == "INTACT_NATIVE_RESOLUTION"
                and (
                    (subtier == "EXACT_MATCH" and evaluation.get("exact_pixel_match") is True)
                    or (subtier == "PERCEPTUAL_MATCH" and _perceptual_match(evaluation))
                )
                and target in ids
                and workflow_id in ids
                and finding["base_tier"] == "E2"
            ):
                applied = finding
            elif (
                promoted == evaluated == "E4"
                and mode == "NO_MEMORY_TARGET"
                and target is None
                and subtier == "EXECUTABLE_REPRODUCTION"
                and workflow_id in ids
                and finding["base_tier"] == "E5"
                and (named is None or finding["finding_id"] in named)
            ):
                executable.append(finding)
            elif (
                promoted is None
                and evaluated == "E4"
                and mode == "NO_MEMORY_TARGET"
                and target is None
                and subtier == "EXECUTABLE_REPRODUCTION"
                and workflow_id in ids
                and finding["base_tier"] == "E3"
                and finding["regeneration"] is None
            ):
                # Not a promotion: eligible_e4_finding_ids names the E5 findings an E4 run may raise, and does
                # not restrict which E3 findings a run of their workflow is attached to.
                executable.append(finding)
            elif (
                promoted is None
                and evaluated == "PARTIAL_CORROBORATION"
                and mode == "PARTIAL_ROW_BOUNDED"
                and target in ids
                and workflow_id in ids
                and finding["base_tier"] == "E3"
            ):
                applied = finding
        if executable:
            for finding in executable:
                if promoted == "E4":
                    finding["tier"] = "E4"
                finding["regeneration"] = regen.get("handoff_id")
            notes.append(
                {
                    "handoff_id": regen.get("handoff_id"),
                    "applied_to": ", ".join(f["finding_id"] for f in executable),
                    "applied_to_all": [f["finding_id"] for f in executable],
                    "note": subtier or evaluated or regen.get("status"),
                }
            )
            continue
        if applied is not None:
            if promoted in ("E1", "E4"):
                applied["tier"] = promoted
            applied["regeneration"] = regen.get("handoff_id")
        notes.append(
            {
                "handoff_id": regen.get("handoff_id"),
                "applied_to": applied["finding_id"] if applied else None,
                "note": evaluation.get("evaluation_subtier") or evaluation.get("evaluated_tier") or regen.get("status"),
            }
        )
    if case_dir is not None and (case_dir / "replay-search.json").is_file():
        from comfytrace.replayevidence import apply_replay_evidence

        notes += apply_replay_evidence(findings, manifest, manifest_sha256, case_dir, regenerations)
    return findings, notes


MATCH_STATUSES = ("EXACT_MATCH", "ROW_SLICE_EXACT", "PERCEPTUAL_MATCH", "ROW_SLICE_PERCEPTUAL")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compared_rows(evaluation: dict[str, Any], artifact: dict[str, Any] | None) -> tuple[int | None, str | None]:
    """How many recovered rows a row-bounded comparison used, and that as text ("1 of 512 rows")."""
    rows = next(
        (evaluation[k] for k in ("compared_rows", "decoded_rows_evaluated") if type(evaluation.get(k)) is int), None
    )
    if rows is None:
        rows = ((artifact or {}).get("derived_preview_metadata") or {}).get("decoded_rows")
        rows = rows if type(rows) is int else None
    if rows is None:
        return None, None
    total = ((artifact or {}).get("derived_preview_metadata") or {}).get("total_rows")
    if type(total) is int and total:
        return rows, f"{rows} of {total} rows"
    return rows, f"{rows} row{'' if rows == 1 else 's'}"


def _difference_text(evaluation: dict[str, Any]) -> str:
    parts = []
    if is_number(evaluation.get("mean_abs_diff")):
        parts.append(f"mean absolute difference {evaluation['mean_abs_diff']:.2f}")
    if is_number(evaluation.get("max_abs_diff")):
        parts.append(f"max difference {evaluation['max_abs_diff']:g}")
    if is_number(evaluation.get("fraction_above_tolerance")):
        parts.append(f"{evaluation['fraction_above_tolerance']:.1%} of pixels above tolerance")
    return "; ".join(parts)


def comparison_fact(
    evaluation: dict[str, Any], artifact: dict[str, Any] | None = None, has_image: bool = True
) -> dict[str, Any]:
    """What a replay's comparison showed about its tested artifact (``target_artifact_id``) and nothing else.
    ``match`` is True only for a match the recorded scores support. Handles current records (INCONCLUSIVE) and
    older ones (NON_MATCHING, E4, PARTIAL_CORROBORATION, E1)."""
    tier = evaluation.get("evaluated_tier")
    subtier = evaluation.get("evaluation_subtier")
    mode = evaluation.get("evaluation_mode")
    target = evaluation.get("target_artifact_id")
    rows, rows_text = _compared_rows(evaluation, artifact) if mode == "PARTIAL_ROW_BOUNDED" else (None, None)
    ssim, phash = evaluation.get("ssim_score"), evaluation.get("phash_hamming_distance")
    scores = ", ".join(
        text
        for ok, text in (
            (is_number(ssim), f"SSIM {ssim:.3f}" if is_number(ssim) else ""),
            (is_number(phash), f"pHash distance {phash}" if is_number(phash) else ""),
        )
        if ok
    )
    fact: dict[str, Any] = {
        "status": subtier or tier or "UNRECORDED",
        "match": False,
        "target_artifact_id": target,
        "rows": rows,
        "rows_text": rows_text,
        "ssim": ssim if is_number(ssim) else None,
        "phash": phash if is_number(phash) else None,
        **{
            k: evaluation[k]
            for k in ("mean_abs_diff", "max_abs_diff", "fraction_above_tolerance")
            if is_number(evaluation.get(k))
        },
    }
    over_rows = f" over its {rows_text} recovered" if rows_text else " over its recovered rows"
    if mode == "NO_MEMORY_TARGET" or subtier == "EXECUTABLE_REPRODUCTION" or (tier == "E4" and target is None):
        fact["status"] = "NO_MEMORY_TARGET"
        fact["label"] = (
            "workflow ran in the lab; no comparable memory pixels: the memory image could not be pixel-compared"
            if has_image
            else "workflow ran in the lab; there is no memory image to compare it with"
        )
    elif subtier == "EXACT_MATCH" and evaluation.get("exact_pixel_match") is True:
        fact.update(match=True, label=f"exact pixel match with {target}")
    elif subtier == "ROW_SLICE_EXACT" and evaluation.get("row_slice_pixel_match") is True:
        fact.update(match=True, label=f"exact match of {target}{over_rows}")
    elif subtier == "PERCEPTUAL_MATCH" and _perceptual_match(evaluation):
        fact.update(match=True, label=f"close perceptual match with {target} ({scores})")
    elif subtier == "ROW_SLICE_PERCEPTUAL" and _perceptual_match(evaluation):
        fact.update(match=True, label=f"close perceptual match of {target}{over_rows} ({scores})")
    elif subtier in MATCH_STATUSES:
        fact["status"] = "UNSUPPORTED_MATCH"
        fact["label"] = (
            f"recorded as {subtier} for {target}, but the recorded scores do not support it; not treated as a match"
        )
    elif tier == "INCONCLUSIVE" or subtier in ("INSUFFICIENT_ROWS", "INSUFFICIENT_DIMENSIONS"):
        fact["status"] = "INCONCLUSIVE"
        fact["subtier"] = subtier
        thin = (
            f"{rows_text} are too few for a 2D SSIM"
            if subtier == "INSUFFICIENT_ROWS" and rows_text
            else "too few rows for a 2D SSIM"
            if subtier == "INSUFFICIENT_ROWS"
            else "the image is too thin for a 2D SSIM"
        )
        diffs = _difference_text(evaluation)
        fact["label"] = f"comparison of {target} inconclusive: {thin}; no match is claimed" + (
            f" ({diffs})" if diffs else ""
        )
    elif subtier == "ANIMATED_COMPARISON_UNAVAILABLE":
        fact["label"] = (
            f"sequence comparison unavailable for {target}: an animation needs its whole sequence "
            "compared; this is not a pixel contradiction"
        )
    elif tier == "NON_MATCHING" or subtier in ("BELOW_THRESHOLDS", "DIMENSION_MISMATCH"):
        fact["status"] = "NON_MATCHING"
        fact["subtier"] = subtier
        where = f"{target}{f' ({rows_text})' if rows_text else ''}"
        if subtier == "DIMENSION_MISMATCH":
            dims = evaluation.get("dimension_mismatch") or {}
            detail = ", ".join(f"{str(k).replace('_', ' ')} {v}" for k, v in dims.items())
            fact["label"] = f"compared {where} and differed: the dimensions do not agree" + (
                f" ({detail})" if detail else ""
            )
        elif (
            mode == "PARTIAL_ROW_BOUNDED"
            and rows is not None
            and rows < 3
            and is_number(ssim)
            and ssim == 0
            and evaluation.get("row_slice_pixel_match") is False
        ):
            # A record made before SSIM was withheld below its 3-pixel window wrote 0.0 for a 1-2 row strip. That
            # number was never measured; the record keeps it and its verdict, the page does not repeat it.
            fact.update(
                ssim=None,
                phash=None,
                legacy_ssim_placeholder=True,
                status="INCONCLUSIVE",
                recorded_verdict=subtier or tier,
            )
            fact["label"] = (
                f"comparison of {where} inconclusive: the recovered strip is not pixel-identical to the "
                f"replay's rows, and a 2D SSIM is unavailable for fewer than 3 rows, so no similarity "
                f"score was measured (recorded verdict {subtier or tier} and SSIM 0.0 are kept in the "
                "record as written)"
            )
        else:
            limits = ", ".join(
                text
                for ok, text in (
                    (is_number(evaluation.get("ssim_threshold")), f"SSIM at least {evaluation.get('ssim_threshold')}"),
                    (
                        is_number(evaluation.get("phash_threshold")),
                        f"pHash distance at most {evaluation.get('phash_threshold')}",
                    ),
                )
                if ok
            )
            fact["label"] = f"compared {where} and differed: {scores or 'scores not recorded'}" + (
                f", outside the thresholds ({limits})" if limits else ""
            )
            if rows is not None and rows < 3 and is_number(ssim):
                fact["label"] += f"; a 2D SSIM over {rows} row{'' if rows == 1 else 's'} carries little weight"
        diffs = _difference_text(evaluation)
        if diffs:
            fact["label"] += f" ({diffs})"
    else:
        fact["status"] = "UNRECOGNISED"
        fact["label"] = (
            f"comparison result {subtier or tier or 'not recorded'} is not recognised; not treated as a match"
        )
    return fact
