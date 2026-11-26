"""Link memory images to jobs by content, through images regenerated from the recovered prompts.

Memory pixel holders (lineage L5) carry no metadata. A regenerated image is produced from a recovered API prompt
(lineage L1), so agreement between the two is corroboration across independent lineages. Thresholds are fixed
here, before any comparison:

* a pair agrees when at least AGREE_FRACTION of the overlapping rows have a mean absolute error within the row
  limit for its scale (1x: 2 levels, 2x and 4x box-downscaled: 4 levels): agreement ROWS;
* at 1x, a lossy copy (the frontend's WebP/JPEG preview, decoded) agrees when its PSNR is at least LOSSY_AGREE_DB
  and its median row error at most LOSSY_ROW_LIMIT: agreement LOSSY. On T2 such copies scored 32.3 to 37.7 dB
  against their own job's regeneration and at most 15.5 dB against any other job (T105). A LOSSY copy is never
  exact;
* a memory image is linked (SUPPORTED) to the job whose regeneration agrees best, only when no other job's
  regeneration also agrees; otherwise it is AMBIGUOUS;
* a regenerated image is MATCHED when at least one pixel-bearing memory image agrees with it, else UNCHECKED;
* an image decoded from a latent found in memory (``decode-latents``) agrees with a regeneration of the same size
  when their PSNR is at least LATENT_AGREE_DB, and is linked by the same exactly-one rule. Both are computed
  images, so the sampler's CPU-versus-GPU divergence is in the comparison; on run 1 and T2 a decoded latent scored
  36.0 to 47.9 dB against its own job's regeneration and at most 10.0 dB against any other job (T95).

Regenerated and decoded images are never counted as recovered images. Writes ``image-links.json``; the manifest
is untouched.

A job whose regeneration agrees row for row with a complete whole-frame pixel buffer or tensor
(OBSERVED_EXACT) gives a tier promotion: that memory image and the job's complete recovered API prompt form one
E2 finding, by REGENERATION_WHOLE_FRAME_MATCH. The report applies it to findings of the manifest named by
``manifest_sha256``, as it applies regeneration records.

A confirmed close match earns E3, never E2: every other memory image linked to the job (a LOSSY copy, or a
partial, flagged, shifted or scaled one that agrees row for row with exactly one job's regeneration) forms one E3
finding with the job's API prompt, by REGENERATION_CLOSE_MATCH. So does an exact image whose job has only a
partial prompt. When the job already has a tiered finding the copies join it and its tier stays.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from comfytrace import latents
from comfytrace.casefiles import LINKS_NAME
from comfytrace.correlate import replay_graph_peers, unreplayed_complete_jobs
from comfytrace.findings import (
    comparison_only_output,
    partial_rows_distinguish_inputs,
    regeneration_permits_promotion,
    target_is_input,
)
from comfytrace.memrecover import (
    OUTPUT_CONTEXT_FIELD,
    TERMINAL_WITHOUT_RECORDED_OUTPUTS,
    _row_errors,
    terminal_without_outputs_ids,
)

AGREE_FRACTION = 0.9
ROW_LIMIT = {1: 2.0, 2: 4.0, 4: 4.0}
LOSSY_AGREE_DB, LOSSY_ROW_LIMIT = 30.0, 4.0
MEMORY_KINDS = ("PIXEL_BUFFER", "TENSOR_IMAGE", "DECODED_BITMAP")
LATENT_AGREE_DB = 20.0


def memory_images(case_dir: Path) -> list[dict[str, Any]]:
    """Memory pixel records, rebuilt from their raw spans after checking the sidecar's raw hash."""
    manifest = json.loads((case_dir / "manifest.json").read_text())
    evidence = {r["record_id"]: r for r in json.loads((case_dir / "recovery-evidence.json").read_text())["records"]}
    out = []
    for art in manifest["findings"]["artifacts"]:
        if art["carve_type"] not in MEMORY_KINDS:
            continue
        raw = b"".join((case_dir / p).read_bytes() for p in art["raw_file_paths"])
        expected = evidence.get(art["artifact_id"], {}).get("sha256", {}).get("raw")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{art['artifact_id']}: raw bytes do not match recovery-evidence.json")
        meta = art["pixel_metadata"]
        channels = meta["source"].get("channels", 3) if art["carve_type"] == "PIXEL_BUFFER" else 3
        row = meta["width"] * channels
        gap_rows: list[int] = []
        if art.get("gaps"):
            # Raw spans hold the genuine rows only; a gap's rows are rebuilt as zeros and never compared.
            full, cursor = bytearray(meta["height"] * row), 0
            for gap in art["gaps"]:
                genuine = gap["relative_offset"] - cursor
                full[cursor : cursor + genuine] = raw[:genuine]
                raw, cursor = raw[genuine:], gap["relative_offset"] + gap["length_bytes"]
                gap_rows += range(gap["relative_offset"] // row, (cursor + row - 1) // row)
            full[cursor : cursor + len(raw)] = raw
            raw = bytes(full)
        rgb = np.frombuffer(raw, np.uint8).reshape(meta["height"], meta["width"], channels)[:, :, :3]
        out.append({"artifact_id": art["artifact_id"], "kind": art["carve_type"], "rgb": rgb,
                    "complete": art["completeness"] == "COMPLETE", "suspect_rows": meta.get("suspect_rows", []),
                    "pixel_sha256": art.get("pixel_sha256"), "container_sha256": art.get("container_sha256"),
                    "ambiguous_workflows": art.get("ambiguous_candidate_workflows", [])
                    if art.get("link_status") == "AMBIGUOUS" else [],
                    "gap_rows": gap_rows})
    return out


def case_regenerations(case_dir: Path, *, unbound: bool = False) -> list[dict[str, Any]]:
    """Completed ``regenerate`` records of the case, each image checked against its recorded hash.

    Only replays of workflows with a recovered prompt ID name a job. ``unbound`` also returns replays of graphs
    without one (``prompt_id`` None), for a pixel search that locates copies without naming their producer."""
    manifest_bytes = (case_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    prompt_ids = {w["workflow_id"]: w["execution_parameters"].get("prompt_id")
                  for w in manifest["findings"]["workflows"]}
    out = []
    for record_path in sorted((case_dir / "regenerations").glob("REGEN-*.json")):
        record = json.loads(record_path.read_text())
        if record.get("target_manifest_sha256") != manifest_sha:
            # Workflow ids are per case: a record made against another manifest names another case's jobs.
            raise ValueError(f"{record_path.name}: made against another manifest")
        prompt_id = prompt_ids.get(record["workflow_id"])
        if (record["status"] != "COMPLETED" or not record.get("derived_regenerated_image")
                or not (prompt_id or (unbound and record["workflow_id"] in prompt_ids))):
            continue
        prompt_id = prompt_id or None
        evaluation = record["evaluation"]
        verified = regeneration_permits_promotion(record)
        rasters = record.get("output_rasters") or [{**(evaluation.get("compared_image") or {}),
                                                    "file": record["derived_regenerated_image"],
                                                    "sha256": evaluation.get("derived_regenerated_image_sha256")}]
        for raster in rasters:
            path = Path(raster["file"])
            source = (case_dir / path).resolve()
            if path.is_absolute() or not source.is_relative_to((case_dir / "exports/derived").resolve()):
                raise ValueError(f"{record_path.name}: regenerated output is outside exports/derived")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != raster["sha256"]:
                raise ValueError(f"{raster['file']}: does not match {record_path.name}")
            with Image.open(io.BytesIO(data)) as image:
                if (getattr(image, "n_frames", 1) > 1 or getattr(image, "custom_mimetype", None) == "image/apng"
                        or (image.format == "PNG" and data[24] == 16)):
                    continue  # This linkage compares still, 8-bit memory pixels.
                rgb = np.asarray(image.convert("RGB"))
            out.append({**raster, "handoff_id": record["handoff_id"], "prompt_id": prompt_id,
                        "verified": verified and not comparison_only_output(record, raster.get("node_id")),
                        "input_evidence": record,
                        "record": {"handoff_id": record["handoff_id"], "workflow_id": record["workflow_id"],
                                   "model_match_level": record["preflight_check"].get("model_match_level"),
                                   "model_sha256": record["preflight_check"].get("model_sha256")},
                        "rgb": rgb})
    return out


def latent_decodes(case_dir: Path) -> list[dict[str, Any]]:
    """Completed ``decode-latents`` records, each image checked against its recorded hash and each latent against
    this case's hash-checked latents sidecar, which must come from this case's memory image."""
    out: list[dict[str, Any]] = []
    records = sorted((case_dir / "latent-decodes").glob("LATDEC-*.json"))
    if not records:
        return out
    manifest = json.loads((case_dir / "manifest.json").read_text())
    sidecar = json.loads((case_dir / latents.LATENTS_NAME).read_text())
    if sidecar["memory_sha256"] != manifest["case_metadata"]["memory_image"]["sha256"]:
        raise ValueError(f"{latents.LATENTS_NAME} was made from another memory image")
    known = {e["latent_id"]: e["sha256"] for e in latents.load(case_dir)}
    for record_path in records:
        record = json.loads(record_path.read_text())
        if record["status"] != "COMPLETED":
            continue
        if known.get(record["latent_id"]) != record["latent_sha256"]:
            raise ValueError(f"{record_path.name}: latent {record['latent_id']} is not this case's")
        data = (case_dir / record["derived_image"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["derived_image_sha256"]:
            raise ValueError(f"{record['derived_image']}: does not match {record_path.name}")
        out.append({"decode_id": record["decode_id"], "latent_id": record["latent_id"], "file": record["derived_image"],
                    "sha256": record["derived_image_sha256"], "checkpoint": record["checkpoint"]["name"],
                    "model_match_level": record["checkpoint"]["model_match_level"],
                    "rgb": np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))})
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return round(10 * float(np.log10(255 ** 2 / mse)), 2) if mse > 0 else float("inf")


def regenerated_images(regen_dir: Path) -> list[dict[str, Any]]:
    out = []
    for record_path in sorted(regen_dir.glob("*.json")):
        record = json.loads(record_path.read_text())
        image_path = record_path.with_suffix(".png")
        data = image_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get("output_sha256"):
            raise ValueError(f"{image_path}: does not match its record's output_sha256")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        out.append({**{key: record.get(key) for key in ("node_id", "batch_index", "handoff_id")},
                    "prompt_id": record["prompt_id"], "file": image_path.name, "sha256": record["output_sha256"],
                    "record": {k: v for k, v in record.items() if k != "runner"}, "rgb": rgb})
    return out


def compare(regen: np.ndarray, memory: np.ndarray, ignore_rows: Sequence[int] = ()) -> dict[str, Any] | None:
    """Aligns a memory image with a regenerated one at scale 1, 2 or 4 and reports per-row agreement. Rows in
    ``ignore_rows`` (gap rows with no genuine pixels) are not compared."""
    if regen.shape[1] % memory.shape[1] or regen.shape[1] // memory.shape[1] not in ROW_LIMIT:
        return None
    factor = regen.shape[1] // memory.shape[1]
    result = _row_errors(regen, memory, factor, ignore_rows)
    if result is None:
        return None
    dy, err = result
    valid = err[~np.isnan(err)]
    if not len(valid):
        return None
    agree = valid <= ROW_LIMIT[factor]
    rows = np.nonzero(~np.isnan(err))[0]
    h = regen.shape[0] // factor * factor
    down = regen[:h].astype(np.float64).reshape(h // factor, factor, memory.shape[1], factor, 3).mean((1, 3))
    a = down[rows + dy]
    b = memory[rows].astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    out = {"scale": factor, "row_offset": int(dy), "rows_compared": int(len(valid)), "rows_agreeing": int(agree.sum()),
           "median_row_error": round(float(np.median(valid)), 3),
           "psnr_db": round(float(10 * np.log10(255 ** 2 / mse)), 2) if mse > 0 else None,
           "agrees": bool(agree.mean() >= AGREE_FRACTION)}
    out["agreement"] = "ROWS" if out["agrees"] else None
    if factor == 1:
        out["identical_pixel_fraction"] = round(float((np.abs(a - b).max(-1) == 0).mean()), 5)
        if not out["agrees"] and out["psnr_db"] is not None and out["psnr_db"] >= LOSSY_AGREE_DB \
                and out["median_row_error"] <= LOSSY_ROW_LIMIT:
            out["agrees"], out["agreement"] = True, "LOSSY"
    return out


def _raster_identity(raster: dict[str, Any]) -> dict[str, Any]:
    identity = {key: raster.get(key) for key in ("prompt_id", "file", "sha256", "handoff_id", "node_id", "batch_index")}
    return {**identity, "verified": raster.get("verified", False)}


def _output_attribution(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "UNATTRIBUTED"
    positions = {(r["prompt_id"], r["node_id"], r["batch_index"]) for r in matches}
    if len(positions) > 1:
        return "AMBIGUOUS"
    _, node, index = next(iter(positions))
    return "SUPPORTED" if node is not None and index is not None else "UNKNOWN"


def _graph_match(metrics: dict[str, Any], complete: bool) -> str:
    """What an agreeing copy of a job without recorded outputs is: its whole frame, or a partial/scaled one."""
    return "WHOLE_FRAME_GRAPH_MATCH" if complete and metrics["whole_frame"] else "PARTIAL_OR_SCALED_GRAPH_MATCH"


def link(case_dir: Path, regen_dir: Path | None = None) -> dict[str, Any]:
    memory = memory_images(case_dir)
    regens = case_regenerations(case_dir) if regen_dir is None else regenerated_images(regen_dir)
    manifest_bytes = (case_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    recorded = manifest.get("findings", {}).get("workflows", [])
    # Jobs whose own history ended without any recorded output. Their regenerations (exact or partial) never
    # attribute, link or promote a memory image; what agrees with them is kept as labelled context.
    unproduced = terminal_without_outputs_ids(w.get("execution_parameters") or {} for w in recorded)
    peers = replay_graph_peers(recorded, unproduced)
    unproduced_workflows = {w.get("workflow_id") for w in recorded
                            if (w.get("execution_parameters") or {}).get("prompt_id") in unproduced}
    verified_workflows = {r["record"]["workflow_id"] for r in regens
                          if r.get("verified") and r["prompt_id"] not in unproduced}
    untested_jobs = unreplayed_complete_jobs(recorded, {r["prompt_id"] for r in regens if r.get("verified")},
                                             unproduced)
    candidates = []
    checks: list[list[dict[str, Any]]] = [[] for r in regens]
    context_checks: list[list[dict[str, Any]]] = [[] for r in regens]
    context_by_job: dict[str, list[dict[str, Any]]] = {}
    for m in memory:
        results = []
        context_matches = []
        for index, r in enumerate(regens):
            metrics = compare(r["rgb"], m["rgb"], m.get("gap_rows", []))
            if metrics is not None:
                # Exact only when the copy is the job's whole frame at native resolution, row for row.
                metrics["whole_frame"] = (metrics["scale"] == 1 and metrics["row_offset"] == 0
                                          and m["rgb"].shape == r["rgb"].shape
                                          and metrics["rows_compared"] == r["rgb"].shape[0]
                                          and np.array_equal(m["rgb"], r["rgb"]))
                replay = r.get("input_evidence") or {}
                if metrics["agrees"] and (target_is_input(replay.get("preflight_check") or {}, m)
                                          or (not (m["complete"] and metrics["whole_frame"])
                                              and not partial_rows_distinguish_inputs(replay, m["artifact_id"]))):
                    metrics["input_ambiguity"] = True
                agreed = metrics["agrees"] and not metrics.get("input_ambiguity")
                if r["prompt_id"] in unproduced:
                    if agreed:
                        relation = _graph_match(metrics, m["complete"])
                        context_matches.append({**_raster_identity(r), "metrics": metrics, "relation": relation,
                                                "output_credit": "NONE"})
                        context_checks[index].append({"artifact_id": m["artifact_id"], "kind": m["kind"],
                                                      "relation": relation, **metrics})
                        context_by_job.setdefault(r["prompt_id"], []).append({
                            "artifact_id": m["artifact_id"], "kind": m["kind"], "agreement": metrics["agreement"],
                            "complete": m["complete"], "relation": relation, "output_credit": "NONE"})
                    continue
                results.append((_raster_identity(r), metrics))
                if agreed:
                    checks[index].append({"artifact_id": m["artifact_id"], "kind": m["kind"], **metrics})
        # Several regenerations of one job are one job.
        results.sort(key=lambda pair: (not pair[1]["agrees"], not pair[0]["verified"], not pair[1]["whole_frame"],
                                       pair[1]["agreement"] != "ROWS", pair[1]["median_row_error"],
                                       -pair[1]["rows_compared"]))
        matches = [{**r, "metrics": met} for r, met in results if met["agrees"]]
        replayed_agreement = {r["prompt_id"] for r in matches}
        indistinguishable = {peer for pid in replayed_agreement for peer in peers.get(pid, ())}
        agreeing = sorted(replayed_agreement | indistinguishable)
        # A workflow of a job that ended without outputs cannot be the producer, so it need not be replayed.
        missing = set(m.get("ambiguous_workflows", [])) - verified_workflows - unproduced_workflows
        if matches and not any(r["verified"] and m["complete"] and r["metrics"]["whole_frame"] for r in matches):
            missing |= untested_jobs
        input_ambiguity = any(match["metrics"].get("input_ambiguity") for match in matches)
        supported = len(agreeing) == 1 and not missing and not input_ambiguity
        status = "SUPPORTED" if supported else "AMBIGUOUS" if agreeing else "UNATTRIBUTED"
        job = agreeing[0] if supported else None
        selected = results[0][0] if job else None
        best = results[0][1] if results else None
        candidates.append({"artifact_id": m["artifact_id"], "kind": m["kind"], "complete": m["complete"],
                           "whole_frame": bool(job and best and best["whole_frame"]),
                           "agreement": best["agreement"] if job and best else None,
                           "matching_rasters": matches, "selected_raster": selected,
                           "output_attribution": (
                               "AMBIGUOUS" if input_ambiguity or indistinguishable or (missing and matches)
                               else _output_attribution(matches)),
                           "unreplayed_workflow_ids": sorted(missing),
                           "attribution": status, "prompt_id": job, "agreeing_jobs": agreeing,
                           "best": best,
                           **({"same_graph_candidate_jobs": sorted(indistinguishable)} if indistinguishable else {}),
                           **({"contextual_matches": context_matches} if context_matches else {})})
    decoded = []
    for d in latent_decodes(case_dir):
        scores: dict[str, float] = {}
        matches = []
        context_latents = []
        for r in regens:
            if r["rgb"].shape == d["rgb"].shape:
                db = psnr(r["rgb"], d["rgb"])
                if r["prompt_id"] in unproduced:
                    if db >= LATENT_AGREE_DB:
                        context_latents.append({**_raster_identity(r), "psnr_db": db, "output_credit": "NONE"})
                    continue
                scores[r["prompt_id"]] = max(scores.get(r["prompt_id"], float("-inf")), db)
                if db >= LATENT_AGREE_DB:
                    matches.append({**_raster_identity(r), "psnr_db": db})
        replayed_agreement = {pid for pid, db in scores.items() if db >= LATENT_AGREE_DB}
        indistinguishable = {peer for pid in replayed_agreement for peer in peers.get(pid, ())}
        agreeing = sorted(replayed_agreement | indistinguishable)
        status = "SUPPORTED" if len(agreeing) == 1 and not untested_jobs else (
            "AMBIGUOUS" if agreeing else "UNATTRIBUTED")
        decoded.append({**{k: v for k, v in d.items() if k != "rgb"}, "origin": "DECODED_FROM_MEMORY_LATENT",
                        "attribution": status, "prompt_id": agreeing[0] if status == "SUPPORTED" else None,
                        "matching_rasters": matches,
                        "unreplayed_workflow_ids": sorted(untested_jobs) if matches else [],
                        "output_attribution": "AMBIGUOUS" if status == "AMBIGUOUS" else _output_attribution(matches),
                        "selected_raster": _raster_identity(max(matches, key=lambda r: r["psnr_db"]))
                        if status == "SUPPORTED" else None, "agreeing_jobs": agreeing, "psnr_db_by_job": scores,
                        **({"contextual_matches": context_latents} if context_latents else {})})
    regenerated = []
    for r, evidence, context in zip(regens, checks, context_checks, strict=True):
        if r["prompt_id"] in unproduced:
            # Memory agrees with a job that ended without outputs: context, never a residue check of an output.
            regenerated.append({**_raster_identity(r), "origin": "REGENERATED",
                                "residue_check": "CONTEXT_ONLY" if context else "UNCHECKED",
                                "agreeing_memory_images": [], "contextual_memory_images": context,
                                OUTPUT_CONTEXT_FIELD: TERMINAL_WITHOUT_RECORDED_OUTPUTS, "record": r["record"]})
            continue
        matched = "MATCHED" if evidence else "UNCHECKED"
        regenerated.append({**_raster_identity(r),
                            "origin": "REGENERATED", "residue_check": matched,
                            "agreeing_memory_images": evidence, "record": r["record"]})
    jobs = _jobs(candidates, decoded, regenerated, unproduced, context_by_job)
    extra = ({"terminal_without_recorded_outputs": {
        "field": OUTPUT_CONTEXT_FIELD, "value": TERMINAL_WITHOUT_RECORDED_OUTPUTS, "prompt_ids": sorted(unproduced),
        "meaning": "The job's own history ends interrupted or in error with explicitly empty outputs. Its "
                   "regenerations earn no attribution, link or promotion; agreeing memory images are listed as "
                   "contextual matches only."}} if unproduced else {})
    return {"schema_version": 1, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "thresholds": {"agree_fraction": AGREE_FRACTION, "row_limit_by_scale": ROW_LIMIT,
                           "decoded_latent_psnr_db": LATENT_AGREE_DB,
                           "lossy_copy": {"psnr_db": LOSSY_AGREE_DB, "median_row_error": LOSSY_ROW_LIMIT},
                           "link_rule": "SUPPORTED requires one agreeing job and distinction from known inputs"},
            "memory_images": candidates, "decoded_latents": decoded, "regenerated": regenerated,
            "jobs": jobs, "promotions": _promotions(jobs, candidates, manifest), **extra}


CLOSE_MATCH = "REGENERATION_CLOSE_MATCH"


def _promotions(jobs: list[dict[str, Any]], candidates: list[dict[str, Any]],
                manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Tier promotions per job.

    E2: each exact whole-frame memory image of an OBSERVED_EXACT job, paired with every complete API prompt
    recovered for that job (byte-distinct copies of one prompt are one finding in the report).

    E3: the job's other linked memory images (close matches: LOSSY, partial, flagged, shifted or scaled copies,
    and exact ones when the job has no complete prompt), together, with every API prompt recovered for the job,
    complete or partial. A close match is linked but not exact, which is what E3 means."""
    by_id = {c["artifact_id"]: c for c in candidates}
    out = []
    for job in jobs:
        if job.get(OUTPUT_CONTEXT_FIELD) == TERMINAL_WITHOUT_RECORDED_OUTPUTS:
            continue  # no regeneration of a job without recorded outputs gives a tier
        recovered = sorted((w["workflow_id"], w.get("completeness")) for w in manifest["findings"].get("workflows", [])
                           if w.get("prompt_format") == "API_GRAPH"
                           and (w.get("execution_parameters") or {}).get("prompt_id") == job["prompt_id"])
        complete = [workflow_id for workflow_id, completeness in recovered if completeness == "COMPLETE"]
        observed_verified = [
            o for o in job["observed"] if (by_id[o["artifact_id"]]["selected_raster"] or {}).get("verified")
        ]
        exact: list[str] = []
        for observed in observed_verified if job["image_status"] == "OBSERVED_EXACT" and complete else []:
            c = by_id[observed["artifact_id"]]
            if c["kind"] in ("PIXEL_BUFFER", "TENSOR_IMAGE") and c["complete"] and c["whole_frame"]:
                exact.append(c["artifact_id"])
                out.append({"tier": "E2", "basis": "REGENERATION_WHOLE_FRAME_MATCH", "prompt_id": job["prompt_id"],
                            "artifact_id": c["artifact_id"], "workflow_ids": complete,
                            "agreement": c["best"], "regenerated": c["selected_raster"]["file"],
                            **{k: c[k] for k in ("selected_raster", "matching_rasters", "output_attribution")}})
        close = [o["artifact_id"] for o in observed_verified if o["artifact_id"] not in exact]
        if close and recovered:
            selected = by_id[close[0]]["selected_raster"]
            if any(by_id[a]["selected_raster"] != selected for a in close):
                selected = None
            out.append({"tier": "E3", "basis": CLOSE_MATCH, "prompt_id": job["prompt_id"], "artifact_ids": close,
                        "workflow_ids": [workflow_id for workflow_id, _ in recovered],
                        "agreement": {a: {"agreement": by_id[a]["agreement"], **(by_id[a]["best"] or {})}
                                      for a in close},
                        "regenerated": selected["file"] if selected else None, "selected_raster": selected,
                        "raster_evidence_by_artifact": {
                            a: {k: by_id[a][k] for k in ("selected_raster", "matching_rasters", "output_attribution")}
                            for a in close}})
    return out


def _jobs(candidates: list[dict[str, Any]], decoded: list[dict[str, Any]], regenerated: list[dict[str, Any]],
          unproduced: Collection[str] = (),
          context: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    jobs = []
    for prompt_id in dict.fromkeys(r["prompt_id"] for r in regenerated):
        outputs = [r for r in regenerated if r["prompt_id"] == prompt_id]
        linked = [c for c in candidates if c["prompt_id"] == prompt_id and c["attribution"] == "SUPPORTED"]
        observed = [{"artifact_id": c["artifact_id"], "kind": c["kind"], "agreement": c["agreement"]} for c in linked]
        exact = [c for c in linked if c["kind"] in ("PIXEL_BUFFER", "TENSOR_IMAGE") and c["complete"]
                 and c["whole_frame"]]
        latent_links = [d for d in decoded if d["prompt_id"] == prompt_id and d["attribution"] == "SUPPORTED"]
        latent = [{"decode_id": d["decode_id"], "latent_id": d["latent_id"], "file": d["file"]} for d in latent_links]
        status = ("OBSERVED_EXACT" if exact else "OBSERVED_PARTIAL_OR_SCALED" if linked
                  else "DECODED_FROM_MEMORY_LATENT" if latent else "REGENERATED_ONLY")
        evidence = exact or linked or latent_links
        selected = evidence[0]["selected_raster"] if evidence else None
        r = next((out for out in outputs if _raster_identity(out) == selected), outputs[0])
        entry = {"prompt_id": prompt_id, "image_status": status, "observed": observed,
                 "decoded_latents": latent, "rasters": [_raster_identity(output) for output in outputs],
                 "regenerated": {"file": r["file"], "residue_check": r["residue_check"]}}
        if prompt_id in unproduced:
            # Nothing credits this job (``observed`` and ``decoded_latents`` stay empty); matches are context.
            entry = {**entry, OUTPUT_CONTEXT_FIELD: TERMINAL_WITHOUT_RECORDED_OUTPUTS,
                     "contextual_observed": (context or {}).get(prompt_id, [])}
        jobs.append(entry)
    return jobs


def write_links(case_dir: Path, regen_dir: Path | None = None) -> Path:
    target = case_dir / LINKS_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; links are written once")
    document = link(case_dir, regen_dir)
    target.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    card = case_dir / "scorecard.json"
    if card.is_file():
        from comfytrace.evidence import apply_image_links

        evidence_path = case_dir / "recovery-evidence.json"
        sources = ({r["record_id"]: r.get("memory_source") for r in
                    json.loads(evidence_path.read_text(encoding="utf-8")).get("records", [])}
                   if evidence_path.is_file() else {})
        updated = apply_image_links(json.loads(card.read_text(encoding="utf-8")), document, sources)
        card.write_text(json.dumps(updated, indent=1, default=str) + "\n", encoding="utf-8")
    return target
