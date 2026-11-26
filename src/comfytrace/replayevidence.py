"""Apply whole-frame replay search evidence to the report's in-memory findings."""

from __future__ import annotations

import hashlib
import json
import mmap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from comfytrace.casefiles import sha256_file
from comfytrace.correlate import replay_graph_peers
from comfytrace.exceptions import MissingDependencyError
from comfytrace.findings import target_is_input
from comfytrace.lime import LimePhysicalImage, parse_lime_segments
from comfytrace.link import case_regenerations
from comfytrace.memrecover import terminal_without_outputs_ids
from comfytrace.regenerate import TieredEvaluator, recovered_prompts
from comfytrace.replaysearch import REPORT_NAME, read_physical


def physical_bytes(memory: Path, offset: int, length: int) -> bytes | None:
    """Exactly ``length`` bytes at physical address ``offset`` of the capture, or None. A LiME v1 capture is read
    through its segment table, so the address is a real RAM address and a span touching a hole reads nothing; a
    flat image is indexed by file offset."""
    segments = parse_lime_segments(memory)
    with memory.open("rb") as handle:
        if memory.stat().st_size == 0:
            return None
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            view: Any = LimePhysicalImage(mapped, segments) if segments is not None else mapped
            return read_physical(view, offset, length)


def _frame(case_dir: Path, memory: Path, hit: dict[str, Any], shape: tuple[int, ...]) -> Image.Image | None:
    layout = hit.get("layout")
    digest = hit.get("raw_sha256")
    if layout not in ("RGB", "BGR", "RGBA", "BGRA", "BGRX") or not isinstance(digest, str):
        return None
    root = (case_dir / "exports/raw/replay-search").resolve()
    path = (case_dir / hit["raw_file"]).resolve()
    if path.parent != root or path.name != f"{digest}.bin":
        return None
    height, width, _ = shape
    size = height * width * len(layout)
    if not path.is_file() or size != hit.get("length_bytes") or path.stat().st_size != size:
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        return None
    offset = hit.get("physical_offset")
    if type(offset) is not int or physical_bytes(memory, offset, size) != data:
        return None
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, len(layout))
    if layout.endswith("A") and not np.all(pixels[:, :, 3] == 255):
        return None
    rgb = pixels[:, :, [2, 1, 0]] if layout.startswith("B") else pixels[:, :, :3]
    return Image.fromarray(rgb)


def apply_replay_evidence(
    findings: list[dict[str, Any]], manifest: dict[str, Any], manifest_sha: str, case_dir: Path,
    regenerations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = case_dir / REPORT_NAME
    if not path.is_file():
        return []
    scan = json.loads(path.read_bytes())
    if scan.get("manifest_sha256") != manifest_sha:
        return []
    if scan.get("memory_sha256") != manifest["case_metadata"]["memory_image"]["sha256"]:
        return []
    hits = [hit for hit in scan.get("matches", []) if hit.get("scale") == 1 and hit.get("whole_frame") is True
            and isinstance(hit.get("raw_file"), str)]
    if not hits or not isinstance(scan.get("memory_path"), str):
        return []
    memory = Path(scan["memory_path"])
    if not memory.is_file() or sha256_file(memory) != scan["memory_sha256"]:
        return []

    # A job that ended without recorded outputs produced nothing to promote, and needs no replay coverage.
    recorded = manifest["findings"].get("workflows", [])
    unproduced = terminal_without_outputs_ids(w.get("execution_parameters") or {} for w in recorded)
    peers = replay_graph_peers(recorded, unproduced)
    workflows = {workflow["workflow_id"] for workflow in recovered_prompts(manifest)
                 if (workflow.get("execution_parameters") or {}).get("prompt_id") not in unproduced}
    run_ids = set(scan.get("searched_handoffs", []))
    by_handoff = {record["handoff_id"]: record for record in regenerations}
    run_ids.intersection_update(by_handoff)
    rasters = [r for r in case_regenerations(case_dir) if r["verified"] and r["handoff_id"] in run_ids
               and r["record"]["workflow_id"] in workflows]
    if {r["record"]["workflow_id"] for r in rasters} != workflows:
        return []
    images = [(r, Image.fromarray(r["rgb"])) for r in rasters]
    keys = {(r["handoff_id"], r["file"], r["sha256"]): r for r in rasters}
    jobs_by_range = defaultdict(set)
    for hit in hits:
        key = (hit.get("handoff_id"), hit.get("regenerated_image"), hit.get("regenerated_sha256"))
        if raster := keys.get(key):
            jobs_by_range[hit.get("physical_offset"), hit.get("length_bytes")].add(raster["prompt_id"])
    evaluator = TieredEvaluator()
    notes = []
    for hit in hits:
        if len(jobs_by_range[hit.get("physical_offset"), hit.get("length_bytes")]) > 1:
            continue
        key = (hit.get("handoff_id"), hit.get("regenerated_image"), hit.get("regenerated_sha256"))
        reference = keys.get(key)
        if (reference is None or reference["prompt_id"] in peers
                or (frame := _frame(case_dir, memory, hit, reference["rgb"].shape)) is None):
            continue
        pixel_sha = hashlib.sha256(frame.tobytes()).hexdigest()
        compared = []
        try:
            for raster, image in images:
                if image.size != frame.size:
                    continue
                tier, subtier, metrics = evaluator.evaluate_intact(image, frame, pixel_sha)
                if tier == "E1":
                    compared.append((raster, subtier, metrics))
        except MissingDependencyError:
            continue
        if {r["prompt_id"] for r, _, _ in compared} != {reference["prompt_id"]}:
            continue
        outcome = next(((subtier, metrics) for r, subtier, metrics in compared if r is reference), None)
        if outcome is None:
            continue
        preflight = by_handoff[reference["handoff_id"]]["preflight_check"]
        if target_is_input(preflight, {"pixel_sha256": pixel_sha}):
            continue
        # Older records might contain only PNG container hashes for conditioning images.
        if set(preflight.get("input_images") or {}) - set(preflight.get("input_image_pixel_sha256") or {}):
            continue
        workflow = reference["record"]["workflow_id"]
        eligible = [f for f in findings if workflow in f["artifact_ids"] and f["base_tier"] in ("E2", "E3", "E5")]
        if len(eligible) != 1 or eligible[0]["tier"] == "E1":
            continue
        subtier, metrics = outcome
        finding = eligible[0]
        finding.update(tier="E1", base_tier="E2", image_link="RAW_REPLAY_WHOLE_FRAME",
                       regeneration=reference["handoff_id"],
                       replay_evidence={**hit, "subtier": subtier, "metrics": metrics, "pixel_sha256": pixel_sha})
        notes.append({"handoff_id": reference["handoff_id"], "applied_to": finding["finding_id"], "note": subtier})
    return notes
