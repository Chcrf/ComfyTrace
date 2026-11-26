"""Search a raw acquisition for packed pixel copies of case-bound regenerations."""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from PIL import Image

from comfytrace.casefiles import load_regenerations
from comfytrace.ingest import supports_raw_search
from comfytrace.link import case_regenerations

log = logging.getLogger(__name__)
PAGE = 4096
CHUNK = 16 * 1024 * 1024
ANCHOR_PIXELS = 6
ANCHORS_PER_LAYOUT = 4096
MIN_PIXELS = 256
MAX_MEAN_ERROR = 4
MIN_NEAR_FRACTION = 0.90
REPORT_NAME = "replay-search.json"


@dataclass(frozen=True)
class Reference:
    workflow: str
    prompt: str | None  # None: a graph without a recovered prompt ID, whose hits name no producer
    handoff: str
    file: str
    sha256: str
    layout: str
    scale: int
    pixels: np.ndarray

    @property
    def channels(self) -> int:
        return self.pixels.shape[1]


def _references(case_dir: Path, selected: set[str], handoffs: set[str] | None) -> list[Reference]:
    refs = []
    workflows = set()
    for raster in case_regenerations(case_dir, unbound=True):
        workflow = raster["record"]["workflow_id"]
        if not raster["verified"] or (selected and workflow not in selected):
            continue
        if handoffs is not None and raster["handoff_id"] not in handoffs:
            continue
        workflows.add(workflow)
        rgb = Image.fromarray(raster["rgb"])
        for scale in (1, 2):
            if rgb.width % scale or rgb.height % scale:
                continue
            image = rgb if scale == 1 else rgb.resize((rgb.width // scale, rgb.height // scale), Image.Resampling.BOX)
            if image.width < ANCHOR_PIXELS:
                continue
            for layout in ("RGB", "BGR", "RGBA", "BGRA", "BGRX"):
                mode = "RGBA" if len(layout) == 4 else "RGB"
                pixels = np.asarray(image.convert(mode)).reshape(-1, len(layout))
                if layout.startswith("B"):
                    pixels = pixels[:, [2, 1, 0, 3] if len(layout) == 4 else [2, 1, 0]].copy()
                if layout == "BGRX":
                    pixels[:, 3] = 0
                refs.append(Reference(workflow, raster["prompt_id"], raster["handoff_id"], raster["file"],
                                      raster["sha256"], layout, scale, pixels))
    if missing := selected - workflows:
        raise ValueError(f"no verified regeneration for: {', '.join(sorted(missing))}")
    if not refs:
        raise ValueError("no verified regenerated images to search for")
    return refs


def _anchors(refs: list[Reference]) -> dict[bytes, list[tuple[int, int]]]:
    # Spread anchors across each image. Discard repeated patterns within the same layout.
    anchors: dict[bytes, list[tuple[int, int]]] = {}
    for idx, ref in enumerate(refs):
        step = max(1, len(ref.pixels) // ANCHORS_PER_LAYOUT)
        seen: dict[bytes, int | None] = {}
        for pos in range(0, len(ref.pixels) - ANCHOR_PIXELS + 1, step):
            pixels = ref.pixels[pos:pos + ANCHOR_PIXELS]
            if int(np.ptp(pixels[:, :3], axis=0).max()) < 3:
                continue
            pattern = pixels.tobytes()
            seen[pattern] = None if pattern in seen else pos
        for pattern, pos in seen.items():
            if pos is not None:
                anchors.setdefault(pattern, []).append((idx, pos))
    return anchors


def _compare(data: bytes, pixels: np.ndarray, channels: int) -> tuple[float, float] | None:
    observed = np.frombuffer(data, dtype=np.uint8).reshape(-1, channels)
    diff = np.abs(observed[:, :3].astype(np.int16) - pixels[:, :3])
    mean = float(diff.mean())
    if mean > MAX_MEAN_ERROR:
        return None
    near = float((diff.max(axis=1) <= 8).mean())
    if near < MIN_NEAR_FRACTION:
        return None
    return mean, near


def _verify(memory: mmap.mmap, ref: Reference, offset: int, pixel: int) -> dict | None:
    channels = ref.channels
    origin = offset - pixel * channels
    page = offset // PAGE * PAGE
    first = max(0, (page - origin + channels - 1) // channels)
    last = min(len(ref.pixels), (min(page + PAGE, len(memory)) - origin) // channels)
    if last - first < MIN_PIXELS or np.ptp(ref.pixels[first:last, :3], axis=0).max() < 16:
        return None
    start = origin + first * channels
    data = read_physical(memory, start, (last - first) * channels)
    if data is None:
        return None
    scores = _compare(data, ref.pixels[first:last], channels)
    if scores is None:
        return None
    full = False
    whole = read_physical(memory, origin, ref.pixels.nbytes)
    if whole is not None:
        # Check both ends before allocating the whole-frame difference.
        ends_match = (
            _compare(whole[:MIN_PIXELS * channels], ref.pixels[:MIN_PIXELS], channels)
            and _compare(whole[-MIN_PIXELS * channels:], ref.pixels[-MIN_PIXELS:], channels)
        )
        if ends_match and (whole_scores := _compare(whole, ref.pixels, channels)) is not None:
            start, data, scores, first, last = origin, whole, whole_scores, 0, len(ref.pixels)
            full = True
    return {
        "workflow_id": ref.workflow, "prompt_id": ref.prompt, "handoff_id": ref.handoff,
        "producer": "RECOVERED_PROMPT_ID" if ref.prompt else "UNRESOLVED",
        "regenerated_image": ref.file, "regenerated_sha256": ref.sha256, "layout": ref.layout, "scale": ref.scale,
        "physical_offset": start, "length_bytes": len(data), "raw_sha256": hashlib.sha256(data).hexdigest(),
        "first_reference_pixel": first, "matched_pixels": last - first, "reference_pixels": len(ref.pixels),
        "whole_frame": full, "mean_absolute_error": scores[0], "near_pixel_fraction": scores[1],
    }


def read_physical(memory: Any, offset: int, length: int) -> bytes | None:
    """Exactly ``length`` genuine bytes at physical address ``offset`` of ``memory`` (a flat image as bytes or an
    mmap, or a ``LimePhysicalImage``), or None when the range leaves the image or touches an unmapped hole."""
    if offset < 0 or length < 0 or offset + length > len(memory):
        return None
    data = bytes(memory[offset:offset + length])
    return data if len(data) == length else None


def verify_linked_pngs(
    case_dir: Path, memory: Any, manifest: dict, manifest_sha: str, handoffs: set[str],
) -> list[dict]:
    """Reread intact PNG targets from their recorded physical pages.

    ``memory`` is indexed by physical address: the flat image itself, or a ``LimePhysicalImage`` over a LiME
    capture. A page or span that is not genuinely present (beyond the image, in a hole) verifies nothing."""
    evidence_file = case_dir / "recovery-evidence.json"
    if not evidence_file.is_file():
        return []
    evidence = {item["record_id"]: item for item in json.loads(evidence_file.read_bytes())["records"]}
    artifacts = {item["artifact_id"]: item for item in manifest["findings"]["artifacts"]}
    verified = []
    for replay in load_regenerations(case_dir / "regenerations"):
        if replay["handoff_id"] not in handoffs or replay["target_manifest_sha256"] != manifest_sha:
            continue
        target = replay["evaluation"].get("target_artifact_id")
        artifact = artifacts.get(target)
        if not artifact or artifact["carve_type"] != "ENCODED_PNG" or artifact["completeness"] != "COMPLETE":
            continue
        source = artifact["raw_file_paths"][0]
        raw = (case_dir / source).resolve()
        if not raw.is_relative_to((case_dir / "exports/raw").resolve()) or not raw.is_file():
            continue
        expected = raw.read_bytes()
        digest = hashlib.sha256(expected).hexdigest()
        if digest != artifact["container_sha256"]:
            continue
        record = evidence.get(target, {})
        pages = (record.get("assembly") or {}).get("page_offsets", [])
        spans = []
        if pages:
            offsets = [int(page, 16) for page in pages]
            contents = [read_physical(memory, page, PAGE) for page in offsets]
            if any(content is None for content in contents):
                continue
            data = b"".join(contents)  # type: ignore[arg-type]
            start = data.find(expected, 0, PAGE - 1 + len(expected))
            if start >= 0:
                remaining = len(expected)
                for page in offsets:
                    take = min(PAGE - start, remaining)
                    spans.append({"physical_offset": page + start, "length_bytes": take})
                    remaining -= take
                    if not remaining:
                        break
                    start = 0
        else:
            for physical in artifact.get("physical_offsets", []):
                offset = int(physical, 16)
                if read_physical(memory, offset, len(expected)) == expected:
                    spans = [{"physical_offset": offset, "length_bytes": len(expected)}]
                    break
        if spans:
            verified.append({"artifact_id": target, "workflow_id": replay["workflow_id"],
                             "handoff_id": replay["handoff_id"], "raw_file": source,
                             "raw_sha256": digest, "spans": spans})
    return verified


def search(case_dir: Path, memory_path: Path, workflows: set[str], *, handoffs: set[str] | None = None) -> Path:
    manifest_bytes = (case_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not supports_raw_search(manifest["case_metadata"], memory_path):
        raise ValueError("replay search requires a flat physical capture")
    from ahocorasick_rs import BytesAhoCorasick

    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected = manifest["case_metadata"]["memory_image"]["sha256"]
    refs = _references(case_dir, workflows, handoffs)
    anchors = _anchors(refs)
    if not anchors:
        raise ValueError("none of the selected images has usable pixel anchors")
    patterns = list(anchors)
    matcher = BytesAhoCorasick(patterns)
    matches = {}
    whole_origins = set()
    from comfytrace.lime import LimePhysicalImage, parse_lime_segments

    segments = parse_lime_segments(memory_path)
    with memory_path.open("rb") as handle:
        log.info("Verifying dump SHA-256 before searching %d pixel anchors", len(patterns))
        digest = hashlib.sha256()
        for data in iter(lambda: handle.read(CHUNK), b""):
            digest.update(data)
        if digest.hexdigest() != expected:
            raise ValueError(f"{memory_path} is not the image this case was extracted from")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as raw_mmap:
            memory: Any = LimePhysicalImage(raw_mmap, segments) if segments is not None else raw_mmap
            linked_pngs = verify_linked_pngs(case_dir, memory, manifest, manifest_sha, {ref.handoff for ref in refs})
            ranges = (
                [(seg.s_addr, seg.end_exclusive) for seg in segments]
                if segments is not None
                else [(0, len(memory))]
            )
            for seg_start, seg_end in ranges:
                for base in range(seg_start, seg_end, CHUNK):
                    if base % (2 ** 31) == 0 and base:
                        log.info("Searched %.0f GiB of raw memory", base / (2 ** 30))
                    chunk_end = min(seg_end, base + CHUNK + ANCHOR_PIXELS * 4 - 1)
                    chunk = memory[base:chunk_end]
                    tested = set()
                    for pattern_id, start, _ in matcher.find_matches_as_indexes(chunk, overlapping=True):
                        if start >= CHUNK:
                            continue
                        offset = base + start
                        for idx, pixel in anchors[patterns[pattern_id]]:
                            ref = refs[idx]
                            origin = offset - pixel * ref.channels
                            key = idx, origin, offset // PAGE
                            if key in tested or (idx, origin) in whole_origins:
                                continue
                            tested.add(key)
                            if found := _verify(memory, ref, offset, pixel):
                                if found["whole_frame"]:
                                    whole_origins.add((idx, origin))
                                    if found["scale"] == 1:
                                        digest = found["raw_sha256"]
                                        path = Path("exports/raw/replay-search") / f"{digest}.bin"
                                        start = found["physical_offset"]
                                        _publish(case_dir / path, memory[start:start + found["length_bytes"]])
                                        found["raw_file"] = path.as_posix()
                                matches[key] = found
    hits = [hit for (idx, origin, _), hit in matches.items()
            if hit["whole_frame"] or (idx, origin) not in whole_origins]
    hits.sort(key=lambda hit: (hit["workflow_id"], not hit["whole_frame"], hit["physical_offset"]))
    results = {
        "schema_version": 1, "memory_path": str(memory_path.resolve()), "memory_sha256": expected,
        "manifest_sha256": manifest_sha, "verified_linked_pngs": linked_pngs,
        "searched_workflows": sorted({ref.workflow for ref in refs}),
        "searched_handoffs": sorted({ref.handoff for ref in refs}), "anchors": len(patterns),
        "unbound_workflows": sorted({ref.workflow for ref in refs if not ref.prompt}),
        "matches": hits,
    }
    path = case_dir / REPORT_NAME
    _publish(path, (json.dumps(results, indent=2) + "\n").encode("utf-8"))
    return path


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(data)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
