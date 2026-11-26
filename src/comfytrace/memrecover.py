"""Turns pure-memory recovery output into manifest records.

Stage 1 (CPython objects, torch tensors, raw pixel buffers) and stage 2's decoded-bitmap scan produce image
candidates with no embedded metadata. Each candidate is checked against every other: an independent copy
(same size, or a 2x box-downscaled thumbnail) that agrees row by row corroborates it, and rows that disagree
are listed as suspect. Confidence is reported, never assumed:

* ``CORROBORATED``: an independent copy agrees on at least 80% of the overlapping rows;
* ``SINGLE_COPY``: no other copy exists, so exactness is unproven.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image

from comfytrace.bitmaps import DecodedBitmap
from comfytrace.carve import (
    IdAllocator,
    canonical_json,
    execution_parameters,
    graph_dependencies,
    graph_is_complete,
)
from comfytrace.models import (
    CarveType,
    Completeness,
    GapDescriptor,
    LinkClass,
    RawArtifactRecord,
    WorkflowRecord,
)
from comfytrace.pyheap import PixelBuffer, QueueItem
from comfytrace.torchmem import ImageTensor

MIN_OVERLAP = 0.75
AGREE_FRACTION = 0.8


@dataclass
class ImageCandidate:
    kind: str
    rgb: np.ndarray
    raw: bytes
    process_id: int | None
    virtual_address: int | None
    physical_offsets: list[int]
    source: dict[str, Any]
    complete: bool
    corroborations: list[dict[str, Any]] = field(default_factory=list)
    suspect_rows: set[int] = field(default_factory=set)
    stale_rows: frozenset[int] = frozenset()
    pages: list[int] = field(default_factory=list)  # physical page numbers holding the pixels
    page_origins: list[dict[str, Any]] = field(default_factory=list)
    gap_rows: list[tuple[int, int, str]] = field(default_factory=list)  # (first, end, reason): no genuine pixels
    execution_cache: dict[str, Any] | None = None  # the ComfyUI cache entry holding this tensor (execcache)
    # Gap rows as they read from frames of uncertain ownership (bitmaps.Gaps.uncertain): row -> pixels, the
    # (first, end, frames) ranges, and what other copies say about them. Never part of ``rgb`` or ``raw``.
    uncertain_rows: dict[int, np.ndarray] = field(default_factory=dict)
    uncertain_ranges: list[list[Any]] = field(default_factory=list)
    uncertain_checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _pages(physical_of: Any, address: int, length: int) -> list[int]:
    """Physical page numbers behind [address, address + length) of a process; non-resident pages are left out."""
    pages = []
    for page in range(address >> 12, (address + length - 1 >> 12) + 1):
        phys = physical_of(page << 12)
        if phys is not None:
            pages.append(phys >> 12)
    return pages


def from_pixel_buffer(buf: PixelBuffer, physical_of: Any) -> ImageCandidate:
    arr = np.frombuffer(buf.data, np.uint8).reshape(buf.height, buf.width, buf.channels)[:, :, :3]
    phys = physical_of(buf.address)
    return ImageCandidate(CarveType.PIXEL_BUFFER, np.ascontiguousarray(arr), buf.data, buf.process_id, buf.address,
                          [phys] if phys is not None else [],
                          {"object": "CPython bytes", "state": buf.state, "channels": buf.channels,
                           "row_continuity_score": round(buf.row_score, 2)}, True,
                          suspect_rows=set(buf.stale_rows), stale_rows=buf.stale_rows,
                          pages=_pages(physical_of, buf.address, len(buf.data)))


def from_tensor(t: ImageTensor, physical_of: Any, execution_cache: dict[str, Any] | None = None) -> ImageCandidate:
    """``execution_cache``: the cache entry that holds this tensor as a node output (``execcache.image_ties``)."""
    phys = physical_of(t.data_address)
    tie = None if execution_cache is None else {**execution_cache, "batch_index": t.batch_index}
    return ImageCandidate(CarveType.TENSOR_IMAGE, t.rgb, t.rgb.tobytes(), t.process_id, t.data_address,
                          [phys] if phys is not None else [],
                          {"tensorimpl": f"0x{t.tensorimpl:X}", "storage": f"0x{t.storage:X}", "sizes": t.sizes,
                           "strides": t.strides, "batch_index": t.batch_index,
                           "rendering": "clip(255*x).astype(uint8)"}, True,
                          suspect_rows=set(t.stale_rows), stale_rows=t.stale_rows,
                          pages=_pages(physical_of, *t.byte_range), execution_cache=tie)


def from_bitmap(b: DecodedBitmap) -> ImageCandidate:
    gap_rows = [(int(first), int(end), str(reason)) for first, end, reason in b.notes.get("gap_rows", [])]
    genuine = np.ones(b.rows, bool)
    for first, end, _ in gap_rows:
        genuine[first:end] = False
    # Raw bytes are the genuine rows only; the rows of a gap exist only as zeros in the in-memory array.
    return ImageCandidate(CarveType.DECODED_BITMAP, b.rgb, b.rgb[genuine].tobytes(), None, None,
                          [p << 12 for p in b.physical_pages[:1]],
                          {"region": b.region, "ordering": b.ordering, "byte_offset_in_region": b.byte_offset,
                           "physical_pages": len(b.physical_pages),
                           **{k: v for k, v in b.notes.items() if k != "uncertain_rows"}}, False,
                          pages=list(b.physical_pages), gap_rows=gap_rows, uncertain_rows=dict(b.uncertain_rows),
                          uncertain_ranges=[list(r) for r in b.notes.get("uncertain_rows", [])])


def _row_errors(big: np.ndarray, small: np.ndarray, factor: int,
                ignore_rows=()) -> tuple[int, np.ndarray] | None:
    h = big.shape[0] // factor * factor
    down = big[:h].astype(np.float32).reshape(h // factor, factor, small.shape[1], factor, 3).mean((1, 3))
    best: tuple[tuple[float, float, int, int], int, np.ndarray] | None = None
    # A partial copy can sit anywhere in the other image; a short overlap of flat rows would agree by chance.
    need = max(16, int(min(small.shape[0], down.shape[0]) * MIN_OVERLAP))
    genuine = np.ones(small.shape[0], dtype=bool)
    genuine[np.asarray(ignore_rows, dtype=int)] = False
    for dy in range(-small.shape[0] + need, down.shape[0] - need + 1):
        lo, hi = max(0, -dy), min(small.shape[0], down.shape[0] - dy)
        mask = genuine[lo:hi]
        if mask.sum() < need:
            continue
        err = np.abs(down[lo + dy : hi + dy] - small[lo:hi].astype(np.float32)).mean((1, 2))
        # Median first (robust to damaged rows), then mean: textures with repeated rows tie on the median.
        valid = err[mask]
        score = (float(np.median(valid)), float(valid.mean()), -len(valid), abs(dy))
        if best is None or score < best[0]:
            err[~mask] = np.nan
            best = (score, dy, np.concatenate([np.full(lo, np.nan), err, np.full(small.shape[0] - hi, np.nan)]))
    return None if best is None else (best[1], best[2])


def corroborate(candidates: list[ImageCandidate]) -> None:
    """Compares every pair whose widths are equal or in a 2:1 ratio.

    A row is suspect when it disagrees with some copy and agrees with none: two disagreeing copies cannot say
    which of them is damaged, but a third copy that agrees clears that row.
    """
    disagreed: dict[int, set[int]] = {id(c): set() for c in candidates}
    agreed: dict[int, set[int]] = {id(c): set() for c in candidates}
    source_pages = {id(c): set(c.pages) for c in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            # A fork or shared mapping can expose one physical copy under two process addresses.
            if not source_pages[id(a)].isdisjoint(source_pages[id(b)]):
                continue
            if (a.process_id is not None and a.process_id == b.process_id and a.virtual_address is not None
                    and b.virtual_address is not None and a.virtual_address < b.virtual_address + len(b.raw)
                    and b.virtual_address < a.virtual_address + len(a.raw)):
                continue
            for big, small in ((a, b), (b, a)):
                if big.width % small.width or big.width // small.width not in (1, 2):
                    continue
                factor = big.width // small.width
                result = _row_errors(big.rgb, small.rgb, factor)
                if result is None:
                    continue
                dy, err = result
                valid = err[~np.isnan(err)]
                limit = 0.5 if factor == 1 else 3.0
                agree = valid <= limit * 2
                if not len(valid) or agree.mean() < AGREE_FRACTION:
                    continue
                bad = {r for r in range(len(err)) if not np.isnan(err[r]) and err[r] > limit * 2}
                good = {r for r in range(len(err)) if not np.isnan(err[r])} - bad
                for rows, target in ((bad, disagreed), (good, agreed)):
                    target[id(small)] |= rows
                    target[id(big)] |= {factor * (r + dy) + k for r in rows for k in range(factor)}
                _check_uncertain(small, big, factor, dy, limit * 2)
                if factor == 1:
                    _check_uncertain(big, small, 1, -dy, limit * 2)
                note = {"rows_compared": int(len(valid)), "rows_agreeing": int(agree.sum()),
                        "scale": factor, "row_offset": dy, "median_error": round(float(np.median(valid)), 3)}
                big.corroborations.append({"with": id(small), **note})
                small.corroborations.append({"with": id(big), **note})
                break
    for c in candidates:
        c.suspect_rows |= disagreed[id(c)] - agreed[id(c)]


def _check_uncertain(c: ImageCandidate, other: ImageCandidate, factor: int, dy: int, limit: float) -> None:
    """Compares ``c``'s rows of uncertain ownership with the genuine rows of an agreeing copy (``other``,
    box-downscaled by ``factor``; row r of ``c`` lies at row r + dy of it). An independent copy that agrees says
    the frame still held the picture; it is recorded, and the rows stay apart."""
    if not c.uncertain_rows:
        return
    h = other.rgb.shape[0] // factor * factor
    down = other.rgb[:h].astype(np.float32).reshape(h // factor, factor, c.width, factor, 3).mean((1, 3))
    genuine = np.ones(other.rgb.shape[0], bool)
    for first, end, _ in other.gap_rows:
        genuine[first:end] = False
    genuine[list(other.suspect_rows)] = False
    usable = genuine[:h].reshape(h // factor, factor).all(1)
    compared = agreeing = 0
    for row, pixels in c.uncertain_rows.items():
        at = row + dy
        if 0 <= at < len(down) and usable[at]:
            compared += 1
            agreeing += bool(np.abs(down[at] - pixels.astype(np.float32)).mean() <= limit)
    if compared:
        c.uncertain_checks.append({"with": id(other), "scale": factor, "rows_compared": compared,
                                   "rows_agreeing": agreeing})


UNCERTAIN_MEANING = ("The page table of the mapping that holds this image still names these frames, but each "
                     "frame's PFN entry names another PTE: the frame may have been reused since, or the two were "
                     "copied at different moments of the acquisition. The bytes read there are kept apart. They "
                     "are not part of the image, are not hashed or compared with it, and no finding rests on them. "
                     "Rows numbered from the image's height on lie on such pages directly after its last "
                     "recovered row; the image's size is not changed by them.")


def _uncertain_layer(c: ImageCandidate, assigned: dict[int, str]) -> tuple[
        dict[str, Any] | None, list[tuple[int, int, bytes]], bytes | None]:
    """Evidence entry, (first, end, RGB bytes) spans and DERIVED preview of the rows of uncertain ownership."""
    if not c.uncertain_rows:
        return None, [], None
    gap = np.zeros(c.height, bool)
    for first, end, _ in c.gap_rows:
        gap[first:end] = True
    steps = np.abs(np.diff(c.rgb.astype(np.int16), axis=0)).mean((1, 2))
    between_genuine = steps[~gap[1:] & ~gap[:-1]]
    typical = round(float(np.median(between_genuine)), 2) if len(between_genuine) else None
    # Rows numbered from the image's height on would follow its last recovered row; the preview is that tall.
    preview = np.full((max(c.height, max(end for _, end, _ in c.uncertain_ranges)), c.width, 3), (255, 0, 255),
                      np.uint8)
    preview[: c.height] = c.rgb
    for first, end, _ in c.gap_rows:
        preview[first:end] = (255, 0, 255)
    rows, spans = [], []
    for first, end, frames in c.uncertain_ranges:
        pixels = np.stack([c.uncertain_rows[r] for r in range(first, end)])
        preview[first:end] = pixels
        edges = []
        if 0 < first <= c.height and not gap[first - 1]:
            edges.append(float(np.abs(pixels[0].astype(np.int16) - c.rgb[first - 1]).mean()))
        if end < c.height and not gap[end]:
            edges.append(float(np.abs(pixels[-1].astype(np.int16) - c.rgb[end]).mean()))
        data = pixels.tobytes()
        spans.append((first, end, data))
        rows.append({"row_range": [first, end], "reason": "STALE_MAPPING", "frames_read": frames,
                     "position": "inside the image" if end <= c.height else "after the last recovered row",
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "continuity": {"edge_row_error": round(max(edges), 2) if edges else None,
                                    "typical_row_error": typical}})
    checks = [{**{k: v for k, v in check.items() if k != "with"}, "copy": assigned[check["with"]]}
              for check in c.uncertain_checks]
    return ({"meaning": UNCERTAIN_MEANING, "rows": rows, "agreeing_copies": checks,
             "rows_total": sum(end - first for first, end, _ in spans)}, spans, _png(preview))


def _gapped(c: ImageCandidate) -> tuple[list[tuple[int, bytes]], list[GapDescriptor], np.ndarray]:
    """Raw spans of the genuine rows (offsets in the full RGB layout), a gap descriptor per run of gap rows, and the
    DERIVED preview with gap rows drawn magenta. Without gaps: one span and the pixels as they are."""
    if not c.gap_rows:
        return [(0, c.raw)], [], c.rgb
    row = c.width * 3
    spans, gaps, cursor = [], [], 0
    for first, end, reason in c.gap_rows:
        if first > cursor:
            spans.append((cursor * row, c.rgb[cursor:first].tobytes()))
        gaps.append(GapDescriptor(first * row, (end - first) * row, reason))
        cursor = end
    if cursor < c.height:
        spans.append((cursor * row, c.rgb[cursor:].tobytes()))
    preview = c.rgb.copy()
    for first, end, _ in c.gap_rows:
        preview[first:end] = (255, 0, 255)
    return spans, gaps, preview


def _png(rgb: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, "PNG")
    return out.getvalue()


def image_records(candidates: list[ImageCandidate]) -> list[RawArtifactRecord]:
    ids: dict[str, IdAllocator] = {CarveType.PIXEL_BUFFER: IdAllocator("PIXBUF"),
                                   CarveType.TENSOR_IMAGE: IdAllocator("TENSOR"),
                                   CarveType.DECODED_BITMAP: IdAllocator("BITMAP")}
    assigned = {id(c): ids[c.kind]() for c in candidates}
    records = []
    for c in candidates:
        spans, gaps, preview = _gapped(c)
        confidence = "CORROBORATED" if c.corroborations else "SINGLE_COPY"
        corroborations = [{**{k: v for k, v in note.items() if k != "with"}, "copy": assigned[note["with"]]}
                          for note in c.corroborations]
        uncertain, uncertain_spans, uncertain_preview = _uncertain_layer(c, assigned)
        raw_digest = hashlib.sha256(c.raw).hexdigest()
        pixel_digest = raw_digest if c.gap_rows else hashlib.sha256(c.rgb.tobytes()).hexdigest()
        records.append(RawArtifactRecord(
            artifact_id=assigned[id(c)],
            # As for workflows: an image with any hole is partial, whatever its extent.
            completeness=(Completeness.COMPLETE if c.complete and not c.suspect_rows and not c.gap_rows
                          else Completeness.PARTIAL),
            link_status=LinkClass.UNASSOCIATED,
            associated_workflow_id=None,
            ambiguous_candidate_workflows=[],
            is_derived=False,
            carve_type=c.kind,
            source_layer="process_virtual" if c.process_id is not None else "physical_UNATTRIBUTED",
            process_id=c.process_id,
            virtual_address=f"0x{c.virtual_address:016X}" if c.virtual_address is not None else None,
            physical_offsets=[f"0x{p:016X}" for p in c.physical_offsets],
            recovered_bytes=len(c.raw),
            container_sha256=None,
            pixel_sha256=pixel_digest,
            pixel_metadata={"mode": "RGB", "width": c.width, "height": c.height, "confidence": confidence,
                            "corroborations": corroborations, "suspect_rows": sorted(c.suspect_rows),
                            "source": c.source},
            validation={"pixels_readable": True, "independent_copy_agrees": bool(c.corroborations)},
            gaps=gaps,
            raw_file_paths=[],
            derived_preview_path=None,
            derived_preview_metadata={"rendering": "RGB pixels as found; suspect rows not altered"
                                      + ("; gap rows drawn magenta" if c.gap_rows else "")},
            spans=spans,
            preview_png=_png(preview),
            dimensions=(c.width, c.height),
            evidence={**_image_evidence(c, corroborations),
                      "sha256": {"raw": raw_digest, "pixels": pixel_digest},
                      **({"ownership_uncertain": uncertain} if uncertain else {})},
            page_origins=c.page_origins,
            uncertain_spans=uncertain_spans,
            uncertain_preview_png=uncertain_preview,
        ))
    return records


def _image_evidence(c: ImageCandidate, corroborations: list[dict[str, Any]]) -> dict[str, Any]:
    scaled = any(note["scale"] != 1 for note in c.corroborations) and c.kind == CarveType.DECODED_BITMAP
    if c.kind == CarveType.TENSOR_IMAGE:
        liveness = "LIVE"
    elif c.kind == CarveType.PIXEL_BUFFER:
        liveness = "LIVE" if c.source.get("state") == "LIVE" else "UNKNOWN"
    else:
        liveness = "UNKNOWN"
    # Rows on a page whose PFN entry disagrees with its mapping are bytes of another moment of acquisition.
    holes = [{"row_range": [r, r + 1], "reason": "STALE_MAPPING" if r in c.stale_rows else "SUSPECT_ROW"}
             for r in sorted(c.suspect_rows)]
    holes += [{"row_range": [first, end], "reason": reason} for first, end, reason in c.gap_rows]
    cache = {"execution_cache": c.execution_cache} if c.execution_cache is not None else {}
    return {
        **cache,
        "origin": "OBSERVED",
        "fidelity": "EXACT" if c.kind != CarveType.DECODED_BITMAP else ("SCALED" if scaled else "UNKNOWN"),
        "attribution": "UNATTRIBUTED",
        "liveness": liveness,
        "lineage": "L5",
        "holes": holes,
        "corroborations": corroborations,
        "observations": [{"holder_id": c.kind, "process_id": c.process_id,
                          "virtual_address": f"0x{c.virtual_address:X}" if c.virtual_address is not None else None,
                          "physical_offsets": [f"0x{p:X}" for p in c.physical_offsets], "source": c.source}],
    }


_OUTCOMES = {"execution_success": "success", "execution_error": "error", "execution_interrupted": "interrupted"}


def _utc(milliseconds: Any) -> str | None:
    """A ComfyUI timestamp (int(time.time() * 1000) on the host) as UTC ISO 8601, or None if it is not one."""
    if not isinstance(milliseconds, int) or isinstance(milliseconds, bool) or not 0 < milliseconds < 10**14:
        return None
    moment = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def job_timeline(item: QueueItem) -> dict[str, Any]:
    """When ComfyUI queued, started and finished a job, read from its own records: the request's
    ``extra_data.create_time`` and the timestamped messages of the history entry's status. Times are the host
    clock in UTC to the millisecond. A message naming another prompt is ignored; an absent one leaves its time
    out."""
    out: dict[str, Any] = {}
    if (queued := _utc((item.extra_data or {}).get("create_time"))) is not None:
        out["queued_utc"] = queued
    status = (item.history or {}).get("status")
    messages = status.get("messages") if isinstance(status, dict) else None
    stamps: dict[str, int] = {}
    for message in messages if isinstance(messages, list) else []:
        if not (isinstance(message, (list, tuple)) and len(message) == 2 and isinstance(message[1], dict)):
            continue
        event, data = message
        if data.get("prompt_id") not in (None, item.prompt_id) or _utc(data.get("timestamp")) is None:
            continue
        if event == "execution_start":
            stamps.setdefault("started", data["timestamp"])
        elif event in _OUTCOMES:
            stamps["finished"] = data["timestamp"]
            out["outcome"] = _OUTCOMES[event]
        elif event == "execution_cached" and isinstance(data.get("nodes"), list):
            out["cached_nodes"] = [str(node) for node in data["nodes"]]
    for name, stamp in stamps.items():
        out[f"{name}_utc"] = _utc(stamp)
    if {"started", "finished"} <= stamps.keys():
        out["run_seconds"] = round((stamps["finished"] - stamps["started"]) / 1000, 3)
    return out


# Machine field of a workflow's ``execution_parameters``. Its one positive value says the job's own history shows a
# terminal interrupted/error outcome with ``completed`` false and explicitly empty outputs, so no image can be
# credited as a recorded output of the job. A manifest without the field (or with another value) keeps the
# earlier behavior: absence is never evidence either way.
OUTPUT_CONTEXT_FIELD = "output_context"
TERMINAL_WITHOUT_RECORDED_OUTPUTS = "TERMINAL_WITHOUT_RECORDED_OUTPUTS"


def _images_of(node: Any) -> list[Any]:
    """A history output node's ``images`` list; a decode hole or other non-list value gives none."""
    images = node.get("images") if isinstance(node, dict) else None
    return images if isinstance(images, list) else []

_MARKS, _CONTRADICTS, _UNKNOWN = "MARKS", "CONTRADICTS", "UNKNOWN"


def _history_reading(item: QueueItem) -> str:
    """What one copy's own history says. MARKS: a readable terminal interrupted/error history with ``completed``
    False and explicitly empty outputs. CONTRADICTS: any recorded output, a success, or another status. UNKNOWN: a
    bare copy or a history that proves nothing (no status, no terminal message, outputs not stated)."""
    history = item.history
    if not isinstance(history, dict) or not history:
        return _UNKNOWN
    outputs = history.get("outputs")
    if isinstance(outputs, dict) and outputs:
        return _CONTRADICTS
    status = history.get("status")
    if not isinstance(status, dict):
        return _UNKNOWN
    outcome = job_timeline(item).get("outcome")
    status_str = status.get("status_str")
    if status.get("completed") is True or outcome == "success" or (
            isinstance(status_str, str) and status_str != "error"):
        return _CONTRADICTS
    if (status_str == "error" and status.get("completed") is False and outcome in ("interrupted", "error")
            and isinstance(outputs, dict)):
        return _MARKS
    return _UNKNOWN


def terminal_without_outputs_jobs(items: Iterable[QueueItem]) -> set[str]:
    """Prompt ids whose informative copies all show a terminal interrupted/error job with empty outputs, and at
    least one does. A bare or unreadable copy neither makes a job marked nor contradicts a reliable history; a
    single copy with outputs, a success or any other status prevents the mark."""
    readings: dict[str, set[str]] = {}
    for item in items:
        readings.setdefault(item.prompt_id, set()).add(_history_reading(item))
    return {prompt_id for prompt_id, seen in readings.items() if _MARKS in seen and _CONTRADICTS not in seen}


def terminal_without_outputs_ids(parameters: Iterable[Mapping[str, Any]]) -> set[str]:
    """The same verdict read back from ``execution_parameters`` of recorded workflows (a manifest). A prompt id
    is marked when some record carries the positive field and no record of that prompt holds an output file name,
    a completed or non-error history or a different value of the field; records without the field say nothing."""
    seen: dict[str, set[str]] = {}
    for params in parameters:
        prompt_id = params.get("prompt_id") if isinstance(params, Mapping) else None
        if not isinstance(prompt_id, str):
            continue
        value = params.get(OUTPUT_CONTEXT_FIELD)
        timeline = params.get("job_timeline")
        contradicts = (bool(params.get("output_filenames")) or params.get("history_completed") is True
                       or params.get("history_status") not in (None, "error")
                       or (value is not None and value != TERMINAL_WITHOUT_RECORDED_OUTPUTS)
                       or (isinstance(timeline, Mapping) and timeline.get("outcome") == "success"))
        seen.setdefault(prompt_id, set()).add(_CONTRADICTS if contradicts
                                              else _MARKS if value == TERMINAL_WITHOUT_RECORDED_OUTPUTS
                                              else _UNKNOWN)
    return {prompt_id for prompt_id, kinds in seen.items() if _MARKS in kinds and _CONTRADICTS not in kinds}


def workflow_records(items: Iterable[QueueItem], next_id: IdAllocator) -> list[WorkflowRecord]:
    """One record per distinct API prompt and per distinct UI workflow, with every copy's address listed. Every
    record of a job whose copies agree it is terminal without recorded outputs carries
    ``execution_parameters[OUTPUT_CONTEXT_FIELD]``."""
    out: dict[tuple[str, str, str], WorkflowRecord] = {}
    items = list(items)
    marked = terminal_without_outputs_jobs(items)
    for item in items:
        runtime: dict[str, Any] = {"prompt_id": item.prompt_id}
        history: dict[str, Any] = item.history or {}
        status: dict[str, Any] = history["status"] if isinstance(history.get("status"), dict) else {}
        outputs: dict[str, Any] = history["outputs"] if isinstance(history.get("outputs"), dict) else {}
        timeline = job_timeline(item)
        files = sorted({str(img["filename"]) for node in outputs.values() if isinstance(node, dict)
                        for img in _images_of(node) if isinstance(img, dict) and img.get("filename")})
        location = {"process_id": item.process_id, "virtual_address": f"0x{item.address:016X}",
                    "physical_offset": None, "queue_number": item.number, "object_refcount": item.refcnt}
        observation = {"holder_id": "python.queue_item", "object_kind": f"tuple[{item.length}]",
                       "process_id": item.process_id, "virtual_address": f"0x{item.address:X}",
                       "refcount": item.refcnt, "owner": item.owner, "liveness": item.liveness}
        for fmt, graph, holes in (("API_GRAPH", item.prompt, item.prompt_holes),
                                  ("UI_WORKFLOW", item.workflow, item.workflow_holes)):
            if graph is None:
                continue
            digest = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
            # Content deduplication never merges jobs: the prompt id is part of the key.
            key = (fmt, digest, item.prompt_id)
            if key in out:
                out[key].execution_parameters["source_locations"].append(location)
                if timeline:
                    merged = out[key].execution_parameters.setdefault("job_timeline", {})
                    conflicts = merged.setdefault("conflicts", {})
                    for field, value in timeline.items():
                        if field == "run_seconds":
                            continue  # recompute from the merged endpoints
                        if field in conflicts:
                            if value not in conflicts[field]:
                                conflicts[field].append(value)
                        elif field in merged and merged[field] != value:
                            conflicts[field] = [merged.pop(field), value]
                        else:
                            merged[field] = value
                    merged.pop("run_seconds", None)
                    if "started_utc" in merged and "finished_utc" in merged:
                        start = datetime.fromisoformat(merged["started_utc"].replace("Z", "+00:00"))
                        finish = datetime.fromisoformat(merged["finished_utc"].replace("Z", "+00:00"))
                        merged["run_seconds"] = round((finish - start).total_seconds(), 3)
                    if not conflicts:
                        merged.pop("conflicts")
                # A bare queue copy can come first: the history fields of a later copy still belong to the job.
                merged_params = out[key].execution_parameters
                if status and "history_status" not in merged_params:
                    merged_params["history_status"] = status.get("status_str")
                    merged_params["history_completed"] = status.get("completed")
                if files:
                    merged_params["output_filenames"] = sorted({*merged_params.get("output_filenames", []), *files})
                out[key].evidence["observations"].append(observation)
                if item.liveness == "LIVE":
                    out[key].evidence["liveness"] = "LIVE"
                continue
            params: dict[str, Any] = execution_parameters(graph, runtime) if fmt == "API_GRAPH" else dict(runtime)
            if status:
                params["history_status"] = status.get("status_str")
                params["history_completed"] = status.get("completed")
            if files:
                params["output_filenames"] = files
            if timeline:
                params["job_timeline"] = timeline
            if item.prompt_id in marked:
                params[OUTPUT_CONTEXT_FIELD] = TERMINAL_WITHOUT_RECORDED_OUTPUTS
            params["source_locations"] = [location]
            complete = (graph_is_complete(fmt, graph, params) if fmt == "API_GRAPH" else True) and not holes
            out[key] = WorkflowRecord(
                workflow_id=next_id(),
                completeness=Completeness.COMPLETE if complete else Completeness.PARTIAL,
                prompt_format=fmt,
                encoding_detected="CPYTHON_OBJECTS",
                source_process_id=item.process_id,
                virtual_address=f"0x{item.address:016X}",
                physical_offsets=[],
                canonical_json_sha256=digest,
                execution_parameters=params,
                dependencies=(graph_dependencies(graph) if fmt == "API_GRAPH"
                              else {"node_classes": [], "model_assets": []}),
                raw_file_path="",
                raw_bytes=json.dumps(graph, ensure_ascii=False).encode("utf-8"),
                graph=graph,
                evidence={"origin": "OBSERVED", "submission_class": "SUBMITTED", "lineage": "L1",
                          "liveness": item.liveness, "holes": list(holes), "prompt_id": item.prompt_id,
                          "observations": [observation]},
            )
    return list(out.values())
