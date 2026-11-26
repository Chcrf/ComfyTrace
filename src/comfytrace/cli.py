"""ComfyTrace extraction, model identification and advanced case commands."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import logging
import shlex
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from comfytrace import __version__
from comfytrace.carve import IdAllocator, PngCarver, SpanSet, WorkflowCarver
from comfytrace.casefiles import (
    EVIDENCE_NAME,
    MANIFEST_NAME,
    METADATA_NAME,
    EvidentiaryExporter,
    ManifestWriter,
    sha256_file,
)
from comfytrace.correlate import (
    OTHER_IMAGES,
    CorrelationResult,
    correlate,
    embedded_prompt,
    job_sizes,
    place_fragments,
    png_workflows,
)
from comfytrace.ingest import (
    MODE_PHYSICAL,
    PAGE_SHIFT,
    IngestionResult,
    RecoverySupport,
    SymbolManager,
    ingest_memory,
)
from comfytrace.media import is_embedded_text
from comfytrace.metadata import metadata_document
from comfytrace.models import (
    SCHEMA_VERSION,
    Completeness,
    CorrelationLink,
    ForensicManifest,
    LinkClass,
    RawArtifactRecord,
    Tier,
    WorkflowRecord,
)
from comfytrace.outputs import OutputFile, OutputFileCarver
from comfytrace.report import HtmlReporter

log = logging.getLogger("comfytrace")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_PARTIAL = 2
EXIT_DEGRADED = 3
EXIT_EMPTY = 4

Ingestor = Callable[..., IngestionResult]


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1: argparse's default 2 would read as "partial findings"."""

    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        self.exit(EXIT_FAILURE, f"{self.prog}: error: {message}\n")


def _add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runner-url", help="Explicit ComfyUI endpoint. Defaults to local discovery with comfy-cli.")
    parser.add_argument("--comfy-cli", type=Path, metavar="PATH", help="Path to the comfy-cli executable")
    parser.add_argument("--install-comfyui", action="store_true", help="Install ComfyUI with comfy-cli if missing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Builds the CLI parser. Shows ``extract`` and ``model-id`` in top-level help; advanced subcommands stay
    callable for scripting and tests."""
    parser = _Parser(prog="comfytrace", description="ComfyUI memory forensics.")
    parser.add_argument("--version", action="version", version=f"comfytrace {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{extract,model-id,env-id}")

    def ssim_threshold(value: str) -> float:
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError("SSIM threshold must be within [0, 1]")
        return threshold

    def phash_threshold(value: str) -> int:
        threshold = int(value)
        if not 0 <= threshold <= 64:
            raise argparse.ArgumentTypeError("pHash threshold must be an integer within [0, 64]")
        return threshold

    ext = sub.add_parser("extract", help="carve, correlate and report ComfyUI evidence from a memory image")
    ext.add_argument("--memory", required=True, type=Path, help="physical memory capture or Linux ELF process core")
    ext.add_argument("--symbols-dir", type=Path, help="offline ISF cache, required for physical captures")
    ext.add_argument("--image-os", choices=("windows", "linux"), default="windows",
                     help="operating system in the capture (default: windows)")
    ext.add_argument("--image-format", choices=("physical", "elf-core"), default="physical",
                     help="capture type (default: physical)")
    ext.add_argument("--allow-symbol-download", action="store_true", help="permit remote symbol downloads")
    ext.add_argument("--outdir", required=True, type=Path, help="new case directory")
    ext.add_argument("--model", type=Path, metavar="PATH",
                     help="checkpoint file or models directory; auto-replays recovered workflows when given")
    _add_runner_arguments(ext)
    ext.add_argument("--inputs-dir", type=Path, help="conditioning images for img2img/inpaint workflows")
    ext.add_argument("--ssim-threshold", type=ssim_threshold, default=0.95)
    ext.add_argument("--phash-threshold", type=phash_threshold, default=4)
    ext.add_argument("--allow-node", action="append", default=[], metavar="CLASS",
                     help="authorize a custom node class (repeatable)")
    ext.add_argument("--allow-sampler-node", action="append", default=[], metavar="CLASS",
                     help="authorize a custom sampler class; only vetted samplers prove an output was generated")
    ext.add_argument("--allow-pickle-models", action="store_true", help="permit .ckpt/.pt/.pth checkpoints")
    ext.add_argument("--timeout", type=int, default=300, help="runner execution timeout in seconds")
    ext.add_argument("--case-id", help="case identifier recorded in the manifest")
    ext.add_argument("--investigator", help="examiner name recorded in the manifest")
    ext.add_argument("--comfyui-pid", action="append", type=int, default=[], metavar="PID",
                     help="examiner-identified Python PID for an unreadable command line (repeatable)")
    ext.add_argument("--omit-unrelated-exports", action="store_true",
                     help="record images with no detected ComfyUI connection without exporting their bytes")
    ext.add_argument("--content-stitch-png", action="store_true",
                     help="search for continuations of torn PNGs carrying ComfyUI workflow metadata (DERIVED)")
    ext.add_argument("--content-stitch-seconds", type=int, default=600, metavar="SECONDS",
                     help="positive time budget for content search after the page index is built (default: 600)")

    def physical_offset(value: str) -> int:
        try:
            offset = int(value, 16 if value.lower().startswith("0x") else 10)
        except ValueError as err:
            raise argparse.ArgumentTypeError("provide a nonnegative physical offset in decimal or hex") from err
        if offset < 0:
            raise argparse.ArgumentTypeError("provide a nonnegative physical offset in decimal or hex")
        return offset

    ext.add_argument("--content-stitch-offset", action="append", type=physical_offset, default=[], metavar="OFFSET",
                     help="select a physical PNG signature offset for content stitching (repeatable)")

    mid = sub.add_parser("model-id", help="check whether local model files' weights are in ComfyUI process memory")
    mid.add_argument("--memory", required=True, type=Path, help="memory capture file")
    mid.add_argument("--symbols-dir", type=Path, help="offline ISF cache for physical captures")
    mid.add_argument("--image-os", choices=("windows", "linux"), default="windows",
                     help="operating system in the capture (default: windows)")
    mid.add_argument("--image-format", choices=("physical", "elf-core"), default="physical",
                     help="capture type (default: physical)")
    mid.add_argument("--allow-symbol-download", action="store_true", help="permit remote symbol downloads")
    mid.add_argument("--comfyui-pid", action="append", type=int, default=[], metavar="PID",
                     help="examiner-confirmed Python PID (repeatable)")
    mid.add_argument("--models-dir", "--model", dest="models_dir", required=True, type=Path,
                     help="checkpoint file or models directory to check against process memory")
    mid.add_argument("--case-dir", type=Path,
                     help="optional case directory written by extract (records model-identity.json)")

    env = sub.add_parser("env-id", help="detect captured ComfyUI software stack and compare against a replay runner")
    env.add_argument("--memory", required=True, type=Path, help="memory capture file")
    env.add_argument("--symbols-dir", type=Path, help="offline ISF cache for physical captures")
    env.add_argument("--image-os", choices=("windows", "linux"), default="windows",
                     help="operating system in the capture (default: windows)")
    env.add_argument("--image-format", choices=("physical", "elf-core"), default="physical",
                     help="capture type (default: physical)")
    env.add_argument("--allow-symbol-download", action="store_true", help="permit remote symbol downloads")
    env.add_argument("--comfyui-pid", action="append", type=int, default=[], metavar="PID",
                     help="examiner-confirmed Python PID (repeatable)")
    env.add_argument("--case-dir", type=Path,
                     help="optional case directory written by extract (records environment-identity.json)")
    env.add_argument("--runner-url", help="optional running ComfyUI endpoint to compare against")
    env.add_argument("--timeout", type=int, default=30, help="runner query timeout in seconds (default: 30)")

    # Advanced / internal subcommands (hidden from top-level --help, still callable directly).
    reg = sub.add_parser("regenerate")
    replay = sub.add_parser("replay-case")
    replay.add_argument("--memory", required=True, type=Path, help="original raw physical memory image")
    replay.add_argument("--image-os", choices=("windows", "linux"), default="windows",
                        help="operating system in the capture when extracting a new case")
    replay.add_argument("--image-format", choices=("physical", "elf-core"), default="physical",
                        help="capture type when extracting a new case")
    replay.add_argument("--comfyui-pid", action="append", type=int, default=[], metavar="PID",
                        help="examiner-confirmed PID when extracting a new case")
    replay.add_argument("--symbols-dir", type=Path,
                        help="offline ISF cache, for extracting a new case and for --model-id/--latents")
    replay.add_argument("--model-id", action="store_true",
                        help="first check which local model files' weights are in ComfyUI process memory")
    replay.add_argument("--latents", action="store_true",
                        help="also recover retained latents from process memory and decode them for linking")

    reg.add_argument("--case-dir", required=True, type=Path)
    which = reg.add_mutually_exclusive_group(required=True)
    which.add_argument("--workflow", type=Path, help="recovered API prompt workflow JSON")
    which.add_argument("--all", action="store_true",
                       help="every complete API prompt recovered from CPython objects, one run per job")
    replay.add_argument("--case-dir", type=Path, required=True)
    for replay_parser in (reg, replay):
        _add_runner_arguments(replay_parser)
        replay_parser.add_argument("--models-dir", type=Path,
                                   help="local copies of the runner's models, for hash verification")
        replay_parser.add_argument("--inputs-dir", type=Path, help="conditioning images for img2img/inpaint workflows")
        replay_parser.add_argument("--ssim-threshold", type=ssim_threshold, default=0.95)
        replay_parser.add_argument("--phash-threshold", type=phash_threshold, default=4)
        replay_parser.add_argument("--allow-node", action="append", default=[], metavar="CLASS",
                         help="authorize a custom node class (repeatable)")
        replay_parser.add_argument("--allow-sampler-node", action="append", default=[], metavar="CLASS",
                         help="authorize a custom sampler class; only vetted samplers prove an output was generated")
        replay_parser.add_argument("--allow-pickle-models", action="store_true",
                                   help="permit .ckpt/.pt/.pth checkpoints")
        replay_parser.add_argument("--timeout", type=int, default=300, help="runner execution timeout in seconds")
    reg.add_argument("--compare-to", type=Path, help="memory-carved image to compare against")

    lat = sub.add_parser("latents")
    lat.add_argument("--case-dir", required=True, type=Path, help="case directory written by extract")
    lat.add_argument("--memory", required=True, type=Path, help="the memory image the case was extracted from")
    lat.add_argument("--symbols-dir", type=Path, help="offline ISF cache for physical captures")

    dec = sub.add_parser("decode-latents")
    dec.add_argument("--case-dir", required=True, type=Path)
    _add_runner_arguments(dec)
    dec.add_argument("--models-dir", required=True, type=Path,
                     help="local model files, at the relative names the recovered prompts use")
    dec.add_argument("--timeout", type=int, default=600, help="runner execution timeout in seconds")

    lnk = sub.add_parser("link")
    lnk.add_argument("--case-dir", required=True, type=Path)
    lnk.add_argument("--regenerated-dir", type=Path,
                     help="regenerated PNGs, each with a JSON record naming its prompt_id and output_sha256")

    search = sub.add_parser("search-replay")
    search.add_argument("--case-dir", type=Path, required=True)
    search.add_argument("--memory", type=Path, required=True)
    search.add_argument("--workflow-id", action="append", default=[], help="workflow to search for (repeatable)")

    rep = sub.add_parser("report")
    rep.add_argument("--case-dir", required=True, type=Path)
    return parser.parse_args(argv)


def extract_exit_code(degraded: bool, tier_summary: dict[str, int], artifact_count: int) -> int:
    """Precedence: operational failure (1, raised elsewhere) > degraded (3) > partial (2) > empty (4) > ok (0)."""
    if degraded:
        return EXIT_DEGRADED
    partial_keys = (Tier.E3, Tier.E6, "untiered_orphan_images", "untiered_ambiguous_images")
    if any(tier_summary.get(k, 0) for k in partial_keys):
        return EXIT_PARTIAL
    if artifact_count == 0:
        return EXIT_EMPTY
    return EXIT_OK


def _carve(ingestion: IngestionResult, known_workflows: Iterable[WorkflowRecord] = (), *,
           content_stitch_png: bool = False, content_stitch_offsets: frozenset[int] = frozenset(),
           content_stitch_seconds: int = 600) -> tuple[
        list[RawArtifactRecord], list[WorkflowRecord], list[tuple[str, OutputFile]]]:
    """PNG, other media, prompt-graph and output-file carving over every address space. ``known_workflows``:
    workflows recovered before carving (stage 1), whose job sizes also decide which text-less WEBP is kept."""
    from comfytrace.media import MEDIA_TYPES, MediaCarver, media_ids

    image_ids = IdAllocator("CARVE-PNG")
    shared_media_ids = media_ids()
    media_found: list[tuple[Any, Any, MediaCarver]] = []
    media_counts: dict[str, int] = {}
    workflow_carver = WorkflowCarver(id_factory=IdAllocator("WF"))
    output_carver = OutputFileCarver()
    pfn = ingestion.recovery.pfn
    images: list[RawArtifactRecord] = []
    workflows: list[WorkflowRecord] = []
    for space in ingestion.spaces:
        png_carver = PngCarver(image_ids, physical_mode=space.physical_mode)
        key = "physical" if space.physical_mode else space.process_id
        ranges = _page_ranges(space.skip_pages)
        owned = SpanSet(ranges) if ranges else None
        if ranges:
            png_carver.claimed[key] = SpanSet(ranges)
        # Shares the PNG carver's claimed spans: the JSON carver leaves the text inside a container alone.
        media_carver = MediaCarver(png_carver.claimed, space.physical_mode, shared_media_ids)
        bytes_scanned = 0
        progress_at = time.monotonic() + 30
        for block in space.blocks:
            if not block.data:
                continue
            bytes_scanned += len(block.data)
            if time.monotonic() >= progress_at:
                log.info("carving %s: %.1f MiB of scan windows, %d images and %d workflows so far",
                         "physical memory" if space.physical_mode else f"PID {space.process_id}",
                         bytes_scanned / (1024 * 1024), len(images), len(workflows))
                progress_at = time.monotonic() + 30
            where: dict[str, Any] = (
                {"base_physical_offset": block.address} if space.physical_mode
                else {"virtual_address": block.address, "process_id": space.process_id}
            )
            new_images = png_carver.carve_stream(block.data, source=space.source, exclude=owned, **where)
            media_found += [(candidate, space, media_carver) for candidate in media_carver.carve_stream(
                block.data, source=space.source, exclude=owned, **where)]
            new_workflows = workflow_carver.carve_stream(
                block.data, source=space.source, exclude=png_carver.claimed.get(key, SpanSet()), **where
            )
            output_carver.carve_stream(block.data, **where)
            if space.translate is not None:
                for record in new_images:
                    _attach_physical(record.physical_offsets, record.virtual_address, space.translate)
                for wf in new_workflows:
                    _attach_physical(wf.physical_offsets, wf.virtual_address, space.translate)
            if pfn is not None:
                for record in new_images:
                    _png_origin(record, space, pfn)
            images += new_images
            workflows += new_workflows
        if space.physical_mode:
            for record in images:
                if record.process_id is None:
                    record.source_layer = "physical_UNATTRIBUTED"
        for kind, count in media_carver.found.items():
            media_counts[kind] = media_counts.get(kind, 0) + count

    def media_record(candidate: Any, space: Any, carver: MediaCarver) -> None:
        record = carver.record(candidate)
        if record is None:
            return
        if space.translate is not None:
            _attach_physical(record.physical_offsets, record.virtual_address, space.translate)
        if record.process_id is None:
            record.source_layer = "physical_UNATTRIBUTED"
        if pfn is not None:
            _png_origin(record, space, pfn)
        images.append(record)

    # A container that carries ComfyUI text, or was read from a ComfyUI process, is a record whatever else is
    # known; the PNG-text step below then reads its prompt.
    for candidate, space, carver in media_found:
        if candidate.keep_always:
            media_record(candidate, space, carver)
    if pfn is not None and ingestion.recovery.physical is not None:
        _page_order_pngs(images, ingestion.recovery.physical, pfn, PngCarver(image_ids, physical_mode=False))
    image_path = getattr(ingestion.recovery, "image_path", None)
    if image_path is not None and ingestion.recovery.physical is not None:
        _stitch_texts(images, image_path, ingestion.recovery.physical)
    if (content_stitch_png or content_stitch_offsets) and ingestion.recovery.physical is not None:
        _content_stitch_pngs(images, ingestion.recovery.physical, offsets=content_stitch_offsets,
                             require_workflow=not content_stitch_offsets, seconds=content_stitch_seconds)
    images = _merge_duplicate_images(images)
    pieces = list(workflow_carver.fragments.values())
    started = time.perf_counter()
    workflows = place_fragments(pieces, workflows, images, workflow_carver.next_id)
    workflows += png_workflows(images, IdAllocator("WFPNG"))
    # Any other container is one of the machine's many browser and application media files. Only a WEBP with
    # exactly a recovered job's size is kept; the rest are counted, not recorded.
    sizes = job_sizes(images, [*known_workflows, *workflows])
    kept = 0
    for candidate, space, carver in media_found:
        if not candidate.keep_always and candidate.parsed.complete and candidate.dimensions in sizes:
            media_record(candidate, space, carver)
            kept += 1
    recorded = sum(1 for image in images if image.carve_type in MEDIA_TYPES)
    notes = getattr(ingestion.recovery, "notes", None)
    if notes is not None:
        notes["media"] = {"containers_walked": media_counts, "recorded": recorded,
                          "recorded_for_job_size_only": kept, "not_recorded": len(media_found) - recorded}
    log.info("other media: %s walked, %d recorded", media_counts, recorded)
    log.info("JSON fragments: %d cut pieces, %d fragment records; %.1f s carving, %.1f s placing", len(pieces),
             sum(1 for wf in workflows if "fragment" in wf.evidence), workflow_carver.fragment_seconds,
             time.perf_counter() - started)
    outputs = output_carver.records()
    if pfn is not None:
        _workflow_origins(workflows, ingestion.spaces, pfn)
        _output_origins(outputs, ingestion.spaces, pfn)
    return images, workflows, outputs


def _copy_pages(start: int, ranges: Iterable[tuple[int, int]],
                translate: Callable[[int], int | None] | None) -> list[int]:
    """Physical page numbers under the byte ranges (offset from ``start``, length) of one copy. ``start`` is a
    physical offset, or a virtual address resolved page by page through ``translate``."""
    pages = []
    for relative, length in ranges:
        if length <= 0:
            continue
        lo = start + relative
        for page in range(lo >> PAGE_SHIFT, (lo + length - 1 >> PAGE_SHIFT) + 1):
            if translate is None:
                pages.append(page)
            elif (physical := translate(page << PAGE_SHIFT)) is not None:
                pages.append(physical >> PAGE_SHIFT)
    return pages


def _png_origin(record: RawArtifactRecord, space: Any, pfn: Any) -> None:
    ranges = [(relative, len(data)) for relative, data in record.spans]
    if space.physical_mode:
        pages = _copy_pages(int(record.physical_offsets[0], 16), ranges, None)
    elif space.translate is not None and record.virtual_address is not None:
        pages = _copy_pages(int(record.virtual_address, 16), ranges, space.translate)
    else:
        return
    record.page_origins = [pfn.copy_origin(pages, record.recovered_bytes)]


def _workflow_origins(workflows: list[WorkflowRecord], spaces: list[Any], pfn: Any) -> None:
    """Each carved copy of a workflow's bytes is contiguous in the space it was read from."""
    translate = {s.process_id: s.translate for s in spaces if not s.physical_mode and s.translate is not None}
    for wf in workflows:
        if is_embedded_text(wf.encoding_detected):
            continue  # its pages are its PNGs' (or other containers') pages, already classified
        for location in wf.execution_parameters.get("source_locations", []):
            pid, va, physical = location.get("process_id"), location.get("virtual_address"), \
                location.get("physical_offset")
            lo, hi = location.get("record_range", (0, len(wf.raw_bytes)))
            size = hi - lo
            if pid is None and physical:
                pages = _copy_pages(int(physical, 16), [(0, size)], None)
            elif va and pid in translate:
                pages = _copy_pages(int(va, 16), [(0, size)], translate[pid])
            else:
                continue
            wf.page_origins.append(pfn.copy_origin(pages, size))


def _page_order_pngs(images: list[RawArtifactRecord], physical: Any, pfn: Any, carver: PngCarver) -> None:
    """Supplementary layer: a PNG torn at a page edge of physical memory is read on in the order of the mapping
    that owns its first page (a file's prototype PTEs, or a process's page tables) and replaces the torn record
    when that verifies more bytes. Most such copies are the file cache's: their memory source says so."""
    from comfytrace.carve import parse_png
    from comfytrace.media import MEDIA_TYPES
    from comfytrace.models import GapReason
    from comfytrace.pageorder import page_order_stream

    rebuilt = 0
    for i, record in enumerate(images):
        if record.process_id is not None or record.completeness == Completeness.COMPLETE \
                or not record.physical_offsets or record.dimensions is None or record.carve_type in MEDIA_TYPES:
            continue
        start = int(record.physical_offsets[0], 16)
        width, height = record.dimensions
        stream = page_order_stream(physical, pfn, start, width * height * 8 + (1 << 20))
        if stream is None or len(stream[2]) < 2:
            continue
        data, method, frames = stream
        parsed = parse_png(data, False, GapReason.PAGE_NOT_PRESENT, start)
        if parsed is None or sum(len(piece) for _, piece in parsed.spans) <= record.recovered_bytes:
            continue
        new = carver.record_from(parsed, start, False, None, record.artifact_id)
        used = frames[: (parsed.end + (start & 0xFFF) + 0xFFF) >> 12]
        new.physical_offsets, new.source_layer = list(record.physical_offsets), record.source_layer
        new.page_origins = [pfn.copy_origin(used, new.recovered_bytes)]
        new.evidence = {"assembly": {"method": method, "pages": len(used),
                                     "page_offsets": [f"0x{f << 12:X}" for f in used]}}
        images[i] = new
        rebuilt += 1
    log.info("page-order PNGs: %d torn PNGs read on in mapping order", rebuilt)


def _stitch_texts(images: list[RawArtifactRecord], image_path: Path, physical: Any) -> None:
    """Supplementary DERIVED layer: a ComfyUI text chunk cut at a page edge is completed by a page of the image that
    starts with the rest of the chunk, its CRC-32 and the next chunk header. Template-identical heads accept the
    tails of other jobs, so with several the workflow's seed must equal the seed of the PNG's own prompt; still
    ambiguous, nothing is stitched. The result is never raw evidence."""
    from comfytrace.carve import METADATA_KEYS, _decode_text_chunk
    from comfytrace.models import GapReason
    from comfytrace.pageorder import stitch_text_chunks, torn_text_chunk

    page = 1 << PAGE_SHIFT

    def to_page_edge(record: RawArtifactRecord) -> bytes:
        """The record's verified bytes and, for a PNG torn between physical pages, the bytes after them on the
        same page: the head of the chunk that failed. A record keeps only whole verified chunks, so without them
        the torn chunk's type, length and first bytes would be unknown. They feed the DERIVED stitched text only,
        where the chunk's CRC-32 then covers them."""
        piece = record.spans[0][1]
        if not record.physical_offsets or not record.gaps or record.gaps[0].reason != GapReason.NONCONTIGUOUS_PHYSICAL:
            return piece
        end = int(record.physical_offsets[0], 16) + len(piece)
        return piece + physical[end:end + (-end % page)]

    torn = [t for r in images if r.process_id is None and r.completeness != Completeness.COMPLETE and r.spans
            and r.spans[0][0] == 0
            and (t := torn_text_chunk(r.artifact_id, to_page_edge(r))) is not None]
    if not torn:
        return
    found = stitch_text_chunks(image_path, torn)
    by_id = {r.artifact_id: r for r in images}
    stitched = 0
    for t in torn:
        record = by_id[t.record_id]
        decoded = {}
        for tail in found[t.record_id]:
            text = _decode_text_chunk(t.ctype, t.head + tail)
            if text is not None and text[0] in METADATA_KEYS and text[0] not in record.embedded_text:
                decoded[text[1]] = text[0]
        if len(decoded) > 1:
            seed = embedded_prompt(record).parameters.get("seed")
            decoded = {v: k for v, k in decoded.items() if seed is not None and _workflow_seed(v) == seed}
        if len(decoded) == 1:
            [(value, key)] = decoded.items()
            record.stitched_text[key] = value
            record.evidence = {**record.evidence, "stitched_text": {
                "key": key, "continuations_found": len(found[t.record_id]), "crc32_verified": True,
                "class": "DERIVED: completed by content search, not by any mapping"}}
            stitched += 1
    log.info("text stitching: %d torn ComfyUI text chunks, %d completed by a CRC-verified page", len(torn), stitched)


def _content_stitch_pngs(
    images: list[RawArtifactRecord], physical: Any, *, offsets: frozenset[int] = frozenset(),
    require_workflow: bool = False, seconds: int = 600,
) -> None:
    """Supplementary DERIVED layer: recover scattered continuations of physical PNGs by content.

    The raw record's spans, offsets, and classified page origins stay unchanged. Searched frames are named only
    in the assembly provenance because content identity does not establish their former owner.
    """
    from comfytrace.carve import PngCarver, parse_png
    from comfytrace.models import CarveType, GapReason
    from comfytrace.pagestitch import PAGE, PageIndex, stitch_png, stitchable_png_prefix

    candidates: list[tuple[int, RawArtifactRecord, int]] = []
    for position, record in enumerate(images):
        if (record.process_id is not None or record.carve_type != CarveType.PARTIAL_PNG
                or not record.physical_offsets or len(record.spans) != 1 or record.spans[0][0] != 0
                or not record.gaps or record.gaps[0].reason not in (
                    GapReason.PAGE_NOT_PRESENT, GapReason.NONCONTIGUOUS_PHYSICAL)):
            continue
        start = int(record.physical_offsets[0], 16)
        if offsets and start not in offsets:
            continue
        if require_workflow and {"prompt", "workflow"}.isdisjoint(
                record.embedded_text.keys() | record.stitched_text.keys()):
            continue
        candidates.append((position, record, start))

    candidates.sort(key=lambda candidate: -len(candidate[1].spans[0][1]))
    log.info("content-stitch PNGs: %d candidates before prefix validation", len(candidates))
    index = None
    deadline = None
    tried = rebuilt = complete = 0
    carver = PngCarver(physical_mode=False)
    for sequence, (position, original, start) in enumerate(candidates, 1):
        if deadline is not None and time.perf_counter() >= deadline:
            log.info("content-stitch PNGs: time budget reached after %d of %d candidates", sequence - 1,
                     len(candidates))
            break
        raw_prefix = original.spans[0][1]
        first_page = start & ~(PAGE - 1)
        phase = start - first_page
        page_count = (phase + len(raw_prefix) + PAGE - 1) // PAGE
        source_order = original.evidence.get("assembly", {}).get("page_offsets")
        prefix_pages = ([int(offset, 16) for offset in source_order[:page_count]] if source_order else
                        list(range(first_page, first_page + page_count * PAGE, PAGE)))
        if (len(prefix_pages) != page_count or not prefix_pages or prefix_pages[0] != first_page
                or len(set(prefix_pages)) != len(prefix_pages)
                or any(offset < 0 or offset % PAGE or offset + PAGE > len(physical) for offset in prefix_pages)):
            continue
        prefix = b"".join(bytes(physical[offset:offset + PAGE]) for offset in prefix_pages)[phase:]
        if not prefix.startswith(raw_prefix) or not stitchable_png_prefix(prefix):
            continue
        if index is None:
            log.info("content-stitch PNGs: building one page index")
            index = PageIndex.build(physical)
            deadline = time.perf_counter() + seconds
        assert deadline is not None
        tried += 1
        log.info("content-stitch PNGs: %d/%d testing %s with %d recovered prefix bytes",
                 sequence, len(candidates), original.artifact_id, original.recovered_bytes)
        item_deadline = deadline if offsets else min(deadline, time.perf_counter() + 120)
        result = stitch_png(prefix, phase, physical, index, excluded_page_offsets=frozenset(prefix_pages),
                            deadline=item_deadline)
        if time.perf_counter() >= item_deadline:
            log.info("content-stitch PNGs: time budget reached for %s", original.artifact_id)
        if result is None:
            continue
        start = int(original.physical_offsets[0], 16)
        parsed = parse_png(result.data, False, GapReason.PAGE_NOT_PRESENT, start)
        if parsed is None:
            continue
        replacement = carver.record_from(parsed, start, False, None, original.artifact_id)
        replacement.is_derived = True
        replacement.physical_offsets = list(original.physical_offsets)
        replacement.source_layer = original.source_layer
        replacement.page_origins = list(original.page_origins)
        replacement.spans = list(original.spans)
        replacement.stitched_text = {**original.stitched_text, **{
            key: value for key, value in replacement.embedded_text.items() if key not in original.embedded_text
        }}
        replacement.embedded_text = dict(original.embedded_text)
        replacement.derived_reconstruction = result.data
        replacement.recovered_bytes = len(result.data)
        replacement.evidence = {**original.evidence, "assembly": {
            "method": "CONTENT_STITCHED",
            "derived": True,
            "pages": len(result.page_offsets),
            "page_offsets": [f"0x{offset:X}" for offset in result.page_offsets],
            "prefix_page_offsets": [f"0x{offset:X}" for offset in prefix_pages],
            "prefix_bytes": len(prefix),
            "last_page_bytes": len(result.data) - len(prefix) - (len(result.page_offsets) - 1) * PAGE,
            "source_gaps": [gap.to_dict() for gap in original.gaps],
            "sealed_bytes": len(result.data),
            "candidate_counts": result.candidate_counts,
        }}
        images[position] = replacement
        rebuilt += 1
        complete += replacement.completeness == Completeness.COMPLETE
    log.info("content-stitch PNGs: %d searched, %d reconstructed (%d complete)", tried, rebuilt, complete)


def _workflow_seed(text: str) -> Any:
    from comfytrace.fragments import ui_parameters, ui_workflow_nodes

    try:
        return ui_parameters(ui_workflow_nodes(json.loads(text))).get("seed")
    except (ValueError, RecursionError):
        return None


def _output_origins(outputs: list[tuple[str, OutputFile]], spaces: list[Any], pfn: Any) -> None:
    translate = {s.process_id: s.translate for s in spaces if not s.physical_mode and s.translate is not None}
    for _, entry in outputs:
        for copy in entry.copies:
            if "physical_offset" in copy:
                pages = _copy_pages(int(copy["physical_offset"], 16), [(0, copy["bytes"])], None)
            elif copy.get("process_id") in translate:
                pages = _copy_pages(int(copy["virtual_address"], 16), [(0, copy["bytes"])],
                                    translate[copy["process_id"]])
            else:
                continue
            copy["origin"] = pfn.copy_origin(pages, copy["bytes"])
            entry.page_origins.append(copy["origin"])


def _python_stage(recovery: RecoverySupport) -> tuple[list[WorkflowRecord], list[Any], dict[str, Any], list[Any],
                                                      list[dict[str, Any]]]:
    """Stage 1: when a ComfyUI Python process is readable as CPython objects, recover its queue/history items,
    raw pixel buffers, IMAGE tensors and execution-cache entries. Otherwise the stage records why it was
    skipped. The last value lists the cache entries, each with the jobs whose prompt holds its signature."""
    from comfytrace import execcache, memrecover, pyheap, torchmem

    summary: dict[str, Any] = {"processes": []}
    if recovery.notes.get("python_objects"):
        summary["status"] = recovery.notes["python_objects"]
    workflows: list[WorkflowRecord] = []
    candidates: list[Any] = []
    items = []
    items_by_pid: dict[int, list[Any]] = {}
    cache_entries: list[Any] = []
    for mem in recovery.python_processes:
        mem.prefetch()
        result = pyheap.extract(mem)
        row = result.summary()
        row["stale_mapped_pages"] = dict(collections.Counter(mem.stale.values()))
        row["compression_store_pages"] = mem.store_status
        if result.status == "AVAILABLE":
            items += result.items
            items_by_pid.setdefault(mem.pid, []).extend(result.items)
            candidates += [memrecover.from_pixel_buffer(b, mem.physical_of) for b in result.pixel_buffers]
            tensors = torchmem.image_tensors(mem)
            row["image_tensors"] = len(tensors)
            entries, row["execution_cache"] = execcache.extract(mem)
            cache_entries += entries
            ties = execcache.image_ties(entries)
            candidates += [memrecover.from_tensor(t, mem.physical_of, ties.get((mem.pid, t.tensorimpl)))
                           for t in tensors]
        else:
            log.info("PID %d: CPython objects unavailable (%s)", mem.pid, result.reason)
        summary["processes"].append(row)
    for candidate in candidates:
        if candidate.execution_cache is not None:
            candidate.execution_cache["jobs"] = execcache.job_matches(
                candidate.execution_cache["graph"], items_by_pid.get(candidate.process_id, []),
                candidate.execution_cache.get("output_socket"))
    cache = [{**entry.to_dict(), "jobs": execcache.job_matches(entry.graph, items_by_pid.get(entry.process_id, []))}
             for entry in cache_entries]
    if recovery.python_processes:
        readable = any(p["status"] == "AVAILABLE" for p in summary["processes"])
        summary["status"] = "AVAILABLE" if readable else "UNAVAILABLE: no process readable as CPython objects"
    elif "status" not in summary:
        summary["status"] = "UNAVAILABLE: no ComfyUI Python process"
    workflows = memrecover.workflow_records(items, IdAllocator("WFPY"))
    log.info("stage 1 (python objects): %s; %d workflows, %d image candidates, %d execution-cache entries",
             summary["status"], len(workflows), len(candidates), len(cache))
    return workflows, candidates, summary, items, cache


def _name_cached_images(cache: list[dict[str, Any]], images: list[RawArtifactRecord]) -> None:
    """Each cache entry lists the image records made from the tensors it holds."""
    entries = {(entry["process_id"], entry["cache_dict"], entry["key_address"]): entry for entry in cache}
    for image in images:
        tie = image.evidence.get("execution_cache")
        if not tie:
            continue
        for holder in [tie, *tie.get("also_held_by", [])]:
            entry = entries.get((image.process_id, holder["cache_dict"], holder["key_address"]))
            if entry is not None:
                entry.setdefault("image_artifacts", []).append(
                    {"artifact_id": image.artifact_id, "batch_index": tie.get("batch_index")})


def _bitmap_stage(recovery: RecoverySupport) -> tuple[list[Any], dict[str, Any]]:
    """Stage 2 (raw scan, continued): decoded 32-bit bitmaps anywhere in physical memory."""
    from comfytrace import bitmaps, memrecover

    if recovery.physical is None:
        return [], {"status": "SKIPPED: no raw physical image"}
    found = bitmaps.scan(recovery.physical, recovery.pfn, recovery.image_path, recovery.scattered)
    log.info("decoded-bitmap scan: %d candidate bitmaps", len(found))
    return [memrecover.from_bitmap(b) for b in found], {
        "status": "DONE", "ordering": recovery.notes.get("bitmap_ordering"), "bitmaps": len(found)}


def _page_ranges(pages: frozenset[int]) -> list[tuple[int, int]]:
    """Contiguous byte ranges already carved through a process space."""
    ordered = iter(sorted(pages))
    start = next(ordered, None)
    if start is None:
        return []
    ranges = []
    previous = start
    for page in ordered:
        if page != previous + 1:
            ranges.append((start << PAGE_SHIFT, (previous + 1) << PAGE_SHIFT))
            start = page
        previous = page
    ranges.append((start << PAGE_SHIFT, (previous + 1) << PAGE_SHIFT))
    return ranges


def _attach_physical(offsets: list[str], virtual_address: str | None, translate: Callable[[int], int | None]) -> None:
    if virtual_address is None:
        return
    physical = translate(int(virtual_address, 16))
    if physical is not None:
        offsets.append(f"0x{physical:016X}")


PREFIX_KEY = 64


def _merge_duplicate_images(images: list[RawArtifactRecord]) -> list[RawArtifactRecord]:
    """Byte-identical complete PNGs found at several locations are one artifact, and so is a torn copy whose
    genuine bytes are a prefix of one; locations accumulate."""
    kept: dict[str, RawArtifactRecord] = {}
    out = []
    for record in images:
        if record.derived_reconstruction is not None:
            out.append(record)
            continue
        digest = record.container_sha256
        if digest is not None and digest in kept:
            first = kept[digest]
            if first.process_id is not None or record.process_id is not None:
                locations = first.evidence.setdefault("process_copies", [])
                for image in ((first, record) if not locations else (record,)):
                    if image.process_id is None:
                        continue
                    location = {"process_id": image.process_id, "virtual_address": image.virtual_address,
                                "physical_offsets": list(image.physical_offsets)}
                    if location not in locations:
                        locations.append(location)
            for offset in record.physical_offsets:
                if offset not in first.physical_offsets:
                    first.physical_offsets.append(offset)
            first.page_origins += record.page_origins
            continue
        if digest is not None:
            kept[digest] = record
        out.append(record)
    by_prefix: dict[bytes, list[RawArtifactRecord]] = {}
    for r in out:
        if r.completeness == Completeness.COMPLETE and r.derived_reconstruction is None and r.spans:
            by_prefix.setdefault(r.spans[0][1][:PREFIX_KEY], []).append(r)
    merged = []
    for record in out:
        complete = None
        # Only an unattributed torn copy that fits exactly one intact image is merged: a process copy keeps
        # its attribution, and batch images sharing their metadata prefix stay apart.
        if (record.derived_reconstruction is None and record.process_id is None
                and record.completeness != Completeness.COMPLETE
                and len(record.spans) == 1 and record.spans[0][0] == 0 and len(record.spans[0][1]) >= PREFIX_KEY):
            piece = record.spans[0][1]
            fits = [c for c in by_prefix.get(piece[:PREFIX_KEY], []) if c.spans[0][1].startswith(piece)]
            if len(fits) == 1:
                complete = fits[0]
        if complete is None:
            merged.append(record)
            continue
        # A torn copy whose genuine bytes are a prefix of an intact image is another location of that image.
        for offset in record.physical_offsets:
            if offset not in complete.physical_offsets:
                complete.physical_offsets.append(offset)
        complete.page_origins += record.page_origins
    return merged


SCORECARD_NAME = "scorecard.json"


def _write_evidence(case_dir: Path, images: list[RawArtifactRecord], workflows: list[WorkflowRecord],
                    queue_items: list[Any], memory_sha: str, outputs: list[tuple[str, OutputFile]],
                    memory_only: CorrelationResult, links: list[CorrelationLink],
                    execution_cache: list[dict[str, Any]] | None = None) -> None:
    from comfytrace import evidence

    tool = {"version": __version__,
            "source_tree_sha256": evidence.source_tree_sha256(Path(__file__).resolve().parent)}
    for name, document in (
        (EVIDENCE_NAME, evidence.evidence_document(images, workflows, tool, outputs, memory_only.tier_summary,
                                                   memory_only.findings, execution_cache,
                                                   memory_only_links=memory_only.links)),
        (SCORECARD_NAME, evidence.scorecard(queue_items, workflows, memory_sha, links, images)),
        (METADATA_NAME, metadata_document(workflows)),
    ):
        target = case_dir / name
        payload = json.dumps(document, indent=1, default=str) + "\n"
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)


def _export(case_dir: Path, images: Iterable[RawArtifactRecord], workflows: list[WorkflowRecord]) -> None:
    exporter = EvidentiaryExporter(case_dir)
    for record in images:
        exporter.export_raw_artifact(record, record.spans)
        if record.derived_reconstruction is not None:
            suffix = "png" if record.completeness == Completeness.COMPLETE else "bin"
            record.derived_preview_path = exporter.export_derived_preview(
                f"content_stitched_{record.artifact_id}.{suffix}", record.derived_reconstruction
            )
            digest = hashlib.sha256(record.derived_reconstruction).hexdigest()
            record.derived_preview_metadata = {
                **(record.derived_preview_metadata or {}),
                "class": "DERIVED: CRC-sealed content reconstruction",
                "rendering": "CRC-sealed reconstruction from content-matched physical pages",
                "sha256": digest,
                "preview_sha256": digest,
                "raw_recovered_bytes": sum(len(data) for _offset, data in record.spans),
                "raw_span_sha256": [hashlib.sha256(data).hexdigest() for _offset, data in record.spans],
            }
        elif record.completeness != Completeness.COMPLETE and record.derived_preview_metadata is not None:
            record.derived_preview_metadata["raw_span_sha256"] = [
                hashlib.sha256(data).hexdigest() for _offset, data in record.spans
            ]
        if record.preview_png is not None and record.derived_reconstruction is None:
            record.derived_preview_path = exporter.export_derived_preview(
                f"preview_{record.artifact_id}.png", record.preview_png
            )
            if record.derived_preview_metadata is not None:
                record.derived_preview_metadata["preview_sha256"] = hashlib.sha256(record.preview_png).hexdigest()
        exporter.export_uncertain(record)
    for workflow in workflows:
        exporter.export_workflow(workflow)


def _resolve_models_dir(model_path: Path) -> Path:
    """Accepts either a models directory or a single checkpoint file path (returning its parent directory)."""
    path = Path(model_path)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    raise ValueError(f"--model path does not exist: {path}")


def _auto_replay(args: argparse.Namespace, case_dir: Path, degraded: bool, ingestor: Ingestor | None) -> int:
    """When ``extract --model`` is given and at least one complete API workflow was recovered, automatically run
    replay-case (model-id/latents when a live process was recovered, regenerate, search-replay, link, workbench)."""
    from comfytrace.regenerate import replayable_prompts
    from comfytrace.replaycase import run_replay_case as replay

    models_dir = _resolve_models_dir(args.model)
    manifest = json.loads((case_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not replayable_prompts(manifest):
        log.info("--model given, but no complete API workflow was recovered; skipping auto-replay")
        return EXIT_OK
    can_live = not degraded and (
        getattr(args, "image_format", "physical") == "elf-core" or args.symbols_dir is not None
    )
    replay_args = argparse.Namespace(
        memory=args.memory,
        case_dir=case_dir,
        symbols_dir=args.symbols_dir,
        image_os=getattr(args, "image_os", "windows"),
        image_format=getattr(args, "image_format", "physical"),
        comfyui_pid=getattr(args, "comfyui_pid", []),
        models_dir=models_dir,
        model_id=can_live,
        latents=can_live,
        runner_url=getattr(args, "runner_url", None),
        comfy_cli=getattr(args, "comfy_cli", None),
        install_comfyui=getattr(args, "install_comfyui", False),
        inputs_dir=getattr(args, "inputs_dir", None),
        ssim_threshold=getattr(args, "ssim_threshold", 0.95),
        phash_threshold=getattr(args, "phash_threshold", 4),
        allow_node=getattr(args, "allow_node", []),
        allow_sampler_node=getattr(args, "allow_sampler_node", []),
        allow_pickle_models=getattr(args, "allow_pickle_models", False),
        timeout=getattr(args, "timeout", 300),
    )
    return replay(replay_args, ingestor=ingestor)


def run_extract(args: argparse.Namespace, ingestor: Ingestor | None = None) -> int:
    """Runs the extraction pipeline and returns the standardized exit code (0, 1, 2, 3, 4)."""
    case_dir: Path = args.outdir
    started = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    try:
        if (case_dir / MANIFEST_NAME).exists():
            raise FileExistsError(f"{case_dir / MANIFEST_NAME} exists; manifests are immutable, choose a new --outdir")
        memory: Path = args.memory
        if not memory.is_file():
            raise FileNotFoundError(f"memory image not found: {memory}")
        image_format = getattr(args, "image_format", "physical")
        if image_format == "physical" and args.symbols_dir is None:
            raise ValueError("pass --symbols-dir for a physical capture")
        if image_format == "elf-core":
            if args.allow_symbol_download:
                raise ValueError("ELF process cores do not use kernel symbols")
            if getattr(args, "content_stitch_png", False) or getattr(args, "content_stitch_offset", ()):
                raise ValueError("Content stitching requires a flat physical capture")
        case_dir.mkdir(parents=True, exist_ok=True)
        log.info("hashing memory image %s", memory)
        memory_sha = sha256_file(memory)
        manager = SymbolManager(args.symbols_dir, allow_download=args.allow_symbol_download,
                                image_os=getattr(args, "image_os", "windows"), image_format=image_format)
        examiner_pids = getattr(args, "comfyui_pid", [])
        if any(pid <= 0 for pid in examiner_pids):
            raise ValueError("--comfyui-pid must be a positive process ID")
        stitch_seconds = getattr(args, "content_stitch_seconds", 600)
        if stitch_seconds <= 0:
            raise ValueError("--content-stitch-seconds must be positive")
        ingestion = (ingest_memory(memory, manager, examiner_pids=examiner_pids) if ingestor is None
                     else ingestor(memory, manager))
        try:
            py_workflows, candidates, stage1, queue_items, execution_cache = _python_stage(ingestion.recovery)
            images, workflows, outputs = _carve(
                ingestion, py_workflows, content_stitch_png=getattr(args, "content_stitch_png", False),
                content_stitch_offsets=frozenset(getattr(args, "content_stitch_offset", ())),
                content_stitch_seconds=stitch_seconds,
            )
            bitmap_candidates, stage2 = _bitmap_stage(ingestion.recovery)
            pfn = ingestion.recovery.pfn
            if pfn is not None:
                for candidate in candidates + bitmap_candidates:
                    candidate.page_origins = [pfn.copy_origin(candidate.pages, len(candidate.raw))]
        finally:
            ingestion.close()
        from comfytrace import memrecover

        candidates += bitmap_candidates
        memrecover.corroborate(candidates)
        images += memrecover.image_records(candidates)
        _name_cached_images(execution_cache, images)
        workflows = py_workflows + workflows
        log.info("correlation: %d image records and %d workflow records", len(images), len(workflows))
        result = correlate(images, workflows)
        from comfytrace import evidence

        memory_only = correlate(*evidence.memory_view(images, workflows))
        log.info("tier summary, memory-backed records only: %s", memory_only.tier_summary)
        omitted: set[str] = set()
        if getattr(args, "omit_unrelated_exports", False):
            omitted = {aid for finding in result.findings if finding.rationale.startswith(OTHER_IMAGES)
                       for aid in finding.artifact_ids}
        _export(case_dir, (image for image in images if image.artifact_id not in omitted), workflows)
        degraded = ingestion.mode == MODE_PHYSICAL
        # Images with no ComfyUI connection are not ComfyUI artifacts: a dump holding only those found nothing.
        comfy_images = len(images) - result.tier_summary["untiered_other_images"]
        exit_code = extract_exit_code(degraded, result.tier_summary, comfy_images + len(workflows))
        manifest = ForensicManifest(
            schema_version=SCHEMA_VERSION,
            case_metadata={
                "case_id": args.case_id or f"CASE-{started:%Y%m%dT%H%M%SZ}",
                "investigator": args.investigator,
                "execution_timestamp_utc": started.isoformat().replace("+00:00", "Z"),
                "comfytrace_version": __version__,
                "command_line": getattr(args, "command_line", None) or "comfytrace " + shlex.join(sys.argv[1:]),
                "memory_image": {"path": str(memory), "size_bytes": memory.stat().st_size, "sha256": memory_sha},
                "symbol_resolution": ingestion.symbol_resolution,
                "exit_code": exit_code,
                **({"unrelated_image_exports_omitted": len(omitted)}
                   if getattr(args, "omit_unrelated_exports", False) else {}),
                "carving_mode": ingestion.mode,
                "degradation_reason": ingestion.degradation_reason,
                **({"examiner_comfyui_pids": sorted(set(examiner_pids))} if examiner_pids else {}),
                "recovery_stages": {"python_objects": stage1,
                                    "raw_scan": {"carving": "DONE", "decoded_bitmaps": stage2,
                                                 "other_media": ingestion.recovery.notes.get("media")},
                                    "evidence_files": [EVIDENCE_NAME, SCORECARD_NAME, METADATA_NAME]},
            },
            process_context=ingestion.process_context,
            findings={
                "tier_summary": result.tier_summary,
                "findings": result.findings,
                "artifacts": images,
                "workflows": workflows,
                "correlation_links": result.links,
                "supporting_artifacts": [],
            },
        )
        _write_evidence(case_dir, images, workflows, queue_items, memory_sha, outputs, memory_only, result.links,
                        execution_cache)
        manifest_path = ManifestWriter(case_dir).write_manifest(manifest)
        HtmlReporter(case_dir).render_report(manifest_path)
        if getattr(args, "model", None) is not None:
            replay_code = _auto_replay(args, case_dir, degraded, ingestor)
            if replay_code != EXIT_OK:
                log.warning("auto-replay exited with status %d", replay_code)
    except Exception as exc:  # noqa: BLE001 - every operational failure maps to exit code 1
        log.error("extraction failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE
    _log_summary(result.tier_summary, exit_code, case_dir, images, workflows)
    return exit_code


def _log_summary(summary: dict[str, int], code: int, case_dir: Path, images: list[RawArtifactRecord],
                 workflows: list[WorkflowRecord]) -> None:
    complete = sum(1 for i in images if i.completeness == Completeness.COMPLETE)
    linked = sum(1 for i in images if i.link_status in (LinkClass.STRICT, LinkClass.SUPPORTED))
    log.info("images: %d (%d complete, %d linked); workflows: %d", len(images), complete, linked, len(workflows))
    log.info("tier summary: %s", ", ".join(f"{k}={v}" for k, v in summary.items()))
    log.info("case written to %s (exit %d)", case_dir, code)


def run_report(args: argparse.Namespace) -> int:
    """Re-renders workbench.html from manifest.json and regenerations/. Returns 0 or 1."""
    try:
        manifest = args.case_dir / MANIFEST_NAME
        path = HtmlReporter(args.case_dir).render_report(manifest, args.case_dir / "regenerations")
    except Exception as exc:  # noqa: BLE001
        log.error("report failed: %s", exc)
        return EXIT_FAILURE
    log.info("workbench written to %s", path)
    return EXIT_OK


def run_replay_case(args: argparse.Namespace) -> int:
    from comfytrace.replaycase import run_replay_case as replay

    try:
        if not (args.case_dir / MANIFEST_NAME).is_file():
            extract_command = ["extract", "--memory", str(args.memory), "--outdir", str(args.case_dir),
                               "--image-os", getattr(args, "image_os", "windows"), "--image-format",
                               getattr(args, "image_format", "physical")]
            if args.symbols_dir is not None:
                extract_command.extend(("--symbols-dir", str(args.symbols_dir)))
            for pid in getattr(args, "comfyui_pid", ()):
                extract_command.extend(("--comfyui-pid", str(pid)))
            extract_args = parse_args(extract_command)
            code = run_extract(extract_args)
            if code == EXIT_FAILURE:
                return code
        return replay(args)
    except Exception as exc:  # noqa: BLE001
        log.error("replay-case failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE


def run_search_replay(args: argparse.Namespace) -> int:
    from comfytrace.replaysearch import search

    try:
        path = search(args.case_dir, args.memory, set(args.workflow_id))
    except ModuleNotFoundError as exc:
        if exc.name != "ahocorasick_rs":
            raise
        log.error("search-replay requires the [search] extra")
        return EXIT_FAILURE
    except (OSError, ValueError) as exc:
        log.error("search-replay failed: %s", exc)
        return EXIT_FAILURE
    log.info("replay search results written to %s", path)
    return EXIT_OK


def run_link(args: argparse.Namespace) -> int:
    """Writes image-links.json. Returns 0, or 1 on failure."""
    from comfytrace.link import write_links

    try:
        path = write_links(args.case_dir, args.regenerated_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("link failed: %s", exc)
        return EXIT_FAILURE
    log.info("image links written to %s", path)
    return EXIT_OK


def _ingest_for_inspection(args: argparse.Namespace, ingestor: Ingestor) -> tuple[Any, str | None]:
    """Ingest ``args.memory`` for standalone ``model-id`` / ``env-id`` or verify against ``args.case_dir``."""
    from comfytrace.ingest import ingest_case

    if getattr(args, "case_dir", None) is not None:
        manifest = json.loads((args.case_dir / "manifest.json").read_text(encoding="utf-8"))
        expected = manifest["case_metadata"]["memory_image"]["sha256"]
        memory_sha = sha256_file(args.memory)
        if memory_sha != expected:
            raise ValueError(f"{args.memory} is not the image this case was extracted from")
        return ingest_case(args.memory, args.symbols_dir, manifest["case_metadata"], ingestor), memory_sha
    memory: Path = args.memory
    if not memory.is_file():
        raise FileNotFoundError(f"memory image not found: {memory}")
    image_format = getattr(args, "image_format", "physical")
    if image_format == "physical" and args.symbols_dir is None:
        raise ValueError("pass --symbols-dir for a physical capture")
    allow_download = getattr(args, "allow_symbol_download", False)
    if image_format == "elf-core" and allow_download:
        raise ValueError("ELF process cores do not use kernel symbols")
    examiner_pids = tuple(getattr(args, "comfyui_pid", ()))
    if any(pid <= 0 for pid in examiner_pids):
        raise ValueError("--comfyui-pid must be a positive process ID")
    manager = SymbolManager(args.symbols_dir, allow_download=allow_download,
                            image_os=getattr(args, "image_os", "windows"), image_format=image_format)
    return ingestor(memory, manager, examiner_pids=examiner_pids), None


def _no_process_memory(ingestion: Any) -> bool:
    """True, after logging why, when ingestion recovered no Python process memory to inspect."""
    if ingestion.recovery.python_processes:
        return False
    log.error("no ComfyUI Python process memory was recovered: %s",
              getattr(ingestion, "degradation_reason", None) or "no reason recorded")
    return True


def run_model_id(args: argparse.Namespace, ingestor: Ingestor = ingest_memory) -> int:
    """Checks local model weights against ComfyUI process memory; writes model-identity.json when --case-dir is set.
    Returns 0, 1 on failure, or 3 (and writes nothing) when no process memory was recovered."""
    from comfytrace.modelid import identity_document, local_model_names, model_names, write_identity

    try:
        case_dir = getattr(args, "case_dir", None)
        if case_dir is not None:
            models_dir = _resolve_models_dir(args.models_dir)
            names = None
            candidates = len(model_names(case_dir))
        else:
            models_dir, names = local_model_names(args.models_dir)
            candidates = len(names)
        log.info("checking %d candidate model file(s) against process memory", candidates)
        ingestion, memory_sha = _ingest_for_inspection(args, ingestor)
        try:
            degraded = _no_process_memory(ingestion)
            document = identity_document(case_dir, ingestion.recovery.python_processes, models_dir, memory_sha, names)
        finally:
            ingestion.close()
        path = None if degraded or case_dir is None else write_identity(case_dir, document)
    except Exception as exc:  # noqa: BLE001
        log.error("model-id failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE
    for model in document["models"]:
        line = f"{model['name']}: {model['verdict']}"
        if path is None:
            print(line)
        else:
            log.info("%s", line)
    if path is not None:
        log.info("model identity written to %s", path)
    return EXIT_DEGRADED if degraded else EXIT_OK


def run_env_id(args: argparse.Namespace, ingestor: Ingestor = ingest_memory) -> int:
    """Detects captured ComfyUI environment, optionally compares against a runner, and writes
    environment-identity.json when --case-dir is set."""
    from comfytrace.envid import (
        RUNNER_UNREPORTED,
        environment_document,
        format_comparison_table,
        write_environment_identity,
    )
    from comfytrace.regenerate import RunnerClient

    try:
        case_dir = getattr(args, "case_dir", None)
        runner_url = getattr(args, "runner_url", None)
        # Validate the case and recover memory first: a bad manifest must fail before any network call.
        ingestion, memory_sha = _ingest_for_inspection(args, ingestor)
        try:
            if _no_process_memory(ingestion):
                print("Verdict: NO_PROCESS_MEMORY")
                return EXIT_DEGRADED
            runner_env = None
            if runner_url:
                runner_env = RunnerClient(runner_url, timeout=getattr(args, "timeout", 30)).environment()
            document = environment_document(ingestion.recovery.python_processes, memory_sha, runner_env, runner_url,
                                            getattr(ingestion, "process_context", None))
        finally:
            ingestion.close()
        if document["verdict"] == RUNNER_UNREPORTED:
            print(format_comparison_table(document))
            log.error("runner at %s reported none of the compared fields; nothing written", runner_url)
            return EXIT_FAILURE
        path = write_environment_identity(case_dir, document) if case_dir is not None else None
    except Exception as exc:  # noqa: BLE001
        log.error("env-id failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE
    print(format_comparison_table(document))
    if path is not None:
        log.info("environment identity written to %s", path)
    return EXIT_OK


def run_latents(args: argparse.Namespace, ingestor: Ingestor = ingest_memory) -> int:
    """Writes latents.json and the latents' raw bytes. Returns 0, or 1 on failure."""
    from comfytrace import latents
    from comfytrace.ingest import ingest_case
    from comfytrace.regenerate import recovered_prompts

    try:
        manifest = json.loads((args.case_dir / "manifest.json").read_text(encoding="utf-8"))
        memory_sha = sha256_file(args.memory)
        if memory_sha != manifest["case_metadata"]["memory_image"]["sha256"]:
            raise ValueError(f"{args.memory} is not the image this case was extracted from")
        if (args.case_dir / latents.LATENTS_NAME).exists():
            raise FileExistsError(f"{args.case_dir / latents.LATENTS_NAME} exists; latents are written once")
        graphs = [json.loads((args.case_dir / w["raw_file_path"]).read_text(encoding="utf-8"))
                  for w in recovered_prompts(manifest)]
        sides = latents.latent_sides(graphs)
        ingestion = ingest_case(args.memory, args.symbols_dir, manifest["case_metadata"], ingestor)
        try:
            blocks = [b for mem in ingestion.recovery.python_processes for b in latents.find(mem, sides)]
        finally:
            ingestion.close()
        path = latents.write(args.case_dir, latents.document(args.case_dir, blocks, memory_sha, sides))
    except Exception as exc:  # noqa: BLE001
        log.error("latents failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE
    log.info("%d latents written to %s", len(blocks), path)
    return EXIT_OK


def run_decode_latents(args: argparse.Namespace) -> int:
    """Writes latent-decodes/. Returns 0, or 1 on failure or refusal."""
    from comfytrace.latentdecode import run_decode

    try:
        return run_decode(args)
    except Exception as exc:  # noqa: BLE001
        log.error("decode-latents failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE


def run_regenerate(args: argparse.Namespace) -> int:
    """Runs the optional regeneration handoff. Returns 0, 1 or 2."""
    from comfytrace.regenerate import run_all, run_regeneration

    if args.all and args.compare_to is not None:
        log.error("--compare-to names one target and cannot be combined with --all")
        return EXIT_FAILURE
    try:
        return run_all(args) if args.all else run_regeneration(args)
    except Exception as exc:  # noqa: BLE001
        log.error("regeneration failed: %s", exc)
        log.debug("%s", traceback.format_exc())
        return EXIT_FAILURE


def main(argv: list[str] | None = None) -> int:
    """Entry point. Maps any unhandled exception to exit code 1 and exits with the command's code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)
    for noisy in ("volatility3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.command_line = "comfytrace " + shlex.join(sys.argv[1:] if argv is None else argv)
    handlers = {"extract": run_extract, "regenerate": run_regenerate, "report": run_report, "link": run_link,
                "model-id": run_model_id, "env-id": run_env_id, "latents": run_latents,
                "decode-latents": run_decode_latents, "search-replay": run_search_replay,
                "replay-case": run_replay_case}
    try:
        code = handlers[args.command](args)
    except Exception:  # noqa: BLE001
        log.error("fatal: %s", traceback.format_exc())
        code = EXIT_FAILURE
    sys.exit(code)
