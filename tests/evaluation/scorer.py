"""Post-extraction evaluation scorer for ComfyTrace case directories.

Compares an already-extracted local case directory (read-only) against withheld job/output ground truth.
Ground truth is supplied ONLY here after case generation and never influences extraction.

Key properties:
- Stable evidence identity: matches artifacts by container_sha256, raw_sha256, pixel_sha256, or normalized
  physical offsets, and workflows by prompt_id or canonical_json_sha256, falling back to transient IDs only
  if explicitly given in ground truth (and disclosing every fallback in identity_stability metrics).
- Explicit denominators: every ratio reports {numerator, denominator, rate, definition}, with rate=None when
  denominator is zero.
- Wrong vs ambiguous links: AMBIGUOUS abstentions are never conflated with WRONG attributions; wrong STRICT
  links, cross-job links, distractor links, and uncatalogued links are all counted separately.
- Tier correctness: evaluates both ground-truth tier expectations (expected_tier, allowed_tiers,
  min_expected_tier, max_allowed_tier; over-tiered vs under-tiered) and ComfyTrace finding tier invariants
  (including the 16-pixel minimum output rule, close-match E3 vs exact E2, and wrong-link tier invalidation).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from comfytrace.correlate import MIN_OUTPUT_SIDE, OTHER_IMAGES
from comfytrace.media import is_embedded_text
from comfytrace.models import CarveType
from comfytrace.report import (
    EVIDENCE_NAME,
    LINKS_NAME,
    MANIFEST_NAME,
    apply_image_links,
    effective_findings,
    load_regenerations,
    sha256_file,
    verify_image_link_promotions,
)

SCORER_SCHEMA_VERSION = 1
SCORECARD_NAME = "scorecard.json"
EvalView = Literal["effective", "extraction", "memory_only"]

EXACT_MEMORY_KINDS = frozenset({
    CarveType.PIXEL_BUFFER.value,
    CarveType.TENSOR_IMAGE.value,
})

RASTER_DIGEST_KEYS = (
    "raster_sha256",
    "per_raster_sha256",
    "output_raster_sha256",
    "matched_raster_sha256",
    "regenerated_raster_sha256",
    "raster_digest",
)

OUTPUT_IDENTITY_NESTED_KEYS = ("output_identity", "matched_output", "output", "best", "selected_raster")

TIER_RANK: dict[str | None, int] = {
    None: -1,
    "UNTIERED": 0,
    "E6": 1,
    "E5": 2,
    "E4": 3,
    "E3": 4,
    "E2": 5,
    "E1": 6,
}

ENCODED_IMAGE_TYPES = frozenset({
    CarveType.ENCODED_PNG.value,
    CarveType.PARTIAL_PNG.value,
    CarveType.ENCODED_WEBP.value,
    CarveType.PARTIAL_WEBP.value,
})

MEMORY_BACKED_SOURCES = frozenset({"MEMORY", "MEMORY_PARTIAL"})


def _norm_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _norm_sha_set(single: Any = None, plural: Any = None) -> frozenset[str]:
    out: set[str] = set()
    if (s := _norm_sha(single)) is not None:
        out.add(s)
    if isinstance(plural, list):
        for item in plural:
            if (s := _norm_sha(item)) is not None:
                out.add(s)
    return frozenset(out)


def _norm_offset(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"0x{value:X}" if value >= 0 else None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.startswith("-"):
        return None
    try:
        parsed = int(cleaned, 16)
    except ValueError:
        return None
    return f"0x{parsed:X}" if parsed >= 0 else None


def _norm_node_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _parse_batch_index(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid batch_index {value!r} for {field_name}")
    return value


def _norm_offset_set(values: Any) -> frozenset[str]:
    if not isinstance(values, list):
        return frozenset()
    return frozenset(off for v in values if (off := _norm_offset(v)) is not None)


def _norm_str_set(single: Any = None, plural: Any = None) -> frozenset[str]:
    out: set[str] = set()
    if isinstance(single, str) and single.strip():
        out.add(single.strip())
    if isinstance(plural, list):
        for item in plural:
            if isinstance(item, str) and item.strip():
                out.add(item.strip())
    return frozenset(out)


def _ratio(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(numerator / denominator, 6) if denominator > 0 else None,
        "definition": definition,
    }


def _file_raw_sha256(case_dir: Path, relative_paths: Iterable[str]) -> str | None:
    paths = [case_dir / p for p in relative_paths]
    if not paths or not all(p.is_file() for p in paths):
        return None
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _dimensions(artifact: Mapping[str, Any]) -> tuple[int, int] | None:
    meta = artifact.get("pixel_metadata")
    if not isinstance(meta, dict):
        return None
    w, h = meta.get("width"), meta.get("height")
    if isinstance(w, int) and isinstance(h, int) and not isinstance(w, bool) and not isinstance(h, bool):
        return (w, h)
    return None


def _is_below_output_size(artifact: Mapping[str, Any]) -> bool:
    dims = _dimensions(artifact)
    return artifact.get("carve_type") in ENCODED_IMAGE_TYPES and dims is not None and min(dims) < MIN_OUTPUT_SIDE


@dataclass(frozen=True)
class ArtifactIdentity:
    artifact_id: str
    carve_type: str
    completeness: str
    container_sha256: str | None
    pixel_sha256: str | None
    raster_sha256: str | None
    raw_sha256: str | None
    has_gaps: bool
    physical_offsets: frozenset[str]
    stable_id: str
    identity_kind: str
    # A partial PNG recovered as one span from its signature: that span's length and SHA-256.
    prefix_length: int | None = None
    prefix_sha256: str | None = None


PREFIX_PAST_IDAT = 1024
"""A verified prefix must reach this far into the first IDAT chunk's data: header and text chunks are shared by
every output of one job (and by reruns of one graph), the encoded pixels are not."""


def _prefix_identity(artifact: Mapping[str, Any]) -> tuple[int | None, str | None]:
    if artifact.get("carve_type") != CarveType.PARTIAL_PNG.value or artifact.get("completeness") != "PARTIAL":
        return None, None
    spans = (artifact.get("derived_preview_metadata") or {}).get("raw_span_sha256") or []
    gaps = artifact.get("gaps") or []
    length = artifact.get("recovered_bytes")
    if (len(spans) != 1 or not gaps or not isinstance(length, int) or isinstance(length, bool)
            or gaps[0].get("relative_offset") != length):
        return None, None
    return length, _norm_sha(spans[0])


def _idat_data_offset(data: bytes) -> int | None:
    at = 8
    while at + 8 <= len(data):
        length = int.from_bytes(data[at:at + 4], "big")
        if data[at + 4:at + 8] == b"IDAT":
            return at + 8
        at += 12 + length
    return None


def _verifies_prefix(art: ArtifactIdentity, original: bytes | None) -> bool:
    if original is None or art.prefix_length is None or art.prefix_sha256 is None:
        return False
    idat = _idat_data_offset(original)
    return (idat is not None and idat + PREFIX_PAST_IDAT <= art.prefix_length < len(original)
            and hashlib.sha256(original[:art.prefix_length]).hexdigest() == art.prefix_sha256)


@dataclass(frozen=True)
class WorkflowIdentity:
    workflow_id: str
    prompt_format: str
    completeness: str
    encoding_detected: str
    prompt_ids: frozenset[str]
    canonical_json_sha256: str | None
    raw_sha256: str | None
    stable_id: str
    identity_kind: str


@dataclass(frozen=True)
class GtOutputSpec:
    output_id: str
    job_id: str
    container_sha256s: frozenset[str]
    pixel_sha256s: frozenset[str]
    raster_sha256s: frozenset[str]
    raw_sha256s: frozenset[str]
    physical_offsets: frozenset[str]
    carve_types: frozenset[str]
    artifact_ids: frozenset[str]
    output_node_id: str | None
    batch_index: int | None
    filename: str | None
    expected_ambiguous: bool
    expected_tier: str | None
    allowed_tiers: frozenset[str] | None
    min_expected_tier: str | None
    max_allowed_tier: str | None
    # The original file's bytes (``container_file``), which verify a partial carve by its exact prefix.
    container_bytes: bytes | None = None


@dataclass(frozen=True)
class GtJobSpec:
    job_id: str
    prompt_ids: frozenset[str]
    canonical_prompt_sha256s: frozenset[str]
    canonical_ui_sha256s: frozenset[str]
    workflow_ids: frozenset[str]
    expected_tier: str | None
    allowed_tiers: frozenset[str] | None
    min_expected_tier: str | None
    max_allowed_tier: str | None
    outputs: tuple[GtOutputSpec, ...]


@dataclass(frozen=True)
class GtDistractorSpec:
    distractor_id: str
    reason: str | None
    container_sha256s: frozenset[str]
    pixel_sha256s: frozenset[str]
    raster_sha256s: frozenset[str]
    raw_sha256s: frozenset[str]
    physical_offsets: frozenset[str]
    carve_types: frozenset[str]
    artifact_ids: frozenset[str]
    allow_link_without_tier: bool
    allowed_link_job_ids: frozenset[str] | None
    expected_tier: str | None
    allowed_tiers: frozenset[str] | None
    min_expected_tier: str | None
    max_allowed_tier: str | None


def _parse_tier(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in TIER_RANK:
        raise ValueError(f"invalid tier {value!r} for {field_name}")
    return value


def _parse_allowed_tiers(value: Any, field_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of tier names")
    parsed = {_parse_tier(t, field_name) for t in value}
    return frozenset(t for t in parsed if t is not None)


def _container_bytes(raw_out: Mapping[str, Any], base: Path | None, out_id: str,
                     container_shas: frozenset[str]) -> bytes | None:
    name = raw_out.get("container_file")
    if not isinstance(name, str) or base is None:
        return None
    data = (base / name).read_bytes()
    if hashlib.sha256(data).hexdigest() not in container_shas:
        raise ValueError(f"{out_id}.container_file does not match its container_sha256")
    return data


def _parse_groundtruth(gt: Mapping[str, Any], base: Path | None = None,
                       ) -> tuple[str | None, list[GtJobSpec], list[GtDistractorSpec]]:
    if not isinstance(gt, Mapping):
        raise ValueError("ground truth must be a JSON object")
    dump_sha = _norm_sha(gt.get("dump_sha256"))
    raw_jobs = gt.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError("ground truth 'jobs' must be a list")
    jobs: list[GtJobSpec] = []
    seen_jobs: set[str] = set()
    seen_outputs: set[str] = set()
    for raw_job in raw_jobs:
        if not isinstance(raw_job, Mapping):
            raise ValueError("each ground-truth job entry must be a JSON object")
        job_id = str(raw_job["job_id"])
        if job_id in seen_jobs:
            raise ValueError(f"duplicate ground-truth job_id: {job_id}")
        seen_jobs.add(job_id)
        raw_outputs = raw_job.get("outputs", [])
        if not isinstance(raw_outputs, list):
            raise ValueError(f"ground truth job {job_id!r} 'outputs' must be a list")
        outputs: list[GtOutputSpec] = []
        for raw_out in raw_outputs:
            if not isinstance(raw_out, Mapping):
                raise ValueError(f"each output entry in job {job_id!r} must be a JSON object")
            out_id = str(raw_out["output_id"])
            if out_id in seen_outputs:
                raise ValueError(f"duplicate ground-truth output_id: {out_id}")
            seen_outputs.add(out_id)
            container_shas = _norm_sha_set(raw_out.get("container_sha256"), raw_out.get("container_sha256s"))
            outputs.append(GtOutputSpec(
                output_id=out_id,
                job_id=job_id,
                container_bytes=_container_bytes(raw_out, base, out_id, container_shas),
                container_sha256s=container_shas,
                pixel_sha256s=_norm_sha_set(raw_out.get("pixel_sha256"), raw_out.get("pixel_sha256s")),
                raster_sha256s=_norm_sha_set(raw_out.get("raster_sha256"), raw_out.get("raster_sha256s")),
                raw_sha256s=_norm_sha_set(raw_out.get("raw_sha256"), raw_out.get("raw_sha256s")),
                physical_offsets=_norm_offset_set(raw_out.get("physical_offsets")),
                carve_types=_norm_str_set(raw_out.get("carve_type"), raw_out.get("carve_types")),
                artifact_ids=_norm_str_set(raw_out.get("artifact_id"), raw_out.get("artifact_ids")),
                output_node_id=_norm_node_id(raw_out.get("output_node_id")),
                batch_index=_parse_batch_index(raw_out.get("batch_index"), f"{out_id}.batch_index"),
                filename=raw_out.get("filename"),
                expected_ambiguous=bool(raw_out.get("expected_ambiguous", False)),
                expected_tier=_parse_tier(raw_out.get("expected_tier"), f"{out_id}.expected_tier"),
                allowed_tiers=_parse_allowed_tiers(raw_out.get("allowed_tiers"), f"{out_id}.allowed_tiers"),
                min_expected_tier=_parse_tier(raw_out.get("min_expected_tier"), f"{out_id}.min_expected_tier"),
                max_allowed_tier=_parse_tier(raw_out.get("max_allowed_tier"), f"{out_id}.max_allowed_tier"),
            ))
        jobs.append(GtJobSpec(
            job_id=job_id,
            prompt_ids=_norm_str_set(raw_job.get("prompt_id"), raw_job.get("prompt_ids")),
            canonical_prompt_sha256s=_norm_sha_set(
                raw_job.get("canonical_prompt_sha256"), raw_job.get("canonical_prompt_sha256s"),
            ),
            canonical_ui_sha256s=_norm_sha_set(
                raw_job.get("canonical_ui_sha256"), raw_job.get("canonical_ui_sha256s"),
            ),
            workflow_ids=_norm_str_set(raw_job.get("workflow_id"), raw_job.get("workflow_ids")),
            expected_tier=_parse_tier(raw_job.get("expected_tier"), f"{job_id}.expected_tier"),
            allowed_tiers=_parse_allowed_tiers(raw_job.get("allowed_tiers"), f"{job_id}.allowed_tiers"),
            min_expected_tier=_parse_tier(raw_job.get("min_expected_tier"), f"{job_id}.min_expected_tier"),
            max_allowed_tier=_parse_tier(raw_job.get("max_allowed_tier"), f"{job_id}.max_allowed_tier"),
            outputs=tuple(outputs),
        ))

    raw_distractors = gt.get("distractors", [])
    if not isinstance(raw_distractors, list):
        raise ValueError("ground truth 'distractors' must be a list")
    distractors: list[GtDistractorSpec] = []
    seen_distractors: set[str] = set()
    for raw_dist in raw_distractors:
        if not isinstance(raw_dist, Mapping):
            raise ValueError("each ground-truth distractor entry must be a JSON object")
        dist_id = str(raw_dist["distractor_id"])
        if dist_id in seen_distractors:
            raise ValueError(f"duplicate ground-truth distractor_id: {dist_id}")
        seen_distractors.add(dist_id)
        max_tier = _parse_tier(raw_dist.get("max_allowed_tier", "UNTIERED"), f"{dist_id}.max_allowed_tier")
        allowed_jobs = None
        if "allowed_link_job_ids" in raw_dist:
            job_ids = raw_dist["allowed_link_job_ids"]
            if not isinstance(job_ids, list) or any(
                not isinstance(jid, str) or jid not in seen_jobs for jid in job_ids
            ):
                raise ValueError(f"{dist_id}.allowed_link_job_ids must list known ground-truth job IDs")
            allowed_jobs = frozenset(job_ids)
        distractors.append(GtDistractorSpec(
            distractor_id=dist_id,
            reason=raw_dist.get("reason"),
            container_sha256s=_norm_sha_set(raw_dist.get("container_sha256"), raw_dist.get("container_sha256s")),
            pixel_sha256s=_norm_sha_set(raw_dist.get("pixel_sha256"), raw_dist.get("pixel_sha256s")),
            raster_sha256s=_norm_sha_set(raw_dist.get("raster_sha256"), raw_dist.get("raster_sha256s")),
            raw_sha256s=_norm_sha_set(raw_dist.get("raw_sha256"), raw_dist.get("raw_sha256s")),
            physical_offsets=_norm_offset_set(raw_dist.get("physical_offsets")),
            carve_types=_norm_str_set(raw_dist.get("carve_type"), raw_dist.get("carve_types")),
            artifact_ids=_norm_str_set(raw_dist.get("artifact_id"), raw_dist.get("artifact_ids")),
            allow_link_without_tier=bool(raw_dist.get("allow_link_without_tier", False)),
            allowed_link_job_ids=allowed_jobs,
            expected_tier=_parse_tier(raw_dist.get("expected_tier"), f"{dist_id}.expected_tier"),
            allowed_tiers=_parse_allowed_tiers(raw_dist.get("allowed_tiers"), f"{dist_id}.allowed_tiers"),
            min_expected_tier=_parse_tier(raw_dist.get("min_expected_tier"), f"{dist_id}.min_expected_tier"),
            max_allowed_tier=max_tier,
        ))
    return dump_sha, jobs, distractors


def _artifact_identity(
    case_dir: Path,
    artifact: Mapping[str, Any],
    evidence_record: Mapping[str, Any] | None,
) -> ArtifactIdentity:
    artifact_id = str(artifact["artifact_id"])
    carve_type = str(artifact.get("carve_type") or "UNKNOWN")
    completeness = str(artifact.get("completeness") or "UNKNOWN")
    container_sha = _norm_sha(artifact.get("container_sha256"))
    pixel_sha = _norm_sha(artifact.get("pixel_sha256"))
    has_gaps = bool(artifact.get("gaps"))
    ev_sha = (evidence_record or {}).get("sha256")
    ev_sha_map = ev_sha if isinstance(ev_sha, dict) else {}
    raw_sha = _norm_sha(ev_sha_map.get("raw"))
    if raw_sha is None:
        raw_sha = _file_raw_sha256(case_dir, artifact.get("raw_file_paths") or ())
    explicit_raster_sha = (
        _norm_sha(artifact.get("raster_sha256"))
        or _norm_sha((evidence_record or {}).get("raster_sha256"))
        or _norm_sha(ev_sha_map.get("raster"))
    )
    raster_sha = explicit_raster_sha or (_norm_sha(ev_sha_map.get("pixels")) if not has_gaps else None)
    offsets = _norm_offset_set(artifact.get("physical_offsets") or [])
    if container_sha is not None:
        stable_id, kind = f"container_sha256:{container_sha}", "container_sha256"
    elif raw_sha is not None:
        stable_id, kind = f"raw_sha256:{raw_sha}", "raw_sha256"
    elif explicit_raster_sha is not None and pixel_sha is None:
        stable_id, kind = f"raster_sha256:{explicit_raster_sha}:{carve_type}", "raster_sha256"
    elif pixel_sha is not None:
        stable_id, kind = f"pixel_sha256:{pixel_sha}:{carve_type}", "pixel_sha256"
    elif raster_sha is not None:
        stable_id, kind = f"raster_sha256:{raster_sha}:{carve_type}", "raster_sha256"
    elif offsets:
        stable_id, kind = f"physical_offset:{carve_type}:{','.join(sorted(offsets))}", "physical_offset"
    else:
        stable_id, kind = f"transient:{artifact_id}", "transient"
    prefix_length, prefix_sha = _prefix_identity(artifact)
    return ArtifactIdentity(
        artifact_id=artifact_id,
        carve_type=carve_type,
        completeness=completeness,
        prefix_length=prefix_length,
        prefix_sha256=prefix_sha,
        container_sha256=container_sha,
        pixel_sha256=pixel_sha,
        raster_sha256=raster_sha,
        raw_sha256=raw_sha,
        has_gaps=has_gaps,
        physical_offsets=offsets,
        stable_id=stable_id,
        identity_kind=kind,
    )


def _workflow_identity(
    case_dir: Path,
    workflow: Mapping[str, Any],
    evidence_record: Mapping[str, Any] | None,
) -> WorkflowIdentity:
    workflow_id = str(workflow["workflow_id"])
    prompt_format = str(workflow.get("prompt_format") or "UNKNOWN")
    completeness = str(workflow.get("completeness") or "UNKNOWN")
    encoding = str(workflow.get("encoding_detected") or "UNKNOWN")
    params = workflow.get("execution_parameters") or {}
    pids = set(_norm_str_set(params.get("prompt_id"), params.get("additional_prompt_ids")))
    if isinstance(evidence_record, Mapping) and isinstance(evidence_record.get("prompt_id"), str):
        ev_pid = evidence_record["prompt_id"].strip()
        if ev_pid:
            pids.add(ev_pid)
    prompt_ids = frozenset(pids)
    canonical_sha = _norm_sha(workflow.get("canonical_json_sha256"))
    raw_path = workflow.get("raw_file_path")
    raw_sha = _file_raw_sha256(case_dir, [raw_path]) if isinstance(raw_path, str) and raw_path else None
    if prompt_ids:
        stable_id, kind = f"prompt_id:{sorted(prompt_ids)[0]}", "prompt_id"
    elif canonical_sha is not None:
        stable_id, kind = f"canonical_json_sha256:{canonical_sha}", "canonical_json_sha256"
    elif raw_sha is not None:
        stable_id, kind = f"raw_sha256:{raw_sha}", "raw_sha256"
    else:
        stable_id, kind = f"transient:{workflow_id}", "transient"
    return WorkflowIdentity(
        workflow_id=workflow_id,
        prompt_format=prompt_format,
        completeness=completeness,
        encoding_detected=encoding,
        prompt_ids=prompt_ids,
        canonical_json_sha256=canonical_sha,
        raw_sha256=raw_sha,
        stable_id=stable_id,
        identity_kind=kind,
    )


def _match_workflow_to_jobs(wf: WorkflowIdentity, jobs: list[GtJobSpec]) -> tuple[list[str], str | None, bool]:
    """Returns (matched_job_ids, matched_by, prompt_hash_conflict)."""
    # 1. Strongest: prompt_id match
    if wf.prompt_ids:
        by_pid = [j for j in jobs if wf.prompt_ids & j.prompt_ids]
        if by_pid:
            conflict = False
            if wf.prompt_format == "API_GRAPH" and wf.completeness == "COMPLETE" and wf.canonical_json_sha256:
                for job in by_pid:
                    if job.canonical_prompt_sha256s and wf.canonical_json_sha256 not in job.canonical_prompt_sha256s:
                        conflict = True
            return [j.job_id for j in by_pid], "prompt_id", conflict

    # 2. Canonical prompt graph hash (API_GRAPH) or UI graph hash (UI_WORKFLOW).
    # Never match across disjoint non-empty prompt_ids (same prompt template, different job execution).
    def _prompt_id_compatible(job: GtJobSpec) -> bool:
        return not (wf.prompt_ids and job.prompt_ids and not (wf.prompt_ids & job.prompt_ids))

    if wf.canonical_json_sha256 is not None:
        if wf.prompt_format == "API_GRAPH":
            by_hash = [
                j.job_id for j in jobs
                if wf.canonical_json_sha256 in j.canonical_prompt_sha256s and _prompt_id_compatible(j)
            ]
            if by_hash:
                return by_hash, "canonical_prompt_sha256", False
        elif wf.prompt_format == "UI_WORKFLOW":
            by_ui = [
                j.job_id for j in jobs
                if wf.canonical_json_sha256 in j.canonical_ui_sha256s and _prompt_id_compatible(j)
            ]
            if by_ui:
                return by_ui, "canonical_ui_sha256", False

    # 3. Last-resort transient fallback (disclosed in identity_stability)
    by_id = [j.job_id for j in jobs if wf.workflow_id in j.workflow_ids and _prompt_id_compatible(j)]
    if by_id:
        return by_id, "transient_workflow_id", False
    return [], None, False


def _match_artifact_to_gt(
    art: ArtifactIdentity,
    outputs: list[GtOutputSpec],
    distractors: list[GtDistractorSpec],
) -> tuple[str | None, list[str], str | None]:
    """Matches an artifact to GT outputs or GT distractors using stable identity keys first.

    Returns ``(target_kind, matched_ids, matched_by)`` where ``target_kind`` is ``"output"``,
    ``"distractor"``, or ``None``.
    """
    # Check each identity tier from most specific stable key down to transient fallback.
    def check_spec(spec: GtOutputSpec | GtDistractorSpec, key: str) -> bool:
        if key == "container_sha256":
            return art.container_sha256 is not None and art.container_sha256 in spec.container_sha256s
        if key == "raw_sha256":
            return art.raw_sha256 is not None and art.raw_sha256 in spec.raw_sha256s
        if key == "raster_sha256":
            if not spec.raster_sha256s:
                return False
            if art.raster_sha256 is not None and art.raster_sha256 in spec.raster_sha256s:
                return True
            return not art.has_gaps and art.pixel_sha256 is not None and art.pixel_sha256 in spec.raster_sha256s
        if key == "pixel_sha256":
            return art.pixel_sha256 is not None and art.pixel_sha256 in spec.pixel_sha256s
        if key == "container_prefix":
            return _verifies_prefix(art, getattr(spec, "container_bytes", None))
        if key == "physical_offset":
            if not (art.physical_offsets and (art.physical_offsets & spec.physical_offsets)):
                return False
            return not spec.carve_types or art.carve_type in spec.carve_types
        if key == "transient_artifact_id":
            return art.artifact_id in spec.artifact_ids
        return False

    for key in (
        "container_sha256",
        "raw_sha256",
        "raster_sha256",
        "pixel_sha256",
        "container_prefix",
        "physical_offset",
        "transient_artifact_id",
    ):
        out_hits = [o.output_id for o in outputs if check_spec(o, key)]
        dist_hits = [d.distractor_id for d in distractors if check_spec(d, key)]
        if out_hits and not dist_hits:
            return "output", out_hits, key
        if dist_hits and not out_hits:
            return "distractor", dist_hits, key
        if out_hits and dist_hits:
            raise ValueError(
                f"ground truth conflict on {key} for {art.artifact_id}: matches outputs {out_hits} "
                f"and distractors {dist_hits}"
            )
    return None, [], None


def _evaluate_tier_expectation(
    achieved_tier: str | None,
    *,
    observed_tier: str | None = None,
    expected_tier: str | None,
    allowed_tiers: frozenset[str] | None,
    min_expected_tier: str | None,
    max_allowed_tier: str | None,
) -> dict[str, Any]:
    has_any = any(x is not None for x in (expected_tier, allowed_tiers, min_expected_tier, max_allowed_tier))
    if not has_any:
        return {
            "has_expectation": False,
            "has_upper_bound": False,
            "has_lower_bound": False,
            "status": "NO_EXPECTATION",
            "achieved_tier": achieved_tier,
            "observed_tier": observed_tier,
        }

    upper_bounds: list[int] = []
    lower_bounds: list[int] = []
    if expected_tier is not None:
        upper_bounds.append(TIER_RANK[expected_tier])
        lower_bounds.append(TIER_RANK[expected_tier])
    if allowed_tiers is not None:
        ranks = [TIER_RANK[t] for t in allowed_tiers]
        upper_bounds.append(max(ranks))
        lower_bounds.append(min(ranks))
    if max_allowed_tier is not None:
        upper_bounds.append(TIER_RANK[max_allowed_tier])
    if min_expected_tier is not None:
        lower_bounds.append(TIER_RANK[min_expected_tier])

    achieved_rank = TIER_RANK[achieved_tier]
    upper_tier_candidates = [t for t in (achieved_tier, observed_tier) if t is not None]
    highest_observed = max(upper_tier_candidates, key=lambda t: TIER_RANK[t]) if upper_tier_candidates else None
    upper = min(upper_bounds) if upper_bounds else None
    lower = max(lower_bounds) if lower_bounds else None

    if upper is not None and highest_observed is not None and TIER_RANK[highest_observed] > upper:
        status = "OVER_TIERED"
    elif lower is not None and achieved_rank < lower:
        status = "UNDER_TIERED"
    elif expected_tier is not None and achieved_tier != expected_tier:
        status = "MISMATCH"
    elif allowed_tiers is not None and achieved_tier not in allowed_tiers:
        status = "MISMATCH"
    else:
        status = "PASS"

    return {
        "has_expectation": True,
        "has_upper_bound": upper is not None,
        "has_lower_bound": lower is not None,
        "status": status,
        "achieved_tier": achieved_tier,
        "observed_tier": observed_tier,
        "expected_tier": expected_tier,
        "allowed_tiers": sorted(allowed_tiers) if allowed_tiers is not None else None,
        "min_expected_tier": min_expected_tier,
        "max_allowed_tier": max_allowed_tier,
    }


def _extract_output_identity_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    include_pixel_sha_key: bool = False,
) -> tuple[frozenset[str], frozenset[int], frozenset[str]]:
    """Extracts predicted (output_node_ids, batch_indices, raster_sha256s) from an attribution record."""
    if not isinstance(mapping, Mapping):
        return frozenset(), frozenset(), frozenset()
    nodes: set[str] = set()
    batches: set[int] = set()
    rasters: set[str] = set()

    def _collect(obj: Mapping[str, Any], allow_pixel_key: bool) -> None:
        for key in ("output_node_id", "node_id"):
            if (nid := _norm_node_id(obj.get(key))) is not None:
                nodes.add(nid)
        if isinstance(obj.get("output_node_ids"), list):
            for raw_nid in obj["output_node_ids"]:
                if (nid := _norm_node_id(raw_nid)) is not None:
                    nodes.add(nid)
        raw_batch = obj.get("batch_index")
        if isinstance(raw_batch, int) and not isinstance(raw_batch, bool) and raw_batch >= 0:
            batches.add(raw_batch)
        for key in RASTER_DIGEST_KEYS:
            if (digest := _norm_sha(obj.get(key))) is not None:
                rasters.add(digest)
        if allow_pixel_key:
            for key in ("pixel_sha256", "regenerated_pixel_sha256", "output_pixel_sha256", "matched_pixel_sha256"):
                if (digest := _norm_sha(obj.get(key))) is not None:
                    rasters.add(digest)

    _collect(mapping, include_pixel_sha_key)
    for nested_key in OUTPUT_IDENTITY_NESTED_KEYS:
        nested = mapping.get(nested_key)
        if isinstance(nested, Mapping):
            _collect(nested, include_pixel_sha_key or nested_key != "best")
            if nested_key == "selected_raster" and (digest := _norm_sha(nested.get("sha256"))) is not None:
                rasters.add(digest)
    return frozenset(nodes), frozenset(batches), frozenset(rasters)


def _load_scorecard_output_identities(
    case_path: Path,
    case_dump_sha: str | None,
    view: EvalView,
) -> dict[str, dict[str, tuple[frozenset[str], frozenset[int], frozenset[str]]]]:
    """Reads explicit output-specific candidates (role=='OUTPUT' or explicit output identity) from scorecard.json."""
    scorecard_path = case_path / SCORECARD_NAME
    if not scorecard_path.is_file():
        return {}
    doc = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        return {}
    sc_dump_sha = _norm_sha(doc.get("dump_sha256"))
    if case_dump_sha is not None and sc_dump_sha is not None and sc_dump_sha != case_dump_sha:
        return {}

    per_art: dict[str, dict[str, tuple[set[str], set[int], set[str]]]] = {}
    for row in doc.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        row_pid = str(row.get("prompt_id") or "").strip()
        row_node = _norm_node_id(row.get("output_node_id"))
        row_batch = row.get("batch_index")
        valid_row_batch = isinstance(row_batch, int) and not isinstance(row_batch, bool) and row_batch >= 0
        candidates = (row.get("image") or {}).get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, Mapping) or not cand.get("artifact_id"):
                continue
            if cand.get("role") in ("JOB", "INTERMEDIATE", "AMBIGUOUS_OUTPUT"):
                continue
            basis = str(cand.get("basis") or "")
            if view == "extraction" and basis == "REGENERATION_COMPARISON":
                continue
            c_nodes, c_batches, c_rasters = _extract_output_identity_from_mapping(cand, include_pixel_sha_key=True)
            is_explicit_output = cand.get("role") == "OUTPUT" or bool(c_nodes or c_batches or c_rasters)
            if not is_explicit_output:
                continue
            aid = str(cand["artifact_id"])
            bucket = per_art.setdefault(aid, {}).setdefault(row_pid, (set(), set(), set()))
            if c_nodes:
                bucket[0].update(c_nodes)
            elif row_node is not None:
                bucket[0].add(row_node)
            if c_batches:
                bucket[1].update(c_batches)
            elif valid_row_batch:
                bucket[1].add(row_batch)
            bucket[2].update(c_rasters)

    return {
        aid: {pid: (frozenset(n), frozenset(b), frozenset(r)) for pid, (n, b, r) in by_pid.items()}
        for aid, by_pid in per_art.items()
    }


def _resolve_predicted_links(
    artifacts: list[dict[str, Any]],
    manifest_links: list[dict[str, Any]],
    wf_to_gt_jobs: dict[str, list[str]],
    wf_identities: Mapping[str, WorkflowIdentity],
    pid_to_gt_jobs: dict[str, list[str]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    scorecard_output_ids: Mapping[str, Mapping[str, tuple[frozenset[str], frozenset[int], frozenset[str]]]],
    image_links_doc: dict[str, Any] | None,
    view: EvalView,
) -> dict[str, dict[str, Any]]:
    """Computes each artifact's predicted attribution status, candidate GT jobs, output identity, and source."""
    links_by_art: dict[str, list[dict[str, Any]]] = {}
    for link in manifest_links:
        links_by_art.setdefault(str(link["artifact_id"]), []).append(link)

    img_link_by_art: dict[str, dict[str, Any]] = {}
    regen_by_file: dict[str, dict[str, Any]] = {}
    if view in ("effective", "memory_only") and image_links_doc is not None:
        for entry in image_links_doc.get("memory_images", []):
            if isinstance(entry, dict) and entry.get("artifact_id"):
                img_link_by_art[str(entry["artifact_id"])] = entry
        for regen_entry in image_links_doc.get("regenerated", []):
            if isinstance(regen_entry, dict) and regen_entry.get("file"):
                regen_by_file[str(regen_entry["file"])] = regen_entry

    predictions: dict[str, dict[str, Any]] = {}
    for art in artifacts:
        art_id = str(art["artifact_id"])
        art_links = links_by_art.get(art_id, [])
        ev_rec = evidence_by_id.get(art_id)

        # 1. Extraction-time attribution from manifest (collect BOTH associated_workflow_id and correlation_links)
        ext_status = str(art.get("link_status") or "UNASSOCIATED")
        ext_class: str | None = None
        ext_wf_ids: list[str] = []
        ext_invalid_rule = False
        active_art_links = [lnk for lnk in art_links if lnk.get("link_class") in ("STRICT", "SUPPORTED")]
        if ext_status in ("STRICT", "SUPPORTED"):
            ext_class = ext_status
            wf_set: set[str] = set()
            if art.get("associated_workflow_id"):
                wf_set.add(str(art["associated_workflow_id"]))
            wf_set.update(str(lnk["workflow_id"]) for lnk in art_links if lnk.get("link_class") == ext_status)
            ext_wf_ids = sorted(wf_set)
            if not ext_wf_ids:
                ext_invalid_rule = True
        elif ext_status == "AMBIGUOUS":
            ext_wf_ids = sorted(
                set(str(w) for w in (art.get("ambiguous_candidate_workflows") or []))
                | {str(lnk["workflow_id"]) for lnk in art_links if lnk.get("link_class") == "AMBIGUOUS"}
            )
            if active_art_links:
                ext_invalid_rule = True
        else:
            if active_art_links or art.get("associated_workflow_id"):
                ext_invalid_rule = True

        ext_gt_jobs = sorted({jid for wid in ext_wf_ids for jid in wf_to_gt_jobs.get(wid, [])})
        ext_unmapped_wfs = sorted(wid for wid in ext_wf_ids if not wf_to_gt_jobs.get(wid))
        if ext_status in ("STRICT", "SUPPORTED") and len(ext_wf_ids) > 1:
            distinct_wf_job_sets = {tuple(sorted(wf_to_gt_jobs.get(w, []))) for w in ext_wf_ids}
            if len(distinct_wf_job_sets) > 1:
                ext_invalid_rule = True

        # Extraction-side output identity (links, evidence_record, execution_cache, scorecard OUTPUT rows)
        ext_nodes: set[str] = set()
        ext_batches: set[int] = set()
        ext_rasters: set[str] = set()
        for lnk in art_links:
            if lnk.get("link_class") == ext_status:
                ln, lb, lr = _extract_output_identity_from_mapping(lnk, include_pixel_sha_key=True)
                ext_nodes.update(ln)
                ext_batches.update(lb)
                ext_rasters.update(lr)
        if ev_rec is not None:
            en, eb, er = _extract_output_identity_from_mapping(ev_rec, include_pixel_sha_key=False)
            ext_nodes.update(en)
            ext_batches.update(eb)
            ext_rasters.update(er)
            tie = ev_rec.get("execution_cache")
            if isinstance(tie, Mapping):
                tn, tb, tr = _extract_output_identity_from_mapping(tie, include_pixel_sha_key=False)
                ext_nodes.update(tn)
                ext_batches.update(tb)
                ext_rasters.update(tr)
        if ext_status in ("STRICT", "SUPPORTED") and art_id in scorecard_output_ids:
            linked_pids = {pid for wid in ext_wf_ids for pid in wf_identities[wid].prompt_ids}
            for sc_pid, (sn, sb, sr) in scorecard_output_ids[art_id].items():
                if not linked_pids or not sc_pid or sc_pid in linked_pids:
                    ext_nodes.update(sn)
                    ext_batches.update(sb)
                    ext_rasters.update(sr)

        # 2. Optional image-links.json attribution (only in effective/memory_only views)
        il = img_link_by_art.get(art_id)
        il_status: str | None = None
        il_pids: list[str] = []
        il_gt_jobs: list[str] = []
        il_unmapped_pids: list[str] = []
        il_invalid_rule = False
        il_output_ambig = False
        il_nodes: set[str] = set()
        il_batches: set[int] = set()
        il_rasters: set[str] = set()
        if il is not None:
            attr = str(il.get("attribution") or "UNATTRIBUTED")
            raw_agreeing = il.get("agreeing_jobs")
            agreeing_jobs = sorted({str(p) for p in raw_agreeing}) if isinstance(raw_agreeing, list) else None
            if attr == "SUPPORTED":
                pid = str(il["prompt_id"]).strip() if il.get("prompt_id") is not None else ""
                if not pid:
                    il_status = "SUPPORTED"
                    il_invalid_rule = True
                else:
                    il_status = "SUPPORTED"
                    il_pids = [pid]
                    if agreeing_jobs is not None and (len(agreeing_jobs) != 1 or agreeing_jobs != [pid]):
                        il_invalid_rule = True
                inodes, ibatches, irasters = _extract_output_identity_from_mapping(il, include_pixel_sha_key=True)
                il_nodes.update(inodes)
                il_batches.update(ibatches)
                il_rasters.update(irasters)
                matched_file = il.get("regenerated_file") or (
                    il.get("best").get("file") if isinstance(il.get("best"), Mapping) else None
                )
                if isinstance(matched_file, str) and matched_file in regen_by_file:
                    rn, rb, rr = _extract_output_identity_from_mapping(
                        regen_by_file[matched_file], include_pixel_sha_key=True,
                    )
                    il_nodes.update(rn)
                    il_batches.update(rb)
                    il_rasters.update(rr)
                raw_agreeing_outputs = il.get("agreeing_outputs")
                out_attr = str(il.get("output_attribution") or "").upper()
                if out_attr == "AMBIGUOUS":
                    il_output_ambig = True
                elif isinstance(raw_agreeing_outputs, list) and len(raw_agreeing_outputs) > 1:
                    if out_attr == "SUPPORTED" or il_nodes or il_batches or il_rasters:
                        il_invalid_rule = True
                    else:
                        il_output_ambig = True
            elif attr == "AMBIGUOUS":
                il_status = "AMBIGUOUS"
                il_pids = agreeing_jobs or []
            else:
                il_status = "UNASSOCIATED"
            il_gt_jobs = sorted({jid for pid in il_pids for jid in pid_to_gt_jobs.get(pid, [])})
            il_unmapped_pids = sorted(pid for pid in il_pids if not pid_to_gt_jobs.get(pid))

        # 3. Combine extraction and image-links predictions cleanly (never hiding conflicts or unmapped targets)
        conflict = False
        invalid_rule = False
        output_ambig = False
        pred_nodes: frozenset[str] = frozenset()
        pred_batches: frozenset[int] = frozenset()
        pred_rasters: frozenset[str] = frozenset()
        unmapped_targets: list[str] = []

        if ext_status in ("STRICT", "SUPPORTED"):
            status = ext_status
            link_class = ext_class
            pred_jobs = ext_gt_jobs
            unmapped_targets = list(ext_unmapped_wfs)
            invalid_rule = ext_invalid_rule
            source = "extraction"
            pred_nodes = frozenset(ext_nodes)
            pred_batches = frozenset(ext_batches)
            pred_rasters = frozenset(ext_rasters)
            if il_status == "SUPPORTED":
                source = "extraction+image_links"
                invalid_rule = invalid_rule or il_invalid_rule
                unmapped_targets = sorted(set(unmapped_targets) | set(il_unmapped_pids))
                if set(ext_gt_jobs) != set(il_gt_jobs) or bool(ext_unmapped_wfs) != bool(il_unmapped_pids):
                    conflict = True
                    pred_jobs = sorted(set(ext_gt_jobs) | set(il_gt_jobs))
                if ext_nodes and il_nodes and not (ext_nodes & il_nodes):
                    conflict = True
                if ext_batches and il_batches and ext_batches != il_batches:
                    conflict = True
                if ext_rasters and il_rasters and ext_rasters != il_rasters:
                    conflict = True
                pred_nodes = frozenset(ext_nodes | il_nodes)
                pred_batches = frozenset(ext_batches | il_batches)
                pred_rasters = frozenset(ext_rasters | il_rasters)
            elif il_status == "AMBIGUOUS":
                source = "extraction+image_links"
                if ext_gt_jobs and il_gt_jobs and not (set(ext_gt_jobs) & set(il_gt_jobs)):
                    conflict = True
        elif il_status == "SUPPORTED":
            if il_output_ambig and not il_invalid_rule:
                status = "AMBIGUOUS"
                link_class = "AMBIGUOUS"
                output_ambig = True
            else:
                status = "SUPPORTED"
                link_class = "REGENERATION_LINK"
            pred_jobs = il_gt_jobs
            unmapped_targets = list(il_unmapped_pids)
            invalid_rule = il_invalid_rule
            source = "image_links"
            pred_nodes, pred_batches, pred_rasters = frozenset(il_nodes), frozenset(il_batches), frozenset(il_rasters)
        elif ext_status == "AMBIGUOUS" or il_status == "AMBIGUOUS":
            status = "AMBIGUOUS"
            link_class = "AMBIGUOUS"
            pred_jobs = sorted(set(ext_gt_jobs) | set(il_gt_jobs))
            unmapped_targets = sorted(set(ext_unmapped_wfs) | set(il_unmapped_pids))
            invalid_rule = ext_invalid_rule
            source = "extraction" if ext_status == "AMBIGUOUS" else "image_links"
            if ext_status == "AMBIGUOUS" and il_status == "AMBIGUOUS":
                source = "extraction+image_links"
        else:
            status = "UNASSOCIATED"
            link_class = None
            pred_jobs = []
            invalid_rule = ext_invalid_rule
            source = "none"

        predictions[art_id] = {
            "status": status,
            "link_class": link_class,
            "source": source,
            "workflow_ids": ext_wf_ids,
            "image_link_prompt_ids": il_pids,
            "predicted_gt_job_ids": pred_jobs,
            "unmapped_targets": unmapped_targets,
            "conflicting_sources": conflict,
            "invalid_link_rule": invalid_rule,
            "output_identity_ambiguous": output_ambig,
            "predicted_output_node_ids": sorted(pred_nodes),
            "predicted_batch_indices": sorted(pred_batches),
            "predicted_raster_sha256s": sorted(pred_rasters),
            "image_link_entry": il,
        }
    return predictions


def _evaluate_output_identity(
    pred: Mapping[str, Any],
    matched_outputs: list[GtOutputSpec],
    job_all_outputs: list[GtOutputSpec],
) -> tuple[str, list[str], dict[str, Any]]:
    """Compares predicted output-level identity against matched GT output(s) of the same job.

    Returns ``(status, verified_output_ids, details)`` where ``status`` is one of:
    ``"MATCHED"``, ``"MISMATCHED"``, ``"AMBIGUOUS"``, ``"UNKNOWN_PREDICTED"``, ``"UNKNOWN_GROUNDTRUTH"``.
    """
    pred_nodes = set(pred.get("predicted_output_node_ids") or ())
    pred_batches = set(pred.get("predicted_batch_indices") or ())
    pred_rasters = set(pred.get("predicted_raster_sha256s") or ())
    details: dict[str, Any] = {
        "predicted_output_node_ids": sorted(pred_nodes),
        "predicted_batch_indices": sorted(pred_batches),
        "predicted_raster_sha256s": sorted(pred_rasters),
    }
    if pred.get("output_identity_ambiguous"):
        return "AMBIGUOUS", [], details
    if not pred_nodes and not pred_batches and not pred_rasters:
        return "UNKNOWN_PREDICTED", [], details
    if len(pred_nodes) > 1 or len(pred_batches) > 1 or len(pred_rasters) > 1:
        return "MISMATCHED", [], details

    matched_ids: list[str] = []
    unknown_gt_ids: list[str] = []
    for out in matched_outputs:
        results: list[str] = []
        if pred_nodes:
            if out.output_node_id is None:
                results.append("UNKNOWN_GT")
            elif pred_nodes == {out.output_node_id}:
                results.append("MATCH")
            else:
                results.append("MISMATCH")
        if pred_batches:
            if out.batch_index is None:
                results.append("UNKNOWN_GT")
            elif pred_batches == {out.batch_index}:
                results.append("MATCH")
            else:
                results.append("MISMATCH")
        if pred_rasters:
            if out.raster_sha256s:
                results.append("MATCH" if pred_rasters <= out.raster_sha256s else "MISMATCH")
            elif out.pixel_sha256s and pred_rasters <= out.pixel_sha256s:
                results.append("MATCH")
            elif any(
                pred_rasters & (other.raster_sha256s | other.pixel_sha256s | other.container_sha256s)
                for other in job_all_outputs
                if other.output_id != out.output_id
            ):
                results.append("MISMATCH")
            else:
                results.append("UNKNOWN_GT")

        if "MISMATCH" in results:
            continue
        if "MATCH" in results:
            matched_ids.append(out.output_id)
        else:
            unknown_gt_ids.append(out.output_id)

    if matched_ids:
        return "MATCHED", matched_ids, details
    if unknown_gt_ids:
        return "UNKNOWN_GROUNDTRUTH", [], details
    return "MISMATCHED", [], details


INDEPENDENT_STRICT_KEYS = frozenset({
    "PROMPT_ID",
    "CANONICAL_PROMPT_HASH",
    "EXECUTION_GRAPH_MATCH",
    "EXECUTION_CACHE_SIGNATURE",
})
NON_INDEPENDENT_STRICT_KEYS = frozenset({"EMBEDDED_PROMPT", "INPUT_REVISION_UNVERIFIED"})


def _independent_extraction_credit(
    art_id: str,
    pred: Mapping[str, Any],
    true_jobs: list[str],
    links: Iterable[Mapping[str, Any]],
    workflows_by_id: Mapping[str, Mapping[str, Any]],
    wf_to_gt_jobs: Mapping[str, list[str]],
) -> tuple[bool, str]:
    """Whether a CORRECT artifact link is corroborated by separately recovered extraction evidence.

    Credit needs an extraction-time STRICT correlation link, scoped to the evaluated view, from a workflow that
    is not this image's own embedded text, whose workflow maps to exactly the true job, and whose matched keys
    include a recognised independent key and neither EMBEDDED_PROMPT nor INPUT_REVISION_UNVERIFIED.
    Regenerated-only (image-links) attribution, AMBIGUOUS/SUPPORTED links and absent keys never earn credit.
    """
    if pred.get("status") != "STRICT" or pred.get("link_class") != "STRICT":
        return False, "NOT_EXTRACTION_STRICT"
    if pred.get("source") not in ("extraction", "extraction+image_links"):
        return False, "NOT_EXTRACTION_STRICT"
    reasons: list[str] = []
    for link in links:
        if str(link.get("artifact_id")) != art_id or link.get("link_class") != "STRICT":
            continue
        wid = str(link.get("workflow_id"))
        workflow = workflows_by_id.get(wid)
        raw_keys = link.get("matched_keys")
        keys = {k for k in raw_keys if isinstance(k, str)} if isinstance(raw_keys, list) else set()
        if workflow is None:
            reasons.append("WORKFLOW_OUT_OF_VIEW")
        elif keys & NON_INDEPENDENT_STRICT_KEYS:
            reasons.append("SELF_DESCRIBED_OR_UNVERIFIED_INPUT")
        elif not keys & INDEPENDENT_STRICT_KEYS:
            reasons.append("NO_INDEPENDENT_KEYS")
        elif set(wf_to_gt_jobs.get(wid, [])) != set(true_jobs):
            reasons.append("WORKFLOW_NOT_TRUE_JOB")
        else:
            encoding = workflow.get("encoding_detected")
            locations = (workflow.get("execution_parameters") or {}).get("source_locations") or []
            own_png = any(isinstance(loc, Mapping) and str(loc.get("png")) == art_id for loc in locations)
            if not isinstance(encoding, str) or not encoding.strip():
                reasons.append("WORKFLOW_ENCODING_ABSENT")
            elif is_embedded_text(encoding) or own_png:
                reasons.append("OWN_EMBEDDED_METADATA")
            else:
                return True, "INDEPENDENT_STRICT_EXTRACTION"
    return False, reasons[0] if reasons else "NO_STRICT_LINK"


def _valid_whole_frame_entry(
    art: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if entry.get("attribution") != "SUPPORTED":
        return False
    pid = entry.get("prompt_id")
    if not isinstance(pid, str) or not pid.strip():
        return False
    agreeing = entry.get("agreeing_jobs")
    if isinstance(agreeing, list) and (len(agreeing) != 1 or agreeing != [pid]):
        return False
    if entry.get("kind") not in EXACT_MEMORY_KINDS or art.get("carve_type") not in EXACT_MEMORY_KINDS:
        return False
    if entry.get("complete") is not True or art.get("completeness") != "COMPLETE":
        return False
    if entry.get("whole_frame") is not True or entry.get("agreement") != "ROWS":
        return False
    if art.get("gaps"):
        return False
    meta = art.get("pixel_metadata")
    if isinstance(meta, Mapping) and meta.get("suspect_rows"):
        return False
    return True


def _valid_close_match_entry(entry: Mapping[str, Any] | None) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if entry.get("attribution") != "SUPPORTED":
        return False
    pid = entry.get("prompt_id")
    if not isinstance(pid, str) or not pid.strip():
        return False
    agreeing = entry.get("agreeing_jobs")
    if isinstance(agreeing, list) and (len(agreeing) != 1 or agreeing != [pid]):
        return False
    return entry.get("agreement") in ("ROWS", "LOSSY")


def _audit_image_link_promotions(
    image_links_doc: Mapping[str, Any] | None,
    artifacts_by_id: Mapping[str, Mapping[str, Any]],
    workflows_by_id: Mapping[str, Mapping[str, Any]],
    wf_identities: Mapping[str, WorkflowIdentity],
    *,
    scoped_only: bool,
) -> list[dict[str, Any]]:
    """Audits every promotion declared in image-links.json against ComfyTrace promotion invariants."""
    if not isinstance(image_links_doc, Mapping):
        return []
    mem_by_id = {
        str(e["artifact_id"]): e
        for e in image_links_doc.get("memory_images", [])
        if isinstance(e, Mapping) and e.get("artifact_id")
    }
    checks: list[dict[str, Any]] = []
    for idx, promo in enumerate(image_links_doc.get("promotions", [])):
        if not isinstance(promo, Mapping):
            checks.append({
                "index": idx, "tier": None, "basis": None, "valid": False, "issues": ["MALFORMED_PROMOTION"],
            })
            continue
        tier = promo.get("tier")
        basis = promo.get("basis")
        prompt_id = str(promo.get("prompt_id") or "").strip()
        wf_ids = [str(w) for w in (promo.get("workflow_ids") or [])]
        if "artifact_id" in promo and promo["artifact_id"] is not None:
            art_ids = [str(promo["artifact_id"])]
        else:
            art_ids = [str(a) for a in (promo.get("artifact_ids") or [])]

        if scoped_only and art_ids and not any(a in artifacts_by_id for a in art_ids):
            continue

        issues: list[str] = []
        if tier == "E2":
            if basis != "REGENERATION_WHOLE_FRAME_MATCH":
                issues.append("E2_PROMOTION_WRONG_BASIS")
            if len(art_ids) != 1:
                issues.append("E2_PROMOTION_MISSING_ARTIFACT")
            for aid in art_ids:
                art = artifacts_by_id.get(aid)
                entry = mem_by_id.get(aid)
                if art is None:
                    issues.append("PROMOTION_UNKNOWN_ARTIFACT")
                elif not _valid_whole_frame_entry(art, entry):
                    issues.append("E2_PROMOTION_NOT_EXACT_WHOLE_FRAME")
                elif entry is not None and prompt_id and entry.get("prompt_id") != prompt_id:
                    issues.append("PROMOTION_PROMPT_ID_MISMATCH")
            if not wf_ids:
                issues.append("E2_PROMOTION_MISSING_WORKFLOW")
            else:
                has_complete_api = False
                for wid in wf_ids:
                    wf = workflows_by_id.get(wid)
                    if wf is None:
                        issues.append("PROMOTION_UNKNOWN_WORKFLOW")
                        continue
                    if wf.get("prompt_format") == "API_GRAPH" and wf.get("completeness") == "COMPLETE":
                        has_complete_api = True
                    if prompt_id and prompt_id not in wf_identities[wid].prompt_ids:
                        issues.append("PROMOTION_WORKFLOW_PROMPT_MISMATCH")
                if not has_complete_api:
                    issues.append("E2_PROMOTION_WITHOUT_COMPLETE_API_WORKFLOW")
        elif tier == "E3":
            if basis != "REGENERATION_CLOSE_MATCH":
                issues.append("E3_PROMOTION_WRONG_BASIS")
            if not art_ids:
                issues.append("E3_PROMOTION_MISSING_ARTIFACT")
            for aid in art_ids:
                if aid not in artifacts_by_id:
                    issues.append("PROMOTION_UNKNOWN_ARTIFACT")
                    continue
                entry = mem_by_id.get(aid)
                if not _valid_close_match_entry(entry):
                    issues.append("E3_PROMOTION_INVALID_CLOSE_MATCH")
                elif entry is not None and prompt_id and entry.get("prompt_id") != prompt_id:
                    issues.append("PROMOTION_PROMPT_ID_MISMATCH")
            if not wf_ids:
                issues.append("E3_PROMOTION_MISSING_WORKFLOW")
            else:
                has_api = False
                for wid in wf_ids:
                    wf = workflows_by_id.get(wid)
                    if wf is None:
                        issues.append("PROMOTION_UNKNOWN_WORKFLOW")
                        continue
                    if wf.get("prompt_format") == "API_GRAPH":
                        has_api = True
                    if prompt_id and prompt_id not in wf_identities[wid].prompt_ids:
                        issues.append("PROMOTION_WORKFLOW_PROMPT_MISMATCH")
                if not has_api:
                    issues.append("E3_PROMOTION_WITHOUT_API_WORKFLOW")
        else:
            issues.append(f"INVALID_IMAGE_LINK_PROMOTION_TIER:{tier}")

        deduped_issues = list(dict.fromkeys(issues))
        checks.append({
            "index": idx,
            "tier": tier,
            "basis": basis,
            "prompt_id": prompt_id or None,
            "artifact_ids": art_ids,
            "workflow_ids": wf_ids,
            "valid": not deduped_issues,
            "issues": deduped_issues,
        })
    return checks


def score_case(
    case_dir: Path | str,
    groundtruth: Path | str | Mapping[str, Any],
    *,
    view: EvalView = "effective",
    strict_dump_sha: bool = True,
) -> dict[str, Any]:
    """Scores an extracted ComfyTrace case directory against withheld ground truth.

    Args:
        case_dir: Path to an existing ComfyTrace case directory (opened read-only).
        groundtruth: Path to a withheld ground-truth JSON file, or an already-parsed mapping.
        view: ``"effective"`` (manifest + valid image-links + valid regenerations),
              ``"extraction"`` (raw manifest.json only), or
              ``"memory_only"`` (memory-backed findings from recovery-evidence.json + valid image-links).
        strict_dump_sha: If True and ground truth includes ``dump_sha256``, raise ``ValueError`` when it does
                         not match ``manifest.json``'s ``memory_image.sha256``.
    """
    if view not in ("effective", "extraction", "memory_only"):
        raise ValueError(f"unsupported view: {view!r}")
    case_path = Path(case_dir)
    manifest_path = case_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found in case directory: {manifest_path}")

    gt_doc: Mapping[str, Any]
    if isinstance(groundtruth, Mapping):
        gt_doc = groundtruth
    else:
        gt_doc = json.loads(Path(groundtruth).read_text(encoding="utf-8"))

    gt_dump_sha, jobs, distractors = _parse_groundtruth(
        gt_doc, None if isinstance(groundtruth, Mapping) else Path(groundtruth).parent)
    outputs_all = [out for job in jobs for out in job.outputs]
    outputs_by_id = {out.output_id: out for out in outputs_all}
    outputs_by_job: dict[str, list[GtOutputSpec]] = {}
    for out in outputs_all:
        outputs_by_job.setdefault(out.job_id, []).append(out)
    distractors_by_id = {d.distractor_id: d for d in distractors}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    metadata = manifest.get("case_metadata") or {}
    case_dump_sha = _norm_sha(metadata.get("memory_image", {}).get("sha256"))
    if strict_dump_sha and gt_dump_sha is not None and case_dump_sha != gt_dump_sha:
        raise ValueError(
            f"dump SHA-256 mismatch: case manifest has {case_dump_sha}, ground truth expects {gt_dump_sha}"
        )
    for field in ("carving_mode", "degradation_reason"):
        expected_field = f"expected_{field}"
        if expected_field in gt_doc and metadata.get(field) != gt_doc[expected_field]:
            raise ValueError(
                f"{field.replace('_', ' ')} mismatch: case manifest has {metadata.get(field)!r}, "
                f"ground truth expects {gt_doc[expected_field]!r}"
            )

    evidence_path = case_path / EVIDENCE_NAME
    evidence_doc = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else None
    evidence_by_id = {
        str(r["record_id"]): r for r in (evidence_doc or {}).get("records", []) if isinstance(r, dict)
    }

    links_path = case_path / LINKS_NAME
    raw_image_links = json.loads(links_path.read_text(encoding="utf-8")) if links_path.is_file() else None
    valid_image_links = (
        raw_image_links
        if raw_image_links is not None and raw_image_links.get("manifest_sha256") == manifest_sha
        else None
    )
    regenerations = load_regenerations(case_path / "regenerations")
    verified_links, verification_notes = verify_image_link_promotions(
        valid_image_links, manifest, regenerations, manifest_sha, case_path,
    )
    regen_by_id = {
        str(r["handoff_id"]): r
        for r in regenerations
        if isinstance(r, Mapping) and r.get("handoff_id")
    }
    scorecard_output_ids = _load_scorecard_output_identities(case_path, case_dump_sha, view)

    # Select findings & scope for the chosen view (never mutating manifest on disk or in memory)
    applied_notes: list[dict[str, Any]] = []
    if view == "extraction":
        active_findings = [
            dict(f, base_tier=f["tier"], extraction_tier=f["tier"], regeneration=None, image_link=None)
            for f in copy.deepcopy(manifest["findings"].get("findings", []))
        ]
        active_links_doc = None
    elif view == "effective":
        active_findings, applied_notes = effective_findings(
            copy.deepcopy(manifest), regenerations, manifest_sha, case_path, raw_image_links,
        )
        active_links_doc = valid_image_links
    else:  # memory_only
        if evidence_doc is None or evidence_doc.get("memory_only_findings") is None:
            raise ValueError(f"view='memory_only' requires {EVIDENCE_NAME} with memory_only_findings")
        active_findings = [
            dict(f, base_tier=f["tier"], extraction_tier=f["tier"], regeneration=None, image_link=None)
            for f in copy.deepcopy(evidence_doc["memory_only_findings"])
        ]
        if raw_image_links is not None:
            links_to_apply = raw_image_links if valid_image_links is None else verified_links
            applied_notes = [*verification_notes, *apply_image_links(active_findings, links_to_apply, manifest_sha)]
        active_links_doc = valid_image_links

    all_artifacts: list[dict[str, Any]] = list(manifest["findings"].get("artifacts", []))
    all_workflows: list[dict[str, Any]] = list(manifest["findings"].get("workflows", []))
    manifest_links: list[dict[str, Any]] = list(manifest["findings"].get("correlation_links", []))

    if view == "memory_only":
        def _is_memory_artifact(art: Mapping[str, Any]) -> bool:
            ev = evidence_by_id.get(str(art["artifact_id"]))
            return (ev is not None and ev.get("memory_source") in MEMORY_BACKED_SOURCES
                    and (ev.get("assembly") or {}).get("method") != "CONTENT_STITCHED")

        def _is_memory_workflow(wf: Mapping[str, Any]) -> bool:
            encoding = wf.get("encoding_detected")
            if encoding == "PNG_TEXT_STITCHED":
                return False
            if encoding == "CPYTHON_OBJECTS":
                return True
            ev = evidence_by_id.get(str(wf["workflow_id"]))
            return ev is not None and ev.get("memory_source") in MEMORY_BACKED_SOURCES

        artifacts = []
        for a in all_artifacts:
            if not _is_memory_artifact(a):
                continue
            ev = evidence_by_id.get(str(a["artifact_id"]))
            if ev is not None and ev.get("memory_source") == "MEMORY_PARTIAL":
                artifacts.append({**a, "completeness": "PARTIAL"})
            else:
                artifacts.append(a)
        workflows = []
        for w in all_workflows:
            if not _is_memory_workflow(w):
                continue
            ev = evidence_by_id.get(str(w["workflow_id"]))
            if (
                w.get("encoding_detected") != "CPYTHON_OBJECTS"
                and ev is not None
                and ev.get("memory_source") == "MEMORY_PARTIAL"
            ):
                workflows.append({**w, "completeness": "PARTIAL"})
            else:
                workflows.append(w)
        kept_art_ids = {str(a["artifact_id"]) for a in artifacts}
        kept_wf_ids = {str(w["workflow_id"]) for w in workflows}
        memory_links = (evidence_doc or {}).get("memory_only_correlation_links")
        scoped_links = [
            lnk for lnk in (manifest_links if memory_links is None else memory_links)
            if str(lnk["artifact_id"]) in kept_art_ids and str(lnk["workflow_id"]) in kept_wf_ids
        ]
        if memory_links is not None:
            memory_links_by_artifact: dict[str, list[dict[str, Any]]] = {}
            for lnk in scoped_links:
                memory_links_by_artifact.setdefault(lnk["artifact_id"], []).append(lnk)
            memory_artifacts = []
            for artifact in artifacts:
                links_for_artifact = memory_links_by_artifact.get(artifact["artifact_id"], [])
                assigned = next(
                    (lnk for lnk in links_for_artifact if lnk["link_class"] in ("STRICT", "SUPPORTED")), None,
                )
                ambiguous = [lnk["workflow_id"] for lnk in links_for_artifact if lnk["link_class"] == "AMBIGUOUS"]
                memory_artifacts.append({
                    **artifact,
                    "link_status": assigned["link_class"] if assigned else "AMBIGUOUS" if ambiguous else "UNASSOCIATED",
                    "associated_workflow_id": assigned["workflow_id"] if assigned else None,
                    "ambiguous_candidate_workflows": ambiguous,
                })
            artifacts = memory_artifacts
    else:
        artifacts = all_artifacts
        workflows = all_workflows
        scoped_links = manifest_links

    artifacts_by_id = {str(a["artifact_id"]): a for a in artifacts}
    workflows_by_id = {str(w["workflow_id"]): w for w in workflows}

    # Build stable identities
    art_identities = {
        str(a["artifact_id"]): _artifact_identity(case_path, a, evidence_by_id.get(str(a["artifact_id"])))
        for a in artifacts
    }
    wf_identities = {
        str(w["workflow_id"]): _workflow_identity(case_path, w, evidence_by_id.get(str(w["workflow_id"])))
        for w in workflows
    }

    # Match workflows to GT jobs
    wf_to_gt_jobs: dict[str, list[str]] = {}
    wf_match_by: dict[str, str | None] = {}
    wf_hash_conflict: dict[str, bool] = {}
    wf_match_kind_counts: Counter[str] = Counter()
    for wid, ident in wf_identities.items():
        matched_jobs, matched_by, conflict = _match_workflow_to_jobs(ident, jobs)
        wf_to_gt_jobs[wid] = matched_jobs
        wf_match_by[wid] = matched_by
        wf_hash_conflict[wid] = conflict
        if matched_by is not None:
            wf_match_kind_counts[matched_by] += 1

    pid_to_gt_jobs: dict[str, list[str]] = {}
    for job in jobs:
        for pid in job.prompt_ids:
            pid_to_gt_jobs.setdefault(pid, []).append(job.job_id)

    # Match artifacts to GT outputs / distractors (NEVER using predicted links!)
    art_gt_match: dict[str, tuple[str | None, list[str], str | None]] = {}
    art_match_kind_counts: Counter[str] = Counter()
    for aid, ident in art_identities.items():
        target_kind, matched_ids, matched_by = _match_artifact_to_gt(ident, outputs_all, distractors)
        art_gt_match[aid] = (target_kind, matched_ids, matched_by)
        if matched_by is not None:
            art_match_kind_counts[matched_by] += 1

    # Resolve predicted links & output-level identity per artifact
    predictions = _resolve_predicted_links(
        artifacts,
        scoped_links,
        wf_to_gt_jobs,
        wf_identities,
        pid_to_gt_jobs,
        evidence_by_id,
        scorecard_output_ids,
        active_links_doc,
        view,
    )

    # Classify every artifact's link outcome (wrong vs ambiguous vs correct vs unassociated vs unscored)
    artifact_reports: list[dict[str, Any]] = []
    art_link_verdict: dict[str, str] = {}
    art_link_subreason: dict[str, str] = {}
    art_output_identity_status: dict[str, str | None] = {}
    art_verified_output_ids: dict[str, list[str]] = {}
    art_independent: dict[str, bool] = {}

    link_counts: Counter[str] = Counter()
    output_identity_counts: Counter[str] = Counter()
    wrong_by_class: Counter[str] = Counter()
    strict_gt_known_attributed = 0
    supported_gt_known_attributed = 0
    job_matched_gt_output_artifacts = 0
    disallowed_gt_distractor_artifacts = 0

    for art in artifacts:
        aid = str(art["artifact_id"])
        ident = art_identities[aid]
        pred = predictions[aid]
        target_kind, matched_ids, matched_by = art_gt_match[aid]
        status = pred["status"]
        link_class = pred["link_class"]
        pred_jobs = pred["predicted_gt_job_ids"]

        true_jobs: list[str] = []
        if target_kind == "output":
            true_jobs = sorted({outputs_by_id[oid].job_id for oid in matched_ids})

        verdict = "UNSCORED"
        subreason = "UNMATCHED_UNASSOCIATED"
        out_id_status: str | None = None
        verified_oids: list[str] = []

        if target_kind == "output":
            expected_ambig = (
                len(true_jobs) > 1 or any(outputs_by_id[oid].expected_ambiguous for oid in matched_ids)
            )
            if status in ("STRICT", "SUPPORTED"):
                if link_class == "STRICT":
                    strict_gt_known_attributed += 1
                else:
                    supported_gt_known_attributed += 1

                if pred["conflicting_sources"]:
                    verdict, subreason = "WRONG", "WRONG_CONFLICTING_SOURCES"
                elif pred["invalid_link_rule"]:
                    verdict, subreason = "WRONG", "WRONG_INVALID_ATTRIBUTION_RULE"
                elif expected_ambig:
                    verdict, subreason = "WRONG", "WRONG_OVERCONFIDENT_AMBIGUOUS"
                elif not pred_jobs:
                    verdict, subreason = "WRONG", "WRONG_UNMATCHED_WORKFLOW"
                elif pred["unmapped_targets"]:
                    verdict, subreason = "WRONG", "WRONG_UNMATCHED_WORKFLOW"
                elif len(true_jobs) == 1 and set(pred_jobs) == set(true_jobs):
                    job_matched_gt_output_artifacts += 1
                    matched_outputs = [outputs_by_id[oid] for oid in matched_ids]
                    out_id_status, verified_oids, _details = _evaluate_output_identity(
                        pred, matched_outputs, outputs_by_job.get(true_jobs[0], []),
                    )
                    output_identity_counts[out_id_status] += 1
                    if out_id_status == "MISMATCHED":
                        verdict, subreason = "WRONG", "WRONG_SAME_JOB_OUTPUT"
                    else:
                        verdict, subreason = "CORRECT", f"CORRECT_{link_class}"
                else:
                    verdict, subreason = "WRONG", "WRONG_CROSS_JOB"
            elif status == "AMBIGUOUS":
                verdict = "AMBIGUOUS"
                if pred["output_identity_ambiguous"]:
                    out_id_status = "AMBIGUOUS"
                    output_identity_counts["AMBIGUOUS"] += 1
                    if len(true_jobs) == 1 and set(pred_jobs) == set(true_jobs):
                        subreason = "AMBIGUOUS_EXPECTED" if expected_ambig else "AMBIGUOUS_SAME_JOB_OUTPUT"
                    elif set(true_jobs) & set(pred_jobs):
                        subreason = "AMBIGUOUS_INCLUDES_TRUE_JOB"
                    else:
                        subreason = "AMBIGUOUS_EXCLUDES_TRUE_JOB"
                elif set(true_jobs) <= set(pred_jobs) and expected_ambig and (
                    len(pred_jobs) + len(pred["unmapped_targets"]) >= 2 or len(true_jobs) > 1
                ):
                    subreason = "AMBIGUOUS_EXPECTED"
                elif set(true_jobs) & set(pred_jobs):
                    subreason = "AMBIGUOUS_INCLUDES_TRUE_JOB"
                else:
                    subreason = "AMBIGUOUS_EXCLUDES_TRUE_JOB"
            else:
                verdict, subreason = "UNASSOCIATED", "UNASSOCIATED_GT_OUTPUT"

        elif target_kind == "distractor":
            matched_distractors = [distractors_by_id[did] for did in matched_ids]
            allow_link = all(dist.allow_link_without_tier for dist in matched_distractors)
            scopes = [
                dist.allowed_link_job_ids for dist in matched_distractors if dist.allowed_link_job_ids is not None
            ]
            if scopes and status in ("STRICT", "SUPPORTED"):
                allow_link = (allow_link and bool(pred_jobs) and not pred["unmapped_targets"]
                              and set(pred_jobs) <= frozenset().union(*scopes))
            if not allow_link:
                disallowed_gt_distractor_artifacts += 1
            if status in ("STRICT", "SUPPORTED"):
                if allow_link:
                    # e.g. <16px test PNG carrying a prompt: ComfyTrace keeps its link and leaves it UNTIERED.
                    verdict, subreason = "ALLOWED_UNTIERED_LINK", "DISTRACTOR_ALLOWED_LINK_WITHOUT_TIER"
                else:
                    if link_class == "STRICT":
                        strict_gt_known_attributed += 1
                    else:
                        supported_gt_known_attributed += 1
                    verdict, subreason = "WRONG", "WRONG_DISTRACTOR_LINKED"
            elif status == "AMBIGUOUS":
                verdict, subreason = "AMBIGUOUS", "AMBIGUOUS_DISTRACTOR"
            else:
                verdict, subreason = "CORRECT_UNASSOCIATED", "DISTRACTOR_UNASSOCIATED"
        else:
            # Artifact not listed in withheld ground truth
            if status in ("STRICT", "SUPPORTED"):
                verdict, subreason = "UNSCORED_ATTRIBUTED", "UNMATCHED_ARTIFACT_ATTRIBUTED"
            elif status == "AMBIGUOUS":
                verdict, subreason = "AMBIGUOUS", "AMBIGUOUS_UNMATCHED"
            else:
                verdict, subreason = "UNSCORED", "UNMATCHED_UNASSOCIATED"

        independent, independent_reason = False, "NOT_CORRECT_GT_OUTPUT_LINK"
        if verdict == "CORRECT":
            independent, independent_reason = _independent_extraction_credit(
                aid, pred, true_jobs, scoped_links, workflows_by_id, wf_to_gt_jobs,
            )
        art_independent[aid] = independent

        art_link_verdict[aid] = verdict
        art_link_subreason[aid] = subreason
        art_output_identity_status[aid] = out_id_status
        art_verified_output_ids[aid] = verified_oids
        link_counts[verdict] += 1
        link_counts[subreason] += 1
        if verdict == "WRONG":
            wrong_by_class[link_class or "UNKNOWN"] += 1

        artifact_reports.append({
            "artifact_id": aid,
            "stable_id": ident.stable_id,
            "identity_kind": ident.identity_kind,
            "carve_type": ident.carve_type,
            "completeness": ident.completeness,
            "gt_match_kind": target_kind,
            "gt_matched_ids": matched_ids,
            "gt_matched_by": matched_by,
            "true_gt_job_ids": true_jobs,
            "predicted_status": status,
            "predicted_link_class": link_class,
            "predicted_source": pred["source"],
            "predicted_gt_job_ids": pred_jobs,
            "unmapped_targets": pred["unmapped_targets"],
            "predicted_output_node_ids": pred["predicted_output_node_ids"],
            "predicted_batch_indices": pred["predicted_batch_indices"],
            "predicted_raster_sha256s": pred["predicted_raster_sha256s"],
            "output_identity_status": out_id_status,
            "link_verdict": verdict,
            "link_subreason": subreason,
            "independent_extraction_credit": independent,
            "independent_extraction_reason": independent_reason,
        })

    # Audit declared image-links.json promotions directly
    promotion_checks = _audit_image_link_promotions(
        active_links_doc,
        artifacts_by_id,
        workflows_by_id,
        wf_identities,
        scoped_only=(view == "memory_only"),
    )
    withheld_promotions = {note["promotion"] for note in verification_notes}
    for promotion_check in promotion_checks:
        if withheld_promotions.intersection(promotion_check.get("artifact_ids", [])):
            promotion_check["issues"].append("UNVERIFIED_REPLAY_OR_RASTER")
            promotion_check["valid"] = False

    # Validate every finding's tier against ComfyTrace invariants + link & promotion correctness
    finding_checks: list[dict[str, Any]] = []
    finding_valid_map: dict[str, bool] = {}

    def _has_complete_api_workflow(wids: list[str]) -> bool:
        return any(
            workflows_by_id[w].get("completeness") == "COMPLETE"
            and workflows_by_id[w].get("prompt_format") == "API_GRAPH"
            for w in wids
        )

    def _validate_whole_frame_link(img_ids: list[str], wf_ids: list[str], issues: list[str]) -> None:
        il_pairs = [
            (i, predictions[i]["image_link_entry"])
            for i in img_ids
            if predictions[i]["image_link_entry"] is not None
        ]
        if not il_pairs:
            issues.append("E2_MISSING_IMAGE_LINK_ENTRY")
            return
        valid_pairs = [(i, e) for i, e in il_pairs if _valid_whole_frame_entry(artifacts_by_id[i], e)]
        if not valid_pairs:
            issues.append("E2_CLOSE_OR_LOSSY_IMAGE_LINK")
            return
        wf_pids = {
            pid
            for w in wf_ids
            if workflows_by_id[w].get("completeness") == "COMPLETE"
            and workflows_by_id[w].get("prompt_format") == "API_GRAPH"
            for pid in wf_identities[w].prompt_ids
        }
        if not any(e.get("prompt_id") in wf_pids for _i, e in valid_pairs):
            issues.append("E2_PROMOTION_PROMPT_MISMATCH")

    for finding in active_findings:
        fid = str(finding["finding_id"])
        tier = str(finding["tier"])
        rationale = str(finding.get("rationale") or "")
        is_other_group = rationale.startswith(OTHER_IMAGES)
        member_ids = [str(x) for x in finding.get("artifact_ids", [])]
        img_ids = [x for x in member_ids if x in artifacts_by_id]
        wf_ids = [x for x in member_ids if x in workflows_by_id]
        issues: list[str] = []

        if is_other_group:
            if tier != "UNTIERED":
                issues.append("OTHER_IMAGES_MUST_BE_UNTIERED")
        elif tier == "E1":
            regen_id = finding.get("regeneration")
            if not regen_id:
                issues.append("E1_WITHOUT_REGENERATION")
            if finding.get("base_tier") != "E2":
                issues.append("E1_BASE_TIER_NOT_E2")
            if not any(
                artifacts_by_id[i].get("completeness") == "COMPLETE" and not _is_below_output_size(artifacts_by_id[i])
                for i in img_ids
            ):
                issues.append("E1_WITHOUT_COMPLETE_IMAGE")
            if not _has_complete_api_workflow(wf_ids):
                issues.append("E1_WITHOUT_COMPLETE_WORKFLOW")
            if finding.get("image_link") == "REGENERATION_WHOLE_FRAME_MATCH":
                _validate_whole_frame_link(img_ids, wf_ids, issues)
            elif finding.get("image_link") == "REGENERATION_CLOSE_MATCH" and finding.get("extraction_tier") != "E2":
                issues.append("E2_CLOSE_OR_LOSSY_IMAGE_LINK")
            if regen_id and str(regen_id) in regen_by_id:
                regen = regen_by_id[str(regen_id)]
                ev_block = regen.get("evaluation") or {}
                target_id = ev_block.get("target_artifact_id")
                regen_wf = regen.get("workflow_id")
                if not isinstance(target_id, str) or target_id not in img_ids:
                    issues.append("E1_INVALID_REGENERATION_TARGET")
                else:
                    target_art = artifacts_by_id[target_id]
                    if (
                        target_art.get("completeness") != "COMPLETE"
                        or _is_below_output_size(target_art)
                        or art_gt_match[target_id][0] == "distractor"
                        or art_link_verdict.get(target_id) in ("WRONG", "AMBIGUOUS", "UNASSOCIATED")
                    ):
                        issues.append("E1_INVALID_REGENERATION_TARGET")
                    compared_meta = ev_block.get("compared_image")
                    if isinstance(compared_meta, Mapping) and art_gt_match[target_id][0] == "output":
                        cn, cb, cr = _extract_output_identity_from_mapping(compared_meta, include_pixel_sha_key=True)
                        if cn or cb or cr:
                            t_outs = [outputs_by_id[oid] for oid in art_gt_match[target_id][1]]
                            t_jobs = sorted({o.job_id for o in t_outs})
                            if len(t_jobs) == 1:
                                c_status, _c_oids, _c_det = _evaluate_output_identity(
                                    {
                                        "predicted_output_node_ids": sorted(cn),
                                        "predicted_batch_indices": sorted(cb),
                                        "predicted_raster_sha256s": sorted(cr),
                                        "output_identity_ambiguous": False,
                                    },
                                    t_outs,
                                    outputs_by_job.get(t_jobs[0], []),
                                )
                                if c_status == "MISMATCHED":
                                    issues.append("E1_REGENERATION_OUTPUT_MISMATCH")
                if not isinstance(regen_wf, str) or regen_wf not in wf_ids:
                    issues.append("E1_REGENERATION_WORKFLOW_MISMATCH")
        elif tier == "E2":
            if not any(
                artifacts_by_id[i].get("completeness") == "COMPLETE" and not _is_below_output_size(artifacts_by_id[i])
                for i in img_ids
            ):
                issues.append("E2_WITHOUT_COMPLETE_IMAGE")
            if not _has_complete_api_workflow(wf_ids):
                issues.append("E2_WITHOUT_COMPLETE_WORKFLOW")
            if finding.get("image_link") == "REGENERATION_WHOLE_FRAME_MATCH":
                _validate_whole_frame_link(img_ids, wf_ids, issues)
            elif finding.get("image_link") == "REGENERATION_CLOSE_MATCH" and finding.get("extraction_tier") != "E2":
                issues.append("E2_CLOSE_OR_LOSSY_IMAGE_LINK")
        elif tier == "E3":
            if not any(not _is_below_output_size(artifacts_by_id[i]) for i in img_ids):
                issues.append("E3_WITHOUT_ELIGIBLE_IMAGE")
            if not wf_ids:
                issues.append("E3_WITHOUT_WORKFLOW")
            if finding.get("image_link") == "REGENERATION_CLOSE_MATCH":
                il_pairs = [
                    (i, predictions[i]["image_link_entry"])
                    for i in img_ids
                    if predictions[i]["image_link_entry"] is not None
                ]
                valid_close = [(i, e) for i, e in il_pairs if _valid_close_match_entry(e)]
                wf_api_pids = {
                    pid
                    for w in wf_ids
                    if workflows_by_id[w].get("prompt_format") == "API_GRAPH"
                    for pid in wf_identities[w].prompt_ids
                }
                if not valid_close or not any(e.get("prompt_id") in wf_api_pids for _i, e in valid_close):
                    issues.append("E3_INVALID_CLOSE_MATCH_PROMOTION")
        elif tier == "E4":
            regen_id = finding.get("regeneration")
            if not regen_id:
                issues.append("E4_WITHOUT_REGENERATION")
            if finding.get("base_tier") not in (None, "E5"):
                issues.append("E4_BASE_TIER_NOT_E5")
            if img_ids:
                issues.append("E4_CONTAINS_IMAGES")
            if not _has_complete_api_workflow(wf_ids):
                issues.append("E4_WITHOUT_COMPLETE_WORKFLOW")
            if regen_id and str(regen_id) in regen_by_id:
                regen = regen_by_id[str(regen_id)]
                ev_block = regen.get("evaluation") or {}
                if (
                    regen.get("workflow_id") not in wf_ids
                    or ev_block.get("promoted_finding_tier") != "E4"
                    or ev_block.get("evaluation_mode") != "NO_MEMORY_TARGET"
                ):
                    issues.append("E4_INVALID_REGENERATION")
        elif tier == "E5":
            if img_ids:
                issues.append("E5_CONTAINS_IMAGES")
            if not _has_complete_api_workflow(wf_ids):
                issues.append("E5_WITHOUT_COMPLETE_WORKFLOW")
        elif tier == "E6":
            if img_ids:
                issues.append("E6_CONTAINS_IMAGES")
            if not wf_ids:
                issues.append("E6_WITHOUT_WORKFLOW")
        elif tier != "UNTIERED":
            issues.append(f"UNKNOWN_TIER:{tier}")

        if tier in ("E1", "E2", "E3"):
            if any(_is_below_output_size(artifacts_by_id[i]) for i in img_ids):
                issues.append("BELOW_MIN_OUTPUT_SIZE_TIERED")
            if any(art_link_verdict.get(i) == "WRONG" for i in img_ids):
                issues.append("WRONG_LINK_TIER")
            if any(art_link_verdict.get(i) == "AMBIGUOUS" for i in img_ids):
                issues.append("AMBIGUOUS_LINK_TIER")
            if any(predictions[i]["status"] == "UNASSOCIATED" for i in img_ids):
                issues.append("UNASSOCIATED_IMAGE_TIERED")
            if any(art_gt_match[i][0] == "distractor" for i in img_ids):
                issues.append("DISTRACTOR_TIERED")

            # Verify finding's GT-matched images and GT-matched workflows belong to the same GT job
            gt_img_job_sets = [
                {outputs_by_id[oid].job_id for oid in art_gt_match[i][1]}
                for i in img_ids
                if art_gt_match[i][0] == "output"
            ]
            if gt_img_job_sets:
                all_wf_gt_jobs = {jid for w in wf_ids for jid in wf_to_gt_jobs.get(w, [])}
                pid_wf_gt_jobs = {
                    jid for w in wf_ids if wf_match_by.get(w) == "prompt_id" for jid in wf_to_gt_jobs.get(w, [])
                }
                union_img_jobs = set().union(*gt_img_job_sets)
                if not all_wf_gt_jobs:
                    issues.append("FINDING_JOB_MISMATCH")
                elif any(not (jset & all_wf_gt_jobs) for jset in gt_img_job_sets):
                    issues.append("FINDING_JOB_MISMATCH")
                elif pid_wf_gt_jobs and not (pid_wf_gt_jobs <= union_img_jobs):
                    issues.append("FINDING_JOB_MISMATCH")

            # Verify a single finding does not conflate two distinct GT outputs
            gt_img_output_sets = [
                set(art_verified_output_ids[i] or art_gt_match[i][1])
                for i in img_ids
                if art_gt_match[i][0] == "output"
            ]
            if len(gt_img_output_sets) > 1:
                for idx_a, oset_a in enumerate(gt_img_output_sets):
                    if any(not (oset_a & oset_b) for oset_b in gt_img_output_sets[idx_a + 1 :]):
                        issues.append("MIXED_GT_OUTPUTS_IN_FINDING")
                        break

        deduped_issues = list(dict.fromkeys(issues))
        valid = not deduped_issues
        finding_valid_map[fid] = valid
        finding_checks.append({
            "finding_id": fid,
            "tier": tier,
            "extraction_tier": finding.get("extraction_tier"),
            "image_link": finding.get("image_link"),
            "regeneration": finding.get("regeneration"),
            "image_count": len(img_ids),
            "workflow_count": len(wf_ids),
            "valid": valid,
            "issues": deduped_issues,
        })

    # Map artifacts and workflows to their findings
    findings_by_member: dict[str, list[dict[str, Any]]] = {}
    for finding in active_findings:
        for member_id in finding.get("artifact_ids", []):
            findings_by_member.setdefault(str(member_id), []).append(finding)

    # Evaluate GT outputs
    output_reports: list[dict[str, Any]] = []
    tier_expectation_results: list[dict[str, Any]] = []

    for out in outputs_all:
        possible_aids = [
            aid for aid, (kind, ids, _by) in art_gt_match.items()
            if kind == "output" and out.output_id in ids
        ]
        matched_aids = [
            aid for aid in possible_aids
            if len(art_gt_match[aid][1]) == 1 or art_verified_output_ids[aid] == [out.output_id]
        ]
        complete_aids = [aid for aid in matched_aids if artifacts_by_id[aid].get("completeness") == "COMPLETE"]
        # When known output identity narrowed a multi-match artifact to specific output_id(s), only credit those.
        correct_aids = [
            aid for aid in matched_aids
            if art_link_verdict[aid] == "CORRECT"
            and (not art_verified_output_ids[aid] or out.output_id in art_verified_output_ids[aid])
        ]
        independent_aids = [aid for aid in correct_aids if art_independent.get(aid)]
        wrong_aids = [aid for aid in matched_aids if art_link_verdict[aid] == "WRONG"]
        ambig_aids = [aid for aid in matched_aids if art_link_verdict[aid] == "AMBIGUOUS"]
        unassoc_aids = [aid for aid in matched_aids if art_link_verdict[aid] == "UNASSOCIATED"]

        # Valid achieved tier comes ONLY from policy-valid findings with a valid link to this output;
        # observed_tier captures any assigned tier (including invalid/over-tiered) so OVER_TIERED is never missed.
        valid_tiers: list[str] = []
        observed_tiers: list[str] = []
        wrongly_linked_findings: list[str] = []
        for aid in matched_aids:
            for finding in findings_by_member.get(aid, []):
                fid = str(finding["finding_id"])
                ftier = str(finding["tier"])
                observed_tiers.append(ftier)
                if art_link_verdict[aid] == "WRONG" and ftier in ("E1", "E2", "E3"):
                    wrongly_linked_findings.append(fid)
                    continue
                if not finding_valid_map.get(fid, False):
                    continue
                if ftier == "UNTIERED":
                    valid_tiers.append("UNTIERED")
                elif aid in correct_aids:
                    if ftier == "E1":
                        regen_id = finding.get("regeneration")
                        regen = regen_by_id.get(str(regen_id)) if regen_id else None
                        target_id = (regen.get("evaluation") or {}).get("target_artifact_id") if regen else None
                        if target_id is None or target_id in matched_aids:
                            valid_tiers.append("E1")
                    else:
                        valid_tiers.append(ftier)

        achieved_tier = max(valid_tiers, key=lambda t: TIER_RANK[t]) if valid_tiers else None
        observed_tier = max(observed_tiers, key=lambda t: TIER_RANK[t]) if observed_tiers else None
        tier_eval = _evaluate_tier_expectation(
            achieved_tier,
            observed_tier=observed_tier,
            expected_tier=out.expected_tier,
            allowed_tiers=out.allowed_tiers,
            min_expected_tier=out.min_expected_tier,
            max_allowed_tier=out.max_allowed_tier,
        )
        if tier_eval["has_expectation"]:
            tier_expectation_results.append({"scope": "output", "id": out.output_id, **tier_eval})

        output_reports.append({
            "output_id": out.output_id,
            "job_id": out.job_id,
            "output_node_id": out.output_node_id,
            "batch_index": out.batch_index,
            "recovered": bool(matched_aids),
            "complete_recovered": bool(complete_aids),
            "correctly_linked": bool(correct_aids),
            "matched_artifact_count": len(matched_aids),
            "unresolved_artifact_ids": [aid for aid in possible_aids if aid not in matched_aids
                                        and len(art_verified_output_ids[aid]) != 1],
            "matched_stable_ids": [art_identities[a].stable_id for a in matched_aids],
            "matched_by": [art_gt_match[a][2] for a in matched_aids],
            "output_identity_statuses": [art_output_identity_status[a] for a in matched_aids],
            "correct_link_count": len(correct_aids),
            "independently_corroborated": bool(independent_aids),
            "independent_extraction_artifact_ids": sorted(independent_aids),
            "wrong_link_count": len(wrong_aids),
            "ambiguous_link_count": len(ambig_aids),
            "unassociated_count": len(unassoc_aids),
            "wrongly_linked_findings": sorted(set(wrongly_linked_findings)),
            "achieved_tier": achieved_tier,
            "observed_tier": observed_tier,
            "tier_check": tier_eval,
        })

    # Evaluate GT jobs
    job_reports: list[dict[str, Any]] = []
    outputs_report_by_job: dict[str, list[dict[str, Any]]] = {}
    for rep in output_reports:
        outputs_report_by_job.setdefault(rep["job_id"], []).append(rep)

    for job in jobs:
        matched_wids = [wid for wid, jids in wf_to_gt_jobs.items() if job.job_id in jids]
        complete_api_wids = [
            wid for wid in matched_wids
            if workflows_by_id[wid].get("prompt_format") == "API_GRAPH"
            and workflows_by_id[wid].get("completeness") == "COMPLETE"
            and not wf_hash_conflict[wid]
        ]
        job_outputs = outputs_report_by_job.get(job.job_id, [])
        job_gt_output_ids = {o.output_id for o in job.outputs}

        valid_job_tiers: list[str] = []
        observed_job_tiers: list[str] = []
        for wid in matched_wids:
            for finding in findings_by_member.get(wid, []):
                fid = str(finding["finding_id"])
                ftier = str(finding["tier"])
                observed_job_tiers.append(ftier)
                if not finding_valid_map.get(fid, False):
                    continue
                if ftier in ("E1", "E2", "E3"):
                    f_imgs = [x for x in finding.get("artifact_ids", []) if x in artifacts_by_id]
                    has_verified_job_output = any(
                        art_link_verdict.get(i) == "CORRECT"
                        and art_gt_match[i][0] == "output"
                        and bool(set(art_gt_match[i][1]) & job_gt_output_ids)
                        for i in f_imgs
                    )
                    if not has_verified_job_output:
                        continue
                    valid_job_tiers.append(ftier)
                elif ftier in ("E4", "E5", "E6"):
                    if not wf_hash_conflict[wid]:
                        valid_job_tiers.append(ftier)
        for orep in job_outputs:
            if orep["achieved_tier"] is not None:
                valid_job_tiers.append(orep["achieved_tier"])
            if orep["observed_tier"] is not None:
                observed_job_tiers.append(orep["observed_tier"])

        achieved_job_tier = max(valid_job_tiers, key=lambda t: TIER_RANK[t]) if valid_job_tiers else None
        observed_job_tier = max(observed_job_tiers, key=lambda t: TIER_RANK[t]) if observed_job_tiers else None
        job_tier_eval = _evaluate_tier_expectation(
            achieved_job_tier,
            observed_tier=observed_job_tier,
            expected_tier=job.expected_tier,
            allowed_tiers=job.allowed_tiers,
            min_expected_tier=job.min_expected_tier,
            max_allowed_tier=job.max_allowed_tier,
        )
        if job_tier_eval["has_expectation"]:
            tier_expectation_results.append({"scope": "job", "id": job.job_id, **job_tier_eval})

        job_reports.append({
            "job_id": job.job_id,
            "recovered": bool(matched_wids),
            "complete_api_recovered": bool(complete_api_wids),
            "has_correctly_linked_output": any(o["correctly_linked"] for o in job_outputs),
            "matched_workflow_count": len(matched_wids),
            "matched_workflow_stable_ids": [wf_identities[w].stable_id for w in matched_wids],
            "matched_by": [wf_match_by[w] for w in matched_wids],
            "prompt_hash_conflicts": [w for w in matched_wids if wf_hash_conflict[w]],
            "output_count": len(job_outputs),
            "outputs_recovered": sum(1 for o in job_outputs if o["recovered"]),
            "outputs_correctly_linked": sum(1 for o in job_outputs if o["correctly_linked"]),
            "achieved_tier": achieved_job_tier,
            "observed_tier": observed_job_tier,
            "tier_check": job_tier_eval,
        })

    # Evaluate GT distractors
    distractor_reports: list[dict[str, Any]] = []
    for dist in distractors:
        matched_aids = [
            aid for aid, (kind, ids, _by) in art_gt_match.items()
            if kind == "distractor" and dist.distractor_id in ids
        ]
        dist_observed_tiers = [
            str(finding["tier"])
            for aid in matched_aids
            for finding in findings_by_member.get(aid, [])
        ]
        dist_valid_tiers = [
            str(finding["tier"])
            for aid in matched_aids
            for finding in findings_by_member.get(aid, [])
            if finding_valid_map.get(str(finding["finding_id"]), False)
        ]
        achieved_dist_tier = max(dist_valid_tiers, key=lambda t: TIER_RANK[t]) if dist_valid_tiers else None
        observed_dist_tier = max(dist_observed_tiers, key=lambda t: TIER_RANK[t]) if dist_observed_tiers else None
        dist_tier_eval = _evaluate_tier_expectation(
            achieved_dist_tier,
            observed_tier=observed_dist_tier,
            expected_tier=dist.expected_tier,
            allowed_tiers=dist.allowed_tiers,
            min_expected_tier=dist.min_expected_tier,
            max_allowed_tier=dist.max_allowed_tier,
        )
        if dist_tier_eval["has_expectation"]:
            tier_expectation_results.append({"scope": "distractor", "id": dist.distractor_id, **dist_tier_eval})
        wrongly_linked = [aid for aid in matched_aids if art_link_verdict[aid] == "WRONG"]
        distractor_reports.append({
            "distractor_id": dist.distractor_id,
            "reason": dist.reason,
            "recovered": bool(matched_aids),
            "matched_artifact_count": len(matched_aids),
            "matched_stable_ids": [art_identities[a].stable_id for a in matched_aids],
            "wrongly_linked_count": len(wrongly_linked),
            "achieved_tier": achieved_dist_tier,
            "observed_tier": observed_dist_tier,
            "tier_check": dist_tier_eval,
        })

    # Compute explicit, honest denominators and summary metrics
    total_gt_jobs = len(jobs)
    recovered_gt_jobs = sum(1 for j in job_reports if j["recovered"])
    complete_api_gt_jobs = sum(1 for j in job_reports if j["complete_api_recovered"])
    jobs_with_linked_output = sum(1 for j in job_reports if j["has_correctly_linked_output"])

    total_gt_outputs = len(outputs_all)
    recovered_gt_outputs = sum(1 for o in output_reports if o["recovered"])
    complete_gt_outputs = sum(1 for o in output_reports if o["complete_recovered"])
    correctly_linked_gt_outputs = sum(1 for o in output_reports if o["correctly_linked"])
    independent_gt_outputs = sum(1 for o in output_reports if o["independently_corroborated"])

    matched_gt_output_artifacts = sum(1 for r in artifact_reports if r["gt_match_kind"] == "output")
    matched_gt_distractor_artifacts = sum(1 for r in artifact_reports if r["gt_match_kind"] == "distractor")
    unmatched_artifacts = sum(1 for r in artifact_reports if r["gt_match_kind"] is None)

    correct_links = link_counts["CORRECT"]
    wrong_links = link_counts["WRONG"]
    wrong_strict_links = wrong_by_class["STRICT"]
    wrong_supported_links = wrong_links - wrong_strict_links
    wrong_on_gt_outputs = sum(
        1 for r in artifact_reports if r["gt_match_kind"] == "output" and r["link_verdict"] == "WRONG"
    )
    wrong_distractor_links = link_counts["WRONG_DISTRACTOR_LINKED"]
    ambiguous_on_gt_outputs = sum(
        1 for r in artifact_reports if r["gt_match_kind"] == "output" and r["link_verdict"] == "AMBIGUOUS"
    )
    unassociated_on_gt_outputs = link_counts["UNASSOCIATED"]
    attributed_gt_known = correct_links + wrong_links
    unmatched_attributed = link_counts["UNSCORED_ATTRIBUTED"]
    total_attributed = attributed_gt_known + unmatched_attributed

    out_id_matched = output_identity_counts["MATCHED"]
    out_id_mismatched = output_identity_counts["MISMATCHED"]
    out_id_checked = out_id_matched + out_id_mismatched
    out_id_ambiguous = output_identity_counts["AMBIGUOUS"]
    out_id_unknown_pred = output_identity_counts["UNKNOWN_PREDICTED"]
    out_id_unknown_gt = output_identity_counts["UNKNOWN_GROUNDTRUTH"]

    total_gt_tier_checks = len(tier_expectation_results)
    gt_tier_pass = sum(1 for t in tier_expectation_results if t["status"] == "PASS")
    gt_tier_upper_denom = sum(1 for t in tier_expectation_results if t["has_upper_bound"])
    gt_tier_over = sum(1 for t in tier_expectation_results if t["status"] == "OVER_TIERED")
    gt_tier_lower_denom = sum(1 for t in tier_expectation_results if t["has_lower_bound"])
    gt_tier_under = sum(1 for t in tier_expectation_results if t["status"] == "UNDER_TIERED")
    gt_tier_mismatch = sum(1 for t in tier_expectation_results if t["status"] == "MISMATCH")

    total_findings = len(finding_checks)
    valid_findings = sum(1 for f in finding_checks if f["valid"])

    total_promotions = len(promotion_checks)
    valid_promotions = sum(1 for p in promotion_checks if p["valid"])

    total_art_gt_matches = sum(art_match_kind_counts.values())
    stable_art_gt_matches = total_art_gt_matches - art_match_kind_counts["transient_artifact_id"]
    total_wf_gt_matches = sum(wf_match_kind_counts.values())
    stable_wf_gt_matches = total_wf_gt_matches - wf_match_kind_counts["transient_workflow_id"]

    metrics = {
        "job_recovery_rate": _ratio(
            recovered_gt_jobs, total_gt_jobs, "recovered_gt_jobs / total_gt_jobs",
        ),
        "complete_api_job_recovery_rate": _ratio(
            complete_api_gt_jobs, total_gt_jobs, "complete_api_recovered_gt_jobs / total_gt_jobs",
        ),
        "job_linked_output_recovery_rate": _ratio(
            jobs_with_linked_output, total_gt_jobs, "gt_jobs_with_correctly_linked_output / total_gt_jobs",
        ),
        "output_recovery_rate": _ratio(
            recovered_gt_outputs, total_gt_outputs, "recovered_gt_outputs / total_gt_outputs",
        ),
        "output_complete_recovery_rate": _ratio(
            complete_gt_outputs, total_gt_outputs, "complete_recovered_gt_outputs / total_gt_outputs",
        ),
        "output_correctly_linked_rate": _ratio(
            correctly_linked_gt_outputs, total_gt_outputs, "correctly_linked_gt_outputs / total_gt_outputs",
        ),
        "output_linked_given_recovered_rate": _ratio(
            correctly_linked_gt_outputs, recovered_gt_outputs, "correctly_linked_gt_outputs / recovered_gt_outputs",
        ),
        "output_independently_corroborated_rate": _ratio(
            independent_gt_outputs, total_gt_outputs,
            "gt_outputs_with_a_CORRECT_strict_extraction_link_from_separately_recovered_evidence / "
            "total_gt_outputs (view-scoped numerator: memory_only counts only memory-backed artifacts, workflows "
            "and memory_only_correlation_links, but the denominator stays every catalogued output; EMBEDDED_PROMPT, "
            "INPUT_REVISION_UNVERIFIED, own-embedded-text workflows, absent keys, AMBIGUOUS/SUPPORTED links and "
            "regeneration-only matches earn no credit)",
        ),
        "output_independently_corroborated_given_correct_rate": _ratio(
            independent_gt_outputs, correctly_linked_gt_outputs,
            "independently_corroborated_gt_outputs / correctly_linked_gt_outputs (share of legacy-correct outputs "
            "that are not merely self-described)",
        ),
        "link_precision": _ratio(
            correct_links, attributed_gt_known,
            "correct_links / attributed_gt_known_artifacts (excludes AMBIGUOUS abstentions and uncatalogued)",
        ),
        "wrong_link_rate": _ratio(
            wrong_links, attributed_gt_known,
            "wrong_links / attributed_gt_known_artifacts (excludes AMBIGUOUS abstentions and uncatalogued)",
        ),
        "wrong_strict_link_rate": _ratio(
            wrong_strict_links, strict_gt_known_attributed,
            "wrong_strict_links / strict_attributed_gt_known_artifacts",
        ),
        "attributed_gt_coverage_rate": _ratio(
            attributed_gt_known, total_attributed,
            "gt_known_attributed_artifacts / total_attributed_artifacts (excludes allowed_untiered and AMBIGUOUS)",
        ),
        "artifact_correct_link_rate": _ratio(
            correct_links, matched_gt_output_artifacts,
            "correct_links / matched_gt_output_artifacts",
        ),
        "artifact_wrong_link_rate": _ratio(
            wrong_on_gt_outputs, matched_gt_output_artifacts,
            "wrong_links_on_gt_outputs / matched_gt_output_artifacts",
        ),
        "artifact_ambiguous_rate": _ratio(
            ambiguous_on_gt_outputs, matched_gt_output_artifacts,
            "ambiguous_links_on_gt_outputs / matched_gt_output_artifacts",
        ),
        "artifact_unassociated_rate": _ratio(
            unassociated_on_gt_outputs, matched_gt_output_artifacts,
            "unassociated_gt_output_artifacts / matched_gt_output_artifacts",
        ),
        "distractor_false_link_rate": _ratio(
            wrong_distractor_links, disallowed_gt_distractor_artifacts,
            "wrong_distractor_links / matched_disallowed_gt_distractor_artifacts (excludes allow_link_without_tier)",
        ),
        "output_identity_known_rate": _ratio(
            out_id_checked, job_matched_gt_output_artifacts,
            "output_identities_checked_against_gt / job_matched_gt_output_artifacts",
        ),
        "output_identity_accuracy": _ratio(
            out_id_matched, out_id_checked,
            "output_identities_matched / output_identities_checked_against_gt (excludes unknown identities)",
        ),
        "gt_tier_pass_rate": _ratio(
            gt_tier_pass, total_gt_tier_checks,
            "gt_items_meeting_tier_expectation / gt_items_with_tier_expectation",
        ),
        "gt_tier_over_tiered_rate": _ratio(
            gt_tier_over, gt_tier_upper_denom,
            "gt_items_over_tiered / gt_items_with_tier_upper_bound",
        ),
        "gt_tier_under_tiered_rate": _ratio(
            gt_tier_under, gt_tier_lower_denom,
            "gt_items_under_tiered / gt_items_with_tier_lower_bound",
        ),
        "finding_policy_valid_rate": _ratio(
            valid_findings, total_findings,
            "policy_valid_findings / total_evaluated_findings",
        ),
        "promotion_valid_rate": _ratio(
            valid_promotions, total_promotions,
            "valid_image_link_promotions / total_evaluated_image_link_promotions",
        ),
        "stable_gt_artifact_match_rate": _ratio(
            stable_art_gt_matches, total_art_gt_matches,
            "gt_artifact_matches_via_stable_identity / total_gt_artifact_matches",
        ),
        "stable_gt_workflow_match_rate": _ratio(
            stable_wf_gt_matches, total_wf_gt_matches,
            "gt_workflow_matches_via_stable_identity / total_gt_workflow_matches",
        ),
    }

    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "case": {
            "case_dir": str(case_path),
            "case_id": metadata.get("case_id"),
            "view": view,
            "dump_sha256": case_dump_sha,
            "manifest_sha256": manifest_sha,
            "exit_code": metadata.get("exit_code"),
            "carving_mode": metadata.get("carving_mode"),
            "degradation_reason": metadata.get("degradation_reason"),
            "image_links_applied": active_links_doc is not None,
            "image_links_ignored_mismatch": (
                raw_image_links is not None and valid_image_links is None and view in ("effective", "memory_only")
            ),
            "regeneration_records_found": len(regenerations),
            "sidecar_notes": applied_notes,
        },
        "groundtruth": {
            "case_id": gt_doc.get("case_id"),
            "dump_sha256": gt_dump_sha,
            "total_jobs": total_gt_jobs,
            "total_outputs": total_gt_outputs,
            "total_distractors": len(distractors),
        },
        "counts": {
            "case_artifacts_evaluated": len(artifacts),
            "case_workflows_evaluated": len(workflows),
            "matched_gt_output_artifacts": matched_gt_output_artifacts,
            "matched_gt_distractor_artifacts": matched_gt_distractor_artifacts,
            "matched_disallowed_gt_distractor_artifacts": disallowed_gt_distractor_artifacts,
            "unmatched_case_artifacts": unmatched_artifacts,
            "unmatched_attributed_artifacts": unmatched_attributed,
        },
        "links": {
            "correct": correct_links,
            "wrong": wrong_links,
            "wrong_strict": wrong_strict_links,
            "wrong_supported": wrong_supported_links,
            "wrong_breakdown": {
                "cross_job": link_counts["WRONG_CROSS_JOB"],
                "same_job_output": link_counts["WRONG_SAME_JOB_OUTPUT"],
                "overconfident_ambiguous": link_counts["WRONG_OVERCONFIDENT_AMBIGUOUS"],
                "invalid_attribution_rule": link_counts["WRONG_INVALID_ATTRIBUTION_RULE"],
                "distractor_linked": wrong_distractor_links,
                "unmatched_workflow": link_counts["WRONG_UNMATCHED_WORKFLOW"],
                "conflicting_sources": link_counts["WRONG_CONFLICTING_SOURCES"],
            },
            "ambiguous": link_counts["AMBIGUOUS"],
            "ambiguous_breakdown": {
                "expected": link_counts["AMBIGUOUS_EXPECTED"],
                "includes_true_job": link_counts["AMBIGUOUS_INCLUDES_TRUE_JOB"],
                "excludes_true_job": link_counts["AMBIGUOUS_EXCLUDES_TRUE_JOB"],
                "same_job_output": link_counts["AMBIGUOUS_SAME_JOB_OUTPUT"],
                "distractor": link_counts["AMBIGUOUS_DISTRACTOR"],
                "unmatched": link_counts["AMBIGUOUS_UNMATCHED"],
            },
            "unassociated_gt_outputs": unassociated_on_gt_outputs,
            "allowed_untiered_distractor_links": link_counts["ALLOWED_UNTIERED_LINK"],
            "unmatched_attributed_artifacts": unmatched_attributed,
        },
        "output_identity": {
            "job_matched_artifacts": job_matched_gt_output_artifacts,
            "matched": out_id_matched,
            "mismatched": out_id_mismatched,
            "ambiguous": out_id_ambiguous,
            "unknown_predicted": out_id_unknown_pred,
            "unknown_groundtruth": out_id_unknown_gt,
        },
        "tiers": {
            "gt_expectations_checked": total_gt_tier_checks,
            "pass": gt_tier_pass,
            "over_tiered": gt_tier_over,
            "under_tiered": gt_tier_under,
            "mismatch": gt_tier_mismatch,
            "policy_valid_findings": valid_findings,
            "policy_invalid_findings": total_findings - valid_findings,
            "gt_checks": tier_expectation_results,
        },
        "promotions": {
            "total": total_promotions,
            "valid": valid_promotions,
            "invalid": total_promotions - valid_promotions,
            "checks": promotion_checks,
        },
        "identity_stability": {
            "artifact_matches_by_kind": dict(sorted(art_match_kind_counts.items())),
            "workflow_matches_by_kind": dict(sorted(wf_match_kind_counts.items())),
            "case_artifact_identity_kinds": dict(
                sorted(Counter(i.identity_kind for i in art_identities.values()).items())
            ),
            "case_workflow_identity_kinds": dict(
                sorted(Counter(i.identity_kind for i in wf_identities.values()).items())
            ),
        },
        "metrics": metrics,
        "jobs": job_reports,
        "outputs": output_reports,
        "distractors": distractor_reports,
        "artifacts": artifact_reports,
        "finding_checks": finding_checks,
    }


def _fmt_ratio(m: Mapping[str, Any]) -> str:
    num, den, rate = m["numerator"], m["denominator"], m["rate"]
    pct = f"{rate * 100:.1f}%" if rate is not None else "n/a"
    return f"{num}/{den} ({pct})"


def format_summary(report: Mapping[str, Any]) -> str:
    """Formats a concise human-readable summary with explicit denominators."""
    case = report["case"]
    m = report["metrics"]
    links = report["links"]
    out_id = report["output_identity"]
    tiers = report["tiers"]
    lines = [
        f"ComfyTrace post-extraction evaluation ({case['view']} view)",
        f"  Case: {case['case_id']} (exit {case['exit_code']}, mode {case['carving_mode']})",
        f"  Jobs recovered:           {_fmt_ratio(m['job_recovery_rate'])} "
        f"(complete API: {_fmt_ratio(m['complete_api_job_recovery_rate'])})",
        f"  Outputs recovered:        {_fmt_ratio(m['output_recovery_rate'])} "
        f"(correctly linked: {_fmt_ratio(m['output_correctly_linked_rate'])})",
        f"  Link precision:           {_fmt_ratio(m['link_precision'])} "
        f"(GT coverage: {_fmt_ratio(m['attributed_gt_coverage_rate'])}, "
        f"uncatalogued attributed: {links['unmatched_attributed_artifacts']})",
        f"  Wrong links:              {links['wrong']} (STRICT: {links['wrong_strict']}, "
        f"SUPPORTED/other: {links['wrong_supported']}, same-job output: {links['wrong_breakdown']['same_job_output']})",
        f"  Output identity:          matched {_fmt_ratio(m['output_identity_accuracy'])} "
        f"(known: {_fmt_ratio(m['output_identity_known_rate'])}, "
        f"unknown pred: {out_id['unknown_predicted']}, unknown GT: {out_id['unknown_groundtruth']})",
        f"  Ambiguous abstentions:    {links['ambiguous']} "
        f"(expected: {links['ambiguous_breakdown']['expected']}, "
        f"includes truth: {links['ambiguous_breakdown']['includes_true_job']}, "
        f"excludes truth: {links['ambiguous_breakdown']['excludes_true_job']})",
        f"  Tier expectations passed: {_fmt_ratio(m['gt_tier_pass_rate'])} "
        f"(over-tiered: {tiers['over_tiered']}, under-tiered: {tiers['under_tiered']})",
        f"  Finding policy valid:     {_fmt_ratio(m['finding_policy_valid_rate'])} "
        f"(promotions valid: {_fmt_ratio(m['promotion_valid_rate'])})",
        f"  Stable identity matches:  artifacts {_fmt_ratio(m['stable_gt_artifact_match_rate'])}, "
        f"workflows {_fmt_ratio(m['stable_gt_workflow_match_rate'])}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="comfytrace-eval-scorer",
        description="Post-extraction scorer comparing a ComfyTrace case directory against withheld ground truth.",
    )
    parser.add_argument("--case-dir", required=True, type=Path, help="existing ComfyTrace case directory (read-only)")
    parser.add_argument("--groundtruth", required=True, type=Path, help="withheld job/output ground-truth JSON")
    parser.add_argument(
        "--view",
        choices=("effective", "extraction", "memory_only"),
        default="effective",
        help="findings view to evaluate (default: effective)",
    )
    parser.add_argument("--json-out", type=Path, help="optional path to write the full machine-readable score JSON")
    parser.add_argument("--allow-dump-sha-mismatch", action="store_true", help="do not fail when dump SHA-256 differs")
    parser.add_argument("--fail-on-wrong-links", action="store_true", help="exit 2 if any wrong link is detected")
    parser.add_argument(
        "--fail-on-over-tier",
        action="store_true",
        help="exit 2 if any item is over-tiered or any finding/promotion violates policy",
    )
    args = parser.parse_args(argv)

    try:
        report = score_case(
            args.case_dir,
            args.groundtruth,
            view=args.view,
            strict_dump_sha=not args.allow_dump_sha_mismatch,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(format_summary(report))
    if args.fail_on_wrong_links and report["links"]["wrong"] > 0:
        return 2
    if args.fail_on_over_tier and (
        report["tiers"]["over_tiered"] > 0
        or report["tiers"]["policy_invalid_findings"] > 0
        or report["promotions"]["invalid"] > 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
