"""Case files, immutable evidence exports and regeneration records."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from comfytrace.carve import GapTracker
from comfytrace.models import (
    CarveType,
    Completeness,
    ForensicManifest,
    RawArtifactRecord,
    RegenerationRecord,
    WorkflowRecord,
)

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "workbench.html"
EVIDENCE_NAME = "recovery-evidence.json"
LINKS_NAME = "image-links.json"
METADATA_NAME = "workflow-metadata.json"
DERIVED_PREFIX = "DERIVED_"


def derived_file(case_dir: Path, name: Any) -> Path | None:
    """Resolve a case-relative replay file within this case's derived exports."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        return None
    try:
        root = case_dir.resolve()
        path = (root / name).resolve()
        if path.is_relative_to(root / "exports" / "derived") and path.is_file():
            return path
    except (ValueError, OSError, RuntimeError):
        pass
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches_derived_file(case_dir: Path | None, name: Any, expected: Any) -> bool:
    """True when ``name`` resolves under ``exports/derived`` and its SHA-256 equals ``expected``."""
    path = derived_file(case_dir, name) if case_dir is not None and expected else None
    try:
        return path is not None and sha256_file(path) == expected
    except OSError:
        return False


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)


class EvidentiaryExporter:
    """exports/raw/ holds only genuine carved bytes; exports/derived/ holds everything synthesized."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)
        self.raw_dir = self.case_dir / "exports" / "raw"
        self.derived_dir = self.case_dir / "exports" / "derived"

    def export_raw_artifact(self, artifact: RawArtifactRecord, payload: bytes | list[tuple[int, bytes]]) -> list[str]:
        """A gap-free COMPLETE image is written as <id>.png (another container as <id>.webp, .mp4, .webm or
        .flac); anything with gaps as span_NNN.bin slices.
        Returns exported paths relative to the case directory.
        Raises: OSError if disk write fails."""
        spans = artifact.spans or ([(0, payload)] if isinstance(payload, bytes) else payload)
        extensions: dict[str, str] = {
            CarveType.ENCODED_PNG: "png",
            CarveType.PARTIAL_PNG: "png",
            CarveType.ENCODED_WEBP: "webp",
            CarveType.ENCODED_MP4: "mp4",
            CarveType.ENCODED_WEBM: "webm",
            CarveType.ENCODED_FLAC: "flac",
        }
        extension = extensions.get(artifact.carve_type)
        derived_assembly = (artifact.evidence.get("assembly") or {}).get("derived") is True
        if (
            not derived_assembly
            and artifact.completeness == Completeness.COMPLETE
            and not artifact.gaps
            and extension is not None
            and len(spans) == 1
            and spans[0][0] == 0
        ):
            target = self.raw_dir / f"{artifact.artifact_id}.{extension}"
            _write_new(target, spans[0][1])
            paths = [target.relative_to(self.case_dir).as_posix()]
        else:
            paths = GapTracker(self.case_dir).export_raw_spans(artifact.artifact_id, spans)
        artifact.raw_file_paths = paths
        return paths

    def export_workflow(self, workflow: WorkflowRecord) -> str:
        """Writes the carved JSON bytes verbatim, in their original encoding. Text completed by search
        (PNG_TEXT_STITCHED) is DERIVED material and goes to exports/derived."""
        if workflow.encoding_detected == "PNG_TEXT_STITCHED":
            target = self.raw_dir.parent / "derived" / f"DERIVED_stitched_{workflow.workflow_id}.json"
        else:
            target = self.raw_dir / f"{workflow.workflow_id}.json"
        _write_new(target, workflow.raw_bytes)
        workflow.raw_file_path = target.relative_to(self.case_dir).as_posix()
        return workflow.raw_file_path

    def export_uncertain(self, artifact: RawArtifactRecord) -> None:
        """Writes an image's rows of uncertain ownership to exports/uncertain/<id>/rows_<first>-<end>.rgb and a
        DERIVED preview showing them in place, and records both paths in the record's evidence. Nothing goes to
        exports/raw: that folder holds only bytes known to be the artifact's."""
        layer = artifact.evidence.get("ownership_uncertain")
        if not layer or not artifact.uncertain_spans:
            return
        for row, (first, end, data) in zip(layer["rows"], artifact.uncertain_spans, strict=True):
            target = self.case_dir / "exports" / "uncertain" / artifact.artifact_id / f"rows_{first:04d}-{end:04d}.rgb"
            _write_new(target, data)
            row["file"] = target.relative_to(self.case_dir).as_posix()
        if artifact.uncertain_preview_png is not None:
            layer["derived_preview"] = self.export_derived_preview(
                f"preview_{artifact.artifact_id}_with_uncertain_rows.png", artifact.uncertain_preview_png
            )
            layer["derived_preview_sha256"] = hashlib.sha256(artifact.uncertain_preview_png).hexdigest()

    def export_derived_preview(self, filename: str, preview_bytes: bytes) -> str:
        """Writes a reconstruction to exports/derived/ under a DERIVED_ prefix.
        Raises: OSError if disk write fails."""
        name = filename if filename.startswith(DERIVED_PREFIX) else DERIVED_PREFIX + filename
        target = self.derived_dir / name
        _write_new(target, preview_bytes)
        return target.relative_to(self.case_dir).as_posix()


class ManifestWriter:
    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)

    def write_manifest(self, manifest: ForensicManifest) -> Path:
        """Writes manifest.json once and marks it read-only. An existing manifest is never replaced.
        Raises: OSError if disk write fails or a manifest already exists."""
        path = self.case_dir / MANIFEST_NAME
        _write_new(path, (json.dumps(manifest.to_dict(), indent=2) + "\n").encode("utf-8"))
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return path


def save_regeneration_record(case_dir: Path, record: RegenerationRecord) -> Path:
    """Writes <case_dir>/regenerations/REGEN-<id>.json, never overwriting an earlier record.
    Raises: OSError if disk write fails."""
    path = Path(case_dir) / "regenerations" / f"{record.handoff_id}.json"
    _write_new(path, (json.dumps(record.to_dict(), indent=2) + "\n").encode("utf-8"))
    return path


def next_regeneration_id(case_dir: Path) -> str:
    folder = Path(case_dir) / "regenerations"
    taken = {p.stem for p in folder.glob("REGEN-*.json")} if folder.is_dir() else set()
    n = 1
    while f"REGEN-{n:03d}" in taken:
        n += 1
    return f"REGEN-{n:03d}"


def load_regenerations(folder: Path) -> list[dict[str, Any]]:
    if not folder.is_dir():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("REGEN-*.json"))]
