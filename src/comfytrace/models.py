"""Domain records for ComfyTrace manifests (Schemas 1-6 of the design document).

Every record serializes through ``to_dict()`` to exactly the keys of its schema, in schema order.
Vocabulary fields are ``str`` enums so they compare equal to their JSON spelling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0.0"


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class Completeness(_StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class LinkClass(_StrEnum):
    STRICT = "STRICT"
    SUPPORTED = "SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    UNASSOCIATED = "UNASSOCIATED"


class Tier(_StrEnum):
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"
    UNTIERED = "UNTIERED"


class GapReason(_StrEnum):
    PAGE_NOT_PRESENT = "PAGE_NOT_PRESENT"
    TRANSLATION_FAULT = "TRANSLATION_FAULT"
    PAGED_OUT = "PAGED_OUT"
    CRC_FAILURE = "CRC_FAILURE"
    NONCONTIGUOUS_PHYSICAL = "NONCONTIGUOUS_PHYSICAL"
    TRUNCATED_STREAM = "TRUNCATED_STREAM"
    STALE_MAPPING = "STALE_MAPPING"  # extension: the frame now belongs to another mapping (reused since)


class PageOrigin(_StrEnum):
    """What a physical page belonged to when memory was captured, read from its PFN entry."""

    PRIVATE = "PRIVATE"  # behind a hardware PTE: process-private or kernel memory
    SHARED_MEMORY = "SHARED_MEMORY"  # a pagefile-backed section: shared memory with no file behind it
    FILE_DATA = "FILE_DATA"  # a data file's section: file cache or a mapped view
    FILE_IMAGE = "FILE_IMAGE"  # an executable image's section
    FILE_UNCONFIRMED = "FILE_UNCONFIRMED"  # a subsection PTE, used only by file sections, with no readable flags
    FREED = "FREED"  # Zeroed, Free or Bad list: the former owner is gone
    UNKNOWN = "UNKNOWN"  # no PFN entry or layout to read


class CarveType(_StrEnum):
    ENCODED_PNG = "ENCODED_PNG"
    PARTIAL_PNG = "PARTIAL_PNG"
    PIXEL_BUFFER = "PIXEL_BUFFER"
    TENSOR_IMAGE = "TENSOR_IMAGE"
    DECODED_BITMAP = "DECODED_BITMAP"
    # Outputs ComfyUI saves in other containers (media.py); PARTIAL_* when the container walk did not close.
    ENCODED_WEBP = "ENCODED_WEBP"
    PARTIAL_WEBP = "PARTIAL_WEBP"
    ENCODED_MP4 = "ENCODED_MP4"
    PARTIAL_MP4 = "PARTIAL_MP4"
    ENCODED_WEBM = "ENCODED_WEBM"
    PARTIAL_WEBM = "PARTIAL_WEBM"
    ENCODED_FLAC = "ENCODED_FLAC"
    PARTIAL_FLAC = "PARTIAL_FLAC"


LINKED_CLASSES = frozenset({LinkClass.STRICT, LinkClass.SUPPORTED})


def _plain(value: Any) -> Any:
    """Deep-copies a value into JSON-native types (enums become their string values)."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return copy.deepcopy(value)


def internal(default: Any = None, factory: Any = None) -> Any:
    """A runtime-only field: carried between pipeline stages, never serialized into a schema."""
    meta = {"internal": True}
    if factory is not None:
        return field(default_factory=factory, repr=False, compare=False, metadata=meta)
    return field(default=default, repr=False, compare=False, metadata=meta)


class _Record:
    """Serializes schema fields in declaration order, which is the schema's key order."""

    @classmethod
    def schema_fields(cls) -> list[str]:
        return [f.name for f in fields(cls) if not f.metadata.get("internal")]  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self.schema_fields()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        names = set(cls.schema_fields())
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(frozen=True)
class GapDescriptor(_Record):
    """A run of bytes that could not be recovered. ComfyTrace never fills a gap with synthetic bytes."""

    relative_offset: int
    length_bytes: int
    reason: str
    synthetic_padding_applied: bool = False

    def __post_init__(self) -> None:
        if self.synthetic_padding_applied:
            raise ValueError("synthetic padding is prohibited; synthetic_padding_applied must be False")
        if self.relative_offset < 0 or self.length_bytes < 0:
            raise ValueError("gap offset and length must be non-negative")


@dataclass
class RawArtifactRecord(_Record):
    """Schema 2/3: a carved image. Carries ``completeness`` and never an evidence tier."""

    artifact_id: str
    completeness: str
    link_status: str
    associated_workflow_id: str | None
    ambiguous_candidate_workflows: list[str]
    is_derived: bool
    carve_type: str
    source_layer: str
    process_id: int | None
    virtual_address: str | None
    physical_offsets: list[str]
    recovered_bytes: int
    container_sha256: str | None
    pixel_sha256: str | None
    pixel_metadata: dict[str, Any] | None
    validation: dict[str, bool]
    gaps: list[GapDescriptor]
    raw_file_paths: list[str]
    derived_preview_path: str | None
    derived_preview_metadata: dict[str, Any] | None
    spans: list[tuple[int, bytes]] = internal(factory=list)
    embedded_text: dict[str, str] = internal(factory=dict)
    stitched_text: dict[str, str] = internal(factory=dict)  # DERIVED: text chunks completed by CRC-checked search
    preview_png: bytes | None = internal()
    # Full or CRC-sealed bytes reconstructed by content search. Exported only under exports/derived.
    derived_reconstruction: bytes | None = internal()
    evidence: dict[str, Any] = internal(factory=dict)
    page_origins: list[dict[str, Any]] = internal(factory=list)
    dimensions: tuple[int, int] | None = internal()
    # Rows read from frames whose ownership is uncertain: (first row, end row, RGB bytes). Exported apart from
    # exports/raw and never part of the record's hashes.
    uncertain_spans: list[tuple[int, int, bytes]] = internal(factory=list)
    uncertain_preview_png: bytes | None = internal()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawArtifactRecord:
        data = dict(data)
        data["gaps"] = [g if isinstance(g, GapDescriptor) else GapDescriptor.from_dict(g) for g in data["gaps"]]
        return super().from_dict(data)  # type: ignore[no-any-return]


@dataclass
class WorkflowRecord(_Record):
    """Schema 4: a recovered ComfyUI prompt graph or graph fragment."""

    workflow_id: str
    completeness: str
    prompt_format: str
    encoding_detected: str
    source_process_id: int | None
    virtual_address: str | None
    physical_offsets: list[str]
    canonical_json_sha256: str
    execution_parameters: dict[str, Any]
    dependencies: dict[str, list[str]]
    raw_file_path: str
    raw_bytes: bytes = internal(b"")
    graph: dict[str, Any] = internal(factory=dict)
    evidence: dict[str, Any] = internal(factory=dict)
    page_origins: list[dict[str, Any]] = internal(factory=list)


@dataclass
class CorrelationLink(_Record):
    link_id: str
    workflow_id: str
    artifact_id: str
    link_class: str
    matched_keys: list[str]
    confidence_notes: str


@dataclass
class Finding(_Record):
    """A correlated cluster of artifacts. Evidence tiers live here, not on artifacts."""

    finding_id: str
    tier: str
    artifact_ids: list[str]
    link_ids: list[str]
    rationale: str


@dataclass
class RegenerationRecord(_Record):
    """Schema 5: written to ``regenerations/REGEN-<id>.json``; never merged into the manifest."""

    handoff_id: str
    workflow_id: str
    target_manifest_sha256: str
    status: str
    security_audit: dict[str, Any]
    preflight_check: dict[str, Any]
    runner_environment: dict[str, str]
    evaluation: dict[str, Any]
    derived_regenerated_image: str | None
    output_rasters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        document = super().to_dict()
        if not self.output_rasters:
            document.pop("output_rasters")
        return document


@dataclass
class SupportingArtifactRecord(_Record):
    """Schema 6: untiered tensor/latent/cache span. Extension point; no carver emits these yet."""

    artifact_id: str
    category: str
    tier: str
    completeness: str
    source_layer: str
    process_id: int | None
    virtual_address: str | None
    physical_offsets: list[str]
    recovered_bytes: int
    sha256: str
    raw_file_paths: list[str]

    def __post_init__(self) -> None:
        if self.tier != Tier.UNTIERED:
            raise ValueError("supporting artifacts are always UNTIERED")


@dataclass
class ForensicManifest(_Record):
    """Schema 1: the top-level ``manifest.json``."""

    schema_version: str
    case_metadata: dict[str, Any]
    process_context: list[dict[str, Any]]
    findings: dict[str, Any] = field(default_factory=dict)


TIER_SUMMARY_KEYS = (
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "untiered_orphan_images",
    "untiered_ambiguous_images",
    "untiered_supporting_artifacts",
    "untiered_other_images",  # extension: images with no ComfyUI connection, counted as one group
)


def empty_tier_summary() -> dict[str, int]:
    return {k: 0 for k in TIER_SUMMARY_KEYS}
