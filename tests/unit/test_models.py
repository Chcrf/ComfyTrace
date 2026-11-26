import dataclasses
import json
from pathlib import Path

import pytest

from comfytrace.models import (
    CorrelationLink,
    Finding,
    ForensicManifest,
    GapDescriptor,
    RawArtifactRecord,
    RegenerationRecord,
    SupportingArtifactRecord,
    WorkflowRecord,
)

SCHEMAS = Path(__file__).parent.parent / "fixtures" / "schemas"


def _example(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.json").read_text())


@pytest.mark.parametrize(
    ("model", "example"),
    [
        (RawArtifactRecord, "raw_artifact_intact"),
        (RawArtifactRecord, "raw_artifact_gapped"),
        (WorkflowRecord, "workflow"),
        (RegenerationRecord, "regeneration"),
        (SupportingArtifactRecord, "supporting_artifact"),
        (ForensicManifest, "manifest"),
    ],
)
def test_manifest_schema_serialization_parity(model, example):
    data = _example(example)
    record = model.from_dict(data)
    out = record.to_dict()
    assert out == data
    assert list(out) == list(data), "key order follows the schema"
    json.dumps(out)


def test_nested_manifest_links_and_findings_round_trip():
    manifest = _example("manifest")
    link = CorrelationLink.from_dict(manifest["findings"]["correlation_links"][0])
    finding = Finding.from_dict(manifest["findings"]["findings"][0])
    assert link.to_dict() == manifest["findings"]["correlation_links"][0]
    assert finding.to_dict() == manifest["findings"]["findings"][0]


def test_finding_cluster_validation():
    finding = Finding(
        finding_id="FIND-001",
        tier="E2",
        artifact_ids=["CARVE-PNG-001", "WF-001"],
        link_ids=["LINK-001"],
        rationale="linked",
    )
    data = finding.to_dict()
    assert data["artifact_ids"] == ["CARVE-PNG-001", "WF-001"]
    assert data["link_ids"] == ["LINK-001"]
    data["artifact_ids"].append("X")
    assert finding.artifact_ids == ["CARVE-PNG-001", "WF-001"], "to_dict returns a copy"


def test_gap_descriptor_immutability():
    gap = GapDescriptor(relative_offset=4096, length_bytes=4096, reason="PAGE_NOT_PRESENT")
    assert gap.synthetic_padding_applied is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        gap.synthetic_padding_applied = True  # type: ignore[misc]
    with pytest.raises(ValueError):
        GapDescriptor(relative_offset=0, length_bytes=1, reason="PAGE_NOT_PRESENT", synthetic_padding_applied=True)


def test_raw_artifact_has_no_tier_field():
    assert "tier" not in {f.name for f in dataclasses.fields(RawArtifactRecord)}


def test_supporting_artifact_must_be_untiered():
    data = _example("supporting_artifact")
    data["tier"] = "E6"
    with pytest.raises(ValueError):
        SupportingArtifactRecord.from_dict(data)
