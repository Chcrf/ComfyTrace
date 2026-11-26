import hashlib
import json
import mmap
from pathlib import Path

import numpy as np
from PIL import Image

from comfytrace.lime import LIME_HEADER, LIME_MAGIC, LimePhysicalImage, parse_lime_segments
from comfytrace.replaysearch import verify_linked_pngs


def test_linked_png_is_verified_in_physical_page_order(tmp_path):
    asset = tmp_path / "exports/raw/png-1.png"
    asset.parent.mkdir(parents=True)
    rgb = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(asset)
    png = asset.read_bytes()
    digest = hashlib.sha256(png).hexdigest()
    artifact = {"artifact_id": "PNG-1", "carve_type": "ENCODED_PNG", "completeness": "COMPLETE",
                "container_sha256": digest, "raw_file_paths": ["exports/raw/png-1.png"]}
    manifest = {"findings": {"artifacts": [artifact]}}
    evidence = {"records": [{"record_id": "PNG-1", "assembly": {"page_offsets": ["0x2000", "0x0"]}}]}
    (tmp_path / "recovery-evidence.json").write_text(json.dumps(evidence))
    regen = tmp_path / "regenerations"
    regen.mkdir()
    (regen / "REGEN-001.json").write_text(json.dumps({
        "handoff_id": "REGEN-001", "target_manifest_sha256": "manifest", "workflow_id": "WFPY-001",
        "evaluation": {"target_artifact_id": "PNG-1"},
    }))
    raw = bytearray(3 * 4096)
    raw[3 * 4096 - 96:] = png[:96]
    raw[:len(png) - 96] = png[96:]
    (proof,) = verify_linked_pngs(tmp_path, bytes(raw), manifest, "manifest", {"REGEN-001"})
    assert proof["raw_sha256"] == digest
    assert proof["spans"] == [{"physical_offset": 3 * 4096 - 96, "length_bytes": 96},
                              {"physical_offset": 0, "length_bytes": len(png) - 96}]
    raw[1] ^= 1
    assert verify_linked_pngs(tmp_path, bytes(raw), manifest, "manifest", {"REGEN-001"}) == []


LOW, HIGH = 0x0, 0x100000  # two RAM ranges with a PCI hole between them


def _lime(path: Path, ranges: list[tuple[int, bytes]]) -> Path:
    out = bytearray()
    for start, data in ranges:
        out += LIME_HEADER.pack(LIME_MAGIC, 1, start, start + len(data) - 1, 0) + data
    path.write_bytes(bytes(out))
    return path


def _png_case(tmp_path: Path, page_offsets: list[str] | None, physical_offsets: list[str] | None = None):
    asset = tmp_path / "exports/raw/png-1.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(3).integers(0, 256, (24, 24, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(asset)
    png = asset.read_bytes()
    artifact = {"artifact_id": "PNG-1", "carve_type": "ENCODED_PNG", "completeness": "COMPLETE",
                "container_sha256": hashlib.sha256(png).hexdigest(), "raw_file_paths": ["exports/raw/png-1.png"],
                "physical_offsets": physical_offsets or []}
    assembly = {"page_offsets": page_offsets} if page_offsets else {}
    (tmp_path / "recovery-evidence.json").write_text(json.dumps({"records": [
        {"record_id": "PNG-1", "assembly": assembly}]}))
    regen = tmp_path / "regenerations"
    regen.mkdir(exist_ok=True)
    (regen / "REGEN-001.json").write_text(json.dumps({
        "handoff_id": "REGEN-001", "target_manifest_sha256": "manifest", "workflow_id": "WFPY-001",
        "evaluation": {"target_artifact_id": "PNG-1"}}))
    return png, {"findings": {"artifacts": [artifact]}}


def _verify_lime(tmp_path: Path, capture: Path, manifest: dict) -> list[dict]:
    segments = parse_lime_segments(capture)
    assert segments is not None
    with capture.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        return verify_linked_pngs(tmp_path, LimePhysicalImage(mapped, segments), manifest, "manifest",
                                  {"REGEN-001"})


def test_linked_png_on_a_lime_capture_is_read_at_its_physical_address(tmp_path):
    # The PNG lies at physical HIGH (above the hole). In the file it sits at another offset, after the first
    # segment and two headers; a read of the file at HIGH would find nothing there.
    png, manifest = _png_case(tmp_path, None, [hex(HIGH)])
    low = np.random.default_rng(4).integers(0, 256, 0x3000, np.uint8).tobytes()
    capture = _lime(tmp_path / "dump.lime", [(LOW, low), (HIGH, png + b"\0" * (-len(png) % 4096))])
    (proof,) = _verify_lime(tmp_path, capture, manifest)
    assert proof["spans"] == [{"physical_offset": HIGH, "length_bytes": len(png)}]
    assert verify_linked_pngs(tmp_path, capture.read_bytes(), manifest, "manifest", {"REGEN-001"}) == [], \
        "the raw file offset is not the physical address"


def test_linked_png_pages_in_a_lime_hole_verify_nothing(tmp_path):
    png, manifest = _png_case(tmp_path, [hex(HIGH)])
    capture = _lime(tmp_path / "dump.lime", [(LOW, b"\0" * 0x3000), (HIGH, png.ljust(4096, b"\0"))])
    (proof,) = _verify_lime(tmp_path, capture, manifest)
    assert proof["spans"] == [{"physical_offset": HIGH, "length_bytes": len(png)}]
    # The recorded pages: one in RAM, one in the hole (0x3000 is past the first segment's end).
    _, manifest = _png_case(tmp_path, [hex(HIGH), "0x3000"])
    assert _verify_lime(tmp_path, capture, manifest) == []
    # A physical span that runs from RAM into the hole is short, so it is rejected, not padded.
    _, manifest = _png_case(tmp_path, None, [hex(0x3000 - 16)])
    assert _verify_lime(tmp_path, capture, manifest) == []


def test_replay_evidence_reads_lime_spans_at_physical_addresses(tmp_path):
    from comfytrace.replayevidence import physical_bytes

    low = bytes(range(256)) * 48
    high = bytes(reversed(range(256))) * 16
    capture = _lime(tmp_path / "dump.lime", [(LOW, low), (HIGH, high)])
    assert physical_bytes(capture, HIGH + 16, 64) == high[16:80]
    assert physical_bytes(capture, len(low) - 8, 16) is None, "a span running into the PCI hole is rejected"
    assert physical_bytes(capture, len(low) + 64, 16) is None, "a span inside the hole is rejected"
    assert physical_bytes(capture, HIGH + len(high) - 8, 16) is None
    flat = tmp_path / "dump.raw"
    flat.write_bytes(low + high)
    assert physical_bytes(flat, len(low), 16) == high[:16], "a flat image is indexed by file offset"
