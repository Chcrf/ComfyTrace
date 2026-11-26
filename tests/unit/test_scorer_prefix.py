"""The scorer credits a partial PNG only when its one leading span is an exact prefix of one known output that
reaches into the encoded pixels. Synthetic PNG bytes only."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.evaluation.scorer import (
    PREFIX_PAST_IDAT,
    _artifact_identity,
    _idat_data_offset,
    _match_artifact_to_gt,
    _parse_groundtruth,
)


def _png(seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (96, 96, 3)).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG", pnginfo=None)
    return out.getvalue()


def _truth(tmp_path: Path, *files: bytes) -> list:
    outputs = []
    for n, data in enumerate(files):
        path = tmp_path / f"out{n}.png"
        path.write_bytes(data)
        outputs.append({"output_id": f"job:7:{n}", "container_sha256": hashlib.sha256(data).hexdigest(),
                        "container_file": path.name})
    _, jobs, _ = _parse_groundtruth({"jobs": [{"job_id": "job", "outputs": outputs}]}, tmp_path)
    return [o for job in jobs for o in job.outputs]


def _partial(data: bytes, length: int, spans: int = 1) -> dict:
    return {"artifact_id": "CARVE-PNG-1", "carve_type": "PARTIAL_PNG", "completeness": "PARTIAL",
            "recovered_bytes": length, "physical_offsets": ["0x1000"],
            "gaps": [{"relative_offset": length, "length_bytes": 4096, "reason": "PAGE_NOT_PRESENT"}],
            "derived_preview_metadata": {"raw_span_sha256": [hashlib.sha256(data[:length]).hexdigest()] * spans}}


def test_an_exact_prefix_into_the_pixels_matches_its_own_output_only(tmp_path):
    first, second = _png(1), _png(2)
    outputs = _truth(tmp_path, first, second)
    length = _idat_data_offset(first) + PREFIX_PAST_IDAT + 100
    art = _artifact_identity(tmp_path, _partial(first, length), None)
    assert _match_artifact_to_gt(art, outputs, []) == ("output", ["job:7:0"], "container_prefix")


@pytest.mark.parametrize("change", ["header_only", "two_spans", "wrong_bytes", "whole_file"])
def test_a_prefix_that_proves_nothing_or_does_not_agree_matches_nothing(tmp_path, change):
    first = _png(1)
    outputs = _truth(tmp_path, first)
    length = {"header_only": _idat_data_offset(first) + 10, "whole_file": len(first)}.get(
        change, _idat_data_offset(first) + PREFIX_PAST_IDAT + 100)
    record = _partial(first, length, spans=2 if change == "two_spans" else 1)
    if change == "wrong_bytes":
        record["derived_preview_metadata"]["raw_span_sha256"] = ["00" * 32]
    art = _artifact_identity(tmp_path, record, None)
    assert _match_artifact_to_gt(art, outputs, []) == (None, [], None)


def test_a_container_file_that_is_not_the_declared_output_is_refused(tmp_path):
    (tmp_path / "out.png").write_bytes(_png(1))
    with pytest.raises(ValueError, match="container_file"):
        _parse_groundtruth({"jobs": [{"job_id": "job", "outputs": [
            {"output_id": "job:7:0", "container_sha256": "ab" * 32, "container_file": "out.png"}]}]}, tmp_path)
