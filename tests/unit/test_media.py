"""ComfyUI outputs that are not PNG: WEBP, MP4, WebM and FLAC written the way ComfyUI writes them
(tests/fixtures/media/make_fixtures.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from comfytrace import media
from comfytrace.carve import Block, IdAllocator, PhysicalSource
from comfytrace.cli import _carve
from comfytrace.correlate import correlate
from comfytrace.ingest import MODE_VIRTUAL, AddressSpace, IngestionResult, RecoverySupport
from comfytrace.models import CarveType, Completeness, LinkClass
from comfytrace.report import EvidentiaryExporter

FIXTURES = Path(__file__).parents[1] / "fixtures" / "media"
FILES = {"still.webp": "WEBP", "animated.webp": "WEBP", "video.mp4": "MP4", "video.webm": "WEBM",
         "audio.flac": "FLAC"}
PAGE = 4096


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.mark.parametrize(("name", "kind"), FILES.items())
def test_a_whole_container_is_walked_to_its_end_and_gives_up_its_prompt(name: str, kind: str):
    data = fixture(name)
    parsed = media.PARSERS[kind](data + bytes(300) + b"unrelated bytes that follow in memory" * 20)
    assert parsed is not None and parsed.complete and parsed.end == len(data) and parsed.halt is None
    assert json.loads(parsed.text["prompt"])["3"]["inputs"]["seed"] == 4242
    assert json.loads(parsed.text["workflow"])["nodes"][0]["type"] == "KSampler"


@pytest.mark.parametrize(("name", "kind"), FILES.items())
def test_a_container_cut_short_is_partial_and_keeps_only_verified_bytes(name: str, kind: str):
    data = fixture(name)
    parsed = media.PARSERS[kind](data[: len(data) * 2 // 3])
    assert parsed is not None and not parsed.complete and parsed.more and parsed.end < len(data) * 2 // 3


def test_formats_say_what_their_walk_proves():
    webp = media.parse_webp(fixture("animated.webp"))
    assert webp.meta["frames"] == 3 and (webp.meta["width"], webp.meta["height"]) == (64, 64)
    assert webp.validation == {"container_walk_closed": True, "payload_checksum": media.NO_CHECKSUM,
                               "exif_strings_valid": True, "decodes": True}
    flac = media.parse_flac(fixture("audio.flac"))
    assert flac.validation == {"frame_crc16_valid": True, "frames_verified": flac.meta["frames"],
                               "samples_verified": flac.meta["total_samples"]}
    assert flac.meta["sample_rate"] == 44100 and flac.meta["channels"] == 2
    mp4 = media.parse_mp4(fixture("video.mp4") + bytes(64))
    assert mp4.meta["boxes"] == ["ftyp", "free", "mdat", "moov"] and mp4.validation["last_box_is_structured"]
    webm = media.parse_webm(fixture("video.webm"))
    assert webm.validation["segment_size_declared"] and webm.validation["container_walk_closed"]


def test_a_damaged_flac_frame_fails_its_crc_and_a_damaged_webp_does_not_decode():
    data = bytearray(fixture("audio.flac"))
    data[-300] ^= 0x40  # inside the last audio frame
    parsed = media.parse_flac(bytes(data) + bytes(64))
    assert not parsed.complete and parsed.halt == "FRAME_CRC_FAILURE" and parsed.end < len(data)
    assert parsed.validation["frames_verified"] == media.parse_flac(fixture("audio.flac")).meta["frames"] - 1
    still = bytearray(fixture("still.webp"))
    at = still.index(b"VP8L") + 14
    still[at : at + 24] = bytes(24)  # no checksum catches this; the decoder does
    parsed = media.parse_webp(bytes(still))
    assert not parsed.complete and parsed.validation["decodes"] is False and parsed.halt.startswith("PIXEL_DECODE")


def test_text_that_is_not_json_is_not_comfyui_metadata():
    text: dict[str, str] = {}
    media._json_text(text, "comment", "made with ffmpeg")
    media._json_text(text, "track", "7")  # valid JSON, but any encoder writes such a tag
    media._json_text(text, "PROMPT", '{"3": {}}')
    assert text == {"prompt": '{"3": {}}'}


def _space(memory: bytes) -> AddressSpace:
    return AddressSpace(None, [Block(0, memory)], PhysicalSource(memory))


def test_a_container_with_comfyui_text_is_a_record_and_its_prompt_is_not_carved_twice(tmp_path):
    webp, flac = fixture("still.webp"), fixture("audio.flac")
    memory = bytearray(12 * PAGE)
    memory[PAGE + 100 : PAGE + 100 + len(webp)] = webp
    memory[4 * PAGE : 4 * PAGE + len(flac)] = flac
    ingestion = IngestionResult(MODE_VIRTUAL, {}, [], [_space(bytes(memory))], recovery=RecoverySupport())
    images, workflows, _ = _carve(ingestion)
    assert [(i.artifact_id, i.carve_type, i.completeness) for i in images] == [
        ("CARVE-WEBP-001", CarveType.ENCODED_WEBP, Completeness.COMPLETE),
        ("CARVE-FLAC-001", CarveType.ENCODED_FLAC, Completeness.COMPLETE)]
    picture, audio = images
    assert picture.container_sha256 == hashlib.sha256(webp).hexdigest() and picture.dimensions == (64, 64)
    assert picture.physical_offsets == [f"0x{PAGE + 100:016X}"] and picture.source_layer == "physical_UNATTRIBUTED"
    assert picture.pixel_sha256 is not None and audio.pixel_sha256 is None
    # One prompt and one UI workflow, each read from both containers; the JSON carver made no record of its own.
    assert sorted((w.workflow_id[:5], w.prompt_format, w.encoding_detected) for w in workflows) == [
        ("WFPNG", "API_GRAPH", "WEBP_EMBEDDED_TEXT"), ("WFPNG", "UI_WORKFLOW", "WEBP_EMBEDDED_TEXT")]
    prompt = next(w for w in workflows if w.prompt_format == "API_GRAPH")
    assert [loc["png"] for loc in prompt.execution_parameters["source_locations"]] == [
        "CARVE-WEBP-001", "CARVE-FLAC-001"]
    assert ingestion.recovery.notes["media"]["recorded"] == 2
    result = correlate(images, workflows)
    assert {(link.artifact_id, link.link_class, tuple(link.matched_keys)) for link in result.links} == {
        ("CARVE-WEBP-001", LinkClass.STRICT, ("EMBEDDED_PROMPT",)),
        ("CARVE-FLAC-001", LinkClass.STRICT, ("EMBEDDED_PROMPT",))}
    assert result.tier_summary["E2"] == 2, "a picture and a sound are two outputs of the job"
    exporter = EvidentiaryExporter(tmp_path)
    assert exporter.export_raw_artifact(audio, audio.spans) == ["exports/raw/CARVE-FLAC-001.flac"]
    assert (tmp_path / "exports/raw/CARVE-FLAC-001.flac").read_bytes() == flac


def test_other_media_is_counted_and_only_a_webp_of_a_job_size_is_kept():
    from PIL import Image

    def plain_webp(size: tuple[int, int], shade: int) -> bytes:
        import io

        out = io.BytesIO()
        Image.new("RGB", size, (shade, 40, 90)).save(out, "WEBP", lossless=True)
        return out.getvalue()

    job_sized, browser = plain_webp((64, 64), 10), plain_webp((48, 20), 200)
    with_text = fixture("still.webp")  # its prompt generates 64 x 64
    memory = bytearray(8 * PAGE)
    for at, blob in ((100, job_sized), (PAGE + 7, browser), (2 * PAGE, with_text)):
        memory[at : at + len(blob)] = blob
    ingestion = IngestionResult(MODE_VIRTUAL, {}, [], [_space(bytes(memory))], recovery=RecoverySupport())
    images, _, _ = _carve(ingestion)
    assert sorted(i.container_sha256 for i in images) == sorted(
        hashlib.sha256(blob).hexdigest() for blob in (job_sized, with_text))
    assert ingestion.recovery.notes["media"] == {
        "containers_walked": {"WEBP": 3, "FLAC": 0, "MP4": 0, "WEBM": 0}, "recorded": 2,
        "recorded_for_job_size_only": 1, "not_recorded": 1}


def test_a_torn_container_keeps_its_text_and_says_where_it_tore():
    webp = fixture("still.webp")
    exif = webp.index(b"EXIF")
    torn = webp[: exif + 8 + 300]  # the page edge falls inside the EXIF chunk
    memory = bytearray(4 * PAGE)
    start = 2 * PAGE - len(torn)
    memory[start : 2 * PAGE] = torn
    memory[2 * PAGE :] = b"\xaa" * (2 * PAGE)
    carver = media.MediaCarver(physical_mode=True, ids={"WEBP": IdAllocator("CARVE-WEBP")})
    [candidate] = carver.carve_stream(bytes(memory), base_physical_offset=0, source=PhysicalSource(bytes(memory)))
    # The chunk chain closes on the declared size and the picture decodes, but the prompt string ran into other
    # bytes: only the workflow string, which ends before the tear, is intact.
    assert candidate.parsed.halt == "EXIF_STRINGS_INVALID" and sorted(candidate.parsed.text) == ["workflow"]
    record = carver.record(candidate)
    assert (record.carve_type, record.completeness) == (CarveType.PARTIAL_WEBP, Completeness.PARTIAL)
    assert record.container_sha256 is None and record.validation["exif_strings_valid"] is False
    [gap] = record.gaps
    assert gap.reason == "NONCONTIGUOUS_PHYSICAL" and gap.relative_offset == record.recovered_bytes < len(torn)
    assert record.spans == [(0, webp[: record.recovered_bytes])], "only bytes of the file, up to the last intact string"
    # The same bytes without any ComfyUI text are one of the machine's many media pieces: not kept.
    plain = bytearray(memory)
    plain[start + exif : 2 * PAGE] = b"\xaa" * (2 * PAGE - start - exif)
    assert media.MediaCarver(physical_mode=True).carve_stream(bytes(plain), base_physical_offset=0) == []


def test_a_movie_header_or_tags_element_that_runs_into_other_bytes_is_not_complete():
    mp4 = bytearray(fixture("video.mp4"))
    mp4[mp4.index(b"udta") - 4 : mp4.index(b"udta") + 40] = b"\xaa" * 44
    parsed = media.parse_mp4(bytes(mp4) + bytes(64))
    assert not parsed.complete and parsed.halt == "MOVIE_HEADER_INVALID" and parsed.text == {}
    webm = bytearray(fixture("video.webm"))
    at = webm.index(b"PROMPT")
    webm[at - 12 : at + 60] = b"\xaa" * 72
    parsed = media.parse_webm(bytes(webm))
    assert not parsed.complete and parsed.halt == "TAGS_INVALID"
