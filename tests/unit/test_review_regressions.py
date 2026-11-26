"""Regressions from the cross-provider review of the cache-reuse, terminal-job and pixel-identity changes.

All inputs are synthetic: tiny PNGs, fake CPython caches and queue items built in memory."""

from __future__ import annotations

import copy
import io
import json
import struct
import zlib

import numpy as np
from PIL import Image

from comfytrace import evidence, execcache, memrecover
from comfytrace.carve import IdAllocator, PngCarver, native_pixel_digest
from comfytrace.correlate import CACHE_KEY, CACHE_REUSED_KEY, correlate
from comfytrace.memrecover import OUTPUT_CONTEXT_FIELD, TERMINAL_WITHOUT_RECORDED_OUTPUTS
from comfytrace.models import Completeness, LinkClass
from tests.synthetic import api_prompt
from tests.unit.test_correlate_identity import stored_png
from tests.unit.test_execcache import JOB_A, JOB_B, CachePython, cached_job, image_tensor, queue_item
from tests.unit.test_terminal_without_outputs import CUT, DONE, OTHER, interrupted, item, success

STAMP = 1794065346982


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


# ------------------------------------------------------------------------------------------- cache reuse


def _reused_by(queue, nodes):
    queue.history["status"]["messages"] = [
        ["execution_start", {"prompt_id": queue.prompt_id, "timestamp": STAMP}],
        ["execution_cached", {"prompt_id": queue.prompt_id, "nodes": nodes, "timestamp": STAMP}],
        ["execution_success", {"prompt_id": queue.prompt_id, "timestamp": STAMP + 10}]]
    return queue


def _cache_case(items, cached_prompt):
    py = CachePython()
    made = cached_job(py, cached_prompt)
    entries, _ = execcache.extract(py.memory())
    tie = execcache.image_ties(entries)[(4242, made["impl"])]
    tie["jobs"] = execcache.job_matches(tie["graph"], items, tie["output_socket"])
    images = memrecover.image_records([memrecover.from_tensor(image_tensor(made["impl"]), lambda va: None, tie)])
    return images, memrecover.workflow_records(items, IdAllocator("WFPY")), tie


def test_the_job_that_computed_a_cached_tensor_keeps_it_beside_a_job_that_reused_it():
    job = api_prompt(seed=222)
    first = queue_item(JOB_A, job, 0x1000)
    second = _reused_by(queue_item(JOB_B, copy.deepcopy(job), 0x2000), ["4", "3", "8", "9"])
    images, workflows, tie = _cache_case([first, second], job)
    assert {j["prompt_id"]: j["reused"] for j in tie["jobs"]} == {JOB_A: False, JOB_B: True}
    [link] = correlate(images, workflows).links
    linked = next(w for w in workflows if w.workflow_id == link.workflow_id)
    assert linked.execution_parameters["prompt_id"] == JOB_A
    assert (link.link_class, link.matched_keys) == (LinkClass.STRICT, [CACHE_KEY])


def test_a_job_that_only_reused_a_cached_tensor_gets_no_strict_link_or_output_position():
    job = api_prompt(seed=222)
    second = _reused_by(queue_item(JOB_B, job), ["4", "3", "8", "9"])
    images, workflows, _ = _cache_case([second], job)
    result = correlate(images, workflows)
    [link] = result.links
    assert link.link_class == LinkClass.SUPPORTED and CACHE_REUSED_KEY in link.matched_keys
    card = evidence.scorecard([second], workflows, "ab" * 32, result.links, images)
    roles = {c.get("role") for row in card["rows"] for c in row["image"]["candidates"]}
    assert roles == {"JOB"}
    assert all("batch_index" not in c for row in card["rows"] for c in row["image"]["candidates"])


def _buffer_copy(image, artifact_id: str):
    import dataclasses

    from comfytrace.models import CarveType

    return dataclasses.replace(image, artifact_id=artifact_id, carve_type=CarveType.PIXEL_BUFFER, evidence={},
                               link_status=LinkClass.UNASSOCIATED, associated_workflow_id=None)


def test_a_pixel_buffer_with_the_cached_outputs_exact_pixels_joins_that_output():
    from comfytrace.correlate import PIXEL_COPY_KEY

    job = api_prompt(seed=222)
    images, workflows, _ = _cache_case([queue_item(JOB_A, job)], job)
    [tensor] = images
    buffer = _buffer_copy(tensor, "PIXBUF-001")
    result = correlate([tensor, buffer], workflows)
    by_image = {link.artifact_id: link for link in result.links}
    assert by_image["PIXBUF-001"].link_class == LinkClass.SUPPORTED
    assert by_image["PIXBUF-001"].matched_keys == [PIXEL_COPY_KEY]
    assert by_image["PIXBUF-001"].workflow_id == by_image[tensor.artifact_id].workflow_id
    [finding] = [f for f in result.findings if "PIXBUF-001" in f.artifact_ids]
    assert tensor.artifact_id in finding.artifact_ids and finding.tier == "E2", "one output, still exact"


def test_pixels_shared_by_two_submissions_link_no_copy():
    job = api_prompt(seed=222)
    renamed = copy.deepcopy(job)
    renamed["9"]["inputs"]["filename_prefix"] = "second"
    images, workflows, _ = _cache_case([queue_item(JOB_A, job, 0x1000), queue_item(JOB_B, renamed, 0x2000)], job)
    [tensor] = images
    result = correlate([tensor, _buffer_copy(tensor, "PIXBUF-001")], workflows)
    assert "PIXBUF-001" not in {link.artifact_id for link in result.links}


def test_consumers_follow_only_the_output_socket_that_carried_the_tensor():
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
              "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
              "3": {"class_type": "MaskToImage", "inputs": {"mask": ["1", 1]}}}
    assert execcache.consumers(prompt, "1") == ["2", "3"]
    assert execcache.consumers(prompt, "1", 0) == ["2"]
    assert execcache.consumers(prompt, "1", 1) == ["3"]


# ------------------------------------------------------------------------------- terminal without outputs


def test_an_interrupted_twin_neither_takes_nor_blurs_a_completed_jobs_png():
    graph = item(DONE, None).prompt
    png = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(stored_png(graph, 100), base_physical_offset=0x10000)
    workflows = memrecover.workflow_records([item(DONE, success()), item(CUT, interrupted())], IdAllocator("WFPY"))
    by_job = {w.execution_parameters["prompt_id"]: w.workflow_id for w in workflows if w.prompt_format == "API_GRAPH"}
    links = correlate(png, workflows).links
    assert [(link.workflow_id, link.link_class) for link in links] == [(by_job[DONE], LinkClass.STRICT)]


def test_an_interrupted_job_alone_does_not_take_a_png_with_its_graph():
    graph = item(CUT, None).prompt
    png = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(stored_png(graph, 100), base_physical_offset=0x10000)
    workflows = memrecover.workflow_records([item(CUT, interrupted())], IdAllocator("WFPY"))
    assert all(w.execution_parameters[OUTPUT_CONTEXT_FIELD] == TERMINAL_WITHOUT_RECORDED_OUTPUTS for w in workflows)
    links = correlate(png, workflows).links
    assert all(w.workflow_id not in {link.workflow_id for link in links} for w in workflows)


def test_a_fully_cached_resubmission_cannot_be_the_producer_of_a_tied_png():
    first = item(DONE, success())
    graph = first.prompt
    png = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(stored_png(graph, 100), base_physical_offset=0x10000)
    reused = item(OTHER, success(OTHER))
    reused.history["status"]["messages"].insert(1, ["execution_cached", {
        "prompt_id": OTHER, "nodes": sorted(graph), "timestamp": STAMP}])
    workflows = memrecover.workflow_records([first, reused], IdAllocator("WFPY"))
    by_job = {w.execution_parameters["prompt_id"]: w.workflow_id for w in workflows if w.prompt_format == "API_GRAPH"}
    links = correlate(png, workflows).links
    assert [(link.workflow_id, link.link_class) for link in links] == [(by_job[DONE], LinkClass.STRICT)]
    # Without the cache record the two submissions stay tied.
    plain = memrecover.workflow_records([item(DONE, success()), item(OTHER, success(OTHER))], IdAllocator("WFPY"))
    png = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(stored_png(graph, 100), base_physical_offset=0x10000)
    assert {link.link_class for link in correlate(png, plain).links} == {LinkClass.AMBIGUOUS}


def test_a_serialized_history_entry_names_its_own_cached_nodes_only():
    from comfytrace.carve import execution_parameters, extract_prompt_graph

    graph = api_prompt(seed=5)
    pid = "50a8582f-6479-46b4-ad3b-9c5cfae675c8"
    entry = {"prompt": [1, pid, graph, {}, ["9"]], "outputs": {}, "status": {"messages": [
        ["execution_start", {"prompt_id": pid}],
        ["execution_cached", {"prompt_id": "another-prompt", "nodes": ["1"]}],
        ["execution_cached", {"prompt_id": pid, "nodes": sorted(graph)}]]}}
    fmt, found, runtime = extract_prompt_graph(json.loads(json.dumps(entry)))
    assert fmt == "API_GRAPH" and found == graph
    assert execution_parameters(found, runtime)["job_timeline"] == {"cached_nodes": sorted(graph)}
    no_cache = {**entry, "status": {"messages": [["execution_start", {"prompt_id": pid}]]}}
    assert "job_timeline" not in extract_prompt_graph(no_cache)[2]


# -------------------------------------------------------------------------------------- history decoding


def test_a_history_output_whose_images_could_not_be_read_is_skipped_not_fatal():
    queue = item(DONE, success())
    queue.history["outputs"]["9"]["images"] = None
    [api, *_] = memrecover.workflow_records([queue], IdAllocator("WFPY"))
    assert "output_filenames" not in api.execution_parameters
    card = evidence.scorecard([queue], memrecover.workflow_records([queue], IdAllocator("WFPY")), "ab" * 32, [])
    assert card["rows"][0]["output_node_id"] is None


def test_a_bare_queue_copy_first_does_not_hide_the_history_of_a_later_copy():
    records = memrecover.workflow_records([item(DONE, None), item(DONE, success())], IdAllocator("WFPY"))
    [api] = [r for r in records if r.prompt_format == "API_GRAPH"]
    params = api.execution_parameters
    assert (params["history_status"], params["history_completed"]) == ("success", True)
    assert params["output_filenames"] == ["ComfyUI_00001_.png"] and len(params["source_locations"]) == 2


# ------------------------------------------------------------------------------------------ pixel identity


def _png16(transparent: bool) -> bytes:
    rows = b"".join(b"\x00" + struct.pack(">HHH", 1000, 2000, 3000) * 4 for _ in range(4))
    out = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 16, 2, 0, 0, 0))
    if transparent:
        out += _chunk(b"tRNS", struct.pack(">HHH", 1000, 2000, 3000))
    return out + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b"")


def test_a_single_colour_transparency_key_is_part_of_the_pixel_identity():
    opaque, keyed = native_pixel_digest(_png16(False)), native_pixel_digest(_png16(True))
    assert opaque[0] != keyed[0] and keyed[1]["transparency_key"] == "03e807d00bb8"
    rgb = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), 9, np.uint8)).save(rgb, format="PNG")
    keyed8 = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), 9, np.uint8)).save(keyed8, format="PNG", transparency=(9, 9, 9))
    assert native_pixel_digest(rgb.getvalue())[0] != native_pixel_digest(keyed8.getvalue())[0]


def test_an_apng_declaring_more_frames_than_it_holds_is_not_intact():
    frames = [Image.fromarray(np.full((8, 8, 3), v, np.uint8)) for v in (10, 200)]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    data = bytearray(buffer.getvalue())
    at = data.find(b"acTL")
    data[at + 4:at + 8] = struct.pack(">I", 65)
    data[at + 12:at + 16] = struct.pack(">I", zlib.crc32(bytes(data[at:at + 12])))
    [record] = PngCarver().carve_stream(bytes(data))
    assert record.completeness == Completeness.PARTIAL
    [intact] = PngCarver().carve_stream(buffer.getvalue())
    assert intact.completeness == Completeness.COMPLETE and json.dumps(intact.pixel_metadata)
