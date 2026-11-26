"""Regressions found by the independent review of the a8c36aa fix branch."""

import json

from comfytrace.carve import PngCarver, ProcessLayerSource, WorkflowCarver
from comfytrace.cli import _merge_duplicate_images
from comfytrace.correlate import correlate
from tests.synthetic import PAGE, FakeProcLayer, api_prompt, noise_image, png_bytes, prompt_request_body

BASE = 0x20000000


def _layer(memory: bytes, missing=()):
    layer = FakeProcLayer(memory.ljust(-(-len(memory) // PAGE) * PAGE, b"\x00"), {})
    layer.map_virtual(BASE, list(range(len(layer.physical) // PAGE)))
    for page in missing:
        del layer.page_table[BASE // PAGE + page]
    return layer


def _seed(record):
    return json.loads(record.embedded_text["prompt"])["3"]["inputs"]["seed"]


def test_png_never_resumes_into_another_image_after_a_lost_header():
    a = png_bytes(noise_image(100, 100, seed=1), {"prompt": json.dumps(api_prompt(seed=111))}, idat_chunk=700)
    b = png_bytes(noise_image(40, 40, seed=2), {"prompt": json.dumps(api_prompt(seed=222))}, idat_chunk=700)
    memory = a[:PAGE] + b"\x00" * PAGE + b
    source = ProcessLayerSource(_layer(memory, missing=(1,)))
    carver = PngCarver()
    first = carver.carve_stream(a[:PAGE], virtual_address=BASE, process_id=42, source=source)
    second = carver.carve_stream(memory[2 * PAGE:], virtual_address=BASE + 2 * PAGE, process_id=42, source=source)
    [img_a], [img_b] = first, second
    assert img_a.spans == [(0, a[:PAGE])] and _seed(img_a) == 111
    assert _seed(img_b) == 222 and img_b.completeness == "COMPLETE"


def test_process_png_tail_beyond_residency_is_page_not_present():
    png = png_bytes(noise_image(100, 100), {}, idat_chunk=50000)
    pages = -(-len(png) // PAGE)
    layer = _layer(png, missing=(pages - 2, pages - 1))
    [record] = PngCarver().carve_stream(png[:PAGE], virtual_address=BASE, process_id=42,
                                        source=ProcessLayerSource(layer))
    start, iend = (pages - 2) * PAGE, 12
    assert [(g.relative_offset, g.length_bytes, str(g.reason)) for g in record.gaps] == [
        (start, len(png) - iend - start, "PAGE_NOT_PRESENT")], "gap is the rest of the known IDAT chunk"


def test_stray_quote_earlier_in_the_run_does_not_hide_a_workflow():
    body = prompt_request_body(api_prompt())
    for prefix in (b'Content-Disposition: inline; name="\n', b'{"note": "overwritten by allocator \n'):
        [record] = WorkflowCarver().carve_stream(prefix + body)
        assert record.raw_bytes == body and record.prompt_format == "API_GRAPH"


def test_duplicate_keys_keep_each_record_graph_equal_to_its_raw_bytes():
    graph_a, graph_b = api_prompt(seed=1), api_prompt(seed=2)
    blob = f'{{"a": 1, "b": {json.dumps(graph_a)}, "a": {json.dumps(graph_b)}}}'.encode()
    records = WorkflowCarver().carve_stream(blob)
    assert records
    for record in records:
        assert record.graph == json.loads(record.raw_bytes)


def test_copy_with_matching_prompt_id_carries_the_link():
    graph = api_prompt()
    first, second = prompt_request_body(graph, "P1"), prompt_request_body(graph, "P2")
    workflows = WorkflowCarver().carve_stream(first + b"\x00" + second)
    [image] = PngCarver().carve_stream(png_bytes(noise_image(32, 32), {"prompt": json.dumps(graph),
                                                                       "prompt_id": "P2"}))
    [link] = correlate([image], workflows).links
    assert link.workflow_id == workflows[1].workflow_id
    assert "PROMPT_ID" in link.matched_keys


def test_torn_copy_of_an_intact_image_is_a_location_not_a_second_finding():
    png = png_bytes(noise_image(100, 100), {"prompt": json.dumps(api_prompt())})
    [intact] = PngCarver().carve_stream(png, virtual_address=BASE, process_id=42)
    [torn] = PngCarver(physical_mode=True).carve_stream(png[:PAGE] + b"\xff" * PAGE, base_physical_offset=0x9000)
    assert torn.completeness == "PARTIAL"
    [merged] = _merge_duplicate_images([intact, torn])
    assert merged is intact and "0x0000000000009000" in intact.physical_offsets


def _batch_pair():
    meta = {"prompt": json.dumps(api_prompt())}
    a, b = (png_bytes(noise_image(100, 100, seed=s), meta) for s in (1, 2))
    shared = next(i for i in range(len(a)) if a[i] != b[i])
    assert shared > 128
    return a, b, shared


def test_torn_process_copy_keeps_its_attribution():
    a, _, shared = _batch_pair()
    [intact] = PngCarver(physical_mode=True).carve_stream(a, base_physical_offset=0x100000)
    [torn] = PngCarver().carve_stream(a[:shared - 8], virtual_address=BASE, process_id=42)
    assert torn.completeness == "PARTIAL"
    assert torn in _merge_duplicate_images([intact, torn])


def test_torn_copy_fitting_two_batch_images_is_not_merged():
    a, b, shared = _batch_pair()
    carver = PngCarver(physical_mode=True)
    intact_a = carver.carve_stream(a, base_physical_offset=0x100000)[0]
    intact_b = carver.carve_stream(b, base_physical_offset=0x200000)[0]
    [torn] = PngCarver(physical_mode=True).carve_stream(a[:shared - 8], base_physical_offset=0x300000)
    kept = _merge_duplicate_images([intact_a, intact_b, torn])
    assert torn in kept
    assert "0x0000000000300000" not in intact_a.physical_offsets + intact_b.physical_offsets


def test_unterminated_nested_candidates_stay_within_the_decode_budget():
    import time

    blob = b'{"a":' * 900 + b'{"class_type": "' + b"x" * 9_000_000
    start = time.perf_counter()
    assert WorkflowCarver().carve_stream(blob) == []
    assert time.perf_counter() - start < 5.0


def test_cheap_fallback_failures_do_not_starve_a_later_workflow():
    body = prompt_request_body(api_prompt())
    for enc in ("utf-8", "utf-16-le"):
        blob = (b"{!" * 6 + body + b" " * 3_000_000).decode().encode(enc)
        [record] = WorkflowCarver().carve_stream(blob)
        assert json.loads(record.raw_bytes.decode(enc)) == json.loads(body)
