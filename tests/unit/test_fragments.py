"""ComfyUI JSON cut at page edges: what a piece still proves and where it is placed."""

import json
from types import SimpleNamespace

from comfytrace.cli import _carve
from comfytrace.evidence import memory_source
from comfytrace.fragments import CUT, api_nodes, carve_fragments, parse_fragment, ui_nodes, ui_parameters
from tests.synthetic import api_prompt

PAGE = 4096


def ui_workflow(seed: int) -> dict:
    """A frontend workflow as ComfyUI saves it: only the KSampler's seed differs between jobs."""
    return {
        "id": "a05a97c8-9275-4862-bc90-c96272da9e7c", "revision": 0, "last_node_id": 9, "last_link_id": 9,
        "nodes": [
            {"id": 4, "type": "CheckpointLoaderSimple", "pos": [26.0, 474.0], "size": [315, 98], "flags": {},
             "order": 0, "mode": 0, "inputs": [],
             "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}, {"name": "CLIP", "type": "CLIP",
                                                                            "links": [3, 5]}],
             "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
             "widgets_values": ["v1-5-pruned-emaonly-fp16.safetensors"]},
            {"id": 3, "type": "KSampler", "pos": [863.0, 186.0], "size": [315, 262], "flags": {}, "order": 4,
             "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 1},
                                   {"name": "positive", "type": "CONDITIONING", "link": 4},
                                   {"name": "negative", "type": "CONDITIONING", "link": 6},
                                   {"name": "latent_image", "type": "LATENT", "link": 2}],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7]}],
             "properties": {"Node name for S&R": "KSampler"},
             "widgets_values": [seed, "fixed", 20, 8, "euler", "simple", 1]},
            {"id": 8, "type": "VAEDecode", "pos": [1209.0, 188.0], "size": [210, 46], "flags": {}, "order": 5,
             "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 7},
                                   {"name": "vae", "type": "VAE", "link": 8}],
             "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9]}],
             "properties": {"Node name for S&R": "VAEDecode"}, "widgets_values": []},
            {"id": 9, "type": "SaveImage", "pos": [1451.0, 189.0], "size": [210, 270], "flags": {}, "order": 6,
             "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 9}], "outputs": [],
             "properties": {}, "widgets_values": ["ComfyUI"]},
        ],
        "links": [[1, 4, 0, 3, 0, "MODEL"], [7, 3, 0, 8, 0, "LATENT"], [9, 8, 0, 9, 0, "IMAGE"]],
        "groups": [], "config": {}, "extra": {"ds": {"scale": 0.76, "offset": [-37.7, -52.8]},
                                              "frontendVersion": "1.57.6"}, "version": 0.4,
    }


def compact(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def spaced(obj) -> bytes:
    return json.dumps(obj).encode()


def test_a_workflow_tail_cut_inside_a_string_still_yields_its_sampler_settings():
    text = spaced(ui_workflow(812940175938402)).decode()
    cut = text.index('"negative"') + 3  # inside the KSampler input's name string
    fragment = parse_fragment(text[cut:] + "R")  # a stray byte after the closing brace, as on a reused page
    assert fragment.cut_before and not fragment.cut_after
    assert fragment.end == len(text) - cut
    params = ui_parameters(ui_nodes(fragment.value))
    assert params == {"seed": 812940175938402, "seed_control": "fixed", "steps": 20, "cfg_scale": 8,
                      "sampler_name": "euler", "scheduler": "simple"}
    assert fragment.value["extra"]["frontendVersion"] == "1.57.6"


def test_a_prompt_head_cut_inside_a_node_keeps_the_nodes_before_the_cut():
    text = compact(api_prompt(seed=5)).decode()
    cut = text.index("golden retriever")
    fragment = parse_fragment(text[:cut])
    assert not fragment.cut_before and fragment.cut_after
    nodes = dict(api_nodes(fragment.value))
    assert nodes["3"]["inputs"]["seed"] == 5
    assert nodes["4"] == api_prompt()["4"]
    assert "text" not in nodes["6"]["inputs"]  # the cut string is dropped, not guessed


def test_values_whose_key_lies_before_the_cut_are_kept_under_the_cut_key():
    fragment = parse_fragment('"type":"CLIP","link":72}],"outputs":[{"name":"CO')
    assert fragment.value == {CUT: [{"type": "CLIP", "link": 72}], "outputs": [{}]}
    assert fragment.cut_before and fragment.cut_after


def _physical(pages: dict[int, bytes]):
    raw = bytearray(14 * PAGE)
    for page, data in pages.items():
        raw[page * PAGE:page * PAGE + len(data)] = data
    return SimpleNamespace(physical_mode=True, process_id=None, source=None, translate=None,
                           blocks=[SimpleNamespace(address=0, data=bytes(raw))], skip_pages=frozenset())


def test_page_sized_pieces_become_partial_copies_or_fragment_records_by_what_they_agree_with():
    x, y, z = ui_workflow(111111111111111), ui_workflow(222222222222222), ui_workflow(333)
    whole_x, whole_y = compact(x), compact(y)
    assert len(whole_x) < PAGE
    common_tail = whole_x[whole_x.index(b'{"id":8'):]
    assert whole_y.endswith(common_tail)
    spaced_y = spaced(y)
    api_head = compact(api_prompt(seed=7))
    api_head = api_head[:api_head.index(b"golden")]
    ingestion = SimpleNamespace(
        spaces=[_physical({1: whole_x, 3: whole_y,
                           5: whole_x[-1500:],                          # inside X only
                           7: spaced_y[spaced_y.index(b'"negative"') + 3:],  # Y, another serialization
                           9: common_tail[5:],                          # inside both X and Y
                           11: api_head,                                # no API graph agrees
                           12: compact(api_prompt(seed=333)),
                           13: spaced(z)[spaced(z).index(b'"negative"') + 3:]})],  # only an API prompt has 333
        recovery=SimpleNamespace(physical=None, pfn=SimpleNamespace(
            copy_origin=lambda pages, size: {"pages": {"PRIVATE" if pages == [5] else "FILE_DATA": 1},
                                             "bytes": size})))
    _, workflows, _ = _carve(ingestion)
    by_offset = {wf.physical_offsets[0]: wf for wf in workflows}
    record_x = by_offset[f"0x{PAGE:016X}"]
    assert record_x.prompt_format == "UI_WORKFLOW"
    partial = [loc for loc in record_x.execution_parameters["source_locations"] if "record_range" in loc]
    [location] = partial
    lo, hi = location["record_range"]
    skipped = int(location["physical_offset"], 16) - 5 * PAGE  # a cut leading token is dropped
    assert 0 <= skipped < 8 and (lo, hi) == (len(whole_x) - 1500 + skipped, len(whole_x))
    assert memory_source(record_x.page_origins, len(record_x.raw_bytes)) == "MEMORY_PARTIAL"

    record_y = by_offset[f"0x{3 * PAGE:016X}"]
    fragments = {wf.physical_offsets[0]: wf for wf in workflows if "fragment" in wf.evidence}
    assert len(fragments) == 4
    one_seed = next(wf for wf in fragments.values() if wf.prompt_format == "UI_FRAGMENT"
                    and wf.execution_parameters.get("seed") == 222222222222222)
    assert one_seed.evidence["fragment"]["attribution"] == "ONE_SEED"
    assert one_seed.evidence["fragment"]["agrees_with"] == [record_y.workflow_id]
    ambiguous = fragments[f"0x{9 * PAGE:016X}"]
    assert ambiguous.evidence["fragment"]["attribution"] == "AMBIGUOUS"
    assert sorted(ambiguous.evidence["fragment"]["agrees_with"]) == sorted([record_x.workflow_id,
                                                                          record_y.workflow_id])
    api = fragments[f"0x{11 * PAGE:016X}"]
    assert api.prompt_format == "API_FRAGMENT"
    assert api.execution_parameters["seed"] == 7 and api.dependencies["model_assets"] == [
        "v1-5-pruned-emaonly.safetensors"]
    assert api.evidence["fragment"]["attribution"] == "NO_REFERENCE_AGREES"
    api_333 = by_offset[f"0x{12 * PAGE:016X}"]
    by_seed = next(wf for wf in fragments.values() if wf.execution_parameters.get("seed") == 333)
    assert by_seed.evidence["fragment"]["agrees_with"] == []
    assert by_seed.evidence["fragment"]["same_seed"] == [api_333.workflow_id]
    assert by_seed.evidence["fragment"]["attribution"] == "ONE_SEED"
    # The single nodes the whole-object carver found inside the API piece are replaced by the piece.
    assert not [wf for wf in workflows if wf.prompt_format == "API_NODE_FRAGMENT"]


def test_a_whole_object_after_stray_bytes_is_not_a_cut_piece_and_invalid_utf8_ends_one():
    whole = compact(api_prompt())
    head = whole[:whole.index(b"golden")]
    tail = compact(ui_workflow(5))[-900:]
    run = b'x"q' + whole + b"q" + head + b"\xff\xc3" + tail
    anchors = [0x1000 + i for i in range(len(run)) if run.startswith((b'"class_type"', b'"inputs"'), i)]
    pieces = [(offset - 0x1000, raw) for offset, raw, _ in carve_fragments(run, 0x1000, [], anchors)]
    assert pieces[0] == (run.index(head, len(whole)), head)
    assert pieces[1][1] == tail[len(tail) - len(pieces[1][1]):] and pieces[1][0] + len(pieces[1][1]) == len(run)
    assert len(pieces) == 2


def test_numbers_outside_json_grammar_do_not_stop_carving():
    fragment = parse_fragment('00, 1e5, -0.5], "class_type": "KSampler", "inputs": {"seed": 007}')
    assert fragment is not None and fragment.value[CUT] == [0, 0, 100000.0, -0.5]


def test_a_lone_surrogate_escape_ends_the_parse():
    text = '"a"}, "3": {"class_type": "KSampler", "inputs": {"text": "x\\ud800y", "seed": 1}}'
    fragment = parse_fragment(text)
    assert fragment.value == {CUT: {}, "3": {"class_type": "KSampler", "inputs": {}}}
    assert fragment.end <= text.index('"x')


def test_mutated_page_pieces_never_crash_carving_or_placement():
    import random

    from comfytrace.carve import IdAllocator, WorkflowCarver
    from comfytrace.correlate import place_fragments
    from comfytrace.fragments import TextFragment

    rng = random.Random(7)
    sources = [spaced(api_prompt(seed=1)), compact(ui_workflow(5))]
    references = [r for n, src in enumerate(sources)
                  for r in WorkflowCarver().carve_stream(b"\0" + src + b"\0", base_physical_offset=n << 20)]
    tokens = [b'"', b"\\", b"{", b"}", b"[", b"]", b":", b",", b"00", b"1e999", b"\\ud800", b"\xff", b"\xc3",
              b"nul", b'"class_type"', b'"inputs"']
    for _ in range(1500):
        src = rng.choice(sources)
        a, b = sorted(rng.sample(range(len(src)), 2))
        run = bytearray(src[a:b])
        for _ in range(rng.randrange(6)):
            i = rng.randrange(len(run) + 1)
            run[i:i] = rng.choice(tokens)
        run = bytes(run)
        anchors = [i for i in range(len(run)) if run.startswith((b'"class_type"', b'"inputs"'), i)]
        pieces: dict[bytes, TextFragment] = {}
        for offset, raw, fragment in carve_fragments(run, 0, [], anchors):
            assert run[offset:offset + len(raw)] == raw
            pieces.setdefault(raw, TextFragment(raw, fragment, [])).physical_offsets.append(offset)
        for record in place_fragments(list(pieces.values()), list(references), [], IdAllocator("FR")):
            json.dumps(record.execution_parameters, ensure_ascii=False).encode()


def test_a_piece_ends_at_its_page_boundary_even_when_the_next_page_is_printable():
    head = compact(ui_workflow(9))[:-200]
    pad = PAGE - 1000
    run = b"\0" * pad + head + b'AAAA,CAAA;;"}' * 200
    anchors = [0x7000 + i for i in range(len(run)) if run.startswith(b'"inputs"', i)]
    pieces = list(carve_fragments(run, 0x7000, [], anchors))
    offset, raw, fragment = pieces[0]
    assert (offset, offset + len(raw)) == (0x7000 + pad, 0x8000) and raw == head[:1000] and fragment.cut_after
    assert all(o >= 0x8000 and o + len(r) <= 0x9000 for o, r, _ in pieces[1:])


def test_a_workflow_stored_as_a_json_string_is_carved_whole_and_dated():
    from comfytrace.carve import WorkflowCarver

    workflow = ui_workflow(6391048257)
    draft = json.dumps({"Comfy.Workflow.Draft.v2:a.json": json.dumps(
        {"data": json.dumps(workflow), "updatedAt": 1789966409517})}).encode()
    inner = json.dumps({"data": json.dumps(workflow), "updatedAt": 1789966409517}).encode()
    carver = WorkflowCarver()
    [record] = carver.carve_stream(b"\0" * 16 + inner + b"\0", base_physical_offset=0x10000)
    assert record.encoding_detected == "UTF-8_JSON_STRING" and record.prompt_format == "UI_WORKFLOW"
    assert record.graph == workflow and record.execution_parameters["saved_at_utc"] == "2026-09-21T04:53:29.517+00:00"
    assert (record.execution_parameters["seed"], record.execution_parameters["seed_control"]) == (6391048257, "fixed")
    assert json.loads(json.loads(record.raw_bytes)) == workflow  # raw bytes are the escaped string as found
    assert carver.carve_stream(b"\0" + draft + b"\0", base_physical_offset=0x40000) == [], (
        "a string nested two levels deep is not decoded twice")


def test_a_json_string_cut_at_a_page_edge_becomes_a_fragment_with_its_escaped_bytes():
    from comfytrace.carve import WorkflowCarver

    escaped = json.dumps(json.dumps(ui_workflow(6391048257))).encode()[1:-1]
    head = escaped[:escaped.index(b'\\"id\\": 8')]
    run = b"\0" * (PAGE - len(head)) + head  # the string's content runs to the page end and stops
    carver = WorkflowCarver()
    assert carver.carve_stream(run + b"\0", base_physical_offset=0x20000) == []
    [piece] = carver.fragments.values()
    assert piece.encoding == "UTF-8_JSON_STRING" and piece.raw == head
    assert piece.physical_offsets == [0x20000 + PAGE - len(head)] and piece.fragment.cut_after
    assert ui_parameters(ui_nodes(piece.fragment.value))["seed"] == 6391048257


def test_served_output_files_are_recorded_with_size_write_time_and_page_origin():
    response = (b"HTTP/1.1 200 OK\x00Content-Disposition: filename=\"ComfyUI_temp_hfkbq_00003_.png\"\x00"
                b"Content-Type: image/png\x00Etag: \"18d742d2a55c42ac-a8ba6\"\x00Content-Length: 691110\x00"
                b"Server: Python/3.13 aiohttp/3.14.3\x00\x00\x00HTTP/1.1 200 OK\x00Etag: \"1-2\"\x00")
    other = response.replace(b"aiohttp", b"nginx").replace(b"hfkbq", b"zzzzz")
    space = _physical({2: response, 4: other, 6: b"x/view?filename=ComfyUI_temp_hfkbq_00003_.png&type=temp&t="
                                                b"1789966409517 "})
    ingestion = SimpleNamespace(spaces=[space], recovery=SimpleNamespace(physical=None, pfn=SimpleNamespace(
        copy_origin=lambda pages, size: {"pages": {"FILE_DATA" if pages == [2] else "PRIVATE": 1}, "bytes": size})))
    _, _, outputs = _carve(ingestion)
    [(record_id, output)] = outputs
    entry = output.to_dict(record_id)
    assert entry["filename"] == "ComfyUI_temp_hfkbq_00003_.png" and entry["folder"] == "temp"
    [version] = entry["versions"]
    assert version["size"] == version["content_length"] == 691110
    assert version["file_mtime_utc"].startswith("2026-09-21T06:40:49.387")
    assert entry["ui_requests"] == [
        {"type": "temp", "subfolder": "", "requested_at_utc": "2026-09-21T04:53:29.517+00:00"}]
    assert [o["pages"] for o in output.page_origins] == [{"FILE_DATA": 1}, {"PRIVATE": 1}]
    from comfytrace.evidence import evidence_document

    [doc_entry] = [r for r in evidence_document([], [], {}, outputs)["records"] if r["kind"] == "output_file"]
    # The name was seen in memory (the URL); the size and write time only in a file-backed response header.
    assert doc_entry["memory_source"] == "MEMORY" and doc_entry["versions"][0]["memory_source"] == "FILE_BACKED"
