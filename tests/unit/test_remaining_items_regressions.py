"""Regressions for the seven remaining-items review defects; partial workflows are unchanged."""

import copy
import json

import numpy as np
import pytest

from comfytrace import evidence, execcache, link, media, memrecover
from comfytrace.carve import IdAllocator
from comfytrace.correlate import correlate
from comfytrace.fragments import ui_parameters, ui_workflow_nodes
from tests.synthetic import api_prompt
from tests.unit.test_execcache import JOB_B, CachePython, cached_job, image_tensor, queue_item
from tests.unit.test_link import _case, _case_regeneration, _photo, _regen
from tests.unit.test_media import fixture
from tests.unit.test_subgraphs import resolve, subgraph_workflow


def test_close_whole_frame_must_not_earn_e2(tmp_path):
    original = _photo(1)
    close = np.minimum(original.astype(np.int16) + 1, 255).astype(np.uint8)
    case = _case(tmp_path, [('PIXBUF-001', 'PIXEL_BUFFER', close)])
    manifest_path = case / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['findings']['workflows'] = [{'workflow_id': 'WFPY-001', 'prompt_format': 'API_GRAPH',
        'completeness': 'COMPLETE', 'execution_parameters': {'prompt_id': 'job'}}]
    manifest_path.write_text(json.dumps(manifest))
    _case_regeneration(case, 'REGEN-001', 'WFPY-001', 'job', original)
    result = link.link(case)
    assert result['memory_images'][0]['best']['identical_pixel_fraction'] < 0.01
    assert [p['tier'] for p in result['promotions']] == ['E3'], result['promotions']


def test_duplicate_queue_copy_does_not_hide_history_timeline():
    prompt = api_prompt()
    bare = queue_item('job', prompt, 0x1000)
    bare.extra_data = {'create_time': 1794065346900}
    bare.history = None
    held = copy.deepcopy(bare)
    held.address = 0x2000
    held.history = {'status': {'messages': [
        ['execution_start', {'prompt_id': 'job', 'timestamp': 1794065346982}],
        ['execution_success', {'prompt_id': 'job', 'timestamp': 1794065421254}]]}}
    [record] = memrecover.workflow_records([bare, held], IdAllocator('WFPY'))
    assert record.execution_parameters['job_timeline'].get('run_seconds') == 74.272, record.execution_parameters


def batch_card():
    prompt = api_prompt(seed=222, batch_size=2)
    py = CachePython()
    item = queue_item(JOB_B, prompt)
    made = cached_job(py, prompt, batch=2)
    entries, _ = execcache.extract(py.memory())
    tie = execcache.image_ties(entries)[(4242, made['impl'])]
    tie['jobs'] = execcache.job_matches(tie['graph'], [item])
    images = memrecover.image_records([
        memrecover.from_tensor(image_tensor(made['impl'], b, seed=b), lambda va: None, tie) for b in (0, 1)])
    workflows = memrecover.workflow_records([item], IdAllocator('WFPY'))
    correlations = correlate(images, workflows)
    return evidence.scorecard([item], workflows, 'ab'*32, correlations.links, images)


def test_cache_batch_control():
    card = batch_card()
    assert [[c['artifact_id'] for c in r['image']['candidates']] for r in card['rows']] == [
        ['TENSOR-001'], ['TENSOR-002']]


def test_link_refresh_preserves_known_batch_assignment():
    card = batch_card()
    links = {'jobs': [{'prompt_id': JOB_B, 'image_status': 'OBSERVED_EXACT', 'observed': [
        {'artifact_id': 'TENSOR-001', 'kind': 'TENSOR_IMAGE', 'agreement': 'ROWS'},
        {'artifact_id': 'TENSOR-002', 'kind': 'TENSOR_IMAGE', 'agreement': 'ROWS'}]}]}
    updated = evidence.apply_image_links(card, links)
    assert [[c['artifact_id'] for c in r['image']['candidates']] for r in updated['rows']] == [
        ['TENSOR-001'], ['TENSOR-002']]
    assert evidence.apply_image_links(copy.deepcopy(updated), links) == updated


def test_link_refresh_includes_candidate_completeness():
    card = batch_card()
    links = {'memory_images': [{'artifact_id': 'BITMAP-001', 'complete': False}],
        'jobs': [{'prompt_id': JOB_B, 'image_status': 'OBSERVED_PARTIAL_OR_SCALED', 'observed': [
            {'artifact_id': 'BITMAP-001', 'kind': 'DECODED_BITMAP', 'agreement': 'ROWS'}]}]}
    updated = evidence.apply_image_links(card, links, {'BITMAP-001': 'MEMORY'})
    candidate = next(c for c in updated['rows'][0]['image']['candidates'] if c['artifact_id'] == 'BITMAP-001')
    assert candidate.get('completeness') == 'PARTIAL', candidate


@pytest.mark.parametrize(('attribution', 'identity', 'expected_batches'), [
    ('SUPPORTED', {'verified': True, 'prompt_id': JOB_B}, [1]),
    ('SUPPORTED', {'prompt_id': JOB_B}, []),
    ('SUPPORTED', {'verified': True, 'prompt_id': 'different-job'}, []),
    ('AMBIGUOUS', {'verified': True, 'prompt_id': JOB_B}, []),
])
def test_regenerated_candidate_requires_matching_output_position(attribution, identity, expected_batches):
    card = batch_card()
    node_id = card['rows'][1]['output_node_id']
    links = {
        'memory_images': [{
            'artifact_id': 'BITMAP-001', 'complete': False, 'output_attribution': attribution,
            'selected_raster': {'node_id': node_id, 'batch_index': 1, **identity},
        }],
        'jobs': [{'prompt_id': JOB_B, 'image_status': 'OBSERVED_PARTIAL_OR_SCALED', 'observed': [
            {'artifact_id': 'BITMAP-001', 'kind': 'DECODED_BITMAP', 'agreement': 'ROWS'},
        ]}],
    }
    updated = evidence.apply_image_links(card, links)
    credited_batches = [
        row['batch_index'] for row in updated['rows']
        if any(c['artifact_id'] == 'BITMAP-001' and c['role'] == 'OUTPUT' for c in row['image']['candidates'])
    ]
    assert credited_batches == expected_batches


def test_job_linked_image_still_earns_its_output_from_a_verified_replay_match():
    card = batch_card()
    node_id = card['rows'][1]['output_node_id']
    job_level = {'artifact_id': 'PNG-001', 'basis': 'CORRELATION_STRICT', 'carve_type': 'ENCODED_PNG',
                 'completeness': 'COMPLETE', 'memory_source': 'FILE_CACHE', 'role': 'JOB'}
    for row in card['rows']:
        row['image']['candidates'].append(dict(job_level))
    links = {
        'memory_images': [
            {'artifact_id': 'PNG-001', 'complete': True, 'output_attribution': 'SUPPORTED',
             'selected_raster': {'node_id': node_id, 'batch_index': 1, 'verified': True, 'prompt_id': JOB_B}},
            # The execution cache places TENSOR-001 at batch 0; a replay pointing it at batch 1 moves nothing.
            {'artifact_id': 'TENSOR-001', 'complete': True, 'output_attribution': 'SUPPORTED',
             'selected_raster': {'node_id': node_id, 'batch_index': 1, 'verified': True, 'prompt_id': JOB_B}},
        ],
        'jobs': [{'prompt_id': JOB_B, 'image_status': 'OBSERVED_EXACT', 'observed': [
            {'artifact_id': 'PNG-001', 'kind': 'ENCODED_PNG', 'agreement': 'ROWS'},
            {'artifact_id': 'TENSOR-001', 'kind': 'TENSOR_IMAGE', 'agreement': 'ROWS'}]}],
    }
    updated = evidence.apply_image_links(card, links)
    first, second = (row['image']['candidates'] for row in updated['rows'])
    assert [(c['artifact_id'], c['basis'], c['role']) for c in first if c['artifact_id'] == 'PNG-001'] == [
        ('PNG-001', 'CORRELATION_STRICT', 'JOB')], "batch 0 keeps only the job-level link"
    replayed = [c for c in second if c['artifact_id'] == 'PNG-001' and c['basis'] == 'REGENERATION_COMPARISON']
    assert [(c['role'], c['also_linked_by']) for c in replayed] == [('OUTPUT', ['CORRELATION_STRICT'])]
    assert any(c['artifact_id'] == 'PNG-001' and c['basis'] == 'CORRELATION_STRICT' for c in second)
    assert updated['rows'][1]['image']['status'] == 'LINKED'
    assert not any(c['artifact_id'] == 'TENSOR-001' for c in second), "the cache's placement is kept"
    assert evidence.apply_image_links(copy.deepcopy(updated), links) == updated


def test_subgraph_parameter_reader_honors_connected_external_seed():
    ui = subgraph_workflow()
    ui['nodes'].append({'id': 30, 'type': 'PrimitiveInt', 'pos': [0, 0], 'widgets_values': [999, 'fixed'],
        'outputs': [{'name': 'INT', 'type': 'INT', 'links': [105]}]})
    ui['nodes'][1]['inputs'][1]['link'] = 105
    ui['links'].append([105, 30, 0, 10, 1, 'INT'])
    resolved = resolve(ui)['samplers'][0]['fields']['seed']['value']
    assert resolved == 999
    assert ui_parameters(ui_workflow_nodes(ui)).get('seed') == resolved


def test_adjacent_mp4_files_are_separate_containers():
    first = fixture('video.mp4')
    second = first.replace(b'4242', b'7777')
    parsed = media.parse_mp4(first + second + bytes(64))
    assert parsed.complete
    assert parsed.end == len(first), (parsed.end, len(first), parsed.text)
    assert json.loads(parsed.text['prompt'])['3']['inputs']['seed'] == 4242


def test_intact_mp4_at_end_of_source_is_complete():
    data = fixture('video.mp4')
    parsed = media.parse_mp4(data)
    assert parsed.complete, (parsed.end, len(data), parsed.halt)


def test_subgraph_standard_control():
    fields = resolve(subgraph_workflow())['samplers'][0]['fields']
    assert fields['seed']['value'] == 777
    assert fields['model_name']['value'] == 'model.safetensors'


@pytest.mark.parametrize("changed_pixels", [0, 1])
def test_e2_uses_unrounded_pixel_equality(tmp_path, changed_pixels):
    original = _photo(1)
    memory = original.copy()
    if changed_pixels:
        memory[0, 0, 0] ^= 1
    case = _case(tmp_path, [("PIXBUF-001", "PIXEL_BUFFER", memory)])
    result = link.link(case, _regen(tmp_path, [("job", original)]))
    # A single changed pixel rounds to an identical fraction of 1.0.
    assert result["memory_images"][0]["best"]["identical_pixel_fraction"] == 1.0
    assert result["jobs"][0]["image_status"] == (
        "OBSERVED_EXACT" if not changed_pixels else "OBSERVED_PARTIAL_OR_SCALED")


def test_timeline_fields_merge_in_either_observation_order():
    bare = queue_item("job", api_prompt())
    bare.extra_data = {"create_time": 1794065346900}
    bare.history = None
    history = copy.deepcopy(bare)
    history.extra_data = {}
    history.history = {"status": {"messages": [
        ["execution_start", {"timestamp": 1794065346982}],
        ["execution_success", {"timestamp": 1794065421254}]]}}
    results = []
    for items in ([bare, history], [history, bare]):
        [record] = memrecover.workflow_records(items, IdAllocator("WFPY"))
        results.append(record.execution_parameters["job_timeline"])
    assert results[0] == results[1]
    assert results[0]["queued_utc"] == "2026-11-07T15:29:06.900Z"
    assert results[0]["run_seconds"] == 74.272


def test_conflicting_timeline_endpoints_do_not_keep_a_runtime():
    items = []
    for stamp in (1794065346982, 1794065347982, 1794065346982):
        item = queue_item("job", api_prompt())
        item.history = {"status": {"messages": [
            ["execution_start", {"timestamp": stamp}],
            ["execution_success", {"timestamp": 1794065421254}]]}}
        items.append(item)
    [record] = memrecover.workflow_records(items, IdAllocator("WFPY"))
    timeline = record.execution_parameters["job_timeline"]
    assert "started_utc" not in timeline and "run_seconds" not in timeline
    assert len(timeline["conflicts"]["started_utc"]) == 2


@pytest.mark.parametrize("complete", [True, False])
def test_scorecard_added_candidates_keep_completeness_and_source(complete):
    card = batch_card()
    links = {"memory_images": [{"artifact_id": "BITMAP-001", "complete": complete}],
             "jobs": [{"prompt_id": JOB_B, "observed": [
                 {"artifact_id": "BITMAP-001", "kind": "DECODED_BITMAP", "agreement": "ROWS"}]}]}
    updated = evidence.apply_image_links(card, links, {"BITMAP-001": "MEMORY"})
    for row in updated["rows"]:
        candidate = next(c for c in row["image"]["candidates"] if c["artifact_id"] == "BITMAP-001")
        assert candidate["completeness"] == ("COMPLETE" if complete else "PARTIAL")
        assert candidate["memory_source"] == "MEMORY"
    assert evidence.apply_image_links(copy.deepcopy(updated), links, {"BITMAP-001": "MEMORY"}) == updated


@pytest.mark.parametrize("broken", [False, True])
def test_subgraph_external_scalar_links_and_broken_links(broken):
    ui = subgraph_workflow()
    ui["nodes"].append({"id": 30, "type": "PrimitiveInt", "pos": [0, 0], "widgets_values": [999, "fixed"],
                        "outputs": [{"name": "INT", "type": "INT", "links": [105]}]})
    ui["nodes"][1]["inputs"][1]["link"] = 105
    if not broken:
        ui["links"].append([105, 30, 0, 10, 1, "INT"])
    params = ui_parameters(ui_workflow_nodes(ui))
    assert params.get("seed") == (None if broken else 999)
    assert params["steps"] == 20
    assert params["sampler_name"] == "dpmpp_2m"


def test_adjacent_mp4s_export_with_their_own_prompt():
    first = fixture("video.mp4")
    second = first.replace(b"4242", b"7777")
    carver = media.MediaCarver()
    candidates = carver.carve_stream(first + second, base_physical_offset=0)
    records = [carver.record(c) for c in candidates]
    assert [r.recovered_bytes for r in records] == [len(first), len(second)]
    assert [json.loads(r.embedded_text["prompt"])["3"]["inputs"]["seed"] for r in records] == [4242, 7777]


@pytest.mark.parametrize("cut", [1, 7, 20])
def test_mp4_cut_inside_final_box_stays_partial(cut):
    data = fixture("video.mp4")
    parsed = media.parse_mp4(data[:-cut])
    assert not parsed.complete and parsed.more
    assert parsed.halt == "TRUNCATED_STREAM"


def test_subgraph_normalization_preserves_custom_named_widgets():
    ui = subgraph_workflow()
    ui["nodes"] = [{"id": 50, "type": "CustomSampler", "pos": [0, 0],
                    "widgets_values_named": {"seed": 999, "steps": 30}}]
    ui["links"] = []
    assert ui_parameters(ui_workflow_nodes(ui)) == {"seed": 999, "steps": 30}

