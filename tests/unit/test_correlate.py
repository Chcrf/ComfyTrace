import itertools
import json
import random

from comfytrace.carve import IdAllocator, PngCarver, WorkflowCarver
from comfytrace.correlate import INPUT_REVISION_UNVERIFIED, KeyMatcher, correlate, png_workflows
from comfytrace.evidence import scorecard
from comfytrace.pyheap import QueueItem
from tests.synthetic import api_prompt, garbage, noise_image, png_bytes, prompt_request_body

IMAGE_IDS = IdAllocator("CARVE-PNG")
PROMPT_ID = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"


def carve_image(graph=None, extra_text=None, size=(64, 48), seed=1, truncate=False):
    text = {"prompt": json.dumps(graph)} if graph is not None else {}
    text.update(extra_text or {})
    png = png_bytes(noise_image(*size, seed=seed), text, idat_chunk=512)
    if truncate:
        png = png[: len(png) - 600]
    [record] = PngCarver(IMAGE_IDS).carve_stream(png)
    return record


_BASES = itertools.count(0x100000, 0x100000)


def carve_workflow(blob: bytes, carver=None):
    [record] = (carver or WorkflowCarver()).carve_stream(blob, base_physical_offset=next(_BASES))
    return record


def test_canonical_json_hash_strict_link():
    graph = api_prompt()
    image = carve_image(graph)
    reordered = json.dumps(dict(reversed(list(graph.items()))), indent=4).encode()
    workflow = carve_workflow(reordered)
    assert KeyMatcher().canonicalize_prompt_json(graph) == KeyMatcher().canonicalize_prompt_json(
        json.loads(reordered)
    )
    link = KeyMatcher().evaluate_link(workflow, image)
    assert link.link_class == "STRICT"
    assert "CANONICAL_PROMPT_HASH" in link.matched_keys

    result = correlate([image], [workflow])
    [finding] = result.findings
    assert finding.tier == "E2"
    assert finding.artifact_ids == [image.artifact_id, workflow.workflow_id]
    assert result.tier_summary["E2"] == 1 and result.tier_summary["E5"] == 0
    assert image.link_status == "STRICT" and image.associated_workflow_id == workflow.workflow_id


def test_prompt_id_runtime_key_matching():
    image = carve_image(extra_text={"prompt_id": PROMPT_ID})
    workflow = carve_workflow(prompt_request_body(api_prompt(seed=999), prompt_id=PROMPT_ID))
    link = KeyMatcher().evaluate_link(workflow, image)
    assert (link.link_class, link.matched_keys) == ("STRICT", ["PROMPT_ID"])

    other = carve_workflow(prompt_request_body(api_prompt(seed=999), prompt_id="00000000-0000-0000-0000-000000000000"))
    assert KeyMatcher().evaluate_link(other, image).link_class == "UNASSOCIATED"


def test_prompt_id_cannot_override_a_conflicting_graph():
    graph = api_prompt(seed=999)
    workflow = carve_workflow(prompt_request_body(graph, prompt_id=PROMPT_ID))
    for contradictory in (api_prompt(seed=1), api_prompt(seed=999)):
        contradictory["9"]["inputs"]["filename_prefix"] = "different"
        image = carve_image(contradictory, extra_text={"prompt_id": PROMPT_ID})
        link = KeyMatcher().evaluate_link(workflow, image)
        assert link.link_class == "UNASSOCIATED"
        assert "disagree" in link.confidence_notes


def test_prompt_id_accepts_comfyui_runtime_annotations_but_compares_known_input_revisions():
    submitted = api_prompt()
    recorded = api_prompt()
    embedded = api_prompt()
    for graph in (submitted, recorded, embedded):
        graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    recorded["10"]["is_changed"] = ["known-input-fingerprint"]
    embedded["10"]["is_changed"] = ["other-input-fingerprint"]
    image = carve_image(embedded, extra_text={"prompt_id": PROMPT_ID})

    before_run = carve_workflow(prompt_request_body(submitted, prompt_id=PROMPT_ID))
    assert KeyMatcher().evaluate_link(before_run, image).matched_keys == ["EXECUTION_GRAPH_MATCH", "PROMPT_ID"]

    after_run = carve_workflow(prompt_request_body(recorded, prompt_id=PROMPT_ID))
    assert KeyMatcher().evaluate_link(after_run, image).link_class == "UNASSOCIATED"


def test_input_fingerprint_added_at_runtime_preserves_independent_graph_match():
    submitted = api_prompt()
    submitted["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    embedded = json.loads(json.dumps(submitted))
    embedded["10"]["is_changed"] = ["runtime-fingerprint"]
    image = carve_image(embedded)
    workflow = carve_workflow(json.dumps(submitted).encode())
    assert workflow.canonical_json_sha256 != KeyMatcher()._embedded_for(image).canonical_sha256
    link = KeyMatcher().evaluate_link(workflow, image)
    assert (link.link_class, link.matched_keys) == ("STRICT", ["EXECUTION_GRAPH_MATCH", INPUT_REVISION_UNVERIFIED])


def test_exact_observed_input_revision_beats_an_unbound_less_detailed_graph():
    submitted = api_prompt()
    submitted["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    observed = json.loads(json.dumps(submitted))
    observed["10"]["is_changed"] = ["runtime-fingerprint"]
    carver = WorkflowCarver()
    earlier = carve_workflow(json.dumps(submitted).encode(), carver)
    exact = carve_workflow(json.dumps(observed).encode(), carver)

    image = carve_image(observed)
    result = correlate([image], [earlier, exact])

    assert [(link.link_class, link.workflow_id) for link in result.links] == [("STRICT", exact.workflow_id)]

    # Without a received fingerprint, a workflow carrying extra facts is still a possible source.
    image = carve_image(submitted)
    assert {link.link_class for link in correlate([image], [earlier, exact]).links} == {"AMBIGUOUS"}

    # An exact graph can be linked independently of a submission that lacks the input revision.
    bound = carve_workflow(prompt_request_body(submitted, prompt_id=PROMPT_ID), carver)
    image = carve_image(observed)
    assert {link.workflow_id for link in correlate([image], [bound, exact]).links} == {exact.workflow_id}

    # A pre-run copy and an exact annotated copy of one identified submission are one candidate.
    bound_exact = carve_workflow(prompt_request_body(observed, prompt_id=PROMPT_ID), carver)
    image = carve_image(observed, extra_text={"prompt_id": PROMPT_ID})
    assert {link.workflow_id for link in correlate([image], [bound, bound_exact]).links} == {bound_exact.workflow_id}


def test_unbound_input_revision_cannot_be_credited_to_a_prerun_submission():
    submitted = api_prompt()
    submitted["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    observed = json.loads(json.dumps(submitted))
    observed["10"]["is_changed"] = ["previous-input-revision"]
    carver = WorkflowCarver()
    earlier = carve_workflow(json.dumps(submitted).encode(), carver)
    bound = carve_workflow(prompt_request_body(submitted, prompt_id=PROMPT_ID), carver)
    bound.evidence["prompt_id"] = PROMPT_ID
    image = carve_image(observed)

    result = correlate([image], [bound, earlier])

    [link] = result.links
    assert link.workflow_id == earlier.workflow_id and INPUT_REVISION_UNVERIFIED in link.matched_keys
    finding = next(f for f in result.findings if image.artifact_id in f.artifact_ids)
    assert bound.workflow_id not in finding.artifact_ids
    item = QueueItem(4242, PROMPT_ID, 0, 0x1000, 1, 5, submitted)
    item.history = {"outputs": {"9": {"images": [{"filename": "new.png"}]}}}
    [row] = scorecard([item], [bound, earlier], "fixture", result.links, [image])["rows"]
    assert row["image"]["candidates"] == []


def test_an_imported_comfyui_png_needs_independent_evidence_of_its_origin():
    donor = api_prompt(seed=9032)
    image = carve_image(donor)
    graph = api_prompt(seed=9033)
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "imported.png"},
                   "is_changed": [image.container_sha256]}
    carver = WorkflowCarver()
    consumer = carve_workflow(json.dumps(graph).encode(), carver)
    retained_text = png_workflows([image], IdAllocator("WFPNG"))

    result = correlate([image], [consumer, *retained_text])

    assert result.links == []
    assert image.evidence["workflow_input_uses"] == [{"workflow_id": consumer.workflow_id, "input_node_id": "10"}]
    finding = next(f for f in result.findings if image.artifact_id in f.artifact_ids)
    assert finding.tier == "UNTIERED" and "records these file bytes as an input" in finding.rationale

    independent = carve_workflow(json.dumps(donor).encode(), carver)
    result = correlate([image], [consumer, independent, *retained_text])
    [link] = result.links
    assert (link.workflow_id, link.matched_keys) == (independent.workflow_id, ["CANONICAL_PROMPT_HASH"])
    assert image.evidence["workflow_input_uses"][0]["workflow_id"] == consumer.workflow_id


def test_partial_graph_can_contradict_a_matching_prompt_id():
    workflow = carve_workflow(prompt_request_body(api_prompt(seed=999)))
    workflow.completeness = "PARTIAL"
    workflow.graph = {"3": {"class_type": "KSampler", "inputs": {"seed": 999}}}
    image = carve_image(api_prompt(seed=1), extra_text={"prompt_id": PROMPT_ID})
    assert KeyMatcher().evaluate_link(workflow, image).link_class == "UNASSOCIATED"
    result = correlate([image], [workflow])
    finding = next(f for f in result.findings if image.artifact_id in f.artifact_ids)
    assert finding.tier == "UNTIERED" and "execution graphs disagree" in finding.rationale
    assert image.evidence["attribution_conflicts"] == [
        {"workflow_id": workflow.workflow_id, "reason": "PROMPT_ID_GRAPH_CONFLICT"},
    ]


def test_different_known_prompt_ids_veto_graph_and_parameter_matches():
    prompt = api_prompt()
    image = carve_image(prompt, extra_text={"prompt_id": PROMPT_ID})
    for denoise in (1.0, 0.9):
        prompt["3"]["inputs"]["denoise"] = denoise
        workflow = carve_workflow(prompt_request_body(prompt, prompt_id="00000000-0000-0000-0000-000000000000"))
        assert KeyMatcher().evaluate_link(workflow, image).link_class == "UNASSOCIATED"


def test_repeated_graph_needs_a_runtime_key_to_identify_one_job():
    graph = api_prompt()
    other_id = "00000000-0000-0000-0000-000000000000"
    carver = WorkflowCarver()
    first = carve_workflow(prompt_request_body(graph, prompt_id=PROMPT_ID), carver)
    second = carve_workflow(prompt_request_body(graph, prompt_id=other_id), carver)
    copy = carve_workflow(json.dumps(graph).encode(), carver)
    ambiguous = carve_image(graph)
    first_image = carve_image(graph, extra_text={"prompt_id": PROMPT_ID})
    second_image = carve_image(graph, extra_text={"prompt_id": other_id})

    result = correlate([ambiguous, first_image, second_image], [first, second, copy])

    assert ambiguous.link_status == "AMBIGUOUS"
    assert set(ambiguous.ambiguous_candidate_workflows) >= {first.workflow_id, second.workflow_id}
    for image, expected, excluded in ((first_image, first, second), (second_image, second, first)):
        assert image.associated_workflow_id == expected.workflow_id
        [finding] = [f for f in result.findings if image.artifact_id in f.artifact_ids]
        assert finding.tier == "E2" and excluded.workflow_id not in finding.artifact_ids


def test_supported_link_requires_full_agreement_without_contradiction():
    image = carve_image(api_prompt(seed=5, text="variant A"))
    same_params = carve_workflow(json.dumps(api_prompt(seed=5, text="variant A", steps=20)).encode()
                                 .replace(b'"denoise": 1.0', b'"denoise": 0.9'))
    link = KeyMatcher().evaluate_link(same_params, image)
    assert link.link_class == "SUPPORTED"
    assert {"SEED", "SAMPLER_NAME", "STEPS", "WIDTH", "HEIGHT"} <= set(link.matched_keys)

    contradicting = carve_workflow(json.dumps(api_prompt(seed=5, text="variant B")).encode())
    assert KeyMatcher().evaluate_link(contradicting, image).link_class == "UNASSOCIATED"
    wrong_size = carve_workflow(json.dumps(api_prompt(seed=5, text="variant A", width=512)).encode())
    assert KeyMatcher().evaluate_link(wrong_size, image).link_class == "UNASSOCIATED"
    bare = carve_image(None, seed=3)
    assert KeyMatcher().evaluate_link(same_params, bare).link_class == "UNASSOCIATED"


def test_partial_image_link_forms_e3_and_unlinked_workflows_form_e5_e6():
    graph = api_prompt(seed=11)
    partial = carve_image(graph, truncate=True)
    assert partial.completeness == "PARTIAL"
    carver = WorkflowCarver()
    linked = carve_workflow(json.dumps(graph).encode(), carver)
    lonely = carve_workflow(json.dumps(api_prompt(seed=12, text="other")).encode(), carver)
    fragment = carve_workflow(json.dumps({"class_type": "KSampler", "inputs": {"seed": 3}}).encode(), carver)
    result = correlate([partial], [linked, lonely, fragment])
    tiers = {tuple(f.artifact_ids): f.tier for f in result.findings}
    assert tiers == {
        (partial.artifact_id, linked.workflow_id): "E3",
        (lonely.workflow_id,): "E5",
        (fragment.workflow_id,): "E6",
    }


def test_ambiguous_image_multiple_candidate_workflows():
    params = dict(seed=77, text="shared prompt")
    image = carve_image(api_prompt(**params))
    carver = WorkflowCarver()
    wf_a = carve_workflow(json.dumps(api_prompt(**params)).encode().replace(b'"denoise": 1.0', b'"denoise": 0.5'),
                          carver)
    wf_b = carve_workflow(json.dumps(api_prompt(**params)).encode().replace(b'"denoise": 1.0', b'"denoise": 0.6'),
                          carver)
    result = correlate([image], [wf_a, wf_b])
    image_finding = next(f for f in result.findings if image.artifact_id in f.artifact_ids)
    assert image_finding.tier == "UNTIERED"
    assert image.link_status == "AMBIGUOUS"
    assert image.ambiguous_candidate_workflows == sorted([wf_a.workflow_id, wf_b.workflow_id])
    assert {link.link_class for link in result.links} == {"AMBIGUOUS"}
    assert len(image_finding.link_ids) == 2
    wf_tiers = {f.artifact_ids[0]: f.tier for f in result.findings if f is not image_finding}
    assert wf_tiers == {wf_a.workflow_id: "E5", wf_b.workflow_id: "E5"}
    assert result.tier_summary["untiered_ambiguous_images"] == 1


def test_strict_beats_supported_without_ambiguity():
    graph = api_prompt(seed=21)
    image = carve_image(graph)
    carver = WorkflowCarver()
    exact = carve_workflow(json.dumps(graph).encode(), carver)
    near = carve_workflow(json.dumps(graph).encode().replace(b'"denoise": 1.0', b'"denoise": 0.7'), carver)
    result = correlate([image], [exact, near])
    assert image.link_status == "STRICT" and image.associated_workflow_id == exact.workflow_id
    assert [link.workflow_id for link in result.links] == [exact.workflow_id]


def test_multi_workflow_negative_test_zero_false_strict_links():
    rng = random.Random(1234)
    pairs = 0
    false_strict = 0
    for trial in range(40):
        graphs = [api_prompt(seed=rng.randrange(2**48), text=f"scene {trial}-{k}", steps=rng.choice([20, 25]))
                  for k in range(5)]
        images = [carve_image(g, size=(16, 12), seed=k) for k, g in enumerate(graphs)]
        recovered_ids = rng.sample(range(5), 5)
        carver = WorkflowCarver()
        workflows = [carve_workflow(json.dumps(graphs[k]).encode(), carver) for k in recovered_ids]
        truth = {w.workflow_id: recovered_ids[n] for n, w in enumerate(workflows)}
        matcher = KeyMatcher()
        for w in workflows:
            for k, image in enumerate(images):
                pairs += 1
                if matcher.evaluate_link(w, image).link_class == "STRICT" and truth[w.workflow_id] != k:
                    false_strict += 1
    assert pairs == 1000
    assert false_strict == 0


def test_batch_generation_multi_image_findings():
    graph = api_prompt(seed=31, batch_size=4)
    images = [carve_image(graph, seed=k) for k in range(4)]
    workflow = carve_workflow(json.dumps(graph).encode())
    result = correlate(images, [workflow])
    tiers = [f.tier for f in result.findings]
    assert tiers == ["E2"] * 4, "a batch is not ambiguity; the workflow is not separately E5"
    assert all(f.artifact_ids[1] == workflow.workflow_id for f in result.findings)
    assert result.tier_summary["E2"] == 4 and result.tier_summary["E5"] == 0


def test_unassociated_images_never_receive_tier_e6():
    images = [carve_image(None, seed=k) for k in range(3)] + [carve_image(None, seed=9, truncate=True)]
    fragment = carve_workflow(garbage(50) + json.dumps({"class_type": "SaveImage", "inputs": {}}).encode())
    result = correlate(images, [fragment])
    for finding in result.findings:
        if any(a.startswith("CARVE-PNG") for a in finding.artifact_ids):
            assert finding.tier == "UNTIERED"
    assert result.tier_summary["untiered_orphan_images"] == 4
    assert result.tier_summary["E6"] == 1


def test_byte_distinct_copies_of_one_graph_link_once_not_ambiguous():
    graph = api_prompt()
    as_float = json.loads(json.dumps(graph), parse_int=float)
    blob = prompt_request_body(graph, None) + b"\x00" + json.dumps(as_float).encode()
    workflows = WorkflowCarver().carve_stream(blob)
    assert len(workflows) == 2 and len({w.canonical_json_sha256 for w in workflows}) == 1
    [image] = PngCarver().carve_stream(png_bytes(noise_image(64, 48), {"prompt": json.dumps(as_float)}))
    result = correlate([image], workflows)
    assert [link.link_class for link in result.links] == ["STRICT"]
    [finding] = result.findings
    assert finding.artifact_ids == [image.artifact_id, *(w.workflow_id for w in workflows)]
    assert result.tier_summary["E5"] == 0


def test_the_memory_only_view_drops_file_backed_records_and_leaves_the_full_view_alone():
    from comfytrace import evidence

    graph = api_prompt()
    image = carve_image(graph)
    carver = WorkflowCarver()
    workflow = carve_workflow(json.dumps(graph).encode(), carver)
    partial = carve_workflow(json.dumps(api_prompt(seed=9)).encode(), carver)
    image.page_origins = [{"pages": {"FILE_DATA": 2}, "bytes": image.recovered_bytes}]  # a file-cache PNG
    workflow.page_origins = [{"pages": {"PRIVATE": 1}, "bytes": len(workflow.raw_bytes)}]
    partial.page_origins = [{"pages": {"FILE_DATA": 1}, "bytes": len(partial.raw_bytes)},
                            {"pages": {"FREED": 1}, "bytes": 40}]  # only a piece of it is in memory
    full = correlate([image], [workflow, partial]).tier_summary
    assert (full["E2"], full["E5"]) == (1, 1)
    images, workflows = evidence.memory_view([image], [workflow, partial])
    memory_only = correlate(images, workflows).tier_summary
    assert images == [] and (memory_only["E2"], memory_only["E5"], memory_only["E6"]) == (0, 1, 1)
    assert image.link_status == "STRICT" and image.associated_workflow_id == workflow.workflow_id
    assert partial.completeness == "COMPLETE"


def test_images_with_no_comfyui_connection_are_one_group_and_the_rest_stay_orphans():
    workflow = carve_workflow(json.dumps(api_prompt(width=512, height=512)).encode())
    icon, job_size, other = (carve_image(None, size=s, seed=k) for k, s in enumerate([(256, 256), (512, 512),
                                                                                       (300, 200)]))
    from_process = carve_image(None, size=(300, 200), seed=7)
    from_process.process_id = 1420
    bitmap = carve_image(None, size=(256, 120), seed=8)  # a half-size preview, in part: its width counts
    bitmap.carve_type = "DECODED_BITMAP"
    with_text = carve_image(api_prompt(seed=5, width=640, height=640), size=(64, 48), seed=9)
    images = [icon, job_size, other, from_process, bitmap, with_text]
    result = correlate(images, [workflow])
    [group] = [f for f in result.findings if f.rationale.startswith("Other images in memory")]
    assert group.artifact_ids == [icon.artifact_id, other.artifact_id] and group.tier == "UNTIERED"
    assert result.tier_summary["untiered_other_images"] == 2
    assert result.tier_summary["untiered_orphan_images"] == 4  # job size, process, bitmap, ComfyUI text
    # With no job size known, nothing tells an output from another picture: every image stays an orphan.
    alone = correlate([carve_image(None, size=(16, 16), seed=3)], [])
    assert (alone.tier_summary["untiered_orphan_images"], alone.tier_summary["untiered_other_images"]) == (1, 0)


def test_versions_of_one_ui_workflow_are_one_finding_and_another_workflow_is_its_own():
    from tests.unit.test_fragments import ui_workflow

    carver = WorkflowCarver()
    first, second = ui_workflow(111), ui_workflow(222)
    second["nodes"][0]["pos"] = [40.0, 500.0]  # moved on the canvas and re-seeded: the same workflow
    different = ui_workflow(333)
    different["links"] = different["links"][:-1]
    records = [carve_workflow(json.dumps(w).encode(), carver) for w in (first, second, different)]
    assert {r.prompt_format for r in records} == {"UI_WORKFLOW"}
    result = correlate([], records)
    assert [f.artifact_ids for f in result.findings] == [[records[0].workflow_id, records[1].workflow_id],
                                                          [records[2].workflow_id]]
    assert "2 recovered UI records share the same workflow topology" in result.findings[0].rationale and "111, 222" in \
        result.findings[0].rationale
    assert result.tier_summary["E6"] == 2



def test_page_fragments_are_grouped_by_the_seeds_they_agree_with():
    import dataclasses

    base = carve_workflow(json.dumps(api_prompt()).encode())

    def fragment(n, seeds):
        attribution = "ONE_SEED" if len(seeds) == 1 else "AMBIGUOUS" if seeds else "NO_REFERENCE_AGREES"
        return dataclasses.replace(base, workflow_id=f"WF-{n:03d}", prompt_format="UI_FRAGMENT",
                                   completeness="PARTIAL", canonical_json_sha256=f"{n}",
                                   evidence={"fragment": {"agreeing_seeds": seeds, "attribution": attribution}})

    records = [fragment(1, [7]), fragment(2, []), fragment(3, [7]), fragment(4, [7, 8]), fragment(5, [])]
    result = correlate([], records)
    assert [f.artifact_ids for f in result.findings] == [["WF-001", "WF-003"], ["WF-002", "WF-005"], ["WF-004"]]
    assert "agreeing only with seed 7" in result.findings[0].rationale
    assert "agree with no recovered workflow" in result.findings[1].rationale
    assert result.tier_summary["E6"] == 3


def test_ui_subgraphs_keep_distinct_canvas_and_inner_topologies():
    carver = WorkflowCarver()

    def blueprint(uid, name):
        return {"revision": 0, "last_node_id": 1, "last_link_id": 0, "links": [], "version": 0.4,
                "nodes": [{"id": 1, "type": uid, "pos": [0, 0], "size": [270, 420], "inputs": [], "outputs": []}],
                "definitions": {"subgraphs": [{"id": uid, "name": name, "nodes": [{"id": 2, "type": "ImageBlend"}]}]}}

    first = blueprint("00000000-0000-0000-0000-000000000001", "first")
    separate = blueprint("00000000-0000-0000-0000-000000000002", "separate")
    edited = blueprint("00000000-0000-0000-0000-000000000001", "edited")
    edited["definitions"]["subgraphs"][0]["nodes"].append({"id": 3, "type": "ImageCrop"})
    records = [carve_workflow(json.dumps(graph).encode(), carver) for graph in (first, separate, edited)]
    result = correlate([], records)
    assert [finding.artifact_ids for finding in result.findings] == [[r.workflow_id] for r in records]
    assert result.tier_summary["E6"] == 3


def test_a_png_alone_is_a_job_finding_through_the_prompt_it_carries():
    from comfytrace.correlate import png_workflows

    graph = api_prompt(seed=77, width=64, height=48)
    image = carve_image(graph, extra_text={"workflow": json.dumps(
        {"nodes": [{"id": 3, "type": "KSampler", "widgets_values": [77, "fixed", 20, 8, "euler", "normal", 1]}],
         "links": []})})
    prompt, ui = png_workflows([image], IdAllocator("WFPNG"))
    assert (prompt.prompt_format, prompt.encoding_detected) == ("API_GRAPH", "PNG_TEXT")
    assert prompt.completeness == "COMPLETE"
    assert ui.prompt_format == "UI_WORKFLOW" and ui.execution_parameters["seed"] == 77
    result = correlate([image], [prompt, ui])
    [link] = result.links
    assert (link.workflow_id, link.link_class, link.matched_keys) == (prompt.workflow_id, "STRICT", ["EMBEDDED_PROMPT"])
    [finding] = result.findings
    assert finding.tier == "E2" and finding.artifact_ids == [image.artifact_id, prompt.workflow_id, ui.workflow_id]


def test_an_external_workflow_wins_over_the_prompt_a_png_carries():
    from comfytrace.correlate import png_workflows

    graph = api_prompt(seed=78)
    drifted = json.loads(json.dumps(graph))
    drifted["3"]["inputs"]["denoise"] = 0.75
    image = carve_image(drifted)
    external = carve_workflow(json.dumps(graph).encode())
    [embedded] = png_workflows([image], IdAllocator("WFPNG"))
    result = correlate([image], [external, embedded])
    [link] = result.links
    assert (link.workflow_id, link.link_class) == (external.workflow_id, "SUPPORTED")
    [finding] = result.findings
    assert finding.artifact_ids == [image.artifact_id, external.workflow_id, embedded.workflow_id]


def test_copies_of_one_output_are_one_finding_and_header_only_copies_join_it():
    from comfytrace.carve import PngCarver

    graph = api_prompt(seed=79)
    png = png_bytes(noise_image(64, 48, seed=3), {"prompt": json.dumps(graph)}, idat_chunk=512)
    carver = PngCarver(IdAllocator("CARVE-PNG"))
    whole = carver.carve_stream(png, base_physical_offset=0x10000)
    torn = carver.carve_stream(png[: len(png) - 900], base_physical_offset=0x80000)
    header = carver.carve_stream(png[: png.index(b"IDAT") - 4], base_physical_offset=0x100000)
    batch_png = png_bytes(noise_image(64, 48, seed=4), {"prompt": json.dumps(graph)}, idat_chunk=512)
    batch = carver.carve_stream(batch_png, base_physical_offset=0x200000)
    workflow = carve_workflow(json.dumps(graph).encode())
    [header_finding] = correlate(header, [workflow]).findings
    assert "Zero PNG pixel rows decoded." in header_finding.rationale
    [finding] = correlate(whole + torn + header, [workflow]).findings
    assert finding.artifact_ids == [i.artifact_id for i in whole + torn + header] + [workflow.workflow_id]
    assert finding.tier == "E2" and "3 copies of one output" in finding.rationale
    # With a batch image the job has two outputs: a copy with no verified pixels cannot be given to either.
    result = correlate(whole + torn + header + batch, [workflow])
    assert sorted(f.artifact_ids[:-1] for f in result.findings) == sorted(
        [[whole[0].artifact_id, torn[0].artifact_id], [header[0].artifact_id], [batch[0].artifact_id]])
