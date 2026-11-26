"""Sampler metadata read through the graph's connections, with byte ranges into the exported workflow."""

import json

from comfytrace.carve import execution_parameters
from comfytrace.metadata import metadata_document, resolve_workflow
from comfytrace.models import WorkflowRecord


def api_graph() -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "unrelated.safetensors"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "connected.safetensors"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": ["8", 0], "steps": 20, "cfg": 7, "sampler_name": "euler",
                                                    "scheduler": "normal", "denoise": 0.6, "model": ["2", 0],
                                                    "positive": ["4", 0], "negative": ["5", 0],
                                                    "latent_image": ["6", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["2", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["2", 1]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        "8": {"class_type": "PrimitiveInt", "inputs": {"value": 999}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["2", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ComfyUI"}},
    }


def workflow(raw: bytes, graph: dict, fmt: str = "API_GRAPH", encoding: str = "UTF-8",
             workflow_id: str = "WF-001") -> WorkflowRecord:
    return WorkflowRecord(workflow_id=workflow_id, completeness="COMPLETE", prompt_format=fmt,
                          encoding_detected=encoding, source_process_id=None, virtual_address=None,
                          physical_offsets=[], canonical_json_sha256="", execution_parameters={},
                          dependencies={"node_classes": [], "model_assets": []},
                          raw_file_path=f"exports/raw/{workflow_id}.json", raw_bytes=raw, graph=graph)


def test_values_follow_the_connections_and_point_at_their_bytes():
    graph = api_graph()
    raw = json.dumps({"client_id": "c", "prompt": graph}).encode()
    record = resolve_workflow(workflow(raw, graph))
    [sampler] = record["samplers"]
    fields = sampler["fields"]
    assert fields["model_name"]["value"] == "connected.safetensors" and fields["model_name"]["status"] == "linked"
    assert execution_parameters(graph, {})["model_name"] == "unrelated.safetensors", "the direct read takes the first"
    assert (fields["seed"]["value"], fields["seed"]["status"]) == (999, "linked")
    assert (fields["positive"]["value"], fields["negative"]["value"]) == ("a cat", "blurry")
    assert (fields["width"]["value"], fields["height"]["value"]) == (512, 768)
    assert fields["vae"]["value"] == {"builtin_vae_of_checkpoint": "connected.safetensors"}
    assert fields["output_prefixes"]["value"] == ["ComfyUI"]
    assert record["graph_location"] == "/prompt" and record["evidence_basis"] == "EXPORTED_DUMP_BYTES"
    for name in ("steps", "cfg", "seed", "positive"):
        a, b = fields[name]["evidence"][-1]["byte_range"]
        assert json.loads(raw[a:b]) == fields[name]["value"]


def test_a_positional_ui_widget_is_inferred_and_a_named_one_direct():
    ui = {"nodes": [
        {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["m.safetensors"]},
        {"id": 3, "type": "KSampler", "inputs": [{"name": "model", "type": "MODEL", "link": 7}],
         "widgets_values": [123, "fixed", 20, 7, "euler", "normal", 1.0]},
        {"id": 4, "type": "KSampler", "widgets_values_named": {"seed": 5, "steps": 9}},
    ], "links": [[7, 1, 0, 3, 0, "MODEL"]]}
    record = resolve_workflow(workflow(json.dumps(ui).encode(), ui, fmt="UI_WORKFLOW"))
    positional, named = record["samplers"]
    assert (positional["fields"]["seed"]["value"], positional["fields"]["seed"]["status"]) == (123, "inferred")
    assert positional["fields"]["model_name"]["value"] == "m.safetensors"
    assert (named["fields"]["seed"]["value"], named["fields"]["seed"]["status"]) == (5, "direct")
    assert named["fields"]["model_name"]["status"] == "missing", "no connection, nothing guessed"


def test_a_quoted_json_string_draft_is_read_inside_its_string():
    graph = api_graph()
    raw = json.dumps(json.dumps(graph)).encode()
    record = resolve_workflow(workflow(raw, graph, encoding="UTF-8_JSON_STRING"))
    assert record["samplers"][0]["fields"]["steps"]["value"] == 20


def test_a_broken_link_is_an_issue_not_a_value():
    graph = api_graph()
    graph["3"]["inputs"]["model"] = ["42", 0]
    record = resolve_workflow(workflow(json.dumps(graph).encode(), graph))
    assert record["graph_completeness"] == "invalid_or_incomplete"
    assert record["samplers"][0]["fields"]["model_name"]["status"] == "missing"


def test_fragments_and_unparsable_bytes_are_listed_as_not_resolved():
    graph = api_graph()
    whole = workflow(json.dumps(graph).encode(), graph)
    cut = workflow(json.dumps(graph).encode()[:200], graph, workflow_id="WF-002")
    fragment = workflow(b'{"seed": 1}', {}, fmt="UI_FRAGMENT", workflow_id="WF-003")
    document = metadata_document([whole, cut, fragment])
    assert [r["workflow_id"] for r in document["workflows"]] == ["WF-001"]
    assert document["not_resolved"] == ["WF-002", "WF-003"]
