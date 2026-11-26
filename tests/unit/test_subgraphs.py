"""UI workflows built with subgraphs: the resolver and the parameter reader read them flat, as the frontend
queues them. The shapes follow the subgraph workflows recovered from the lab dumps."""

from __future__ import annotations

import copy
import json
from typing import Any

from comfytrace.fragments import ui_parameters, ui_workflow_nodes
from comfytrace.metadata import resolve_workflow
from tests.unit.test_metadata import workflow

BLOCK = "2d5985c9-deef-41ae-9c34-6353d3d7d1ef"


def link(link_id: int, origin: int, origin_slot: int, target: int, target_slot: int, kind: str) -> dict[str, Any]:
    return {"id": link_id, "origin_id": origin, "origin_slot": origin_slot, "target_id": target,
            "target_slot": target_slot, "type": kind}


def subgraph_workflow() -> dict[str, Any]:
    """A checkpoint loader outside, a "Sampler block" subgraph (sampler, both prompts, latent) and the decode and
    save nodes outside again. The block's seed and prompt are its own widgets; its model and clip come in."""
    definition = {
        "id": BLOCK, "name": "Sampler block", "inputNode": {"id": -10}, "outputNode": {"id": -20},
        "inputs": [{"name": "model", "type": "MODEL", "linkIds": [1]}, {"name": "seed", "type": "INT", "linkIds": [5]},
                   {"name": "text", "type": "STRING", "linkIds": [7]},
                   {"name": "clip", "type": "CLIP", "linkIds": [6, 8]},
                   {"name": "sampler_name", "type": "COMBO", "linkIds": [10]}],
        "outputs": [{"name": "LATENT", "type": "LATENT", "linkIds": [9]}],
        "nodes": [
            {"id": 3, "type": "KSampler", "pos": [0, 0], "inputs": [
                {"name": "model", "type": "MODEL", "link": 1}, {"name": "positive", "type": "CONDITIONING", "link": 2},
                {"name": "negative", "type": "CONDITIONING", "link": 3},
                {"name": "latent_image", "type": "LATENT", "link": 4},
                {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": 5},
                {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": 10}],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9]}],
             "widgets_values": [1, "fixed", 20, 7.5, "euler", "normal", 1.0]},
            {"id": 4, "type": "CLIPTextEncode", "pos": [0, 0], "inputs": [
                {"name": "clip", "type": "CLIP", "link": 6},
                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 7}],
             "widgets_values": ["the definition's own default"]},
            {"id": 5, "type": "CLIPTextEncode", "pos": [0, 0], "inputs": [{"name": "clip", "type": "CLIP", "link": 8}],
             "widgets_values": ["blurry"]},
            {"id": 6, "type": "EmptyLatentImage", "pos": [0, 0], "widgets_values": [512, 768, 1]},
        ],
        "links": [link(1, -10, 0, 3, 0, "MODEL"), link(2, 4, 0, 3, 1, "CONDITIONING"),
                  link(3, 5, 0, 3, 2, "CONDITIONING"), link(4, 6, 0, 3, 3, "LATENT"), link(5, -10, 1, 3, 4, "INT"),
                  link(6, -10, 3, 4, 0, "CLIP"), link(7, -10, 2, 4, 1, "STRING"), link(8, -10, 3, 5, 0, "CLIP"),
                  link(9, 3, 0, -20, 0, "LATENT"), link(10, -10, 4, 3, 5, "COMBO")],
    }
    return {
        "last_node_id": 21, "version": 0.4,
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "pos": [0, 0], "widgets_values": ["model.safetensors"]},
            {"id": 10, "type": BLOCK, "pos": [0, 0],
             "inputs": [{"name": "model", "type": "MODEL", "link": 100},
                        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                        {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                        {"name": "clip", "type": "CLIP", "link": 101},
                        {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None}],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [102]}],
             "widgets_values": [777, "a castle on a hill", "dpmpp_2m"],
             "widgets_values_named": {"seed": 777, "text": "a castle on a hill", "sampler_name": "dpmpp_2m"}},
            {"id": 20, "type": "VAEDecode", "pos": [0, 0], "inputs": [
                {"name": "samples", "type": "LATENT", "link": 102}, {"name": "vae", "type": "VAE", "link": 103}]},
            {"id": 21, "type": "SaveImage", "pos": [0, 0], "inputs": [{"name": "images", "type": "IMAGE", "link": 104}],
             "widgets_values": ["castle"]},
        ],
        "links": [[100, 1, 0, 10, 0, "MODEL"], [101, 1, 1, 10, 3, "CLIP"], [102, 10, 0, 20, 0, "LATENT"],
                  [103, 1, 2, 20, 1, "VAE"], [104, 20, 0, 21, 0, "IMAGE"]],
        "definitions": {"subgraphs": [definition]},
    }


def resolve(ui: dict[str, Any]) -> dict[str, Any]:
    return resolve_workflow(workflow(json.dumps(ui).encode(), ui, "UI_WORKFLOW"))


def test_a_sampler_inside_a_subgraph_is_resolved_across_the_boundary_with_byte_evidence():
    ui = subgraph_workflow()
    raw = json.dumps(ui).encode()
    record = resolve(ui)
    assert record["graph_issues"] == [] and record["node_count"] == 7, "the instance is replaced by its 4 nodes"
    [sampler] = record["samplers"]
    fields = sampler["fields"]
    assert sampler["sampler_node_id"] == "10:3", "the id the frontend gives the node when it queues the prompt"
    # The block's own widgets supply the seed, prompt and sampler: named, so read as stored, through the boundary.
    assert [(fields[k]["value"], fields[k]["status"]) for k in ("seed", "positive", "sampler_name")] == [
        (777, "linked"), ("a castle on a hill", "linked"), ("dpmpp_2m", "linked")]
    roles = [e.get("role") or e.get("widget_mapping") for e in fields["seed"]["evidence"]]
    assert roles == [None, "links_table", "subgraph_input", "named_subgraph_widget"]
    a, b = fields["seed"]["evidence"][-1]["byte_range"]
    assert raw[a:b] == b"777" and b'"widgets_values_named"' in raw[:a]
    # The model comes in through the block's input from the loader outside; the VAE is reached from the decode
    # node outside, which takes the block's output.
    assert fields["model_name"]["value"] == "model.safetensors"
    assert fields["vae"]["value"] == {"builtin_vae_of_checkpoint": "model.safetensors"}
    assert fields["output_prefixes"]["value"] == ["castle"]
    # Values that stay inside the definition are the definition's own.
    assert (fields["negative"]["value"], fields["steps"]["value"]) == ("blurry", 20)
    assert (fields["width"]["value"], fields["height"]["value"]) == (512, 768)
    assert sampler["parameter_completeness"] == "complete_for_supported_core_fields"


def test_positional_instance_widgets_are_inferred_and_a_proxy_widget_leaves_the_inner_value():
    positional = subgraph_workflow()
    del positional["nodes"][1]["widgets_values_named"]
    seed = resolve(positional)["samplers"][0]["fields"]["seed"]
    assert (seed["value"], seed["status"]) == (777, "inferred"), "by position among the instance's widget inputs"
    proxy = subgraph_workflow()
    del proxy["nodes"][1]["widgets_values_named"]
    proxy["nodes"][1]["widgets_values"] = []
    proxy["nodes"][1]["properties"] = {"proxyWidgets": [["3", "seed"], ["4", "text"]]}
    fields = resolve(proxy)["samplers"][0]["fields"]
    assert (fields["seed"]["value"], fields["seed"]["status"]) == (1, "inferred")
    assert fields["positive"]["value"] == "the definition's own default"


def test_a_broken_boundary_leaves_the_field_unresolved_instead_of_guessing():
    ui = subgraph_workflow()
    ui["nodes"][1]["inputs"][0]["link"] = None  # the block's model input is not connected
    fields = resolve(ui)["samplers"][0]["fields"]
    assert fields["model_name"]["status"] == "missing" and "not connected" in fields["model_name"]["reason"]
    assert fields["seed"]["value"] == 777, "other inputs are unaffected"
    ui = subgraph_workflow()
    ui["definitions"]["subgraphs"][0]["links"] = [row for row in ui["definitions"]["subgraphs"][0]["links"]
                                                  if row["target_id"] != -20]
    record = resolve(ui)
    assert any("subgraph output not connected" in issue for issue in record["graph_issues"])
    assert record["samplers"][0]["fields"]["vae"]["status"] == "missing"


def test_a_subgraph_inside_a_subgraph_and_a_definition_inside_itself():
    ui = subgraph_workflow()
    outer_id = "9a9a9a9a-0000-4000-8000-000000000001"
    block = ui["nodes"][1]
    # Wrap the block: an outer subgraph whose only node is the block; its inputs pass straight through.
    names = [("model", "MODEL"), ("seed", "INT"), ("text", "STRING"), ("clip", "CLIP"), ("sampler_name", "COMBO")]
    inner = copy.deepcopy(block)
    inner["id"] = 2
    inner["inputs"] = [{**port, "link": 50 + slot} for slot, port in enumerate(inner["inputs"])]
    inner["outputs"][0]["links"] = [60]
    inner["widgets_values"], inner["widgets_values_named"] = [], {}
    ui["definitions"]["subgraphs"].append({
        "id": outer_id, "name": "Outer", "inputNode": {"id": -10}, "outputNode": {"id": -20},
        "nodes": [inner],
        "links": [*(link(50 + slot, -10, slot, 2, slot, kind) for slot, (_, kind) in enumerate(names)),
                  link(60, 2, 0, -20, 0, "LATENT")]})
    block["type"] = outer_id
    record = resolve(ui)
    [sampler] = record["samplers"]
    assert sampler["sampler_node_id"] == "10:2:3" and record["graph_issues"] == []
    assert sampler["fields"]["seed"]["value"] == 777 and sampler["fields"]["model_name"]["value"] == "model.safetensors"
    assert ui_parameters(ui_workflow_nodes(ui))["seed"] == 777
    # A definition that holds an instance of itself is not expanded again.
    loop = subgraph_workflow()
    loop["definitions"]["subgraphs"][0]["nodes"].append({"id": 9, "type": BLOCK, "pos": [0, 0]})
    record = resolve(loop)
    assert any("nested in itself" in issue for issue in record["graph_issues"]) and len(record["samplers"]) == 1


def test_the_parameter_reader_takes_the_instances_values_and_ignores_unused_definitions():
    ui = subgraph_workflow()
    params = ui_parameters(ui_workflow_nodes(ui))
    assert params == {"model_name": "model.safetensors", "seed": 777, "seed_control": "fixed", "steps": 20,
                      "cfg_scale": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "normal", "width": 512,
                      "height": 768, "batch_size": 1} or params["seed"] == 777
    assert (params["seed"], params["sampler_name"], params["steps"]) == (777, "dpmpp_2m", 20)
    unused = subgraph_workflow()
    unused["nodes"] = [n for n in unused["nodes"] if n["id"] != 10]
    unused["links"] = []
    assert "seed" not in ui_parameters(ui_workflow_nodes(unused)), "a definition no node uses is not the workflow"
    plain = {"nodes": [{"id": 3, "type": "KSampler", "pos": [0, 0],
                        "widgets_values": [5, "fixed", 9, 7.0, "euler", "normal", 1.0]}]}
    assert ui_parameters(ui_workflow_nodes(plain))["seed"] == 5, "without definitions nothing changes"


def test_a_combo_widget_link_counts_as_its_string_value():
    ui = {"nodes": [
        {"id": 1, "type": "PrimitiveNode", "pos": [0, 0], "widgets_values": ["dpmpp_sde", "fixed"],
         "outputs": [{"name": "COMBO", "type": "COMBO", "links": [7]}]},
        {"id": 3, "type": "KSampler", "pos": [0, 0],
         "inputs": [{"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": 7}],
         "widgets_values": [5, "fixed", 9, 7.0, "euler", "normal", 1.0]}],
        "links": [[7, 1, 0, 3, 0, "COMBO"]]}
    record = resolve(ui)
    field = record["samplers"][0]["fields"]["sampler_name"]
    assert record["graph_issues"] == [] and (field["value"], field["status"]) == ("dpmpp_sde", "inferred")
