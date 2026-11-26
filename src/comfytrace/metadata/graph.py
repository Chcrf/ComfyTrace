"""Occurrence-scoped graph and explicit core-node schemas.

Widget orders adapted from fragment-extractor (read-only reference). They describe
classic core ComfyUI nodes with control_after_generate after seed widgets. No
installed ComfyUI version is inferred from a memory occurrence.

Subgraphs (ComfyTrace addition). A UI workflow may hold subgraph definitions under
``definitions.subgraphs``; a node whose ``type`` is a definition's id is an instance of it.
The graph is read flat, as the frontend queues it: an instance is replaced by the definition's
nodes, with ids ``<instance id>:<inner id>``. A link from the definition's input node (id -10,
slot k) continues with whatever feeds the instance's input k; a link from an instance's output
slot j continues with the definition's link into its output node (id -20, slot j). When the
instance's input k is an unconnected widget, the value is the instance's own widget value
(``widgets_values_named``, else ``widgets_values`` by position among its widget inputs, which
is ``inferred``); when the instance stores none (a proxy widget shows the inner node's own
widget), the inner node keeps its own value. Every boundary crossed is listed as evidence.
"""
from collections import Counter
from dataclasses import dataclass, field

from .decoding import ambiguous_path, pointer

UI_ORDER = {
    "KSampler": "seed control_after_generate steps cfg sampler_name scheduler denoise",
    "KSamplerAdvanced": "add_noise noise_seed control_after_generate steps cfg sampler_name scheduler start_at_step end_at_step return_with_leftover_noise",
    "SamplerCustom": "add_noise noise_seed control_after_generate cfg",
    "SamplerCustomAdvanced": "",
    "RandomNoise": "noise_seed control_after_generate", "DisableNoise": "",
    "KSamplerSelect": "sampler_name", "BasicScheduler": "scheduler steps denoise",
    "CFGGuider": "cfg", "BasicGuider": "", "FluxGuidance": "guidance",
    "CheckpointLoaderSimple": "ckpt_name", "CheckpointLoader": "config_name ckpt_name",
    "UNETLoader": "unet_name weight_dtype", "LoraLoader": "lora_name strength_model strength_clip",
    "LoraLoaderModelOnly": "lora_name strength_model", "VAELoader": "vae_name",
    "CLIPLoader": "clip_name type", "DualCLIPLoader": "clip_name1 clip_name2 type",
    "CLIPSetLastLayer": "stop_at_clip_layer", "CLIPTextEncode": "text",
    "CLIPTextEncodeSDXL": "width height crop_w crop_h target_width target_height text_g text_l",
    "EmptyLatentImage": "width height batch_size", "EmptySD3LatentImage": "width height batch_size",
    "LoadImage": "image upload", "LoadImageMask": "image channel upload",
    "SaveImage": "filename_prefix", "PreviewImage": "",
    "ImageScaleBy": "upscale_method scale_by", "ImageScale": "upscale_method width height crop",
    "LatentUpscaleBy": "upscale_method scale_by", "LatentUpscale": "upscale_method width height crop",
    "UpscaleModelLoader": "model_name", "ImageUpscaleWithModel": "",
    "PrimitiveInt": "value control_after_generate", "PrimitiveFloat": "value",
    "PrimitiveString": "value", "PrimitiveBoolean": "value", "PrimitiveNode": "value control_after_generate",
    "Reroute": "", "VAEDecode": "", "VAEEncode": "",
    "ConditioningCombine": "", "ConditioningConcat": "", "ConditioningZeroOut": "",
}
SAMPLERS = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
# Output slot types, not heuristics based on class-name substrings.
OUTPUTS = {
    **{t: ("LATENT",) for t in SAMPLERS}, "SamplerCustom": ("LATENT", "LATENT"),
    "SamplerCustomAdvanced": ("LATENT", "LATENT"),
    "CheckpointLoaderSimple": ("MODEL", "CLIP", "VAE"), "CheckpointLoader": ("MODEL", "CLIP", "VAE"),
    "UNETLoader": ("MODEL",), "LoraLoader": ("MODEL", "CLIP"), "LoraLoaderModelOnly": ("MODEL",),
    "VAELoader": ("VAE",), "CLIPLoader": ("CLIP",), "DualCLIPLoader": ("CLIP",),
    "CLIPSetLastLayer": ("CLIP",), "CLIPTextEncode": ("CONDITIONING",),
    "CLIPTextEncodeSDXL": ("CONDITIONING",), "FluxGuidance": ("CONDITIONING",),
    "ConditioningCombine": ("CONDITIONING",), "ConditioningConcat": ("CONDITIONING",),
    "ConditioningZeroOut": ("CONDITIONING",),
    "EmptyLatentImage": ("LATENT",), "EmptySD3LatentImage": ("LATENT",),
    "LatentUpscale": ("LATENT",), "LatentUpscaleBy": ("LATENT",),
    "VAEEncode": ("LATENT",), "VAEDecode": ("IMAGE",), "LoadImage": ("IMAGE", "MASK"),
    "LoadImageMask": ("MASK",), "ImageScale": ("IMAGE",), "ImageScaleBy": ("IMAGE",),
    "ImageUpscaleWithModel": ("IMAGE",), "UpscaleModelLoader": ("UPSCALE_MODEL",),
    "RandomNoise": ("NOISE",), "DisableNoise": ("NOISE",), "CFGGuider": ("GUIDER",),
    "BasicGuider": ("GUIDER",), "BasicScheduler": ("SIGMAS",), "KSamplerSelect": ("SAMPLER",),
    "PrimitiveInt": ("INT",), "PrimitiveFloat": ("FLOAT",), "PrimitiveString": ("STRING",),
    "PrimitiveBoolean": ("BOOLEAN",), "PrimitiveNode": ("*",), "Reroute": ("*",),
    "SaveImage": (), "PreviewImage": (),
}
CONNECTIONS = {
    **{t: dict(model="MODEL", positive="CONDITIONING", negative="CONDITIONING", latent_image="LATENT")
       for t in ("KSampler", "KSamplerAdvanced", "SamplerCustom")},
    "SamplerCustomAdvanced": dict(noise="NOISE", guider="GUIDER", sampler="SAMPLER", sigmas="SIGMAS", latent_image="LATENT"),
    "LoraLoader": dict(model="MODEL", clip="CLIP"), "LoraLoaderModelOnly": dict(model="MODEL"),
    "CLIPTextEncode": dict(clip="CLIP"), "CLIPTextEncodeSDXL": dict(clip="CLIP"),
    "CLIPSetLastLayer": dict(clip="CLIP"), "FluxGuidance": dict(conditioning="CONDITIONING"),
    "ConditioningZeroOut": dict(conditioning="CONDITIONING"),
    "ConditioningCombine": dict(conditioning_1="CONDITIONING", conditioning_2="CONDITIONING"),
    "ConditioningConcat": dict(conditioning_to="CONDITIONING", conditioning_from="CONDITIONING"),
    "CFGGuider": dict(model="MODEL", positive="CONDITIONING", negative="CONDITIONING"),
    "BasicGuider": dict(model="MODEL", conditioning="CONDITIONING"),
    "BasicScheduler": dict(model="MODEL"), "VAEDecode": dict(samples="LATENT", vae="VAE"),
    "VAEEncode": dict(pixels="IMAGE", vae="VAE"), "SaveImage": dict(images="IMAGE"),
    "PreviewImage": dict(images="IMAGE"), "ImageScale": dict(image="IMAGE"),
    "ImageScaleBy": dict(image="IMAGE"), "LatentUpscale": dict(samples="LATENT"),
    "LatentUpscaleBy": dict(samples="LATENT"), "ImageUpscaleWithModel": dict(image="IMAGE", upscale_model="UPSCALE_MODEL"),
    "Reroute": {"": "*"},
}
CONNECTIONS["SamplerCustom"].update(sampler="SAMPLER", sigmas="SIGMAS")
INT_FIELDS = set("seed noise_seed steps start_at_step end_at_step width height batch_size crop_w crop_h target_width target_height stop_at_clip_layer".split())
FLOAT_FIELDS = set("cfg denoise guidance strength_model strength_clip scale_by".split())
BOOL_FIELDS = {"add_noise", "return_with_leftover_noise"}


def expected_type(node_type, name):
    if name in CONNECTIONS.get(node_type, {}):
        return CONNECTIONS[node_type][name]
    if name not in UI_ORDER.get(node_type, "").split():
        return None
    if name == "value":
        return OUTPUTS.get(node_type, ("*",))[0]
    return "INT" if name in INT_FIELDS else "FLOAT" if name in FLOAT_FIELDS else "*" if name in BOOL_FIELDS else "STRING"


def integer(v):
    return isinstance(v, int) and not isinstance(v, bool)


def same_type(found, declared):
    """A link or input slot type agrees with the schema's. The frontend types a combo widget's slot COMBO; its
    value is one of a list of strings."""
    return found in (declared, "*") or (found == "COMBO" and declared == "STRING")


def link_shape(v):
    return isinstance(v, list) and len(v) == 2 and (isinstance(v[0], str) or integer(v[0])) and integer(v[1])


SUBGRAPH_INPUT, SUBGRAPH_OUTPUT = -10, -20
MAX_SUBGRAPH_DEPTH = 8


@dataclass
class Edge:
    source: str
    slot: int
    evidence: list
    error: str | None = None
    link_id: object = None  # the link that leaves the source node, when a subgraph boundary was crossed


@dataclass
class Instance:
    """A node that stands for a subgraph: where it sits and the scope its definition's nodes get."""
    scope: str
    local: str
    raw: dict
    path: tuple
    inner: str


@dataclass
class Node:
    id: str
    instance: str
    type: str
    raw: dict
    path: tuple
    values: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    conflicts: set = field(default_factory=set)
    duplicate: bool = False
    scope: str = ""  # "" at the top level, "<instance id>:" inside a subgraph instance
    local: str = ""  # the id the node has in its own graph
    crossed: dict = field(default_factory=dict)  # input name -> evidence of the subgraph boundary its value crossed


class Graph:
    def __init__(self, occurrence):
        self.occurrence = occurrence
        self.id = occurrence.id if occurrence.kind.endswith("graph") else None
        self.nodes, self.by_id, self.issues = [], {}, []
        self.partial = bool(getattr(occurrence, "metadata", {}).get("partial"))
        self.incomplete_paths = set(getattr(occurrence, "metadata", {}).get("incomplete_paths", []))
        self.ui = occurrence.kind.startswith("ui")
        self.instances, self.scope_instance, self.scopes, self.tables = {}, {}, {}, {}
        obj, p = occurrence.value, occurrence.path
        enclosure_conflict = occurrence.document.ambiguous_enclosure or ambiguous_path(occurrence.document, p)
        if enclosure_conflict:
            self.issues.append("duplicate enclosing record member")
        if occurrence.kind == "api_graph":
            entries = [(str(k), v, p + (k,)) for k, v in obj.items()]
            duplicate_ids = getattr(obj, "duplicates", set())
        elif occurrence.kind == "ui_graph":
            entries = self._ui_entries(obj, p)
            duplicate_ids = {k for k, count in Counter(e[0] for e in entries).items() if count > 1}
            if "nodes" in getattr(obj, "duplicates", set()) or "links" in getattr(obj, "duplicates", set()):
                self.issues.append("duplicate nodes/links member; graph connections invalid")
        else:
            entries = [(str(obj.get("id", "standalone")), obj, p)]
            duplicate_ids = set()
        for i, (nid, raw, path, *where) in enumerate(entries):
            scope, local = where or ("", nid)
            node = Node(nid, f"{occurrence.id}/node/{i}", raw.get("type" if self.ui else "class_type"), raw, path,
                        duplicate=nid in duplicate_ids or enclosure_conflict or (self.ui and "nodes" in getattr(obj, "duplicates", set())),
                        scope=scope, local=local)
            self.nodes.append(node)
            self.by_id.setdefault(nid, []).append(node)
            self._values(node)
        for nid in duplicate_ids:
            self.issues.append(f"duplicate node ID {nid}")
        if self.ui and self.id:
            self.tables = {scope: self._table(graph.get("links", []), path) for scope, (graph, path) in self.scopes.items()}
        for n in self.nodes:
            self._edges(n, self.tables.get(n.scope, {}))
        for n in self.nodes:
            for name, edge in n.edges.items():
                self._validate(n, name, edge)
                if edge.error:
                    self.issues.append(f"{n.id}.{name}: {edge.error}")
        self.issues.extend(self._cycles())

    def ev(self, n, path, **extra):
        ev = self.occurrence.evidence(path)
        ev.update(node_id=n.id, node_instance=n.instance, **extra)
        return ev

    def _values(self, n):
        if not self.ui:
            data = n.raw["inputs"]
            for key, val in data.items():
                path = n.path + ("inputs", key)
                n.values[key] = (val, path, "api_input")
            n.conflicts.update(getattr(data, "duplicates", set()))
        else:
            named = n.raw.get("widgets_values_named")
            widgets = n.raw.get("widgets_values")
            if isinstance(widgets, dict):
                for key, val in widgets.items():
                    n.values[key] = (val, n.path + ("widgets_values", key), "named_widget")
                n.conflicts.update(getattr(widgets, "duplicates", set()))
            if isinstance(named, dict):
                for key, val in named.items():
                    if key in n.values and n.values[key][0] != val:
                        n.conflicts.add(key)
                    n.values[key] = (val, n.path + ("widgets_values_named", key), "named_widget")
                n.conflicts.update(getattr(named, "duplicates", set()))
            elif isinstance(widgets, list):
                for i, (key, val) in enumerate(zip(UI_ORDER.get(n.type, "").split(), widgets, strict=False)):
                    n.values.setdefault(key, (val, n.path + ("widgets_values", i), "positional_core_v1"))
        for key in getattr(n.raw, "duplicates", set()):
            if self.ui and key == "id":
                n.duplicate = True
                self.issues.append(f"duplicate node ID member {n.id}")
            if key in ("inputs", "class_type", "type", "widgets_values", "widgets_values_named"):
                n.conflicts.update(n.values)
                if key in ("inputs", "class_type", "type"):
                    n.duplicate = True

    def _ui_entries(self, obj, p):
        """(scoped id, node, path, scope, local id) of every node, with subgraph instances replaced by the nodes
        of their definitions."""
        definitions = {}
        subgraphs = obj.get("definitions").get("subgraphs") if isinstance(obj.get("definitions"), dict) else None
        for i, definition in enumerate(subgraphs if isinstance(subgraphs, list) else []):
            if isinstance(definition, dict) and isinstance(definition.get("id"), str) and isinstance(definition.get("nodes"), list):
                if definition["id"] in definitions:
                    self.issues.append(f"duplicate subgraph definition {definition['id']}")
                definitions.setdefault(definition["id"], (definition, p + ("definitions", "subgraphs", i)))
        entries = []

        def walk(graph, path, scope, chain):
            self.scopes[scope] = (graph, path)
            for i, v in enumerate(graph["nodes"]):
                if not isinstance(v, dict) or "id" not in v:
                    self.issues.append(f"{scope}nodes[{i}]: not a node")
                    continue
                local = str(v["id"])
                definition = definitions.get(v.get("type")) if isinstance(v.get("type"), str) else None
                if definition is None:
                    entries.append((scope + local, v, path + ("nodes", i), scope, local))
                elif v["type"] in chain or len(chain) >= MAX_SUBGRAPH_DEPTH:
                    self.issues.append(f"{scope}{local}: subgraph nested in itself or too deep")
                elif scope + local in self.instances:
                    self.issues.append(f"duplicate node ID {scope}{local}")
                else:
                    self.instances[scope + local] = Instance(scope, local, v, path + ("nodes", i), scope + local + ":")
                    self.scope_instance[scope + local + ":"] = scope + local
                    walk(definition[0], definition[1], scope + local + ":", chain | {v["type"]})

        walk(obj, p, "", frozenset())
        return entries

    def _origin(self, n, key, scope, origin, slot, lid):
        """Follows a link's origin across subgraph boundaries. Returns (scoped source id, slot, id of the link
        leaving the source, evidence of each boundary, widget, error). ``widget`` is None for a connection,
        ("value", (value, path, mode)) when the instance's own widget supplies the input, ("inner",) when the
        inner node keeps its own widget value and ("unconnected",) when nothing feeds the input."""
        ev = []
        for _ in range(2 * MAX_SUBGRAPH_DEPTH + 2):
            if origin == SUBGRAPH_INPUT and scope in self.scope_instance:
                inst = self.instances[self.scope_instance[scope]]
                inputs = inst.raw.get("inputs")
                port = inputs[slot] if isinstance(inputs, list) and integer(slot) and 0 <= slot < len(inputs) and isinstance(inputs[slot], dict) else None
                if port is None:
                    return "", -1, lid, ev, None, "subgraph input slot outside the instance's inputs"
                ev.append(self.ev(n, inst.path + ("inputs", slot), input=key, role="subgraph_input"))
                outer = port.get("link")
                if outer is None:
                    return "", -1, lid, ev, self._instance_widget(inst, slot, port), None
                rows = self.tables.get(inst.scope, {}).get(str(outer), [])
                if len(rows) != 1:
                    return "", -1, lid, ev, None, "duplicate link ID" if rows else "missing link ID"
                row, path = rows[0]
                ev.append(self.ev(n, path, input=key, role="links_table"))
                if str(row[3]) != inst.local or row[4] != slot:
                    return "", -1, lid, ev, None, "invalid link target endpoint or input slot"
                scope, origin, slot, lid = inst.scope, row[1], row[2], outer
                continue
            inst = self.instances.get(scope + str(origin))
            if inst is None:
                return scope + str(origin), slot, lid, ev, None, None
            rows = [(row, path) for group in self.tables.get(inst.inner, {}).values() for row, path in group
                    if row[3] == SUBGRAPH_OUTPUT and row[4] == slot]
            if len(rows) != 1:
                return "", -1, lid, ev, None, "duplicate subgraph output link" if rows else "subgraph output not connected"
            row, path = rows[0]
            ev.append(self.ev(n, path, input=key, role="subgraph_output"))
            scope, origin, slot, lid = inst.inner, row[1], row[2], row[0]
        return "", -1, lid, ev, None, "subgraph boundary chain too long"

    @staticmethod
    def _instance_widget(inst, slot, port):
        widget = port.get("widget")
        if not isinstance(widget, dict):
            return ("unconnected",)
        name = widget.get("name", port.get("name"))
        named, values = inst.raw.get("widgets_values_named"), inst.raw.get("widgets_values")
        if isinstance(named, dict) and name in named:
            return ("value", (named[name], inst.path + ("widgets_values_named", name), "named_subgraph_widget"))
        if isinstance(values, list) and values:
            index = sum(1 for other in inst.raw["inputs"][:slot] if isinstance(other, dict) and isinstance(other.get("widget"), dict))
            if index < len(values):
                return ("value", (values[index], inst.path + ("widgets_values", index), "positional_subgraph_widget"))
        return ("inner",)

    def _table(self, rows, base=None):
        base = self.occurrence.path if base is None else base
        table = {}
        if not isinstance(rows, list):
            self.issues.append("invalid links table")
            return table
        for i, row in enumerate(rows):
            if pointer(base + ("links", i)) in self.incomplete_paths:
                self.issues.append(f"incomplete links row {i}")
                continue
            duplicate_members = bool(getattr(row, "duplicates", set()))
            if isinstance(row, dict):
                row = [row.get(k) for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")]
            if not isinstance(row, list) or len(row) < 6 or not (integer(row[0]) or isinstance(row[0], str)):
                self.issues.append(f"invalid links row {i}")
                continue
            table.setdefault(str(row[0]), []).append((row, base + ("links", i)))
            if duplicate_members:
                table[str(row[0])].append((row, base + ("links", i)))
                self.issues.append(f"duplicate members in links row {i}")
        return table

    def _edges(self, n, table):
        if not self.ui:
            for key, (val, path, _) in list(n.values.items()):
                if pointer(path) in self.incomplete_paths and expected_type(n.type, key):
                    n.edges[key] = Edge("", -1, [self.ev(n, path, input=key)], "incomplete input value; no link inferred")
                    del n.values[key]
                elif expected_type(n.type, key) and link_shape(val):
                    n.edges[key] = Edge(str(val[0]), val[1], [self.ev(n, path, input=key)])
                    del n.values[key]
        else:
            inputs = n.raw.get("inputs", [])
            if not isinstance(inputs, list):
                self.issues.append(f"{n.id}: invalid inputs array")
                return
            names = [inp.get("name") for inp in inputs if isinstance(inp, dict) and isinstance(inp.get("name"), str)]
            n.conflicts.update(name for name, count in Counter(names).items() if count > 1)
            for slot, inp in enumerate(inputs):
                if not isinstance(inp, dict) or not isinstance(inp.get("name"), str):
                    continue
                key, lid = inp["name"], inp.get("link")
                if getattr(inp, "duplicates", set()):
                    n.conflicts.add(key)
                if lid is None:
                    continue
                ev = [self.ev(n, n.path + ("inputs", slot), input=key, link_id=lid)]
                if key in n.edges:
                    n.conflicts.add(key)
                rows = table.get(str(lid), [])
                if len(rows) != 1:
                    n.edges[key] = Edge("", -1, ev, "duplicate link ID" if rows else "missing link ID")
                    continue
                row, path = rows[0]
                ev.append(self.ev(n, path, input=key, role="links_table"))
                error = None
                if str(row[3]) != n.local or not integer(row[4]) or row[4] != slot:
                    error = "invalid link target endpoint or input slot"
                declared = expected_type(n.type, key)
                if n.type in UI_ORDER and declared is None:
                    error = "unsupported input name for supported node schema"
                if "slot_index" in inp and inp["slot_index"] != slot:
                    error = "recorded input slot_index disagrees with input position"
                if inp.get("type") is not None and declared not in (None, "*") and not same_type(inp["type"], declared):
                    error = "input name/type disagrees with supported node schema"
                if declared not in (None, "*") and not same_type(row[5], declared):
                    error = "link type disagrees with input name"
                if "duplicate nodes/links member; graph connections invalid" in self.issues:
                    error = "duplicate graph structure"
                source, source_slot, link_id = n.scope + str(row[1]), row[2], None
                if error is None and self.instances:
                    source, source_slot, link_id, crossed, widget, error = self._origin(n, key, n.scope, row[1], row[2], lid)
                    ev += crossed
                    if widget is not None and widget[0] != "unconnected":
                        if widget[0] == "value":  # the subgraph's own widget supplies this input: a value, no edge
                            n.values[key] = widget[1]
                            n.crossed[key] = ev
                        continue
                    if widget is not None:
                        error = "subgraph input not connected"
                n.edges[key] = Edge(source, source_slot, ev, error, link_id)

    def output_type(self, node, slot, visited=frozenset()):
        if node.instance in visited or len(visited) >= 64:
            return None, "cycle or depth limit in reroute type chain"
        if node.duplicate:
            return None, "duplicate source node ID"
        if not integer(slot) or slot < 0:
            return None, "invalid output slot"
        outputs = OUTPUTS.get(node.type)
        if outputs is not None and slot >= len(outputs):
            return None, "output slot outside supported schema"
        if node.type == "Reroute":
            edge = node.edges.get("")
            if edge is None:
                return None, "missing reroute input"
            if edge.error:
                return None, edge.error
            sources = self.by_id.get(edge.source, [])
            if not sources:
                return None, "missing reroute source node"
            if len(sources) != 1:
                return None, "duplicate reroute source node"
            return self.output_type(sources[0], edge.slot, visited | {node.instance})
        return outputs[slot] if outputs is not None else None, None

    def _validate(self, target, name, edge):
        if edge.error:
            return
        sources = self.by_id.get(edge.source, [])
        if not sources:
            edge.error = "missing source node ID"
            return
        if len(sources) != 1 or sources[0].duplicate:
            edge.error = "duplicate source node ID"
            return
        if target.duplicate or name in target.conflicts:
            edge.error = "duplicate target node/input"
            return
        source = sources[0]
        if not integer(edge.slot) or edge.slot < 0:
            edge.error = "invalid output slot"
            return
        output_type, error = self.output_type(source, edge.slot)
        if error:
            edge.error = error
            return
        expected = expected_type(target.type, name)
        if expected and output_type and expected != "*" and output_type not in (expected, "*"):
            edge.error = "output type does not match input name (including reroutes)"
        declared = source.raw.get("outputs")
        if self.ui and isinstance(declared, list):
            if edge.slot >= len(declared):
                edge.error = "output slot outside recorded outputs"
            elif isinstance(declared[edge.slot], dict):
                output = declared[edge.slot]
                if "slot_index" in output and output["slot_index"] != edge.slot:
                    edge.error = "recorded output slot_index mismatch"
                link_ids = output.get("links")
                lid = edge.link_id if edge.link_id is not None else edge.evidence[0].get("link_id")
                if isinstance(link_ids, list) and lid not in link_ids:
                    edge.error = "origin outputs do not list recorded link ID"
                dtype = output.get("type")
                expected = expected_type(target.type, name)
                if dtype and expected and "*" not in (dtype, expected) and not same_type(dtype, expected):
                    edge.error = "recorded output type mismatch"

    def _cycles(self):
        visited, active, issues = set(), set(), []
        def visit(n):
            if n.instance in active:
                issues.append(f"cycle through node {n.id}")
                return
            if n.instance in visited:
                return
            visited.add(n.instance)
            active.add(n.instance)
            for edge in n.edges.values():
                if not edge.error:
                    visit(self.by_id[edge.source][0])
            active.remove(n.instance)
        # Iterative resolution is bounded; skip recursion overflow in adversarial graphs.
        try:
            for n in self.nodes:
                visit(n)
        except RecursionError:
            issues.append("graph traversal depth exceeded")
        return issues

    def record(self):
        return {"id": self.id, "occurrence_id": self.occurrence.id,
                "association": "graph_structure" if self.id else "unassociated_standalone",
                "node_count": len(self.nodes), "graph_completeness":
                "partial" if self.partial else "standalone" if not self.id else "structurally_complete" if not self.issues else "invalid_or_incomplete",
                "issues": self.issues,
                "nodes": [{"id": n.id, "instance": n.instance, "type": n.type,
                           "duplicate_id": n.duplicate, "location": self.ev(n, n.path)} for n in self.nodes]}
