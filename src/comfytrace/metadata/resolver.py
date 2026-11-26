"""Shared Part 1/Part 2 resolver. Only explicit supported node semantics are used."""
from .decoding import pointer
from .graph import FLOAT_FIELDS, INT_FIELDS, SAMPLERS, UI_ORDER, integer

FIELDS = "seed control_after_generate steps cfg sampler_name scheduler denoise model_loader model_name loras positive negative guidance width height batch_size vae source_images output_prefixes upscale noise sigmas guider add_noise start_at_step end_at_step return_with_leftover_noise".split()
PRIMITIVES = {"PrimitiveInt", "PrimitiveFloat", "PrimitiveString", "PrimitiveBoolean", "PrimitiveNode"}
LOADERS = {"CheckpointLoaderSimple": "ckpt_name", "CheckpointLoader": "ckpt_name", "UNETLoader": "unet_name"}
UPSCALE = {"ImageScale", "ImageScaleBy", "LatentUpscale", "LatentUpscaleBy", "ImageUpscaleWithModel"}


def result(value=None, status="missing", reason=None, evidence=None, **extra):
    evidence = evidence or []
    path = [{k: ev[k] for k in ("node_instance", "input", "location") if k in ev} for ev in evidence]
    return {"value": value, "status": status, "reason": reason,
            "evidence": [dict(ev, traversal_path=path[:i + 1]) for i, ev in enumerate(evidence)], **extra}


class Unresolved(Exception):
    def __init__(self, reason, evidence, ambiguous=False):
        self.reason, self.evidence, self.ambiguous = reason, evidence, ambiguous

    def record(self):
        return result(status="ambiguous" if self.ambiguous else "missing",
                      reason=self.reason, evidence=self.evidence)


def combine(records, absent="no supported connected evidence"):
    if not records:
        return result(reason=absent)
    known = [r for r in records if r["status"] in ("direct", "linked", "inferred")]
    ev = [e for r in records for e in r["evidence"]]
    if any(r["status"] == "ambiguous" for r in records) or any(r["value"] != known[0]["value"] for r in known[1:]):
        return result(status="ambiguous", reason="conflicting connected values", evidence=ev,
                      alternatives=[r["value"] for r in known])
    if len(known) != len(records):
        return result(reason="one or more connected branches unresolved", evidence=ev,
                      partial_values=[r["value"] for r in known])
    status = "inferred" if any(r["status"] == "inferred" for r in records) else "linked" if any(r["status"] == "linked" for r in records) else "direct"
    return result(known[0]["value"], status, evidence=ev)


class Resolver:
    def __init__(self, graph):
        self.g = graph

    def evidence(self, n, path=None):
        return [self.g.ev(n, n.path if path is None else path)]

    def follow(self, n, name, trail=()):
        edge = n.edges.get(name)
        ev = list(trail) + self.evidence(n)
        if edge is None:
            raise Unresolved(f'{n.type}#{n.id} has no recorded {name} connection', ev)
        ev = list(trail) + edge.evidence
        if edge.error:
            raise Unresolved(edge.error, ev, ambiguous="duplicate" in edge.error)
        if n.duplicate:
            raise Unresolved("duplicate node semantics", ev, True)
        return self.g.by_id[edge.source][0], ev

    def guard(self, n, trail, visited):
        if n.instance in visited or len(visited) >= 64:
            raise Unresolved("cycle or traversal depth limit", list(trail) + self.evidence(n))
        if n.duplicate:
            raise Unresolved("duplicate node ID or structure", list(trail) + self.evidence(n), True)
        return visited | {n.instance}

    def scalar(self, n, name, trail=(), visited=frozenset()):
        try:
            visited = self.guard(n, trail, visited)
            if n.type not in UI_ORDER:
                raise Unresolved(f"unsupported node semantics: {n.type}", list(trail) + self.evidence(n))
            if name in n.conflicts:
                raise Unresolved(f"conflicting or duplicate {name}", list(trail) + self.evidence(n), True)
            if name in n.edges:
                src, ev = self.follow(n, name, trail)
                if src.type not in PRIMITIVES | {"Reroute"}:
                    raise Unresolved(f"{src.type} is not a supported primitive provider", ev + self.evidence(src))
                rec = self.scalar(src, "" if src.type == "Reroute" else "value", ev, visited)
                value = rec["value"]
                if rec["status"] in ("direct", "linked", "inferred"):
                    valid = integer(value) if name in INT_FIELDS else (isinstance(value, (int, float)) and not isinstance(value, bool)) if name in FLOAT_FIELDS else isinstance(value, str) if name in {"text", "sampler_name", "scheduler"} else True
                    if not valid:
                        return result(reason=f"wrong linked value type for {name}", evidence=rec["evidence"])
                return rec
            if n.type == "Reroute":
                src, ev = self.follow(n, "", trail)
                if src.type not in PRIMITIVES | {"Reroute"}:
                    raise Unresolved("unsupported reroute value provider", ev)
                return self.scalar(src, "value" if src.type in PRIMITIVES else "", ev, visited)
            if name not in n.values:
                raise Unresolved(f"{n.type}#{n.id} has no {name} value", list(trail) + self.evidence(n))
            value, loc, mode = n.values[name]
            if pointer(loc) in self.g.incomplete_paths:
                raise Unresolved("incomplete value; recovered prefix is not a complete parameter", list(trail) + self.evidence(n, loc))
            crossed = n.crossed.get(name, [])
            ev = list(trail) + crossed + [self.g.ev(n, loc, parameter=name, widget_mapping=mode)]
            if value is None:
                raise Unresolved(f"{name} is explicitly null", ev)
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            valid = integer(value) if name in INT_FIELDS else numeric if name in FLOAT_FIELDS else True
            if name in {"sampler_name", "scheduler", "text", "text_g", "text_l", "image", "filename_prefix", "ckpt_name", "unet_name", "vae_name", "lora_name", "model_name", "upscale_method"}:
                valid = isinstance(value, str)
            if not valid:
                raise Unresolved(f"wrong value type for {name}; mapping rejected", ev)
            status = "inferred" if mode.startswith("positional") else "linked" if trail or crossed else "direct"
            return result(value, status, "positional core-node mapping assumed" if status == "inferred" else None, ev)
        except Unresolved as exc:
            return exc.record()

    def safe(self, operation):
        try:
            return operation()
        except Unresolved as exc:
            return exc.record()

    def model(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type in LOADERS:
            name = self.scalar(n, LOADERS[n.type], trail)
            loader = result(n.type, "linked" if trail else "direct", evidence=list(trail) + self.evidence(n))
            return name, loader, [], []
        if n.type not in {"LoraLoader", "LoraLoaderModelOnly", "Reroute"}:
            raise Unresolved(f"unsupported model semantics: {n.type}", list(trail) + self.evidence(n))
        try:
            src, ev = self.follow(n, "" if n.type == "Reroute" else "model", trail)
            name, loader, loras, components = self.model(src, ev, visited)
        except Unresolved as exc:
            name, loader, loras, components = exc.record(), exc.record(), [], []
        if n.type != "Reroute":
            fields = {k: self.scalar(n, k, trail) for k in ("lora_name", "strength_model", "strength_clip")
                      if k != "strength_clip" or n.type == "LoraLoader"}
            loras.append({"node_instance": n.instance, **{k: r["value"] for k, r in fields.items()}})
            components.append({"node_instance": n.instance, "fields": fields})
        return name, loader, loras, components

    def clip_loras(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type in {"CheckpointLoaderSimple", "CheckpointLoader", "CLIPLoader", "DualCLIPLoader"}:
            return []
        if n.type not in {"LoraLoader", "CLIPSetLastLayer", "Reroute"}:
            raise Unresolved(f"unsupported CLIP semantics: {n.type}", list(trail) + self.evidence(n))
        src, ev = self.follow(n, "" if n.type == "Reroute" else "clip", trail)
        items = self.clip_loras(src, ev, visited)
        if n.type == "LoraLoader":
            items.append({"node_instance": n.instance,
                          "fields": {k: self.scalar(n, k, trail) for k in ("lora_name", "strength_model", "strength_clip")}})
        return items

    def conditioning_loras(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type in {"CLIPTextEncode", "CLIPTextEncodeSDXL"}:
            return self.clip_loras(*self.follow(n, "clip", trail))
        names = {"FluxGuidance": ["conditioning"], "ConditioningZeroOut": ["conditioning"],
                 "ConditioningCombine": ["conditioning_1", "conditioning_2"],
                 "ConditioningConcat": ["conditioning_to", "conditioning_from"], "Reroute": [""]}.get(n.type)
        if names is None:
            raise Unresolved(f"unsupported conditioning semantics: {n.type}", list(trail) + self.evidence(n))
        items = []
        for name in names:
            src, ev = self.follow(n, name, trail)
            items.extend(self.conditioning_loras(src, ev, visited))
        return items

    def input_media_nodes(self, sampler):
        nodes, seen = [], set()
        def walk(n, ev):
            if n.instance in seen or len(seen) >= 64:
                return
            seen.add(n.instance)
            nodes.append((n, ev))
            input_name = {"LatentUpscale": "samples", "LatentUpscaleBy": "samples", "Reroute": "",
                          "VAEEncode": "pixels", "ImageScale": "image", "ImageScaleBy": "image",
                          "ImageUpscaleWithModel": "image"}.get(n.type)
            if input_name is not None:
                try:
                    walk(*self.follow(n, input_name, ev))
                except Unresolved:
                    pass
        try:
            walk(*self.follow(sampler, "latent_image"))
        except Unresolved:
            pass
        return nodes

    def conditioning(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type == "CLIPTextEncode":
            return self.scalar(n, "text", trail), []
        if n.type == "CLIPTextEncodeSDXL":
            fields = {k: self.scalar(n, k, trail) for k in ("text_g", "text_l")}
            ok = all(r["status"] in ("linked", "direct", "inferred") for r in fields.values())
            return result({k: r["value"] for k, r in fields.items()} if ok else None,
                          "inferred" if ok and any(r["status"] == "inferred" for r in fields.values()) else "linked" if ok else "missing",
                          None if ok else "incomplete SDXL text", [e for r in fields.values() for e in r["evidence"]], components=fields), []
        if n.type in {"FluxGuidance", "ConditioningZeroOut", "ConditioningCombine", "ConditioningConcat", "Reroute"}:
            names = {"FluxGuidance": ["conditioning"], "ConditioningZeroOut": ["conditioning"],
                     "ConditioningCombine": ["conditioning_1", "conditioning_2"],
                     "ConditioningConcat": ["conditioning_to", "conditioning_from"], "Reroute": [""]}[n.type]
            records, guidance = [], []
            for key in names:
                src, ev = self.follow(n, key, trail)
                rec, guides = self.conditioning(src, ev, visited)
                records.append(rec)
                guidance.extend(guides)
            if n.type == "FluxGuidance":
                guidance.append(self.scalar(n, "guidance", trail))
            if n.type in {"ConditioningCombine", "ConditioningConcat", "ConditioningZeroOut"}:
                ok = all(r["status"] in ("direct", "linked", "inferred") for r in records)
                return result({"operation": n.type, "inputs": [r["value"] for r in records]} if ok else None,
                              "inferred" if ok and any(r["status"] == "inferred" for r in records) else "linked" if ok else "missing",
                              None if ok else "unresolved conditioning branch",
                              [e for r in records for e in r["evidence"]], components=records), guidance
            return records[0], guidance
        raise Unresolved(f"unsupported conditioning semantics: {n.type}", list(trail) + self.evidence(n))

    def latent(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type in {"EmptyLatentImage", "EmptySD3LatentImage"}:
            return {k: self.scalar(n, k, trail) for k in ("width", "height", "batch_size")}, []
        if n.type in {"LatentUpscale", "LatentUpscaleBy", "Reroute"}:
            src, ev = self.follow(n, "" if n.type == "Reroute" else "samples", trail)
            dims, sources = self.latent(src, ev, visited)
            if n.type == "LatentUpscale":
                for k in ("width", "height"):
                    r = self.scalar(n, k, trail)
                    # Zero means auto/aspect-preserving; do not invent resulting dimensions.
                    dims[k] = r if r["value"] != 0 else result(reason="automatic upscale dimension requires image geometry", evidence=r["evidence"])
            if n.type == "LatentUpscaleBy":
                scale = self.scalar(n, "scale_by", trail)
                for k in ("width", "height"):
                    dims[k] = result(reason="scaled latent dimension requires ComfyUI rounding semantics", evidence=dims[k]["evidence"] + scale["evidence"])
            return dims, sources
        if n.type == "VAEEncode":
            src, ev = self.follow(n, "pixels", trail)
            sources = self.images(src, ev, visited)
            return {k: result(reason="source image geometry/batch unavailable from JSON", evidence=ev + self.evidence(n))
                    for k in ("width", "height", "batch_size")}, sources
        raise Unresolved(f"unsupported latent semantics: {n.type}", list(trail) + self.evidence(n))

    def images(self, n, trail=(), visited=frozenset()):
        visited = self.guard(n, trail, visited)
        if n.type == "LoadImage":
            return [self.scalar(n, "image", trail)]
        if n.type in {"ImageScale", "ImageScaleBy", "ImageUpscaleWithModel", "Reroute"}:
            src, ev = self.follow(n, "" if n.type == "Reroute" else "image", trail)
            return self.images(src, ev, visited)
        raise Unresolved(f"unsupported source-image semantics: {n.type}", list(trail) + self.evidence(n))

    def vae(self, n, trail):
        if n.type == "VAELoader":
            return self.scalar(n, "vae_name", trail)
        if n.type in {"CheckpointLoaderSimple", "CheckpointLoader"}:
            r = self.scalar(n, "ckpt_name", trail)
            if r["status"] in ("direct", "linked", "inferred"):
                r["value"] = {"builtin_vae_of_checkpoint": r["value"]}
            return r
        return result(reason=f"unsupported VAE semantics: {n.type}", evidence=list(trail) + self.evidence(n))

    def descendants(self, sampler):
        # Do not cross another sampler; its output belongs to that sampler's record.
        queue, seen, found = [(sampler, [])], {sampler.instance}, []
        allowed = {"VAEDecode", "SaveImage", "PreviewImage", "ImageScale", "ImageScaleBy", "ImageUpscaleWithModel", "LatentUpscale", "LatentUpscaleBy", "Reroute"}
        while queue:
            cur, trail = queue.pop(0)
            for target in self.g.nodes:
                if target.instance in seen or target.type not in allowed:
                    continue
                for _name, edge in target.edges.items():
                    if not edge.error and edge.source == cur.id:
                        ev = trail + edge.evidence
                        found.append((target, ev))
                        queue.append((target, ev))
                        seen.add(target.instance)
                        break
        return found

    def collection(self, records, reason, empty_evidence=None):
        if not records:
            return result([], "linked", evidence=empty_evidence) if empty_evidence else result(reason=reason)
        ev = [e for r in records for e in r["evidence"]]
        ok = all(r["status"] in ("direct", "linked", "inferred") for r in records)
        status = "inferred" if ok and any(r["status"] == "inferred" for r in records) else "linked" if ok else "ambiguous" if any(r["status"] == "ambiguous" for r in records) else "missing"
        return result([r["value"] for r in records] if ok else None, status,
                      None if ok else "one or more collection members unresolved", ev, components=records)

    def resolve(self, s):
        fields = {k: result(reason=f"no supported connected evidence for {k}", evidence=self.evidence(s)) for k in FIELDS}
        for k in ("seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"):
            name = "noise_seed" if k == "seed" and s.type in {"KSamplerAdvanced", "SamplerCustom"} else k
            fields[k] = self.scalar(s, name)
        if s.type == "KSampler":
            for k in ("noise", "sigmas", "guider", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"):
                fields[k] = result(status="not_applicable", reason="not exposed by KSampler", evidence=self.evidence(s))
        elif s.type == "KSamplerAdvanced":
            for k in ("denoise", "noise", "sigmas", "guider"):
                fields[k] = result(status="not_applicable", reason="not exposed by KSamplerAdvanced", evidence=self.evidence(s))
        if s.type == "SamplerCustom":
            for k in ("noise", "guider"):
                fields[k] = result(status="not_applicable", reason="not exposed by SamplerCustom", evidence=self.evidence(s))
        holder, holder_ev = s, []
        if s.type == "SamplerCustomAdvanced":
            def noise():
                n, ev = self.follow(s, "noise")
                if n.type == "DisableNoise":
                    fields["seed"] = result(status="not_applicable", reason="DisableNoise connected", evidence=ev + self.evidence(n))
                    fields["add_noise"] = result(False, "linked", evidence=ev + self.evidence(n))
                    return result({"type": n.type}, "linked", evidence=ev + self.evidence(n))
                if n.type != "RandomNoise":
                    raise Unresolved(f"unsupported noise semantics: {n.type}", ev)
                fields["seed"] = self.scalar(n, "noise_seed", ev)
                fields["control_after_generate"] = self.scalar(n, "control_after_generate", ev)
                fields["add_noise"] = result(True, "linked", evidence=ev + self.evidence(n))
                return result({"type": n.type, "seed": fields["seed"]["value"]}, fields["seed"]["status"],
                              fields["seed"]["reason"], fields["seed"]["evidence"])
            fields["noise"] = self.safe(noise)
            if fields["noise"]["status"] in ("missing", "ambiguous"):
                fields["seed"] = fields["noise"].copy()
            try:
                holder, holder_ev = self.follow(s, "guider")
                if holder.type not in {"BasicGuider", "CFGGuider"}:
                    raise Unresolved(f"unsupported guider semantics: {holder.type}", holder_ev)
                fields["guider"] = result(holder.type, "linked", evidence=holder_ev + self.evidence(holder))
                fields["cfg"] = self.scalar(holder, "cfg", holder_ev) if holder.type == "CFGGuider" else result(status="not_applicable", reason="BasicGuider has no CFG", evidence=holder_ev)
            except Unresolved as exc:
                fields["guider"] = exc.record()
                holder = None
        if s.type in {"SamplerCustom", "SamplerCustomAdvanced"}:
            def selection():
                n, ev = self.follow(s, "sampler")
                if n.type != "KSamplerSelect":
                    raise Unresolved(f"unsupported sampler selection: {n.type}", ev)
                return self.scalar(n, "sampler_name", ev)
            fields["sampler_name"] = self.safe(selection)
            def sigmas():
                n, ev = self.follow(s, "sigmas")
                if n.type != "BasicScheduler":
                    raise Unresolved(f"unsupported sigma schedule: {n.type}", ev)
                components = {k: self.scalar(n, k, ev) for k in ("scheduler", "steps", "denoise")}
                fields.update(components)
                ok = all(v["status"] in ("direct", "linked", "inferred") for v in components.values())
                inferred = any(v["status"] == "inferred" for v in components.values())
                return result({"type": "BasicScheduler", "parameters": {k: v["value"] for k, v in components.items()}} if ok else None,
                              "inferred" if ok and inferred else "linked" if ok else "missing",
                              "schedule parameters; numeric sigmas are not recomputed" if ok else "incomplete schedule parameters",
                              ev + self.evidence(n), components=components)
            fields["sigmas"] = self.safe(sigmas)
            if fields["sigmas"]["status"] in ("missing", "ambiguous") and "components" not in fields["sigmas"]:
                for k in ("steps", "scheduler", "denoise"):
                    fields[k] = fields["sigmas"].copy()
            for k in ("start_at_step", "end_at_step", "return_with_leftover_noise"):
                fields[k] = result(status="not_applicable", reason="not exposed by custom sampler", evidence=self.evidence(s))
        model_lora_components, clip_lora_components, clip_lora_errors = [], [], []
        if holder is not None:
            try:
                n, ev = self.follow(holder, "model", holder_ev)
                name, loader, loras, components = self.model(n, ev)
                model_lora_components = components
                fields["model_name"], fields["model_loader"] = name, loader
                evs = [e for c in components for r in c["fields"].values() for e in r["evidence"]]
                ok = all(r["status"] in ("linked", "direct", "inferred") for c in components for r in c["fields"].values())
                fields["loras"] = result(loras if ok else None, "inferred" if ok and any(r["status"] == "inferred" for c in components for r in c["fields"].values()) else "linked" if ok else "missing",
                                         None if ok else "incomplete LoRA parameters", evs or name["evidence"], components=components)
            except Unresolved as exc:
                fields["model_name"] = exc.record()
                fields["model_loader"] = exc.record()
                fields["loras"] = exc.record()
            guides = []
            for side in ("positive", "negative"):
                key = "conditioning" if holder.type == "BasicGuider" and side == "positive" else side
                if holder.type == "BasicGuider" and side == "negative":
                    fields[side] = result(status="not_applicable", reason="BasicGuider has no negative input", evidence=holder_ev + self.evidence(holder))
                    continue
                try:
                    n, ev = self.follow(holder, key, holder_ev)
                    fields[side], guidance = self.conditioning(n, ev)
                    guides.extend(guidance)
                    try:
                        clip_lora_components.extend(self.conditioning_loras(n, ev))
                    except Unresolved as exc:
                        clip_lora_errors.append(exc.record())
                except Unresolved as exc:
                    fields[side] = exc.record()
            if guides:
                fields["guidance"] = combine(guides)
            all_components = {c["node_instance"]: c for c in model_lora_components + clip_lora_components}
            if all_components:
                parts = list(all_components.values())
                ok = all(r["status"] in ("direct", "linked", "inferred") for c in parts for r in c["fields"].values())
                inferred = any(r["status"] == "inferred" for c in parts for r in c["fields"].values())
                values = [{"node_instance": c["node_instance"], **{k: r["value"] for k, r in c["fields"].items()}} for c in parts]
                fields["loras"] = result(values if ok else None, "inferred" if ok and inferred else "linked" if ok else "missing",
                                         None if ok else "incomplete LoRA parameters",
                                         [e for c in parts for r in c["fields"].values() for e in r["evidence"]], components=parts)
            if fields["model_name"]["status"] in ("missing", "ambiguous"):
                if fields["loras"]["value"] == []:
                    fields["loras"] = result(reason="upstream model unresolved; absence of LoRAs not established", evidence=fields["loras"]["evidence"])
                fields["loras"]["coverage"] = "partial; upstream model unresolved"
            if clip_lora_errors:
                fields["loras"]["unresolved_clip_branches"] = clip_lora_errors
                if fields["loras"]["value"] == []:
                    fields["loras"] = result(reason="CLIP branches unresolved; cannot establish absence of LoRAs",
                                             evidence=fields["loras"]["evidence"], unresolved_clip_branches=clip_lora_errors)
        try:
            n, ev = self.follow(s, "latent_image")
            dims, sources = self.latent(n, ev)
            fields.update(dims)
            fields["source_images"] = self.collection(sources, "source image not resolved", ev if not sources else None)
        except Unresolved as exc:
            for k in ("width", "height", "batch_size", "source_images"):
                fields[k] = exc.record()
        vaes, outputs, upscales = [], [], []
        for n, ev in self.input_media_nodes(s) + self.descendants(s):
            if n.type in {"VAEDecode", "VAEEncode"}:
                vaes.append(self.safe(lambda n=n, ev=ev: self.vae(*self.follow(n, "vae", ev))))
            if n.type == "SaveImage":
                outputs.append(self.scalar(n, "filename_prefix", ev))
            if n.type in UPSCALE:
                parts = {k: self.scalar(n, k, ev) for k in UI_ORDER[n.type].split()}
                if n.type == "ImageUpscaleWithModel":
                    def upscale_model(n=n, ev=ev):
                        src, path = self.follow(n, "upscale_model", ev)
                        if src.type != "UpscaleModelLoader":
                            raise Unresolved("unsupported upscale model loader", path)
                        return self.scalar(src, "model_name", path)
                    parts["model_name"] = self.safe(upscale_model)
                ok = all(r["status"] in ("direct", "linked", "inferred") for r in parts.values())
                upscales.append(result({"node_instance": n.instance, "type": n.type, **{k: r["value"] for k, r in parts.items()}} if ok else None,
                                       "inferred" if ok and any(r["status"] == "inferred" for r in parts.values()) else "linked" if ok else "missing",
                                       None if ok else "incomplete upscale parameters", [e for r in parts.values() for e in r["evidence"]], components=parts))
        fields["vae"] = combine(vaes, "no supported downstream VAE connection")
        fields["output_prefixes"] = self.collection(outputs, "no connected SaveImage prefix")
        fields["upscale"] = self.collection(upscales, "no supported connected upscale settings")
        context = {"occurrence_id": self.g.occurrence.id, "graph_id": self.g.id, "sampler_id": s.instance}
        def attach_context(item):
            if isinstance(item, dict):
                if "status" in item and "evidence" in item:
                    item.update(context)
                for value in list(item.values()):
                    attach_context(value)
            elif isinstance(item, list):
                for value in item:
                    attach_context(value)
        attach_context(fields)
        required = ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "model_name", "positive", "negative", "width", "height", "batch_size")
        known = {"direct", "linked", "inferred", "not_applicable"}
        return {**context, "sampler_node_id": s.id, "sampler_type": s.type,
                "association": "graph_structure" if self.g.id else "unassociated_standalone",
                "parameter_completeness": "complete_for_supported_core_fields" if all(fields[k]["status"] in known for k in required) else "partial",
                "graph_completeness": "partial" if self.g.partial else self.g.record()["graph_completeness"],
                "acquisition_method": self.g.occurrence.method,
                "ui_state_caveat": "UI widgets may be editor/next-seed state" if self.g.ui else None,
                "fields": fields}

    def records(self):
        return [self.resolve(n) for n in self.g.nodes if n.type in SAMPLERS]
