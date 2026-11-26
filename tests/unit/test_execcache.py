"""ComfyUI's execution cache read from synthetic CPython objects: signatures, cache entries, ties and links."""

from __future__ import annotations

import copy
import json
import struct
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from comfytrace import cli, evidence, execcache, memrecover, pyheap, torchmem
from comfytrace.carve import IdAllocator
from comfytrace.correlate import CACHE_KEY, correlate
from comfytrace.models import Completeness, LinkClass
from comfytrace.torchmem import DTYPE_OFFSET, NDIM_OFFSET, NUMEL_OFFSET, SIZES_OFFSET, ImageTensor
from tests.synthetic import api_prompt
from tests.unit.test_memory_recovery import FakePython

JOB_A, JOB_B = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"


def is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def signature(prompt: dict[str, Any], node_id: str) -> list[Any]:
    """ComfyUI's CacheKeySetInputSignature.get_node_signature, before to_hashable."""
    ancestors: list[str] = []
    order: dict[str, int] = {}

    def walk(current: str) -> None:
        inputs = prompt[current]["inputs"]
        for key in sorted(inputs):
            if is_link(inputs[key]) and inputs[key][0] not in order:
                ancestors.append(inputs[key][0])
                order[inputs[key][0]] = len(ancestors) - 1
                walk(inputs[key][0])

    def immediate(current: str) -> list[Any]:
        node = prompt[current]
        out: list[Any] = [node["class_type"], 0]  # is_changed False
        for key in sorted(node["inputs"]):
            value = node["inputs"][key]
            out.append((key, ("ANCESTOR", order[value[0]], value[1]) if is_link(value) else value))
        return out

    walk(node_id)
    return [immediate(node_id), *(immediate(a) for a in ancestors)]


class CachePython(FakePython):
    """FakePython plus what the execution cache is made of: frozensets, general-key dicts, a CacheEntry named
    tuple and torch.Tensor wrappers around a TensorImpl."""

    def __init__(self) -> None:
        super().__init__()
        name_va = self.dll.alloc(16)
        self.dll.raw(name_va, b"frozenset\0")
        self.dll.raw(name_va - 1, b"\0")
        self.types["frozenset"] = self.dll.alloc(64)
        self.dll.q(self.types["frozenset"] + 24, name_va)
        self.dll.q(self.types["frozenset"] + 8, self.types["type"])
        self.heap_types = {name: self._heap_type(name) for name in ("CacheEntry", "Tensor", "ModelPatcher")}

    def _heap_type(self, name: str) -> int:
        name_va = self.heap.alloc(len(name) + 1)
        self.heap.raw(name_va, name.encode() + b"\0")
        va = self.heap.alloc(64)
        self.heap.q(va + 8, self.types["type"])
        self.heap.q(va + 24, name_va)
        return va

    def words(self, *values: int) -> int:
        va = self.heap.alloc(8 * len(values))
        for i, value in enumerate(values):
            self.heap.q(va + 8 * i, value)
        return va

    def frozenset(self, members: list[int]) -> int:
        slots = 8
        while slots * 3 // 5 < len(members):
            slots *= 2
        table = self.heap.alloc(16 * slots)
        for i, member in enumerate(members):
            self.heap.q(table + 16 * i, member)
            self.heap.q(table + 16 * i + 8, 0x1234 + i)
        return self.words(1, self.types["frozenset"], len(members), len(members), slots - 1, table, 0, 0)

    def hashable(self, value: Any) -> int:
        """ComfyUI's to_hashable, written as objects."""
        if value is None or isinstance(value, str | int | float):
            return self.obj(value)
        pairs = sorted(value.items()) if isinstance(value, dict) else list(enumerate(value))
        return self.frozenset([self.words(1, self.types["tuple"], 2, self.obj(k), self.hashable(v))
                               for k, v in pairs])

    def general_dict(self, pairs: list[tuple[int, int]]) -> int:
        """A dict whose keys are not all str: 24-byte entries (hash, key, value)."""
        log2_size = 3
        while (1 << log2_size) * 2 // 3 < len(pairs):
            log2_size += 1
        keys = self.heap.alloc(32 + (1 << log2_size) + 24 * len(pairs))
        struct.pack_into("<BBB", self.heap.buf, keys + 8 - self.heap.base, log2_size, log2_size, 0)
        self.heap.q(keys + 24, len(pairs))
        entries = keys + 32 + (1 << log2_size)
        for i, (key, value) in enumerate(pairs):
            self.heap.q(entries + 24 * i, 0x5000 + i)
            self.heap.q(entries + 24 * i + 8, key)
            self.heap.q(entries + 24 * i + 16, value)
        return self.words(1, self.types["dict"], len(pairs), 0, keys, 0)

    def tensor(self, sizes: list[int]) -> tuple[int, int]:
        """(torch.Tensor object, its TensorImpl)."""
        impl = self.heap.alloc(0xB0)
        self.heap.q(impl + NDIM_OFFSET, len(sizes))
        for i, size in enumerate(sizes):
            self.heap.q(impl + SIZES_OFFSET + 8 * i, size)
        self.heap.q(impl + NUMEL_OFFSET, int(np.prod(sizes)))
        self.heap.q(impl + DTYPE_OFFSET, 6)
        return self.words(1, self.heap_types["Tensor"], impl), impl

    def cache_entry(self, ui: Any, outputs: list[list[int]]) -> int:
        slots = []
        for items in outputs:
            vector = self.words(*items) if items else self.heap.alloc(8)
            slots.append(self.words(1, self.types["list"], len(items), vector, max(1, len(items))))
        vector = self.words(*slots) if slots else self.heap.alloc(8)
        out_list = self.words(1, self.types["list"], len(slots), vector, max(1, len(slots)))
        return self.words(1, self.heap_types["CacheEntry"], 2, self.obj(ui), out_list)

    def key_set(self, node_ids: dict[str, int]) -> int:
        """A str-keyed dict whose values are the given addresses, as ComfyUI's CacheKeySet.keys: node id -> the
        frozenset that keys the cache."""
        va = self.obj(dict.fromkeys(node_ids, 0))
        keys = self.heap.q_at(va + 32)
        entries = keys + 32 + (1 << self.heap.buf[keys + 9 - self.heap.base])
        for i, target in enumerate(node_ids.values()):
            self.heap.q(entries + 16 * i + 8, target)
        return va


def cached_job(py: CachePython, prompt: dict[str, Any], batch: int = 1) -> dict[str, Any]:
    """The cache as ComfyUI leaves it after running ``prompt``: one entry per node, the picture tensor under the
    VAEDecode node and again, with the UI's file list, under SaveImage."""
    keys = {node_id: py.hashable(signature(prompt, node_id)) for node_id in prompt}
    image, impl = py.tensor([batch, 48, 64, 3])
    latent, _ = py.tensor([batch, 4, 6, 8])
    model = py.words(1, py.heap_types["ModelPatcher"])
    files = [{"filename": f"ComfyUI_{i:05d}_.png", "subfolder": "", "type": "output"} for i in range(1, batch + 1)]
    latent_dict = py.key_set({"samples": latent})  # a LATENT is a dict holding the tensor
    values = {
        "4": py.cache_entry(None, [[model], [model], [model]]),
        "3": py.cache_entry(None, [[latent_dict]]),
        "8": py.cache_entry(None, [[image]]),
        "9": py.cache_entry({"meta": {"node_id": "9"}, "output": {"images": files}}, [[image]]),
    }
    cache = py.general_dict([(keys[node_id], value) for node_id, value in values.items()])
    stamps = py.general_dict([(keys[node_id], py.obj(1793895741.72)) for node_id in values])
    py.key_set({node_id: keys[node_id] for node_id in ("3", "8", "9")})
    return {"cache": cache, "stamps": stamps, "impl": impl, "keys": keys}


def queue_item(prompt_id: str, prompt: dict[str, Any], address: int = 0x1000) -> pyheap.QueueItem:
    item = pyheap.QueueItem(4242, prompt_id, 0, address, 1, 5, prompt, {})
    item.history = {"status": {"status_str": "success", "completed": True},
                    "outputs": {"9": {"images": [{"filename": "ComfyUI_00001_.png"},
                                                 {"filename": "ComfyUI_00002_.png"}]}}}
    return item


def image_tensor(impl: int, batch_index: int = 0, seed: int = 0) -> ImageTensor:
    rgb = np.random.default_rng(seed).integers(0, 256, (48, 64, 3)).astype(np.uint8)
    return ImageTensor(4242, impl, 0x9000, 0x200000, [1, 48, 64, 3], [9216, 192, 3, 1], batch_index, rgb, True,
                       frozenset(), (0x200000, 4096))


def test_a_cache_entry_is_read_with_its_key_its_node_id_its_time_and_what_it_holds():
    py = CachePython()
    prompt = api_prompt(seed=222)
    made = cached_job(py, prompt)
    entries, summary = execcache.extract(py.memory())
    assert summary["status"] == "FOUND" and summary["entries"] == 4 and summary["tensors_held"] == 2
    by_class = {e.class_type: e for e in entries}
    assert sorted(by_class) == ["CheckpointLoaderSimple", "KSampler", "SaveImage", "VAEDecode"]
    decode = by_class["VAEDecode"]
    # Node "0" is the cached node, "1" its first ancestor in ComfyUI's order (samples -> the KSampler).
    assert decode.graph["0"]["inputs"] == {"samples": ["1", 0], "vae": ["3", 2]}
    assert decode.graph["1"]["class_type"] == "KSampler" and decode.graph["1"]["inputs"]["seed"] == 222
    assert decode.graph["5"]["inputs"]["text"] == prompt["6"]["inputs"]["text"]
    assert decode.value_kind == "CacheEntry" and decode.node_ids == ["8"], "the key set names the node"
    assert decode.bookkeeping == {"timestamp_utc": "2026-11-05T16:22:21.720Z"}, "time.time() of the host, in UTC"
    [[image]] = decode.outputs
    assert (image["type"], image["tensorimpl"], image["sizes"]) == ("Tensor", made["impl"], [1, 48, 64, 3])
    [[latent]] = by_class["KSampler"].outputs
    assert latent["members"]["samples"]["sizes"] == [1, 4, 6, 8]
    assert [[o["type"] for o in slot] for slot in by_class["CheckpointLoaderSimple"].outputs] == [["ModelPatcher"]] * 3
    assert by_class["SaveImage"].ui_filenames == ["ComfyUI_00001_.png"]
    assert by_class["SaveImage"].ui["meta"] == {"node_id": "9"}


def test_the_tie_names_the_node_that_made_the_tensor_and_the_one_job_with_its_key():
    py = CachePython()
    job_a, job_b = api_prompt(seed=111), api_prompt(seed=222)
    made = cached_job(py, job_b)
    entries, _ = execcache.extract(py.memory())
    tie = execcache.image_ties(entries)[(4242, made["impl"])]
    # SaveImage holds the same tensor; the maker has the shorter key.
    assert tie["class_type"] == "VAEDecode" and [h["class_type"] for h in tie["also_held_by"]] == ["SaveImage"]
    assert execcache.matching_nodes(tie["graph"], job_a) == [], "another seed upstream is another key"
    assert execcache.matching_nodes(tie["graph"], job_b) == ["8"]
    jobs = execcache.job_matches(tie["graph"], [queue_item(JOB_A, job_a), queue_item(JOB_B, job_b)])
    assert jobs == [{"prompt_id": JOB_B, "node_ids": ["8"], "feeds": ["9"], "reused": False}]


def test_matching_compares_inputs_and_topology_with_a_prerun_prompt():
    prompt = api_prompt(seed=5)
    graph = execcache.signature_graph(json.loads(json.dumps(signature(prompt, "8"))))
    assert execcache.matching_nodes(graph, prompt) == ["8"]
    graph["1"]["is_changed"] = "0f3a"  # a pre-run prompt omits the node's runtime annotation
    assert execcache.matching_nodes(graph, prompt) == ["8"]
    for change in (lambda p: p["3"]["inputs"].update(steps=21), lambda p: p["7"]["inputs"].update(text="x"),
                   lambda p: p["8"]["inputs"].update(vae=["4", 1]), lambda p: p["3"]["inputs"].update(extra=1),
                   lambda p: p["3"].update(class_type="KSamplerAdvanced")):
        other = copy.deepcopy(prompt)
        change(other)
        assert execcache.matching_nodes(graph, other) == []
    other = copy.deepcopy(prompt)
    other["9"]["inputs"]["filename_prefix"] = "elsewhere"  # downstream of the node: not part of its key
    assert execcache.matching_nodes(graph, other) == ["8"]


def test_matching_cache_signature_rejects_a_conflicting_load_image_ancestor():
    prompt = api_prompt(seed=5)
    prompt["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
    prompt["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    graph = execcache.signature_graph(json.loads(json.dumps(signature(prompt, "8"))))
    cached = next(node for node in graph.values() if node["class_type"] == "LoadImage")
    cached["is_changed"] = ["current"]
    assert execcache.matching_nodes(graph, prompt) == ["8"]
    prompt["10"]["is_changed"] = ["previous"]
    assert execcache.matching_nodes(graph, prompt) == []
    assert execcache.job_matches(graph, [queue_item(JOB_B, prompt)]) == []
    prompt["10"]["is_changed"] = ["current"]
    assert execcache.matching_nodes(graph, prompt) == ["8"]


def test_evidence_sidecar_keeps_cache_settings_and_job_matches():
    from comfytrace.carve import execution_parameters

    py = CachePython()
    job = api_prompt(seed=222)
    cached_job(py, job)
    entries, _ = execcache.extract(py.memory())
    document = [{**e.to_dict(), "jobs": execcache.job_matches(e.graph, [queue_item(JOB_B, job)])} for e in entries]
    assert json.loads(json.dumps(document)) == document, "the sidecar entry is plain JSON"
    decode = next(entry for entry in document if entry["class_type"] == "VAEDecode")
    params = execution_parameters(decode["graph"], {})
    assert params["seed"] == 222 and params["width"] == 64 and params["height"] == 48
    assert decode["outputs"][0][0]["type"] == "Tensor" and decode["outputs"][0][0]["sizes"] == [1, 48, 64, 3]
    assert decode["jobs"] == [{"prompt_id": JOB_B, "node_ids": ["8"], "feeds": ["9"], "reused": False}]
    assert evidence.evidence_document([], [], {}, execution_cache=document)["execution_cache"] == document
    assert "execution_cache" not in evidence.evidence_document([], [], {}, execution_cache=None)


def test_a_dict_keyed_by_other_frozensets_is_not_a_cache():
    py = CachePython()
    plain = py.frozenset([py.obj("a"), py.obj("b")])
    py.general_dict([(plain, py.obj(1))])
    entries, summary = execcache.extract(py.memory())
    assert entries == [] and summary["status"] == "NOT_FOUND" and summary["undecodable_keys"] == 0


def _records(py: CachePython, prompts: dict[str, dict[str, Any]], cached: dict[str, Any]) -> tuple[list, list]:
    items = [queue_item(prompt_id, prompt, 0x1000 * (n + 1)) for n, (prompt_id, prompt) in enumerate(prompts.items())]
    made = cached_job(py, cached)
    entries, _ = execcache.extract(py.memory())
    tie = execcache.image_ties(entries)[(4242, made["impl"])]
    tie["jobs"] = execcache.job_matches(tie["graph"], items)
    candidate = memrecover.from_tensor(image_tensor(made["impl"]), lambda va: None, tie)
    return memrecover.image_records([candidate]), memrecover.workflow_records(items, IdAllocator("WFPY"))


def test_a_cached_tensor_and_its_jobs_complete_prompt_are_an_e2_finding_without_regeneration():
    job_a, job_b = api_prompt(seed=111), api_prompt(seed=222)
    images, workflows = _records(CachePython(), {JOB_A: job_a, JOB_B: job_b}, job_b)
    [image] = images
    assert image.evidence["execution_cache"]["class_type"] == "VAEDecode"
    result = correlate(images, workflows)
    [link] = result.links
    assert (link.link_class, link.matched_keys) == (LinkClass.STRICT, [CACHE_KEY])
    linked = next(w for w in workflows if w.workflow_id == link.workflow_id)
    assert linked.execution_parameters["prompt_id"] == JOB_B and linked.completeness == Completeness.COMPLETE
    assert result.tier_summary["E2"] == 1 and result.tier_summary["untiered_orphan_images"] == 0
    assert image.link_status == LinkClass.STRICT and image.associated_workflow_id == linked.workflow_id


def test_a_tensor_without_cache_provenance_stays_unassociated():
    items = [queue_item(JOB_A, api_prompt(seed=111)), queue_item(JOB_B, api_prompt(seed=222))]
    candidate = memrecover.from_tensor(image_tensor(0x2000), lambda va: None, None)
    images = memrecover.image_records([candidate])
    workflows = memrecover.workflow_records(items, IdAllocator("WFPY"))
    result = correlate(images, workflows)

    [image] = images
    assert image.link_status == LinkClass.UNASSOCIATED
    assert image.associated_workflow_id is None
    assert result.tier_summary["E2"] == 0
    assert result.tier_summary["untiered_orphan_images"] == 1


def test_the_cached_tensor_and_the_jobs_own_png_with_the_same_pixels_are_one_output():
    from PIL import Image

    from comfytrace.carve import PngCarver
    from tests.synthetic import png_bytes

    job_a, job_b = api_prompt(seed=111), api_prompt(seed=222)
    images, workflows = _records(CachePython(), {JOB_A: job_a, JOB_B: job_b}, job_b)
    saved = png_bytes(Image.fromarray(image_tensor(0).rgb), {"prompt": json.dumps(job_b)})
    [png] = PngCarver(IdAllocator("CARVE-PNG")).carve_stream(saved)
    assert png.pixel_sha256 == images[0].pixel_sha256, "ComfyUI saves exactly the pixels the tensor renders to"
    result = correlate([png, *images], workflows)
    assert result.tier_summary["E2"] == 1 and result.tier_summary["E3"] == 0
    [job] = [f for f in result.findings if f.tier == "E2"]
    assert job.artifact_ids[:2] == ["CARVE-PNG-001", "TENSOR-001"]
    assert "CANONICAL_PROMPT_HASH, EXECUTION_CACHE_SIGNATURE" in job.rationale and "2 copies" in job.rationale
    # Another picture of the same job (a batch image) stays a finding of its own.
    other = memrecover.image_records([memrecover.from_tensor(
        image_tensor(0, 1, seed=9), lambda va: None, images[0].evidence["execution_cache"])])
    assert correlate([png, *other], workflows).tier_summary["E2"] == 2


def test_jobs_that_differ_only_downstream_share_the_key_and_leave_the_image_ambiguous():
    job = api_prompt(seed=222)
    renamed = copy.deepcopy(job)
    renamed["9"]["inputs"]["filename_prefix"] = "second"
    images, workflows = _records(CachePython(), {JOB_A: job, JOB_B: renamed}, job)
    result = correlate(images, workflows)
    assert {link.link_class for link in result.links} == {LinkClass.AMBIGUOUS}
    assert result.tier_summary["E2"] == 0 and result.tier_summary["untiered_ambiguous_images"] == 1
    # Identical execution graphs can still describe different submissions.
    images, workflows = _records(CachePython(), {JOB_A: job, JOB_B: copy.deepcopy(job)}, job)
    result = correlate(images, workflows)
    assert len(result.links) == 2
    assert {link.link_class for link in result.links} == {LinkClass.AMBIGUOUS}


def test_a_runtime_annotation_on_one_copy_does_not_create_a_second_cache_producer():
    job = api_prompt(seed=222)
    job["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
    job["10"] = {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}}
    observed = copy.deepcopy(job)
    observed["10"]["is_changed"] = ["current"]
    images, _ = _records(CachePython(), {JOB_A: job}, job)
    cache = images[0].evidence["execution_cache"]
    next(node for node in cache["graph"].values() if node["class_type"] == "LoadImage")["is_changed"] = ["current"]
    workflows = memrecover.workflow_records(
        [queue_item(JOB_A, job), queue_item(JOB_A, observed, 0x2000)], IdAllocator("WFPY"),
    )
    unbound = copy.deepcopy(workflows[0])
    unbound.workflow_id = "WF-001"
    unbound.execution_parameters.pop("prompt_id")

    result = correlate(images, [unbound, *workflows])
    [link] = result.links
    assert link.link_class == LinkClass.STRICT and result.tier_summary["E2"] == 1
    assert link.workflow_id == workflows[1].workflow_id

    # An actual change downstream of the cached tensor still leaves competing graph evidence.
    observed["9"]["inputs"]["filename_prefix"] = "different-placement"
    conflicting = memrecover.workflow_records(
        [queue_item(JOB_A, job), queue_item(JOB_A, observed, 0x2000)], IdAllocator("WFPY"),
    )
    result = correlate(images, conflicting)
    assert len(result.links) == 2 and result.tier_summary["E2"] == 0
    assert all(link.link_class == LinkClass.AMBIGUOUS for link in result.links)


def test_cache_signature_and_virtual_addresses_stay_with_their_own_process():
    job = api_prompt(seed=222)
    images, workflows = _records(CachePython(), {JOB_A: job, JOB_B: copy.deepcopy(job)}, job)
    [image] = images
    workflows[1].source_process_id = 4343
    for location in workflows[1].execution_parameters["source_locations"]:
        location["process_id"] = 4343

    [link] = correlate(images, workflows).links
    assert link.workflow_id == workflows[0].workflow_id and link.link_class == LinkClass.STRICT

    tie = image.evidence["execution_cache"]
    holders = [{"process_id": pid, "cache_dict": tie["cache_dict"], "key_address": tie["key_address"]}
               for pid in (4242, 4343)]
    cli._name_cached_images(holders, images)
    assert holders[0]["image_artifacts"][0]["artifact_id"] == image.artifact_id
    assert "image_artifacts" not in holders[1]


def test_python_stage_scopes_job_matches_by_process(monkeypatch):
    job = api_prompt(seed=222)
    python = CachePython()
    made = cached_job(python, job)
    entries, _ = execcache.extract(python.memory())
    workers = [SimpleNamespace(pid=pid, prefetch=lambda: None, stale={}, store_status={}, physical_of=lambda _: None)
               for pid in (4242, 4343)]
    items = {pid: replace(queue_item(job_id, job), process_id=pid) for pid, job_id in ((4242, JOB_A), (4343, JOB_B))}

    monkeypatch.setattr(pyheap, "extract", lambda mem: SimpleNamespace(
        items=[items[mem.pid]], pixel_buffers=[], status="AVAILABLE", summary=lambda: {"status": "AVAILABLE"}))
    monkeypatch.setattr(execcache, "extract", lambda mem: ([replace(e, process_id=mem.pid) for e in entries], {}))
    monkeypatch.setattr(torchmem, "image_tensors",
                        lambda mem: [replace(image_tensor(made["impl"]), process_id=mem.pid)])

    _, candidates, _, _, cache = cli._python_stage(SimpleNamespace(notes={}, python_processes=workers))

    assert {c.process_id: [j["prompt_id"] for j in c.execution_cache["jobs"]] for c in candidates} == {
        4242: [JOB_A], 4343: [JOB_B],
    }
    assert all([j["prompt_id"] for j in entry["jobs"]] == [items[entry["process_id"]].prompt_id] for entry in cache)


def test_the_scorecard_lists_a_cached_tensor_on_its_own_output_row_and_batch_index():
    job = api_prompt(seed=222, batch_size=2)
    py = CachePython()
    item = queue_item(JOB_B, job)
    made = cached_job(py, job, batch=2)
    entries, _ = execcache.extract(py.memory())
    tie = execcache.image_ties(entries)[(4242, made["impl"])]
    tie["jobs"] = execcache.job_matches(tie["graph"], [item])
    candidates = [memrecover.from_tensor(image_tensor(made["impl"], b, seed=b), lambda va: None, tie)
                  for b in (0, 1)]
    images = memrecover.image_records(candidates)
    workflows = memrecover.workflow_records([item], IdAllocator("WFPY"))
    result = correlate(images, workflows)
    assert result.tier_summary["E2"] == 2, "batch images are separate findings"
    card = evidence.scorecard([item], workflows, "ab" * 32, result.links, images)
    rows = {r["batch_index"]: r["image"]["candidates"] for r in card["rows"]}
    assert [[c["artifact_id"] for c in rows[b]] for b in (0, 1)] == [["TENSOR-001"], ["TENSOR-002"]]
    assert rows[1][0]["role"] == "OUTPUT" and rows[1][0]["batch_index"] == 1 and rows[1][0]["cache_node_ids"] == ["8"]
    assert card["summary"]["outputs_with_execution_cache_image"] == 2


def test_a_cached_tensor_feeding_several_outputs_has_no_unique_output_position():
    job = api_prompt(seed=222)
    job["10"] = {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
    item = queue_item(JOB_A, job)
    item.history["outputs"]["9"]["images"] = item.history["outputs"]["9"]["images"][:1]
    item.history["outputs"]["10"] = {"images": [{"filename": "preview.png"}]}
    images, workflows = _records(CachePython(), {JOB_A: job}, job)
    card = evidence.scorecard([item], workflows, "ab" * 32, correlate(images, workflows).links, images)

    assert {row["image"]["status"] for row in card["rows"]} == {"JOB_LINKED"}
    for row in card["rows"]:
        [candidate] = row["image"]["candidates"]
        assert candidate["candidate_output_nodes"] == ["10", "9"]
        assert candidate["role"] == "AMBIGUOUS_OUTPUT" and "batch_index" not in candidate


def _static_object(py: CachePython, name: str | None) -> int:
    """A header-only object in the DLL with None's shape (basicsize 16, itemsize 0), like Ellipsis; ``name`` None
    leaves its tp_name pointing at an unreadable page."""
    if name is None:
        name_va = 0x3000_0000_0000
    else:
        name_va = py.dll.alloc(16)
        py.dll.raw(name_va, name.encode() + b"\0")
    kind = py.dll.alloc(64)
    py.dll.q(kind + 8, py.types["type"])
    py.dll.q(kind + 24, name_va)
    py.dll.q(kind + 32, 16)
    obj = py.dll.alloc(16)
    py.dll.q(obj + 8, kind)
    return obj


def _heap_without_none_type(py: CachePython) -> pyheap.CPythonHeap:
    """A decoder that must read the type name directly because NoneType was absent from the type table."""
    heap = pyheap.CPythonHeap(py.memory())
    heap.types.pop("NoneType")
    heap.names = {address: name for name, address in heap.types.items()}
    return heap


def test_a_signature_slot_that_cannot_be_read_is_never_none():
    py = CachePython()
    pair = lambda value: py.words(1, py.types["tuple"], 2, py.obj("seed"), value)  # noqa: E731
    gone = py.frozenset([pair(0x3000_0000_0000)])  # a slot pointing at an unmapped page
    null = py.frozenset([pair(0)])
    renamed = py.frozenset([pair(_static_object(py, None))])  # its type's name cannot be read
    plain = py.hashable({"seed": None})
    for heap in (pyheap.CPythonHeap(py.memory()), _heap_without_none_type(py)):
        for broken in (gone, null, renamed):
            try:
                value = execcache.decode_hashable(heap, broken)
            except pyheap.Unreadable:
                continue
            raise AssertionError(f"an unreadable slot decoded as {value!r}")
        assert execcache.decode_hashable(heap, plain) == {"seed": None}, "a real None stays"


def test_a_static_object_shaped_like_none_is_not_none_when_its_name_says_otherwise():
    py = CachePython()
    ellipsis = _static_object(py, "ellipsis")
    pair = py.words(1, py.types["tuple"], 2, py.obj("seed"), ellipsis)
    signature_item = py.frozenset([pair])
    plain = py.hashable({"seed": None})
    for heap in (pyheap.CPythonHeap(py.memory()), _heap_without_none_type(py)):
        try:
            value = execcache.decode_hashable(heap, signature_item)
        except pyheap.Unreadable:
            continue
        raise AssertionError(f"Ellipsis decoded as {value!r}")
    heap = _heap_without_none_type(py)
    assert execcache.decode_hashable(heap, plain) == {"seed": None}
    assert execcache.decode_hashable(heap, py.none) is None


def test_an_unhashable_or_non_finite_value_never_strictly_matches_a_prompt():
    py = CachePython()
    unhashable = py.words(1, py._heap_type("Unhashable"))
    items = py.frozenset([py.words(1, py.types["tuple"], 2, py.obj(0), unhashable),
                          py.words(1, py.types["tuple"], 2, py.obj(1), py.obj(float("nan")))])
    heap = pyheap.CPythonHeap(py.memory())
    assert execcache.decode_hashable(heap, items) == [{execcache.UNHASHABLE: "Unhashable"}, {execcache.NAN: "float"}]

    prompt = api_prompt(seed=5)
    graph = execcache.signature_graph(json.loads(json.dumps(signature(prompt, "8"))))
    assert execcache.matching_nodes(graph, prompt) == ["8"]
    for marker in ({execcache.UNHASHABLE: "Unhashable"}, {execcache.NAN: "float"}):
        cached = copy.deepcopy(graph)
        cached["1"]["inputs"]["seed"] = marker
        other = copy.deepcopy(prompt)
        other["3"]["inputs"]["seed"] = copy.deepcopy(marker)  # even a prompt that spells out the same marker
        assert execcache.matching_nodes(cached, other) == [], marker
    plain = copy.deepcopy(graph)
    plain["1"]["inputs"]["seed"] = "<NaN>"  # an ordinary string of that spelling is still an ordinary literal
    other = copy.deepcopy(prompt)
    other["3"]["inputs"]["seed"] = "<NaN>"
    assert execcache.matching_nodes(plain, other) == ["8"]
    marked = copy.deepcopy(graph)
    marked["1"]["is_changed"] = {execcache.UNHASHABLE: "Unhashable"}
    assert execcache.matching_nodes(marked, prompt) == ["8"], "a pre-run prompt records no is_changed"
    other = copy.deepcopy(prompt)
    other["3"]["is_changed"] = {execcache.UNHASHABLE: "Unhashable"}
    assert execcache.matching_nodes(marked, other) == []
