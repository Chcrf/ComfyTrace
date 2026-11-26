"""Key-gated workflow/image correlation and finding-level evidence tiers.

Only strong keys link evidence: a canonical prompt-graph hash or an uncontradicted prompt_id (STRICT), or
complete agreement of seed, sampler, steps and image size with no contradicting field (SUPPORTED).
Memory proximity is never a key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from comfytrace.carve import (
    IdAllocator,
    canonical_json,
    execution_parameters,
    extract_prompt_graph,
    graph_dependencies,
    graph_is_complete,
)
from comfytrace.execcache import matching_nodes
from comfytrace.fragments import (
    CUT,
    Reference,
    TextFragment,
    agrees_with,
    api_nodes,
    ui_node_type,
    ui_nodes,
    ui_parameters,
    ui_workflow_nodes,
)
from comfytrace.media import MEDIA_TYPES, embedded_encoding, is_embedded_text
from comfytrace.memrecover import OUTPUT_CONTEXT_FIELD, TERMINAL_WITHOUT_RECORDED_OUTPUTS
from comfytrace.models import (
    LINKED_CLASSES,
    CarveType,
    Completeness,
    CorrelationLink,
    Finding,
    LinkClass,
    RawArtifactRecord,
    Tier,
    WorkflowRecord,
    empty_tier_summary,
)

SUPPORTING_KEYS = ("seed", "sampler_name", "steps")
CONSISTENCY_KEYS = ("scheduler", "cfg_scale", "model_name", "filename_prefix", "positive_prompt", "negative_prompt")
CLASS_RANK: dict[str, int] = {LinkClass.STRICT: 2, LinkClass.SUPPORTED: 1}
PROMPT_ID_GRAPH_CONFLICT = "PROMPT_ID_GRAPH_CONFLICT"
INPUT_REVISION_UNVERIFIED = "INPUT_REVISION_UNVERIFIED"
GRAPH_CONFLICT_NOTE = "The prompt IDs match, but the recovered execution graphs disagree."
INPUT_USE_KEY = "workflow_input_uses"


@dataclass(frozen=True)
class EmbeddedPrompt:
    graph: dict[str, Any] | None
    canonical_sha256: str | None
    prompt_id: str | None
    client_id: str | None
    parameters: dict[str, Any]


def embedded_prompt(image: RawArtifactRecord) -> EmbeddedPrompt:
    """Reads the ComfyUI metadata an image carries in its tEXt/zTXt/iTXt chunks."""
    text = image.embedded_text
    graph = None
    runtime: dict[str, Any] = {}
    if "prompt" in text:
        try:
            classified = extract_prompt_graph(json.loads(text["prompt"]))
        except (json.JSONDecodeError, RecursionError):
            classified = None
        if classified is not None and classified[0] == "API_GRAPH":
            graph, runtime = classified[1], classified[2]
    prompt_id = text.get("prompt_id") or runtime.get("prompt_id")
    canonical_sha256 = None
    if graph:
        try:
            canonical_sha256 = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
        except (UnicodeError, ValueError, RecursionError):
            graph = None
    return EmbeddedPrompt(
        graph=graph,
        canonical_sha256=canonical_sha256,
        prompt_id=prompt_id,
        client_id=text.get("client_id") or runtime.get("client_id"),
        parameters=execution_parameters(graph, runtime) if graph else {},
    )


def _prompt_graphs_disagree(first: dict[str, Any], second: dict[str, Any], *, complete: bool) -> bool:
    if complete and first.keys() != second.keys():
        return True
    for node_id in first.keys() & second.keys():
        node = first[node_id]
        other = second[node_id]
        if not isinstance(node, dict) or not isinstance(other, dict):
            if complete:
                return True
            continue
        # ComfyUI adds runtime annotations after submission, including fingerprints of input files.
        fields = node.keys() & other.keys() & {"class_type", "is_changed"}
        if any(canonical_json(node[field]) != canonical_json(other[field]) for field in fields):
            return True
        inputs, other_inputs = node.get("inputs"), other.get("inputs")
        if complete and ("class_type" not in fields or canonical_json(inputs) != canonical_json(other_inputs)):
            return True
        if not complete and isinstance(inputs, dict) and isinstance(other_inputs, dict):
            if any(canonical_json(inputs[key]) != canonical_json(other_inputs[key])
                   for key in inputs.keys() & other_inputs.keys()):
                return True
    return False


class KeyMatcher:
    def __init__(self) -> None:
        self._embedded: dict[str, EmbeddedPrompt] = {}

    def canonicalize_prompt_json(self, prompt_graph: dict[str, Any]) -> str:
        """Deterministic sorted-key JSON without whitespace, for cryptographic hashing."""
        return canonical_json(prompt_graph)

    def _embedded_for(self, image: RawArtifactRecord) -> EmbeddedPrompt:
        if image.artifact_id not in self._embedded:
            self._embedded[image.artifact_id] = embedded_prompt(image)
        return self._embedded[image.artifact_id]

    def has_link_evidence(self, image: RawArtifactRecord) -> bool:
        emb = self._embedded_for(image)
        return bool(emb.prompt_id is not None or emb.canonical_sha256 is not None or emb.graph or emb.parameters)

    def evaluate_link(self, workflow: WorkflowRecord, image: RawArtifactRecord) -> CorrelationLink:
        """Classifies one workflow/image pair as STRICT, SUPPORTED or UNASSOCIATED. AMBIGUOUS is decided
        across all pairs of an image by resolve_links. The returned link has no link_id yet."""
        if workflow.prompt_format != "API_GRAPH":
            return self._link(workflow, image, LinkClass.UNASSOCIATED, [],
                              "Workflow canvas records provide context. Image attribution requires an API prompt.")
        emb = self._embedded_for(image)
        wf = workflow.execution_parameters
        if wf.get(OUTPUT_CONTEXT_FIELD) == TERMINAL_WITHOUT_RECORDED_OUTPUTS:
            # The job's own history ended interrupted or in error with explicitly empty outputs: no image is its
            # output, and its copy of a graph must not make another job's image ambiguous.
            return self._link(workflow, image, LinkClass.UNASSOCIATED, [TERMINAL_WITHOUT_RECORDED_OUTPUTS],
                              "The job ended without recorded outputs; its graph is context only.")
        known_ids = {wf.get("prompt_id"), *(wf.get("additional_prompt_ids") or [])} - {None}
        same_id = emb.prompt_id is not None and emb.prompt_id in known_ids
        if emb.prompt_id is not None and known_ids and not same_id:
            return self._link(workflow, image, LinkClass.UNASSOCIATED, [], "The known prompt IDs disagree.")
        strict_keys = []
        if emb.canonical_sha256 is not None and emb.canonical_sha256 == workflow.canonical_json_sha256:
            strict_keys.append("CANONICAL_PROMPT_HASH")
        elif workflow.prompt_format == "API_GRAPH" and workflow.graph and emb.graph:
            complete = (workflow.completeness == Completeness.COMPLETE
                        and graph_is_complete("API_GRAPH", emb.graph, emb.parameters))
            if (complete or same_id) and _prompt_graphs_disagree(emb.graph, workflow.graph, complete=complete):
                if same_id:
                    return self._link(workflow, image, LinkClass.UNASSOCIATED,
                                      [PROMPT_ID_GRAPH_CONFLICT], GRAPH_CONFLICT_NOTE)
            elif complete:
                strict_keys.append("EXECUTION_GRAPH_MATCH")
                if not same_id and not _covers_input_fingerprints(workflow.graph, emb.graph):
                    if known_ids:
                        return self._link(workflow, image, LinkClass.UNASSOCIATED, [INPUT_REVISION_UNVERIFIED],
                                          "The recovered prompt lacks an observed input revision for this image.")
                    strict_keys.append(INPUT_REVISION_UNVERIFIED)
        if same_id:
            strict_keys.append("PROMPT_ID")
        if strict_keys:
            return self._link(workflow, image, LinkClass.STRICT, strict_keys,
                              "Exact match on " + " and ".join(strict_keys) + ".")

        img = dict(emb.parameters)
        if image.dimensions is not None and emb.parameters:
            img["width"], img["height"] = image.dimensions
        agreed = []
        for key in (*SUPPORTING_KEYS, "width", "height"):
            if key not in img or key not in wf or img[key] != wf[key]:
                return self._link(workflow, image, LinkClass.UNASSOCIATED, [], "No complete parameter agreement.")
            agreed.append(key.upper())
        for key in CONSISTENCY_KEYS:
            if key in img and key in wf:
                if img[key] != wf[key]:
                    return self._link(workflow, image, LinkClass.UNASSOCIATED, [],
                                      f"Parameters agree but {key} contradicts.")
                agreed.append(key.upper())
        if workflow.graph and emb.graph and _input_fingerprints_conflict(workflow.graph, emb.graph):
            return self._link(workflow, image, LinkClass.UNASSOCIATED, [],
                              "Parameters agree but input revision contradicts.")
        if emb.client_id is not None and emb.client_id == wf.get("client_id"):
            agreed.append("CLIENT_ID")
        return self._link(workflow, image, LinkClass.SUPPORTED, agreed,
                          "Embedded generation parameters agree with the recovered workflow without contradiction.")

    @staticmethod
    def _link(workflow: WorkflowRecord, image: RawArtifactRecord, cls: str, keys: list[str],
              notes: str) -> CorrelationLink:
        return CorrelationLink(link_id="", workflow_id=workflow.workflow_id, artifact_id=image.artifact_id,
                               link_class=cls, matched_keys=keys, confidence_notes=notes)


WorkflowCopyKey = tuple[str, tuple[str, ...]]


def replay_graph_peers(workflows: Iterable[dict[str, Any]], excluded_jobs: set[str]) -> dict[str, set[str]]:
    """Other submissions that a replay of the same canonical API graph cannot distinguish."""
    by_graph: dict[str, set[str]] = {}
    for workflow in workflows:
        if workflow.get("prompt_format") != "API_GRAPH":
            continue
        digest = workflow.get("canonical_json_sha256")
        prompt_id = (workflow.get("execution_parameters") or {}).get("prompt_id")
        if isinstance(digest, str) and digest and isinstance(prompt_id, str) and prompt_id not in excluded_jobs:
            by_graph.setdefault(digest, set()).add(prompt_id)
    peers: dict[str, set[str]] = {}
    for jobs in by_graph.values():
        if len(jobs) > 1:
            for prompt_id in jobs:
                peers.setdefault(prompt_id, set()).update(jobs - {prompt_id})
    return peers


def unreplayed_complete_jobs(workflows: Iterable[dict[str, Any]], verified_jobs: set[str],
                             excluded_jobs: set[str]) -> set[str]:
    """Workflow IDs for known submissions that haven't been eliminated as close-match alternatives."""
    missing = set()
    for workflow in workflows:
        prompt_id = (workflow.get("execution_parameters") or {}).get("prompt_id")
        if (workflow.get("prompt_format") == "API_GRAPH" and workflow.get("completeness") == "COMPLETE"
                and isinstance(prompt_id, str) and prompt_id and prompt_id not in verified_jobs
                and prompt_id not in excluded_jobs):
            missing.add(workflow["workflow_id"])
    return missing


def workflow_copy_keys(workflows: Iterable[WorkflowRecord]) -> dict[str, WorkflowCopyKey]:
    """Group copies by graph and known submission, keeping repeated executions distinct."""
    keys: dict[str, WorkflowCopyKey] = {}
    submissions: dict[str, set[tuple[str, ...]]] = {}
    for w in workflows:
        digest = w.canonical_json_sha256 or w.workflow_id
        params = w.execution_parameters
        ids = tuple(sorted({pid for pid in (params.get("prompt_id"), *(params.get("additional_prompt_ids") or []))
                            if isinstance(pid, str) and pid}))
        keys[w.workflow_id] = (digest, ids)
        if ids:
            submissions.setdefault(digest, set()).add(ids)
    for wid, (digest, ids) in keys.items():
        known = submissions.get(digest, set())
        if not ids and len(known) == 1:
            keys[wid] = (digest, next(iter(known)))
    return keys


def _covers_input_fingerprints(exact: dict[str, Any], alternative: dict[str, Any]) -> bool:
    return all(
        "is_changed" not in node or (
            "is_changed" in exact.get(node_id, {})
            and canonical_json(node["is_changed"]) == canonical_json(exact[node_id]["is_changed"])
        )
        for node_id, node in alternative.items()
    )


def _input_fingerprints_conflict(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return any(
        isinstance(first[node_id], dict) and isinstance(second[node_id], dict)
        and "is_changed" in first[node_id] and "is_changed" in second[node_id]
        and canonical_json(first[node_id]["is_changed"]) != canonical_json(second[node_id]["is_changed"])
        for node_id in first.keys() & second.keys()
    )


def _fully_cached(workflow: WorkflowRecord) -> bool:
    cached = (workflow.execution_parameters.get("job_timeline") or {}).get("cached_nodes")
    return (workflow.prompt_format == "API_GRAPH" and bool(workflow.graph) and isinstance(cached, list)
            and {str(node) for node in workflow.graph} <= {str(node) for node in cached})


def _preferred_cache_copy(links: list[CorrelationLink], workflows: dict[str, WorkflowRecord]) -> CorrelationLink | None:
    if len(links) < 2 or any(link.matched_keys != [CACHE_KEY] for link in links):
        return None
    copies = [workflows[link.workflow_id] for link in links]
    ids = {copy.execution_parameters.get("prompt_id") for copy in copies}
    if len(ids) != 1 or None in ids or any(copy.completeness != Completeness.COMPLETE for copy in copies):
        return None
    preferred = max(copies, key=lambda copy: sum("is_changed" in node for node in copy.graph.values()))
    if any(_prompt_graphs_disagree(preferred.graph, copy.graph, complete=True)
           or not _covers_input_fingerprints(preferred.graph, copy.graph) for copy in copies):
        return None
    return next(link for link in links if link.workflow_id == preferred.workflow_id)


def resolve_links(images: list[RawArtifactRecord], candidates: Iterable[CorrelationLink],
                  workflows: list[WorkflowRecord]) -> list[CorrelationLink]:
    """Keeps the strongest link per submission, declaring ties between submissions ambiguous."""
    copy_keys = workflow_copy_keys(workflows)
    by_id = {workflow.workflow_id: workflow for workflow in workflows}
    # Any copy of a submission (its history entry, typically) can say it ran fully from cache.
    reused_ids = {pid for w in workflows if _fully_cached(w) for pid in copy_keys[w.workflow_id][1]}
    by_image: dict[str, list[CorrelationLink]] = {}
    for link in candidates:
        if link.link_class in LINKED_CLASSES:
            by_image.setdefault(link.artifact_id, []).append(link)
    final: list[CorrelationLink] = []
    for image in images:
        links = by_image.get(image.artifact_id, [])
        if not links:
            continue
        top = max(CLASS_RANK[link.link_class] for link in links)
        best = [link for link in links if CLASS_RANK[link.link_class] == top]
        best = [link for link in best if "PROMPT_ID" in link.matched_keys] or best
        # A submission whose own history lists every node of its graph as execution_cached executed nothing and
        # wrote no file: among tied submissions it cannot be the producer, as long as another one remains.
        executed = [link for link in best if not copy_keys[link.workflow_id][1]
                    or not set(copy_keys[link.workflow_id][1]) <= reused_ids]
        if executed and len(executed) < len(best):
            best = executed
        exact = [link for link in best if "CANONICAL_PROMPT_HASH" in link.matched_keys]
        identities = {copy_keys[link.workflow_id][1] for link in best}
        if exact and len(identities) == 1 and len(next(iter(identities))) <= 1:
            reference = by_id[exact[0].workflow_id].graph
            if all("CANONICAL_PROMPT_HASH" in link.matched_keys
                   or ("EXECUTION_GRAPH_MATCH" in link.matched_keys
                       and _covers_input_fingerprints(reference, by_id[link.workflow_id].graph)) for link in best):
                best = exact
        representatives: dict[WorkflowCopyKey, CorrelationLink] = {}
        for link in best:
            key = copy_keys[link.workflow_id]
            previous = representatives.get(key)
            if previous is None or len(link.matched_keys) > len(previous.matched_keys) or (
                len(link.matched_keys) == len(previous.matched_keys)
                and link.matched_keys == [CACHE_KEY]
                and by_id[link.workflow_id].execution_parameters.get("prompt_id")
                and not by_id[previous.workflow_id].execution_parameters.get("prompt_id")
            ):
                representatives[key] = link
        best = list(representatives.values())
        if len(identities) == 1 and len(next(iter(identities))) == 1:
            if preferred := _preferred_cache_copy(best, by_id):
                preferred.confidence_notes += " Compatible copies of this submission carry the same execution graph."
                best = [preferred]
        if len({link.workflow_id for link in best}) > 1:
            for link in best:
                link.confidence_notes = (
                    f"{link.link_class} keys {link.matched_keys} also match other workflows; "
                    "not attributed. " + link.confidence_notes
                )
                link.link_class = LinkClass.AMBIGUOUS
        final.extend(best)
    for n, link in enumerate(final, 1):
        link.link_id = f"LINK-{n:03d}"
    return final


MIN_OUTPUT_SIDE = 16
"""ComfyUI's latent nodes make no picture smaller than this on a side (EmptyLatentImage: minimum 16). An encoded
image below it that carries a prompt is a test or placeholder file, not a job's output."""

OTHER_IMAGES = "Other images in memory"
"""Rationale prefix of the one finding that groups images with no ComfyUI connection."""


def job_sizes(images: list[RawArtifactRecord], workflows: list[WorkflowRecord]) -> set[tuple[int, int]]:
    """Image sizes the recovered jobs produce: from workflow parameters and from prompts embedded in images."""
    sizes = set()
    for params in [w.execution_parameters for w in workflows] + [embedded_prompt(i).parameters for i in images]:
        if isinstance(params.get("width"), int) and isinstance(params.get("height"), int):
            sizes.add((params["width"], params["height"]))
    return sizes


def comfy_relevant(image: RawArtifactRecord, sizes: set[tuple[int, int]]) -> bool:
    """Whether an image may be ComfyUI evidence: it carries ComfyUI text, it was read from a ComfyUI process (a
    pixel buffer, a tensor, or bytes carved in its address space), or it has the size of a recovered job: a PNG
    exactly (ComfyUI writes outputs at full size; 256 x 256 and 128 x 128 PNGs are mostly Windows icons), a decoded
    bitmap by width at full, half or quarter scale (the frontend's previews, often recovered only in part). With
    no job size known,
    nothing tells an output from another picture, so every image may be one."""
    if image.embedded_text.get("prompt") or image.embedded_text.get("workflow"):
        return True
    if image.carve_type in (CarveType.PIXEL_BUFFER, CarveType.TENSOR_IMAGE) or image.process_id is not None:
        return True
    if image.dimensions is None or not sizes:
        return not sizes
    width, height = image.dimensions
    if image.carve_type == CarveType.DECODED_BITMAP:
        return width in {w // k for w, _ in sizes for k in (1, 2, 4)}
    return (width, height) in sizes


def _ui_family(graph: dict[str, Any]) -> str:
    """Group UI records by canvas ID and topology, including the contents of subgraph definitions."""
    def topology(canvas: dict[str, Any]) -> tuple[list[tuple[str, str]], list[str]]:
        nodes = canvas.get("nodes")
        links = canvas.get("links")
        return (sorted((str(n.get("id")), str(n.get("type")))
                       for n in nodes if isinstance(n, dict)) if isinstance(nodes, list) else [],
                sorted(canonical_json(link) for link in links) if isinstance(links, list) else [])

    definitions = graph.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    children = (sorted((str(g.get("id")), topology(g)) for g in subgraphs if isinstance(g, dict))
                if isinstance(subgraphs, list) else [])
    return hashlib.sha256(canonical_json([graph.get("id"), topology(graph), children]).encode()).hexdigest()


def _partial_group(workflow: WorkflowRecord) -> tuple[str, Any] | None:
    """Group unlinked UI records by topology and fragments by their agreeing seeds."""
    fragment = workflow.evidence.get("fragment")
    if fragment is not None or workflow.prompt_format == "API_NODE_FRAGMENT":
        return ("FRAGMENTS", tuple(fragment["agreeing_seeds"]) if fragment else ())
    if workflow.prompt_format == "UI_WORKFLOW":
        return "UI_FAMILY", _ui_family(workflow.graph)
    return None


def _group_rationale(key: tuple[str, Any], members: list[WorkflowRecord]) -> str:
    n = len(members)
    if key[0] == "UI_FAMILY":
        seeds = sorted({w.execution_parameters.get("seed") for w in members} - {None}, key=str)
        values = ", ".join(map(str, seeds)) or "unavailable"
        return f"{n} recovered UI records share the same workflow topology. Recovered seeds are {values}."
    seeds = key[1]
    if not seeds:
        return f"{n} page fragments or single nodes that agree with no recovered workflow; no image recovered."
    if len(seeds) == 1:
        return f"{n} page fragments agreeing only with seed {seeds[0]}; no corresponding image recovered."
    return (f"{n} page fragments agreeing with seeds {', '.join(map(str, seeds))} (template-identical); no "
            "corresponding image recovered.")


class TierAssigner:
    def __init__(self) -> None:
        self._count = 0

    def _next_id(self) -> str:
        self._count += 1
        return f"FIND-{self._count:03d}"

    def build_findings(
        self, images: list[RawArtifactRecord], workflows: list[WorkflowRecord], links: list[CorrelationLink]
    ) -> tuple[list[Finding], dict[str, int]]:
        """Builds one finding per job: the images of one output linked to a workflow (with every copy of that
        workflow, and the PNG copies of the same output), plus the job's prompt text read from those PNGs and the
        page fragments agreeing only with its seed. Batch images (different pixels) are separate findings.
        Assigns finding-level tiers and updates each image's link_status, associated_workflow_id and
        ambiguous_candidate_workflows. Raises: None."""
        workflows_by_id = {w.workflow_id: w for w in workflows}
        copy_keys = workflow_copy_keys(workflows)
        copies: dict[WorkflowCopyKey, list[str]] = {}
        unbound: dict[str, list[str]] = {}
        for w in workflows:
            copies.setdefault(copy_keys[w.workflow_id], []).append(w.workflow_id)
            if not (w.execution_parameters.get("prompt_id") or w.execution_parameters.get("additional_prompt_ids")):
                unbound.setdefault(copy_keys[w.workflow_id][0], []).append(w.workflow_id)
        links_by_image: dict[str, list[CorrelationLink]] = {}
        for link in links:
            links_by_image.setdefault(link.artifact_id, []).append(link)
        findings: list[Finding] = []
        summary = empty_tier_summary()
        linked_workflows: set[str] = set()
        sizes = job_sizes(images, workflows)
        other_images: list[str] = []
        jobs: dict[tuple[WorkflowCopyKey, str], dict[str, Any]] = {}
        header_only: list[tuple[WorkflowCopyKey, RawArtifactRecord, CorrelationLink]] = []
        linked: list[tuple[WorkflowCopyKey, RawArtifactRecord, CorrelationLink]] = []

        for image in images:
            image_links = links_by_image.get(image.artifact_id, [])
            chosen = [link for link in image_links if link.link_class in LINKED_CLASSES]
            ambiguous = [link for link in image_links if link.link_class == LinkClass.AMBIGUOUS]
            if chosen and _below_output_size(image):
                # The link stands, but the image earns no tier, and its prompt is tiered on its own below.
                [link] = chosen
                image.link_status = link.link_class
                image.associated_workflow_id = link.workflow_id
                width, height = image.dimensions or (0, 0)
                findings.append(Finding(
                    self._next_id(), Tier.UNTIERED, [image.artifact_id], [link.link_id],
                    f"Image of {width} x {height} pixels {link.link_class}-linked to {link.workflow_id} via "
                    f"{', '.join(link.matched_keys)}: below {MIN_OUTPUT_SIDE} pixels a side, smaller than any "
                    "picture ComfyUI's latent nodes produce, so it is not counted as a job's output.",
                ))
                summary["untiered_orphan_images"] += 1
            elif chosen:
                [link] = chosen
                workflow = workflows_by_id[link.workflow_id]
                graph = copy_keys[workflow.workflow_id]
                if INPUT_REVISION_UNVERIFIED in link.matched_keys:
                    graph = (graph[0], ())
                    copies[graph] = unbound[graph[0]]
                linked_workflows.update(copies[graph])
                image.link_status = link.link_class
                image.associated_workflow_id = workflow.workflow_id
                linked.append((graph, image, link))
            elif ambiguous:
                image.link_status = LinkClass.AMBIGUOUS
                image.associated_workflow_id = None
                image.ambiguous_candidate_workflows = sorted({link.workflow_id for link in ambiguous})
                findings.append(Finding(
                    self._next_id(), Tier.UNTIERED, [image.artifact_id], [link.link_id for link in ambiguous],
                    "Ambiguous image matching multiple candidate workflows "
                    f"({', '.join(image.ambiguous_candidate_workflows)}) without a distinguishing key.",
                ))
                summary["untiered_ambiguous_images"] += 1
            elif not image.evidence.get(INPUT_USE_KEY) and not comfy_relevant(image, sizes):
                image.link_status = LinkClass.UNASSOCIATED
                image.associated_workflow_id = None
                other_images.append(image.artifact_id)
            else:
                image.link_status = LinkClass.UNASSOCIATED
                image.associated_workflow_id = None
                note = " " + GRAPH_CONFLICT_NOTE if image.evidence.get("attribution_conflicts") else ""
                if image.evidence.get(INPUT_USE_KEY):
                    note += " A recovered workflow records these file bytes as an input."
                findings.append(Finding(
                    self._next_id(), Tier.UNTIERED, [image.artifact_id], [],
                    f"{image.completeness.title()} image with no corroborating workflow key (orphan).{note}",
                ))
                summary["untiered_orphan_images"] += 1
        # A decoded holder (tensor, pixel buffer, bitmap) with exactly the decoded pixels of one of the job's
        # complete encoded images is a copy of that output, not another output.
        identities = _output_identities([(graph, image) for graph, image, _ in linked])
        for graph, image, link in linked:
            identity = identities[(graph, image.artifact_id)]
            if identity is None:
                header_only.append((graph, image, link))
            else:
                self._join(jobs, (graph, identity), image, link)
        # A copy with no pixel data verified belongs to the job's only output; with several (a batch) it is its own.
        for graph, image, link in header_only:
            outputs = [key for key in jobs if key[0] == graph]
            output = outputs[0] if len(outputs) == 1 else (graph, f"no-pixels:{image.artifact_id}")
            self._join(jobs, output, image, link)

        by_png: dict[str, Finding] = {}
        by_seed: dict[Any, list[Finding]] = {}
        for (graph, _identity), job in jobs.items():
            workflow = workflows_by_id[job["links"][0].workflow_id]
            job_images: list[RawArtifactRecord] = job["images"]
            complete_image = any(i.completeness == Completeness.COMPLETE and not i.is_derived for i in job_images)
            # A content-stitched container is a DERIVED reconstruction. Even when all CRCs and pixels verify, it
            # cannot raise the extraction tier above the policy for a partial/derived image.
            tier = Tier.E2 if complete_image and workflow.completeness == Completeness.COMPLETE else Tier.E3
            # E3 needs recovered visual evidence. When every linked image is a partial with explicitly zero decoded
            # rows, the link stands on container provenance alone: the job is tiered by its prompt (E5 for a
            # complete API graph, which a replay can raise to E4; else E6), keeping the containers and links.
            pixelless = all(_zero_decoded_rows(image) for image in job_images)
            if pixelless:
                complete_prompt = (workflow.completeness == Completeness.COMPLETE
                                   and workflow.prompt_format == "API_GRAPH")
                tier = Tier.E5 if complete_prompt else Tier.E6
            others = [wid for wid in copies[graph] if wid != workflow.workflow_id]
            keys = sorted({k for link in job["links"] for k in link.matched_keys})
            classes = sorted({link.link_class for link in job["links"]})
            rationale = (f"{len(job_images)} {'copy' if len(job_images) == 1 else 'copies'} of one output "
                         f"({', '.join(sorted({i.completeness.lower() for i in job_images}))}) "
                         f"{'/'.join(classes)}-linked to {workflow.completeness.lower()} workflow "
                         f"{workflow.workflow_id} via {', '.join(keys)}.")
            if all(image.carve_type == CarveType.PARTIAL_PNG
                   and (image.derived_preview_metadata or {}).get("decoded_rows") == 0 for image in job_images):
                rationale += " Zero PNG pixel rows decoded."
            if pixelless:
                counts = "; ".join(f"{image.artifact_id}: 0 of "
                                   f"{image.dimensions[1] if image.dimensions else 'an unknown number of'} rows decoded"
                                   for image in job_images)
                rationale += (f" Container provenance only ({counts}); no visual pixels recovered, so the finding is "
                              f"tiered {tier} by its prompt, not E3.")
            if others:
                rationale += f" Byte-distinct copies of the same canonical graph: {', '.join(others)}."
            if any(image.evidence.get("attribution_conflicts") for image in job_images):
                rationale += " " + GRAPH_CONFLICT_NOTE
            ids = [i.artifact_id for i in job_images] + [workflow.workflow_id, *others]
            finding = Finding(self._next_id(), tier, ids, [link.link_id for link in job["links"]], rationale)
            findings.append(finding)
            summary[tier] += 1
            for image in job_images:
                by_png[image.artifact_id] = finding
            by_seed.setdefault(workflow.execution_parameters.get("seed"), []).append(finding)

        if other_images:
            findings.append(Finding(
                self._next_id(), Tier.UNTIERED, other_images, [],
                f"{OTHER_IMAGES}: {len(other_images)} images with no ComfyUI text, not read from a ComfyUI process and "
                "without a recovered job's size (browser, system and application images). Each keeps its hashes "
                "and exports; none is ComfyUI evidence.",
            ))
            summary["untiered_other_images"] = len(other_images)

        # Unlinked workflows in their order. The PNG text of a linked output and the fragments agreeing only with
        # a linked job's seed join that job; complete copies of one prompt and grouped partial material are one
        # finding each, in the place of the first member.
        groups: dict[Any, list[WorkflowRecord]] = {}
        order: list[Any] = []
        for workflow in workflows:
            if workflow.workflow_id in linked_workflows:
                continue
            pngs = [loc.get("png") for loc in workflow.execution_parameters.get("source_locations", [])]
            owner = next((by_png[p] for p in pngs if p in by_png), None)
            if owner is not None:
                owner.artifact_ids.append(workflow.workflow_id)
                continue
            key: Any
            if workflow.completeness == Completeness.COMPLETE and workflow.prompt_format == "API_GRAPH":
                key = ("PROMPT", copy_keys[workflow.workflow_id])
            else:
                key = _partial_group(workflow)
                single_job = key is not None and key[0] == "FRAGMENTS" and len(key[1]) == 1
                if single_job and len(by_seed.get(key[1][0], [])) == 1:
                    [owner] = by_seed[key[1][0]]
                    owner.artifact_ids.append(workflow.workflow_id)
                    continue
            if key is None:
                order.append(workflow)
            elif key in groups:
                groups[key].append(workflow)
            else:
                groups[key] = [workflow]
                order.append(key)
        for entry in order:
            members = groups[entry] if isinstance(entry, tuple) else [entry]
            if isinstance(entry, tuple) and entry[0] == "PROMPT":
                tier, rationale = Tier.E5, "Complete prompt graph; no corresponding image recovered."
                if len(members) > 1:
                    rationale += f" {len(members)} byte-distinct copies of the same canonical graph."
            elif isinstance(entry, tuple) and len(members) > 1:
                tier, rationale = Tier.E6, _group_rationale(entry, members)
            elif members[0].prompt_format == "UI_WORKFLOW":
                tier, rationale = Tier.E6, "Recovered UI workflow context; no corresponding image recovered."
            else:
                tier, rationale = Tier.E6, (f"Partial workflow material ({members[0].prompt_format}); no "
                                            "corresponding image recovered.")
            findings.append(Finding(self._next_id(), tier, [w.workflow_id for w in members], [], rationale))
            summary[tier] += 1
        return findings, summary

    @staticmethod
    def _join(jobs: dict[tuple[WorkflowCopyKey, str], dict[str, Any]], key: tuple[WorkflowCopyKey, str],
              image: RawArtifactRecord,
              link: CorrelationLink) -> None:
        job = jobs.setdefault(key, {"images": [], "links": []})
        job["images"].append(image)
        job["links"].append(link)


_HOLDERS = (CarveType.PIXEL_BUFFER, CarveType.TENSOR_IMAGE, CarveType.DECODED_BITMAP)


def _below_output_size(image: RawArtifactRecord) -> bool:
    """An encoded image (PNG or WEBP) smaller than MIN_OUTPUT_SIDE on either side."""
    encoded = image.carve_type in (CarveType.ENCODED_PNG, CarveType.PARTIAL_PNG, CarveType.ENCODED_WEBP,
                                   CarveType.PARTIAL_WEBP)
    return encoded and image.dimensions is not None and min(image.dimensions) < MIN_OUTPUT_SIDE


def _zero_decoded_rows(image: RawArtifactRecord) -> bool:
    """A partial image whose decoder recorded exactly zero pixel rows. Records without row metadata (legacy or
    synthetic ones) are unknown, not zero."""
    rows = (image.derived_preview_metadata or {}).get("decoded_rows")
    return (image.completeness != Completeness.COMPLETE and isinstance(rows, int) and not isinstance(rows, bool)
            and rows == 0)


def _verified_head(image: RawArtifactRecord) -> bytes:
    """The verified bytes the record holds from the start of its container."""
    return image.spans[0][1] if image.spans and image.spans[0][0] == 0 else b""


def _has_encoded_pixels(image: RawArtifactRecord) -> bool:
    """Whether the verified bytes include the start of the encoded pixel data: a PNG's first IDAT chunk (up to 1 KiB
    of it), or the first KiB of another container (WEBP and MP4 put their metadata last, and a FLAC header carries
    the audio's MD5). A header and text chunks alone are shared by every output of a job."""
    data = _verified_head(image)
    if image.carve_type in MEDIA_TYPES:
        return len(data) >= 1024
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        if data[pos + 4 : pos + 8] == b"IDAT":
            return len(data) - (pos + 8) >= min(length, 1024)
        pos += 12 + length
    return False


def _agrees_with(image: RawArtifactRecord, container: bytes) -> bool:
    """Whether every verified span of the record is the same bytes at the same offset of a complete container."""
    return bool(image.spans) and all(container[offset : offset + len(piece)] == piece for offset, piece in image.spans)


def _output_identities(linked: list[tuple[WorkflowCopyKey, RawArtifactRecord]]) -> dict[
        tuple[WorkflowCopyKey, str], str | None]:
    """What tells the outputs of one job apart, for each (job graph, image). Outputs of a batch share their header,
    text chunks and often a long encoded prefix, so only a complete container's own digest names an output.

    - A complete encoded container is its container digest; byte-identical copies are one output.
    - A partial whose verified bytes (its first pixel data included) agree with exactly one complete container of
      the job is a copy of it. With none, or with several, it is its own uncertain output: a shared prefix does not
      say which container it came from.
    - A decoded holder joins the one complete container of the job that decodes to exactly its pixels; with several
      such containers it stays separate, named by its pixels.
    - None when no pixel data was verified: such a copy joins the job's only output (see build_findings)."""
    containers: dict[WorkflowCopyKey, dict[str, bytes | None]] = {}
    by_pixels: dict[tuple[WorkflowCopyKey, str], set[str]] = {}
    for graph, image in linked:
        if image.carve_type in _HOLDERS or image.completeness != Completeness.COMPLETE or not image.container_sha256:
            continue
        head = _verified_head(image)
        known = containers.setdefault(graph, {})
        if known.get(image.container_sha256) is None:
            known[image.container_sha256] = head if hashlib.sha256(head).hexdigest() == image.container_sha256 else None
        if image.pixel_sha256:
            by_pixels.setdefault((graph, image.pixel_sha256), set()).add(image.container_sha256)
    identities: dict[tuple[WorkflowCopyKey, str], str | None] = {}
    for graph, image in linked:
        identity: str | None
        if image.carve_type in _HOLDERS:
            matching = by_pixels.get((graph, image.pixel_sha256 or ""), set())
            identity = (f"container:{next(iter(matching))}" if len(matching) == 1
                        else f"pixels:{image.pixel_sha256}" if image.pixel_sha256 else None)
        elif image.completeness == Completeness.COMPLETE and image.container_sha256:
            identity = f"container:{image.container_sha256}"
        elif not _has_encoded_pixels(image):
            identity = None
        else:
            matching = {sha for sha, data in containers.get(graph, {}).items()
                        if data is not None and _agrees_with(image, data)}
            identity = f"container:{next(iter(matching))}" if len(matching) == 1 else f"partial:{image.artifact_id}"
        identities[(graph, image.artifact_id)] = identity
    return identities


@dataclass
class CorrelationResult:
    links: list[CorrelationLink]
    findings: list[Finding]
    tier_summary: dict[str, int]


def correlate(images: list[RawArtifactRecord], workflows: list[WorkflowRecord]) -> CorrelationResult:
    """Link images by recovered workflow keys and distinguish recorded inputs from self-described outputs."""
    matcher = KeyMatcher()
    external = [w for w in workflows if w.prompt_format == "API_GRAPH" and not is_embedded_text(w.encoding_detected)]
    conflicts: dict[str, list[dict[str, str]]] = {}
    input_uses: dict[str, list[dict[str, str]]] = {}
    for workflow in external:
        for node_id, node in workflow.graph.items():
            if not isinstance(node, dict) or node.get("class_type") not in ("LoadImage", "LoadImageMask"):
                continue
            hashes = node.get("is_changed")
            if isinstance(hashes, list) and len(hashes) == 1 and isinstance(hashes[0], str):
                input_uses.setdefault(hashes[0].lower(), []).append({"workflow_id": workflow.workflow_id,
                                                                  "input_node_id": node_id})

    def candidates() -> Iterable[CorrelationLink]:
        for image in images:
            if not matcher.has_link_evidence(image):
                continue
            for workflow in external:
                candidate = matcher.evaluate_link(workflow, image)
                if PROMPT_ID_GRAPH_CONFLICT in candidate.matched_keys:
                    conflicts.setdefault(candidate.artifact_id, []).append({
                        "workflow_id": candidate.workflow_id, "reason": PROMPT_ID_GRAPH_CONFLICT,
                    })
                if candidate.link_class in LINKED_CLASSES:
                    yield candidate
        yield from _cache_links(images, external)

    links = resolve_links(images, candidates(), external)
    for image in images:
        uses = input_uses.get((image.container_sha256 or "").lower())
        if (uses or INPUT_USE_KEY in image.evidence or image.artifact_id in conflicts
                or "attribution_conflicts" in image.evidence):
            evidence = {**image.evidence}
            evidence.pop("attribution_conflicts", None)
            evidence.pop(INPUT_USE_KEY, None)
            if image.artifact_id in conflicts:
                evidence["attribution_conflicts"] = conflicts[image.artifact_id]
            if uses:
                evidence[INPUT_USE_KEY] = uses
            image.evidence = evidence
    decided = {link.artifact_id for link in links if link.link_class in (*LINKED_CLASSES, LinkClass.AMBIGUOUS)}
    own = {loc["png"]: w for w in workflows
           if is_embedded_text(w.encoding_detected) and w.prompt_format == "API_GRAPH"
           for loc in w.execution_parameters.get("source_locations", []) if loc.get("png")}
    for image in images:
        if image.artifact_id not in decided and image.artifact_id in own and not image.evidence.get(INPUT_USE_KEY):
            holder = own[image.artifact_id]
            note = ("The prompt reconstructed from this PNG's text" if holder.encoding_detected == "PNG_TEXT_STITCHED"
                    else "The prompt this PNG carries in its own text chunk")
            links.append(CorrelationLink(f"LINK-{len(links) + 1:03d}", holder.workflow_id,
                                         image.artifact_id, LinkClass.STRICT, ["EMBEDDED_PROMPT"],
                                         f"{note}; no separately recovered workflow links it."))
    links += _pixel_copy_links(images, workflows, links)
    findings, summary = TierAssigner().build_findings(images, workflows, links)
    return CorrelationResult(links, findings, summary)


CACHE_KEY = "EXECUTION_CACHE_SIGNATURE"
CACHE_REUSED_KEY = "EXECUTION_CACHE_REUSED"
PIXEL_COPY_KEY = "PIXEL_IDENTITY_WITH_LINKED_OUTPUT"


def _pixel_copy_links(images: list[RawArtifactRecord], workflows: list[WorkflowRecord],
                      links: list[CorrelationLink]) -> list[CorrelationLink]:
    """An otherwise unlinked decoded holder (pixel buffer, bitmap, tensor) whose exact decoded pixels are those of
    a complete image STRICT-linked to one submission is a copy of that output (SUPPORTED, joining its finding).
    Pixels shared by images of several submissions, or by an image recorded as an input, link nothing."""
    by_id = {image.artifact_id: image for image in images}
    copy_keys = workflow_copy_keys(workflows)
    decided = {link.artifact_id for link in links}
    anchors: dict[str, dict[WorkflowCopyKey, str]] = {}
    for link in links:
        image = by_id.get(link.artifact_id)
        if (link.link_class == LinkClass.STRICT and image is not None and image.pixel_sha256
                and image.completeness == Completeness.COMPLETE and not image.evidence.get(INPUT_USE_KEY)
                and link.workflow_id in copy_keys):
            anchors.setdefault(image.pixel_sha256, {}).setdefault(copy_keys[link.workflow_id], link.workflow_id)
    tainted = {image.pixel_sha256 for image in images if image.evidence.get(INPUT_USE_KEY) and image.pixel_sha256}
    out = []
    for image in images:
        submissions = anchors.get(image.pixel_sha256 or "", {})
        if (image.artifact_id in decided or image.carve_type not in _HOLDERS or len(submissions) != 1
                or image.completeness != Completeness.COMPLETE or image.pixel_sha256 in tainted):
            continue
        workflow_id = next(iter(submissions.values()))
        out.append(CorrelationLink(
            f"LINK-{len(links) + len(out) + 1:03d}", workflow_id, image.artifact_id, LinkClass.SUPPORTED,
            [PIXEL_COPY_KEY], "Exactly the decoded pixels of an image STRICT-linked to this submission; a copy of "
                              "that output, not a separate one."))
    return out


def _cache_links(images: list[RawArtifactRecord], workflows: list[WorkflowRecord]) -> list[CorrelationLink]:
    """An image tensor ComfyUI's execution cache holds as a node's output is STRICT-linked to every API prompt
    with a node of exactly that cache key: the same class, inputs and upstream nodes (execcache). The key is
    ComfyUI's own record of what produced the tensor, so the link needs no file, no text and no regeneration.
    Prompts that differ only elsewhere in the graph share the key; resolve_links then leaves the image
    AMBIGUOUS unless they are copies of one canonical graph.

    A prompt whose own history lists every matching node as ``execution_cached`` returned the tensor without
    computing it. When another linked prompt computed it, only the computing prompt is linked; when none did, the
    reusing prompts keep a SUPPORTED link (EXECUTION_CACHE_REUSED), never a STRICT one."""
    links = []
    for image in images:
        cache = image.evidence.get("execution_cache")
        if not cache or not cache.get("graph"):
            continue
        found: list[tuple[CorrelationLink, bool]] = []
        for workflow in workflows:
            if (workflow.prompt_format != "API_GRAPH"
                    or workflow.execution_parameters.get(OUTPUT_CONTEXT_FIELD) == TERMINAL_WITHOUT_RECORDED_OUTPUTS):
                continue
            locations = workflow.execution_parameters.get("source_locations", [])
            holders = {workflow.source_process_id, *(loc.get("process_id") for loc in locations)} - {None}
            if image.process_id is not None and holders and image.process_id not in holders:
                continue
            nodes = matching_nodes(cache["graph"], workflow.graph)
            if nodes:
                cached = set((workflow.execution_parameters.get("job_timeline") or {}).get("cached_nodes") or [])
                found.append((CorrelationLink(
                    "", workflow.workflow_id, image.artifact_id, LinkClass.STRICT, [CACHE_KEY],
                    f"ComfyUI's execution cache holds this tensor as output {cache.get('output_socket')} of a "
                    f"{cache.get('class_type')} node whose cache key equals node {', '.join(nodes)} of this prompt "
                    "with everything upstream of it."), set(nodes) <= cached))
        computed = [link for link, reused in found if not reused]
        if computed:
            links += computed
            continue
        for link, _ in found:
            link.link_class = LinkClass.SUPPORTED
            link.matched_keys.append(CACHE_REUSED_KEY)
            link.confidence_notes += " The prompt's history lists that node as execution_cached: it reused an " \
                                     "earlier job's tensor rather than computing it."
            links.append(link)
    return links


def png_workflows(images: list[RawArtifactRecord], ids: IdAllocator) -> list[WorkflowRecord]:
    """The prompts and UI workflows ComfyUI writes into its PNGs, as workflow records (encoding PNG_TEXT): one per
    distinct graph, listing every PNG it was read from. Its pages are those PNGs' pages, so its memory source
    follows theirs. The canonical prompt hash then links each PNG to its own prompt (STRICT)."""
    records: dict[tuple[str, str, str | None, bool], WorkflowRecord] = {}
    for image in images:
        for name in ("prompt", "workflow"):
            text, encoding = image.embedded_text.get(name), embedded_encoding(image.carve_type)
            if not text and name in image.stitched_text:
                text, encoding = image.stitched_text[name], "PNG_TEXT_STITCHED"
            if not text:
                continue
            try:
                classified = extract_prompt_graph(json.loads(text))
            except (ValueError, RecursionError):
                classified = None
            if classified is None or classified[0] not in ("API_GRAPH", "UI_WORKFLOW"):
                continue
            fmt, graph, runtime = classified
            if prompt_id := image.embedded_text.get("prompt_id"):
                runtime.setdefault("prompt_id", prompt_id)
            try:
                digest = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
                raw = text.encode("utf-8")
            except (UnicodeError, ValueError, RecursionError):
                continue
            location = {"process_id": image.process_id, "virtual_address": image.virtual_address,
                        "physical_offset": image.physical_offsets[0] if image.physical_offsets else None,
                        "png": image.artifact_id}
            key = (fmt, digest, runtime.get("prompt_id"), encoding == "PNG_TEXT_STITCHED")
            record = records.get(key)
            if record is not None:
                record.execution_parameters["source_locations"].append(location)
                record.page_origins += image.page_origins
                continue
            if fmt == "API_GRAPH":
                params = execution_parameters(graph, runtime)
                complete = graph_is_complete(fmt, graph, params)
                dependencies = graph_dependencies(graph)
            else:
                params = {**{k: runtime[k] for k in ("prompt_id", "client_id") if k in runtime},
                          **ui_parameters(ui_workflow_nodes(graph))}
                complete = False
                dependencies = {"node_classes": [], "model_assets": []}
            params["source_locations"] = [location]
            records[key] = WorkflowRecord(
                workflow_id=ids(), completeness=Completeness.COMPLETE if complete else Completeness.PARTIAL,
                prompt_format=fmt, encoding_detected=encoding, source_process_id=image.process_id,
                virtual_address=image.virtual_address, physical_offsets=list(image.physical_offsets[:1]),
                canonical_json_sha256=digest, execution_parameters=params, dependencies=dependencies,
                raw_file_path="", raw_bytes=raw, graph=graph, page_origins=list(image.page_origins))
    return list(records.values())


def references(workflows: list[WorkflowRecord], images: list[RawArtifactRecord]) -> list[Reference]:
    out = []
    for wf in workflows:
        if wf.prompt_format == "API_GRAPH":
            out.append(Reference(wf.workflow_id, wf.graph, [], wf.execution_parameters.get("seed")))
        elif wf.prompt_format == "UI_WORKFLOW":
            # Agreement is with the nodes as stored; the seed is the one the workflow would queue.
            out.append(Reference(wf.workflow_id, {}, ui_nodes(wf.graph),
                                 ui_parameters(ui_workflow_nodes(wf.graph)).get("seed")))
    for image in images:
        for name in ("prompt", "workflow"):
            try:
                classified = extract_prompt_graph(json.loads(image.embedded_text.get(name) or "null"))
            except (ValueError, RecursionError):
                continue
            if classified is not None and classified[0] == "API_GRAPH":
                graph = classified[1]
                out.append(Reference(f"{image.artifact_id}/{name}", graph, [],
                                     execution_parameters(graph, {}).get("seed")))
            elif classified is not None and classified[0] == "UI_WORKFLOW":
                out.append(Reference(f"{image.artifact_id}/{name}", {}, ui_nodes(classified[1]),
                                     ui_parameters(ui_workflow_nodes(classified[1])).get("seed")))
    return out


def _location(offset: int, record_range: tuple[int, int] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"process_id": None, "virtual_address": None, "physical_offset": f"0x{offset:016X}"}
    if record_range is not None:
        out["record_range"] = list(record_range)
    return out


def _fragment_record(
    piece: TextFragment, refs: list[Reference], seeds: set[Any], ids: IdAllocator
) -> WorkflowRecord:
    value = piece.fragment.value
    api, ui = api_nodes(value), ui_nodes(value)
    graph = {key or f"?{n}": node for n, (key, node) in enumerate(api)}
    params = {**ui_parameters(ui), **execution_parameters(graph, {})}
    params["source_locations"] = [_location(offset) for offset in piece.physical_offsets]
    agreeing = [r for r in refs if agrees_with(value, r)]
    own = params.get("seed")
    same_seed = [r for r in refs if own is not None and r.seed == own]
    agreeing_seeds = {r.seed for r in agreeing + same_seed if r.seed is not None}
    if not agreeing and not same_seed:
        attribution = "NO_REFERENCE_AGREES"
    elif len(agreeing_seeds) == 1 and len(seeds) > 1:
        attribution = "ONE_SEED"
    else:
        attribution = "AMBIGUOUS"
    classes = {str(n.get("class_type")) for _, n in api if "class_type" in n} | {
        t for n in ui if (t := ui_node_type(n)) is not None}
    return WorkflowRecord(
        workflow_id=ids(),
        completeness=Completeness.PARTIAL,
        prompt_format="API_FRAGMENT" if api else "UI_FRAGMENT",
        encoding_detected=piece.encoding,
        source_process_id=None,
        virtual_address=None,
        physical_offsets=[loc["physical_offset"] for loc in params["source_locations"]],
        canonical_json_sha256=hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest(),
        execution_parameters=params,
        dependencies={"node_classes": sorted(classes),
                      "model_assets": graph_dependencies(
                          {k: n for k, n in graph.items() if isinstance(n.get("inputs"), dict)
                           and isinstance(n.get("class_type"), str)})["model_assets"]},
        raw_file_path="",
        raw_bytes=piece.raw,
        graph=value if isinstance(value, dict) else {CUT: value},
        evidence={"fragment": {"cut_before": piece.fragment.cut_before, "cut_after": piece.fragment.cut_after,
                               "agrees_with": [r.record_id for r in agreeing],
                               "same_seed": [r.record_id for r in same_seed],
                               "agreeing_seeds": sorted(agreeing_seeds, key=str), "attribution": attribution}},
    )


def place_fragments(
    pieces: list[TextFragment], workflows: list[WorkflowRecord], images: list[RawArtifactRecord], ids: IdAllocator
) -> list[WorkflowRecord]:
    """Places each carved piece. Bytes found inside the copies of exactly one recovered graph become a partial
    copy of it: a source location carrying ``record_range``. Any other piece becomes a fragment record. Either
    way the piece replaces the single-node records carved from inside it. Returns the updated workflow list."""
    refs = references(workflows, images)
    seeds = {r.seed for r in refs if r.seed is not None}
    whole = [wf for wf in workflows if wf.prompt_format in ("API_GRAPH", "UI_WORKFLOW") and wf.raw_bytes]
    added: list[WorkflowRecord] = []
    for piece in pieces:
        holders = [wf for wf in whole if piece.raw in wf.raw_bytes]
        if holders and len({wf.canonical_json_sha256 for wf in holders}) == 1:
            for wf in holders:
                at = wf.raw_bytes.find(piece.raw)
                wf.execution_parameters["source_locations"] += [
                    _location(offset, (at, at + len(piece.raw))) for offset in piece.physical_offsets]
            continue
        added.append(_fragment_record(piece, refs, seeds, ids))
    spans = [(offset, offset + len(piece.raw)) for piece in pieces for offset in piece.physical_offsets]

    def superseded(wf: WorkflowRecord) -> bool:
        if wf.prompt_format != "API_NODE_FRAGMENT" or wf.source_process_id is not None or not wf.physical_offsets:
            return False
        return all(any(lo <= int(o, 16) and int(o, 16) + len(wf.raw_bytes) <= hi for lo, hi in spans)
                   for o in wf.physical_offsets)

    return [wf for wf in workflows if not superseded(wf)] + added
