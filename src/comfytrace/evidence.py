"""Recovery evidence sidecar and per-output scorecard (design revision 3, section 10).

``manifest.json`` keeps schema 1.0.0. Everything the pure-memory stages know beyond it lives in
``recovery-evidence.json`` (one entry per manifest record) and ``scorecard.json`` (one row per job output named by
a backend history item). Nothing here decides a link; it reports what each holder and lineage gave.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable
from typing import Any, TypeVar

from comfytrace.carve import canonical_json
from comfytrace.correlate import INPUT_REVISION_UNVERIFIED, WorkflowCopyKey, workflow_copy_keys
from comfytrace.media import is_embedded_text
from comfytrace.memrecover import TERMINAL_WITHOUT_RECORDED_OUTPUTS, terminal_without_outputs_jobs
from comfytrace.models import (
    LINKED_CLASSES,
    Completeness,
    CorrelationLink,
    LinkClass,
    PageOrigin,
    RawArtifactRecord,
    WorkflowRecord,
)
from comfytrace.pyheap import QueueItem

Rec = TypeVar("Rec", RawArtifactRecord, WorkflowRecord)
EVIDENCE_SCHEMA = 1
SCORECARD_SCHEMA = 1
LINKED = {LinkClass.STRICT: "STRICT", LinkClass.SUPPORTED: "SUPPORTED"}


def _carved_evidence(kind: str, record: Any) -> dict[str, Any]:
    """Records from the PNG and JSON carvers: observed bytes whose lineage the carver cannot tell."""
    out: dict[str, Any] = {"origin": "OBSERVED", "lineage": "UNCLASSIFIED", "holes": [],
                           "liveness": "UNKNOWN", "observations": []}
    if kind == "workflow":
        out["submission_class"] = "FOREIGN_OR_UNKNOWN"
        out["observations"] = [{"holder_id": "carved_json", **loc}
                               for loc in record.execution_parameters.get("source_locations", [])]
    else:
        out["attribution"] = LINKED.get(record.link_status, "UNATTRIBUTED")
        out["fidelity"] = "EXACT" if record.completeness == "COMPLETE" else "UNKNOWN"
        out["holes"] = [{"byte_range": [g.relative_offset, g.relative_offset + g.length_bytes], "reason": g.reason}
                        for g in record.gaps]
        fmt = record.carve_type.split("_", 1)[1].lower() if record.carve_type.endswith(MEDIA_SUFFIXES) else "png"
        out["observations"] = [{"holder_id": f"carved_{fmt}", "process_id": record.process_id,
                                "virtual_address": record.virtual_address,
                                "physical_offsets": record.physical_offsets}]
    return out


MEDIA_SUFFIXES = ("_WEBP", "_MP4", "_WEBM", "_FLAC")
MEMORY_PAGES = frozenset({PageOrigin.PRIVATE, PageOrigin.SHARED_MEMORY, PageOrigin.FREED})
FILE_PAGES = frozenset({PageOrigin.FILE_DATA, PageOrigin.FILE_IMAGE, PageOrigin.FILE_UNCONFIRMED})


def memory_source(copies: list[dict[str, Any]], nbytes: int) -> str:
    """Whether a record stands on memory alone, given the page origins of each copy (``PfnSupport.copy_origin``).

    MEMORY: some copy lies wholly on memory pages and holds all ``nbytes``. MEMORY_PARTIAL: only shorter such
    copies exist. FILE_BACKED: every copy lies wholly on file-backed pages, so it rests on a file object.
    MIXED: no copy is wholly in memory and some page is file-backed. UNKNOWN: pages of unknown origin only.
    NOT_CLASSIFIED: no page was classified (no PFN database, or a record decoded from Python objects)."""
    classified = [c for c in copies if c["pages"]]
    if not classified:
        return "NOT_CLASSIFIED"
    in_memory = [c["bytes"] for c in classified if set(c["pages"]) <= MEMORY_PAGES]
    if in_memory:
        return "MEMORY" if max(in_memory) >= nbytes else "MEMORY_PARTIAL"
    if all(set(c["pages"]) <= FILE_PAGES for c in classified):
        return "FILE_BACKED"
    if any(set(c["pages"]) & FILE_PAGES for c in classified):
        return "MIXED"
    return "UNKNOWN"


def record_bytes(record: RawArtifactRecord | WorkflowRecord) -> int:
    if isinstance(record, WorkflowRecord):
        return len(record.raw_bytes)
    return record.recovered_bytes or 0


def _memory_copy(record: Rec) -> Rec | None:
    if (isinstance(record, RawArtifactRecord) and record.derived_reconstruction is not None
            or isinstance(record, WorkflowRecord) and record.encoding_detected == "PNG_TEXT_STITCHED"):
        return None
    source = memory_source(record.page_origins, record_bytes(record))
    if source == "MEMORY" or (isinstance(record, WorkflowRecord) and record.encoding_detected == "CPYTHON_OBJECTS"):
        return dataclasses.replace(record)
    if source == "MEMORY_PARTIAL":
        return dataclasses.replace(record, completeness=Completeness.PARTIAL)
    return None


def memory_view(images: list[RawArtifactRecord], workflows: list[WorkflowRecord]) -> tuple[
        list[RawArtifactRecord], list[WorkflowRecord]]:
    """The records that stand on memory, as the correlator should see them when file objects are assumed gone:
    MEMORY records, MEMORY_PARTIAL ones marked PARTIAL (only part of them is in memory), and workflows decoded
    from a live process's Python objects. Copies, so correlating them leaves the full view's link fields alone."""
    return ([c for r in images if (c := _memory_copy(r)) is not None],
            [w for r in workflows if (w := _memory_copy(r)) is not None])


def _matching_values(observed: Any, complete: Any, path: str, holes: set[str]) -> int | None:
    if path in holes:
        return 0
    if isinstance(observed, dict) and isinstance(complete, dict):
        pairs = ((value, complete[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
                 for key, value in observed.items() if key in complete)
        if not observed.keys() <= complete.keys():
            return None
    elif isinstance(observed, list) and isinstance(complete, list) and len(observed) == len(complete):
        pairs = ((first, second, f"{path}/{index}")
                 for index, (first, second) in enumerate(zip(observed, complete, strict=True)))
    else:
        return 1 if canonical_json(observed) == canonical_json(complete) else None
    count = 0
    for first, second, child_path in pairs:
        matched = _matching_values(first, second, child_path, holes)
        if matched is None:
            return None
        count += matched
    return count


def _queue_graph_corroboration(workflows: list[WorkflowRecord]) -> dict[str, list[dict[str, Any]]]:
    """A complete serialized queue item can agree with a partial CPython item without acquiring its provenance."""
    by_prompt: dict[str, list[WorkflowRecord]] = {}
    for workflow in workflows:
        prompt_id = workflow.execution_parameters.get("prompt_id")
        if (isinstance(prompt_id, str) and workflow.prompt_format == "API_GRAPH"
                and not is_embedded_text(workflow.encoding_detected)):
            by_prompt.setdefault(prompt_id, []).append(workflow)
    result = {}
    for records in by_prompt.values():
        complete = [workflow for workflow in records if workflow.completeness == Completeness.COMPLETE]
        if len({workflow.canonical_json_sha256 for workflow in complete}) != 1:
            continue
        for queue in complete:
            if queue.execution_parameters.get("runtime_id_source") != "SERIALIZED_QUEUE_ITEM":
                continue
            supporting = []
            conflicting_processes = set()
            for workflow in records:
                if (workflow.completeness != Completeness.PARTIAL or workflow.encoding_detected != "CPYTHON_OBJECTS"
                        or workflow.source_process_id is None or not workflow.evidence.get("holes")):
                    continue
                locations = workflow.execution_parameters.get("source_locations", [])
                pids = {workflow.source_process_id, *(copy.get("process_id") for copy in locations)} - {None}
                holes = {item["json_pointer"] for item in workflow.evidence["holes"]}
                known = {key: ({name: value for name, value in node.items() if name in {"class_type", "inputs"}}
                               if isinstance(node, dict) else node) for key, node in workflow.graph.items()}
                matched = _matching_values(known, queue.graph, "", holes)
                if matched is None:
                    conflicting_processes.update(pids)
                elif matched > len(known):
                    supporting.extend({"workflow_id": workflow.workflow_id, "process_id": pid,
                                       "matched_graph_values": matched} for pid in sorted(pids))
            supporting = [copy for copy in supporting if copy["process_id"] not in conflicting_processes]
            if supporting:
                result[queue.workflow_id] = supporting
    return result


def evidence_document(images: list[RawArtifactRecord], workflows: list[WorkflowRecord], tool: dict[str, Any],
                      outputs: Iterable[tuple[str, Any]] = (), memory_only: dict[str, int] | None = None,
                      memory_only_findings: Iterable[Any] | None = None,
                      execution_cache: list[dict[str, Any]] | None = None, *,
                      memory_only_links: Iterable[CorrelationLink] | None = None) -> dict[str, Any]:
    """``memory_only_findings``: the findings of the memory-only correlation, kept so that ``report`` can apply
    image links to them. ``execution_cache``: ComfyUI's cache entries read from process memory (execcache)."""
    records = []
    queue_corroboration = _queue_graph_corroboration(workflows)
    for kind, items in (("workflow", workflows), ("image", images)):
        for record in items:
            evidence = dict(record.evidence) if "origin" in record.evidence else {
                **_carved_evidence(kind, record), **record.evidence}
            record_id = record.workflow_id if isinstance(record, WorkflowRecord) else record.artifact_id
            entry = {"record_id": record_id, "kind": kind, "completeness": record.completeness, **evidence}
            if record_id in queue_corroboration:
                entry["queue_graph_corroboration"] = queue_corroboration[record_id]
            hashes = dict(entry.get("sha256", {}))
            if "raw" not in hashes:
                digest = hashlib.sha256()
                if isinstance(record, WorkflowRecord):
                    digest.update(record.raw_bytes)
                else:
                    for _offset, data in record.spans:
                        digest.update(data)
                hashes["raw"] = digest.hexdigest()
            entry["sha256"] = hashes
            if isinstance(record, RawArtifactRecord) and record.derived_reconstruction is not None:
                entry["sha256"]["derived"] = hashlib.sha256(record.derived_reconstruction).hexdigest()
            entry["observations"] = [
                {**obs, "observation_id": obs.get("observation_id", f"{record_id}/obs-{n}")}
                for n, obs in enumerate(entry.get("observations", []), 1)
            ]
            entry["page_origins"] = record.page_origins
            entry["memory_source"] = memory_source(record.page_origins, record_bytes(record))
            records.append(entry)
    for record_id, output in outputs:
        # Each copy observes the same metadata independently, so one copy in memory is enough. The file name is
        # in every copy; a version's size and write time only in the response headers carrying its ETag.
        for key, version in output.versions.items():
            seen = [c["origin"] for c in output.copies if c.get("version") == key and "origin" in c]
            version["memory_source"] = memory_source(seen, 0)
        records.append({**output.to_dict(record_id), "kind": "output_file", "page_origins": output.page_origins,
                        "memory_source": memory_source(output.page_origins, 0)})
    summary: dict[str, dict[str, int]] = {}
    for entry in records:
        counts = summary.setdefault(entry["kind"], {})
        counts[entry["memory_source"]] = counts.get(entry["memory_source"], 0) + 1
    document: dict[str, Any] = {"schema_version": EVIDENCE_SCHEMA, "tool": tool, "memory_sources": summary,
                                "memory_only_tier_summary": memory_only}
    if memory_only_findings is not None:
        document["memory_only_findings"] = [f.to_dict() if hasattr(f, "to_dict") else f
                                            for f in memory_only_findings]
    if memory_only_links is not None:
        document["memory_only_correlation_links"] = [link.to_dict() for link in memory_only_links]
    if execution_cache is not None:
        document["execution_cache"] = execution_cache
    return {**document, "records": records}


def _graph_status(record: WorkflowRecord | None, decoded: bool, present: bool) -> dict[str, Any]:
    if record is not None:
        return {"status": "RECOVERED", "record_id": record.workflow_id, "completeness": record.completeness,
                "holes": len(record.evidence.get("holes", []))}
    if not decoded:
        return {"status": "UNREADABLE"}
    return {"status": "NOT_PRESENT" if not present else "NOT_RECOVERED"}


def scorecard(items: Iterable[QueueItem], workflows: list[WorkflowRecord], memory_sha256: str,
              links: Iterable[CorrelationLink] = (), images: Iterable[RawArtifactRecord] = ()) -> dict[str, Any]:
    """One row per (prompt_id, output node, batch index) of every backend queue/history item. A row's image
    candidates are the images a STRICT or SUPPORTED correlation link ties to a workflow holding exactly the job's
    API prompt (same canonical hash); ``link`` later adds the memory images its regeneration comparison ties to
    the job (``apply_image_links``). Each candidate says what the image is (carve type, completeness) and its
    memory source, since a PNG header on file-cache pages also carries the prompt. Candidates are per job, except
    a tensor that ComfyUI's execution cache holds as a node's output: it is listed on the rows of the output
    nodes that node feeds, at its own batch index (``_row_candidates``)."""
    by_prompt: dict[tuple[str, str], list[WorkflowRecord]] = {}
    for wf in workflows:
        prompt_id = wf.evidence.get("prompt_id")
        if prompt_id:
            by_prompt.setdefault((prompt_id, wf.prompt_format), []).append(wf)

    def best(prompt_id: str, fmt: str) -> WorkflowRecord | None:
        candidates = by_prompt.get((prompt_id, fmt), [])
        return min(candidates, key=lambda w: (w.completeness != "COMPLETE", len(w.evidence.get("holes", []))),
                   default=None)

    described = {img.artifact_id: {"carve_type": img.carve_type, "completeness": img.completeness,
                                   "memory_source": memory_source(img.page_origins, record_bytes(img))}
                 for img in images}
    ties = {img.artifact_id: img.evidence["execution_cache"] for img in images
            if img.evidence.get("execution_cache")}
    copy_keys = workflow_copy_keys(wf for wf in workflows if wf.prompt_format == "API_GRAPH")
    by_key: dict[WorkflowCopyKey, list[dict[str, Any]]] = {}
    for link in links:
        key = copy_keys.get(link.workflow_id)
        if key is not None and link.link_class in LINKED_CLASSES and INPUT_REVISION_UNVERIFIED not in link.matched_keys:
            by_key.setdefault(key, []).append({"artifact_id": link.artifact_id,
                                                   "basis": f"CORRELATION_{link.link_class}",
                                                   "matched_keys": list(link.matched_keys),
                                                   **described.get(link.artifact_id, {})})

    def correlated(prompt_id: str) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for wf in by_prompt.get((prompt_id, "API_GRAPH"), []):
            for candidate in by_key.get(copy_keys[wf.workflow_id], []):
                found.setdefault(candidate["artifact_id"], candidate)
        return sorted(found.values(), key=lambda c: c["artifact_id"])

    jobs: dict[str, list[QueueItem]] = {}
    for item in items:
        jobs.setdefault(item.prompt_id, []).append(item)
    unproduced = terminal_without_outputs_jobs(item for copies in jobs.values() for item in copies)
    rows = []
    for prompt_id, copies in sorted(jobs.items()):
        live = any(c.liveness == "LIVE" for c in copies)
        decoded_extra = any(c.extra_data is not None for c in copies)
        has_workflow = any(c.workflow is not None for c in copies) or any(
            h["json_pointer"].startswith("/extra_pnginfo") for c in copies for h in c.extra_holes)
        api = _graph_status(best(prompt_id, "API_GRAPH"), any(c.prompt is not None for c in copies), True)
        ui = _graph_status(best(prompt_id, "UI_WORKFLOW"), decoded_extra, has_workflow)
        outputs: list[tuple[str | None, int | None, str | None]] = []
        for c in copies:
            hist_outputs = (c.history or {}).get("outputs")
            if isinstance(hist_outputs, dict):
                for node_id, node in hist_outputs.items():
                    images = node.get("images") if isinstance(node, dict) else None
                    for index, img in enumerate(images if isinstance(images, list) else []):
                        name = img.get("filename") if isinstance(img, dict) else None
                        outputs.append((str(node_id), index, name))
        output_nodes = {node_id for node_id, _, _ in outputs}
        for node_id, batch_index, filename in sorted(set(outputs), key=str) or [(None, None, None)]:
            candidates = _row_candidates(correlated(prompt_id), ties, prompt_id, node_id, batch_index, output_nodes)
            context: dict[str, Any] = {}
            if prompt_id in unproduced:
                # The job's own history says it ended without any output: what matched its graph is context.
                context = {"output_context": TERMINAL_WITHOUT_RECORDED_OUTPUTS}
                image = _image_column([], contextual=_contextual(candidates))
            else:
                image = _image_column(candidates)
            rows.append({
                "dump_sha256": memory_sha256,
                "prompt_id": prompt_id,
                "output_node_id": node_id,
                "batch_index": batch_index,
                "output_filename": filename,
                "liveness": "LIVE" if live else "UNKNOWN",
                "holders": sorted({"python.queue_item" + (" (history)" if c.owner else "") for c in copies}),
                "lineages": ["L1"],
                "observations": [f"0x{c.address:X}" for c in copies],
                "api_prompt": api,
                "ui_workflow": ui,
                **context,
                "image": image,
            })
    return {"schema_version": SCORECARD_SCHEMA, "dump_sha256": memory_sha256, "rows": rows,
            "summary": _summary(rows)}


def _row_candidates(candidates: list[dict[str, Any]], ties: dict[str, dict[str, Any]], prompt_id: str,
                    node_id: str | None, batch_index: int | None, output_nodes: set[str | None],
                    ) -> list[dict[str, Any]]:
    """The job's candidates that belong on one output row. A tensor the execution cache ties to a node of this
    job is the output of the output nodes that node feeds directly, at the tensor's own batch index: it is
    listed only there (role OUTPUT). When it feeds no output node of the job it is a picture the job computed
    on the way (role INTERMEDIATE) and is listed on every row. Every other candidate is per job."""
    out = []
    for candidate in candidates:
        tie = ties.get(candidate["artifact_id"])
        job = next((j for j in (tie or {}).get("jobs", []) if j["prompt_id"] == prompt_id), None)
        if tie is None or job is None or node_id is None:
            out.append({**candidate, "role": "JOB"})
            continue
        fed = set(job["feeds"]) & output_nodes
        if job.get("reused"):
            # The job's history lists the node as execution_cached: it returned a tensor an earlier job
            # computed, which earns the job, never an exact output node or batch position.
            out.append({**candidate, "cache_node_ids": job["node_ids"], "role": "JOB", "cache_reused": True})
            continue
        described = {**candidate, "cache_node_ids": job["node_ids"], "batch_index": tie.get("batch_index")}
        if not fed:
            out.append({**described, "role": "INTERMEDIATE"})
        elif node_id in fed and batch_index in (None, tie.get("batch_index")):
            if len(fed) == 1:
                out.append({**described, "role": "OUTPUT"})
            else:
                out.append({**candidate, "cache_node_ids": job["node_ids"], "role": "AMBIGUOUS_OUTPUT",
                            "candidate_output_nodes": sorted(node for node in fed if node is not None)})
    return out


CONTEXT_ROLE = "CONTEXT_NOT_OUTPUT"


def _contextual(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Images that matched a job ending without recorded outputs, kept as observations that earn it no credit."""
    return [{**candidate, "role": CONTEXT_ROLE} for candidate in candidates]


def _image_column(candidates: list[dict[str, Any]], regeneration: dict[str, Any] | None = None, *,
                  contextual: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output_linked = any(candidate.get("role") == "OUTPUT" for candidate in candidates)
    status = "LINKED" if output_linked else "JOB_LINKED" if candidates else "NOT_LINKED"
    column: dict[str, Any] = {"status": status, "candidates": candidates}
    if contextual:
        column["contextual_observations"] = contextual
    if regeneration is not None:
        column["regeneration"] = regeneration
    return column


def apply_image_links(card: dict[str, Any], links: dict[str, Any],
                      sources: dict[str, str] | None = None) -> dict[str, Any]:
    """Refreshes the scorecard with the job's image status and output-specific matches from image-links.json.

    Older link files carry job-level candidates. Applying the same links again gives the same scorecard.
    """
    jobs = {job["prompt_id"]: job for job in links.get("jobs", [])}
    memory = {c["artifact_id"]: c for c in links.get("memory_images", [])}
    assigned: dict[tuple[str, str], set[tuple[Any, Any]]] = {}
    for row in card["rows"]:
        for candidate in row["image"].get("candidates", []):
            if candidate.get("role") == "OUTPUT" and candidate.get("basis") != "REGENERATION_COMPARISON":
                assigned.setdefault((row["prompt_id"], candidate["artifact_id"]), set()).add(
                    (row["output_node_id"], row["batch_index"]))
    for row in card["rows"]:
        image = row["image"]
        kept = [c for c in image.get("candidates", []) if c.get("basis") != "REGENERATION_COMPARISON"]
        job = jobs.get(row["prompt_id"])
        if row.get("output_context") == TERMINAL_WITHOUT_RECORDED_OUTPUTS:
            # No link, old or new, credits a job that ended without recorded outputs; its matches stay context.
            context = [c for c in image.get("contextual_observations", [])
                       if c.get("basis") != "REGENERATION_COMPARISON"]
            context += [{**c, "basis": "REGENERATION_COMPARISON",
                         "memory_source": (sources or {}).get(c.get("artifact_id")), "role": CONTEXT_ROLE}
                        for c in (job or {}).get("contextual_observed", [])]
            row["image"] = _image_column([], None if job is None else {
                "image_status": job.get("image_status"), "image_links_manifest_sha256": links.get("manifest_sha256")},
                contextual=context)
            continue
        if job is None:
            row["image"] = _image_column(kept)
            continue
        prior: dict[str, list[dict[str, Any]]] = {}
        for c in kept:
            prior.setdefault(c["artifact_id"], []).append(c)
        position = row["output_node_id"], row["batch_index"]
        added = []
        for observed in job.get("observed", []):
            artifact_id = observed["artifact_id"]
            cache_positions = assigned.get((row["prompt_id"], artifact_id))
            if cache_positions is not None and position not in cache_positions:
                continue
            existing = prior.get(artifact_id, [])
            # An image already listed per job (by correlation, not by the execution cache) can still earn this
            # output by a verified replay match; one the cache places (output, intermediate, ambiguous, reused)
            # keeps the cache's placement.
            if any(c.get("role") != "JOB" or c.get("cache_node_ids") for c in existing):
                continue
            match = memory.get(artifact_id, {})
            attribution = match.get("output_attribution")
            raster = match.get("selected_raster") or {}
            role = "JOB"
            if attribution is not None and position != (None, None):
                if (attribution == "SUPPORTED" and raster.get("verified") is True
                        and raster.get("prompt_id") == row["prompt_id"]):
                    if position != (raster.get("node_id"), raster.get("batch_index")):
                        continue
                    role = "OUTPUT"
            if existing and role != "OUTPUT":
                continue  # the job-level link already says as much
            completeness = observed.get("completeness")
            if "complete" in match:
                completeness = "COMPLETE" if match["complete"] else "PARTIAL"
            candidate = {"artifact_id": artifact_id, "basis": "REGENERATION_COMPARISON",
                         "carve_type": observed.get("kind"), "agreement": observed.get("agreement"),
                         "memory_source": (sources or {}).get(artifact_id),
                         "completeness": completeness, "role": role}
            if existing:
                candidate["also_linked_by"] = sorted({c.get("basis") for c in existing if c.get("basis")})
            added.append(candidate)
        row["image"] = _image_column(kept + added, {"image_status": job.get("image_status"),
                                                    "image_links_manifest_sha256": links.get("manifest_sha256")})
    card["summary"] = _summary(card["rows"])
    return card


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    jobs = {r["prompt_id"]: r for r in rows}
    outputs = [r for r in rows if r["output_node_id"] is not None and r["batch_index"] is not None]
    unproduced = {pid for pid, r in jobs.items() if r.get("output_context") == TERMINAL_WITHOUT_RECORDED_OUTPUTS}
    return {
        **({"jobs_terminal_without_recorded_outputs": len(unproduced)} if unproduced else {}),
        "jobs_in_backend_holders": len(jobs),
        "live_jobs": sum(1 for r in jobs.values() if r["liveness"] == "LIVE"),
        "api_prompt_recovered": sum(1 for r in jobs.values() if r["api_prompt"]["status"] == "RECOVERED"),
        "ui_workflow_recovered": sum(1 for r in jobs.values() if r["ui_workflow"]["status"] == "RECOVERED"),
        "ui_workflow_not_present": sum(1 for r in jobs.values() if r["ui_workflow"]["status"] == "NOT_PRESENT"),
        "outputs": len(outputs),
        "jobs_with_linked_image": len({r["prompt_id"] for r in rows if r["image"]["candidates"]}),
        "outputs_with_linked_image": sum(1 for r in outputs if r["image"]["status"] == "LINKED"),
        "outputs_with_linked_memory_image": sum(1 for r in rows if any(
            c.get("role") == "OUTPUT" and c.get("memory_source") == "MEMORY" for c in r["image"]["candidates"])),
        "outputs_with_execution_cache_image": sum(1 for r in rows if any(
            c.get("role") == "OUTPUT" and c.get("cache_node_ids") for c in r["image"]["candidates"])),
    }


def source_tree_sha256(root: Any) -> str:
    """Hash of the package source files, including uncommitted edits."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
