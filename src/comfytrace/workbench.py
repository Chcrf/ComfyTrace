"""The examiner's workbench: ComfyTrace's single interactive, offline HTML report (``workbench.html``).

It renders effective findings (``findings.effective_findings``: regenerations and image links applied)
through ``_Report`` lookups (lead workflow, parameters, applied replay, bounded hash-checked previews).
The page is a fixed template (``templates/workbench.html``); this module builds the JSON the template reads.
Evidence semantics:

- an image recovered as bytes from memory or the file cache is *direct recovery*, complete or partial;
- ``derived`` marks reconstructions only (``is_derived`` records, content-stitched PNGs);
- a regenerated image is a *lab replay*, shown beside the recovered one, never as if it had been found in memory.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import quote

from comfytrace.carve import canonical_json, extract_prompt_graph, native_pixel_digest
from comfytrace.casefiles import (
    EVIDENCE_NAME,
    LINKS_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    derived_file,
    load_regenerations,
)
from comfytrace.exceptions import IngestionError
from comfytrace.findings import effective_findings
from comfytrace.reportview import (
    _CARVE_LABEL,
    _SOURCE_TEXT,
    _TIER_MEANING,
    _TIER_NAME,
    _TIERS,
    _basename,
    _human_bytes,
    _Report,
    _where,
)

WORKBENCH_NAME = REPORT_NAME
UNTIERED_LISTED = 200  # untiered findings listed on the page; the rest are counted and stay in manifest.json
# Plain-English origin of a workflow record, by its ``encoding_detected``. Unlisted media-container encodings
# (``<TYPE>_EMBEDDED_TEXT``) are named by ``_encoding_text``; anything else is shown as recorded.
_ENCODING_TEXT = {
    "CPYTHON_OBJECTS": "decoded from ComfyUI's Python objects",
    "RESERIALIZED_PYTHON_OBJECTS": "decoded from ComfyUI's Python objects",
    "SERIALIZED_QUEUE_ITEM": "decoded from ComfyUI's job queue",
    "PNG_TEXT": "read from a PNG's text chunk",
    "DECOMPRESSED_OR_DECODED_PNG_TEXT": "read from a PNG's text chunk",
    "CONTAINER_METADATA_TEXT": "read from media container metadata",
    "PNG_TEXT_STITCHED": "reconstructed from a PNG's torn text chunk",
    "UTF-8": "carved JSON text",
    "UTF-8_JSON_STRING": "decoded from a Python UTF-8 JSON string",
    "UTF-16LE": "carved JSON text (UTF-16)",
    "UTF-16LE_JSON_STRING": "decoded from a Python UTF-16 JSON string",
}
_DATA_MARK = "__COMFYTRACE_DATA__"
_TITLE_MARK = "__COMFYTRACE_TITLE__"


def _text(value: Any) -> str:
    return "—" if value is None or value == "" else str(value)


def _encoding_text(encoding: str | None) -> str:
    """Where a workflow record's text came from, in plain words; an unknown encoding is shown as recorded."""
    if not encoding:
        return "—"
    if encoding in _ENCODING_TEXT:
        return _ENCODING_TEXT[encoding]
    if encoding.endswith("_EMBEDDED_TEXT"):
        return f"read from a {encoding.removesuffix('_EMBEDDED_TEXT')} file's embedded metadata"
    return encoding


class _Builder:
    def __init__(self, rep: _Report) -> None:
        self.rep = rep
        self.links_by_artifact: dict[str, list[dict[str, Any]]] = {}
        for link in rep.link_records.values():
            self.links_by_artifact.setdefault(link.get("artifact_id", ""), []).append(link)

    # ---- images ---------------------------------------------------------------------------------------------

    def _preview(self, a: dict[str, Any]) -> dict[str, Any]:
        """The recovered bytes when they are a complete PNG/WebP, else the case's derived rendering of the pixels."""
        raw = a.get("raw_file_paths") or []
        dm = a.get("derived_preview_metadata") or {}
        complete = a["completeness"] == "COMPLETE"
        if complete and len(raw) == 1 and raw[0].lower().endswith((".png", ".webp")):
            return self.rep.previews.load(raw[0], a.get("container_sha256"), a.get("pixel_sha256"))
        if a.get("derived_preview_path"):
            return self.rep.previews.load(a["derived_preview_path"], dm.get("preview_sha256"),
                                          a.get("pixel_sha256") if complete else None)
        return {"omitted": "no previewable rendering was exported for this artifact"}

    def _export_link(self, path: str | None) -> str | None:
        if not path or Path(path).is_absolute():
            return None
        try:
            root = self.rep.case_dir.resolve()
            file = (root / path).resolve()
            if file.is_relative_to(root / "exports") and file.is_file():
                return quote(file.relative_to(root).as_posix())
        except (OSError, ValueError, RuntimeError):
            pass
        return None

    def image(self, a: dict[str, Any]) -> dict[str, Any]:
        ident = a["artifact_id"]
        pm = a.get("pixel_metadata") or {}
        dm = a.get("derived_preview_metadata") or {}
        complete = a["completeness"] == "COMPLETE"
        pct = 100
        rows = None
        if not complete and type(dm.get("total_rows")) is int and dm["total_rows"] > 0:
            decoded = dm.get("decoded_rows") if type(dm.get("decoded_rows")) is int else 0
            # Rounded to a percentage, 1 row of 512 reads as 0%: the row count is what is shown.
            pct = max(0, min(99, int(100 * decoded / dm["total_rows"])))
            rows = f"{decoded} of {dm['total_rows']} rows"
        width, height = pm.get("width"), pm.get("height")
        if not width and isinstance(a.get("dimensions"), list) and len(a["dimensions"]) == 2:
            width, height = a["dimensions"]
        links = self.links_by_artifact.get(ident, [])
        keys = sorted({k for link in links for k in link.get("matched_keys", [])})
        record = self.rep.records.get(ident, {})
        assembly = (record.get("assembly") or a.get("evidence", {}).get("assembly") or {})
        loaded = self._preview(a)
        return {
            "id": ident,
            "kind": _CARVE_LABEL.get(a["carve_type"], a["carve_type"]),
            "completeness": "COMPLETE" if complete else (f"PARTIAL · {rows}" if rows else "PARTIAL"),
            "partialPct": pct,
            "rows": rows,
            "size": f"{width}x{height}" if width and height else "—",
            "link": a.get("link_status") or "UNASSOCIATED",
            "key": ", ".join(keys) or "—",
            "linkDigest": ", ".join(link["link_id"] for link in links) or "—",
            "source": _SOURCE_TEXT.get(self.rep.source(ident) or "", "page origin not recorded"),
            "process": _where(a) or "—",
            "offset": a.get("virtual_address") or next(iter(a.get("physical_offsets") or []), None) or "—",
            "pixelSha256": a.get("pixel_sha256"),
            "containerSha256": a.get("container_sha256"),
            "derived": bool(a.get("is_derived") or assembly.get("derived")),
            "derivedNote": (f"Reconstructed by {assembly['method'].lower().replace('_', ' ')}"
                            if assembly.get("derived") and assembly.get("method") else None),
            "uri": loaded.get("uri"),
            "previewNote": loaded.get("pixel_note") or (
                "No recorded hash verifies this preview." if loaded.get("uri") and loaded.get("check") == "none"
                else None),
            "omitted": loaded.get("omitted"),
            "file": self._export_link(next(iter(a.get("raw_file_paths") or []), None)),
            "comparison": None,
        }

    # ---- workflow, job and replay ---------------------------------------------------------------------------

    def _graph(self, workflow: dict[str, Any]) -> dict[str, Any]:
        rel = workflow.get("raw_file_path")
        if workflow.get("prompt_format") != "API_GRAPH" or not rel:
            return {}
        try:
            path = (self.rep.case_dir / rel).resolve()
            if not path.is_relative_to(self.rep.case_dir.resolve()):
                return {}
            classified = extract_prompt_graph(json.loads(path.read_bytes()))
        except (OSError, ValueError, RuntimeError):
            return {}
        if not classified or classified[0] != "API_GRAPH":
            return {}
        graph = classified[1]
        expected = workflow.get("canonical_json_sha256")
        try:
            actual = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
        except (ValueError, RecursionError):
            return {}
        if expected and actual != expected:
            return {}
        return graph

    def workflow(self, workflows: list[dict[str, Any]]) -> dict[str, Any] | None:
        lead = self.rep.lead(workflows)
        if lead is None:
            return None
        p = self.rep.params(workflows)
        return {
            "id": lead["workflow_id"],
            "format": {"API_GRAPH": "API graph", "UI_WORKFLOW": "UI workflow", "API_FRAGMENT": "API fragment",
                       "UI_FRAGMENT": "UI fragment", "API_NODE_FRAGMENT": "API node fragment"}.get(
                           lead["prompt_format"], lead["prompt_format"]),
            "source": _encoding_text(lead.get("encoding_detected")),
            "sha256": lead.get("canonical_json_sha256"),
            "params": {
                "checkpoint": _text(p.get("model_name")), "seed": _text(p.get("seed")),
                "steps": _text(p.get("steps")), "cfg": _text(p.get("cfg_scale")),
                "sampler": _text(p.get("sampler_name")), "scheduler": _text(p.get("scheduler")),
                "width": _text(p.get("width")), "height": _text(p.get("height")),
                "positive": _text(p.get("positive_prompt")), "negative": _text(p.get("negative_prompt")),
            },
            "copies": [{"id": w["workflow_id"],
                        "source": _encoding_text(w.get("encoding_detected")),
                        "sha256": w.get("canonical_json_sha256") or "—"}
                       for w in workflows if w is not lead],
            "graph": self._graph(lead),
        }

    def job(self, workflows: list[dict[str, Any]]) -> dict[str, Any]:
        p = self.rep.params(workflows)
        prompt_id = p.get("prompt_id")
        line = ((self.rep.timelines.get(str(prompt_id), {}).get("execution_parameters") or {})
                .get("job_timeline") or {})
        run = line.get("run_seconds")
        return {"promptId": _text(prompt_id), "queued": _text(line.get("queued_utc") or line.get("started_utc")),
                "duration": f"{run} s" if run is not None else "—",
                "status": _text(line.get("outcome") or p.get("history_status"))}

    def replay(self, summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The finding's lead verified lab replay (``_Report.replay_summaries``) as three separate facts. The hash
        shown is the derived PNG file's SHA-256, not a pixel digest."""
        if not summaries:
            return None
        lead = summaries[0]
        regen = self.rep.regen_by_id.get(lead["handoff_id"]) or {}
        evaluation = regen.get("evaluation") or {}
        execution, comparison, promotion = lead["execution"], lead["comparison"], lead["promotion"]
        loaded: dict[str, Any] = {"omitted": "no regenerated image is recorded"}
        regen_pixel_sha: str | None = None
        if execution["derived_path"]:
            resolved = derived_file(self.rep.case_dir, execution["derived_path"])
            loaded = (self.rep.previews.load(execution["derived_path"], execution["derived_file_sha256"], None, True)
                      if resolved is not None
                      else {"omitted": "the regenerated image is not under exports/derived"})
            if resolved is not None and loaded.get("check") == "match":
                try:
                    regen_pixel_sha, _ = native_pixel_digest(resolved.read_bytes())
                except (OSError, ValueError, SyntaxError):
                    regen_pixel_sha = None
        ssim = comparison.get("ssim")
        raw = lead["observation"] == "RAW_WHOLE_FRAME"
        result = (comparison["status"] if raw
                  else evaluation.get("evaluation_subtier") or evaluation.get("evaluated_tier") or regen.get("status"))
        selection = evaluation.get("target_selection")
        selection = ({k: selection[k] for k in ("selection_reason", "candidate_artifact_ids", "excluded_candidates")
                      if k in selection} if isinstance(selection, dict) else None)
        return {"id": lead["handoff_id"], "workflowId": lead["workflow_id"], "scope": lead["scope"],
                "observation": lead["observation"], "result": result,
                "comparisonStatus": comparison["status"], "match": comparison["match"],
                "exact": comparison["match"] and comparison["status"] in ("EXACT_MATCH", "ROW_SLICE_EXACT"),
                "ssim": f"{ssim:.3f}" if isinstance(ssim, (int, float)) else None,
                "rows": comparison.get("rows_text"), "testedArtifact": comparison.get("target_artifact_id"),
                "recordedVerdict": comparison.get("recorded_verdict"),
                "rawFrame": ({k: comparison.get(k) for k in ("raw_file", "raw_sha256", "physical_offset",
                                                              "pixel_sha256")} if raw else None),
                "targetSelection": selection or None,
                "summary": comparison["label"], "execution": execution["label"], "promotion": promotion["label"],
                "promoted": promotion["changed"], "applied": promotion["applied"],
                "outputNode": execution["output_node"], "batchIndex": execution["batch_index"],
                "derivedPath": execution["derived_path"], "derivedFileSha256": execution["derived_file_sha256"],
                "derivedPixelSha256": regen_pixel_sha,
                "others": [{"id": s["handoff_id"], "scope": s["scope"], "observation": s["observation"],
                            "summary": s["comparison"]["label"], "promotion": s["promotion"]["label"]}
                           for s in summaries[1:]],
                "uri": loaded.get("uri"), "omitted": loaded.get("omitted")}

    def finding(self, f: dict[str, Any]) -> dict[str, Any]:
        images, workflows, links = self.rep.parts(f)
        shown = [self.image(a) for a in images]
        containers = self.rep.containers(f, images)
        own_boxes = {c["artifact_id"] for c in containers if c["from_finding"] is None}
        for im in shown:
            im["container"] = im["id"] in own_boxes
        summaries = self.rep.replay_summaries(f)
        for im in shown:
            for s in summaries:
                fact = s["artifacts"].get(im["id"])
                if fact and (im["comparison"] is None or (im["comparison"]["status"] == "NOT_COMPARED"
                                                          and fact["status"] != "NOT_COMPARED")):
                    im["comparison"] = {"status": fact["status"], "label": fact["label"], "match": fact["match"],
                                        "replay": s["handoff_id"]}
        tier = f["tier"] if f["tier"] in _TIERS else "UT"
        label, desc = (self.rep.untiered_label(f) if tier == "UT" else (_TIER_NAME[tier], _TIER_MEANING[tier]))
        digest = self.rep.digest(f)
        # The tier the finding had from extraction alone, before a lab replay or image link raised it: the page
        # names a promotion instead of showing an extraction rationale that argues for the lower tier.
        base = f.get("base_tier", f["tier"])
        base_tier = base if base in _TIERS else ("UT" if base == "UNTIERED" else tier)
        return {"id": f["finding_id"], "tier": tier, "baseTier": base_tier, "tierLabel": label, "tierDesc": desc,
                "rationale": f["rationale"], "job": self.job(workflows), "groupId": _group_id(digest),
                "workflowIds": [w["workflow_id"] for w in workflows],
                "links": [{"id": link["link_id"], "class": link.get("link_class"), "workflow": link.get("workflow_id"),
                           "artifact": link.get("artifact_id"), "keys": link.get("matched_keys") or []}
                          for link in links],
                "closeMatchScope": f.get("close_match_scope"), "outputCandidates": f.get("output_candidates") or [],
                "containerProvenance": [{"artifactId": c["artifact_id"], "rows": c["rows_text"],
                                         "fromFinding": c["from_finding"], "linkIds": c["link_ids"]}
                                        for c in containers if c["from_finding"] is not None],
                "images": shown, "workflow": self.workflow(workflows), "replay": self.replay(summaries)}

    # ---- workflow groups ------------------------------------------------------------------------------------

    def groups(self, pairs: list[tuple[dict[str, Any], dict[str, Any]]],
               tiered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Findings grouped by the canonical digest of their lead workflow: the graph and its lab replays once,
        every child finding with its own tier, artifacts and artifact-scoped comparison beneath."""
        members: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for raw, shown in pairs:
            digest = self.rep.digest(raw)
            if digest:
                members.setdefault(digest, []).append((raw, shown))
        fragments = self._one_seed_fragments(tiered)
        out = []
        for digest, children in members.items():
            lead = next((shown["workflow"] for _, shown in children if shown["workflow"]), None) or {}
            copies = sorted({w["workflow_id"] for w in self.rep.workflows.values()
                             if w.get("canonical_json_sha256") == digest})
            prompts = sorted({str(p) for wid in copies
                              if (p := (self.rep.workflows[wid].get("execution_parameters") or {}).get("prompt_id"))})
            unbound = [wid for wid in copies if not (self.rep.workflows[wid].get("execution_parameters") or {})
                       .get("prompt_id")]
            replays: dict[str, dict[str, Any]] = {}
            rank = {"RAW_WHOLE_FRAME": 0, "TESTED": 0, "NO_TARGET": 1, "UNTESTED_SIBLING": 2}
            applied_to: dict[str, list[str]] = {}
            for raw, _ in children:
                for s in self.rep.replay_summaries(raw):
                    key = f"{s['handoff_id']}:{s['observation']}"
                    if s["promotion"]["applied"]:
                        # Each observation's own result: a run whose later raw whole-frame match raised a finding to
                        # E1 is "applied" on that observation only, not on the run's original comparison.
                        applied_to.setdefault(key, []).append(raw["finding_id"])
                    known = replays.get(key)
                    if known is None or rank[s["scope"]] < rank[known["scope"]]:
                        replays[key] = {
                            "id": s["handoff_id"], "workflowId": s["workflow_id"], "scope": s["scope"],
                            "observation": s["observation"],
                            "execution": s["execution"]["label"], "outputNode": s["execution"]["output_node"],
                            "batchIndex": s["execution"]["batch_index"], "derivedPath": s["execution"]["derived_path"],
                            "derivedFileSha256": s["execution"]["derived_file_sha256"],
                            "testedArtifact": s["comparison"].get("target_artifact_id"),
                            "testedFinding": self.rep.owners.get(s["comparison"].get("target_artifact_id") or ""),
                            "comparisonStatus": s["comparison"]["status"], "comparison": s["comparison"]["label"],
                            "recordedVerdict": s["comparison"].get("recorded_verdict")}
            for key, entry in replays.items():
                entry["appliedTo"] = ", ".join(applied_to.get(key, [])) or None
            seeds = {str(seed) for seed in [lead.get("params", {}).get("seed")] if seed not in (None, "—")}
            xrefs = [{"findingId": fid, "tier": tier, "seed": seed, "workflowIds": wids,
                      "note": f"{len(wids)} fragment(s) agreeing only with seed {seed} of this graph; kept as its own "
                              f"{tier} finding with its own provenance, not merged into this group"}
                     for fid, tier, seed, wids in fragments
                     if seed in seeds and fid not in {raw["finding_id"] for raw, _ in children}]
            out.append({
                "id": _group_id(digest), "digest": digest, "leadWorkflowId": lead.get("id"),
                "format": lead.get("format"), "params": lead.get("params") or {}, "copies": copies,
                "submissions": {"promptIds": prompts, "unboundWorkflowIds": unbound,
                                "note": (f"{len(prompts)} known submission(s) by prompt ID. A lab run shows the graph "
                                         "runs; it does not show which submission produced each output."
                                         if prompts else
                                         "No prompt ID was recovered for this graph: its records are unbound copies "
                                         "and name no submission. A lab run shows the graph runs, nothing more.")},
                "replays": sorted(replays.values(), key=lambda r: (str(r["id"]), r["observation"])),
                "seedXrefs": xrefs,
                "children": [self._child(shown) for _, shown in children],
            })
        return out

    @staticmethod
    def _child(shown: dict[str, Any]) -> dict[str, Any]:
        return {"id": shown["id"], "tier": shown["tier"], "artifactIds": [i["id"] for i in shown["images"]],
                "workflowIds": shown["workflowIds"], "linkIds": [link["id"] for link in shown["links"]],
                "comparisons": {i["id"]: (i["comparison"] or {}).get("label") or "no lab replay compared it"
                                for i in shown["images"]},
                "containerProvenance": shown["containerProvenance"]}

    def _one_seed_fragments(self, tiered: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
        """E6 findings whose fragments all agree with exactly one recovered seed: (finding, tier, seed, fragments)."""
        out = []
        for f in tiered:
            if f["tier"] != "E6":
                continue
            _, workflows, _ = self.rep.parts(f)
            fragments = [self.rep.records.get(w["workflow_id"], {}).get("fragment") or {} for w in workflows]
            seeds = {str(s) for fr in fragments if fr.get("attribution") == "ONE_SEED"
                     for s in fr.get("agreeing_seeds") or []}
            if len(seeds) == 1 and fragments and all(fr.get("attribution") == "ONE_SEED" for fr in fragments):
                out.append((f["finding_id"], f["tier"], next(iter(seeds)), [w["workflow_id"] for w in workflows]))
        return out


def _group_id(digest: str | None) -> str | None:
    return f"WG-{digest}" if digest else None


def workbench_data(rep: _Report) -> dict[str, Any]:
    """The JSON the workbench template reads: case header, tier legend and counts, findings strongest first, and
    the workflow groups they belong to."""
    builder = _Builder(rep)
    ranked = rep.ordered()  # tiered first, so the preview budget is spent on the strongest findings
    tiered = [f for f in ranked if f["tier"] in _TIERS]
    untiered = [f for f in ranked if f["tier"] not in _TIERS]
    meta = rep.meta
    image = meta.get("memory_image") or {}
    totals = rep.summary_counts(ranked)
    tiers = {t: {"code": t, "short": _TIER_NAME[t], "desc": _TIER_MEANING[t],
                 "count": totals["findings_by_tier"][t]} for t in _TIERS}
    # Untiered findings are counted as findings; the "Other images" bucket is one finding of many artifacts.
    tiers["UT"] = {"code": "UT", "short": "Untiered images", "desc": _TIER_MEANING["UNTIERED"],
                   "count": len(untiered), "otherImagesFindings": totals["other_images_findings"],
                   "otherImagesArtifacts": totals["other_images_artifacts"]}
    listed = untiered[:UNTIERED_LISTED]
    shown_tiered = [builder.finding(f) for f in tiered]
    shown_untiered = [builder.finding(f) for f in listed]
    return {
        "case": {"tool": "ComfyTrace", "version": _text(meta.get("comfytrace_version")),
                 "caseId": _text(meta.get("case_id")), "dumpName": _basename(image.get("path")) or "—",
                 "dumpSize": _human_bytes(image.get("size_bytes")), "dumpSha256": image.get("sha256") or "",
                 "mode": _text(meta.get("carving_mode")), "manifestSha256": rep.sha},
        "tiers": tiers,
        "counts": totals,
        "findings": shown_tiered,
        "untiered": shown_untiered,
        "groups": builder.groups([*zip(tiered, shown_tiered, strict=True), *zip(listed, shown_untiered, strict=True)],
                                 tiered),
    }


def render_workbench_html(data: dict[str, Any]) -> str:
    template = resources.files("comfytrace").joinpath("templates/workbench.html").read_text(encoding="utf-8")
    # Inside <script type="application/json"> only "</" can end the block early; JSON allows "<\/" for it.
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"{data['case']['caseId']} · {data['case']['dumpName']}")
    before, after = template.split(_DATA_MARK, 1)
    return before.replace(_TITLE_MARK, title, 1) + payload + after


def write_workbench(case_dir: Path, manifest_path: Path | None = None, regenerations_dir: Path | None = None,
                    output_html: Path | None = None) -> Path:
    """Write the case workbench atomically. Unreadable case inputs raise IngestionError."""
    case_dir = Path(case_dir)
    manifest_path = manifest_path or case_dir / MANIFEST_NAME
    try:
        serialized = Path(manifest_path).read_bytes()
        manifest = json.loads(serialized)
        manifest["findings"]["findings"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IngestionError(f"cannot read manifest {manifest_path}: {exc}") from exc
    manifest_sha = hashlib.sha256(serialized).hexdigest()
    regen_dir = regenerations_dir or case_dir / "regenerations"
    try:
        regenerations = load_regenerations(regen_dir)
    except (OSError, ValueError) as exc:
        raise IngestionError(f"cannot read regenerations in {regen_dir}: {exc}") from exc
    if not all(isinstance(record, dict) for record in regenerations):
        raise IngestionError(f"cannot read regenerations in {regen_dir}: expected JSON objects")
    sidecars: dict[str, dict[str, Any]] = {}
    for name in (EVIDENCE_NAME, LINKS_NAME):
        sidecar_path = case_dir / name
        if sidecar_path.is_file():
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise IngestionError(f"cannot read sidecar {sidecar_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise IngestionError(f"cannot read sidecar {sidecar_path}: expected JSON object")
            sidecars[name] = loaded
    findings, notes = effective_findings(manifest, regenerations, manifest_sha, case_dir, sidecars.get(LINKS_NAME))
    rep = _Report(case_dir, manifest, manifest_sha, findings, regenerations, notes, sidecars.get(EVIDENCE_NAME))
    out = Path(output_html) if output_html is not None else case_dir / WORKBENCH_NAME
    rendered = render_workbench_html(workbench_data(rep))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=out.parent, prefix=f".{out.name}.", suffix=".tmp", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(rendered)
        os.replace(tmp_path, out)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return out
