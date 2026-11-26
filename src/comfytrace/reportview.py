"""Case lookups and verified previews for the examiner workbench."""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import re
from pathlib import Path
from typing import Any

from comfytrace.carve import native_pixel_digest
from comfytrace.casefiles import matches_derived_file
from comfytrace.correlate import OTHER_IMAGES
from comfytrace.findings import (
    comparison_fact,
    comparison_only_output,
    derived_image_matches,
    is_number,
    regeneration_permits_promotion,
)


def _timeline_jobs(workflows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The workflow record that carries each job's timeline, by prompt_id (an API graph wins over a copy)."""
    jobs: dict[str, dict[str, Any]] = {}
    for w in workflows:
        params = w.get("execution_parameters") or {}
        if "job_timeline" in params and (params.get("prompt_id") not in jobs or w["prompt_format"] == "API_GRAPH"):
            jobs[params["prompt_id"]] = w
    return jobs


_TIERS = ["E1", "E2", "E3", "E4", "E5", "E6"]

_TIER_RANK = {tier: n for n, tier in enumerate([*_TIERS, "UNTIERED"])}

_TIER_NAME = {
    "E1": "Reproduced",
    "E2": "Image and prompt recovered together",
    "E3": "Linked, but partial or a close match",
    "E4": "Prompt shown to run",
    "E5": "Complete prompt, no image",
    "E6": "Partial prompt, no image",
    "UNTIERED": "Image not tied to a job",
}

_TIER_MEANING = {
    "E1": "An image recovered intact from memory was reproduced by re-running the recovered prompt on a "
    "ComfyUI runner.",
    "E2": "A complete image and a complete prompt were recovered, and a strong key ties them: an identical prompt "
    "hash or prompt ID, or an exact whole-frame match with the job's regenerated image.",
    "E3": "An image and a prompt are linked, but the image or the prompt is partial, or the image is a close match "
    "rather than an exact one.",
    "E4": "A complete prompt with no recovered image was re-run successfully. It can run; that does not show an "
    "image existed in memory.",
    "E5": "A complete prompt graph with its execution settings; no recovered image is tied to it.",
    "E6": "Part of a prompt (a node fragment or a saved UI workflow); no recovered image is tied to it.",
    "UNTIERED": "An image that no strong key ties to a recovered prompt, or that matches several equally.",
}

_SOURCE_TEXT = {
    "MEMORY": "on ordinary memory pages",
    "MEMORY_PARTIAL": "only partly on ordinary memory pages",
    "FILE_BACKED": "only on file-backed pages (a file may stand behind it)",
    "MIXED": "on a mix of file-backed and other pages, none wholly in memory",
    "UNKNOWN": "on pages of unknown origin",
    "NOT_CLASSIFIED": "page origin not classified",
}

_CARVE_LABEL = {
    "ENCODED_PNG": "PNG file",
    "PARTIAL_PNG": "torn PNG file",
    "ENCODED_WEBP": "WebP file",
    "PARTIAL_WEBP": "torn WebP file",
    "ENCODED_MP4": "MP4 video",
    "PARTIAL_MP4": "torn MP4 video",
    "ENCODED_WEBM": "WebM video",
    "PARTIAL_WEBM": "torn WebM video",
    "ENCODED_FLAC": "FLAC audio",
    "PARTIAL_FLAC": "torn FLAC audio",
    "PIXEL_BUFFER": "pixel buffer in memory",
    "TENSOR_IMAGE": "image tensor in ComfyUI's memory",
    "DECODED_BITMAP": "decoded bitmap in memory",
}


def _natural(text: Any) -> tuple[tuple[int, int, str], ...]:
    """Sort key that puts FIND-2 before FIND-10."""
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"(\d+)", str(text)) if p)


def _human_bytes(count: Any) -> str:
    if not isinstance(count, int):
        return str(count)
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return str(count)


def _basename(path: Any) -> str:
    return re.split(r"[\\/]", str(path))[-1] if path else ""


def _where(item: dict[str, Any]) -> str | None:
    pid = item.get("process_id", item.get("source_process_id"))
    offsets = item.get("physical_offsets") or []
    bits = []
    if pid is not None:
        bits.append(f"PID {pid}")
    if item.get("virtual_address"):
        bits.append(f"VA {item['virtual_address']}")
    if offsets:
        bits.append("physical " + ", ".join(offsets[:2]) + (f" (+{len(offsets) - 2} more)" if len(offsets) > 2 else ""))
    return " · ".join(bits) or None


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _image_mime(data: bytes) -> str | None:
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class _Previews:
    """Bounded, offline previews: a file is embedded as a data URI only when it is a PNG or WebP under the case
    folder, within a per-image and a whole-report byte budget; a larger image is scaled down (and says so)."""

    PER_IMAGE = 300_000
    TOTAL = 6_000_000
    MAX_COUNT = 40
    READ_LIMIT = 40_000_000
    SIDES = (640, 400, 240)

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir).resolve()
        self.used = 0
        self.count = 0
        self.cache: dict[Any, dict[str, Any]] = {}

    def load(
        self, rel: str, expected: str | None = None, pixel_sha256: str | None = None, require_hash: bool = False
    ) -> dict[str, Any]:
        """Reads one image for embedding and checks the bytes it actually holds against the hash the case recorded:
        ``expected`` is the SHA-256 of the file itself, ``pixel_sha256`` the digest of its decoded pixels (used only
        when no file hash was recorded). ``check`` is "match", "pixels" (pixels match) or "none" (nothing recorded
        to check against); a recorded hash that does not match, or a missing one when ``require_hash`` is set,
        withholds the image."""
        expected = expected.lower() if isinstance(expected, str) and expected else None
        pixel_sha256 = pixel_sha256.lower() if isinstance(pixel_sha256, str) and pixel_sha256 else None
        key = (rel, expected, pixel_sha256, require_hash)
        if key not in self.cache:
            self.cache[key] = self._load(rel, expected, pixel_sha256, require_hash)
        return self.cache[key]

    @staticmethod
    def _pixels_match(data: bytes, expected: str) -> bool:
        try:
            actual, _ = native_pixel_digest(data)
            return actual == expected
        except Exception:  # noqa: BLE001 - any decoding failure just means the pixels could not be checked
            return False

    def _load(self, rel: str, expected: str | None, pixel_sha256: str | None, require_hash: bool) -> dict[str, Any]:
        try:
            path = (self.case_dir / rel).resolve()
            if Path(rel).is_absolute() or not path.is_relative_to(self.case_dir) or not path.is_file():
                return {"omitted": "the file is not in the case folder"}
            if path.stat().st_size > self.READ_LIMIT:
                return {"omitted": "the file is too large to preview"}
            data = path.read_bytes()
        except (OSError, ValueError, RuntimeError):
            return {"omitted": "the file could not be read"}
        info: dict[str, Any] = {
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "check": "none",
        }
        if expected:
            if info["sha256"] != expected:
                return {
                    **info,
                    "check": "mismatch",
                    "expected": expected,
                    "omitted": "the file's SHA-256 does not match the hash recorded for it, so it may have been "
                    "altered or replaced since extraction",
                }
            info["check"] = "match"
        elif require_hash:
            return {**info, "omitted": "no SHA-256 was recorded for this image, so its bytes cannot be verified"}
        mime = _image_mime(data)
        if mime is None:
            return {**info, "omitted": "the file is not a PNG or WebP image"}
        if not expected and pixel_sha256:
            if self._pixels_match(data, pixel_sha256):
                info["check"] = "pixels"
            else:
                info["pixel_note"] = (
                    "the decoded pixels do not reproduce the recorded pixel SHA-256 (the image may be re-rendered)"
                )
        shown, scaled = data, None
        if len(data) > self.PER_IMAGE:
            thumb = self._thumbnail(data)
            if thumb is None:
                return {
                    **info,
                    "omitted": f"the image is over the {self.PER_IMAGE // 1000} KB preview limit and "
                    "could not be scaled down",
                }
            (shown, scaled), mime = thumb, "image/png"
        if self.count >= self.MAX_COUNT or self.used + len(shown) > self.TOTAL:
            return {**info, "omitted": "the report's preview budget is used up"}
        self.used += len(shown)
        self.count += 1
        return {**info, "uri": f"data:{mime};base64,{base64.b64encode(shown).decode('ascii')}", "scaled": scaled}

    def _thumbnail(self, data: bytes) -> tuple[bytes, tuple[tuple[int, int], tuple[int, int]]] | None:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as img:
                img.load()
                original = img.size
                for side in self.SIDES:
                    small = img.copy()
                    small.thumbnail((side, side))
                    if small.mode not in ("RGB", "RGBA", "L", "LA"):
                        small = small.convert("RGB")
                    buffer = io.BytesIO()
                    small.save(buffer, format="PNG", optimize=True)
                    if buffer.tell() <= self.PER_IMAGE:
                        return buffer.getvalue(), (small.size, original)
        except Exception:  # noqa: BLE001 - any decoding failure just means no preview
            return None
        return None


class _Report:
    """Case records indexed for the workbench."""

    def __init__(
        self,
        case_dir: Path,
        manifest: dict[str, Any],
        manifest_sha: str,
        findings: list[dict[str, Any]],
        regenerations: list[dict[str, Any]],
        regen_notes: list[dict[str, Any]],
        evidence: dict[str, Any] | None,
    ) -> None:
        self.case_dir, self.sha = Path(case_dir), manifest_sha
        self.findings, self.regenerations = findings, regenerations
        body = manifest["findings"]
        self.meta = manifest["case_metadata"]
        self.records = {r["record_id"]: r for r in (evidence or {}).get("records", [])}
        self.artifacts = {a["artifact_id"]: a for a in body.get("artifacts", [])}
        self.workflows = {w["workflow_id"]: w for w in body.get("workflows", [])}
        self.link_records = {link["link_id"]: link for link in body.get("correlation_links", [])}
        self.other_images = {a for f in findings if f["rationale"].startswith(OTHER_IMAGES) for a in f["artifact_ids"]}
        self.regen_by_id = {r.get("handoff_id"): r for r in regenerations}
        self.regen_note = {n["handoff_id"]: n for n in regen_notes if "handoff_id" in n}
        self.owners = {
            a: f["finding_id"]
            for f in findings
            for a in f["artifact_ids"]
            if a in self.artifacts and a not in self.other_images
        }
        self._replay_problems: dict[int, str | None] = {}
        self.timelines = _timeline_jobs(list(self.workflows.values()))
        self.previews = _Previews(case_dir)

    def source(self, record_id: str) -> str | None:
        return self.records.get(record_id, {}).get("memory_source")

    def parts(self, finding: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        ids = finding["artifact_ids"]
        images = [self.artifacts[i] for i in ids if i in self.artifacts and i not in self.other_images]
        images.sort(key=lambda a: a["completeness"] != "COMPLETE")
        workflows = [self.workflows[i] for i in ids if i in self.workflows]
        links = [self.link_records[i] for i in finding.get("link_ids", []) if i in self.link_records]
        return images, workflows, links

    @staticmethod
    def _score(workflow: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
        params = workflow.get("execution_parameters") or {}
        return (
            workflow["prompt_format"] == "API_GRAPH",
            workflow["completeness"] == "COMPLETE",
            "prompt_id" in params,
            "positive_prompt" in params,
        )

    def lead(self, workflows: list[dict[str, Any]]) -> dict[str, Any] | None:
        return max(workflows, key=self._score) if workflows else None

    def params(self, workflows: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for w in sorted(workflows, key=self._score):
            merged.update(w.get("execution_parameters") or {})
        return merged

    def digest(self, finding: dict[str, Any]) -> str | None:
        """The canonical digest of a finding's lead workflow: findings that share it share one workflow graph."""
        _, workflows, _ = self.parts(finding)
        return (self.lead(workflows) or {}).get("canonical_json_sha256") or None

    def replay_problem(self, regen: dict[str, Any]) -> str | None:
        """Why a regeneration record cannot be shown as a verified lab run of this case, or None when it can."""
        key = id(regen)
        if key not in self._replay_problems:
            self._replay_problems[key] = self._replay_problem(regen)
        return self._replay_problems[key]

    def _replay_problem(self, regen: dict[str, Any]) -> str | None:
        evaluation = regen.get("evaluation") or {}
        if regen.get("target_manifest_sha256") != self.sha:
            return "made against another manifest"
        if not regeneration_permits_promotion(regen):
            return "needs a completed run, a verified preflight and permitted model formats"
        if not derived_image_matches(self.case_dir, regen):
            return "the derived regenerated image is missing or its SHA-256 does not match"
        if comparison_only_output(regen, (evaluation.get("selected_raster") or {}).get("node_id")):
            return "the selected raster is a reference-comparison output"
        return None

    def replay_summaries(self, finding: dict[str, Any]) -> list[dict[str, Any]]:
        """Every verified lab replay of a finding's workflow graph, as three separate facts: did it run
        (execution), what did its comparison show about the artifact it tested (comparison, scoped to
        ``target_artifact_id``: a sibling finding's images are "not compared"), and did it raise this finding's tier
        (promotion). The applied replay comes first, then one that tested this finding's own image."""
        images, workflows, _ = self.parts(finding)
        image_ids = [a["artifact_id"] for a in images]
        own = {w["workflow_id"] for w in workflows}
        digest = self.digest(finding)
        owners = self.owners
        out = []
        for regen in self.regenerations:
            workflow_id = regen.get("workflow_id")
            same_graph = bool(digest) and (self.workflows.get(workflow_id) or {}).get("canonical_json_sha256") == digest
            if (workflow_id not in own and not same_graph) or self.replay_problem(regen) is not None:
                continue
            handoff = regen.get("handoff_id")
            evaluation = regen.get("evaluation") or {}
            selected, _, _ = self.regen_rasters(regen)
            position = {
                **(evaluation.get("compared_image") or {}),
                **(evaluation.get("selected_raster") or {}),
                **((selected or {}).get("detail") or {}),
            }
            target = evaluation.get("target_artifact_id")
            candidate_evals = evaluation.get("candidate_evaluations") or {}
            no_target = target is None or evaluation.get("evaluation_mode") == "NO_MEMORY_TARGET"
            not_compared = (
                "not compared: the replay had no comparable memory pixels"
                if no_target
                else f"this artifact was not compared; {handoff} compared {target}"
            )
            artifacts: dict[str, dict[str, Any]] = {}
            for i in image_ids:
                if not no_target and i == target:
                    artifacts[i] = comparison_fact(evaluation, self.artifacts.get(i), True)
                elif isinstance(candidate_evals.get(i), dict):
                    artifacts[i] = comparison_fact(candidate_evals[i], self.artifacts.get(i), True)
                else:
                    artifacts[i] = {"status": "NOT_COMPARED", "match": False, "label": not_compared}
            tested_candidate = next((i for i in image_ids if isinstance(candidate_evals.get(i), dict)), None)
            if no_target and tested_candidate is None:
                scope = "NO_TARGET"
                comparison = comparison_fact(evaluation, None, bool(images))
            elif not no_target and target in image_ids:
                scope = "TESTED"
                comparison = artifacts[target]
            elif tested_candidate is not None:
                scope = "TESTED"
                comparison = artifacts[tested_candidate]
            else:
                scope = "UNTESTED_SIBLING"
                where = f" ({owners[target]})" if target in owners else ""
                comparison = {
                    "status": "NOT_COMPARED",
                    "match": False,
                    "target_artifact_id": target,
                    "target_finding_id": owners.get(target),
                    "label": f"this artifact was not compared; {handoff} compared {target}{where}",
                }
            applied = bool(handoff) and finding.get("regeneration") == handoff
            base, tier = finding["base_tier"], finding["tier"]
            note = self.regen_note.get(handoff, {}).get("note")
            raw = finding.get("replay_evidence") or {}
            superseded = applied and raw.get("handoff_id") == handoff
            if superseded:
                # replay-search found this run's output as a whole frame in the raw dump: that later observation
                # raised the tier, not the comparison recorded when the run was made.
                applied = False
                comparison = {**comparison, "label": f"original lab comparison: {comparison['label']}"}
                promotion_label = (
                    f"this original comparison raised no tier; the raw whole-frame match of {handoff}'s "
                    f"output (a separate observation) raised the finding to {tier}"
                )
            elif applied and tier != base:
                promotion_label = f"raised this finding from {base} to {tier}"
            elif applied:
                promotion_label = f"applied to this finding; the tier stays {tier}"
            elif scope == "UNTESTED_SIBLING":
                promotion_label = f"not applied to this finding (it tested another artifact); the tier stays {tier}"
            elif comparison["match"]:
                promotion_label = f"not applied ({note or 'no promotion rule matched'}); the tier stays {tier}"
            else:
                promotion_label = f"not applied: this comparison raises no tier; the tier stays {tier}"
            out.append(
                {
                    "handoff_id": handoff,
                    "workflow_id": workflow_id,
                    "scope": scope,
                    "observation": "ORIGINAL_EVALUATION",
                    "execution": {
                        "handoff_id": handoff,
                        "workflow_id": workflow_id,
                        "same_workflow_record": workflow_id in own,
                        "output_node": position.get("node_id"),
                        "batch_index": position.get("batch_index"),
                        "derived_path": (selected or {}).get("path"),
                        "derived_file_sha256": (selected or {}).get("sha256"),
                        "label": self._execution_label(regen, selected, position, workflow_id in own),
                    },
                    "comparison": comparison,
                    "artifacts": artifacts,
                    "promotion": {
                        "applied": applied,
                        "from": base,
                        "to": tier,
                        "changed": applied and tier != base,
                        "label": promotion_label,
                    },
                }
            )
        if (observation := self.raw_observation(finding, image_ids)) is not None:
            out.append(observation)
        rank = {"RAW_WHOLE_FRAME": 0, "TESTED": 1, "NO_TARGET": 2, "UNTESTED_SIBLING": 3}
        out.sort(key=lambda s: (not s["promotion"]["applied"], rank[s["scope"]]))
        return out

    def raw_raster(self, finding: dict[str, Any]) -> dict[str, Any] | None:
        """The regenerated output a raw whole-frame match (``replay_evidence``) matched, with its recorded position,
        when its file is under exports/derived and its SHA-256 still matches."""
        raw = finding.get("replay_evidence") or {}
        path, digest = raw.get("regenerated_image"), raw.get("regenerated_sha256")
        regen = self.regen_by_id.get(raw.get("handoff_id"))
        if regen is None or not matches_derived_file(self.case_dir, path, digest):
            return None
        listed = [*(regen.get("output_rasters") or []), (regen.get("evaluation") or {}).get("selected_raster") or {}]
        detail = next(
            (r for r in listed if isinstance(r, dict) and r.get("file") == path and r.get("sha256") == digest), {}
        )
        return {
            "path": path,
            "sha256": digest,
            "selected": True,
            "detail": {k: detail[k] for k in ("node_id", "batch_index") if detail.get(k) is not None},
        }

    def raw_observation(self, finding: dict[str, Any], image_ids: list[str]) -> dict[str, Any] | None:
        """The later raw whole-frame comparison (replay-search) that raised a finding to E1, as its own
        execution / comparison / promotion, with the matched output, not the run's originally selected one."""
        raw = finding.get("replay_evidence") or {}
        raster = self.raw_raster(finding) if raw else None
        if raster is None:
            return None
        handoff, regen = raw.get("handoff_id"), self.regen_by_id[raw.get("handoff_id")]
        metrics = raw.get("metrics") or {}
        subtier = raw.get("subtier")
        ssim, phash = metrics.get("ssim_score"), metrics.get("phash_hamming_distance")
        offset = raw.get("physical_offset")
        where = (
            f"the raw memory frame at physical offset {offset:#x} ({raw.get('length_bytes')} bytes, raw SHA-256 "
            f"{raw.get('raw_sha256')})"
            if type(offset) is int
            else "a raw memory frame"
        )
        if subtier == "EXACT_MATCH" and metrics.get("exact_pixel_match") is True:
            comparison = {"status": "EXACT_MATCH", "match": True, "label": f"whole-frame exact pixel match of {where}"}
        elif subtier == "PERCEPTUAL_MATCH" and is_number(ssim) and is_number(phash):
            comparison = {
                "status": "PERCEPTUAL_MATCH",
                "match": True,
                "label": f"whole-frame close perceptual match of {where} (SSIM {ssim:.3f}, pHash distance {phash})",
            }
        else:
            comparison = {
                "status": subtier or "UNRECORDED",
                "match": False,
                "label": f"raw whole-frame comparison of {where} recorded as {subtier}; not treated as a match",
            }
        comparison.update(
            target_artifact_id=None,
            raw_file=raw.get("raw_file"),
            raw_sha256=raw.get("raw_sha256"),
            physical_offset=offset,
            pixel_sha256=raw.get("pixel_sha256"),
            ssim=ssim if is_number(ssim) else None,
            phash=phash if is_number(phash) else None,
        )
        position = raster["detail"]
        tier = finding["tier"]
        applied = finding.get("regeneration") == handoff and comparison["match"]
        return {
            "handoff_id": handoff,
            "workflow_id": regen.get("workflow_id"),
            "scope": "RAW_WHOLE_FRAME",
            "observation": "RAW_WHOLE_FRAME",
            "execution": {
                "handoff_id": handoff,
                "workflow_id": regen.get("workflow_id"),
                "same_workflow_record": True,
                "output_node": position.get("node_id"),
                "batch_index": position.get("batch_index"),
                "derived_path": raster["path"],
                "derived_file_sha256": raster["sha256"],
                "label": self._execution_label(regen, raster, position, True),
            },
            "comparison": comparison,
            "artifacts": {
                i: {
                    "status": "NOT_COMPARED",
                    "match": False,
                    "label": f"this artifact was not compared; the raw frame at {where} was",
                }
                for i in image_ids
            },
            "promotion": {
                "applied": applied,
                "from": finding["extraction_tier"],
                "to": tier,
                "changed": applied,
                "label": (
                    f"raised this finding to {tier} (extraction tier {finding['extraction_tier']}) "
                    "through the raw whole-frame match"
                    if applied
                    else f"not applied; the tier stays {tier}"
                ),
            },
        }

    @staticmethod
    def _execution_label(
        regen: dict[str, Any], selected: dict[str, Any] | None, position: dict[str, Any], same_record: bool
    ) -> str:
        where = ", ".join(
            f"{label} {position[key]}"
            for label, key in (("output node", "node_id"), ("batch index", "batch_index"))
            if position.get(key) is not None
        )
        text = f"{regen.get('handoff_id')} re-ran {regen.get('workflow_id')} in ComfyUI"
        if selected and selected.get("path"):
            text += (
                f" and produced the derived raster {selected['path']}"
                f"{f' ({where})' if where else ''}, derived file SHA-256 {selected.get('sha256') or 'not recorded'}"
            )
        if not same_record:
            text += (
                ". The replayed record is another copy of this finding's workflow graph; a lab run does not show "
                "which submission produced this finding"
            )
        return text

    def untiered_label(self, finding: dict[str, Any]) -> tuple[str, str]:
        """The short name and description of an untiered finding: why it earns no tier, in its own terms."""
        if finding["rationale"].startswith(OTHER_IMAGES):
            return (
                "Other images (not ComfyUI evidence)",
                f"An aggregate of {len(finding['artifact_ids'])} image artifact(s) with no ComfyUI connection. "
                "It is one finding, not one image.",
            )
        _, _, links = self.parts(finding)
        if "pixels a side" in finding["rationale"] and links:
            link_text = "; ".join(
                f"{link['link_id']} ({link['link_class']}, {', '.join(link['matched_keys'])}) to {link['workflow_id']}"
                for link in links
            )
            return (
                "Linked image excluded by size",
                f"{finding['rationale']} The link stands: {link_text}. The prompt is tiered on its own.",
            )
        if "Ambiguous image" in finding["rationale"]:
            return "Image matching several prompts equally", _TIER_MEANING["UNTIERED"]
        return _TIER_NAME["UNTIERED"], _TIER_MEANING["UNTIERED"]

    def summary_counts(self, ranked: list[dict[str, Any]]) -> dict[str, Any]:
        """Findings by tier, distinct workflow graphs and image artifacts, counted separately: several findings
        (one per output) can share one workflow graph, and one finding can hold several image copies."""
        by_tier = collections.Counter(f["tier"] for f in ranked)
        linked = [f for f in ranked if f["tier"] in ("E1", "E2", "E3")]
        tiered = [f for f in ranked if f["tier"] in _TIERS]
        return {
            "findings_by_tier": {t: by_tier[t] for t in [*_TIERS, "UNTIERED"]},
            "linked_findings": len(linked),
            "linked_workflow_graphs": len({self.digest(f) for f in linked} - {None}),
            "linked_image_artifacts": len({a["artifact_id"] for f in linked for a in self.parts(f)[0]}),
            "prompt_only_findings": sum(by_tier[t] for t in ("E4", "E5", "E6")),
            "workflow_graphs": len({self.digest(f) for f in tiered} - {None}),
            "untiered_findings": by_tier["UNTIERED"],
            "other_images_findings": sum(1 for f in self.findings if f["rationale"].startswith(OTHER_IMAGES)),
            "other_images_artifacts": len(self.other_images),
        }

    _RASTER_PATH_KEYS = ("file", "path", "derived_regenerated_image", "image")
    _RASTER_SHA_KEYS = ("sha256", "derived_regenerated_image_sha256")
    _RASTER_DETAIL_KEYS = ("node_id", "batch_index", "ssim_score", "phash_hamming_distance")

    @classmethod
    def _raster(cls, entry: Any) -> dict[str, Any] | None:
        """One regenerated image as a record names it: a path, or an object with a path and optionally its recorded
        SHA-256 and which output of the job it is."""
        if isinstance(entry, str) and entry:
            return {"path": entry, "sha256": None, "detail": {}, "selected": False}
        if isinstance(entry, dict):
            path = next((entry[k] for k in cls._RASTER_PATH_KEYS if isinstance(entry.get(k), str) and entry[k]), None)
            sha = next((entry[k] for k in cls._RASTER_SHA_KEYS if isinstance(entry.get(k), str) and entry[k]), None)
            return {
                "path": path,
                "sha256": sha,
                "selected": entry.get("selected") is True,
                "detail": {k: entry[k] for k in cls._RASTER_DETAIL_KEYS if k in entry},
            }
        return None

    def _pick_raster(
        self,
        raw_list: Any,
        chosen: Any,
        legacy: Any = None,
        legacy_sha: Any = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The selected image and the listed ones of a record. ``chosen`` is an entry, a path or an index into the
        list; a chosen entry without a hash takes the listed one's. With none chosen the entry flagged selected,
        then the legacy single image (a record made before these lists existed), is used."""
        parsed = [self._raster(e) for e in raw_list] if isinstance(raw_list, list) else []
        listed = [r for r in parsed if r and r["path"]]
        selected: dict[str, Any] | None = None
        if isinstance(chosen, int) and not isinstance(chosen, bool):
            selected = parsed[chosen] if 0 <= chosen < len(parsed) else None
        else:
            selected = self._raster(chosen)
            if selected and selected["path"] and not selected["sha256"]:
                selected["sha256"] = next((m["sha256"] for m in listed if m["path"] == selected["path"]), None)
        if selected is None or not selected["path"]:
            selected = next((m for m in listed if m["selected"]), None)
        if selected is None or not selected["path"]:
            selected = (
                {"path": legacy, "sha256": legacy_sha, "detail": {}, "selected": True}
                if isinstance(legacy, str) and legacy
                else None
            )
        elif not selected["sha256"] and selected["path"] == legacy:
            selected["sha256"] = legacy_sha
        return selected, listed

    def regen_rasters(self, regen: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
        """The regenerated image to show for a record, the images the record lists, and what listing means:
        "matched" for ``matching_rasters``, "produced" for ``output_rasters``. Both lists and ``selected_raster`` are
        additive, read from the record or its evaluation; a record made before them has only
        ``derived_regenerated_image`` (with its hash in the evaluation), which is then the one image."""
        evaluation = regen.get("evaluation") or {}

        def field(name: str) -> Any:
            return regen[name] if regen.get(name) is not None else evaluation.get(name)

        role = "matched" if isinstance(field("matching_rasters"), list) and field("matching_rasters") else "produced"
        selected, listed = self._pick_raster(
            field("matching_rasters") if role == "matched" else field("output_rasters"),
            field("selected_raster"),
            regen.get("derived_regenerated_image"),
            evaluation.get("derived_regenerated_image_sha256"),
        )
        return selected, listed, role

    @staticmethod
    def zero_rows(artifact: dict[str, Any]) -> bool:
        """A partial image whose decoder recorded exactly zero pixel rows: a container, not visual evidence."""
        rows = (artifact.get("derived_preview_metadata") or {}).get("decoded_rows")
        return artifact.get("completeness") != "COMPLETE" and type(rows) is int and rows == 0

    def containers(self, finding: dict[str, Any], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The zero-row PNG containers behind a finding: its own (an E4-E6 tiered by its prompt) and those kept as
        container provenance from absorbed findings, each with the link IDs that tie it."""
        out = []
        if finding["tier"] in ("E4", "E5", "E6"):
            for a in images:
                if self.zero_rows(a):
                    out.append(
                        {
                            "artifact_id": a["artifact_id"],
                            "from_finding": None,
                            "link_ids": [
                                link["link_id"]
                                for link in self.link_records.values()
                                if link.get("artifact_id") == a["artifact_id"]
                                and link["link_id"] in finding.get("link_ids", [])
                            ],
                        }
                    )
        for p in finding.get("container_provenance") or []:
            for a in p["artifact_ids"]:
                out.append(
                    {
                        "artifact_id": a,
                        "from_finding": p["finding_id"],
                        "link_ids": [
                            link
                            for link in p["link_ids"]
                            if self.link_records.get(link, {}).get("artifact_id") in (a, None)
                        ],
                    }
                )
        for c in out:
            dm = (self.artifacts.get(c["artifact_id"]) or {}).get("derived_preview_metadata") or {}
            c["rows_text"] = (
                f"{dm.get('decoded_rows', 0)} of {dm['total_rows']} rows"
                if type(dm.get("total_rows")) is int
                else "0 rows"
            )
        return out

    def ordered(self) -> list[dict[str, Any]]:
        def key(f: dict[str, Any]) -> tuple[Any, ...]:
            images, _, _ = self.parts(f)
            return _TIER_RANK.get(f["tier"], len(_TIER_RANK)), not images, _natural(f["finding_id"])

        return sorted((f for f in self.findings if not f["rationale"].startswith(OTHER_IMAGES)), key=key)
