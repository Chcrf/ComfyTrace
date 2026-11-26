"""Optional, preflight-gated regeneration handoff to a ComfyUI runner (Tiers E1 and E4).

Regeneration never touches manifest.json. Each run writes regenerations/REGEN-<id>.json, bound to the
manifest by its SHA-256, and saves regenerated rasters under exports/derived/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath
from typing import Any

from comfytrace.carve import (
    IMAGE_EDIT_INPUTS,
    MODEL_INPUT,
    SAMPLER_CLASSES,
    PngCarver,
    extract_prompt_graph,
    graph_dependencies,
    image_edit_outputs,
    native_pixel_digest,
)
from comfytrace.casefiles import (
    MANIFEST_NAME,
    EvidentiaryExporter,
    next_regeneration_id,
    save_regeneration_record,
    sha256_file,
)
from comfytrace.exceptions import (
    ExecutionTimeoutError,
    MissingDependencyError,
    PromptSubmissionError,
    RegenerationError,
    RunnerConnectionError,
)
from comfytrace.findings import target_is_input
from comfytrace.localrunner import resolve_runner
from comfytrace.models import Completeness, RawArtifactRecord, RegenerationRecord, WorkflowRecord

log = logging.getLogger(__name__)

PASSED = "PASSED"
SECURITY_REJECTED = "SECURITY_REJECTED"
MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
UNAVAILABLE = "UNAVAILABLE"

SAFETENSORS_SUFFIXES = (".safetensors",)
PICKLE_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin", ".pkl")

CORE_NODES = frozenset({
    "KSampler", "KSamplerAdvanced", "CheckpointLoaderSimple", "CheckpointLoader", "unCLIPCheckpointLoader",
    "SamplerCustom", "SamplerCustomAdvanced", "RandomNoise", "DisableNoise", "CFGGuider", "BasicGuider",
    "KSamplerSelect", "BasicScheduler",
    "CLIPTextEncode", "CLIPSetLastLayer", "CLIPLoader", "DualCLIPLoader", "CLIPVisionLoader", "CLIPVisionEncode",
    "CLIPTextEncodeSDXL",
    "VAEDecode", "VAEEncode", "VAEDecodeTiled", "VAEEncodeTiled", "VAEEncodeForInpaint", "VAELoader",
    "UNETLoader", "LoraLoader", "LoraLoaderModelOnly", "ModelMergeSimple", "EmptyLatentImage", "LatentUpscale",
    "LatentUpscaleBy",
    "LatentFromBatch", "RepeatLatentBatch", "LatentComposite", "LatentBlend", "LatentRotate", "LatentFlip",
    "LatentCrop", "SetLatentNoiseMask", "SaveImage", "PreviewImage", "LoadImage", "LoadImageMask",
    "ImageScale", "ImageScaleBy", "ImageInvert", "ImageBatch", "ImagePadForOutpaint", "EmptyImage",
    "ImageCompositeMasked", "ImageBlend", "ImageCrop", "ImageToMask", "Canny",
    "ConditioningAverage", "ConditioningCombine", "ConditioningConcat", "ConditioningSetArea",
    "ConditioningSetAreaPercentage", "ConditioningSetAreaStrength", "ConditioningSetMask",
    "ConditioningZeroOut", "ConditioningSetTimestepRange", "ControlNetApply", "ControlNetApplyAdvanced",
    "ControlNetLoader", "DiffControlNetLoader", "StyleModelLoader", "StyleModelApply", "unCLIPConditioning",
    "GLIGENLoader", "GLIGENTextBoxApply", "InpaintModelConditioning",
    "UpscaleModelLoader", "ImageUpscaleWithModel",
})

RASTER_TYPES = frozenset({"IMAGE", "LATENT"})
MAX_COMPARED_IMAGES = 64
# IMAGE/LATENT inputs of core nodes, used when the runner's /object_info does not declare input types.
CORE_RASTER_INPUTS: dict[str, frozenset[str]] = {
    cls: frozenset(names) for cls, names in {
        "KSampler": {"latent_image"}, "KSamplerAdvanced": {"latent_image"}, "SamplerCustom": {"latent_image"},
        "SamplerCustomAdvanced": {"latent_image"}, "VAEDecode": {"samples"}, "VAEDecodeTiled": {"samples"},
        "VAEEncode": {"pixels"}, "VAEEncodeTiled": {"pixels"}, "VAEEncodeForInpaint": {"pixels"},
        "InpaintModelConditioning": {"pixels"}, "LatentUpscale": {"samples"}, "LatentUpscaleBy": {"samples"},
        "LatentFromBatch": {"samples"}, "RepeatLatentBatch": {"samples"},
        "LatentComposite": {"samples_to", "samples_from"}, "LatentBlend": {"samples1", "samples2"},
        "LatentRotate": {"samples"}, "LatentFlip": {"samples"}, "LatentCrop": {"samples"},
        "SetLatentNoiseMask": {"samples"}, "SaveImage": {"images"}, "PreviewImage": {"images"},
        "ImageBatch": {"image1", "image2"}, **IMAGE_EDIT_INPUTS,
    }.items()
}


def _model_format(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(SAFETENSORS_SUFFIXES):
        return "SAFETENSORS"
    if lowered.endswith(PICKLE_SUFFIXES):
        return "PICKLE"
    return "UNKNOWN"


def _registered_choices(object_info: dict[str, Any]) -> set[str]:
    """Every string option the runner lists for any node input (model filenames among them)."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    found.add(item)
                else:
                    walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for schema in object_info.values():
        if isinstance(schema, dict):
            walk(schema.get("input", {}))
    return found


def _safe_filename(name: Any) -> bool:
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(c in name for c in '/\\:<>"|?*')
            and not any(ord(c) < 32 for c in name)
            and not name.endswith((".", " ", "[output]", "[temp]", "[input]")))


def _relative_parts(name: str) -> tuple[str, ...] | None:
    if not name or PureWindowsPath(name).drive or name.startswith(("/", "\\")):
        return None
    parts = name.replace("\\", "/").split("/")
    return tuple(parts) if all(_safe_filename(part) for part in parts) else None


def _contained_path(root: Path, parts: tuple[str, ...]) -> Path | None:
    try:
        resolved_root = root.resolve()
        candidate = resolved_root.joinpath(*parts).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    return candidate if candidate.is_relative_to(resolved_root) and candidate != resolved_root else None


def input_path(inputs_dir: Path, name: str) -> Path | None:
    """Resolves a single input filename inside --inputs-dir without separator normalization."""
    return _contained_path(inputs_dir, (name,)) if _safe_filename(name) else None


def _comfy_token_weights(text: str) -> list[str]:
    """Port of ComfyUI sd1_clip parse_parentheses/token_weights: the text segments that are tokenized."""
    pieces, current, depth = [], "", 0
    for char in text:
        if char == "(":
            if depth == 0 and current:
                pieces.append(current)
            current = "(" if depth == 0 else current + char
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                pieces.append(current + ")")
                current = ""
            else:
                current += char
        else:
            current += char
    if current:
        pieces.append(current)
    out = []
    for piece in pieces:
        if len(piece) >= 2 and piece[0] == "(" and piece[-1] == ")":
            inner = piece[1:-1]
            colon = inner.rfind(":")
            if colon > 0:
                try:
                    float(inner[colon + 1:])
                    inner = inner[:colon]
                except ValueError:
                    pass
            out += _comfy_token_weights(inner)
        else:
            out.append(piece)
    return out


def _embedding_candidates(text: str) -> list[tuple[str, ...]]:
    """Per embedding word, the names ComfyUI's tokenizer tries to load, in order: the first
    whitespace-delimited word, then its comma-stripped form if that one does not resolve."""
    escaped = text.replace("\\)", "\0\1").replace("\\(", "\0\2")
    words: list[tuple[str, ...]] = []
    for segment in _comfy_token_weights(escaped):
        segment = segment.replace("\0\1", ")").replace("\0\2", "(")
        split = re.split(r"(?<=\s)embedding:", segment)
        parts = [split[0]] + ["embedding:" + rest for rest in split[1:]]
        for word in parts:
            if not word.startswith("embedding:"):
                continue
            tokens = word[len("embedding:"):].strip("\n").split()
            name = tokens[0] if tokens else ""
            stripped = name.strip(",")
            words.append((name,) if stripped == name else (name, stripped))
    return words


def _regeneration_dependencies(graph: dict[str, Any]) -> dict[str, Any]:
    deps: dict[str, Any] = dict(graph_dependencies(graph))
    assets = set(deps["model_assets"])
    embeddings: list[tuple[str, ...]] = []
    for node in graph.values():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, str) and "embedding:" in value:
                if not MODEL_INPUT.search(key):
                    assets.discard(value)
                embeddings += _embedding_candidates(value)
    deps["loader_assets"] = sorted(assets)
    deps["model_assets"] = sorted(assets | {words[0] for words in embeddings})
    deps["embedding_candidates"] = sorted(set(embeddings))
    return deps


EMBEDDING_EXTENSIONS = (".safetensors", ".pt", ".bin")


def _embedding_files(root: Path, parts: tuple[str, ...]) -> tuple[set[str], bool]:
    """Port of ComfyUI load_embed: every directory under the embeddings root, symlinked ones included,
    is searched in no fixed order for the exact name and then the name plus each extension. Names are
    matched lexically, as the loader's suffix dispatch sees them. Returns each directory's pick and
    whether any pick's real target lies outside the root."""
    lexical_root = Path(os.path.abspath(root))
    real_root = lexical_root.resolve()
    found, escaped, visited = set(), False, set()
    for directory, subdirs, _ in os.walk(lexical_root, followlinks=True):
        real = os.path.realpath(directory)
        if real in visited:
            subdirs[:] = []
            continue
        visited.add(real)
        base = Path(directory).joinpath(*parts)
        hit = base if base.is_file() else next(
            (c for c in (base.with_name(base.name + ext) for ext in EMBEDDING_EXTENSIONS) if c.is_file()), None)
        if hit is not None:
            found.add(hit.relative_to(lexical_root).as_posix())
            escaped |= not hit.resolve().is_relative_to(real_root)
    return found, escaped


def _resolve_embeddings(deps: dict[str, Any], models_dir: Path | None) -> str | None:
    """Replaces each embedding word in deps["model_assets"] by the file ComfyUI would load.
    Returns a rejection reason when that file cannot be determined unambiguously."""
    # Loader inputs keep their own role even when an embedding word has the same spelling.
    assets = set(deps.get("loader_assets", deps["model_assets"]))
    embedding_files: set[str] = set()
    for words in deps["embedding_candidates"]:
        if any(_relative_parts(word) is None or "\\" in word for word in words):
            # A backslash is a separator only on Windows runners; on POSIX it is part of the filename.
            return f"embedding name {words[0]!r} is not a contained relative path"
        if any(c in words[0] for c in "<["):
            # ComfyUI releases disagree on whether these characters end the embedding name.
            return f"embedding name {words[0]!r} resolves differently across ComfyUI versions"
        if models_dir is None:
            # /object_info never lists embeddings, so only --models-dir can tell which file is loaded.
            return f"embedding {words[0]!r} needs --models-dir to resolve the file ComfyUI loads"
        # ComfyUI retries the comma-stripped name after any load failure, so every candidate is reachable.
        files: set[str] = set()
        for word in words:
            picks, escaped = _embedding_files(models_dir, _relative_parts(word) or ())
            if escaped:
                return f"embedding name {word!r} resolves outside --models-dir"
            files |= picks
        # Aliases of one file opened by the same loader are one dependency.
        identities = {(os.path.realpath(models_dir / name), _model_format(name)) for name in files}
        if len(identities) > 1:
            return f"embedding name {words[0]!r} may load any of: {', '.join(sorted(files))}"
        if not files:
            return f"embedding {words[0]!r} not found in --models-dir"
        picked = min(files)
        assets.add(picked)
        embedding_files.add(picked)
    deps["embedding_files"] = sorted(embedding_files)
    deps["model_assets"] = sorted(assets)
    return None


def _load_images(graph: dict[str, Any]) -> list[str]:
    loaders = ("LoadImage", "LoadImageMask")
    return sorted({n["inputs"]["image"] for n in graph.values()
                   if n.get("class_type") in loaders and isinstance(n["inputs"].get("image"), str)})


def _links(node: dict[str, Any]) -> list[tuple[str, str]]:
    return [(name, str(value[0])) for name, value in node["inputs"].items()
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)]


def _declared_types(entry: Any) -> dict[str, str]:
    inputs = entry.get("input", {}) if isinstance(entry, dict) else {}
    return {name: spec[0] for section in ("required", "optional") for name, spec in (inputs.get(section) or {}).items()
            if isinstance(spec, list) and spec and isinstance(spec[0], str)}


def _raster_inputs(cls: str, object_info: dict[str, Any]) -> frozenset[str]:
    """IMAGE/LATENT inputs as the runner declares them, else the core table. Undeclared custom inputs
    carry no provenance, so a filename or text edge cannot prove an image was sampled."""
    declared = _declared_types(object_info.get(cls))
    if declared:
        return frozenset(name for name, kind in declared.items() if kind in RASTER_TYPES)
    return CORE_RASTER_INPUTS.get(cls, frozenset())


def _is_image_output(cls: str, object_info: dict[str, Any]) -> bool:
    entry = object_info.get(cls)
    return cls in ("SaveImage", "PreviewImage") or (
        isinstance(entry, dict) and entry.get("output_node") is True and "IMAGE" in _declared_types(entry).values())


def _has_cycle(graph: dict[str, Any]) -> bool:
    state: dict[str, int] = {}
    for start in graph:
        if start in state:
            continue
        stack = [(start, iter(_links(graph[start])))]
        state[start] = 1
        while stack:
            node, edges = stack[-1]
            for _, up in edges:
                if up not in graph:
                    continue
                if state.get(up) == 1:
                    return True
                if up not in state:
                    state[up] = 1
                    stack.append((up, iter(_links(graph[up]))))
                    break
            else:
                state[node] = 2
                stack.pop()
    return False


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sampler_runs(node: dict[str, Any], graph: dict[str, Any]) -> bool:
    """False when a core sampler's literal settings skip sampling. To my understanding ComfyUI then
    returns the input latent unchanged (denoise 0, or KSamplerAdvanced starting at or past its end),
    making an img2img graph a VAE round trip of the input. Modular samplers require an observable schedule."""
    inputs = node["inputs"]
    if node["class_type"] == "KSampler":
        steps, denoise = _number(inputs.get("steps")), _number(inputs.get("denoise", 1.0))
        return steps is not None and denoise is not None and steps >= 1 and denoise > 0
    if node["class_type"] == "KSamplerAdvanced":
        steps = _number(inputs.get("steps"))
        start, end = _number(inputs.get("start_at_step", 0)), _number(inputs.get("end_at_step", 10000))
        if steps is None or start is None or end is None:
            return False
        return steps >= 1 and start < min(end, steps)
    if node["class_type"] in ("SamplerCustom", "SamplerCustomAdvanced"):
        link = inputs.get("sigmas")
        if not isinstance(link, list) or len(link) != 2 or link[1] != 0:
            return False
        scheduler = graph.get(str(link[0]))
        if scheduler is None or scheduler["class_type"] != "BasicScheduler":
            return False
        steps = _number(scheduler["inputs"].get("steps"))
        denoise = _number(scheduler["inputs"].get("denoise"))
        return steps is not None and denoise is not None and steps >= 1 and 0 < denoise <= 1
    return True


def _sampled_nodes(graph: dict[str, Any], samplers: frozenset[str], object_info: dict[str, Any],
                   ) -> tuple[set[str], set[str]]:
    """Return fully sampled nodes and mixed-source composites eligible for reference comparisons only."""
    def raster_slot(node_id: Any, slot: int) -> bool:
        node = graph.get(str(node_id))
        entry = object_info.get(node["class_type"]) if node else None
        outputs = entry.get("output") if isinstance(entry, dict) else None
        if not isinstance(outputs, list):
            return slot >= 0
        return 0 <= slot < len(outputs) and outputs[slot] in RASTER_TYPES

    def raster_edges(node_id: str) -> list[tuple[str, int]]:
        node = graph[node_id]
        wanted = _raster_inputs(node["class_type"], object_info)
        return [(str(value[0]), value[1]) for name, value in node["inputs"].items() if name in wanted
                and isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)]

    sampled: set[str] = set()
    references: set[str] = set()
    visited: set[str] = set()
    for start in graph:
        stack = [start]
        while stack:
            node_id = stack[-1]
            if node_id in visited:
                stack.pop()
                continue
            if graph[node_id]["class_type"] in samplers and _sampler_runs(graph[node_id], graph):
                sampled.add(node_id)
            else:
                edges = raster_edges(node_id)
                pending = [up for up, _ in edges if up in graph and up not in visited]
                if pending:
                    stack.extend(pending)
                    continue
                if edges and all(up in graph and raster_slot(up, slot) for up, slot in edges):
                    upstream = {up for up, _ in edges}
                    if upstream <= sampled:
                        sampled.add(node_id)
                    elif ((len(edges) == 1 and upstream <= references)
                          or (graph[node_id]["class_type"] in ("ImageCompositeMasked", "ImageBlend") and len(edges) == 2
                              and (upstream & sampled or upstream & references))):
                        references.add(node_id)
            visited.add(node_id)
            stack.pop()
    return sampled, references


def _computed_prompt_inputs(graph: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    """Linked STRING inputs of text encoders. Their text, and any embedding: word in it, only exists at
    run time, so preflight cannot audit the embedding files it loads."""
    found = []
    for node_id, node in graph.items():
        declared = _declared_types(object_info.get(node["class_type"]))
        if "CLIP" not in declared.values() and node["class_type"] != "CLIPTextEncode":
            continue
        strings = {name for name, kind in declared.items() if kind == "STRING"} or {"text"}
        found += [f"{node_id}.{name}" for name, _ in _links(node) if name in strings]
    return sorted(found)


class PreflightChecker:
    def __init__(self, runner_url: str, allowed_nodes: list[str], allow_pickle: bool,
                 models_dir: Path | None, inputs_dir: Path | None = None, sampler_nodes: Iterable[str] = (), *,
                 model_identity: dict[str, Any] | None = None) -> None:
        self.runner_url = runner_url.rstrip("/")
        self.model_identity = {m["name"]: m for m in (model_identity or {}).get("models", [])}
        self.samplers = frozenset(SAMPLER_CLASSES) | frozenset(sampler_nodes)
        self.allowed_nodes = sorted(set(allowed_nodes) | set(sampler_nodes))
        self.allow_pickle = allow_pickle
        self.models_dir = models_dir
        self.inputs_dir = inputs_dir

    def audit(self, workflow: WorkflowRecord) -> dict[str, Any]:
        deps = _regeneration_dependencies(workflow.graph) if workflow.prompt_format == "API_GRAPH" else {
            "node_classes": [], "model_assets": []}
        formats = {name: _model_format(name) for name in deps["model_assets"]}
        return {
            "security_audit": {
                "allowed_custom_nodes": self.allowed_nodes,
                "model_format": _summarize_formats(formats),
                "pickle_models_permitted": self.allow_pickle,
            },
            "preflight_check": {
                "status": "NOT_RUN",
                "runner_object_info_verified": False,
                "model_match_level": "FILENAME_ONLY",
                "missing_node_classes": [],
                "missing_model_assets": [],
            },
        }

    def validate(self, workflow: WorkflowRecord) -> tuple[bool, str, dict[str, Any]]:
        """Default-deny checks for API format, nodes, models, and conditioning inputs.
        Returns (passed, status, audit). Raises: RunnerConnectionError if the runner is unreachable."""
        graph = workflow.graph
        deps = _regeneration_dependencies(graph) if workflow.prompt_format == "API_GRAPH" else {
            "node_classes": [], "model_assets": [], "embedding_candidates": []}
        audit = self.audit(workflow)
        check = audit["preflight_check"]

        def fail(status: str, reason: str) -> tuple[bool, str, dict[str, Any]]:
            check["status"] = status
            check["reason"] = reason
            return False, status, audit

        if workflow.prompt_format != "API_GRAPH":
            return fail(UNAVAILABLE, f"{workflow.prompt_format} is not an API prompt graph; ComfyTrace does not "
                                     "bundle a UI-to-API compiler")
        ambiguous = _resolve_embeddings(deps, self.models_dir)
        if ambiguous:
            return fail(SECURITY_REJECTED, ambiguous)
        formats = {name: _model_format(name) for name in deps["model_assets"]}
        audit["security_audit"]["model_format"] = _summarize_formats(formats)
        unvetted = sorted(set(deps["node_classes"]) - CORE_NODES - set(self.allowed_nodes))
        if unvetted:
            check["unvetted_node_classes"] = unvetted
            return fail(SECURITY_REJECTED, f"custom node classes not authorized: {', '.join(unvetted)}")
        if any(node["class_type"] == "LoadLatent" and not _safe_filename(node["inputs"].get("latent"))
               for node in graph.values()):
            return fail(SECURITY_REJECTED, "latent names must be safe single filenames without directory annotations")
        if _has_cycle(graph):
            return fail(SECURITY_REJECTED, "workflow links form a cycle")
        model_parts = {name: _relative_parts(name) for name in deps["model_assets"]}
        if any(parts is None for parts in model_parts.values()):
            return fail(SECURITY_REJECTED, "model names must be contained relative paths")
        pickles = sorted(n for n, f in formats.items() if f == "PICKLE")
        unknown = sorted(n for n, f in formats.items() if f == "UNKNOWN")
        if unknown:
            return fail(SECURITY_REJECTED, f"unrecognized model formats: {', '.join(unknown)}")
        if pickles and not self.allow_pickle:
            return fail(SECURITY_REJECTED, f"pickle checkpoints require --allow-pickle-models: {', '.join(pickles)}")
        if pickles:
            check["forensic_warning"] = ("pickle checkpoints executed by investigator authorization: "
                                         + ", ".join(pickles))

        needed_inputs = _load_images(graph)
        escaping = [n for n in needed_inputs if not _safe_filename(n) or (
            self.inputs_dir is not None and input_path(self.inputs_dir, n) is None)]
        if escaping:
            return fail(SECURITY_REJECTED, "input image names escape --inputs-dir or contain separators: "
                        + ", ".join(escaping))

        object_info = RunnerClient(self.runner_url).object_info()
        check["runner_object_info_verified"] = True
        check["missing_node_classes"] = sorted(set(deps["node_classes"]) - set(object_info))
        # Embeddings are never listed in /object_info; they were resolved and hashed under --models-dir.
        check["missing_model_assets"] = sorted(set(deps.get("loader_assets", deps["model_assets"]))
                                               - _registered_choices(object_info))
        if check["missing_node_classes"] or check["missing_model_assets"]:
            return fail(MISSING_DEPENDENCY, "runner does not register every required node class and model")
        outputs = [n for n, node in graph.items() if _is_image_output(node["class_type"], object_info)]
        computed = _computed_prompt_inputs(graph, object_info)
        if computed:
            return fail(SECURITY_REJECTED, "prompt text computed at run time cannot be audited for embeddings: "
                        + ", ".join(computed))
        sampled_nodes, reference_nodes = _sampled_nodes(graph, self.samplers, object_info)
        edit_nodes = image_edit_outputs(graph) - sampled_nodes
        reference_nodes.update(edit_nodes)
        compared = sorted(n for n in outputs if n in sampled_nodes or n in reference_nodes)
        if outputs and not compared:
            return fail(SECURITY_REJECTED, "no image output node has a sampler ancestor")
        check["compared_output_nodes"] = compared
        check["comparison_only_output_nodes"] = sorted(n for n in outputs if n in reference_nodes)
        if edit_nodes:
            check["image_edit_output_nodes"] = sorted(edit_nodes)

        if self.models_dir is not None:
            hashes = {}
            for name, parts in model_parts.items():
                local = _contained_path(self.models_dir, parts or ())
                if local is None:
                    return fail(SECURITY_REJECTED, f"model {name!r} escapes --models-dir")
                if not local.is_file():
                    check["missing_model_assets"] = [name]
                    return fail(MISSING_DEPENDENCY, f"model {name!r} missing from --models-dir")
                hashes[name] = sha256_file(local)
            check["model_sha256"] = hashes
            if hashes:
                check["model_match_level"] = "HASHED"
                seen = {name: self.model_identity.get(name, {}) for name in hashes}
                check["model_memory_identity"] = {name: m.get("verdict", "NOT_ASSESSED") for name, m in seen.items()}
                # Only weights observed in this dump's ComfyUI process memory make the local file the one it used.
                if all(m.get("verdict") == "OBSERVED" and m.get("sha256") == hashes[name] for name, m in seen.items()):
                    check["model_match_level"] = "OBSERVED_IN_MEMORY"

        if needed_inputs:
            missing = [n for n in needed_inputs if self.inputs_dir is None or not (self.inputs_dir / n).is_file()]
            if missing:
                return fail(UNAVAILABLE, "Missing source input images: " + ", ".join(missing))
            check["input_images"] = {n: sha256_file(self.inputs_dir / n) for n in needed_inputs}  # type: ignore[operator]
            check["input_image_pixel_sha256"] = {}
            for name in needed_inputs:
                try:
                    digest, _ = native_pixel_digest((self.inputs_dir / name).read_bytes())  # type: ignore[operator]
                except (OSError, ValueError, SyntaxError):
                    continue
                check["input_image_pixel_sha256"][name] = digest
        check["status"] = PASSED
        return True, PASSED, audit


def _summarize_formats(formats: dict[str, str]) -> str:
    kinds = set(formats.values())
    if not kinds:
        return "NONE"
    return kinds.pop() if len(kinds) == 1 else "MIXED"


class RunnerClient:
    def __init__(self, runner_url: str, timeout: int = 300, poll_interval: float = 1.0) -> None:
        self.runner_url = runner_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import requests

        try:
            default_timeout = min(self.timeout, 30) if self.timeout > 0 else 30
            response = requests.request(
                method, self.runner_url + path, timeout=kwargs.pop("timeout", default_timeout), **kwargs,
            )
        except requests.RequestException as exc:
            raise RunnerConnectionError(f"runner unreachable at {self.runner_url}: {exc}") from exc
        return response

    def object_info(self) -> dict[str, Any]:
        response = self._request("GET", "/object_info")
        if response.status_code != 200:
            raise RunnerConnectionError(f"/object_info returned HTTP {response.status_code}")
        return dict(response.json())

    def environment(self) -> dict[str, str]:
        """Runner environment as the runner reports it. Fields it does not report are UNREPORTED."""
        env = dict.fromkeys(("comfyui_version", "python_version", "pytorch_version", "cuda_driver_version",
                             "gpu_hardware"), "UNREPORTED")
        try:
            response = self._request("GET", "/system_stats")
            stats = response.json() if response.status_code == 200 else {}
        except (RunnerConnectionError, ValueError):
            return env
        system = stats.get("system", {})
        for key in ("comfyui_version", "python_version", "pytorch_version", "cuda_driver_version"):
            if system.get(key):
                env[key] = str(system[key]).split(" ")[0]
        devices = stats.get("devices") or []
        if devices and devices[0].get("name"):
            env["gpu_hardware"] = str(devices[0]["name"])
        return env

    def upload_input(self, path: Path, name: str) -> str:
        """Uploads a conditioning image under a validated filename with no subfolder.
        Returns the name the runner stored it under, which the graph must then reference."""
        if not _safe_filename(name):
            raise PromptSubmissionError(f"unsafe input image name: {name!r}")
        with open(path, "rb") as handle:
            response = self._request("POST", "/upload/image", files={"image": (name, handle, "image/png")},
                                     data={"overwrite": "true", "subfolder": "", "type": "input"})
        if response.status_code != 200:
            raise PromptSubmissionError(f"input upload of {name} rejected: HTTP {response.status_code}")
        try:
            stored = response.json()
        except ValueError as exc:
            raise PromptSubmissionError("input upload returned invalid JSON") from exc
        if (not isinstance(stored, dict) or not _safe_filename(stored.get("name"))
                or stored.get("name") != name
                or stored.get("subfolder", "") != "" or stored.get("type", "input") != "input"):
            raise PromptSubmissionError("input upload returned an unsafe filename, subfolder, or type")
        return str(stored["name"])

    def queue_prompt(self, api_prompt_graph: dict[str, Any]) -> str:
        """Submits the API prompt graph to /prompt and returns the runner's prompt_id.
        Raises: RunnerConnectionError if unreachable; PromptSubmissionError if rejected."""
        response = self._request("POST", "/prompt", json={"prompt": api_prompt_graph, "client_id": "comfytrace"})
        if response.status_code != 200:
            detail = response.text[:500]
            raise PromptSubmissionError(f"/prompt rejected the graph: HTTP {response.status_code} {detail}")
        prompt_id = response.json().get("prompt_id")
        if not isinstance(prompt_id, str):
            raise PromptSubmissionError("/prompt response carried no prompt_id")
        return prompt_id

    def retrieve_output_rasters(self, prompt_id: str, nodes: Iterable[str] | None = None,
                                limit: int = MAX_COMPARED_IMAGES) -> list[tuple[dict[str, Any], bytes]]:
        """Polls /history/{prompt_id} and downloads every output image of `nodes` (any node when None)
        through /view, saved outputs before previews, at most `limit`. Each comes with its node id,
        batch index, filename and type.
        Raises: RunnerConnectionError if unreachable; ExecutionTimeoutError past the timeout."""
        deadline = time.monotonic() + self.timeout

        def remaining() -> float:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise ExecutionTimeoutError(f"no output for {prompt_id} within {self.timeout}s")
            return min(30.0, budget)

        while True:
            response = self._request("GET", f"/history/{prompt_id}", timeout=remaining())
            remaining()
            entry = (response.json() if response.status_code == 200 else {}).get(prompt_id)
            status = (entry or {}).get("status", {})
            messages = status.get("messages") or []
            if (status.get("status_str") in ("error", "errored", "failed", "failure", "cancelled", "interrupted")
                    or any(m and m[0] in ("execution_error", "execution_interrupted") for m in messages)):
                raise PromptSubmissionError(f"runner reported an execution error for {prompt_id}")
            if status.get("completed") is True:
                # PreviewImage results are "temp"; "input" images were never produced by this run.
                wanted = None if nodes is None else set(nodes)
                images = [(node_id, index, img) for node_id, out in (entry or {}).get("outputs", {}).items()
                          if wanted is None or node_id in wanted
                          for index, img in enumerate(out.get("images", [])) if img.get("type") in ("output", "temp")]
                if not images:
                    raise PromptSubmissionError(f"runner completed {prompt_id} without an output image")
                images.sort(key=lambda item: item[2]["type"] != "output")
                rasters = []
                for node_id, index, image in images[:limit]:
                    name, subfolder = image.get("filename"), image.get("subfolder", "")
                    if (not _safe_filename(name) or not isinstance(subfolder, str) or "\\" in subfolder
                            or (subfolder and _relative_parts(subfolder) is None)):
                        raise PromptSubmissionError("runner returned an unsafe output filename or subfolder")
                    params = {"filename": name, "subfolder": subfolder, "type": image["type"]}
                    view = self._request("GET", "/view", params=params, timeout=remaining())
                    remaining()
                    if view.status_code != 200:
                        raise RunnerConnectionError(f"/view returned HTTP {view.status_code}")
                    rasters.append(({"node_id": node_id, "batch_index": index, **params}, bytes(view.content)))
                return rasters
            time.sleep(min(self.poll_interval, remaining()))


def _perceptual_backends() -> tuple[Any, Any]:
    try:
        import imagehash
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise MissingDependencyError(
            "perceptual comparison needs the [regen] extra: pip install 'comfytrace[regen]'"
        ) from exc
    return imagehash, structural_similarity


INCONCLUSIVE = "INCONCLUSIVE"
INSUFFICIENT_ROWS = "INSUFFICIENT_ROWS"
INSUFFICIENT_DIMENSIONS = "INSUFFICIENT_DIMENSIONS"
WINDOW_TOO_SMALL = "window_too_small"
NO_DECODED_ROWS = "no_decoded_rows"
DIFFERENCE_TOLERANCE = 2
"""Levels on an 8-bit channel above which a pixel counts as different in ``fraction_above_tolerance``."""
MIN_SSIM_WINDOW = 3


def _ssim_window(image: Any) -> int:
    """The SSIM window an image admits: odd, at most 7, at most its shorter side. Below 3 there is no SSIM."""
    side = min(image.width, image.height)
    return min(7, side if side % 2 else side - 1)


def _pixel_differences(a: Any, b: Any) -> dict[str, Any]:
    """Descriptive differences of two equally sized rasters on 8-bit RGBA channels. No similarity claim."""
    import numpy as np

    diff = np.abs(np.asarray(a.convert("RGBA"), dtype=np.int16) - np.asarray(b.convert("RGBA"), dtype=np.int16))
    if diff.size == 0:
        return {"mean_abs_diff": None, "max_abs_diff": None, "fraction_above_tolerance": None,
                "difference_tolerance": DIFFERENCE_TOLERANCE}
    return {"mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()),
            "fraction_above_tolerance": float((diff > DIFFERENCE_TOLERANCE).any(axis=-1).mean()),
            "difference_tolerance": DIFFERENCE_TOLERANCE}


class TieredEvaluator:
    def __init__(self, ssim_threshold: float = 0.95, phash_threshold: int = 4) -> None:
        self.ssim_threshold = ssim_threshold
        self.phash_threshold = phash_threshold

    def _perceptual(self, a: Any, b: Any) -> tuple[float | None, int]:
        """SSIM and pHash distance. SSIM is None when the rasters are too small for an SSIM window (fewer than
        3 pixels on a side): a 1-2 row strip has no structure to compare, and no score is made up for it."""
        import numpy as np

        imagehash, structural_similarity = _perceptual_backends()
        win = _ssim_window(a)
        ssim: float | None = None
        if win >= MIN_SSIM_WINDOW:
            ssim = float(structural_similarity(np.asarray(a.convert("RGBA")), np.asarray(b.convert("RGBA")),
                                               channel_axis=2, data_range=255, win_size=win))
        distance = int(imagehash.phash(a, hash_size=8) - imagehash.phash(b, hash_size=8))
        return ssim, distance

    def _scores(self, metrics: dict[str, Any], a: Any, b: Any, exact: bool) -> None:
        """Fills ssim_score and phash_hamming_distance. Missing [regen] extras only matter when a perceptual
        score could decide the outcome: not for an exact match, and not below the SSIM window."""
        try:
            metrics["ssim_score"], metrics["phash_hamming_distance"] = self._perceptual(a, b)
        except MissingDependencyError:
            if not exact and _ssim_window(a) >= MIN_SSIM_WINDOW:
                raise
            metrics["perceptual_scores"] = "UNAVAILABLE: [regen] extra not installed"

    def _perceptual_match(self, metrics: dict[str, Any]) -> bool:
        ssim, phash = metrics.get("ssim_score"), metrics.get("phash_hamming_distance")
        return (ssim is not None and phash is not None
                and ssim >= self.ssim_threshold and phash <= self.phash_threshold)

    @staticmethod
    def _inconclusive(metrics: dict[str, Any], a: Any, b: Any, reason: str = WINDOW_TOO_SMALL) -> None:
        metrics["ssim_score"] = None
        metrics["ssim_unavailable_reason"] = reason
        metrics.update(_pixel_differences(a, b))

    def evaluate_reproduction(self, regen_bytes: bytes, target_image: RawArtifactRecord | None,
                              target_png: bytes | None = None) -> tuple[str, str | None, dict[str, Any]]:
        """Compares a regenerated raster with a memory-recovered target at native resolution.
        `target_png` holds the carved PNG for an intact target, or the DERIVED row preview for a partial one.
        Returns (evaluated_tier: E1 | E4 | PARTIAL_CORROBORATION | NON_MATCHING, subtier, metrics).
        Raises: MissingDependencyError when perceptual comparison is needed without the [regen] extra."""
        from PIL import Image

        regen = Image.open(io.BytesIO(regen_bytes))
        if regen.format != "PNG":
            raise RegenerationError(f"runner output must be PNG, received {regen.format}")
        regen.load()
        if target_image is None:
            return "E4", "EXECUTABLE_REPRODUCTION", {
                "evaluation_mode": "NO_MEMORY_TARGET",
                "regenerated_dimensions": {"width": regen.width, "height": regen.height, "mode": regen.mode},
            }
        if target_png is None:
            raise RegenerationError(f"no target raster supplied for {target_image.artifact_id}")
        target = Image.open(io.BytesIO(target_png))
        target.load()
        if target_image.completeness == Completeness.COMPLETE:
            if (any(getattr(image, "n_frames", 1) > 1 or getattr(image, "custom_mimetype", None) == "image/apng"
                    for image in (target, regen)) or (target_image.pixel_metadata or {}).get("animated")):
                # An assessability limit, not a pixel contradiction: nothing was compared.
                return "NON_MATCHING", "ANIMATED_COMPARISON_UNAVAILABLE", {
                    "evaluation_mode": "INTACT_NATIVE_RESOLUTION", "exact_pixel_match": False,
                    "comparison_unavailable_reason": "animated_sequence",
                    "reason": "An animation requires a comparison of the entire sequence.",
                }
            native_match = None
            if ((target.format == "PNG" and (target.mode == "P" or target_png[24] == 16))
                    or regen.mode == "P" or regen_bytes[24] == 16):
                try:
                    original, original_meta = native_pixel_digest(target_png)
                    regenerated, regen_meta = native_pixel_digest(regen_bytes)
                    native_match = original == regenerated == target_image.pixel_sha256 and original_meta == regen_meta
                except (OSError, ValueError, SyntaxError):
                    native_match = False
            return self.evaluate_intact(regen, target, target_image.pixel_sha256, native_match=native_match)
        return self._partial(regen, target, target_image)

    def evaluate_intact(
        self, regen: Any, target: Any, pixel_sha256: str | None, *, native_match: bool | None = None,
    ) -> tuple[str, str | None, dict[str, Any]]:
        metrics: dict[str, Any] = {"evaluation_mode": "INTACT_NATIVE_RESOLUTION", "mode_conversion": None,
                                   "exact_pixel_match": False, "ssim_score": None, "phash_hamming_distance": None}
        if regen.size != target.size:
            metrics["dimension_mismatch"] = {"regenerated": list(regen.size), "target": list(target.size)}
            return "NON_MATCHING", "DIMENSION_MISMATCH", metrics
        if regen.mode != target.mode:
            metrics["mode_conversion"] = f"{regen.mode}->{target.mode}"
            regen = regen.convert(target.mode)
        native_equal = (hashlib.sha256(regen.tobytes()).hexdigest() == pixel_sha256
                        if native_match is None else native_match)
        rendered_equal = regen.convert("RGBA").tobytes() == target.convert("RGBA").tobytes()
        metrics["native_pixel_sha256_match"] = native_equal
        metrics["exact_pixel_match"] = native_equal and rendered_equal
        self._scores(metrics, regen, target, metrics["exact_pixel_match"])
        if metrics["exact_pixel_match"]:
            return "E1", "EXACT_MATCH", metrics
        if _ssim_window(regen) < MIN_SSIM_WINDOW:
            self._inconclusive(metrics, regen, target)
            return INCONCLUSIVE, INSUFFICIENT_DIMENSIONS, metrics
        if self._perceptual_match(metrics):
            return "E1", "PERCEPTUAL_MATCH", metrics
        return "NON_MATCHING", "BELOW_THRESHOLDS", metrics

    def _partial(self, regen: Any, preview: Any, record: RawArtifactRecord, *, exact_only: bool = False,
                 ) -> tuple[str, str | None, dict[str, Any]]:
        metadata = record.derived_preview_metadata or {}
        declared = metadata.get("decoded_rows")
        rows = int(declared) if declared is not None else preview.height
        metrics: dict[str, Any] = {"evaluation_mode": "PARTIAL_ROW_BOUNDED", "decoded_rows_evaluated": rows,
                                   "cropped_dimensions": {"width": regen.width, "height": rows},
                                   "row_slice_pixel_match": False, "exact_pixel_match": False,
                                   "ssim_score": None, "phash_hamming_distance": None, "mode_conversion": None}
        expected_width = record.dimensions[0] if record.dimensions else preview.width
        if regen.width != expected_width or regen.height < rows:
            metrics["dimension_mismatch"] = {"regenerated": list(regen.size), "target_width": expected_width,
                                             "decoded_rows": rows}
            return "NON_MATCHING", "DIMENSION_MISMATCH", metrics
        if rows <= 0:
            # No decoded pixels: nothing was compared, so nothing contradicts the replay either.
            metrics["ssim_unavailable_reason"] = NO_DECODED_ROWS
            return INCONCLUSIVE, INSUFFICIENT_ROWS, metrics
        cropped = regen.crop((0, 0, regen.width, rows))
        if cropped.mode != preview.mode:
            metrics["mode_conversion"] = f"{cropped.mode}->{preview.mode}"
            cropped = cropped.convert(preview.mode)
        preview = preview.crop((0, 0, preview.width, rows))
        native_rows = not (metadata.get("sample_precision_reduced") or metadata.get("transparency_dropped"))
        metrics["row_slice_pixel_match"] = native_rows and cropped.tobytes() == preview.tobytes()
        if exact_only:
            # Byte equality needs no SSIM window: unequal bytes differ however few rows there are.
            return (("PARTIAL_CORROBORATION", "ROW_SLICE_EXACT", metrics) if metrics["row_slice_pixel_match"]
                    else ("NON_MATCHING", "BELOW_THRESHOLDS", metrics))
        self._scores(metrics, cropped, preview, metrics["row_slice_pixel_match"])
        if metrics["row_slice_pixel_match"]:
            return "PARTIAL_CORROBORATION", "ROW_SLICE_EXACT", metrics
        if _ssim_window(cropped) < MIN_SSIM_WINDOW:
            # A 1-2 row strip that is not byte-exact can neither corroborate nor contradict perceptually.
            self._inconclusive(metrics, cropped, preview)
            return INCONCLUSIVE, INSUFFICIENT_ROWS, metrics
        if self._perceptual_match(metrics):
            return "PARTIAL_CORROBORATION", "ROW_SLICE_PERCEPTUAL", metrics
        return "NON_MATCHING", "BELOW_THRESHOLDS", metrics

    def compare_partial_inputs(
        self, target: RawArtifactRecord, preview_png: bytes, inputs_dir: Path,
        expected: dict[str, str], *, exact_only: bool,
    ) -> dict[str, Any]:
        """Check whether the target's readable rows also match the supplied input images."""
        from PIL import Image

        metadata = target.derived_preview_metadata or {}
        if expected and (metadata.get("sample_precision_reduced") or metadata.get("transparency_dropped")):
            return {"status": "UNVERIFIED", "verified_inputs": {}}
        verified = {}
        inconclusive = []
        with Image.open(io.BytesIO(preview_png)) as preview:
            for name, digest in expected.items():
                local = input_path(inputs_dir, name)
                try:
                    data = local.read_bytes() if local is not None else b""
                    if hashlib.sha256(data).hexdigest() != digest:
                        continue
                    with Image.open(io.BytesIO(data)) as image:
                        evaluated, _, metrics = self._partial(image, preview, target, exact_only=exact_only)
                except (OSError, MissingDependencyError):
                    continue
                if metrics.get("dimension_mismatch"):
                    continue
                verified[name] = digest
                if evaluated == "PARTIAL_CORROBORATION":
                    return {"status": "MATCHES_INPUT", "verified_inputs": verified, "matching_input": name}
                if evaluated == INCONCLUSIVE:
                    inconclusive.append(name)
        result: dict[str, Any] = {
            "status": "DISTINCT" if verified == expected and not inconclusive else "UNVERIFIED",
            "verified_inputs": verified,
        }
        if inconclusive:
            # An input the rows cannot be told apart from never counts as distinct.
            result["inconclusive_inputs"] = sorted(inconclusive)
        return result


def load_workflow_file(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    data = path.read_bytes()
    # utf-8-sig also reads plain UTF-8; PowerShell 5.1's ``-Encoding utf8`` writes the byte-order mark.
    for codec in ("utf-8-sig", "utf-16-le", "utf-16"):
        try:
            obj = json.loads(data.decode(codec))
            if isinstance(obj, str):
                obj = json.loads(obj)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        classified = extract_prompt_graph(obj)
        if classified is not None:
            return classified
    raise RegenerationError(f"{path} is not a ComfyUI prompt graph in UTF-8 or UTF-16")


def _finding_for(manifest: dict[str, Any], ids: list[str]) -> dict[str, Any] | None:
    for finding in manifest["findings"]["findings"]:
        if set(ids).issubset(finding["artifact_ids"]):
            return finding
    return None


def run_regeneration(args: argparse.Namespace) -> int:
    case_dir: Path = args.case_dir
    manifest_path = case_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    fmt, graph, runtime = load_workflow_file(args.workflow)

    from comfytrace.carve import canonical_json

    digest = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
    workflows = manifest["findings"]["workflows"]
    wanted = getattr(args, "workflow_id", None)
    wf_dict = next((w for w in workflows if w["canonical_json_sha256"] == digest
                    and wanted in (None, w["workflow_id"])), None)
    if wf_dict is None:
        raise RegenerationError("workflow is not a recovered workflow of this case (no canonical hash match)")
    workflow = WorkflowRecord.from_dict(wf_dict)
    workflow.graph = graph
    workflow.prompt_format = fmt

    target: RawArtifactRecord | None = None
    target_png: bytes | None = None
    if args.compare_to is not None:
        target, target_png = _resolve_target(case_dir, manifest, Path(args.compare_to))

    runner_url = resolve_runner(args)
    handoff_id = next_regeneration_id(case_dir)
    checker = PreflightChecker(runner_url, args.allow_node, args.allow_pickle_models, args.models_dir,
                               args.inputs_dir, args.allow_sampler_node,
                               model_identity=_model_identity(case_dir, manifest))
    audit = checker.audit(workflow)
    record = RegenerationRecord(
        handoff_id=handoff_id, workflow_id=workflow.workflow_id, target_manifest_sha256=manifest_sha,
        status="FAILED", security_audit=audit["security_audit"], preflight_check=audit["preflight_check"],
        runner_environment=dict.fromkeys(("comfyui_version", "python_version", "pytorch_version",
                                          "cuda_driver_version", "gpu_hardware"), "UNREPORTED"),
        evaluation=_evaluation_block(manifest, workflow, target, "NOT_EVALUATED", None,
                                     {"evaluation_mode": "NOT_EVALUATED"}, args.ssim_threshold, args.phash_threshold),
        derived_regenerated_image=None,
    )
    _record_target_selection(record.evaluation, args)
    client = RunnerClient(runner_url, timeout=args.timeout)
    try:
        passed, status, audit = checker.validate(workflow)
        record.security_audit, record.preflight_check = audit["security_audit"], audit["preflight_check"]
        if not passed:
            record.status = status
            save_regeneration_record(case_dir, record)
            log.error("preflight %s: %s", status, record.preflight_check.get("reason"))
            return 1
        record.runner_environment = client.environment()
        submitted = json.loads(json.dumps(graph))
        uploads = {}
        for name in _load_images(graph):
            local = input_path(args.inputs_dir, name)
            if local is None:
                raise RegenerationError(f"input {name!r} escapes --inputs-dir")
            uploads[name] = client.upload_input(local, name)
        for node in submitted.values():
            image = node["inputs"].get("image")
            if node.get("class_type") in ("LoadImage", "LoadImageMask") and isinstance(image, str) and image in uploads:
                node["inputs"]["image"] = uploads[node["inputs"]["image"]]
        if uploads:
            record.preflight_check["input_upload_names"] = uploads
        rasters = client.retrieve_output_rasters(client.queue_prompt(submitted),
                                                 record.preflight_check.get("compared_output_nodes"))
        evaluator = TieredEvaluator(args.ssim_threshold, args.phash_threshold)
        # An inconclusive comparison says less than a match and more than a contradiction.
        rank = {"EXACT_MATCH": 0, "PERCEPTUAL_MATCH": 1, "ROW_SLICE_EXACT": 2,
                "ROW_SLICE_PERCEPTUAL": 3, "EXECUTABLE_REPRODUCTION": 4,
                INSUFFICIENT_ROWS: 5, INSUFFICIENT_DIMENSIONS: 5}
        scored = []
        missing_perceptual = None
        for index, (_, data) in enumerate(rasters):
            try:
                scored.append((index, *evaluator.evaluate_reproduction(data, target, target_png)))
            except MissingDependencyError as exc:
                missing_perceptual = exc
        if missing_perceptual and not any(score[2] in ("EXACT_MATCH", "ROW_SLICE_EXACT") for score in scored):
            raise missing_perceptual
        references = set(record.preflight_check.get("comparison_only_output_nodes", []))
        selected, evaluated, subtier, metrics = min(
            scored, key=lambda item: (rank.get(item[2] or "", 6), rasters[item[0]][0].get("node_id") in references),
        )
        meta = rasters[selected][0]
        if meta.get("node_id") in references and evaluated not in ("NON_MATCHING", INCONCLUSIVE):
            evaluated = "REFERENCE_COMPARISON"
        record.evaluation = _evaluation_block(manifest, workflow, target, evaluated, subtier, metrics,
                                              args.ssim_threshold, args.phash_threshold)
        _record_target_selection(record.evaluation, args)
        if evaluated == "REFERENCE_COMPARISON":
            if meta.get("node_id") in record.preflight_check.get("image_edit_output_nodes", ()):
                record.evaluation["note"] = "The replay reproduced an image edit and provides a reference raster."
            else:
                record.evaluation["note"] = "The sampled image was combined with other raster inputs before saving."
        record.evaluation.update(compared_image=meta, compared_image_count=len(rasters))
        input_hashes = record.preflight_check.get("input_images") or {}
        if evaluated == "PARTIAL_CORROBORATION" and input_hashes and target is not None and target_png is not None:
            record.evaluation["input_comparison"] = evaluator.compare_partial_inputs(
                target, target_png, args.inputs_dir, input_hashes, exact_only=subtier == "ROW_SLICE_EXACT",
            )
        extra_paths = getattr(args, "extra_compare_to", None) or []
        if extra_paths:
            candidate_evals: dict[str, dict[str, Any]] = {}
            for extra_path in extra_paths:
                try:
                    ex_target, ex_png = _resolve_target(case_dir, manifest, Path(extra_path))
                    if ex_target.artifact_id == (target.artifact_id if target else None):
                        continue
                    ex_scored = [(idx, *evaluator.evaluate_reproduction(data, ex_target, ex_png))
                                 for idx, (_, data) in enumerate(rasters)]
                    ex_sel, ex_eval, ex_sub, ex_metrics = min(
                        ex_scored,
                        key=lambda item: (rank.get(item[2] or "", 6), rasters[item[0]][0].get("node_id") in references),
                    )
                    ex_meta = rasters[ex_sel][0]
                    if ex_meta.get("node_id") in references and ex_eval not in ("NON_MATCHING", INCONCLUSIVE):
                        ex_eval = "REFERENCE_COMPARISON"
                    ex_block = _evaluation_block(manifest, workflow, ex_target, ex_eval, ex_sub, ex_metrics,
                                                 args.ssim_threshold, args.phash_threshold)
                    ex_block.update(compared_image=ex_meta, compared_image_count=len(rasters))
                    if ex_eval == "PARTIAL_CORROBORATION" and input_hashes and ex_png is not None:
                        ex_block["input_comparison"] = evaluator.compare_partial_inputs(
                            ex_target, ex_png, args.inputs_dir, input_hashes, exact_only=ex_sub == "ROW_SLICE_EXACT",
                        )
                    candidate_evals[ex_target.artifact_id] = ex_block
                except Exception as exc:  # noqa: BLE001
                    log.warning("additional candidate comparison for %s skipped: %s", extra_path, exc)
            if candidate_evals:
                record.evaluation["candidate_evaluations"] = candidate_evals
        if target is not None and evaluated == "E1" and target_is_input(record.preflight_check, target.to_dict()):
            record.evaluation.update(promoted_finding_tier=None,
                                     target_finding_tier_retained=(_finding_for(
                                         manifest, [target.artifact_id, workflow.workflow_id]) or {}).get("tier"),
                                     note="Target image was supplied as a conditioning input; E1 promotion withheld.")
        label = f"{evaluated}_" if evaluated in ("NON_MATCHING", INCONCLUSIVE) else ""
        name = f"regen_{label}{workflow.workflow_id}.png"
        if (case_dir / "exports" / "derived" / f"DERIVED_{name}").exists():
            name = f"regen_{label}{workflow.workflow_id}_{handoff_id}.png"
        exporter = EvidentiaryExporter(case_dir)
        for index, (output, data) in enumerate(rasters):
            filename = name if index == selected else f"regen_{workflow.workflow_id}_{handoff_id}_output_{index}.png"
            raster_path = exporter.export_derived_preview(filename, data)
            raster = {"file": raster_path, "sha256": hashlib.sha256(data).hexdigest(), "handoff_id": handoff_id,
                      "node_id": output.get("node_id"), "batch_index": output.get("batch_index")}
            record.output_rasters.append(raster)
        primary = record.output_rasters[selected]
        record.derived_regenerated_image = primary["file"]
        record.evaluation.update(selected_raster=primary, derived_regenerated_image_sha256=primary["sha256"])
    except Exception as exc:
        record.status = "FAILED"
        if record.preflight_check["status"] == "NOT_RUN":
            record.preflight_check.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        record.evaluation["error"] = f"{type(exc).__name__}: {exc}"
        record.evaluation["promoted_finding_tier"] = None
        save_regeneration_record(case_dir, record)
        raise
    record.status = "COMPLETED"
    path = save_regeneration_record(case_dir, record)
    log.info("%s: %s (%s); record %s", workflow.workflow_id, evaluated, subtier, path)
    log.info("re-render the report with: comfytrace report --case-dir %s", case_dir)
    return 0 if record.evaluation["promoted_finding_tier"] in ("E1", "E4") else 2


def _record_target_selection(evaluation: dict[str, Any], args: argparse.Namespace) -> None:
    """How replay-case chose the comparison target (candidates, exclusions, reason), when it chose one."""
    selection = getattr(args, "target_selection", None)
    if selection is not None:
        evaluation["target_selection"] = json.loads(json.dumps(selection))


def _model_identity(case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """model-identity.json, when it was computed from this case's own memory image."""
    path = case_dir / "model-identity.json"
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if document.get("memory_sha256") != manifest["case_metadata"]["memory_image"]["sha256"]:
        log.warning("%s was computed from another memory image; ignored", path)
        return None
    return document


def recovered_prompts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Complete API prompts decoded from CPython objects, one per backend job."""
    return [w for w in manifest["findings"]["workflows"]
            if w["encoding_detected"] == "CPYTHON_OBJECTS" and w["prompt_format"] == "API_GRAPH"
            and w["completeness"] == Completeness.COMPLETE]


def replayable_prompts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every complete API graph a replay can run: the backend prompts of ``recovered_prompts`` first, then complete
    API graphs recovered elsewhere (a PNG's text chunk, carved JSON) whose canonical graph none of those already
    covers. The latter may carry no original prompt ID; their replays can still show where the pixels lie, but
    they name no producing submission."""
    out = [w for w in recovered_prompts(manifest) if w.get("raw_file_path")]
    seen = {w.get("canonical_json_sha256") for w in out} - {None}
    others = [w for w in manifest["findings"]["workflows"]
              if w["prompt_format"] == "API_GRAPH" and w["completeness"] == Completeness.COMPLETE
              and w["encoding_detected"] != "CPYTHON_OBJECTS" and w.get("raw_file_path")
              and w.get("canonical_json_sha256")]
    # Of several copies of one graph, keep one that names its submission (a serialized queue item, say).
    others.sort(key=lambda w: not prompt_id_of(w))
    for w in others:
        if w["canonical_json_sha256"] not in seen:
            seen.add(w["canonical_json_sha256"])
            out.append(w)
    return out


def prompt_id_of(workflow: dict[str, Any]) -> str | None:
    return (workflow.get("execution_parameters") or {}).get("prompt_id") or None


def run_all(args: argparse.Namespace) -> int:
    """Regenerates every recovered API prompt of the case. Returns 0 when every run completed, else 1."""
    manifest = json.loads((args.case_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    workflows = recovered_prompts(manifest)
    runner_url = resolve_runner(args) if workflows else None
    failed = 0
    for wf in workflows:
        single = argparse.Namespace(**{**vars(args), "runner_url": runner_url,
                                        "workflow": args.case_dir / wf["raw_file_path"],
                                       "workflow_id": wf["workflow_id"], "compare_to": None})
        try:
            failed += run_regeneration(single) == 1
        except Exception as exc:  # noqa: BLE001 - one job's failure is recorded and the rest still run
            log.error("%s: %s", wf["workflow_id"], exc)
            failed += 1
    log.info("regenerated %d recovered prompts, %d failed or rejected", len(workflows), failed)
    return 0 if not failed else 1


def _resolve_target(case_dir: Path, manifest: dict[str, Any], compare_to: Path) -> tuple[RawArtifactRecord, bytes]:
    """The comparison target must be memory-carved evidence of this case, verified against the manifest."""
    resolved = compare_to.resolve()
    for artifact in manifest["findings"]["artifacts"]:
        paths = [*artifact["raw_file_paths"], artifact.get("derived_preview_path")]
        if any(p and (case_dir / p).resolve() == resolved for p in paths):
            record = RawArtifactRecord.from_dict(artifact)
            record.dimensions = (
                (artifact["pixel_metadata"]["width"], artifact["pixel_metadata"]["height"])
                if artifact.get("pixel_metadata") else None
            )
            if record.completeness == Completeness.COMPLETE:
                data = (case_dir / artifact["raw_file_paths"][0]).read_bytes()
                if hashlib.sha256(data).hexdigest() != artifact["container_sha256"]:
                    raise RegenerationError(f"{compare_to} no longer matches its manifest container_sha256")
                return record, data
            return record, _rebuild_partial_rows(case_dir, artifact)
    raise RegenerationError(f"{compare_to} is not a carved image recorded in this case's manifest")


def _rebuild_partial_rows(case_dir: Path, artifact: dict[str, Any]) -> bytes:
    """Re-decodes a partial target from its raw spans after checking them against the manifest. The DERIVED
    preview file on disk is never trusted as comparison input."""
    meta = artifact.get("derived_preview_metadata") or {}
    expected = meta.get("raw_span_sha256")
    spans = [(case_dir / p).read_bytes() for p in artifact["raw_file_paths"]]
    if not expected or [hashlib.sha256(s).hexdigest() for s in spans] != expected:
        raise RegenerationError(f"{artifact['artifact_id']}: raw spans do not match the manifest's span hashes")
    [rebuilt] = PngCarver().carve_stream(spans[0])
    rows = (rebuilt.derived_preview_metadata or {}).get("decoded_rows")
    if rebuilt.preview_png is None or rows != meta.get("decoded_rows"):
        raise RegenerationError(f"{artifact['artifact_id']}: raw spans no longer decode to the recorded rows")
    return rebuilt.preview_png


def _evaluation_block(manifest: dict[str, Any], workflow: WorkflowRecord, target: RawArtifactRecord | None,
                      evaluated: str, subtier: str | None, metrics: dict[str, Any],
                      ssim_threshold: float, phash_threshold: int) -> dict[str, Any]:
    block: dict[str, Any] = {
        "target_artifact_id": target.artifact_id if target else None,
        "evaluation_mode": metrics.pop("evaluation_mode"),
        "exact_pixel_match": metrics.pop("exact_pixel_match", False),
        "ssim_score": metrics.pop("ssim_score", None),
        "phash_hamming_distance": metrics.pop("phash_hamming_distance", None),
        "ssim_threshold": ssim_threshold,
        "phash_threshold": phash_threshold,
        "evaluated_tier": evaluated,
        "evaluation_subtier": subtier,
        "promoted_finding_tier": None,
        "target_finding_tier_retained": None,
    }
    for key in ("decoded_rows_evaluated", "cropped_dimensions", "row_slice_pixel_match"):
        if key in metrics:
            block[key] = metrics.pop(key)
    block.update(metrics)
    if target is None:
        # Every finding holding the workflow is considered, so the outcome never depends on manifest order: a
        # workflow can sit in an E2 finding (a holder copy, say) and in a pixel-less E5 one at once.
        holding = sorted((f for f in manifest["findings"]["findings"] if workflow.workflow_id in f["artifact_ids"]),
                         key=lambda f: (str(f["tier"]), str(f.get("finding_id"))))
        eligible = sorted(str(f.get("finding_id")) for f in holding if f["tier"] == "E5")
        block["eligible_e4_finding_ids"] = eligible
        if len(eligible) == 1:
            block["target_finding_id"] = eligible[0]
        if evaluated == "E4" and eligible:
            block["promoted_finding_tier"] = "E4"
            block["note"] = "Demonstrates workflow executability, not that an image existed in memory."
        else:
            block["target_finding_tier_retained"] = holding[0]["tier"] if holding else None
            block["note"] = "Workflow is not an unlinked E5 finding; executability recorded without promotion."
        return block
    finding = _finding_for(manifest, [target.artifact_id, workflow.workflow_id])
    base = finding["tier"] if finding else None
    if evaluated == "E1" and base == "E2":
        block["promoted_finding_tier"] = "E1"
    else:
        block["target_finding_tier_retained"] = base
    if finding is None:
        block["note"] = "Target image is not linked to this workflow in the manifest; no promotion."
    return block
