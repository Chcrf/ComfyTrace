"""Generation metadata per sampler, read through the graph's connections, with byte evidence for every field.

``decoding``, ``graph`` and ``resolver`` come from the team's metadata prototype (``memory_metadata``), changed
only for lint. This module runs them on the workflows ComfyTrace recovered: it finds the graph inside a
workflow's exported bytes and resolves each sampler's seed, prompts, model, LoRAs, VAE, size and settings by
following the links that feed it. Each field carries a status (``direct``, ``linked``, ``inferred`` for a
positional UI widget, ``ambiguous``, ``missing``, ``not_applicable``) and the byte range in the exported file
that holds it. Unsupported nodes and broken links leave a field unresolved instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from comfytrace.metadata.decoding import Document, decode, deserialize, parse, pointer, unescape_body
from comfytrace.metadata.graph import SAMPLERS, Graph
from comfytrace.metadata.resolver import Resolver
from comfytrace.models import WorkflowRecord

_KINDS = {"API_GRAPH": "api_graph", "UI_WORKFLOW": "ui_graph"}
_ENCLOSURES = {"prompt", "workflow", "data", "extra_data", "extra_pnginfo", "payload", "result", "records"}
_DEPTH = 3


@dataclass
class _Occurrence:
    """The interface ``Graph`` and ``Resolver`` read: one graph inside one decoded document."""

    id: str
    kind: str
    document: Document
    path: tuple
    value: Any
    raw: bytes
    raw_start: int = 0
    method: str = "comfytrace_workflow"
    metadata: dict[str, Any] = field(default_factory=dict)

    def evidence(self, path: tuple) -> dict[str, Any]:
        a, b = self.document.byte_range(path)
        return {"location": pointer(path), "byte_range": [a, b], "transformations": self.document.transformations,
                "sha256": hashlib.sha256(self.raw[a - self.raw_start:b - self.raw_start]).hexdigest()}


def _shape(obj: Any) -> str | None:
    if not isinstance(obj, dict) or not obj:
        return None
    if isinstance(obj.get("nodes"), list) and all(
            isinstance(n, dict) and "id" in n and isinstance(n.get("type"), str) for n in obj["nodes"]):
        return "ui_graph"
    if all(isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs"), dict)
           for v in obj.values()):
        return "api_graph"
    return None


def _graphs(doc: Document, depth: int = _DEPTH, path: tuple = ()) -> list[tuple[Document, tuple, Any, str]]:
    """Every API graph or UI workflow in the document: at its root, under a known enclosing key (a /prompt body,
    a /history entry) or inside a JSON string holding JSON."""
    obj: Any = doc.value
    for part in path:
        obj = obj[part]
    kind = _shape(obj)
    if kind:
        return [(doc, path, obj, kind)]
    found: list[tuple[Document, tuple, Any, str]] = []
    if isinstance(obj, str) and depth and obj.lstrip().startswith(("{", "[")):
        try:
            found += _graphs(deserialize(doc, path), depth - 1)
        except (ValueError, UnicodeError, RecursionError):
            pass
    elif isinstance(obj, dict):
        for key in obj:
            if key in _ENCLOSURES:
                found += _graphs(doc, depth, path + (key,))
    elif isinstance(obj, list):
        for i in range(len(obj)):
            found += _graphs(doc, depth, path + (i,))
    return found


def _document(workflow: WorkflowRecord) -> Document:
    """The workflow's exported bytes decoded with each character's byte range in that file.
    Raises: ValueError, UnicodeError when the bytes are not one complete JSON value."""
    raw = workflow.raw_bytes
    if workflow.encoding_detected == "UTF-16LE":
        return parse(decode(raw, 0, "utf-16-le"))
    try:
        # A JSON-string draft carved with its quotes is one JSON string; _graphs reads the JSON inside it.
        return parse(decode(raw))
    except ValueError:
        if not workflow.encoding_detected.endswith("JSON_STRING"):
            raise
    return parse(unescape_body(decode(raw)))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_workflow(workflow: WorkflowRecord) -> dict[str, Any] | None:
    """The sampler records of one complete API graph or UI workflow, or None for a fragment or a record whose
    exported bytes do not parse as complete JSON (their parameters stay those ComfyTrace read directly)."""
    kind = _KINDS.get(workflow.prompt_format)
    if kind is None or not workflow.raw_bytes:
        return None
    try:
        doc = _document(workflow)
        found = [g for g in _graphs(doc) if g[3] == kind]
    except (ValueError, UnicodeError, RecursionError):
        return None
    if not found:
        return None
    target = _canonical(workflow.graph)
    gdoc, path, value, _ = next((g for g in found if _canonical(g[2]) == target), found[0])
    occurrence = _Occurrence(workflow.workflow_id, kind, gdoc, path, value, workflow.raw_bytes)
    graph = Graph(occurrence)
    samplers = Resolver(graph).records()
    for record in samplers:
        _strip_context(record["fields"])
    return {
        "workflow_id": workflow.workflow_id,
        "raw_file_path": workflow.raw_file_path,
        "evidence_basis": ("RESERIALIZED_PYTHON_OBJECTS" if workflow.encoding_detected == "CPYTHON_OBJECTS"
                           else "DECOMPRESSED_OR_DECODED_PNG_TEXT" if workflow.encoding_detected.startswith("PNG_TEXT")
                           else "CONTAINER_METADATA_TEXT" if workflow.encoding_detected.endswith("_EMBEDDED_TEXT")
                           else "EXPORTED_DUMP_BYTES"),
        "graph_location": pointer(path),
        "graph_completeness": graph.record()["graph_completeness"],
        "graph_issues": graph.issues,
        "node_count": len(graph.nodes),
        "sampler_count": sum(n.type in SAMPLERS for n in graph.nodes),
        "samplers": samplers,
    }


def _strip_context(item: Any) -> None:
    """The resolver repeats the graph and sampler ids on every nested field; the sampler record already has them."""
    if isinstance(item, dict):
        for key in ("occurrence_id", "graph_id", "sampler_id"):
            if "status" in item and "evidence" in item:
                item.pop(key, None)
        for value in item.values():
            _strip_context(value)
    elif isinstance(item, list):
        for value in item:
            _strip_context(value)


def metadata_document(workflows: list[WorkflowRecord]) -> dict[str, Any]:
    records = [r for r in (resolve_workflow(w) for w in workflows) if r is not None]
    return {
        "method": "Connection-aware sampler resolution (team metadata prototype: graph.py, resolver.py) over "
                  "each recovered workflow's exported bytes. Byte ranges are offsets into raw_file_path.",
        "statuses": {"direct": "value on the sampler node itself", "linked": "value reached through a link",
                     "inferred": "value read from a positional UI widget list (core node layout assumed)",
                     "ambiguous": "conflicting connected values", "missing": "no supported connected evidence",
                     "not_applicable": "the sampler type has no such input"},
        "not_resolved": [w.workflow_id for w in workflows if w.workflow_id not in {r["workflow_id"] for r in records}],
        "workflows": records,
    }
