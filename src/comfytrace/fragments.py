"""ComfyUI JSON cut at a page edge: the partial tree it still proves, and what it agrees with.

A prompt or workflow whose copy no longer lies on contiguous pages survives in physical memory only as
page-sized pieces. ``parse_fragment`` reads such a piece without knowing where it starts: values whose
container began before the cut sit under the key ``CUT``, and containers still open at the end are closed.
``correlate.place_fragments`` then compares each piece with the graphs recovered elsewhere.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

PAGE = 4096
CUT = "<cut>"
"""Key of a value whose own key lies before the start of the fragment."""

_TOKEN = re.compile(
    r'\s*(?:(?P<q>")|(?P<p>[{}\[\]:,])|(?P<n>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)|(?P<l>true|false|null))'
)
_ESCAPED = re.compile("[\udc80-\udcff]+")
_SURROGATE = re.compile("[\ud800-\udfff]")
_LITERALS = {"true": True, "false": False, "null": None}
# KSampler's widget order, for workflows saved before the frontend wrote widgets_values_named.
_KSAMPLER_WIDGETS = ("seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise")
# seed_control: with "randomize", "increment" or "decrement" a UI copy saved after queueing shows the next seed,
# not the one used, so a UI seed is evidence of a job only when another record agrees with it.
_UI_PARAMETERS = {"seed": "seed", "noise_seed": "seed", "control_after_generate": "seed_control",
                  "steps": "steps", "cfg": "cfg_scale",
                  "sampler_name": "sampler_name", "scheduler": "scheduler", "ckpt_name": "model_name",
                  "width": "width", "height": "height", "batch_size": "batch_size"}


@dataclass
class _Frame:
    kind: str  # "{", "[", or "?" for the container the fragment starts inside
    items: list[tuple[str | None, Any]] = field(default_factory=list)
    key: str | None = None
    after_colon: bool = False

    def add(self, value: Any) -> None:
        self.items.append((self.key, value))
        self.key, self.after_colon = None, False

    def close(self, kind: str) -> Any:
        if kind == "{":
            return {k: v for k, v in self.items if k is not None}
        return [v for k, v in self.items if k in (None, CUT)]


@dataclass(frozen=True)
class Fragment:
    """``value`` is read from text[start:end]. ``cut_before``: the text starts inside a value. ``cut_after``: it
    ends inside one; the containers still open were closed with what they held, and a string or number cut by
    the end is part of the span but not of ``value``."""

    value: Any
    start: int
    end: int
    cut_before: bool
    cut_after: bool


def _string_end(text: str, start: int) -> int:
    """Index just past the quote closing the string whose body begins at ``start``, or -1 if the text ends first."""
    at = start
    while (quote := text.find('"', at)) >= 0:
        backslashes = quote
        while backslashes > start and text[backslashes - 1] == "\\":
            backslashes -= 1
        if (quote - backslashes) % 2 == 0:
            return quote + 1
        at = quote + 1
    return -1


def _parse(text: str, pos: int) -> Fragment:
    first = end = pos
    stack = [_Frame("?")]
    above = 0
    truncated = False
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            break
        top = stack[-1]
        if m["q"] is not None:
            close = _string_end(text, m.end())
            if close < 0:
                truncated, end = True, len(text)
                break
            try:
                s = json.loads(text[m.start("q"):close])
            except ValueError:
                break
            if _SURROGATE.search(s):  # an escaped lone surrogate has no UTF-8 form
                break
            if top.kind != "[" and top.key is None and not top.after_colon:
                top.items.append((None, s))
            else:
                top.add(s)
            pos = end = close
            continue
        if m["n"] is not None:
            if m.end() == len(text):
                truncated, end = True, len(text)
                break
            top.add(json.loads(m["n"]))
        elif m["l"] is not None:
            top.add(_LITERALS[m["l"]])
        elif m["p"] in "{[":
            stack.append(_Frame(m["p"]))
        elif m["p"] in "}]":
            want = "{" if m["p"] == "}" else "["
            if top.kind not in ("?", want) or top.kind == "?" and want == "[" and any(
                    k not in (None, CUT) for k, _ in top.items):
                break
            if top.kind == "?":
                stack[0] = _Frame("?", [(CUT, top.close(want))])
                above += 1
            else:
                stack.pop()
                stack[-1].add(top.close(want))
        elif m["p"] == ":":
            if top.kind == "?" and not top.items:
                top.key, top.after_colon = CUT, True
            elif top.kind == "[" or not top.items or top.items[-1][0] is not None or top.after_colon \
                    or not isinstance(top.items[-1][1], str):
                break
            else:
                top.key, top.after_colon = top.items.pop()[1], True
        pos = end = m.end()
    cut_after = truncated or len(stack) > 1
    while len(stack) > 1:
        frame = stack.pop()
        frame.key = None
        stack[-1].add(frame.close(frame.kind))
    root = stack[0]
    if any(k not in (None, CUT) for k, _ in root.items):
        value = root.close("{")
    else:
        items = root.close("[")
        value = items[0] if len(items) == 1 else items
    return Fragment(value, first, end, above > 0, cut_after)


def parse_fragment(text: str, pos: int = 0, cut_start: bool = True) -> Fragment | None:
    """The longest reading of ``text[pos:]`` as JSON cut at both ends: starting outside a string, or (when
    ``cut_start``) inside one and resuming after its closing quote. Parsing stops at the first character JSON
    cannot continue with. Offsets in the result index ``text``."""
    quote = text.find('"', pos) if cut_start else -1
    best = None
    for start in (pos, quote + 1) if quote >= 0 else (pos,):
        fragment = _parse(text, start)
        if fragment.end - fragment.start >= 2 and (best is None or fragment.end > best.end):
            best = fragment
    if best is not None and best.start > pos:
        best = Fragment(best.value, best.start, best.end, True, best.cut_after)
    return best


def _walk(value: Any, parent_key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    stack = [(parent_key, value)]
    while stack:
        key, item = stack.pop()
        yield key, item
        if isinstance(item, dict):
            stack.extend(reversed(list(item.items())))
        elif isinstance(item, list):
            stack.extend((None, v) for v in reversed(item))


def api_nodes(value: Any) -> list[tuple[str | None, dict[str, Any]]]:
    """API prompt nodes, with their id when the key survived the cut. A node cut before its class_type still
    counts when its numeric id and its inputs survived."""
    return [(None if key == CUT else key, item) for key, item in _walk(value) if isinstance(item, dict) and (
        isinstance(item.get("class_type"), str)
        or key not in (None, CUT) and key[:1].isdigit() and isinstance(item.get("inputs"), dict))]


def ui_node_type(node: dict[str, Any]) -> str | None:
    """A UI node's type, read from its S&R name when the cut removed the type field."""
    if isinstance(node.get("type"), str):
        return str(node["type"])
    props = node.get("properties")
    name = props.get("Node name for S&R") if isinstance(props, dict) else None
    return name if isinstance(name, str) else None


def ui_nodes(value: Any) -> list[dict[str, Any]]:
    """UI workflow nodes: objects with widget values, or with an integer id and a type."""
    return [item for _, item in _walk(value) if isinstance(item, dict) and "class_type" not in item and (
        isinstance(item.get("widgets_values"), list | dict)
        or isinstance(item.get("id"), int) and isinstance(item.get("type"), str) and "pos" in item)]


def ui_workflow_nodes(graph: Any) -> list[dict[str, Any]]:
    """Read complete subgraph workflows through the metadata resolver's connections.
    Workflows without definitions retain the existing widget reader."""
    subgraphs = graph.get("definitions", {}).get("subgraphs") if isinstance(graph, dict) and isinstance(
        graph.get("definitions"), dict) else None
    if not subgraphs:
        return ui_nodes(graph)
    from comfytrace.metadata import _Occurrence
    from comfytrace.metadata.decoding import decode, parse
    from comfytrace.metadata.graph import UI_ORDER, Graph
    from comfytrace.metadata.resolver import Resolver

    try:
        raw = json.dumps(graph, ensure_ascii=False, allow_nan=False).encode("utf8")
        doc = parse(decode(raw))
    except (ValueError, UnicodeError, RecursionError):
        return ui_nodes(graph)
    resolved = Graph(_Occurrence("parameters", "ui_graph", doc, (), doc.value, raw))
    reader = Resolver(resolved)
    out = []
    for node in resolved.nodes:
        if node.type not in UI_ORDER:
            out.append({**node.raw, "id": node.id})
            continue  # preserve the existing named-widget reader for custom nodes
        values = {}
        for name in UI_ORDER.get(node.type, "").split():
            field = reader.scalar(node, name)
            if field["status"] in ("direct", "linked", "inferred"):
                values[name] = field["value"]
        out.append({**node.raw, "id": node.id, "widgets_values_named": values})
    return out


def ui_parameters(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Generation parameters from UI widget values: named widgets where the frontend wrote them, otherwise the
    KSampler's fixed widget order. Values a node does not carry are omitted."""
    params: dict[str, Any] = {}
    for node in nodes:
        named = node.get("widgets_values_named")
        widgets = node.get("widgets_values")
        if not isinstance(named, dict):
            named = (
                {k: v for k, v in zip(_KSAMPLER_WIDGETS, widgets, strict=False) if k is not None}
                if ui_node_type(node) == "KSampler" and isinstance(widgets, list)
                else {}
            )
        for key, value in named.items():
            if key in _UI_PARAMETERS and not isinstance(value, list | dict):
                params.setdefault(_UI_PARAMETERS[key], value)
    return params


def _agrees(part: Any, whole: Any) -> bool:
    """``part`` is a piece of ``whole``: shared keys agree, and a list is a prefix or suffix of the other list.
    Keys ``whole`` lacks are ignored, so a newer frontend's extra keys do not break agreement."""
    if isinstance(part, dict):
        return isinstance(whole, dict) and all(k == CUT or k not in whole or _agrees(v, whole[k])
                                               for k, v in part.items())
    if isinstance(part, list):
        if not isinstance(whole, list) or len(part) > len(whole):
            return False
        tail = len(whole) - len(part)
        return all(_agrees(p, w) for p, w in zip(part, whole, strict=False)) or all(
            _agrees(p, whole[tail + i]) for i, p in enumerate(part))
    return type(part) is type(whole) and part == whole or (
        isinstance(part, int | float) and isinstance(whole, int | float)
        and not isinstance(part, bool) and not isinstance(whole, bool)
        and part == whole)


@dataclass(frozen=True)
class Reference:
    """A recovered graph a fragment can be compared with: a workflow record, or a PNG's embedded text."""

    record_id: str
    api: dict[str, dict[str, Any]]
    ui: list[dict[str, Any]]
    seed: Any


def _ui_node_agrees(node: dict[str, Any], ui: list[dict[str, Any]]) -> bool:
    if isinstance(node.get("id"), int):
        return any(r.get("id") == node["id"] and _agrees(node, r) for r in ui)
    kind = ui_node_type(node)
    return any((kind is None or ui_node_type(r) == kind) and _agrees(node, r) for r in ui)


def _api_node_agrees(key: str | None, node: dict[str, Any], api: dict[str, dict[str, Any]]) -> bool:
    if key is not None:
        return key in api and _agrees(node, api[key])
    return any(r.get("class_type") == node["class_type"] and _agrees(node, r) for r in api.values())


def agrees_with(value: Any, ref: Reference) -> bool:
    """Every node the fragment holds in the reference's format agrees with a node of the reference."""
    api, ui = api_nodes(value), ui_nodes(value)
    checked = (ref.api and api) or (ref.ui and ui)
    return bool(checked) and (not ref.api or all(_api_node_agrees(k, n, ref.api) for k, n in api)) and (
        not ref.ui or all(_ui_node_agrees(n, ref.ui) for n in ui))


@dataclass
class TextFragment:
    """A carved piece of ComfyUI JSON, with every physical offset it was found at. ``encoding`` is UTF-8, or
    UTF-8_JSON_STRING when the JSON was stored as a string inside other JSON (``raw`` keeps the escapes)."""

    raw: bytes
    fragment: Fragment
    physical_offsets: list[int]
    encoding: str = "UTF-8"


ESCAPED_ANCHORS = (b'\\"class_type\\"', b'\\"nodes\\"', b'\\"widgets_values\\"')
"""ComfyUI keys inside a JSON string, as the frontend stores workflow drafts: {"data":"{\\"id\\":...}"}."""
_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})')
_ESCAPED_CHAR = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


@dataclass(frozen=True)
class EscapedLiteral:
    """A JSON string in a run whose content holds an escaped ComfyUI key. ``start`` and ``end`` are byte offsets
    of the opening quote and just past the closing one; a quote outside the run makes the literal cut there and
    the offset is the run's edge."""

    start: int
    end: int
    cut_before: bool
    cut_after: bool


def _unescaped_quote(run: bytes, at: int) -> bool:
    backslashes = 0
    while at - backslashes - 1 >= 0 and run[at - backslashes - 1] == 0x5C:
        backslashes += 1
    return backslashes % 2 == 0


def escaped_literals(run: bytes, anchors: list[int]) -> list[EscapedLiteral]:
    """The string literals around ``anchors`` (offsets in ``run`` of ESCAPED_ANCHORS). An escaped quote can only
    occur inside a string, so the nearest unescaped quote before an anchor opens its string."""
    out: list[EscapedLiteral] = []
    for anchor in sorted(anchors):
        if out and anchor < out[-1].end:
            continue
        start = run.rfind(b'"', 0, anchor)
        while start >= 0 and not _unescaped_quote(run, start):
            start = run.rfind(b'"', 0, start)
        end = run.find(b'"', anchor + 1)
        while end >= 0 and not _unescaped_quote(run, end):
            end = run.find(b'"', end + 1)
        out.append(EscapedLiteral(max(start, 0), end + 1 if end >= 0 else len(run), start < 0, end < 0))
    return out


def unescape(text: str) -> str:
    """JSON string content (without its quotes) to the text it holds, tolerating cuts: an escape cut by the end
    is dropped, a quote whose backslash lay before the text stays a quote, and a surrogate pair cut in half
    becomes U+FFFD."""
    text = re.sub(r"\\(u[0-9a-fA-F]{0,3})?$", "", text)
    out = _ESCAPE.sub(lambda m: chr(int(m[1][1:], 16)) if m[1][0] == "u" else _ESCAPED_CHAR[m[1]], text)
    return out.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def carve_fragments(run: bytes, run_base: int, covered: list[tuple[int, int]], anchors: list[int]) -> Iterator[
        tuple[int, bytes, Fragment]]:
    """Pieces of ComfyUI JSON in ``run`` (absolute physical base ``run_base``) outside the ``covered`` absolute
    spans of whole graphs. Invalid UTF-8 ends a piece, as a control character does, and so does a page boundary:
    the next physical page is no evidence of continuation. Yields (offset, bytes, fragment) for each cut piece
    that holds an anchor and at least one API or UI node."""
    first_boundary = -run_base % PAGE
    cuts = [(lo - run_base, hi - run_base) for lo, hi in covered]
    cuts += [(b, b) for b in range(first_boundary, len(run), PAGE) if b > 0]
    segments, cursor = [], 0
    for lo, hi in [*sorted(cuts), (len(run), len(run))]:
        if lo > cursor:
            segments.append((cursor, lo))
        cursor = max(cursor, hi)
    for lo, hi in segments:
        if not any(lo <= a - run_base < hi for a in anchors):
            continue
        text = run[lo:hi].decode("utf-8", "surrogateescape")
        offset = run_base + lo
        cursor = 0
        for bad in [*_ESCAPED.finditer(text), None]:
            stop = bad.start() if bad is not None else len(text)
            valid = text[cursor:stop]
            size = len(valid.encode("utf-8"))
            if any(offset <= a < offset + size for a in anchors):
                yield from _pieces(valid, offset)
            offset += size + (bad.end() - bad.start() if bad is not None else 0)
            cursor = stop if bad is None else bad.end()


def _pieces(text: str, base: int) -> Iterator[tuple[int, bytes, Fragment]]:
    """Fragments of one valid-UTF-8 stretch. Only the first may start inside a string; after the bytes that
    end a fragment, the next one starts at the following brace."""
    pos, byte_pos = 0, base
    fragment = parse_fragment(text)
    while True:
        if fragment is not None:
            start = byte_pos + len(text[pos:fragment.start].encode("utf-8"))
            raw = text[fragment.start:fragment.end].encode("utf-8")
            if (fragment.cut_before or fragment.cut_after) and (api_nodes(fragment.value)
                                                                or ui_nodes(fragment.value)):
                yield start, raw, fragment
            pos, byte_pos = fragment.end, start + len(raw)
        following = text.find("{", pos + 1)
        if following < 0:
            return
        byte_pos += len(text[pos:following].encode("utf-8"))
        pos = following
        fragment = parse_fragment(text, pos, cut_start=False)
