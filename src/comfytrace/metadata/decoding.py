"""Strict JSON with exact original-byte provenance through JSON string decoding."""
import json
import math
import re
from dataclasses import dataclass, field


class Object(dict):
    def __init__(self):
        super().__init__()
        self.duplicates = set()


def pointer(path):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path)


@dataclass
class Document:
    text: str
    starts: list
    ends: list
    transformations: list
    value: object = None
    spans: dict = field(default_factory=dict)
    ambiguous_enclosure: bool = False

    def byte_range(self, path=()):
        a, b = self.spans[path]
        return [self.starts[a], self.ends[b - 1]]


def decode(raw, offset=0, encoding="utf-8"):
    text = raw.decode(encoding, errors="strict")
    starts, ends, pos = [], [], offset
    for char in text:
        starts.append(pos)
        pos += len(char.encode(encoding))
        ends.append(pos)
    return Document(text, starts, ends, [encoding])


def parse(doc, max_depth=64):
    text, spans = doc.text, {}
    whitespace = re.compile(r"[ \r\n\t]*")
    number = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

    def value(pos, path, depth):
        if depth > max_depth:
            raise ValueError("JSON nesting limit exceeded")
        pos = whitespace.match(text, pos).end()
        start = pos
        if pos >= len(text):
            raise ValueError("incomplete JSON")
        c = text[pos]
        if c == '"':
            result, pos = json.decoder.scanstring(text, pos + 1, True)
        elif c in "{[":
            is_object = c == "{"
            result = Object() if is_object else []
            closer = "}" if is_object else "]"
            pos = whitespace.match(text, pos + 1).end()
            if pos >= len(text):
                raise ValueError("incomplete container")
            if text[pos] != closer:
                while True:
                    if is_object:
                        if text[pos:pos + 1] != '"':
                            raise ValueError("expected member name")
                        key, pos = json.decoder.scanstring(text, pos + 1, True)
                        pos = whitespace.match(text, pos).end()
                        if text[pos:pos + 1] != ":":
                            raise ValueError("expected colon")
                        child, pos = value(pos + 1, path + (key,), depth + 1)
                        if key in result:
                            result.duplicates.add(key)
                        result[key] = child
                    else:
                        child, pos = value(pos, path + (len(result),), depth + 1)
                        result.append(child)
                    pos = whitespace.match(text, pos).end()
                    if text[pos:pos + 1] == closer:
                        break
                    if text[pos:pos + 1] != ",":
                        raise ValueError("expected comma or closing delimiter")
                    pos = whitespace.match(text, pos + 1).end()
            pos += 1
        else:
            match = number.match(text, pos)
            if match:
                result = json.loads(match.group())
                if isinstance(result, float) and not math.isfinite(result):
                    raise ValueError("non-finite JSON number")
                pos = match.end()
            else:
                token = next((t for t in ("true", "false", "null") if text.startswith(t, pos)), None)
                if token is None:
                    raise ValueError("invalid JSON value")
                result, pos = {"true": True, "false": False, "null": None}[token], pos + len(token)
        spans[path] = (start, pos)
        return result, pos

    doc.value, end = value(0, (), 0)
    if text[end:].strip():
        raise ValueError("trailing content")
    doc.spans = spans
    return doc


def ambiguous_path(doc, path):
    obj = doc.value
    for part in path:
        if isinstance(obj, dict) and part in getattr(obj, "duplicates", set()):
            return True
        obj = obj[part]
    return False


def deserialize(doc, path):
    """Map decoded characters to their complete original escape sequences."""
    a, b = doc.spans[path]
    chars, starts, ends = [], [], []
    i = a + 1
    while i < b - 1:
        begin = i
        if doc.text[i] != "\\":
            char, i = doc.text[i], i + 1
        elif doc.text[i + 1] == "u":
            i += 6
            unit = int(doc.text[begin + 2:i], 16)
            if 0xD800 <= unit <= 0xDBFF and doc.text[i:i + 2] == "\\u":
                low = int(doc.text[i + 2:i + 6], 16)
                if 0xDC00 <= low <= 0xDFFF:
                    i += 6
            char = json.loads('"' + doc.text[begin:i] + '"')
        else:
            i += 2
            char = json.loads('"' + doc.text[begin:i] + '"')
        for c in char:
            chars.append(c)
            starts.append(doc.starts[begin])
            ends.append(doc.ends[i - 1])
    return parse(Document("".join(chars), starts, ends,
                          doc.transformations + ["json-string:" + pointer(path)],
                          ambiguous_enclosure=doc.ambiguous_enclosure or ambiguous_path(doc, path)))

def unescape_body(doc):
    """One bounded layer for an escaped JSON buffer without surrounding quotes.

    Parsing the resulting complete object is still mandatory. No truncation or
    punctuation repair is performed. Mapping survives every removed backslash.
    """
    chars, starts, ends = [], [], []
    i = 0
    while i < len(doc.text):
        begin = i
        if doc.text[i] != "\\":
            char, i = doc.text[i], i + 1
        else:
            try:
                if doc.text[i + 1] == "u":
                    i += 6
                    unit = int(doc.text[begin + 2:i], 16)
                    if 0xD800 <= unit <= 0xDBFF and doc.text[i:i + 2] == "\\u":
                        low = int(doc.text[i + 2:i + 6], 16)
                        if 0xDC00 <= low <= 0xDFFF:
                            i += 6
                else:
                    i += 2
                char = json.loads('"' + doc.text[begin:i] + '"')
            except (ValueError, IndexError):
                break
        for c in char:
            chars.append(c)
            starts.append(doc.starts[begin])
            ends.append(doc.ends[i - 1])
    return Document("".join(chars), starts, ends, doc.transformations + ["json-escape-body"], ambiguous_enclosure=doc.ambiguous_enclosure)
