"""Image files ComfyUI's server wrote, from traces of serving them: aiohttp's /view response headers and the
frontend's /view request URLs.

These are not image content. They name each file, its folder, its size and its modification time: aiohttp's
ETag is the file's mtime in nanoseconds and its size, both in hex. The headers usually survive in Chromium's
HTTP cache, which is file-backed, so every copy carries its page origin like any other record: under the
anti-forensics assumption these records add context but no finding rests on them.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

HEADER = re.compile(rb'Content-Disposition: filename="([^"\x00-\x1f]{1,200})"')
HEADER_PREFIX = b'Content-Disposition: filename="'
VIEW_URL = {"UTF-8": re.compile(rb"/view\?([A-Za-z0-9_.~%&=+\-]{1,600})"),
            "UTF-16LE": re.compile(rb"/\x00v\x00i\x00e\x00w\x00\?\x00((?:[A-Za-z0-9_.~%&=+\-]\x00){1,600})")}
_FIELD = re.compile(rb"([A-Za-z][A-Za-z0-9-]{1,40}): ([^\x00\r\n]{0,300})")
_ETAG = re.compile(r'^"([0-9a-f]{1,17})-([0-9a-f]{1,12})"$')
HEADER_WINDOW = 1024


@dataclass
class OutputFile:
    """One file name with its served versions (by ETag), the UI's requests for it and every copy's location."""

    filename: str
    versions: dict[str, dict[str, Any]] = field(default_factory=dict)
    ui_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    copies: list[dict[str, Any]] = field(default_factory=list)
    page_origins: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, record_id: str) -> dict[str, Any]:
        types = {r["type"] for r in self.ui_requests.values() if r.get("type")}
        return {"record_id": record_id, "filename": self.filename,
                "folder": types.pop() if len(types) == 1 else ("temp" if "_temp_" in self.filename else None),
                "versions": sorted(self.versions.values(), key=lambda v: v.get("file_mtime_utc") or ""),
                "ui_requests": sorted(self.ui_requests.values(), key=lambda r: r.get("requested_at_utc") or ""),
                "copies": [{k: v for k, v in c.items() if k != "origin"} for c in self.copies]}


def _iso_ms(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="milliseconds")


def _headers(window: bytes) -> dict[str, str]:
    return {m[1].decode().lower(): m[2].decode("latin-1").strip() for m in _FIELD.finditer(window)}


def _version(headers: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"etag": headers.get("etag"), "last_modified": headers.get("last-modified"),
                           "served_at": headers.get("date"), "server": headers.get("server")}
    length = headers.get("content-length", "")
    out["content_length"] = int(length) if length.isascii() and length.isdigit() else None
    etag = _ETAG.match(headers.get("etag", ""))
    if etag:
        ns, size = int(etag[1], 16), int(etag[2], 16)
        out["size"] = size
        try:
            out["file_mtime_utc"] = dt.datetime.fromtimestamp(ns // 10**9, dt.timezone.utc).isoformat().replace(
                "+00:00", f".{ns % 10**9:09d}Z")
        except (OverflowError, OSError, ValueError):
            pass
    return out


def _header_matches(buffer: bytes) -> Iterator[re.Match[bytes]]:
    """HEADER.finditer(buffer), with the literal start of the pattern located by bytes.find: the same matches in
    the same order (after a match the search goes on at its end, after a failed start one byte further)."""
    at = buffer.find(HEADER_PREFIX)
    while at >= 0:
        m = HEADER.match(buffer, at)
        if m is not None:
            yield m
        at = buffer.find(HEADER_PREFIX, m.end() if m is not None else at + 1)


class OutputFileCarver:
    """Collects /view responses served by aiohttp and /view request URLs, grouped by file name."""

    def __init__(self) -> None:
        self.files: dict[str, OutputFile] = {}
        self._seen: set[tuple[Any, int]] = set()

    def carve_stream(self, buffer: bytes, virtual_address: int | None = None, base_physical_offset: int | None = None,
                     process_id: int | None = None) -> None:
        base = virtual_address if virtual_address is not None else (base_physical_offset or 0)
        space = process_id if virtual_address is not None else "physical"

        def copy(offset: int, length: int, kind: str) -> dict[str, Any] | None:
            if (space, base + offset) in self._seen:
                return None
            self._seen.add((space, base + offset))
            where = ({"process_id": process_id, "virtual_address": f"0x{base + offset:X}"}
                     if virtual_address is not None else {"physical_offset": f"0x{base + offset:016X}"})
            return {"kind": kind, "bytes": length, **where}

        for m in _header_matches(buffer):
            lo = buffer.rfind(b"HTTP/1.", max(0, m.start() - 512), m.start())
            lo = m.start() if lo < 0 else lo
            # One response's block ends at a blank line (CRLF or Chromium's NUL-separated form) or the next status
            # line; reading on would take the next response's ETag.
            ends = [buffer.find(stop, m.end(), m.end() + HEADER_WINDOW) for stop in (b"\x00\x00", b"\r\n\r\n",
                                                                                   b"HTTP/1.")]
            window = buffer[lo : min([e for e in ends if e >= 0] or [m.end() + HEADER_WINDOW])]
            headers = _headers(window)
            if "aiohttp" not in headers.get("server", ""):
                continue
            where = copy(lo, len(window), "HTTP_RESPONSE")
            if where is None:
                continue
            entry = self.files.setdefault(m[1].decode("utf-8", "replace"), OutputFile(m[1].decode("utf-8", "replace")))
            version = _version(headers)
            loc = where.get("physical_offset") or f"pid{where.get('process_id')}@{where.get('virtual_address')}"
            key = version["etag"] or f"no-etag@{loc}"
            entry.versions.setdefault(key, version)
            entry.copies.append({**where, "version": key})
        for encoding, pattern in VIEW_URL.items():
            for m in pattern.finditer(buffer):
                query = parse_qs(m[1].decode("utf-16-le" if encoding == "UTF-16LE" else "ascii", "replace"))
                name = query.get("filename", [""])[0]
                if not name:
                    continue
                where = copy(m.start(), m.end() - m.start(), f"VIEW_URL_{encoding}")
                if where is None:
                    continue
                entry = self.files.setdefault(name, OutputFile(name))
                stamp = query.get("t", query.get("rand", ["-"]))[0]
                request = entry.ui_requests.setdefault(stamp, {"type": query.get("type", [None])[0],
                                                               "subfolder": query.get("subfolder", [""])[0]})
                if stamp.isascii() and stamp.isdigit() and 10**12 <= int(stamp) < 10**13:
                    request["requested_at_utc"] = _iso_ms(int(stamp))
                entry.copies.append(where)

    def records(self) -> list[tuple[str, OutputFile]]:
        return [(f"OUT-{n:03d}", entry) for n, (_, entry) in enumerate(sorted(self.files.items()), 1)]
