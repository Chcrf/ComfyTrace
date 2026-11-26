"""Stage 2 (raw scan, continued): ComfyUI outputs that are not PNG.

ComfyUI also saves WEBP and animated WEBP (``SaveAnimatedWEBP``), MP4 (``SaveVideo``), WebM (``SaveWEBM``) and
FLAC (``SaveAudio``), and writes the prompt and workflow into each:

* WEBP: the EXIF chunk's ASCII tags, ``prompt:{json}`` in 0x0110 and ``<key>:{json}`` from 0x010F downwards;
* MP4: ``moov/udta/meta`` with a ``keys`` box naming ``prompt`` and ``workflow`` and an ``ilst`` of UTF-8 values
  (PyAV with ``movflags=use_metadata_tags``);
* WebM: Matroska ``Tags/Tag/SimpleTag`` with TagName ``PROMPT`` / ``WORKFLOW``;
* FLAC: the VORBIS_COMMENT block, ``prompt=...`` and ``workflow=...``.

Each container is walked strictly from its first byte. What that proves differs by format, and each record says
which checks it passed:

* FLAC frames carry a CRC-16, so audio bytes are verified as PNG chunks are. The last frame has no following
  frame header to delimit it: its block size must be exactly the samples STREAMINFO still expects, and its end is
  the first place, at or after the stream's minimum frame size, where the CRC-16 fits. (Later fits prove
  nothing: once a frame and its CRC have been read the running CRC is zero, and it stays zero over the zero
  bytes that usually follow a file in memory.)
* WEBP, MP4 and WebM have no checksum. The walk must close exactly at the size the container declares (or, for
  MP4, on its last box), which fails when the pages read are not the file's pages in order. The part that
  holds the metadata is checked as far as its structure allows: a WEBP must decode in Pillow and every string
  of its EXIF chunk must be a terminated UTF-8 string; an MP4's ``moov`` box and a WebM's ``Tags`` element must
  be tiled exactly by their children; ComfyUI text must parse as JSON. Encoded video and audio between two
  verified headers is unverified, and ``validation.payload_checksum`` says so.

Nothing is invented: bytes are kept up to the last verified structure, and the rest is a gap.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from comfytrace.carve import ByteSource, IdAllocator, SpanSet, _hex_addr
from comfytrace.models import (
    CarveType,
    Completeness,
    GapDescriptor,
    GapReason,
    LinkClass,
    RawArtifactRecord,
)

log = logging.getLogger(__name__)

MAX_MEDIA_BYTES = 256 * 1024 * 1024
PAGE_SIZE = 4096
WEBP_CHUNKS = frozenset({b"VP8 ", b"VP8L", b"VP8X", b"ANIM", b"ANMF", b"ALPH", b"ICCP", b"EXIF", b"XMP "})
MP4_BRANDS = frozenset({b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"av01", b"M4V ",
                        b"M4A ", b"qt  ", b"dash"})
MP4_TOP = frozenset({b"ftyp", b"free", b"skip", b"wide", b"mdat", b"moov", b"moof", b"mfra", b"sidx", b"styp",
                     b"uuid", b"meta"})
MP4_CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"ilst"})
EBML_HEADER = b"\x1a\x45\xdf\xa3"
SEGMENT = 0x18538067
MATROSKA_LEVEL1 = frozenset({0x114D9B74, 0x1549A966, 0x1654AE6B, 0x1254C367, 0x1F43B675, 0x1C53BB6B, 0x1941A469,
                             0x1043A770, 0xEC, 0xBF})
TAGS, TAG, SIMPLE_TAG, TAG_NAME, TAG_STRING = 0x1254C367, 0x7373, 0x67C8, 0x45A3, 0x4487
TEXT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
EXTENSIONS = {"WEBP": "webp", "MP4": "mp4", "WEBM": "webm", "FLAC": "flac"}
ENCODED = {"WEBP": CarveType.ENCODED_WEBP, "MP4": CarveType.ENCODED_MP4, "WEBM": CarveType.ENCODED_WEBM,
           "FLAC": CarveType.ENCODED_FLAC}
PARTIAL = {"WEBP": CarveType.PARTIAL_WEBP, "MP4": CarveType.PARTIAL_MP4, "WEBM": CarveType.PARTIAL_WEBM,
           "FLAC": CarveType.PARTIAL_FLAC}
MEDIA_TYPES = frozenset({*ENCODED.values(), *PARTIAL.values()})
NO_CHECKSUM = "none in this format: bytes between verified headers are unverified"


@dataclass
class Parsed:
    """One container walked from its first byte. ``end``: bytes verified. ``needed``: the size the container
    declares, when it declares one. ``more``: the walk stopped because the bytes ran out."""

    kind: str
    end: int
    complete: bool = False
    halt: str | None = None
    needed: int | None = None
    more: bool = False
    text: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


def _json_text(text: dict[str, str], key: str, value: str) -> None:
    """Keeps ``key: value`` when the value is a JSON object or array, as ComfyUI's prompt and workflow are. A bare
    number or string is what any encoder writes (a duration, a track number) and is not ComfyUI text."""
    key = key.lower()
    if key in text or not TEXT_KEY.fullmatch(key):
        return
    try:
        if not isinstance(json.loads(value), dict | list):
            return
    except (ValueError, RecursionError):
        return
    text[key] = value


# -- WEBP -------------------------------------------------------------------------------------------------------


def exif_text(payload: bytes) -> tuple[dict[str, str], bool, int]:
    """ComfyUI's ``<key>:<json>`` strings in the ASCII tags of the EXIF chunk's first IFD. Also whether the IFD
    and every ASCII value are intact (inside the chunk, UTF-8, terminated by their NUL at the declared length),
    and how many bytes of the chunk that holds for: a chunk that runs into other bytes fails here."""
    lead = 6 if payload[:6] == b"Exif\x00\x00" else 0
    payload = payload[lead:]
    order = {b"II*\x00": "<", b"MM\x00*": ">"}.get(payload[:4])
    text: dict[str, str] = {}
    if order is None or len(payload) < 14:
        return text, False, 0
    (ifd,) = struct.unpack_from(order + "I", payload, 4)
    if ifd + 2 > len(payload):
        return text, False, lead + 8
    (count,) = struct.unpack_from(order + "H", payload, ifd)
    if ifd + 2 + 12 * count > len(payload):
        return text, False, lead + 8
    intact, valid = True, ifd + 2 + 12 * count
    for i in range(count):
        at = ifd + 2 + 12 * i
        _tag, kind, length, offset = struct.unpack_from(order + "HHII", payload, at)
        if kind != 2:
            continue
        raw = payload[offset : offset + length] if length > 4 else payload[at + 8 : at + 8 + length]
        try:
            if len(raw) != length or not raw.endswith(b"\x00"):
                raise ValueError("not a terminated string of the declared length")
            key, _, value = raw[:-1].decode("utf-8").partition(":")
        except (UnicodeDecodeError, ValueError):
            intact = False
            continue
        if length > 4:
            valid = max(valid, offset + length)
        _json_text(text, key, value)
    return text, intact, lead + valid


def parse_webp(data: bytes) -> Parsed | None:
    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP" or data[12:16] not in WEBP_CHUNKS:
        return None
    total = struct.unpack_from("<I", data, 4)[0] + 8
    if total < 20 or total > MAX_MEDIA_BYTES:
        return None
    out = Parsed("WEBP", 12, needed=total, meta={"format": "WEBP", "chunks": []})
    pos = 12
    while pos < total:
        if pos + 8 > len(data):
            out.halt, out.more = "TRUNCATED_STREAM", True
            break
        fourcc, length = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        end = pos + 8 + length + (length & 1)
        if fourcc not in WEBP_CHUNKS or end > total:
            out.halt = "INVALID_CHUNK"
            break
        out.end = pos + 8  # the header is verified; the payload is, once the next header lines up
        if end > len(data):
            out.halt, out.more = "TRUNCATED_STREAM", True
            break
        payload = data[pos + 8 : pos + 8 + length]
        if fourcc == b"VP8X" and length >= 10:
            out.meta.update(width=1 + int.from_bytes(payload[4:7], "little"),
                            height=1 + int.from_bytes(payload[7:10], "little"), animated=bool(payload[0] & 2))
        elif fourcc == b"VP8 " and length >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            out.meta.setdefault("width", struct.unpack_from("<H", payload, 6)[0] & 0x3FFF)
            out.meta.setdefault("height", struct.unpack_from("<H", payload, 8)[0] & 0x3FFF)
        elif fourcc == b"VP8L" and length >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            out.meta.setdefault("width", (bits & 0x3FFF) + 1)
            out.meta.setdefault("height", ((bits >> 14) & 0x3FFF) + 1)
        elif fourcc == b"EXIF":
            text, intact, valid = exif_text(payload)
            out.text.update(text)
            if not intact:
                out.end, out.halt = pos + 8 + valid, "EXIF_STRINGS_INVALID"
                break
        out.meta["chunks"].append(fourcc.decode("latin-1").strip())
        pos = out.end = end
    walked = pos == total
    out.validation = {"container_walk_closed": walked, "payload_checksum": NO_CHECKSUM}
    if "EXIF" in out.meta["chunks"] or out.halt == "EXIF_STRINGS_INVALID":
        out.validation["exif_strings_valid"] = out.halt != "EXIF_STRINGS_INVALID"
    if walked:
        try:
            with Image.open(io.BytesIO(data[:total])) as image:
                frames = getattr(image, "n_frames", 1)
                for index in range(frames):
                    image.seek(index)
                    image.load()
                image.seek(0)
                first = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                out.meta.update(frames=frames, mode=first.mode)
                if frames == 1:
                    out.meta["pixel_sha256"] = hashlib.sha256(first.tobytes()).hexdigest()
                else:
                    out.meta.update(animated=True, pixel_digest="UNSUPPORTED",
                                    pixel_digest_unsupported_reason="ANIMATED_WEBP_REQUIRES_CONTAINER_IDENTITY")
            out.validation["decodes"] = True
            out.complete = True
        except Exception as exc:  # noqa: BLE001 - any decoder failure means not intact
            out.validation["decodes"] = False
            out.halt = f"PIXEL_DECODE_FAILED: {type(exc).__name__}"
    return out


# -- FLAC -------------------------------------------------------------------------------------------------------

_CRC8 = [0] * 256
_CRC16 = [0] * 256
for _n in range(256):
    _c8, _c16 = _n, _n << 8
    for _ in range(8):
        _c8 = ((_c8 << 1) ^ 0x07) & 0xFF if _c8 & 0x80 else (_c8 << 1) & 0xFF
        _c16 = ((_c16 << 1) ^ 0x8005) & 0xFFFF if _c16 & 0x8000 else (_c16 << 1) & 0xFFFF
    _CRC8[_n], _CRC16[_n] = _c8, _c16
_BLOCK_SIZES = {1: 192, **{n: 576 << (n - 2) for n in range(2, 6)}, **{n: 256 << (n - 8) for n in range(8, 16)}}


def _crc16(data: bytes, crc: int = 0) -> int:
    table = _CRC16
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ table[(crc >> 8) ^ byte]
    return crc


def _flac_frame_header(data: bytes, pos: int, variable: int) -> tuple[int, int, int] | None:
    """(header length, block size, frame or sample number) of a valid frame header at ``pos``, else None."""
    if pos + 6 > len(data) or data[pos] != 0xFF or data[pos + 1] != 0xF8 | variable:
        return None
    size_code, rate_code = data[pos + 2] >> 4, data[pos + 2] & 0x0F
    if size_code == 0 or rate_code == 15 or data[pos + 3] & 1 or (data[pos + 3] >> 4) > 10:
        return None
    first = data[pos + 4]
    extra = 0 if first < 0x80 else next((n for n, mask in enumerate((0xE0, 0xF0, 0xF8, 0xFC, 0xFE, 0xFF), 1)
                                         if first & mask == (mask << 1) & 0xFF), -1)
    if extra < 0 or pos + 5 + extra > len(data):
        return None
    number = first if not extra else first & (0x3F >> extra)
    for byte in data[pos + 5 : pos + 5 + extra]:
        if byte & 0xC0 != 0x80:
            return None
        number = number << 6 | byte & 0x3F
    at = pos + 5 + extra
    if size_code in (6, 7):
        width = size_code - 5
        if at + width > len(data):
            return None
        block = int.from_bytes(data[at : at + width], "big") + 1
        at += width
    else:
        block = _BLOCK_SIZES[size_code]
    at += {12: 1, 13: 2, 14: 2}.get(rate_code, 0)
    if at + 1 > len(data):
        return None
    crc = 0
    for byte in data[pos:at]:
        crc = _CRC8[crc ^ byte]
    return (at + 1 - pos, block, number) if crc == data[at] else None


def vorbis_comments(block: bytes) -> dict[str, str]:
    text: dict[str, str] = {}
    try:
        (vendor,) = struct.unpack_from("<I", block, 0)
        (count,) = struct.unpack_from("<I", block, 4 + vendor)
        pos = 8 + vendor
        for _ in range(min(count, 1024)):
            (length,) = struct.unpack_from("<I", block, pos)
            entry = block[pos + 4 : pos + 4 + length]
            if len(entry) != length:
                break
            key, _, value = entry.decode("utf-8").partition("=")
            _json_text(text, key, value)
            pos += 4 + length
    except (struct.error, UnicodeDecodeError):
        pass
    return text


def parse_flac(data: bytes) -> Parsed | None:
    if data[:4] != b"fLaC" or len(data) < 42 or data[4] & 0x7F != 0 or data[5:8] != b"\x00\x00\x22":
        return None
    info = data[8:42]
    min_frame, max_frame = int.from_bytes(info[4:7], "big"), int.from_bytes(info[7:10], "big")
    packed = int.from_bytes(info[10:18], "big")
    rate, channels, bits = packed >> 44, (packed >> 41 & 7) + 1, (packed >> 36 & 31) + 1
    samples = packed & (1 << 36) - 1
    if rate == 0:
        return None
    out = Parsed("FLAC", 4, meta={"format": "FLAC", "sample_rate": rate, "channels": channels,
                                  "bits_per_sample": bits, "total_samples": samples,
                                  "audio_md5": info[18:34].hex(),
                                  "duration_seconds": round(samples / rate, 3) if samples else None})
    pos, last = 4, False
    while not last:
        if pos + 4 > len(data):
            out.halt, out.more = "TRUNCATED_STREAM", True
            return out
        last, kind, length = bool(data[pos] & 0x80), data[pos] & 0x7F, int.from_bytes(data[pos + 1 : pos + 4], "big")
        if kind == 127:
            out.halt = "INVALID_METADATA_BLOCK"
            return out
        if pos + 4 + length > len(data):
            out.halt, out.more = "TRUNCATED_STREAM", True
            return out
        if kind == 4:
            out.text.update(vorbis_comments(data[pos + 4 : pos + 4 + length]))
        pos = out.end = pos + 4 + length
    first = _flac_frame_header(data, pos, 0) or _flac_frame_header(data, pos, 1)
    if first is None:
        out.halt, out.more = "NO_AUDIO_FRAME", pos + 16 > len(data)
        return out
    variable = data[pos + 1] & 1
    limit = max_frame or 1 << 20
    frames = decoded = 0
    while True:
        header = _flac_frame_header(data, pos, variable)
        if header is None:
            out.halt, out.more = "INVALID_FRAME_HEADER", pos + 16 > len(data)
            break
        length, block, _number = header
        final = bool(samples) and decoded + block >= samples
        if bool(samples) and decoded + block > samples:
            out.halt = "SAMPLES_EXCEED_STREAMINFO"
            break
        frame_end = None
        if not final:
            # The next frame's header (sync, CRC-8) delimits this one; the CRC-16 then verifies its bytes.
            at = data.find(b"\xff" + bytes((0xF8 | variable,)), pos + max(length + 2, min_frame))
            while 0 <= at <= pos + limit:
                following = _flac_frame_header(data, at, variable)
                if following is not None and _crc16(data[pos : at - 2]) == int.from_bytes(data[at - 2 : at], "big"):
                    frame_end = at
                    break
                at = data.find(b"\xff" + bytes((0xF8 | variable,)), at + 1)
        if frame_end is None and (final or not samples):
            # The last frame, or a stream without a sample count: the first place where the CRC-16 fits.
            crc = _crc16(data[pos : pos + max(length, min_frame - 2)])
            for i in range(pos + max(length, min_frame - 2), min(len(data) - 1, pos + limit - 1)):
                if crc == data[i] << 8 | data[i + 1]:
                    frame_end = i + 2
                    break
                crc = ((crc << 8) & 0xFFFF) ^ _CRC16[(crc >> 8) ^ data[i]]
            if frame_end is not None:
                final = True
        if frame_end is None:
            out.halt, out.more = "FRAME_CRC_FAILURE", pos + limit > len(data)
            break
        frames, decoded, pos = frames + 1, decoded + block, frame_end
        out.end = pos
        if final:
            break
    out.meta["frames"] = frames
    out.validation = {"frame_crc16_valid": out.halt is None, "frames_verified": frames,
                      "samples_verified": decoded}
    if out.halt is None and samples and decoded < samples:
        out.halt = "SAMPLES_MISSING"
    out.complete = out.halt is None and bool(samples)
    if out.halt is None and not samples:
        out.halt = "TOTAL_SAMPLES_UNKNOWN"
    return out


# -- MP4 --------------------------------------------------------------------------------------------------------


def _boxes(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """(type, payload start, payload end) of the boxes laid end to end in data[start:end]; stops at a bad size."""
    out, pos = [], start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1 and pos + 16 <= end:
            size, header = struct.unpack_from(">Q", data, pos + 8)[0], 16
        if size < header or pos + size > end:
            break
        out.append((kind, pos + header, pos + size))
        pos += size
    return out


def _tiled(data: bytes, start: int, end: int) -> bool:
    """Whether the boxes laid end to end in data[start:end] fill it exactly."""
    found = _boxes(data, start, end)
    return bool(found) and found[-1][2] == end


def mp4_text(data: bytes, start: int, end: int) -> dict[str, str]:
    """Values of ``moov/udta/meta`` keyed by its ``keys`` box (QuickTime ``mdta`` metadata)."""
    text: dict[str, str] = {}
    stack = [(start, end)]
    while stack:
        lo, hi = stack.pop()
        for kind, a, b in _boxes(data, lo, hi):
            if kind in MP4_CONTAINERS:
                stack.append((a, b))
            elif kind == b"meta":
                # A full box in ISO files (4 bytes of version and flags), a plain container in QuickTime ones.
                inner = _boxes(data, a + 4, b) or _boxes(data, a, b)
                names: list[str] = []
                for child, ca, cb in inner:
                    if child == b"keys" and cb - ca >= 8:
                        names = [data[ka:kb].decode("utf-8", "replace") for _, ka, kb in _boxes(data, ca + 8, cb)]
                for child, ca, cb in inner:
                    if child != b"ilst":
                        continue
                    for item, ia, ib in _boxes(data, ca, cb):
                        index = int.from_bytes(item, "big")
                        for value, va, vb in _boxes(data, ia, ib):
                            if value == b"data" and vb - va >= 8 and 1 <= index <= len(names):
                                try:
                                    _json_text(text, names[index - 1], data[va + 8 : vb].decode("utf-8"))
                                except UnicodeDecodeError:
                                    pass
    return text


def parse_mp4(data: bytes) -> Parsed | None:
    if len(data) < 16 or data[4:8] != b"ftyp" or data[8:12] not in MP4_BRANDS:
        return None
    first = struct.unpack_from(">I", data, 0)[0]
    if not 16 <= first <= 1024 or first % 4:
        return None
    out = Parsed("MP4", 0, meta={"format": "MP4", "major_brand": data[8:12].decode("latin-1").strip(), "boxes": []})
    pos = 0
    kinds: list[bytes] = []
    while True:
        if pos + 8 > len(data):
            # A closed movie at EOF needs no unrelated bytes to prove its end.
            out.more = not (pos == len(data) and b"moov" in kinds
                            and (b"mdat" in kinds or b"moof" in kinds))
            break
        size, kind = struct.unpack_from(">I4s", data, pos)
        if kind == b"ftyp" and kinds:
            break  # another file starts here; its metadata belongs to that file
        header = 8
        if size == 1:
            if pos + 16 > len(data):
                out.more = True
                break
            size, header = struct.unpack_from(">Q", data, pos + 8)[0], 16
        if kind not in MP4_TOP or size < header or size > MAX_MEDIA_BYTES:
            break  # no box here: the file ended at pos
        if pos + size > len(data):
            out.end, out.halt, out.more = pos + header, "TRUNCATED_STREAM", True
            out.needed = pos + size
            break
        if kind == b"moov":
            if not _tiled(data, pos + header, pos + size):
                out.end, out.halt = pos + header, "MOVIE_HEADER_INVALID"
                break
            out.text.update(mp4_text(data, pos + header, pos + size))
        kinds.append(kind)
        out.meta["boxes"].append(kind.decode("latin-1").strip())
        pos = out.end = pos + size
    out.validation = {"box_walk_reached_a_clean_end": out.halt is None and not out.more,
                      "movie_header_present": b"moov" in kinds, "media_data_present": b"mdat" in kinds,
                      "last_box_is_structured": bool(kinds) and kinds[-1] in (b"moov", b"mfra"),
                      "payload_checksum": NO_CHECKSUM}
    if out.halt is None and not out.more:
        if b"moov" in kinds and (b"mdat" in kinds or b"moof" in kinds):
            out.complete = True
        else:
            out.halt = "MOVIE_HEADER_OR_MEDIA_DATA_MISSING"
    elif out.halt is None:
        out.halt = "TRUNCATED_STREAM"
    return out


# -- WebM / Matroska --------------------------------------------------------------------------------------------


def _ebml(data: bytes, pos: int, end: int) -> tuple[int, int | None, int] | None:
    """(element id, payload size or None when unknown, payload start) of the EBML element at ``pos``."""
    if pos >= end or not data[pos]:
        return None
    width = 9 - data[pos].bit_length()
    if width > 4 or pos + width >= end or not data[pos + width]:
        return None
    ident = int.from_bytes(data[pos : pos + width], "big")
    size_width = 9 - data[pos + width].bit_length()
    start = pos + width + size_width
    if start > end:
        return None
    raw = int.from_bytes(data[pos + width : start], "big")
    size = raw & (1 << 7 * size_width) - 1
    return ident, None if size == (1 << 7 * size_width) - 1 else size, start


def _elements(data: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    out, pos = [], start
    while pos < end:
        element = _ebml(data, pos, end)
        if element is None or element[1] is None or element[2] + element[1] > end:
            break
        out.append((element[0], element[2], element[2] + element[1]))
        pos = element[2] + element[1]
    return out


def matroska_text(data: bytes, start: int, end: int) -> dict[str, str]:
    text: dict[str, str] = {}
    for tag, ta, tb in _elements(data, start, end):
        for simple, sa, sb in _elements(data, ta, tb) if tag == TAG else []:
            if simple != SIMPLE_TAG:
                continue
            fields = {ident: data[a:b] for ident, a, b in _elements(data, sa, sb)}
            if TAG_NAME in fields and TAG_STRING in fields:
                try:
                    _json_text(text, fields[TAG_NAME].decode("utf-8"), fields[TAG_STRING].decode("utf-8"))
                except UnicodeDecodeError:
                    pass
    return text


def parse_webm(data: bytes) -> Parsed | None:
    if data[:4] != EBML_HEADER:
        return None
    header = _ebml(data, 0, len(data))
    if header is None or header[1] is None or not 4 <= header[1] <= 256 or header[2] + header[1] > len(data):
        return None
    doctype = next((data[a:b] for ident, a, b in _elements(data, header[2], header[2] + header[1])
                    if ident == 0x4282), b"")
    if doctype not in (b"webm", b"matroska"):
        return None
    pos = header[2] + header[1]
    out = Parsed("WEBM", pos, meta={"format": "WEBM", "doctype": doctype.decode(), "elements": []})
    segment = _ebml(data, pos, len(data))
    if segment is None or segment[0] != SEGMENT:
        out.halt, out.more = "NO_SEGMENT", pos + 12 > len(data)
        return out
    _, size, pos = segment
    if size is not None and pos + size > MAX_MEDIA_BYTES:
        return None
    total = None if size is None else pos + size
    out.needed, out.end = total, pos
    while total is None or pos < total:
        element = _ebml(data, pos, len(data))
        if element is None:
            out.more = pos + 12 > len(data)
            break
        ident, length, start = element
        if ident not in MATROSKA_LEVEL1 or length is None or (total is not None and start + length > total):
            break
        out.end = start
        if start + length > len(data):
            out.more = True
            break
        if ident == TAGS:
            children = _elements(data, start, start + length)
            if not children or children[-1][2] != start + length:
                out.halt = "TAGS_INVALID"
                break
            out.text.update(matroska_text(data, start, start + length))
        out.meta["elements"].append(f"{ident:X}")
        pos = out.end = start + length
    out.validation = {"segment_size_declared": total is not None, "container_walk_closed": pos == total,
                      "payload_checksum": NO_CHECKSUM}
    if pos == total and 0x1F43B675 in {int(e, 16) for e in out.meta["elements"]}:
        out.complete = True
    elif out.halt is None:
        out.halt = ("TRUNCATED_STREAM" if out.more else "SEGMENT_SIZE_UNKNOWN" if total is None
                    else "NO_CLUSTER" if pos == total else "INVALID_ELEMENT")
    return out


# -- carving ----------------------------------------------------------------------------------------------------

PARSERS = {"WEBP": parse_webp, "FLAC": parse_flac, "MP4": parse_mp4, "WEBM": parse_webm}
# needle -> (format, offset of the container's first byte before the needle)
NEEDLES = {b"WEBPVP8": ("WEBP", 8), b"fLaC\x00\x00\x00\x22": ("FLAC", 0), b"fLaC\x80\x00\x00\x22": ("FLAC", 0),
           b"ftyp": ("MP4", 4), EBML_HEADER: ("WEBM", 0)}
PREFETCH = 1 << 20  # bytes read first for a candidate that runs past its window


def media_ids() -> dict[str, IdAllocator]:
    return {kind: IdAllocator(f"CARVE-{kind}") for kind in PARSERS}


def embedded_encoding(carve_type: str) -> str:
    """``encoding_detected`` of a prompt or workflow read from a record's own metadata."""
    return f"{carve_type.split('_', 1)[1]}_EMBEDDED_TEXT" if carve_type in MEDIA_TYPES else "PNG_TEXT"


def is_embedded_text(encoding: str) -> bool:
    """Whether a workflow record's text was read from inside an image, video or audio record."""
    return encoding.startswith("PNG_TEXT") or encoding.endswith("_EMBEDDED_TEXT")


@dataclass
class MediaCandidate:
    """A carved container before it is known whether it is ComfyUI evidence. ``data`` is kept only when the
    container carries ComfyUI text or was read from a ComfyUI process; otherwise ``record`` reads it again."""

    parsed: Parsed
    absolute: int
    is_virtual: bool
    process_id: int | None
    physical_mode: bool
    data: bytes | None
    source: ByteSource | None

    @property
    def dimensions(self) -> tuple[int, int] | None:
        meta = self.parsed.meta
        return (meta["width"], meta["height"]) if "width" in meta and "height" in meta else None

    @property
    def keep_always(self) -> bool:
        return bool(self.parsed.text) or self.is_virtual


class MediaCarver:
    """Finds WEBP, MP4, WebM and FLAC containers and walks each. ``claimed`` is shared with the PNG carver, so
    the JSON carver does not carve again the prompt text a container holds."""

    def __init__(self, claimed: dict[Any, SpanSet] | None = None, physical_mode: bool = False,
                 ids: dict[str, IdAllocator] | None = None) -> None:
        self.physical_mode = physical_mode
        self.claimed = claimed if claimed is not None else {}
        self._seen: set[tuple[Any, int]] = set()
        self.ids = ids if ids is not None else media_ids()
        self.found: dict[str, int] = dict.fromkeys(PARSERS, 0)

    def carve_stream(self, buffer: bytes, virtual_address: int | None = None,
                     base_physical_offset: int | None = None, process_id: int | None = None,
                     source: ByteSource | None = None, exclude: SpanSet | None = None) -> list[MediaCandidate]:
        base: int = virtual_address if virtual_address is not None else (base_physical_offset or 0)
        space = process_id if virtual_address is not None else "physical"
        out = []
        for needle, (kind, lead) in NEEDLES.items():
            idx = buffer.find(needle)
            while idx != -1:
                start = idx - lead
                absolute = base + start
                if start >= 0 and (space, absolute) not in self._seen and not (
                        exclude is not None and absolute in exclude):
                    candidate = self._carve_one(kind, buffer, start, absolute, virtual_address is not None,
                                                process_id, source)
                    if candidate is not None:
                        self._seen.add((space, absolute))
                        self.claimed.setdefault(space, SpanSet()).add(absolute, absolute + candidate.parsed.end)
                        self.found[kind] += 1
                        out.append(candidate)
                idx = buffer.find(needle, idx + 1)
        return out

    def _carve_one(self, kind: str, buffer: bytes, start: int, absolute: int, is_virtual: bool,
                   process_id: int | None, source: ByteSource | None) -> MediaCandidate | None:
        data = buffer[start : start + MAX_MEDIA_BYTES]
        parsed = PARSERS[kind](data)
        # A candidate cut by the end of the window is read on from the source, once at its declared size.
        while parsed is not None and parsed.more and source is not None:
            want = min(MAX_MEDIA_BYTES, max(parsed.needed or 0, len(data) + PREFETCH, 2 * len(data)))
            more, _reason = source.read_prefix(absolute, want)
            if len(more) <= len(data):
                break
            data = more
            parsed = PARSERS[kind](data)
        if parsed is None or parsed.end < 16:
            return None
        if not parsed.complete and self.physical_mode and not parsed.text:
            return None  # a torn container without ComfyUI text is one of thousands of browser media pieces
        candidate = MediaCandidate(parsed, absolute, is_virtual, process_id, self.physical_mode, None, source)
        if candidate.keep_always:
            candidate.data = data[: parsed.end]
        return candidate

    def record(self, candidate: MediaCandidate) -> RawArtifactRecord | None:
        """The manifest record of a candidate. Returns None when its bytes can no longer be read."""
        parsed, kind = candidate.parsed, candidate.parsed.kind
        data = candidate.data
        if data is None and candidate.source is not None:
            data, _reason = candidate.source.read_prefix(candidate.absolute, parsed.end)
        if data is None or len(data) < parsed.end:
            return None
        data = data[: parsed.end]
        meta = {k: v for k, v in parsed.meta.items() if k != "pixel_sha256"}
        gaps = []
        if not parsed.complete:
            missing = max(0, (parsed.needed or parsed.end) - parsed.end)
            # In physical memory the piece that failed (to the declared size, or of unknown length) continues
            # on the next physical page, which need not be the file's next page.
            last = candidate.absolute + (parsed.needed or parsed.end + PAGE_SIZE) - 1
            crosses_a_page = (
                (missing > 0 or parsed.needed is None)
                and (candidate.absolute + parsed.end - 1) // PAGE_SIZE != last // PAGE_SIZE
            )
            reason = (GapReason.NONCONTIGUOUS_PHYSICAL if candidate.physical_mode and crosses_a_page
                      else GapReason.TRUNCATED_STREAM)
            gaps.append(GapDescriptor(parsed.end, missing, reason))
            meta["halt_reason"] = parsed.halt
        record = RawArtifactRecord(
            artifact_id=self.ids[kind](),
            completeness=Completeness.COMPLETE if parsed.complete else Completeness.PARTIAL,
            link_status=LinkClass.UNASSOCIATED,
            associated_workflow_id=None,
            ambiguous_candidate_workflows=[],
            is_derived=False,
            carve_type=ENCODED[kind] if parsed.complete else PARTIAL[kind],
            source_layer=f"virtual_Process{candidate.process_id}" if candidate.is_virtual else "physical",
            process_id=candidate.process_id if candidate.is_virtual else None,
            virtual_address=_hex_addr(candidate.absolute) if candidate.is_virtual else None,
            physical_offsets=[] if candidate.is_virtual else [_hex_addr(candidate.absolute) or ""],
            recovered_bytes=len(data),
            container_sha256=hashlib.sha256(data).hexdigest() if parsed.complete else None,
            pixel_sha256=parsed.meta.get("pixel_sha256"),
            pixel_metadata=meta,
            validation={**parsed.validation, "has_gaps": bool(gaps)},
            gaps=gaps,
            raw_file_paths=[],
            derived_preview_path=None,
            derived_preview_metadata=None,
            spans=[(0, data)],
            embedded_text=dict(parsed.text),
            dimensions=candidate.dimensions,
        )
        return record
