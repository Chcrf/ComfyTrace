"""Deterministic streaming carvers for PNG images and ComfyUI prompt graphs.

Raw evidence is only ever genuine bytes. Where bytes are missing, a GapDescriptor records it;
nothing is padded. Partial pixel previews are produced separately as DERIVED material.
"""

from __future__ import annotations

import bisect
import datetime as dt
import functools
import hashlib
import io
import json
import logging
import mmap
import re
import struct
import time
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeGuard

from comfytrace.exceptions import IngestionError
from comfytrace.fragments import (
    ESCAPED_ANCHORS,
    PAGE,
    TextFragment,
    carve_fragments,
    escaped_literals,
    parse_fragment,
    ui_parameters,
    ui_workflow_nodes,
    unescape,
)
from comfytrace.models import (
    CarveType,
    Completeness,
    GapDescriptor,
    GapReason,
    LinkClass,
    RawArtifactRecord,
    WorkflowRecord,
)

log = logging.getLogger(__name__)

BLOCK_SIZE = 32 * 1024 * 1024
BLOCK_OVERLAP = 64 * 1024
PAGE_SIZE = 4096
MAX_PNG_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_BYTES = 256 * 1024 * 1024
MAX_JSON_CANDIDATE_BYTES = 10 * 1024 * 1024
JSON_LOOKBACK_BYTES = 1024 * 1024

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")
METADATA_KEYS = ("prompt", "workflow", "prompt_id", "client_id")

UTF8_ANCHORS = (b'"class_type"', b'"inputs"', b'"nodes"')
UTF16_ANCHOR = b'"\x00c\x00l\x00a\x00s\x00s\x00_\x00t\x00y\x00p\x00e\x00"\x00'

PAGE_FAULT_REASONS = frozenset(
    {GapReason.PAGE_NOT_PRESENT, GapReason.TRANSLATION_FAULT, GapReason.PAGED_OUT}
)

# ---------------------------------------------------------------------------------------------
# Byte sources and streaming


class ByteSource(Protocol):
    """Random access to the genuine bytes of one address space."""

    def read_prefix(self, offset: int, length: int) -> tuple[bytes, str | None]:
        """Returns the longest genuinely readable prefix of [offset, offset+length) and, when it is
        shorter than requested, the gap reason that stopped it (None means end of source)."""
        ...

    def read_before(self, offset: int, length: int) -> bytes:
        """Returns the longest genuinely readable suffix of [offset-length, offset)."""
        ...


class PhysicalSource:
    """A raw or LiME physical dump file. Reads never cross the end of a segment/file and never pad."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def read_prefix(self, offset: int, length: int) -> tuple[bytes, str | None]:
        return bytes(self.data[offset : offset + length]), None

    def read_before(self, offset: int, length: int) -> bytes:
        if hasattr(self.data, "read_before"):
            return bytes(self.data.read_before(offset, length))
        return bytes(self.data[max(0, offset - length) : offset])


def _fault_reason(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "SwappedInvalidAddressException":
        return GapReason.PAGED_OUT
    if name == "PagedInvalidAddressException":
        return GapReason.PAGE_NOT_PRESENT
    return GapReason.TRANSLATION_FAULT


def _invalid_address_types() -> tuple[type[BaseException], ...]:
    try:
        from volatility3.framework.exceptions import InvalidAddressException

        return (InvalidAddressException,)
    except ImportError:  # pragma: no cover - volatility3 is a hard dependency
        return ()


class ProcessLayerSource:
    """A Volatility process layer. All reads use pad=False; faults end the prefix instead of padding."""

    def __init__(self, proc_layer: Any) -> None:
        self.layer = proc_layer
        self._faults = _invalid_address_types()

    def _read(self, offset: int, length: int) -> bytes:
        try:
            return self.layer.read(offset, length, pad=False)  # type: ignore[no-any-return]
        finally:
            # Volatility's shared 512-entry cache otherwise retains entire VAD/PNG buffers.
            clear = getattr(self.layer.read, "cache_clear", None)
            if clear is not None:
                clear()

    def read_prefix(self, offset: int, length: int) -> tuple[bytes, str | None]:
        try:
            return self._read(offset, length), None
        except self._faults:
            pass
        out = bytearray()
        cursor = offset
        end = offset + length
        while cursor < end:
            step = min(PAGE_SIZE - (cursor % PAGE_SIZE), end - cursor)
            try:
                out += self._read(cursor, step)
            except self._faults as exc:
                return bytes(out), _fault_reason(exc)
            cursor += step
        return bytes(out), None

    def read_before(self, offset: int, length: int) -> bytes:
        pieces: list[bytes] = []
        cursor = offset
        floor = max(0, offset - length)
        while cursor > floor:
            step = min(cursor - floor, (cursor - 1) % PAGE_SIZE + 1)
            try:
                pieces.append(self._read(cursor - step, step))
            except self._faults:
                break
            cursor -= step
        return b"".join(reversed(pieces))

    def read_spans(
        self, offset: int, length: int
    ) -> tuple[list[tuple[int, bytes]], list[GapDescriptor], str | None]:
        """Genuine spans, interior holes, and why reading stopped before offset + length (if it did)."""
        spans: list[tuple[int, bytes]] = []
        gaps: list[GapDescriptor] = []
        cursor = offset
        for lo, hi in _coalesced_runs(self.layer, offset, length):
            if lo > cursor:
                gaps.append(GapDescriptor(cursor - offset, lo - cursor, GapReason.PAGE_NOT_PRESENT))
            while lo < hi:
                data, reason = self.read_prefix(lo, hi - lo)
                if data:
                    relative = lo - offset
                    if spans and spans[-1][0] + len(spans[-1][1]) == relative:
                        start, previous = spans[-1]
                        spans[-1] = (start, previous + data)
                    else:
                        spans.append((relative, data))
                lo += len(data)
                if reason is None:
                    break
                end = min(hi, (lo // PAGE_SIZE + 1) * PAGE_SIZE)
                gaps.append(GapDescriptor(lo - offset, end - lo, reason))
                lo = end
            cursor = hi
        return spans, gaps, GapReason.PAGE_NOT_PRESENT if cursor < offset + length else None


@dataclass(frozen=True)
class Block:
    """A window of genuine bytes at an absolute address, plus the gaps found before it."""

    address: int
    data: bytes
    gaps: tuple[GapDescriptor, ...] = ()


class StreamReader:
    def __init__(self, dump_path: Path) -> None:
        self.dump_path = Path(dump_path)

    def open_physical(self) -> tuple[mmap.mmap, Any]:
        try:
            handle = open(self.dump_path, "rb")  # noqa: SIM115 - lifetime owned by the caller
        except OSError as exc:
            raise IngestionError(f"cannot open memory image {self.dump_path}: {exc}") from exc
        try:
            return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ), handle
        except (OSError, ValueError) as exc:
            handle.close()
            raise IngestionError(f"cannot map memory image {self.dump_path}: {exc}") from exc

    def stream_physical(
        self, block_size: int = BLOCK_SIZE, overlap: int = BLOCK_OVERLAP
    ) -> Iterator[tuple[int, bytes]]:
        """Streams the physical dump in sliding windows (block_size, with overlap), read from the file: mapping
        and unmapping every page of a window costs more than the read's copy.
        Raises: IngestionError if dump path is unreadable."""
        if overlap >= block_size:
            raise ValueError("overlap must be smaller than block size")
        mapped, handle = self.open_physical()
        try:
            size = len(mapped)
            offset = 0
            while offset < size:
                handle.seek(offset)
                yield offset, handle.read(block_size)
                if offset + block_size >= size:
                    break
                offset += block_size - overlap
        finally:
            mapped.close()
            handle.close()

    def stream_process_vad(
        self,
        proc_layer: Any,
        va_start: int,
        va_end: int,
        block_size: int = BLOCK_SIZE,
        overlap: int = BLOCK_OVERLAP,
    ) -> Iterator[tuple[int, bytes, list[GapDescriptor]]]:
        """Streams a VAD (inclusive bounds) by driving proc_layer.mapping(..., ignore_errors=True).
        Unmapped ranges become PAGE_NOT_PRESENT gaps; read faults become PAGED_OUT/TRANSLATION_FAULT
        gaps. Gap offsets are relative to va_start. Yields (virtual offset, genuine bytes, gaps that
        precede those bytes); trailing gaps arrive with an empty byte string."""
        for block in self.iter_vad_blocks(proc_layer, va_start, va_end, block_size, overlap):
            yield block.address, block.data, list(block.gaps)

    def iter_vad_blocks(
        self,
        proc_layer: Any,
        va_start: int,
        va_end: int,
        block_size: int = BLOCK_SIZE,
        overlap: int = BLOCK_OVERLAP,
    ) -> Iterator[Block]:
        source = ProcessLayerSource(proc_layer)
        pending: list[GapDescriptor] = []
        cursor = va_start
        for run_start, run_end in _coalesced_runs(proc_layer, va_start, va_end - va_start + 1):
            if run_start > cursor:
                pending.append(GapDescriptor(cursor - va_start, run_start - cursor, GapReason.PAGE_NOT_PRESENT))
            offset = run_start
            while offset < run_end:
                length = min(block_size, run_end - offset)
                data, reason = source.read_prefix(offset, length)
                if data:
                    yield Block(offset, data, tuple(pending))
                    pending = []
                if reason is not None:
                    bad_start = offset + len(data)
                    bad_end = min(run_end, (bad_start // PAGE_SIZE + 1) * PAGE_SIZE)
                    pending.append(GapDescriptor(bad_start - va_start, bad_end - bad_start, reason))
                    offset = bad_end
                    continue
                if offset + length >= run_end:
                    break
                offset += max(length - overlap, 1)
            cursor = max(cursor, run_end)
        if cursor <= va_end:
            pending.append(GapDescriptor(cursor - va_start, va_end + 1 - cursor, GapReason.PAGE_NOT_PRESENT))
        if pending:
            yield Block(va_end + 1, b"", tuple(pending))


def _coalesced_runs(proc_layer: Any, start: int, length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for offset, sublength, *_ in proc_layer.mapping(start, length, ignore_errors=True):
        if runs and offset == runs[-1][1]:
            runs[-1] = (runs[-1][0], offset + sublength)
        else:
            runs.append((offset, offset + sublength))
    return runs


# ---------------------------------------------------------------------------------------------
# PNG carving

COLOR_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


@dataclass(frozen=True)
class Ihdr:
    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace: int

    @property
    def bits_per_pixel(self) -> int:
        return COLOR_CHANNELS[self.color_type] * self.bit_depth

    @property
    def filter_bpp(self) -> int:
        """Bytes per complete pixel for scanline filtering (at least one)."""
        return max(1, self.bits_per_pixel // 8)

    @property
    def row_bytes(self) -> int:
        return (self.width * self.bits_per_pixel + 7) // 8

    def as_meta(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "color_type": self.color_type,
            "interlace": self.interlace,
        }

    @staticmethod
    def parse(data: bytes) -> Ihdr | None:
        if len(data) != 13:
            return None
        width, height, depth, ctype, comp, filt, interlace = struct.unpack(">IIBBBBB", data)
        valid_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
        if not (0 < width < 2**31 and 0 < height < 2**31) or depth not in valid_depths.get(ctype, ()):
            return None
        if comp != 0 or filt != 0 or interlace not in (0, 1):
            return None
        return Ihdr(width, height, depth, ctype, interlace)


@dataclass
class _PngParse:
    end: int
    complete: bool
    ihdr: Ihdr
    idat: bytes
    palette: bytes | None
    text: dict[str, str]
    halt_reason: str | None
    lost_bytes: int
    crc32_valid: bool = True
    spans: list[tuple[int, bytes]] = field(default_factory=list)
    gaps: list[GapDescriptor] = field(default_factory=list)
    transparency: bool = False


def _chunk_type_ok(ctype: bytes) -> bool:
    return len(ctype) == 4 and all(65 <= c <= 90 or 97 <= c <= 122 for c in ctype)


def _decode_text_chunk(ctype: bytes, data: bytes) -> tuple[str, str] | None:
    try:
        if ctype == b"tEXt":
            key, sep, value = data.partition(b"\x00")
            if not sep:
                return None
            return key.decode("latin-1"), value.decode("utf-8", errors="strict")
        if ctype == b"zTXt":
            key, sep, rest = data.partition(b"\x00")
            if not sep or not rest or rest[0] != 0:
                return None
            return key.decode("latin-1"), _inflate_text(rest[1:]).decode("utf-8", errors="strict")
        if ctype == b"iTXt":
            key, sep, rest = data.partition(b"\x00")
            if not sep or len(rest) < 2:
                return None
            compressed, method = rest[0], rest[1]
            if compressed not in (0, 1) or method != 0:
                return None
            _lang, sep_lang, rest = rest[2:].partition(b"\x00")
            _tkey, sep_tkey, text = rest.partition(b"\x00")
            if not sep_lang or not sep_tkey:
                return None
            if compressed:
                text = _inflate_text(text)
            return key.decode("latin-1"), text.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, zlib.error, IndexError, ValueError):
        return None
    return None


def _inflate_text(data: bytes) -> bytes:
    """Inflates a compressed text chunk, refusing truncated streams and output beyond the JSON candidate ceiling."""
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, MAX_JSON_CANDIDATE_BYTES + 1)
    if len(out) > MAX_JSON_CANDIDATE_BYTES or inflater.unconsumed_tail:
        raise ValueError("compressed text chunk exceeds the size ceiling")
    if not inflater.eof:
        raise ValueError("compressed text chunk is truncated")
    return out


def parse_png(
    data: bytes, physical_mode: bool, tail_reason: str | None, origin: int = 0,
    spans: list[tuple[int, bytes]] | None = None, gaps: list[GapDescriptor] | None = None,
) -> _PngParse | None:
    """Parses genuine spans without joining across holes. After a lost chunk header, process mode
    resumes only at a recognizable chunk with a verified CRC; previews stop at the first hole."""
    spans = spans if spans is not None else [(0, data)]
    gaps = list(gaps or ())
    starts = [offset for offset, _ in spans]
    available_end = max((offset + len(piece) for offset, piece in spans), default=0)

    def prefix(offset: int, length: int) -> bytes:
        i = bisect.bisect_right(starts, offset) - 1
        if i < 0:
            return b""
        lo, piece = spans[i]
        return piece[offset - lo:offset - lo + length]

    header = prefix(0, 33)
    if len(header) < 33 or header[:8] != PNG_MAGIC:
        return None
    length, ctype = struct.unpack(">I4s", header[8:16])
    if ctype != b"IHDR" or length != 13 or zlib.crc32(header[12:29]) != struct.unpack(">I", header[29:33])[0]:
        return None
    ihdr = Ihdr.parse(header[16:29])
    if ihdr is None:
        return None
    pos = 33
    idat = bytearray()
    palette = None
    text: dict[str, str] = {}
    crc_valid = True
    preview_stopped = False
    transparency = False

    def result(end: int, complete: bool, reason: str | None, lost: int = 0) -> _PngParse:
        kept = [(lo, piece[:end - lo]) for lo, piece in spans if lo < end]
        missing = [gap for gap in gaps if gap.relative_offset < end]
        if lost:
            gap_reason = reason if reason in PAGE_FAULT_REASONS else _tear_reason(origin + end,
                                                                                 origin + end + lost, physical_mode)
            missing.append(GapDescriptor(end, lost, gap_reason or GapReason.CRC_FAILURE))
        return _PngParse(end, complete and not missing, ihdr, bytes(idat), palette, text, reason, lost,
                         crc_valid, kept, missing, transparency)

    while True:
        header = prefix(pos, 8)
        if len(header) < 8:
            gap = next((g for g in gaps if g.relative_offset < pos + 8
                        and pos < g.relative_offset + g.length_bytes), None)
            if gap is not None:
                # A lost chunk header breaks the chain; bytes after the hole may belong to anything.
                crc_valid = False
                return result(gap.relative_offset + gap.length_bytes, False, gap.reason)
            crc_valid = crc_valid and not header
            return result(available_end, False, tail_reason or GapReason.TRUNCATED_STREAM, 8 - len(header))
        length, ctype = struct.unpack(">I4s", header)
        if length >= 2**31 or not _chunk_type_ok(ctype) or ctype == b"IHDR":
            crc_valid = False
            reason = _tear_reason(origin + pos, origin + pos + 8, physical_mode)
            return result(pos, False, reason, 8)
        chunk_end = pos + 12 + length
        chunk = prefix(pos, 12 + length)
        if len(chunk) < 12 + length:
            if not preview_stopped and ctype == b"IDAT":
                idat += chunk[8:8 + length]
            crc_valid = False
            crossed = [g for g in gaps if pos <= g.relative_offset < chunk_end]
            if crossed and chunk_end <= available_end:
                preview_stopped = True
                pos = chunk_end
                continue
            if crossed:
                tail_gap = next((g for g in crossed if g.relative_offset >= available_end), None)
                reason = tail_gap.reason if tail_gap is not None else (tail_reason or GapReason.TRUNCATED_STREAM)
                return result(available_end, False, reason, chunk_end - available_end)
            return result(available_end, False, tail_reason or GapReason.TRUNCATED_STREAM, chunk_end - available_end)
        body = chunk[8:-4]
        if zlib.crc32(chunk[4:-4]) != struct.unpack(">I", chunk[-4:])[0]:
            crc_valid = False
            return result(pos, False, _tear_reason(origin + pos, origin + chunk_end, physical_mode), 12 + length)
        if (ctype == b"IEND" and length != 0) or (ctype[0] < 97 and ctype not in (b"IHDR", b"PLTE", b"IDAT", b"IEND")):
            return result(chunk_end, False, "INVALID_CRITICAL_CHUNK")
        if ctype == b"IDAT" and not preview_stopped:
            idat += body
        elif ctype == b"PLTE":
            palette = body
        elif ctype == b"tRNS":
            transparency = True
        elif ctype in TEXT_CHUNKS:
            decoded = _decode_text_chunk(ctype, body)
            if decoded and decoded[0] in METADATA_KEYS:
                text.setdefault(decoded[0], decoded[1])
        pos = chunk_end
        if ctype == b"IEND":
            return result(pos, True, gaps[0].reason if gaps else None)


def _tear_reason(start: int, end: int, physical_mode: bool) -> str:
    """In physical carving, a failure in a chunk that crosses a 4 KB frame boundary of the dump (absolute
    offsets) is reported as a page tear; one inside a single frame is a plain CRC failure."""
    if physical_mode and start // PAGE_SIZE != (end - 1) // PAGE_SIZE:
        return GapReason.NONCONTIGUOUS_PHYSICAL
    return GapReason.CRC_FAILURE


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def defilter_rows(raw: bytes, ihdr: Ihdr, rows: int) -> bytes:
    """Reverses PNG scanline filters for `rows` complete rows of a non-interlaced image."""
    stride = ihdr.row_bytes
    bpp = ihdr.filter_bpp
    out = bytearray()
    prev = bytearray(stride)
    for r in range(rows):
        base = r * (stride + 1)
        ftype = raw[base]
        line = bytearray(raw[base + 1 : base + 1 + stride])
        if ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                upleft = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(left, prev[i], upleft)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"invalid filter type {ftype} on row {r}")
        out += line
        prev = line
    return bytes(out)


_RAW_MODES = {
    (0, 1): ("L", "1"), (0, 2): ("L", "L;2"), (0, 4): ("L", "L;4"), (0, 8): ("L", "L"),
    (2, 8): ("RGB", "RGB"), (3, 1): ("P", "P;1"), (3, 2): ("P", "P;2"), (3, 4): ("P", "P;4"),
    (3, 8): ("P", "P"), (4, 8): ("LA", "LA"), (6, 8): ("RGBA", "RGBA"),
}


def render_rows_png(pixels: bytes, ihdr: Ihdr, rows: int, palette: bytes | None) -> bytes:
    """Encodes defiltered rows as a DERIVED preview PNG. 16-bit samples keep their high byte."""
    from PIL import Image

    key = (ihdr.color_type, ihdr.bit_depth)
    if ihdr.bit_depth == 16:
        pixels = pixels[0::2]
        key = (ihdr.color_type, 8)
    mode, rawmode = _RAW_MODES[key]
    img = Image.frombytes(mode, (ihdr.width, rows), pixels, "raw", rawmode)
    if mode == "P" and palette:
        img.putpalette(palette)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLES_BE16_SCHEME = "png-samples-be16-v1"


class PixelDigestUnsupported(ValueError):
    """A CRC-verified PNG that Pillow decodes, but whose full sample identity cannot be digested honestly.
    `meta` is the pixel metadata to record in place of a digest; the container itself stays valid evidence."""

    def __init__(self, reason: str, meta: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.meta = {**meta, "pixel_digest": "UNSUPPORTED", "pixel_digest_unsupported_reason": reason}


def _png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """(type, body) of each chunk of an already CRC-verified PNG."""
    pos = len(PNG_MAGIC)
    while pos + 8 <= len(data):
        length, ctype = struct.unpack_from(">I4s", data, pos)
        yield ctype, data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IEND":
            return


def _sixteen_bit_sample_digest(png_bytes: bytes, ihdr: Ihdr, img: Any, meta: dict[str, Any]) -> str:
    """SHA-256 of the big-endian interleaved 16-bit samples exactly as the PNG stores them (filters reversed,
    no filter bytes). Pillow's RGB/RGBA/LA 16-bit modes keep only the high byte, so its pixels cannot be
    hashed. Cross-checked against Pillow's own decode of the high bytes before anything is trusted."""
    import numpy as np

    if ihdr.interlace:
        raise PixelDigestUnsupported("ADAM7_INTERLACED_16BIT", meta)
    expected = ihdr.height * (1 + ihdr.row_bytes)
    if expected > MAX_PREVIEW_BYTES:
        raise PixelDigestUnsupported("IMAGE_TOO_LARGE_FOR_SAMPLE_DIGEST", meta)
    idat = b"".join(body for ctype, body in _png_chunks(png_bytes) if ctype == b"IDAT")
    inflater = zlib.decompressobj()
    try:
        stream = inflater.decompress(idat, expected + 1)
    except zlib.error as exc:
        raise PixelDigestUnsupported("IDAT_INFLATE_FAILED", meta) from exc
    if len(stream) != expected or not inflater.eof or inflater.unused_data:
        raise PixelDigestUnsupported("IDAT_LENGTH_MISMATCH", meta)
    try:
        samples = defilter_rows(stream, ihdr, ihdr.height)
    except ValueError as exc:
        raise PixelDigestUnsupported("INVALID_SCANLINE_FILTER", meta) from exc
    channels = COLOR_CHANNELS[ihdr.color_type]
    high = (np.frombuffer(samples, dtype=">u2").astype(np.uint16) >> 8).astype(np.uint8)
    if channels == 2:  # Pillow expands 16-bit grey+alpha to RGBA
        high = high.reshape(-1, 2)
        high = np.stack([high[:, 0], high[:, 0], high[:, 0], high[:, 1]], axis=1)
    if high.tobytes() != img.tobytes():
        raise PixelDigestUnsupported("HIGH_BYTE_CROSSCHECK_FAILED", meta)
    return hashlib.sha256(samples).hexdigest()


def native_pixel_digest(png_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Validates a complete PNG with Pillow and hashes its decoded pixels in native mode.

    Ordinary 8-bit and lower-depth images, and native 16-bit greyscale, hash `img.tobytes()`.
    Paletted images hash rendered RGBA to include palette colors and transparency.
    16-bit RGB/RGBA/grey+alpha would be silently cut to 8 bits there, so they hash the PNG's full big-endian
    samples instead. Animated PNG identity is represented by the container hash.
    Raises PixelDigestUnsupported (a ValueError) when the full sample identity cannot be digested."""
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as probe:
        probe.verify()
    with Image.open(io.BytesIO(png_bytes)) as img:
        img.load()
        meta: dict[str, Any] = {"mode": img.mode, "width": img.width, "height": img.height}
        if img.format != "PNG":
            return hashlib.sha256(img.tobytes()).hexdigest(), meta
        ihdr = Ihdr.parse(png_bytes[16:29])
        listed = list(_png_chunks(png_bytes))
        chunks = dict(reversed(listed))  # first chunk of each type
        animation = chunks.get(b"acTL")
        if animation is not None:
            frames = struct.unpack(">I", animation[:4])[0] if len(animation) >= 4 else None
            decode_count = getattr(img, "n_frames", 1)
            controls = sum(1 for ctype, _ in listed if ctype == b"fcTL")
            if type(frames) is not int or frames < 1 or frames != controls:
                # acTL must declare exactly the frames the container holds (one fcTL each), else it is not intact.
                raise ValueError(f"acTL declares {frames} frames; the container holds {controls}")
            checked = (decode_count <= 64 and decode_count * img.width * img.height * 4 <= MAX_PREVIEW_BYTES)
            if checked:
                for index in range(1, decode_count):
                    img.seek(index)
                    img.load()
            meta.update(animated=True, frames=frames, animation_scanlines_verified=checked)
            raise PixelDigestUnsupported("ANIMATED_PNG_REQUIRES_CONTAINER_IDENTITY", meta)
        if img.mode == "P":
            meta["pixel_digest"] = "png-palette-rgba-v1"
            return hashlib.sha256(img.convert("RGBA").tobytes()).hexdigest(), meta
        # A tRNS chunk outside palette images names one fully transparent colour that the samples do not carry:
        # equal samples with and without it are different pictures.
        key = chunks.get(b"tRNS")
        if ihdr is None or ihdr.bit_depth != 16 or (ihdr.color_type == 0 and img.mode == "I;16"):
            return _with_transparency_key(hashlib.sha256(img.tobytes()).hexdigest(), key, meta), meta
        meta.update(bit_depth=16, color_type=ihdr.color_type, pixel_digest=SAMPLES_BE16_SCHEME)
        if ihdr.color_type == 0:
            raise PixelDigestUnsupported("GREY16_NOT_I16_MODE", meta)
        return _with_transparency_key(_sixteen_bit_sample_digest(png_bytes, ihdr, img, meta), key, meta), meta


def _with_transparency_key(digest: str, key: bytes | None, meta: dict[str, Any]) -> str:
    if key is None:
        return digest
    meta["transparency_key"] = key.hex()
    return hashlib.sha256(bytes.fromhex(digest) + b"tRNS" + key).hexdigest()


class IdAllocator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"{self.prefix}-{self.count:03d}"


def _hex_addr(value: int | None) -> str | None:
    return None if value is None else f"0x{value:016X}"


class PngCarver:
    def __init__(self, id_factory: IdAllocator | None = None, physical_mode: bool = False) -> None:
        self.next_id = id_factory or IdAllocator("CARVE-PNG")
        self.physical_mode = physical_mode
        self._seen: set[tuple[Any, int]] = set()
        self.claimed: dict[Any, SpanSet] = {}

    def carve_stream(
        self,
        buffer: bytes,
        virtual_address: int | None = None,
        base_physical_offset: int | None = None,
        process_id: int | None = None,
        source: ByteSource | None = None,
        exclude: SpanSet | None = None,
    ) -> list[RawArtifactRecord]:
        """Carves intact and partial PNGs with CRC-32 validation and metadata extraction.
        `source` lets a candidate that runs past the end of this window be re-read in full.
        A signature inside `exclude` (absolute ranges carved elsewhere) is skipped.
        Raises: None."""
        base: int = virtual_address if virtual_address is not None else (base_physical_offset or 0)
        space = process_id if virtual_address is not None else "physical"
        records = []
        idx = buffer.find(PNG_MAGIC)
        while idx != -1:
            absolute = base + idx
            if (space, absolute) not in self._seen and not (exclude is not None and absolute in exclude):
                record = self._carve_one(buffer, idx, absolute, virtual_address is not None, process_id, source)
                if record is not None:
                    self._seen.add((space, absolute))
                    for offset, piece in record.spans:
                        self.claimed.setdefault(space, SpanSet()).add(absolute + offset, absolute + offset + len(piece))
                    records.append(record)
            idx = buffer.find(PNG_MAGIC, idx + 1)
        return records

    def _carve_one(
        self,
        buffer: bytes,
        idx: int,
        absolute: int,
        is_virtual: bool,
        process_id: int | None,
        source: ByteSource | None,
    ) -> RawArtifactRecord | None:
        data = buffer[idx : idx + MAX_PNG_BYTES]
        parsed = parse_png(data, self.physical_mode, None, absolute)
        ran_off_window = (parsed is None and len(data) < 64) or (
            parsed is not None and parsed.halt_reason == GapReason.TRUNCATED_STREAM
        )
        if ran_off_window and source is not None:
            if is_virtual and not self.physical_mode and isinstance(source, ProcessLayerSource):
                spans, gaps, tail = source.read_spans(absolute, MAX_PNG_BYTES)
                parsed = parse_png(data, False, tail, absolute, spans, gaps)
            else:
                data, tail_reason = source.read_prefix(absolute, MAX_PNG_BYTES)
                parsed = parse_png(data, self.physical_mode, tail_reason, absolute)
        if parsed is None:
            return None
        return self.record_from(parsed, absolute, is_virtual, process_id)

    def record_from(self, parsed: _PngParse, absolute: int, is_virtual: bool, process_id: int | None,
                    artifact_id: str | None = None) -> RawArtifactRecord:
        """The record of a parsed PNG: COMPLETE when every chunk verified and the pixels decode, else PARTIAL with
        its gaps and a DERIVED preview of the rows that decode."""
        verified = parsed.spans[0][1]
        artifact_id = artifact_id or self.next_id()
        record = RawArtifactRecord(
            artifact_id=artifact_id,
            completeness=Completeness.PARTIAL,
            link_status=LinkClass.UNASSOCIATED,
            associated_workflow_id=None,
            ambiguous_candidate_workflows=[],
            is_derived=False,
            carve_type=CarveType.PARTIAL_PNG,
            source_layer=f"virtual_Process{process_id}" if is_virtual else "physical",
            process_id=process_id if is_virtual else None,
            virtual_address=_hex_addr(absolute) if is_virtual else None,
            physical_offsets=[] if is_virtual else [_hex_addr(absolute) or ""],
            recovered_bytes=sum(len(piece) for _, piece in parsed.spans),
            container_sha256=None,
            pixel_sha256=None,
            pixel_metadata=None,
            validation={"crc32_valid": parsed.crc32_valid, "scanlines_valid": False, "has_gaps": bool(parsed.gaps)},
            gaps=parsed.gaps,
            raw_file_paths=[],
            derived_preview_path=None,
            derived_preview_metadata=None,
            spans=parsed.spans,
            embedded_text=parsed.text,
            dimensions=(parsed.ihdr.width, parsed.ihdr.height),
        )
        if parsed.complete:
            try:
                pixel_sha, meta = native_pixel_digest(verified)
            except PixelDigestUnsupported as exc:
                # The container is CRC-verified and Pillow decodes it; only the sample digest is unavailable.
                log.debug("%s: pixel digest unsupported: %s", artifact_id, exc.reason)
                pixel_sha, meta = None, exc.meta
                self._mark_complete(record, verified, pixel_sha, meta)
                return record
            except Exception as exc:  # noqa: BLE001 - any decoder failure means not intact
                log.debug("%s: CRC-valid container failed pixel decode: %s", artifact_id, exc)
                parsed.complete = False
                parsed.halt_reason = "PIXEL_DECODE_FAILED"
            else:
                self._mark_complete(record, verified, pixel_sha, meta)
                return record
        self._attach_preview(record, parsed)
        return record

    @staticmethod
    def _mark_complete(record: RawArtifactRecord, verified: bytes, pixel_sha: str | None,
                       meta: dict[str, Any]) -> None:
        record.completeness = Completeness.COMPLETE
        record.carve_type = CarveType.ENCODED_PNG
        record.container_sha256 = hashlib.sha256(verified).hexdigest()
        record.pixel_sha256 = pixel_sha
        record.pixel_metadata = meta
        record.validation = {"crc32_valid": True, "scanlines_valid": meta.get("animation_scanlines_verified", True),
                             "has_gaps": False}

    def _attach_preview(self, record: RawArtifactRecord, parsed: _PngParse) -> None:
        ihdr = parsed.ihdr
        meta: dict[str, Any] = {"decoded_rows": 0, "total_rows": ihdr.height, "scanline_halt_reason": None}
        record.derived_preview_metadata = meta
        if ihdr.bit_depth == 16:
            meta["sample_precision_reduced"] = True
        if parsed.transparency:
            meta["transparency_dropped"] = True
        if ihdr.interlace:
            meta["scanline_halt_reason"] = "ADAM7_INTERLACED_PARTIAL_PREVIEW_UNSUPPORTED"
            return
        pixels, rows, halt = self.decompress_partial_idat(parsed.idat, {**ihdr.as_meta()})
        meta["decoded_rows"] = rows
        gap_reason = record.gaps[0].reason if record.gaps else None
        if halt == "INCOMPLETE_ZLIB_STREAM" and gap_reason is not None:
            halt = "PAGE_GAP_ENCOUNTERED" if gap_reason in PAGE_FAULT_REASONS else str(gap_reason)
        meta["scanline_halt_reason"] = halt or parsed.halt_reason
        if rows:
            try:
                record.preview_png = render_rows_png(pixels, ihdr, rows, parsed.palette)
            except Exception as exc:  # noqa: BLE001 - a preview is optional derived material
                meta["scanline_halt_reason"] = f"PREVIEW_RENDER_FAILED: {exc}"

    def decompress_partial_idat(self, idat_bytes: bytes, ihdr_meta: dict[str, Any]) -> tuple[bytes, int, str | None]:
        """Decompresses a possibly truncated IDAT stream up to the first error with zlib.decompressobj().
        Returns (defiltered rows, decoded row count, halt reason or None when the stream ended cleanly).
        Raises: None."""
        ihdr = Ihdr(ihdr_meta["width"], ihdr_meta["height"], ihdr_meta["bit_depth"],
                    ihdr_meta["color_type"], ihdr_meta.get("interlace", 0))
        if ihdr.interlace:
            return b"", 0, "ADAM7_INTERLACED_PARTIAL_PREVIEW_UNSUPPORTED"
        expected = ihdr.height * (1 + ihdr.row_bytes)
        limit = min(expected, MAX_PREVIEW_BYTES)
        inflater = zlib.decompressobj()
        out = bytearray()
        halt: str | None = None
        step = 4096
        for i in range(0, len(idat_bytes), step):
            checkpoint = inflater.copy()
            try:
                out += inflater.decompress(idat_bytes[i : i + step], limit - len(out))
            except zlib.error:
                inflater = checkpoint
                for byte in idat_bytes[i:i + step]:
                    try:
                        out += inflater.decompress(bytes((byte,)), limit - len(out))
                    except zlib.error:
                        break
                    if len(out) >= limit:
                        break
                halt = "ZLIB_DATA_ERROR"
                break
            if len(out) >= limit:
                break
        if halt is None and len(out) >= limit:
            halt = None if limit == expected else "PREVIEW_SIZE_LIMIT"
        elif halt is None and not inflater.eof:
            halt = "INCOMPLETE_ZLIB_STREAM"
        rows = min(len(out) // (1 + ihdr.row_bytes), ihdr.height)
        try:
            return defilter_rows(bytes(out), ihdr, rows), rows, halt
        except ValueError:
            good = 0
            for r in range(rows):
                if out[r * (1 + ihdr.row_bytes)] > 4:
                    break
                good += 1
            return defilter_rows(bytes(out), ihdr, good), good, "INVALID_FILTER_TYPE"


# ---------------------------------------------------------------------------------------------
# Workflow carving

SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
CHECKPOINT_INPUTS = ("ckpt_name", "unet_name", "model_name")
PARAMETER_ORDER = (
    "prompt_id", "seed", "sampler_name", "scheduler", "steps", "cfg_scale", "model_name", "positive_prompt",
    "negative_prompt", "width", "height", "batch_size", "filename_prefix",
)
COMPLETE_PARAMETERS = ("seed", "sampler_name", "steps", "model_name", "positive_prompt")
IMAGE_EDIT_INPUTS = {
    "Canny": ("image",), "ImageBlend": ("image1", "image2"),
    "ImageCompositeMasked": ("destination", "source"), "ImageCrop": ("image",),
    "ImageInvert": ("image",), "ImagePadForOutpaint": ("image",), "ImageScale": ("image",),
    "ImageScaleBy": ("image",), "ImageUpscaleWithModel": ("image",),
}


def _is_api_graph(obj: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs"), dict)
                for v in obj.values())
    )


def _is_api_node(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("class_type"), str) and isinstance(obj.get("inputs"), dict)


def _is_ui_workflow(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("nodes"), list) and any(
        isinstance(n, dict) and "type" in n for n in obj["nodes"]
    )


_QUEUE_UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
_QUEUE_ID = re.compile(_QUEUE_UUID)
_QUEUE_START = re.compile(r'\[\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,\s*"' + _QUEUE_UUID + r'"\s*,\s*\{')


def _history_cached_nodes(status: Any, prompt_id: Any) -> list[str] | None:
    """The nodes a serialized /history entry's own ``execution_cached`` message names (messages of another prompt
    are ignored), or None when the entry carries no such message."""
    messages = status.get("messages") if isinstance(status, dict) else None
    found: list[str] | None = None
    for message in messages if isinstance(messages, list) else []:
        if (isinstance(message, list) and len(message) == 2 and message[0] == "execution_cached"
                and isinstance(message[1], dict) and isinstance(message[1].get("nodes"), list)
                and message[1].get("prompt_id") in (None, prompt_id)):
            found = [*(found or []), *(str(node) for node in message[1]["nodes"])]
    return found


def extract_prompt_graph(obj: Any) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Classifies a decoded JSON object. Returns (prompt_format, graph, runtime keys) or None if it is
    not ComfyUI material. Handles /prompt request bodies and /history entries that wrap the graph."""
    if _is_api_graph(obj):
        return "API_GRAPH", obj, {}
    if (isinstance(obj, list) and len(obj) == 5 and type(obj[0]) in (int, float)
            and isinstance(obj[1], str) and _QUEUE_ID.fullmatch(obj[1]) and _is_api_graph(obj[2])
            and isinstance(obj[3], dict) and isinstance(obj[4], list)
            and all(isinstance(node, str) and node in obj[2] for node in obj[4])):
        return "API_GRAPH", obj[2], {"prompt_id": obj[1]}
    if isinstance(obj, dict):
        runtime = {k: obj[k] for k in ("prompt_id", "client_id") if isinstance(obj.get(k), str)}
        inner = obj.get("prompt")
        if _is_api_graph(inner):
            return "API_GRAPH", inner, runtime
        if isinstance(inner, list) and len(inner) >= 3 and _is_api_graph(inner[2]):
            if isinstance(inner[1], str):
                runtime.setdefault("prompt_id", inner[1])
            if (cached := _history_cached_nodes(obj.get("status"), runtime.get("prompt_id"))) is not None:
                runtime["job_timeline"] = {"cached_nodes": cached}
            return "API_GRAPH", inner[2], runtime
        if _is_ui_workflow(obj):
            return "UI_WORKFLOW", obj, runtime
        if _is_api_node(obj):
            return "API_NODE_FRAGMENT", {"?": obj}, runtime
    return None


def canonical_json(obj: Any) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_CONDITIONING_INPUTS = {
    "ConditioningCombine": ("conditioning_1", "conditioning_2"),
    "ConditioningConcat": ("conditioning_to", "conditioning_from"),
    "ConditioningSetArea": ("conditioning",),
    "ConditioningSetAreaPercentage": ("conditioning",),
    "ConditioningSetAreaStrength": ("conditioning",),
    "ConditioningSetMask": ("conditioning",),
    "ConditioningSetTimestepRange": ("conditioning",),
    "ControlNetApply": ("conditioning",),
}


def _resolve_text(graph: dict[str, Any], ref: Any) -> str | None:
    pending = [ref]
    pieces: list[str] = []
    seen: set[tuple[str, int]] = set()
    while pending:
        link = pending.pop()
        if isinstance(link, str):
            pieces.append(link)
            continue
        if not isinstance(link, list) or len(link) != 2 or type(link[1]) is not int:
            return None
        node_id, socket = str(link[0]), link[1]
        if (node_id, socket) in seen:
            continue
        seen.add((node_id, socket))
        node = graph.get(node_id)
        if not _is_api_node(node):
            return None
        inputs = node["inputs"]
        cls = node["class_type"]
        if cls == "CLIPTextEncodeSDXL":
            fields = ("text_g", "text_l")
        elif cls in ("ControlNetApplyAdvanced", "InpaintModelConditioning") and socket in (0, 1):
            fields = (("positive", "negative")[socket],)
        elif cls == "PrimitiveString" and socket == 0 and type(inputs.get("value")) is str:
            fields = ("value",)
        else:
            fields = _CONDITIONING_INPUTS.get(cls, ("text",))
        pending.extend(inputs.get(field) for field in reversed(fields))
    return "\n".join(dict.fromkeys(pieces)) if pieces else None


def _upstream_inputs(graph: dict[str, Any], link: Any, *classes: str) -> dict[str, Any]:
    if (not isinstance(link, list) or len(link) != 2 or type(link[1]) is not int or link[1] != 0
            or not isinstance(link[0], (str, int))):
        return {}
    node = graph.get(str(link[0]))
    if _is_api_node(node) and node["class_type"] in classes:
        return node["inputs"]
    return {}


def _sampler_parameters(graph: dict[str, Any], ctype: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if ctype not in ("SamplerCustom", "SamplerCustomAdvanced"):
        return inputs
    selector = _upstream_inputs(graph, inputs.get("sampler"), "KSamplerSelect")
    schedule = _upstream_inputs(graph, inputs.get("sigmas"), "BasicScheduler")
    params = {**inputs, **{k: schedule[k] for k in ("scheduler", "steps") if k in schedule}}
    if "sampler_name" in selector:
        params["sampler_name"] = selector["sampler_name"]
    if ctype == "SamplerCustomAdvanced":
        noise = _upstream_inputs(graph, inputs.get("noise"), "RandomNoise")
        guider = _upstream_inputs(graph, inputs.get("guider"), "CFGGuider", "BasicGuider")
        params.update({k: guider[k] for k in ("cfg", "positive", "negative") if k in guider})
        if "conditioning" in guider:
            params.setdefault("positive", guider["conditioning"])
        if "noise_seed" in noise:
            params["noise_seed"] = noise["noise_seed"]
    return params


def execution_parameters(graph: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Extracts generation parameters from an API prompt graph. Missing values are omitted, not guessed."""
    params: dict[str, Any] = {}
    for key in ("prompt_id", "client_id"):
        if key in runtime:
            params[key] = runtime[key]
    if isinstance(runtime.get("job_timeline"), dict):
        params["job_timeline"] = runtime["job_timeline"]
    for node in graph.values():
        if not _is_api_node(node):
            continue
        inputs = node["inputs"]
        ctype = node["class_type"]
        if ctype in SAMPLER_CLASSES and "sampler_name" not in params:
            sampler = _sampler_parameters(graph, ctype, inputs)
            for src, dst in (("seed", "seed"), ("noise_seed", "seed"), ("sampler_name", "sampler_name"),
                             ("scheduler", "scheduler"), ("steps", "steps"), ("cfg", "cfg_scale")):
                value = sampler.get(src)
                if isinstance(value, list) and src in ("seed", "noise_seed", "steps"):
                    value = _upstream_inputs(graph, value, "PrimitiveInt").get("value")
                if type(value) in (str, int, float) and dst not in params:
                    params[dst] = value
            pos = _resolve_text(graph, sampler.get("positive"))
            neg = _resolve_text(graph, sampler.get("negative"))
            if pos is not None:
                params["positive_prompt"] = pos
            if neg is not None:
                params["negative_prompt"] = neg
        for key in CHECKPOINT_INPUTS:
            if isinstance(inputs.get(key), str) and "model_name" not in params:
                params["model_name"] = inputs[key]
        if ctype in ("SaveImage", "SaveImageAdvanced", "SaveAnimatedPNG", "SaveAnimatedWEBP") and isinstance(
            inputs.get("filename_prefix"), str,
        ):
            params.setdefault("filename_prefix", inputs["filename_prefix"])
        if ctype == "EmptyLatentImage":
            for key in ("width", "height", "batch_size"):
                if type(inputs.get(key)) is int:
                    params.setdefault(key, inputs[key])
    order = {k: i for i, k in enumerate(PARAMETER_ORDER)}
    if image_edit_outputs(graph):
        params["workflow_operation"] = "IMAGE_EDIT"
    return dict(sorted(params.items(), key=lambda kv: order.get(kv[0], len(order))))


MODEL_INPUT = re.compile(r"(?:^|_)(?:ckpt|lora|vae|unet|model|clip|clip_vision|control_net|controlnet|"
                         r"upscale_model|style_model|gligen|ipadapter|embedding)_?name\d*$")
MODEL_FILE_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".pkl", ".gguf", ".onnx")


def graph_dependencies(graph: dict[str, Any]) -> dict[str, list[str]]:
    """Node classes plus every model file the graph loads. A string input counts as a model asset when its
    name is a loader input (ckpt_name, clip_name1, ...) or its value has a model-file extension, so an
    unlisted loader input cannot slip a checkpoint past the format policy."""
    classes = sorted({n["class_type"] for n in graph.values() if _is_api_node(n)})
    assets = sorted(
        {
            v
            for n in graph.values()
            if _is_api_node(n)
            for k, v in n["inputs"].items()
            if isinstance(v, str) and (MODEL_INPUT.search(k) or v.lower().endswith(MODEL_FILE_SUFFIXES))
        }
    )
    return {"node_classes": classes, "model_assets": assets}


def image_edit_outputs(graph: dict[str, Any]) -> set[str]:
    """Recognize built-in sampler-free raster edits immediately upstream of an image sink."""
    if any(node["class_type"] in SAMPLER_CLASSES for node in graph.values() if _is_api_node(node)):
        return set()

    def linked_node(link: Any) -> dict[str, Any]:
        if isinstance(link, list) and len(link) == 2 and type(link[1]) is int and link[1] >= 0:
            node = graph.get(str(link[0]))
            if _is_api_node(node):
                return node
        return {}

    outputs = set()
    for node_id, node in graph.items():
        if not _is_api_node(node) or node["class_type"] not in ("SaveImage", "PreviewImage"):
            continue
        link = node["inputs"].get("images")
        edit = linked_node(link)
        required_inputs = IMAGE_EDIT_INPUTS.get(edit.get("class_type"))
        if required_inputs and link[1] == 0 and all(linked_node(edit["inputs"].get(name)) for name in required_inputs):
            outputs.add(node_id)
    return outputs


def graph_is_complete(fmt: str, graph: dict[str, Any], params: dict[str, Any]) -> bool:
    """Require resolvable references and either generation settings or a recognized image edit."""
    if fmt != "API_GRAPH":
        return False
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and type(value[1]) is int:
                if str(value[0]) not in graph:
                    return False
    return all(k in params for k in COMPLETE_PARAMETERS) or bool(image_edit_outputs(graph))


@dataclass(frozen=True)
class _Encoding:
    name: str
    codec: str
    unit: int
    brace: bytes
    anchors: tuple[bytes, ...]


JSON_STRING = _Encoding("UTF-8_JSON_STRING", "utf-8", 1, b"{", ESCAPED_ANCHORS)
"""A workflow stored as a string inside other JSON (the frontend's workflow drafts); raw bytes keep the escapes."""
ENCODINGS = (
    _Encoding("UTF-8", "utf-8", 1, b"{", UTF8_ANCHORS),
    _Encoding("UTF-16LE", "utf-16-le", 2, b"{\x00", (UTF16_ANCHOR,)),
    JSON_STRING,
)
_UPDATED_AT = re.compile(rb'\s*,\s*"updatedAt"\s*:\s*(\d{12,13})\b')


class WorkflowCarver:
    def __init__(
        self,
        max_candidate_bytes: int = MAX_JSON_CANDIDATE_BYTES,
        lookback_window_bytes: int = JSON_LOOKBACK_BYTES,
        id_factory: IdAllocator | None = None,
    ) -> None:
        self.max_candidate_bytes = max_candidate_bytes
        self.lookback = lookback_window_bytes
        self.next_id = id_factory or IdAllocator("WF")
        self._spans: dict[Any, SpanSet] = {}
        self._copies: dict[tuple[str, str], list[WorkflowRecord]] = {}
        self._locations: dict[str, set[tuple[Any, ...]]] = {}
        self._physical_locations: dict[str, set[str]] = {}
        self._graph_spans: dict[Any, SpanSet] = {}
        self.fragments: dict[bytes, TextFragment] = {}
        """Physical-mode pieces of ComfyUI JSON cut at a run edge, keyed by their bytes."""
        self.fragment_seconds = 0.0

    def carve_stream(
        self,
        buffer: bytes,
        virtual_address: int | None = None,
        base_physical_offset: int | None = None,
        process_id: int | None = None,
        source: ByteSource | None = None,
        exclude: SpanSet | Iterable[tuple[int, int]] = (),
    ) -> list[WorkflowRecord]:
        """Dual-stream (UTF-8 and UTF-16LE) carver. For each anchor it considers raw_decode at every
        preceding brace within the look-back window and keeps the largest object containing the anchor.
        Nested candidates share a decoded tree, with a total decode-work budget per run.
        Byte-identical copies in the same encoding are merged into the first record. `exclude` holds absolute
        ranges already claimed (embedded PNG text chunks), which are not standalone workflows.
        Raises: None."""
        is_virtual = virtual_address is not None
        base: int = virtual_address if virtual_address is not None else (base_physical_offset or 0)
        space = process_id if is_virtual else "physical"
        claimed = self._spans.setdefault(space, SpanSet())
        excluded = exclude if isinstance(exclude, SpanSet) else SpanSet(exclude)
        records: list[WorkflowRecord] = []
        for enc in ENCODINGS:
            positions = [
                idx
                for idx in _find_any(buffer, _anchor_pattern(enc.anchors))
                if (base + idx) % enc.unit == 0 and (base + idx) not in excluded
            ]
            i = 0
            previous_end = 0
            while i < len(positions):
                lo, hi = _text_run(buffer, positions[i], enc, previous_end, len(buffer))
                j = bisect.bisect_left(positions, hi, i)
                group, i, previous_end = positions[i:j], j, hi
                data, dbase = buffer, base
                if source is not None and ((lo == 0 and base > 0) or hi == len(buffer)):
                    left = source.read_before(base, self.lookback) if lo == 0 else b""
                    left = left[len(left) % enc.unit :]
                    right = source.read_prefix(base + hi, self.max_candidate_bytes)[0] if hi == len(buffer) else b""
                    data, dbase = left + buffer[lo:hi] + right, base + lo - len(left)
                    lo, hi = _text_run(data, base + group[0] - dbase, enc, 0, len(data))
                anchors = [base + p for p in group if (base + p) not in claimed]
                if enc is JSON_STRING:
                    records += self._carve_json_strings(data[lo:hi], dbase + lo, anchors, is_virtual, process_id,
                                                        source, claimed)
                    continue
                graphs = self._graph_spans.setdefault(space, SpanSet())
                for start, raw, obj in self._sweep_run(data[lo:hi], dbase + lo, anchors, enc, claimed):
                    record = self._record(start, raw, obj, enc, is_virtual, process_id, source)
                    if record is not None:
                        records.append(record)
                    if not _is_api_node(obj):
                        graphs.add(start, start + len(raw))
                if not is_virtual and enc.unit == 1:
                    self._keep_fragments(data[lo:hi], dbase + lo, graphs, [base + p for p in group])
        return records

    def _carve_json_strings(
        self, run: bytes, run_base: int, anchors: list[int], is_virtual: bool, process_id: int | None,
        source: ByteSource | None, claimed: SpanSet,
    ) -> list[WorkflowRecord]:
        """Workflows stored as JSON strings. A whole string that decodes to a ComfyUI graph is a record whose raw
        bytes are the escaped string; a draft's following "updatedAt" (ms) becomes ``saved_at_utc``. Any other
        string, in physical mode, is kept page by page as fragments of its unescaped text."""
        records: list[WorkflowRecord] = []
        for literal in escaped_literals(run, [a - run_base for a in anchors]):
            start = run_base + literal.start
            if start in claimed:
                continue
            raw = run[literal.start : literal.end]
            obj = None
            if not literal.cut_before and not literal.cut_after:
                try:
                    decoded = json.loads(json.loads(raw.decode("utf-8")))
                    obj = decoded if _no_surrogates(decoded) else None
                except (ValueError, RecursionError):
                    obj = None
            if extract_prompt_graph(obj) is not None:
                claimed.add(start, start + len(raw))
                record = self._record(start, raw, obj, JSON_STRING, is_virtual, process_id, source)
                saved = _UPDATED_AT.match(run, literal.end)
                if record is not None and saved is not None:
                    stamp = dt.datetime.fromtimestamp(int(saved[1]) / 1000, dt.timezone.utc)
                    record.execution_parameters["saved_at_utc"] = stamp.isoformat(timespec="milliseconds")
                if record is not None:
                    records.append(record)
            elif not is_virtual:
                self._keep_string_fragments(run, run_base, literal.start + (not literal.cut_before),
                                            literal.end - (not literal.cut_after))
        return records

    def _keep_string_fragments(self, run: bytes, run_base: int, lo: int, hi: int) -> None:
        """Each page's share of a JSON string's content, unescaped and read as a cut JSON fragment; the piece's
        raw bytes are the escaped content on that page."""
        cuts = [lo, *range(lo + (-(run_base + lo) % PAGE or PAGE), hi, PAGE), hi]
        for a, b in zip(cuts, cuts[1:], strict=False):
            content = run[a:b]
            if not any(anchor in content for anchor in ESCAPED_ANCHORS):
                continue
            fragment = parse_fragment(unescape(content.decode("utf-8", "surrogateescape")))
            if fragment is None or not (fragment.cut_before or fragment.cut_after):
                continue
            piece = self.fragments.setdefault(content, TextFragment(content, fragment, [], JSON_STRING.name))
            if run_base + a not in piece.physical_offsets:
                piece.physical_offsets.append(run_base + a)

    def _keep_fragments(self, run: bytes, run_base: int, graphs: SpanSet, anchors: list[int]) -> None:
        """Keeps the pieces of the run outside whole graphs. The same bytes seen again at another offset add
        that offset; seen again at the same offset (a block overlap) they are ignored."""
        started = time.perf_counter()
        covered = graphs.overlapping(run_base, run_base + len(run))
        for offset, raw, fragment in carve_fragments(run, run_base, covered, anchors):
            piece = self.fragments.setdefault(raw, TextFragment(raw, fragment, []))
            if offset not in piece.physical_offsets:
                piece.physical_offsets.append(offset)
        self.fragment_seconds += time.perf_counter() - started

    def _sweep_run(
        self, run: bytes, run_base: int, anchors: list[int], enc: _Encoding, claimed: SpanSet
    ) -> Iterator[tuple[int, bytes, Any]]:
        """Finds the largest candidate containing each anchor within the byte look-back and size cap.
        A lexical interval index avoids decoding unrelated objects or copying nested subtrees."""
        errors = "surrogatepass" if enc.unit == 2 else "surrogateescape"
        text = run.decode(enc.codec, errors=errors)
        duplicate_keys = [False]

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            obj = dict(pairs)
            if len(obj) != len(pairs):
                duplicate_keys[0] = True
            return obj

        decoder = json.JSONDecoder(object_pairs_hook=pairs_hook)
        ends = _object_ends(text)
        braces = sorted(ends)
        queue_starts = [match.start() for match in _QUEUE_START.finditer(text)]
        candidates = sorted([*_find_all(text, "{"), *queue_starts]) if queue_starts else _find_all(text, "{")
        parsed_upto = 0
        active: list[int] = []
        decoded: dict[int, tuple[Any, int]] = {}
        rejected: set[int] = set()
        budget = max(4 * len(run), self.max_candidate_bytes)
        cursor_byte, cursor_char = run_base, 0
        lookback_chars = self.lookback // enc.unit
        for anchor_abs in anchors:
            if anchor_abs < cursor_byte or anchor_abs in claimed:
                continue
            cursor_char += len(run[cursor_byte - run_base:anchor_abs - run_base].decode(enc.codec, errors=errors))
            cursor_byte = anchor_abs
            a = cursor_char
            while parsed_upto < len(candidates) and candidates[parsed_upto] <= a:
                b = candidates[parsed_upto]
                parsed_upto += 1
                if b not in ends:
                    # The lexical index can miss a brace after a stray quote; decode it directly.
                    if b < a - lookback_chars or budget <= 0:
                        continue
                    found, examined = _decode_object(decoder, text, b, self.max_candidate_bytes // enc.unit)
                    budget -= examined * enc.unit
                    if found is None:
                        continue
                    ends[b] = found[1]
                    decoded[b] = found
                active.append(b)
            active = [b for b in active if ends[b] > a and b >= a - lookback_chars]
            decoded = {b: value for b, value in decoded.items() if value[1] > a}
            for b in active:
                if b in rejected:
                    continue
                end = ends[b]
                if end - b > self.max_candidate_bytes // enc.unit:
                    rejected.add(b)
                    continue
                behind = len(text[b:a].encode(enc.codec, errors=errors))
                if behind > self.lookback:
                    continue
                if b not in decoded:
                    cost = (end - b) * enc.unit
                    if cost > budget:
                        continue
                    budget -= cost
                    duplicate_keys[0] = False
                    found, _ = _decode_object(decoder, text, b, self.max_candidate_bytes // enc.unit)
                    if found is None or found[1] != end:
                        rejected.add(b)
                        continue
                    children: list[Any] = []
                    stack = [found[0]]
                    while stack:
                        child = stack.pop()
                        if isinstance(child, dict):
                            children.append(child)
                            stack.extend(reversed(list(child.values())))
                        elif isinstance(child, list):
                            stack.extend(reversed(child))
                    keys = braces[bisect.bisect_left(braces, b):bisect.bisect_left(braces, end)]
                    # Duplicate JSON keys discard subtrees, invalidating the preorder correspondence.
                    if not duplicate_keys[0] and len(keys) == len(children):
                        decoded.update((key, (child, ends[key])) for key, child in zip(keys, children, strict=True))
                    decoded[b] = found
                obj = decoded[b][0]
                if extract_prompt_graph(obj) is None:
                    rejected.add(b)
                    continue
                raw = text[b:end].encode(enc.codec, errors=errors)
                if len(raw) > self.max_candidate_bytes or not _strict_ok(raw, enc.codec):
                    rejected.add(b)
                    continue
                start = anchor_abs - behind
                claimed.add(start, start + len(raw))
                yield start, raw, obj
                break

    def _record(
        self, start: int, raw: bytes, obj: Any, enc: _Encoding, is_virtual: bool, process_id: int | None,
        source: ByteSource | None,
    ) -> WorkflowRecord | None:
        classified = extract_prompt_graph(obj)
        if classified is None:
            return None
        fmt, graph, runtime = classified
        if fmt == "API_NODE_FRAGMENT":
            graph = obj
        try:
            digest = hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
        except (UnicodeError, ValueError, RecursionError):
            return None
        location = _hex_addr(start) or ""
        physical = None if is_virtual else location
        if is_virtual and isinstance(source, ProcessLayerSource):
            try:
                mapped = next(iter(source.layer.mapping(start, 1, ignore_errors=True)), None)
                if mapped is not None and mapped[0] == start:
                    physical = _hex_addr(mapped[2])
            except source._faults:
                pass
        provenance = {"process_id": process_id if is_virtual else None,
                      "virtual_address": location if is_virtual else None, "physical_offset": physical}
        location_key = tuple(provenance.values())
        copy_key = (enc.name, hashlib.sha256(raw).hexdigest())
        copies = self._copies.setdefault(copy_key, [])
        for existing in copies:
            if existing.raw_bytes != raw:
                continue
            locations = existing.execution_parameters["source_locations"]
            seen = self._locations[existing.workflow_id]
            if location_key not in seen:
                seen.add(location_key)
                locations.append(provenance)
            offsets = self._physical_locations[existing.workflow_id]
            if physical is not None and physical not in offsets:
                offsets.add(physical)
                existing.physical_offsets.append(physical)
            return None
        params: dict[str, Any]
        if fmt == "UI_WORKFLOW":
            params = {k: runtime[k] for k in ("prompt_id", "client_id") if k in runtime}
            params.update(ui_parameters(ui_workflow_nodes(graph)))
        else:
            params = execution_parameters(graph if fmt != "API_NODE_FRAGMENT" else {"?": obj}, runtime)
        if isinstance(obj, list):
            params["runtime_id_source"] = "SERIALIZED_QUEUE_ITEM"
        params["source_locations"] = [provenance]
        complete = graph_is_complete(fmt, graph, params)
        workflow_id = self.next_id()
        record = WorkflowRecord(
            workflow_id=workflow_id,
            completeness=Completeness.COMPLETE if complete else Completeness.PARTIAL,
            prompt_format=fmt,
            encoding_detected=enc.name,
            source_process_id=process_id if is_virtual else None,
            virtual_address=location if is_virtual else None,
            physical_offsets=[] if is_virtual else [location],
            canonical_json_sha256=digest,
            execution_parameters=params,
            dependencies=graph_dependencies(graph) if fmt == "API_GRAPH" else {"node_classes": [], "model_assets": []},
            raw_file_path="",
            raw_bytes=raw,
            graph=graph,
        )
        copies.append(record)
        self._locations[workflow_id] = {location_key}
        self._physical_locations[workflow_id] = {physical} if physical is not None else set()
        return record


_UTF8_CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_UTF16_CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]\x00", re.S)


def _text_run(data: bytes, idx: int, enc: _Encoding, floor: int, ceiling: int) -> tuple[int, int]:
    """Bounds of the control-character-free run around data[idx], searched no further left than `floor`
    nor right than `ceiling`. JSON text cannot contain raw control characters other than tab, LF and CR,
    so any object that contains the anchor lies inside this run. Backward search grows geometrically and
    forward search stops at the first control character, so cost is proportional to the run."""
    pattern = _UTF16_CONTROL if enc.unit == 2 else _UTF8_CONTROL
    floor = max(0, floor)
    floor += (idx - floor) % enc.unit

    def aligned(m: re.Match[bytes]) -> bool:
        return enc.unit == 1 or (m.start() - idx) % 2 == 0

    lo = floor
    window, upper = 256, idx
    while True:
        start = max(floor, idx - window)
        last = None
        for m in pattern.finditer(data, start, upper):
            if aligned(m):
                last = m
        if last is not None:
            lo = last.end()
            break
        if start == floor:
            break
        upper, window = start + 1, window * 4
    hi = ceiling - (ceiling - idx) % enc.unit
    for m in pattern.finditer(data, idx, ceiling):
        if aligned(m):
            hi = m.start()
            break
    return lo, hi


_JSON_STRUCTURE = re.compile(r'"[^"\\\x00-\x1f]*(?:\\.[^"\\\x00-\x1f]*)*"|[{}]')


def _object_ends(text: str) -> dict[int, int]:
    stack: list[int] = []
    ends: dict[int, int] = {}
    for match in _JSON_STRUCTURE.finditer(text):
        token = match.group()
        if token == "{":
            stack.append(match.start())
        elif token == "}" and stack:
            ends[stack.pop()] = match.end()
    return ends


def _no_surrogates(obj: Any) -> bool:
    try:
        stack = [obj]
        while stack:
            value = stack.pop()
            if isinstance(value, str):
                value.encode("utf-8")
            elif isinstance(value, dict):
                stack.extend(value)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    except (ValueError, RecursionError):
        return False
    return True


def _strict_ok(raw: bytes, codec: str) -> bool:
    try:
        obj = json.loads(raw.decode(codec, errors="strict"))
    except (ValueError, RecursionError):
        return False
    return _no_surrogates(obj)


def _decode_object(
    decoder: json.JSONDecoder, text: str, start: int, cap: int
) -> tuple[tuple[Any, int] | None, int]:
    """Decode a candidate object or queue item in growing slices. JSONDecodeError scans the whole
    document to compute a line number, so parsing the full run would make every failure O(run).
    Also returns the length of the last slice parsed, which bounds the work done within 4/3."""
    size = 256
    while True:
        chunk = text[start : start + size]
        try:
            obj, end = decoder.raw_decode(chunk)
        except json.JSONDecodeError as exc:
            truncated = exc.pos >= len(chunk) - 8 or exc.msg.startswith("Unterminated string")
            if not truncated or start + size >= len(text) or size >= cap:
                return None, len(chunk)
            size = min(size * 4, cap)
            continue
        except (RecursionError, ValueError):
            return None, len(chunk)
        return ((obj, start + end) if isinstance(obj, (dict, list)) else None), len(chunk)


@functools.cache
def _anchor_pattern(anchors: tuple[bytes, ...]) -> re.Pattern[bytes]:
    return re.compile(b"|".join(re.escape(anchor) for anchor in anchors))


def _find_any(haystack: bytes, pattern: re.Pattern[bytes]) -> list[int]:
    """Every offset where one of the pattern's literals starts, ascending: the union of ``_find_all`` over the
    literals, in one pass over the buffer. The search restarts one byte after each hit, so literals that share
    a byte (the quote between two keys) or overlap themselves are all found."""
    out = []
    match = pattern.search(haystack)
    while match is not None:
        out.append(match.start())
        match = pattern.search(haystack, match.start() + 1)
    return out


def _find_all(haystack: Any, needle: Any) -> list[int]:
    out = []
    idx = haystack.find(needle)
    while idx != -1:
        out.append(idx)
        idx = haystack.find(needle, idx + 1)
    return out


class SpanSet:
    """Absolute byte ranges already attributed to an artifact, with logarithmic membership tests."""

    def __init__(self, spans: Iterable[tuple[int, int]] = ()) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        for lo, hi in spans:
            self.add(lo, hi)

    def add(self, lo: int, hi: int) -> None:
        if hi <= lo:
            return
        i = bisect.bisect_left(self._ends, lo)
        j = bisect.bisect_right(self._starts, hi)
        if i < j:
            lo = min(lo, self._starts[i])
            hi = max(hi, self._ends[j - 1])
        self._starts[i:j] = [lo]
        self._ends[i:j] = [hi]

    def __contains__(self, point: int) -> bool:
        i = bisect.bisect_right(self._starts, point) - 1
        return i >= 0 and point < self._ends[i]

    def overlapping(self, lo: int, hi: int) -> list[tuple[int, int]]:
        i = bisect.bisect_right(self._ends, lo)
        j = bisect.bisect_left(self._starts, hi)
        return list(zip(self._starts[i:j], self._ends[i:j], strict=True))


class GapTracker:
    def __init__(self, outdir: Path) -> None:
        self.outdir = Path(outdir)

    def export_raw_spans(self, artifact_id: str, spans: list[tuple[int, bytes]]) -> list[str]:
        """Writes each genuine span to exports/raw/<artifact_id>/span_NNN.bin, unpadded.
        Returns paths relative to the case directory.
        Raises: OSError if disk write fails."""
        folder = self.outdir / "exports" / "raw" / artifact_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for n, (_offset, data) in enumerate(spans):
            target = folder / f"span_{n:03d}.bin"
            with open(target, "xb") as handle:
                handle.write(data)
            paths.append(target.relative_to(self.outdir).as_posix())
        return paths
