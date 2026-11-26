"""Content-verified recovery of PNG pages scattered through physical memory.

The index is built in one sequential pass. Stitching accepts a page only while zlib,
PNG framing, scanline filter bytes, chunk CRCs, and the IHDR-derived output size agree.
Continuity scores only rank the bounded beam; they never make a page valid.
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PAGE = 4096
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
IEND_TAIL = b"IEND\xaeB\x60\x82"
_SUPPORTED_CHANNELS = {0: 1, 2: 3, 6: 4}
_IS_ASCII_ALPHA = bytes(65 <= b <= 90 or 97 <= b <= 122 for b in range(256))
_DEADLINE_CHECK_INTERVAL = 256


@dataclass(frozen=True)
class PageIndex:
    """Compact physical offsets retained after one sequential pass of the image."""

    total_pages: int
    nonzero_offsets: np.ndarray
    high_entropy_offsets: np.ndarray
    boundary_offsets: np.ndarray
    entropy_threshold: float
    build_seconds: float
    _offset_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _cached_offsets(self, kind: str) -> tuple[tuple[int, ...], int, int]:
        cached = self._offset_cache.get(kind)
        if cached is not None:
            return cached
        if kind == "high":
            arr = self.high_entropy_offsets
        elif kind == "general":
            arr = self._general_array()
        elif kind == "fallback_interior":
            if self.nonzero_offsets.size and self.high_entropy_offsets.size:
                arr = self.nonzero_offsets[np.isin(self.nonzero_offsets, self.high_entropy_offsets, invert=True)]
            else:
                arr = self.nonzero_offsets
        elif kind == "fallback_general":
            general = self._general_array()
            if self.nonzero_offsets.size and general.size:
                arr = self.nonzero_offsets[np.isin(self.nonzero_offsets, general, invert=True)]
            else:
                arr = self.nonzero_offsets
        else:
            raise ValueError(f"unknown offset cache kind: {kind}")

        if arr.size:
            result = (tuple(arr.tolist()), int(arr.min()), int(arr.max()))
        else:
            result = ((), 0, -1)
        self._offset_cache[kind] = result
        return result

    def _general_array(self) -> np.ndarray:
        cached = self._offset_cache.get("_general_arr")
        if cached is None:
            if not self.boundary_offsets.size:
                cached = self.high_entropy_offsets
            elif not self.high_entropy_offsets.size:
                cached = self.boundary_offsets
            else:
                cached = np.union1d(self.boundary_offsets, self.high_entropy_offsets)
            self._offset_cache["_general_arr"] = cached
        return cached

    @classmethod
    def build(cls, physical: Any, *, entropy_threshold: float = 7.5, chunk_pages: int = 16384) -> PageIndex:
        started = time.perf_counter()
        total_pages = len(physical) // PAGE
        nonzero: list[np.ndarray] = []
        high: list[np.ndarray] = []
        boundaries: set[int] = set()

        counts = np.arange(PAGE + 1, dtype=np.float64)
        probabilities = counts / PAGE
        entropy_lut = np.zeros(PAGE + 1, dtype=np.float32)
        entropy_lut[1:] = (-probabilities[1:] * np.log2(probabilities[1:])).astype(np.float32)
        marker_tail = b""

        page_ranges = (
            [((seg.s_addr + PAGE - 1) // PAGE, seg.end_exclusive // PAGE) for seg in physical.segments]
            if hasattr(physical, "segments")
            else [(0, total_pages)]
        )
        for range_start, range_end in page_ranges:
            marker_tail = b""
            for base_page in range(range_start, range_end, chunk_pages):
                pages_here = min(chunk_pages, range_end - base_page)
                start = base_page * PAGE
                block = bytes(physical[start : start + pages_here * PAGE])
                actual_pages = len(block) // PAGE
                if not actual_pages:
                    marker_tail = b""
                    continue
                block = block[: actual_pages * PAGE]
                pages = np.frombuffer(block, dtype=np.uint8).reshape(actual_pages, PAGE)
                zero_counts = (pages == 0).sum(axis=1)
                nz = np.flatnonzero(zero_counts < PAGE)
                if nz.size:
                    nonzero.append((nz.astype(np.uint64) + base_page) * PAGE)

                candidates = np.flatnonzero(zero_counts <= 128)
                for batch_start in range(0, candidates.size, 2048):
                    batch = candidates[batch_start : batch_start + 2048]
                    selected = pages[batch]
                    not_ff = (selected == 255).sum(axis=1) <= 128
                    batch = batch[not_ff]
                    selected = selected[not_ff]
                    if batch.size:
                        row_tags = np.arange(batch.size, dtype=np.int32)[:, None] << 8
                        tagged = (selected.astype(np.int32) + row_tags).ravel()
                        histograms = np.bincount(tagged, minlength=batch.size * 256).reshape(-1, 256)
                        entropy = entropy_lut[histograms].sum(axis=1)
                        kept = batch[entropy >= entropy_threshold]
                        if kept.size:
                            high.append((kept.astype(np.uint64) + base_page) * PAGE)

                combined = marker_tail + block
                combined_start = start - len(marker_tail)
                for marker in (b"IDAT", IEND_TAIL):
                    pos = combined.find(marker)
                    while pos >= 0:
                        absolute = combined_start + pos
                        first = absolute // PAGE
                        last = (absolute + len(marker) - 1) // PAGE
                        for pfn in range(first, last + 1):
                            if 0 <= pfn < total_pages:
                                boundaries.add(pfn * PAGE)
                        pos = combined.find(marker, pos + 1)
                marker_tail = block[-(len(IEND_TAIL) - 1) :]

        def joined(parts: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts).astype(np.uint64, copy=False) if parts else np.empty(0, dtype=np.uint64)

        return cls(
            total_pages=total_pages,
            nonzero_offsets=joined(nonzero),
            high_entropy_offsets=joined(high),
            boundary_offsets=np.asarray(sorted(boundaries), dtype=np.uint64),
            entropy_threshold=entropy_threshold,
            build_seconds=time.perf_counter() - started,
        )


@dataclass(frozen=True)
class StitchResult:
    """Only bytes sealed by a newly verified chunk CRC are returned."""

    data: bytes
    page_offsets: list[int]
    candidate_counts: list[int]
    complete: bool


@dataclass(frozen=True)
class _PngInfo:
    width: int
    height: int
    bpp: int
    row_bytes: int
    stride: int
    total_inflated: int


@dataclass
class _State:
    mode: str
    chunk_type: bytes
    chunk_crc: int
    data_remaining: int
    crc_bytes: bytes
    header_bytes: bytes
    inflater: Any
    inflated: int
    saw_idat: bool
    previous_row: np.ndarray
    partial_row: bytes
    pages: list[bytes] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    last_page_used: int = PAGE
    verified_crcs: int = 0
    sealed_page_count: int = 0
    sealed_last_used: int = 0
    gradient_sum: float = 0.0
    gradient_rows: int = 0
    last_score: float = 255.0
    used_offsets: frozenset[int] = field(default_factory=frozenset)

    @property
    def rank(self) -> tuple[int, float, float]:
        average = self.gradient_sum / self.gradient_rows if self.gradient_rows else self.last_score
        return -self.verified_crcs, average, self.last_score

    def reconstructed(self, prefix: bytes) -> bytes:
        if not self.pages:
            return prefix
        return prefix + b"".join(self.pages[:-1]) + self.pages[-1][: self.last_page_used]

    def sealed(self, prefix: bytes) -> tuple[bytes, list[int]]:
        if not self.sealed_page_count:
            return prefix, []
        count = self.sealed_page_count
        data = prefix + b"".join(self.pages[: count - 1]) + self.pages[count - 1][: self.sealed_last_used]
        return data, self.offsets[:count]


def _chunk_type_ok(value: bytes) -> bool:
    return len(value) == 4 and all(_IS_ASCII_ALPHA[byte] for byte in value)


def _filters_ok(output: bytes, inflated_before: int, stride: int) -> bool:
    if not output:
        return True
    remainder = inflated_before % stride
    first = 0 if remainder == 0 else stride - remainder
    filters = output[first::stride]
    return not filters or max(filters) <= 4


def _defilter(row_type: int, raw: np.ndarray, previous: np.ndarray, bpp: int) -> np.ndarray:
    if row_type == 0:
        return raw.copy()
    if row_type == 2:
        return np.add(raw, previous, dtype=np.uint8)
    if row_type == 1:
        return np.cumsum(raw.reshape(-1, bpp), axis=0, dtype=np.uint8).ravel()
    line = bytearray(raw.tobytes())
    prior = previous.tobytes()
    n = len(line)
    head = min(bpp, n)
    if row_type == 3:
        for i in range(head):
            line[i] = (line[i] + (prior[i] >> 1)) & 0xFF
        for i in range(bpp, n):
            line[i] = (line[i] + ((line[i - bpp] + prior[i]) >> 1)) & 0xFF
    else:
        for i in range(head):
            line[i] = (line[i] + prior[i]) & 0xFF
        for i in range(bpp, n):
            a = line[i - bpp]
            b = prior[i]
            c = prior[i - bpp]
            pa = b - c
            if pa < 0:
                pa = -pa
            pb = a - c
            if pb < 0:
                pb = -pb
            pc = a + b - (c << 1)
            if pc < 0:
                pc = -pc
            predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            line[i] = (line[i] + predictor) & 0xFF
    return np.frombuffer(line, dtype=np.uint8)


def _score_rows(previous: np.ndarray, partial: bytes, output: bytes, info: _PngInfo) -> tuple[
        np.ndarray, bytes, float, int]:
    buffered = partial + output
    count = len(buffered) // info.stride
    if not count:
        return previous, buffered, 255.0, 0
    score = 0.0
    current = previous
    current_i16 = current.astype(np.int16)
    row_bytes = info.row_bytes
    bpp = info.bpp
    horiz_count = (info.width - 1) * bpp
    for row in range(count):
        start = row * info.stride
        row_type = buffered[start]
        raw = np.frombuffer(buffered, dtype=np.uint8, count=row_bytes, offset=start + 1)
        pixels = _defilter(row_type, raw, current, bpp)
        pixels_i16 = pixels.astype(np.int16)
        vertical = int(np.abs(pixels_i16 - current_i16).sum()) / row_bytes
        if horiz_count:
            grouped = pixels_i16.reshape(-1, bpp)
            horizontal = int(np.abs(grouped[1:] - grouped[:-1]).sum()) / horiz_count
        else:
            horizontal = 0.0
        score += vertical + horizontal
        current = pixels
        current_i16 = pixels_i16
    return current, buffered[count * info.stride :], score / count, count


def _initial_state(prefix: bytes, max_inflated: int) -> tuple[_PngInfo, _State] | None:
    if len(prefix) < 33 or prefix[:8] != PNG_MAGIC:
        return None
    ihdr_len, ihdr_type = struct.unpack(">I4s", prefix[8:16])
    if ihdr_len != 13 or ihdr_type != b"IHDR" or zlib.crc32(prefix[12:29]) != int.from_bytes(prefix[29:33], "big"):
        return None
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", prefix[16:29])
    if (not width or not height or depth != 8 or color not in _SUPPORTED_CHANNELS or compression != 0
            or filtering != 0 or interlace != 0):
        return None
    bpp = _SUPPORTED_CHANNELS[color]
    row_bytes = width * bpp
    total = height * (row_bytes + 1)
    if total <= 0 or total > max_inflated:
        return None
    info = _PngInfo(width, height, bpp, row_bytes, row_bytes + 1, total)
    inflater = zlib.decompressobj()
    inflated = 0
    saw_idat = False
    previous = np.zeros(row_bytes, dtype=np.uint8)
    partial = b""
    pos = 33

    def state(mode: str, ctype: bytes = b"", crc: int = 0, remaining: int = 0,
              crc_bytes: bytes = b"", header: bytes = b"") -> _State:
        return _State(mode, ctype, crc, remaining, crc_bytes, header, inflater, inflated, saw_idat,
                      previous, partial)

    while pos < len(prefix):
        if pos + 8 > len(prefix):
            return info, state("HEADER", header=prefix[pos:])
        length, ctype = struct.unpack(">I4s", prefix[pos : pos + 8])
        if length >= 1 << 31 or not _chunk_type_ok(ctype) or ctype == b"IHDR":
            return None
        pos += 8
        crc = zlib.crc32(ctype)
        amount = min(length, len(prefix) - pos)
        part = prefix[pos : pos + amount]
        crc = zlib.crc32(part, crc)
        if ctype == b"IDAT":
            saw_idat = True
            try:
                output = inflater.decompress(part)
            except zlib.error:
                return None
            if inflated + len(output) > total or not _filters_ok(output, inflated, info.stride):
                return None
            previous, partial, _, _ = _score_rows(previous, partial, output, info)
            inflated += len(output)
        pos += amount
        if amount < length:
            return info, state("DATA", ctype, crc, length - amount)
        amount = min(4, len(prefix) - pos)
        stored = prefix[pos : pos + amount]
        pos += amount
        if amount < 4:
            return info, state("CRC", ctype, crc, crc_bytes=stored)
        if int.from_bytes(stored, "big") != crc:
            return None
        if ctype == b"IEND":
            return info, state("DONE", ctype, crc, crc_bytes=stored)
    return info, state("HEADER")


def _advance_interior(current: _State, page: bytes | memoryview, offset: int, info: _PngInfo) -> _State | None:
    before = current.inflated
    stride = info.stride
    total_inflated = info.total_inflated
    max_output = total_inflated - before
    remainder = before % stride
    first_filter = 0 if remainder == 0 else stride - remainder
    inflater = current.inflater.copy()
    try:
        first = inflater.decompress(page, first_filter + 1)
        if len(first) > first_filter and first[first_filter] > 4:
            return None
        produced = len(first)
        if produced > max_output:
            return None
        tail = inflater.unconsumed_tail
        if tail:
            chunks = [first]
            while tail:
                chunk = inflater.decompress(tail, stride)
                if len(chunk) == stride and chunk[-1] > 4:
                    return None
                produced += len(chunk)
                if produced > max_output:
                    return None
                chunks.append(chunk)
                tail = inflater.unconsumed_tail
            output = b"".join(chunks)
        else:
            output = first
    except zlib.error:
        return None
    inflated = before + produced
    remaining = current.data_remaining - PAGE
    if inflater.eof and (inflater.unused_data or inflated != total_inflated or remaining > 0):
        return None
    previous, partial, score, rows = _score_rows(current.previous_row, current.partial_row, output, info)
    page_bytes = page if isinstance(page, bytes) else bytes(page)
    return _State(
        "CRC" if remaining == 0 else "DATA", b"IDAT", zlib.crc32(page_bytes, current.chunk_crc), remaining, b"", b"",
        inflater, inflated, True, previous, partial, current.pages + [page_bytes], current.offsets + [offset], PAGE,
        current.verified_crcs, current.sealed_page_count, current.sealed_last_used,
        current.gradient_sum + score * rows, current.gradient_rows + rows, score, current.used_offsets | {offset},
    )


def _advance_general(current: _State, page: bytes | memoryview, offset: int, info: _PngInfo) -> _State | None:
    mode, ctype = current.mode, current.chunk_type
    crc, remaining = current.chunk_crc, current.data_remaining
    crc_bytes, header = current.crc_bytes, current.header_bytes
    inflated, saw_idat = current.inflated, current.saw_idat
    previous, partial = current.previous_row, current.partial_row
    verified = current.verified_crcs
    sealed_count, sealed_used = current.sealed_page_count, current.sealed_last_used
    gradient_sum, gradient_rows = current.gradient_sum, current.gradient_rows
    inflater = None
    pos = 0
    step_gradient = 0.0
    step_rows = 0
    new_page_count = len(current.pages) + 1

    if mode == "DATA" and remaining + 4 <= PAGE:
        stored = int.from_bytes(page[remaining : remaining + 4], "big")
        if zlib.crc32(page[:remaining], crc) != stored:
            return None
    elif mode == "DATA" and remaining < PAGE:
        partial_len = PAGE - remaining
        if page[remaining:] != zlib.crc32(page[:remaining], crc).to_bytes(4, "big")[:partial_len]:
            return None
    elif mode == "CRC":
        needed = 4 - len(crc_bytes)
        if page[:needed] != crc.to_bytes(4, "big")[len(crc_bytes) :]:
            return None

    while pos < PAGE:
        if mode == "DATA":
            amount = min(remaining, PAGE - pos)
            part = page[pos : pos + amount]
            crc = zlib.crc32(part, crc)
            if ctype == b"IDAT" and part:
                inflater = inflater or current.inflater.copy()
                try:
                    output = inflater.decompress(part, info.total_inflated - inflated + 1)
                except zlib.error:
                    return None
                if inflated + len(output) > info.total_inflated or not _filters_ok(output, inflated, info.stride):
                    return None
                previous, partial, score, rows = _score_rows(previous, partial, output, info)
                step_gradient += score * rows
                step_rows += rows
                gradient_sum += score * rows
                gradient_rows += rows
                inflated += len(output)
                if inflater.eof and (inflater.unused_data or inflated != info.total_inflated or remaining - amount > 0):
                    return None
            pos += amount
            remaining -= amount
            if remaining == 0:
                mode, crc_bytes = "CRC", b""
        elif mode == "CRC":
            needed = 4 - len(crc_bytes)
            amount = min(needed, PAGE - pos)
            crc_bytes += bytes(page[pos : pos + amount])
            pos += amount
            if len(crc_bytes) == 4:
                if int.from_bytes(crc_bytes, "big") != crc:
                    return None
                verified += 1
                sealed_count, sealed_used = new_page_count, pos
                gradient_sum, gradient_rows = 0.0, 0
                if ctype == b"IEND":
                    checked = inflater or current.inflater
                    if not checked.eof or checked.unused_data or inflated != info.total_inflated:
                        return None
                    page_bytes = page if isinstance(page, bytes) else bytes(page)
                    return _State(
                        "DONE", b"IEND", crc, 0, crc_bytes, b"", checked.copy(), inflated, saw_idat,
                        previous, partial, current.pages + [page_bytes], current.offsets + [offset], pos, verified,
                        sealed_count, sealed_used, 0.0, 0, step_gradient / step_rows if step_rows else 0.0,
                        current.used_offsets | {offset},
                    )
                mode, header = "HEADER", b""
            elif crc_bytes != crc.to_bytes(4, "big")[: len(crc_bytes)]:
                return None
        else:
            needed = 8 - len(header)
            amount = min(needed, PAGE - pos)
            header += bytes(page[pos : pos + amount])
            pos += amount
            if len(header) == 8:
                length, next_type = struct.unpack(">I4s", header)
                if length >= 1 << 25 or not _chunk_type_ok(next_type) or next_type == b"IHDR":
                    return None
                if next_type == b"IEND" and length != 0:
                    return None
                checked = inflater or current.inflater
                if saw_idat and next_type != b"IDAT" and (
                        not checked.eof or checked.unused_data or inflated != info.total_inflated):
                    return None
                if next_type == b"IDAT":
                    if checked.eof and length:
                        return None
                    saw_idat = True
                ctype, crc, remaining = next_type, zlib.crc32(next_type), length
                mode = "DATA" if length else "CRC"
                crc_bytes, header = b"", b""

    checked = inflater or current.inflater.copy()
    page_bytes = page if isinstance(page, bytes) else bytes(page)
    return _State(
        mode, ctype, crc, remaining, crc_bytes, header, checked, inflated, saw_idat, previous, partial,
        current.pages + [page_bytes], current.offsets + [offset], PAGE, verified, sealed_count, sealed_used,
        gradient_sum, gradient_rows, step_gradient / step_rows if step_rows else current.last_score,
        current.used_offsets | {offset},
    )


def stitch_png(
    prefix_bytes: bytes,
    file_page_phase: int,
    physical: Any,
    index: PageIndex,
    beam: int = 8,
    *,
    max_steps: int = 200,
    max_inflated: int = 256 << 20,
    excluded_page_offsets: frozenset[int] = frozenset(),
    deadline: float | None = None,
) -> StitchResult | None:
    """Recover pages after a PNG prefix, returning only newly CRC-sealed bytes.

    ``file_page_phase`` is the PNG signature's offset within its first physical frame. The prefix must end on a
    physical frame boundary, so every searched continuation is one complete frame.

    When ``deadline`` (an absolute ``time.perf_counter()`` timestamp) is reached, any incomplete search step is
    discarded conservatively so ambiguous or unverified candidates cannot be accepted.
    """
    if beam < 1 or max_steps < 1 or not 0 <= file_page_phase < PAGE:
        return None
    if (file_page_phase + len(prefix_bytes)) % PAGE:
        return None
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    initialized = _initial_state(prefix_bytes, max_inflated)
    if initialized is None:
        return None
    info, initial = initialized
    if initial.mode == "DONE":
        return None
    initial.used_offsets = excluded_page_offsets

    active = [initial]
    best_data = prefix_bytes
    best_offsets: list[int] = []
    candidate_counts: list[int] = []

    physical_len = len(physical)
    max_valid_offset = physical_len - PAGE
    view: memoryview | None = None
    try:
        raw_view = memoryview(physical)
        try:
            if raw_view.format == "B" and raw_view.ndim == 1:
                view = raw_view
            else:
                view = raw_view.cast("B")
                raw_view.release()
        except TypeError:
            raw_view.release()
            view = None
    except TypeError:
        view = None

    high_offsets, high_min, high_max = index._cached_offsets("high")
    high_in_bounds = high_min >= 0 and high_max <= max_valid_offset
    general_cached: tuple[tuple[int, ...], bool] | None = None
    fallback_interior_cached: tuple[tuple[int, ...], bool] | None = None
    fallback_general_cached: tuple[tuple[int, ...], bool] | None = None

    stride = info.stride
    total_inflated = info.total_inflated
    alpha = _IS_ASCII_ALPHA
    crc32 = zlib.crc32
    zlib_error = zlib.error
    check_interval = _DEADLINE_CHECK_INTERVAL

    def _scan_interior(
        state: _State,
        offsets: tuple[int, ...],
        in_bounds: bool,
        next_states: list[_State],
    ) -> tuple[int, bool]:
        before = state.inflated
        max_output = total_inflated - before
        remainder = before % stride
        first_filter = 0 if remainder == 0 else stride - remainder
        first_limit = first_filter + 1
        remaining = state.data_remaining - PAGE
        next_mode = "CRC" if remaining == 0 else "DATA"
        copy_inflater = state.inflater.copy
        used_offsets = state.used_offsets
        chunk_crc = state.chunk_crc
        found = 0
        countdown = check_interval

        for offset in offsets:
            if deadline is not None:
                countdown -= 1
                if countdown == 0:
                    countdown = check_interval
                    if time.perf_counter() >= deadline:
                        return found, True
            if offset in used_offsets:
                continue
            if view is not None:
                if not in_bounds and (offset < 0 or offset > max_valid_offset):
                    continue
                page: bytes | memoryview = view[offset : offset + PAGE]
            else:
                if offset < 0 or offset > max_valid_offset:
                    continue
                page = bytes(physical[offset : offset + PAGE])
                if len(page) != PAGE:
                    continue

            inflater = copy_inflater()
            try:
                first = inflater.decompress(page, first_limit)
                if len(first) > first_filter and first[first_filter] > 4:
                    continue
                produced = len(first)
                if produced > max_output:
                    continue
                tail = inflater.unconsumed_tail
                if tail:
                    chunks = [first]
                    bad = False
                    while tail:
                        chunk = inflater.decompress(tail, stride)
                        if len(chunk) == stride and chunk[-1] > 4:
                            bad = True
                            break
                        produced += len(chunk)
                        if produced > max_output:
                            bad = True
                            break
                        chunks.append(chunk)
                        tail = inflater.unconsumed_tail
                    if bad:
                        continue
                    output = b"".join(chunks)
                else:
                    output = first
            except zlib_error:
                continue

            inflated = before + produced
            if inflater.eof and (inflater.unused_data or inflated != total_inflated or remaining > 0):
                continue

            previous, partial, score, rows = _score_rows(state.previous_row, state.partial_row, output, info)
            page_bytes = page if isinstance(page, bytes) else bytes(page)
            next_states.append(
                _State(
                    next_mode, b"IDAT", crc32(page_bytes, chunk_crc), remaining, b"", b"",
                    inflater, inflated, True, previous, partial, state.pages + [page_bytes],
                    state.offsets + [offset], PAGE, state.verified_crcs, state.sealed_page_count,
                    state.sealed_last_used, state.gradient_sum + score * rows, state.gradient_rows + rows,
                    score, used_offsets | {offset},
                )
            )
            found += 1
        return found, False

    def _scan_general(
        state: _State,
        offsets: tuple[int, ...],
        in_bounds: bool,
        next_states: list[_State],
        completed: list[_State],
        record_seal: Any,
    ) -> tuple[int, str]:
        mode = state.mode
        rem = state.data_remaining
        crc0 = state.chunk_crc
        used_offsets = state.used_offsets
        found = 0
        countdown = check_interval

        # Precompute cheap byte-level rejection filter for the active state.
        # 0: none, 1: CRC tail match, 2: DATA + next header + CRC, 3: DATA + CRC,
        # 4: DATA + partial CRC, 5: HEADER (len 0), 6: HEADER (len 1..7)
        filter_kind = 0
        crc_needed = 0
        expected_crc_tail = b""
        hpos = 0
        partial_len = 0
        hdr_len = 0
        hdr_alpha_start = 0
        hdr_alpha_end = 0

        if mode == "CRC":
            expected_crc = crc0.to_bytes(4, "big")
            have = len(state.crc_bytes)
            if have and state.crc_bytes != expected_crc[:have]:
                return 0, "ok"
            filter_kind = 1
            crc_needed = 4 - have
            expected_crc_tail = expected_crc[have:]
        elif mode == "DATA":
            if rem + 12 <= PAGE and state.chunk_type != b"IEND":
                filter_kind = 2
                hpos = rem + 4
            elif rem + 4 <= PAGE:
                filter_kind = 3
                hpos = rem + 4
            elif rem < PAGE:
                filter_kind = 4
                partial_len = PAGE - rem
        elif mode == "HEADER":
            hdr = state.header_bytes
            hdr_len = len(hdr)
            if hdr_len == 0:
                filter_kind = 5
            else:
                if hdr[0] >= 2 or any(not alpha[byte] for byte in hdr[4:]):
                    return 0, "ok"
                filter_kind = 6
                hdr_alpha_start = max(4, hdr_len) - hdr_len
                hdr_alpha_end = 8 - hdr_len

        for offset in offsets:
            if deadline is not None:
                countdown -= 1
                if countdown == 0:
                    countdown = check_interval
                    if time.perf_counter() >= deadline:
                        return found, "timeout"
            if offset in used_offsets:
                continue
            if view is not None:
                if not in_bounds and (offset < 0 or offset > max_valid_offset):
                    continue
                page: bytes | memoryview = view[offset : offset + PAGE]
            else:
                if offset < 0 or offset > max_valid_offset:
                    continue
                page = bytes(physical[offset : offset + PAGE])
                if len(page) != PAGE:
                    continue

            if filter_kind == 1:
                if page[:crc_needed] != expected_crc_tail:
                    continue
            elif filter_kind == 2:
                if page[hpos] >= 2:
                    continue
                if not (
                    alpha[page[hpos + 4]] and alpha[page[hpos + 5]]
                    and alpha[page[hpos + 6]] and alpha[page[hpos + 7]]
                ):
                    continue
                if crc32(page[:rem], crc0) != int.from_bytes(page[rem:hpos], "big"):
                    continue
            elif filter_kind == 3:
                if crc32(page[:rem], crc0) != int.from_bytes(page[rem:hpos], "big"):
                    continue
            elif filter_kind == 4:
                if page[rem:] != crc32(page[:rem], crc0).to_bytes(4, "big")[:partial_len]:
                    continue
            elif filter_kind == 5:
                if page[0] >= 2:
                    continue
                if not (alpha[page[4]] and alpha[page[5]] and alpha[page[6]] and alpha[page[7]]):
                    continue
            elif filter_kind == 6:
                if any(not alpha[page[i]] for i in range(hdr_alpha_start, hdr_alpha_end)):
                    continue

            advanced = _advance_general(state, page, offset, info)
            if advanced is None:
                continue
            found += 1
            if not record_seal(state, advanced):
                return found, "ambiguous"
            if advanced.mode == "DONE":
                completed.append(advanced)
            else:
                next_states.append(advanced)
        return found, "ok"

    try:
        for _step in range(max_steps):
            if not active:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            next_states: list[_State] = []
            completed: list[_State] = []
            valid = 0
            step_best_data = best_data
            step_best_offsets = best_offsets
            timed_out = False

            def seal_is_consistent(parent: _State, child: _State) -> bool:
                nonlocal step_best_data, step_best_offsets
                if not child.sealed_page_count or child.verified_crcs <= parent.verified_crcs:
                    return True
                data, offsets = child.sealed(prefix_bytes)
                if data.startswith(step_best_data):
                    if len(data) > len(step_best_data):
                        step_best_data, step_best_offsets = data, offsets
                    return True
                return step_best_data.startswith(data)

            for state in active:
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break
                interior = state.mode == "DATA" and state.chunk_type == b"IDAT" and state.data_remaining >= PAGE
                if interior:
                    found, hit_deadline = _scan_interior(state, high_offsets, high_in_bounds, next_states)
                    valid += found
                    if hit_deadline:
                        timed_out = True
                        break
                    if not next_states and not completed:
                        if deadline is not None and time.perf_counter() >= deadline:
                            timed_out = True
                            break
                        if fallback_interior_cached is None:
                            fb_offsets, fb_min, fb_max = index._cached_offsets("fallback_interior")
                            fallback_interior_cached = (fb_offsets, fb_min >= 0 and fb_max <= max_valid_offset)
                        found, hit_deadline = _scan_interior(
                            state, fallback_interior_cached[0], fallback_interior_cached[1], next_states,
                        )
                        valid += found
                        if hit_deadline:
                            timed_out = True
                            break
                else:
                    if general_cached is None:
                        gen_offsets, gen_min, gen_max = index._cached_offsets("general")
                        general_cached = (gen_offsets, gen_min >= 0 and gen_max <= max_valid_offset)
                    found, status = _scan_general(
                        state, general_cached[0], general_cached[1], next_states, completed, seal_is_consistent,
                    )
                    valid += found
                    if status == "ambiguous":
                        return None
                    if status == "timeout":
                        timed_out = True
                        break
                    if not next_states and not completed:
                        if deadline is not None and time.perf_counter() >= deadline:
                            timed_out = True
                            break
                        if fallback_general_cached is None:
                            fb_offsets, fb_min, fb_max = index._cached_offsets("fallback_general")
                            fallback_general_cached = (fb_offsets, fb_min >= 0 and fb_max <= max_valid_offset)
                        found, status = _scan_general(
                            state, fallback_general_cached[0], fallback_general_cached[1],
                            next_states, completed, seal_is_consistent,
                        )
                        valid += found
                        if status == "ambiguous":
                            return None
                        if status == "timeout":
                            timed_out = True
                            break

            if timed_out:
                break

            best_data, best_offsets = step_best_data, step_best_offsets
            candidate_counts.append(valid)
            if completed:
                winner = min(completed, key=lambda state: state.rank)
                data = winner.reconstructed(prefix_bytes)
                if any(state.reconstructed(prefix_bytes) != data for state in completed if state is not winner):
                    # A valid checksum establishes the bytes of a candidate. Different valid tails can share the prefix.
                    return None
                return StitchResult(data, winner.offsets, candidate_counts, True)
            if not next_states:
                break
            next_states.sort(key=lambda state: state.rank)
            active = next_states[:beam]
    finally:
        if view is not None:
            view.release()

    if len(best_data) <= len(prefix_bytes):
        return None
    return StitchResult(best_data, best_offsets, candidate_counts, False)


def stitchable_png_prefix(prefix_bytes: bytes, *, max_inflated: int = 256 << 20) -> bool:
    """Whether a page-aligned prefix stops inside IDAT and uses a supported IHDR format."""
    initialized = _initial_state(prefix_bytes, max_inflated)
    if initialized is None:
        return False
    _info, state = initialized
    return state.chunk_type == b"IDAT" and state.mode in {"DATA", "CRC"}
