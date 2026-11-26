"""Read a captured x86-64 Linux user-process ELF core without filling absent pages."""

from __future__ import annotations

import bisect
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfytrace.exceptions import IngestionError, ProcessNotFoundError
from comfytrace.ingest import MODE_VIRTUAL, IngestionResult, RecoverySupport, process_space
from comfytrace.ingest_linux import LinuxProcessWalker

HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
PROGRAM = struct.Struct("<IIQQQQQQ")
NOTE = struct.Struct("<III")
FILE_HEADER = struct.Struct("<QQ")
FILE_ENTRY = struct.Struct("<QQQ")
PT_LOAD, PT_NOTE = 1, 4
NT_PRPSINFO, NT_FILE = 3, 0x46494C45
MAX_NOTE_BYTES = 16 << 20
USER_SPACE_END = 1 << 56  # x86-64 LA57 user limit; 4-level user space (2^47) lies inside it


@dataclass(frozen=True)
class Segment:
    start: int
    memory_end: int
    file_end: int
    offset: int


def _notes(data: bytes) -> Iterator[tuple[int, bytes]]:
    cursor = 0
    while cursor + NOTE.size <= len(data):
        name_size, desc_size, kind = NOTE.unpack_from(data, cursor)
        name_start = cursor + NOTE.size
        desc_start = name_start + ((name_size + 3) & ~3)
        desc_end = desc_start + desc_size
        cursor = desc_start + ((desc_size + 3) & ~3)
        if desc_end > len(data) or cursor > len(data):
            raise ValueError("truncated ELF core note")
        if data[name_start:name_start + name_size].rstrip(b"\0") == b"CORE":
            yield kind, data[desc_start:desc_end]


def _mapped_files(data: bytes) -> list[tuple[int, int, str]]:
    count, page_size = FILE_HEADER.unpack_from(data)
    end = FILE_HEADER.size + count * FILE_ENTRY.size
    if page_size == 0 or end > len(data):
        raise ValueError("invalid ELF core mapped-file table")
    names = data[end:].split(b"\0")
    if len(names) != count + 1 or names[-1]:
        raise ValueError("truncated ELF core mapped-file names")
    result = []
    for index, (start, stop, _offset) in enumerate(FILE_ENTRY.iter_unpack(data[FILE_HEADER.size:end])):
        if start >= stop or not names[index]:
            raise ValueError("invalid ELF core mapped-file range")
        result.append((start, stop, names[index].decode("utf-8", errors="replace")))
    result.sort()
    if any(first[1] > second[0] for first, second in zip(result, result[1:], strict=False)):
        raise ValueError("overlapping ELF core mapped-file ranges")
    return result


class CoreLayer:
    """Virtual ranges backed by PT_LOAD file bytes. The mapped offsets are file offsets."""

    def __init__(self, source: Any) -> None:
        ident, kind, machine, version, _, table, _, _, _, entry_size, count, *_ = HEADER.unpack_from(source)
        if (ident[:7] != b"\x7fELF\x02\x01\x01" or (kind, machine, version) != (4, 62, 1)
                or entry_size != PROGRAM.size or not 0 < count < 65535 or table < HEADER.size
                or table + count * entry_size > len(source)):
            raise ValueError("expected an x86-64 little-endian Linux ELF process core")
        self.source = source
        self.segments: list[Segment] = []
        process = None
        files = None
        note_bytes = 0
        metadata_ranges = [(0, HEADER.size), (table, table + count * entry_size)]
        for index in range(count):
            tag, _, offset, va, _pa, filesz, memsz, _align = PROGRAM.unpack_from(source, table + index * entry_size)
            if filesz and offset + filesz > len(source):
                raise ValueError("ELF core segment extends beyond the capture")
            # Canonical user space reaches 2^56 under 5-level paging; kernel (vsyscall) segments lie above it.
            if tag == PT_LOAD and memsz and va < USER_SPACE_END:
                if filesz > memsz or va + memsz > USER_SPACE_END:
                    raise ValueError("inconsistent ELF core virtual range")
                self.segments.append(Segment(va, va + memsz, va + filesz, offset))
            if tag == PT_NOTE:
                note_bytes += filesz
                if note_bytes > MAX_NOTE_BYTES:
                    raise ValueError("ELF core note data is too large")
                metadata_ranges.append((offset, offset + filesz))
                for note, data in _notes(source[offset:offset + filesz]):
                    if note == NT_FILE:
                        mapped_files = _mapped_files(data)
                        if files is not None and files != mapped_files:
                            raise ValueError("conflicting ELF core mapped-file tables")
                        files = mapped_files
                    elif note == NT_PRPSINFO and len(data) >= 136:
                        pid, = struct.unpack_from("<i", data, 24)
                        name, args = (part.split(b"\0", 1)[0].decode("utf-8", errors="replace")
                                      for part in (data[40:56], data[56:136]))
                        observed = pid, name, args
                        if process is not None and process != observed:
                            raise ValueError("conflicting ELF core process records")
                        process = observed
        self.segments.sort(key=lambda segment: segment.start)
        if (process is None or process[0] <= 0 or not self.segments
                or any(a.memory_end > b.start for a, b in zip(self.segments, self.segments[1:], strict=False))):
            raise ValueError("ELF core lacks an unambiguous process or virtual range table")
        ranges = sorted((s.offset, s.offset + s.file_end - s.start) for s in self.segments if s.file_end > s.start)
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("ELF core load segments reuse captured file bytes")
        range_ends = [end for _, end in ranges]
        for start, end in metadata_ranges:
            position = bisect.bisect_right(range_ends, start)
            if position < len(ranges) and ranges[position][0] < end:
                raise ValueError("ELF core load segment overlaps capture metadata")
        self.pid, self.process_name, self.command_line = process
        self.starts = [segment.start for segment in self.segments]
        self.regions = []
        file_starts = [entry[0] for entry in files or ()]
        for segment in self.segments:
            cursor = segment.start
            while cursor < segment.memory_end:
                position = bisect.bisect_right(file_starts, cursor) - 1
                if files and position >= 0 and cursor < files[position][1]:
                    _, stop, path = files[position]
                    end = min(stop, segment.memory_end)
                else:
                    following = file_starts[position + 1:position + 2]
                    end = min(following[0], segment.memory_end) if following else segment.memory_end
                    # The core lists named mappings, but does not establish anonymous page ownership.
                    path = "<ELF core mapping origin unverified>"
                self.regions.append((cursor, end - 1, path))
                cursor = end

    def mapping(self, offset: int, length: int,
                ignore_errors: bool = False) -> Iterator[tuple[int, int, int, int, str]]:
        first = max(0, bisect.bisect_right(self.starts, offset) - 1)
        for index in range(first, len(self.segments)):
            segment = self.segments[index]
            if segment.start >= offset + length:
                break
            start, end = max(segment.start, offset), min(segment.file_end, offset + length)
            if end > start:
                yield start, end - start, segment.offset + start - segment.start, end - start, "ELF_CORE"

    def read(self, offset: int, length: int, pad: bool = False) -> bytes:
        from volatility3.framework.exceptions import InvalidAddressException

        out = bytearray()
        for va, size, file_offset, _, _ in self.mapping(offset, length):
            if va != offset + len(out):
                break
            out.extend(self.source[file_offset:file_offset + size])
        if len(out) != length:
            raise InvalidAddressException("ELF_CORE", offset + len(out))
        return bytes(out)


def ingest_core(memory_path: Path, examiner_pids: Iterable[int]) -> IngestionResult:
    from comfytrace.carve import StreamReader
    from comfytrace.procmem import ProcessMemory

    mapped, handle = StreamReader(memory_path).open_physical()

    def close() -> None:
        mapped.close()
        handle.close()

    try:
        layer = CoreLayer(mapped)
        if set(examiner_pids) != {layer.pid}:
            raise ProcessNotFoundError(f"ELF process core requires --comfyui-pid {layer.pid} to confirm its scope.")
        recovery = RecoverySupport(notes={"bitmap_ordering": "UNAVAILABLE: process-only ELF core"})
        if LinuxProcessWalker._is_python(layer.process_name):
            recovery.python_processes.append(
                ProcessMemory.from_layer(layer.pid, layer, layer.regions, None, image_os="linux"))
        regions = [(start, end) for start, end, _ in layer.regions]
        space = process_space(layer.pid, layer, regions, StreamReader(memory_path), None, file_offsets=False)
        context = {"pid": layer.pid, "ppid": None, "process_name": layer.process_name,
                   "command_line": layer.command_line, "dtb": None, "threads": None, "vad_count": len(regions),
                   "role": "ROOT_LAUNCHER", "discovery_basis": "EXAMINER_PID"}
        return IngestionResult(MODE_VIRTUAL, {"status": "NOT_REQUIRED", "mode": "ELF_PROCESS_CORE",
                                             "image_os": "linux", "image_format": "elf-core"},
                               [context], [space], close=close, recovery=recovery)
    except (ValueError, struct.error) as exc:
        close()
        raise IngestionError(f"Cannot read ELF process core: {exc}") from exc
    except BaseException:
        close()
        raise
