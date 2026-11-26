"""Synthetic ComfyUI artifacts and a page-table-backed fake of the Volatility process layer.

The fake process layer resolves virtual pages to page frames inside a real dump file, so tests
exercise genuine translation: contiguous virtual memory may sit on scattered physical frames.
"""

from __future__ import annotations

import io
import json
import random
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from volatility3.framework.exceptions import PagedInvalidAddressException, SwappedInvalidAddressException

PAGE = 4096


def api_prompt(seed: int = 167346845307033, text: str = "cinematic portrait of a golden retriever puppy, 8k",
               steps: int = 20, sampler: str = "euler_ancestral", width: int = 64, height: int = 48,
               batch_size: int = 1, ckpt: str = "v1-5-pruned-emaonly.safetensors") -> dict[str, Any]:
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": 7.0, "sampler_name": sampler, "scheduler": "normal",
            "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch_size}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, distorted", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
    }


def prompt_request_body(graph: dict[str, Any], prompt_id: str = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02") -> bytes:
    """A /prompt request body as the ComfyUI server holds it: graph plus runtime keys."""
    body = {"client_id": "5f0c3c1e9a7b4e0f", "prompt": graph, "prompt_id": prompt_id}
    return json.dumps(body, indent=1).encode("utf-8")


def noise_image(width: int, height: int, mode: str = "RGB", seed: int = 1) -> Image.Image:
    rng = random.Random(seed)
    channels = len(mode)
    data = bytes(
        (x * 3 + y * 5 + rng.randrange(48)) & 0xFF for y in range(height) for x in range(width) for _ in range(channels)
    )
    return Image.frombytes(mode, (width, height), data)


def png_bytes(img: Image.Image, text: dict[str, str] | None = None, idat_chunk: int = 1024,
              interlace: bool = False) -> bytes:
    """Encodes with Pillow (adaptive scanline filters), then re-splits IDAT into small chunks so a tear
    lands mid-stream the way it does for multi-page images in RAM."""
    info = PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info, optimize=False)
    raw = buf.getvalue()
    chunks, idat = [], b""
    pos = 8
    while pos < len(raw):
        length, ctype = struct.unpack(">I4s", raw[pos:pos + 8])
        body = raw[pos + 8:pos + 8 + length]
        if ctype == b"IHDR" and interlace:
            body = body[:12] + b"\x01"
        chunks.append((ctype, body))
        if ctype == b"IDAT":
            idat += body
        pos += 12 + length
    if interlace:
        idat = zlib.compress(b"\x00" * (len(img.tobytes()) + img.height * 7))
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    emitted = False
    for ctype, body in chunks:
        if ctype == b"IDAT":
            if emitted:
                continue
            emitted = True
            for i in range(0, len(idat), idat_chunk):
                out += _chunk(b"IDAT", idat[i:i + idat_chunk])
            continue
        out += _chunk(ctype, body)
    return bytes(out)


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def garbage(n: int, seed: int = 7) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(n))


class DumpBuilder:
    """Lays out a flat physical image in 4 KB frames."""

    def __init__(self, frames: int, seed: int = 3) -> None:
        self.data = bytearray(garbage(frames * PAGE, seed))

    def put(self, frame: int, payload: bytes, offset_in_frame: int = 0) -> int:
        start = frame * PAGE + offset_in_frame
        self.data[start:start + len(payload)] = payload
        return start

    def write(self, path: Path) -> Path:
        path.write_bytes(bytes(self.data))
        return path


@dataclass
class FakeProcess:
    UniqueProcessId: int
    InheritedFromUniqueProcessId: int
    ImageFileName: str
    cmdline: str | None
    ActiveThreads: int = 8
    vads: list[tuple[int, int]] = field(default_factory=list)
    layer: Any = None

    @property
    def Pcb(self) -> Any:  # noqa: N802 - mirrors the _EPROCESS member name
        return type("Pcb", (), {"DirectoryTableBase": 0x1AB000 + self.UniqueProcessId})()


class FakeVad:
    def __init__(self, start: int, end: int) -> None:
        self._start, self._end = start, end

    def get_start(self) -> int:
        return self._start

    def get_end(self) -> int:
        return self._end


class FakeProcLayer:
    """Virtual pages -> frames of a physical dump. Missing pages fault; 'swapped' pages fault as paged out."""

    name = "fake_proc_layer"

    def __init__(self, physical: bytes, page_table: dict[int, int], swapped: set[int] | None = None) -> None:
        self.physical = physical
        self.page_table = page_table
        self.swapped = swapped or set()
        self.reads: list[tuple[int, int, bool]] = []

    def map_virtual(self, va: int, frames: list[int]) -> None:
        for i, frame in enumerate(frames):
            self.page_table[va // PAGE + i] = frame

    def mapping(self, offset: int, length: int, ignore_errors: bool = False):
        end = offset + length
        cursor = offset
        while cursor < end:
            page = cursor // PAGE
            step = min(PAGE - cursor % PAGE, end - cursor)
            if page in self.page_table and page not in self.swapped:
                phys = self.page_table[page] * PAGE + cursor % PAGE
                yield cursor, step, phys, step, "memory_layer"
            elif not ignore_errors:
                raise PagedInvalidAddressException(self.name, cursor, 0, 0)
            cursor += step

    def read(self, offset: int, length: int, pad: bool = False) -> bytes:
        self.reads.append((offset, length, pad))
        if pad:
            raise AssertionError("pad=True is prohibited")
        out = bytearray()
        cursor, end = offset, offset + length
        while cursor < end:
            page = cursor // PAGE
            step = min(PAGE - cursor % PAGE, end - cursor)
            if page in self.swapped:
                raise SwappedInvalidAddressException(self.name, cursor, 0, 0, 0)
            if page not in self.page_table:
                raise PagedInvalidAddressException(self.name, cursor, 0, 0)
            phys = self.page_table[page] * PAGE + cursor % PAGE
            out += self.physical[phys:phys + step]
            cursor += step
        return bytes(out)


def pages_for(payload_len: int) -> int:
    return (payload_len + PAGE - 1) // PAGE
