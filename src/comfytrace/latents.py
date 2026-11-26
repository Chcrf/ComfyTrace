"""Latents of finished jobs in ComfyUI process memory.

Unless ComfyUI runs with ``--gpu-only``, a sampler's output LATENT is moved to the CPU as a float32 tensor of
shape [batch, 4, height/8, width/8]. Its storage sits in a private allocation on a 64 KiB boundary and, once
freed, loses only its first 8 bytes to the allocator's free-list link (measured on run 1 and T2, T95). Decoding
it with the job's VAE gives that job's image before PNG encoding.

A block is a latent only when every test holds:

* every value is finite and below LIMIT in magnitude, apart from the first two float32 slots when they hold a
  user-space pointer (the free-list link word);
* fewer than 1% of values are zero, and each channel's standard deviation lies in CHANNEL_STD;
* neighbouring values along a row are correlated, in every channel (CHANNEL_LAG_MIN) and on average (LAG_MIN):
  sampler output is smooth, CPU noise is not, and a read that spans a noise block fails in that channel;
* in every channel, values one claimed row apart are correlated (CHANNEL_LAG_MIN), more than values half a row
  apart, and not much less (WIDTH_MARGIN) than values two rows apart. A block read at the wrong shape fails:
  several smaller latents correlate best at half the width, half of a larger one at twice the width.

The bytes are reported as found. Decoding is a separate step on a runner (``decode-latents``); a decoded image is
derived evidence, never a recovered image.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from comfytrace.procmem import PAGE, ProcessMemory

SLOT = 64 * 1024
CHANNELS = 4
LIMIT = 64.0
CHANNEL_STD = (0.5, 12.0)
LAG_MIN = 0.2
CHANNEL_LAG_MIN = 0.1
WIDTH_MARGIN = 0.15
DEFAULT_SIDES = ((64, 64), (96, 96), (128, 128))
LATENTS_NAME = "latents.json"
USER_POINTER = (0x10000, 0x7FFF_FFFF_0000)


@dataclass
class LatentBlock:
    process_id: int
    address: int
    height: int
    width: int
    raw: bytes
    physical_offsets: list[int]
    first_word: str
    stats: dict[str, list[float]]
    stale_pages: list[int] = field(default_factory=list)

    def values(self) -> np.ndarray:
        """float32 [4, height, width]; when the first word is a free-list link, its two slots copy the third."""
        a = np.frombuffer(self.raw, np.float32).copy()
        if self.first_word == "POINTER":
            a[:2] = a[2]
        return a.reshape(CHANNELS, self.height, self.width)


def _correlation(x: np.ndarray, lag: int) -> np.ndarray:
    """Per channel, the lag correlation of its values read as one flat sequence."""
    c = x.reshape(x.shape[0], -1)
    c = c - c.mean(1, keepdims=True)
    return np.asarray((c[:, lag:] * c[:, :-lag]).mean(1) / (c.var(1) + 1e-12))


def assess(raw: bytes, height: int, width: int) -> tuple[str, dict[str, list[float]]] | None:
    """(first-word kind, statistics) when the bytes pass every latent test at this shape, else None."""
    if height < 3 or width < 2:
        return None  # no neighbour half a row away, or two rows away, to test against
    a = np.frombuffer(raw, np.float32).copy()
    first = int.from_bytes(raw[:8], "little")
    kind = "POINTER" if USER_POINTER[0] <= first < USER_POINTER[1] else "DATA"
    if kind == "POINTER":
        a[:2] = a[2]
    if not np.isfinite(a).all() or float(np.abs(a).max()) >= LIMIT or float((a == 0).mean()) >= 0.01:
        return None
    x = a.reshape(CHANNELS, height, width).astype(np.float64)
    std = x.reshape(CHANNELS, -1).std(1)
    if not ((std >= CHANNEL_STD[0]) & (std <= CHANNEL_STD[1])).all():
        return None
    lag1 = _correlation(x, 1)
    vertical = {w: _correlation(x, w) for w in (width // 2, width, width * 2)}
    if lag1.min() < CHANNEL_LAG_MIN or lag1.mean() < LAG_MIN:
        return None
    row = vertical[width]
    if (row < CHANNEL_LAG_MIN).any() or (row <= vertical[width // 2]).any() or (
            vertical[width * 2] > row + WIDTH_MARGIN).any():
        return None
    return kind, {"lag1": [round(float(v), 4) for v in lag1], "vertical": [round(float(v), 4) for v in vertical[width]],
                  "channel_std": [round(float(s), 3) for s in std]}


def find(mem: ProcessMemory, sides: list[tuple[int, int]]) -> list[LatentBlock]:
    """Latents of the given (height, width) sides in resident private memory of ``mem``, largest first."""
    shapes = sorted(set(sides), key=lambda hw: -hw[0] * hw[1])
    found: list[LatentBlock] = []
    taken_until = 0
    for start, length, _phys in mem.runs:
        first = (max(start, taken_until) + SLOT - 1) // SLOT * SLOT
        for address in range(first, start + length, SLOT):
            # Runs can cross VAD boundaries: each slot is checked against the private VAD that holds it.
            vad = mem.vad(address)
            if vad is None or vad[2] or address < taken_until or not _plausible(mem.try_read(address + PAGE, PAGE)):
                continue
            for height, width in shapes:
                size = CHANNELS * height * width * 4
                if address + size - 1 > vad[1]:
                    continue
                raw = mem.try_read(address, size)
                verdict = assess(raw, height, width) if raw is not None else None
                if verdict is None:
                    continue
                offsets = [mem.physical_of(address + o) for o in range(0, size, PAGE)]
                found.append(LatentBlock(mem.pid, address, height, width, raw,  # type: ignore[arg-type]
                                         [o for o in offsets if o is not None], *verdict,
                                         stale_pages=mem.stale_pages(address, size)))
                taken_until = address + size
                break
    return found


def _plausible(page: bytes | None) -> bool:
    """Cheap test on the slot's second page before any full read."""
    if page is None:
        return False
    a = np.frombuffer(page, np.float32)
    if not np.isfinite(a).all() or float(np.abs(a).max()) >= LIMIT:
        return False
    return bool(CHANNEL_STD[0] <= float(a.std()) <= CHANNEL_STD[1])


def latent_sides(graphs: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Latent (height, width) of every EmptyLatentImage-like node in the recovered prompts, plus the defaults."""
    sides = set(DEFAULT_SIDES)
    for graph in graphs:
        for node in graph.values():
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            w, h = inputs.get("width"), inputs.get("height")
            if "Latent" in str(node.get("class_type", "")) and isinstance(w, int) and isinstance(h, int):
                if w % 8 == 0 and h % 8 == 0 and 0 < w <= 8192 and 0 < h <= 8192:
                    sides.add((h // 8, w // 8))
    return sorted(sides)


def document(case_dir: Path, blocks: list[LatentBlock], memory_sha256: str,
             sides: list[tuple[int, int]]) -> dict[str, Any]:
    """Writes each block's bytes under exports/raw and returns the sidecar document."""
    raw_dir = case_dir / "exports" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    latents = []
    for i, b in enumerate(blocks, 1):
        latent_id = f"LATENT-{i:03d}"
        path = raw_dir / f"{latent_id}.f32"
        with path.open("xb") as handle:  # exported evidence is never overwritten
            handle.write(b.raw)
        latents.append({"latent_id": latent_id, "process_id": b.process_id, "virtual_address": f"0x{b.address:X}",
                        "physical_offsets": [f"0x{p:X}" for p in b.physical_offsets],
                        "shape": [1, CHANNELS, b.height, b.width], "dtype": "float32",
                        "first_word": b.first_word, "statistics": b.stats,
                        "stale_pages": [f"0x{p:X}" for p in b.stale_pages],
                        "raw_file_path": str(path.relative_to(case_dir)),
                        "sha256": hashlib.sha256(b.raw).hexdigest(), "origin": "OBSERVED"})
    return {"schema_version": 1, "memory_sha256": memory_sha256,
            "method": {"slot_bytes": SLOT, "channels": CHANNELS, "value_limit": LIMIT, "channel_std": CHANNEL_STD,
                       "row_correlation_min": LAG_MIN, "channel_row_correlation_min": CHANNEL_LAG_MIN,
                       "width_margin": WIDTH_MARGIN, "shapes_tested": [list(s) for s in sides],
                       "holder": "resident pages of private VADs of ComfyUI Python processes"},
            "latents": latents}


def write(case_dir: Path, doc: dict[str, Any]) -> Path:
    target = case_dir / LATENTS_NAME
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(doc, indent=1) + "\n")
    except FileExistsError:
        raise FileExistsError(f"{target} exists; latents are written once") from None
    return target


def load(case_dir: Path) -> list[dict[str, Any]]:
    """Sidecar entries whose raw files still match their recorded hashes. Raises ValueError otherwise."""
    doc = json.loads((case_dir / LATENTS_NAME).read_text(encoding="utf-8"))
    for entry in doc["latents"]:
        if hashlib.sha256((case_dir / entry["raw_file_path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['raw_file_path']} does not match {LATENTS_NAME}")
    return list(doc["latents"])
