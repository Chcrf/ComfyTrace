"""Model identity from memory: are a local model file's weights in a ComfyUI process's private memory?

A recovered prompt names its checkpoint, but a name is not an identity: a renamed or fine-tuned file would
regenerate different images. Here the local file's own tensor bytes are the reference. Windows of WINDOW bytes
are sampled every STEP bytes of every tensor, and each window yields 8-byte keys at every element shift, in its
stored dtype (F16, BF16, F32) and widened to F32. A window is observed when at least NEED of its keys sit at one
consistent address, so single-word coincidences in unrelated data cannot count. A control set (the lowest bit of
every element flipped) must stay unobserved.

Only resident pages of private (unnamed) VADs of the ComfyUI Python process count. Pages of a file-backed VAD
(the safetensors mmap, DLLs) are file objects and are reported but never counted; pages outside the process are
not scanned. The denoiser (UNet) identifies a checkpoint; VAE and text-encoder weights are often shared between
checkpoints and are reported only.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comfytrace.procmem import ProcessMemory

STEP = 64 * 1024
WINDOW = 64
MIN_DISTINCT = 24
NEED = {"stored": 6, "f32": 12}
MIN_WINDOWS = 16
IDENTITY_NAME = "model-identity.json"
ELEMENT = {"F16": (np.uint16, 2), "BF16": (np.uint16, 2), "F32": (np.uint32, 4)}
COMPONENTS = (("model.diffusion_model.", "unet"), ("first_stage_model.", "vae"), ("cond_stage_model.", "clip"),
              ("conditioner.", "clip"), ("text_encoders.", "clip"))
OBSERVED = "OBSERVED"
SHARED_ONLY = "SHARED_COMPONENTS_ONLY"
NOT_OBSERVED = "NOT_OBSERVED"
CONTROL_FAILED = "INCONCLUSIVE_CONTROL"
NO_PROCESS_MEMORY = "NO_PROCESS_MEMORY"
SAFETENSORS_EXTENSIONS = (".safetensors", ".sft")
MODEL_EXTENSIONS = (*SAFETENSORS_EXTENSIONS, ".ckpt", ".pt", ".pth", ".gguf")


def component(tensor: str) -> str:
    return next((label for prefix, label in COMPONENTS if tensor.startswith(prefix)), "other")


def _widen(raw: np.ndarray, dtype: str) -> np.ndarray | None:
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32).view(np.uint32)
    if dtype == "BF16":
        return raw.astype(np.uint32) << 16
    return None


@dataclass
class Fingerprint:
    """Sorted 8-byte keys of a model file and, per key, which window and encoding it came from."""

    name: str
    path: Path
    sha256: str
    windows: list[tuple[str, int]]
    keys: np.ndarray
    window: np.ndarray
    offset: np.ndarray
    widened: np.ndarray
    control: np.ndarray

    @classmethod
    def of(cls, name: str, path: Path, step: int | None = None) -> Fingerprint:
        """Raises ValueError for a file that is not a complete safetensors container: no usable header, or a
        sampled tensor whose bytes lie beyond the end of the file (an interrupted download)."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(16 << 20), b""):
                digest.update(block)
        file_size = path.stat().st_size
        data = path.read_bytes() if file_size < (64 << 20) else None
        with open(path, "rb") as f:
            prefix = f.read(8)
            n = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else None
            if n is None or n > min(100 << 20, file_size - 8):
                raise ValueError(f"{path.name}: not a safetensors header")
            try:
                header = json.loads(f.read(n))
            except ValueError as exc:  # not JSON, or not UTF-8
                raise ValueError(f"{path.name}: not a safetensors header") from exc
            if not isinstance(header, dict):
                raise ValueError(f"{path.name}: not a safetensors header")
            base = 8 + n
            windows: list[tuple[str, int]] = []
            cols: dict[str, list[int]] = {"key": [], "window": [], "offset": [], "widened": [], "control": []}
            for tensor, info in header.items():
                dtype = info.get("dtype") if isinstance(info, dict) else None
                if tensor == "__metadata__" or not isinstance(dtype, str) or dtype not in ELEMENT:
                    continue
                kind, size = ELEMENT[dtype]
                offsets = info.get("data_offsets")
                if (not isinstance(offsets, list) or len(offsets) != 2 or not all(type(o) is int for o in offsets)
                        or not 0 <= offsets[0] <= offsets[1] or base + offsets[1] > file_size):
                    raise ValueError(f"{path.name}: tensor {tensor!r} lies outside the file")
                lo, hi = offsets
                for off in range(0, hi - lo - WINDOW + 1, step or STEP):
                    if data is not None:
                        chunk = data[base + lo + off: base + lo + off + WINDOW]
                    else:
                        f.seek(base + lo + off)
                        chunk = f.read(WINDOW)
                    elements = np.frombuffer(chunk, kind)
                    if len(np.unique(elements)) < MIN_DISTINCT * 2 // size:
                        continue
                    index = len(windows)
                    windows.append((tensor, off))
                    for control in (False, True):
                        stored = elements ^ 1 if control else elements
                        wide = _widen(stored, info["dtype"])
                        encodings = [(stored.tobytes(), size, False)]
                        if wide is not None:
                            encodings.append((wide.tobytes(), 4, True))
                        for raw, stride, widened in encodings:
                            for s in range(0, len(raw) - 7, stride):
                                cols["key"].append(struct.unpack_from("<Q", raw, s)[0])
                                cols["window"].append(index)
                                cols["offset"].append(s)
                                cols["widened"].append(widened)
                                cols["control"].append(control)
        keys = np.array(cols["key"], np.uint64)
        order = np.argsort(keys, kind="stable")
        return cls(name, path, digest.hexdigest(), windows, keys[order],
                   np.array(cols["window"], np.int32)[order], np.array(cols["offset"], np.int32)[order],
                   np.array(cols["widened"], bool)[order], np.array(cols["control"], bool)[order])


def observe(fp: Fingerprint, mem: ProcessMemory) -> Counter[tuple[int, bool, bool, int]]:
    """Matched-key counts per (window, widened, control, address of the window's first byte)."""
    counts: Counter[tuple[int, bool, bool, int]] = Counter()
    if not len(fp.keys):
        return counts
    for va, data in mem.chunks(64 << 20):
        head = (-va) % 8
        words = np.frombuffer(data, np.uint64, (len(data) - head) // 8, head)
        left = np.searchsorted(fp.keys, words, "left")
        right = np.searchsorted(fp.keys, words, "right")
        for p in np.nonzero(right > left)[0]:
            address = va + head + 8 * int(p)
            for k in range(int(left[p]), int(right[p])):
                counts[(int(fp.window[k]), bool(fp.widened[k]), bool(fp.control[k]),
                        address - int(fp.offset[k]))] += 1
    return counts


def assess(fp: Fingerprint, memories: list[ProcessMemory]) -> dict[str, Any]:
    sampled: Counter[str] = Counter(component(t) for t, _ in fp.windows)
    found: dict[str, dict[str, set[int]]] = {}
    control: set[int] = set()
    disputed: dict[tuple[int, int, int], dict[str, Any]] = {}
    for mem in memories:
        for (window, widened, is_control, address), n in observe(fp, mem).items():
            if n < NEED["f32" if widened else "stored"]:
                continue
            # The whole matched window, not just its first byte, must lie in one VAD. A window widened to FP32 holds
            # twice as many bytes in memory as in the file.
            span = 2 * WINDOW if widened else WINDOW
            vad = mem.vad(address)
            if vad is None or address + span - 1 > vad[1]:
                continue
            if is_control:
                control.add(window)
                continue
            comp = component(fp.windows[window][0])
            stale = mem.stale_pages(address, span)
            if stale:
                # A page whose PFN entry disputes its mapping may hold another moment's bytes: never identity.
                found.setdefault(comp, {}).setdefault("pfn_disputed", set()).add(window)
                tensor, offset = fp.windows[window]
                disputed.setdefault((mem.pid, window, address), {
                    "process_id": mem.pid, "component": comp, "tensor": tensor, "tensor_offset": offset,
                    "address": hex(address), "holder": "mapped_file" if vad[2] else "private",
                    "pages": [{"page": hex(p), "reason": mem.stale[p]} for p in stale]})
                continue
            holder = "mapped_file" if vad[2] else "private"
            found.setdefault(comp, {}).setdefault(holder, set()).add(window)
    components = {}
    for comp, total in sorted(sampled.items()):
        holders = found.get(comp, {})
        components[comp] = {"windows_sampled": total,
                            "private_windows": len(holders.get("private", ())),
                            "mapped_file_windows_not_counted": len(holders.get("mapped_file", ())),
                            "pfn_disputed_windows_not_counted": len(holders.get("pfn_disputed", ()))}
    identifying = "unet" if "unet" in sampled else "other"
    shared = [c for c in components if c != identifying]
    if control:
        verdict = CONTROL_FAILED
    elif components.get(identifying, {}).get("private_windows", 0) >= MIN_WINDOWS:
        verdict = OBSERVED
    elif any(components[c]["private_windows"] >= MIN_WINDOWS for c in shared):
        verdict = SHARED_ONLY
    else:
        verdict = NOT_OBSERVED
    return {"name": fp.name, "local_path": str(fp.path), "sha256": fp.sha256, "verdict": verdict,
            "identifying_component": identifying, "components": components, "control_windows": len(control),
            "pfn_disputed_matches": len(disputed),
            "pfn_disputed_match_details": [disputed[k] for k in sorted(disputed)],
            "process_ids": [m.pid for m in memories]}


def model_names(case_dir: Path) -> list[str]:
    from comfytrace.carve import graph_dependencies

    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    names: set[str] = set()
    for wf in manifest["findings"]["workflows"]:
        if wf["prompt_format"] == "API_GRAPH" and wf["encoding_detected"] == "CPYTHON_OBJECTS":
            graph = json.loads((case_dir / wf["raw_file_path"]).read_text(encoding="utf-8"))
            names |= set(graph_dependencies(graph)["model_assets"])
    return sorted(names)


def local_model_names(model_path: Path) -> tuple[Path, list[str]]:
    """Return ``(models_dir, relative_names)`` for a single checkpoint file or a directory of model files.

    Formats other than safetensors are listed so that ``identity_document`` reports them as ``UNSUPPORTED_FORMAT``
    instead of silently skipping them. Raises ValueError when a directory holds no model file."""
    path = Path(model_path)
    if path.is_file():
        return path.parent, [path.name]
    if path.is_dir():
        found = sorted(
            p.relative_to(path).as_posix()
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
        )
        if not found:
            raise ValueError(f"--model directory holds no model files ({', '.join(MODEL_EXTENSIONS)}): {path}")
        return path, found
    raise ValueError(f"--model path does not exist: {path}")


def identity_document(case_dir: Path | None, memories: list[ProcessMemory], models_dir: Path,
                      memory_sha256: str | None, names: list[str] | None = None) -> dict[str, Any]:
    from comfytrace.regenerate import _contained_path, _relative_parts

    target_names = names if names is not None else (model_names(case_dir) if case_dir is not None else [])
    models = []
    for name in target_names:
        if not memories:
            # No ComfyUI process memory was recovered: absence of evidence, never NOT_OBSERVED.
            models.append({"name": name, "verdict": NO_PROCESS_MEMORY})
            continue
        parts = _relative_parts(name)
        local = _contained_path(models_dir, parts) if parts else None
        if local is None or not local.is_file():
            models.append({"name": name, "verdict": "NO_LOCAL_COPY"})
            continue
        if not name.lower().endswith(SAFETENSORS_EXTENSIONS):
            models.append({"name": name, "local_path": str(local), "verdict": "UNSUPPORTED_FORMAT",
                           "reason": "only safetensors files can be fingerprinted"})
            continue
        try:
            fingerprint = Fingerprint.of(name, local)
        except ValueError as exc:
            # A truncated or damaged file is not the file ComfyUI loaded; it must not cost the other models theirs.
            models.append({"name": name, "local_path": str(local), "verdict": "UNSUPPORTED_FORMAT",
                           "reason": str(exc)})
            continue
        models.append(assess(fingerprint, memories))
    return {"schema_version": 1, "memory_sha256": memory_sha256,
            "method": {"window_bytes": WINDOW, "step_bytes": STEP, "keys_needed": NEED,
                       "min_identifying_windows": MIN_WINDOWS,
                       "counted_holder": "resident pages of private VADs of ComfyUI Python processes"},
            "models": models}


def write_identity(case_dir: Path, document: dict[str, Any]) -> Path:
    target = case_dir / IDENTITY_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; model identity is written once")
    target.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return target
