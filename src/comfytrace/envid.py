"""Runtime environment identity from memory: recover the captured ComfyUI software stack and compare it against a
lab replay runner.

Matching the captured ComfyUI version, PyTorch build, Python minor version, and GPU hardware removes known stack drift
that turns bit-exact diffusion replay into a perceptual match (SSIM ~0.98-0.999). It does not guarantee a bit-for-bit
reproduction: the CUDA driver, kernel selection and the GPU device index / allocator are not part of the verdict.

Only resident pages of private (unnamed) VADs are scanned, the same holder that ``modelid`` counts. Values are
accepted only when they end at a terminator (newline, NUL, quote, backslash or comma), so a banner truncated by a
chunk boundary or a missing page is rejected instead of being reported as a shorter version.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from comfytrace import __version__
from comfytrace.cpython.runtime import PYTHON_VERSION_MISMATCH
from comfytrace.cpython.runtime import detect as detect_runtime
from comfytrace.procmem import ProcessMemory

ENV_IDENTITY_NAME = "environment-identity.json"
ENV_IDENTITY_SCHEMA = 1

EXACT_STACK_MATCH = "EXACT_STACK_MATCH"
PARTIAL_STACK_MATCH = "PARTIAL_STACK_MATCH"
STACK_MISMATCH = "STACK_MISMATCH"
CAPTURED_ONLY = "CAPTURED_ONLY"
RUNNER_UNREPORTED = "RUNNER_UNREPORTED"
UNAVAILABLE = "UNAVAILABLE"

COUNTED_HOLDER = "resident pages of private VADs of ComfyUI Python processes"
COMPARED_FIELDS = ("comfyui_version", "pytorch_version", "python_version", "gpu_hardware")
COMFYUI_FIELDS = ("comfyui_version", "pytorch_version", "gpu_hardware", "vram_state")
OVERLAP = 256

# A value must end at one of these (or at the end of the scanned buffer, which is then discarded as truncated).
_TERM = rb"""(?=[\r\n\x00"'\\,]|$)"""
_VERSION = rb"(\d+\.\d+\.\d+(?:[.\-+a-zA-Z0-9]{0,24})?)"
_COMFYUI_RE = re.compile(rb"ComfyUI version:[ \t]*" + _VERSION + _TERM)
_PYTORCH_RE = re.compile(rb"pytorch version:[ \t]*" + _VERSION + _TERM)
_DEVICE_RE = re.compile(
    rb"Device:[ \t]*((?:cuda:\d+|cpu|mps|xpu:\d+|privateuseone(?::\d+)?|npu:\d+|mlu:\d+)"
    rb"[ \t:A-Za-z0-9_.\-()/]{0,96})" + _TERM)
_VRAM_RE = re.compile(rb"Total VRAM\s+(\d{2,7})\s*MB,\s*total RAM\s+(\d{2,7})\s*MB" + _TERM)
_VRAM_STATE_RE = re.compile(
    rb"Set vram state to:[ \t]*(NORMAL_VRAM|HIGH_VRAM|LOW_VRAM|NO_VRAM|SHARED|DISABLED)" + _TERM)
_ATTENTION_RE = re.compile(rb"(Using [A-Za-z0-9 _\-]{0,60}attention[A-Za-z0-9 _\-]{0,40})" + _TERM)
_PY_BANNER_RE = re.compile(rb"Python version:[ \t]*(3\.\d+\.\d+)" + rb"""(?=[ \r\n\x00"'\\,]|$)""")
_TOKENS = (b"version:", b"Device:", b"Total VRAM", b"vram state", b"attention")

_GPU_RE = re.compile(
    r"^(?P<dev>[a-z]+)(?::(?P<idx>\d+))?\s*(?P<name>.*?)(?:\s*:\s*(?P<alloc>\w+))?$", re.IGNORECASE)
_BARE_GPU_RE = re.compile(r"^(?P<name>.*?)(?:\s*:\s*(?P<alloc>\w+))?$", re.IGNORECASE)
_KNOWN_DEVICES = frozenset({"cuda", "cpu", "mps", "xpu", "privateuseone", "npu", "mlu", "hip", "directml"})


def _clean_text(raw: bytes) -> str | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text or any(bad in text for bad in ("{", "}", "%", ".format", "cannot ", "not a supported")):
        return None
    return " ".join(text.split())


def _pick_most_common(counts: Counter[str]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _scan_chunks(mem: ProcessMemory) -> Iterator[tuple[int, bytes]]:
    """Resident bytes of private (unnamed) VADs. Objects without VADs (simple test doubles) fall back to all chunks."""
    vads = getattr(mem, "vads", None)
    if not vads:
        yield from mem.chunks()
        return
    for start, end, name in vads:
        if not name:
            yield from mem.chunks(lo=start, hi=end + 1)


def _hits(mem: ProcessMemory) -> dict[str, Counter[str]]:
    hits: dict[str, Counter[str]] = {k: Counter() for k in (
        "comfyui_version", "pytorch_version", "gpu_hardware", "vram_summary", "vram_state", "attention_backend",
        "python_banner_version")}
    simple = (("comfyui_version", _COMFYUI_RE), ("pytorch_version", _PYTORCH_RE), ("gpu_hardware", _DEVICE_RE),
              ("attention_backend", _ATTENTION_RE), ("python_banner_version", _PY_BANNER_RE))
    prev_va: int | None = None
    prev_len = 0
    tail = b""
    for va, data in _scan_chunks(mem):
        carry = tail if prev_va is not None and prev_va + prev_len == va else b""
        buffer = carry + data if carry else data
        prev_va, prev_len, tail = va, len(data), buffer[-OVERLAP:]
        if not any(token in buffer for token in _TOKENS):
            continue

        def accepted(match: re.Match[bytes], n: int = len(carry), size: int = len(buffer)) -> bool:
            # Ending at the buffer's end means the value may continue past a page gap or into the next chunk (where the
            # carried tail finds it again); ending inside the carried tail means the previous buffer already counted it.
            return n <= match.end() < size

        for key, regex in simple:
            for match in regex.finditer(buffer):
                if accepted(match) and (value := _clean_text(match.group(1))):
                    hits[key][value] += 1
        for match in _VRAM_RE.finditer(buffer):
            if accepted(match):
                vram, ram = match.group(1).decode("ascii"), match.group(2).decode("ascii")
                hits["vram_summary"][f"Total VRAM {vram} MB, total RAM {ram} MB"] += 1
        for match in _VRAM_STATE_RE.finditer(buffer):
            if accepted(match):
                hits["vram_state"][match.group(1).decode("ascii")] += 1
    return hits


def detect_process_environment(mem: ProcessMemory) -> dict[str, Any]:
    """Extract the captured ComfyUI runtime stack from one live Python process's private virtual memory."""
    rt = detect_runtime(mem)
    status = rt.unsupported
    python_version = f"{rt.version[0]}.{rt.version[1]}" if rt.version else None
    if status and status.startswith(PYTHON_VERSION_MISMATCH):
        python_version = None  # the file name and the exported Py_Version disagree: neither is trusted
    source = "RUNTIME_IMAGE" if python_version else None

    hits = _hits(mem)
    banner = _pick_most_common(hits["python_banner_version"])
    if python_version is None and banner:
        python_version, source = ".".join(banner.split(".")[:2]), "STARTUP_BANNER"

    return {
        "pid": mem.pid,
        "python_version": python_version,
        "python_version_source": source,
        "python_runtime_status": status,
        "python_image": getattr(rt.image, "name", rt.image),
        "python_banner_version": banner,
        "comfyui_version": _pick_most_common(hits["comfyui_version"]),
        "pytorch_version": _pick_most_common(hits["pytorch_version"]),
        "gpu_hardware": _pick_most_common(hits["gpu_hardware"]),
        "vram_summary": _pick_most_common(hits["vram_summary"]),
        "vram_state": _pick_most_common(hits["vram_state"]),
        "attention_backend": _pick_most_common(hits["attention_backend"]),
        "candidates": {k: dict(sorted(v.items())) for k, v in hits.items() if v},
    }


def parse_gpu(value: str | None) -> dict[str, Any] | None:
    """Split ``cuda:0 NVIDIA GeForce RTX 3070 : cudaMallocAsync`` into device type, index, name and allocator."""
    if not value or value == "UNREPORTED":
        return None
    text = " ".join(value.strip().split())
    match = _GPU_RE.match(text)
    if match and match["dev"].lower() in _KNOWN_DEVICES:
        device = match["dev"].lower()
        index = int(match["idx"]) if match["idx"] else None
        name, alloc = match["name"], match["alloc"]
    else:
        bare = _BARE_GPU_RE.match(text)
        assert bare is not None
        device, index, name, alloc = None, None, bare["name"], bare["alloc"]
    return {"device": device, "index": index, "name": " ".join(name.lower().split()), "allocator": alloc}


def _gpu_same(captured: dict[str, Any], runner: dict[str, Any]) -> bool:
    """Same device type and name. The device index and the allocator are advisory and never decide a verdict."""
    if captured["device"] and runner["device"] and captured["device"] != runner["device"]:
        return False
    return bool(captured["name"] == runner["name"])


def _gpu_key(value: str) -> str:
    parsed = parse_gpu(value)
    return value if parsed is None else f"{parsed['device']}|{parsed['name']}"


def _python_minor(value: str | None) -> str | None:
    if not value or value == "UNREPORTED":
        return None
    match = re.match(r"(\d+\.\d+)", value.strip())
    return match.group(1) if match else value.strip()


def _norm(value: str | None) -> str | None:
    if not value or value == "UNREPORTED":
        return None
    return " ".join(value.strip().lower().split())


def _has_comfyui_field(record: dict[str, Any]) -> bool:
    return any(record.get(k) for k in COMFYUI_FIELDS)


def _context_by_pid(process_context: Iterable[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    return {c["pid"]: c for c in process_context or () if isinstance(c, dict) and c.get("pid") is not None}


def _is_comfyui_process(context: dict[str, Any] | None) -> bool:
    if not context:
        return False
    cmdline = str(context.get("command_line") or "").lower()
    return context.get("role") == "WORKER_CHILD" or "comfy" in cmdline or "main.py" in cmdline


def find_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fields for which ComfyUI-bearing processes report different non-null values."""
    pool = [r for r in records if _has_comfyui_field(r)] or records
    normalizers = {"python_version": _python_minor, "gpu_hardware": _gpu_key}
    conflicts = []
    for key in (*COMPARED_FIELDS, "vram_state", "vram_summary"):
        grouped: dict[str, dict[str, Any]] = {}
        for record in pool:
            if (value := record.get(key)) is None:
                continue
            normalized = normalizers.get(key, _norm)(value)
            entry = grouped.setdefault(str(normalized), {"value": value, "pids": []})
            entry["pids"].append(record["pid"])
        if len(grouped) > 1:
            conflicts.append({"field": key, "values": list(grouped.values())})
    return conflicts


def detect_captured_environment(
    processes: Iterable[ProcessMemory],
    process_context: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Scan all candidate Python processes and return ``(best_captured_env, per_process_records)``.

    ``process_context`` (``IngestionResult.process_context``) lets ComfyUI worker processes win over unrelated Python
    processes. Disagreeing non-null values are recorded in ``best["conflicts"]``; a Python version taken from a process
    with no ComfyUI-specific field is marked ``python_version_attribution: UNATTRIBUTED``."""
    contexts = _context_by_pid(process_context)
    records = [detect_process_environment(pm) for pm in processes]
    if not records:
        empty = {
            "pid": None, "python_version": None, "python_image": None,
            "comfyui_version": None, "pytorch_version": None, "gpu_hardware": None,
            "vram_summary": None, "vram_state": None, "python_version_attribution": None, "conflicts": [],
        }
        return empty, []
    keys = (*COMFYUI_FIELDS, "python_version")

    def rank(record: dict[str, Any]) -> tuple[bool, bool, int, int]:
        return (_has_comfyui_field(record), _is_comfyui_process(contexts.get(record["pid"])),
                sum(1 for k in keys if record.get(k)), -(record["pid"] or 0))

    best = dict(max(records, key=rank))
    best["python_version_attribution"] = (
        None if best.get("python_version") is None
        else "COMFYUI_PROCESS" if _has_comfyui_field(best) else "UNATTRIBUTED")
    best["conflicts"] = find_conflicts(records)
    return best, records


def compare_environments(captured: dict[str, Any], runner: dict[str, Any] | None) -> dict[str, Any]:
    """Compare captured process stack against a live replay runner's ``runner_environment``."""
    if not any(captured.get(k) for k in COMPARED_FIELDS):
        return {"verdict": UNAVAILABLE, "mismatches": [], "fields": {}}
    conflicted = {c["field"] for c in captured.get("conflicts") or ()}
    unattributed = captured.get("python_version_attribution") == "UNATTRIBUTED"
    if runner is None:
        fields = {
            k: {"captured": captured.get(k), "runner": None,
                "status": "CAPTURED_ONLY" if captured.get(k) else "UNREPORTED"}
            for k in COMPARED_FIELDS
        }
        return {"verdict": CAPTURED_ONLY, "mismatches": [], "fields": fields}
    compared: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    unreported = False
    runner_values = {k: (None if runner.get(k) == "UNREPORTED" else runner.get(k)) for k in COMPARED_FIELDS}
    if not any(runner_values.values()):
        return {"verdict": RUNNER_UNREPORTED, "mismatches": [],
                "fields": {k: {"captured": captured.get(k), "runner": None, "status": "UNREPORTED"}
                           for k in COMPARED_FIELDS}}
    for key in COMPARED_FIELDS:
        cap_val, run_val = captured.get(key), runner_values[key]
        extra: dict[str, Any] = {}
        if key == "python_version" and unattributed and cap_val is not None:
            status = "UNATTRIBUTED"
            unreported = True
        elif cap_val is None or run_val is None:
            status = "UNREPORTED"
            unreported = True
        else:
            if key == "python_version":
                same = _python_minor(cap_val) == _python_minor(run_val)
            elif key == "gpu_hardware":
                cap_gpu, run_gpu = parse_gpu(cap_val), parse_gpu(run_val)
                same = cap_gpu is not None and run_gpu is not None and _gpu_same(cap_gpu, run_gpu)
                if cap_gpu and run_gpu:
                    extra["advisory"] = {
                        "captured_device_index": cap_gpu["index"], "runner_device_index": run_gpu["index"],
                        "captured_allocator": cap_gpu["allocator"], "runner_allocator": run_gpu["allocator"]}
            else:
                same = _norm(cap_val) == _norm(run_val)
            status = "MATCH" if same else "MISMATCH"
            if not same:
                mismatches.append(key)
        compared[key] = {"captured": cap_val, "runner": run_val, "status": status, **extra}
    if mismatches:
        verdict = STACK_MISMATCH
    elif unreported or conflicted & set(COMPARED_FIELDS):
        verdict = PARTIAL_STACK_MATCH
    else:
        verdict = EXACT_STACK_MATCH
    return {"verdict": verdict, "mismatches": mismatches, "fields": compared}


def environment_document(
    processes: Iterable[ProcessMemory],
    memory_sha256: str | None = None,
    runner_env: dict[str, Any] | None = None,
    runner_url: str | None = None,
    process_context: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    captured, per_process = detect_captured_environment(processes, process_context)
    comparison = compare_environments(captured, runner_env)
    candidates = {str(r["pid"]): r["candidates"] for r in per_process if r.get("candidates")}
    per_process = [{k: v for k, v in r.items() if k != "candidates"} for r in per_process]
    captured = {k: v for k, v in captured.items() if k != "candidates"}
    return {
        "schema_version": ENV_IDENTITY_SCHEMA,
        "memory_sha256": memory_sha256,
        "verdict": comparison["verdict"],
        "method": {"counted_holder": COUNTED_HOLDER, "comfytrace_version": __version__, "candidates": candidates},
        "captured_environment": captured,
        "conflicts": captured.get("conflicts", []),
        "runner_url": runner_url,
        "runner_environment": runner_env,
        "comparison": comparison,
        "processes": per_process,
    }


def write_environment_identity(case_dir: Path, document: dict[str, Any]) -> Path:
    """Writes ``environment-identity.json`` into ``case_dir`` once. Raises FileExistsError if it is already there."""
    target = Path(case_dir) / ENV_IDENTITY_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; environment identity is written once")
    payload = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".envid_", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        try:
            os.link(tmp_name, target)  # never replaces an existing file
        except FileExistsError:
            raise
        except OSError:  # filesystem without hard links: claim the name exclusively, then fill it
            claim = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(claim, "wb") as handle:
                handle.write(payload)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return target


def format_comparison_table(document: dict[str, Any]) -> str:
    """Format a concise Captured vs. Runner summary table for terminal display."""
    comp = document["comparison"]
    captured = document["captured_environment"]
    runner = document.get("runner_environment")
    labels = (
        ("comfyui_version", "ComfyUI"),
        ("pytorch_version", "PyTorch build"),
        ("python_version", "Python"),
        ("gpu_hardware", "GPU / Device"),
    )
    lines = [f"Verdict: {document['verdict']}"]
    if captured.get("pid") is not None:
        lines.append(f"Captured PID: {captured['pid']}")
    if runner is None:
        for key, label in labels:
            lines.append(f"  {label:15s}: {captured.get(key) or 'UNREPORTED'}")
    else:
        for key, label in labels:
            row = comp["fields"].get(key) or {}
            lines.append(
                f"  {label:15s}: captured={row.get('captured') or 'UNREPORTED'} | "
                f"runner={row.get('runner') or 'UNREPORTED'} [{row.get('status', 'UNREPORTED')}]"
            )
    if captured.get("vram_summary"):
        lines.append(f"  {'VRAM / RAM':15s}: {captured['vram_summary']}"
                     + (f" ({captured['vram_state']})" if captured.get("vram_state") else ""))
    if captured.get("python_version_attribution") == "UNATTRIBUTED":
        lines.append("  Note: the Python version comes from a process with no ComfyUI banner (UNATTRIBUTED)")
    for conflict in document.get("conflicts") or captured.get("conflicts") or ():
        lines.append(f"  Conflict: {conflict['field']} differs between processes "
                     f"({', '.join(str(v['value']) for v in conflict['values'])})")
    gpu = (comp["fields"].get("gpu_hardware") or {}).get("advisory")
    if gpu and (gpu["captured_device_index"] != gpu["runner_device_index"]
                or gpu["captured_allocator"] != gpu["runner_allocator"]):
        lines.append("  Note: GPU index and allocator differ and are ignored in the verdict")
    return "\n".join(lines)
