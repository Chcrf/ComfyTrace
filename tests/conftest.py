"""Synthetic memory images for the T1/T2/T3 evaluation states and the multi-workflow negative test.

T1, T2 and MULTI_WORKFLOW_NEGATIVE are process-attributed states. A real Windows kernel image cannot be
synthesized here, so their process list and page tables are served by `fake_ingestor` at the ingestion
boundary, while every byte still comes from the dump file. T3 runs the real Volatility path, which
fails offline symbol resolution and falls back to physical carving.
"""

from __future__ import annotations

import http.server
import io
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from PIL import Image

from comfytrace.carve import StreamReader
from comfytrace.ingest import (
    MODE_VIRTUAL,
    IngestionResult,
    ProcessWalker,
    SymbolManager,
    VolatilityEngine,
    ingest_memory,
    process_space,
)
from tests.synthetic import (
    PAGE,
    DumpBuilder,
    FakeProcess,
    FakeProcLayer,
    FakeVad,
    api_prompt,
    noise_image,
    pages_for,
    png_bytes,
    prompt_request_body,
)

State = Literal["T1", "T2", "T3", "MULTI_WORKFLOW_NEGATIVE"]
WORKER_PID, ROOT_PID = 13484, 19556
ROOT_CMD = r"C:\ComfyUI\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build"
WORKER_CMD = r"C:\ComfyUI\python_embeded\python.exe main.py --listen"


@dataclass
class ProcessLayout:
    processes: list[FakeProcess]
    layers: dict[int, FakeProcLayer]


LAYOUTS: dict[Path, ProcessLayout] = {}


class _Frames:
    """Hands out scattered, non-reused physical frames so virtual contiguity never implies physical."""

    def __init__(self, total: int) -> None:
        self.free = list(range(total - 1, 2, -3)) + list(range(total - 2, 2, -3)) + list(range(total - 3, 2, -3))

    def take(self, n: int) -> list[int]:
        out, self.free = self.free[:n], self.free[n:]
        return out


def _place(dump: DumpBuilder, layer: FakeProcLayer, frames: _Frames, va: int, payload: bytes,
           trailing_unmapped: int = 1) -> tuple[int, int]:
    """Maps payload at va onto scattered frames; returns the VAD (inclusive) including unmapped tail pages."""
    count = pages_for(len(payload))
    chosen = frames.take(count)
    for i, frame in enumerate(chosen):
        dump.put(frame, payload[i * PAGE:(i + 1) * PAGE])
    layer.map_virtual(va, chosen)
    return va, va + (count + trailing_unmapped) * PAGE - 1


def _processes(vads: list[tuple[int, int]]) -> list[FakeProcess]:
    return [
        FakeProcess(4, 0, "System", None),
        FakeProcess(7000, 4, "explorer.exe", None),
        FakeProcess(ROOT_PID, 7000, "python.exe", ROOT_CMD, ActiveThreads=6, vads=[(0x10000, 0x10FFF)]),
        FakeProcess(WORKER_PID, ROOT_PID, "python.exe", WORKER_CMD, ActiveThreads=26, vads=vads),
        FakeProcess(15000, 7000, "python.exe", r"python.exe C:\tools\backup.py"),
    ]


def build_memory_image(tmp_path: Path, state: State) -> Path:
    path = tmp_path / f"memdump_{state}.raw"
    dump = DumpBuilder(frames=256)
    frames = _Frames(256)
    if state == "T3":
        img = noise_image(96, 96, seed=31)
        png = png_bytes(img, idat_chunk=900)
        start = 40 * PAGE + 200
        dump.data[start:start + len(png)] = png
        torn = (start // PAGE + 3) * PAGE
        dump.data[torn:torn + PAGE] = bytes((i * 7 + 3) & 0xFF for i in range(PAGE))
        return dump.write(path)

    layer = FakeProcLayer(b"", {})
    root_layer = FakeProcLayer(b"", {})
    vads = []
    if state == "T1":
        graph = api_prompt()
        ui_workflow = json.dumps({"nodes": [{"id": 3, "type": "KSampler"}], "links": []})
        png = png_bytes(noise_image(64, 48, seed=5), {"prompt": json.dumps(graph), "workflow": ui_workflow})
        vads.append(_place(dump, layer, frames, 0x213BC000000, png))
        vads.append(_place(dump, layer, frames, 0x213B8E20000, prompt_request_body(graph)))
    elif state == "T2":
        png = png_bytes(noise_image(48, 48, seed=8))
        vads.append(_place(dump, layer, frames, 0x1F0A0000000, b"\x00" * 128 + png))
        fragment = json.dumps({"class_type": "KSampler", "inputs": {"seed": 167346845307033, "steps": 20}})
        heap = b"\x00\x00" + fragment.encode() + b"\x00\x13\x07"
        start, end = _place(dump, layer, frames, 0x1F0B0000000, heap.ljust(3 * PAGE, b"\x01"), trailing_unmapped=0)
        layer.swapped.add(start // PAGE + 1)
        vads.append((start, end))
    elif state == "MULTI_WORKFLOW_NEGATIVE":
        graphs = [api_prompt(seed=1000 + k, text=f"subject {k}") for k in range(3)]
        drifted = json.loads(json.dumps(graphs[2]))
        drifted["3"]["inputs"]["denoise"] = 0.75
        for k, g in enumerate([graphs[0], graphs[1], drifted]):
            png = png_bytes(noise_image(64, 48, seed=40 + k), {"prompt": json.dumps(g)})
            vads.append(_place(dump, layer, frames, 0x300000000 + k * 0x1000000, png))
        for k, g in enumerate(graphs):
            body = prompt_request_body(g, prompt_id=f"00000000-0000-0000-0000-00000000000{k}")
            vads.append(_place(dump, layer, frames, 0x400000000 + k * 0x1000000, body))
    else:
        raise ValueError(state)
    path = dump.write(path)
    physical = path.read_bytes()
    layer.physical = physical
    root_layer.physical = physical
    LAYOUTS[path] = ProcessLayout(_processes(vads), {WORKER_PID: layer, ROOT_PID: root_layer})
    return path


def fake_ingest(memory: Path, manager: SymbolManager) -> IngestionResult:
    """Process-attributed ingestion over a registered layout; real ingestion for anything else."""
    layout = LAYOUTS.get(Path(memory))
    if layout is None:
        return ingest_memory(memory, manager)
    context = MagicMock()
    context.modules = {"kernel": MagicMock(symbol_table_name="symtab")}
    walker = ProcessWalker(
        context,
        "kernel",
        list_processes=lambda ctx, name: iter(layout.processes),
        get_cmdline=lambda ctx, table, proc: proc.cmdline,
        list_vads=lambda proc: [FakeVad(s, e) for s, e in proc.vads],
    )
    hierarchy = walker.enumerate_comfyui_hierarchy()
    engine = VolatilityEngine.__new__(VolatilityEngine)
    reader = StreamReader(memory)
    spaces = []
    for proc in hierarchy:
        layer = layout.layers[proc["pid"]]
        vads = walker.get_process_vads(proc["pid"])
        spaces.append(process_space(proc["pid"], layer, vads, reader, engine))
    status = {"mode": manager.mode, "symbols_dir": str(manager.symbols_dir), "remote_download_permitted": False,
              "status": "SUCCESS"}
    return IngestionResult(MODE_VIRTUAL, status, hierarchy, spaces)


@pytest.fixture
def mock_memory_image(tmp_path: Path) -> Callable[[State], Path]:
    """Factory for synthetic memory images in states T1, T2, T3 and MULTI_WORKFLOW_NEGATIVE.
    - T1: active ComfyUI worker with an intact PNG (embedded prompt) and the matching /prompt body.
      Expected: 1 Tier E2 finding (FIND-001), exit 0.
    - T2: residual heap with an intact metadata-free PNG and a lone KSampler node fragment.
      Expected: 1 Tier E6 finding, 1 untiered orphan image, exit 2.
    - T3: unresolvable symbols; a PNG torn by a reallocated physical page. Expected: 1 untiered orphan
      image with a NONCONTIGUOUS_PHYSICAL gap, exit 3.
    - MULTI_WORKFLOW_NEGATIVE: 3 workflows, 3 images, one image's graph drifted. Expected: 0 false STRICT."""
    yield lambda state: build_memory_image(tmp_path, state)
    LAYOUTS.clear()


@pytest.fixture
def fake_ingestor() -> Callable[[Path, SymbolManager], IngestionResult]:
    return fake_ingest


@pytest.fixture
def mock_volatility_context() -> MagicMock:
    """A Volatility-like context whose kernel module, process list, command lines and VADs are fakes."""
    context = MagicMock()
    context.modules = {"kernel": MagicMock(symbol_table_name="symtab")}
    context.processes = _processes([(0x20000, 0x2FFFF)])
    return context


@pytest.fixture
def symbols_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "symbols"
    folder.mkdir()
    return folder


RUNNER_PROMPT_ID = "d6c4a8f1-34e2-4f32-b7d1-08f23491ba02"


def _runner_png() -> bytes:
    buf = io.BytesIO()
    noise_image(64, 48, seed=5).save(buf, format="PNG")
    return buf.getvalue()


class RunnerState:
    def __init__(self) -> None:
        self.image: bytes = _runner_png()
        self.object_info: dict[str, Any] = {
            name: {"input": {}} for name in
            ("KSampler", "CheckpointLoaderSimple", "EmptyLatentImage", "CLIPTextEncode", "VAEDecode", "SaveImage")
        }
        self.object_info["CheckpointLoaderSimple"] = {
            "input": {"required": {"ckpt_name": [["v1-5-pruned-emaonly.safetensors", "legacy.ckpt"]]}}
        }
        self.submitted: list[dict[str, Any]] = []
        self.system_stats = {"system": {"comfyui_version": "0.35.0", "python_version": "3.12.2 (main)",
                                        "pytorch_version": "2.8.0+cu128"},
                             "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3070 Laptop GPU : cudaMallocAsync"}]}


@pytest.fixture
def mock_runner_server() -> Iterator[tuple[str, RunnerState]]:
    """Ephemeral local HTTP server with canned ComfyUI endpoints: /object_info, /prompt,
    /history/{id}, /view and /system_stats. Yields (base URL, mutable state)."""
    state = RunnerState()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _json(self, payload: Any, code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/object_info":
                self._json(state.object_info)
            elif self.path == "/system_stats":
                self._json(state.system_stats)
            elif self.path.startswith(f"/history/{RUNNER_PROMPT_ID}"):
                self._json({RUNNER_PROMPT_ID: {"outputs": {"9": {"images": [
                    {"filename": "out.png", "subfolder": "", "type": "output"}]}}, "status": {"completed": True}}})
            elif self.path.startswith("/view"):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(state.image)))
                self.end_headers()
                self.wfile.write(state.image)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            if self.path == "/prompt":
                state.submitted.append(payload)
                self._json({"prompt_id": RUNNER_PROMPT_ID, "number": 1})
            else:
                self._json({"error": "not found"}, 404)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()


def load_png(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.load()
        return img.copy()
