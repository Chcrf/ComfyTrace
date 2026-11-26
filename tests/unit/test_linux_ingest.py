from __future__ import annotations

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
from volatility3.framework.exceptions import InvalidAddressException

from comfytrace import ingest, ingest_linux
from comfytrace.cli import parse_args, run_extract
from comfytrace.elfcore import HEADER, PROGRAM, CoreLayer
from comfytrace.ingest import SymbolManager, ingest_case
from comfytrace.ingest_linux import UNKNOWN_BACKING, LinuxProcessWalker
from tests.synthetic import api_prompt, noise_image, png_bytes, prompt_request_body

START = 0x100000
PID = 3215


def _note(kind: int, payload: bytes) -> bytes:
    return struct.pack("<III", 5, len(payload), kind) + b"CORE\0\0\0\0" + payload + bytes(-len(payload) % 4)


def _core(payload: bytes, *, memory_size: int | None = None) -> bytes:
    status = bytearray(136)
    struct.pack_into("<i", status, 24, PID)
    status[40:46] = b"python"
    argv = b"python /opt/ComfyUI/main.py"
    status[56:56 + len(argv)] = argv
    file_note = struct.pack("<QQQQQ", 1, 4096, START, START + 64, 0) + b"/lib/libfixture.so\0"
    notes = _note(3, status) + _note(0x46494C45, file_note)
    offset = HEADER.size + PROGRAM.size * 2
    header = HEADER.pack(b"\x7fELF\x02\x01\x01" + bytes(9), 4, 62, 1, 0, HEADER.size, 0, 0,
                         HEADER.size, PROGRAM.size, 2, 0, 0, 0)
    table = PROGRAM.pack(4, 0, offset, 0, 0, len(notes), 0, 4) + PROGRAM.pack(
        1, 6, offset + len(notes), START, 0, len(payload), memory_size or len(payload), 4096)
    return header + table + notes + payload


def test_elf_core_exposes_only_dumped_bytes_and_preserves_unknown_backing():
    core = CoreLayer(_core(b"x" * 4096, memory_size=8192))
    assert core.pid == PID and core.regions[0] == (START, START + 63, "/lib/libfixture.so")
    assert core.regions[1][2] == "<ELF core mapping origin unverified>"
    assert core.read(START + 4090, 6) == b"x" * 6
    with pytest.raises(InvalidAddressException):
        core.read(START + 4090, 7, pad=True)
    with pytest.raises(ValueError, match="extends beyond"):
        CoreLayer(_core(b"x" * 4096)[:-1])


def test_elf_core_cli_carves_workflow_and_image_without_physical_offsets(tmp_path):
    graph = api_prompt(width=32, height=32, text="A little wooden sailboat")
    png = png_bytes(noise_image(32, 32), {"prompt": json.dumps(graph)})
    dump = tmp_path / "python.core"
    dump.write_bytes(_core(bytes(64) + prompt_request_body(graph) + bytes(64) + png))
    case = tmp_path / "case"
    args = ["extract", "--memory", str(dump), "--outdir", str(case), "--image-os", "linux",
            "--image-format", "elf-core"]
    assert run_extract(parse_args(args + ["--comfyui-pid", str(PID + 1)])) == 1
    assert run_extract(parse_args(args + ["--comfyui-pid", str(PID)])) in (0, 2)
    manifest = json.loads((case / "manifest.json").read_bytes())
    assert manifest["case_metadata"]["symbol_resolution"]["image_format"] == "elf-core"
    [image] = manifest["findings"]["artifacts"]
    assert image["process_id"] == PID and image["physical_offsets"] == []
    assert image["container_sha256"] == hashlib.sha256(png).hexdigest()
    assert (case / image["raw_file_paths"][0]).read_bytes() == png
    assert (case / "workbench.html").is_file() and not (case / "report.html").exists()


def test_linux_task_matching_uses_script_location_and_rejects_generic_main(monkeypatch):
    walker = LinuxProcessWalker(SimpleNamespace(), "kernel")
    monkeypatch.setattr(walker, "_cwd", lambda proc: proc.cwd)
    assert walker._matches_command(SimpleNamespace(cwd="/opt/ComfyUI"), "python3 -u main.py --cpu")
    assert walker._matches_command(SimpleNamespace(cwd="/tmp"), "python3 '/Shared folder/ComfyUI/main.py'")
    assert not walker._matches_command(SimpleNamespace(cwd="/tmp"), "python3 main.py --comfyui")
    assert not walker._matches_command(SimpleNamespace(cwd="/opt/ComfyUI"), "python3 -c main.py")
    file_vma = SimpleNamespace(vm_start=100, vm_end=200, vm_file=1, get_name=lambda *_: None)
    assert walker._region(None, file_vma) == (100, 199, UNKNOWN_BACKING)


def test_supplementary_ingestion_reuses_recorded_platform_and_scope(tmp_path):
    def check(path, symbol_manager, *, examiner_pids):
        assert path == tmp_path and symbol_manager.symbols_dir is None
        assert symbol_manager.image_os == "linux" and symbol_manager.image_format == "elf-core"
        return examiner_pids

    meta = {"symbol_resolution": {"image_os": "linux", "image_format": "elf-core"},
            "examiner_comfyui_pids": [PID]}
    assert ingest_case(tmp_path, None, meta, check) == [PID]


@pytest.mark.parametrize("is_flat", [True, False])
def test_linux_kernel_adapter_reuses_scoped_virtual_layers_without_windows_structures(tmp_path, monkeypatch, is_flat):
    dump = tmp_path / "memory.raw"
    dump.write_bytes(b"W" * 4096 + b"L" * 4096)

    class Layer:
        def mapping(self, start, length, ignore_errors=False):
            yield start, length, 4096 + start - START, length, "capture"

        def read(self, _offset, length, pad=False):
            assert not pad
            return b"L" * length

    class Walker:
        def __init__(self, context, kernel):
            assert kernel == "kernel"

        def enumerate_comfyui_hierarchy(self):
            return [{"pid": PID, "process_name": "python3.12", "dtb": None}]

        def get_process_layer(self, pid):
            assert pid == PID
            return Layer()

        def get_vad_details(self, _pid):
            return [(START, START + 4095, None)]

        def get_process_vads(self, _pid):
            return [(START, START + 4095)]

        _is_python = staticmethod(LinuxProcessWalker._is_python)

    def refuse(*_args):
        pytest.fail("Linux ingestion called Windows PFN recovery")

    monkeypatch.setattr(ingest.VolatilityEngine, "initialize_context", lambda _: object())
    monkeypatch.setattr(ingest_linux, "LinuxProcessWalker", Walker)
    monkeypatch.setattr(ingest, "is_raw_physical", lambda _: is_flat)
    monkeypatch.setattr(ingest, "kernel_support", refuse)
    result = ingest.ingest_memory(dump, SymbolManager(tmp_path, image_os="linux"))
    try:
        assert result.symbol_resolution["image_os"] == "linux" and result.mode == ingest.MODE_VIRTUAL
        [process] = result.recovery.python_processes
        assert process.read(START, 16) == b"L" * 16
        assert process.physical_of(START) == (4096 if is_flat else None)
        assert result.spaces[0].physical_mode is False
        assert result.spaces[0].source.read_prefix(START, 16) == (b"L" * 16, None)
        if is_flat:
            assert result.spaces[1].skip_pages == frozenset({1})
        else:
            assert len(result.spaces) == 1 and result.spaces[0].translate is None
            assert not ingest.supports_raw_search({"symbol_resolution": result.symbol_resolution})
    finally:
        result.close()


def test_segmented_capture_never_falls_back_to_physical_file_offsets(tmp_path, monkeypatch):
    dump = tmp_path / "capture.lime"
    dump.write_bytes(b"EMiL" + bytes(4092))
    monkeypatch.setattr(ingest.VolatilityEngine, "initialize_context", lambda _: object())

    class NoProcesses:
        unreadable_python_command_lines = False

        def __init__(self, *_args):
            pass

        def enumerate_comfyui_hierarchy(self):
            return []

    monkeypatch.setattr(ingest_linux, "LinuxProcessWalker", NoProcesses)
    result = ingest.ingest_memory(dump, SymbolManager(tmp_path, image_os="linux"))
    assert result.degradation_reason == "NO_COMFYUI_PROCESS"
    assert result.symbol_resolution["capture_layout"] == "SEGMENTED"
    assert not result.spaces and result.recovery.physical is None
    assert not ingest.supports_raw_search({}, dump)
