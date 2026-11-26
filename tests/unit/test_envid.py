"""Unit tests for captured vs. runner environment identity detection (``envid``) and the standalone
``env-id`` / ``model-id`` commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfytrace import envid, modelid
from comfytrace.cli import EXIT_DEGRADED, EXIT_FAILURE, EXIT_OK, parse_args, run_env_id, run_model_id
from comfytrace.envid import (
    CAPTURED_ONLY,
    ENV_IDENTITY_NAME,
    EXACT_STACK_MATCH,
    PARTIAL_STACK_MATCH,
    RUNNER_UNREPORTED,
    STACK_MISMATCH,
    UNAVAILABLE,
    compare_environments,
    detect_captured_environment,
    detect_process_environment,
    environment_document,
    format_comparison_table,
    parse_gpu,
    write_environment_identity,
)
from comfytrace.procmem import ProcessMemory

BASE = 0x100000
GPU = "cuda:0 NVIDIA GeForce RTX 3070 Laptop GPU : cudaMallocAsync"
BANNER = (
    b"ComfyUI version: 0.44.0\n"
    b"pytorch version: 2.12.1+cu130\n"
    b"Device: " + GPU.encode() + b"\n"
    b"Total VRAM 8192 MB, total RAM 8191 MB\n"
    b"Set vram state to: NORMAL_VRAM\n"
)


class _FakeProc:
    """A process whose resident memory is a list of ``(va, bytes)`` chunks. No ``vads`` attribute: all chunks count."""

    def __init__(self, pid: int, blobs: list[bytes] | None = None, image_name: str | None = "python313.dll", *,
                 chunks: list[tuple[int, bytes]] | None = None, image_os: str = "windows") -> None:
        self.pid = pid
        self._chunks = chunks if chunks is not None else [(BASE + i * 0x10000, b) for i, b in enumerate(blobs or [])]
        self.image_os = image_os
        self._image_name = image_name
        self.images: list[SimpleNamespace] = []
        self.exported = None

    def image(self, pred):
        if self._image_name and pred(self._image_name):
            return SimpleNamespace(name=self._image_name, base=0x1000)
        return None

    def elf_symbol(self, image, symbol):
        return None if self.exported is None else 0x5000

    def q(self, va):
        return self.exported

    def chunks(self):
        yield from self._chunks


def _linux_proc(pid: int, banner: bytes, lib: str = "libpython3.12.so.1.0", exported: int | None = None):
    proc = _FakeProc(pid, [banner], image_name=None, image_os="linux")
    proc.images = [SimpleNamespace(name=lib, kind="elf", base=0x1000)]
    proc.exported = exported
    return proc


def _contiguous(*parts: bytes) -> list[tuple[int, bytes]]:
    out, va = [], BASE
    for part in parts:
        out.append((va, part))
        va += len(part)
    return out


def test_detect_captured_environment_extracts_concrete_stack_and_ignores_templates():
    banner = (
        b"ComfyUI version: {}\n"
        b'ComfyUI version: {}".format(comfyui_version.__version__)\n'
        b"ComfyUI version: 0.44.0\n"
        b"pytorch version: 2.12.1+cu130\n"
        b"Device: CPU is not a supported choice, passing through unchanged.\n"
        b"Device: cuda:0 NVIDIA GeForce RTX 3070 Laptop GPU : cudaMallocAsync\n"
        b"Total VRAM 8192 MB, total RAM 8191 MB\n"
        b"Set vram state to: NORMAL_VRAM\n"
    )
    captured, records = detect_captured_environment([_FakeProc(1420, [banner])])
    assert len(records) == 1
    assert captured["pid"] == 1420
    assert captured["python_version"] == "3.13"
    assert captured["comfyui_version"] == "0.44.0"
    assert captured["pytorch_version"] == "2.12.1+cu130"
    assert captured["gpu_hardware"] == "cuda:0 NVIDIA GeForce RTX 3070 Laptop GPU : cudaMallocAsync"
    assert captured["vram_summary"] == "Total VRAM 8192 MB, total RAM 8191 MB"
    assert captured["vram_state"] == "NORMAL_VRAM"
    assert captured["python_version_attribution"] == "COMFYUI_PROCESS"


@pytest.mark.parametrize("line, expected", [
    (b"Device: cpu\n", "cpu"),
    (b"Device: mps\n", "mps"),
    (b"Device: cuda:0 AMD Radeon RX 7900 XTX : native\n", "cuda:0 AMD Radeon RX 7900 XTX : native"),
    (b"Device: xpu:0 Intel(R) Arc(TM) A770 Graphics\n", "xpu:0 Intel(R) Arc(TM) A770 Graphics"),
    (b"Device: privateuseone:0\n", "privateuseone:0"),
    (b"Device: npu:0 Ascend910B\n", "npu:0 Ascend910B"),
    (b"Device: mlu:1 MLU370\n", "mlu:1 MLU370"),
])
def test_non_nvidia_devices_are_recovered(line, expected):
    env = detect_process_environment(_FakeProc(1, [b"pad " + line + b"pad"]))
    assert env["gpu_hardware"] == expected


def test_advisory_lines_are_recorded_without_deciding_the_stack():
    banner = (b"Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025) [MSC v.1943 64 bit (AMD64)]\n"
              b"Using pytorch attention\n")
    env = detect_process_environment(_FakeProc(1, [banner]))
    assert env["python_banner_version"] == "3.12.10"
    assert env["attention_backend"] == "Using pytorch attention"
    assert env["python_version"] == "3.13" and env["python_version_source"] == "RUNTIME_IMAGE"


def test_banner_split_across_contiguous_chunks_is_found_once():
    head = b"x" * 100 + b"ComfyUI version: 0.44.0-rc"
    chunks = _contiguous(head, b"1\npytorch version: 2.12.1+cu130\n")
    env = detect_process_environment(_FakeProc(1, chunks=chunks))
    assert env["comfyui_version"] == "0.44.0-rc1"
    assert env["pytorch_version"] == "2.12.1+cu130"
    assert env["candidates"]["comfyui_version"] == {"0.44.0-rc1": 1}

    whole = _contiguous(b"y" * 50 + b"Device: cuda:0 RTX 3070 : native\nmore", b" and more\n")
    env = detect_process_environment(_FakeProc(1, chunks=whole))
    assert env["candidates"]["gpu_hardware"] == {"cuda:0 RTX 3070 : native": 1}, "counted once, not per buffer"


def test_value_cut_by_a_page_gap_is_rejected_not_truncated():
    chunks = [(BASE, b"junk ComfyUI version: 0.44.0-rc"), (BASE + 0x5000, b"1\nDevice: cuda:0 NVIDIA GeForce RTX 30")]
    env = detect_process_environment(_FakeProc(1, chunks=chunks))
    assert env["comfyui_version"] is None and env["gpu_hardware"] is None
    # The same text at the very end of the last chunk, with nothing after it, is just as unproven.
    env = detect_process_environment(_FakeProc(1, chunks=[(BASE, b"ComfyUI version: 0.44.0")]))
    assert env["comfyui_version"] is None


def test_overlong_or_unterminated_tokens_are_rejected():
    text = b"ComfyUI version: 0.44.0" + b"a" * 30 + b"\npytorch version: 2.12.1+cu130\xff\xfe\n"
    env = detect_process_environment(_FakeProc(1, [text]))
    assert env["comfyui_version"] is None
    assert env["pytorch_version"] is None
    ok = detect_process_environment(_FakeProc(1, [b'"ComfyUI version: 0.44.0\x00pytorch version: 2.11.0\\n']))
    assert ok["comfyui_version"] == "0.44.0" and ok["pytorch_version"] == "2.11.0"


def test_chunks_without_banner_tokens_skip_the_regexes(monkeypatch):
    class _Boom:
        def finditer(self, _data):
            raise AssertionError("regex ran on a chunk with none of the cheap pre-check tokens")

    for name in ("_COMFYUI_RE", "_PYTORCH_RE", "_DEVICE_RE", "_VRAM_RE", "_VRAM_STATE_RE", "_ATTENTION_RE",
                 "_PY_BANNER_RE"):
        monkeypatch.setattr(envid, name, _Boom())
    env = detect_process_environment(_FakeProc(1, [b"\x00" * 4096 + b"plain heap bytes"]))
    assert env["comfyui_version"] is None


def test_only_private_vads_are_scanned_on_a_real_process_memory():
    private = b"ComfyUI version: 0.44.0\n" + b"\0" * 4072
    mapped = b"ComfyUI version: 0.99.9\n" + b"\0" * 4072
    assert len(private) == len(mapped) == 4096
    a, b = 0x10000000, 0x20000000
    mem = ProcessMemory(7, [(a, 4096, 0), (b, 4096, 4096)], private + mapped,
                        vads=[(a, a + 4095, None), (b, b + 4095, r"\models\notes.txt")])
    env = detect_process_environment(mem)
    assert env["comfyui_version"] == "0.44.0"
    assert env["candidates"]["comfyui_version"] == {"0.44.0": 1}


def test_linux_libpython_runtime_and_untrusted_mismatch():
    ok = detect_process_environment(_linux_proc(5, BANNER))
    assert ok["python_version"] == "3.12" and ok["python_image"] == "libpython3.12.so.1.0"
    assert ok["python_runtime_status"] is None

    # Py_Version says 3.11 while the file name says 3.12: neither is trusted.
    mismatch = detect_process_environment(_linux_proc(6, BANNER, exported=0x030B00F0))
    assert mismatch["python_runtime_status"].startswith("PYTHON_VERSION_MISMATCH")
    assert mismatch["python_version"] is None

    banner = BANNER + b"Python version: 3.12.10 (tags/v3.12.10) [GCC 13.2.0]\n"
    fallback = detect_process_environment(_linux_proc(6, banner, exported=0x030B00F0))
    assert fallback["python_version"] == "3.12" and fallback["python_version_source"] == "STARTUP_BANNER"


def test_comfyui_worker_is_preferred_and_disagreement_is_recorded_as_a_conflict():
    unrelated = _FakeProc(10, [b"ComfyUI version: 0.40.0\npytorch version: 2.10.0+cu126\n"], "python311.dll")
    worker = _FakeProc(20, [BANNER])
    root = _FakeProc(30, [b"ComfyUI version: 0.44.0\n"])
    context = [{"pid": 20, "role": "WORKER_CHILD", "command_line": "python main.py"},
               {"pid": 30, "role": "COMFYUI_ROOT", "command_line": "python -m venv"}]
    best, records = detect_captured_environment([unrelated, root, worker], context)
    assert best["pid"] == 20 and len(records) == 3
    conflicts = {c["field"]: c for c in best["conflicts"]}
    assert set(conflicts) == {"comfyui_version", "pytorch_version", "python_version"}
    assert {tuple(sorted(v["pids"])) for v in conflicts["comfyui_version"]["values"]} == {(10,), (20, 30)}

    # A conflict never silently becomes an exact match.
    runner = {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130", "python_version": "3.13.1",
              "gpu_hardware": GPU}
    assert compare_environments(best, runner)["verdict"] == PARTIAL_STACK_MATCH
    table = format_comparison_table(environment_document([unrelated, root, worker], None, runner, "u", context))
    assert "Conflict: comfyui_version" in table


def test_python_version_from_a_process_without_comfyui_fields_is_unattributed():
    best, _ = detect_captured_environment([_FakeProc(10, [b"no banner here"], "python311.dll")])
    assert best["python_version"] == "3.11" and best["python_version_attribution"] == "UNATTRIBUTED"
    comparison = compare_environments(best, {"comfyui_version": "UNREPORTED", "pytorch_version": "UNREPORTED",
                                            "python_version": "3.11.9", "gpu_hardware": "UNREPORTED"})
    assert comparison["fields"]["python_version"]["status"] == "UNATTRIBUTED"
    assert comparison["verdict"] == PARTIAL_STACK_MATCH and comparison["mismatches"] == []


def test_compare_environments_distinguishes_exact_match_mismatch_and_captured_only():
    captured = {
        "pid": 1420,
        "comfyui_version": "0.44.0",
        "pytorch_version": "2.12.1+cu130",
        "python_version": "3.13",
        "gpu_hardware": GPU,
    }
    matched_runner = {
        "comfyui_version": "0.44.0",
        "pytorch_version": "2.12.1+cu130",
        "python_version": "3.13.18",
        "gpu_hardware": GPU,
    }
    mismatched_runner = dict(matched_runner, comfyui_version="0.46.2", pytorch_version="2.11.0+cu128",
                             python_version="3.12.10")

    exact = compare_environments(captured, matched_runner)
    assert exact["verdict"] == EXACT_STACK_MATCH
    assert exact["mismatches"] == []

    diff = compare_environments(captured, mismatched_runner)
    assert diff["verdict"] == STACK_MISMATCH
    assert set(diff["mismatches"]) == {"comfyui_version", "pytorch_version", "python_version"}
    assert diff["fields"]["gpu_hardware"]["status"] == "MATCH"

    only = compare_environments(captured, None)
    assert only["verdict"] == CAPTURED_ONLY
    assert {f["status"] for f in only["fields"].values()} == {"CAPTURED_ONLY"}

    sparse = compare_environments({"comfyui_version": "0.44.0"}, None)
    assert sparse["fields"]["pytorch_version"]["status"] == "UNREPORTED"
    assert sparse["fields"]["comfyui_version"]["status"] == "CAPTURED_ONLY"

    empty = compare_environments({}, matched_runner)
    assert empty["verdict"] == UNAVAILABLE


def test_partial_and_runner_unreported_verdicts():
    captured = {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130", "python_version": "3.13",
                "gpu_hardware": GPU}
    partial = compare_environments(captured, {"comfyui_version": "0.44.0", "pytorch_version": "UNREPORTED",
                                              "python_version": "3.13.1", "gpu_hardware": "UNREPORTED"})
    assert partial["verdict"] == PARTIAL_STACK_MATCH and partial["mismatches"] == []
    wrong = compare_environments(captured, {"comfyui_version": "0.46.0", "pytorch_version": "UNREPORTED"})
    assert wrong["verdict"] == STACK_MISMATCH, "a proven mismatch outranks unreported fields"
    silent = compare_environments(captured, dict.fromkeys(captured, "UNREPORTED"))
    assert silent["verdict"] == RUNNER_UNREPORTED


def test_gpu_index_and_allocator_are_advisory_and_never_mismatch():
    captured = {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130", "python_version": "3.13",
                "gpu_hardware": "cuda:0 RTX 3070 : cudaMallocAsync"}
    runner = dict(captured, gpu_hardware="cuda:1 RTX 3070 : native")
    comparison = compare_environments(captured, runner)
    assert comparison["verdict"] == EXACT_STACK_MATCH
    advisory = comparison["fields"]["gpu_hardware"]["advisory"]
    assert advisory == {"captured_device_index": 0, "runner_device_index": 1,
                        "captured_allocator": "cudaMallocAsync", "runner_allocator": "native"}
    table = format_comparison_table(environment_document([_FakeProc(1, [BANNER])], None, runner, "u"))
    assert "ignored in the verdict" in table

    for other in ("cuda:0 RTX 4090 : cudaMallocAsync", "cpu", "mps"):
        assert compare_environments(captured, dict(runner, gpu_hardware=other))["verdict"] == STACK_MISMATCH
    # A runner that reports only the marketing name still matches the same GPU.
    assert compare_environments(captured, dict(runner, gpu_hardware="RTX 3070"))["verdict"] == EXACT_STACK_MATCH


def test_parse_gpu_handles_cpu_mps_rocm_and_unreported():
    assert parse_gpu("cpu") == {"device": "cpu", "index": None, "name": "", "allocator": None}
    assert parse_gpu("mps")["device"] == "mps"
    rocm = parse_gpu("cuda:0 AMD Radeon RX 7900 XTX : native")
    assert rocm == {"device": "cuda", "index": 0, "name": "amd radeon rx 7900 xtx", "allocator": "native"}
    assert parse_gpu("CUDA:2   NVIDIA  A100")["index"] == 2
    assert parse_gpu("UNREPORTED") is None and parse_gpu(None) is None
    assert envid.compare_environments({"gpu_hardware": "cpu"}, {"gpu_hardware": "CPU"})["verdict"] \
        == PARTIAL_STACK_MATCH


def test_environment_document_carries_method_and_memory_sha():
    doc = environment_document([_FakeProc(1420, [BANNER])], "a" * 64, None, None)
    assert doc["memory_sha256"] == "a" * 64 and "dump_sha256" not in doc
    assert doc["method"]["counted_holder"] == envid.COUNTED_HOLDER
    assert doc["method"]["comfytrace_version"]
    assert doc["method"]["candidates"]["1420"]["comfyui_version"] == {"0.44.0": 1}
    assert "candidates" not in doc["captured_environment"] and "candidates" not in doc["processes"][0]
    assert doc["conflicts"] == []
    json.dumps(doc)


def test_write_environment_identity_is_write_once(tmp_path):
    doc = {
        "schema_version": 1,
        "memory_sha256": "a" * 64,
        "verdict": STACK_MISMATCH,
        "captured_environment": {"pid": 1420, "comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130",
                                 "python_version": "3.13", "gpu_hardware": "cuda:0 RTX 3070",
                                 "vram_summary": "Total VRAM 8192 MB, total RAM 8191 MB", "vram_state": "NORMAL_VRAM"},
        "runner_url": "http://127.0.0.1:8188",
        "runner_environment": {"comfyui_version": "0.46.2", "pytorch_version": "2.11.0+cu128",
                               "python_version": "3.12.10", "gpu_hardware": "cuda:0 RTX 3070"},
        "comparison": compare_environments(
            {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130",
             "python_version": "3.13", "gpu_hardware": "cuda:0 RTX 3070"},
            {"comfyui_version": "0.46.2", "pytorch_version": "2.11.0+cu128",
             "python_version": "3.12.10", "gpu_hardware": "cuda:0 RTX 3070"},
        ),
        "processes": [],
    }
    path = write_environment_identity(tmp_path, doc)
    assert path == tmp_path / ENV_IDENTITY_NAME
    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == STACK_MISMATCH
    table = format_comparison_table(doc)
    assert "STACK_MISMATCH" in table
    assert "captured=2.12.1+cu130 | runner=2.11.0+cu128 [MISMATCH]" in table

    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_environment_identity(tmp_path, dict(doc, verdict="SOMETHING_ELSE"))
    assert path.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == [ENV_IDENTITY_NAME], "no temporary file is left behind"


# --- standalone commands -------------------------------------------------------------------------------------------


class _Ingestor:
    """Has the real ``ingest_memory`` signature, so a call with the wrong arguments fails the test."""

    def __init__(self, processes=(), context=(), reason=None) -> None:
        self.calls: list[tuple[Path, object, tuple[int, ...]]] = []
        self.closed = 0
        self._result = SimpleNamespace(recovery=SimpleNamespace(python_processes=list(processes)),
                                       process_context=list(context), degradation_reason=reason,
                                       close=self._close)

    def _close(self) -> None:
        self.closed += 1

    def __call__(self, memory_path, symbol_manager, *, examiner_pids=()):
        self.calls.append((memory_path, symbol_manager, tuple(examiner_pids)))
        return self._result


def _args(tmp_path, command="env-id", *extra):
    memory = tmp_path / "dump.raw"
    memory.write_bytes(b"image")
    extra_args = list(extra)
    if command == "model-id" and "--model" not in extra_args:
        extra_args += ["--model", str(tmp_path)]
    return parse_args([command, "--memory", str(memory), "--symbols-dir", str(tmp_path), *extra_args])


def _runner(monkeypatch, environment):
    from comfytrace.regenerate import RunnerClient

    calls = []

    def fake(self):
        calls.append(self.runner_url)
        return dict(environment)

    monkeypatch.setattr(RunnerClient, "environment", fake)
    return calls


def test_cli_env_id_argument_parsing_and_missing_manifest(tmp_path, monkeypatch):
    calls = _runner(monkeypatch, {})
    args = parse_args(["env-id", "--case-dir", str(tmp_path), "--memory", str(tmp_path / "m.raw"),
                       "--runner-url", "http://127.0.0.1:8188"])
    assert args.command == "env-id"
    assert args.runner_url == "http://127.0.0.1:8188"
    assert run_env_id(args) == 1
    assert calls == [], "a bad case fails before the runner is contacted"

    standalone_env = parse_args(["env-id", "--memory", str(tmp_path / "m.raw"), "--symbols-dir", str(tmp_path)])
    assert standalone_env.case_dir is None
    standalone_mid = parse_args(["model-id", "--memory", str(tmp_path / "m.raw"),
                                 "--symbols-dir", str(tmp_path), "--model", str(tmp_path)])
    assert standalone_mid.case_dir is None


def test_standalone_env_id_ingests_like_extract_and_compares(tmp_path, monkeypatch, capsys):
    calls = _runner(monkeypatch, {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130",
                                  "python_version": "3.13.1", "gpu_hardware": "cuda:1 NVIDIA GeForce RTX 3070 Laptop "
                                  "GPU : native", "cuda_driver_version": "UNREPORTED"})
    ingestor = _Ingestor([_FakeProc(20, [BANNER])], [{"pid": 20, "role": "WORKER_CHILD", "command_line": "main.py"}])
    args = _args(tmp_path, "env-id", "--runner-url", "http://runner", "--comfyui-pid", "20", "--comfyui-pid", "21")
    assert run_env_id(args, ingestor) == EXIT_OK
    ((memory, manager, pids),) = ingestor.calls
    assert memory == args.memory and pids == (20, 21)
    assert (manager.symbols_dir, manager.image_os, manager.image_format, manager.allow_download) \
        == (tmp_path, "windows", "physical", False)
    assert ingestor.closed == 1 and calls == ["http://runner"]
    assert "Verdict: EXACT_STACK_MATCH" in capsys.readouterr().out
    assert not (tmp_path / ENV_IDENTITY_NAME).exists()


def test_standalone_commands_validate_pids_and_inputs_before_ingesting(tmp_path):
    for command in ("env-id", "model-id"):
        for bad in ("0", "-3"):
            ingestor = _Ingestor([_FakeProc(1, [BANNER])])
            runner = run_env_id if command == "env-id" else run_model_id
            assert runner(_args(tmp_path, command, "--comfyui-pid", bad), ingestor) == EXIT_FAILURE
            assert ingestor.calls == []
    missing = parse_args(["env-id", "--memory", str(tmp_path / "none.raw"), "--symbols-dir", str(tmp_path)])
    ingestor = _Ingestor()
    assert run_env_id(missing, ingestor) == EXIT_FAILURE and ingestor.calls == []
    no_symbols = parse_args(["env-id", "--memory", str(tmp_path / "m.raw")])
    (tmp_path / "m.raw").write_bytes(b"x")
    assert run_env_id(no_symbols, ingestor) == EXIT_FAILURE and ingestor.calls == []


def test_standalone_elf_core_needs_no_symbols_dir(tmp_path):
    memory = tmp_path / "core.raw"
    memory.write_bytes(b"core")
    args = parse_args(["env-id", "--memory", str(memory), "--image-os", "linux", "--image-format", "elf-core",
                       "--comfyui-pid", "9"])
    ingestor = _Ingestor([_linux_proc(9, BANNER)])
    assert run_env_id(args, ingestor) == EXIT_OK
    ((_, manager, pids),) = ingestor.calls
    assert manager.symbols_dir is None and manager.image_format == "elf-core" and pids == (9,)
    downloading = parse_args(["env-id", "--memory", str(memory), "--image-os", "linux", "--image-format", "elf-core",
                              "--allow-symbol-download"])
    assert run_env_id(downloading, ingestor) == EXIT_FAILURE


def test_unreachable_runner_is_a_failure_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _runner(monkeypatch, dict.fromkeys(("comfyui_version", "python_version", "pytorch_version",
                                        "cuda_driver_version", "gpu_hardware"), "UNREPORTED"))
    case = tmp_path / "case"
    case.mkdir()
    ingestor = _Ingestor([_FakeProc(20, [BANNER])])
    args = _args(tmp_path, "env-id", "--runner-url", "http://127.0.0.1:1", "--case-dir", str(case))
    memory_sha = hashlib.sha256(b"image").hexdigest()
    (case / "manifest.json").write_text(json.dumps({"case_metadata": {"memory_image": {"sha256": memory_sha}}}))
    monkeypatch.setattr("comfytrace.ingest.ingest_case", lambda *a, **k: ingestor(a[0], None))
    assert run_env_id(args, ingestor) == EXIT_FAILURE
    assert "RUNNER_UNREPORTED" in capsys.readouterr().out
    assert not (case / ENV_IDENTITY_NAME).exists()
    assert ingestor.closed == 1


def test_case_dir_env_id_checks_the_image_then_writes_once(tmp_path, monkeypatch):
    _runner(monkeypatch, {"comfyui_version": "0.44.0", "pytorch_version": "2.12.1+cu130", "python_version": "3.13.1",
                          "gpu_hardware": GPU})
    case = tmp_path / "case"
    case.mkdir()
    args = _args(tmp_path, "env-id", "--runner-url", "http://runner", "--case-dir", str(case))
    sha = hashlib.sha256(b"image").hexdigest()
    (case / "manifest.json").write_text(json.dumps({"case_metadata": {"memory_image": {"sha256": sha}}}))
    ingestor = _Ingestor([_FakeProc(20, [BANNER])])
    monkeypatch.setattr("comfytrace.ingest.ingest_case", lambda *a, **k: ingestor(a[0], None))
    assert run_env_id(args, ingestor) == EXIT_OK
    document = json.loads((case / ENV_IDENTITY_NAME).read_text())
    assert document["verdict"] == EXACT_STACK_MATCH and document["memory_sha256"] == sha
    assert run_env_id(args, ingestor) == EXIT_FAILURE, "environment identity is written once"
    (case / ENV_IDENTITY_NAME).unlink()
    args.memory.write_bytes(b"another image")
    assert run_env_id(args, ingestor) == EXIT_FAILURE and not (case / ENV_IDENTITY_NAME).exists()


def test_no_process_memory_is_degraded_not_observed(tmp_path, monkeypatch, capsys, caplog):
    calls = _runner(monkeypatch, {})
    ingestor = _Ingestor([], reason="no ComfyUI process found")
    with caplog.at_level("ERROR", logger="comfytrace"):
        assert run_env_id(_args(tmp_path, "env-id", "--runner-url", "http://runner"), ingestor) == EXIT_DEGRADED
    assert "no ComfyUI process found" in caplog.text and calls == [], "no runner call without memory to compare"
    assert "NO_PROCESS_MEMORY" in capsys.readouterr().out and ingestor.closed == 1

    models = tmp_path / "models"
    models.mkdir()
    (models / "a.safetensors").write_bytes(b"x")
    ingestor = _Ingestor([], reason="no ComfyUI process found")
    assert run_model_id(_args(tmp_path, "model-id", "--model", str(models)), ingestor) == EXIT_DEGRADED
    assert "a.safetensors: NO_PROCESS_MEMORY" in capsys.readouterr().out
    assert ingestor.closed == 1


def test_local_model_names_edge_cases(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        modelid.local_model_names(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("not a model")
    with pytest.raises(ValueError, match="no model files"):
        modelid.local_model_names(empty)

    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    for name in ("a.safetensors", "b.CKPT", "sub/c.pt", "sub/d.pth", "e.gguf", "f.sft", "notes.txt"):
        (models / name).write_bytes(b"x")
    root, names = modelid.local_model_names(models)
    assert root == models
    assert names == ["a.safetensors", "b.CKPT", "e.gguf", "f.sft", "sub/c.pt", "sub/d.pth"]
    assert modelid.local_model_names(models / "a.safetensors") == (models, ["a.safetensors"])


def test_non_safetensors_models_are_reported_unsupported_not_skipped(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("old.ckpt", "q.gguf"):
        (models / name).write_bytes(b"x")
    root, names = modelid.local_model_names(models)
    document = modelid.identity_document(None, [object()], root, "0" * 64, names)  # type: ignore[list-item]
    assert {m["name"]: m["verdict"] for m in document["models"]} == {
        "old.ckpt": "UNSUPPORTED_FORMAT", "q.gguf": "UNSUPPORTED_FORMAT"}
    assert all("only safetensors" in m["reason"] for m in document["models"])
    absent = modelid.identity_document(None, [object()], root, None, ["gone.ckpt"])  # type: ignore[list-item]
    assert absent["models"][0]["verdict"] == "NO_LOCAL_COPY"
    degraded = modelid.identity_document(None, [], root, None, ["old.ckpt"])
    assert degraded["models"] == [{"name": "old.ckpt", "verdict": modelid.NO_PROCESS_MEMORY}]


def test_standalone_model_id_reports_candidates_and_fails_on_an_empty_directory(tmp_path, capsys, caplog):
    models = tmp_path / "models"
    models.mkdir()
    (models / "old.ckpt").write_bytes(b"x")
    ingestor = _Ingestor([_FakeProc(1, [BANNER])])
    with caplog.at_level("INFO", logger="comfytrace"):
        assert run_model_id(_args(tmp_path, "model-id", "--model", str(models)), ingestor) == EXIT_OK
    assert "old.ckpt: UNSUPPORTED_FORMAT" in capsys.readouterr().out
    assert "checking 1 candidate model file(s)" in caplog.text
    ((_, manager, pids),) = ingestor.calls
    assert manager.symbols_dir == tmp_path and pids == ()

    empty = tmp_path / "empty"
    empty.mkdir()
    ingestor = _Ingestor([_FakeProc(1, [BANNER])])
    assert run_model_id(_args(tmp_path, "model-id", "--model", str(empty)), ingestor) == EXIT_FAILURE
    assert ingestor.calls == [], "no memory is ingested for a directory with nothing to check"


def test_namespace_without_optional_flags_still_works(tmp_path):
    memory = tmp_path / "dump.raw"
    memory.write_bytes(b"image")
    args = argparse.Namespace(memory=memory, symbols_dir=tmp_path, case_dir=None, runner_url=None)
    ingestor = _Ingestor([_FakeProc(1, [BANNER])])
    assert run_env_id(args, ingestor) == EXIT_OK
    assert ingestor.calls[0][2] == ()
