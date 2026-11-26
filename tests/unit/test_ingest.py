import socket
from types import SimpleNamespace

import pytest
import volatility3.symbols
from volatility3.framework import constants, exceptions

from comfytrace.carve import StreamReader
from comfytrace.exceptions import ProcessNotFoundError, SymbolResolutionError
from comfytrace.ingest import ROLE_ROOT, ROLE_WORKER, ProcessWalker, SymbolManager, VolatilityEngine
from tests.synthetic import PAGE, FakeProcess, FakeProcLayer, FakeVad


@pytest.fixture
def volatility_globals(monkeypatch, tmp_path):
    # A private identifier cache: the user's is shared with any extraction running at the same time.
    monkeypatch.setattr(constants, "CACHE_PATH", str(tmp_path / "volatility-cache"))
    (tmp_path / "volatility-cache").mkdir()
    monkeypatch.setattr(constants, "OFFLINE", False)
    monkeypatch.setattr(constants, "REMOTE_ISF_URL", "https://example.invalid/isf")
    monkeypatch.setattr(volatility3.symbols, "__path__", list(volatility3.symbols.__path__))


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_symbol_manager_offline_cache(tmp_path, volatility_globals, no_network):
    isf = tmp_path / "windows" / "ntkrnlmp.pdb" / "3844DBB920174967BE7AA4A2C20430FA-2.json.xz"
    isf.parent.mkdir(parents=True)
    isf.write_bytes(b"\xfd7zXZ")
    manager = SymbolManager(tmp_path, allow_download=False)
    assert manager.isf_files() == [isf]
    manager.apply_network_policy()
    assert constants.OFFLINE is True
    assert constants.REMOTE_ISF_URL is None
    assert str(tmp_path.resolve()) in volatility3.symbols.__path__


def test_symbol_manager_rejects_silent_download(tmp_path, volatility_globals, no_network):
    empty = tmp_path / "symbols"
    empty.mkdir()
    with pytest.raises(SymbolResolutionError):
        SymbolManager(empty, allow_download=False).resolve_symbol_table(context=None)
    with pytest.raises(SymbolResolutionError):
        SymbolManager(None, allow_download=False).resolve_symbol_table(context=None)


def test_an_offline_download_attempt_cannot_remove_a_cached_isf(tmp_path, volatility_globals, no_network):
    from volatility3.framework.symbols.windows.pdbutil import PDBUtility

    isf = tmp_path / "symbols" / "windows" / "ntkrnlmp.pdb" / "B661C6FE72991D160FE37C48B8BC5470-1.json.xz"
    isf.parent.mkdir(parents=True)
    isf.write_bytes(b"\xfd7zXZ cached")
    SymbolManager(tmp_path / "symbols", allow_download=False).apply_network_policy()
    with pytest.raises(exceptions.OfflineException):
        PDBUtility.download_pdb_isf(None, "B661C6FE72991D160FE37C48B8BC5470", 1, "ntkrnlmp.pdb")
    assert isf.read_bytes() == b"\xfd7zXZ cached"


def test_real_volatility_automagic_failure_maps_to_symbol_error(tmp_path, volatility_globals, no_network):
    """Drives the real volatility3 2.28.2 automagic against a non-Windows image with an offline cache."""
    symbols = tmp_path / "symbols" / "windows"
    symbols.mkdir(parents=True)
    (symbols / "placeholder.json").write_text("{}")
    dump = tmp_path / "not-windows.raw"
    dump.write_bytes(b"\x00" * (2 * 1024 * 1024))
    engine = VolatilityEngine(dump, SymbolManager(tmp_path / "symbols"))
    with pytest.raises(SymbolResolutionError):
        engine.initialize_context()
    assert constants.OFFLINE is True


def _walker(processes):
    context = SimpleNamespace(modules={"kernel": SimpleNamespace(symbol_table_name="symtab")}, layers={})
    calls = []

    def get_cmdline(ctx, table, proc):
        calls.append(table)
        return proc.cmdline

    walker = ProcessWalker(
        context,
        "kernel",
        list_processes=lambda ctx, name: iter(processes),
        get_cmdline=get_cmdline,
        list_vads=lambda proc: [FakeVad(s, e) for s, e in proc.vads],
    )
    return walker, calls


def test_process_walker_cmdline_extraction():
    processes = [
        FakeProcess(4, 0, "System", None),
        FakeProcess(7000, 4, "explorer.exe", None),
        FakeProcess(19000, 7000, "cmd.exe", None),
        FakeProcess(19556, 19000, "python.exe",
                    r"C:\ComfyUI\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build",
                    vads=[(0x10000, 0x1FFFF)]),
        FakeProcess(13484, 19556, "python.exe", r"C:\ComfyUI\python_embeded\python.exe main.py --listen",
                    ActiveThreads=26, vads=[(0x20000, 0x2FFFF), (0x40000, 0x4FFFF)]),
        FakeProcess(14000, 13484, "conhost.exe", "conhost.exe 0xffffffff"),
        FakeProcess(15000, 7000, "python.exe", r"python.exe C:\tools\other_script.py"),
    ]
    walker, calls = _walker(processes)
    hierarchy = {p["pid"]: p for p in walker.enumerate_comfyui_hierarchy()}
    assert set(hierarchy) == {19556, 13484, 14000}
    assert hierarchy[19556]["role"] == ROLE_ROOT
    assert hierarchy[13484]["role"] == ROLE_WORKER
    assert hierarchy[13484]["ppid"] == 19556
    assert hierarchy[13484]["threads"] == 26
    assert hierarchy[13484]["vad_count"] == 2
    assert hierarchy[13484]["command_line"].endswith("main.py --listen")
    assert hierarchy[14000]["role"] == ROLE_WORKER
    assert set(calls) == {"symtab"}
    assert walker.get_process_vads(13484) == [(0x20000, 0x2FFFF), (0x40000, 0x4FFFF)]
    with pytest.raises(ProcessNotFoundError):
        walker.get_process_vads(99999)


def test_examiner_pid_selects_python_with_unreadable_command_line():
    walker, _ = _walker([
        FakeProcess(4, 0, "System", None),
        FakeProcess(7000, 4, "python.exe", None),
        FakeProcess(7004, 7000, "python.exe", None),
        FakeProcess(9000, 4, "python.exe", r"python.exe C:\tools\different.py"),
    ])
    assert walker.enumerate_comfyui_hierarchy() == []
    assert walker.unreadable_python_command_lines is True

    results = walker.enumerate_comfyui_hierarchy(examiner_pids=(7000,))
    assert [(p["pid"], p["discovery_basis"]) for p in results] == [
        (7000, "EXAMINER_PID"), (7004, "DESCENDANT"),
    ]
    assert all(p["command_line"] is None for p in results)
    for pid in (4, 9000, 9004):
        with pytest.raises(ProcessNotFoundError, match=str(pid)):
            walker.enumerate_comfyui_hierarchy(examiner_pids=(pid,))


def test_vad_virtual_mapping_pad_false():
    physical = bytes(range(256)) * (16 * PAGE // 256)
    layer = FakeProcLayer(physical, {})
    base = 0x7FF000000000
    layer.map_virtual(base, [5, 2])
    layer.map_virtual(base + 3 * PAGE, [9])
    layer.swapped.add(base // PAGE + 4)
    layer.page_table[base // PAGE + 4] = 1
    blocks = list(StreamReader("unused").stream_process_vad(layer, base, base + 6 * PAGE - 1))

    assert all(pad is False for _, _, pad in layer.reads)
    data = b"".join(chunk for _, chunk, _ in blocks)
    assert data == physical[5 * PAGE:6 * PAGE] + physical[2 * PAGE:3 * PAGE] + physical[9 * PAGE:10 * PAGE]
    gaps = [g for _, _, gs in blocks for g in gs]
    assert [(g.relative_offset, g.length_bytes, str(g.reason)) for g in gaps] == [
        (2 * PAGE, PAGE, "PAGE_NOT_PRESENT"),
        (4 * PAGE, 2 * PAGE, "PAGE_NOT_PRESENT"),
    ]
    assert all(g.synthetic_padding_applied is False for g in gaps)


def test_read_fault_inside_mapping_becomes_paged_out_gap():
    physical = bytes(8 * PAGE)
    layer = FakeProcLayer(physical, {})
    base = 0x10000
    layer.map_virtual(base, [0, 1, 2])
    original_read = layer.read

    def read(offset, length, pad=False):
        if offset <= base + PAGE < offset + length:
            layer.swapped.add((base + PAGE) // PAGE)
        return original_read(offset, length, pad)

    layer.read = read
    blocks = list(StreamReader("unused").stream_process_vad(layer, base, base + 3 * PAGE - 1))
    gaps = [g for _, _, gs in blocks for g in gs]
    assert [(g.relative_offset, g.length_bytes, str(g.reason)) for g in gaps] == [(PAGE, PAGE, "PAGED_OUT")]
    assert sum(len(chunk) for _, chunk, _ in blocks) == 2 * PAGE


def test_translate_virtual_to_physical():
    layer = FakeProcLayer(bytes(4 * PAGE), {})
    layer.map_virtual(0x5000, [3])
    engine = VolatilityEngine.__new__(VolatilityEngine)
    assert engine.translate_virtual_to_physical(layer, 0x5010) == 3 * PAGE + 0x10
    assert engine.translate_virtual_to_physical(layer, 0x9000) is None


class _VolatilityString(str):
    """Mimics volatility3 objects.String: a str subclass that cannot be deep-copied."""

    def __deepcopy__(self, memo):
        raise AttributeError("Unable to find _PrimitiveObject__new_value")


def test_volatility_string_cmdline_serializes_into_manifest():
    from comfytrace.models import ForensicManifest

    cmd = _VolatilityString(r"C:\ComfyUI\.venv\Scripts\python.exe -s ComfyUI\main.py")
    walker, _ = _walker([FakeProcess(1420, 4, "python.exe", cmd, vads=[(0x10000, 0x1FFFF)])])
    hierarchy = walker.enumerate_comfyui_hierarchy()
    manifest = ForensicManifest(schema_version="1.0.0", case_metadata={}, process_context=hierarchy, findings={})
    assert manifest.to_dict()["process_context"][0]["command_line"] == str(cmd)


def _ingest_with(monkeypatch, tmp_path, hierarchy=None, enumerate_error=None, layer_error=None):
    from comfytrace import ingest
    from comfytrace.exceptions import IngestionError

    dump = tmp_path / "mem.raw"
    dump.write_bytes(b"\x00" * (2 * PAGE))
    monkeypatch.setattr(ingest.VolatilityEngine, "initialize_context", lambda self: SimpleNamespace(modules={}))

    def enumerate_hierarchy(self):
        if enumerate_error:
            raise IngestionError(enumerate_error)
        return hierarchy or []

    def no_layer(self, pid):
        raise layer_error

    monkeypatch.setattr(ingest.ProcessWalker, "enumerate_comfyui_hierarchy", enumerate_hierarchy)
    if layer_error:
        monkeypatch.setattr(ingest.ProcessWalker, "get_process_layer", no_layer)
    result = ingest.ingest_memory(dump, SymbolManager(tmp_path))
    result.close()
    return result


def test_unreadable_process_list_falls_back_to_physical(monkeypatch, tmp_path):
    from comfytrace.ingest import MODE_PHYSICAL

    result = _ingest_with(monkeypatch, tmp_path, enumerate_error="process list traversal failed: smeared")
    assert (result.mode, result.degradation_reason) == (MODE_PHYSICAL, "PROCESS_LIST_UNREADABLE")
    assert [s.process_id for s in result.spaces] == [None]


def test_no_readable_process_layer_falls_back_to_physical(monkeypatch, tmp_path):
    from comfytrace.ingest import MODE_PHYSICAL

    hierarchy = [{"pid": 1420, "role": ROLE_ROOT}]
    result = _ingest_with(monkeypatch, tmp_path, hierarchy=hierarchy, layer_error=ProcessNotFoundError("gone"))
    assert (result.mode, result.degradation_reason) == (MODE_PHYSICAL, "PROCESS_LAYERS_UNREADABLE")
    assert [s.process_id for s in result.spaces] == [None]


def test_process_mode_adds_physical_pass_over_unowned_pages(monkeypatch, tmp_path):
    from comfytrace import ingest
    from comfytrace.ingest import MODE_VIRTUAL

    layer = FakeProcLayer(b"", {0x10: 7, 0x11: 3})
    monkeypatch.setattr(ingest, "is_raw_physical", lambda layer: True)
    monkeypatch.setattr(ingest.ProcessWalker, "get_process_layer", lambda self, pid: layer)
    monkeypatch.setattr(ingest.ProcessWalker, "get_process_vads", lambda self, pid: [(0x10000, 0x12FFF)])
    result = _ingest_with(monkeypatch, tmp_path, hierarchy=[{"pid": 1420, "role": ROLE_ROOT}])
    assert result.mode == MODE_VIRTUAL and result.degradation_reason is None
    assert [s.process_id for s in result.spaces] == [1420, None]
    assert result.spaces[-1].skip_pages == frozenset({7, 3})


def test_non_raw_image_skips_the_unowned_page_pass(monkeypatch, tmp_path):
    from comfytrace import ingest

    layer = FakeProcLayer(b"", {0x10: 7})
    monkeypatch.setattr(ingest.ProcessWalker, "get_process_layer", lambda self, pid: layer)
    monkeypatch.setattr(ingest.ProcessWalker, "get_process_vads", lambda self, pid: [(0x10000, 0x10FFF)])
    result = _ingest_with(monkeypatch, tmp_path, hierarchy=[{"pid": 1420, "role": ROLE_ROOT}])
    assert [s.process_id for s in result.spaces] == [1420]
