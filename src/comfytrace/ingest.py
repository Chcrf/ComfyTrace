"""Volatility 3 headless ingestion: symbol gating, kernel context, ComfyUI process discovery and VADs."""

from __future__ import annotations

import logging
import mmap
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfytrace.exceptions import IngestionError, ProcessNotFoundError, SymbolResolutionError

log = logging.getLogger(__name__)

PYTHON_IMAGE_NAMES = frozenset({"python.exe", "pythonw.exe"})
COMFYUI_CMDLINE_MARKERS = ("main.py", "python_embeded")
ISF_SUFFIXES = (".json", ".json.xz", ".json.gz", ".json.bz2")

ROLE_ROOT = "ROOT_LAUNCHER"
ROLE_WORKER = "WORKER_CHILD"


def is_comfyui_cmdline(cmdline: str | None) -> bool:
    if not cmdline:
        return False
    lowered = cmdline.replace("\\", "/").lower()
    return any(marker in lowered for marker in COMFYUI_CMDLINE_MARKERS)


class SymbolManager:
    """Gates Volatility symbol resolution to an offline ISF cache unless downloads are authorized."""

    def __init__(self, symbols_dir: Path | None, allow_download: bool = False, *, image_os: str = "windows",
                 image_format: str = "physical") -> None:
        if image_os not in {"windows", "linux"}:
            raise ValueError(f"Unsupported image OS: {image_os}")
        if image_format != "physical" and (image_os, image_format) != ("linux", "elf-core"):
            raise ValueError("The elf-core image format requires --image-os linux")
        self.symbols_dir = Path(symbols_dir) if symbols_dir is not None else None
        self.allow_download = allow_download
        self.image_os = image_os
        self.image_format = image_format
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    @property
    def mode(self) -> str:
        return "REMOTE_PERMITTED" if self.allow_download else "OFFLINE_CACHE"

    def isf_files(self) -> list[Path]:
        if self.symbols_dir is None or not self.symbols_dir.is_dir():
            return []
        return sorted(p for p in self.symbols_dir.rglob("*") if p.is_file() and p.name.endswith(ISF_SUFFIXES))

    def apply_network_policy(self) -> None:
        """Points Volatility at the local cache and disables every remote fetch unless authorized."""
        import volatility3.symbols
        from volatility3.framework import constants

        constants.OFFLINE = not self.allow_download
        paths = [str(self.symbols_dir.resolve())] if self.symbols_dir is not None else []
        if not self.allow_download:
            constants.REMOTE_ISF_URL = None
            # When automagic misses an ISF (a stale or concurrently rewritten identifier cache), Volatility's PDB
            # download opens its output in the first symbol path, truncating any cached ISF there, and deletes
            # it once the offline fetch fails. That first path is an empty scratch directory, never the cache.
            self._scratch = tempfile.TemporaryDirectory(prefix="comfytrace-symbols-")
            paths.insert(0, self._scratch.name)
        volatility3.symbols.__path__ = [*paths, *constants.SYMBOL_BASEPATHS]

    def resolve_symbol_table(self, context: Any) -> str:
        """Resolves Windows kernel symbol tables from offline ISF cache or authorized download.
        Returns the kernel module name registered in the context.
        Raises: SymbolResolutionError if symbols cannot be resolved offline without --allow-symbol-download."""
        if not self.allow_download and not self.isf_files():
            raise SymbolResolutionError(
                f"no ISF symbol files under {self.symbols_dir!s}; pass --symbols-dir with a populated "
                "cache or --allow-symbol-download"
            )
        self.apply_network_policy()

        import volatility3.plugins
        from volatility3 import framework
        from volatility3.framework import automagic, exceptions, plugins
        if self.image_os == "linux":
            from volatility3.plugins.linux.pslist import PsList
        else:
            from volatility3.plugins.windows.pslist import PsList

        framework.import_files(volatility3.plugins, True)
        chosen = automagic.choose_automagic(automagic.available(context), PsList)
        try:
            plugin = plugins.construct_plugin(context, chosen, PsList, "plugins", None, None)
        except (exceptions.UnsatisfiedException, exceptions.OfflineException) as exc:
            raise SymbolResolutionError(f"kernel symbols could not be resolved: {exc}") from exc
        kernel_module_name = plugin.config.get("kernel")
        if not kernel_module_name or kernel_module_name not in context.modules:
            raise SymbolResolutionError("kernel module was not constructed by Volatility automagic")
        return str(kernel_module_name)


class VolatilityEngine:
    def __init__(self, memory_path: Path, symbol_manager: SymbolManager) -> None:
        self.memory_path = Path(memory_path)
        self.symbol_manager = symbol_manager
        self.context: Any = None
        self.kernel_module_name: str | None = None

    def initialize_context(self) -> Any:
        """Initializes headless Volatility 3 Context and registers symbol paths.
        Raises: IngestionError on invalid memory image format or unreadable path;
        SymbolResolutionError when the kernel cannot be resolved."""
        if not self.memory_path.is_file():
            raise IngestionError(f"memory image not found or unreadable: {self.memory_path}")
        import volatility3
        from volatility3.framework import contexts
        from volatility3.framework.configuration import requirements

        volatility3.framework.require_interface_version(2, 0, 0)
        context = contexts.Context()
        context.config["automagic.LayerStacker.single_location"] = requirements.URIRequirement.location_from_file(
            str(self.memory_path.resolve())
        )
        self.kernel_module_name = self.symbol_manager.resolve_symbol_table(context)
        self.context = context
        return context

    def translate_virtual_to_physical(self, proc_layer: Any, virtual_address: int) -> int | None:
        """Translates a user-mode virtual address to the physical offset in the layer beneath the process layer.
        Returns None on translation fault or unmapped page."""
        try:
            for offset, _sublength, mapped_offset, _mapped_length, _layer in proc_layer.mapping(
                virtual_address, 1, ignore_errors=True
            ):
                if offset <= virtual_address:
                    return int(mapped_offset + (virtual_address - offset))
        except Exception:  # noqa: BLE001 - any translation failure means "unmapped" for this lookup
            return None
        return None


def _field(obj: Any, *path: str, default: Any = None) -> Any:
    try:
        for name in path:
            obj = getattr(obj, name)
        return obj
    except Exception:  # noqa: BLE001 - smeared kernel structures raise many exception types
        return default


def _image_name(proc: Any) -> str:
    raw = _field(proc, "ImageFileName")
    try:
        return str(raw.cast("string", max_length=raw.vol.count, errors="replace"))
    except Exception:  # noqa: BLE001
        return str(raw) if raw is not None else ""


class ProcessWalker:
    def __init__(
        self,
        context: Any,
        kernel_module_name: str = "kernel",
        list_processes: Callable[..., Iterable[Any]] | None = None,
        get_cmdline: Callable[..., str | None] | None = None,
        list_vads: Callable[[Any], Iterable[Any]] | None = None,
    ) -> None:
        self.context = context
        self.kernel_module_name = kernel_module_name
        if list_processes is None or get_cmdline is None or list_vads is None:
            from volatility3.plugins.windows import cmdline, pslist, vadinfo

            list_processes = list_processes or pslist.PsList.list_processes
            get_cmdline = get_cmdline or cmdline.CmdLine.get_cmdline
            list_vads = list_vads or vadinfo.VadInfo.list_vads
        self._list_processes = list_processes
        self._get_cmdline = get_cmdline
        self._list_vads = list_vads
        self.unreadable_python_command_lines = False
        self._inventory: dict[int, Any] | None = None
        self._regions: dict[int, list[tuple[int, int, str | None]]] = {}

    @staticmethod
    def _pid(proc: Any) -> int:
        return int(_field(proc, "UniqueProcessId", default=-1))

    @staticmethod
    def _parent(proc: Any) -> int:
        return int(_field(proc, "InheritedFromUniqueProcessId", default=-1))

    _name = staticmethod(_image_name)

    @staticmethod
    def _is_python(name: str) -> bool:
        return name.lower() in PYTHON_IMAGE_NAMES

    def _matches_command(self, proc: Any, cmdline: str | None) -> bool:
        return is_comfyui_cmdline(cmdline)

    @staticmethod
    def _process_fields(proc: Any) -> dict[str, Any]:
        dtb = _field(proc, "Pcb", "DirectoryTableBase")
        return {"threads": int(_field(proc, "ActiveThreads", default=0) or 0),
                "dtb": hex(int(dtb)) if dtb is not None else None}

    @property
    def _kernel_table(self) -> str:
        return str(self.context.modules[self.kernel_module_name].symbol_table_name)

    def _processes(self) -> list[Any]:
        try:
            return list(self._list_processes(self.context, self.kernel_module_name))
        except Exception as exc:
            raise IngestionError(f"process list traversal failed: {exc}") from exc

    def _proc_by_pid(self, pid: int) -> Any:
        if self._inventory is None:
            self._inventory = {self._pid(proc): proc for proc in self._processes()}
        try:
            return self._inventory[pid]
        except KeyError:
            raise ProcessNotFoundError(f"PID {pid} not present in process list") from None

    def enumerate_comfyui_hierarchy(self, examiner_pids: Iterable[int] = ()) -> list[dict[str, Any]]:
        """Locates ComfyUI launchers (python running main.py or a python_embeded layout) and every
        descendant worker. A ComfyUI process whose ancestor is also ComfyUI is a WORKER_CHILD.
        Raises: IngestionError if process traversal fails unexpectedly."""
        procs = self._processes()
        by_pid: dict[int, Any] = {}
        parent: dict[int, int] = {}
        cmdlines: dict[int, str | None] = {}
        python_pids: set[int] = set()
        for proc in procs:
            pid = self._pid(proc)
            by_pid[pid] = proc
            parent[pid] = self._parent(proc)
            if self._is_python(self._name(proc)):
                python_pids.add(pid)
            cmdlines[pid] = self._cmdline(proc) if pid in python_pids else None
        self._inventory = by_pid

        self.unreadable_python_command_lines = any(not cmdlines[pid] for pid in python_pids)
        from_command = {pid for pid in python_pids if self._matches_command(by_pid[pid], cmdlines[pid])}
        hints = set(examiner_pids)
        for pid in sorted(hints - from_command):
            if pid not in python_pids:
                raise ProcessNotFoundError(f"Examiner-selected PID {pid} is not a listed Python process.")
            if cmdlines[pid]:
                raise ProcessNotFoundError(
                    f"The command line for examiner-selected PID {pid} does not identify ComfyUI.")
        comfy = from_command | hints

        def ancestors(pid: int) -> Iterable[int]:
            seen = {pid}
            while (pid := parent.get(pid, -1)) in by_pid and pid not in seen:
                seen.add(pid)
                yield pid

        selected: dict[int, str] = {}
        for pid in comfy:
            selected[pid] = ROLE_WORKER if any(a in comfy for a in ancestors(pid)) else ROLE_ROOT
        for pid in by_pid:
            if pid not in selected and any(a in comfy for a in ancestors(pid)):
                selected[pid] = ROLE_WORKER

        hierarchy = []
        for pid in sorted(selected):
            proc = by_pid[pid]
            name = self._name(proc)
            if cmdlines[pid] is None and not self._is_python(name):
                cmdlines[pid] = self._cmdline(proc)
            hierarchy.append(
                {
                    "pid": pid,
                    "ppid": parent[pid],
                    "process_name": name,
                    "command_line": cmdlines[pid],
                    **({"discovery_basis": ("COMMAND_LINE" if pid in from_command else
                                            "EXAMINER_PID" if pid in hints else "DESCENDANT")} if hints else {}),
                    "role": selected[pid],
                    **self._process_fields(proc),
                    "vad_count": len(self.get_vad_details(pid)),
                }
            )
        return hierarchy

    def _cmdline(self, proc: Any) -> str | None:
        # volatility returns objects.String, which cannot be deep-copied into the manifest.
        try:
            raw = self._get_cmdline(self.context, self._kernel_table, proc)
        except Exception:  # noqa: BLE001 - a paged-out PEB must not abort discovery
            log.debug("cmdline unreadable for PID %s", _field(proc, "UniqueProcessId"))
            return None
        return None if raw is None else str(raw)

    def _safe_vads(self, proc: Any) -> Iterable[Any]:
        try:
            yield from self._list_vads(proc)
        except Exception:  # noqa: BLE001 - a corrupt VAD tree yields what was walked
            log.debug("VAD walk aborted for PID %s", _field(proc, "UniqueProcessId"))

    def _region(self, proc: Any, vad: Any) -> tuple[int, int, str | None]:
        name = _field(vad, "get_file_name")
        try:
            path = name() if name else None
        except Exception:  # noqa: BLE001 - unreadable mapped file metadata
            path = None
        return int(vad.get_start()), int(vad.get_end()), str(path) if isinstance(path, str) else None

    def get_vad_details(self, pid: int) -> list[tuple[int, int, str | None]]:
        """Return each mapped region's inclusive bounds and mapped path, when readable."""
        if pid not in self._regions:
            proc = self._proc_by_pid(pid)
            regions = []
            for vad in self._safe_vads(proc):
                try:
                    start, end, path = self._region(proc, vad)
                except Exception:  # noqa: BLE001 - one corrupt mapping does not invalidate the other regions
                    log.debug("unreadable mapping in PID %d", pid)
                    continue
                if 0 <= start <= end:
                    regions.append((start, end, path))
            self._regions[pid] = regions
        return self._regions[pid]

    def get_process_vads(self, pid: int) -> list[tuple[int, int]]:
        """Returns inclusive virtual address boundaries [(vad.get_start(), vad.get_end()), ...].
        Raises: ProcessNotFoundError if PID is invalid or terminated."""
        return [(start, end) for start, end, _ in self.get_vad_details(pid)]

    def get_process_layer(self, pid: int) -> Any:
        """Builds (or reuses) the Volatility process layer for a PID.
        Raises: ProcessNotFoundError if PID is invalid or its layer cannot be constructed."""
        proc = self._proc_by_pid(pid)
        try:
            return self.context.layers[proc.add_process_layer()]
        except Exception as exc:
            raise ProcessNotFoundError(f"cannot construct process layer for PID {pid}: {exc}") from exc


MODE_VIRTUAL = "PROCESS_VIRTUAL"
MODE_PHYSICAL = "PHYSICAL_FALLBACK"


PAGE_SHIFT = 12


@dataclass
class AddressSpace:
    """One carving target: a ComfyUI process's VADs, or the whole physical image when unattributed.
    skip_pages holds physical page numbers already carved through a process space; a physical finding
    that starts in one of them is that process's copy and is not reported twice."""

    process_id: int | None
    blocks: Iterable[Any]
    source: Any
    translate: Callable[[int], int | None] | None = None
    skip_pages: frozenset[int] = frozenset()

    @property
    def physical_mode(self) -> bool:
        return self.process_id is None


@dataclass
class RecoverySupport:
    """Inputs for the pure-memory stages. Volatility contributes only supporting data (DTBs, VAD names, kernel
    symbols); every ComfyUI artefact is decoded by ComfyTrace's own walkers from the raw image."""

    physical: Any = None
    scattered: Any = None  # the same image mapped without read-around, for page-by-page reads of process memory
    kernel: Any = None
    pfn: Any = None
    store: Any = None
    python_processes: list[Any] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    image_path: Path | None = None  # the file ``physical`` maps, for passes that read it front to back


@dataclass
class IngestionResult:
    mode: str
    symbol_resolution: dict[str, Any]
    process_context: list[dict[str, Any]]
    spaces: list[AddressSpace]
    degradation_reason: str | None = None
    close: Callable[[], None] = field(default=lambda: None)
    recovery: RecoverySupport = field(default_factory=RecoverySupport)


def physical_space(
    memory_path: Path, skip_pages: frozenset[int] = frozenset()
) -> tuple[AddressSpace, Callable[[], None]]:
    from comfytrace.carve import BLOCK_OVERLAP, BLOCK_SIZE, Block, PhysicalSource, StreamReader
    from comfytrace.lime import LimePhysicalImage, parse_lime_segments

    reader = StreamReader(memory_path)
    mapped, handle = reader.open_physical()
    segments = parse_lime_segments(memory_path)
    if segments is not None:
        view = LimePhysicalImage(mapped, segments)

        def blocks() -> Iterable[Any]:
            with open(memory_path, "rb") as stream:
                for offset, data in view.iter_blocks(stream, BLOCK_SIZE, BLOCK_OVERLAP):
                    yield Block(offset, data)

        source = PhysicalSource(view)
    else:
        def blocks() -> Iterable[Any]:
            for offset, data in reader.stream_physical():
                yield Block(offset, data)

        source = PhysicalSource(mapped)

    def close() -> None:
        mapped.close()
        handle.close()

    return AddressSpace(None, blocks(), source, skip_pages=skip_pages), close


def physical_ingestion(memory_path: Path, symbol_resolution: dict[str, Any], reason: str,
                       context: Any = None, kernel_module_name: str | None = None,
                       walker: Any = None) -> IngestionResult:
    if symbol_resolution.get("capture_layout") == "SEGMENTED":
        log.warning("%s: segmented capture has no usable process scope; physical carving is unavailable", reason)
        recovery = RecoverySupport(notes={"python_objects": f"UNAVAILABLE: {reason}",
                                          "bitmap_ordering": "UNAVAILABLE: segmented capture"})
        return IngestionResult(MODE_PHYSICAL, symbol_resolution, [], [], reason, recovery=recovery)
    space, close = physical_space(memory_path)
    log.warning("degraded execution: physical-only carving (%s); findings are UNATTRIBUTED", reason)
    recovery, close_recovery = open_recovery(memory_path)
    recovery.image_path = memory_path
    recovery.notes["python_objects"] = f"UNAVAILABLE: {reason}"
    recovery.notes["bitmap_ordering"] = "PHYSICAL_CONTIGUITY (no kernel symbols)"
    if context is not None and kernel_module_name:
        if symbol_resolution.get("image_os") == "linux":
            linux_kernel_support(context, kernel_module_name, recovery, walker)
        else:
            kernel_support(context, kernel_module_name, recovery)
    elif symbol_resolution.get("image_os") == "linux":
        recovery.notes["bitmap_ordering"] = "PHYSICAL_CONTIGUITY (no kernel symbols)"

    def close_all() -> None:
        close()
        close_recovery()

    return IngestionResult(MODE_PHYSICAL, symbol_resolution, [], [space], reason, close_all, recovery)


def open_recovery(memory_path: Path) -> tuple[RecoverySupport, Callable[[], None]]:
    from comfytrace.carve import StreamReader
    from comfytrace.lime import LimePhysicalImage, parse_lime_segments

    mapped, handle = StreamReader(memory_path).open_physical()
    segments = parse_lime_segments(memory_path)
    # A process's pages lie all over the image. Through a plain mapping every page that is not cached makes the
    # kernel read a whole read-ahead window around it (megabytes), which pushes the process's other pages out of
    # the page cache again when the image is larger than RAM. This mapping reads one page per miss instead.
    scattered_mmap = None
    random_access = getattr(mmap, "MADV_RANDOM", None)
    if random_access is not None:
        scattered_mmap = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        madvise = getattr(scattered_mmap, "madvise", None)
        if madvise is not None:
            madvise(random_access)

    physical: Any = LimePhysicalImage(mapped, segments) if segments is not None else mapped
    scattered: Any = (
        LimePhysicalImage(scattered_mmap, segments)
        if (scattered_mmap is not None and segments is not None)
        else scattered_mmap
    )

    def close() -> None:
        if scattered_mmap is not None:
            scattered_mmap.close()
        mapped.close()
        handle.close()

    return RecoverySupport(physical=physical, scattered=scattered), close


def linux_kernel_support(context: Any, kernel_module_name: str, recovery: RecoverySupport,
                         walker: Any = None) -> None:
    """Linux kernel page tables, ``vmemmap`` PFN provenance/ordering, and ``zswap`` compressed-store support."""
    from comfytrace.linux_memstore import open_linux_store_reader
    from comfytrace.linux_pfn import open_linux_pfn_support
    from comfytrace.pagetable import PageTables

    try:
        kernel = context.modules[kernel_module_name]
        layer = context.layers[kernel.layer_name]
        levels = 5 if "5Level" in type(layer).__name__ or layer.config.get("pgtable_l5") else 4
        if levels == 4:
            try:
                l5_sym = kernel.get_symbol("__pgtable_l5_enabled")
                l5_raw = layer.read(int(kernel.offset) + int(l5_sym.address), 4, pad=False)
                if len(l5_raw) == 4 and int.from_bytes(l5_raw, "little") != 0:
                    levels = 5
            except Exception:  # noqa: BLE001 - absent on non-LA57 kernels
                pass
        recovery.kernel = PageTables(recovery.physical, int(layer.config["page_map_offset"]),
                                     os_mode="linux", levels=levels)
        loader = ((lambda: walker.collect_process_tables(recovery.kernel))
                  if walker is not None and hasattr(walker, "collect_process_tables") else None)
        recovery.pfn = open_linux_pfn_support(context, kernel_module_name, recovery.kernel,
                                              process_tables_loader=loader)
        recovery.notes["bitmap_ordering"] = "PAGE_CACHE_ORDER"
    except Exception as exc:  # noqa: BLE001 - missing symbols only lose page provenance/ordering
        log.warning("Linux vmemmap PFN support unavailable (%s); decoded bitmaps ordered by physical contiguity", exc)
        recovery.notes["bitmap_ordering"] = f"PHYSICAL_CONTIGUITY ({exc})"
    if recovery.kernel is not None:
        recovery.store, recovery.notes["compression_store"] = open_linux_store_reader(
            context, kernel_module_name, recovery.kernel
        )


def kernel_support(context: Any, kernel_module_name: str, recovery: RecoverySupport) -> None:
    """Kernel page tables and PFN-database facts from symbols (supporting data only)."""
    from comfytrace.bitmaps import PfnSupport
    from comfytrace.pagetable import PageTables, canonical

    try:
        kernel = context.modules[kernel_module_name]
        layer = context.layers[kernel.layer_name]
        recovery.kernel = PageTables(recovery.physical, int(layer.config["page_map_offset"]))
        base = canonical(kernel.offset)
        pfn_db = recovery.kernel.read_kernel_qword(base + kernel.get_symbol("MmPfnDatabase").address)
        pte_base = recovery.kernel.read_kernel_qword(base + kernel.get_symbol("MmPteBase").address)
        mmpfn = context.symbol_space.get_type(kernel.symbol_table_name + "!_MMPFN")
        if pfn_db is None or pte_base is None:
            raise ValueError("MmPfnDatabase or MmPteBase not readable")
        recovery.pfn = PfnSupport(recovery.kernel, pfn_db, int(mmpfn.size),
                                  int(mmpfn.vol.members["PteAddress"][0]), pte_base, *_pfn_fields(context, mmpfn),
                                  section=_section_layout(context, kernel.symbol_table_name, mmpfn),
                                  pte_frame=_pte_frame_field(context, mmpfn))
        recovery.notes["bitmap_ordering"] = "PROTOTYPE_PTE_ORDER"
    except Exception as exc:  # noqa: BLE001 - missing symbols only lose page ordering
        log.warning("PFN database unavailable (%s); decoded bitmaps ordered by physical contiguity", exc)
        recovery.notes["bitmap_ordering"] = f"PHYSICAL_CONTIGUITY ({exc})"
    if recovery.kernel is not None:
        recovery.store, recovery.notes["compression_store"] = _store_reader(context, kernel_module_name, recovery)


def _store_reader(context: Any, kernel_module_name: str, recovery: RecoverySupport) -> tuple[Any, str]:
    """The compression-store reader when the image passes its layout checks, and a note saying why not."""
    from comfytrace.memstore import Corrupt, StoreReader
    from comfytrace.pagetable import canonical

    try:
        kernel = context.modules[kernel_module_name]
        table = kernel.symbol_table_name
        base = canonical(kernel.offset)
        smglobals = base + kernel.get_symbol("?SmGlobals@@3U_SM_GLOBALS@@A").address
        partition = context.symbol_space.get_type(table + "!_MI_PARTITION")
        stores_offset, stores = partition.vol.members["Store"]
        number_offset, _ = context.symbol_space.get_type(stores.vol.type_name).vol.members["VirtualPageFileNumber"]
        where = base + kernel.get_symbol("MiSystemPartition").address + stores_offset + number_offset
        pagefile = recovery.kernel.read_kernel_qword(where)
        name_offset = context.symbol_space.get_type(table + "!_EPROCESS").vol.members["ImageFileName"][0]
    except Exception as exc:  # noqa: BLE001 - symbols for the store are optional
        return None, f"UNAVAILABLE: store symbols not resolved ({exc})"
    if pagefile is None:
        return None, "UNAVAILABLE: virtual store pagefile number unreadable"

    def owner_name(eprocess: int) -> str | None:
        # ImageFileName can straddle a page boundary of a pool-allocated EPROCESS: read it virtually.
        raw = recovery.kernel.read_virtual(eprocess + name_offset, 15) if eprocess else None
        return None if raw is None else raw.split(b"\0")[0].decode("latin-1")

    try:
        reader = StoreReader.open(recovery.kernel, smglobals, pagefile & 0xF, owner_name)
    except Corrupt as exc:
        return None, f"UNAVAILABLE: {exc}"
    return reader, f"AVAILABLE: virtual store pagefile {pagefile & 0xF}"


def _pfn_fields(context: Any, mmpfn: Any) -> tuple[int | None, tuple[int, int] | None]:
    """(byte offset of _MMPFN.u3.e1.PageLocation, (offset of _MMPFN.u4, bit of PrototypePte)), each None when
    this kernel's layout differs; page mappings are then not checked against the PFN database."""
    def member(template: Any, name: str) -> tuple[int, Any]:
        resolved = context.symbol_space.get_type(template.vol.type_name) if hasattr(template.vol, "type_name") \
            and "members" not in template.vol else template
        offset, child = resolved.vol.members[name]
        return int(offset), child

    location = prototype = None
    try:
        u3_offset, u3 = member(mmpfn, "u3")
        e1_offset, e1 = member(u3, "e1")
        offset, field = member(e1, "PageLocation")
        if field.vol.start_bit == 0 and field.vol.end_bit == 3:
            location = u3_offset + e1_offset + offset
        u4_offset, u4 = member(mmpfn, "u4")
        offset, field = member(u4, "PrototypePte")
        if field.vol.end_bit - field.vol.start_bit == 1:
            prototype = (u4_offset + offset, int(field.vol.start_bit))
    except Exception:  # noqa: BLE001 - another kernel layout only loses the mapping check
        pass
    if location is None or prototype is None:
        log.warning("_MMPFN PageLocation or PrototypePte not found; page mappings are not checked against the PFN "
                    "database")
    return location, prototype


def _pte_frame_field(context: Any, mmpfn: Any) -> tuple[int, int, int] | None:
    """Where _MMPFN.u4.PteFrame lies: (byte offset, first bit, bits). None when this kernel's layout differs;
    private decoded bitmaps then cannot be put in order."""
    try:
        u4_offset, u4 = mmpfn.vol.members["u4"]
        u4_type = context.symbol_space.get_type(u4.vol.type_name)
        offset, field = u4_type.vol.members["PteFrame"]
        return int(u4_offset) + int(offset), int(field.vol.start_bit), int(field.vol.end_bit - field.vol.start_bit)
    except Exception:  # noqa: BLE001 - another kernel layout only loses private bitmap ordering
        log.warning("_MMPFN.u4.PteFrame not found; private decoded bitmaps are not ordered")
        return None


def _section_layout(context: Any, table: str, mmpfn: Any) -> Any:
    """Offsets that tell file-backed section pages from shared memory, or None when this kernel's layout differs;
    section pages are then of UNKNOWN origin."""
    from comfytrace.bitmaps import SectionLayout

    try:
        area = context.symbol_space.get_type(table + "!_CONTROL_AREA")
        u_offset, u = area.vol.members["u"]
        flags_offset, flags = context.symbol_space.get_type(u.vol.type_name).vol.members["Flags"]
        bits = context.symbol_space.get_type(flags.vol.type_name).vol.members
        if any(bits[name][1].vol.end_bit - bits[name][1].vol.start_bit != 1 for name in ("File", "Image")) or \
                bits["File"][0] != bits["Image"][0]:
            raise ValueError("File and Image are not single bits of one word")
        subsection = context.symbol_space.get_type(table + "!_SUBSECTION")
        return SectionLayout(
            original_pte_offset=int(mmpfn.vol.members["OriginalPte"][0]),
            control_area_offset=int(subsection.vol.members["ControlArea"][0]),
            flags_offset=int(u_offset) + int(flags_offset) + int(bits["File"][0]),
            file_bit=int(bits["File"][1].vol.start_bit), image_bit=int(bits["Image"][1].vol.start_bit),
            file_name=_file_name_layout(context, table, area))
    except Exception:  # noqa: BLE001 - another kernel layout only loses the file/shared distinction
        log.warning("_CONTROL_AREA flags or _MMPFN.OriginalPte not found; section pages are of unknown origin")
        return None


def _file_name_layout(context: Any, table: str, area: Any) -> Any:
    from comfytrace.bitmaps import FileNameLayout

    try:
        fast_ref = context.symbol_space.get_type(table + "!_EX_FAST_REF")
        file_object = context.symbol_space.get_type(table + "!_FILE_OBJECT")
        unicode_string = context.symbol_space.get_type(table + "!_UNICODE_STRING")
        bits = fast_ref.vol.members["RefCnt"][1].vol
        if bits.start_bit != 0 or not 0 < bits.end_bit <= 4:
            return None
        return FileNameLayout(
            file_pointer=int(area.vol.members["FilePointer"][0]),
            reference_bits=int(bits.end_bit),
            file_name=int(file_object.vol.members["FileName"][0]),
            length=int(unicode_string.vol.members["Length"][0]),
            maximum_length=int(unicode_string.vol.members["MaximumLength"][0]),
            buffer=int(unicode_string.vol.members["Buffer"][0]),
        )
    except Exception:  # noqa: BLE001 - missing optional symbols leave filename provenance unavailable
        return None


def python_memories(walker: ProcessWalker, hierarchy: list[dict[str, Any]], recovery: RecoverySupport) -> None:
    from comfytrace.pagetable import PageTables
    from comfytrace.procmem import ProcessMemory

    for proc in hierarchy:
        if str(proc.get("process_name", "")).lower() not in PYTHON_IMAGE_NAMES or not proc.get("dtb"):
            continue
        try:
            vads = walker.get_vad_details(proc["pid"])
        except (ProcessNotFoundError, IngestionError) as exc:
            log.warning("PID %s: VADs unreadable (%s)", proc["pid"], exc)
            continue
        tables = PageTables(recovery.physical, int(proc["dtb"], 16), kernel=recovery.kernel)
        image = recovery.scattered if recovery.scattered is not None else recovery.physical
        recovery.python_processes.append(
            ProcessMemory.from_page_tables(proc["pid"], tables, vads, image, recovery.pfn, recovery.store))


def is_raw_physical(layer: Any) -> bool:
    """True when the layer beneath a process/kernel layer exposes direct physical addresses (a flat raw
    ``FileLayer`` or a LiME v1 ``LimeLayer``)."""
    from volatility3.framework.layers.lime import LimeLayer
    from volatility3.framework.layers.physical import FileLayer

    try:
        underlying = layer.context.layers[layer.config["memory_layer"]]
        return isinstance(underlying, (FileLayer, LimeLayer))
    except Exception:  # noqa: BLE001 - any doubt means the offsets cannot be compared
        return False


def has_capture_container_header(memory_path: Path) -> bool:
    """Recognize containers whose file positions cannot be interpreted as flat physical addresses."""
    with memory_path.open("rb") as source:
        header = source.read(8)
    return header.startswith((b"EMiL", b"\x7fELF", b"PAGEDUMP", b"PAGEDU64"))


def owned_pages(layer: Any, vads: list[tuple[int, int]]) -> set[int]:
    """Physical page numbers backing a process's resident VAD pages."""
    pages: set[int] = set()
    for start, end in vads:
        try:
            for _off, _len, mapped_offset, mapped_length, _layer in layer.mapping(
                start, end - start + 1, ignore_errors=True
            ):
                if mapped_length:
                    last = (mapped_offset + mapped_length - 1) >> PAGE_SHIFT
                    pages.update(range(mapped_offset >> PAGE_SHIFT, last + 1))
        except Exception:  # noqa: BLE001 - an unreadable VAD only means its pages stay in the physical pass
            log.debug("page ownership unavailable for VAD 0x%x-0x%x", start, end)
    return pages


def process_space(
    pid: int, layer: Any, vads: list[tuple[int, int]], reader: Any, engine: VolatilityEngine | None, *,
    file_offsets: bool = True,
) -> AddressSpace:
    """Binds one process layer to its VAD block stream and address translation."""
    from comfytrace.carve import ProcessLayerSource

    def blocks() -> Iterable[Any]:
        for start, end in vads:
            yield from reader.iter_vad_blocks(layer, start, end)

    translate = (lambda va: engine.translate_virtual_to_physical(layer, va)) if file_offsets and engine else None
    return AddressSpace(pid, blocks(), ProcessLayerSource(layer), translate)


def ingest_memory(memory_path: Path, symbol_manager: SymbolManager, *,
                  examiner_pids: Iterable[int] = ()) -> IngestionResult:
    """Resolves kernel symbols and ComfyUI processes; falls back to physical carving when either fails.
    Raises: IngestionError when the image itself is unreadable."""
    from comfytrace.carve import StreamReader

    status: dict[str, Any] = {
        "image_os": symbol_manager.image_os,
        "mode": symbol_manager.mode,
        "symbols_dir": str(symbol_manager.symbols_dir) if symbol_manager.symbols_dir is not None else None,
        "remote_download_permitted": symbol_manager.allow_download,
        "status": "SUCCESS",
    }
    if not Path(memory_path).is_file():
        raise IngestionError(f"memory image not found or unreadable: {memory_path}")
    requested = set(examiner_pids)
    if symbol_manager.image_format == "elf-core":
        from comfytrace.elfcore import ingest_core

        return ingest_core(memory_path, requested)
    if has_capture_container_header(memory_path):
        from comfytrace.lime import parse_lime_segments

        if parse_lime_segments(memory_path) is not None:
            status["capture_layout"] = "LIME"
        else:
            status["capture_layout"] = "SEGMENTED"
    engine = VolatilityEngine(memory_path, symbol_manager)
    try:
        context = engine.initialize_context()
    except SymbolResolutionError as exc:
        if requested:
            raise ProcessNotFoundError("Cannot apply examiner PIDs without resolved kernel symbols.") from exc
        status["status"] = "FAILED"
        status["error"] = str(exc)
        return physical_ingestion(memory_path, status, "KERNEL_SYMBOLS_UNRESOLVED")
    kernel = engine.kernel_module_name or "kernel"
    try:
        layer_name = context.modules[kernel].layer_name
        kernel_layer = context.layers[layer_name] if isinstance(layer_name, str) else None
    except (AttributeError, KeyError, TypeError):
        kernel_layer = None
    if kernel_layer is not None and not is_raw_physical(kernel_layer):
        status["capture_layout"] = "SEGMENTED"
    linux = symbol_manager.image_os == "linux"
    if linux:
        from comfytrace.ingest_linux import LinuxProcessWalker

        walker = LinuxProcessWalker(context, kernel)
    else:
        walker = ProcessWalker(context, kernel)
    try:
        hierarchy = (walker.enumerate_comfyui_hierarchy(examiner_pids=requested) if requested
                     else walker.enumerate_comfyui_hierarchy())
    except ProcessNotFoundError:
        raise
    except IngestionError as exc:
        if requested:
            raise ProcessNotFoundError("Cannot apply examiner PIDs to an unreadable process list.") from exc
        log.warning("%s", exc)
        return physical_ingestion(memory_path, status, "PROCESS_LIST_UNREADABLE", context, engine.kernel_module_name)
    if not hierarchy:
        reason = "PYTHON_COMMAND_LINES_UNREADABLE" if walker.unreadable_python_command_lines else "NO_COMFYUI_PROCESS"
        return physical_ingestion(memory_path, status, reason, context, engine.kernel_module_name, walker=walker)

    reader = StreamReader(memory_path)
    spaces = []
    layers = {}
    readable_pids: set[int] = set()
    owned: set[int] = set()
    raw = status.get("capture_layout") != "SEGMENTED"
    for proc in hierarchy:
        pid = proc["pid"]
        try:
            layer = walker.get_process_layer(pid)
            vads = walker.get_process_vads(pid)
        except (ProcessNotFoundError, IngestionError) as exc:
            log.warning("skipping PID %s: %s", pid, exc)
            continue
        file_offsets = status.get("capture_layout") != "SEGMENTED" and is_raw_physical(layer)
        spaces.append(process_space(pid, layer, vads, reader, engine, file_offsets=file_offsets))
        layers[pid] = layer
        readable_pids.add(pid)
        raw = raw and file_offsets
        if raw:
            owned |= owned_pages(layer, vads)
    if unavailable := requested - readable_pids:
        ids = ", ".join(map(str, sorted(unavailable)))
        raise ProcessNotFoundError(f"Process layer unavailable for examiner PIDs {ids}.")
    if not spaces:
        return physical_ingestion(memory_path, status, "PROCESS_LAYERS_UNREADABLE", context, engine.kernel_module_name)
    if not raw:
        log.warning("image is not a raw physical dump; skipping the pass over unowned pages")
        status["capture_layout"] = "SEGMENTED"
        result = IngestionResult(MODE_VIRTUAL, status, hierarchy, spaces)
        if linux:
            _linux_python_memories(walker, hierarchy, layers, result.recovery)
            result.recovery.notes["bitmap_ordering"] = "UNAVAILABLE: segmented capture"
        else:
            result.recovery.notes["python_objects"] = "UNAVAILABLE: image is not a raw physical dump"
        return result
    # Freed and file-cache pages hold most generated PNGs once ComfyUI has moved on; no process owns them.
    unowned, close = physical_space(memory_path, frozenset(owned))
    recovery, close_recovery = open_recovery(memory_path)
    recovery.image_path = memory_path
    if linux:
        linux_kernel_support(context, kernel, recovery, walker)
        _linux_python_memories(walker, hierarchy, layers, recovery)
    else:
        kernel_support(context, kernel, recovery)
        python_memories(walker, hierarchy, recovery)

    def close_all() -> None:
        close()
        close_recovery()

    return IngestionResult(MODE_VIRTUAL, status, hierarchy, [*spaces, unowned], close=close_all, recovery=recovery)


def _linux_python_memories(walker: ProcessWalker, hierarchy: list[dict[str, Any]],
                           layers: dict[int, Any], recovery: RecoverySupport) -> None:
    from comfytrace.pagetable import PageTables
    from comfytrace.procmem import ProcessMemory

    physical = recovery.scattered if recovery.scattered is not None else recovery.physical
    for proc in hierarchy:
        pid = proc["pid"]
        if pid not in layers or not walker._is_python(proc["process_name"]):
            continue
        vads = walker.get_vad_details(pid)
        if recovery.kernel is not None and proc.get("dtb") and physical is not None:
            tables = PageTables(recovery.physical, int(proc["dtb"], 16), kernel=recovery.kernel,
                                os_mode="linux", levels=recovery.kernel.levels)
            recovery.python_processes.append(
                ProcessMemory.from_page_tables(pid, tables, vads, physical, recovery.pfn, recovery.store,
                                               image_os="linux")
            )
        else:
            recovery.python_processes.append(
                ProcessMemory.from_layer(pid, layers[pid], vads, physical, image_os="linux")
            )


def ingest_case(memory_path: Path, symbols_dir: Path | None, case_metadata: dict[str, Any],
                ingestor: Callable[..., IngestionResult] = ingest_memory) -> IngestionResult:
    """Reopen a case's captured OS and examiner-selected scope for supplementary memory recovery."""
    source = case_metadata.get("symbol_resolution") or {}
    image_format = source.get("image_format", "physical")
    if image_format == "physical" and symbols_dir is None:
        raise ValueError("pass --symbols-dir for process recovery from a physical capture")
    manager = SymbolManager(symbols_dir, image_os=source.get("image_os", "windows"), image_format=image_format)
    if pids := case_metadata.get("examiner_comfyui_pids"):
        return ingestor(memory_path, manager, examiner_pids=pids)
    return ingestor(memory_path, manager)


def supports_raw_search(case_metadata: dict[str, Any], memory_path: Path | None = None) -> bool:
    from comfytrace.lime import parse_lime_segments

    source = case_metadata.get("symbol_resolution") or {}
    if source.get("image_format") == "elf-core" or source.get("capture_layout") == "SEGMENTED":
        return False
    if memory_path is None:
        return True
    if not has_capture_container_header(memory_path):
        return True
    return parse_lime_segments(memory_path) is not None
