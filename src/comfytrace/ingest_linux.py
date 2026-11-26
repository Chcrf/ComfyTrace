"""Linux task and VMA access through Volatility's captured-kernel layers."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from comfytrace.ingest import ProcessWalker, _field

PYTHON_NAME = re.compile(r"python(?:3(?:\.\d+)?)?")
UNKNOWN_BACKING = "<unresolved Linux mapping>"


class LinuxProcessWalker(ProcessWalker):
    def __init__(self, context: Any, kernel_module_name: str) -> None:
        from volatility3.plugins.linux.pslist import PsList

        super().__init__(context, kernel_module_name, list_processes=PsList.list_tasks,
                         get_cmdline=self._read_arguments, list_vads=self._mapped_regions)

    def _processes(self) -> list[Any]:
        from comfytrace.exceptions import IngestionError

        try:
            return list(self._list_processes(self.context, self.kernel_module_name))
        except Exception as exc:  # noqa: BLE001 - kernel task list may be unreadable
            raise IngestionError(f"Linux task list traversal failed: {exc}") from exc

    @staticmethod
    def _pid(proc: Any) -> int:
        return int(proc.pid)

    @staticmethod
    def _parent(proc: Any) -> int:
        try:
            return int(proc.get_parent_pid())
        except Exception:  # noqa: BLE001 - parent links are independent of a readable task
            return -1

    @staticmethod
    def _name(proc: Any) -> str:
        from volatility3.framework.objects.utility import array_to_string

        return str(array_to_string(proc.comm))

    @staticmethod
    def _is_python(name: str) -> bool:
        return PYTHON_NAME.fullmatch(name) is not None

    def _dtb(self, proc: Any) -> int | None:
        try:
            mm = proc.mm
            if not mm:
                return None
            pgd = int(mm.pgd)
            if not pgd:
                return None
            kernel_layer_name = self.context.modules[self.kernel_module_name].layer_name
            kernel_layer = self.context.layers[kernel_layer_name]
            for offset, _sublen, mapped_offset, _mlen, _layer in kernel_layer.mapping(pgd, 1, ignore_errors=True):
                if offset <= pgd:
                    return int(mapped_offset + (pgd - offset))
        except Exception:  # noqa: BLE001 - unreadable mm or pgd leaves dtb unavailable
            return None
        return None

    def _process_fields(self, proc: Any) -> dict[str, Any]:
        dtb = self._dtb(proc)
        return {
            "threads": int(_field(proc, "signal", "nr_threads", default=0) or 0),
            "dtb": hex(dtb) if dtb is not None else None,
        }

    def collect_process_tables(self, kernel: Any) -> list[tuple[int, Any, list[tuple[int, int]]]]:
        """Collect ``(dtb, PageTables, [(0, user_limit - 1)])`` for user tasks to support private page ordering.

        Avoids walking every process's VMA maple tree in Volatility (which is slow across hundreds of tasks);
        walking each process's hardware page tables directly in ``PageTables`` is orders of magnitude faster.
        """
        from comfytrace.pagetable import PageTables

        tables: list[tuple[int, Any, list[tuple[int, int]]]] = []
        seen_dtbs: set[int] = set()
        user_limit = (1 << 56) if kernel.levels == 5 else (1 << 47)
        try:
            procs = list(self._inventory.values()) if self._inventory is not None else self._processes()
        except Exception:  # noqa: BLE001
            return tables
        for proc in procs:
            dtb = self._dtb(proc)
            if dtb is None or dtb in seen_dtbs:
                continue
            seen_dtbs.add(dtb)
            tables.append((
                dtb,
                PageTables(kernel.physical, dtb, kernel=kernel, os_mode="linux", levels=kernel.levels),
                [(0, user_limit - 1)],
            ))
        return tables

    @staticmethod
    def _read_arguments(context: Any, _table: str, proc: Any) -> str | None:
        mm = proc.mm
        if not mm:
            return None
        start, end = int(mm.arg_start), int(mm.arg_end)
        if not 0 < end - start <= 65536:
            return None
        layer = context.layers[proc.add_process_layer()]
        data = layer.read(start, end - start, pad=False)
        if len(data) != end - start or not data.endswith(b"\0"):
            return None
        return shlex.join(data[:-1].decode("utf-8", errors="replace").split("\0"))

    @staticmethod
    def _cwd(proc: Any) -> str | None:
        from volatility3.framework.symbols.linux import LinuxUtilities

        try:
            fs = proc.fs
            return LinuxUtilities.do_get_path(fs.get_root_dentry(), fs.get_root_mnt(), fs.pwd.dentry, fs.pwd.mnt)
        except Exception:  # noqa: BLE001 - working directory is optional process evidence
            return None

    def _matches_command(self, proc: Any, cmdline: str | None) -> bool:
        if not cmdline:
            return False
        args = shlex.split(cmdline)[1:]
        while args and args[0] in {"-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "--"}:
            args = args[1:]
        if args[:2] == ["-m", "comfyui"]:
            return True
        if not args or args[0].startswith("-"):
            return False
        script = PurePosixPath(args[0])
        if script.name != "main.py":
            return False
        if not script.is_absolute():
            script = PurePosixPath(self._cwd(proc) or "") / script
        return any(part.lower() == "comfyui" for part in script.parent.parts)

    @staticmethod
    def _mapped_regions(proc: Any) -> Iterable[Any]:
        if proc.mm:
            yield from proc.mm.get_vma_iter()

    def _region(self, proc: Any, vad: Any) -> tuple[int, int, str | None]:
        path = None
        try:
            if vad.vm_file:
                path = vad.get_name(self.context, proc) or UNKNOWN_BACKING
            elif "s" in vad.get_protection():
                path = UNKNOWN_BACKING
        except Exception:  # noqa: BLE001 - absence of a name is insufficient to claim private ownership
            path = UNKNOWN_BACKING
        return int(vad.vm_start), int(vad.vm_end) - 1, str(path) if path else None
