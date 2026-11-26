"""CPython 3.12, release ABI, x86-64: Windows (python312.dll) and Linux (libpython3.12 / static python3.12).

3.12 keeps split-dict values at offset 0 of PyDictValues.
"""

from comfytrace.cpython.base import CPythonLayout

CPYTHON_312 = CPythonLayout(keys=(("windows", (3, 12), ""), ("linux", (3, 12), "")), split_values_offset=0)

LAYOUTS = (CPYTHON_312,)
