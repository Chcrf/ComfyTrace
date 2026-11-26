"""CPython 3.13, release ABI, x86-64, Windows (python313.dll).

3.13 prefixes PyDictValues with an 8-byte header (capacity, size, embedded, valid), so split values start at 8.
"""

from comfytrace.cpython.base import CPythonLayout

CPYTHON_313 = CPythonLayout(keys=(("windows", (3, 13), ""),), split_values_offset=8)

LAYOUTS = (CPYTHON_313,)
