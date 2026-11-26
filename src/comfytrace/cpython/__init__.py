"""CPython object layouts, runtime detection and the profile registry.

``base`` interprets CPython structs byte by byte through an immutable ``CPythonLayout``; ``runtime`` finds the
interpreter image of a process and its version; ``registry`` maps ``(image_os, (major, minor), build_flags)`` to
the layout registered for it by a ``profiles.vNNN`` module.
"""

from comfytrace.cpython.base import CorruptObject, CPythonLayout

__all__ = ["CPythonLayout", "CorruptObject"]
