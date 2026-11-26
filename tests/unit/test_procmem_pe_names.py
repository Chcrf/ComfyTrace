"""An image whose VAD lost its file name is still identified by its own PE export directory. Synthetic bytes."""

from __future__ import annotations

import struct

from comfytrace.procmem import PE_EXPORT_PATH, MappedImage, ProcessMemory
from comfytrace.pyheap import PYTHON_DLL

BASE = 0x7FFD_A4D7_0000
SIZE = 0x3000


def _pe(export_name: bytes | None, magic: int = 0x20B) -> bytes:
    image = bytearray(SIZE)
    image[:2] = b"MZ"
    image[0x3C:0x40] = struct.pack("<I", 0x80)
    image[0x80:0x84] = b"PE\0\0"
    optional = 0x80 + 24
    image[optional:optional + 2] = struct.pack("<H", magic)
    directories = optional + (112 if magic == 0x20B else 96)
    if export_name is not None:
        image[directories:directories + 4] = struct.pack("<I", 0x1000)
        image[0x1000 + 12:0x1000 + 16] = struct.pack("<I", 0x1100)
        image[0x1100:0x1100 + len(export_name) + 1] = export_name + b"\0"
    return bytes(image)


def _memory(image: bytes, name: str | None) -> ProcessMemory:
    return ProcessMemory(4242, [(BASE, len(image), 0)], image, vads=[(BASE, BASE + len(image) - 1, name)])


def test_an_unnamed_vad_is_named_by_its_export_directory():
    mem = _memory(_pe(b"python312.dll"), None)
    found = mem.image(lambda name: bool(PYTHON_DLL.fullmatch(name)))
    assert found is not None and (found.name, found.path, found.base, found.end) == (
        "python312.dll", PE_EXPORT_PATH, BASE, BASE + SIZE)


def test_a_named_image_wins_and_non_images_are_ignored():
    named = _memory(_pe(b"other.dll"), r"\Windows\python312.dll")
    named.images.append(MappedImage("python312.dll", r"\Windows\python312.dll", BASE, BASE + SIZE))
    assert named.image(lambda name: name.startswith("python")).path == r"\Windows\python312.dll"
    assert named.pe_images() == [], "a VAD with a file name is not re-read"
    assert _memory(_pe(None), None).image(lambda name: True) is None, "no export directory, no name"
    assert _memory(b"\0" * SIZE, None).image(lambda name: True) is None, "not a PE image"
    assert _memory(_pe(b"bad name\x01.dll"), None).image(lambda name: True) is None, "names must be plain"
    assert _memory(_pe(b"python312.dll", magic=0x10B), None).image(
        lambda name: name == "python312.dll") is not None, "PE32 headers place the directories 16 bytes earlier"
