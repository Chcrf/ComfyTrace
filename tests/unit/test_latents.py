"""Latents of finished jobs in process memory: found by shape and smoothness, CPU noise and misread shapes refused."""

import json
import struct

import numpy as np
import pytest

from comfytrace import latents
from comfytrace.procmem import ProcessMemory

BASE = 0x20000000


def _smooth(seed: int, h: int, w: int) -> np.ndarray:
    """A latent-like field: four channels of smoothed noise with standard deviation about 4."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((latents.CHANNELS, h + 4, w + 4))
    for axis in (1, 2):
        x = sum(np.roll(x, k, axis) for k in range(5)) / 5
    x = x[:, 2:-2, 2:-2]
    return (4 * x / x.std()).astype(np.float32)


def _memory(blocks: list[bytes], mapped: bytes = b"") -> ProcessMemory:
    """Consecutive 64 KiB slots of private memory, then an optional file-backed view."""
    private = b"".join(b + b"\0" * (-len(b) % latents.SLOT) for b in blocks)
    runs = [(BASE, len(private), 0)]
    vads = [(BASE, BASE + len(private) - 1, None)]
    if mapped:
        runs.append((BASE + 0x1000000, len(mapped), len(private)))
        vads.append((BASE + 0x1000000, BASE + 0x1000000 + len(mapped) - 1, r"\x\file.bin"))
    return ProcessMemory(5, runs, private + mapped, vads=vads)


def _freed(values: np.ndarray, link: int) -> bytes:
    raw = bytearray(values.tobytes())
    raw[:8] = struct.pack("<Q", link)
    return bytes(raw)


def test_freed_latents_are_found_at_their_own_shape_and_noise_is_not():
    small, large = _smooth(1, 64, 64), _smooth(2, 128, 128)
    noise = np.random.default_rng(3).standard_normal((4, 64, 64)).astype(np.float32)
    mem = _memory([noise.tobytes(), _freed(small, BASE + 0x40000), noise.tobytes(), large.tobytes()])
    found = latents.find(mem, [(64, 64), (128, 128)])
    assert [(b.address - BASE, b.height, b.first_word) for b in found] == [
        (0x10000, 64, "POINTER"), (0x30000, 128, "DATA")]
    np.testing.assert_array_equal(found[0].values()[:, 1:], small[:, 1:])
    np.testing.assert_array_equal(found[1].values(), large)


def test_half_of_a_larger_latent_is_not_read_as_a_smaller_one():
    half = _smooth(4, 128, 128).tobytes()[: 4 * 64 * 64 * 4]
    assert latents.find(_memory([half]), [(64, 64), (128, 128)]) == []


def test_four_small_latents_in_a_row_are_not_read_as_one_large_one():
    blocks = [_smooth(10 + i, 64, 64).tobytes() for i in range(4)]
    found = latents.find(_memory(blocks), [(64, 64), (128, 128)])
    assert [(b.address - BASE, b.height) for b in found] == [(i * latents.SLOT, 64) for i in range(4)]


def test_a_slot_whose_run_continues_into_a_file_view_is_not_taken_from_it():
    private, mapped = b"\0" * latents.SLOT, _smooth(8, 64, 64).tobytes()
    mem = ProcessMemory(1, [(BASE, len(private + mapped), 0)], private + mapped,
                        vads=[(BASE, BASE + latents.SLOT - 1, None),
                              (BASE + latents.SLOT, BASE + 2 * latents.SLOT - 1, "checkpoint.safetensors")])
    assert latents.find(mem, [(64, 64)]) == []


def test_only_a_pointer_first_word_is_exempt():
    raw = bytearray(_smooth(9, 64, 64).tobytes())
    assert latents.assess(bytes(raw), 64, 64) is not None
    raw[:8] = struct.pack("<ff", float("nan"), float("inf"))
    assert latents.assess(bytes(raw), 64, 64) is None


def test_exported_latent_bytes_are_never_overwritten(tmp_path):
    block = latents.LatentBlock(5, BASE, 64, 64, b"first", [], "DATA", {})
    latents.document(tmp_path, [block], "0" * 64, [(64, 64)])
    block.raw = b"second"
    with pytest.raises(FileExistsError):
        latents.document(tmp_path, [block], "0" * 64, [(64, 64)])
    assert (tmp_path / "exports" / "raw" / "LATENT-001.f32").read_bytes() == b"first"


def test_file_backed_views_are_not_searched():
    mem = _memory([np.zeros(16384, np.float32).tobytes()], mapped=_smooth(5, 64, 64).tobytes())
    assert latents.find(mem, [(64, 64)]) == []


@pytest.mark.parametrize("damage", ["nan", "zeros", "flat_channel"])
def test_damaged_blocks_are_refused(damage):
    x = _smooth(6, 64, 64)
    if damage == "nan":
        x[2, 10, 10] = np.nan
    elif damage == "zeros":
        x[1] = 0
    else:
        x[3] = 0.01 * x[3]
    assert latents.find(_memory([x.tobytes()]), [(64, 64)]) == []


def test_sides_come_from_recovered_prompts_and_the_sidecar_is_checked(tmp_path):
    graph = {"5": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
             "6": {"class_type": "KSampler", "inputs": {"latent_image": ["5", 0]}}}
    assert (64, 96) in latents.latent_sides([graph])
    blocks = latents.find(_memory([_smooth(7, 64, 64).tobytes()]), [(64, 64)])
    doc = latents.document(tmp_path, blocks, "ab" * 32, [(64, 64)])
    latents.write(tmp_path, doc)
    with pytest.raises(FileExistsError):
        latents.write(tmp_path, doc)
    (entry,) = latents.load(tmp_path)
    assert entry["shape"] == [1, 4, 64, 64] and json.loads((tmp_path / latents.LATENTS_NAME).read_text())
    (tmp_path / entry["raw_file_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        latents.load(tmp_path)


def test_a_frame_beyond_the_image_is_unreadable_not_short():
    from comfytrace.procmem import Unreadable

    mem = ProcessMemory(5, [(BASE, 0x2000, 0x1000)], b"\0" * 0x1800, vads=[(BASE, BASE + 0x1FFF, None)])
    with pytest.raises(Unreadable):
        mem.read(BASE, 0x2000)
    assert mem.try_read(BASE + 0x1000, 0x1000) is None
    assert latents.find(mem, [(64, 64)]) == []
