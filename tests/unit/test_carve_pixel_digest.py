"""Full-sample pixel identity for 16-bit PNG (ComfyUI SaveImageAdvanced) and the APNG default-image scope."""

import hashlib
import io
import random
import struct
import zlib

import pytest
from PIL import Image

from comfytrace.carve import PixelDigestUnsupported, PngCarver, native_pixel_digest

CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _filtered(raw: bytes, prev: bytes, bpp: int, ftype: int) -> bytes:
    out = bytearray()
    for i, x in enumerate(raw):
        a = raw[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ftype == 0:
            p = 0
        elif ftype == 1:
            p = a
        elif ftype == 2:
            p = b
        elif ftype == 3:
            p = (a + b) >> 1
        else:
            pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
            p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        out.append((x - p) & 0xFF)
    return bytes([ftype]) + bytes(out)


def png16(width: int, height: int, ctype: int, samples: list[int], filters: tuple[int, ...] = (0,),
          interlace: int = 0, extra: list[tuple[bytes, bytes]] | None = None) -> bytes:
    """A valid 16-bit PNG written by hand (Pillow cannot write 16-bit RGB/RGBA)."""
    ch = CHANNELS[ctype]
    stride = width * ch * 2
    rows = [struct.pack(f">{width * ch}H", *samples[r * width * ch:(r + 1) * width * ch]) for r in range(height)]
    prev = bytes(stride)
    raw = b""
    for r, row in enumerate(rows):
        raw += _filtered(row, prev, ch * 2, filters[r % len(filters)])
        prev = row
    head = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, ctype, 0, 0, interlace))
    mid = b"".join(_chunk(t, b) for t, b in (extra or []))
    return (b"\x89PNG\r\n\x1a\n" + head + mid + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def _samples(n: int, seed: int = 1, mask: int = 0xFFFF) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 65536) & mask for _ in range(n)]


def _pillow_native_sha(png: bytes) -> str:
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        return hashlib.sha256(img.tobytes()).hexdigest()


@pytest.mark.parametrize("ctype", [2, 4, 6])
def test_low_byte_only_difference_is_not_collapsed(ctype):
    n = 5 * 3 * CHANNELS[ctype]
    high = _samples(n, mask=0xFF00)
    a = png16(5, 3, ctype, high)
    b = png16(5, 3, ctype, [s | (i % 251 + 1) for i, s in enumerate(high)])
    # Demonstrates the defect: Pillow's native bytes cannot tell these two files apart.
    assert _pillow_native_sha(a) == _pillow_native_sha(b)
    digest_a, meta_a = native_pixel_digest(a)
    digest_b, _ = native_pixel_digest(b)
    assert digest_a != digest_b
    assert meta_a["bit_depth"] == 16 and meta_a["pixel_digest"] == "png-samples-be16-v1"
    assert (meta_a["width"], meta_a["height"]) == (5, 3)


@pytest.mark.parametrize("ctype", [2, 4, 6])
def test_digest_is_sha256_of_the_big_endian_samples_for_every_filter(ctype):
    samples = _samples(7 * 6 * CHANNELS[ctype], seed=ctype)
    png = png16(7, 6, ctype, samples, filters=(0, 1, 2, 3, 4, 4))
    digest, _ = native_pixel_digest(png)
    assert digest == hashlib.sha256(struct.pack(f">{len(samples)}H", *samples)).hexdigest()


def test_carver_keeps_container_and_full_sample_identity_for_16bit_rgba():
    high = _samples(4 * 4 * 4, mask=0xFF00)
    a = png16(4, 4, 6, high)
    b = png16(4, 4, 6, [s | 1 for s in high])
    [rec_a] = PngCarver().carve_stream(b"\x00" * 64 + a)
    [rec_b] = PngCarver().carve_stream(b + b"\x00" * 64)
    assert rec_a.completeness == rec_b.completeness == "COMPLETE"
    assert rec_a.container_sha256 == hashlib.sha256(a).hexdigest()
    assert rec_a.pixel_sha256 != rec_b.pixel_sha256
    assert rec_a.spans == [(0, a)]


def test_unsupported_variant_abstains_but_keeps_the_container():
    # A 1x1 image has a single Adam7 pass, so its IDAT is identical to the non-interlaced stream.
    png = png16(1, 1, 2, [0x1234, 0x5678, 0x9ABC], interlace=1)
    with pytest.raises(PixelDigestUnsupported) as info:
        native_pixel_digest(png)
    assert isinstance(info.value, ValueError)
    [record] = PngCarver().carve_stream(png)
    assert record.completeness == "COMPLETE"
    assert record.container_sha256 == hashlib.sha256(png).hexdigest()
    assert record.spans == [(0, png)] and record.recovered_bytes == len(png)
    assert record.pixel_sha256 is None
    assert record.pixel_metadata["pixel_digest"] == "UNSUPPORTED"
    assert record.pixel_metadata["pixel_digest_unsupported_reason"] == "ADAM7_INTERLACED_16BIT"
    assert (record.pixel_metadata["width"], record.pixel_metadata["height"]) == (1, 1)


def test_corrupt_idat_is_not_reclassified_as_a_digest_abstention():
    png = bytearray(png16(4, 4, 6, _samples(64)))
    png[png.index(b"IDAT") + 10] ^= 0xFF
    [record] = PngCarver().carve_stream(bytes(png))
    assert record.completeness == "PARTIAL" and record.pixel_sha256 is None


def test_legacy_8bit_and_grey16_digests_are_unchanged():
    for mode in ("RGB", "RGBA", "L", "LA"):
        img = Image.effect_noise((17, 9), 80).convert(mode)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        digest, meta = native_pixel_digest(buf.getvalue())
        assert digest == hashlib.sha256(img.tobytes()).hexdigest()
        assert meta == {"mode": mode, "width": 17, "height": 9}
    grey = png16(3, 2, 0, _samples(6))
    digest, meta = native_pixel_digest(grey)
    assert meta == {"mode": "I;16", "width": 3, "height": 2}
    assert digest == _pillow_native_sha(grey)


def test_palette_color_and_transparency_contribute_to_the_pixel_digest():
    image = Image.new("P", (4, 4), 0)
    a, b, c = io.BytesIO(), io.BytesIO(), io.BytesIO()
    image.putpalette([0, 0, 0] * 256)
    image.save(a, format="PNG")
    image.putpalette([10, 0, 0] + [0, 0, 0] * 255)
    image.save(b, format="PNG")
    image.save(c, format="PNG", transparency=0)
    assert len({_pillow_native_sha(buf.getvalue()) for buf in (a, b, c)}) == 1
    assert len({native_pixel_digest(buf.getvalue())[0] for buf in (a, b, c)}) == 3


def test_apng_retains_full_container_identity_and_abstains_from_pixel_identity():
    frames = [Image.new("RGB", (4, 4), colour) for colour in ((10, 0, 0), (0, 200, 0), (0, 0, 99))]
    animated, static = io.BytesIO(), io.BytesIO()
    frames[0].save(animated, format="PNG", save_all=True, append_images=frames[1:])
    frames[0].save(static, format="PNG")
    with pytest.raises(PixelDigestUnsupported, match="ANIMATED_PNG_REQUIRES_CONTAINER_IDENTITY"):
        native_pixel_digest(animated.getvalue())
    assert "animated" not in native_pixel_digest(static.getvalue())[1]
    [record] = PngCarver().carve_stream(animated.getvalue())
    assert record.completeness == "COMPLETE" and record.pixel_sha256 is None
    assert record.pixel_metadata["animated"] is True and record.pixel_metadata["frames"] == 3
    assert record.container_sha256 == hashlib.sha256(animated.getvalue()).hexdigest()


def test_replay_exact_match_checks_16bit_samples_and_refuses_partial_animation_comparison():
    from comfytrace.regenerate import TieredEvaluator

    evaluator = TieredEvaluator()
    high = _samples(4 * 4 * 3, mask=0xFF00)
    a = png16(4, 4, 2, high)
    b = png16(4, 4, 2, [s | 1 for s in high])
    [target] = PngCarver().carve_stream(a)
    assert evaluator.evaluate_reproduction(a, target, a)[:2] == ("E1", "EXACT_MATCH")
    _, matched, metrics = evaluator.evaluate_reproduction(b, target, a)
    assert matched != "EXACT_MATCH" and metrics["exact_pixel_match"] is False

    frames = [Image.new("RGB", (4, 4), color) for color in ("black", "white")]
    animated = io.BytesIO()
    frames[0].save(animated, format="PNG", save_all=True, append_images=frames[1:])
    data = animated.getvalue()
    [target] = PngCarver().carve_stream(data)
    assert evaluator.evaluate_reproduction(data, target, data)[:2] == (
        "NON_MATCHING", "ANIMATED_COMPARISON_UNAVAILABLE",
    )


def test_reduced_precision_preview_rows_do_not_get_native_exact_credit():
    from comfytrace.regenerate import TieredEvaluator

    png = png16(8, 8, 2, _samples(8 * 8 * 3))
    [target] = PngCarver().carve_stream(png[:-12])
    assert target.completeness == "PARTIAL" and target.derived_preview_metadata["sample_precision_reduced"] is True
    _, subtier, metrics = TieredEvaluator().evaluate_reproduction(png, target, target.preview_png)
    assert subtier != "ROW_SLICE_EXACT" and metrics["row_slice_pixel_match"] is False
