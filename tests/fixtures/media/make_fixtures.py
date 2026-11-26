"""Writes the media fixtures the way ComfyUI's save nodes do (Pillow EXIF for WEBP; PyAV container metadata for
video and audio). Needs ``pip install av``; the fixtures are committed, so tests do not need it.

    python tests/fixtures/media/make_fixtures.py
"""

import io
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
PROMPT = {"3": {"class_type": "KSampler", "inputs": {"seed": 4242, "steps": 20, "cfg": 8.0, "sampler_name": "euler",
                                                     "scheduler": "normal", "denoise": 1.0, "model": ["4", 0],
                                                     "positive": ["6", 0], "negative": ["7", 0],
                                                     "latent_image": ["5", 0]}},
          "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
          "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
          "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lighthouse at dusk", "clip": ["4", 1]}},
          "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
          "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
          "9": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}}}
WORKFLOW = {"last_node_id": 9, "last_link_id": 9, "nodes": [{"id": 3, "type": "KSampler", "pos": [0, 0],
                                                             "widgets_values": [4242, "fixed", 20, 8.0, "euler",
                                                                                "normal", 1.0]}],
            "links": [], "version": 0.4}


def frames(count: int, size: int = 64) -> list[np.ndarray]:
    y, x = np.mgrid[0:size, 0:size]
    return [np.stack([(x * 4 + 9 * i) % 256, (y * 4) % 256, ((x + y) * 2 + 5 * i) % 256], -1).astype(np.uint8)
            for i in range(count)]


def webp(animated: bool) -> bytes:
    """comfy_extras/nodes_images.py SaveAnimatedWEBP: prompt in EXIF 0x0110, extra_pnginfo from 0x010F down."""
    images = [Image.fromarray(f) for f in frames(3 if animated else 1)]
    exif = images[0].getexif()
    exif[0x0110] = f"prompt:{json.dumps(PROMPT)}"
    exif[0x010F] = f"workflow:{json.dumps(WORKFLOW)}"
    out = io.BytesIO()
    images[0].save(out, "WEBP", save_all=True, duration=125, append_images=images[1:], exif=exif, lossless=True,
                   quality=80, method=4)
    return out.getvalue()


def video(fmt: str, codec: str, options: dict[str, str]) -> bytes:
    """comfy_api VideoFromComponents.save_to (mp4, movflags use_metadata_tags) and SaveWEBM (vp9)."""
    out = io.BytesIO()
    with av.open(out, mode="w", format=fmt, options=options) as container:
        container.metadata["prompt"] = json.dumps(PROMPT)
        container.metadata["workflow"] = json.dumps(WORKFLOW)
        stream = container.add_stream(codec, rate=Fraction(8, 1))
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for frame in frames(6):
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        container.mux(stream.encode(None))
    return out.getvalue()


def flac() -> bytes:
    """comfy_extras/nodes_audio.py SaveAudio: metadata set on the output container."""
    out = io.BytesIO()
    with av.open(out, mode="w", format="flac") as container:
        container.metadata["prompt"] = json.dumps(PROMPT)
        container.metadata["workflow"] = json.dumps(WORKFLOW)
        stream = container.add_stream("flac", rate=44100, layout="stereo")
        t = np.arange(44100 // 4) / 44100
        wave = (np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]) * 0.3).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(wave, format="fltp", layout="stereo")
        frame.sample_rate = 44100
        resampler = av.AudioResampler(format=stream.format, layout="stereo", rate=44100)
        for part in resampler.resample(frame):
            for packet in stream.encode(part):
                container.mux(packet)
        container.mux(stream.encode(None))
    return out.getvalue()


if __name__ == "__main__":
    for name, data in (("still.webp", webp(False)), ("animated.webp", webp(True)),
                       ("video.mp4", video("mp4", "libx264", {"movflags": "use_metadata_tags"})),
                       ("video.webm", video("webm", "libvpx-vp9", {})), ("audio.flac", flac())):
        (HERE / name).write_bytes(data)
        print(name, len(data))
