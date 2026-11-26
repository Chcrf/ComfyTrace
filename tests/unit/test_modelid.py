"""Model identity from process memory: which local weights are present, in which holder."""

import json
import struct

import numpy as np

from comfytrace import modelid
from comfytrace.procmem import ProcessMemory

STEP = 512
PRIVATE, MAPPED = 0x10000000, 0x20000000


def _weights(seed: int, n: int = 16384) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.05).astype(np.float16)


def _checkpoint(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": "F16", "shape": [len(array)], "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return path


def _memory(private: bytes, mapped: bytes = b"") -> ProcessMemory:
    noise = np.random.default_rng(99).integers(0, 256, 8192, np.uint8).tobytes()
    private = noise[:24] + private + noise
    pad = lambda b: b + b"\0" * (-len(b) % 4096)  # noqa: E731
    private, mapped = pad(private), pad(mapped or b"\0")
    physical = private + mapped
    return ProcessMemory(7, [(PRIVATE, len(private), 0), (MAPPED, len(mapped), len(private))], physical,
                         vads=[(PRIVATE, PRIVATE + len(private) - 1, None),
                               (MAPPED, MAPPED + len(mapped) - 1, r"\models\checkpoints\m.safetensors")])


def _assess(tmp_path, tensors, mem):
    path = _checkpoint(tmp_path / "m.safetensors", tensors)
    return modelid.assess(modelid.Fingerprint.of("m.safetensors", path, step=STEP), [mem])


def test_denoiser_in_private_memory_is_observed_and_file_views_are_not_counted(tmp_path):
    unet, vae = _weights(1), _weights(2)
    result = _assess(tmp_path, {"model.diffusion_model.w": unet, "first_stage_model.w": vae},
                     _memory(unet.tobytes(), mapped=vae.tobytes()))
    assert result["verdict"] == modelid.OBSERVED and result["control_windows"] == 0
    assert result["components"]["unet"]["private_windows"] == result["components"]["unet"]["windows_sampled"]
    assert result["components"]["vae"]["private_windows"] == 0
    assert result["components"]["vae"]["mapped_file_windows_not_counted"] > 0


def test_weights_widened_to_fp32_are_observed(tmp_path):
    unet = _weights(3)
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, _memory(unet.astype(np.float32).tobytes()))
    assert result["verdict"] == modelid.OBSERVED


def test_a_different_denoiser_with_a_shared_vae_is_not_an_identity(tmp_path):
    vae = _weights(4)
    other_unet = _weights(5)
    result = _assess(tmp_path, {"model.diffusion_model.w": _weights(6), "first_stage_model.w": vae},
                     _memory(other_unet.tobytes() + vae.tobytes()))
    assert result["verdict"] == modelid.SHARED_ONLY
    assert result["components"]["unet"]["private_windows"] == 0


def test_bit_flipped_weights_trip_the_control(tmp_path):
    unet = _weights(7)
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, _memory((unet.view(np.uint16) ^ 1).tobytes()))
    assert result["verdict"] == modelid.CONTROL_FAILED


def test_absent_weights_are_not_observed(tmp_path):
    result = _assess(tmp_path, {"model.diffusion_model.w": _weights(8)}, _memory(_weights(9).tobytes()))
    assert result["verdict"] == modelid.NOT_OBSERVED and result["control_windows"] == 0


def test_windows_on_pfn_disputed_pages_cannot_establish_identity(tmp_path):
    unet = _weights(11)
    mem = _memory(unet.tobytes())
    # Every page that holds the denoiser's bytes is disputed by its PFN entry.
    mem.stale = {PRIVATE + p: "PFN_PTE_MISMATCH" for p in range(0, 9 * 4096, 4096)}
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)
    unet_counts = result["components"]["unet"]
    assert result["verdict"] == modelid.NOT_OBSERVED
    assert unet_counts["private_windows"] == 0
    assert unet_counts["pfn_disputed_windows_not_counted"] == unet_counts["windows_sampled"]
    assert result["pfn_disputed_matches"] == len(result["pfn_disputed_match_details"]) > 0
    detail = result["pfn_disputed_match_details"][0]
    assert detail["process_id"] == 7 and detail["pages"][0]["reason"] == "PFN_PTE_MISMATCH"


def test_one_disputed_page_inside_a_window_disqualifies_it(tmp_path):
    unet = _weights(12)
    clean = _assess(tmp_path, {"model.diffusion_model.w": unet}, _memory(unet.tobytes()))
    mem = _memory(unet.tobytes())
    # Only the second page is disputed: the windows on it are listed, not counted; the rest still count.
    mem.stale = {PRIVATE + 4096: "PFN_PTE_MISMATCH"}
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)
    counts = result["components"]["unet"]
    assert counts["pfn_disputed_windows_not_counted"] > 0
    assert counts["private_windows"] + counts["pfn_disputed_windows_not_counted"] >= \
        clean["components"]["unet"]["private_windows"]
    assert counts["private_windows"] < clean["components"]["unet"]["private_windows"]
    addresses = [int(d["address"], 16) for d in result["pfn_disputed_match_details"]]
    assert all(a < PRIVATE + 2 * 4096 and a + modelid.WINDOW > PRIVATE + 4096 for a in addresses)


def test_a_window_whose_tail_lies_on_a_disputed_page_is_not_counted(tmp_path):
    unet = _weights(14)
    # 456 bytes of padding put the eighth sampled window at PRIVATE + 4064: it begins on a clean page and ends
    # on the next one, which is disputed.
    mem = _memory(b"\0" * 456 + unet.tobytes())
    mem.stale = {PRIVATE + 4096: "PFN_PTE_MISMATCH"}
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)
    addresses = {int(d["address"], 16) for d in result["pfn_disputed_match_details"]}
    assert PRIVATE + 4064 in addresses


def test_a_window_running_past_its_vad_is_not_counted(tmp_path):
    unet = _weights(13)
    mem = _memory(unet.tobytes())
    full = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)["components"]["unet"]["private_windows"]
    # Split the private VAD inside the denoiser's bytes: a window straddling the split is in no single VAD.
    start, end, name = mem.vads[0]
    cut = PRIVATE + 24 + STEP * 3 + modelid.WINDOW // 2
    mem.vads = sorted([(start, cut - 1, name), (cut, end, name), mem.vads[1]])
    mem._vad_starts = [v[0] for v in mem.vads]
    split = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)["components"]["unet"]["private_windows"]
    assert split == full - 1


def test_command_checks_the_image_and_reads_model_names_from_recovered_prompts(tmp_path, monkeypatch):
    import argparse
    import hashlib
    from types import SimpleNamespace

    from comfytrace import cli

    unet = _weights(10)
    models = tmp_path / "models"
    models.mkdir()
    _checkpoint(models / "m.safetensors", {"model.diffusion_model.w": unet})
    case = tmp_path / "case"
    (case / "exports").mkdir(parents=True)
    graph = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}}}
    (case / "exports" / "wf.json").write_text(json.dumps(graph))
    image = tmp_path / "dump.raw"
    image.write_bytes(b"image")
    manifest = {"case_metadata": {"memory_image": {"sha256": hashlib.sha256(b"image").hexdigest()}},
                "findings": {"workflows": [{"prompt_format": "API_GRAPH", "encoding_detected": "CPYTHON_OBJECTS",
                                            "raw_file_path": "exports/wf.json"}]}}
    (case / "manifest.json").write_text(json.dumps(manifest))
    mem = _memory(unet.tobytes())
    ingestor = lambda *_: SimpleNamespace(recovery=SimpleNamespace(python_processes=[mem]),  # noqa: E731
                                          close=lambda: None)
    args = argparse.Namespace(case_dir=case, memory=image, symbols_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(modelid, "STEP", STEP)
    assert cli.run_model_id(args, ingestor) == 0
    (document,) = json.loads((case / modelid.IDENTITY_NAME).read_text())["models"]
    assert document["name"] == "m.safetensors" and document["verdict"] == modelid.OBSERVED
    assert cli.run_model_id(args, ingestor) == 1, "model identity is written once"
    (case / modelid.IDENTITY_NAME).unlink()
    image.write_bytes(b"another image")
    assert cli.run_model_id(args, ingestor) == 1 and not (case / modelid.IDENTITY_NAME).exists()
