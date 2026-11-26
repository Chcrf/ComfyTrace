"""Pre-freeze audit of the replay pipeline: model identity, latents, decode resume and replay-case orchestration.

Each test pins a defect found by reading the modules and reproducing it before the fix."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from comfytrace import latentdecode, latents, modelid
from comfytrace.exceptions import RegenerationError
from comfytrace.procmem import ProcessMemory
from comfytrace.regenerate import load_workflow_file

STEP = 512
PRIVATE, MAPPED = 0x10000000, 0x20000000


def _weights(seed: int, n: int = 16384) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * 0.05).astype(np.float16)


def _checkpoint(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": "F16", "shape": [len(array)], "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return path


def _memory(private: bytes, pad: int = 0) -> ProcessMemory:
    noise = np.random.default_rng(99).integers(0, 256, 8192, np.uint8).tobytes()
    private = noise[:24] + b"\0" * pad + private + noise
    mapped = b"\0" * 4096
    private += b"\0" * (-len(private) % 4096)
    return ProcessMemory(7, [(PRIVATE, len(private), 0), (MAPPED, len(mapped), len(private))], private + mapped,
                         vads=[(PRIVATE, PRIVATE + len(private) - 1, None),
                               (MAPPED, MAPPED + len(mapped) - 1, r"\models\m.safetensors")])


def _assess(tmp_path: Path, tensors: dict[str, np.ndarray], mem: ProcessMemory) -> dict:
    path = _checkpoint(tmp_path / "m.safetensors", tensors)
    return modelid.assess(modelid.Fingerprint.of("m.safetensors", path, step=STEP), [mem])


# --- modelid: a widened (FP16/BF16 -> FP32) window is 2 * WINDOW bytes in memory, and all of it must be checked ---


def test_a_widened_window_split_across_vads_is_not_counted(tmp_path):
    unet = _weights(3)
    mem = _memory(unet.astype(np.float32).tobytes())
    full = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)["components"]["unet"]["private_windows"]
    start, end, name = mem.vads[0]
    # FP32 copies of the sampled windows sit 2 * STEP apart and are 2 * WINDOW long. Cut inside the fourth one's tail.
    cut = PRIVATE + 24 + 2 * STEP * 3 + modelid.WINDOW + modelid.WINDOW // 2
    mem.vads = sorted([(start, cut - 1, name), (cut, end, name), mem.vads[1]])
    mem._vad_starts = [v[0] for v in mem.vads]
    split = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)["components"]["unet"]["private_windows"]
    assert split == full - 1


def test_a_widened_window_whose_tail_lies_on_a_disputed_page_is_not_counted(tmp_path):
    unet = _weights(14)
    # 936 bytes of padding put the fourth FP32 window at PRIVATE + 4032: its first half ends exactly at the page
    # boundary and its second half is on the next page, which is disputed.
    mem = _memory(unet.astype(np.float32).tobytes(), pad=936)
    mem.stale = {PRIVATE + 4096: "PFN_PTE_MISMATCH"}
    result = _assess(tmp_path, {"model.diffusion_model.w": unet}, mem)
    addresses = {int(d["address"], 16) for d in result["pfn_disputed_match_details"]}
    assert PRIVATE + 4032 in addresses


# --- modelid: one unreadable checkpoint must not take the identity of the others with it ---


def _identity_case(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "case"
    (case / "exports").mkdir(parents=True)
    graph = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "good.safetensors"}},
             "5": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "bad.safetensors"}}}
    (case / "exports" / "wf.json").write_text(json.dumps(graph))
    (case / "manifest.json").write_text(json.dumps({"findings": {"workflows": [
        {"prompt_format": "API_GRAPH", "encoding_detected": "CPYTHON_OBJECTS", "raw_file_path": "exports/wf.json"}]}}))
    models = tmp_path / "models"
    models.mkdir()
    return case, models


def _truncated(models: Path, unet: np.ndarray) -> bytes:
    whole = _checkpoint(models / "whole.bin", {"model.diffusion_model.w": unet}).read_bytes()
    (models / "whole.bin").unlink()
    return whole[: len(whole) // 2]


@pytest.mark.parametrize("damage", ["empty", "header_longer_than_file", "not_json", "header_is_a_list",
                                    "tensor_data_cut_short"])
def test_an_unreadable_checkpoint_gets_its_own_verdict_and_the_others_are_still_assessed(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(modelid, "STEP", STEP)
    unet = _weights(21)
    case, models = _identity_case(tmp_path)
    _checkpoint(models / "good.safetensors", {"model.diffusion_model.w": unet})
    bad = {"empty": b"",
           "header_longer_than_file": struct.pack("<Q", 10 ** 6) + b"{",
           "not_json": struct.pack("<Q", 5) + b"zzzzz",
           "header_is_a_list": struct.pack("<Q", 2) + b"[]",
           "tensor_data_cut_short": _truncated(models, unet)}[damage]
    (models / "bad.safetensors").write_bytes(bad)
    document = modelid.identity_document(case, [_memory(unet.tobytes())], models, "00" * 32)
    verdicts = {m["name"]: m for m in document["models"]}
    assert verdicts["good.safetensors"]["verdict"] == modelid.OBSERVED
    assert verdicts["bad.safetensors"]["verdict"] == "UNSUPPORTED_FORMAT" and verdicts["bad.safetensors"]["reason"]


# --- latents ---


def test_a_one_column_shape_is_refused_not_a_crash():
    values = (np.random.default_rng(5).standard_normal(latents.CHANNELS * 64) * 4).astype(np.float32)
    assert latents.assess(values.tobytes(), 64, 1) is None


# --- latentdecode: a record that parses but is not a record proves nothing, like an unreadable one ---


def test_a_decode_record_that_is_not_a_record_does_not_stop_the_resume(tmp_path):
    out = tmp_path / latentdecode.DECODES_DIR
    out.mkdir()
    model = {"name": "m.safetensors", "sha256": "ab" * 32}
    good = {"status": "COMPLETED", "latent_id": "LATENT-001", "latent_sha256": "cd" * 32, "checkpoint": model}
    (out / "LATDEC-001.json").write_text("[]")
    (out / "LATDEC-002.json").write_text(json.dumps({"status": "COMPLETED", "latent_id": "LATENT-001",
                                                     "latent_sha256": "cd" * 32, "checkpoint": {}}))
    (out / "LATDEC-003.json").write_text(json.dumps(good))
    done, highest = latentdecode.completed_pairs(out)
    assert done == {("LATENT-001", "cd" * 32, "m.safetensors", "ab" * 32)} and highest == 3


# --- regenerate: a UTF-8 workflow file with a byte-order mark (PowerShell 5.1 -Encoding utf8) ---


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le"])
def test_a_workflow_file_is_read_in_every_encoding_a_windows_editor_writes(tmp_path, encoding):
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    path = tmp_path / "wf.json"
    path.write_bytes(json.dumps(graph).encode(encoding))
    assert load_workflow_file(path)[:2] == ("API_GRAPH", graph)


def test_a_file_that_is_no_graph_is_still_refused(tmp_path):
    path = tmp_path / "wf.json"
    path.write_bytes(codecs.BOM_UTF8 + b"[1, 2]")
    with pytest.raises(RegenerationError):
        load_workflow_file(path)


# --- replay-case: a failure of the optional decode stage must not cost the link and the report ---


def test_a_decode_stage_that_raises_does_not_stop_linking_and_the_report(tmp_path, monkeypatch):
    pytest.importorskip("ahocorasick_rs")
    from comfytrace import link, replaycase
    from comfytrace.cli import parse_args

    memory = tmp_path / "dump.raw"
    memory.write_bytes(b"memory image")
    case_dir = tmp_path / "case"
    (case_dir / "exports/raw").mkdir(parents=True)
    (case_dir / "exports/raw/WFPY-001.json").write_text("{}")
    (case_dir / "manifest.json").write_text(json.dumps({
        "case_metadata": {"memory_image": {"sha256": hashlib.sha256(b"memory image").hexdigest()}},
        "findings": {"artifacts": [], "findings": [], "workflows": [{
            "workflow_id": "WFPY-001", "encoding_detected": "CPYTHON_OBJECTS", "prompt_format": "API_GRAPH",
            "completeness": "COMPLETE", "canonical_json_sha256": "a" * 64, "raw_file_path": "exports/raw/WFPY-001.json",
            "execution_parameters": {"prompt_id": "job-a"}}]}}))
    (case_dir / latents.LATENTS_NAME).write_text(json.dumps({"latents": [{"latent_id": "LATENT-001"}]}))
    calls: list[str] = []

    def regenerate(args):
        (case_dir / "regenerations").mkdir(exist_ok=True)
        (case_dir / "regenerations/REGEN-001.json").write_text(json.dumps({"status": "COMPLETED"}))
        return 0

    def decode(args):
        raise ValueError("checkpoint 'm.safetensors' has no contained local copy under --models-dir")

    def write_links(case_dir_):
        calls.append("link")
        (case_dir_ / link.LINKS_NAME).write_text("{}")

    monkeypatch.setattr(replaycase, "resolve_runner", lambda args: "http://runner")
    monkeypatch.setattr(replaycase, "run_regeneration", regenerate)
    monkeypatch.setattr(replaycase, "search", lambda *a, **k: calls.append("search-replay"))
    monkeypatch.setattr(replaycase, "HtmlReporter", lambda d: SimpleNamespace(
        render_report=lambda path: calls.append("report") or d / "workbench.html"))
    monkeypatch.setattr(latentdecode, "run_decode", decode)
    monkeypatch.setattr(link, "write_links", write_links)
    args = parse_args(["replay-case", "--memory", str(memory), "--case-dir", str(case_dir), "--runner-url",
                       "http://runner", "--latents", "--symbols-dir", str(tmp_path), "--models-dir", str(tmp_path)])
    assert isinstance(args, argparse.Namespace)
    assert replaycase.run_replay_case(args, lambda *_: SimpleNamespace(
        recovery=SimpleNamespace(python_processes=[]), close=lambda: None)) == 1, "an incomplete decode fails the run"
    assert calls == ["search-replay", "link", "report"]
