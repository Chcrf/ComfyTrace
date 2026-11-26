"""Decoding latents found in memory: what is sent to the runner, what is refused, what is recorded."""

import argparse
import hashlib
import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from comfytrace import latentdecode, latents

MEMORY_SHA = "cd" * 32


class FakeRunner:
    sent: list = []
    uploads: dict = {}

    def __init__(self, url, timeout=300):
        pass

    def object_info(self):
        return {n: {"input": {"required": {"ckpt_name": [["m.safetensors"]]}}} for n in latentdecode.NODES}

    def environment(self):
        return {"comfyui_version": "test"}

    def upload_input(self, path, name):
        FakeRunner.uploads[name] = path.read_bytes()
        return name

    def queue_prompt(self, graph):
        FakeRunner.sent.append(graph)
        return "p1"

    def retrieve_output_rasters(self, prompt_id, nodes):
        out = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), 7, np.uint8)).save(out, "PNG")
        return [({"node_id": "4", "filename": "x.png"}, out.getvalue())]


def _case(tmp_path, verdict="OBSERVED"):
    case = tmp_path / "case"
    (case / "exports").mkdir(parents=True)
    graph = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}}}
    (case / "exports" / "wf.json").write_text(json.dumps(graph))
    (case / "manifest.json").write_text(json.dumps({
        "case_metadata": {"memory_image": {"sha256": MEMORY_SHA}},
        "findings": {"workflows": [{"prompt_format": "API_GRAPH", "encoding_detected": "CPYTHON_OBJECTS",
                                    "raw_file_path": "exports/wf.json"}]}}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.safetensors").write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    (case / "model-identity.json").write_text(json.dumps({"memory_sha256": MEMORY_SHA, "models": [
        {"name": "m.safetensors", "sha256": digest, "verdict": verdict,
         "components": {"vae": {"private_windows": 9}}}]}))
    values = np.arange(4 * 64 * 64, dtype=np.float32) / 1000
    raw = bytearray(values.tobytes())
    raw[:8] = struct.pack("<Q", 0x20000)
    block = latents.LatentBlock(5, 0x20000, 64, 64, bytes(raw), [0x1000], "POINTER", {})
    latents.write(case, latents.document(case, [block], MEMORY_SHA, [(64, 64)]))
    return case, models


def _args(case, models):
    return argparse.Namespace(case_dir=case, runner_url="http://runner", models_dir=models, timeout=5)


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch):
    FakeRunner.sent, FakeRunner.uploads = [], {}
    monkeypatch.setattr(latentdecode, "RunnerClient", FakeRunner)


def test_latent_is_sent_as_found_and_the_decode_is_recorded(tmp_path):
    case, models = _case(tmp_path)
    assert latentdecode.run_decode(_args(case, models)) == 0
    (payload,) = FakeRunner.uploads.values()
    n = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8:8 + n])
    assert header["latent_tensor"]["shape"] == [1, 4, 64, 64] and "latent_format_version_0" in header
    sent = np.frombuffer(payload[8 + n:], np.float32)
    expected = np.arange(4 * 64 * 64, dtype=np.float32) / 1000
    expected[:2] = expected[2]
    np.testing.assert_array_equal(sent, expected)
    (graph,) = FakeRunner.sent
    assert graph["2"] == {"class_type": "LoadLatent", "inputs": {"latent": "comfytrace_LATENT-001.latent"}}
    record = json.loads((case / "latent-decodes" / "LATDEC-001.json").read_text())
    assert record["status"] == "COMPLETED" and record["origin"] == "DECODED_FROM_MEMORY_LATENT"
    assert record["checkpoint"]["model_match_level"] == "OBSERVED_IN_MEMORY"
    image = (case / record["derived_image"]).read_bytes()
    assert hashlib.sha256(image).hexdigest() == record["derived_image_sha256"]
    with pytest.raises(FileExistsError):
        latentdecode.run_decode(_args(case, models))


def test_a_shared_vae_alone_leaves_the_model_hashed_and_another_dump_is_refused(tmp_path):
    case, models = _case(tmp_path, verdict="SHARED_COMPONENTS_ONLY")
    assert latentdecode.run_decode(_args(case, models)) == 0
    record = json.loads((case / "latent-decodes" / "LATDEC-001.json").read_text())
    assert record["checkpoint"]["model_match_level"] == "HASHED"
    other, models = _case(tmp_path / "b")
    doc = json.loads((other / latents.LATENTS_NAME).read_text())
    doc["memory_sha256"] = "ef" * 32
    (other / latents.LATENTS_NAME).write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        latentdecode.run_decode(_args(other, models))


def test_a_pickle_checkpoint_is_never_sent_to_the_runner(tmp_path):
    case, models = _case(tmp_path)
    graph = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.ckpt"}}}
    (case / "exports" / "wf.json").write_text(json.dumps(graph))
    (models / "m.ckpt").write_bytes(b"pickle")
    assert latentdecode.run_decode(_args(case, models)) == 1
    assert not FakeRunner.sent and not (case / "latent-decodes").exists()


def test_an_interrupted_run_resumes_only_the_missing_decodes(tmp_path, monkeypatch):
    case, models = _case(tmp_path)
    out = case / latentdecode.DECODES_DIR

    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeRunner, "retrieve_output_rasters", interrupted)
    with pytest.raises(KeyboardInterrupt):
        latentdecode.run_decode(_args(case, models))
    assert out.is_dir() and not list(out.glob("LATDEC-*.json")), "the run stopped before any record"
    assert not latentdecode.decodes_complete(case)
    monkeypatch.undo()
    monkeypatch.setattr(latentdecode, "RunnerClient", FakeRunner)
    assert latentdecode.run_decode(_args(case, models)) == 0, "an empty directory is resumed, not refused"
    record = json.loads((out / "LATDEC-001.json").read_text())
    assert record["status"] == "COMPLETED" and latentdecode.decodes_complete(case)
    # A completed pair is never decoded twice: with the marker gone, a resume has nothing left to do.
    (out / latentdecode.COMPLETE_NAME).unlink()
    FakeRunner.sent.clear()
    assert latentdecode.run_decode(_args(case, models)) == 0
    assert not FakeRunner.sent and sorted(p.name for p in out.glob("LATDEC-*.json")) == ["LATDEC-001.json"]
    assert latentdecode.decodes_complete(case)


def test_failed_decode_writes_a_receipt_and_returns_failure(tmp_path, monkeypatch):
    case, models = _case(tmp_path)

    def runner_failed(*_args):
        raise RuntimeError("test runner produced no raster")

    monkeypatch.setattr(FakeRunner, "retrieve_output_rasters", runner_failed)
    assert latentdecode.run_decode(_args(case, models)) == 1
    record = json.loads((case / "latent-decodes/LATDEC-001.json").read_bytes())
    assert record["status"] == "FAILED" and "derived_image" not in record
