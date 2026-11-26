"""Decode latents found in memory with the VAE of each checkpoint the recovered prompts name.

The decode uses a ComfyUI runner. ComfyTrace hashes model files under --models-dir and copies the model's
memory identity from model-identity.json of this dump. The graph is fixed:
CheckpointLoaderSimple -> LoadLatent -> VAEDecode -> SaveImage. The latent is sent as a ``.latent`` file holding
the bytes found in memory (free-list link slots restored as recorded by ``latents``).

Each decode writes ``latent-decodes/LATDEC-NNN.json`` and ``exports/derived/DERIVED_latent_<id>.png``. The image
is DECODED_FROM_MEMORY_LATENT: the latent was observed, the pixels were computed by the examiner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from comfytrace import latents
from comfytrace.casefiles import EvidentiaryExporter, sha256_file
from comfytrace.localrunner import resolve_runner
from comfytrace.modelid import model_names
from comfytrace.regenerate import RunnerClient, _contained_path, _model_identity, _registered_choices, _relative_parts

log = logging.getLogger(__name__)

DECODES_DIR = "latent-decodes"
COMPLETE_NAME = "COMPLETE.json"  # written after the last pair; its absence means the run did not finish
ORIGIN = "DECODED_FROM_MEMORY_LATENT"
NODES = ("CheckpointLoaderSimple", "LoadLatent", "VAEDecode", "SaveImage")


def latent_file(values: np.ndarray) -> bytes:
    """A safetensors file as ComfyUI's SaveLatent writes it: ``latent_tensor`` plus the format marker, so
    LoadLatent applies no scaling."""
    data = np.ascontiguousarray(values[None], np.float32).tobytes()
    header = {"latent_tensor": {"dtype": "F32", "shape": [1, *values.shape], "data_offsets": [0, len(data)]},
              "latent_format_version_0": {"dtype": "F32", "shape": [0], "data_offsets": [len(data), len(data)]}}
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + data


def decode_graph(checkpoint: str, latent_name: str) -> dict[str, Any]:
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "2": {"class_type": "LoadLatent", "inputs": {"latent": latent_name}},
            "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 2]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "comfytrace_latent"}}}


def checkpoints(case_dir: Path, models_dir: Path, identity: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Checkpoints named by the recovered prompts, with local hash and memory identity. Raises ValueError for a
    name that escapes --models-dir or has no local copy."""
    seen = {m["name"]: m for m in (identity or {}).get("models", [])}
    out = []
    for name in model_names(case_dir):
        # Pickle checkpoints run code when loaded; decode-latents takes safetensors only.
        if not name.lower().endswith((".safetensors", ".sft")):
            continue
        parts = _relative_parts(name)
        local = _contained_path(models_dir, parts) if parts else None
        if local is None or not local.is_file():
            raise ValueError(f"checkpoint {name!r} has no contained local copy under --models-dir")
        digest = sha256_file(local)
        memory = seen.get(name, {})
        level = ("OBSERVED_IN_MEMORY" if memory.get("verdict") == "OBSERVED" and memory.get("sha256") == digest
                 else "HASHED")
        out.append({"name": name, "sha256": digest, "model_match_level": level,
                    "model_memory_identity": memory.get("verdict", "NOT_ASSESSED"),
                    "vae_private_windows": memory.get("components", {}).get("vae", {}).get("private_windows")})
    return out


def run_decode(args: argparse.Namespace) -> int:
    """Decodes every latent of the case with every recovered checkpoint. Returns 0, or 1 when refused."""
    case_dir: Path = args.case_dir
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    doc = json.loads((case_dir / latents.LATENTS_NAME).read_text(encoding="utf-8"))
    if doc["memory_sha256"] != manifest["case_metadata"]["memory_image"]["sha256"]:
        raise ValueError(f"{latents.LATENTS_NAME} was written from another memory image")
    entries = latents.load(case_dir)
    if not entries:
        log.info("no retained latents to decode")
        return 0
    out_dir = case_dir / DECODES_DIR
    if decodes_complete(case_dir):
        raise FileExistsError(f"{out_dir / COMPLETE_NAME} exists; latents are decoded once per case")
    models = checkpoints(case_dir, args.models_dir, _model_identity(case_dir, manifest))
    if not models:
        log.error("no safetensors checkpoint in the recovered prompts; pickle checkpoints are never loaded here")
        return 1
    client = RunnerClient(resolve_runner(args), timeout=args.timeout)
    info = client.object_info()
    missing = [n for n in NODES if n not in info]
    unregistered = [m["name"] for m in models if m["name"] not in _registered_choices(info)]
    if missing or unregistered:
        raise ValueError(f"runner lacks node classes {missing} or checkpoints {unregistered}")
    environment = client.environment()
    # An interrupted run left records for some pairs: those that completed are kept and not redone, every
    # other pair is decoded under a new decode ID after the highest one written.
    done, count = completed_pairs(out_dir)
    if done:
        log.info("resuming %s: %d decodes already completed", DECODES_DIR, len(done))
    out_dir.mkdir(exist_ok=True)
    exporter = EvidentiaryExporter(case_dir)
    failed = 0
    for entry in entries:
        pending = [m for m in models if _pair(entry["latent_id"], entry["sha256"], m) not in done]
        if not pending:
            continue
        _, c, h, w = entry["shape"]
        raw = (case_dir / entry["raw_file_path"]).read_bytes()
        values = np.frombuffer(raw, np.float32).copy()
        if entry["first_word"] == "POINTER":
            values[:2] = values[2]
        payload = latent_file(values.reshape(c, h, w))
        name = f"comfytrace_{entry['latent_id']}.latent"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(payload)
            stored = client.upload_input(path, name)
        for model in pending:
            count += 1
            decode_id = f"LATDEC-{count:03d}"
            graph = decode_graph(model["name"], stored)
            record: dict[str, Any] = {
                "decode_id": decode_id, "latent_id": entry["latent_id"], "latent_sha256": entry["sha256"],
                "latent_file_sha256": hashlib.sha256(payload).hexdigest(), "checkpoint": model,
                "runner_environment": environment, "graph": graph,
                "origin": ORIGIN, "status": "FAILED"}
            try:
                (meta, image), *_ = client.retrieve_output_rasters(client.queue_prompt(graph), ["4"])
                suffix = "" if len(models) == 1 else f"_{decode_id}"
                if (exporter.derived_dir / f"DERIVED_latent_{entry['latent_id']}{suffix}.png").exists():
                    suffix = f"_{decode_id}"  # left by an interrupted run before its record; never overwritten
                record["derived_image"] = exporter.export_derived_preview(
                    f"latent_{entry['latent_id']}{suffix}.png", image)
                record["derived_image_sha256"] = hashlib.sha256(image).hexdigest()
                record["runner_output"] = meta
                record["status"] = "COMPLETED"
            except Exception as exc:  # noqa: BLE001 - recorded; the remaining latents still decode
                failed += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                log.error("%s: %s", decode_id, record["error"])
            _write_atomic(out_dir / f"{decode_id}.json", json.dumps(record, indent=1) + "\n")
            log.info("%s: %s with %s: %s", decode_id, entry["latent_id"], model["name"], record["status"])
    # Written last: only a run that reached every pair marks the stage done.
    _write_atomic(out_dir / COMPLETE_NAME, json.dumps(
        {"decodes": count, "failed_this_run": failed, "memory_sha256": doc["memory_sha256"]}, indent=1) + "\n")
    return int(failed > 0)


def decodes_complete(case_dir: Path) -> bool:
    """True once a decode run has reached every (latent, checkpoint) pair; a bare or partial directory is not."""
    return (case_dir / DECODES_DIR / COMPLETE_NAME).is_file()


def _pair(latent_id: str, latent_sha256: str, model: dict[str, Any]) -> tuple[str, str, str, str]:
    return latent_id, latent_sha256, model["name"], model["sha256"]


def completed_pairs(out_dir: Path) -> tuple[set[tuple[str, str, str, str]], int]:
    """(latent, checkpoint) pairs with a COMPLETED record in ``out_dir``, and the highest decode number used."""
    done: set[tuple[str, str, str, str]] = set()
    highest = 0
    for path in sorted(out_dir.glob("LATDEC-*.json")) if out_dir.is_dir() else []:
        try:
            highest = max(highest, int(path.stem.split("-", 1)[1]))
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue  # an unreadable record proves nothing; its pair is decoded again
        if not isinstance(record, dict) or record.get("status") != "COMPLETED":
            continue
        model = record.get("checkpoint")
        # A record that parses but does not name its pair proves nothing either.
        if (isinstance(model, dict) and isinstance(model.get("name"), str) and isinstance(model.get("sha256"), str)
                and isinstance(record.get("latent_id"), str) and isinstance(record.get("latent_sha256"), str)):
            done.add(_pair(record["latent_id"], record["latent_sha256"], model))
    return done, highest


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False,
                                     suffix=".tmp") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
