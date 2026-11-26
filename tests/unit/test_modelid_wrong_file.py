"""A checkpoint with the right filename but different weights must never earn OBSERVED_IN_MEMORY.

Everything is offline and tiny: toy F16 safetensors files, a synthetic ProcessMemory standing in for the dump, the
local HTTP stub runner from conftest, and the real ``modelid`` / ``PreflightChecker`` / ``latentdecode`` /
``TieredEvaluator`` interfaces.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from comfytrace import modelid
from comfytrace.cli import parse_args, run_extract, run_regenerate
from comfytrace.latentdecode import checkpoints
from comfytrace.procmem import ProcessMemory
from comfytrace.regenerate import PreflightChecker, TieredEvaluator, _model_identity
from tests.synthetic import api_prompt, noise_image

STEP = 512
NAME = "v1-5-pruned-emaonly.safetensors"
MEMORY_SHA = "ab" * 32
PRIVATE, MAPPED = 0x10000000, 0x20000000
UNET, VAE = "model.diffusion_model.w", "first_stage_model.w"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return path


def _memory(private: bytes, mapped: bytes = b"") -> ProcessMemory:
    noise = np.random.default_rng(99).integers(0, 256, 8192, np.uint8).tobytes()
    private = noise[:24] + private + noise

    def pad(b: bytes) -> bytes:
        return b + b"\0" * (-len(b) % 4096)

    private, mapped = pad(private), pad(mapped or b"\0")
    return ProcessMemory(7, [(PRIVATE, len(private), 0), (MAPPED, len(mapped), len(private))], private + mapped,
                         vads=[(PRIVATE, PRIVATE + len(private) - 1, None),
                               (MAPPED, MAPPED + len(mapped) - 1, rf"\models\checkpoints\{NAME}")])


def _recovered_case(root: Path) -> Path:
    """A case whose only recovered prompt loads NAME; this is where modelid reads the model names from."""
    case = root / "named_case"
    (case / "exports").mkdir(parents=True)
    (case / "exports" / "wf.json").write_text(json.dumps(api_prompt(ckpt=NAME)))
    manifest = {"case_metadata": {"memory_image": {"sha256": MEMORY_SHA}},
                "findings": {"workflows": [{"prompt_format": "API_GRAPH", "encoding_detected": "CPYTHON_OBJECTS",
                                            "raw_file_path": "exports/wf.json"}]}}
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def _identity(case: Path, models_dir: Path, memory: ProcessMemory, monkeypatch, sha: str = MEMORY_SHA) -> dict:
    monkeypatch.setattr(modelid, "STEP", STEP)
    (document,) = modelid.identity_document(case, [memory], models_dir, sha)["models"]
    return document


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def worlds(tmp_path):
    """Two different files under one filename: ``local`` is what the examiner holds, ``used`` is what ran."""
    used_tensors = {UNET: _weights(1), VAE: _weights(2)}
    local_tensors = {UNET: _weights(21), VAE: _weights(22)}
    used = _checkpoint(tmp_path / "used" / NAME, used_tensors)
    local = _checkpoint(tmp_path / "local" / NAME, local_tensors)
    assert used.name == local.name and _sha(used) != _sha(local)
    return SimpleNamespace(used=used, local=local, used_tensors=used_tensors, local_tensors=local_tensors,
                           case=_recovered_case(tmp_path))


# ---------------------------------------------------------------------------------------------------------------
# modelid: verdicts for a same-name file whose weights are not the ones in the dump
# ---------------------------------------------------------------------------------------------------------------


def test_the_real_file_in_memory_is_observed_and_bound_to_its_own_hash(worlds, monkeypatch):
    """Positive control: without it the negatives below could pass for the wrong reason."""
    memory = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.used.parent, memory, monkeypatch)
    assert document["verdict"] == modelid.OBSERVED and document["sha256"] == _sha(worlds.used)


def test_same_name_different_weights_is_not_observed(worlds, monkeypatch):
    memory = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.local.parent, memory, monkeypatch)
    assert document["name"] == NAME
    assert document["verdict"] == modelid.NOT_OBSERVED
    assert document["sha256"] == _sha(worlds.local) != _sha(worlds.used)
    assert all(c["private_windows"] == 0 for c in document["components"].values())
    assert document["control_windows"] == 0


def test_a_shared_component_alone_is_weaker_than_observed(worlds, monkeypatch):
    """A fine-tune keeps the VAE: the local VAE is in the dump, the local denoiser is not."""
    memory = _memory(worlds.used_tensors[UNET].tobytes() + worlds.local_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.local.parent, memory, monkeypatch)
    assert document["verdict"] == modelid.SHARED_ONLY != modelid.OBSERVED
    assert document["components"]["unet"]["private_windows"] == 0
    assert document["components"]["vae"]["private_windows"] >= modelid.MIN_WINDOWS


def test_a_mere_fragment_of_the_denoiser_is_not_enough(worlds, monkeypatch):
    """Fewer than MIN_WINDOWS windows of the local denoiser (a few tensors shared with another file) do not count."""
    fragment = worlds.local_tensors[UNET][:2048]
    document = _identity(worlds.case, worlds.local.parent, _memory(fragment.tobytes()), monkeypatch)
    assert 0 < document["components"]["unet"]["private_windows"] < modelid.MIN_WINDOWS
    assert document["verdict"] == modelid.NOT_OBSERVED


def test_local_weights_only_in_a_file_backed_view_do_not_earn_observed(worlds, monkeypatch):
    """The dump's private memory holds another file's weights; the local file is merely mmapped under the name."""
    memory = _memory(worlds.used_tensors[UNET].tobytes(),
                     mapped=worlds.local_tensors[UNET].tobytes() + worlds.local_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.local.parent, memory, monkeypatch)
    assert document["verdict"] == modelid.NOT_OBSERVED
    assert document["components"]["unet"]["mapped_file_windows_not_counted"] > 0
    assert document["components"]["unet"]["private_windows"] == 0


# ---------------------------------------------------------------------------------------------------------------
# preflight (real PreflightChecker against the local stub runner) and decode-latents' checkpoint level
# ---------------------------------------------------------------------------------------------------------------


def _preflight(url, models_dir, identity_document) -> dict:
    workflow = SimpleNamespace(graph=api_prompt(ckpt=NAME), prompt_format="API_GRAPH")
    checker = PreflightChecker(url, [], False, models_dir, model_identity=identity_document)
    passed, status, audit = checker.validate(workflow)
    assert (passed, status) == (True, "PASSED")
    return audit["preflight_check"]


def _identity_file(case: Path, document: dict) -> dict:
    (case / modelid.IDENTITY_NAME).write_text(json.dumps({"memory_sha256": MEMORY_SHA, "models": [document]}))
    return _model_identity(case, json.loads((case / "manifest.json").read_text()))


def test_preflight_keeps_a_wrong_file_at_hashed_but_a_correct_one_is_observed(worlds, mock_runner_server, monkeypatch):
    url, _ = mock_runner_server
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    right = _identity_file(worlds.case, _identity(worlds.case, worlds.used.parent, dump, monkeypatch))
    wrong = _identity_file(worlds.case, _identity(worlds.case, worlds.local.parent, dump, monkeypatch))

    control = _preflight(url, worlds.used.parent, right)
    assert control["model_match_level"] == "OBSERVED_IN_MEMORY"

    check = _preflight(url, worlds.local.parent, wrong)
    assert check["model_match_level"] == "HASHED"
    assert check["model_memory_identity"] == {NAME: modelid.NOT_OBSERVED}
    assert check["model_sha256"] == {NAME: _sha(worlds.local)}


def test_preflight_keeps_shared_components_only_at_hashed(worlds, mock_runner_server, monkeypatch):
    url, _ = mock_runner_server
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.local_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.local.parent, dump, monkeypatch)
    check = _preflight(url, worlds.local.parent, _identity_file(worlds.case, document))
    assert document["verdict"] == modelid.SHARED_ONLY
    assert check["model_match_level"] == "HASHED"
    assert check["model_memory_identity"] == {NAME: modelid.SHARED_ONLY}


def test_an_observed_identity_of_the_real_file_does_not_transfer_to_a_swapped_same_name_file(
    worlds, mock_runner_server, monkeypatch,
):
    """OBSERVED was earned by ``used``; the examiner's models dir now holds ``local`` under the same name."""
    url, _ = mock_runner_server
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.used.parent, dump, monkeypatch)
    assert document["verdict"] == modelid.OBSERVED
    identity = _identity_file(worlds.case, document)

    check = _preflight(url, worlds.local.parent, identity)
    assert check["model_match_level"] == "HASHED"
    assert check["model_memory_identity"] == {NAME: modelid.OBSERVED}, "the verdict is reported, the level is not"
    assert check["model_sha256"] == {NAME: _sha(worlds.local)} and document["sha256"] != _sha(worlds.local)


def test_identity_from_another_dump_is_ignored_even_for_the_right_file(worlds, mock_runner_server, monkeypatch):
    url, _ = mock_runner_server
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    document = _identity(worlds.case, worlds.used.parent, dump, monkeypatch)
    path = worlds.case / modelid.IDENTITY_NAME
    path.write_text(json.dumps({"memory_sha256": "cd" * 32, "models": [document]}))
    manifest = json.loads((worlds.case / "manifest.json").read_text())
    assert _model_identity(worlds.case, manifest) is None
    assert _preflight(url, worlds.used.parent, None)["model_match_level"] == "HASHED"


def test_decode_latents_checkpoint_level_follows_the_same_rule(worlds, monkeypatch):
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    manifest = json.loads((worlds.case / "manifest.json").read_text())

    def level(models_dir: Path, identity_document: dict) -> str:
        _identity_file(worlds.case, identity_document)
        (entry,) = checkpoints(worlds.case, models_dir, _model_identity(worlds.case, manifest))
        return entry["model_match_level"]

    assert level(worlds.used.parent, _identity(worlds.case, worlds.used.parent, dump, monkeypatch)) == (
        "OBSERVED_IN_MEMORY")
    assert level(worlds.local.parent, _identity(worlds.case, worlds.local.parent, dump, monkeypatch)) == "HASHED"
    stale = _identity(worlds.case, worlds.used.parent, dump, monkeypatch)
    assert level(worlds.local.parent, stale) == "HASHED"


# ---------------------------------------------------------------------------------------------------------------
# evaluation: a deliberately different replay image gets no exact-match promotion, whatever the model level
# ---------------------------------------------------------------------------------------------------------------


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_evaluator_gives_a_different_image_no_exact_match():
    target = noise_image(64, 48, seed=5)
    different = noise_image(64, 48, seed=77)
    tier, subtier, metrics = TieredEvaluator().evaluate_intact(
        different, target, hashlib.sha256(target.tobytes()).hexdigest())
    assert (tier, subtier) == ("NON_MATCHING", "BELOW_THRESHOLDS")
    assert metrics["exact_pixel_match"] is False and metrics["native_pixel_sha256_match"] is False
    tier, subtier, _ = TieredEvaluator().evaluate_intact(
        target.copy(), target, hashlib.sha256(target.tobytes()).hexdigest())
    assert (tier, subtier) == ("E1", "EXACT_MATCH"), "control: the same pixels do match"


@pytest.fixture
def t1_case(tmp_path, mock_memory_image, fake_ingestor, symbols_dir) -> Path:
    case = tmp_path / "case01"
    args = parse_args(["extract", "--memory", str(mock_memory_image("T1")), "--symbols-dir", str(symbols_dir),
                       "--outdir", str(case)])
    assert run_extract(args, ingestor=fake_ingestor) == 0
    return case


@pytest.mark.parametrize(("local_is_real_file", "image_differs"), [(True, True), (False, True), (False, False)])
def test_model_identity_and_image_match_are_independent_and_no_exact_promotion_for_a_different_image(
    t1_case, mock_runner_server, worlds, monkeypatch, local_is_real_file, image_differs,
):
    """Full regenerate against the stub runner: the replay image decides promotion, never the model level, and a
    same-name wrong file cannot be upgraded to OBSERVED_IN_MEMORY by anything the image does."""
    url, state = mock_runner_server
    manifest = json.loads((t1_case / "manifest.json").read_text())
    memory_sha = manifest["case_metadata"]["memory_image"]["sha256"]
    dump = _memory(worlds.used_tensors[UNET].tobytes() + worlds.used_tensors[VAE].tobytes())
    models_dir = (worlds.used if local_is_real_file else worlds.local).parent
    monkeypatch.setattr(modelid, "STEP", STEP)
    document = modelid.identity_document(worlds.case, [dump], models_dir, memory_sha)
    modelid.write_identity(t1_case, document)
    if image_differs:
        state.image = _png(noise_image(64, 48, seed=99))

    argv = ["regenerate", "--case-dir", str(t1_case), "--workflow", str(t1_case / "exports/raw/WF-001.json"),
            "--runner-url", url, "--models-dir", str(models_dir),
            "--compare-to", str(t1_case / "exports/raw/CARVE-PNG-001.png")]
    exit_code = run_regenerate(parse_args(argv))

    record = json.loads((t1_case / "regenerations" / "REGEN-001.json").read_text())
    check, evaluation = record["preflight_check"], record["evaluation"]
    assert check["model_match_level"] == ("OBSERVED_IN_MEMORY" if local_is_real_file else "HASHED")
    assert check["model_memory_identity"] == {NAME: modelid.OBSERVED if local_is_real_file else modelid.NOT_OBSERVED}
    if image_differs:
        assert exit_code == 2
        assert evaluation["evaluated_tier"] == "NON_MATCHING" and evaluation["evaluation_subtier"] != "EXACT_MATCH"
        assert evaluation["exact_pixel_match"] is False
        assert evaluation["promoted_finding_tier"] is None and evaluation["target_finding_tier_retained"] == "E2"
        assert record["derived_regenerated_image"].startswith("exports/derived/DERIVED_regen_NON_MATCHING_")
    else:
        assert exit_code == 0
        assert evaluation["evaluation_subtier"] == "EXACT_MATCH" and evaluation["promoted_finding_tier"] == "E1"
