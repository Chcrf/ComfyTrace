# ComfyTrace

ComfyTrace recovers ComfyUI workflows, prompts, generated images, cached tensors, and retained latents from Windows and Linux memory captures. It writes a read-only manifest, raw and derived exports, and an offline workbench (`workbench.html`).

Extraction runs on the CPU with no GPU, no model weights, and no network access unless you pass `--allow-symbol-download` or run lab replay.

## Quickstart

### 1. Install

ComfyTrace runs on Python 3.10 through 3.13. Structured heap recovery supports captured Windows CPython 3.12 or 3.13 and Linux CPython 3.12.

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[regen,search]"
.\.venv\Scripts\comfytrace.exe --version
```

```bash
# Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[regen,search]'
comfytrace --version
```

| Extra | Added packages and behavior |
| :--- | :--- |
| none | Core extraction (`volatility3==2.28.2`), carving, correlation, and `workbench.html` |
| `regen` | Perceptual comparison (`imagehash`, `scikit-image`) for lab replay |
| `search` | Bulk pixel anchor search (`ahocorasick-rs`) |
| `runner` | Local ComfyUI discovery and install through `comfy-cli` when `--runner-url` is omitted |
| `dev` | `pytest`, `ruff`, and `mypy` |

### 2. Run

```bash
# Inspect the captured ComfyUI stack and local checkpoint weights before extracting
comfytrace env-id --memory dump.raw --symbols-dir ./symbols --runner-url http://127.0.0.1:8188
comfytrace model-id --memory dump.raw --symbols-dir ./symbols --model ./checkpoints

# Pure memory extraction when the checkpoint is unknown
comfytrace extract --memory dump.raw --symbols-dir ./symbols --outdir ./case01

# Extract and automatically replay recovered workflows using a checkpoint file or directory
comfytrace extract --memory dump.raw --symbols-dir ./symbols --outdir ./case01 \
    --model ./checkpoints/v1-5-pruned-emaonly-fp16.safetensors --runner-url http://127.0.0.1:8188
```

Open `case01/workbench.html` in a browser to inspect findings, side-by-side lab replays, Pixel and Derived file SHA-256 hashes, and provenance.

## Platform support

| Capture format | Command flags | Supported recovery |
| :--- | :--- | :--- |
| Windows physical (`.raw`) | `--image-os windows --symbols-dir ./symbols` | Process enumeration, page-table walks, PFN provenance and page ordering, CPython 3.12 and 3.13 heap and PyTorch execution cache, carving, decoded bitmaps, and retained latents |
| Linux whole-RAM (`.raw` or LiME v1 `.lime`) | `--image-os linux --symbols-dir ./symbols` | Process enumeration, 4-level and 5-level page-table walks, PFN and page-cache provenance, backing file paths, CPython 3.12 heap and execution cache, and raw replay search |
| Linux ELF core (`.core`) | `--image-os linux --image-format elf-core --comfyui-pid <PID>` | Single-process carving, CPython and PyTorch recovery, replay, and reporting without kernel symbols |

Place a matching Volatility 3 Linux ISF JSON under `symbols/linux/` built from the matching `vmlinux` and `System.map` with `dwarf2json`. Decompress AVML captures to uncompressed LiME v1 before running `extract`.

Live `zswap` and `zram` recovery is disabled in v1.0.0. If kernel symbols are missing or ComfyUI has exited, `extract` falls back to physical carving (`PHYSICAL_FALLBACK`, exit code `3`).

## CLI reference

Top-level `comfytrace --help` lists `extract`, `model-id`, and `env-id`.

### `comfytrace extract`

| Option | Description |
| :--- | :--- |
| `--memory PATH` | Memory capture file. Required. |
| `--outdir PATH` | New case output directory. Required. Never overwrites an existing manifest. |
| `--symbols-dir PATH` | Offline Volatility 3 ISF cache. Required for physical captures. |
| `--image-os {windows,linux}` | Captured operating system. Defaults to `windows`. |
| `--image-format {physical,elf-core}` | Capture type. Defaults to `physical`. |
| `--allow-symbol-download` | Permit Volatility 3 remote symbol downloads. |
| `--model PATH` | Checkpoint file or models directory. Runs automatic replay (`model-id`, `latents`, `regenerate`, `decode-latents`, `search-replay`, `link`, and `workbench.html`). |
| `--runner-url URL` | Running ComfyUI endpoint such as `http://127.0.0.1:8188`. If omitted during replay, ComfyTrace falls back to local `comfy-cli` discovery (`--comfy-cli`, `--install-comfyui`). |
| `--inputs-dir PATH` | Conditioning input images for img2img, inpaint, and edit workflows. |
| `--comfyui-pid PID` | Examiner-confirmed Python PID. Repeatable, and required for `elf-core`. |
| `--omit-unrelated-exports` | Record unrelated OS and application images without exporting their raw bytes. |
| `--content-stitch-png` | Search physical memory for torn PNG continuations (`DERIVED`). |
| `--case-id ID`, `--investigator NAME` | Case metadata recorded in `manifest.json`. |

### `comfytrace model-id`

```bash
comfytrace model-id --memory dump.raw --symbols-dir ./symbols --model ./checkpoints
```

Fingerprints local safetensors weights against ComfyUI private process memory (`--model` is an alias for `--models-dir`). Non-safetensors files (`.ckpt`, `.pt`, `.pth`, `.gguf`) report `UNSUPPORTED_FORMAT`, and an empty directory exits with code `1`. Pass `--case-dir ./case01` to write `model-identity.json` once into an existing case.

| Verdict | Meaning |
| :--- | :--- |
| `OBSERVED` | Identifying UNet weight windows matched in private ComfyUI memory with 0 control hits |
| `SHARED_COMPONENTS_ONLY` | Only shared VAE or CLIP weights matched |
| `NOT_OBSERVED` | Below threshold in private process memory |
| `NO_PROCESS_MEMORY` | No live ComfyUI Python process memory was recovered |
| `UNSUPPORTED_FORMAT` / `NO_LOCAL_COPY` | Unsupported checkpoint format or missing file |

### `comfytrace env-id`

```bash
comfytrace env-id --memory dump.raw --symbols-dir ./symbols --runner-url http://127.0.0.1:8188
```

Recovers the captured ComfyUI version, PyTorch build string (`2.12.1+cu130`), Python version, and GPU device from private ComfyUI process memory and compares them against an optional replay runner. Pass `--case-dir ./case01` to write `environment-identity.json` once into an existing case.

| Verdict | Meaning |
| :--- | :--- |
| `EXACT_STACK_MATCH` | ComfyUI, PyTorch build, Python minor version, and GPU device name all match |
| `PARTIAL_STACK_MATCH` | No mismatch, but some fields are unreported, unattributed, or conflict across processes |
| `STACK_MISMATCH` | One or more compared fields differ between the captured process and the runner |
| `CAPTURED_ONLY` | No `--runner-url` was supplied |
| `RUNNER_UNREPORTED` | The runner returned no compared fields or was unreachable |
| `NO_PROCESS_MEMORY` | No live ComfyUI Python process memory was recovered |
| `UNAVAILABLE` | No stack fields were recovered from process memory |

GPU comparison checks device type and name only; device index (`cuda:0` versus `cuda:1`) and allocator (`cudaMallocAsync` versus `native`) are advisory. The `cu130` suffix in the PyTorch string names the wheel's CUDA toolkit build, not the installed NVIDIA driver.

<details>
<summary>ComfyUI runner setup and matched-stack replay</summary>

For a quick CPU-only runner (perceptual replay), start ComfyUI with:

```bash
python main.py --cpu --listen 127.0.0.1 --port 8188 --disable-all-custom-nodes --disable-api-nodes
```

Add `--cache-none` if you replay the same workflow more than once in one runner session.

When `env-id` reports `STACK_MISMATCH`, diffusion replay typically yields perceptual matches (`SSIM` around 0.98 to 0.999) instead of identical RGBA hashes. To match the captured stack (for example `ComfyUI 0.44.0`, `PyTorch 2.12.1+cu130`, and `Python 3.13`):

```bash
# 1. Check out the exact ComfyUI tag reported by env-id
git clone https://github.com/comfyanonymous/ComfyUI.git comfyui-match
git -C comfyui-match checkout v0.44.0

# 2. Create an isolated virtual environment with the matching Python minor version
uv venv --python 3.13 comfyui-match/.venv

# 3. Install the exact PyTorch and CUDA wheel (split 2.12.1+cu130 into 2.12.1 and cu130)
uv pip install --python comfyui-match/.venv/bin/python \
    "torch==2.12.1" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python comfyui-match/.venv/bin/python -r comfyui-match/requirements.txt

# 4. Start the matched runner on a dedicated port
comfyui-match/.venv/bin/python comfyui-match/main.py \
    --listen 127.0.0.1 --port 8189 --disable-all-custom-nodes --disable-api-nodes

# 5. Confirm EXACT_STACK_MATCH and run replay against the matched runner
comfytrace env-id --memory dump.raw --symbols-dir ./symbols --runner-url http://127.0.0.1:8189
comfytrace extract --memory dump.raw --symbols-dir ./symbols --outdir ./case01 \
    --model ./checkpoints --runner-url http://127.0.0.1:8189
```

On Windows PowerShell, use `comfyui-match\.venv\Scripts\python.exe` in place of `comfyui-match/.venv/bin/python`. Matching a different GPU model may still produce small pixel drift because GPU kernels and drivers are not identical across hardware.

</details>

<details>
<summary>Advanced subcommands</summary>

These subcommands are hidden from top-level `--help` and remain callable directly:

| Command | Purpose |
| :--- | :--- |
| `comfytrace replay-case` | Run `model-id`, `latents`, `regenerate`, `decode-latents`, `search-replay`, `link`, and `workbench.html` |
| `comfytrace regenerate` | Replay a single `--workflow` JSON or `--all` CPython prompts on a ComfyUI runner |
| `comfytrace latents` | Extract retained latent tensor blocks from ComfyUI process memory |
| `comfytrace decode-latents` | Decode recovered latents through the checkpoint VAE into `DERIVED` PNGs |
| `comfytrace search-replay` | Search raw physical memory for pixel copies of regenerated images |
| `comfytrace link` | Link orphan memory images to jobs through regenerated outputs |
| `comfytrace report` | Re-render `workbench.html` from `manifest.json` and `regenerations/` |

</details>

## Evidence tiers

| Tier | Meaning |
| :---: | :--- |
| `E1` | Intact memory image, complete workflow, and verified lab replay match (`EXACT_MATCH` or `PERCEPTUAL_MATCH`) |
| `E2` | Complete memory image and complete workflow linked by a `STRICT` or `SUPPORTED` key |
| `E3` | Linked finding with a partial image that has decoded pixels, or with a partial workflow; replay adds `PARTIAL_CORROBORATION` or `INCONCLUSIVE` below 3 rows |
| `E4` | Complete workflow with no recovered memory pixels (`E5`), proved executable by lab replay (`EXECUTABLE_REPRODUCTION`) |
| `E5` | Complete workflow recovered from memory, including 0-row PNG containers, with no recovered pixels and no replay |
| `E6` | Partial workflow, UI context, or parameter fragment with no image |
| `UT` | Untiered artifacts, including orphan images, size-excluded 1x1 placeholders, and unrelated OS or application images |

## Case output layout

```text
case01/
├── workbench.html            # Interactive examiner workbench
├── manifest.json             # Read-only forensic manifest (schema 1.0.0)
├── recovery-evidence.json    # Page origins, memory-only view, execution cache, and provenance
├── scorecard.json            # Per-job and per-output recovery scorecard
├── workflow-metadata.json    # Normalized sampler parameters (seed, steps, CFG, model, prompts)
├── model-identity.json       # Optional: local model weight presence in process memory
├── environment-identity.json # Optional: captured ComfyUI/PyTorch/Python/GPU stack and runner comparison
├── latents.json              # Optional: retained latent tensors from process memory
├── replay-search.json        # Optional: raw dump pixel search hits
├── image-links.json          # Optional: orphan-to-job replay links
├── regenerations/            # Optional: REGEN-NNN.json replay records and preflight audits
├── latent-decodes/           # Optional: VAE-decoded latent outputs
└── exports/
    ├── raw/                  # Verbatim carved bytes (PNGs, JSONs, span_NNN.bin slices)
    ├── derived/              # Synthesized previews, stitched text, and lab-replayed images (DERIVED_*)
    └── uncertain/            # Pixel rows of uncertain ownership, separated from exports/raw
```

## Exit codes

When multiple conditions apply, the highest-precedence code wins in order `1`, `3`, `2`, `4`, `0`.

| Code | `extract` | `model-id` and `env-id` | `regenerate` |
| :---: | :--- | :--- | :--- |
| `0` | Clean recovery (`E2` and `E5` only) | Completed | Earned `E1` match or `E4` executability |
| `1` | Operational failure, existing manifest, or usage error | Error, empty model dir, existing sidecar, or `RUNNER_UNREPORTED` (nothing written) | Preflight rejection, missing dependency, unreachable runner, or failed run |
| `2` | `E3` or `E6` findings, orphan images, or ambiguous images | Not used | Ran without an earned `E1` or `E4` (partial corroboration, inconclusive, or below thresholds) |
| `3` | Physical-only fallback (`NO_COMFYUI_PROCESS` or unresolved kernel symbols) | `NO_PROCESS_MEMORY` (nothing written) | Not used |
| `4` | No ComfyUI-related evidence found | Not used | Not used |

## Development and testing

```bash
pip install -e ".[dev,regen,search]"
ruff check src tests
pytest -q -m "not groundtruth"
```

<details>
<summary>Module layout</summary>

| Subsystem | Modules |
| :--- | :--- |
| OS and memory ingestion | `ingest.py`, `ingest_linux.py`, `lime.py`, `elfcore.py`, `pagetable.py`, `procmem.py`, `linux_pfn.py`, `memstore.py`, `linux_memstore.py` |
| CPython and PyTorch recovery | `cpython/` (`base.py`, `runtime.py`, `registry.py`, `profiles/v312.py`, `profiles/v313.py`), `pyheap.py`, `execcache.py`, `torchmem.py`, `memrecover.py` |
| Carving and reconstruction | `carve.py`, `media.py`, `bitmaps.py`, `fragments.py`, `outputs.py`, `pageorder.py`, `pagestitch.py`, `metadata/` |
| Correlation and evidence | `correlate.py`, `evidence.py`, `jobview.py`, `models.py`, `exceptions.py` |
| Replay, latents, and identity | `regenerate.py`, `replaycase.py`, `replaysearch.py`, `replayevidence.py`, `link.py`, `latents.py`, `latentdecode.py`, `modelid.py`, `envid.py`, `localrunner.py` |
| Case I/O, workbench, and CLI | `casefiles.py`, `findings.py`, `reportview.py`, `workbench.py`, `templates/workbench.html`, `report.py`, `cli.py` |

</details>
