# Evaluation & Reproducibility Usage Note

This package provides portable real-capture path resolution (`paths.py`) and a read-only post-extraction scorer (`scorer.py`).

## Withheld Ground-Truth Boundary

- `comfytrace extract` receives only `--memory`, `--symbols-dir`, and `--outdir`. It never reads `COMFYTRACE_EVAL_*` variables or ground-truth files.
- Withheld ground-truth JSON files are supplied **only** to `tests.evaluation.scorer` after extraction completes.

## Portable Real-Case Registry & Path Configuration

Registered lab cases live in `tests/fixtures/groundtruth_cases.json` (override with `COMFYTRACE_EVAL_CASES_FILE`).

Resolution precedence (first match wins, Windows and Linux):

1. Per-case environment variables (`<CASE_TOKEN>` = uppercased alphanumerics with `_`, e.g. `gen-only-run-1` -> `GEN_ONLY_RUN_1`):
   - `COMFYTRACE_EVAL_MEMORY_<CASE_TOKEN>`
   - `COMFYTRACE_EVAL_SYMBOLS_<CASE_TOKEN>`
   - `COMFYTRACE_EVAL_GROUNDTRUTH_<CASE_TOKEN>`
2. Per-case overrides in the JSON file named by `COMFYTRACE_EVAL_CONFIG` (`cases.<case_id>.{memory,symbols_dir,groundtruth}`). Relative paths resolve from the config file's directory.
3. Shared directory environment variables:
   - `COMFYTRACE_EVAL_DUMPS_DIR` + `memory_filename`
   - `COMFYTRACE_EVAL_SYMBOLS_DIR` (+ optional `symbols_subdir`)
   - `COMFYTRACE_EVAL_GROUNDTRUTH_DIR` + `groundtruth_filename` (or `<case_id>.groundtruth.json` / `<case_id>.json`)
4. Shared directories in `COMFYTRACE_EVAL_CONFIG` (`dumps_dir`, `symbols_dir`, `groundtruth_dir`).
5. Case registry `platform_defaults.<windows|linux|darwin>.*` or legacy `memory` / `symbols_dir`.

### Missing-Path Behavior

- Unconfigured or registry-default missing dumps/symbols skip `pytest -m groundtruth`.
- Any **explicitly configured** memory dump, symbols directory (`COMFYTRACE_EVAL_SYMBOLS_*` / `COMFYTRACE_EVAL_SYMBOLS_DIR` / config), or withheld ground truth (`COMFYTRACE_EVAL_GROUNDTRUTH_*` / `COMFYTRACE_EVAL_GROUNDTRUTH_DIR` / config) **fails** if missing so broken paths never silently pass.

## Running Real-Capture Tests

```powershell
# PowerShell example
$env:PYTHONPATH = "src"
$env:COMFYTRACE_EVAL_DUMPS_DIR = "D:\captures"
$env:COMFYTRACE_EVAL_SYMBOLS_DIR = "D:\symbols"
# Optional post-extraction scoring:
$env:COMFYTRACE_EVAL_GROUNDTRUTH_DIR = "D:\groundtruth"
pytest -m groundtruth tests/integration/test_evaluation_matrix.py
```

When a registered case has withheld ground truth, use `expected_score_limits` to set the expected
job and output counts. Use `expected_jobs`, `expected_outputs`, `min_recovered_jobs`,
`min_recovered_outputs`, and `min_correct_output_links` for case-specific thresholds. The test
defaults to zero wrong links and requires some recovery when the ground truth lists jobs or outputs.

## Running the Post-Extraction Scorer Directly

```powershell
$env:PYTHONPATH = "src"
python -m tests.evaluation.scorer `
  --case-dir path\to\extracted_case `
  --groundtruth tests\fixtures\evaluation\sample_withheld_groundtruth.json `
  --view effective `
  --json-out path\to\score.json `
  --fail-on-wrong-links `
  --fail-on-over-tier
```

- `--view`: `effective` (manifest + valid `image-links.json` + valid `regenerations/`), `extraction` (raw `manifest.json` only), or `memory_only` (`recovery-evidence.json` `memory_only_findings` + valid `image-links.json`).
- See `python -m tests.evaluation.scorer --help` for all CLI flags and `tests/fixtures/evaluation/sample_withheld_groundtruth.json` for the ground-truth JSON layout (`output_node_id`, `batch_index`, `raster_sha256` / `pixel_sha256`, `container_sha256`, `raw_sha256s`, `physical_offsets`, `expected_ambiguous`, tier expectations, and `distractors`).
- Output-level identities (`output_node_id`, `batch_index`, per-raster digest) are scored when present in both attribution and ground truth; missing identities are surfaced explicitly as `UNKNOWN_PREDICTED` or `UNKNOWN_GROUNDTRUTH` rather than assumed correct.
