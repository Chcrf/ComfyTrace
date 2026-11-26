"""Regression checks for the data embedded in the offline case report."""

import hashlib
import json
import re
import subprocess
import sys

import pytest

from comfytrace import workbench
from comfytrace.carve import canonical_json
from comfytrace.exceptions import IngestionError
from comfytrace.workbench import write_workbench


def _minimal_case(tmp_path):
    manifest = {"case_metadata": {"case_id": "CASE-INT"}, "findings": {"workflows": [], "findings": []}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return tmp_path


def test_workbench_reads_utf16_graph_and_withholds_changed_export(tmp_path):
    graph = {"1": {"class_type": "Example", "inputs": {"text": "original"}}}
    workflow = {"workflow_id": "WF-1", "raw_file_path": "exports/raw/WF-1.json",
                "encoding_detected": "UTF-16LE", "prompt_format": "API_GRAPH", "completeness": "COMPLETE",
                "canonical_json_sha256": hashlib.sha256(canonical_json(graph).encode()).hexdigest()}
    file = tmp_path / workflow["raw_file_path"]
    file.parent.mkdir(parents=True)
    file.write_bytes(json.dumps({"prompt": graph}).encode("utf-16-le"))
    manifest = {"case_metadata": {"case_id": "__COMFYTRACE_DATA__"},
                "findings": {"workflows": [workflow], "findings": [
                    {"finding_id": "FIND-1", "tier": "E5", "artifact_ids": ["WF-1"], "rationale": "workflow"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    def embedded_graph():
        page = write_workbench(tmp_path).read_text(encoding="utf8")
        payload = re.search(r'<script id="ct-data" type="application/json">(.*?)</script>', page, re.S)
        assert payload
        data = json.loads(payload[1])
        assert data["case"]["caseId"] == "__COMFYTRACE_DATA__"
        return data["findings"][0]["workflow"]["graph"]

    assert embedded_graph() == graph
    graph["1"]["inputs"]["text"] = "changed"
    file.write_text(json.dumps(graph), encoding="utf8")
    assert embedded_graph() == {}


@pytest.mark.parametrize(
    ("relpath", "payload"),
    [
        ("recovery-evidence.json", "{not-json"),
        ("image-links.json", "[]"),
        ("regenerations/REGEN-001.json", "{truncated"),
        ("regenerations/REGEN-001.json", "[]"),
    ],
)
def test_write_workbench_wraps_corrupt_sidecars_and_regenerations_as_ingestion_error(tmp_path, relpath, payload):
    _minimal_case(tmp_path)
    existing = tmp_path / "workbench.html"
    existing.write_text("prior-workbench", encoding="utf8")
    bad = tmp_path / relpath
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(payload, encoding="utf8")

    with pytest.raises(IngestionError):
        write_workbench(tmp_path)

    assert existing.read_text(encoding="utf8") == "prior-workbench"


def test_write_workbench_is_atomic_and_cleans_temp_file_on_replace_error(tmp_path, monkeypatch):
    _minimal_case(tmp_path)
    out = write_workbench(tmp_path)
    prior = out.read_text(encoding="utf8")
    assert not list(tmp_path.glob("*.tmp"))

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(workbench.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_workbench(tmp_path)

    assert out.read_text(encoding="utf8") == prior
    assert not list(tmp_path.glob("*.tmp"))


def test_casefiles_and_replay_modules_do_not_import_report_or_workbench():
    probe = (
        "import sys, comfytrace.casefiles, comfytrace.latentdecode, comfytrace.replaysearch; "
        "assert 'comfytrace.report' not in sys.modules; "
        "assert 'comfytrace.workbench' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_the_template_is_one_offline_file_whose_script_cannot_end_early():
    from importlib import resources

    template = resources.files("comfytrace").joinpath("templates/workbench.html").read_text(encoding="utf-8")
    assert re.findall(r"http://(?!www\.w3\.org/2000/svg)", template) == [] and "https://" not in template
    assert "<link" not in template and "@import" not in template
    assert re.findall(r'(?:src|href)="(?!#|data:|\$\{src\})[^"]*"', template) == []
    # The data block and the one inline script are the only two script elements; no literal "</script>" in code.
    assert template.count("<script") == 2 and template.count("</script>") == 2
