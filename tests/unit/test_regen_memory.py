"""Regeneration of every recovered job in one command."""

import argparse
import json

from comfytrace import regenerate
from comfytrace.cli import parse_args, run_regenerate


def test_all_runs_each_recovered_job_once_and_counts_failures(tmp_path, monkeypatch):
    workflows = [{"workflow_id": f"WFPY-00{i}", "encoding_detected": enc, "prompt_format": fmt,
                  "completeness": done, "raw_file_path": f"exports/raw/WFPY-00{i}.json"}
                 for i, (enc, fmt, done) in enumerate([("CPYTHON_OBJECTS", "API_GRAPH", "COMPLETE"),
                                                       ("CPYTHON_OBJECTS", "API_GRAPH", "COMPLETE"),
                                                       ("CPYTHON_OBJECTS", "UI_WORKFLOW", "COMPLETE"),
                                                       ("CPYTHON_OBJECTS", "API_GRAPH", "PARTIAL"),
                                                       ("JSON_CARVED", "API_GRAPH", "COMPLETE")], 1)]
    (tmp_path / "manifest.json").write_text(json.dumps({"findings": {"workflows": workflows}}))
    calls = []
    discoveries = []
    url = "http://local-runner:8188"

    def discover(args):
        discoveries.append(args.case_dir)
        return url

    def fake(args):
        calls.append((args.workflow_id, args.workflow.name, args.compare_to, args.runner_url))
        return 1 if args.workflow_id == "WFPY-002" else 2

    monkeypatch.setattr(regenerate, "resolve_runner", discover)
    monkeypatch.setattr(regenerate, "run_regeneration", fake)
    assert regenerate.run_all(argparse.Namespace(case_dir=tmp_path, compare_to=None)) == 1
    assert discoveries == [tmp_path]
    assert calls == [("WFPY-001", "WFPY-001.json", None, url), ("WFPY-002", "WFPY-002.json", None, url)]


def test_all_refuses_a_single_comparison_target(tmp_path):
    args = parse_args(["regenerate", "--case-dir", str(tmp_path), "--all", "--runner-url", "http://x",
                       "--compare-to", str(tmp_path / "a.png")])
    assert run_regenerate(args) == 1
