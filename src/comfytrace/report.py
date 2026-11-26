"""HTML report entry point and compatibility imports for case consumers."""

from __future__ import annotations

from pathlib import Path

from comfytrace.casefiles import (
    DERIVED_PREFIX,
    EVIDENCE_NAME,
    LINKS_NAME,
    MANIFEST_NAME,
    METADATA_NAME,
    REPORT_NAME,
    EvidentiaryExporter,
    ManifestWriter,
    load_regenerations,
    next_regeneration_id,
    save_regeneration_record,
    sha256_file,
)
from comfytrace.findings import (
    MATCH_STATUSES,
    # Retained as compatibility bindings for existing in-repo tests.
    _regen_positions,
    _verified_replay_positions,
    apply_image_links,
    comparison_fact,
    comparison_only_output,
    effective_findings,
    memory_only_after_links,
    partial_rows_distinguish_inputs,
    regeneration_permits_promotion,
    regeneration_preflight_verified,
    target_is_input,
    tier_counts,
    verify_image_link_promotions,
)
from comfytrace.reportview import (
    # Retained as compatibility bindings for existing in-repo tests.
    _Previews,
    _Report,
)
from comfytrace.workbench import write_workbench

# Private symbols (_Previews, _Report, _regen_positions, _verified_replay_positions) remain exported here
# solely as compatibility bindings for existing in-repo tests; production code imports casefiles/findings directly.
__all__ = [
    "DERIVED_PREFIX",
    "EVIDENCE_NAME",
    "EvidentiaryExporter",
    "HtmlReporter",
    "LINKS_NAME",
    "MANIFEST_NAME",
    "MATCH_STATUSES",
    "METADATA_NAME",
    "ManifestWriter",
    "REPORT_NAME",
    "_Previews",
    "_Report",
    "_regen_positions",
    "_verified_replay_positions",
    "apply_image_links",
    "comparison_fact",
    "comparison_only_output",
    "effective_findings",
    "load_regenerations",
    "memory_only_after_links",
    "next_regeneration_id",
    "partial_rows_distinguish_inputs",
    "regeneration_permits_promotion",
    "regeneration_preflight_verified",
    "save_regeneration_record",
    "sha256_file",
    "target_is_input",
    "tier_counts",
    "verify_image_link_promotions",
]


class HtmlReporter:
    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)

    def render_report(
        self, manifest_path: Path, regenerations_dir: Path | None = None, output_html: Path | None = None
    ) -> Path:
        """Renders the self-contained interactive workbench (``workbench.html``) from manifest.json plus any
        regeneration records. Raises: OSError, IngestionError (unreadable or non-ComfyTrace manifest)."""
        return write_workbench(self.case_dir, Path(manifest_path), regenerations_dir, output_html)
