from __future__ import annotations

from typing import Any

from ..pipeline import run_classification


def run_job(
    job_id: str,
    input_path: str,
    output_prefix: str,
    mode: str,
    task: str,
    threshold: float,
    read_type: str,
    min_length: int,
    coverage_source: str,
    circularity_check: bool,
    polish: str,
) -> dict[str, Any]:
    result = run_classification(
        input_path=input_path,
        output_prefix=output_prefix,
        mode=mode,
        task=task,  # type: ignore[arg-type]
        threshold=threshold,
        allow_fallback=True,
        read_type=read_type,  # type: ignore[arg-type]
        min_length=min_length,
        coverage_source=coverage_source,  # type: ignore[arg-type]
        circularity_check=circularity_check,
        polish=polish,  # type: ignore[arg-type]
    )

    return {
        "job_id": job_id,
        "requested_mode": result.requested_mode,
        "used_mode": result.used_mode,
        "task": result.task,
        "fallback_reason": result.fallback_reason,
        "summary": result.summary,
        "artifacts": {
            "tsv": str(result.artifacts.tsv),
            "plasmids_fasta": str(result.artifacts.plasmids_fasta),
            "chromosomes_fasta": str(result.artifacts.chromosomes_fasta),
            "unclassified_fasta": str(result.artifacts.unclassified_fasta),
            "phage_fasta": str(result.artifacts.phage_fasta),
            "ambiguous_fasta": str(result.artifacts.ambiguous_fasta),
            "report_json": str(result.artifacts.report_json),
            "report_html": str(result.artifacts.report_html),
        },
    }
