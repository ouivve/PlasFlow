from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import shutil

from .constants import CoverageSource, PolishType, ReadType
from .io import ContigRecord, read_fasta


@dataclass(frozen=True)
class PreprocessConfig:
    min_length: int = 1000
    read_type: ReadType = "short"
    circularity_check: bool = True
    coverage_source: CoverageSource = "header"
    polish: PolishType = "none"


@dataclass
class PreprocessResult:
    records: list[ContigRecord]
    qc: dict[str, Any]
    circular_flags: dict[str, bool]
    coverage: dict[str, float | None]
    warnings: list[str]


_COVERAGE_PATTERNS = [
    re.compile(r"(?:^|[\s|;,_-])(?:cov|coverage|depth)\s*(?:=|:|_)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"(?:cov|coverage|depth)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def parse_coverage_from_header(header: str) -> float | None:
    text = str(header).strip()
    if not text:
        return None
    for pattern in _COVERAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
    return None


def is_circular_by_overlap(
    sequence: str,
    min_overlap: int = 40,
    max_mismatch_rate: float = 0.02,
) -> bool:
    seq = str(sequence).upper()
    n = len(seq)
    if n < 500:
        return False

    upper_overlap = min(250, n // 2)
    if upper_overlap < min_overlap:
        return False

    for overlap in range(upper_overlap, min_overlap - 1, -1):
        prefix = seq[:overlap]
        suffix = seq[-overlap:]
        mismatches = sum(1 for a, b in zip(prefix, suffix) if a != b)
        allowed = int(overlap * max_mismatch_rate)
        if mismatches <= allowed:
            return True
    return False


def _apply_optional_polish(records: list[ContigRecord], polish: PolishType, warnings: list[str]) -> None:
    if polish == "none":
        return
    tool = shutil.which(polish)
    if tool is None:
        warnings.append(f"Requested polish='{polish}' but tool is not installed. Skipping polish step.")
        return
    warnings.append(
        f"Requested polish='{polish}' and tool was found at '{tool}', "
        "but external polishing integration is deferred in this milestone. Skipping polish step."
    )


def run_preprocessing(input_path: str | Path, config: PreprocessConfig) -> PreprocessResult:
    if config.min_length < 1:
        raise ValueError("min_length must be >= 1")

    warnings: list[str] = []
    records = read_fasta(input_path)
    total_input = len(records)
    total_bases_input = sum(rec.length for rec in records)

    _apply_optional_polish(records, config.polish, warnings)

    filtered = [rec for rec in records if rec.length >= int(config.min_length)]
    removed_by_length = total_input - len(filtered)
    total_bases_retained = sum(rec.length for rec in filtered)

    circular_flags: dict[str, bool] = {}
    if config.circularity_check:
        for rec in filtered:
            circular_flags[rec.name] = is_circular_by_overlap(rec.sequence)
    else:
        circular_flags = {rec.name: False for rec in filtered}

    coverage: dict[str, float | None] = {}
    if config.coverage_source == "header":
        for rec in filtered:
            coverage[rec.name] = parse_coverage_from_header(rec.header)
    else:
        coverage = {rec.name: None for rec in filtered}

    if not filtered:
        warnings.append("No contigs left after preprocessing filters.")

    qc = {
        "read_type": config.read_type,
        "min_length": int(config.min_length),
        "coverage_source": config.coverage_source,
        "polish": config.polish,
        "circularity_check": bool(config.circularity_check),
        "input_contigs": total_input,
        "retained_contigs": len(filtered),
        "removed_contigs_by_length": removed_by_length,
        "input_bases": total_bases_input,
        "retained_bases": total_bases_retained,
        "retained_length_mean": (total_bases_retained / len(filtered)) if filtered else 0.0,
        "circular_contigs": sum(1 for value in circular_flags.values() if value),
        "coverage_observed_contigs": sum(1 for value in coverage.values() if value is not None),
    }

    return PreprocessResult(
        records=filtered,
        qc=qc,
        circular_flags=circular_flags,
        coverage=coverage,
        warnings=warnings,
    )
