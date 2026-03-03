from __future__ import annotations

from pathlib import Path

from plasflow_v2.preprocessing import (
    PreprocessConfig,
    is_circular_by_overlap,
    parse_coverage_from_header,
    run_preprocessing,
)


def test_parse_coverage_from_header_patterns() -> None:
    assert parse_coverage_from_header("contig_1 cov=18.2") == 18.2
    assert parse_coverage_from_header("contig_2 coverage:42.0") == 42.0
    assert parse_coverage_from_header("contig_3 depth_7.5") == 7.5
    assert parse_coverage_from_header("contig_without_cov") is None


def test_circularity_overlap_detection_boundaries() -> None:
    prefix = "ACGT" * 12
    middle = "TTAACCGG" * 80
    seq_circular = prefix + middle + prefix
    assert is_circular_by_overlap(seq_circular, min_overlap=40, max_mismatch_rate=0.02) is True

    seq_not_circular = ("ACGT" * 120) + ("TTTT" * 20)
    assert is_circular_by_overlap(seq_not_circular, min_overlap=40, max_mismatch_rate=0.0) is False


def test_run_preprocessing_filters_and_collects_qc(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(
        ">c1 cov=20.5\n" + "ACGT" * 400 + "\n"
        ">c2 depth:5\n" + "AT" * 100 + "\n",
        encoding="utf-8",
    )
    result = run_preprocessing(
        fasta,
        PreprocessConfig(min_length=1000, read_type="long", coverage_source="header", circularity_check=True),
    )
    assert len(result.records) == 1
    assert result.records[0].name == "c1"
    assert result.coverage["c1"] == 20.5
    assert result.qc["removed_contigs_by_length"] == 1
    assert result.qc["read_type"] == "long"
