from __future__ import annotations

from pathlib import Path
import pytest

from plasflow_v2.pipeline import run_classification


def _sample_fasta(path: Path) -> None:
    path.write_text(
        ">contig_A\n" + "ACGT" * 300 + "\n"
        ">contig_B\n" + "GGGCGCGCGC" * 120 + "\n"
        ">contig_C\n" + "ATATATATAT" * 120 + "\n",
        encoding="utf-8",
    )


def test_classify_v2_outputs_artifacts(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    _sample_fasta(fasta)

    result = run_classification(
        input_path=fasta,
        output_prefix=tmp_path / "result",
        mode="v2",
        task="legacy28",
        threshold=0.7,
    )

    assert result.used_mode == "v2"
    assert result.artifacts.tsv.exists()
    assert result.artifacts.plasmids_fasta.exists()
    assert result.artifacts.chromosomes_fasta.exists()
    assert result.artifacts.unclassified_fasta.exists()
    assert result.artifacts.phage_fasta.exists()
    assert result.artifacts.ambiguous_fasta.exists()
    assert result.artifacts.report_json.exists()
    assert result.artifacts.report_html.exists()

    first_line = result.artifacts.tsv.read_text(encoding="utf-8").splitlines()[0]
    assert "contig_name" in first_line
    assert "label" in first_line


def test_classify_v1_with_fallback(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    _sample_fasta(fasta)

    result = run_classification(
        input_path=fasta,
        output_prefix=tmp_path / "v1_result",
        mode="v1",
        task="legacy28",
        threshold=0.7,
        allow_fallback=True,
    )

    assert result.used_mode in {"v1", "v2"}
    if result.used_mode == "v2":
        assert result.fallback_reason is not None


def test_domain4_requires_bundle_without_heuristic_fallback(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    _sample_fasta(fasta)
    with pytest.raises(FileNotFoundError):
        run_classification(
            input_path=fasta,
            output_prefix=tmp_path / "domain4_result",
            mode="v2",
            task="domain4",
            threshold=0.7,
        )
