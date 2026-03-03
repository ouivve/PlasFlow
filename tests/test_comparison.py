from __future__ import annotations

from pathlib import Path

from plasflow_v2.comparison import compare_modes


def test_compare_modes_with_ground_truth(tmp_path: Path) -> None:
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(
        ">c1\n" + "ACGT" * 100 + "\n" +
        ">c2\n" + "GGCC" * 120 + "\n",
        encoding="utf-8",
    )

    ground_truth = tmp_path / "ground_truth.tsv"
    ground_truth.write_text(
        "contig_name\tlabel\n"
        "c1\tplasmid.other\n"
        "c2\tchromosome.other\n",
        encoding="utf-8",
    )

    result = compare_modes(
        input_path=fasta,
        output_dir=tmp_path / "compare",
        threshold=0.7,
        ground_truth=ground_truth,
    )

    assert "runs" in result
    assert "v2" in result["runs"]
    assert result["runs"]["v2"]["ok"] is True
    assert "ground_truth_eval" in result
    assert "v2" in result["ground_truth_eval"]
    assert result["ground_truth_eval"]["v2"]["common_contigs"] == 2
