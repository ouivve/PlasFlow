from __future__ import annotations

import gzip
from pathlib import Path

from plasflow_v2.io import read_fasta


def test_read_fasta_plain_and_gz(tmp_path: Path) -> None:
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">c1 some info\nACGTACGT\n>c2\nNNNACG\n", encoding="utf-8")

    gz_path = tmp_path / "sample.fasta.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(fasta.read_text(encoding="utf-8"))

    plain = read_fasta(fasta)
    gz = read_fasta(gz_path)

    assert [r.name for r in plain] == ["c1", "c2"]
    assert [r.length for r in plain] == [8, 6]
    assert [r.sequence for r in plain] == ["ACGTACGT", "NNNACG"]
    assert [r.name for r in gz] == ["c1", "c2"]
