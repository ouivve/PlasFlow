from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import gzip


@dataclass
class ContigRecord:
    contig_id: int
    name: str
    sequence: str
    header: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)


def open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_fasta(path: str | Path) -> list[ContigRecord]:
    p = Path(path)
    records: list[ContigRecord] = []
    header: str | None = None
    seq_chunks: list[str] = []

    with open_text_auto(p) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):  # Start of a new FASTA entry.
                if header is not None:
                    header_name = header.split()[0]
                    records.append(
                        ContigRecord(
                            contig_id=len(records),
                            name=header_name,
                            sequence="".join(seq_chunks).upper(),
                            header=header,
                        )
                    )
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        header_name = header.split()[0]
        records.append(
            ContigRecord(
                contig_id=len(records),
                name=header_name,
                sequence="".join(seq_chunks).upper(),
                header=header,
            )
        )

    return records


def write_fasta(records: list[ContigRecord], path: str | Path, append_label: dict[str, str] | None = None) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as handle:
        for rec in records:
            label = ""
            if append_label and rec.name in append_label:
                label = f" {append_label[rec.name]}"
            handle.write(f">{rec.name}{label}\n")
            sequence = rec.sequence
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i:i + 80] + "\n")


def index_by_name(records: list[ContigRecord]) -> dict[str, ContigRecord]:
    return {rec.name: rec for rec in records}
