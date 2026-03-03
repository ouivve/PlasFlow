from __future__ import annotations

from pathlib import Path
import json

from plasflow_v2.datasets import dataset_split_counts, load_dataset_rows


def test_dataset_manifest_loading_and_dedup(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    source.write_text(
        "sequence\tlabel\taccession\n"
        "ACGTACGT\tplasmid.other\tA1\n"
        "ACGTACGT\tplasmid.other\tA1\n"
        "GGGGCCCC\tchromosome.other\tA2\n",
        encoding="utf-8",
    )

    manifest = {
        "name": "demo",
        "random_seed": 42,
        "deduplicate": True,
        "split": {
            "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
            "group_col": "accession",
        },
        "sources": [
            {
                "name": "demo_source",
                "path": str(source),
                "format": "tsv",
                "sequence_col": "sequence",
                "label_col": "label",
            }
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = load_dataset_rows(manifest_path)
    assert len(rows) == 2
    assert {row.domain_label for row in rows} == {"plasmid", "chromosome"}

    counts = dataset_split_counts(rows)
    assert sum(counts.values()) == 2
