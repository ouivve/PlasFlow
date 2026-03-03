from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any
import csv
import json

from .metrics import binary_domain_from_label, domain4_from_label


VALID_SPLITS = {"train", "val", "test"}


@dataclass
class DatasetRow:
    sequence: str
    label: str
    domain_label: str
    split: str
    group_id: str
    source: str


def _load_manifest(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "YAML manifest requires PyYAML. Install extras: pip install 'plasflow-v2[train]'"
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("dataset manifest must be a JSON/YAML object")
        return data

    raise ValueError("dataset-manifest must be .json/.yaml/.yml")


def _as_ratio_split(split_cfg: dict[str, Any]) -> tuple[float, float, float]:
    ratios = split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    train = float(ratios.get("train", 0.8))
    val = float(ratios.get("val", 0.1))
    test = float(ratios.get("test", 0.1))
    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios sum must be > 0")
    return train / total, val / total, test / total


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in VALID_SPLITS:
        return lowered
    return None


def _load_source_rows(source: dict[str, Any], root: Path) -> list[dict[str, str]]:
    source_name = str(source.get("name", "source"))
    path_value = source.get("path")
    if not path_value:
        raise ValueError(f"dataset source '{source_name}' is missing path")

    path = (root / str(path_value)).resolve() if not Path(str(path_value)).is_absolute() else Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"dataset source not found: {path}")

    fmt = str(source.get("format", "tsv")).lower()
    if fmt != "tsv":
        raise ValueError(f"dataset source format '{fmt}' is unsupported (only tsv)")

    sequence_col = str(source.get("sequence_col", "sequence"))
    label_col = str(source.get("label_col", "label"))

    out: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq = (row.get(sequence_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if not seq or not label:
                continue
            out.append(dict(row))
    return out


def load_dataset_rows(dataset_manifest: str | Path) -> list[DatasetRow]:
    manifest_path = Path(dataset_manifest)
    manifest = _load_manifest(manifest_path)

    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset manifest must include a non-empty 'sources' list")

    split_cfg = manifest.get("split", {}) if isinstance(manifest.get("split", {}), dict) else {}
    train_r, val_r, test_r = _as_ratio_split(split_cfg)
    split_col_default = split_cfg.get("split_col")
    group_col_default = split_cfg.get("group_col")
    seed = int(manifest.get("random_seed", 42))
    min_length = int(manifest.get("min_length", 1))
    deduplicate = bool(manifest.get("deduplicate", True))

    rows: list[DatasetRow] = []
    seen_sequences: set[str] = set()

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source in dataset manifest must be an object")

        source_name = str(source.get("name", "source"))
        source_rows = _load_source_rows(source, manifest_path.parent)
        sequence_col = str(source.get("sequence_col", "sequence"))
        label_col = str(source.get("label_col", "label"))
        split_col = source.get("split_col", split_col_default)
        group_col = source.get("group_col", group_col_default)
        forced_split = _normalize_split(source.get("split"))

        for raw in source_rows:
            sequence = (raw.get(sequence_col) or "").strip().upper()
            label = (raw.get(label_col) or "").strip()
            if len(sequence) < min_length:
                continue
            if deduplicate and sequence in seen_sequences:
                continue

            domain = binary_domain_from_label(label) or domain4_from_label(label)

            split = forced_split
            if split is None and split_col:
                split = _normalize_split(raw.get(str(split_col)))

            group_id = ""
            if group_col:
                group_id = (raw.get(str(group_col)) or "").strip()
            if not group_id:
                group_id = md5(sequence.encode("utf-8")).hexdigest()[:16]

            if split is None:
                bucket_raw = f"{seed}:{group_id}".encode("utf-8")
                bucket = int(md5(bucket_raw).hexdigest()[:8], 16) / 0xFFFFFFFF
                if bucket < train_r:
                    split = "train"
                elif bucket < train_r + val_r:
                    split = "val"
                else:
                    split = "test"

            rows.append(
                DatasetRow(
                    sequence=sequence,
                    label=label,
                    domain_label=domain,
                    split=split,
                    group_id=group_id,
                    source=source_name,
                )
            )
            if deduplicate:
                seen_sequences.add(sequence)

    if not rows:
        raise ValueError("no valid rows were produced from dataset manifest")

    return rows


def dataset_split_counts(rows: list[DatasetRow]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        if row.split in counts:
            counts[row.split] += 1
    return counts


def write_rows_as_tsv(rows: list[DatasetRow], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sequence", "label", "domain_label", "split", "group_id", "source"])
        for row in rows:
            writer.writerow([row.sequence, row.label, row.domain_label, row.split, row.group_id, row.source])
