from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import csv

DEFAULT_THRESHOLD = 0.7
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
LEGACY_SCRIPT_DEFAULT = PROJECT_ROOT / "PlasFlow.py"
LEGACY_MODELS_DEFAULT = PROJECT_ROOT / "models"
DEFAULT_LABELS_FILE = LEGACY_MODELS_DEFAULT / "class_labels_df.tsv"
DEFAULT_MODELS_V2_DIR = PROJECT_ROOT / "models_v2"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

ReadType = Literal["short", "long", "hybrid", "complete"]
CoverageSource = Literal["header", "none"]
PolishType = Literal["none", "racon", "medaka"]
TaskType = Literal["legacy28", "binary_domain", "domain4"]
ModeType = Literal["v1", "v2"]

BINARY_DOMAIN_LABELS: list[str] = ["plasmid", "chromosome"]
DOMAIN4_LABELS: list[str] = ["plasmid", "chromosome", "phage", "ambiguous"]


# Fallback list if labels file is missing.
FALLBACK_LABELS: list[str] = [
    "chromosome.Acidobacteria",
    "chromosome.Actinobacteria",
    "chromosome.Bacteroidetes",
    "chromosome.Chlamydiae",
    "chromosome.Chlorobi",
    "chromosome.Chloroflexi",
    "chromosome.Cyanobacteria",
    "chromosome.DeinococcusThermus",
    "chromosome.Firmicutes",
    "chromosome.Fusobacteria",
    "chromosome.Nitrospirae",
    "chromosome.other",
    "chromosome.Planctomycetes",
    "chromosome.Proteobacteria",
    "chromosome.Spirochaetes",
    "chromosome.Tenericutes",
    "chromosome.Thermotogae",
    "chromosome.Verrucomicrobia",
    "plasmid.Actinobacteria",
    "plasmid.Bacteroidetes",
    "plasmid.Chlamydiae",
    "plasmid.Cyanobacteria",
    "plasmid.DeinococcusThermus",
    "plasmid.Firmicutes",
    "plasmid.Fusobacteria",
    "plasmid.other",
    "plasmid.Proteobacteria",
    "plasmid.Spirochaetes",
]


@dataclass(frozen=True)
class LabelSpec:
    id_to_label: dict[int, str]
    labels: list[str]

    @property
    def label_to_id(self) -> dict[str, int]:
        return {v: k for k, v in self.id_to_label.items()}

    @property
    def taxons(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for label in self.labels:
            tax = label.split(".", 1)[1]
            if tax not in seen:
                seen.add(tax)
                ordered.append(tax)
        return ordered


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    return value


def load_label_spec(labels_file: Path | None = None) -> LabelSpec:
    path = labels_file or DEFAULT_LABELS_FILE
    if not path.exists():
        mapping = {idx: label for idx, label in enumerate(FALLBACK_LABELS)}
        return LabelSpec(id_to_label=mapping, labels=FALLBACK_LABELS[:])

    id_to_label: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            idx = int(_clean_value(row["id"]))
            label = _clean_value(row["label"])
            id_to_label[idx] = label

    if not id_to_label:
        mapping = {idx: label for idx, label in enumerate(FALLBACK_LABELS)}
        return LabelSpec(id_to_label=mapping, labels=FALLBACK_LABELS[:])

    labels = [label for _, label in sorted(id_to_label.items(), key=lambda x: x[0])]
    return LabelSpec(id_to_label=id_to_label, labels=labels)


def plasmid_labels(labels: Iterable[str]) -> list[str]:
    return [label for label in labels if label.startswith("plasmid.")]


def chromosome_labels(labels: Iterable[str]) -> list[str]:
    return [label for label in labels if label.startswith("chromosome.")]


def normalize_mode(mode: str) -> ModeType:
    raw = str(mode).strip().lower()
    if raw in {"v1", "legacy"}:
        return "v1"
    if raw in {"v2", "modern"}:
        return "v2"
    raise ValueError("mode must be one of: v1, v2 (legacy/modern aliases are still accepted)")


def load_task_label_spec(task: TaskType, labels_file: Path | None = None) -> LabelSpec:
    if task == "legacy28":
        return load_label_spec(labels_file=labels_file)
    if task == "binary_domain":
        mapping = {idx: label for idx, label in enumerate(BINARY_DOMAIN_LABELS)}
        return LabelSpec(id_to_label=mapping, labels=BINARY_DOMAIN_LABELS[:])
    if task == "domain4":
        mapping = {idx: label for idx, label in enumerate(DOMAIN4_LABELS)}
        return LabelSpec(id_to_label=mapping, labels=DOMAIN4_LABELS[:])
    raise ValueError(f"Unsupported task: {task}")
