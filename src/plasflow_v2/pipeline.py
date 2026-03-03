from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from .classifier import ModernClassifier
from .constants import (
    CoverageSource,
    DEFAULT_THRESHOLD,
    ModeType,
    PolishType,
    ReadType,
    TaskType,
    load_task_label_spec,
    normalize_mode,
)
from .io import ContigRecord, write_fasta
from .legacy_runner import run_legacy_classifier
from .metrics import uncertainty_components
from .preprocessing import PreprocessConfig, PreprocessResult, run_preprocessing
from .reporting import build_summary, write_report_html, write_report_json


@dataclass
class ClassificationArtifacts:
    tsv: Path
    plasmids_fasta: Path
    chromosomes_fasta: Path
    unclassified_fasta: Path
    phage_fasta: Path
    ambiguous_fasta: Path
    report_json: Path
    report_html: Path


@dataclass
class ClassificationResult:
    requested_mode: ModeType
    used_mode: ModeType
    task: TaskType
    threshold: float
    artifacts: ClassificationArtifacts
    summary: dict[str, Any]
    fallback_reason: str | None


def normalize_output_prefix(output: str | Path) -> Path:
    raw = Path(output)
    if raw.suffix == ".tsv":
        return raw.with_suffix("")
    return raw


def artifact_paths(output_prefix: str | Path) -> ClassificationArtifacts:
    prefix = normalize_output_prefix(output_prefix)
    return ClassificationArtifacts(
        tsv=prefix.with_suffix(".tsv"),
        plasmids_fasta=Path(f"{prefix}_plasmids.fasta"),
        chromosomes_fasta=Path(f"{prefix}_chromosomes.fasta"),
        unclassified_fasta=Path(f"{prefix}_unclassified.fasta"),
        phage_fasta=Path(f"{prefix}_phage.fasta"),
        ambiguous_fasta=Path(f"{prefix}_ambiguous.fasta"),
        report_json=prefix.with_suffix(".report.json"),
        report_html=prefix.with_suffix(".report.html"),
    )


def _threshold_relabel(row: dict[str, Any], labels: list[str], threshold: float, task: TaskType) -> None:
    if task == "legacy28":
        label_name = str(row["label"])
        current_prob = float(row.get(label_name, 0.0))
        if current_prob >= threshold:
            return
        taxname = label_name.split(".", 1)[1] if "." in label_name else "unclassified"
        plasmids_sum = sum(float(row.get(col, 0.0)) for col in labels if col.startswith("plasmid."))
        chromosomes_sum = sum(float(row.get(col, 0.0)) for col in labels if col.startswith("chromosome."))
        tax_sum = sum(float(row.get(col, 0.0)) for col in labels if col.endswith(f".{taxname}"))

        if plasmids_sum > threshold:
            row["label"] = "plasmid.unclassified"
        elif chromosomes_sum > threshold:
            row["label"] = "chromosome.unclassified"
        elif tax_sum > threshold:
            row["label"] = f"unclassified.{taxname}"
        else:
            row["label"] = "unclassified.unclassified"
        return

    # binary/domain4: low-confidence predictions are forced to ambiguous.
    if float(row.get("max_probability", 0.0)) < threshold:
        row["label"] = "ambiguous"


def _to_row(
    rec: ContigRecord,
    pred_id: int,
    probs: list[float],
    labels: list[str],
    circular_flags: dict[str, bool],
    coverage: dict[str, float | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "contig_id": rec.contig_id,
        "contig_name": rec.name,
        "contig_length": rec.length,
        "id": pred_id,
        "label": labels[pred_id],
        "is_circular": bool(circular_flags.get(rec.name, False)),
        "coverage": coverage.get(rec.name),
    }
    for idx, label in enumerate(labels):
        row[label] = float(probs[idx])

    unc = uncertainty_components(probs)
    row.update(unc)
    return row


def _write_tsv(rows: list[dict[str, Any]], labels: list[str], output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "",
        "contig_id",
        "contig_name",
        "contig_length",
        "id",
        "label",
        "is_circular",
        "coverage",
        "max_prob",
        "margin",
        "entropy",
        "uncertainty_score",
        *labels,
    ]
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for idx, row in enumerate(rows):
            base = [
                idx,
                row["contig_id"],
                row["contig_name"],
                row["contig_length"],
                row["id"],
                row["label"],
                row.get("is_circular", False),
                row.get("coverage"),
                row.get("max_prob"),
                row.get("margin"),
                row.get("entropy"),
                row.get("uncertainty_score"),
            ]
            probs = [row.get(label, 0.0) for label in labels]
            writer.writerow(base + probs)


def _write_split_fastas(records: list[ContigRecord], rows: list[dict[str, Any]], artifacts: ClassificationArtifacts) -> None:
    by_name = {rec.name: rec for rec in records}
    label_by_name = {str(row["contig_name"]): str(row["label"]) for row in rows}

    plasmids: list[ContigRecord] = []
    chromosomes: list[ContigRecord] = []
    phage: list[ContigRecord] = []
    ambiguous: list[ContigRecord] = []
    unclassified: list[ContigRecord] = []

    for row in rows:
        name = str(row["contig_name"])
        label = str(row["label"]).lower()
        rec = by_name[name]
        if label.startswith("plasmid"):
            plasmids.append(rec)
        elif label.startswith("chromosome"):
            chromosomes.append(rec)
        elif label.startswith("phage"):
            phage.append(rec)
            unclassified.append(rec)
        elif label.startswith("ambiguous") or label.startswith("unclassified"):
            ambiguous.append(rec)
            unclassified.append(rec)
        else:
            ambiguous.append(rec)
            unclassified.append(rec)

    write_fasta(plasmids, artifacts.plasmids_fasta, append_label=label_by_name)
    write_fasta(chromosomes, artifacts.chromosomes_fasta, append_label=label_by_name)
    write_fasta(unclassified, artifacts.unclassified_fasta, append_label=label_by_name)
    write_fasta(phage, artifacts.phage_fasta, append_label=label_by_name)
    write_fasta(ambiguous, artifacts.ambiguous_fasta, append_label=label_by_name)


def _run_modern(
    preprocessed: PreprocessResult,
    output_prefix: Path,
    threshold: float,
    task: TaskType,
    requested_mode: ModeType,
    fallback_reason: str | None,
) -> ClassificationResult:
    records = preprocessed.records
    label_spec = load_task_label_spec(task)
    classifier = ModernClassifier(task=task)
    prediction = classifier.predict(records, label_spec)
    model_metrics = classifier.model_metrics()

    rows = []
    for rec, pred_id, probs in zip(records, prediction.predicted_ids, prediction.probabilities):
        row = _to_row(
            rec=rec,
            pred_id=pred_id,
            probs=probs,
            labels=label_spec.labels,
            circular_flags=preprocessed.circular_flags,
            coverage=preprocessed.coverage,
        )
        _threshold_relabel(row, label_spec.labels, threshold, task=task)
        rows.append(row)

    artifacts = artifact_paths(output_prefix)
    _write_tsv(rows, label_spec.labels, artifacts.tsv)
    _write_split_fastas(records, rows, artifacts)

    summary = build_summary(
        rows=rows,
        labels=label_spec.labels,
        threshold=threshold,
        requested_mode=requested_mode,
        used_mode="v2",
        fallback_reason=fallback_reason,
        metrics=model_metrics,
        task=task,
        preprocessing={
            "read_type": preprocessed.qc.get("read_type"),
            "min_length": preprocessed.qc.get("min_length"),
            "coverage_source": preprocessed.qc.get("coverage_source"),
            "circularity_check": preprocessed.qc.get("circularity_check"),
            "polish": preprocessed.qc.get("polish"),
        },
        qc=preprocessed.qc,
        warnings=preprocessed.warnings,
    )
    write_report_json(summary, artifacts.report_json)
    write_report_html(summary, artifacts.report_html)

    return ClassificationResult(
        requested_mode=requested_mode,
        used_mode="v2",
        task=task,
        threshold=threshold,
        artifacts=artifacts,
        summary=summary,
        fallback_reason=fallback_reason,
    )


def _write_preprocessed_fasta(records: list[ContigRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(records, output_path, append_label=None)


def run_classification(
    input_path: str | Path,
    output_prefix: str | Path,
    mode: str = "v1",
    task: TaskType = "legacy28",
    threshold: float = DEFAULT_THRESHOLD,
    allow_fallback: bool = True,
    read_type: ReadType = "short",
    min_length: int = 1000,
    coverage_source: CoverageSource = "header",
    circularity_check: bool = True,
    polish: PolishType = "none",
) -> ClassificationResult:
    input_file = Path(input_path)
    prefix = normalize_output_prefix(output_prefix)
    canonical_mode = normalize_mode(mode)

    if task not in {"legacy28", "binary_domain", "domain4"}:
        raise ValueError("task must be one of: legacy28, binary_domain, domain4")

    if canonical_mode == "v1" and task != "legacy28":
        raise ValueError("mode=v1 supports only task=legacy28")

    preprocess_cfg = PreprocessConfig(
        min_length=min_length,
        read_type=read_type,
        circularity_check=circularity_check,
        coverage_source=coverage_source,
        polish=polish,
    )
    preprocessed = run_preprocessing(input_file, preprocess_cfg)

    if canonical_mode == "v2":
        return _run_modern(
            preprocessed=preprocessed,
            output_prefix=prefix,
            threshold=threshold,
            task=task,
            requested_mode=canonical_mode,
            fallback_reason=None,
        )

    # v1 mode: run legacy on preprocessed FASTA.
    artifacts = artifact_paths(prefix)
    preprocessed_fasta = prefix.with_suffix(".preprocessed.fasta")
    _write_preprocessed_fasta(preprocessed.records, preprocessed_fasta)
    legacy_res = run_legacy_classifier(
        input_path=preprocessed_fasta,
        output_tsv=artifacts.tsv,
        threshold=threshold,
    )

    if legacy_res.ok:
        rows: list[dict[str, Any]] = []
        with artifacts.tsv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                normalized = dict(row)
                normalized.pop("", None)
                rows.append(normalized)

        label_spec = load_task_label_spec("legacy28")
        summary = build_summary(
            rows=rows,
            labels=label_spec.labels,
            threshold=threshold,
            requested_mode=canonical_mode,
            used_mode="v1",
            fallback_reason=None,
            metrics=None,
            task=task,
            preprocessing={
                "read_type": preprocessed.qc.get("read_type"),
                "min_length": preprocessed.qc.get("min_length"),
                "coverage_source": preprocessed.qc.get("coverage_source"),
                "circularity_check": preprocessed.qc.get("circularity_check"),
                "polish": preprocessed.qc.get("polish"),
            },
            qc=preprocessed.qc,
            warnings=preprocessed.warnings,
        )
        write_report_json(summary, artifacts.report_json)
        write_report_html(summary, artifacts.report_html)

        return ClassificationResult(
            requested_mode=canonical_mode,
            used_mode="v1",
            task=task,
            threshold=threshold,
            artifacts=artifacts,
            summary=summary,
            fallback_reason=None,
        )

    if not allow_fallback:
        raise RuntimeError(legacy_res.reason or "v1 mode failed")

    return _run_modern(
        preprocessed=preprocessed,
        output_prefix=prefix,
        threshold=threshold,
        task="legacy28",
        requested_mode=canonical_mode,
        fallback_reason=legacy_res.reason,
    )
