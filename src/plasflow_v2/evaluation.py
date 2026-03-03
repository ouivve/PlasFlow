from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from .classifier import ModernClassifier
from .constants import DEFAULT_THRESHOLD, TaskType, load_task_label_spec
from .io import ContigRecord
from .metrics import (
    aggregate_calibration_metrics,
    best_threshold_by_macro_f1,
    binary_domain_from_label,
    domain4_from_label,
    domain4_metrics,
    evaluate_binary_predictions,
    expected_calibration_error_multiclass,
    predict_with_threshold,
    uncertainty_components,
)


def _resolve_bundle_dir(model_dir: str | Path) -> Path:
    candidate = Path(model_dir)
    if (candidate / "domain_model.joblib").exists():
        return candidate
    if (candidate / "current" / "domain_model.joblib").exists():
        return candidate / "current"
    if (candidate / "model.joblib").exists():
        return candidate
    if (candidate / "current" / "model.joblib").exists():
        return candidate / "current"
    raise FileNotFoundError(f"No model bundle found under: {candidate}")


def _load_labeled_sequences(
    input_tsv: Path,
    sequence_col: str,
    label_col: str,
    name_col: str,
    task: TaskType,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader):
            sequence = (row.get(sequence_col) or "").strip().upper()
            label = (row.get(label_col) or "").strip()
            if not sequence or not label:
                continue
            if task == "binary_domain" and binary_domain_from_label(label) is None:
                continue
            name = (row.get(name_col) or "").strip() or f"eval_{idx}"
            rows.append({"name": name, "sequence": sequence, "label": label})
    return rows


def _write_confusion_matrix_csv(
    confusion_matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for row_idx, label in enumerate(labels):
            writer.writerow([label, *confusion_matrix[row_idx]])


def _write_threshold_curve_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "macro_f1", "precision_macro", "recall_macro", "accuracy"])
        for row in rows:
            writer.writerow(
                [
                    row["threshold"],
                    row["macro_f1"],
                    row["precision_macro"],
                    row["recall_macro"],
                    row["accuracy"],
                ]
            )


def _domain4_threshold_curve(
    y_true: list[str],
    probabilities: list[list[float]],
    labels: list[str],
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    if step <= 0:
        raise ValueError("step must be > 0")

    curve: list[dict[str, float]] = []
    t = start
    while t <= end + 1e-9:
        preds: list[str] = []
        for row in probabilities:
            best_idx = max(range(len(row)), key=lambda i: row[i])
            best_prob = float(row[best_idx])
            pred = labels[best_idx]
            if best_prob < t:
                pred = "ambiguous"
            preds.append(pred)
        metrics = domain4_metrics(y_true, preds)
        curve.append(
            {
                "threshold": round(t, 4),
                "macro_f1": float(metrics["macro_f1"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
        t += step

    if not curve:
        return DEFAULT_THRESHOLD, {"macro_f1": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "accuracy": 0.0}, []
    best = max(curve, key=lambda row: (row["macro_f1"], -abs(row["threshold"] - DEFAULT_THRESHOLD)))
    return float(best["threshold"]), best, curve


def _uncertainty_summary(probabilities: list[list[float]]) -> dict[str, float]:
    if not probabilities:
        return {
            "mean_max_prob": 0.0,
            "mean_margin": 0.0,
            "mean_entropy": 0.0,
            "mean_uncertainty_score": 0.0,
        }
    scores = [uncertainty_components(row) for row in probabilities]
    total = len(scores)
    return {
        "mean_max_prob": sum(row["max_prob"] for row in scores) / total,
        "mean_margin": sum(row["margin"] for row in scores) / total,
        "mean_entropy": sum(row["entropy"] for row in scores) / total,
        "mean_uncertainty_score": sum(row["uncertainty_score"] for row in scores) / total,
    }


def _normalize_task(task: str) -> TaskType:
    if task not in {"binary_domain", "domain4"}:
        raise ValueError("Only task=binary_domain|domain4 is supported")
    return task  # type: ignore[return-value]


def evaluate_modern_model(
    input_tsv: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    task: str = "binary_domain",
    sequence_col: str = "sequence",
    label_col: str = "label",
    name_col: str = "contig_name",
    threshold: float | None = None,
) -> dict[str, Any]:
    task_t = _normalize_task(task)
    rows = _load_labeled_sequences(
        input_tsv=Path(input_tsv),
        sequence_col=sequence_col,
        label_col=label_col,
        name_col=name_col,
        task=task_t,
    )
    if not rows:
        raise ValueError("No labeled sequences found in input TSV")

    bundle_dir = _resolve_bundle_dir(model_dir)
    classifier = ModernClassifier(bundle_dir=bundle_dir, task=task_t)
    label_spec = load_task_label_spec(task_t)

    records = [
        ContigRecord(contig_id=index, name=row["name"], sequence=row["sequence"])
        for index, row in enumerate(rows)
    ]
    prediction = classifier.predict(records, label_spec)
    recommended = classifier.recommended_threshold() or DEFAULT_THRESHOLD
    chosen_threshold = float(threshold) if threshold is not None else float(recommended)

    uncertainty = _uncertainty_summary(prediction.probabilities)

    if task_t == "binary_domain":
        if "plasmid" not in label_spec.label_to_id:
            raise ValueError("binary task label spec must contain 'plasmid'")
        p_idx = label_spec.label_to_id["plasmid"]
        p_plasmid = [float(row[p_idx]) for row in prediction.probabilities]
        y_true = [binary_domain_from_label(row["label"]) or "chromosome" for row in rows]
        y_pred = predict_with_threshold(p_plasmid, chosen_threshold)
        metrics = evaluate_binary_predictions(y_true, y_pred, p_plasmid)
        best_threshold, best_row, threshold_rows = best_threshold_by_macro_f1(y_true, p_plasmid)
        metrics_payload = {
            "binary_domain": {
                "macro_f1": metrics["macro_f1"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "accuracy": metrics["accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
                "support": metrics["support"],
            },
            "calibration": {
                **aggregate_calibration_metrics(y_true, p_plasmid),
                "recommended_threshold": recommended,
            },
        }
        matrix_labels = ["plasmid", "chromosome"]
    else:
        y_true = [domain4_from_label(row["label"]) for row in rows]
        y_pred: list[str] = []
        for probs in prediction.probabilities:
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            best_prob = float(probs[best_idx])
            pred = label_spec.labels[best_idx]
            if best_prob < chosen_threshold:
                pred = "ambiguous"
            y_pred.append(pred)
        metrics = domain4_metrics(y_true, y_pred)
        best_threshold, best_row, threshold_rows = _domain4_threshold_curve(
            y_true=y_true,
            probabilities=prediction.probabilities,
            labels=label_spec.labels,
        )
        metrics_payload = {
            "domain4": {
                "macro_f1": metrics["macro_f1"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "accuracy": metrics["accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
                "support": metrics["support"],
            },
            "calibration": {
                "ece": expected_calibration_error_multiclass(
                    y_true,
                    prediction.probabilities,
                    labels=label_spec.labels,
                    mapper=None,
                ),
                "brier_score": None,
                "recommended_threshold": recommended,
            },
        }
        matrix_labels = label_spec.labels

    payload = {
        "task": task_t,
        "model_dir": str(bundle_dir),
        "input_tsv": str(input_tsv),
        "support": len(y_true),
        "threshold_used": chosen_threshold,
        "recommended_threshold_from_model": recommended,
        "best_threshold_from_eval": best_threshold,
        "best_threshold_row": best_row,
        "metrics": metrics_payload,
        "uncertainty_summary": uncertainty,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    confusion = (
        metrics_payload.get("binary_domain", {}).get("confusion_matrix")
        or metrics_payload.get("domain4", {}).get("confusion_matrix")
    )
    _write_confusion_matrix_csv(confusion, matrix_labels, out.with_name("confusion_matrix.csv"))
    _write_threshold_curve_csv(threshold_rows, out.with_name("threshold_curve.csv"))

    payload["artifacts"] = {
        "eval_json": str(out),
        "confusion_matrix_csv": str(out.with_name("confusion_matrix.csv")),
        "threshold_curve_csv": str(out.with_name("threshold_curve.csv")),
    }
    return payload
