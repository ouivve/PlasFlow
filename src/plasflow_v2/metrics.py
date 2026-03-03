from __future__ import annotations

import math
from typing import Any


BINARY_DOMAIN_LABELS = ("plasmid", "chromosome")
DOMAIN4_LABELS = ("plasmid", "chromosome", "phage", "ambiguous")


def binary_domain_from_label(label: str) -> str | None:
    normalized = str(label).strip().lower()
    if normalized.startswith("plasmid"):
        return "plasmid"
    if normalized.startswith("chromosome"):
        return "chromosome"
    return None


def domain4_from_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized.startswith("plasmid"):
        return "plasmid"
    if normalized.startswith("chromosome"):
        return "chromosome"
    if normalized.startswith("phage"):
        return "phage"
    if normalized.startswith("ambiguous") or normalized.startswith("unclassified"):
        return "ambiguous"
    return "ambiguous"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def confusion_matrix_labels(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | tuple[str, ...],
    mapper: Any | None = None,
) -> list[list[int]]:
    index = {name: i for i, name in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        t = mapper(truth) if mapper else str(truth).strip().lower()
        p = mapper(pred) if mapper else str(pred).strip().lower()
        if t is None or p is None:
            continue
        if t not in index or p not in index:
            continue
        matrix[index[t]][index[p]] += 1
    return matrix


def _per_class_metrics(matrix: list[list[int]], class_index: int) -> dict[str, float]:
    tp = float(matrix[class_index][class_index])
    fp = float(sum(matrix[row][class_index] for row in range(len(matrix)) if row != class_index))
    fn = float(sum(matrix[class_index][col] for col in range(len(matrix[class_index])) if col != class_index))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def multiclass_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | tuple[str, ...],
    mapper: Any | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix_labels(y_true, y_pred, labels=labels, mapper=mapper)
    per_class = {
        class_name: _per_class_metrics(matrix, class_index)
        for class_index, class_name in enumerate(labels)
    }

    precision_macro = sum(per_class[name]["precision"] for name in labels) / len(labels)
    recall_macro = sum(per_class[name]["recall"] for name in labels) / len(labels)
    f1_macro = sum(per_class[name]["f1"] for name in labels) / len(labels)

    total = sum(sum(row) for row in matrix)
    accuracy = _safe_div(float(sum(matrix[i][i] for i in range(len(matrix)))), float(total))

    return {
        "labels": list(labels),
        "support": total,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": f1_macro,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def confusion_matrix_binary(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    return confusion_matrix_labels(
        y_true,
        y_pred,
        labels=BINARY_DOMAIN_LABELS,
        mapper=binary_domain_from_label,
    )


def binary_domain_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return multiclass_metrics(
        y_true,
        y_pred,
        labels=BINARY_DOMAIN_LABELS,
        mapper=binary_domain_from_label,
    )


def domain4_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return multiclass_metrics(
        y_true,
        y_pred,
        labels=DOMAIN4_LABELS,
        mapper=domain4_from_label,
    )


def brier_score_binary(y_true: list[str], p_plasmid: list[float]) -> float:
    if not y_true:
        return 0.0
    values: list[float] = []
    for label, prob in zip(y_true, p_plasmid):
        truth = 1.0 if binary_domain_from_label(label) == "plasmid" else 0.0
        values.append((truth - float(prob)) ** 2)
    return sum(values) / len(values) if values else 0.0


def expected_calibration_error_binary(y_true: list[str], p_plasmid: list[float], bins: int = 10) -> float:
    if not y_true or bins <= 0:
        return 0.0

    bucket_total = [0 for _ in range(bins)]
    bucket_prob = [0.0 for _ in range(bins)]
    bucket_truth = [0.0 for _ in range(bins)]

    for label, prob in zip(y_true, p_plasmid):
        prob = min(max(float(prob), 0.0), 1.0)
        idx = min(int(prob * bins), bins - 1)
        bucket_total[idx] += 1
        bucket_prob[idx] += prob
        bucket_truth[idx] += 1.0 if binary_domain_from_label(label) == "plasmid" else 0.0

    total = sum(bucket_total)
    if total == 0:
        return 0.0

    ece = 0.0
    for idx in range(bins):
        count = bucket_total[idx]
        if count == 0:
            continue
        avg_prob = bucket_prob[idx] / count
        avg_truth = bucket_truth[idx] / count
        ece += (count / total) * abs(avg_truth - avg_prob)
    return ece


def predict_with_threshold(p_plasmid: list[float], threshold: float) -> list[str]:
    return ["plasmid" if float(prob) >= threshold else "chromosome" for prob in p_plasmid]


def threshold_curve(
    y_true: list[str],
    p_plasmid: list[float],
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    if step <= 0:
        raise ValueError("step must be > 0")

    t = start
    while t <= end + 1e-9:
        preds = predict_with_threshold(p_plasmid, t)
        metrics = binary_domain_metrics(y_true, preds)
        out.append(
            {
                "threshold": round(t, 4),
                "macro_f1": float(metrics["macro_f1"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
        t += step

    return out


def best_threshold_by_macro_f1(
    y_true: list[str],
    p_plasmid: list[float],
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
    default_threshold: float = 0.7,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    curve = threshold_curve(y_true, p_plasmid, start=start, end=end, step=step)
    if not curve:
        return default_threshold, {"macro_f1": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "accuracy": 0.0}, []

    best = max(curve, key=lambda row: (row["macro_f1"], -abs(row["threshold"] - default_threshold)))
    return float(best["threshold"]), best, curve


def aggregate_calibration_metrics(y_true: list[str], p_plasmid: list[float]) -> dict[str, float]:
    return {
        "ece": expected_calibration_error_binary(y_true, p_plasmid),
        "brier_score": brier_score_binary(y_true, p_plasmid),
    }


def expected_calibration_error_multiclass(
    y_true: list[str],
    probabilities: list[list[float]],
    labels: list[str] | tuple[str, ...] = DOMAIN4_LABELS,
    mapper: Any | None = domain4_from_label,
    bins: int = 10,
) -> float:
    if not y_true or not probabilities or bins <= 0:
        return 0.0

    correct_sum = [0.0 for _ in range(bins)]
    conf_sum = [0.0 for _ in range(bins)]
    count = [0 for _ in range(bins)]
    label_order = list(labels)

    for truth, probs in zip(y_true, probabilities):
        if not probs:
            continue
        clipped = [min(max(float(x), 0.0), 1.0) for x in probs]
        pred_idx = max(range(len(clipped)), key=lambda idx: clipped[idx])
        confidence = clipped[pred_idx]
        pred_label = label_order[pred_idx] if pred_idx < len(label_order) else None
        mapped_truth = mapper(truth) if mapper else truth
        is_correct = 1.0 if pred_label == mapped_truth else 0.0
        bucket = min(int(confidence * bins), bins - 1)
        count[bucket] += 1
        conf_sum[bucket] += confidence
        correct_sum[bucket] += is_correct

    total = sum(count)
    if total == 0:
        return 0.0

    ece = 0.0
    for idx in range(bins):
        if count[idx] == 0:
            continue
        avg_conf = conf_sum[idx] / count[idx]
        avg_acc = correct_sum[idx] / count[idx]
        ece += (count[idx] / total) * abs(avg_acc - avg_conf)
    return ece


def uncertainty_components(probabilities: list[float]) -> dict[str, float]:
    if not probabilities:
        return {
            "max_prob": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "uncertainty_score": 1.0,
        }

    clipped = [max(float(p), 1e-12) for p in probabilities]
    total = sum(clipped)
    norm = [p / total for p in clipped]
    ordered = sorted(norm, reverse=True)
    max_prob = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    margin = max_prob - second

    entropy = -sum(p * math.log(p) for p in norm)
    max_entropy = math.log(len(norm)) if len(norm) > 1 else 1.0
    entropy_norm = min(max(entropy / max_entropy, 0.0), 1.0)

    score = 0.5 * (1.0 - max_prob) + 0.3 * (1.0 - margin) + 0.2 * entropy_norm
    return {
        "max_prob": float(max_prob),
        "margin": float(margin),
        "entropy": float(entropy_norm),
        "uncertainty_score": float(min(max(score, 0.0), 1.0)),
    }


def evaluate_binary_predictions(
    y_true: list[str],
    y_pred: list[str],
    p_plasmid: list[float] | None = None,
) -> dict[str, Any]:
    metrics = binary_domain_metrics(y_true, y_pred)
    if p_plasmid is not None:
        metrics.update(aggregate_calibration_metrics(y_true, p_plasmid))
    return metrics
