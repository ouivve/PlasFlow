from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json
import math

from .constants import DEFAULT_MODELS_V2_DIR, TaskType
from .datasets import DatasetRow, dataset_split_counts, load_dataset_rows
from .features import build_feature_manifest, vectorize_sequence
from .metrics import (
    aggregate_calibration_metrics,
    best_threshold_by_macro_f1,
    binary_domain_from_label,
    domain4_from_label,
    domain4_metrics,
    evaluate_binary_predictions,
    expected_calibration_error_multiclass,
    predict_with_threshold,
)


def _load_rows_from_tsv(
    input_tsv: Path,
    sequence_col: str,
    label_col: str,
    split_col: str,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with input_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sequence = (row.get(sequence_col) or "").strip().upper()
            label = (row.get(label_col) or "").strip()
            if not sequence or not label:
                continue

            split = (row.get(split_col) or "train").strip().lower()
            if split not in {"train", "val", "test"}:
                split = "train"

            domain = binary_domain_from_label(label) or domain4_from_label(label)
            rows.append(
                DatasetRow(
                    sequence=sequence,
                    label=label,
                    domain_label=domain,
                    split=split,
                    group_id=row.get("group_id") or sha256(sequence.encode("utf-8")).hexdigest()[:16],
                    source=row.get("source") or "input_tsv",
                )
            )
    return rows


def _class_balance(rows: list[DatasetRow], mapper: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(mapper(row.label))
        out[key] = out.get(key, 0) + 1
    return out


def _taxon_prior(rows: list[DatasetRow]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = {"plasmid": {}, "chromosome": {}}
    for row in rows:
        parts = row.label.split(".", 1)
        domain = binary_domain_from_label(row.label)
        if domain is None:
            continue
        taxon = parts[1] if len(parts) == 2 else "other"
        domain_map = counts.setdefault(domain, {})
        domain_map[taxon] = domain_map.get(taxon, 0) + 1

    out: dict[str, dict[str, float]] = {}
    for domain, domain_counts in counts.items():
        total = sum(domain_counts.values())
        if total <= 0:
            out[domain] = {"other": 1.0}
            continue
        out[domain] = {taxon: value / total for taxon, value in domain_counts.items()}
    return out


def _ensure_train_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from lightgbm import LGBMClassifier
        import joblib
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise RuntimeError(
            "Training requires lightgbm + scikit-learn + joblib. Install extras: pip install 'plasflow-v2[train]'"
        ) from exc

    return LGBMClassifier, joblib, IsotonicRegression, LogisticRegression


def _fit_calibrator(
    method: str,
    val_probs: list[float],
    y_val: list[int],
    isotonic_cls: Any,
    logistic_cls: Any,
) -> dict[str, Any]:
    chosen = method.strip().lower()
    if chosen not in {"isotonic", "platt"}:
        raise ValueError("calibration must be 'isotonic' or 'platt'")

    if chosen == "isotonic":
        model = isotonic_cls(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(val_probs, y_val)
        return {"type": "isotonic", "model": model}

    model = logistic_cls(max_iter=200)
    model.fit([[p] for p in val_probs], y_val)
    return {"type": "platt", "model": model}


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [v / total for v in exps]


def _fit_temperature_calibrator(
    val_probs: list[list[float]],
    y_val_labels: list[str],
    class_order: list[str],
) -> dict[str, Any]:
    class_index = {label: idx for idx, label in enumerate(class_order)}
    y_idx = [class_index.get(label, -1) for label in y_val_labels]
    filtered = [(probs, idx) for probs, idx in zip(val_probs, y_idx) if idx >= 0]
    if not filtered:
        return {"type": "temperature", "temperature": 1.0}

    candidates = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    best_t = 1.0
    best_nll = float("inf")
    for temp in candidates:
        total_nll = 0.0
        for probs, target_idx in filtered:
            logits = [math.log(max(float(p), 1e-12)) / temp for p in probs]
            cal = _softmax(logits)
            total_nll += -math.log(max(cal[target_idx], 1e-12))
        avg_nll = total_nll / len(filtered)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_t = temp
    return {"type": "temperature", "temperature": best_t}


def apply_calibrator(calibrator_payload: dict[str, Any] | None, probs: list[float]) -> list[float]:
    if not calibrator_payload:
        return [float(p) for p in probs]

    cal_type = str(calibrator_payload.get("type", "")).lower()
    model = calibrator_payload.get("model")
    if model is None:
        return [float(p) for p in probs]

    if cal_type == "isotonic":
        return [float(x) for x in model.predict(probs)]
    if cal_type == "platt":
        out = model.predict_proba([[p] for p in probs])
        return [float(row[1]) for row in out]

    return [float(p) for p in probs]


def apply_calibrator_matrix(calibrator_payload: dict[str, Any] | None, probs: list[list[float]]) -> list[list[float]]:
    if not calibrator_payload:
        return [[float(v) for v in row] for row in probs]
    if str(calibrator_payload.get("type", "")).lower() != "temperature":
        return [[float(v) for v in row] for row in probs]
    temp = float(calibrator_payload.get("temperature", 1.0))
    if temp <= 0:
        temp = 1.0
    calibrated: list[list[float]] = []
    for row in probs:
        logits = [math.log(max(float(p), 1e-12)) / temp for p in row]
        calibrated.append(_softmax(logits))
    return calibrated


def _multiclass_brier_score(y_true: list[str], probs: list[list[float]], class_order: list[str]) -> float:
    if not y_true:
        return 0.0
    index = {label: idx for idx, label in enumerate(class_order)}
    values: list[float] = []
    for label, row in zip(y_true, probs):
        target_idx = index.get(label)
        if target_idx is None:
            continue
        row_sum = 0.0
        for idx, prob in enumerate(row):
            truth = 1.0 if idx == target_idx else 0.0
            row_sum += (truth - float(prob)) ** 2
        values.append(row_sum / max(len(row), 1))
    return sum(values) / len(values) if values else 0.0


def _predict_multiclass_labels(probs: list[list[float]], class_order: list[str]) -> list[str]:
    labels: list[str] = []
    for row in probs:
        idx = max(range(len(row)), key=lambda i: row[i])
        labels.append(class_order[idx])
    return labels


def _select_task_label(raw_label: str, task: TaskType) -> str | None:
    if task == "binary_domain":
        return binary_domain_from_label(raw_label)
    if task == "domain4":
        return domain4_from_label(raw_label)
    return raw_label


def _ensure_task_supported(task: str) -> TaskType:
    if task not in {"binary_domain", "domain4"}:
        raise ValueError("Only task=binary_domain|domain4 is supported in v2 training")
    return task  # type: ignore[return-value]


def train_modern_model(
    dataset_manifest: str | Path | None = None,
    input_tsv: str | Path | None = None,
    model_dir: str | Path | None = None,
    task: str = "binary_domain",
    model_name: str = "lightgbm",
    sequence_col: str = "sequence",
    label_col: str = "label",
    split_col: str = "split",
    calibration: str = "isotonic",
    random_seed: int = 42,
) -> dict[str, Any]:
    task_t = _ensure_task_supported(task)
    if model_name != "lightgbm":
        raise ValueError("Only model=lightgbm is supported in v2")
    if dataset_manifest is None and input_tsv is None:
        raise ValueError("Either dataset_manifest or input_tsv is required")

    LGBMClassifier, joblib, IsotonicRegression, LogisticRegression = _ensure_train_dependencies()

    if dataset_manifest is not None:
        rows = load_dataset_rows(dataset_manifest)
        dataset_fingerprint = sha256(Path(dataset_manifest).read_bytes()).hexdigest()
    else:
        input_path = Path(str(input_tsv))
        rows = _load_rows_from_tsv(input_path, sequence_col=sequence_col, label_col=label_col, split_col=split_col)
        dataset_fingerprint = sha256(input_path.read_bytes()).hexdigest()

    if not rows:
        raise ValueError("No valid training rows found")

    processed: list[tuple[DatasetRow, str]] = []
    for row in rows:
        mapped = _select_task_label(row.label, task_t)
        if mapped is None:
            continue
        processed.append((row, mapped))
    if not processed:
        raise ValueError("No training rows compatible with selected task")

    split_counts_raw = dataset_split_counts([row for row, _ in processed])
    if split_counts_raw["train"] == 0:
        raise ValueError("training split is empty")
    if split_counts_raw["val"] == 0:
        raise ValueError("validation split is empty; provide val rows in dataset or split config")

    feature_manifest = build_feature_manifest(k_values=(4, 5, 6), include_scalar=True, canonical=True)

    train_pairs = [(row, label) for row, label in processed if row.split == "train"]
    val_pairs = [(row, label) for row, label in processed if row.split == "val"]
    test_pairs = [(row, label) for row, label in processed if row.split == "test"]

    x_train = [vectorize_sequence(row.sequence, feature_manifest) for row, _ in train_pairs]
    y_train = [label for _, label in train_pairs]

    x_val = [vectorize_sequence(row.sequence, feature_manifest) for row, _ in val_pairs]
    y_val = [label for _, label in val_pairs]

    if len(set(y_train)) < 2:
        raise ValueError("training split must contain >=2 classes")
    if len(set(y_val)) < 2:
        raise ValueError("validation split must contain >=2 classes")

    model_kwargs: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_seed,
        "n_jobs": 1,
    }
    if task_t == "binary_domain":
        model_kwargs["objective"] = "binary"
    else:
        model_kwargs["objective"] = "multiclass"
        model_kwargs["num_class"] = 4

    model = LGBMClassifier(**model_kwargs)
    model.fit(x_train, y_train)
    class_order = [str(v) for v in list(model.classes_)]

    recommended_threshold = 0.7
    threshold_curve_preview: list[dict[str, Any]] | None = None

    if task_t == "binary_domain":
        if "plasmid" not in class_order:
            raise ValueError("Binary model classes must include 'plasmid'")
        p_idx = class_order.index("plasmid")
        val_raw = [float(row[p_idx]) for row in model.predict_proba(x_val)]
        y_val_binary = [1 if label == "plasmid" else 0 for label in y_val]
        calibrator = _fit_calibrator(
            method=calibration,
            val_probs=val_raw,
            y_val=y_val_binary,
            isotonic_cls=IsotonicRegression,
            logistic_cls=LogisticRegression,
        )
        val_calibrated = apply_calibrator(calibrator, val_raw)
        best_threshold, best_row, threshold_rows = best_threshold_by_macro_f1(y_val, val_calibrated)
        val_pred = predict_with_threshold(val_calibrated, best_threshold)
        val_metrics = evaluate_binary_predictions(y_val, val_pred, val_calibrated)
        recommended_threshold = best_threshold
        threshold_curve_preview = threshold_rows[::10] if len(threshold_rows) > 120 else threshold_rows
        calibration_metrics = {
            "method": calibrator.get("type"),
            "ece": aggregate_calibration_metrics(y_val, val_calibrated)["ece"],
            "brier_score": aggregate_calibration_metrics(y_val, val_calibrated)["brier_score"],
            "recommended_threshold": best_threshold,
            "best_validation_row": best_row,
            "threshold_curve_preview": threshold_curve_preview,
        }
    else:
        val_raw_matrix = [[float(v) for v in row] for row in model.predict_proba(x_val)]
        calibrator = _fit_temperature_calibrator(val_raw_matrix, y_val, class_order)
        val_calibrated_matrix = apply_calibrator_matrix(calibrator, val_raw_matrix)
        val_pred = _predict_multiclass_labels(val_calibrated_matrix, class_order)
        val_metrics = domain4_metrics(y_val, val_pred)
        calibration_metrics = {
            "method": calibrator.get("type"),
            "temperature": calibrator.get("temperature"),
            "ece": expected_calibration_error_multiclass(y_val, val_calibrated_matrix, labels=class_order, mapper=None),
            "brier_score": _multiclass_brier_score(y_val, val_calibrated_matrix, class_order),
            "recommended_threshold": recommended_threshold,
            "best_validation_row": None,
            "threshold_curve_preview": None,
        }

    test_metrics: dict[str, Any] | None = None
    if test_pairs:
        x_test = [vectorize_sequence(row.sequence, feature_manifest) for row, _ in test_pairs]
        y_test = [label for _, label in test_pairs]
        test_raw = [[float(v) for v in row] for row in model.predict_proba(x_test)]
        if task_t == "binary_domain":
            p_idx = class_order.index("plasmid")
            raw_p = [row[p_idx] for row in test_raw]
            test_calibrated = apply_calibrator(calibrator, raw_p)
            test_pred = predict_with_threshold(test_calibrated, recommended_threshold)
            test_metrics = evaluate_binary_predictions(y_test, test_pred, test_calibrated)
        else:
            test_calibrated = apply_calibrator_matrix(calibrator, test_raw)
            test_pred = _predict_multiclass_labels(test_calibrated, class_order)
            test_metrics = domain4_metrics(y_test, test_pred)

    target_dir = Path(model_dir) if model_dir else DEFAULT_MODELS_V2_DIR / "current"
    target_dir.mkdir(parents=True, exist_ok=True)

    domain_model_path = target_dir / "domain_model.joblib"
    calibrator_path = target_dir / "calibrator.joblib"
    feature_manifest_path = target_dir / "feature_manifest.json"
    metadata_path = target_dir / "metadata.json"

    joblib.dump(model, domain_model_path)
    joblib.dump(calibrator, calibrator_path)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    metadata: dict[str, Any] = {
        "model_bundle_version": 2,
        "task": task_t,
        "model": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "dataset_manifest": str(dataset_manifest) if dataset_manifest else None,
        "source_tsv": str(input_tsv) if input_tsv else None,
        "split_counts": split_counts_raw,
        "class_balance_train": _class_balance([row for row, _ in train_pairs], mapper=lambda label: _select_task_label(label, task_t)),
        "taxon_prior": _taxon_prior([row for row, _ in train_pairs]),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibration": calibration_metrics,
        "class_order": class_order,
        "artifacts": {
            "domain_model": str(domain_model_path),
            "calibrator": str(calibrator_path),
            "feature_manifest": str(feature_manifest_path),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_dir": str(target_dir),
        "domain_model": str(domain_model_path),
        "calibrator": str(calibrator_path),
        "feature_manifest": str(feature_manifest_path),
        "metadata": str(metadata_path),
        "task": task_t,
        "recommended_threshold": recommended_threshold,
    }
