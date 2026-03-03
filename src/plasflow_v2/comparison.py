from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json

from .metrics import evaluate_binary_predictions
from .pipeline import run_classification


@dataclass
class ModeRun:
    mode: str
    ok: bool
    duration_sec: float
    output_prefix: Path | None
    error: str | None


def _read_labels(tsv_path: Path, name_col: str = "contig_name", label_col: str = "label") -> dict[str, str]:
    labels: dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = row.get(name_col)
            label = row.get(label_col)
            if name and label:
                labels[str(name)] = str(label)
    return labels


def _evaluate_against_ground_truth(predicted: dict[str, str], ground_truth: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(predicted.keys()) & set(ground_truth.keys()))
    if not common:
        return {
            "common_contigs": 0,
            "metrics": None,
        }

    y_true = [ground_truth[name] for name in common]
    y_pred = [predicted[name] for name in common]

    metrics = evaluate_binary_predictions(y_true, y_pred, p_plasmid=None)
    return {
        "common_contigs": len(common),
        "metrics": {
            "macro_f1": metrics["macro_f1"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "support": metrics["support"],
        },
    }


def compare_modes(
    input_path: str | Path,
    output_dir: str | Path,
    threshold: float,
    ground_truth: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, ModeRun] = {}

    v2_prefix = out_dir / "v2" / "result"
    v2_prefix.parent.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    run_classification(
        input_path=input_path,
        output_prefix=v2_prefix,
        mode="v2",
        task="legacy28",
        threshold=threshold,
        allow_fallback=True,
        min_length=1,
    )
    runs["v2"] = ModeRun(
        mode="v2",
        ok=True,
        duration_sec=perf_counter() - t0,
        output_prefix=v2_prefix,
        error=None,
    )

    v1_prefix = out_dir / "v1" / "result"
    v1_prefix.parent.mkdir(parents=True, exist_ok=True)

    t1 = perf_counter()
    try:
        run_classification(
            input_path=input_path,
            output_prefix=v1_prefix,
            mode="v1",
            task="legacy28",
            threshold=threshold,
            allow_fallback=False,
            min_length=1,
        )
        runs["v1"] = ModeRun(
            mode="v1",
            ok=True,
            duration_sec=perf_counter() - t1,
            output_prefix=v1_prefix,
            error=None,
        )
    except Exception as exc:
        runs["v1"] = ModeRun(
            mode="v1",
            ok=False,
            duration_sec=perf_counter() - t1,
            output_prefix=None,
            error=str(exc),
        )

    comparison: dict[str, Any] = {
        "input_path": str(input_path),
        "threshold": threshold,
        "ground_truth": str(ground_truth) if ground_truth else None,
        "runs": {
            mode: {
                "ok": run.ok,
                "duration_sec": run.duration_sec,
                "output_prefix": str(run.output_prefix) if run.output_prefix else None,
                "error": run.error,
            }
            for mode, run in runs.items()
        },
    }

    if runs["v1"].ok and runs["v1"].output_prefix:
        v2_labels = _read_labels((runs["v2"].output_prefix or v2_prefix).with_suffix(".tsv"))
        v1_labels = _read_labels(runs["v1"].output_prefix.with_suffix(".tsv"))

        common = sorted(set(v2_labels.keys()) & set(v1_labels.keys()))
        if common:
            agree = sum(1 for key in common if v2_labels[key] == v1_labels[key])
            comparison["agreement"] = {
                "common_contigs": len(common),
                "matching_labels": agree,
                "agreement_ratio": agree / len(common),
            }

    if ground_truth:
        gt_labels = _read_labels(Path(ground_truth), name_col="contig_name", label_col="label")
        if not gt_labels:
            gt_labels = _read_labels(Path(ground_truth), name_col="name", label_col="label")

        ground_truth_eval: dict[str, Any] = {}
        for mode_name, run in runs.items():
            if not run.ok or not run.output_prefix:
                continue
            pred = _read_labels(run.output_prefix.with_suffix(".tsv"))
            ground_truth_eval[mode_name] = _evaluate_against_ground_truth(pred, gt_labels)
        comparison["ground_truth_eval"] = ground_truth_eval

    out_path = out_dir / "comparison.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    comparison["report_path"] = str(out_path)
    return comparison
