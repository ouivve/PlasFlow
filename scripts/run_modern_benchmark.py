#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plasflow_v2.comparison import compare_modes
from plasflow_v2.datasets import load_dataset_rows
from plasflow_v2.evaluation import evaluate_modern_model
from plasflow_v2.training import train_modern_model


def _write_test_artifacts(dataset_manifest: Path, outdir: Path) -> tuple[Path, Path, Path]:
    rows = load_dataset_rows(dataset_manifest)
    test_rows = [row for row in rows if row.split == "test"]
    if not test_rows:
        raise ValueError("dataset manifest does not contain test rows")

    fasta_path = outdir / "benchmark_test.fasta"
    gt_path = outdir / "benchmark_ground_truth.tsv"
    eval_tsv_path = outdir / "benchmark_eval_input.tsv"

    outdir.mkdir(parents=True, exist_ok=True)

    with fasta_path.open("w", encoding="utf-8") as fasta, gt_path.open(
        "w", encoding="utf-8", newline=""
    ) as gt, eval_tsv_path.open("w", encoding="utf-8", newline="") as eval_tsv:
        gt_writer = csv.writer(gt, delimiter="\t")
        gt_writer.writerow(["contig_name", "label"])

        eval_writer = csv.writer(eval_tsv, delimiter="\t")
        eval_writer.writerow(["contig_name", "sequence", "label"])

        for idx, row in enumerate(test_rows):
            name = f"bench_{idx}"
            fasta.write(f">{name}\n{row.sequence}\n")
            gt_writer.writerow([name, row.label])
            eval_writer.writerow([name, row.sequence, row.label])

    return fasta_path, gt_path, eval_tsv_path


def _extract_macro_f1(compare_payload: dict) -> float:
    metrics = (
        compare_payload.get("ground_truth_eval", {})
        .get("modern", {})
        .get("metrics", {})
    )
    value = metrics.get("macro_f1")
    if value is None:
        raise ValueError("Could not read baseline macro_f1 from compare output")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline-vs-modern benchmark and produce gate inputs")
    parser.add_argument("--dataset-manifest", required=True, help="Dataset manifest path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.7, help="Classification threshold for baseline")
    parser.add_argument("--calibration", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_fasta, ground_truth_tsv, eval_input_tsv = _write_test_artifacts(
        dataset_manifest=Path(args.dataset_manifest),
        outdir=outdir,
    )

    baseline_compare = compare_modes(
        input_path=test_fasta,
        output_dir=outdir / "baseline_compare",
        threshold=args.threshold,
        ground_truth=ground_truth_tsv,
    )
    baseline_macro_f1 = _extract_macro_f1(baseline_compare)

    baseline_eval_path = outdir / "baseline_eval.json"
    baseline_eval = {
        "source": "heuristic-modern-via-compare",
        "metrics": {
            "binary_domain": {
                "macro_f1": baseline_macro_f1,
                "precision_macro": baseline_compare["ground_truth_eval"]["modern"]["metrics"].get("precision_macro"),
                "recall_macro": baseline_compare["ground_truth_eval"]["modern"]["metrics"].get("recall_macro"),
                "accuracy": baseline_compare["ground_truth_eval"]["modern"]["metrics"].get("accuracy"),
                "confusion_matrix": baseline_compare["ground_truth_eval"]["modern"]["metrics"].get("confusion_matrix"),
                "support": baseline_compare["ground_truth_eval"]["modern"]["metrics"].get("support"),
            }
        },
    }
    baseline_eval_path.write_text(json.dumps(baseline_eval, indent=2), encoding="utf-8")

    model_dir = outdir / "model_bundle"
    train_result = train_modern_model(
        dataset_manifest=Path(args.dataset_manifest),
        model_dir=model_dir,
        task="binary_domain",
        model_name="lightgbm",
        calibration=args.calibration,
        random_seed=args.seed,
    )

    candidate_eval_path = outdir / "candidate_eval.json"
    candidate_eval = evaluate_modern_model(
        input_tsv=eval_input_tsv,
        model_dir=model_dir,
        out_path=candidate_eval_path,
        task="binary_domain",
        sequence_col="sequence",
        label_col="label",
        name_col="contig_name",
    )

    summary = {
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "baseline_eval": str(baseline_eval_path.resolve()),
        "candidate_eval": str(candidate_eval_path.resolve()),
        "baseline_macro_f1": baseline_macro_f1,
        "candidate_macro_f1": candidate_eval["metrics"]["binary_domain"]["macro_f1"],
        "train": train_result,
        "baseline_compare": baseline_compare,
        "candidate": candidate_eval,
    }

    summary_path = outdir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({**summary, "summary_path": str(summary_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
