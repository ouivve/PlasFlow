#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plasflow_v2.datasets import load_dataset_rows
from plasflow_v2.evaluation import evaluate_modern_model
from plasflow_v2.training import train_modern_model


def _write_eval_input_from_manifest(dataset_manifest: Path, out_path: Path) -> Path:
    rows = load_dataset_rows(dataset_manifest)
    test_rows = [row for row in rows if row.split == "test"]
    if not test_rows:
        raise ValueError("dataset manifest does not contain any test rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["contig_name", "sequence", "label"])
        for idx, row in enumerate(test_rows):
            writer.writerow([f"bench_{idx}", row.sequence, row.label])
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modern model and evaluate it in one command")
    parser.add_argument("--dataset-manifest", required=True, help="Dataset manifest (.json/.yaml/.yml)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--eval-input", default=None, help="Optional labeled TSV for evaluation")
    parser.add_argument("--eval-out", default=None, help="Output path for eval JSON")
    parser.add_argument("--calibration", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dir = outdir / "model_bundle"
    eval_out = Path(args.eval_out) if args.eval_out else outdir / "eval.json"

    train_result = train_modern_model(
        dataset_manifest=Path(args.dataset_manifest),
        model_dir=model_dir,
        task="binary_domain",
        model_name="lightgbm",
        calibration=args.calibration,
        random_seed=args.seed,
    )

    if args.eval_input:
        eval_input = Path(args.eval_input)
    else:
        eval_input = _write_eval_input_from_manifest(
            dataset_manifest=Path(args.dataset_manifest),
            out_path=outdir / "eval_input.tsv",
        )

    eval_result = evaluate_modern_model(
        input_tsv=eval_input,
        model_dir=model_dir,
        out_path=eval_out,
        task="binary_domain",
        sequence_col="sequence",
        label_col="label",
        name_col="contig_name",
    )

    payload = {
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "model_dir": str(model_dir.resolve()),
        "eval_input": str(eval_input.resolve()),
        "train": train_result,
        "evaluate": eval_result,
    }

    summary_path = outdir / "train_eval_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({**payload, "summary": str(summary_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
