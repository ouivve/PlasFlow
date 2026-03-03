from __future__ import annotations

import argparse
from pathlib import Path
import json

from .comparison import compare_modes
from .constants import DEFAULT_THRESHOLD, load_task_label_spec
from .evaluation import evaluate_modern_model
from .pipeline import run_classification
from .reporting import generate_report_from_tsv
from .tools import probe_tools
from .training import train_modern_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plasflow",
        description="PlasFlow v2: v1/v2 plasmid/chromosome classification and reporting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="Classify contigs from FASTA")
    classify.add_argument("--input", required=True, help="Input FASTA (supports .gz)")
    classify.add_argument("--output", required=True, help="Output prefix or .tsv path")
    classify.add_argument("--mode", choices=["v1", "v2", "legacy", "modern"], default="v1")
    classify.add_argument("--task", choices=["legacy28", "binary_domain", "domain4"], default="legacy28")
    classify.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    classify.add_argument("--read-type", choices=["short", "long", "hybrid", "complete"], default="short")
    classify.add_argument("--min-length", type=int, default=1000)
    classify.add_argument("--coverage-source", choices=["header", "none"], default="header")
    classify.add_argument("--polish", choices=["none", "racon", "medaka"], default="none")
    classify.add_argument(
        "--no-circularity-check",
        action="store_true",
        help="Disable circularity overlap check",
    )
    classify.add_argument(
        "--no-fallback",
        action="store_true",
        help="In v1 mode, do not fallback to v2 when v1 execution fails",
    )

    report = subparsers.add_parser("report", help="Generate HTML/JSON report from TSV")
    report.add_argument("--input", required=True, help="Input TSV produced by PlasFlow")
    report.add_argument("--out", required=True, help="Output HTML path")
    report.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    report.add_argument("--task", choices=["legacy28", "binary_domain", "domain4"], default="legacy28")

    serve = subparsers.add_parser("serve", help="Run FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)

    train = subparsers.add_parser(
        "train-v2",
        aliases=["train-modern"],
        help="Train and save a v2 model bundle",
    )
    train.add_argument("--dataset-manifest", default=None, help="Dataset manifest path (.json/.yaml/.yml)")
    train.add_argument("--input", default=None, help="Labeled TSV fallback (sequence/label/split columns)")
    train.add_argument("--outdir", default=None, help="Model output directory (default: models_v2/current)")
    train.add_argument("--task", choices=["binary_domain", "domain4"], default="binary_domain")
    train.add_argument("--model", choices=["lightgbm"], default="lightgbm")
    train.add_argument("--sequence-col", default="sequence")
    train.add_argument("--label-col", default="label")
    train.add_argument("--split-col", default="split")
    train.add_argument("--calibration", choices=["isotonic", "platt"], default="isotonic")
    train.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser(
        "evaluate-v2",
        aliases=["evaluate-modern"],
        help="Evaluate a v2 model on labeled TSV",
    )
    evaluate.add_argument("--input", required=True, help="Labeled TSV with sequence and label columns")
    evaluate.add_argument("--model-dir", required=True, help="Model directory (bundle dir or models_v2 root)")
    evaluate.add_argument("--task", choices=["binary_domain", "domain4"], default="binary_domain")
    evaluate.add_argument("--out", required=True, help="Output eval JSON path")
    evaluate.add_argument("--sequence-col", default="sequence")
    evaluate.add_argument("--label-col", default="label")
    evaluate.add_argument("--name-col", default="contig_name")
    evaluate.add_argument("--threshold", type=float, default=None)

    compare = subparsers.add_parser("compare-modes", help="Run v2/v1 and generate comparison report")
    compare.add_argument("--input", required=True, help="Input FASTA")
    compare.add_argument("--outdir", required=True, help="Output directory for mode runs and comparison.json")
    compare.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    compare.add_argument(
        "--ground-truth",
        default=None,
        help="Optional labeled TSV for ground-truth metrics (requires contig_name,label columns)",
    )

    subparsers.add_parser("tools-check", help="Probe optional external tools (racon/medaka/prodigal/blastn/diamond)")

    return parser


def cmd_classify(args: argparse.Namespace) -> int:
    result = run_classification(
        input_path=args.input,
        output_prefix=args.output,
        mode=args.mode,
        task=args.task,
        threshold=args.threshold,
        allow_fallback=not args.no_fallback,
        read_type=args.read_type,
        min_length=args.min_length,
        coverage_source=args.coverage_source,
        circularity_check=not args.no_circularity_check,
        polish=args.polish,
    )

    payload = {
        "requested_mode": result.requested_mode,
        "used_mode": result.used_mode,
        "task": result.task,
        "threshold": result.threshold,
        "fallback_reason": result.fallback_reason,
        "artifacts": {
            "tsv": str(result.artifacts.tsv),
            "plasmids_fasta": str(result.artifacts.plasmids_fasta),
            "chromosomes_fasta": str(result.artifacts.chromosomes_fasta),
            "unclassified_fasta": str(result.artifacts.unclassified_fasta),
            "phage_fasta": str(result.artifacts.phage_fasta),
            "ambiguous_fasta": str(result.artifacts.ambiguous_fasta),
            "report_json": str(result.artifacts.report_json),
            "report_html": str(result.artifacts.report_html),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    labels = load_task_label_spec(args.task).labels
    summary = generate_report_from_tsv(
        tsv_path=args.input,
        html_output=args.out,
        labels=labels,
        threshold=args.threshold,
        task=args.task,
    )
    print(json.dumps({"html": str(Path(args.out)), "summary": summary}, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("plasflow_v2.api.app:create_app", host=args.host, port=args.port, factory=True)
    return 0


def cmd_train_modern(args: argparse.Namespace) -> int:
    result = train_modern_model(
        dataset_manifest=args.dataset_manifest,
        input_tsv=args.input,
        model_dir=args.outdir,
        task=args.task,
        model_name=args.model,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        split_col=args.split_col,
        calibration=args.calibration,
        random_seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_evaluate_modern(args: argparse.Namespace) -> int:
    result = evaluate_modern_model(
        input_tsv=args.input,
        model_dir=args.model_dir,
        out_path=args.out,
        task=args.task,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        name_col=args.name_col,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_compare_modes(args: argparse.Namespace) -> int:
    result = compare_modes(
        input_path=args.input,
        output_dir=args.outdir,
        threshold=args.threshold,
        ground_truth=args.ground_truth,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_tools_check() -> int:
    print(json.dumps({"tools": probe_tools()}, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "classify":
        return cmd_classify(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command in {"train-v2", "train-modern"}:
        return cmd_train_modern(args)
    if args.command in {"evaluate-v2", "evaluate-modern"}:
        return cmd_evaluate_modern(args)
    if args.command == "compare-modes":
        return cmd_compare_modes(args)
    if args.command == "tools-check":
        return cmd_tools_check()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
