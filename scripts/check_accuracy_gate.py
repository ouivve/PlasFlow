#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_macro_f1(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    binary = metrics.get("binary_domain", {}) if isinstance(metrics, dict) else {}
    value = binary.get("macro_f1")
    if value is None:
        raise ValueError(f"macro_f1 missing in {path}")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check macro-F1 improvement gate")
    parser.add_argument("--baseline", required=True, help="Baseline eval JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate eval JSON path")
    parser.add_argument("--min-delta", type=float, default=0.05, help="Required absolute improvement")
    parser.add_argument("--min-candidate", type=float, default=0.0, help="Optional candidate absolute floor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    baseline_f1 = _load_macro_f1(baseline_path)
    candidate_f1 = _load_macro_f1(candidate_path)
    delta = candidate_f1 - baseline_f1

    print(
        json.dumps(
            {
                "baseline_macro_f1": baseline_f1,
                "candidate_macro_f1": candidate_f1,
                "delta": delta,
                "required_delta": args.min_delta,
                "required_candidate_floor": args.min_candidate,
            },
            indent=2,
        )
    )

    if candidate_f1 < args.min_candidate:
        print(
            f"[ACCURACY_GATE] FAILED: candidate macro_f1 {candidate_f1:.4f} is below floor {args.min_candidate:.4f}",
            flush=True,
        )
        return 1

    if delta < args.min_delta:
        print(
            f"[ACCURACY_GATE] FAILED: delta {delta:.4f} is below required {args.min_delta:.4f}",
            flush=True,
        )
        return 1

    print(
        f"[ACCURACY_GATE] PASSED: delta {delta:.4f} >= {args.min_delta:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
