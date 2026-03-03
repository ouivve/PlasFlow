from __future__ import annotations

from plasflow_v2.features import (
    canonical_kmer,
    default_feature_manifest,
    reverse_complement,
    vectorize_sequence,
)
from plasflow_v2.metrics import best_threshold_by_macro_f1, binary_domain_metrics, threshold_curve
from plasflow_v2.reporting import build_summary
from plasflow_v2.training import apply_calibrator


def test_canonical_kmer_is_reverse_complement_invariant() -> None:
    kmer = "ATGCGA"
    rc = reverse_complement(kmer)
    assert canonical_kmer(kmer) == canonical_kmer(rc)


def test_feature_manifest_vectorization_is_reproducible() -> None:
    manifest = default_feature_manifest()
    seq = "ACGT" * 100
    v1 = vectorize_sequence(seq, manifest)
    v2 = vectorize_sequence(seq, manifest)

    assert len(v1) == len(manifest["feature_order"])
    assert v1 == v2


def test_threshold_curve_handles_single_class_without_crash() -> None:
    y_true = ["chromosome"] * 5
    probs = [0.2, 0.3, 0.1, 0.4, 0.35]
    best_threshold, best_row, curve = best_threshold_by_macro_f1(y_true, probs)

    assert 0.05 <= best_threshold <= 0.95
    assert "macro_f1" in best_row
    assert len(curve) > 0


def test_apply_calibrator_supports_isotonic_and_platt() -> None:
    probs = [0.2, 0.8]

    class DummyIsotonic:
        def predict(self, values):
            return values

    class DummyPlatt:
        def predict_proba(self, rows):
            return [[1.0 - float(row[0]), float(row[0])] for row in rows]

    isotonic = apply_calibrator({"type": "isotonic", "model": DummyIsotonic()}, probs)
    platt = apply_calibrator({"type": "platt", "model": DummyPlatt()}, probs)

    assert isotonic == probs
    assert platt == probs


def test_build_summary_includes_extended_metrics_schema() -> None:
    rows = [
        {
            "contig_id": 0,
            "contig_name": "c1",
            "contig_length": 1000,
            "label": "plasmid.other",
            "plasmid.other": 0.8,
            "chromosome.other": 0.2,
        }
    ]
    labels = ["plasmid.other", "chromosome.other"]

    summary = build_summary(
        rows=rows,
        labels=labels,
        threshold=0.7,
        requested_mode="v2",
        used_mode="v2",
        fallback_reason=None,
        metrics={
            "binary_domain": {"macro_f1": 0.91},
            "calibration": {"ece": 0.05, "recommended_threshold": 0.63},
        },
    )

    assert summary["metrics"]["binary_domain"]["macro_f1"] == 0.91
    assert summary["metrics"]["calibration"]["ece"] == 0.05
    assert summary["metrics"]["calibration"]["recommended_threshold"] == 0.63


def test_binary_domain_metrics_confusion_shape() -> None:
    metrics = binary_domain_metrics(
        y_true=["plasmid", "plasmid", "chromosome", "chromosome"],
        y_pred=["plasmid", "chromosome", "chromosome", "plasmid"],
    )
    assert metrics["support"] == 4
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_threshold_curve_has_expected_columns() -> None:
    curve = threshold_curve(
        y_true=["plasmid", "chromosome", "plasmid", "chromosome"],
        p_plasmid=[0.9, 0.2, 0.7, 0.4],
        start=0.5,
        end=0.6,
        step=0.1,
    )
    assert len(curve) == 2
    assert set(curve[0].keys()) == {"threshold", "macro_f1", "precision_macro", "recall_macro", "accuracy"}
