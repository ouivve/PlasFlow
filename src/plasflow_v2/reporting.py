from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any
import csv
import html
import json

from .constants import TaskType
from .metrics import domain4_from_label, uncertainty_components


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _default_metrics() -> dict[str, Any]:
    return {
        "binary_domain": {
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "accuracy": None,
            "confusion_matrix": None,
            "support": None,
        },
        "domain4": {
            "macro_f1": None,
            "precision_macro": None,
            "recall_macro": None,
            "accuracy": None,
            "confusion_matrix": None,
            "support": None,
        },
        "calibration": {
            "ece": None,
            "brier_score": None,
            "recommended_threshold": None,
        },
    }


def _merge_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    out = _default_metrics()
    if not isinstance(metrics, dict):
        return out

    for section in ("binary_domain", "domain4", "calibration"):
        section_values = metrics.get(section)
        if not isinstance(section_values, dict):
            continue
        out_section = out.setdefault(section, {})
        for key, value in section_values.items():
            out_section[key] = value
    return out


def _row_uncertainty(row: dict[str, Any], labels: list[str]) -> dict[str, float]:
    if all(key in row for key in ("max_prob", "margin", "entropy", "uncertainty_score")):
        return {
            "max_prob": _as_float(row.get("max_prob")),
            "margin": _as_float(row.get("margin")),
            "entropy": _as_float(row.get("entropy")),
            "uncertainty_score": _as_float(row.get("uncertainty_score")),
        }
    probs = [_as_float(row.get(label, 0.0)) for label in labels]
    return uncertainty_components(probs)


def build_summary(
    rows: list[dict[str, Any]],
    labels: list[str],
    threshold: float,
    requested_mode: str,
    used_mode: str,
    fallback_reason: str | None,
    metrics: dict[str, Any] | None = None,
    task: TaskType = "legacy28",
    preprocessing: dict[str, Any] | None = None,
    qc: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    label_dist: dict[str, int] = {}
    top_probs: list[float] = []
    uncertain_contigs: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, float]] = []

    for row in rows:
        label = str(row["label"])
        label_dist[label] = label_dist.get(label, 0) + 1
        unc = _row_uncertainty(row, labels)
        uncertainty_rows.append(unc)

        max_prob = unc["max_prob"]
        max_label = ""
        for lbl in labels:
            prob = _as_float(row.get(lbl, 0.0))
            if prob >= _as_float(row.get(max_label, 0.0)):
                max_label = lbl

        top_probs.append(max_prob)
        if max_prob < threshold:
            uncertain_contigs.append(
                {
                    "contig_name": row.get("contig_name", ""),
                    "contig_length": int(float(row.get("contig_length", 0))),
                    "max_probability": max_prob,
                    "top_label": max_label,
                    "assigned_label": label,
                    "margin": unc["margin"],
                    "entropy": unc["entropy"],
                    "uncertainty_score": unc["uncertainty_score"],
                }
            )

    uncertain_contigs.sort(key=lambda x: x["max_probability"])
    plasmid_count = sum(count for lbl, count in label_dist.items() if lbl.startswith("plasmid"))
    chromosome_count = sum(count for lbl, count in label_dist.items() if lbl.startswith("chromosome"))
    phage_count = sum(count for lbl, count in label_dist.items() if lbl.startswith("phage"))
    ambiguous_count = sum(count for lbl, count in label_dist.items() if domain4_from_label(lbl) == "ambiguous")

    total_unc = len(uncertainty_rows) or 1
    uncertainty_summary = {
        "mean_max_prob": sum(row["max_prob"] for row in uncertainty_rows) / total_unc,
        "mean_margin": sum(row["margin"] for row in uncertainty_rows) / total_unc,
        "mean_entropy": sum(row["entropy"] for row in uncertainty_rows) / total_unc,
        "mean_uncertainty_score": sum(row["uncertainty_score"] for row in uncertainty_rows) / total_unc,
    }

    summary = {
        "task": task,
        "requested_mode": requested_mode,
        "used_mode": used_mode,
        "fallback_reason": fallback_reason,
        "threshold": threshold,
        "total_contigs": len(rows),
        "label_distribution": label_dist,
        "plasmid_count": plasmid_count,
        "chromosome_count": chromosome_count,
        "phage_count": phage_count,
        "unclassified_count": len(rows) - plasmid_count - chromosome_count - phage_count,
        "plasmid_fraction": plasmid_count / len(rows) if rows else 0.0,
        "uncertain_count": len(uncertain_contigs),
        "uncertain_fraction": len(uncertain_contigs) / len(rows) if rows else 0.0,
        "top_probability": {
            "min": min(top_probs) if top_probs else 0.0,
            "mean": mean(top_probs) if top_probs else 0.0,
            "median": median(top_probs) if top_probs else 0.0,
            "max": max(top_probs) if top_probs else 0.0,
        },
        "uncertainty_summary": uncertainty_summary,
        "most_uncertain_contigs": uncertain_contigs[:20],
        "metrics": _merge_metrics(metrics),
        "preprocessing": preprocessing or {},
        "qc": qc or {},
        "warnings": warnings or [],
    }
    if task == "domain4":
        dist4 = {"plasmid": 0, "chromosome": 0, "phage": 0, "ambiguous": 0}
        for label, count in label_dist.items():
            mapped = domain4_from_label(label)
            dist4[mapped] = dist4.get(mapped, 0) + int(count)
        summary["class_distribution_4way"] = {
            "plasmid": dist4["plasmid"],
            "chromosome": dist4["chromosome"],
            "phage": dist4["phage"],
            "ambiguous": dist4["ambiguous"],
        }
    return summary


def write_report_json(summary: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _dict_rows_to_html_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "<p>No rows</p>"
    parts = ["<table><thead><tr>"]
    for head in headers:
        parts.append(f"<th>{html.escape(head)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        for head in headers:
            val = row.get(head, "")
            parts.append(f"<td>{html.escape(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _fmt_metric(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return "n/a"
    return format(val, fmt)


def write_report_html(summary: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    label_rows = [
        {"label": label, "count": count}
        for label, count in sorted(summary["label_distribution"].items(), key=lambda x: x[1], reverse=True)
    ]

    uncertain_rows = summary.get("most_uncertain_contigs", [])
    metrics = summary.get("metrics", {})
    binary_metrics = metrics.get("binary_domain", {}) if isinstance(metrics, dict) else {}
    domain4_metrics = metrics.get("domain4", {}) if isinstance(metrics, dict) else {}
    calibration_metrics = metrics.get("calibration", {}) if isinstance(metrics, dict) else {}
    unc_summary = summary.get("uncertainty_summary", {})

    style = """
    body { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; margin: 24px; color: #1d2939; }
    h1, h2 { margin-bottom: 8px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 12px; margin-bottom: 20px; }
    .card { border: 1px solid #d0d5dd; border-radius: 12px; padding: 12px; background: #f8fafc; }
    .k { font-size: 12px; color: #475467; }
    .v { font-size: 20px; font-weight: 700; color: #101828; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
    th, td { border: 1px solid #d0d5dd; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #eaecf0; }
    .meta { color: #475467; font-size: 13px; }
    """

    html_content = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>PlasFlow v2 Report</title>
<style>{style}</style>
</head>
<body>
  <h1>PlasFlow v2 Classification Report</h1>
  <p class=\"meta\">Task: {html.escape(str(summary.get('task', 'legacy28')))} | Requested mode: {html.escape(str(summary['requested_mode']))} | Used mode: {html.escape(str(summary['used_mode']))}</p>
  <p class=\"meta\">Threshold: {summary['threshold']:.3f}</p>
  <div class=\"cards\">
    <div class=\"card\"><div class=\"k\">Total contigs</div><div class=\"v\">{summary['total_contigs']}</div></div>
    <div class=\"card\"><div class=\"k\">Plasmid count</div><div class=\"v\">{summary['plasmid_count']}</div></div>
    <div class=\"card\"><div class=\"k\">Chromosome count</div><div class=\"v\">{summary['chromosome_count']}</div></div>
    <div class=\"card\"><div class=\"k\">Unclassified count</div><div class=\"v\">{summary['unclassified_count']}</div></div>
    <div class=\"card\"><div class=\"k\">Uncertain fraction</div><div class=\"v\">{summary['uncertain_fraction']:.2%}</div></div>
    <div class=\"card\"><div class=\"k\">Binary Macro-F1</div><div class=\"v\">{_fmt_metric(binary_metrics.get('macro_f1'))}</div></div>
    <div class=\"card\"><div class=\"k\">Domain4 Macro-F1</div><div class=\"v\">{_fmt_metric(domain4_metrics.get('macro_f1'))}</div></div>
    <div class=\"card\"><div class=\"k\">ECE</div><div class=\"v\">{_fmt_metric(calibration_metrics.get('ece'))}</div></div>
    <div class=\"card\"><div class=\"k\">Recommended threshold</div><div class=\"v\">{_fmt_metric(calibration_metrics.get('recommended_threshold'), '.3f')}</div></div>
    <div class=\"card\"><div class=\"k\">Mean uncertainty</div><div class=\"v\">{_fmt_metric(unc_summary.get('mean_uncertainty_score'))}</div></div>
  </div>
  <h2>Label distribution</h2>
  {_dict_rows_to_html_table(label_rows, ['label', 'count'])}
  <h2>Most uncertain contigs</h2>
  {_dict_rows_to_html_table(uncertain_rows, ['contig_name', 'contig_length', 'max_probability', 'top_label', 'assigned_label', 'margin', 'entropy', 'uncertainty_score'])}
</body>
</html>
"""

    out.write_text(html_content, encoding="utf-8")


def load_rows_from_tsv(tsv_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(tsv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            normalized = dict(row)
            if "" in normalized:
                normalized.pop("", None)
            rows.append(normalized)
    return rows


def generate_report_from_tsv(
    tsv_path: str | Path,
    html_output: str | Path,
    labels: list[str],
    threshold: float,
    requested_mode: str = "report",
    used_mode: str = "report",
    metrics: dict[str, Any] | None = None,
    task: TaskType = "legacy28",
) -> dict[str, Any]:
    rows = load_rows_from_tsv(tsv_path)
    summary = build_summary(
        rows=rows,
        labels=labels,
        threshold=threshold,
        requested_mode=requested_mode,
        used_mode=used_mode,
        fallback_reason=None,
        metrics=metrics,
        task=task,
    )
    html_path = Path(html_output)
    json_path = html_path.with_suffix(".json")
    write_report_json(summary, json_path)
    write_report_html(summary, html_path)
    return summary
