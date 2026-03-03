from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

from .constants import (
    DEFAULT_MODELS_V2_DIR,
    LabelSpec,
    TaskType,
    chromosome_labels,
    plasmid_labels,
)
from .features import default_feature_manifest, sequence_features, vectorize_sequence
from .io import ContigRecord
from .metrics import domain4_from_label


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize(values: list[float]) -> list[float]:
    clipped = [max(float(v), 1e-12) for v in values]
    total = sum(clipped)
    return [v / total for v in clipped]


TAXON_GC_CENTER: dict[str, float] = {
    "Acidobacteria": 0.60,
    "Actinobacteria": 0.70,
    "Bacteroidetes": 0.42,
    "Chlamydiae": 0.41,
    "Chlorobi": 0.50,
    "Chloroflexi": 0.53,
    "Cyanobacteria": 0.47,
    "DeinococcusThermus": 0.67,
    "Firmicutes": 0.40,
    "Fusobacteria": 0.31,
    "Nitrospirae": 0.56,
    "Planctomycetes": 0.55,
    "Proteobacteria": 0.53,
    "Spirochaetes": 0.40,
    "Tenericutes": 0.27,
    "Thermotogae": 0.46,
    "Verrucomicrobia": 0.55,
    "other": 0.50,
}


@dataclass
class PredictionOutput:
    labels: list[str]
    probabilities: list[list[float]]
    predicted_ids: list[int]


class BaseModernModel:
    model_task: str = "legacy28"

    def predict_proba(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        raise NotImplementedError

    def model_metrics(self) -> dict[str, Any] | None:
        return None

    def recommended_threshold(self) -> float | None:
        return None


class HeuristicModernModel(BaseModernModel):
    def __init__(self, task: TaskType):
        self.model_task = task

    def _taxon_scores(self, gc: float, taxons: list[str], entropy4: float) -> dict[str, float]:
        scores: dict[str, float] = {}
        for tax in taxons:
            center = TAXON_GC_CENTER.get(tax, 0.5)
            spread = 0.09 if tax == "other" else 0.07
            z = (gc - center) / spread
            score = math.exp(-0.5 * z * z) * (0.8 + 0.4 * entropy4)
            scores[tax] = max(score, 1e-9)
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def _plasmid_probability(self, length: float, gc: float, entropy4: float, n_frac: float) -> float:
        len_term = -0.55 * math.log10(max(length, 1.0))
        gc_term = -8.0 * (gc - 0.5) ** 2
        entropy_term = 4.4 * (entropy4 - 0.70)
        n_penalty = -2.4 * n_frac
        raw = 0.8 + len_term + gc_term + entropy_term + n_penalty
        prob = _sigmoid(raw)
        return float(min(max(prob, 0.01), 0.99))

    def _predict_legacy28(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        labels = label_spec.labels
        p_labels = plasmid_labels(labels)
        c_labels = chromosome_labels(labels)

        p_tax = sorted({lbl.split(".", 1)[1] for lbl in p_labels})
        c_tax = sorted({lbl.split(".", 1)[1] for lbl in c_labels})

        out: list[list[float]] = [[0.0 for _ in labels] for _ in records]
        for i, rec in enumerate(records):
            feats = sequence_features(rec.sequence)
            plasmid_prob = self._plasmid_probability(
                length=feats["length"],
                gc=feats["gc"],
                entropy4=feats["entropy4"],
                n_frac=feats["n_frac"],
            )
            chromosome_prob = 1.0 - plasmid_prob
            p_tax_scores = self._taxon_scores(feats["gc"], p_tax, feats["entropy4"])
            c_tax_scores = self._taxon_scores(feats["gc"], c_tax, feats["entropy4"])

            for j, label in enumerate(labels):
                domain, tax = label.split(".", 1)
                if domain == "plasmid":
                    out[i][j] = plasmid_prob * p_tax_scores.get(tax, 1e-9)
                else:
                    out[i][j] = chromosome_prob * c_tax_scores.get(tax, 1e-9)
            out[i] = _normalize(out[i])
        return out

    def _predict_binary(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        out: list[list[float]] = []
        for rec in records:
            feats = sequence_features(rec.sequence)
            plasmid_prob = self._plasmid_probability(
                length=feats["length"],
                gc=feats["gc"],
                entropy4=feats["entropy4"],
                n_frac=feats["n_frac"],
            )
            row = []
            for label in label_spec.labels:
                if label == "plasmid":
                    row.append(plasmid_prob)
                elif label == "chromosome":
                    row.append(1.0 - plasmid_prob)
                else:
                    row.append(1e-9)
            out.append(_normalize(row))
        return out

    def _predict_domain4(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        out: list[list[float]] = []
        for rec in records:
            feats = sequence_features(rec.sequence)
            plasmid_prob = self._plasmid_probability(
                length=feats["length"],
                gc=feats["gc"],
                entropy4=feats["entropy4"],
                n_frac=feats["n_frac"],
            )
            chromosome_prob = 1.0 - plasmid_prob

            phage_score = max(0.01, (1.0 - abs(feats["gc"] - 0.5) * 2.2) * min(1.0, feats["entropy4"] / 0.8) * 0.35)
            ambiguous_score = max(0.01, 0.2 + feats["n_frac"] * 1.5 + (0.45 - abs(plasmid_prob - 0.5)) * 0.25)
            score_map = {
                "plasmid": plasmid_prob * 0.75,
                "chromosome": chromosome_prob * 0.75,
                "phage": phage_score,
                "ambiguous": ambiguous_score,
            }
            row = [score_map.get(label, 1e-9) for label in label_spec.labels]
            out.append(_normalize(row))
        return out

    def predict_proba(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        if self.model_task == "legacy28":
            return self._predict_legacy28(records, label_spec)
        if self.model_task == "binary_domain":
            return self._predict_binary(records, label_spec)
        if self.model_task == "domain4":
            return self._predict_domain4(records, label_spec)
        raise ValueError(f"Unsupported heuristic task: {self.model_task}")


class JoblibModernModel(BaseModernModel):
    def __init__(self, model_path: Path):
        import joblib

        payload: dict[str, Any] = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_order = payload.get("feature_order", ["length", "log_length", "gc", "entropy4", "n_frac"])
        self.model_task = str(payload.get("task", "legacy28"))

    def _vectorize(self, records: list[ContigRecord]) -> list[list[float]]:
        rows: list[list[float]] = []
        for rec in records:
            feats = sequence_features(rec.sequence)
            rows.append([float(feats[k]) for k in self.feature_order])
        return rows

    def predict_proba(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        matrix = self._vectorize(records)
        probs = self.model.predict_proba(matrix)

        out: list[list[float]] = [[0.0 for _ in label_spec.labels] for _ in range(len(matrix))]
        class_values = [str(v) for v in list(self.model.classes_)]
        label_to_id = label_spec.label_to_id

        for class_idx, class_name in enumerate(class_values):
            mapped = domain4_from_label(class_name) if self.model_task == "domain4" else class_name
            if mapped not in label_to_id:
                continue
            out_idx = label_to_id[mapped]
            for row_idx in range(len(out)):
                out[row_idx][out_idx] = float(probs[row_idx][class_idx])

        for i, row in enumerate(out):
            out[i] = _normalize(row)
        return out


class DomainBundleModernModel(BaseModernModel):
    def __init__(self, bundle_dir: Path, task: TaskType):
        import joblib

        self.bundle_dir = bundle_dir
        self.requested_task = task
        self.domain_model = joblib.load(bundle_dir / "domain_model.joblib")
        self.calibrator_payload = None
        calibrator_path = bundle_dir / "calibrator.joblib"
        if calibrator_path.exists():
            self.calibrator_payload = joblib.load(calibrator_path)

        feature_manifest_path = bundle_dir / "feature_manifest.json"
        if feature_manifest_path.exists():
            self.feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        else:
            self.feature_manifest = default_feature_manifest()

        metadata_path = bundle_dir / "metadata.json"
        self.metadata: dict[str, Any] = {}
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.model_task = str(self.metadata.get("task", "binary_domain"))
        self.taxon_prior = self.metadata.get("taxon_prior", {})

    def _calibrate_binary(self, probs: list[float]) -> list[float]:
        payload = self.calibrator_payload
        if not payload:
            return [float(p) for p in probs]
        cal_type = str(payload.get("type", "")).lower()
        model = payload.get("model")
        if model is None:
            return [float(p) for p in probs]
        if cal_type == "isotonic":
            return [float(x) for x in model.predict(probs)]
        if cal_type == "platt":
            out = model.predict_proba([[p] for p in probs])
            return [float(row[1]) for row in out]
        return [float(p) for p in probs]

    def _calibrate_multiclass(self, probs: list[list[float]]) -> list[list[float]]:
        payload = self.calibrator_payload
        if not payload:
            return [_normalize(row) for row in probs]
        if str(payload.get("type", "")).lower() != "temperature":
            return [_normalize(row) for row in probs]
        temperature = float(payload.get("temperature", 1.0))
        if temperature <= 0:
            temperature = 1.0
        calibrated: list[list[float]] = []
        for row in probs:
            logits = [math.log(max(float(p), 1e-12)) / temperature for p in row]
            max_logit = max(logits)
            exps = [math.exp(x - max_logit) for x in logits]
            calibrated.append(_normalize(exps))
        return calibrated

    def _domain_positive_index(self) -> int:
        classes = [str(c) for c in list(getattr(self.domain_model, "classes_", ["chromosome", "plasmid"]))]
        if "plasmid" in classes:
            return classes.index("plasmid")
        return min(1, len(classes) - 1)

    def _taxon_scores(self, gc: float, taxons: list[str], entropy4: float) -> dict[str, float]:
        scores: dict[str, float] = {}
        for tax in taxons:
            center = TAXON_GC_CENTER.get(tax, 0.5)
            spread = 0.09 if tax == "other" else 0.07
            z = (gc - center) / spread
            score = math.exp(-0.5 * z * z) * (0.8 + 0.4 * entropy4)
            scores[tax] = max(score, 1e-9)
        total = sum(scores.values())
        if total <= 0:
            return {tax: 1.0 / max(len(taxons), 1) for tax in taxons}
        return {k: v / total for k, v in scores.items()}

    def _domain_taxon_distribution(self, domain: str, taxons: list[str], gc: float, entropy4: float) -> dict[str, float]:
        domain_prior = self.taxon_prior.get(domain, {}) if isinstance(self.taxon_prior, dict) else {}
        prior = {tax: float(domain_prior.get(tax, 0.0)) for tax in taxons if float(domain_prior.get(tax, 0.0)) > 0}
        if prior:
            total = sum(prior.values())
            return {tax: prior.get(tax, 0.0) / total for tax in taxons}
        return self._taxon_scores(gc=gc, taxons=taxons, entropy4=entropy4)

    def _predict_binary_task(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        labels = label_spec.labels
        matrix = [vectorize_sequence(rec.sequence, self.feature_manifest) for rec in records]
        model_probs = self.domain_model.predict_proba(matrix)
        pos_idx = self._domain_positive_index()
        raw_plasmid = [float(row[pos_idx]) for row in model_probs]
        calibrated_plasmid = self._calibrate_binary(raw_plasmid)

        # If the task labels are binary labels, map directly.
        if set(labels) == {"plasmid", "chromosome"}:
            out = []
            for plasmid_prob in calibrated_plasmid:
                row = []
                for label in labels:
                    if label == "plasmid":
                        row.append(plasmid_prob)
                    elif label == "chromosome":
                        row.append(1.0 - plasmid_prob)
                    else:
                        row.append(1e-9)
                out.append(_normalize(row))
            return out

        # For legacy28 outputs, spread domain probability over taxon priors.
        p_labels = plasmid_labels(labels)
        c_labels = chromosome_labels(labels)
        p_tax = sorted({lbl.split(".", 1)[1] for lbl in p_labels})
        c_tax = sorted({lbl.split(".", 1)[1] for lbl in c_labels})

        out: list[list[float]] = [[0.0 for _ in labels] for _ in records]
        for i, rec in enumerate(records):
            feats = sequence_features(rec.sequence)
            plasmid_prob = float(min(max(calibrated_plasmid[i], 1e-6), 1.0 - 1e-6))
            chromosome_prob = 1.0 - plasmid_prob
            p_tax_scores = self._domain_taxon_distribution("plasmid", p_tax, feats["gc"], feats["entropy4"])
            c_tax_scores = self._domain_taxon_distribution("chromosome", c_tax, feats["gc"], feats["entropy4"])

            for j, label in enumerate(labels):
                domain, tax = label.split(".", 1)
                if domain == "plasmid":
                    out[i][j] = plasmid_prob * p_tax_scores.get(tax, 1e-9)
                else:
                    out[i][j] = chromosome_prob * c_tax_scores.get(tax, 1e-9)
            out[i] = _normalize(out[i])
        return out

    def _predict_domain4_task(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        matrix = [vectorize_sequence(rec.sequence, self.feature_manifest) for rec in records]
        model_probs = [[float(v) for v in row] for row in self.domain_model.predict_proba(matrix)]
        calibrated = self._calibrate_multiclass(model_probs)
        class_values = [str(v) for v in list(self.domain_model.classes_)]
        label_to_id = label_spec.label_to_id
        out: list[list[float]] = [[0.0 for _ in label_spec.labels] for _ in calibrated]

        for class_idx, class_name in enumerate(class_values):
            mapped = domain4_from_label(class_name)
            if mapped not in label_to_id:
                continue
            target_idx = label_to_id[mapped]
            for row_idx in range(len(out)):
                out[row_idx][target_idx] += calibrated[row_idx][class_idx]

        for idx in range(len(out)):
            out[idx] = _normalize(out[idx])
        return out

    def predict_proba(self, records: list[ContigRecord], label_spec: LabelSpec) -> list[list[float]]:
        if self.model_task == "domain4":
            return self._predict_domain4_task(records, label_spec)
        return self._predict_binary_task(records, label_spec)

    def model_metrics(self) -> dict[str, Any] | None:
        validation = self.metadata.get("validation_metrics") or {}
        calibration = self.metadata.get("calibration") or {}
        if self.model_task == "domain4":
            return {
                "domain4": {
                    "macro_f1": validation.get("macro_f1"),
                    "precision_macro": validation.get("precision_macro"),
                    "recall_macro": validation.get("recall_macro"),
                    "accuracy": validation.get("accuracy"),
                    "confusion_matrix": validation.get("confusion_matrix"),
                    "support": validation.get("support"),
                },
                "calibration": {
                    "ece": calibration.get("ece"),
                    "brier_score": calibration.get("brier_score"),
                    "recommended_threshold": calibration.get("recommended_threshold"),
                },
            }
        return {
            "binary_domain": {
                "macro_f1": validation.get("macro_f1"),
                "precision_macro": validation.get("precision_macro"),
                "recall_macro": validation.get("recall_macro"),
                "accuracy": validation.get("accuracy"),
                "confusion_matrix": validation.get("confusion_matrix"),
                "support": validation.get("support"),
            },
            "calibration": {
                "ece": calibration.get("ece"),
                "brier_score": calibration.get("brier_score"),
                "recommended_threshold": calibration.get("recommended_threshold"),
            },
        }

    def recommended_threshold(self) -> float | None:
        calibration = self.metadata.get("calibration") if isinstance(self.metadata, dict) else None
        if not isinstance(calibration, dict):
            return None
        value = calibration.get("recommended_threshold")
        if value is None:
            return None
        return float(value)


class ModernClassifier:
    def __init__(self, models_dir: Path | None = None, bundle_dir: Path | None = None, task: TaskType = "legacy28"):
        self.models_dir = models_dir or DEFAULT_MODELS_V2_DIR
        self.bundle_dir = bundle_dir
        self.task = task
        self.model = self._load_model()

    def _load_from_dir(self, model_dir: Path) -> BaseModernModel | None:
        if (model_dir / "domain_model.joblib").exists():
            return DomainBundleModernModel(model_dir, task=self.task)
        if (model_dir / "model.joblib").exists():
            return JoblibModernModel(model_dir / "model.joblib")
        return None

    def _validate_task_compatibility(self, model: BaseModernModel) -> None:
        model_task = str(getattr(model, "model_task", self.task))
        if model_task == "legacy28" and self.task != "legacy28":
            raise ValueError(f"Model task mismatch: bundle='{model_task}' requested='{self.task}'")
        if model_task == "binary_domain" and self.task == "domain4":
            raise ValueError(f"Model task mismatch: bundle='{model_task}' requested='{self.task}'")
        if model_task == "domain4" and self.task != "domain4":
            raise ValueError(f"Model task mismatch: bundle='{model_task}' requested='{self.task}'")

    def _load_model(self) -> BaseModernModel:
        if self.bundle_dir:
            loaded = self._load_from_dir(self.bundle_dir)
            if loaded:
                self._validate_task_compatibility(loaded)
                return loaded
            raise FileNotFoundError(f"Model bundle not found under: {self.bundle_dir}")

        current_dir = self.models_dir / "current"
        loaded = self._load_from_dir(current_dir)
        if loaded:
            self._validate_task_compatibility(loaded)
            return loaded

        loaded_root = self._load_from_dir(self.models_dir)
        if loaded_root:
            self._validate_task_compatibility(loaded_root)
            return loaded_root

        if self.task == "domain4":
            raise FileNotFoundError(
                "No domain4 model bundle found under models_v2/current. "
                "Heuristic fallback is disabled for task=domain4."
            )
        return HeuristicModernModel(task=self.task)

    def predict(self, records: list[ContigRecord], label_spec: LabelSpec) -> PredictionOutput:
        probs = self.model.predict_proba(records, label_spec)
        predicted = [max(range(len(row)), key=lambda idx: row[idx]) for row in probs]
        return PredictionOutput(labels=label_spec.labels, probabilities=probs, predicted_ids=predicted)

    def model_metrics(self) -> dict[str, Any] | None:
        return self.model.model_metrics()

    def recommended_threshold(self) -> float | None:
        return self.model.recommended_threshold()


def save_model_metadata(models_dir: Path, metadata: dict[str, Any]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
