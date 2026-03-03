# PlasFlow v2

PlasFlow v2 is a Python 3.11 rebuild focused on:
- v1-compatible outputs (`.tsv` + three FASTA bins)
- v2 runtime and packaging
- API + web workflow for local/server usage
- confidence reporting (`.report.json` and `.report.html`)
- reproducible v2 model training/evaluation (`binary_domain`, LightGBM)

This v2 implementation coexists with the v1 script in this repository.

## Quick Start

### 1) Install

```bash
pip install -e .
```

For full dev/train environment:

```bash
./scripts/setup_dev_env.sh
```

### 2) Classify

```bash
plasflow classify \
  --input test/Citrobacter_freundii_strain_CAV1321_scaffolds.fasta \
  --output runs/demo/result \
  --mode v1 \
  --task legacy28 \
  --threshold 0.7
```

If v1 dependencies are unavailable, `v1` mode falls back to `v2` mode by default.

### 3) Generate report from existing TSV

```bash
plasflow report --input runs/demo/result.tsv --out runs/demo/result.report.html
```

### 4) Run API server

```bash
plasflow serve --host 0.0.0.0 --port 8080
```

## CLI

- `plasflow classify --input <fasta> --output <prefix> --mode v1|v2 --task legacy28|binary_domain|domain4 --threshold 0.7`
- `plasflow serve --host 0.0.0.0 --port 8080`
- `plasflow report --input <output.tsv> --out <report.html>`
- `plasflow train-v2 --dataset-manifest <path> --outdir <dir> --task binary_domain --model lightgbm` (`train-modern` alias)
- `plasflow evaluate-v2 --input <labeled.tsv> --model-dir <dir> --task binary_domain --out <eval.json>` (`evaluate-modern` alias)
- `plasflow compare-modes --input <fasta> --outdir <dir> --threshold 0.7 [--ground-truth <labeled.tsv>]`

## Dataset Manifest (train-v2)

`train-v2` accepts JSON/YAML manifest for reproducible data prep.

Example (`dataset_manifest.json`):

```json
{
  "name": "plasflow-binary-domain",
  "random_seed": 42,
  "min_length": 1000,
  "deduplicate": true,
  "split": {
    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "group_col": "accession",
    "split_col": "split"
  },
  "sources": [
    {
      "name": "refseq",
      "path": "data/refseq_labeled.tsv",
      "format": "tsv",
      "sequence_col": "sequence",
      "label_col": "label",
      "license_note": "NCBI RefSeq"
    }
  ]
}
```

Fallback path is still supported:

```bash
plasflow train-v2 --input data/labeled.tsv --outdir models_v2/current
```

## v2 Model Bundle

Default output bundle directory: `models_v2/current/`

- `domain_model.joblib`
- `calibrator.joblib`
- `feature_manifest.json`
- `metadata.json`

`metadata.json` contains:
- dataset fingerprint hash
- random seed
- split counts
- validation/test metrics
- calibration stats (ECE, Brier score)
- recommended threshold

## Evaluation Outputs

`evaluate-v2` writes:
- `<out>.json` (evaluation summary)
- `confusion_matrix.csv`
- `threshold_curve.csv`

Example:

```bash
plasflow evaluate-v2 \
  --input data/test_labeled.tsv \
  --model-dir models_v2/current \
  --task binary_domain \
  --out runs/eval/eval.json
```

## Automated Train + Eval

Single-command training/evaluation pipeline:

```bash
python scripts/run_modern_train_eval.py \
  --dataset-manifest tests/fixtures/benchmark/benchmark_manifest.json \
  --outdir runs/train_eval_demo
```

Artifacts:
- `runs/train_eval_demo/model_bundle/*`
- `runs/train_eval_demo/eval.json`
- `runs/train_eval_demo/confusion_matrix.csv`
- `runs/train_eval_demo/threshold_curve.csv`
- `runs/train_eval_demo/train_eval_summary.json`

## Accuracy Gate (Baseline 대비 +0.05)

Benchmark generation (baseline + candidate):

```bash
python scripts/run_modern_benchmark.py \
  --dataset-manifest tests/fixtures/benchmark/benchmark_manifest.json \
  --outdir runs/ci_benchmark \
  --threshold 0.7
```

Acceptance check:

```bash
python scripts/check_accuracy_gate.py \
  --baseline runs/ci_benchmark/baseline_eval.json \
  --candidate runs/ci_benchmark/candidate_eval.json \
  --min-delta 0.05 \
  --min-candidate 0.60
```

## API

- `POST /api/v1/jobs` (multipart: `file`, `mode`, `threshold`) -> `{ job_id }`
- `GET /api/v1/jobs/{job_id}` -> status and timing
- `GET /api/v1/jobs/{job_id}/artifacts` -> generated artifacts
- `GET /api/v1/jobs/{job_id}/download/{artifact_name}` -> file download

Supported artifact names:
- `tsv`
- `plasmids.fasta`
- `chromosomes.fasta`
- `unclassified.fasta`
- `report.json`
- `report.html`

## Web UI

Web source is in `plasflow-web/` (React + TypeScript).

```bash
cd plasflow-web
npm install
npm run dev
```

Default API base is `/api/v1` (relative path).  
In dev mode, Vite proxy forwards `/api/*` to `VITE_PROXY_TARGET` (default: `http://localhost:8080`).

The UI reads `report.json` and shows model quality signals (Macro-F1, ECE, recommended threshold) when available.

## Docker Compose

```bash
docker compose up --build
```

- API: <http://localhost:8080>
- Web: <http://localhost:5173>

In Docker, web uses:
- `VITE_API_BASE=/api/v1`
- `VITE_PROXY_TARGET=http://api:8080`

This avoids direct browser calls to `localhost:8080` for API requests.

## Tests

```bash
pytest
```

### Web E2E (Playwright)

Run from `plasflow-web/`:

```bash
npm install
npm run e2e:install
npm run e2e:test
```

Optional:
- headed mode: `npm run e2e:test:headed`
- custom base URL: `E2E_BASE_URL=http://localhost:5173 npm run e2e:test`

E2E uses a small FASTA fixture for faster CI runs:
- `plasflow-web/e2e/fixtures/smoke-small.fasta`

CI runs both `pytest` and Docker-backed Playwright E2E on every push/PR:
- workflow: `.github/workflows/ci.yml`

## Notes on Modes

- `v1` executes `PlasFlow.py` through a subprocess adapter.
- `v2` uses the native v2 classifier pipeline and model bundle.
- `legacy`/`modern` aliases are still accepted for compatibility.

In `v1` mode, when the old runtime is unavailable, v2 records the fallback reason and continues with `v2` mode (unless `--no-fallback` is set).
