# PlasFlow (v1 + v2)

PlasFlow is a toolkit for classifying metagenomic contigs with backward-compatible v1 behavior and an extended v2 pipeline.

This repository currently provides:
- `v1` execution path for legacy compatibility (`PlasFlow.py` wrapper)
- `v2` Python pipeline with preprocessing, uncertainty reporting, and API/Web workflows
- tasks: `legacy28`, `binary_domain`, `domain4` (`plasmid/chromosome/phage/ambiguous`)

## Quick Start

### 1) Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,train]'
```

### 2) Classify

```bash
plasflow classify \
  --input test/Citrobacter_freundii_strain_CAV1321_scaffolds.fasta \
  --output runs/demo/result \
  --mode v2 \
  --task legacy28 \
  --threshold 0.7
```

### 3) Generate report

```bash
plasflow report --input runs/demo/result.tsv --out runs/demo/result.report.html
```

## CLI

```bash
plasflow classify --input <fasta> --output <prefix> \
  --mode v1|v2 --task legacy28|binary_domain|domain4 \
  --read-type short|long|hybrid|complete \
  --min-length 1000 --coverage-source header|none \
  --polish none|racon|medaka --threshold 0.7

plasflow train-v2 --dataset-manifest <path> --outdir <dir> --task binary_domain|domain4
plasflow evaluate-v2 --input <labeled.tsv> --model-dir <dir> --task binary_domain|domain4 --out <eval.json>
plasflow compare-modes --input <fasta> --outdir <dir> --threshold 0.7 [--ground-truth <labeled.tsv>]
plasflow tools-check
plasflow serve --host 0.0.0.0 --port 8080
```

Compatibility aliases are accepted:
- `--mode legacy|modern` -> `v1|v2`
- `train-modern` -> `train-v2`
- `evaluate-modern` -> `evaluate-v2`

## API

- `POST /api/v1/jobs` (multipart form)
  - required: `file`
  - optional: `mode`, `task`, `threshold`, `read_type`, `min_length`, `coverage_source`, `circularity_check`, `polish`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/artifacts`
- `GET /api/v1/jobs/{job_id}/download/{artifact_name}`

## Web

```bash
cd plasflow-web
npm install
npm run dev
```

Default proxy target is `http://localhost:8080`.

## Outputs

Core outputs:
- `result.tsv`
- `result_plasmids.fasta`
- `result_chromosomes.fasta`
- `result_unclassified.fasta`
- `result.report.json`
- `result.report.html`

Additional v2/domain4 outputs (when relevant):
- `result_phage.fasta`
- `result_ambiguous.fasta`

## Tests

```bash
. .venv/bin/activate
pytest -q

cd plasflow-web
npm run build
npm run e2e:test
```

## Licensing, Copyright, and Provenance

This repository includes and modifies code from the original PlasFlow project:
- upstream: <https://github.com/smaegol/PlasFlow>
- original paper: Krawczyk PS, Lipinski L, Dziembowski A. Nucleic Acids Res. 2018;46(6):e35. doi:10.1093/nar/gkx1321

The project is distributed under **GNU GPL v3** (see `LICENSE`).

Practical implications for redistribution of this modified repository:
- keep the GPLv3 license text and notices
- keep attribution to the original authors/project
- distribute corresponding source code for conveyed binaries/images
- clearly mark that this is a modified version

For legal interpretation, consult your legal counsel.
