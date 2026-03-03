#!/usr/bin/env bash
set -euo pipefail

WITH_WEB="${1:-}"

python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,train]"

if [[ "$WITH_WEB" == "--with-web" ]]; then
  npm install --prefix plasflow-web
fi

echo "Environment setup completed."
