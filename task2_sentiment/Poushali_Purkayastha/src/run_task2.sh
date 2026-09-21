#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TASK2_CONFIG="${TASK2_CONFIG:-configs/full.yaml}"
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task2_sentiment.ipynb
