#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TASK1_CONFIG="${TASK1_CONFIG:-configs/full.yaml}"
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task1_char_gpt.ipynb
