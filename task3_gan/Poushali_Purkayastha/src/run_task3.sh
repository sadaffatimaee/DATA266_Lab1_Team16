#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TASK3_CONFIG="${TASK3_CONFIG:-configs/full.yaml}"
export TASK3_STAGE="${TASK3_STAGE:-all}"
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task3_cyclegan.ipynb
