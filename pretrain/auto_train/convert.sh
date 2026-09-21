#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <checkpoint_dir> [output_dir] [source_hf_model_path]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-"${CHECKPOINT_DIR}/../converted"}"
# Default source HF model: relative to the pretrain/ dir (REPO_ROOT).
# Override with $3 or by exporting SOURCE_HF_MODEL_PATH.
SOURCE_HF_MODEL_PATH="${3:-${SOURCE_HF_MODEL_PATH:-${REPO_ROOT}/../models/pretrain-8B/converted}}"

# Use the converter that ships with this repo (auto_train/ is one level below
# the pretrain dir which contains tools/model_converter/...). Override by
# setting CONVERTER_PY in the environment.
CONVERTER_PY="${CONVERTER_PY:-${REPO_ROOT}/tools/model_converter/convert_checkpoint_to_hf.py}"

python3 "${CONVERTER_PY}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --source_hf_model_path "${SOURCE_HF_MODEL_PATH}"
