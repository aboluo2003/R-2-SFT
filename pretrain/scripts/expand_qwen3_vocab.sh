#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPAND_SCRIPT="${PRETRAIN_DIR}/tools/model_converter/expand_qwen3_vocab.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ITEMIC_LAYER_N="${ITEMIC_LAYER_N:-3}"
VOCAB_SIZE_PER_LAYER="${VOCAB_SIZE_PER_LAYER:-8192}"

if [[ -z "${HF_MODEL_DIR:-}" ]]; then
    echo "Error: HF_MODEL_DIR is required." >&2
    echo "Example: HF_MODEL_DIR=/path/to/Qwen3-1.7B bash scripts/expand_qwen3_vocab.sh" >&2
    exit 1
fi

if [[ -z "${OUTPUT_MODEL_DIR:-}" ]]; then
    echo "Error: OUTPUT_MODEL_DIR is required." >&2
    echo "Example: OUTPUT_MODEL_DIR=/path/to/Qwen3-1.7B_itemic bash scripts/expand_qwen3_vocab.sh" >&2
    exit 1
fi

if [[ ! -d "${HF_MODEL_DIR}" ]]; then
    echo "Error: input model directory does not exist: ${HF_MODEL_DIR}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${EXPAND_SCRIPT}" \
    --hf_model_dir "${HF_MODEL_DIR}" \
    --output_model_dir "${OUTPUT_MODEL_DIR}" \
    --itemic_layer_n "${ITEMIC_LAYER_N}" \
    --vocab_size_per_layer "${VOCAB_SIZE_PER_LAYER}"
