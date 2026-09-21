#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 2 (cot) — RecIF CoT synthesis (single-host, multi-threaded)
# ============================================================================
# Input:  Step 1 (1_history_text) output parquet (video + ad samples)
# Output: synthesized CoT parquet, schema:
#   sample_key, uid, target_domain, cot,
#   history_pids, history_captions, target_pids, target_captions, history_meta
#
# The teacher LLM must NOT see the target item (target-blind synthesis).
# video samples use prompt.txt, ad samples use prompt_ad.txt (two-segment history).
#
# Usage:
#   API_CONFIG_PATH=/path/to/api_config.json bash run.sh
#   API_CONFIG_PATH=api_config.json NUM_WORKERS=64 MAX_SAMPLES=0 bash run.sh
#
# Required env:
#   API_CONFIG_PATH  JSON with {base_url, api_key, model} (OpenAI-compatible)
#
# Optional env:
#   NUM_WORKERS         concurrency, default 16
#   INPUT_DIR           Step 1 output dir (has default)
#   OUTPUT_DIR          Step 2 output dir (has default)
#   MODEL               model name (overridden by config "model" if present)
#   MAX_SAMPLES         max new samples this run, 0 = unlimited, default 0
#   CHECKPOINT_INTERVAL flush parquet every N rows, default 64
#   DRY_RUN             1 = skip API calls, write placeholder cot
# ============================================================================

# ============== Path prefix ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

DEFAULT_INPUT_DIR="${PREFIX}/data/RecIF-cot/steps/1_history_text"
DEFAULT_OUTPUT_DIR="${PREFIX}/data/RecIF-cot/steps/2_cot"

INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
export INPUT_DIR OUTPUT_DIR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MPI_ENTRY="${SCRIPT_DIR}/main.py"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODEL="${MODEL:-gpt-4o-mini}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-64}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/cot_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -f "$MPI_ENTRY" ]; then
    echo "[ERROR] entry script not found: $MPI_ENTRY" >&2
    exit 1
fi

if [ "${DRY_RUN}" = "0" ] && [ -z "${API_CONFIG_PATH:-}" ]; then
    echo "[ERROR] API_CONFIG_PATH is required (JSON with base_url, api_key, model)" >&2
    echo "        or set DRY_RUN=1 for a local dry run" >&2
    exit 1
fi

case "$OUTPUT_DIR" in
    viewfs://*|hdfs://*) ;;
    *) mkdir -p "$OUTPUT_DIR" ;;
esac

PY_ARGS=(
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --model "$MODEL"
    --num_workers "$NUM_WORKERS"
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --max_samples "$MAX_SAMPLES"
)
if [ "${DRY_RUN}" = "1" ]; then
    PY_ARGS+=(--dry_run)
fi
if [ -n "${API_CONFIG_PATH:-}" ]; then
    PY_ARGS+=(--api_config_path "$API_CONFIG_PATH")
fi

echo "========================================"
echo "Step 2 (cot) — RecIF CoT synthesis"
echo "========================================"
echo "ENTRY             : ${MPI_ENTRY}"
echo "INPUT_DIR         : ${INPUT_DIR}"
echo "OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "MODEL             : ${MODEL}"
echo "NUM_WORKERS       : ${NUM_WORKERS}"
echo "MAX_SAMPLES       : ${MAX_SAMPLES}"
echo "CHECKPOINT_INTERVAL: ${CHECKPOINT_INTERVAL}"
echo "DRY_RUN           : ${DRY_RUN}"
echo "API_CONFIG_PATH   : ${API_CONFIG_PATH:-<none>}"
echo "LOG_FILE          : ${LOG_FILE}"
echo ""

"$PYTHON" "$MPI_ENTRY" "${PY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
