#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 2 (eval PPL) — score target-SID PPL using the warm-up scorer
# ============================================================================
# Input:  preprocessed_sid_cot_ver5/  (from step 1_preprocess.sh)
# Output: ppl_eval_sft_cot_ver5/
#
# Scorer: the 200-step warm-up SFT checkpoint
#   ${PREFIX}/models/r2sft_warmup_200steps/step200/converted/
#   Trained via: cd pretrain && python3 auto_train/auto_train.py auto_train/r2sft_warmup_200steps.yaml
#
# Usage:
#   bash 2_eval_ppl.sh
#   bash 2_eval_ppl.sh /etc/mpi/hostfile
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"

INPUT_URI="${INPUT_URI:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/preprocessed_sid_cot_ver5}"
OUTPUT_URI="${OUTPUT_URI:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_cot_ver5}"
MODEL_PATH="${MODEL_PATH:-${PREFIX}/models/r2sft_warmup_200steps/step200/converted/}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"

UPSTREAM_SH="${SCRIPT_DIR}/../2_run_mpi_ppl_sft.sh"
[ ! -f "${UPSTREAM_SH}" ] && echo "[ERROR] ${UPSTREAM_SH} not found" >&2 && exit 1
[ ! -d "${INPUT_URI}" ] && echo "[ERROR] preprocess output not found: ${INPUT_URI}" >&2 && exit 1
[ ! -d "${MODEL_PATH}" ] && echo "[ERROR] step200 model dir not found: ${MODEL_PATH}" >&2 && exit 1
[ ! -f "${HOSTFILE}" ] && echo "[ERROR] hostfile not found: ${HOSTFILE}" >&2 && exit 1

echo "========================================"
echo "Step 2: eval PPL (warm-up step200 scorer)"
echo "========================================"
echo "INPUT_URI  : ${INPUT_URI}"
echo "OUTPUT_URI : ${OUTPUT_URI}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "HOSTFILE   : ${HOSTFILE}"
echo "========================================"
echo ""

INPUT_URI="${INPUT_URI}" \
OUTPUT_URI="${OUTPUT_URI}" \
MODEL_PATH="${MODEL_PATH}" \
REPRESENTATION_TYPE=sid \
bash "${UPSTREAM_SH}" "${HOSTFILE}"

echo ""
echo "eval PPL done: ${OUTPUT_URI}"
