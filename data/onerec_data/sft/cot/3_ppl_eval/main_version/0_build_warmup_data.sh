#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 0 (warm-up data) — build raw CoT SFT data for the PPL scorer warm-up
# ============================================================================
# Purpose: synthesize the SFT training data for the 200-step warm-up scorer
#   (used later as the SID-PPL scorer in step 2_eval_ppl.sh). Uses unfiltered
#   raw CoT (video + ad) from step 2_cot.
#
# Input:  ${PREFIX}/data/RecIF-cot/steps/2_cot  (step 2 output, video + ad CoT)
# Output: ${PREFIX}/data/output/r2sft_warmup_raw
# Blended: ${PREFIX}/pretrain/examples/blended/r2sft_warmup_raw.json
#
# Data composition (video + ad only, RecIF product data not open-sourced):
#   - video + ad raw CoT (unfiltered)
#   - non-CoT rec (video + ad)
#   - caption2sid + sid2caption (video + ad)
#   - SPLIT_TARGET=1, TARGET_MAX_LEN=3
#
# Usage:
#   bash 0_build_warmup_data.sh
#   bash 0_build_warmup_data.sh /etc/mpi/hostfile
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"

COT_DIR="${COT_DIR:-${PREFIX}/data/RecIF-cot/steps/2_cot}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/data/output/r2sft_warmup_raw}"
BLENDED_JSON_NAME="${BLENDED_JSON_NAME:-r2sft_warmup_raw.json}"
BLENDED_JSON_PATH="${PREFIX}/pretrain/examples/blended/${BLENDED_JSON_NAME}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"

UPSTREAM_SH="${SCRIPT_DIR}/../../5_data_format/run_mpi.sh"
[ ! -f "${UPSTREAM_SH}" ] && echo "[ERROR] ${UPSTREAM_SH} not found" >&2 && exit 1
[ ! -d "${COT_DIR}" ] && echo "[ERROR] raw CoT dir not found: ${COT_DIR}" >&2 && exit 1
[ ! -f "${HOSTFILE}" ] && echo "[ERROR] hostfile not found: ${HOSTFILE}" >&2 && exit 1

FORCE_CLEAN="${FORCE_CLEAN:-1}"
if [ "${FORCE_CLEAN}" = "1" ]; then
  echo "[Cleanup] removing stale r2sft warmup raw output..."
  echo "  - rm -rf ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
  echo "  - rm -f ${BLENDED_JSON_PATH}"
  rm -f "${BLENDED_JSON_PATH}"
fi

echo "========================================"
echo "Step 0: build warm-up scorer SFT data (r2sft warmup raw)"
echo "========================================"
echo "COT_DIR              : ${COT_DIR} (step 2_cot, video+ad raw CoT)"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "BLENDED_JSON_NAME    : ${BLENDED_JSON_NAME}"
echo "COT_TARGET_SOURCE    : bench"
echo "ENABLE_NON_COT_REC   : 1 (video+ad)"
echo "ENABLE_SID2CAPTION   : 1 (video/ad)"
echo "ENABLE_CAPTION2SID   : 1 (video/ad)"
echo "PRODUCT              : disabled (RecIF product data not open-sourced)"
echo "HOSTFILE             : ${HOSTFILE}"
echo "========================================"
echo ""

export COT_DIR OUTPUT_DIR BLENDED_JSON_NAME
export COT_TARGET_SOURCE=bench
export ALLOW_COT_NON_COT_USER_OVERLAP=1
export ENABLE_COT_SPLIT_FILTER=1
export SPLIT_TARGET=1
export TARGET_MAX_LEN=3
export ENABLE_NON_COT_REC=1
export ENABLE_VIDEO_REC=1
export ENABLE_AD_REC=1
export ENABLE_PRODUCT_REC=0
export ENABLE_SID2CAPTION=1
export ENABLE_CAPTION2SID=1
export ENABLE_PRODUCT_SID2CAPTION=0
export ENABLE_PRODUCT_CAPTION2SID=0
export PRODUCT_PID2SID=""
export PRODUCT_CAPTION_DIR=""
export GLOBAL_SHUFFLE=1
export SAMPLE_RATIO=1.0
export SEED=42

bash "${UPSTREAM_SH}" "${HOSTFILE}"

echo ""
echo "========================================"
echo "warm-up data build done"
echo "  output      : ${OUTPUT_DIR}"
echo "  blended_json: ${BLENDED_JSON_PATH}"
echo "========================================"
