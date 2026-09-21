#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 3 (join) — join step2 CoT with step2-eval PPL output (per-target)
# ============================================================================
# Input:
#   COT_INPUT = step 2_cot output (video + ad CoT)
#   PPL_INPUT = ppl_eval_sft_cot_ver5 (from step 2_eval_ppl.sh)
# Output: joined_sft_ver5/
#
# Usage:
#   bash 3_join.sh
#   bash 3_join.sh /etc/mpi/hostfile
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"

COT_INPUT="${COT_INPUT:-${PREFIX}/data/RecIF-cot/steps/2_cot}"
PPL_INPUT="${PPL_INPUT:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_cot_ver5}"
OUTPUT="${OUTPUT:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_sft_ver5}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"

UPSTREAM_SH="${SCRIPT_DIR}/../3_run_join.sh"
[ ! -f "${UPSTREAM_SH}" ] && echo "[ERROR] ${UPSTREAM_SH} not found" >&2 && exit 1
[ ! -d "${COT_INPUT}" ] && echo "[ERROR] CoT input not found: ${COT_INPUT}" >&2 && exit 1
[ ! -d "${PPL_INPUT}" ] && echo "[ERROR] PPL input not found: ${PPL_INPUT}" >&2 && exit 1

echo "========================================"
echo "Step 3: join_back"
echo "========================================"
echo "COT_INPUT : ${COT_INPUT}"
echo "PPL_INPUT : ${PPL_INPUT}"
echo "OUTPUT    : ${OUTPUT}"
echo "HOSTFILE  : ${HOSTFILE}"
echo "========================================"
echo ""

COT_INPUT="${COT_INPUT}" \
PPL_INPUT="${PPL_INPUT}" \
OUTPUT="${OUTPUT}" \
RUN_TYPE=sft \
bash "${UPSTREAM_SH}" "${HOSTFILE}"

echo ""
echo "join done: ${OUTPUT}"
