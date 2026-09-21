#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 4 (filter) — per-target PPL filter
# ============================================================================
# Input:  joined_sft_ver5/  (from step 3_join.sh)
# Output: r2sft_filter/
#
# Two-phase usage:
#   1) annotate_all to get distribution (recommended first run):
#        MAX_LOG_PPL_COT="" MAX_LOG_PPL_X="" bash 4_filter.sh
#      then read log_ppl_cot_stats.p50 from ${OUTPUT_DIR}/filter_summary.json
#   2) threshold filter:
#        MAX_LOG_PPL_COT=<p50> bash 4_filter.sh
#
# Default hygiene gate: MAX_LOG_PPL_X=9.0; MAX_LOG_PPL_COT defaults to empty
# (avoid hardcoding the threshold before the new step200 distribution is known).
#
# Usage:
#   bash 4_filter.sh
#   bash 4_filter.sh /etc/mpi/hostfile
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"

INPUT_DIR="${INPUT_DIR:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_sft_ver5}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/data/RecIF-cot/steps/4_filter_ppl/r2sft_filter}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"

# Use ${VAR-...} (without colon) so empty string "" explicitly disables a gate.
MAX_DIFF="${MAX_DIFF-}"
MAX_LOG_PPL_X="${MAX_LOG_PPL_X-9.0}"
MAX_LOG_PPL_COT="${MAX_LOG_PPL_COT-}"

UPSTREAM_SH="${SCRIPT_DIR}/../../4_filter_ppl/run_mpi_sft.sh"
[ ! -f "${UPSTREAM_SH}" ] && echo "[ERROR] ${UPSTREAM_SH} not found" >&2 && exit 1
[ ! -d "${INPUT_DIR}" ] && echo "[ERROR] joined input not found: ${INPUT_DIR}" >&2 && exit 1

MODE="threshold"
if [ -z "${MAX_DIFF}" ] && [ -z "${MAX_LOG_PPL_X}" ] && [ -z "${MAX_LOG_PPL_COT}" ]; then
  MODE="annotate_all"
elif [ -z "${MAX_LOG_PPL_COT}" ]; then
  MODE="annotate_without_cot_threshold"
fi

echo "========================================"
echo "Step 4: per-target PPL filter"
echo "========================================"
echo "MODE            : ${MODE}"
echo "INPUT_DIR       : ${INPUT_DIR}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "MAX_DIFF        : ${MAX_DIFF:-<none>}"
echo "MAX_LOG_PPL_COT : ${MAX_LOG_PPL_COT:-<none>}"
echo "MAX_LOG_PPL_X   : ${MAX_LOG_PPL_X:-<none>}"
echo "HOSTFILE        : ${HOSTFILE}"
echo "========================================"
echo ""

INPUT_DIR="${INPUT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MAX_DIFF="${MAX_DIFF}" \
MAX_LOG_PPL_COT="${MAX_LOG_PPL_COT}" \
MAX_LOG_PPL_X="${MAX_LOG_PPL_X}" \
bash "${UPSTREAM_SH}" "${HOSTFILE}"

echo ""
echo "4_filter done: ${OUTPUT_DIR}"
echo "summary: ${OUTPUT_DIR}/filter_summary.json"
if [ -z "${MAX_LOG_PPL_COT}" ]; then
  echo ""
  echo "next: read log_ppl_cot_stats.p50 from filter_summary.json, then:"
  echo "  MAX_LOG_PPL_COT=<p50> bash ${SCRIPT_DIR}/4_filter.sh ${HOSTFILE}"
  echo "then:"
  echo "  bash ${SCRIPT_DIR}/5_data_format.sh ${HOSTFILE}"
fi
