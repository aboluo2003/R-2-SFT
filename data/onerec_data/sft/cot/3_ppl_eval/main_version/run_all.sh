#!/bin/bash
set -euo pipefail

# ============================================================================
# Orchestrator — join → filter → data_format (steps 3 → 4 → 5)
# ============================================================================
# Prerequisites:
#   1. bash 1_preprocess.sh
#   2. train the warm-up scorer:  cd pretrain && python3 auto_train/auto_train.py auto_train/r2sft_warmup_200steps.yaml
#   3. bash 2_eval_ppl.sh
#
# This script runs:
#   step 3: 3_join.sh              → joined_sft_ver5
#   step 4: 4_filter.sh            → r2sft_filter
#   step 5: 5_data_format.sh       → r2sft_sft_ppl (★ main output)
#
# Note: for a new step200 scorer, the log_ppl_cot p50 is unknown. Run
#   annotate_all first (without MAX_LOG_PPL_COT), read p50 from
#   filter_summary.json, then re-run with MAX_LOG_PPL_COT=<p50>:
#   MAX_LOG_PPL_COT="" MAX_DIFF="" MAX_LOG_PPL_X="" bash 4_filter.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

JOIN_OUTPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_sft_ver5"
FILTER_OUTPUT="${PREFIX}/data/RecIF-cot/steps/4_filter_ppl/r2sft_filter"
DATA_FORMAT_OUTPUT="${PREFIX}/data/output/r2sft_sft_ppl"
PPL_INPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_cot_ver5"
BLENDED_JSON_NAME="r2sft_sft_ppl.json"

[ ! -d "${PPL_INPUT}" ] && echo "[ERROR] step 2 PPL output not found: ${PPL_INPUT}" >&2 && exit 1
[ ! -f "${HOSTFILE}" ] && echo "[ERROR] hostfile not found: ${HOSTFILE}" >&2 && exit 1

has_parquet() {
  local dir="$1"
  [ -d "${dir}" ] && ls "${dir}"/part-*.parquet >/dev/null 2>&1
}

echo "========================================"
echo "Orchestrator: join → filter → data_format"
echo "========================================"
echo "PPL_INPUT           : ${PPL_INPUT}"
echo "JOIN_OUTPUT         : ${JOIN_OUTPUT}"
echo "FILTER_OUTPUT       : ${FILTER_OUTPUT}"
echo "DATA_FORMAT_OUTPUT  : ${DATA_FORMAT_OUTPUT}"
echo "BLENDED_JSON        : ${BLENDED_JSON_NAME}"
echo "MAX_LOG_PPL_COT     : ${MAX_LOG_PPL_COT:-<none>}"
echo "HOSTFILE            : ${HOSTFILE}"
echo "SKIP_EXISTING       : ${SKIP_EXISTING}"
echo "========================================"
echo ""

echo "[Step 3/5] join..."
if [ "${SKIP_EXISTING}" = "1" ] && has_parquet "${JOIN_OUTPUT}"; then
  echo "  [Skip] join output exists: ${JOIN_OUTPUT}"
else
  bash "${SCRIPT_DIR}/3_join.sh" "${HOSTFILE}"
fi
echo ""

echo "[Step 4/5] per-target filter..."
if [ "${SKIP_EXISTING}" = "1" ] && has_parquet "${FILTER_OUTPUT}"; then
  echo "  [Skip] filter output exists: ${FILTER_OUTPUT}"
else
  bash "${SCRIPT_DIR}/4_filter.sh" "${HOSTFILE}"
fi
echo ""

echo "[Step 5/5] data_format..."
if [ -z "${MAX_LOG_PPL_COT:-}" ]; then
  echo "  [Skip] MAX_LOG_PPL_COT not set: only join/filter done (for distribution / hygiene gate)."
  echo "  Read log_ppl_cot_stats.p50 from ${FILTER_OUTPUT}/filter_summary.json, then run:"
  echo "    MAX_LOG_PPL_COT=<p50> SKIP_EXISTING=1 bash ${SCRIPT_DIR}/run_all.sh ${HOSTFILE}"
  echo "  or directly:"
  echo "    MAX_LOG_PPL_COT=<p50> bash ${SCRIPT_DIR}/4_filter.sh ${HOSTFILE}"
  echo "    bash ${SCRIPT_DIR}/5_data_format.sh ${HOSTFILE}"
else
  if [ "${SKIP_EXISTING}" = "1" ] && has_parquet "${DATA_FORMAT_OUTPUT}"; then
    echo "  [Skip] data_format output exists: ${DATA_FORMAT_OUTPUT}"
  else
    bash "${SCRIPT_DIR}/5_data_format.sh" "${HOSTFILE}"
  fi
fi

echo ""
echo "========================================"
echo "Orchestrator done"
echo "  join        : ${JOIN_OUTPUT}"
echo "  filter      : ${FILTER_OUTPUT}"
echo "  data_format : ${DATA_FORMAT_OUTPUT}"
echo "  blended_json: ${PREFIX}/pretrain/examples/blended/${BLENDED_JSON_NAME}"
echo "========================================"
