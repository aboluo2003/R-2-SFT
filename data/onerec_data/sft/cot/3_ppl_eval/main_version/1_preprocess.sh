#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 1 (preprocess) — prepare CoT samples for PPL eval (bench mode, SID)
# ============================================================================
# Input:  step 2_cot output (video + ad raw CoT)
# Output: preprocessed_sid_cot_ver5/
#
# Explodes each CoT sample into per-target PPL eval inputs. Input/target pids
# are rebuilt from bench (target_source=bench) to align with training format.
#
# Usage:
#   bash 1_preprocess.sh
#   bash 1_preprocess.sh /etc/mpi/hostfile
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../../.." && pwd)}"

COT_INPUT="${COT_INPUT:-${PREFIX}/data/RecIF-cot/steps/2_cot}"
BENCH_URI="${BENCH_URI:-${PREFIX}/data/RecIF/onerec_bench_release.parquet}"
PID2SID_URI="${PID2SID_URI:-${PREFIX}/data/RecIF/video_ad_pid2sid.parquet}"
PRODUCT_PID2SID_URI="${PRODUCT_PID2SID_URI:-}"
OUTPUT_URI="${OUTPUT_URI:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/preprocessed_sid_cot_ver5}"

HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-1000}"

MPI_ENTRY="${SCRIPT_DIR}/../1_preprocess.py"
[ ! -f "${MPI_ENTRY}" ] && echo "[ERROR] ${MPI_ENTRY} not found" >&2 && exit 1
[ ! -d "${COT_INPUT}" ] && echo "[ERROR] raw CoT input not found: ${COT_INPUT}" >&2 && exit 1
[ ! -f "${BENCH_URI}" ] && echo "[ERROR] bench not found: ${BENCH_URI}" >&2 && exit 1
[ ! -f "${PID2SID_URI}" ] && echo "[ERROR] video/ad pid2sid not found: ${PID2SID_URI}" >&2 && exit 1
[ ! -f "${HOSTFILE}" ] && echo "[ERROR] hostfile not found: ${HOSTFILE}" >&2 && exit 1

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l | tr -d ' ')
NP=$((NP * SLOTS_PER_NODE))

export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
JAVA_HOME="${JAVA_HOME:-/home/hadoop/software/java}"
HADOOP_HOME="${HADOOP_HOME:-/home/hadoop/software/hadoop}"
if [ -d "${JAVA_HOME}" ]; then
  export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib/amd64/server:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${HADOOP_HOME}" ]; then
  export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH:-}"
  export CLASSPATH="${CLASSPATH:+$CLASSPATH:}$(${HADOOP_HOME}/bin/hadoop classpath --glob 2>/dev/null || true)"
fi
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-mpi}"

mkdir -p "${OUTPUT_URI}" "${SCRIPT_DIR}/logs"

echo "========================================"
echo "Step 1: preprocess"
echo "========================================"
echo "COT_INPUT           : ${COT_INPUT}"
echo "BENCH_URI           : ${BENCH_URI}"
echo "PID2SID_URI         : ${PID2SID_URI}"
echo "PRODUCT_PID2SID_URI : ${PRODUCT_PID2SID_URI:-<none>}"
echo "OUTPUT_URI          : ${OUTPUT_URI}"
echo "HOSTFILE            : ${HOSTFILE}"
echo "SLOTS_PER_NODE      : ${SLOTS_PER_NODE}"
echo "ROWS_PER_SHARD      : ${ROWS_PER_SHARD}"
echo "========================================"
echo ""

PREPROCESS_ARGS=(
  --input_uri "$COT_INPUT"
  --output_uri "$OUTPUT_URI"
  --representation_type sid
  --pid2sid_uri "$PID2SID_URI"
  --target_source bench
  --bench_uri "$BENCH_URI"
  --rows_per_shard "$ROWS_PER_SHARD"
)
if [ -n "${PRODUCT_PID2SID_URI}" ]; then
  PREPROCESS_ARGS+=(--product_pid2sid_uri "$PRODUCT_PID2SID_URI")
fi

mpirun -np "$NP" \
  --hostfile "$HOSTFILE" \
  --allow-run-as-root \
  --oversubscribe \
  --map-by slot \
  --bind-to none \
  -x PYTHONPATH -x PATH -x LD_LIBRARY_PATH -x CLASSPATH \
  -x JAVA_HOME -x HADOOP_HOME -x HADOOP_USER_NAME \
  -x PYTHONUNBUFFERED -x PYTHONIOENCODING \
  python3 "$MPI_ENTRY" \
    "${PREPROCESS_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/logs/preprocess_cot_ver5_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "preprocess done: ${OUTPUT_URI}"
