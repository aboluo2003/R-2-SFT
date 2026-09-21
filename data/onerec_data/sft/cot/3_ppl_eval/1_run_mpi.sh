#!/bin/bash
set -euo pipefail

# 用法：
#   bash 1_run_mpi.sh [hostfile]
#   REPRESENTATION_TYPE=caption bash 1_run_mpi.sh
#   REPRESENTATION_TYPE=sid bash 1_run_mpi.sh
#
# 功能：
#   使用 mpirun 启动 1_preprocess.py，把 step2/step4 的 CoT 样本整理成统一的 PPL 评估输入。
#   - caption 模式：输出 target_caption | input 的统一字段
#   - sid 模式：输出 target_sid | input_sid 的统一字段
#
# 说明：
#   参考 1_history_text/run_mpi.sh，保留相同的环境透传方式与 Java/Hadoop 处理逻辑。

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出路径 ==============
DEFAULT_INPUT_URI="${PREFIX}/data/RecIF-cot/steps/2_cot"
DEFAULT_PID2SID_URI="${PREFIX}/data/RecIF/video_ad_pid2sid.parquet"

INPUT_URI="${INPUT_URI:-${DEFAULT_INPUT_URI}}"
REPRESENTATION_TYPE="${REPRESENTATION_TYPE:-caption}"
PID2SID_URI="${PID2SID_URI:-${DEFAULT_PID2SID_URI}}"
DEFAULT_OUTPUT_URI="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/preprocessed_${REPRESENTATION_TYPE}"
OUTPUT_URI="${OUTPUT_URI:-${DEFAULT_OUTPUT_URI}}"
export INPUT_URI OUTPUT_URI REPRESENTATION_TYPE PID2SID_URI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
MPI_ENTRY="${MPI_ENTRY:-${SCRIPT_DIR}/1_preprocess.py}"
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-1000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/mpi_ppl_preprocess_${REPRESENTATION_TYPE}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] hostfile 不存在: $HOSTFILE" >&2
    exit 1
fi

if [ ! -f "$MPI_ENTRY" ]; then
    echo "[ERROR] mpi 入口脚本不存在: $MPI_ENTRY" >&2
    exit 1
fi

for f in "$INPUT_URI"; do
    case "$f" in
        viewfs://*|hdfs://*) ;;
        *)
            if [ ! -e "$f" ]; then
                echo "[ERROR] 输入路径不存在: $f" >&2
                exit 1
            fi
            ;;
    esac
done

if [ "$REPRESENTATION_TYPE" = "sid" ]; then
    case "$PID2SID_URI" in
        viewfs://*|hdfs://*) ;;
        *)
            if [ ! -e "$PID2SID_URI" ]; then
                echo "[ERROR] sid 模式要求的 pid2sid 不存在: $PID2SID_URI" >&2
                exit 1
            fi
            ;;
    esac
fi

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l)
NP=$((NP * SLOTS_PER_NODE))

OPTIONAL_VARS=(
    INPUT_DIR
    LOG_LEVEL PYTHONPATH PATH
    JAVA_HOME HADOOP_HOME CLASSPATH LD_LIBRARY_PATH
    INPUT_URI OUTPUT_URI REPRESENTATION_TYPE PID2SID_URI
    ROWS_PER_SHARD
)
ENV_ARGS=()
for var in "${OPTIONAL_VARS[@]}"; do
    if [ -n "${!var+x}" ]; then
        ENV_ARGS+=(-x "$var")
    fi
done

if [ -n "${JAVA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib/amd64/server:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${HADOOP_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH:-}"
    export CLASSPATH="${CLASSPATH:+$CLASSPATH:}$(${HADOOP_HOME}/bin/hadoop classpath --glob)"
fi
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-mpi}"

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${SCRIPT_DIR}"
fi

for var in PYTHONPATH LD_LIBRARY_PATH CLASSPATH HADOOP_USER_NAME; do
    found=0
    for existing in "${ENV_ARGS[@]}"; do
        if [ "$existing" = "$var" ]; then
            found=1
            break
        fi
    done
    if [ $found -eq 0 ]; then
        ENV_ARGS+=(-x "$var")
    fi
done

echo "========================================"
echo "Step 3a (ppl_eval preprocess) — 统一预处理层"
echo "========================================"
echo "NP                  : ${NP}"
echo "hostfile            : ${HOSTFILE}"
echo "MPI_ENTRY           : ${MPI_ENTRY}"
echo "INPUT_URI           : ${INPUT_URI}"
echo "OUTPUT_URI          : ${OUTPUT_URI}"
echo "REPRESENTATION_TYPE : ${REPRESENTATION_TYPE}"
echo "PID2SID_URI         : ${PID2SID_URI}"
echo "ROWS_PER_SHARD      : ${ROWS_PER_SHARD}"
echo "LOG_FILE            : ${LOG_FILE}"
echo ""

case "$OUTPUT_URI" in
    viewfs://*|hdfs://*) ;;
    *) mkdir -p "$OUTPUT_URI" ;;
esac

PY_ARGS=(
    --input_uri "$INPUT_URI"
    --output_uri "$OUTPUT_URI"
    --representation_type "$REPRESENTATION_TYPE"
    --rows_per_shard "$ROWS_PER_SHARD"
)
if [ "$REPRESENTATION_TYPE" = "sid" ]; then
    PY_ARGS+=(--pid2sid_uri "$PID2SID_URI")
fi

mpirun -np "$NP" \
    --hostfile "$HOSTFILE" \
    --allow-run-as-root \
    --oversubscribe \
    --map-by slot \
    --bind-to none \
    "${ENV_ARGS[@]}" \
    python3 "$MPI_ENTRY" \
    "${PY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
