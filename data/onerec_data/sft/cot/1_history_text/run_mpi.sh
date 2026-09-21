#!/bin/bash
set -euo pipefail

# 用法：
#   bash run_mpi.sh [hostfile]
#   export SLOTS_PER_NODE=2
#   bash run_mpi.sh /etc/mpi/hostfile
#
# 功能：
#   使用 mpirun 启动 main.py，按用户粒度并行抽取 video/ad 历史 + caption 文本。
#   策略：方案 A —— 每个 rank 全量读取 pid2caption.parquet 构建本地 dict，
#   用户按 round-robin 分配给各 rank，输出按 rank 分片，每 1000 行一个 parquet。
#
# 必填环境变量（或通过 --bench_uri / --caption_uri / --output_uri 覆盖）：
#   BENCH_URI   onerec_bench_release.parquet 路径
#   CAPTION_URI pid2caption.parquet 路径
#   OUTPUT_URI  输出目录
#
# 可选环境变量：
#   SLOTS_PER_NODE  每节点进程数，默认 2
#   ROWS_PER_SHARD  每输出 parquet 行数，默认 1000
#   MIN_HISTORY_LEN history 保留 caption 后的最小长度，低于此值丢弃，默认 1
#   CAPTION_COLUMN  caption 文本列名：'dense_caption'(默认) 或 'summary'(用摘要版)
#   KEEP_NO_CAPTION_TARGET  0=丢弃无 caption 的 target pid（默认）；1=保留全部 target pid
#   KEEP_NO_CAPTION_HISTORY 0=丢弃无 caption 的 history pid（默认）；1=保留全部 history pid
#                           （与 bench 完全一致，仅长度截断，SID 路径能看到更完整 history）
#   TARGET_MAX_LEN_STEP1    target 截断上限：0=不截断（默认，0717 新）；>0=取前 N 个（原 10）
#   HADOOP_USER_NAME 默认 mpi（仅 viewfs:// 时需要）

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出路径 ==============
DEFAULT_BENCH_URI="${PREFIX}/data/RecIF/onerec_bench_release.parquet"
# DEFAULT_CAPTION_URI="${PREFIX}/data/RecIF/pid2caption.parquet"
DEFAULT_CAPTION_URI="${PREFIX}/data/RecIF-cot/caption_summary/"
DEFAULT_OUTPUT_URI="${PREFIX}/data/RecIF-cot/steps/1a_history_text"

BENCH_URI="${BENCH_URI:-${DEFAULT_BENCH_URI}}"
CAPTION_URI="${CAPTION_URI:-${DEFAULT_CAPTION_URI}}"
CAPTION_COLUMN="${CAPTION_COLUMN:-summary}"
OUTPUT_URI="${OUTPUT_URI:-${DEFAULT_OUTPUT_URI}}"
export BENCH_URI CAPTION_URI CAPTION_COLUMN OUTPUT_URI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
MPI_ENTRY="${MPI_ENTRY:-${SCRIPT_DIR}/main.py}"
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/mpi_history_text_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -f "$HOSTFILE" ]; then
    echo "[ERROR] hostfile 不存在: $HOSTFILE" >&2
    exit 1
fi

if [ ! -f "$MPI_ENTRY" ]; then
    echo "[ERROR] mpi 入口脚本不存在: $MPI_ENTRY" >&2
    exit 1
fi

# 校验输入路径存在（本地路径场景；viewfs:// 跳过）
# CAPTION_URI 可以是文件（pid2caption.parquet）或目录（caption_summary/）
for f in "$BENCH_URI" "$CAPTION_URI"; do
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

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l)
NP=$((NP * SLOTS_PER_NODE))

OPTIONAL_VARS=(
    INPUT_DIR
    LOG_LEVEL PYTHONPATH PATH
    JAVA_HOME HADOOP_HOME CLASSPATH LD_LIBRARY_PATH
    BENCH_URI CAPTION_URI CAPTION_COLUMN OUTPUT_URI
    ROWS_PER_SHARD MIN_HISTORY_LEN
    KEEP_NO_CAPTION_TARGET KEEP_NO_CAPTION_HISTORY TARGET_MAX_LEN_STEP1
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
echo "Step 1 (history_text) — RecIF CoT 消融"
echo "========================================"
echo "NP              : ${NP}"
echo "hostfile        : ${HOSTFILE}"
echo "MPI_ENTRY       : ${MPI_ENTRY}"
echo "BENCH_URI       : ${BENCH_URI}"
echo "CAPTION_URI     : ${CAPTION_URI}"
echo "CAPTION_COLUMN  : ${CAPTION_COLUMN}"
echo "OUTPUT_URI      : ${OUTPUT_URI}"
echo "ROWS_PER_SHARD  : ${ROWS_PER_SHARD:-1000}"
echo "MIN_HISTORY_LEN : ${MIN_HISTORY_LEN:-1}"
echo "KEEP_NO_CAPTION_TARGET  : ${KEEP_NO_CAPTION_TARGET:-0}"
echo "KEEP_NO_CAPTION_HISTORY : ${KEEP_NO_CAPTION_HISTORY:-0}"
echo "TARGET_MAX_LEN_STEP1   : ${TARGET_MAX_LEN_STEP1:-0} (0=不截断)"
echo "LOG_FILE        : ${LOG_FILE}"
echo ""

# 创建输出目录（本地路径场景）
case "$OUTPUT_URI" in
    viewfs://*|hdfs://*) ;;
    *) mkdir -p "$OUTPUT_URI" ;;
esac

mpirun -np "$NP" \
    --hostfile "$HOSTFILE" \
    --allow-run-as-root \
    --oversubscribe \
    --map-by slot \
    --bind-to none \
    "${ENV_ARGS[@]}" \
    python3 "$MPI_ENTRY" \
        --bench_uri "$BENCH_URI" \
        --caption_uri "$CAPTION_URI" \
        --caption_column "$CAPTION_COLUMN" \
        --output_uri "$OUTPUT_URI" \
        --rows_per_shard "${ROWS_PER_SHARD:-1000}" \
        --min_history_len "${MIN_HISTORY_LEN:-1}" \
        --keep_no_caption_target "${KEEP_NO_CAPTION_TARGET:-0}" \
        --keep_no_caption_history "${KEEP_NO_CAPTION_HISTORY:-0}" \
        --target_max_len "${TARGET_MAX_LEN_STEP1:-0}" 2>&1 | tee "$LOG_FILE"
