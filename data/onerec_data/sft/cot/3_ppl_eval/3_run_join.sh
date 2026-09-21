#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 3a join — 把 step3a PPL eval 输出与 step2 CoT 输出 join (MPI, CPU)
# ============================================================================
# 功能:
#   1. 读 step2 CoT 输出 (维度表) 与 step3a PPL eval 输出 (事实表)
#   2. 按 hash(sample_key) % world_size 分片, 每个 rank 本地 join
#   3. 恢复 step5 需要的 history_captions / target_pids / target_captions
#   4. 丢弃渲染中间列, 输出干净的 per-target parquet, 供 4_filter_ppl 消费
#
# 用法:
#   bash 3_run_join.sh                              # RUN_TYPE=pretrain 默认
#   RUN_TYPE=sft bash 3_run_join.sh                 # sft
#   bash 3_run_join.sh /etc/mpi/hostfile            # 指定 hostfile
#   PPL_INPUT=/xxx OUTPUT=/yyy bash 3_run_join.sh    # 自定义路径
#
# 环境变量:
#   RUN_TYPE         pretrain | sft (默认 pretrain; 仅用于选默认 PPL_INPUT/OUTPUT)
#   COT_INPUT        step2 CoT 输出目录
#   PPL_INPUT        step3a PPL eval 输出目录
#   OUTPUT           join 输出目录
#   ROWS_PER_SHARD   每输出 parquet 行数 (默认 2000)
#   SLOTS_PER_NODE   每节点 MPI slot 数 (默认 2)
#   HOSTFILE          MPI hostfile (默认 /etc/mpi/hostfile)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出 ==============
RUN_TYPE="${RUN_TYPE:-pretrain}"
DEFAULT_COT_INPUT="${PREFIX}/data/RecIF-cot/steps/2_cot"

if [ "$RUN_TYPE" = "pretrain" ]; then
  DEFAULT_PPL_INPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sid"
  DEFAULT_OUTPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_pretrain_sid"
elif [ "$RUN_TYPE" = "sft" ]; then
  DEFAULT_PPL_INPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_sid"
  DEFAULT_OUTPUT="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_sft_sid"
else
  echo "[ERROR] 不支持的 RUN_TYPE=${RUN_TYPE} (仅 pretrain | sft)" >&2
  exit 1
fi

COT_INPUT="${COT_INPUT:-${DEFAULT_COT_INPUT}}"
PPL_INPUT="${PPL_INPUT:-${DEFAULT_PPL_INPUT}}"
OUTPUT="${OUTPUT:-${DEFAULT_OUTPUT}}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-2000}"

MPI_ENTRY="${MPI_ENTRY:-${SCRIPT_DIR}/3_join_back.py}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
# 消费掉位置参数 $1（hostfile），避免 "$@" 把 hostfile 当作 3_join_back.py 的参数
if [ $# -gt 0 ]; then
  shift
fi
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/join_${RUN_TYPE}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"

# ============== 校验 ==============
if [ ! -f "$HOSTFILE" ]; then
  echo "[ERROR] hostfile 不存在: $HOSTFILE" >&2
  exit 1
fi
if [ ! -f "$MPI_ENTRY" ]; then
  echo "[ERROR] mpi 入口脚本不存在: $MPI_ENTRY" >&2
  exit 1
fi
for f in "$COT_INPUT" "$PPL_INPUT"; do
  if [ ! -e "$f" ]; then
    echo "[ERROR] 输入路径不存在: $f" >&2
    exit 1
  fi
done

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l | tr -d ' ')
NP=$((NP * SLOTS_PER_NODE))

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

OPTIONAL_VARS=(
  COT_INPUT PPL_INPUT OUTPUT ROWS_PER_SHARD
  PYTHONPATH PATH
)
ENV_ARGS=()
for var in "${OPTIONAL_VARS[@]}"; do
  if [ -n "${!var+x}" ]; then
    ENV_ARGS+=(-x "$var")
  fi
done

# ============== 打印配置 ==============
echo "========================================"
echo "Step 3a join — step2 x step3a-eval join (MPI, CPU)"
echo "========================================"
echo "run_type          : ${RUN_TYPE}"
echo "hostfile          : ${HOSTFILE}"
echo "np (total slots)  : ${NP}"
echo "slots_per_node    : ${SLOTS_PER_NODE}"
echo "MPI_ENTRY         : ${MPI_ENTRY}"
echo "COT_INPUT         : ${COT_INPUT}"
echo "PPL_INPUT         : ${PPL_INPUT}"
echo "OUTPUT            : ${OUTPUT}"
echo "ROWS_PER_SHARD    : ${ROWS_PER_SHARD}"
echo "LOG_FILE          : ${LOG_FILE}"
echo "========================================"
echo ""

mkdir -p "$OUTPUT"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mpirun -np "$NP" \
  --hostfile "$HOSTFILE" \
  --allow-run-as-root \
  --oversubscribe \
  --map-by slot \
  --bind-to none \
  "${ENV_ARGS[@]}" \
  "$PYTHON_BIN" "$MPI_ENTRY" \
    --cot_input "$COT_INPUT" \
    --ppl_input "$PPL_INPUT" \
    --output "$OUTPUT" \
    --rows_per_shard "$ROWS_PER_SHARD" \
    "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Done."
echo "  output_dir : ${OUTPUT}"
echo "  summary   : ${OUTPUT}/join_summary.json"
echo "  log       : ${LOG_FILE}"
echo "========================================"
