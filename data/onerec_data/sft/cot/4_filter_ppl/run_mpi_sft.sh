#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 4a (filter_ppl) — sft PPL 增益筛选 (MPI, CPU)
# ============================================================================
# 输入: 3a join 的 sft 输出 (per-target parquet)
# 输出: per-sample parquet, 可直接作为 step5 的 --cot_dir
#
# 用法:
#   bash run_mpi_sft.sh
#   bash run_mpi_sft.sh /etc/mpi/hostfile
#   MAX_DIFF=-0.10 MERGE_RULE=all bash run_mpi_sft.sh
#   LIMIT=100000 bash run_mpi_sft.sh            # top-k
#   MAX_DIFF="" MAX_LOG_PPL_X="" bash run_mpi_sft.sh  # annotate_all (全保留, 仅打分)
#
# 环境变量 (同 run_mpi_pretrain.sh, 默认输入/输出切到 sft 路径):
#   INPUT_DIR              3a join sft 输出目录
#   OUTPUT_DIR             4a sft 输出目录
#   MAX_DIFF / MAX_LOG_PPL_X / MAX_LOG_PPL_COT / MAX_LOG_PPL_COT_BY_DOMAIN /
#   MERGE_RULE / LIMIT / LIMIT_RATIO /
#   ROWS_PER_OUTPUT_FILE / SLOTS_PER_NODE / HOSTFILE
#
#   ★ MAX_DIFF / MAX_LOG_PPL_X / MAX_LOG_PPL_COT 可设为空串 "" 来显式禁用该门
#     (不传则用默认值)
#   ★ MAX_LOG_PPL_COT_BY_DOMAIN 支持 ad:3.3932,video:4.6238,product:3.8894，
#     命中域优先于 MAX_LOG_PPL_COT，未命中域回退到全局阈值。
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出 ==============
DEFAULT_INPUT_DIR="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/joined_sft_sid"
DEFAULT_OUTPUT_DIR="${PREFIX}/data/RecIF-cot/steps/4_filter_ppl/sft"
export INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

# export 以便 mpirun -x 可正确转发到远程节点（避免 "could not find environment variable" 警告）
# ★ 使用 ${VAR-default} (不带冒号) 使得空串 "" 能显式禁用某个门
export MAX_DIFF="${MAX_DIFF--0.05}"
export MAX_LOG_PPL_X="${MAX_LOG_PPL_X-9.0}"
# 0723 新增：log_ppl_cot 门，默认空=禁用
export MAX_LOG_PPL_COT="${MAX_LOG_PPL_COT-}"
export MAX_LOG_PPL_COT_BY_DOMAIN="${MAX_LOG_PPL_COT_BY_DOMAIN-}"
export MERGE_RULE="${MERGE_RULE:-any}"
export LIMIT="${LIMIT:-0}"
export LIMIT_RATIO="${LIMIT_RATIO:-0.0}"
export ROWS_PER_OUTPUT_FILE="${ROWS_PER_OUTPUT_FILE:-5000}"

MPI_ENTRY="${MPI_ENTRY:-${SCRIPT_DIR}/main.py}"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
# 消费掉位置参数 $1（hostfile），避免 "$@" 把 hostfile 当作 main.py 的参数
if [ $# -gt 0 ]; then
  shift
fi
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/filter_ppl_sft_$(date +%Y%m%d_%H%M%S).log}"

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
if [ ! -e "$INPUT_DIR" ]; then
  echo "[ERROR] 输入目录不存在: $INPUT_DIR" >&2
  exit 1
fi

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l | tr -d ' ')
NP=$((NP * SLOTS_PER_NODE))

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

OPTIONAL_VARS=(
  INPUT_DIR OUTPUT_DIR MAX_DIFF MAX_LOG_PPL_X MAX_LOG_PPL_COT MAX_LOG_PPL_COT_BY_DOMAIN MERGE_RULE LIMIT LIMIT_RATIO
  ROWS_PER_OUTPUT_FILE PYTHONPATH PATH
)
ENV_ARGS=()
for var in "${OPTIONAL_VARS[@]}"; do
  if [ -n "${!var+x}" ]; then
    ENV_ARGS+=(-x "$var")
  fi
done

# ============== 组装 python 参数 ==============
PY_ARGS=(
  --input "$INPUT_DIR"
  --output "$OUTPUT_DIR"
  --merge_rule "$MERGE_RULE"
  --limit "$LIMIT"
  --limit_ratio "$LIMIT_RATIO"
  --rows_per_output_file "$ROWS_PER_OUTPUT_FILE"
)
if [ -n "$MAX_DIFF" ]; then
  PY_ARGS+=(--max_diff "$MAX_DIFF")
fi
if [ -n "$MAX_LOG_PPL_X" ]; then
  PY_ARGS+=(--max_log_ppl_x "$MAX_LOG_PPL_X")
fi
if [ -n "$MAX_LOG_PPL_COT" ]; then
  PY_ARGS+=(--max_log_ppl_cot "$MAX_LOG_PPL_COT")
fi
if [ -n "$MAX_LOG_PPL_COT_BY_DOMAIN" ]; then
  PY_ARGS+=(--max_log_ppl_cot_by_domain "$MAX_LOG_PPL_COT_BY_DOMAIN")
fi

# ============== 打印配置 ==============
echo "========================================"
echo "Step 4a (filter_ppl) — sft PPL 增益筛选 (MPI, CPU)"
echo "========================================"
echo "hostfile             : ${HOSTFILE}"
echo "np (total slots)     : ${NP}"
echo "slots_per_node       : ${SLOTS_PER_NODE}"
echo "MPI_ENTRY            : ${MPI_ENTRY}"
echo "INPUT_DIR            : ${INPUT_DIR}"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "MAX_DIFF             : ${MAX_DIFF:-<none>}"
echo "MAX_LOG_PPL_X        : ${MAX_LOG_PPL_X:-<none>}"
echo "MAX_LOG_PPL_COT      : ${MAX_LOG_PPL_COT:-<none>}"
echo "MAX_LOG_PPL_COT_BY_DOMAIN : ${MAX_LOG_PPL_COT_BY_DOMAIN:-<none>}"
echo "MERGE_RULE           : ${MERGE_RULE}"
echo "LIMIT                : ${LIMIT}"
echo "LIMIT_RATIO          : ${LIMIT_RATIO}"
echo "ROWS_PER_OUTPUT_FILE : ${ROWS_PER_OUTPUT_FILE}"
echo "LOG_FILE             : ${LOG_FILE}"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_DIR"

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
    "${PY_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Done."
echo "  output_dir : ${OUTPUT_DIR}"
echo "  summary   : ${OUTPUT_DIR}/filter_summary.json"
echo "  log       : ${LOG_FILE}"
echo "========================================"
