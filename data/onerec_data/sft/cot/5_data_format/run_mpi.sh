#!/bin/bash
set -euo pipefail

# ============================================================================
# Step 5 (data_format) — 合成 RecIF CoT SFT 数据集（MPI 并行）
# ============================================================================
# 功能：
#   1. 读取 step2 (2_cot) 或 step4 (4_filter) 产出的 CoT 样本
#   2. 从 onerec_bench_release.parquet 构造非 CoT 推荐样本（video/ad/product）
#   3. 从 caption 分片构造 sid2caption / caption2sid 样本（disjoint pid）
#   4. 全局 shuffle + 分片 + 生成 file_list.json
#
# 用法：
#   bash run_mpi.sh
#   bash run_mpi.sh /etc/mpi/hostfile
#
# 必填环境变量：
#   （均有默认值，可按需 override）
#
# 可选环境变量：
#   PREFIX                  OpenOneRec 根目录（服务器路径）
#   COT_DIR                 CoT 输入目录（step2 或 step4 输出）
#   BENCH_PARQUET           onerec_bench_release.parquet 路径
#   CAPTION_DIR             caption 分片目录（pid + dense_caption）
#   PID2SID                 video_ad_pid2sid.parquet 路径
#   PRODUCT_PID2SID         product_pid2sid.parquet 路径
#   OUTPUT_DIR              输出目录（分片 parquet）
#   BLENDED_JSON_NAME       file_list.json 文件名（保存到 ${PREFIX}/pretrain/examples/blended/）
#   TEMP_DIR                临时文件目录（默认 ${OUTPUT_DIR}/_temp）
#   KEEP_TEMP               1=保留临时文件
#
#   ENABLE_NON_COT_REC       1/0，是否构造非 CoT 推荐样本
#   ENABLE_VIDEO_REC         1/0
#   ENABLE_AD_REC            1/0
#   ENABLE_PRODUCT_REC       1/0
#   ENABLE_SID2CAPTION       1/0
#   ENABLE_CAPTION2SID       1/0
#   ALLOW_COT_NON_COT_USER_OVERLAP  1/0，CoT 与非 CoT 是否允许用户重叠
#   ENABLE_COT_SPLIT_FILTER          1/0，CoT 是否按 uid 的 split 过滤（默认 1）
#
#   ★ 0723 新增 product 域 caption 任务（独立于 video/ad caption）:
#   PRODUCT_CAPTION_DIR           product caption 目录（goods_pid2caption），启用 product caption 时必填
#   PRODUCT_CAPTION_COLUMN        product caption 列名（默认 caption）
#   PRODUCT_PID_COLUMN            product pid 列名（默认 new_pid，★ 不要用 old_pid）
#   ENABLE_PRODUCT_SID2CAPTION    1/0，是否构造 product sid2caption（默认 0）
#   ENABLE_PRODUCT_CAPTION2SID    1/0，是否构造 product caption2sid（默认 0）
#   PRODUCT_SID2CAPTION_SIZE      product sid2caption 目标样本数（默认 260000）
#   PRODUCT_CAPTION2SID_SIZE      product caption2sid 目标样本数（默认 260000）
#
#   GLOBAL_SHUFFLE           1/0，全局 shuffle
#   SAMPLE_RATIO             (0, 1]，全局采样比例
#   ROWS_PER_SHARD           每分片行数，默认 1000
#   TARGET_MAX_LEN           每条推荐样本保留的 target 数量上限，默认 3
#   SPLIT_TARGET             0=不拆分 target（默认）；非0=按 target 拆成单 target 样本，最多 TARGET_MAX_LEN 条
#   SID2CAPTION_SIZE         sid2caption 目标样本数，默认 260000
#   CAPTION2SID_SIZE         caption2sid 目标样本数，默认 260000
#   SHUFFLE_CAPTION_FILES    1/0，是否打乱 caption 分片顺序
#
#   SIDD0C_MODE              0/1/2 siddoc 模式（0=原格式，1=纯siddoc替换，2=mix同时造rec+siddoc），默认 0
#
#   SEED                     随机种子
#   SLOTS_PER_NODE           每节点 MPI slot 数
#   HOSTFILE                 MPI hostfile 路径
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出路径 ==============
DEFAULT_COT_DIR="${PREFIX}/data/RecIF-cot/steps/2_cot"
DEFAULT_BENCH_PARQUET="${PREFIX}/data/RecIF/onerec_bench_release.parquet"
DEFAULT_CAPTION_DIR="${PREFIX}/data/RecIF-cot/caption_shards"
DEFAULT_PID2SID="${PREFIX}/data/RecIF/video_ad_pid2sid.parquet"
DEFAULT_PRODUCT_PID2SID="${PREFIX}/data/RecIF/product_pid2sid.parquet"
DEFAULT_OUTPUT_DIR="${PREFIX}/data/output/sft_recif_cot"
DEFAULT_BLENDED_JSON_NAME="recif_cot_sft.json"

COT_DIR="${COT_DIR:-${DEFAULT_COT_DIR}}"
BENCH_PARQUET="${BENCH_PARQUET:-${DEFAULT_BENCH_PARQUET}}"
CAPTION_DIR="${CAPTION_DIR:-${DEFAULT_CAPTION_DIR}}"
PID2SID="${PID2SID:-${DEFAULT_PID2SID}}"
PRODUCT_PID2SID="${PRODUCT_PID2SID-${DEFAULT_PRODUCT_PID2SID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
BLENDED_JSON_NAME="${BLENDED_JSON_NAME:-${DEFAULT_BLENDED_JSON_NAME}}"
BLENDED_JSON_PATH="${BLENDED_JSON_PATH:-${PREFIX}/pretrain/examples/blended/${BLENDED_JSON_NAME}}"
TEMP_DIR="${TEMP_DIR:-${OUTPUT_DIR}/_temp}"

# ============== 任务开关 ==============
ENABLE_NON_COT_REC="${ENABLE_NON_COT_REC:-1}"
ENABLE_VIDEO_REC="${ENABLE_VIDEO_REC:-1}"
ENABLE_AD_REC="${ENABLE_AD_REC:-1}"
ENABLE_PRODUCT_REC="${ENABLE_PRODUCT_REC:-1}"
ENABLE_SID2CAPTION="${ENABLE_SID2CAPTION:-1}"
ENABLE_CAPTION2SID="${ENABLE_CAPTION2SID:-1}"
ALLOW_COT_NON_COT_USER_OVERLAP="${ALLOW_COT_NON_COT_USER_OVERLAP:-0}"
ENABLE_COT_SPLIT_FILTER="${ENABLE_COT_SPLIT_FILTER:-1}"

# ============== product 域 caption 任务开关（0723 新增） ==============
PRODUCT_CAPTION_DIR="${PRODUCT_CAPTION_DIR:-}"
PRODUCT_CAPTION_COLUMN="${PRODUCT_CAPTION_COLUMN:-caption}"
PRODUCT_PID_COLUMN="${PRODUCT_PID_COLUMN:-new_pid}"
ENABLE_PRODUCT_SID2CAPTION="${ENABLE_PRODUCT_SID2CAPTION:-0}"
ENABLE_PRODUCT_CAPTION2SID="${ENABLE_PRODUCT_CAPTION2SID:-0}"
PRODUCT_SID2CAPTION_SIZE="${PRODUCT_SID2CAPTION_SIZE:-260000}"
PRODUCT_CAPTION2SID_SIZE="${PRODUCT_CAPTION2SID_SIZE:-260000}"

# ============== CoT target 来源（0717 新增） ==============
# COT_TARGET_SOURCE: bench（默认，用 bench 原始 target）/ cot（用 4a 输出的筛选后 target）
COT_TARGET_SOURCE="${COT_TARGET_SOURCE:-bench}"

# ============== shuffle / 采样 ==============
GLOBAL_SHUFFLE="${GLOBAL_SHUFFLE:-1}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-1000}"
TARGET_MAX_LEN="${TARGET_MAX_LEN:-10}"
SPLIT_TARGET="${SPLIT_TARGET:-0}"
SID2CAPTION_SIZE="${SID2CAPTION_SIZE:-260000}"
CAPTION2SID_SIZE="${CAPTION2SID_SIZE:-260000}"
SHUFFLE_CAPTION_FILES="${SHUFFLE_CAPTION_FILES:-1}"
SEED="${SEED:-42}"
KEEP_TEMP="${KEEP_TEMP:-0}"
SIDD0C_MODE="${SIDD0C_MODE:-0}"

# ============== MPI / Java / Hadoop 环境 ==============
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
if [ $# -gt 0 ]; then
  shift
fi
MPI_ENTRY="${MPI_ENTRY:-${SCRIPT_DIR}/main.py}"
SLOTS_PER_NODE="${SLOTS_PER_NODE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JAVA_HOME="${JAVA_HOME:-/home/hadoop/software/java}"
HADOOP_HOME="${HADOOP_HOME:-/home/hadoop/software/hadoop}"
HADOOP_USER_NAME="${HADOOP_USER_NAME:-mpi}"
TCP_NIC="${TCP_NIC:-$(ifconfig 2>/dev/null | grep -B1 " $(hostname -i) " | grep -o "^\w*" | head -n1 || echo eth0)}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/data_format_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$LOG_DIR"

# ============== 校验 ==============
if [ ! -f "$HOSTFILE" ]; then
  echo "[ERROR] hostfile 不存在: $HOSTFILE" >&2
  exit 1
fi
if [ ! -f "$MPI_ENTRY" ]; then
  echo "[ERROR] mpi 入口脚本不存在: $MPI_ENTRY" >&2
  exit 1
fi
if [ ! -f "$BENCH_PARQUET" ] && ! [[ "$BENCH_PARQUET" =~ ^viewfs:// ]] && ! [[ "$BENCH_PARQUET" =~ ^hdfs:// ]]; then
  echo "[ERROR] bench parquet 不存在: $BENCH_PARQUET" >&2
  exit 1
fi

NP=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | wc -l | tr -d ' ')
NP=$((NP * SLOTS_PER_NODE))

# ============== Hadoop / Java 环境（viewfs 支持） ==============
if [ -n "${JAVA_HOME:-}" ] && [ -d "${JAVA_HOME}" ]; then
  export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib/amd64/server:${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${HADOOP_HOME:-}" ] && [ -d "${HADOOP_HOME}" ]; then
  export LD_LIBRARY_PATH="${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH:-}"
  export CLASSPATH="${CLASSPATH:+$CLASSPATH:}$(${HADOOP_HOME}/bin/hadoop classpath --glob 2>/dev/null || true)"
fi

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}"
fi

# ============== 需要传递给 MPI 进程的环境变量 ==============
OPTIONAL_VARS=(
  COT_DIR BENCH_PARQUET CAPTION_DIR PID2SID PRODUCT_PID2SID
  OUTPUT_DIR BLENDED_JSON_PATH TEMP_DIR KEEP_TEMP
  ENABLE_NON_COT_REC ENABLE_VIDEO_REC ENABLE_AD_REC ENABLE_PRODUCT_REC
  ENABLE_SID2CAPTION ENABLE_CAPTION2SID ALLOW_COT_NON_COT_USER_OVERLAP ENABLE_COT_SPLIT_FILTER
  COT_TARGET_SOURCE
  PRODUCT_CAPTION_DIR PRODUCT_CAPTION_COLUMN PRODUCT_PID_COLUMN
  ENABLE_PRODUCT_SID2CAPTION ENABLE_PRODUCT_CAPTION2SID
  PRODUCT_SID2CAPTION_SIZE PRODUCT_CAPTION2SID_SIZE
  GLOBAL_SHUFFLE SAMPLE_RATIO ROWS_PER_SHARD TARGET_MAX_LEN SPLIT_TARGET
  SID2CAPTION_SIZE CAPTION2SID_SIZE SHUFFLE_CAPTION_FILES SEED SIDD0C_MODE
  PYTHONPATH PATH JAVA_HOME HADOOP_HOME CLASSPATH LD_LIBRARY_PATH HADOOP_USER_NAME
)
ENV_ARGS=()
for var in "${OPTIONAL_VARS[@]}"; do
  if [ -n "${!var+x}" ]; then
    ENV_ARGS+=(-x "$var")
  fi
done

# ============== 组装 python 参数 ==============
PY_ARGS=(
  --cot_dir "$COT_DIR"
  --bench_parquet "$BENCH_PARQUET"
  --pid2sid "$PID2SID"
  --output_dir "$OUTPUT_DIR"
  --blended_json_path "$BLENDED_JSON_PATH"
  --temp_dir "$TEMP_DIR"
  --enable_non_cot_rec "$ENABLE_NON_COT_REC"
  --enable_video_rec "$ENABLE_VIDEO_REC"
  --enable_ad_rec "$ENABLE_AD_REC"
  --enable_product_rec "$ENABLE_PRODUCT_REC"
  --enable_sid2caption "$ENABLE_SID2CAPTION"
  --enable_caption2sid "$ENABLE_CAPTION2SID"
  --allow_cot_non_cot_user_overlap "$ALLOW_COT_NON_COT_USER_OVERLAP"
  --enable_cot_split_filter "$ENABLE_COT_SPLIT_FILTER"
  --cot_target_source "$COT_TARGET_SOURCE"
  --enable_product_sid2caption "$ENABLE_PRODUCT_SID2CAPTION"
  --enable_product_caption2sid "$ENABLE_PRODUCT_CAPTION2SID"
  --product_sid2caption_size "$PRODUCT_SID2CAPTION_SIZE"
  --product_caption2sid_size "$PRODUCT_CAPTION2SID_SIZE"
  --global_shuffle "$GLOBAL_SHUFFLE"
  --sample_ratio "$SAMPLE_RATIO"
  --rows_per_shard "$ROWS_PER_SHARD"
  --target_max_len "$TARGET_MAX_LEN"
  --split_target "$SPLIT_TARGET"
  --sid2caption_size "$SID2CAPTION_SIZE"
  --caption2sid_size "$CAPTION2SID_SIZE"
  --shuffle_caption_files "$SHUFFLE_CAPTION_FILES"
  --seed "$SEED"
  --siddoc_mode "$SIDD0C_MODE"
)

if [ -n "$CAPTION_DIR" ]; then
  PY_ARGS+=(--caption_dir "$CAPTION_DIR")
fi
if [ -n "$PRODUCT_PID2SID" ]; then
  PY_ARGS+=(--product_pid2sid "$PRODUCT_PID2SID")
fi
if [ -n "$PRODUCT_CAPTION_DIR" ]; then
  PY_ARGS+=(--product_caption_dir "$PRODUCT_CAPTION_DIR")
  PY_ARGS+=(--product_caption_column "$PRODUCT_CAPTION_COLUMN")
  PY_ARGS+=(--product_pid_column "$PRODUCT_PID_COLUMN")
fi
if [ "$KEEP_TEMP" = "1" ]; then
  PY_ARGS+=(--keep_temp)
fi

# ============== 打印配置 ==============
echo "========================================"
echo "Step 5 (data_format) — RecIF CoT SFT 数据集合成（MPI）"
echo "========================================"
echo "hostfile             : ${HOSTFILE}"
echo "np (total slots)     : ${NP}"
echo "slots_per_node       : ${SLOTS_PER_NODE}"
echo "MPI_ENTRY            : ${MPI_ENTRY}"
echo "COT_DIR              : ${COT_DIR}"
echo "BENCH_PARQUET        : ${BENCH_PARQUET}"
echo "CAPTION_DIR          : ${CAPTION_DIR}"
echo "PID2SID               : ${PID2SID}"
echo "PRODUCT_PID2SID      : ${PRODUCT_PID2SID}"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "BLENDED_JSON_PATH    : ${BLENDED_JSON_PATH}"
echo "TEMP_DIR             : ${TEMP_DIR}"
echo "KEEP_TEMP            : ${KEEP_TEMP}"
echo ""
echo "ENABLE_NON_COT_REC   : ${ENABLE_NON_COT_REC}"
echo "ENABLE_VIDEO_REC     : ${ENABLE_VIDEO_REC}"
echo "ENABLE_AD_REC        : ${ENABLE_AD_REC}"
echo "ENABLE_PRODUCT_REC   : ${ENABLE_PRODUCT_REC}"
echo "ENABLE_SID2CAPTION   : ${ENABLE_SID2CAPTION}"
echo "ENABLE_CAPTION2SID   : ${ENABLE_CAPTION2SID}"
echo "ALLOW_OVERLAP        : ${ALLOW_COT_NON_COT_USER_OVERLAP}"
echo "COT_SPLIT_FILTER    : ${ENABLE_COT_SPLIT_FILTER}"
echo "COT_TARGET_SOURCE    : ${COT_TARGET_SOURCE}"
echo ""
echo "PRODUCT_CAPTION_DIR       : ${PRODUCT_CAPTION_DIR:-<none>}"
echo "ENABLE_PRODUCT_SID2CAPTION: ${ENABLE_PRODUCT_SID2CAPTION}"
echo "ENABLE_PRODUCT_CAPTION2SID: ${ENABLE_PRODUCT_CAPTION2SID}"
echo "PRODUCT_SID2CAPTION_SIZE  : ${PRODUCT_SID2CAPTION_SIZE}"
echo "PRODUCT_CAPTION2SID_SIZE  : ${PRODUCT_CAPTION2SID_SIZE}"
echo ""
echo "GLOBAL_SHUFFLE       : ${GLOBAL_SHUFFLE}"
echo "SAMPLE_RATIO         : ${SAMPLE_RATIO}"
echo "ROWS_PER_SHARD       : ${ROWS_PER_SHARD}"
echo "TARGET_MAX_LEN       : ${TARGET_MAX_LEN}"
echo "SPLIT_TARGET         : ${SPLIT_TARGET}"
echo "SID2CAPTION_SIZE     : ${SID2CAPTION_SIZE}"
echo "CAPTION2SID_SIZE     : ${CAPTION2SID_SIZE}"
echo "SHUFFLE_CAPTION_FILES: ${SHUFFLE_CAPTION_FILES}"
echo "SEED                 : ${SEED}"
echo "SIDD0C_MODE          : ${SIDD0C_MODE}"
echo "LOG_FILE             : ${LOG_FILE}"
echo "========================================"
echo ""

# ============== 启动 MPI ==============
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 本地路径场景创建输出目录
case "$OUTPUT_DIR" in
  viewfs://*|hdfs://*) ;;
  *) mkdir -p "$OUTPUT_DIR" ;;
esac
case "$TEMP_DIR" in
  viewfs://*|hdfs://*) ;;
  *) mkdir -p "$TEMP_DIR" ;;
esac
case "$BLENDED_JSON_PATH" in
  viewfs://*|hdfs://*) ;;
  *) mkdir -p "$(dirname "$BLENDED_JSON_PATH")" ;;
esac

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
echo "All done."
echo "  output_dir    : ${OUTPUT_DIR}"
echo "  blended_json  : ${BLENDED_JSON_PATH}"
echo "  log           : ${LOG_FILE}"
echo "========================================"
