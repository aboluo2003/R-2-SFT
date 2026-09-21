#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 0 (caption_summary) — 拆分 + Qwen3-8B vLLM 摘要
# ============================================================================
# 流程：
#   1. 在所有节点上安装依赖（vllm, transformers）
#   2. rank0 单进程拆分 pid2caption.parquet → caption_shards/
#   3. MPI 多机多卡并行摘要：每 rank 绑 1 GPU，vLLM TP=1，文件 round-robin
#
# 用法：
#   bash run_mpi.sh
#   bash run_mpi.sh /etc/mpi/hostfile
#   MODEL_PATH=/path/to/Qwen3-8B SKIP_INSTALL=1 bash run_mpi.sh
#   bash OpenOneRec/data/onerec_data/sft/cot/0_caption_summary/run_mpi.sh
#   SKIP_INSTALL=1 SKIP_SPLIT=1 bash OpenOneRec/data/onerec_data/sft/cot/0_caption_summary/run_mpi.sh
#
# 必填环境变量：
#   （无必填，均有默认值）
#
# 可选环境变量：
#   MODEL_PATH         Qwen3-8B 模型路径
#   INPUT_PARQUET      pid2caption.parquet 路径
#   SHARDS_DIR         拆分输出目录
#   SUMMARY_DIR        摘要输出目录
#   ROWS_PER_SHARD     每分片行数，默认 5000
#   SKIP_INSTALL       1=跳过安装依赖
#   SKIP_SPLIT         1=跳过拆分（已有分片时）
#   BATCH_SIZE         每次 llm.generate 的 prompt 数，默认 512
#   MAX_MODEL_LEN      vLLM 最大上下文长度，默认 2048
#   GPU_MEM_UTIL       vLLM GPU 显存占用比例，默认 0.9
#   MAX_FILES          调试：限制总文件数，0=不限
#   MAX_SAMPLES        调试：限制总样本数，0=不限
#   OVERWRITE          1=强制覆盖已完成的摘要分片
# ============================================================================

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============== 默认路径 ==============
DEFAULT_INPUT_PARQUET="${PREFIX}/data/RecIF/pid2caption.parquet"
DEFAULT_SHARDS_DIR="${PREFIX}/data/RecIF-cot/caption_shards"
DEFAULT_SUMMARY_DIR="${PREFIX}/data/RecIF-cot/caption_summary"
DEFAULT_MODEL_PATH="${PREFIX}/models/Qwen3-8B"

INPUT_PARQUET="${INPUT_PARQUET:-${DEFAULT_INPUT_PARQUET}}"
SHARDS_DIR="${SHARDS_DIR:-${DEFAULT_SHARDS_DIR}}"
SUMMARY_DIR="${SUMMARY_DIR:-${DEFAULT_SUMMARY_DIR}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_FILES="${MAX_FILES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"

# ============== MPI / Java / Hadoop 环境 ==============
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"
JAVA_HOME="${JAVA_HOME:-/home/hadoop/software/java}"
HADOOP_HOME="${HADOOP_HOME:-/home/hadoop/software/hadoop}"
JAVA_LIBJVM_DIR="${JAVA_LIBJVM_DIR:-$JAVA_HOME/jre/lib/amd64/server}"
HADOOP_NATIVE_DIR="${HADOOP_NATIVE_DIR:-$HADOOP_HOME/lib/native}"
HADOOP_USER_NAME="${HADOOP_USER_NAME:-mpi}"
TCP_NIC="${TCP_NIC:-$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*" | head -n1)}"
MASTER_ADDR="${MASTER_ADDR:-${MY_NODE_IP:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-29599}"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/caption_summary_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$LOG_DIR"

# ============== 校验 ==============
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "[ERROR] hostfile 不存在: ${HOSTFILE}" >&2
    exit 1
fi

NODE_COUNT="$(grep -v '^#' "${HOSTFILE}" | awk 'NF > 0 {count++} END {print count + 0}')"
TOTAL_SLOTS="$(grep -v '^#' "${HOSTFILE}" | awk '
  NF > 0 {
    slots = 1
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^slots=/) { split($i, a, "="); slots = a[2] + 0 }
    }
    total += slots
  }
  END {print total + 0}
')"

if [[ "${NODE_COUNT}" -le 0 || "${TOTAL_SLOTS}" -le 0 ]]; then
    echo "[ERROR] 无法从 hostfile 解析出有效节点或 slots" >&2
    exit 1
fi

# ============== 打印配置 ==============
echo "========================================"
echo "Step 0 (caption_summary)"
echo "========================================"
echo "hostfile          : ${HOSTFILE}"
echo "node_count        : ${NODE_COUNT}"
echo "total_slots       : ${TOTAL_SLOTS}"
echo "INPUT_PARQUET     : ${INPUT_PARQUET}"
echo "SHARDS_DIR        : ${SHARDS_DIR}"
echo "SUMMARY_DIR       : ${SUMMARY_DIR}"
echo "MODEL_PATH        : ${MODEL_PATH}"
echo "ROWS_PER_SHARD    : ${ROWS_PER_SHARD}"
echo "BATCH_SIZE        : ${BATCH_SIZE}"
echo "MAX_MODEL_LEN     : ${MAX_MODEL_LEN}"
echo "GPU_MEM_UTIL      : ${GPU_MEM_UTIL}"
echo "SKIP_INSTALL      : ${SKIP_INSTALL}"
echo "SKIP_SPLIT        : ${SKIP_SPLIT}"
echo "LOG_FILE          : ${LOG_FILE}"
echo ""

# ============== Step 1: 安装依赖 ==============
if [[ "${SKIP_INSTALL}" != "1" ]]; then
    echo "========================================"
    echo "Step 1: 安装依赖（所有节点）"
    echo "========================================"
    # 安装 vllm + transformers（兼容 0701_xxx 的 transformers 版本约束）
    mpirun --allow-run-as-root -np "${NODE_COUNT}" \
        --hostfile "${HOSTFILE}" \
        --pernode \
        bash -lc "
            set -euo pipefail
            hostname
            ${PIP_BIN} install --no-cache-dir --upgrade transformers==5.12.1
            ${PIP_BIN} show transformers | sed -n '1,3p'
        "
fi

# ============== Step 2: 拆分（rank0 单进程） ==============
if [[ "${SKIP_SPLIT}" != "1" ]]; then
    echo "========================================"
    echo "Step 2: 拆分 pid2caption.parquet"
    echo "========================================"
    if [[ ! -f "${INPUT_PARQUET}" ]]; then
        echo "[ERROR] 输入文件不存在: ${INPUT_PARQUET}" >&2
        exit 1
    fi
    mkdir -p "${SHARDS_DIR}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/split.py" \
        --input "${INPUT_PARQUET}" \
        --output_dir "${SHARDS_DIR}" \
        --rows_per_shard "${ROWS_PER_SHARD}" 2>&1 | tee -a "${LOG_FILE}"
    echo ""
fi

# ============== Step 3: MPI 并行摘要 ==============
echo "========================================"
echo "Step 3: vLLM Qwen3-8B 摘要（${TOTAL_SLOTS} GPU）"
echo "========================================"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HADOOP_USER_NAME
export LD_LIBRARY_PATH="${HADOOP_NATIVE_DIR}:${JAVA_HOME}/jre/lib/amd64/server:${JAVA_HOME}/lib/server:${JAVA_HOME}/lib/amd64/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CLASSPATH="${CLASSPATH:+$CLASSPATH:}$(${HADOOP_HOME}/bin/hadoop classpath --glob)"

mkdir -p "${SUMMARY_DIR}"

# 组装 summary.py 参数
SUMMARY_ARGS="--input_dir ${SHARDS_DIR} --output_dir ${SUMMARY_DIR} --model_path ${MODEL_PATH}"
SUMMARY_ARGS="${SUMMARY_ARGS} --batch_size ${BATCH_SIZE} --max_model_len ${MAX_MODEL_LEN}"
SUMMARY_ARGS="${SUMMARY_ARGS} --gpu_memory_utilization ${GPU_MEM_UTIL}"
SUMMARY_ARGS="${SUMMARY_ARGS} --max_files ${MAX_FILES} --max_samples ${MAX_SAMPLES}"
if [[ "${OVERWRITE}" = "1" ]]; then
    SUMMARY_ARGS="${SUMMARY_ARGS} --overwrite"
fi

# MPI 启动（环境参数对齐 run_mpi_collect_shared_device_yes.sh）
stdbuf -oL -eL mpirun --allow-run-as-root \
    -hostfile "${HOSTFILE}" \
    -mca btl self,tcp -mca pml ob1 \
    -mca plm_rsh_num_concurrent 600 \
    -mca routed_radix 600 \
    -mca btl_tcp_if_include "${TCP_NIC}" \
    -mca oob_tcp_if_include "${TCP_NIC}" \
    -mca btl_openib_allow_ib false \
    -mca opal_set_max_sys_limits 1 \
    -x OMPI_MCA_btl=self,tcp \
    -x OMPI_MCA_pml=ob1 \
    -x OMPI_MCA_btl_tcp_if_include="${TCP_NIC}" \
    -x OMPI_MCA_oob_tcp_if_include="${TCP_NIC}" \
    -x OMPI_MCA_btl_openib_allow_ib=false \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_SOCKET_IFNAME="${TCP_NIC}" \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_NET_OVERHEAD=1000 \
    -x NCCL_IB_TIMEOUT=20 \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -x NO_COLOR=1 \
    -x TERM=dumb \
    -x COLORTERM=0 \
    -x PYTHONIOENCODING=utf-8 \
    -x VLLM_LOGGING_LEVEL=WARNING \
    -x HF_HUB_DISABLE_TELEMETRY=1 \
    -x TOKENIZERS_PARALLELISM=false \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH \
    -x JAVA_HOME \
    -x HADOOP_HOME \
    -x CLASSPATH \
    -x HADOOP_USER_NAME \
    -x MASTER_ADDR="${MASTER_ADDR}" \
    -x MASTER_PORT="${MASTER_PORT}" \
    bash -c "ORIG_LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-0}; \
        export CUDA_VISIBLE_DEVICES=\${ORIG_LOCAL_RANK}; \
        export LOCAL_RANK=0; \
        export OMPI_COMM_WORLD_LOCAL_RANK=0; \
        echo \"[Rank \${OMPI_COMM_WORLD_RANK:-0}] ORIG_LOCAL_RANK=\${ORIG_LOCAL_RANK}, CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\"; \
        ${PYTHON_BIN} '${SCRIPT_DIR}/summary.py' ${SUMMARY_ARGS}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "All done."
echo "  shards  : ${SHARDS_DIR}"
echo "  summary : ${SUMMARY_DIR}"
echo "  log     : ${LOG_FILE}"
echo "========================================"
