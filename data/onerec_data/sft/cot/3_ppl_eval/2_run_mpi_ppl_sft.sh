#!/usr/bin/env bash
set -euo pipefail

# 多机多卡 PPL 评估启动脚本 (2_eval_ppl.py)。
#
# 约定：
# 1. hostfile 使用 MPI 标准格式，例如：
#    <node-ip-1> slots=8
#    <node-ip-2> slots=8
#    每个 slots 对应一张 GPU，每个 rank 绑定一张卡。
# 2. 本脚本对齐 reason/mpi/cross_reco/ablation/0701_xxx/run_mpi_collect_shared_device_yes.sh
#    的 MPI / NCCL 通信参数，避免随意修改导致 NCCL / MPI 卡死。
# 3. 不使用 vllm：旧版 vllm 无法只返回目标 token 的 logits，全序列 logits 会 OOM。
#    这里走 transformers + flash_attention_2 + logits_to_keep。
#
# 用法：
#   bash 2_run_mpi.sh
#   bash 2_run_mpi.sh /etc/mpi/hostfile
#   REPRESENTATION_TYPE=sid     bash 2_run_mpi.sh
#   REPRESENTATION_TYPE=caption bash 2_run_mpi.sh
#
# 可选环境变量：
#   REPRESENTATION_TYPE  caption | sid（决定默认模型与默认输入目录）
#   INPUT_URI            1_preprocess.py 的输出目录/单文件（默认 preprocessed_${REPRESENTATION_TYPE}）
#   MODEL_PATH           HF 模型路径（sid 默认 OpenOneRec pretrain-8B/converted，
#                        caption 默认 Qwen3-8B）
#   OUTPUT_URI           PPL 评估输出目录（默认 ppl_eval_${REPRESENTATION_TYPE}）
#   BATCH_SIZE           每个 micro-batch 的序列数（默认 32）
#   MAX_SEQ_LEN          单条序列最大长度（默认 32768）
#   DTYPE                默认 bfloat16
#   ATTN_IMPLEMENTATION  默认 flash_attention_2（失败回退 sdpa/eager）
#   USE_SYSTEM_MESSAGE   是否把 system_text 拼进 chat template（默认 1）
#   ENABLE_THINKING      chat template 的 enable_thinking（默认 0）
#   SLOTS_PER_NODE       覆盖 hostfile 中 slots（默认沿用 hostfile）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${1:-${HOSTFILE:-/etc/mpi/hostfile}}"
PY_SCRIPT="${2:-${PY_SCRIPT:-${SCRIPT_DIR}/2_eval_ppl.py}}"

# ============== 路径前缀 ==============
PREFIX="${PREFIX:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

# ============== 默认输入/输出/模型路径 ==============
REPRESENTATION_TYPE="${REPRESENTATION_TYPE:-sid}"
PREPROCESSED_URI="${PREPROCESSED_URI:-${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/preprocessed_${REPRESENTATION_TYPE}}"
INPUT_URI="${INPUT_URI:-${PREPROCESSED_URI}}"

DEFAULT_MODEL_SID="${PREFIX}/models/0708_cot_ver0/step200/converted/"
DEFAULT_MODEL_CAPTION="${PREFIX}/models/Qwen3-8B"
if [ "$REPRESENTATION_TYPE" = "sid" ]; then
  DEFAULT_MODEL_PATH="$DEFAULT_MODEL_SID"
elif [ "$REPRESENTATION_TYPE" = "caption" ]; then
  DEFAULT_MODEL_PATH="$DEFAULT_MODEL_CAPTION"
else
  echo "[ERROR] 不支持的 REPRESENTATION_TYPE=${REPRESENTATION_TYPE} (仅支持 sid | caption)" >&2
  exit 1
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

DEFAULT_OUTPUT_URI="${PREFIX}/data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_${REPRESENTATION_TYPE}"
OUTPUT_URI="${OUTPUT_URI:-${DEFAULT_OUTPUT_URI}}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
# ★ 动态 batch 的 token 预算上限（默认 131072 = 32×4096）。
#   valid_seqs 按长度降序排序后，按当前批最长序列反推单批 bs = min(batch_size, token_budget // max_len)。
#   长 seq 自动小 batch 防 OOM，短 seq 自动大 batch 保吞吐。设为 0 禁用，退化为固定 batch_size。
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-131072}"
# ★ ETA 日志频率：每 N 个文件打一次进度+ETA（默认 10）。
ETA_LOG_INTERVAL="${ETA_LOG_INTERVAL:-10}"
# ★ PyTorch CUDA 分配器：expandable_segments:True 解决碎片化 OOM（报错里 19G reserved-but-unallocated）。
#   允许分配段按需扩展并复用已释放块，避免"有 19G 但凑不出 11.42G 连续块"。纯数据并行推理无 NCCL allreduce，安全。
#   已设值则沿用用户环境（不覆盖）。
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
USE_SYSTEM_MESSAGE="${USE_SYSTEM_MESSAGE:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/mpi_ppl_eval_${REPRESENTATION_TYPE}_$(date +%Y%m%d_%H%M%S).log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ============== Java / Hadoop / 网络 ==============
JAVA_HOME="${JAVA_HOME:-/home/hadoop/software/java}"
HADOOP_HOME="${HADOOP_HOME:-/home/hadoop/software/hadoop}"
JAVA_LIBJVM_DIR="${JAVA_LIBJVM_DIR:-$JAVA_HOME/jre/lib/amd64/server}"
HADOOP_NATIVE_DIR="${HADOOP_NATIVE_DIR:-$HADOOP_HOME/lib/native}"
HADOOP_USER_NAME="${HADOOP_USER_NAME:-mpi}"
# 自动探测本机网卡名（集群上建议直接 export TCP_NIC=ethX 覆盖）
if [ -z "${TCP_NIC:-}" ]; then
  _my_ip="$(hostname -i 2>/dev/null || echo 127.0.0.1)"
  TCP_NIC="$(ifconfig 2>/dev/null | grep -B1 " ${_my_ip} " | grep -o '^[a-zA-Z0-9]*' | head -n1 || true)"
  TCP_NIC="${TCP_NIC:-}"
fi
MASTER_ADDR="${MASTER_ADDR:-${MY_NODE_IP:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-29599}"

# ============== 校验 ==============
if [[ ! -f "${HOSTFILE}" ]]; then
  echo "[ERROR] hostfile 不存在: ${HOSTFILE}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] python 入口不存在: ${PY_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

NODE_COUNT="$(grep -v '^#' "${HOSTFILE}" | awk 'NF > 0 {count++} END {print count + 0}')"
TOTAL_SLOTS="$(grep -v '^#' "${HOSTFILE}" | awk '
  NF > 0 {
    slots = 1
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^slots=/) {
        split($i, a, "=")
        slots = a[2] + 0
      }
    }
    total += slots
  }
  END {print total + 0}
')"
if [[ "${NODE_COUNT}" -le 0 || "${TOTAL_SLOTS}" -le 0 ]]; then
  echo "[ERROR] 无法从 hostfile 解析出有效节点或 slots: ${HOSTFILE}" >&2
  exit 1
fi

# 本地路径校验（viewfs:// / hdfs:// 跳过）
case "${INPUT_URI}" in
  viewfs://*|hdfs://*) ;;
  *) [ -e "${INPUT_URI}" ] || { echo "[ERROR] 输入路径不存在: ${INPUT_URI}" >&2; exit 1; } ;;
esac
case "${MODEL_PATH}" in
  viewfs://*|hdfs://*) ;;
  *) [ -e "${MODEL_PATH}" ] || { echo "[ERROR] 模型路径不存在: ${MODEL_PATH}" >&2; exit 1; } ;;
esac

echo "========================================"
echo "Step 3a (ppl_eval) — 多机多卡 PPL 评估"
echo "========================================"
echo "hostfile            : ${HOSTFILE}"
echo "node_count          : ${NODE_COUNT}"
echo "total_slots(gpu)    : ${TOTAL_SLOTS}"
echo "representation_type : ${REPRESENTATION_TYPE}"
echo "input_uri           : ${INPUT_URI}"
echo "model_path          : ${MODEL_PATH}"
echo "output_uri          : ${OUTPUT_URI}"
echo "batch_size          : ${BATCH_SIZE}"
echo "max_seq_len         : ${MAX_SEQ_LEN}"
echo "max_tokens_per_batch: ${MAX_TOKENS_PER_BATCH}"
echo "eta_log_interval   : ${ETA_LOG_INTERVAL}"
echo "dtype               : ${DTYPE}"
echo "attn_implementation : ${ATTN_IMPLEMENTATION}"
echo "use_system_message  : ${USE_SYSTEM_MESSAGE}"
echo "enable_thinking     : ${ENABLE_THINKING}"
echo "local_files_only    : ${LOCAL_FILES_ONLY}"
echo "pytorch_cuda_alloc  : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "log_file            : ${LOG_FILE}"
echo ""

# 创建本地输出目录
case "${OUTPUT_URI}" in
  viewfs://*|hdfs://*) ;;
  *) mkdir -p "${OUTPUT_URI}" ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF
export HADOOP_USER_NAME
export LD_LIBRARY_PATH="${HADOOP_NATIVE_DIR}:${JAVA_HOME}/jre/lib/amd64/server:${JAVA_HOME}/lib/server:${JAVA_HOME}/lib/amd64/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CLASSPATH="${CLASSPATH:+$CLASSPATH:}$(${HADOOP_HOME}/bin/hadoop classpath --glob)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

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
  -x LD_LIBRARY_PATH \
  -x PATH \
  -x PYTHONPATH \
  -x JAVA_HOME \
  -x HADOOP_HOME \
  -x CLASSPATH \
  -x HADOOP_USER_NAME \
  -x MASTER_ADDR="${MASTER_ADDR}" \
  -x MASTER_PORT="${MASTER_PORT}" \
  -x TOKENIZERS_PARALLELISM=false \
  -x PYTORCH_CUDA_ALLOC_CONF \
  bash -c "ORIG_LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-0}; \
    export CUDA_VISIBLE_DEVICES=\${ORIG_LOCAL_RANK}; \
    export LOCAL_RANK=0; \
    export OMPI_COMM_WORLD_LOCAL_RANK=0; \
    echo \"[Rank \${OMPI_COMM_WORLD_RANK:-0}] ORIG_LOCAL_RANK=\${ORIG_LOCAL_RANK}, CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\"; \
    ${PYTHON_BIN} '${PY_SCRIPT}' \
      --input_uri '${INPUT_URI}' \
      --output_uri '${OUTPUT_URI}' \
      --representation_type '${REPRESENTATION_TYPE}' \
      --model_path '${MODEL_PATH}' \
      --dtype '${DTYPE}' \
      --attn_implementation '${ATTN_IMPLEMENTATION}' \
      --local_files_only '${LOCAL_FILES_ONLY}' \
      --use_system_message '${USE_SYSTEM_MESSAGE}' \
      --enable_thinking '${ENABLE_THINKING}' \
      --batch_size '${BATCH_SIZE}' \
      --max_seq_len '${MAX_SEQ_LEN}' \
      --max_tokens_per_batch '${MAX_TOKENS_PER_BATCH}' \
      --eta_log_interval '${ETA_LOG_INTERVAL}'" \
  2>&1 | tee "${LOG_FILE}"
