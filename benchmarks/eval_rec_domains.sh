#!/bin/bash
# Evaluate OpenOneRec-8B on 3 recommendation domains: ad / product / video
#
# This is a slimmed-down version of benchmarks/eval_script.sh that only runs
# the three cross/within-domain recommendation tasks (no label_pred / item_understand /
# rec_reason / interactive / label_cond), so you can quickly sanity-check the rec metrics
# against the published numbers in OpenOneRec/README.md:
#   - Short Video Rec  Recall@32 = 0.0355
#   - Ad Rec           Recall@32 = 0.0877
#   - Product Rec      Recall@32 = 0.0470
#
# Usage (run from anywhere -- the script auto-locates benchmarks/ and sets PYTHONPATH):
#   bash OpenOneRec/benchmarks/eval_rec_domains.sh <model_path> <eval_name> [enable_thinking]
#
# Positional args (required args have no default -- the script will abort if they are missing):
#   $1 = MODEL_PATH        : path to the HuggingFace-format model directory (required)
#   $2 = eval_name (base)  : base name for result dir / log (required, suffix auto-appended below)
#   $3 = ENABLE_THINKING   : "true" / "false" (optional, defaults to "false")
#
# Optional env vars:
#   EVAL_DOMAINS=ad|product|video|ad,video  Only run selected recommendation domains
#
# The real RESULT_NAME is built as "<eval_name>_<suffix>", where suffix is
# "_think" if ENABLE_THINKING=true, otherwise "_nothink".
# Example:
#   bash OpenOneRec/benchmarks/eval_rec_domains.sh /path/to/OpenOneRec-8B rec_domains false
#   -> RESULT_NAME="rec_domains_nothink"
#
# Other paths can still be overridden via env vars:
#   DATA_DIR=/custom/data bash OpenOneRec/benchmarks/eval_rec_domains.sh /path/to/model rec_domains true
#
# Quick smoke test (10 samples per task):
#   SAMPLE_SIZE=10 bash OpenOneRec/benchmarks/eval_rec_domains.sh /path/to/model rec_domains false
#
# Single-domain eval:
#   EVAL_DOMAINS=ad bash OpenOneRec/benchmarks/eval_rec_domains.sh /path/to/model rec_ad true

set -e

# ============== Auto-locate benchmarks/ dir ==============
# Resolve the directory this script lives in, so it works regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="${SCRIPT_DIR}"

# Make sure `benchmark` package is importable (evaluate.py does `from benchmark import ...`)
export PYTHONPATH="${BENCHMARKS_DIR}:${PYTHONPATH}"

# ============== Configuration ==============
# Default to the repository root; override WORKSPACE_ROOT or DATA_DIR as needed.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${BENCHMARKS_DIR}/.." && pwd)}"

usage() {
    cat <<USAGE
Usage:
  bash OpenOneRec/benchmarks/eval_rec_domains.sh <model_path> <eval_name> [enable_thinking]

Arguments:
  \$1 model_path      (required) path to the HuggingFace-format model directory
  \$2 eval_name       (required) base name for result dir / log (a "_think"/"_nothink" suffix is appended)
  \$3 enable_thinking (optional) "true" / "false", defaults to "false"

Example:
  bash OpenOneRec/benchmarks/eval_rec_domains.sh /path/to/OpenOneRec-8B rec_domains false
  -> RESULT_NAME="rec_domains_nothink"
USAGE
}

# MODEL_PATH (required, no default)
if [ -z "${1:-}" ]; then
    echo "Error: <model_path> is required (positional arg \$1)." >&2
    echo "" >&2
    usage >&2
    exit 1
fi
MODEL_PATH="${1}"

# eval_name base (required, no default)
if [ -z "${2:-}" ]; then
    echo "Error: <eval_name> is required (positional arg \$2)." >&2
    echo "" >&2
    usage >&2
    exit 1
fi
RESULT_NAME_BASE="${2}"

# ENABLE_THINKING (optional, defaults to "false")
ENABLE_THINKING="${3:-false}"

# Build the real RESULT_NAME with a thinking suffix:
#   ENABLE_THINKING=true  -> "<base>_think"
#   ENABLE_THINKING=false -> "<base>_nothink"
if [ "${ENABLE_THINKING}" = "true" ]; then
    RESULT_NAME="${RESULT_NAME_BASE}_think"
else
    RESULT_NAME="${RESULT_NAME_BASE}_nothink"
fi

DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data/RecIF/benchmark_data}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
EVAL_DOMAINS="${EVAL_DOMAINS:-ad,product,video}"

IFS=',' read -r -a TASKS <<< "${EVAL_DOMAINS}"
VALID_TASKS=()
for TASK in "${TASKS[@]}"; do
    TASK="$(echo "${TASK}" | xargs)"
    [ -z "${TASK}" ] && continue
    case "${TASK}" in
        ad|product|video) VALID_TASKS+=("${TASK}") ;;
        *) echo "Error: unsupported EVAL_DOMAINS task: ${TASK} (allowed: ad, product, video)" >&2; exit 1 ;;
    esac
done
if [ "${#VALID_TASKS[@]}" -eq 0 ]; then
    echo "Error: EVAL_DOMAINS produced no tasks: ${EVAL_DOMAINS}" >&2
    exit 1
fi

# Benchmark framework dirs -- use absolute paths under BENCHMARKS_DIR so CWD doesn't matter.
DATA_VERSION="${DATA_VERSION:-v1.0}"

BASE_OUTPUT_DIR="${BENCHMARKS_DIR}/results/${DATA_VERSION}/results_${RESULT_NAME}"
BASE_LOG_NAME="${BENCHMARKS_DIR}/auto_eval_logs/${DATA_VERSION}/${RESULT_NAME}"
EVAL_SCRIPT="${BENCHMARKS_DIR}/scripts/ray-vllm/evaluate.py"

# ============== Pre-flight checks ==============
echo "========================================"
echo "RecIF Recommendation Domain Evaluation"
echo "========================================"
echo "Model:          ${MODEL_PATH}"
echo "Data dir:       ${DATA_DIR}"
echo "Result name:    ${RESULT_NAME} (base: ${RESULT_NAME_BASE})"
echo "Enable thinking: ${ENABLE_THINKING}"
echo "Eval domains:    ${VALID_TASKS[*]}"
[ -n "${SAMPLE_SIZE}" ] && echo "Sample size:    ${SAMPLE_SIZE} (smoke test)"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: model dir not found: ${MODEL_PATH}"
    echo "       Pass the HuggingFace-format model directory as the first positional arg."
    exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: benchmark data dir not found: ${DATA_DIR}"
    echo "       Set DATA_DIR env var to the RecIF benchmark_data directory"
    echo "       (the one containing ad/, product/, video/, sid2pid.json, sid2iid.json)."
    exit 1
fi
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "Error: evaluate.py not found at: ${EVAL_SCRIPT}"
    echo "       Script dir: ${BENCHMARKS_DIR}"
    exit 1
fi

# Verify required data files for selected domains
for sub in "${VALID_TASKS[@]}"; do
    if [ ! -f "${DATA_DIR}/${sub}/${sub}_test.parquet" ]; then
        echo "Error: missing ${DATA_DIR}/${sub}/${sub}_test.parquet"
        exit 1
    fi
done
for mapping in sid2pid.json sid2iid.json; do
    if [ ! -f "${DATA_DIR}/${mapping}" ]; then
        echo "Warning: ${DATA_DIR}/${mapping} not found -- PID-mode metrics will fail."
        echo "         (SID-mode pass@k / recall@k still work without it.)"
    fi
done

mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "${BASE_OUTPUT_DIR}"

# Build thinking args
THINKING_ARGS=""
if [ "${ENABLE_THINKING}" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

# Build optional sample-size arg
SAMPLE_ARG=""
[ -n "${SAMPLE_SIZE}" ] && SAMPLE_ARG="--sample_size ${SAMPLE_SIZE}"

{
    echo "========== Task Configuration =========="
    echo "MODEL_PATH: ${MODEL_PATH}"
    echo "DATA_DIR: ${DATA_DIR}"
    echo "Enable Thinking: ${ENABLE_THINKING}"
    echo "Eval Domains: ${VALID_TASKS[*]}"
    echo "Sample Size: ${SAMPLE_SIZE:-full}"
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

echo "Thinking args: ${THINKING_ARGS}"
echo "Logs:          ${BASE_LOG_NAME}.log"
echo "Results dir:   ${BASE_OUTPUT_DIR}"
echo ""

# ============== Run 3 recommendation tasks ==============
# These three tasks share the same generation config (beam 32, 32 return sequences,
# 1 thinking sequence, batch 1875). See benchmarks/benchmark/tasks/v1_0/recommendation/config.py.

for TASK in "${VALID_TASKS[@]}"; do
    echo "========================================"
    echo "Task: ${TASK}"
    echo "========================================"
    python3 -u "${EVAL_SCRIPT}" \
        --task_types "${TASK}" \
        --gpu_memory_utilization 0.8 \
        --model_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${BASE_OUTPUT_DIR}" \
        --dtype bfloat16 \
        --worker_batch_size 1875 \
        --overwrite \
        --num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 \
        ${THINKING_ARGS} ${SAMPLE_ARG} >> "${BASE_LOG_NAME}.log" 2>&1
    echo "  -> done. See log tail below:"
    tail -n 5 "${BASE_LOG_NAME}.log"
    echo ""
done

# ============== Compute metrics ==============
echo "========================================"
echo "Computing evaluation metrics"
echo "========================================"
EVAL_RESULTS_PATH="${BASE_OUTPUT_DIR}/eval_results.json"
# PYTHONPATH already exported at the top so `from benchmark import Benchmark` works.
python3 -u -c "
from benchmark import Benchmark
task_types = '${EVAL_DOMAINS}'.split(',')
task_types = [t.strip() for t in task_types if t.strip()]
Benchmark.evaluate_dev(
    generation_results_dir='${BASE_OUTPUT_DIR}',
    output_path='${EVAL_RESULTS_PATH}',
    data_dir='${DATA_DIR}',
    overwrite=True,
    task_types=task_types
)
" >> "${BASE_LOG_NAME}.log" 2>&1

echo ""
echo "========================================"
echo "All tasks completed. Results:"
echo "========================================"
if [ -f "${EVAL_RESULTS_PATH}" ]; then
    echo "Results file: ${EVAL_RESULTS_PATH}"
    echo ""
    # Pretty-print the per-task recall@32 / pass@32 metrics
    # Pass the path via env var so the Python code stays a literal block.
    EVAL_RESULTS_PATH="${EVAL_RESULTS_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

results_path = Path(os.environ["EVAL_RESULTS_PATH"])
if not results_path.exists():
    print("Results file not found, check logs.")
    raise SystemExit(0)

with results_path.open() as f:
    data = json.load(f)

task_names = os.environ.get("EVAL_DOMAINS", "ad,product,video").split(",")
task_names = [t.strip() for t in task_names if t.strip()]
print(f"{'Task':<12} {'pass@32':>10} {'pos1@32':>10} {'recall@32':>10}   |   {'pid_pass@32':>12} {'pid_recall@32':>14}")
print("-" * 80)
for model_name, model_results in data.items():
    if model_name.startswith("_"):
        continue
    for task_name in task_names:
        if task_name not in model_results:
            continue
        for split, m in model_results[task_name].items():
            if not isinstance(m, dict):
                continue
            sid_pass = m.get("pass@32", "?")
            sid_pos1 = m.get("position1_pass@32", "?")
            sid_recall = m.get("recall@32", "?")
            pid_pass = m.get("pid_pass@32", "-")
            pid_recall = m.get("pid_recall@32", "-")
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            print(f"{task_name:<12} {fmt(sid_pass):>10} {fmt(sid_pos1):>10} {fmt(sid_recall):>10}   |   {fmt(pid_pass):>12} {fmt(pid_recall):>14}")

print()
print("Published baselines (OpenOneRec-8B, README.md):")
print("  Video   Recall@32 = 0.0355")
print("  Ad      Recall@32 = 0.0877")
print("  Product Recall@32 = 0.0470")
PY
else
    echo "Results file not generated. Check log: ${BASE_LOG_NAME}.log"
fi
echo ""
echo "Full log: ${BASE_LOG_NAME}.log"
