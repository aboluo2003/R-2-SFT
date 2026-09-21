#!/bin/bash
# Recompute evaluation metrics from existing generation results (no regeneration).
#
# Use this after fixing the recall@k bug in
# benchmarks/benchmark/tasks/v1_0/recommendation/utils.py and utils_by_pid.py
# to avoid regenerating all predictions.
#
# Usage (from anywhere):
#   bash OpenOneRec/benchmarks/recompute_metrics.sh
#
# Override via env vars:
#   RESULT_NAME=rec_domains_eval bash OpenOneRec/benchmarks/recompute_metrics.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data/RecIF/benchmark_data}"
RESULT_NAME="${RESULT_NAME:-rec_domains_eval}"
DATA_VERSION="${DATA_VERSION:-v1.0}"

BASE_OUTPUT_DIR="${SCRIPT_DIR}/results/${DATA_VERSION}/results_${RESULT_NAME}"
EVAL_RESULTS_PATH="${BASE_OUTPUT_DIR}/eval_results.json"

if [ ! -d "${BASE_OUTPUT_DIR}" ]; then
    echo "Error: results dir not found: ${BASE_OUTPUT_DIR}"
    echo "       Set RESULT_NAME env var to match the generation run."
    exit 1
fi

echo "========================================"
echo "Recomputing metrics (no regeneration)"
echo "========================================"
echo "Results dir:  ${BASE_OUTPUT_DIR}"
echo "Data dir:     ${DATA_DIR}"
echo "Output json:  ${EVAL_RESULTS_PATH}"
echo ""

# Verify generation files exist for the 3 tasks
MODEL_DIR=""
for d in "${BASE_OUTPUT_DIR}"/*/; do
    [ -d "$d" ] && MODEL_DIR="$d" && break
done
if [ -z "${MODEL_DIR}" ]; then
    echo "Error: no model subdirectory found under ${BASE_OUTPUT_DIR}"
    exit 1
fi
echo "Model dir:    ${MODEL_DIR}"
for task in ad product video; do
    gen_file="${MODEL_DIR}${task}/test_generated.json"
    if [ ! -f "${gen_file}" ]; then
        echo "Error: generation file not found: ${gen_file}"
        exit 1
    fi
    n_samples=$(python3 -c "import json; d=json.load(open('${gen_file}')); print(len(d.get('samples', {})))" 2>/dev/null || echo "?")
    echo "  ${task}: ${n_samples} samples in ${gen_file##*/}"
done
echo ""

python3 -u -c "
from benchmark import Benchmark
Benchmark.evaluate_dev(
    generation_results_dir='${BASE_OUTPUT_DIR}',
    output_path='${EVAL_RESULTS_PATH}',
    data_dir='${DATA_DIR}',
    overwrite=True,
    task_types=['ad', 'product', 'video']
)
"
echo ""

echo "========================================"
echo "Results:"
echo "========================================"
if [ -f "${EVAL_RESULTS_PATH}" ]; then
    EVAL_RESULTS_PATH="${EVAL_RESULTS_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

results_path = Path(os.environ["EVAL_RESULTS_PATH"])
with results_path.open() as f:
    data = json.load(f)

print(f"{'Task':<12} {'pass@32':>10} {'pos1@32':>10} {'recall@32':>10}   |   {'pid_pass@32':>12} {'pid_recall@32':>14}")
print("-" * 80)
for model_name, model_results in data.items():
    if model_name.startswith("_"):
        continue
    for task_name in ["ad", "product", "video"]:
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
    echo "Results file not generated."
fi
