#!/bin/bash
# Install OpenOneRec/benchmarks/requirements.txt on the current node and all MPI hosts.
#
# Uses the same self-reinvocation pattern as datahub/mpi_install.sh:
#   - Without --local-only: run `mpirun --pernode` to re-invoke this script with
#     --local-only on every host in HOSTFILE (one process per node).
#   - With --local-only: cd into the benchmarks dir and run
#     `python3 -m pip install <PIP_FLAGS> -r requirements.txt` on this node only.
#
# Assumes the repo (this script + requirements.txt) is on a filesystem shared
# across all MPI hosts and reachable at the same absolute path on every node.
#
# Usage:
#   bash mpi_install.sh                      # install on current node + all MPI hosts
#   bash mpi_install.sh --local-only         # install on current node only
#   bash mpi_install.sh --hostfile /path     # override hostfile
#   bash mpi_install.sh --python python3.10  # override python binary
#   PIP_FLAGS="--user" bash mpi_install.sh   # pass extra pip flags
#
# Environment overrides:
#   HOSTFILE=/path/to/hostfile PYTHON_BIN=python3.10 PIP_FLAGS="--user" bash mpi_install.sh

set -euo pipefail

HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_FLAGS="${PIP_FLAGS:-}"
RUN_REMOTE=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCH_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
REQ_PATH="${REQ_PATH:-${BENCH_DIR}/requirements.txt}"

usage() {
  cat <<EOF
Usage: bash mpi_install.sh [options]

Install ${REQ_PATH} on the current node and all MPI hosts from ${HOSTFILE}.

Options:
  --hostfile PATH     MPI hostfile path. Default: ${HOSTFILE}
  --python PATH       Python executable. Default: ${PYTHON_BIN}
  --req PATH          requirements file. Default: ${REQ_PATH}
  --local-only        Install only on the current node (skip mpirun).
  -h, --help          Show this help.

Environment overrides:
  HOSTFILE=/path/to/hostfile PYTHON_BIN=python3.10 PIP_FLAGS="--user" bash mpi_install.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostfile)   HOSTFILE="$2"; shift 2 ;;
    --python)     PYTHON_BIN="$2"; shift 2 ;;
    --req)        REQ_PATH="$2"; shift 2 ;;
    --local-only) RUN_REMOTE=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_REMOTE" -eq 1 && ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  echo "Use --hostfile PATH or --local-only." >&2
  exit 1
fi
if [[ ! -f "$REQ_PATH" ]]; then
  echo "Requirements file not found: $REQ_PATH" >&2
  exit 1
fi

echo "Benchmarks dir: $BENCH_DIR"
echo "Requirements:   $REQ_PATH"
echo "Python:         $PYTHON_BIN"
[[ -n "$PIP_FLAGS" ]] && echo "Pip flags:      $PIP_FLAGS"
if [[ "$RUN_REMOTE" -eq 1 ]]; then
  echo "Hostfile: $HOSTFILE"
  echo "Hosts:"
  cat "$HOSTFILE"
else
  echo "Remote install: disabled (--local-only)"
fi

install_one_node() {
  cd "$BENCH_DIR"
  "$PYTHON_BIN" -m pip install $PIP_FLAGS -r "$REQ_PATH"
}

echo
echo "Installing on current node..."
install_one_node

if [[ "$RUN_REMOTE" -eq 1 ]]; then
  echo
  echo "Installing on MPI hosts with mpirun --pernode..."
  REMOTE_ARGS=(--local-only --hostfile "$HOSTFILE" --python "$PYTHON_BIN" --req "$REQ_PATH")
  mpirun --allow-run-as-root \
    --hostfile "$HOSTFILE" \
    --pernode \
    bash "$SCRIPT_DIR/mpi_install.sh" "${REMOTE_ARGS[@]}"
fi

echo
echo "OpenOneRec benchmarks MPI installation finished."
