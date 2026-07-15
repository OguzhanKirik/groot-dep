#!/usr/bin/env bash
# One-command GR00T + Isaac eval (single process by default — avoids GPU/PhysX conflicts).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-adam-u-groot-unified}"
ISAAC_CONDA_ENV="${ISAAC_CONDA_ENV:-adam-u-isaac-6}"
ADAM_REPO="${ADAM_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/home/revel/models/GR00T-N1.7-3B}"
GROOT_SCHEMA="${GROOT_SCHEMA:-real_g1}"
MAX_STEPS="${MAX_STEPS:-1000}"
GUI="${GUI:-0}"
# Keep Isaac's runtime isolated from GR00T's Python/CUDA dependencies.
INPROCESS="${INPROCESS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

echo "[pipeline] Cleaning up stale GR00T/Isaac processes..."
QUIET=1 bash "${SCRIPT_DIR}/cleanup_groot_stack.sh" --stop --wait

if [[ "${INPROCESS}" == "1" ]]; then
  echo "[pipeline] Single-process mode (Isaac first, then GR00T in-process)."
  echo "[pipeline] Using conda env '${CONDA_ENV}'."
  EVAL_ARGS=(
    adam_u_groot/scripts/eval_groot.py
    --mode groot
    --groot-inprocess
    --groot-model-path "${MODEL_PATH}"
    --groot-schema "${GROOT_SCHEMA}"
    --enable_cameras
    --max_steps "${MAX_STEPS}"
  )
else
  GROOT_REPO="${GROOT_REPO:-$HOME/Isaac-GR00T}"
  GROOT_PORT="${GROOT_PORT:-5555}"
  EMBODIMENT_TAG="${EMBODIMENT_TAG:-REAL_G1}"
  SERVER_LOG="${SERVER_LOG:-/tmp/gr00t_server_${GROOT_PORT}.log}"

  echo "[pipeline] Two-process mode (Isaac first, then background GR00T server)."
  echo "[pipeline] GR00T env: '${CONDA_ENV}'; Isaac env: '${ISAAC_CONDA_ENV}'."

  EVAL_ARGS=(
    adam_u_groot/scripts/eval_groot.py
    --mode groot
    --groot-schema "${GROOT_SCHEMA}"
    --groot_port "${GROOT_PORT}"
    --groot-launch-server
    --groot-server-conda-env "${CONDA_ENV}"
    --groot-server-script "${GROOT_REPO}/gr00t/eval/run_gr00t_server.py"
    --groot-model-path "${MODEL_PATH}"
    --groot-server-log "${SERVER_LOG}"
    --enable_cameras
    --max_steps "${MAX_STEPS}"
  )
fi

if [[ "${GUI}" == "1" ]]; then
  EVAL_ARGS+=(--gui)
fi
if [[ $# -gt 0 ]]; then
  EVAL_ARGS+=("$@")
fi

echo "[pipeline] Launching Isaac eval..."
cd "${ADAM_REPO}"
EVAL_CONDA_ENV="${CONDA_ENV}"
if [[ "${INPROCESS}" != "1" ]]; then
  EVAL_CONDA_ENV="${ISAAC_CONDA_ENV}"
fi
conda run -n "${EVAL_CONDA_ENV}" --no-capture-output python "${EVAL_ARGS[@]}"
