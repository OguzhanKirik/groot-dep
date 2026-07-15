#!/usr/bin/env bash
# Simple Adam-U sim — zero actions, no GR00T, no cameras (GUI viewport only).
set -euo pipefail

ISAAC_ENV="${ISAAC_ENV:-adam-u-isaac-6}"
ADAM_REPO="${ADAM_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
MAX_STEPS="${MAX_STEPS:-1000}"
# Headless is reliable; GUI can hang or OOM during env init on some setups.
GUI="${GUI:-0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export OMNI_KIT_ACCEPT_EULA=yes

echo "[zero-sim] Cleaning up stale Isaac/GR00T processes..."
QUIET=1 bash "${ADAM_REPO}/adam_u_groot/scripts/cleanup_groot_stack.sh" --stop --wait

EVAL_ARGS=(adam_u_groot/scripts/eval_groot.py --mode zero --max_steps "${MAX_STEPS}")
if [[ "${GUI}" == "1" ]]; then
  EVAL_ARGS+=(--gui)
else
  EVAL_ARGS+=(--headless)
fi
if [[ $# -gt 0 ]]; then
  EVAL_ARGS+=("$@")
fi

echo "[zero-sim] Using env '${ISAAC_ENV}' (no GR00T, no scene cameras)."
echo "[zero-sim] mode=zero max_steps=${MAX_STEPS} gui=${GUI} (set GUI=1 for Isaac window)"
cd "${ADAM_REPO}"
conda run -n "${ISAAC_ENV}" --no-capture-output python "${EVAL_ARGS[@]}"
