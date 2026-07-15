#!/usr/bin/env bash
# GR00T-free right-arm reach diagnostic for Adam-U in Isaac Lab.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python adam_u_groot/scripts/eval_groot.py \
  --mode scripted_reach \
  --gui \
  --max_steps "${MAX_STEPS:-500}" \
  --reach-hover-height "${REACH_HOVER_HEIGHT:-0.12}" \
  --reach-joint-step "${REACH_JOINT_STEP:-0.03}" \
  "$@"
