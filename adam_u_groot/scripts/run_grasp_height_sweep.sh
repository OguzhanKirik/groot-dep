#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/teleop/grasp_height_sweep}"
MAX_STEPS="${MAX_STEPS:-1800}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# Values are grasp-center height above cube center. For a 5 cm cube these
# correspond to 1.5, 2.0, ..., 4.0 cm above the top surface.
for entry in \
    "1.5cm 0.040" \
    "2.0cm 0.045" \
    "2.5cm 0.050" \
    "3.0cm 0.055" \
    "3.5cm 0.060" \
    "4.0cm 0.065"; do
    read -r label clearance <<<"${entry}"
    echo "[SWEEP] Starting ${label} above cube surface (center clearance=${clearance} m)"
    python adam_u_groot/scripts/teleop_record_adam_u.py \
        --gui \
        --scripted-grasp \
        --scripted-exit-after-attempt \
        --no-real-time \
        --max-steps "${MAX_STEPS}" \
        --scripted-grasp-clearance "${clearance}" \
        --scripted-seed 42 \
        --scripted-rotation-tolerance 0.35 \
        --joint-target-smoothing 0.25 \
        --max-joint-target-step 0.006 \
        --output "${OUTPUT_DIR}/grasp_${label}.hdf5" \
        2>&1 | tee "${OUTPUT_DIR}/grasp_${label}.log"
done

echo "[SWEEP] Finished. Results: ${OUTPUT_DIR}"
