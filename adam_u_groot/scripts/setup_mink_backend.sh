#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/third_party/pnd_models"

python - <<'PY'
import mujoco
if mujoco.__version__ != "3.8.0":
    raise SystemExit(
        f"Refusing to modify the Isaac environment: expected mujoco 3.8.0, "
        f"found {mujoco.__version__}"
    )
PY

# --no-deps is deliberate: Mink 1.2 would upgrade Isaac Sim's pinned MuJoCo.
python -m pip install --no-deps "mink==1.1.0"

if [[ ! -d "${MODEL_DIR}/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/pndbotics/pnd_models.git "${MODEL_DIR}"
    git -C "${MODEL_DIR}" sparse-checkout set adam_u
fi

python - <<'PY'
import mink  # noqa: F401
import mujoco
assert mujoco.__version__ == "3.8.0", mujoco.__version__
print("Mink backend ready; Isaac MuJoCo pin remains", mujoco.__version__)
PY
echo "Model: ${MODEL_DIR}/adam_u/adam_u.xml"
