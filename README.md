# adam_u_rl

## Overview

This repository provides a minimal example of loading the **adam_u robot** into [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and running simple RL training tasks.  
It is intended as a **starting point** for robot manipulation with Isaac Lab — both the RL algorithm and the environment design can (and should) be further extended.

<p align="center">
  <img src="docs/demo.gif" alt="Demo of adam_u grasping task" width="800"/>
</p>

## Features

- ✅ Load **adam_u robot** from URDF into Isaac Lab.  
- ✅ Simple RL environment for grasping using **RSL_RL**.  
- ✅ CLI arguments to select different environments.  

⚠️ **Note**: This repository is primarily a demonstration. The algorithms and environment design are simplified for clarity and should be re-designed for production research.

---

## Installation

Tested on:
- **Ubuntu 22.04**
- **Isaac Sim 5.0** (should also work with 4.0 / 4.5)

### Requirements

- [Isaac Lab (pip installation)](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)  
  Please follow the official documentation for installation and troubleshooting.

Clone this repo:
```bash
git clone https://gitlab.com/pndbotics/manipulation.git
cd adam_u_rl
```

---

## Quick Start

### Train

Navigate to the `manipulation` folder and run:

```bash
python adam_u_rl/scripts/train.py --headless
```

### Evaluate

To evaluate a trained policy:

```bash
python adam_u_rl/scripts/play.py   --checkpoint_path logs/rsl_rl/adam_u_grasp/2025-06-27_17-04-25/model_1499.pt
```

---

## GR00T N1.7 Evaluation (Adam-U)

This repo includes a minimal GR00T integration under `adam_u_groot/` for closed-loop testing in the existing grasp scene.

### What was added

- `AdamUGraspGrootEnvCfg` — same table/cube/Adam-U scene with **front + wrist RGB cameras**
- `GrootAdapter` — maps Isaac Lab obs/actions ↔ GR00T Policy API format
- `eval_groot.py` — eval script with `zero`, `random`, and `groot` modes

### Step 1: Validate the env (no GR00T required)

From the repo root, with your **Isaac Lab conda/env activated**:

**With GUI (recommended for first run):**

```bash
python adam_u_groot/scripts/eval_groot.py --mode zero --enable_cameras --gui --real-time
```

Isaac Lab 6 defaults to **headless** unless you pass `--gui` or `--viz kit`. Do **not** rely on omitting `--headless` to get a window.

**Headless (no window):**

```bash
python adam_u_groot/scripts/eval_groot.py --mode zero --enable_cameras --headless
```

Use `--mode random` to sanity-check action wiring.

### Step 2: Run GR00T policy (one command)

Use the unified conda env **`adam-u-groot-unified`** (Isaac Sim + GR00T):

```bash
conda activate adam-u-groot-unified

GUI=1 MAX_STEPS=1000 bash adam_u_groot/scripts/run_groot_pipeline.sh \
  --groot-schema real_g1 \
  --task "pick up the cube and place it on the green target"
```

This runs Isaac Sim first, then loads GR00T **in the same process** (`--groot-inprocess`) so PhysX is not fighting a background server on the same GPU.

To use the legacy two-process ZMQ server instead: `INPROCESS=0 bash adam_u_groot/scripts/run_groot_pipeline.sh ...` (may crash if server and Isaac share one GPU).

Stop stale servers and Isaac evals before each run:

```bash
bash adam_u_groot/scripts/cleanup_groot_stack.sh --status   # inspect
bash adam_u_groot/scripts/cleanup_groot_stack.sh --wait     # stop + wait for port 5555
# alias:
bash adam_u_groot/scripts/stop_groot_stack.sh
```

`run_groot_pipeline.sh` runs cleanup automatically before launch.

**A) Fast pipeline test (base model, REAL_G1 shim)** — motion is not meaningful; use `--groot-schema real_g1`.

**B) Real Adam-U control (after finetune on NEW_EMBODIMENT)** — use `--groot-schema adam_u` and a finetuned checkpoint (`EMBODIMENT_TAG=NEW_EMBODIMENT MODEL_PATH=/path/to/finetuned ...`).

<details>
<summary>Manual two-process setup (optional)</summary>

```bash
conda activate adam-u-groot-unified

# Terminal 1
python ~/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
  --model-path /home/revel/models/GR00T-N1.7-3B \
  --embodiment-tag REAL_G1 \
  --device cuda:0

# Terminal 2 (stop server first if sharing one GPU with Isaac)
python adam_u_groot/scripts/eval_groot.py \
  --mode groot \
  --groot-schema real_g1 \
  --enable_cameras \
  --gui

# Or in-process (recommended, single terminal):
python adam_u_groot/scripts/eval_groot.py \
  --mode groot \
  --groot-inprocess \
  --groot-model-path /home/revel/models/GR00T-N1.7-3B \
  --groot-schema real_g1 \
  --enable_cameras \
  --gui
```

</details>

Finetune (same env):

```bash
conda activate adam-u-groot-unified
cd ~/Isaac-GR00T
python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/revel/models/GR00T-N1.7-3B \
  --dataset-path /path/to/adam_u_demos \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /home/revel/adam/adam_u_isaac_lab/adam_u_groot/examples/adam_u_modality_register.py
```

> **Note:** The base `nvidia/GR00T-N1.7-3B` checkpoint only supports REAL_G1 (and other built-in tags), not Adam-U directly. Use `--groot-schema real_g1` for wiring tests, or finetune for real behavior. See `adam_u_groot/configs/groot_schemas.py`.

---

# Tutorial: Adam-U Grasping Environment

This tutorial shows how to set up and run a simple **Adam-U grasping task** in Isaac Lab.

---

### 1. Scene Setup

The scene includes:
- A **table** (cuboid top + cylindrical leg).  
- A **target object** (5cm cube) placed on the table.  
- The **Adam-U robot** loaded from URDF with predefined initial joint positions.  
- **Ground plane** and **dome light** for physics and visualization.  

Example (table + cube object):
```python
table_top = sim_utils.CuboidCfg(size=(0.6, 0.5, 0.05))
object = sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05))
robot = sim_utils.UrdfFileCfg(asset_path="assets/robots/adam_u/urdf/adam_u.urdf")
```

---

### 2. Action Space

The robot is controlled through **7 right-arm joints**:
- shoulderPitch_Right, shoulderRoll_Right, shoulderYaw_Right, elbow_Right, wristYaw_Right, wristPitch_Right, wristRoll_Right  

Example:
```python
actions = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[...7 right arm joints...],
    scale=1.0
)
```

---

### 3. Observation Space

The policy receives observations including:
- Right arm joint positions & velocities  
- Right hand position & orientation  
- Finger joint positions & velocities  
- Object 3D position  
- Last actions & robot base position  

Example:
```python
right_arm_pos = ObsTerm(func=mdp.joint_pos, params={...})
object_position = ObsTerm(func=mdp.root_pos_w, params={...})
```

---

### 4. Reward Function

Rewards encourage:
- Staying alive  
- Approaching and lifting the object  
- Smooth actions and low joint velocity  

Example:
```python
distance_to_object = RewTerm(func=compute_distance_reward, weight=2.0)
object_height = RewTerm(func=compute_height_reward, weight=10.0)
```

---

