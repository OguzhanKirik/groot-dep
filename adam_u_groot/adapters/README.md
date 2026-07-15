# Adam-U GR00T adapters

This directory bridges the NVIDIA GR00T `REAL_G1` checkpoint to PNDbotics
Adam-U. It provides safe command conversion and task-space retargeting; it does
not make the G1 checkpoint a native Adam-U policy.

## Data flow

```text
Isaac RGB + Adam-U state → GR00T REAL_G1 → 53-value grouped output
                                             ↓
                           joint-space or EEF-space adapter
                                             ↓
                         Adam-U body[19] + hands[12]
                                             ↓
                         Isaac body[19] + finger joints[24]
```

GR00T's 53 outputs contain arm joints (14), waist (3), G1 hands (14), wrist EEF
poses (18), base height (1), and navigation (3). Adam-U uses 31 low-level values:

```text
body[19] = waist[3] + neck[2] + left arm[7] + right arm[7]
hands[12] = left hand[6] + right hand[6]
```

Base height and navigation are ignored. Neck is held at the configured neutral
pose. Arm-joint and wrist-EEF outputs are never applied simultaneously.

## Joint adapter

`adam_u_action_adapter.py` implements the `joint_space` path:

- maps named G1 waist and seven-joint arm groups into Adam-U order;
- converts each G1 hand[7] into semantic Adam-U hand[6] synergies;
- combines the two G1 thumb-flexion channels instead of dropping one;
- applies configurable signs, zero offsets, units, limits, step limits, velocity
  limits, and smoothing;
- rejects NaN/Inf and invalid dimensions;
- preserves exact Isaac joint order with `preserve_order=True`.

Matching joint counts do not imply matching kinematics. G1 and Adam-U have
different link lengths, joint frames, zero poses, and wrist mounting geometry.

## Workspace / EEF adapter

`eef_pose.py` and Isaac Lab's differential IK controller implement `eef_space`:

1. consume `left_wrist_eef_9d` and `right_wrist_eef_9d`;
2. convert G1-canonical targets to Adam-U world coordinates;
3. ignore GR00T arm-joint outputs;
4. read Adam-U wrist poses and 6x7 Jacobians from Isaac;
5. solve position differential IK using Adam-U's geometry;
6. return persistent, limited seven-joint targets for each arm.

`Gr00tPolicy` already decodes its internally relative EEF prediction to an
absolute target using the supplied wrist state. The adapter therefore treats
returned EEF values as absolute and does not add the current pose twice.

The current deterministic workspace is defined in
`adam_u_rl/envs/scene_layout.py`:

```text
table center: (-0.50, 0.00, 0.85) m
cube center:  (-0.40, 0.00, 0.90) m
```

Both arms start raised, fingers open, neck pitched toward the table, and reset
joint randomization is disabled for repeatable evaluation.

## Virtual G1 observation state

`virtual_g1_state.py` prevents raw Adam-U joint angles from entering REAL_G1
state fields. On every inference it solves:

1. measured Adam-U joints -> Adam-U wrist FK;
2. Adam world wrist -> G1 canonical workspace;
3. G1 wrist target -> virtual G1 joints using the official G1 URDF;
4. previous virtual joints -> next IK seed for continuous solutions.

The seven values use G1 order: shoulder pitch/roll/yaw, elbow, and wrist
roll/pitch/yaw. The solver reports wrist consistency error and warns above 5 cm.

## Control modes

| Mode | Arm source | Purpose |
|---|---|---|
| `joint_space` | G1 arm joints | Safe semantic joint conversion |
| `eef_space` | G1 wrist EEF + Adam-U IK | Geometry-aware workspace retargeting |
| `g1_real` | G1 right arm only + old default offset | Simulation-only legacy comparison |
| `scripted_reach` | Cube position + Adam-U IK | GR00T-free IK/workspace diagnostic |

Run the normal base model with joint conversion:

```bash
GUI=1 MAX_STEPS=1000 bash adam_u_groot/scripts/run_groot_pipeline.sh \
  --groot-schema real_g1 --control-mode joint_space
```

Run EEF retargeting:

```bash
GUI=1 MAX_STEPS=1000 bash adam_u_groot/scripts/run_groot_pipeline.sh \
  --groot-schema real_g1 --control-mode eef_space
```

Run the GR00T-free right-arm diagnostic:

```bash
conda activate adam-u-isaac-6
MAX_STEPS=500 REACH_JOINT_STEP=0.005 \
  bash adam_u_groot/scripts/run_scripted_reach_cube.sh
```

`g1_real` intentionally reproduces the old double-offset behavior. It is unsafe
for hardware and must be used only for simulation A/B testing.

## Main files

- `groot_adapter.py`: observation packing and simulator command assembly.
- `adam_u_action_adapter.py`: mapping, validation, limits, smoothing, and IK boundary.
- `eef_pose.py`: wrist pose conversion and Isaac FK/Jacobian provider.
- `joint_state_reader.py`: named Adam-U joint-state access.
- `../configs/adam_u_action_mapping.py`: calibration and safety configuration.
- `../ADAM_U_ACTION_MAPPING.md`: detailed mapping rationale.
