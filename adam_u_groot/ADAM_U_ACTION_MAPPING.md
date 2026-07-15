# REAL_G1 to Adam-U action mapping

## Native GR00T embodiment contract

The Adam-U `NEW_EMBODIMENT` configuration consumes two independent RGB streams,
`front` and `wrist`, stored as `observation.images.<name>`. State and action use
the same groups: `waist` (3), `neck` (2), `left_arm` (7), `right_arm` (7),
`left_hand` (6), and `right_hand` (6). This is body[19] plus hands[12].

Native Adam-U actions are absolute targets: radians for body joints and
calibrated synergy units for hands. Training data, dataset statistics,
checkpoint processor configuration, and inference must retain this exact
layout. `examples/adam_u_modality_register.py` is the authoritative GR00T
registration and `configs/modality.json` defines matching LeRobot ranges.

For REAL_G1 compatibility, Adam-U arm angles are never inserted directly into
the G1 state fields. `VirtualG1StateAdapter` uses the official Unitree G1 URDF
and Pinocchio IK to find continuous virtual G1 arm angles whose wrist positions
match the transformed Adam-U wrist positions. Each solve is seeded from the
previous virtual state to avoid elbow flips. Thus the G1 joint and EEF states
fed to GR00T describe the same virtual configuration.

Virtual G1 IK also supplies the wrist rot6d from G1 FK; it does not combine G1
joints with Adam-U tool orientation. A null-space bias toward the checkpoint's
mean arm posture avoids valid-but-out-of-distribution elbow/shoulder solutions.

This integration treats the GR00T action dictionary as named groups. It never
concatenates outputs based only on matching dimensions.

## Verified source configuration

The installed `GR00T-N1.7-3B/processor_config.json` declares:

- `left_arm`, `right_arm`: relative, non-EEF, seven values each
- `left_wrist_eef_9d`, `right_wrist_eef_9d`: relative XYZ + rot6d EEF actions
- `left_hand`, `right_hand`: absolute, seven values each
- `waist`, `base_height_command`, `navigate_command`: absolute

`PolicyClient` postprocessing reconstructs arm outputs using the supplied state,
so the integration defaults to treating received arm vectors as absolute. Use
`--raw-relative-arm-actions` only with raw, non-postprocessed model outputs.

## Adam-U controller order

The body command is always 19 values in this order:

1. `waistRoll`, `waistPitch`, `waistYaw`
2. `neckYaw`, `neckPitch`
3. left shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll
4. right shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll

These names, axes, radian units, and limits come from the Adam-U URDF. Source
signs and zero offsets are explicit configuration fields. They currently default
to identity because no measured REAL_G1-to-Adam-U hardware calibration is in
this repository. Real deployment remains locked until
`calibration_verified_for_real_robot=True` is set deliberately.

The hand command is 12 values: six left-hand synergies followed by six right:

1. thumb opposition
2. thumb flexion
3. index flexion
4. middle flexion
5. ring flexion
6. pinky flexion

Checkpoint ranges show G1 channels 0–3 as mirrored index/middle/ring/pinky
flexion, channel 4 as thumb opposition, and channels 5–6 as two thumb-flexion
components. The default configurable 6x7 matrix therefore:

- maps channel 4 to thumb opposition;
- averages channels 5 and 6 into Adam-U thumb flexion, with the observed side sign;
- maps channels 0–3 by finger meaning, correcting the observed left/right sign.

Nothing is dropped arbitrarily. The two G1 thumb-flexion DOFs are combined
because Adam-U's low-level interface exposes one thumb-flexion synergy. Isaac's
12 physical joints per hand are a separate layer: the six synergies are expanded
by name into thumb and MCP/DIP targets.

## Exclusive arm modes

- `joint_space`: consumes `left_arm` and `right_arm`; wrist EEF actions are logged
  as ignored.
- `eef_space`: consumes only the wrist EEF actions. Adam world poses are mapped
  to the REAL_G1 canonical workspace as `[-x, -y, z - 1]` before policy input,
  and returned absolute targets are mapped back before Isaac differential IK.
  IK currently controls position only because the G1-to-Adam wrist/tool-frame
  rotation has not been calibrated. Persistent joint targets resist gravity
  sag. The adapter refuses to start without IK and never falls back to joints.
- `g1_real`: simulation-only diagnostic that reproduces the pre-adapter path:
  only `right_arm[7]` is applied and Isaac intentionally adds Adam-U's default
  right-arm pose. It ignores all other outputs and must never be used on real
  hardware.

Navigation and base height are always ignored because this integration controls
a fixed upper body. Neck joints hold the configurable neutral pose.

Waist ordering is explicitly permuted in both directions: REAL_G1 uses
yaw/roll/pitch, while Adam-U uses roll/pitch/yaw. Persistent IK integrator state
is synchronized to the final clamped and execution-masked target to prevent
windup at joint limits.

The default execution scope is `right_arm_hand`: GR00T still observes the full
REAL_G1-compatible state, but only the right EEF/right arm and right hand are
executed. Waist, neck, left arm, and left hand are latched after reset. Use
`--execution-scope full` only for deliberate bimanual experiments.

EEF differential IK has its own `--eef-ik-joint-step` limiter, defaulting to
0.01 rad per arm joint per simulation step. It is deliberately separate from
the broader body/hand adapter limit `--max-position-step`.

## Safety

Every command is checked for dimensions, finite values, names/order, URDF body
limits, hand-synergy limits, and maximum per-step change. Optional velocity and
exponential smoothing limits are applied before output. Invalid data either
raises immediately (default) or holds the previous safe command when explicitly
configured. Logging records source groups, mapped body/hands, ignored groups,
and every clamp/limit event.
