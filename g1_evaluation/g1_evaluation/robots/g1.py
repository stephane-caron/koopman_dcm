# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 (29-DoF) robot spec.

Probed from ``g1_description`` (robot_descriptions). G1 has a narrow
stance — half foot spacing ≈ 12 cm — which makes the lateral balance
demonstration sensitive. Leg ordering is hip_pitch → hip_roll →
hip_yaw → knee → ankle_pitch → ankle_roll.

Geometry probed at the neutral configuration:

- 29 actuated joints (6 per leg + 3 waist + 7 per arm).
- Foot frames: ``left_ankle_roll_link`` / ``right_ankle_roll_link``
  (BODY frames). The G1 URDF uses 4 sphere-shaped collision elements
  per foot rather than a single box.
- Approximate sphere footprint per foot: sagittal [-0.05, +0.12] m,
  lateral [-0.03, +0.03] m, sole z ≈ -0.03 m from the ankle link.
- Foot spacing at neutral q: ≈ 0.237 m, so half_foot_spacing ≈ 0.119 m.
- CoM height above feet at neutral q: ≈ 0.686 m.
- Initial base_z so the sole rests on z=0: ≈ 0.79 m.
"""

from .base import RobotSpec

_ACTUATED = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


_STANDING_POSTURE = {
    # Legs: slight crouch.
    "left_hip_pitch_joint": -0.30,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.60,
    "left_ankle_pitch_joint": -0.30,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.30,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.60,
    "right_ankle_pitch_joint": -0.30,
    "right_ankle_roll_joint": 0.0,
    # Waist: keep upright.
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # Arms: slight outward to avoid self-collision.
    "left_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.20,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.40,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.40,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


# PyBullet POSITION_CONTROL blending factors (see RobotSpec docstring).
_KP = {n: 0.5 for n in _ACTUATED}
_KD = {n: 1.0 for n in _ACTUATED}
# Slightly stiffer ankles (CoP authority along the foot).
for n in (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
):
    _KP[n] = 0.8


# G1 motor torque limits — conservative figures by joint type; refine
# per published Unitree specs if needed.
_MAX_TORQUE = {n: 80.0 for n in _ACTUATED}
for n in ("left_knee_joint", "right_knee_joint"):
    _MAX_TORQUE[n] = 200.0
for n in (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
):
    _MAX_TORQUE[n] = 150.0
for n in (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
):
    _MAX_TORQUE[n] = 20.0


G1_SPEC = RobotSpec(
    name="g1",
    description="g1_description",
    actuated_joint_names=_ACTUATED,
    left_foot_frame="left_ankle_roll_link",
    right_foot_frame="right_ankle_roll_link",
    initial_base_z=0.80,
    com_height=0.70,
    half_foot_spacing=0.119,
    foot_half_width=0.03,
    foot_half_length=0.085,
    standing_posture=_STANDING_POSTURE,
    kp=_KP,
    kd=_KD,
    max_torque=_MAX_TORQUE,
    # Sole sits ~3 cm below the ankle_roll_link origin (sphere contacts).
    transform_sole_to_frame=(0.035, 0.0, -0.030),
    posture_task_joint_names=[
        # Lock waist + arms in the IK posture task; legs are constrained
        # by the foot frame tasks.
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)
