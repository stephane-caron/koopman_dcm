# SPDX-License-Identifier: Apache-2.0

"""Project-wide settings."""

from pathlib import Path

# Recordings (collection datasets) go to data/
DATA_DIR: Path = Path(__file__).parent.parent / "data"

# Evaluation results go to evaluation/data/, alongside the real-robot
# session logs, keeping them out of the recording directory.
EVALUATION_DATA_DIR: Path = (
    Path(__file__).parent.parent / "evaluation" / "data"
)

# Learned DCMs go to models/
MODELS_DIR: Path = Path(__file__).parent.parent / "models"

# Control loop frequency, in Hz.
CONTROL_FREQUENCY: float = 100.0

# Natural frequency of the learned-LIP ablation, in rad/s. Fitted to the
# cookie.h5 dataset by scripts/estimate_omega_from_dataset.py, which
# holds the observable at the analytic form ξ = c + ċ/ω and optimizes ω
# alone under the same multi-step losses as the learned DCM. Mean over five
# seeds (std 0.02 rad/s). Compare to the kinematic √(g/h) ≈ 5.0 rad/s.
LEARNED_LIP_OMEGA: float = 6.29

# Fall detection
FALL_ANGLE: float = 1.0  # rad

# DCM controllers: admittance mapping the commanded-ZMP error to a leaning
# velocity. Shared by DcmFeedbackController (slider initial value) and DcmMpc
# (fallback before an interface is registered).
ADMITTANCE_DEFAULT: float = 0.4

# Leg controller
MAX_CROUCH_HEIGHT: float = 0.12  # m
MAX_HEIGHT_DIFFERENCE: float = 0.1  # m
DEFAULT_CROUCH_HEIGHT: float = 0.02  # m
MAX_INIT_JOINT_VELOCITY: float = 0.1  # rad/s
MAX_LEANING_VELOCITY: float = 0.1  # m/s
MAX_LEG_TORQUE: float = 5.0  # N·m
LEG_KP_SCALE: float = 1.0
LEG_KD_SCALE: float = 2.0

# Wheel controller
MPC_LEG_LENGTH: float = 0.58  # m
MAX_GROUND_ACCEL: float = 10.0  # m/s²
MAX_GROUND_VELOCITY: float = 3.0  # m/s
MAX_WHEEL_TORQUE: float = 0.8  # N·m
