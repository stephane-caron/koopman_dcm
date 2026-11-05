# SPDX-License-Identifier: Apache-2.0

"""Global settings for g1_evaluation."""

from pathlib import Path

# Recordings (collection datasets, evaluation results) go to data/
DATA_DIR: Path = Path(__file__).parent.parent / "data"

# Trained DCM observables (ONNX) go to models/
MODELS_DIR: Path = Path(__file__).parent.parent / "models"

CONTROL_FREQUENCY: float = 240.0  # Hz; matches PyBullet's default 1/240 s step
DT: float = 1.0 / CONTROL_FREQUENCY
GRAVITY: float = 9.81  # m/s^2

# Anchor low-pass cutoff (foot-midpoint smoothing). Hz.
ANCHOR_LPF_CUTOFF: float = 1.0

# Contact-force guard: hold last ZMP if total normal force below this.
MIN_TOTAL_FZ: float = 5.0  # N

# Fall detection (relative to initial base height).
FALL_HEIGHT_FRACTION: float = 0.5

# Sagittal stabilizer — out-of-scope axis. PD on world-x of the base link
# plus PD on yaw (so the body-x stays aligned with world-x, ensuring the
# sagittal force never bleeds into the lateral axis we study).
SAGITTAL_KP: float = 500.0  # N/m
SAGITTAL_KD: float = 200.0  # N·s/m
YAW_KP: float = 200.0  # N·m/rad
YAW_KD: float = 50.0  # N·m·s/rad

# Admittance (gain converting commanded ZMP error to leaning velocity),
# shared by all controllers.
ADMITTANCE_DEFAULT: float = 1.0
