# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained DCM observable, exported to ONNX, on the robot."""

import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Sanity range for the learned ω. With insufficient frequency variation in
# the training data, training can collapse to a degenerate minimum at
# ξ ≈ c + ċ/ω̂ with ω̂ ≈ 0 (the trivial CoM tracker).
OMEGA_MIN: float = 1.5  # rad/s
OMEGA_MAX: float = 8.0  # rad/s

logger = logging.getLogger("g1_evaluation.onnx_dcm_observable")


class OnnxDCMObservable:
    """Learned DCM from the delay embedding of observations and actions.

    Attributes:
        dcm_line: Latest learned DCM, as a normalized line coordinate.
        hist_size: Delay-embedding horizon H, in control steps.
        obs_dim: Dimension of the observation in the delay embedding.
        omega: Natural frequency learned during training, in rad/s.
    """

    dcm_line: float
    hist_size: int
    obs_dim: int
    omega: float

    def __init__(self, model_path: Path) -> None:
        """Load the ONNX model and read its configuration from metadata.

        Args:
            model_path: Path to the ``.onnx`` model file.
        """
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        props = self._session.get_modelmeta().custom_metadata_map
        self._dt: float = float(props["dt"])
        self.hist_size = int(props["hist_size"])
        self.obs_dim = int(props["obs_dim"])
        self.omega = float(props["omega"])

        self._y_buf: deque[np.ndarray] = deque(maxlen=self.hist_size)
        self._u_buf: deque[float] = deque(maxlen=self.hist_size)
        self.dcm_line = 0.0

    def cycle(
        self,
        dt: float,
        q_actuated: np.ndarray,
        com_anchor: np.ndarray,
        com_line: float,
        action: float,
    ) -> None:
        """Update the learned DCM from the latest observation and action.

        Args:
            dt: Time step in seconds.
            q_actuated: Actuated joint angles, in rad.
            com_anchor: Center-of-mass position in the anchor frame, in m.
            com_line: Normalized lateral CoM coordinate in [-1, 1].
            action: Latest ZMP action scalar.
        """
        y_k = np.concatenate(
            [
                np.asarray(q_actuated, dtype=np.float32),
                np.asarray(com_anchor, dtype=np.float32),
                np.array([com_line], dtype=np.float32),
            ]
        )

        if len(self._y_buf) == self.hist_size:
            if not math.isclose(dt, self._dt, rel_tol=1e-2):
                raise ValueError(
                    f"dt mismatch: cycle called with dt={dt} "
                    f"but model was trained with dt={self._dt}"
                )
            parts: list[np.ndarray] = []
            for y_past, u_past in zip(self._y_buf, self._u_buf):
                parts.append(y_past)
                parts.append(np.array([u_past], dtype=np.float32))
            parts.append(y_k)
            x = np.concatenate(parts)[np.newaxis, :]  # (1, input_dim)
            output = self._session.run(["dcm"], {"history": x})
            self.dcm_line = float(output[0][0])

        self._y_buf.append(y_k)
        self._u_buf.append(float(action))


def load_dcm_observable(path: Path) -> OnnxDCMObservable:
    """Load a trained DCM observable and check its learned ω.

    Args:
        path: ONNX file to load.

    Returns:
        The loaded DCM observable.

    Raises:
        SystemExit: if ω is outside ``[OMEGA_MIN, OMEGA_MAX]``.
    """
    observable = OnnxDCMObservable(path)
    logger.info(
        "Loaded DCM observable from %s: obs_dim=%d hist_size=%d omega=%.3f rad/s",
        path,
        observable.obs_dim,
        observable.hist_size,
        observable.omega,
    )
    if not (OMEGA_MIN <= observable.omega <= OMEGA_MAX):
        raise SystemExit(
            f"Learned omega = {observable.omega:.3f} rad/s is outside the "
            f"sanity range [{OMEGA_MIN:.1f}, {OMEGA_MAX:.1f}]. The training "
            "likely collapsed to a degenerate minimum (ξ ≈ c + ċ/ω̂). Retrain "
            "with a better OMEGA_INIT in train_dcm.py."
        )
    return observable
