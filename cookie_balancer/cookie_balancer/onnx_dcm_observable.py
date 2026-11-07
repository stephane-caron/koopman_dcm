# SPDX-License-Identifier: Apache-2.0

"""Learned DCM evaluated online from the observation-action history."""

import math
import pathlib
from collections import deque

import numpy as np
import onnxruntime as ort


class OnnxDcmObservable:
    """Evaluate an ONNX DCM observable on the observation-action history.

    Attributes:
        dcm_line: Latest estimated DCM line coordinate, in [-1, 1].
        omega: Natural frequency learned during training, in rad/s.
    """

    dcm_line: float
    omega: float

    def __init__(self, model_path: pathlib.Path) -> None:
        """Load the ONNX DCM observable and read config from model metadata.

        Args:
            model_path: Path to the saved ``.onnx`` model file.
        """
        # CPUExecutionProvider matches the real-robot target (aarch64, no CUDA)
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        props = self._session.get_modelmeta().custom_metadata_map
        self._dt: float = float(props["dt"])
        self._hist_size: int = int(props["hist_size"])
        self._obs_dim: int = int(props["obs_dim"])
        self.omega: float = float(props["omega"])

        self._y_buf: deque[np.ndarray] = deque(maxlen=self._hist_size)
        self._u_buf: deque[float] = deque(maxlen=self._hist_size)
        self.dcm_line: float = 0.0

    def reset(self) -> None:
        """Clear history buffer and reset estimated DCM."""
        self._y_buf.clear()
        self._u_buf.clear()
        self.dcm_line = 0.0

    def cycle(
        self,
        dt: float,
        q_actuated: np.ndarray,
        com_anchor: np.ndarray,
        com_line: float,
        action: float,
        skip_inference: bool,
    ) -> None:
        """Update DCM estimate from the latest observation and action.

        Args:
            dt: Time step in seconds.
            q_actuated: Actuated joint angles, in rad.
            com_anchor: Center-of-mass position in the anchor frame, in m.
            com_line: Normalized lateral CoM coordinate in [-1, 1].
            action: Latest ZMP action scalar.
            skip_inference: If set, only populate the history buffer and skip
                inference of the DCM observable.
        """
        y_k = np.concatenate(
            [
                np.asarray(q_actuated, dtype=np.float32),
                np.asarray(com_anchor, dtype=np.float32),
                np.array([com_line], dtype=np.float32),
            ]
        )

        if not skip_inference and len(self._y_buf) == self._hist_size:
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
