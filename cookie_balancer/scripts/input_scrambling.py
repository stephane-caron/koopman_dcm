# SPDX-License-Identifier: Apache-2.0

"""Measure whether the learned DCM reads robot state or its own actions.

The DCM observable consumes a delay embedding
``z_k = (y_{k-h}, u_{k-h}, ..., y_{k-1}, u_{k-1}, y_k)`` that mixes
measurements ``y`` with past actions ``u``. Since training and evaluation
references come from the same family of sinusoids, a DCM observable could
in principle infer the phase of the reference from its own action history,
rather than from the state of the robot.

To check this (Section 4.5 of the paper), this script replaces a group of
inputs, either the past actions or the measurements, by values recorded at
unrelated times: it permutes the group across rows under a single shared
permutation, which preserves its distribution and destroys only its
correlation with the rest of the embedding. It then reports the change in
output ``RMS(Δξ)``, in units of the standard deviation ``σ(ξ)`` of the
output. If scrambling past actions moves the output much less than
scrambling measurements, the DCM observable relies mostly on robot state.

``IS_TRAINING_DATASET`` below decides whether a split is applied:

- True: the DCM observable was fit on ``DATASET``, so the reported rows
  must be its held-out validation split.
- False: ``DATASET`` was recorded after training and held out in full, so
  every valid row is evaluated and no split is applied.

Usage::

    pixi run python scripts/input_scrambling.py
"""

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import onnx
import onnxruntime as ort

from cookie_balancer import settings
from cookie_balancer.train_dcm import (
    build_histories,
    build_valid_segment_starts,
    split_indices,
)

# DCM observable to analyze
MODEL: Path = settings.MODELS_DIR / "dcm-cookie.onnx"

# Dataset of the paper's headline numbers (Section 4.5): out-of-distribution
# periods, recorded after training. For the validation-split numbers, set
# DATASET to the training dataset ``cookie.h5`` and IS_TRAINING_DATASET to
# True.
DATASET: Path = settings.DATA_DIR / "cookie-ood.h5"
IS_TRAINING_DATASET: bool = False

VAL_FRAC: float = 0.1  # validation fraction used at training
SEED: int = 42  # seed for the split and the permutations

# Batch size for inference of the DCM observable
BATCH_SIZE = 4096


def load_dcm_observable(model_path: Path) -> Tuple[ort.InferenceSession, Dict]:
    """Load the ONNX DCM observable with a batch dimension made dynamic.

    The exported graph fixes the batch dimension to 1, which would force
    one inference call per row. Relaxing the dimension to a symbolic name
    lets onnxruntime batch; this is bit-exact against the batch-1 graph
    for the Gemm/Tanh stack the DCM observable uses.

    Args:
        model_path: Path to the ``.onnx`` file of the DCM observable.

    Returns:
        Pair of the inference session and the model metadata dictionary.
    """
    model = onnx.load(str(model_path))
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"
    # The exported graph still declares a length-1 output, so onnxruntime
    # warns on every batched call. The values are unaffected.
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    props = dict(session.get_modelmeta().custom_metadata_map)
    return session, props


def input_groups(hist_size: int, obs_dim: int) -> Dict[str, np.ndarray]:
    """Column indices of the past actions and of the measurements.

    Row layout is ``h`` blocks of ``(y, u)`` followed by the current
    ``y_k``, so column ``i * (obs_dim + 1) + obs_dim`` holds the action
    of block ``i`` and all other columns hold measurements.

    Args:
        hist_size: Number of past ``(y, u)`` pairs in the embedding.
        obs_dim: Dimension of the observation vector of the DCM observable.

    Returns:
        Mapping from group name to the column indices it covers.
    """
    block = obs_dim + 1
    actions = np.array(
        [i * block + obs_dim for i in range(hist_size)], dtype=np.int64
    )
    past_measurements = np.array(
        [i * block + d for i in range(hist_size) for d in range(obs_dim)],
        dtype=np.int64,
    )
    current_measurement = np.arange(
        hist_size * block, hist_size * block + obs_dim, dtype=np.int64
    )
    return {
        "past actions": actions,
        "measurements": np.concatenate(
            [past_measurements, current_measurement]
        ),
    }


def run_dcm_observable(
    session: ort.InferenceSession, rows: np.ndarray
) -> np.ndarray:
    """Evaluate the DCM observable on a batch of delay-embedding rows.

    Args:
        session: Inference session of the DCM observable.
        rows: Array of shape ``(n, input_dim)``.

    Returns:
        Outputs of the DCM observable, of shape ``(n,)``.
    """
    out = np.empty(rows.shape[0], dtype=np.float64)
    for start in range(0, rows.shape[0], BATCH_SIZE):
        chunk = np.ascontiguousarray(
            rows[start : start + BATCH_SIZE], dtype=np.float32
        )
        out[start : start + chunk.shape[0]] = session.run(
            ["dcm"], {"history": chunk}
        )[0].reshape(-1)
    return out


def scramble(
    rows: np.ndarray, columns: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace a group of columns by their values at unrelated times.

    One permutation is applied to the whole group, so that the group stays
    internally consistent (a real history from another time) and only its
    link to the rest of the row is broken.

    Args:
        rows: Delay-embedding rows of shape ``(n, input_dim)``.
        columns: Column indices of the group.
        rng: Random generator of the permutation.

    Returns:
        Scrambled copy of ``rows``.
    """
    out = rows.copy()
    perm = rng.permutation(rows.shape[0])
    out[:, columns] = rows[np.ix_(perm, columns)]
    return out


def main() -> None:
    """Scramble each input group and report the change in output."""
    rng = np.random.default_rng(SEED)
    session, props = load_dcm_observable(MODEL)
    hist_size = int(props["hist_size"])
    obs_dim = int(props["obs_dim"])

    with h5py.File(DATASET, "r") as hf:
        Y = hf["Y"][:]
        U = hf["U"][:]
        terminated = hf["terminated"][:]
        truncated = hf["truncated"][:]

    histories = build_histories(Y[:, :obs_dim], U, hist_size)

    # Rows whose history window crosses no episode boundary. The helper
    # requires a horizon of at least one step, so ask for a length-1
    # segment and drop the final row it reserves for the successor.
    valid = build_valid_segment_starts(
        histories.shape[0], hist_size, 1, terminated, truncated
    )
    if IS_TRAINING_DATASET:
        _, val_idx = split_indices(valid.shape[0], VAL_FRAC, SEED)
        valid = valid[val_idx]
    valid = np.sort(valid)
    scope = "validation split" if IS_TRAINING_DATASET else "held out in full"
    print(f"Dataset {DATASET}: {valid.shape[0]} embedding rows ({scope})")
    print(f"DCM observable {MODEL}: ω̂ = {float(props['omega']):.4f} rad/s")

    rows = histories[valid]
    baseline = run_dcm_observable(session, rows)
    sigma = float(baseline.std())

    print()
    print("| Scrambled inputs | RMS(Δξ) | σ(ξ) | RMS(Δξ) / σ(ξ) |")
    print("|---|---:|---:|---:|")
    for name, columns in input_groups(hist_size, obs_dim).items():
        scrambled = run_dcm_observable(session, scramble(rows, columns, rng))
        rms = float(np.sqrt(np.mean((scrambled - baseline) ** 2)))
        print(f"| {name} | {rms:.4f} | {sigma:.4f} | {rms / sigma:.3f} |")


if __name__ == "__main__":
    main()
