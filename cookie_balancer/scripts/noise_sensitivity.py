# SPDX-License-Identifier: Apache-2.0

"""Measure how measurement noise on the CoM moves the learned DCM.

Companion to ``input_scrambling.py``, same protocol and same
reporting, but perturbing rather than scrambling: for every delay
embedding of the held-out set we add noise to the CoM channels, re-run
the DCM observable, and report how far the output moves.

Usage::

    pixi run python scripts/noise_sensitivity.py
"""

import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import onnx
import onnxruntime as ort
from scipy.signal import butter, sosfiltfilt

from cookie_balancer import settings
from cookie_balancer.observation import make_observation
from cookie_balancer.train_dcm import (
    build_histories,
    build_valid_segment_starts,
    split_indices,
)

# Batch size for inference of the DCM observable
BATCH_SIZE = 8192

# DCM observable to analyze
MODEL: Path = settings.MODELS_DIR / "dcm-cookie.onnx"

# Dataset of the paper (Appendix B.2): the validation split of the training
# dataset. Set IS_TRAINING_DATASET to False for a dataset recorded after
# training, which is then evaluated in full without a split.
DATASET: Path = settings.DATA_DIR / "cookie.h5"
IS_TRAINING_DATASET: bool = True

# Split of the training dataset, as in training
SPLIT: str = "val"  # side of the split to evaluate, "train" or "val"
VAL_FRAC: float = 0.1  # validation fraction used at training

# Synthetic white-noise magnitudes swept alongside the measured
# residual, in millimetres of lateral CoM displacement
LEVELS_MM: tuple[float, ...] = (1.0, 2.0, 5.0)

SAMPLES: int = 10  # noise draws per embedding row
CUTOFF: float = 4.0  # high-pass cutoff isolating the estimator residual, Hz
KP: float = 1.0  # proportional gain of the DCM feedback law

SEED: int = 42  # seed for the split and the draws
MAX_ROWS: int = 0  # subsample to at most this many rows, 0 for no limit

# Markdown report, written in addition to stdout if not None
OUTPUT: Path | None = None

# Discarded at the start of each stationary segment, while the robot is
# still travelling to the new reference offset
SETTLE_DURATION = 2.0  # s

# Shortest stationary segment kept, after settling
MIN_SEGMENT_STEPS = 400


def stationary_segments(
    amplitude: np.ndarray,
    offset: np.ndarray,
    period: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    dt: float,
) -> List[Tuple[int, int]]:
    """Locate the segments where the robot stands at a fixed reference.

    Data collection redraws ``(amplitude, offset, period)`` every chunk,
    so a segment is a maximal run of constant parameters that no episode
    boundary cuts. Those with zero amplitude hold the ZMP reference
    still, and their leading transient is dropped so that what remains is
    the robot holding a constant lean.

    Args:
        amplitude: Reference amplitude per step.
        offset: Reference offset per step.
        period: Reference period per step.
        terminated: Termination flag per step.
        truncated: Truncation flag per step.
        dt: Sampling period in seconds.

    Returns:
        List of ``(start, end)`` index pairs, settling already removed.
    """
    key = np.stack([amplitude, offset, period], axis=1)
    cut = (
        np.any(key[1:] != key[:-1], axis=1) | terminated[:-1] | truncated[:-1]
    )
    starts = np.concatenate([[0], np.flatnonzero(cut) + 1])
    ends = np.concatenate([np.flatnonzero(cut) + 1, [len(amplitude)]])
    settle = int(SETTLE_DURATION / dt)
    return [
        (start + settle, end)
        for start, end in zip(starts, ends)
        if amplitude[start] == 0.0
        and end - start - settle >= MIN_SEGMENT_STEPS
    ]


def load_dcm_observable(model_path: Path) -> Tuple[ort.InferenceSession, Dict]:
    """Load the ONNX DCM observable with a batch dimension made dynamic.

    Args:
        model_path: Path to the ``.onnx`` file of the DCM observable.

    Returns:
        Pair of the inference session and the model metadata dictionary.
    """
    model = onnx.load(str(model_path))
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session, dict(session.get_modelmeta().custom_metadata_map)


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


def com_columns(hist_size: int, obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Locate the CoM columns of a delay-embedding row.

    Row layout is ``hist_size`` blocks of ``(y, u)`` followed by the
    current ``y_k``, so each block starts at ``i * (obs_dim + 1)``.

    Args:
        hist_size: Number of past ``(y, u)`` pairs in the embedding.
        obs_dim: Dimension of the observation vector of the DCM observable.

    Returns:
        Pair of column-index arrays, one per time block: the lateral
        pair ``(com_anchor_y, com_line)`` stacked as ``(n_blocks, 2)``,
        and the other two ``com_anchor`` axes as ``(n_blocks, 2)``.
    """
    sv = make_observation()
    anchor = sv.slice_of("com_anchor").start
    line = sv.slice_of("com_line").start
    block = obs_dim + 1
    starts = [i * block for i in range(hist_size)] + [hist_size * block]
    lateral = np.array(
        [[start + anchor + 1, start + line] for start in starts],
        dtype=np.int64,
    )
    others = np.array(
        [[start + anchor, start + anchor + 2] for start in starts],
        dtype=np.int64,
    )
    return lateral, others


def metres_per_line_unit(Y: np.ndarray) -> float:
    """Regress the metric lateral CoM against its line coordinate.

    Args:
        Y: Observation array of shape ``(n, >= 10)``.

    Returns:
        Metres of lateral CoM travel per unit of ``com_line``.
    """
    sv = make_observation()
    com_line = Y[:, sv.slice_of("com_line").start]
    com_y = Y[:, sv.slice_of("com_anchor").start + 1]
    return float(np.polyfit(com_line, com_y, 1)[0])


def residual_windows(
    path: str, nb_blocks: int, cutoff: float, dt: float
) -> Tuple[np.ndarray, float]:
    """Collect windows of the residual recorded while standing still.

    On a zero-amplitude reference the commanded ZMP is constant, so the
    deviation of the CoM from its segment mean is noise plus the robot's
    own sway. High-passing above ``cutoff`` removes the sway, which is
    slow, and keeps the fast residual that the CoM cannot physically
    produce: what is left is the estimator's own error, with its true
    spectrum and its true correlation from one sample to the next.

    Args:
        path: HDF5 dataset holding the stationary segments.
        nb_blocks: Window length, in samples.
        cutoff: High-pass cutoff, in Hz.
        dt: Sampling period in seconds.

    Returns:
        Pair of the unit-variance windows, of shape
        ``(n_windows, nb_blocks)``, and the standard deviation they were
        normalized by, in metres.
    """
    sv = make_observation()
    with h5py.File(path, "r") as hf:
        com_y = hf["Y"][:, sv.slice_of("com_anchor").start + 1].astype(
            np.float64
        )
        segments = stationary_segments(
            hf["amplitude"][:],
            hf["offset"][:],
            hf["period"][:],
            hf["terminated"][:],
            hf["truncated"][:],
            dt,
        )
    sos = butter(4, cutoff, btype="highpass", fs=1.0 / dt, output="sos")
    windows: List[np.ndarray] = []
    residuals: List[np.ndarray] = []
    for start, end in segments:
        segment = com_y[start:end]
        if len(segment) <= 3 * nb_blocks:
            continue
        residual = sosfiltfilt(sos, segment - segment.mean())
        residuals.append(residual)
        # Every window the segment contains, as a strided view
        windows.append(
            np.lib.stride_tricks.sliding_window_view(residual, nb_blocks)
        )
    sigma = float(np.concatenate(residuals).std())
    return np.concatenate(windows) / sigma, sigma


def perturb(
    rows: np.ndarray,
    lateral: np.ndarray,
    others: np.ndarray,
    sigma_m: float,
    scale: float,
    rng: np.random.Generator,
    source: str,
    windows: np.ndarray,
) -> np.ndarray:
    """Return a copy of ``rows`` with Gaussian noise on the CoM.

    Args:
        rows: Delay-embedding rows of shape ``(n, input_dim)``.
        lateral: ``(n_blocks, 2)`` columns of the lateral CoM pair.
        others: ``(n_blocks, 2)`` columns of the other CoM axes.
        sigma_m: Noise standard deviation, in metres.
        scale: Metres of lateral travel per line unit.
        rng: Random generator.
        source: Either ``measured`` or ``white``.
        windows: Residual windows of shape ``(n_windows, n_blocks)``,
            normalized to unit standard deviation. Required by
            ``measured``.

    Returns:
        Perturbed copy of ``rows``.
    """
    out = rows.copy()
    nb_rows, nb_blocks = rows.shape[0], lateral.shape[0]

    def draw() -> np.ndarray:
        """Return one ``(nb_rows, nb_blocks)`` noise field."""
        if source == "white":
            return rng.normal(0.0, sigma_m, size=(nb_rows, nb_blocks))
        picks = rng.integers(0, windows.shape[0], size=nb_rows)
        return sigma_m * windows[picks]

    # One physical displacement per draw, shared by the two channels
    # that encode the lateral CoM in different units.
    lateral_draw = draw()
    out[:, lateral[:, 0]] += lateral_draw
    out[:, lateral[:, 1]] += lateral_draw / scale
    for axis in range(others.shape[1]):
        out[:, others[:, axis]] += draw()
    return out


def summarize(
    baseline: np.ndarray,
    deltas: List[np.ndarray],
    gain: float,
    sigma_zmp: float,
) -> Dict[str, float]:
    """Pool several noise draws into one row of statistics.

    Args:
        baseline: Outputs of the DCM observable on unperturbed rows.
        deltas: One ``xi_perturbed - xi_baseline`` array per draw.
        gain: Feedback gain ``1 + k_p / omega``.
        sigma_zmp: Spread of the commanded ZMP over the same rows.

    Returns:
        Dictionary of summary statistics.
    """
    stacked = np.concatenate(deltas)
    rms = float(np.sqrt(np.mean(stacked**2)))
    std = float(baseline.std())
    per_draw = [float(np.sqrt(np.mean(d**2))) for d in deltas]
    return {
        "rms_delta": rms,
        "sigma_xi": std,
        "rel_rms": rms / std,
        "spread_draws": float(np.std(per_draw)),
        "bias": float(stacked.mean()) / std,
        "rms_delta_zmp": gain * rms,
        "sigma_zmp": sigma_zmp,
        "rel_rms_zmp": gain * rms / sigma_zmp,
    }


def main() -> None:
    """Run the noise sweep and report results."""
    dataset_path = DATASET
    split = SPLIT if IS_TRAINING_DATASET else None

    rng = np.random.default_rng(SEED)
    session, props = load_dcm_observable(MODEL)
    hist_size = int(props["hist_size"])
    obs_dim = int(props["obs_dim"])
    omega = float(props["omega"])
    gain = 1.0 + KP / omega
    print(
        f"DCM observable {MODEL}: hist_size={hist_size} obs_dim={obs_dim} "
        f"omega={omega:.4f} rad/s, feedback gain {gain:.3f}"
    )

    with h5py.File(dataset_path, "r") as hf:
        Y = hf["Y"][:]
        U = hf["U"][:]
        terminated = hf["terminated"][:]
        truncated = hf["truncated"][:]
        dt = float(hf.attrs["dt"])

    scale = metres_per_line_unit(Y)
    histories = build_histories(Y[:, :obs_dim], U, hist_size)
    valid = build_valid_segment_starts(
        histories.shape[0], hist_size, 1, terminated, truncated
    )
    if split is not None:
        train_idx, val_idx = split_indices(valid.shape[0], VAL_FRAC, SEED)
        valid = valid[val_idx if split == "val" else train_idx]
    valid = np.sort(valid)
    if MAX_ROWS and valid.shape[0] > MAX_ROWS:
        valid = np.sort(rng.choice(valid, size=MAX_ROWS, replace=False))
    scope = f"{split} split" if split is not None else "held out in full"
    print(
        f"Dataset {dataset_path}: evaluating {valid.shape[0]} rows ({scope}), "
        f"1 line unit = {scale:.4f} m"
    )

    rows = histories[valid]
    baseline = run_dcm_observable(session, rows)
    # Row j of the embedding is anchored at step j + hist_size, so this
    # is the ZMP commanded at the instant each row describes.
    commanded_zmp = U[hist_size:][valid]
    sigma_zmp = float(commanded_zmp.std())
    print(
        f"Baseline xi: mean={baseline.mean():+.4f} std={baseline.std():.4f}"
        f" | commanded ZMP: std={sigma_zmp:.4f}"
    )

    windows, sigma_measured = residual_windows(
        dataset_path, hist_size + 1, CUTOFF, dt
    )
    print(
        f"Residual pool: {windows.shape[0]} windows of {windows.shape[1]} "
        f"samples, high-passed above {CUTOFF:g} Hz, "
        f"σ = {1e3 * sigma_measured:.2f} mm"
    )

    # The measured residual at its own magnitude, then white noise at
    # that same magnitude to isolate the spectrum, then white noise at
    # larger magnitudes in case the high-pass estimate is optimistic.
    runs = [("measured", sigma_measured), ("white", sigma_measured)]
    runs += [("white", 1e-3 * level) for level in LEVELS_MM]

    lateral, others = com_columns(hist_size, obs_dim)
    records: List[Dict] = []
    for source, sigma_m in runs:
        deltas = [
            run_dcm_observable(
                session,
                perturb(
                    rows,
                    lateral,
                    others,
                    sigma_m,
                    scale,
                    rng,
                    source,
                    windows,
                ),
            )
            - baseline
            for _ in range(SAMPLES)
        ]
        record = {"level": 1e3 * sigma_m, "correlation": source}
        record.update(summarize(baseline, deltas, gain, sigma_zmp))
        records.append(record)
        print(
            "  %-8s sigma_CoM=%.2f mm  RMS(dxi)=%.4f (%.3f sigma_xi)"
            "  RMS(dZMP)=%.4f (%.3f sigma_ZMP)"
            % (
                source,
                record["level"],
                record["rms_delta"],
                record["rel_rms"],
                record["rms_delta_zmp"],
                record["rel_rms_zmp"],
            )
        )

    provenance = (
        f"`{dataset_path}`, {split} split, {valid.shape[0]} embedding rows"
        if split is not None
        else f"`{dataset_path}`, held out in full, {valid.shape[0]} rows"
    )
    table = [
        "| Time structure | σ(CoM) | RMS Δξ | σ(ξ) | RMS Δξ / σ(ξ) "
        "| RMS ΔZMP | σ(ZMP) | RMS ΔZMP / σ(ZMP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        table.append(
            "| %s | %.2f mm | %.4f | %.4f | **%.3f** | %.4f | %.4f "
            "| **%.3f** |"
            % (
                record["correlation"],
                record["level"],
                record["rms_delta"],
                record["sigma_xi"],
                record["rel_rms"],
                record["rms_delta_zmp"],
                record["sigma_zmp"],
                record["rel_rms_zmp"],
            )
        )

    report = "\n".join(
        [
            "# Sensitivity of the DCM observable to CoM measurement noise",
            "",
            f"- Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
            f"- DCM observable: `{MODEL}` (hist_size={hist_size}, "
            f"obs_dim={obs_dim}, ω̂={omega:.4f} rad/s)",
            f"- Dataset: {provenance}",
            f"- {SAMPLES} noise draws per row",
            f"- Residual pool: {windows.shape[0]} windows high-passed "
            f"above {CUTOFF:g} Hz, σ = {1e3 * sigma_measured:.2f} mm",
            f"- Baseline ξ: mean {baseline.mean():+.4f}, "
            f"std {baseline.std():.4f}",
            "",
            "Only the CoM is perturbed: the input of the DCM observable is "
            "`q_actuated(6) + com_anchor(3) + com_line(1)` and the ZMP "
            "estimate is sliced off before training, so ZMP noise moves "
            "ξ by exactly zero. Deltas are normalized by `std(ξ)` and "
            "not by `mean(ξ)`, which sits near zero for a signed "
            "coordinate and would make the ratio unstable. The ZMP "
            "columns carry the same construction one step down the "
            "loop: the feedback law turns a move of the observable "
            f"into a move of the command, ΔZMP = {gain:.3f} Δξ, which "
            "is then referred to the spread of the commanded ZMP over "
            "the same rows. Everything is the normalized line "
            "coordinate in [-1, 1], the units Table 1 reports, so 1 "
            f"unit is {scale:.4f} m of lateral travel.",
            "",
            "\n".join(table),
            "",
            "`measured` draws windows from the residual recorded while "
            "the robot stood still, so its spectrum and its correlation "
            "from sample to sample are the real ones; `white` redraws "
            "independently at every time block and overstates the "
            "content above 10 Hz, where the measured residual has "
            "little power. The first two rows share the measured σ and "
            "so differ only in time structure. The remaining rows raise "
            "the magnitude, which is honest for white noise because its "
            "magnitude is a parameter either way, and is worth doing "
            "because the measured figure may be optimistic: "
            f"high-passing above {CUTOFF:g} Hz removes the robot's "
            "sway but takes any slowly drifting estimator error with "
            "it. The measured residual is never rescaled, since a "
            "fabricated magnitude on a real waveform would say more "
            "about the choice than about the robot. Nor does any mode "
            "hold a single draw across the window, which would pair a "
            "magnitude read off a high-pass residual with a slow time "
            "structure it does not have. Across the "
            f"{SAMPLES} draws the per-draw RMS varies by at most "
            f"{max(r['spread_draws'] for r in records):.4f}, so the "
            "figures are converged.",
            "",
        ]
    )
    if OUTPUT:
        Path(OUTPUT).write_text(report)
        print(f"\nWrote {OUTPUT}")
    print()
    print(report)


if __name__ == "__main__":
    main()
