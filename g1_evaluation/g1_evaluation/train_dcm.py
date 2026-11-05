# SPDX-License-Identifier: Apache-2.0

"""Train the DCM observable on a collected dataset.

The training loss is the multi-step loss of Section 3.4 of the paper. The
trained model is exported to ONNX, with its configuration and the robot name in
the model metadata.
"""

import logging
from pathlib import Path
from typing import Tuple

import h5py
import onnx
import numpy as np
import torch
from torch import nn

from . import settings
from .observation import make_observation
from .robots import get_spec
from .structured_vector import StructuredVector

logger = logging.getLogger("g1_evaluation.train_dcm")

DATASET: Path = settings.DATA_DIR / "pybullet-g1.h5"  # from download-data
OUTPUT: Path = settings.MODELS_DIR / "dcm-pybullet-g1-new.onnx"

# HIST_SIZE, HIDDEN_DIMS, NB_LAMBDA_STEPS and NB_HORIZON_STEPS are those
# stored in the metadata of models/dcm-pybullet-g1.onnx, the model of the paper
HIST_SIZE: int = 10  # delay-embedding horizon H
HIDDEN_DIMS: tuple[int, ...] = (128, 128)  # hidden layers of the MLP
NB_LAMBDA_STEPS: int = 20  # effective horizon 1 / (1 - lambda)
NB_HORIZON_STEPS: int = 100  # truncation of the multi-step loss
OMEGA_INIT: float = 5.0  # rad/s, initial guess for omega
COM_LOSS_WEIGHT: float = 10.0  # w_CoM
EPOCHS: int = 50
BATCH_SIZE: int = 1024  # segments per mini-batch
LEARNING_RATE: float = 3e-4  # Adam
VAL_FRAC: float = 0.1  # fraction of segments held out for validation
SEED: int = 0


class DCMObservable(nn.Module):
    """MLP from the delay embedding to a scalar learned DCM ξ."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...]) -> None:
        """Initialize the network with identity input normalization.

        Args:
            input_dim: Dimension of the delay embedding.
            hidden_dims: Sizes of the hidden layers.
        """
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.Tanh())
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set the input normalization applied in the forward pass.

        Args:
            mean: Mean of the delay embeddings, shape ``(input_dim,)``.
            std: Standard deviation of the delay embeddings, same shape.
        """
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned DCM on a batch of delay embeddings.

        Args:
            x: Delay embeddings, shape ``(B, input_dim)``.

        Returns:
            Learned DCMs, shape ``(B,)``.
        """
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


def build_histories(Y: np.ndarray, U: np.ndarray, hist_size: int) -> np.ndarray:
    """Assemble the delay embedding at each step of the dataset.

    Args:
        Y: Observation array of shape (N, obs_dim).
        U: Action array of shape (N,).
        hist_size: History length h (>= 1).

    Returns:
        Histories of shape (N - h, (h + 1) * obs_dim + h). Row j is the
        delay embedding at step k = j + h, i.e.
        (y_{k-h}, u_{k-h}, ..., y_{k-1}, u_{k-1}, y_k). Action u_k is
        deliberately excluded so the dynamics residual remains
        non-trivial.
    """
    N, obs_dim = Y.shape
    h = hist_size
    M = N - h
    input_dim = (h + 1) * obs_dim + h
    out = np.empty((M, input_dim), dtype=np.float32)
    for i in range(h):
        start = i * (obs_dim + 1)
        out[:, start : start + obs_dim] = Y[i : i + M]
        out[:, start + obs_dim] = U[i : i + M]
    out[:, h * (obs_dim + 1) :] = Y[h : h + M]
    return out


def lambda_from_nb_lambda_steps(nb_lambda_steps: int) -> float:
    """Compute λ from the number of lambda steps ``1 / (1 − λ)``.

    The number of lambda steps is the effective horizon of the geometric
    weights (1 − λ) λ^{n−1}, hence ``λ = 1 − 1 / nb_lambda_steps``. The
    boundary case ``nb_lambda_steps = 1`` gives ``λ = 0``, i.e. a one-step
    loss.

    Args:
        nb_lambda_steps: Number of lambda steps, at least 1.

    Returns:
        Value of λ in ``[0, 1)``.
    """
    if nb_lambda_steps < 1:
        raise ValueError("nb_lambda_steps must be >= 1")
    return 1.0 - 1.0 / float(nb_lambda_steps)


def split_indices(
    nb_samples: int, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split sample indices into training and validation sets.

    Args:
        nb_samples: Number of samples.
        val_frac: Fraction of samples in the validation set.
        seed: Seed of the random permutation.

    Returns:
        Training indices and validation indices.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nb_samples)
    nb_val = max(1, int(nb_samples * val_frac))
    return perm[nb_val:], perm[:nb_val]


def build_valid_segment_starts(
    M: int,
    hist_size: int,
    nb_horizon_steps: int,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Indices ``j`` such that a length-``nb_horizon_steps+1`` segment of
    histories starting at ``j`` is fully populated and crosses no episode
    boundary.

    A segment occupies history rows ``j, j+1, …, j+nb_horizon_steps``. The
    history window for row ``p`` spans flags ``[p, p+hist_size]``, so the
    segment's flags span ``[j, j+nb_horizon_steps+hist_size]``. A start is
    valid iff that union contains no terminated/truncated flag.

    Args:
        M: Number of history rows (= ``N − hist_size`` from the raw data).
        hist_size: Delay-embedding horizon.
        nb_horizon_steps: Truncation horizon of the multi-step loss
            (length of the segment's action window).
        terminated: Boolean array of shape ``(N,)`` from the env.
        truncated: Boolean array of shape ``(N,)`` from the env.

    Returns:
        ``int64`` array of valid segment-start indices in
        ``[0, M − nb_horizon_steps]``.
    """
    if terminated.shape != truncated.shape:
        raise ValueError(
            "terminated and truncated must have the same shape, got "
            f"{terminated.shape} vs {truncated.shape}"
        )
    if hist_size < 1:
        raise ValueError("hist_size must be >= 1")
    if nb_horizon_steps < 1:
        raise ValueError("nb_horizon_steps must be >= 1")
    N = terminated.shape[0]
    if M != N - hist_size:
        raise ValueError(f"M={M} inconsistent with N={N} and hist_size={hist_size}")
    nb_starts = M - nb_horizon_steps
    if nb_starts <= 0:
        return np.empty(0, dtype=np.int64)
    flags = (terminated | truncated).astype(np.int64)
    cum = np.concatenate(([0], np.cumsum(flags)))
    js = np.arange(nb_starts, dtype=np.int64)
    # Flag window for segment starting at j is
    # [j, j + nb_horizon_steps + hist_size], i.e.
    # nb_horizon_steps + hist_size + 1 entries.
    window_sums = cum[js + nb_horizon_steps + hist_size + 1] - cum[js]
    return js[window_sums == 0]


def run_epoch(
    dcm_observable: DCMObservable,
    histories: torch.Tensor,
    U_aligned: torch.Tensor,
    com_line_idx: int,
    segment_starts: torch.Tensor,
    log_omega: nn.Parameter,
    dt: float,
    com_loss_weight: float,
    lam: float,
    nb_horizon_steps: int,
    batch_size: int,
    opt: torch.optim.Optimizer | None,
    shuffle: bool,
) -> Tuple[float, float, float]:
    """Run one epoch and return the per-sample average losses.

    Returns ``(loss, dcm_loss, com_loss)``, each as the average over the
    epoch's samples. ``loss = dcm_loss + com_loss_weight · com_loss`` is
    the optimisation target; the unweighted DCM and CoM components are
    returned separately so they can be compared across runs with
    different ``com_loss_weight``.

    Args:
        dcm_observable: Network mapping a delay embedding to a scalar ξ.
        histories: All delay embeddings of the dataset, shape
            ``(M, input_dim)``, on the training device. Row ``j`` is the
            delay embedding at step ``hist_size + j``.
        U_aligned: Actions aligned to ``histories``, shape ``(M,)``.
            ``U_aligned[j]`` drives the transition from history row ``j``
            to row ``j+1``.
        com_line_idx: Column index of the CoM-line scalar inside each
            ``histories`` row (used to read ``c_n`` cheaply without a
            parallel tensor).
        segment_starts: Valid segment-start indices into ``histories``,
            shape ``(n_seg,)``. Each start ``j`` indexes a contiguous
            segment ``[j, j+1, …, j+nb_horizon_steps]``.
        log_omega: Scalar ``nn.Parameter`` with ``ω = exp(log_omega)``;
            optimized jointly with the network weights.
        dt: Discrete control-loop time step in seconds.
        com_loss_weight: Weight on the CoM-side multi-step residual term.
        lam: Value of λ in ``[0, 1)``. ``0`` gives a one-step loss.
        nb_horizon_steps: Truncation horizon ``m`` of the multi-step loss,
            in steps of the recursion. The DCM observable is evaluated on
            segments of length ``m + 1``.
        batch_size: Number of segments per mini-batch.
        opt: Adam optimiser, or ``None`` for an evaluation pass.
        shuffle: Whether to randomly permute ``segment_starts`` before
            iterating (typically ``True`` for training, ``False`` for
            validation).

    Returns:
        Average total loss, DCM loss and CoM loss over the epoch.
    """
    is_train = opt is not None
    dcm_observable.train(is_train)
    device = histories.device
    n_seg = segment_starts.shape[0]
    if shuffle:
        perm = torch.randperm(n_seg, device=device)
    else:
        perm = torch.arange(n_seg, device=device)

    if lam == 0.0:
        weights = torch.zeros(nb_horizon_steps, device=device)
        weights[0] = 1.0
    else:
        n_axis = torch.arange(
            1, nb_horizon_steps + 1, device=device, dtype=torch.float32
        )
        weights = (1.0 - lam) * (lam ** (n_axis - 1.0))

    total = 0.0
    dcm_total = 0.0
    com_total = 0.0
    nb_samples = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    offsets_full = torch.arange(nb_horizon_steps + 1, device=device)
    offsets_action = torch.arange(nb_horizon_steps, device=device)
    with ctx:
        for batch_start in range(0, n_seg, batch_size):
            batch_perm = perm[batch_start : batch_start + batch_size]
            starts = segment_starts[batch_perm]  # (B,)
            B = starts.shape[0]

            # (B, nb_horizon_steps+1) → delay embeddings along the segment
            seg_pos = starts.unsqueeze(1) + offsets_full.unsqueeze(0)
            X_seg = histories[seg_pos]  # (B, nb_horizon_steps+1, input_dim)

            # (B, nb_horizon_steps) → actions u_n driving each transition
            u_seg = U_aligned[starts.unsqueeze(1) + offsets_action.unsqueeze(0)]

            # (B, nb_horizon_steps+1) → CoM-line at each segment position,
            # read from the delay embedding (avoids a parallel tensor).
            c_seg = X_seg[..., com_line_idx]

            # Single batched forward over all (B * (nb_horizon_steps+1)) inputs
            xi = dcm_observable(X_seg.reshape(B * (nb_horizon_steps + 1), -1)).reshape(
                B, nb_horizon_steps + 1
            )

            omega = torch.exp(log_omega)
            gamma = torch.exp(-omega * dt)

            # One-step DCM residuals r_k (backward form, γ < 1)
            r = (
                xi[:, :-1] - gamma * xi[:, 1:] - (1.0 - gamma) * u_seg
            )  # (B, nb_horizon_steps)
            # n-step DCM residuals R_n = Σ_{k=0}^{n−1} γ^k r_k: cumulative
            # sum along the horizon. Column n of R is R_{n+1}.
            k_powers = gamma ** torch.arange(
                nb_horizon_steps, device=device, dtype=gamma.dtype
            )
            R = (k_powers.unsqueeze(0) * r).cumsum(dim=1)

            # One-step CoM residuals s_k (forward in time, γ < 1)
            s = (
                c_seg[:, 1:]
                - gamma * c_seg[:, :-1]
                - (1.0 - gamma) / 2.0 * (xi[:, :-1] + xi[:, 1:])
            )  # (B, nb_horizon_steps)
            # n-step CoM residuals S_n = γ^{n−1} s_0 + γ^{n−2} s_1 + … + s_{n−1},
            # computed by the recurrence S_n = γ · S_{n−1} + s_{n−1}. (The
            # cumsum form S_n = γ^{n−1} · cumsum(γ^{−k} s_k) is numerically
            # unsafe as γ^{−k} overflows float32 at large nb_horizon_steps.)
            S_list = [s[:, 0]]
            for n in range(1, nb_horizon_steps):
                S_list.append(gamma * S_list[-1] + s[:, n])
            S = torch.stack(S_list, dim=1)  # (B, nb_horizon_steps)

            dcm_per_n = (R**2).mean(dim=0)
            com_per_n = (S**2).mean(dim=0)
            dcm_loss = (weights * dcm_per_n).sum()
            com_loss = (weights * com_per_n).sum()
            loss = dcm_loss + com_loss_weight * com_loss

            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total += loss.item() * B
            dcm_total += dcm_loss.item() * B
            com_total += com_loss.item() * B
            nb_samples += B

    return (
        total / nb_samples,
        dcm_total / nb_samples,
        com_total / nb_samples,
    )


def train(
    observation_schema: StructuredVector,
    robot: str,
    dataset: str,
    output: str,
    hist_size: int,
    omega_init: float,
    hidden_dims: Tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    com_loss_weight: float,
    nb_lambda_steps: int,
    nb_horizon_steps: int,
) -> None:
    """Train the DCM observable on a collected dataset.

    The ONNX export is wrapped in a ``finally`` block so that an
    interrupted run still exports the best model found so far.

    Args:
        observation_schema: Per-robot observation schema used to validate
            ``obs_dim`` and locate the observations of the delay embedding.
        robot: Short robot name, written into the ONNX metadata.
        dataset: Path of the HDF5 dataset.
        output: Path to save the trained model.
        hist_size: Delay-embedding horizon H, i.e. number of past (y, u)
            pairs.
        omega_init: Initial guess for the LIP natural frequency, in rad/s.
        hidden_dims: MLP hidden layer sizes.
        epochs: Number of training epochs.
        batch_size: Mini-batch size, in segments per batch.
        lr: Adam learning rate.
        val_frac: Fraction of segments used for validation.
        seed: Random seed.
        com_loss_weight: Weight on the multi-step CoM residual.
        nb_lambda_steps: Number of lambda steps ``1 / (1 − λ)``. ``1``
            gives a one-step loss; larger values weight longer horizons.
        nb_horizon_steps: Truncation horizon of the multi-step loss.
    """
    if hist_size < 1:
        raise ValueError("hist_size must be >= 1")
    logger.info("Using seed=%d", seed)
    torch.manual_seed(seed)

    lam = lambda_from_nb_lambda_steps(nb_lambda_steps)
    if nb_horizon_steps < 1:
        raise ValueError("nb_horizon_steps must be >= 1")
    pt_path = Path(output).with_suffix(".pt")
    onnx_path = Path(output).with_suffix(".onnx")
    if pt_path.exists() or onnx_path.exists():
        existing = [str(p) for p in (pt_path, onnx_path) if p.exists()]
        raise FileExistsError(
            f"{existing[0]} already exists: move it away to train a new model."
        )
    logger.info(
        "Multi-step loss: nb_lambda_steps=%d → λ=%.6f, nb_horizon_steps=%d "
        "(truncation tail λ^nb_horizon_steps=%.3e)",
        nb_lambda_steps,
        lam,
        nb_horizon_steps,
        float(lam**nb_horizon_steps) if lam > 0 else 0.0,
    )

    with h5py.File(dataset, "r") as hf:
        Y = hf["Y"][:].astype(np.float32)
        U = hf["U"][:].astype(np.float32)
        if "terminated" not in hf or "truncated" not in hf:
            raise ValueError(
                f"{dataset!r} is missing 'terminated'/'truncated' datasets. "
                "Re-collect data with 'pixi run collect-data'."
            )
        terminated = hf["terminated"][:].astype(bool)
        truncated = hf["truncated"][:].astype(bool)
        if "dt" not in hf.attrs:
            raise ValueError(
                f"{dataset!r} is missing the 'dt' attribute. "
                "Re-collect data with 'pixi run collect-data'."
            )
        dt = float(hf.attrs["dt"])
    N, obs_dim = Y.shape
    logger.info(
        "Loaded %d steps from %s "
        "(obs_dim=%d, dt=%.4f s, terminated=%d, truncated=%d)",
        N,
        dataset,
        obs_dim,
        dt,
        int(terminated.sum()),
        int(truncated.sum()),
    )

    # Validate the observation schema and keep the observations of the
    # delay embedding: [q_actuated, com_anchor, com_line].
    sv = observation_schema
    if obs_dim != sv.dim:
        raise ValueError(
            f"obs_dim={obs_dim} in {dataset!r} does not match the current "
            f"observation schema (expected {sv.dim}). "
            "Re-collect data with 'pixi run collect-data'."
        )
    emb_obs_dim = sv.slice_of("com_line").stop
    Y_emb = Y[:, :emb_obs_dim]
    logger.info(
        "Delay-embedding observation: q_actuated(%d) + com_anchor(3) + "
        "com_line(1) = %d dims per step",
        sv.slice_of("q_actuated").stop,
        emb_obs_dim,
    )

    histories_np = build_histories(Y_emb, U, hist_size)
    M, input_dim = histories_np.shape
    # U_aligned[j] = action driving the transition from history row j
    # to row j+1, i.e. u at global step (hist_size + j).
    U_aligned_np = U[hist_size : hist_size + M].astype(np.float32)
    # com_line is the last element of Y_emb; locate it in the history row.
    com_line_idx = hist_size * (emb_obs_dim + 1) + sv.slice_of("com_line").start

    valid_starts_np = build_valid_segment_starts(
        M, hist_size, nb_horizon_steps, terminated, truncated
    )
    n_seg = valid_starts_np.shape[0]
    if n_seg == 0:
        raise RuntimeError(
            f"No valid segments of length {nb_horizon_steps + 1} fit in "
            f"the dataset after filtering episode boundaries (M={M}, "
            f"hist_size={hist_size})."
        )
    logger.info(
        "Built %d valid segments of length %d " "(M=%d, dropped %d to boundaries)",
        n_seg,
        nb_horizon_steps + 1,
        M,
        max(0, M - nb_horizon_steps) - n_seg,
    )

    train_seg_idx, val_seg_idx = split_indices(n_seg, val_frac, seed)
    # Normalisation from training-set anchor rows (segment starts).
    mean = histories_np[valid_starts_np[train_seg_idx]].mean(axis=0)
    std = histories_np[valid_starts_np[train_seg_idx]].std(axis=0) + 1e-6

    # Trivial-prediction floor for the one-step CoM loss: a DCM observable
    # that contributes nothing (ξ ≡ 0, ω → 0) reduces it to mean((Δc)²).
    # Reported as a benchmark for the n = 1 term of the CoM loss.
    val_anchors = valid_starts_np[val_seg_idx]
    val_com_floor = float(
        np.mean(
            (
                histories_np[val_anchors + 1, com_line_idx]
                - histories_np[val_anchors, com_line_idx]
            )
            ** 2
        )
    )
    logger.info(
        "val_com_floor (1-step trivial baseline) = mean((Δc_line)²) = %.4e",
        val_com_floor,
    )

    device = get_device()
    dcm_observable = DCMObservable(input_dim, hidden_dims).to(device)
    dcm_observable.set_normalization(mean, std)

    log_omega = nn.Parameter(torch.tensor(float(np.log(omega_init)), device=device))
    logger.info("omega_init=%.3f rad/s", omega_init)

    # Move bulk tensors to device once. ~M·input_dim·4 bytes for histories.
    # Per-batch slicing is then a pure-GPU operation.
    histories = torch.from_numpy(histories_np).to(device)
    U_aligned = torch.from_numpy(U_aligned_np).to(device)
    train_starts = torch.from_numpy(valid_starts_np[train_seg_idx]).to(device)
    val_starts = torch.from_numpy(valid_starts_np[val_seg_idx]).to(device)

    opt = torch.optim.Adam(list(dcm_observable.parameters()) + [log_omega], lr=lr)

    best_val = float("inf")
    best_omega: float = omega_init
    try:
        for epoch in range(epochs):
            train_loss, train_dcm, train_com = run_epoch(
                dcm_observable,
                histories,
                U_aligned,
                com_line_idx,
                train_starts,
                log_omega,
                dt,
                com_loss_weight,
                lam,
                nb_horizon_steps,
                batch_size,
                opt,
                shuffle=True,
            )
            val_loss, val_dcm, val_com = run_epoch(
                dcm_observable,
                histories,
                U_aligned,
                com_line_idx,
                val_starts,
                log_omega,
                dt,
                com_loss_weight,
                lam,
                nb_horizon_steps,
                batch_size,
                None,
                shuffle=False,
            )
            omega_cur = float(torch.exp(log_omega).item())
            logger.info(
                "epoch %3d/%d  train=%.3e (dcm=%.3e com=%.3e) | "
                "val=%.3e (dcm=%.3e com=%.3e) | ω=%.4f rad/s",
                epoch + 1,
                epochs,
                train_loss,
                train_dcm,
                train_com,
                val_loss,
                val_dcm,
                val_com,
                omega_cur,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_omega = omega_cur
                pt_path = Path(output).with_suffix(".pt")
                pt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": dcm_observable.state_dict(),
                        "config": {
                            "hist_size": hist_size,
                            "omega": best_omega,
                            "dt": dt,
                            "obs_dim": emb_obs_dim,
                            "input_dim": input_dim,
                            "hidden_dims": list(hidden_dims),
                            "nb_lambda_steps": nb_lambda_steps,
                            "nb_horizon_steps": nb_horizon_steps,
                        },
                    },
                    str(pt_path),
                )
    finally:
        if best_val < float("inf"):
            pt_path = Path(output).with_suffix(".pt")
            logger.info(
                "Best validation loss: %.6e  omega=%.4f rad/s (saved to %s)",
                best_val,
                best_omega,
                pt_path,
            )

            onnx_path = Path(output).with_suffix(".onnx")
            best_observable = DCMObservable(input_dim, hidden_dims)
            best_observable.load_state_dict(
                torch.load(str(pt_path), map_location="cpu", weights_only=True)[
                    "state_dict"
                ]
            )
            best_observable.eval()
            torch.onnx.export(
                best_observable,
                torch.zeros(1, input_dim),
                str(onnx_path),
                input_names=["history"],
                output_names=["dcm"],
                opset_version=17,
            )
            onnx_model = onnx.load(str(onnx_path))
            for key, value in {
                "hist_size": str(hist_size),
                "omega": str(best_omega),
                "dt": str(dt),
                "obs_dim": str(emb_obs_dim),
                "input_dim": str(input_dim),
                "hidden_dims": str(list(hidden_dims)),
                "nb_lambda_steps": str(nb_lambda_steps),
                "nb_horizon_steps": str(nb_horizon_steps),
                "robot": robot,
            }.items():
                entry = onnx_model.metadata_props.add()
                entry.key = key
                entry.value = value
            onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
            data_path = onnx_path.parent / (onnx_path.name + ".data")
            if data_path.exists():
                data_path.unlink()
            logger.info("ONNX model saved to %s", onnx_path)

            # The .pt only carries the best-so-far weights across epochs
            # so that the export above can pick them up: the ONNX now
            # holds the same weights and config. An export failure skips
            # this line, leaving the checkpoint for a manual retry.
            pt_path.unlink(missing_ok=True)


def get_device() -> torch.device:
    """Return the CUDA device if available, falling back to CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("No CUDA device found, training on CPU")
    return torch.device("cpu")


def main() -> None:
    """Train the DCM observable on the collected dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve the observation schema from the dataset's ``robot``
    # attribute, written at collection time.
    with h5py.File(DATASET, "r") as hf:
        if "robot" not in hf.attrs:
            raise SystemExit(
                f"{DATASET} has no 'robot' attribute; re-collect it with "
                "`pixi run collect-data`."
            )
        robot_name = str(hf.attrs["robot"])
    observation_schema = make_observation(get_spec(robot_name))

    train(
        observation_schema=observation_schema,
        robot=robot_name,
        dataset=str(DATASET),
        output=str(OUTPUT),
        hist_size=HIST_SIZE,
        omega_init=OMEGA_INIT,
        hidden_dims=HIDDEN_DIMS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        val_frac=VAL_FRAC,
        seed=SEED,
        com_loss_weight=COM_LOSS_WEIGHT,
        nb_lambda_steps=NB_LAMBDA_STEPS,
        nb_horizon_steps=NB_HORIZON_STEPS,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
