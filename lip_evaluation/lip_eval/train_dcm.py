# SPDX-License-Identifier: Apache-2.0

"""Train the DCM observable jointly with its eigenvalue ω.

The DCM observable ξ_θ is an MLP over delay embeddings of observations and
actions. It is trained on the one-step DCM loss, which asks ξ_θ to follow
the DCM dynamics, plus a weighted CoM prediction loss, which penalizes the
trivial minimum ω → 0.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HIDDEN_DIMS: tuple[int, ...] = (128, 128)  # hidden layers of the MLP
BATCH_SIZE: int = 1024
LEARNING_RATE: float = 1e-3  # Adam
VAL_FRAC: float = 0.1  # fraction of samples held out for validation


def build_histories(Y: np.ndarray, U: np.ndarray, hist_size: int) -> np.ndarray:
    """Assemble per-step delay embeddings.

    Row j is (y_{k-h}, u_{k-h}, ..., y_{k-1}, u_{k-1}, y_k) for k = j + h.

    Args:
        Y: Observation array of shape (num_steps, obs_dim).
        U: ZMP command array of shape (num_steps,).
        hist_size: Delay-embedding horizon H.

    Returns:
        Delay embeddings of shape (num_steps - H, (H + 1) * obs_dim + H).
    """
    N, obs_dim = Y.shape
    h = hist_size
    M = N - h
    input_dim = (h + 1) * obs_dim + h
    out = np.empty((M, input_dim), dtype=np.float32)
    for i in range(h):
        s = i * (obs_dim + 1)
        out[:, s : s + obs_dim] = Y[i : i + M]
        out[:, s + obs_dim] = U[i : i + M]
    out[:, h * (obs_dim + 1) :] = Y[h : h + M]
    return out


class DCMObservable(nn.Module):
    """MLP mapping a delay embedding to the scalar learned DCM ξ_θ."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        """Initialize the MLP with tanh activations.

        Args:
            input_dim: Dimension of delay embeddings.
            hidden_dims: Widths of the hidden layers.
        """
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set the input normalization applied before the MLP.

        Args:
            mean: Mean of delay embeddings over the training set.
            std: Standard deviation of delay embeddings over the training set.
        """
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned DCM.

        Args:
            x: Batch of delay embeddings, of shape (batch_size, input_dim).

        Returns:
            Learned DCM for each delay embedding, of shape (batch_size,).
        """
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


def get_device() -> torch.device:
    """Return the CUDA device if available, falling back to CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  Warning: no CUDA device found, training on CPU")
    return torch.device("cpu")


def train_dcm(
    Y: np.ndarray,
    U: np.ndarray,
    dt: float,
    hist_size: int,
    omega_init: float,
    epochs: int,
    com_loss_weight: float,
    seed: int,
) -> tuple[
    DCMObservable,
    float,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Train the DCM observable and jointly optimize log ω.

    ω is parameterized as exp(log_ω) to stay positive.  Both the weights of
    the DCM observable and log_ω are updated by the same Adam optimizer.

    Args:
        Y: Observation array of shape (num_steps, obs_dim), whose first
            coordinate is the CoM position.
        U: ZMP command array of shape (num_steps,).
        dt: Time step between consecutive observations, in seconds.
        hist_size: Delay-embedding horizon H.
        omega_init: Initial guess for ω, in rad/s.
        epochs: Number of training epochs.
        com_loss_weight: Weight w_CoM of the CoM prediction loss.
        seed: Seed for weight initialization, batching and the
            train/validation split.

    Returns:
        ``(dcm, best_omega, train_losses, val_losses, omegas,
        val_dcm_losses, val_com_losses)`` --- the DCM observable snapshot
        with lowest validation loss, the corresponding ω, and the per-epoch
        curves (combined val loss, ω, and unweighted DCM/CoM val
        components).
    """
    torch.manual_seed(seed)
    device = get_device()

    obs_dim = Y.shape[1]
    histories = build_histories(Y, U, hist_size)
    M = histories.shape[0]
    X0, X1 = histories[:-1], histories[1:]
    u0 = U[hist_size : hist_size + M - 1]
    n_samples, input_dim = X0.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * VAL_FRAC))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    mean = X0[train_idx].mean(axis=0)
    std = X0[train_idx].std(axis=0) + 1e-6

    dcm = DCMObservable(input_dim, HIDDEN_DIMS).to(device)
    dcm.set_normalization(mean, std)

    log_omega = nn.Parameter(torch.tensor(float(np.log(omega_init)), device=device))
    opt = torch.optim.Adam(list(dcm.parameters()) + [log_omega], lr=LEARNING_RATE)

    def make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        """Batch consecutive delay embeddings and commands.

        Args:
            idx: Indices of the samples to load.
            shuffle: If True, reshuffle samples at every epoch.

        Returns:
            Loader of (x0, x1, u0) batches.
        """
        ds = TensorDataset(
            torch.from_numpy(X0[idx]),
            torch.from_numpy(X1[idx]),
            torch.from_numpy(u0[idx]),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    best_val = float("inf")
    best_state: dict | None = None
    best_omega: float = omega_init
    train_losses: list[float] = []
    val_losses: list[float] = []
    omegas: list[float] = []
    val_dcm_losses: list[float] = []
    val_com_losses: list[float] = []

    for epoch in range(epochs):
        dcm.train()
        total, count = 0.0, 0
        for x0, x1, u in train_loader:
            x0, x1, u = x0.to(device), x1.to(device), u.to(device)
            omega = torch.exp(log_omega)
            expanse = torch.exp(omega * dt)
            decay = torch.exp(-omega * dt)
            xi0, xi1 = dcm(x0), dcm(x1)
            pred_xi1 = expanse * (xi0 - u) + u
            dcm_loss = torch.mean((xi1 - pred_xi1) ** 2)
            c0, c1 = x0[:, -obs_dim], x1[:, -obs_dim]
            pred_c1 = decay * c0 + (1.0 - decay) * (xi0 + xi1) / 2.0
            com_loss = torch.mean((c1 - pred_c1) ** 2)
            loss = dcm_loss + com_loss_weight * com_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x0.shape[0]
            count += x0.shape[0]

        dcm.eval()
        vtotal, vtotal_dcm, vtotal_com, vcount = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            omega = torch.exp(log_omega)
            expanse = torch.exp(omega * dt)
            decay = torch.exp(-omega * dt)
            for x0, x1, u in val_loader:
                x0, x1, u = x0.to(device), x1.to(device), u.to(device)
                xi0, xi1 = dcm(x0), dcm(x1)
                pred_xi1 = expanse * (xi0 - u) + u
                dcm_loss = torch.mean((xi1 - pred_xi1) ** 2)
                c0, c1 = x0[:, -obs_dim], x1[:, -obs_dim]
                pred_c1 = decay * c0 + (1.0 - decay) * (xi0 + xi1) / 2.0
                com_loss = torch.mean((c1 - pred_c1) ** 2)
                loss = dcm_loss + com_loss_weight * com_loss
                nb = x0.shape[0]
                vtotal += loss.item() * nb
                vtotal_dcm += dcm_loss.item() * nb
                vtotal_com += com_loss.item() * nb
                vcount += nb

        omega_cur = float(torch.exp(log_omega).item())
        t_loss = total / count
        v_loss = vtotal / vcount
        v_dcm = vtotal_dcm / vcount
        v_com = vtotal_com / vcount
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        omegas.append(omega_cur)
        val_dcm_losses.append(v_dcm)
        val_com_losses.append(v_com)
        print(
            f"  epoch {epoch + 1:3d}/{epochs}"
            f"  train={t_loss:.4e}  val={v_loss:.4e}"
            f"  val_dcm={v_dcm:.4e}  val_com={v_com:.4e}"
            f"  ω={omega_cur:.4f} rad/s"
        )

        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(dcm.state_dict())
            best_omega = omega_cur

    assert best_state is not None
    dcm.load_state_dict(best_state)
    return (
        dcm,
        best_omega,
        train_losses,
        val_losses,
        omegas,
        val_dcm_losses,
        val_com_losses,
    )
