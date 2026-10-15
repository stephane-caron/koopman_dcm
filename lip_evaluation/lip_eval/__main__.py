# SPDX-License-Identifier: Apache-2.0

"""Evaluate the DCM training pipeline on a 1-D linear inverted pendulum.

Generates synthetic data from c̈ = ω²(c − z), trains the DCM observable
ξ_θ(y, u history) jointly with ω, then plots training curves.

Usage:
    pixi run train
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lip_eval.figures import save_pyplot_figure
from lip_eval.train_dcm import train

# ---------------------------------------------------------------------------
# Training run (Appendix A)
# ---------------------------------------------------------------------------

NUM_STEPS: int = 200_000  # closed-loop simulation steps in the dataset
EPOCHS: int = 100
HIST_SIZE: int = 10  # delay-embedding horizon H
OMEGA_INIT: float = 4.0  # rad/s, initial guess for ω
COM_LOSS_WEIGHT: float = 1.0  # w_CoM
SEED: int = 0  # for both data generation and training
PLOT_NAME: str = "lip_eval_losses"

# ---------------------------------------------------------------------------
# LIP model
# ---------------------------------------------------------------------------

OMEGA_TRUE: float = 5.0  # LIP natural frequency sqrt(g/h), h = 0.4 m
DT: float = 1.0 / 200.0  # s
OBS_DIM: int = 6  # y = [cx, cy, cz, ċx, ċy, ċz]
COM_HEIGHT: float = 0.4  # m, constant CoM height


# ---------------------------------------------------------------------------
# LIP dynamics
# ---------------------------------------------------------------------------


def lip_step(
    c: float, cdot: float, z: float, omega: float, dt: float
) -> tuple[float, float]:
    """Exact ZOH integration of LIP dynamics for piecewise-constant z."""
    odt = omega * dt
    d = c - z
    c_next = np.cosh(odt) * d + np.sinh(odt) / omega * cdot + z
    cdot_next = omega * np.sinh(odt) * d + np.cosh(odt) * cdot
    return float(c_next), float(cdot_next)


def dcm_from_state(c: float, cdot: float, omega: float) -> float:
    return c + cdot / omega


def make_obs(c: float, cdot: float) -> np.ndarray:
    """6-D observation [cx, cy, cz, ċx, ċy, ċz]; cy/cz/ċy/ċz are constant."""
    return np.array([c, 0.0, COM_HEIGHT, cdot, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sinusoidal ZMP/DCM reference
# ---------------------------------------------------------------------------


def compute_targets(
    t: float, amplitude: float, offset: float, period: float, omega: float
) -> tuple[float, float]:
    """Return (zmp_ref, dcm_ref) for a sine-wave ZMP reference at time t."""
    nu = 2.0 * np.pi / period
    phase = nu * t
    zmp = amplitude * np.sin(phase) + offset
    shift = np.arctan2(nu, omega)
    dcm = amplitude * omega / np.sqrt(omega**2 + nu**2) * np.sin(phase + shift) + offset
    return zmp, dcm


# ---------------------------------------------------------------------------
# DCM controller
# ---------------------------------------------------------------------------


def compute_zmp(
    xi: float,
    xi_ref: float,
    z_ref: float,
    kp: float = 1.0,
    omega: float = OMEGA_TRUE,
) -> float:
    """z = z_ref − (1 + kp/ω)(ξ_ref − ξ), clipped to [−1, 1]."""
    zmp_gain = 1.0 + kp / omega
    return float(np.clip(z_ref - zmp_gain * (xi_ref - xi), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


_PERIODS = [
    0.3,
    0.5,
    0.8,
    1.2,
    2.0,
]  # sinusoidal ZMP periods used in curriculum
_AMPLITUDES = [0.1, 0.2, 0.3, 0.4, 0.5]
_OFFSETS = [-0.2, -0.1, 0.0, 0.1, 0.2]
_CHUNK_STEPS = 2_000  # re-draw parameters every N steps


def generate_data(
    num_steps: int,
    kp: float = 1.0,
    fall_threshold: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the 1-D LIP under a varied sinusoidal curriculum.

    Parameters are re-drawn from _PERIODS / _AMPLITUDES / _OFFSETS every
    _CHUNK_STEPS steps.  Varied frequencies are required for ω to be
    identifiable: with a single frequency there are approximate local minima
    for any (learned DCM ≈ c + ċ/ω′, ω̂ = ω′).

    Returns:
        Y: Observation array of shape (num_steps, OBS_DIM).
        U: ZMP command array of shape (num_steps,).
    """
    rng = np.random.default_rng(seed)
    c, cdot, t = 0.0, 0.0, 0.0
    Y = np.empty((num_steps, OBS_DIM), dtype=np.float32)
    U = np.empty(num_steps, dtype=np.float32)
    n_resets = 0
    period = float(rng.choice(_PERIODS))
    amplitude = float(rng.choice(_AMPLITUDES))
    offset = float(rng.choice(_OFFSETS))

    for k in range(num_steps):
        if k % _CHUNK_STEPS == 0 and k > 0:
            period = float(rng.choice(_PERIODS))
            amplitude = float(rng.choice(_AMPLITUDES))
            offset = float(rng.choice(_OFFSETS))
            t = 0.0

        z_ref, xi_ref = compute_targets(t, amplitude, offset, period, OMEGA_TRUE)
        xi = dcm_from_state(c, cdot, OMEGA_TRUE)
        z = compute_zmp(xi, xi_ref, z_ref, kp)

        Y[k] = make_obs(c, cdot)
        U[k] = np.float32(z)

        c, cdot = lip_step(c, cdot, z, OMEGA_TRUE, DT)
        t += DT

        if abs(c) > fall_threshold:
            c, cdot, t = 0.0, 0.0, 0.0
            n_resets += 1

    print(f"  {num_steps:,} steps — {n_resets} resets")
    return Y, U


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def build_histories(Y: np.ndarray, U: np.ndarray, hist_size: int) -> np.ndarray:
    """Assemble per-step delayed observable vectors.

    Row j is (y_{k-h}, u_{k-h}, ..., y_{k-1}, u_{k-1}, y_k) for k = j + h.
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


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """MLP mapping observation-action history to a scalar ξ."""

    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 128)
    ) -> None:
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
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training with jointly learned ω
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Return the CUDA device if available, falling back to CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  Warning: no CUDA device found, training on CPU")
    return torch.device("cpu")


def train(
    Y: np.ndarray,
    U: np.ndarray,
    hist_size: int = 10,
    omega_init: float = 4.0,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 0,
    hidden_dims: tuple[int, ...] = (128, 128),
    com_loss_weight: float = 1.0,
) -> tuple[
    Encoder,
    float,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Train encoder and jointly optimize log ω.

    ω is parameterized as exp(log_ω) to stay positive.  Both the encoder
    weights and log_ω are updated by the same Adam optimizer.

    Returns:
        ``(encoder, best_omega, train_losses, val_losses, omegas,
        val_dcm_losses, val_com_losses)`` --- the encoder snapshot with
        lowest validation loss, the corresponding ω, and the per-epoch
        curves (combined val loss, ω, and unweighted DCM/CoM val
        components).
    """
    torch.manual_seed(seed)
    device = get_device()

    histories = build_histories(Y, U, hist_size)
    M = histories.shape[0]
    X0, X1 = histories[:-1], histories[1:]
    u0 = U[hist_size : hist_size + M - 1]
    n_samples, input_dim = X0.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_frac))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    mean = X0[train_idx].mean(axis=0)
    std = X0[train_idx].std(axis=0) + 1e-6

    encoder = Encoder(input_dim, hidden_dims).to(device)
    encoder.set_normalization(mean, std)

    log_omega = nn.Parameter(torch.tensor(float(np.log(omega_init)), device=device))
    opt = torch.optim.Adam(list(encoder.parameters()) + [log_omega], lr=lr)

    def make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X0[idx]),
            torch.from_numpy(X1[idx]),
            torch.from_numpy(u0[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

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
        encoder.train()
        total, count = 0.0, 0
        for x0, x1, u in train_loader:
            x0, x1, u = x0.to(device), x1.to(device), u.to(device)
            omega = torch.exp(log_omega)
            expanse = torch.exp(omega * DT)
            decay = torch.exp(-omega * DT)
            xi0, xi1 = encoder(x0), encoder(x1)
            pred_xi1 = expanse * (xi0 - u) + u
            dcm_loss = torch.mean((xi1 - pred_xi1) ** 2)
            c0, c1 = x0[:, -OBS_DIM], x1[:, -OBS_DIM]
            pred_c1 = decay * c0 + (1.0 - decay) * (xi0 + xi1) / 2.0
            com_loss = torch.mean((c1 - pred_c1) ** 2)
            loss = dcm_loss + com_loss_weight * com_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x0.shape[0]
            count += x0.shape[0]

        encoder.eval()
        vtotal, vtotal_dcm, vtotal_com, vcount = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            omega = torch.exp(log_omega)
            expanse = torch.exp(omega * DT)
            decay = torch.exp(-omega * DT)
            for x0, x1, u in val_loader:
                x0, x1, u = x0.to(device), x1.to(device), u.to(device)
                xi0, xi1 = encoder(x0), encoder(x1)
                pred_xi1 = expanse * (xi0 - u) + u
                dcm_loss = torch.mean((xi1 - pred_xi1) ** 2)
                c0, c1 = x0[:, -OBS_DIM], x1[:, -OBS_DIM]
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
            best_state = copy.deepcopy(encoder.state_dict())
            best_omega = omega_cur

    assert best_state is not None
    encoder.load_state_dict(best_state)
    return (
        encoder,
        best_omega,
        train_losses,
        val_losses,
        omegas,
        val_dcm_losses,
        val_com_losses,
    )


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    omegas: list[float],
    plot_name: str,
    val_dcm_losses: list[float] | None = None,
    val_com_losses: list[float] | None = None,
) -> None:
    """Plot per-epoch train/val losses (left) and ω (right axis).

    If ``val_dcm_losses`` and ``val_com_losses`` are provided, overlays
    the unweighted DCM and CoM validation components so the reader can
    see which term dominates and when each crosses below the other.
    """
    epochs_x = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    (l1,) = ax.plot(epochs_x, train_losses, lw=1.8, label="Train (total)")
    (l2,) = ax.plot(epochs_x, val_losses, lw=1.8, label="Val (total)")
    handles.extend([l1, l2])
    if val_dcm_losses is not None:
        (ldcm,) = ax.plot(
            epochs_x,
            val_dcm_losses,
            lw=1.2,
            ls="--",
            color="tab:red",
            label="Val DCM",
        )
        handles.append(ldcm)
    if val_com_losses is not None:
        (lcom,) = ax.plot(
            epochs_x,
            val_com_losses,
            lw=1.2,
            ls="--",
            color="tab:purple",
            label="Val CoM",
        )
        handles.append(lcom)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("DCM training curves on the LIP")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    (l3,) = ax2.plot(epochs_x, omegas, lw=1.5, color="tab:green", label="ω (rad/s)")
    ax2.axhline(OMEGA_TRUE, color="tab:green", lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylabel("ω (rad/s)")
    handles.append(l3)
    ax.legend(handles=handles, loc="best")
    plt.tight_layout()
    save_pyplot_figure(plot_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Data generation ===")
    Y, U = generate_data(NUM_STEPS, seed=SEED)

    print("\n=== Training ===")
    (
        _,
        omega_learned,
        train_losses,
        val_losses,
        omegas,
        val_dcm_losses,
        val_com_losses,
    ) = train(
        Y,
        U,
        dt=DT,
        hist_size=HIST_SIZE,
        omega_init=OMEGA_INIT,
        epochs=EPOCHS,
        com_loss_weight=COM_LOSS_WEIGHT,
        seed=SEED,
    )
    plot_losses(
        train_losses,
        val_losses,
        omegas,
        plot_name=PLOT_NAME,
        val_dcm_losses=val_dcm_losses,
        val_com_losses=val_com_losses,
    )

    print("\n=== Summary ===")
    print(f"  Learned ω = {omega_learned:.4f} rad/s")
    print(f"  True    ω = {OMEGA_TRUE:.4f} rad/s")
    print(
        f"  Error     = {abs(omega_learned - OMEGA_TRUE):.4f} rad/s"
        f"  ({100 * abs(omega_learned - OMEGA_TRUE) / OMEGA_TRUE:.1f}%)"
    )
    print(f"  Final val DCM loss = {val_dcm_losses[-1]:.4e}")
    print(f"  Final val CoM loss = {val_com_losses[-1]:.4e}")


if __name__ == "__main__":
    main()
