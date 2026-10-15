# SPDX-License-Identifier: Apache-2.0

"""Generate closed-loop data from a 1-D linear inverted pendulum (LIP).

The LIP follows c̈ = ω²(c − z), with CoM position c and ZMP z. A DCM
controller tracks sinusoidal ZMP references whose period, amplitude and
offset are re-drawn every CHUNK_STEPS steps.
"""

import numpy as np

OMEGA_TRUE: float = 5.0  # LIP natural frequency sqrt(g/h), h = 0.4 m
DT: float = 1.0 / 200.0  # s
OBS_DIM: int = 6  # y = [cx, cy, cz, ċx, ċy, ċz]
COM_HEIGHT: float = 0.4  # m, constant CoM height

PERIODS: tuple[float, ...] = (0.3, 0.5, 0.8, 1.2, 2.0)  # s, ZMP reference
AMPLITUDES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
OFFSETS: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
CHUNK_STEPS: int = 2_000  # re-draw reference parameters every N steps


def lip_step(
    c: float, cdot: float, z: float, omega: float, dt: float
) -> tuple[float, float]:
    """Exact ZOH integration of LIP dynamics for piecewise-constant z.

    Args:
        c: CoM position.
        cdot: CoM velocity.
        z: ZMP, held constant over the time step.
        omega: LIP natural frequency, in rad/s.
        dt: Duration of the time step, in seconds.

    Returns:
        CoM position and velocity at the end of the time step.
    """
    odt = omega * dt
    d = c - z
    c_next = np.cosh(odt) * d + np.sinh(odt) / omega * cdot + z
    cdot_next = omega * np.sinh(odt) * d + np.cosh(odt) * cdot
    return float(c_next), float(cdot_next)


def dcm_from_state(c: float, cdot: float, omega: float) -> float:
    """Compute the LIP DCM ξ = c + ċ/ω.

    Args:
        c: CoM position.
        cdot: CoM velocity.
        omega: LIP natural frequency, in rad/s.

    Returns:
        LIP DCM.
    """
    return c + cdot / omega


def make_obs(c: float, cdot: float) -> np.ndarray:
    """Build the 6-D observation [cx, cy, cz, ċx, ċy, ċz].

    Args:
        c: CoM position cx.
        cdot: CoM velocity ċx.

    Returns:
        Observation vector, where cy, cz, ċy and ċz are constant.
    """
    return np.array([c, 0.0, COM_HEIGHT, cdot, 0.0, 0.0], dtype=np.float32)


def compute_targets(
    t: float, amplitude: float, offset: float, period: float, omega: float
) -> tuple[float, float]:
    """Compute ZMP and DCM references for a sine-wave ZMP reference.

    Args:
        t: Time since the start of the reference, in seconds.
        amplitude: Amplitude of the ZMP sine wave.
        offset: Stationary offset of the ZMP sine wave.
        period: Period of the ZMP sine wave, in seconds.
        omega: LIP natural frequency, in rad/s.

    Returns:
        ZMP and DCM references at time t.
    """
    nu = 2.0 * np.pi / period
    phase = nu * t
    zmp = amplitude * np.sin(phase) + offset
    shift = np.arctan2(nu, omega)
    dcm = amplitude * omega / np.sqrt(omega**2 + nu**2) * np.sin(phase + shift) + offset
    return zmp, dcm


def compute_zmp(
    xi: float,
    xi_ref: float,
    z_ref: float,
    kp: float = 1.0,
    omega: float = OMEGA_TRUE,
) -> float:
    """Compute the DCM feedback z = z_ref − (1 + kp/ω)(ξ_ref − ξ).

    Args:
        xi: Measured DCM.
        xi_ref: DCM reference.
        z_ref: ZMP reference.
        kp: DCM feedback gain, in rad/s.
        omega: LIP natural frequency, in rad/s.

    Returns:
        Commanded ZMP, clipped to [−1, 1].
    """
    zmp_gain = 1.0 + kp / omega
    return float(np.clip(z_ref - zmp_gain * (xi_ref - xi), -1.0, 1.0))


def generate_data(
    num_steps: int,
    kp: float = 1.0,
    fall_threshold: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the 1-D LIP under a varied sinusoidal curriculum.

    Parameters are re-drawn from PERIODS / AMPLITUDES / OFFSETS every
    CHUNK_STEPS steps.  Varied frequencies are required for ω to be
    identifiable: with a single frequency there are approximate local minima
    for any (learned DCM ≈ c + ċ/ω′, ω̂ = ω′).

    Args:
        num_steps: Number of simulation steps.
        kp: DCM feedback gain, in rad/s.
        fall_threshold: CoM position beyond which the LIP is reset to rest.
        seed: Seed for drawing reference parameters.

    Returns:
        Y: Observation array of shape (num_steps, OBS_DIM).
        U: ZMP command array of shape (num_steps,).
    """
    rng = np.random.default_rng(seed)
    c, cdot, t = 0.0, 0.0, 0.0
    Y = np.empty((num_steps, OBS_DIM), dtype=np.float32)
    U = np.empty(num_steps, dtype=np.float32)
    n_resets = 0
    period = float(rng.choice(PERIODS))
    amplitude = float(rng.choice(AMPLITUDES))
    offset = float(rng.choice(OFFSETS))

    for k in range(num_steps):
        if k % CHUNK_STEPS == 0 and k > 0:
            period = float(rng.choice(PERIODS))
            amplitude = float(rng.choice(AMPLITUDES))
            offset = float(rng.choice(OFFSETS))
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
