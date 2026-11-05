# SPDX-License-Identifier: Apache-2.0

"""Model predictive control over the lateral DCM.

This is the MPC of Section 5 of the paper, with references from the same
sinusoid as in ``PatternGenerator``. When ``enforce_viability=False``, the DCM
state box is dropped while the rest of the MPC is unchanged: this is the "MPC
w/o viability" controller of Table 2.
"""

import numpy as np
from qpmpc import MPCProblem, solve_mpc

from . import settings


class DcmMpc:
    """Receding-horizon 1-D lateral DCM-state MPC.

    Attributes:
        xi_max: Half-width of the DCM state box, in normalized units.
            Inert when ``enforce_viability`` is False.
        z_max: Half-width of the ZMP input box, in normalized units.
        horizon: Number of timesteps in the receding horizon.
        Q: Stage state cost weight.
        R: Stage input cost weight.
        Q_terminal: Terminal state cost weight.
        enforce_viability: Whether the DCM state box ``|ξ_k| ≤ xi_max``
            is included in the QP. False isolates the viability
            constraint's contribution by dropping it while keeping the
            horizon, cost weights and input box unchanged.
    """

    def __init__(
        self,
        horizon: int = 50,
        xi_max: float = 0.85,
        z_max: float = 1.0,
        Q: float = 1.0,
        R: float = 1.0,
        Q_terminal: float = 1.0,
        solver: str = "proxqp",
        enforce_viability: bool = True,
        key_prefix: str = "mpc",
    ) -> None:
        """Initialize the MPC.

        Args:
            horizon: Number of timesteps in the receding horizon.
            xi_max: Half-width of the DCM state box, in normalized units.
            z_max: Half-width of the ZMP input box, in normalized units.
            Q: Stage state cost weight.
            R: Stage input cost weight. Input tracking is weighted on par
                with state tracking, so that the MPC tracks the reference
                until the viability constraint binds.
            Q_terminal: Terminal state cost weight.
            solver: QP solver passed to ``qpmpc``.
            enforce_viability: Whether the DCM state box is included in
                the QP.
            key_prefix: Prefix of the slider names, distinguishing the two
                MPCs of the balancer.
        """
        self.horizon = horizon
        self.xi_max = xi_max
        self.z_max = z_max
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.solver = solver
        self.enforce_viability = enforce_viability
        self.key_prefix = key_prefix

        if enforce_viability:
            # Stacked state-box + input-box inequalities (4 rows per
            # timestep):
            #     [ +1,  0 ] [ξ_k]     [ xi_max]
            #     [ -1,  0 ] [z_k]  ≤  [ xi_max]
            #     [  0, +1 ]           [  z_max]
            #     [  0, -1 ]           [  z_max]
            self._ineq_state = np.array([[+1.0], [-1.0], [0.0], [0.0]])
            self._ineq_input = np.array([[0.0], [0.0], [+1.0], [-1.0]])
            self._ineq_vec = np.array([xi_max, xi_max, z_max, z_max])
        else:
            # Input-box only (2 rows per timestep); no state constraint.
            self._ineq_state = np.array([[0.0], [0.0]])
            self._ineq_input = np.array([[+1.0], [-1.0]])
            self._ineq_vec = np.array([z_max, z_max])

        # Time counter mirroring PatternGenerator's internal clock so we
        # can evaluate the reference forward over the horizon.
        self._t = 0.0
        # Admittance applied to (z_cmd − z_obs) → leaning_velocity. Same
        # semantics as DcmController; read live from the shared
        # ``admittance`` slider when an interface is registered.
        self._admittance = settings.ADMITTANCE_DEFAULT
        self._interface = None

    def register_interface(self, interface) -> None:
        """Register MPC tuning sliders with the given interface.

        Adds three sliders that override ``self.Q``
        (stage state cost), ``self.R`` (stage input cost), and
        ``self.Q_terminal`` (terminal state cost) at the start of every
        ``cycle``.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        interface.add_slider(
            name=f"{self.key_prefix}_Q",
            label="Q (state)",
            min=1e-4,
            max=10.0,
            step=0.1,
            initial_value=self.Q,
        )
        interface.add_slider(
            name=f"{self.key_prefix}_R",
            label="R (input)",
            min=1e-4,
            max=10.0,
            step=0.001,
            initial_value=self.R,
        )
        interface.add_slider(
            name=f"{self.key_prefix}_Q_terminal",
            label="Q_terminal",
            min=1e-4,
            max=100.0,
            step=0.1,
            initial_value=self.Q_terminal,
        )

    def reset(self) -> None:
        self._t = 0.0

    def _pattern_at(
        self,
        t: float,
        amplitude: float,
        offset: float,
        period: float,
        omega: float,
    ) -> tuple[float, float]:
        """Return (dcm_ref, zmp_ref) at time ``t``, in normalized units.

        Same formulas as ``PatternGenerator.cycle``.

        Args:
            t: Time in seconds.
            amplitude: Amplitude of the ZMP reference.
            offset: Offset of the ZMP reference.
            period: Period of the ZMP reference, in seconds.
            omega: Natural frequency of the DCM dynamics, in rad/s.

        Returns:
            DCM and ZMP references at time ``t``.
        """
        nu = 2.0 * np.pi / max(period, 1e-6)
        phase = nu * t
        zmp = amplitude * np.sin(phase) + offset
        shift = np.arctan2(nu, omega)
        dcm_amplitude = amplitude * omega / np.sqrt(omega**2 + nu**2)
        dcm = dcm_amplitude * np.sin(phase + shift) + offset
        return float(dcm), float(zmp)

    def cycle(
        self,
        dt: float,
        omega: float,
        dcm_observed: float,
        zmp_observed: float,
        pattern,
    ) -> tuple[float, float]:
        """Solve the receding-horizon problem.

        Args:
            dt: Control timestep (also the MPC sampling period).
            omega: Current LIPM natural frequency, rad/s.
            dcm_observed: Measured DCM_y, normalized.
            zmp_observed: Measured ZMP_y, normalized.
            pattern: ``PatternGenerator`` instance (read amplitude / offset
                / period; ``_t`` is mirrored internally for forward eval).

        Returns:
            ``(z_cmd, leaning_velocity)`` — both in normalized units. The
            leaning velocity is ``admittance · (z_cmd − z_obs)``, as in
            ``DcmController``.
        """
        # Update Q/R/Q_terminal + admittance from sliders if registered.
        if self._interface is not None:
            self.Q = self._interface.get_slider(f"{self.key_prefix}_Q")
            self.R = self._interface.get_slider(f"{self.key_prefix}_R")
            self.Q_terminal = self._interface.get_slider(
                f"{self.key_prefix}_Q_terminal"
            )
            self._admittance = self._interface.get_slider("admittance")

        alpha = float(np.exp(omega * dt))
        state_matrix = np.array([[alpha]])
        input_matrix = np.array([[1.0 - alpha]])

        # Build per-step reference trajectories (DCM and ZMP) by
        # forward-evaluation of the pattern's analytic sinusoid. Both
        # are length-N stage references; the terminal cost uses
        # goal_state = ξ_ref_N.
        target_states = np.zeros(self.horizon)
        target_inputs = np.zeros(self.horizon)
        for k in range(self.horizon):
            dcm_ref, zmp_ref = self._pattern_at(
                t=self._t + (k + 1) * dt,
                amplitude=pattern.amplitude,
                offset=pattern.offset,
                period=pattern.period,
                omega=omega,
            )
            target_states[k] = dcm_ref
            target_inputs[k] = zmp_ref
        # Terminal target = state reference at horizon end.
        goal_state = np.array([target_states[-1]])
        self._t += dt

        if self.enforce_viability:
            # Clip initial state to the feasible box: if the measured DCM
            # has already escaped (very aggressive pattern), the QP would
            # be infeasible from k=0. Clipping keeps the MPC active; the
            # next cycles' controls will still try to bring the real DCM
            # back.
            initial_state = np.array(
                [float(np.clip(dcm_observed, -self.xi_max, +self.xi_max))]
            )
        else:
            # No state box to be infeasible against.
            initial_state = np.array([float(dcm_observed)])

        # Build the MPC problem. ineq_vector is repeated (constant box).
        problem = MPCProblem(
            transition_state_matrix=state_matrix,
            transition_input_matrix=input_matrix,
            ineq_state_matrix=self._ineq_state,
            ineq_input_matrix=self._ineq_input,
            ineq_vector=[self._ineq_vec] * self.horizon,
            nb_timesteps=self.horizon,
            terminal_cost_weight=self.Q_terminal,
            stage_state_cost_weight=self.Q,
            stage_input_cost_weight=self.R,
            initial_state=initial_state,
            goal_state=goal_state,
            target_states=target_states,
        )
        plan = solve_mpc(problem, solver=self.solver)
        if plan is None or plan.inputs is None:
            # Solver failed: fall back to the DCM-tracking proportional
            # law (same as DcmController's expression).
            kp = 1.0
            z_cmd = float(
                target_states[0]
                - (1.0 + kp / omega) * (target_states[0] - dcm_observed)
            )
        else:
            z_cmd = float(plan.inputs.flat[0])

        # Saturate to [-z_max, z_max] (the QP should already enforce this
        # but numerical drift can leak past).
        z_cmd = float(np.clip(z_cmd, -self.z_max, self.z_max))
        leaning_velocity = self._admittance * (z_cmd - zmp_observed)
        return z_cmd, leaning_velocity
