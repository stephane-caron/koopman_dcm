# SPDX-License-Identifier: Apache-2.0

"""1-D lateral DCM-state model predictive control.

State ``x = ξ`` (lateral DCM, normalized to [-1, 1]).
Input ``u = z`` (lateral ZMP, normalized).
Dynamics (discrete LIPM-DCM at the sampling period ``T``):

    ξ_{k+1} = α · ξ_k + (1 − α) · z_k,   α = exp(ω · T)

Hard inequality constraints:

    |ξ_k| ≤ ξ_max     (state viability, only when ``enforce_viability``)
    |z_k| ≤ z_max     (input feasibility)

Here ``zmp_line = ±1`` is a wheel contact, so the lateral support polygon edge
is at ``±1``. ``ξ_max`` is set strictly inside that edge to leave a safety
margin. It should be evaluated from static equilibrium tests, whereas z_max can
be evaluated from the dynamic capabilities of the FFDC.

Cost (quadratic): stage tracking ``(ξ − ξ_ref_k)²`` with weight ``Q``,
stage input tracking ``(z − z_ref_k)²`` with weight ``R``, and a terminal
cost ``(ξ_N − ξ_ref_N)²``. Both reference trajectories come from the same
analytic sinusoid as ``PatternGenerator.cycle``.
"""

import numpy as np
from qpmpc import MPCProblem, solve_mpc

from . import settings
from .dcm_controller import DcmController
from .interfaces.interface import Interface
from .pattern_generator import PatternGenerator

# How close a planned variable must come to its bound for the constraint
# to count as active. The QP satisfies constraints to solver tolerance, so
# an exactly-active constraint lands a hair inside the bound.
_CONSTRAINT_TOLERANCE = 1e-3

# Names of the cost sliders, shared by both variants so that the viability
# box is the only difference between them when they are compared.
SLIDER_Q = "mpc_Q"
SLIDER_R = "mpc_R"
SLIDER_Q_TERMINAL = "mpc_Q_terminal"


class DcmMpc(DcmController):
    """Receding-horizon 1-D lateral DCM-state MPC.

    Attributes:
        enforce_viability: Whether the DCM state box ``|ξ_k| ≤ xi_max``
            is included in the QP. False isolates the viability
            constraint's contribution by dropping it while keeping the
            horizon, cost weights and input box unchanged.
        horizon: Number of timesteps in the receding horizon.
        dcm_constraint_active: Whether the DCM viability constraint
            ``|ξ_k| ≤ xi_max`` is active over the horizon this cycle, in
            the usual sense of an active constraint: present in the QP and
            met with equality. Always False when ``enforce_viability`` is
            False, because there is then no such constraint to be active.
        leaning_velocity: Leaning velocity in m/s for the leg controller.
        predicted_peak: Largest ``|ξ_k|`` over the horizon of the plan
            returned by the solver this cycle, in normalized units. NaN
            when the solver failed.
        planned_input_peak: Largest ``|z_k|`` over the same plan. NaN when
            the solver failed.
        zmp_constraint_active: Whether the ZMP input constraint
            ``|z_k| ≤ z_max`` is active over the horizon this cycle. Unlike
            the DCM constraint this one is in the QP for both variants, so
            it is always a genuine activation rather than a counterfactual.
        Q: Stage state cost weight.
        Q_terminal: Terminal state cost weight.
        R: Stage input cost weight.
        xi_max: Half-width of the DCM interval, normalized between 0 and 1.
        z_max: Half-width of the ZMP input box, normalized between 0 and 1.
        zmp_line: Normalized line coordinate, in [-1, 1], of the commanded ZMP.
    """

    def __init__(
        self,
        horizon: int = 50,
        xi_max: float = 0.7,
        z_max: float = 0.9,
        Q: float = 1.0,
        # Input tracking is weighted on par with state tracking — the
        # demo we care about is "track the reference until the viability
        # constraint binds", not "regularise z toward zero".
        R: float = 1.0,
        Q_terminal: float = 1.0,
        solver: str = "proxqp",
        enforce_viability: bool = True,
    ) -> None:
        """Initialize the controller.

        Args:
            horizon: Number of timesteps in the receding horizon.
            xi_max: Half-width of the DCM state box, in normalized units.
            z_max: Half-width of the ZMP input box, in normalized units.
            Q: Stage state cost weight.
            R: Stage input cost weight.
            Q_terminal: Terminal state cost weight.
            solver: QP solver name passed to qpmpc.
            enforce_viability: Whether to include the DCM state box in
                the QP.
        """
        super().__init__()
        self.horizon = horizon
        self._xi_max = xi_max
        self._z_max = z_max
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.solver = solver
        self.enforce_viability = enforce_viability
        self._rebuild_constraints()

        # Admittance applied to (z_cmd − z_obs) → leaning_velocity. Same
        # semantics as DcmFeedbackController; read live from the shared
        # ``admittance`` slider once an interface is registered.
        self._admittance = settings.ADMITTANCE_DEFAULT

        # Diagnostics, refreshed every cycle. The two constraints are
        # tracked separately.
        self.predicted_peak = float("nan")
        self.planned_input_peak = float("nan")
        self.dcm_constraint_active = False
        self.zmp_constraint_active = False

    def _rebuild_constraints(self) -> None:
        """Rebuild the stacked inequality arrays from the current bounds.

        Called whenever ``xi_max`` or ``z_max`` changes, so that the two
        can never fall out of step with the arrays handed to the solver.
        """
        if self.enforce_viability:
            # Stacked state-box + input-box inequalities (4 rows per
            # timestep):
            #     [ +1,  0 ] [ξ_k]     [ xi_max]
            #     [ -1,  0 ] [z_k]  ≤  [ xi_max]
            #     [  0, +1 ]           [  z_max]
            #     [  0, -1 ]           [  z_max]
            self._ineq_state = np.array([[+1.0], [-1.0], [0.0], [0.0]])
            self._ineq_input = np.array([[0.0], [0.0], [+1.0], [-1.0]])
            self._ineq_vec = np.array(
                [self._xi_max, self._xi_max, self._z_max, self._z_max]
            )
        else:
            # Input-box only (2 rows per timestep); no state constraint.
            self._ineq_state = np.array([[0.0], [0.0]])
            self._ineq_input = np.array([[+1.0], [-1.0]])
            self._ineq_vec = np.array([self._z_max, self._z_max])

    @property
    def xi_max(self) -> float:
        """Half-width of the DCM state box, in normalized units."""
        return self._xi_max

    @xi_max.setter
    def xi_max(self, value: float) -> None:
        self._xi_max = float(value)
        self._rebuild_constraints()

    @property
    def z_max(self) -> float:
        """Half-width of the ZMP input box, in normalized units."""
        return self._z_max

    @z_max.setter
    def z_max(self, value: float) -> None:
        self._z_max = float(value)
        self._rebuild_constraints()

    def register_interface(self, interface: Interface) -> None:
        """Register the shared MPC cost sliders with the given interface.

        The two variants deliberately share one set of sliders: they are
        compared against each other, so their stage and terminal costs
        must stay equal and the viability box must remain the only
        difference between them. Whichever instance registers first
        creates the sliders; the other finds them and reads the same
        values.

        Args:
            interface: Active input interface.
        """
        super().register_interface(interface)
        if interface.has_slider(SLIDER_Q):
            return  # the other variant already created the shared set
        with interface.add_folder("MPC costs"):
            interface.add_slider(
                name=SLIDER_Q,
                label="Q (state)",
                min=1e-4,
                max=10.0,
                step=0.1,
                initial_value=self.Q,
            )
            interface.add_slider(
                name=SLIDER_R,
                label="R (input)",
                min=1e-4,
                max=10.0,
                step=0.001,
                initial_value=self.R,
            )
            interface.add_slider(
                name=SLIDER_Q_TERMINAL,
                label="Q_terminal",
                min=1e-4,
                max=100.0,
                step=0.1,
                initial_value=self.Q_terminal,
            )

    def reset(self) -> None:
        """Reset the controller outputs.

        There is no clock to rewind: the reference is evaluated off the
        pattern generator's own clock, so the two cannot drift apart.
        """
        super().reset()

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
            t: Time in seconds since the last reset.
            amplitude: ZMP oscillation amplitude.
            offset: ZMP offset.
            period: Stepping period in seconds.
            omega: Natural frequency of the linear inverted pendulum.

        Returns:
            Tuple of reference DCM and reference ZMP at time ``t``.
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
        pattern: PatternGenerator,
    ) -> tuple[float, float]:
        """Solve the receding-horizon problem for this cycle.

        Args:
            dt: Control timestep, which is also the MPC sampling period.
            omega: Eigenvalue of the divergent component, in rad/s.
            dcm_observed: Estimate of the normalized line DCM, in [-1, 1].
            zmp_observed: Estimate of the normalized line ZMP, in [-1, 1].
            pattern: Pattern generator, from which amplitude, offset and
                period are read; its clock is mirrored internally so the
                reference can be evaluated forward over the horizon.

        Returns:
            Tuple ``(zmp_line, leaning_velocity)``, also saved as
            attributes. The leaning velocity is derived as
            ``admittance · (z_cmd − z_obs)``, mirroring
            :class:`DcmFeedbackController`, so that the downstream leg
            controller is unchanged.
        """
        assert self._interface is not None, "call register_interface first"
        assert omega > 0.0, f"suspicious {omega=}"

        self.Q = self._interface.get_slider(SLIDER_Q)
        self.R = self._interface.get_slider(SLIDER_R)
        self.Q_terminal = self._interface.get_slider(SLIDER_Q_TERMINAL)
        self._admittance = self._interface.get_slider("admittance")

        alpha = float(np.exp(omega * dt))
        state_matrix = np.array([[alpha]])
        input_matrix = np.array([[1.0 - alpha]])

        # Build per-step reference trajectories (DCM and ZMP) by
        # forward-evaluation of the pattern's analytic sinusoid. Both are
        # length-N stage references; the terminal cost uses
        # goal_state = ξ_ref_N.
        target_states = np.zeros(self.horizon)
        target_inputs = np.zeros(self.horizon)
        for k in range(self.horizon):
            dcm_ref, zmp_ref = self._pattern_at(
                t=pattern.t + k * dt,
                amplitude=pattern.amplitude,
                offset=pattern.offset,
                period=pattern.period,
                omega=omega,
            )
            target_states[k] = dcm_ref
            target_inputs[k] = zmp_ref
        goal_state = np.array([target_states[-1]])

        if self.enforce_viability:
            # Clip the initial state to the feasible box: if the measured
            # DCM has already escaped (very aggressive pattern), the QP
            # would be infeasible from k=0. Clipping keeps the MPC active;
            # the next cycles' controls will still try to bring the real
            # DCM back.
            initial_state = np.array(
                [float(np.clip(dcm_observed, -self.xi_max, +self.xi_max))]
            )
        else:  # no state box to be infeasible against
            initial_state = np.array([float(dcm_observed)])

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
        if plan is not None and plan.states is not None:
            self.predicted_peak = float(np.max(np.abs(plan.states)))
            # A constraint absent from the QP cannot be active, however
            # close the trajectory comes to where it would have sat.
            self.dcm_constraint_active = self.enforce_viability and bool(
                self.predicted_peak >= self.xi_max - _CONSTRAINT_TOLERANCE
            )
        else:
            self.predicted_peak = float("nan")
            self.dcm_constraint_active = False
        if plan is not None and plan.inputs is not None:
            self.planned_input_peak = float(np.max(np.abs(plan.inputs)))
            self.zmp_constraint_active = bool(
                self.planned_input_peak >= self.z_max - _CONSTRAINT_TOLERANCE
            )
        else:
            self.planned_input_peak = float("nan")
            self.zmp_constraint_active = False

        if plan is None or plan.inputs is None:
            # Solver failed: fall back to the proportional DCM law, with
            # the same expression as DcmFeedbackController at kp = 1.
            kp = 1.0
            z_cmd = float(
                target_states[0]
                - (1.0 + kp / omega) * (target_states[0] - dcm_observed)
            )
        else:
            z_cmd = float(plan.inputs.flat[0])

        # Saturate to [-z_max, z_max]: the QP should already enforce this,
        # but numerical drift can leak past.
        z_cmd = float(np.clip(z_cmd, -self.z_max, self.z_max))
        self.zmp_line = z_cmd
        self.leaning_velocity = self._admittance * (z_cmd - zmp_observed)
        return self.zmp_line, self.leaning_velocity

    def to_log(self) -> dict:
        """Get instantaneous values for logging.

        Returns:
            Dictionary mapping signal names to their current values.
        """
        return {
            **super().to_log(),
            "predicted_peak": self.predicted_peak,
            "planned_input_peak": self.planned_input_peak,
            "dcm_constraint_active": self.dcm_constraint_active,
            "zmp_constraint_active": self.zmp_constraint_active,
        }
