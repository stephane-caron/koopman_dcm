# SPDX-License-Identifier: Apache-2.0

"""Per-cycle orchestrator for the lateral balancer.

DCM feedback is only for lateral balancing, the focus of this study. Sagittal
balancing is handled separately by a ``SagittalStabilizer``.
"""

from dataclasses import dataclass

import numpy as np

from . import settings
from .dcm_controller import DcmController
from .dcm_mpc import DcmMpc
from .height_difference import HeightDifference
from .ik_controller import IKController
from .interface import Interface
from .pattern_generator import PatternGenerator
from .pinker_bridge import PinkerBridge
from .pybullet_bridge import PyBulletBridge, PyBulletObservation
from .robots.base import RobotSpec
from .sagittal_stabilizer import SagittalStabilizer


@dataclass
class State:
    """Estimated state passed between modules each cycle."""

    com_world: np.ndarray
    com_dot_world: np.ndarray
    omega: float
    zmp_world: np.ndarray
    com_line: float
    zmp_line: float
    dcm_y: float
    anchor_world: np.ndarray
    base_z: float


class Balancer:
    """Main control loop."""

    def __init__(
        self,
        spec: RobotSpec,
        controller: str = "dcm",
        interface: Interface | None = None,
    ) -> None:
        """Build the bridges, modules, and seed initial state.

        Args:
            spec: Robot specification.
            controller: Controller ``"dcm"`` (proportional DCM
                feedback), ``"mpc"`` (DCM-state MPC with the viability
                box), or ``"viability-less-mpc"`` (same MPC with the DCM
                state box ``|ξ| ≤ ξ_max`` removed — isolates the
                viability constraint's contribution from the MPC's other
                ingredients).
            interface: Interface providing the slider values, e.g. the
                ZMP reference of data collection. Defaults to a base
                ``Interface`` with fixed initial values.
        """
        self.spec = spec
        self.controller_name = controller

        self.pb = PyBulletBridge(spec)
        self.pinker = PinkerBridge(spec)

        self.interface = interface if interface is not None else Interface()
        self.pattern = PatternGenerator()
        self.dcm_lat = DcmController()
        self.dcm_mpc = DcmMpc(horizon=50)
        self.dcm_mpc_no_viability = DcmMpc(
            horizon=50, enforce_viability=False, key_prefix="mpc_nv"
        )
        self.pattern.register_interface(self.interface)
        self.dcm_lat.register_interface(self.interface)
        self.dcm_mpc.register_interface(self.interface)
        self.dcm_mpc_no_viability.register_interface(self.interface)

        self.ik = IKController(self.pinker, root_frame="pelvis")
        self.height_diff = HeightDifference(max_value=0.06)
        self.sagittal_stabilizer = SagittalStabilizer()

        self._prev_com: np.ndarray | None = None
        self._anchor_filtered: np.ndarray | None = None
        self._anchor_alpha = 2 * np.pi * settings.ANCHOR_LPF_CUTOFF * settings.DT
        # Optional learned DCM fed to the controllers in place of the LIP
        # DCM ``state.dcm_y``, set by the evaluation. ``None`` uses the
        # LIP DCM.
        self._dcm_y_override: float | None = None

    def reset(self) -> None:
        """Reset PyBullet, settle, then anchor IK targets."""
        self.pb.reset()
        q_hold = np.array(
            [self.spec.standing_posture[n] for n in self.spec.actuated_joint_names]
        )
        v_zero = np.zeros_like(q_hold)
        for _ in range(int(0.5 / settings.DT)):  # 0.5 s settle
            self.pb.apply_action(q_hold, v_zero)
            self.pb.step()
        obs = self.pb.observe()
        self.pinker.update_configuration(
            obs.base_position, obs.base_quat_xyzw, obs.q_actuated
        )
        self.ik.set_initial_targets()

        lf, rf = self.pinker.get_foot_transforms()
        self.height_diff.reset()
        self.pattern.reset()
        self.dcm_lat.reset()
        self.sagittal_stabilizer.anchor(
            base_position=obs.base_position,
            base_quat_xyzw=obs.base_quat_xyzw,
        )
        self._prev_com = self.pinker.compute_com().copy()
        self._anchor_filtered = 0.5 * (lf.translation + rf.translation)

    def _estimate_state(self, obs: PyBulletObservation) -> State:
        """Estimate the CoM, ZMP and LIP DCM from a simulator observation.

        Args:
            obs: Simulator observation.

        Returns:
            Estimated state.
        """
        com = self.pinker.compute_com()
        assert self._prev_com is not None
        com_dot = (com - self._prev_com) / settings.DT
        self._prev_com = com.copy()

        fl = obs.left_contact.total_fz
        fr = obs.right_contact.total_fz
        if fl + fr > settings.MIN_TOTAL_FZ:
            zmp_world = (
                fl * obs.left_contact.position + fr * obs.right_contact.position
            ) / (fl + fr)
        else:
            zmp_world = 0.5 * (obs.left_contact.position + obs.right_contact.position)

        lf, rf = self.pinker.get_foot_transforms()
        anchor_inst = 0.5 * (lf.translation + rf.translation)
        a = float(np.clip(self._anchor_alpha, 0.0, 1.0))
        assert self._anchor_filtered is not None
        self._anchor_filtered = (1.0 - a) * self._anchor_filtered + a * anchor_inst
        anchor = self._anchor_filtered

        com_height = max(com[2] - anchor[2], 0.1)
        omega = float(np.sqrt(settings.GRAVITY / com_height))

        dcm_world = com + com_dot / omega

        half = self.spec.half_foot_spacing
        com_line = (com[1] - anchor[1]) / half
        zmp_line = (zmp_world[1] - anchor[1]) / half
        dcm_y_line = (dcm_world[1] - anchor[1]) / half

        return State(
            com_world=com,
            com_dot_world=com_dot,
            omega=omega,
            zmp_world=zmp_world,
            com_line=com_line,
            zmp_line=zmp_line,
            dcm_y=dcm_y_line,
            anchor_world=anchor,
            base_z=float(obs.base_position[2]),
        )

    def cycle(self) -> tuple[State, float, bool]:
        """Run one control cycle.

        Returns:
            ``(state, u_lateral, fell)``. ``u_lateral`` is the
            recorded action (normalised lateral commanded ZMP).
        """
        dt = settings.DT
        obs = self.pb.observe()
        self.pinker.update_configuration(
            obs.base_position, obs.base_quat_xyzw, obs.q_actuated
        )
        state = self._estimate_state(obs)

        # 1) Pattern generator drives the lateral targets.
        self.pattern.cycle(dt, state.omega)

        # 2) Lateral DCM feedback. All controllers consume the same DCM:
        # ``_dcm_y_override`` replaces the LIP DCM by the learned DCM
        # when set by the evaluation.
        dcm_obs = state.dcm_y if self._dcm_y_override is None else self._dcm_y_override
        if self.controller_name == "mpc":
            z_cmd_y, lean_y = self.dcm_mpc.cycle(
                dt,
                omega=state.omega,
                dcm_observed=dcm_obs,
                zmp_observed=state.zmp_line,
                pattern=self.pattern,
            )
        elif self.controller_name == "viability-less-mpc":
            z_cmd_y, lean_y = self.dcm_mpc_no_viability.cycle(
                dt,
                omega=state.omega,
                dcm_observed=dcm_obs,
                zmp_observed=state.zmp_line,
                pattern=self.pattern,
            )
        else:
            self.dcm_lat.cycle(
                dt,
                omega=state.omega,
                dcm_target=self.pattern.dcm,
                dcm_observed=dcm_obs,
                zmp_target=self.pattern.zmp,
                zmp_observed=state.zmp_line,
            )
            # Clamp the commanded ZMP to the normalized support polygon
            # ``[-1, 1]`` (the DcmController doesn't bound its output),
            # and recompute the leaning velocity from the clamped value
            # so that the height difference integrates a feasible ZMP.
            z_cmd_y = float(np.clip(self.dcm_lat.zmp_line, -1.0, 1.0))
            admittance = self.interface.get_slider("admittance")
            lean_y = admittance * (z_cmd_y - state.zmp_line)

        # 3) Foot height difference: integrate leaning velocity (m/s).
        # leaning_velocity from DcmController is in normalized units; the
        # admittance slider has units 1/s, so leaning_velocity is also
        # dimensionless [-1,1]/s. Multiply by half_foot_spacing to get
        # m/s, then integrate into height_difference (m).
        self.height_diff.integrate(
            leaning_velocity=lean_y * self.spec.half_foot_spacing, dt=dt
        )

        # 4) Foot targets: static DSP, anchored foot poses + height diff.
        self.ik.set_foot_heights(self.height_diff.value)

        # 5) IK solve + integrate.
        ik_velocity = self.ik.solve(dt, solver="proxqp")

        # 6) Apply POSITION_CONTROL.
        q_target = self.pinker.get_q_actuated()
        v_target = self.pinker.get_v_actuated_from_ik(ik_velocity)
        self.pb.apply_action(q_target, v_target)

        # 6b) Sagittal stabilizer: external world-x force + world-z torque
        # on the base link. Out-of-scope axis only; cannot contaminate
        # the lateral study by construction.
        force_x, torque_z = self.sagittal_stabilizer.compute(
            base_position=obs.base_position,
            base_quat_xyzw=obs.base_quat_xyzw,
            base_linvel_world=obs.base_linvel_world,
            base_angvel_world=obs.base_angvel_world,
        )
        self.pb.apply_external_disturbance(
            base_position=obs.base_position,
            force_x=force_x,
            torque_z=torque_z,
        )

        # 7) Step the simulation.
        self.pb.step()
        post_obs = self.pb.observe()
        fell = self.pb.has_fallen(post_obs)

        return state, float(z_cmd_y), fell

    def close(self) -> None:
        self.pb.close()
