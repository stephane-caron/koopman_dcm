# SPDX-License-Identifier: Apache-2.0

"""State estimation from joint encoders, IMU and joint torques."""

import os
import time
from typing import Dict, Optional

import cookie_description
import numpy as np
import pinker
import upkie.model
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask, RollingTask
from qpsolvers import solve_qp
from upkie.logging import logger
from upkie.utils.filters import low_pass_filter

from . import settings
from .interfaces import Interface
from .observation import make_observation
from .standing_posture import make_standing_posture
from .utils import RunningMeanStd, StructuredVector

# Standard gravity, in m/s².
GRAVITY: float = 9.81


def compute_generalized_gravity(
    model: kin.Model, data: kin.Data, q: np.ndarray
) -> np.ndarray:
    """Compute the generalized gravity vector g(q) of the robot.

    The gradient of the potential energy V(q) = m g c_z(q) is g(q) = m g
    J_{com}(q)^T e_z, where m is the total mass, J_{com} is the CoM Jacobian,
    and e_z is the unit vector pointing upward. See for instance "On the
    inertially decoupled structure of the floating base robot dynamics"
    (Garofalo et al., IFAC 2015).

    Args:
        model: Kinematic model of the robot.
        data: Data of the model, updated in place by the computation.
        q: Configuration vector.

    Returns:
        Generalized gravity vector, of dimension ``model.nv``.
    """
    J_com = kin.jacobian_center_of_mass(model, data, q)
    return data.mass[0] * GRAVITY * J_com[2]


class LipDcm:
    """Analytic LIP DCM evaluated at a fixed natural frequency.

    Used to evaluate the learned-LIP ablation: the observable keeps the
    closed-form expression ξ = c + ċ/ω from the linear inverted pendulum model,
    with ω a constant fitted offline to recorded robot data rather than the
    kinematic √(g/h) recomputed at every cycle in the state estimator.

    Exposes the same ``dcm_line`` / ``omega`` pair as the trained
    ``OnnxDcmObservable``, so the balancer can select either interchangeably.

    Attributes:
        dcm_line: Latest DCM line coordinate, in [-1, 1].
        omega: Fixed natural frequency, in rad/s.
    """

    dcm_line: float
    omega: float

    def __init__(self, omega: float) -> None:
        """Initialize the fixed-frequency LIP DCM.

        Args:
            omega: Natural frequency, in rad/s. Must be positive.
        """
        if not omega > 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        self.dcm_line = 0.0
        self.omega = omega

    def reset(self) -> None:
        """Reset the estimated DCM."""
        self.dcm_line = 0.0

    def update(self, com_line: float, comd_line: float) -> None:
        """Update the DCM from the latest CoM line coordinate and velocity.

        The line coordinate is affine in the world position of a point, so
        projecting ξ = c + ċ/ω onto the contact line commutes with the
        projection: the line DCM is exactly ``com_line + comd_line / ω``.

        Args:
            com_line: Normalized lateral CoM coordinate, in [-1, 1].
            comd_line: Time derivative of ``com_line``, in s^-1.
        """
        self.dcm_line = com_line + comd_line / self.omega


class Contact:
    """Estimated state of a wheel contact.

    Attributes:
        force_in_contact: Contact force in the contact frame, in N.
        position_in_anchor: Contact position in the anchor frame, in m.
        position_in_world: Contact position in the world frame, in m.
        rotation_contact_to_world: Rotation matrix from the contact frame to
            the world frame.
    """

    force_in_contact: np.ndarray
    position_in_anchor: np.ndarray
    position_in_world: np.ndarray
    rotation_contact_to_world: np.ndarray

    def __init__(self):
        """Initialize a contact at the origin, without force."""
        self.force_in_contact = np.zeros(3)
        self.position_in_anchor = np.zeros(3)
        self.position_in_world = np.zeros(3)
        self.rotation_contact_to_world = np.eye(3)


class StateEstimator:
    """Estimate full generalized coordinates from sensor observations.

    Attributes:
        base_task: Task to track base orientation.
        com_anchor: Latest CoM position in the anchor frame.
        com_line: Latest CoM line coordinate, in [-1, 1].
        configuration: Pinker configuration for inverse kinematics.
        contacts: Latest contact state keyed by side ("left", "right").
        dcm_anchor: Latest DCM position in the anchor frame.
        dcm_line: Latest DCM line coordinate, in [-1, 1].
        joint_idx_q: Mapping from joint names to configuration-vector indices
            for the kinematic model.
        joint_idx_v: Mapping from joint names to tangent-vector indices
            for the kinematic model.
        left_wheel_task: Left wheel rolling-contact task.
        lip_dcm: LIP DCM estimator.
        kin_model: Kinematic model of the robot, with a free-flyer base.
        nq: Number of generalized coordinates of the kinematic model.
        nv: Number of velocity coordinates of the kinematic model.
        observation: Latest observation as a structured vector.
        omega: Expected value for the natural frequency of a corresponding
            linear inverted pendulum model, tuned.
        posture_task: Task to track observed joint angles.
        q_init: Default standing-posture configuration vector.
        right_wheel_task: Right wheel rolling-contact task.
        sigma_actuated: Weight on the actuated-joint residual in the
            contact-force estimation problem.
        sigma_floating: Weight on the floating-base residual in the
            contact-force estimation problem.
        transform_anchor_to_world: SE3 transform from anchor frame to world.
        visual_model: Visual geometry model of the robot.
        zmp_anchor: Latest ZMP position in the anchor frame.
        zmp_line: Latest ZMP line coordinate, in [-1, 1].
    """

    base_task: FrameTask
    com_anchor: Optional[np.ndarray]
    com_line: float
    configuration: pinker.Configuration
    contacts: Dict[str, Contact]
    dcm_anchor: Optional[np.ndarray]
    dcm_line: float
    joint_idx_q: dict[str, int]
    joint_idx_v: dict[str, int]
    left_wheel_task: RollingTask
    lip_dcm: LipDcm
    nv: int
    observation: StructuredVector
    omega: Optional[float]
    posture_task: PostureTask
    q_init: np.ndarray
    right_wheel_task: RollingTask
    sigma_actuated: float
    sigma_floating: float
    transform_anchor_to_world: kin.SE3
    zmp_anchor: Optional[np.ndarray]
    zmp_line: float

    # Sigma/task-cost sliders are tuned separately (not through the runtime
    # interface); flip this to re-enable exposing them as sliders.
    _TUNABLE_VIA_INTERFACE: bool = False

    def __init__(
        self,
        model: upkie.model.Model,
        lip_omega: float = settings.LEARNED_LIP_OMEGA,
    ) -> None:
        """Initialize state estimator.

        Args:
            model: Upkie model.
            lip_omega: Fixed natural frequency of the learned-LIP ablation
                exposed as ``lip_dcm``, in rad/s.
        """
        self.joint_names = model.joint_names

        # Load robot wrapper from URDF
        robot_wrapper = pinker.load_robot_urdf(
            cookie_description.URDF_PATH,
            root_joint="free_flyer",
            package_dirs=[os.path.dirname(cookie_description.PATH)],
        )

        # Initialize configuration at standing pose
        q_init = make_standing_posture(robot_wrapper)
        q_init[2] = 0.572  # z so that contact frames land at z=0
        self.configuration = pinker.Configuration(
            robot_wrapper.model, robot_wrapper.data, q_init
        )

        # Set up kinematic tasks
        self.posture_task = PostureTask(
            cost=[1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )
        self.base_task = FrameTask(
            "base",
            orientation_cost=[1.0, 1.0, 0.1],  # orientation tracking
            position_cost=[0.0, 0.0, 0.0],  # no position tracking
        )
        self.left_wheel_task = RollingTask(
            "left_wheel_hub",
            floor_frame="universe",
            wheel_radius=model.wheel_radius,
            cost=[1.0, 1.0, 1.0],
        )
        self.right_wheel_task = RollingTask(
            "right_wheel_hub",
            floor_frame="universe",
            wheel_radius=model.wheel_radius,
            cost=[1.0, 1.0, 1.0],
        )

        # Build joint index map: joint name → index in q vector
        self.joint_idx_q = {
            name: robot_wrapper.model.joints[
                robot_wrapper.model.get_joint_id(name)
            ].idx_q
            for name in model.joint_names
        }
        self.joint_idx_v = {
            name: robot_wrapper.model.joints[
                robot_wrapper.model.get_joint_id(name)
            ].idx_v
            for name in model.joint_names
        }
        self.leg_joint_idx_v = {
            leg: [
                self.joint_idx_v[joint]
                for joint in (f"{leg}_hip", f"{leg}_knee", f"{leg}_wheel")
            ]
            for leg in ("left", "right")
        }

        # Quick and dirty perf timers
        self._perf_samples: dict[str, RunningMeanStd] = {
            key: RunningMeanStd()
            for key in (
                "kinematics",
                "contacts",
                "com-dcm-zmp",
            )
        }

        self._interface = None
        self._comd_in_world: Optional[np.ndarray] = None
        self._prev_com_in_world: Optional[np.ndarray] = None
        self.com_anchor = None
        self.com_line = 0.0
        self.comd_line = 0.0
        self.contacts = {"left": Contact(), "right": Contact()}
        self.dcm_anchor = None
        self.dcm_line = 0.0
        self.lip_dcm = LipDcm(lip_omega)
        self.observation = make_observation()
        self.omega = None
        self.kin_model = robot_wrapper.model
        self.q_init = q_init
        self.sigma_actuated = 1.0
        self.sigma_floating = 1.0
        self.transform_anchor_to_world = kin.SE3.Identity()
        self.visual_model = robot_wrapper.visual_model
        self.zmp_anchor = None
        self.zmp_line = 0.0

    @property
    def q_actuated(self) -> np.ndarray:
        """Joint angles of the actuated joints, in rad."""
        return self.configuration.q[7:]

    def reset(self, interface: Optional[Interface]) -> None:
        """Reset state estimator to initial configuration.

        Args:
            interface: Registered interface, as a reminder that it needs to be
                reset before the call to this reset method.
        """
        self._comd_in_world = None
        self._prev_com_in_world = None
        self.lip_dcm.reset()
        self.configuration.update(self.q_init)
        if self._interface is not None:
            assert self._interface == interface
            if hasattr(self._interface, "visualizer"):
                self._interface.visualizer.display(self.configuration.q)

    def register_interface(self, interface: Interface) -> None:
        """Register plots with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface

        if self._TUNABLE_VIA_INTERFACE:
            _posture_cost_init = float(
                np.log10(next(c for c in self.posture_task.cost if c > 0))
            )
            _base_ori_cost_init = float(
                np.log10(self.base_task.orientation_cost[0])
            )
            _rolling_xy_cost_init = float(
                np.log10(self.left_wheel_task.cost[0])
            )
            _rolling_z_cost_init = float(
                np.log10(self.left_wheel_task.cost[2])
            )
            with interface.add_folder(
                "State estimator", expand_by_default=False
            ):
                interface.add_slider(
                    name="sigma_act",
                    label="F: σ_act (log10)",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=float(np.log10(self.sigma_actuated)),
                )
                interface.add_slider(
                    name="sigma_float",
                    label="F: σ_float (log10)",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=float(np.log10(self.sigma_floating)),
                )
                interface.add_slider(
                    name="base_ori_cost",
                    label="IK: base orientation (log10)",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=_base_ori_cost_init,
                )
                interface.add_slider(
                    name="posture_cost",
                    label="IK: posture (log10)",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=_posture_cost_init,
                )
                interface.add_slider(
                    name="rolling_xy_cost",
                    label="IK: rolling xy (log10)",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial_value=_rolling_xy_cost_init,
                )
                interface.add_slider(
                    name="rolling_z_cost",
                    label="IK: rolling z (log10)",
                    min=-3.0,
                    max=6.0,
                    step=0.1,
                    initial_value=_rolling_z_cost_init,
                )

        if not hasattr(interface, "viser_server"):
            return

        self._scene = interface.viser_server.scene
        self._force_handles: dict[str, Optional[object]] = {
            "left": None,
            "right": None,
        }
        self._contact_frame_handles = {
            "left": self._scene.add_frame(
                "/wheel/left_contact", axes_length=0.08, axes_radius=0.003
            ),
            "right": self._scene.add_frame(
                "/wheel/right_contact", axes_length=0.08, axes_radius=0.003
            ),
        }
        self._dcm_world_handle = self._scene.add_icosphere(
            "/dcm/world",
            radius=0.015,
            color=(0.0, 0.3, 0.6),
            position=(0.0, 0.0, 0.0),
        )
        self._dcm_line_handle = self._scene.add_icosphere(
            "/dcm/line",
            radius=0.015,
            color=(0.0, 0.0, 0.8),
            position=(0.0, 0.0, 0.0),
        )
        self._zmp_target_handle = self._scene.add_icosphere(
            "/zmp/target",
            radius=0.015,
            color=(0.0, 0.8, 0.0),
            position=(0.0, 0.0, 0.0),
        )

    def cycle(self, dt: float, observation: dict) -> float:
        """Update state estimate from observation.

        Args:
            dt: Time step in seconds.
            observation: Sensor observation dictionary.
        """
        if self._TUNABLE_VIA_INTERFACE and self._interface is not None:
            sigma_act = self._interface.get_slider("sigma_act")
            sigma_float = self._interface.get_slider("sigma_float")
            self.sigma_actuated = 10.0**sigma_act
            self.sigma_floating = 10.0**sigma_float

            posture_cost = 10.0 ** self._interface.get_slider("posture_cost")
            self.posture_task.cost = posture_cost * np.array(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            )
            base_ori_cost = 10.0 ** self._interface.get_slider("base_ori_cost")
            self.base_task.set_orientation_cost(
                [base_ori_cost, base_ori_cost, base_ori_cost]
            )
            rolling_xy = 10.0 ** self._interface.get_slider("rolling_xy_cost")
            rolling_z = 10.0 ** self._interface.get_slider("rolling_z_cost")
            self.left_wheel_task.cost = [rolling_xy, rolling_xy, rolling_z]
            self.right_wheel_task.cost = [rolling_xy, rolling_xy, rolling_z]

        tau = self.get_torques(observation)

        _t0 = time.perf_counter()
        self.estimate_kinematics(dt, observation)
        _t1 = time.perf_counter()
        self.estimate_contact_positions()
        self.estimate_contact_forces(tau)
        _t2 = time.perf_counter()
        self.estimate_zmp()
        self.estimate_com_dcm(dt)
        _t3 = time.perf_counter()

        # Perf counters
        self._perf_samples["kinematics"].append(_t1 - _t0)
        self._perf_samples["contacts"].append(_t2 - _t1)
        self._perf_samples["com-dcm-zmp"].append(_t3 - _t2)

        # Write observation
        self.observation["q_actuated"] = self.q_actuated
        self.observation["com_anchor"] = self.com_anchor
        self.observation["com_line"] = self.com_line
        self.observation["tau"] = tau
        self.observation["zmp_anchor"] = self.zmp_anchor
        self.observation["zmp_line"] = self.zmp_line

    def get_torques(self, observation: dict) -> np.ndarray:
        """Read joint torque vector from observation dictionary.

        Args:
            observation: Sensor observation dictionary.

        Returns:
            Joint torque 6-dimensional vector.
        """
        return np.array(
            [
                observation["servo"][joint_name]["torque"]
                for leg in ("left", "right")
                for joint_name in (
                    f"{leg}_hip",
                    f"{leg}_knee",
                    f"{leg}_wheel",
                )
            ]
        )

    def estimate_kinematics(self, dt: float, observation: dict) -> None:
        """Update kinematics.

        Args:
            dt: Time step in seconds.
            observation: Sensor observation dictionary.
        """
        # Update posture-task target from servo positions
        q_target = self.configuration.q.copy()
        for joint_name, idx_q in self.joint_idx_q.items():
            q_target[idx_q] = observation["servo"][joint_name]["position"]
        self.posture_task.set_target(q_target)

        # Update base-task target from base orientation
        base_orientation = observation["base_orientation"]
        rotation_base_to_world = base_orientation["rotation_base_to_world"]
        target = self.configuration.get_transform_frame_to_world("base")
        target.rotation = np.array(rotation_base_to_world).reshape((3, 3))
        self.base_task.set_target(target)

        # Solve differential inverse kinematics
        velocity = solve_ik(
            self.configuration,
            [
                self.base_task,
                self.left_wheel_task,
                self.posture_task,
                self.right_wheel_task,
            ],
            dt,
            solver="proxqp",
            safety_break=True,
        )
        if velocity is None:
            raise RuntimeError("could not solve the differential IK problem")

        self.configuration.integrate_inplace(velocity, dt)

    def compute_anchor_frame(self) -> None:
        """Compute anchor frame from contact positions and update transform.

        The anchor frame origin is the midpoint between the two wheel contacts.
        Its y-axis points toward the left contact, its z-axis is the world
        z-axis, and its x-axis follows by right-hand rule (pointing forward).
        """
        left = self.contacts["left"].position_in_world
        right = self.contacts["right"].position_in_world
        origin = 0.5 * (left + right)
        y_axis = left - origin
        y_norm = np.linalg.norm(y_axis[:2])  # project to floor plane
        if y_norm > 1e-6:
            y_axis = np.array([y_axis[0], y_axis[1], 0.0]) / y_norm
        else:
            y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_axis, z_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        self.transform_anchor_to_world = kin.SE3(rotation, origin)

    def estimate_contact_positions(self) -> None:
        """Estimate wheel contact positions from the current configuration."""
        for side, task in (
            ("left", self.left_wheel_task),
            ("right", self.right_wheel_task),
        ):
            tf_hub = self.configuration.get_transform_frame_to_world(
                task.hub_frame
            )
            contact_pos = tf_hub.translation.copy()
            contact_pos[2] -= task.wheel_radius
            self.contacts[side].position_in_world = contact_pos

            # Rim frame has the same orientation as floor frame in the task
            self.contacts[side].rotation_contact_to_world = np.eye(3)

        self.compute_anchor_frame()
        for side in ("left", "right"):
            self.contacts[
                side
            ].position_in_anchor = self.transform_anchor_to_world.act_inv(
                self.contacts[side].position_in_world
            )

    def estimate_contact_forces(self, tau: np.ndarray) -> None:
        """Estimate 3D contact forces via QP solved with qpsolvers.

        Minimizes the sum of squared residuals on:

        - Actuated joints (left/right legs):
            J_la^T f_L + tau_L = g_la   and   J_ra^T f_R + tau_R = g_ra
        - Unactuated floating-base DoFs:
            J_lu^T f_L + J_ru^T f_R = g_u

        Subject to unilateral-contact inequality constraints

            f_L_z >= 0,   f_R_z >= 0,

        which encode the physical fact that the ground can push but
        not pull. Each normal force drops to zero on the side that has
        lifted off, so the four contact cases (no contact, left only,
        right only, both) emerge as different active sets of these
        inequalities.

        Args:
            tau: Joint torques of the six actuated joints, in N·m, ordered
                as the left leg (hip, knee, wheel) then the right leg.
        """
        q = self.configuration.q
        model = self.configuration.model
        data = self.configuration.data
        tau_left = tau[0:3]
        tau_right = tau[3:6]

        J_l = self.left_wheel_task.compute_jacobian(self.configuration)
        J_r = self.right_wheel_task.compute_jacobian(self.configuration)
        J_la = J_l[0:3, :][:, self.leg_joint_idx_v["left"]]  # (3, 3)
        J_ra = J_r[0:3, :][:, self.leg_joint_idx_v["right"]]  # (3, 3)
        J_lu = J_l[0:3, :][:, :6]  # (3, 6)
        J_ru = J_r[0:3, :][:, :6]  # (3, 6)

        g = compute_generalized_gravity(model, data, q)
        g_la = g[self.leg_joint_idx_v["left"]]
        g_ra = g[self.leg_joint_idx_v["right"]]
        g_u = g[:6]

        # Decision variable x = [f_L_x, f_L_y, f_L_z, f_R_x, f_R_y, f_R_z]
        zero_3x3 = np.zeros((3, 3))
        A_la = np.hstack([J_la.T, zero_3x3])  # (3, 6)
        A_ra = np.hstack([zero_3x3, J_ra.T])  # (3, 6)
        A_u = np.hstack([J_lu.T, J_ru.T])  # (6, 6)
        b_la = g_la - tau_left
        b_ra = g_ra - tau_right
        b_u = g_u

        # Stack residuals into ||A x - b||^2_W where W weights actuated and
        # floating-base coordinates according to their expected covariances,
        # see "Dynamic State Estimation using Quadratic Programming",
        # Xinjilefu, Siyuan Feng and Atkeson,  IROS 2014
        w_a = 1.0 / self.sigma_actuated  # std. dev. here, covariance in cost
        w_u = 1.0 / self.sigma_floating  # idem
        A = np.vstack([w_a * A_la, w_a * A_ra, w_u * A_u])
        b = np.hstack([w_a * b_la, w_a * b_ra, w_u * b_u])
        P = A.T @ A
        q_vec = -A.T @ b

        # Unilateral contact: -f_L_z <= 0 and -f_R_z <= 0
        G = np.array(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
        h = np.zeros(2)

        x = solve_qp(P, q_vec, G=G, h=h, solver="quadprog")
        self.contacts["left"].force_in_contact = x[0:3]
        self.contacts["right"].force_in_contact = x[3:6]

    def estimate_zmp(self) -> None:
        """Estimate the ZMP from the contact forces."""
        # Contact frames from rolling tasks have vertical z-axes
        left_force = self.contacts["left"].force_in_contact
        right_force = self.contacts["right"].force_in_contact
        vertical_resultant = left_force[2] + right_force[2]
        alpha_left = left_force[2] / vertical_resultant

        # Line coordinate is +1 for left and -1 for right
        # (same sign as the y-axis ZMP coordinate in the body frame)
        left_position = self.contacts["left"].position_in_world
        right_position = self.contacts["right"].position_in_world
        self.zmp_line = 2.0 * alpha_left - 1.0
        zmp_in_world = (
            alpha_left * left_position + (1.0 - alpha_left) * right_position
        )
        self.zmp_anchor = self.transform_anchor_to_world.act_inv(zmp_in_world)

    def estimate_com_dcm(self, dt: float) -> None:
        """Estimate the CoM and the kinematic DCM of the LIP.

        Args:
            dt: Time step since the last estimate, in seconds.
        """
        com_in_world = kin.center_of_mass(
            self.configuration.model,
            self.configuration.data,
            self.configuration.q,
        )
        if self._prev_com_in_world is None:
            self._prev_com_in_world = com_in_world
        comd_raw = (com_in_world - self._prev_com_in_world) / dt
        if self._comd_in_world is None:
            self._comd_in_world = comd_raw
        else:
            comd_cutoff_period = 0.03  # s
            self._comd_in_world = low_pass_filter(
                prev_output=self._comd_in_world,
                cutoff_period=comd_cutoff_period,
                new_input=comd_raw,
                dt=dt,
            )
        comd_in_world = self._comd_in_world

        # Parallel estimate of omega from kinematics
        # h is between 0.3 m and 0.4 m for Cookie
        # omega is between 5.0 rad/s² and 5.7 rad/s²
        com_in_left = com_in_world - self.contacts["left"].position_in_world
        h = com_in_left[2]
        if h < 0.3:
            logger.warning("CoM height h=%f is too low, clamping to h=0.3", h)
            h = 0.3
        elif h > 0.5:
            logger.warning("CoM height h=%f is too high, clamping to h=0.5", h)
            h = 0.5
        self.omega = np.sqrt(9.81 / h)

        dcm_in_world = com_in_world + comd_in_world / self.omega
        self._prev_com_in_world = com_in_world

        tf = self.transform_anchor_to_world
        self.com_anchor = tf.act_inv(com_in_world)
        self.dcm_anchor = tf.act_inv(dcm_in_world)

        left = self.contacts["left"].position_in_world[0:2]
        right = self.contacts["right"].position_in_world[0:2]
        squared_contact_dist = np.dot(right - left, right - left)

        dcm_2d = dcm_in_world[0:2]
        dcm_numerator = np.dot(dcm_2d - left, right - left)
        beta_left = 1.0 - dcm_numerator / squared_contact_dist
        # beta_left in [0, 1] if DCM projects to [left, right]
        self.dcm_line = 2.0 * beta_left - 1.0

        com_2d = com_in_world[0:2]
        com_numerator = np.dot(com_2d - left, right - left)
        gamma_left = 1.0 - com_numerator / squared_contact_dist
        # gamma_left in [0, 1] if CoM projects to [left, right]
        self.com_line = 2.0 * gamma_left - 1.0

        # Feeds the learned-LIP DCM below, and logging
        self.comd_line = (
            -2.0
            * np.dot(comd_in_world[0:2], right - left)
            / squared_contact_dist
        )

        # Same observable as dcm_line above, but at the fitted frequency
        # instead of the kinematic one
        self.lip_dcm.update(self.com_line, self.comd_line)

    def update_visualizer(
        self,
        dt: float,
        zmp_target: float,
        force_arrow_scale: float = 0.03,
    ) -> None:
        """Update Viser scene: robot mesh, rim frames, ZMP, and force arrows.

        Args:
            dt: Time step in seconds.
            zmp_target: Target normalized line ZMP in [-1, 1].
            force_arrow_scale: Scale to draw force arrows, in m/N.
        """
        if not hasattr(self._interface, "visualizer"):
            return

        self._interface.visualizer.display(self.configuration.q)

        for side in ("left", "right"):
            handle = self._contact_frame_handles[side]
            handle.position = self.contacts[side].position_in_world
            handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        def _line_to_world(zmp_line: float) -> np.ndarray:
            """Map a line coordinate to a point between the contacts.

            Args:
                zmp_line: Line coordinate, +1 at the left contact and -1 at
                    the right one.

            Returns:
                Position of the point in the world frame.
            """
            alpha_left = (zmp_line + 1.0) / 2.0
            left_position = self.contacts["left"].position_in_world
            right_position = self.contacts["right"].position_in_world
            return (
                alpha_left * left_position
                + (1.0 - alpha_left) * right_position
            )

        self._dcm_line_handle.position = _line_to_world(self.dcm_line)
        dcm_in_world = self.transform_anchor_to_world.act(self.dcm_anchor)
        self._dcm_world_handle.position = dcm_in_world
        self._zmp_target_handle.position = _line_to_world(zmp_target)

        force_colors = {
            "left": (0.27, 0.53, 1.0),
            "right": (1.0, 0.27, 0.27),
        }
        for side in ("left", "right"):
            if self._force_handles[side] is not None:
                self._force_handles[side].remove()
            start_in_world = self.contacts[side].position_in_world
            force_in_contact = self.contacts[side].force_in_contact
            rotation = self.contacts[side].rotation_contact_to_world
            force_in_world = rotation @ force_in_contact
            force_in_world *= np.array([0.0, 0.0, 1.0])  # only display f_z
            end_in_world = start_in_world + force_arrow_scale * force_in_world
            self._force_handles[side] = self._scene.add_line_segments(
                f"/forces/{side}",
                points=np.array([[start_in_world, end_in_world]]),
                colors=force_colors[side],
                line_width=3.0,
            )

    def to_log(self) -> dict:
        """Get instantaneous values for logging.

        Returns:
            Dictionary mapping signal names to their current values.
        """
        return {
            k: v
            for k, v in {
                "com_anchor": self.com_anchor,
                "com_line": self.com_line,
                "comd_line": self.comd_line,
                "dcm_line": self.dcm_line,
                "lip_dcm_line": self.lip_dcm.dcm_line,
                "lip_omega": self.lip_dcm.omega,
                "left_contact_force_z": float(
                    self.contacts["left"].force_in_contact[2]
                ),
                "omega": self.omega,
                "q_actuated": self.q_actuated,
                "right_contact_force_z": float(
                    self.contacts["right"].force_in_contact[2]
                ),
                "tau": self.observation["tau"],
            }.items()
            if v is not None
        }

    def log_perfs(self) -> None:
        """Log the duration of each step of the state estimation."""
        if not any(v.n > 0 for v in self._perf_samples.values()):
            return
        logger.info("  State estimator                   ")
        logger.info("-----------------                   ")
        for k, v in self._perf_samples.items():
            if v.n > 1:
                logger.info(
                    "%16s: %5.1f ± %5.1f ms  ",
                    k,
                    1e3 * v.average(),
                    1e3 * v.std(),
                )
