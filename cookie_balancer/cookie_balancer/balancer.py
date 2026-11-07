# SPDX-License-Identifier: Apache-2.0

"""Balancer of the Cookie wheeled biped, sagittal and lateral."""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Union

import numpy as np
from upkie.envs import UpkieServos
from upkie.logging import logger

from . import settings
from .dcm_controller import DcmController
from .dcm_feedback_controller import DcmFeedbackController
from .dcm_mpc import DcmMpc
from .interfaces import Interface, JoystickInterface
from .leg_controller import LegController
from .live_plots.dcm_zmp_plot import DcmZmpPlot
from .onnx_dcm_observable import OnnxDcmObservable
from .pattern_generator import NOMINAL_PERIOD, PatternGenerator
from .state_estimator import LipDcm, StateEstimator
from .utils.running_mean_std import RunningMeanStd
from .wheel_controller import WheelController

CONTROLLERS = ("dcm", "mpc", "viability-less-mpc")


class Balancer:
    """Lateral balancer agent.

    Attributes:
        controller_name: Key of the active DCM controller, one of
            ``CONTROLLERS``.
        dcm_observables: Learned DCMs keyed by model stem.
        dcm_feedback: Proportional DCM feedback controller.
        dcm_mpc: DCM-state MPC with the viability box.
        dcm_mpc_no_viability: Same MPC with the DCM state box removed.
        dcm_zmp_plot: DCM and ZMP live plot, also written to logs.
        dt: Control loop time step in seconds.
        interface: Active input interface.
        leg_controller: Differential-IK leg controller.
        pattern_generator: ZMP/DCM walking pattern generator.
        state_estimator: Kinematics and contact-force state estimator.
        wheel_controller: Ground-plane forward velocity controller.
    """

    def __init__(
        self,
        env: UpkieServos,
        interface_name: Literal["joystick", "none", "viser"] = "none",
        interface: Interface | None = None,
        dcm_models_dir: Path = settings.MODELS_DIR,
        initial_period: float = NOMINAL_PERIOD,
        supervised: bool = False,
        controller: str = "dcm",
        controller_locked: bool = False,
        crouch_height: float = settings.DEFAULT_CROUCH_HEIGHT,
    ) -> None:
        """Initialize the balancer.

        Args:
            env: Any Upkie servos environment.
            interface_name: Interface to use: "joystick", "viser", or "none".
            interface: Interface instance to use, overriding
                ``interface_name`` if not None.
            dcm_models_dir: Directory of the learned DCMs, loaded from their
                ONNX files.
            initial_period: Initial stepping period in seconds for the pattern
                generator slider.
            supervised: If True, add a "ready" selector that the operator must
                set to "yes" before the leg controller activates.
            controller: Initial DCM controller: ``"dcm"`` (proportional
                feedback), ``"mpc"`` (DCM-state MPC with the viability box),
                or ``"viability-less-mpc"`` (same MPC with the DCM state box
                ``|ξ| ≤ ξ_max`` removed). All three are instantiated
                unconditionally so that the runtime selector can switch between
                them at no allocation cost.
            controller_locked: If True, the controller choice is fixed for the
                lifetime of the balancer. If False, a selector is registered
                with the interface and read at every cycle, so that the user
                can switch controllers live.
            crouch_height: Distance the robot goes down while crouching, in
                meters. Lowering the center of mass raises the kinematic
                ω = √(g/h), which is the parameter variation we sweep to check
                the method is not overfit to one configuration.
        """
        if controller not in CONTROLLERS:
            raise ValueError(
                f"unknown controller {controller!r}, "
                f"expected one of {CONTROLLERS}"
            )

        # Modules
        self.dcm_feedback = DcmFeedbackController()
        self.dcm_mpc = DcmMpc(horizon=50)
        self.dcm_mpc_no_viability = DcmMpc(horizon=50, enforce_viability=False)
        self.controller_name = controller
        self.controller_locked = controller_locked
        self.dcm_observables: dict[str, OnnxDcmObservable] = {
            path.stem: OnnxDcmObservable(path)
            for path in sorted(dcm_models_dir.glob("*.onnx"))
        }
        self.pattern_generator = PatternGenerator(
            initial_period=initial_period
        )

        # Plots
        self.dcm_zmp_plot = DcmZmpPlot()

        # Environment
        self._env = env
        model = self._env.unwrapped.model

        # Modules needing model
        self.leg_controller = LegController(
            model, default_crouch_height=crouch_height
        )
        self.state_estimator = StateEstimator(model)
        self.wheel_controller = WheelController(model)

        # Interface
        if interface is not None:
            self.interface: Interface = interface
        elif interface_name == "viser":
            try:
                from .interfaces import ViserInterface
            except ImportError:
                raise RuntimeError("viser is not installed") from None
            self.interface = ViserInterface(
                self.state_estimator.kin_model,
                self.state_estimator.visual_model,
            )
        elif interface_name == "joystick":
            self.interface = JoystickInterface(
                axis_map={
                    "pad_horizontal": ("amplitude", 0.1),
                    "pad_vertical": ("hack_velocity", -0.2),
                    "left_vertical": ("forward_velocity", -1.0),
                    "right_horizontal": ("offset", -1.0),
                },
                button_map={
                    "right_button": "dcm_model",
                    "square_button": "ready",
                },
            )
        else:
            self.interface = Interface()

        # DCM model selector
        dcm_values = ["kinematics", "learned-lip"] + list(self.dcm_observables)
        self.interface.add_selector(
            name="dcm_model",
            label="DCM model",
            values=dcm_values,
            default="kinematics",
        )

        # DCM controller selector (no-op when locked)
        if not controller_locked:
            self.interface.add_selector(
                name="controller",
                label="Controller",
                values=list(CONTROLLERS),
                default=controller,
            )

        # Ready selector (square button on joystick), only in supervised mode
        if supervised:
            self.interface.add_selector(
                name="ready",
                label="Ready?",
                values=["yes", "no"],
                default="no",
            )

        # Module sliders
        self.leg_controller.register_interface(self.interface)
        self.dcm_feedback.register_interface(self.interface)
        self.dcm_mpc.register_interface(self.interface)
        self.dcm_mpc_no_viability.register_interface(self.interface)
        self.wheel_controller.register_interface(self.interface)
        self.pattern_generator.register_interface(self.interface)
        self.state_estimator.register_interface(self.interface)

        # Plots
        plots_context = (
            self.interface.viser_server.gui.add_folder(
                "Plots",
                expand_by_default=True,
            )
            if hasattr(self.interface, "viser_server")
            else nullcontext()
        )
        with plots_context:
            self.dcm_zmp_plot.register_interface(self.interface)

        # Quick and dirty perf timers
        self._perf_samples: dict[str, RunningMeanStd] = {
            key: RunningMeanStd()
            for key in (
                "leg",
                "wheel",
                "env_step",
                "interface",
                "estimators",
                "pattern",
                "dcm",
                "visualizer",
                "plots",
                "log",
                "total",
            )
        }

        # Reset once everything is ready
        _, info = self._env.reset()
        spine_observation = info["spine_observation"]
        if (
            interface_name == "joystick"
            and "joystick" not in spine_observation
        ):
            raise RuntimeError(
                "Joystick interface requested but no joystick "
                "found in spine observation"
            )

        # Save attributes
        self._ctrl_leaning_velocity = 0.0
        self._ctrl_zmp_line = 0.0
        self._spine_observation = spine_observation

    @property
    def dcm_controller(self) -> DcmController:
        """Active DCM controller, as selected by ``controller_name``."""
        if self.controller_name == "mpc":
            return self.dcm_mpc
        if self.controller_name == "viability-less-mpc":
            return self.dcm_mpc_no_viability
        return self.dcm_feedback

    def select_dcm_source(
        self, dcm_model: str
    ) -> Union[StateEstimator, LipDcm, OnnxDcmObservable]:
        """Map a selector value to the DCM source driving the controller.

        The three sources differ only in the observable they expose: the
        kinematic LIP DCM at ω = √(g/h), the learned-LIP ablation at the
        fitted ω̃, and a learned DCM. All expose ``dcm_line`` and
        ``omega``, so the controller consumes them interchangeably.

        Args:
            dcm_model: Value of the ``dcm_model`` selector.

        Returns:
            The selected DCM source. Unknown values fall back to the
            kinematic DCM.
        """
        if dcm_model == "learned-lip":
            return self.state_estimator.lip_dcm
        # "kinematics", or the stem of a learned DCM
        return self.dcm_observables.get(dcm_model, self.state_estimator)

    def cycle(self) -> bool:
        """Run one control cycle.

        Returns:
            True if the episode ended this cycle.

        Note:
            If the episode ended, the balancer will have reset all modules.
            Callers should re-apply any programmatic slider overrides before
            the next cycle.
        """
        dt = self._env.unwrapped.dt
        dcm_model = self.interface.get_selector("dcm_model")
        dcm_source = self.select_dcm_source(dcm_model)

        # Leg controller
        _t0 = time.perf_counter()
        leg_action = self.leg_controller.cycle(
            dt,
            self._spine_observation,
            leaning_velocity=self._ctrl_leaning_velocity,
        )

        # Wheel controller
        _t1 = time.perf_counter()
        wheel_action = self.wheel_controller.cycle(
            dt,
            observation=self._spine_observation,
            apply_hack=self.leg_controller.is_ready,
        )

        # Step the environment
        _t2 = time.perf_counter()
        _, _, terminated, truncated, info = self._env.step(
            action={
                "left_hip": leg_action["left_hip"],
                "left_knee": leg_action["left_knee"],
                "left_wheel": wheel_action["left_wheel"],
                "right_hip": leg_action["right_hip"],
                "right_knee": leg_action["right_knee"],
                "right_wheel": wheel_action["right_wheel"],
            }
        )
        self._spine_observation = info["spine_observation"]

        # User interface
        _t3 = time.perf_counter()
        self.interface.cycle(dt, self._spine_observation)

        # State observer
        _t4 = time.perf_counter()
        self.state_estimator.cycle(dt, self._spine_observation)

        # Learned DCMs: infer only on the selected one to save real-robot
        # cycles.
        for key, observable in self.dcm_observables.items():
            observable.cycle(
                dt,
                q_actuated=self.state_estimator.q_actuated,
                com_anchor=self.state_estimator.com_anchor,
                com_line=self.state_estimator.com_line,
                action=self._ctrl_zmp_line,
                skip_inference=key != dcm_model,
            )

        # Pattern generator
        _t5 = time.perf_counter()
        self.pattern_generator.cycle(dt, omega=dcm_source.omega)

        # DCM controller. The two families take different arguments: the
        # MPC needs the pattern generator itself, to forward-evaluate the
        # reference over its preview horizon.
        _t6 = time.perf_counter()
        if not self.controller_locked:
            self.controller_name = self.interface.get_selector("controller")
        dcm_controller = self.dcm_controller
        if isinstance(dcm_controller, DcmMpc):
            dcm_controller.cycle(
                dt,
                omega=dcm_source.omega,
                dcm_observed=dcm_source.dcm_line,
                zmp_observed=self.state_estimator.zmp_line,
                pattern=self.pattern_generator,
            )
        else:
            dcm_controller.cycle(
                dt,
                omega=dcm_source.omega,
                dcm_target=self.pattern_generator.dcm,
                dcm_observed=dcm_source.dcm_line,
                zmp_target=self.pattern_generator.zmp,
                zmp_observed=self.state_estimator.zmp_line,
            )

        # Save ctrl targets for next cycle and collect data if applicable
        self._ctrl_leaning_velocity = dcm_controller.leaning_velocity
        self._ctrl_zmp_line = dcm_controller.zmp_line
        if hasattr(self.interface, "collect_data"):
            self.interface.collect_data(
                y_k=np.asarray(
                    self.state_estimator.observation, dtype=np.float32
                ),
                u_k=np.float32(self._ctrl_zmp_line),
                terminated=terminated,
                truncated=truncated,
                is_ready=self.leg_controller.is_ready,
            )

        # State visualizer
        _t7 = time.perf_counter()
        self.state_estimator.update_visualizer(
            dt,
            zmp_target=self.pattern_generator.zmp,
        )
        dcm_controller.update_visualizer(self.state_estimator.contacts)

        # Plots
        _t8 = time.perf_counter()
        if self.dcm_zmp_plot is not None:
            self.dcm_zmp_plot.cycle(
                dt,
                dcm_target=self.pattern_generator.dcm,
                dcm_observed=dcm_source.dcm_line,
                zmp_target=self.pattern_generator.zmp,
                zmp_observed=self.state_estimator.zmp_line,
                zmp_command=dcm_controller.zmp_line,
            )

        # Logging
        _t9 = time.perf_counter()
        self._env.log("dcm_controller", dcm_controller.to_log())
        self._env.log(
            "controller_index", CONTROLLERS.index(self.controller_name)
        )
        self._env.log(
            "dcm_model_index", self.interface.get_selector_index("dcm_model")
        )
        self._env.log("leg_controller", self.leg_controller.to_log())
        self._env.log("pattern_generator", self.pattern_generator.to_log())
        self._env.log("state_estimator", self.state_estimator.to_log())
        if "ready" in self.interface._selectors:
            is_ready = self.interface.get_selector("ready") == "yes"
            self._env.log("ready", is_ready)
        if self.dcm_zmp_plot is not None:
            self._env.log("dcm_zmp_plot", self.dcm_zmp_plot.to_log())

        # Log perf timers
        _t10 = time.perf_counter()
        self._perf_samples["leg"].append(_t1 - _t0)
        self._perf_samples["wheel"].append(_t2 - _t1)
        self._perf_samples["env_step"].append(_t3 - _t2)
        self._perf_samples["interface"].append(_t4 - _t3)
        self._perf_samples["estimators"].append(_t5 - _t4)
        self._perf_samples["pattern"].append(_t6 - _t5)
        self._perf_samples["dcm"].append(_t7 - _t6)
        self._perf_samples["visualizer"].append(_t8 - _t7)
        self._perf_samples["plots"].append(_t9 - _t8)
        self._perf_samples["log"].append(_t10 - _t9)
        self._perf_samples["total"].append(_t10 - _t0)

        episode_ended = (
            terminated or truncated or self.interface.reset_requested
        )
        if episode_ended:
            self.reset()
        return episode_ended

    def reset(self) -> None:
        """Reset the environment and all modules."""
        _, info = self._env.reset()
        self._spine_observation = info["spine_observation"]
        self._ctrl_leaning_velocity = 0.0
        self._ctrl_zmp_line = 0.0

        self.interface.reset()
        self.state_estimator.reset(self.interface)
        self.dcm_feedback.reset()
        self.dcm_mpc.reset()
        self.dcm_mpc_no_viability.reset()
        self.leg_controller.reset()
        self.wheel_controller.reset()
        for observable in self.dcm_observables.values():
            observable.reset()

    def log_perfs(self) -> None:
        """Log the duration of each step of the control cycle."""
        if not any(v.n > 0 for v in self._perf_samples.values()):
            return
        logger.info("         Balancer                   ")
        logger.info("-----------------                   ")
        for k, v in self._perf_samples.items():
            if v.n > 1:
                logger.info(
                    "%16s: %5.1f ± %5.1f ms  ",
                    k,
                    1e3 * v.average(),
                    1e3 * v.std(),
                )
        self.state_estimator.log_perfs()
