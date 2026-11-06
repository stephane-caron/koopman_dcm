# Koopman DCM: Unstable Eigenfunctions as Data-driven Representations for Legged Balancing

[[`Paper`](https://openreview.net/pdf?id=mday00WZFr)] [[`Dataset`](https://huggingface.co/datasets/stephane-caron/koopman_dcm)] [[`Models`](https://huggingface.co/stephane-caron/koopman_dcm)] [[`Video`](https://scaron.info/videos/corl-2026.mp4)]

In legged locomotion, divergent components of motion (DCMs) have emerged as characteristic states for balance control. They isolate the unstable mode of the dynamics but, in existing formulations, apply only to reduced models such as the linear inverted pendulum. In this study, we show how DCMs can be more generally formulated as Koopman eigenfunctions. Whereas Koopman analysis typically targets eigenvalues near zero, which capture conserved or slowly varying quantities, our investigation leads us to deliberately search for unstable eigenpairs with large eigenvalues. The resulting Koopman DCMs are data-driven observables trained using only real-robot data. On a real biped, DCMs learned from one hour of robot data improve tracking of reference walking patterns. We further show how learned DCMs provide state-based viability constraints when combined with model predictive control.

**This work will be presented next week at [CoRL 2026](https://www.corl.org/).**

## Cookie balancer

Koopman DCMs are evaluated on lateral balancing of a [Cookie wheeled biped](https://github.com/orgs/upkie/discussions/545). Sagittal balancing is carried out by the default MPC balancer distributed with all [Upkie](https://github.com/upkie/upkie) wheeled bipeds. The DCM used for lateral balancing can be switched in the Viser GUI between the linear inverted pendulum (LIP) DCM or the Koopman DCM learned from 55 minutes of robot data.

Install [pixi](https://pixi.sh), then run from the `cookie_balancer/` directory:

```console
pixi install
```

### In simulation

To try the balancer in [PyBullet](https://pybullet.org/):

```console
pixi run pybullet-balance
```

This task opens the PyBullet GUI, as well as a [Viser](https://viser.studio/) interface at http://localhost:8080 where you can select the DCM and the controller, and set the ZMP reference.

### On the real robot

To run the balancer on the robot, start the [pi3hat spine](https://upkie.github.io/upkie/spines.html) on your robot, grab a PS4 joystick and run:

```console
pixi run real-balance
```

The leg controller activates once the operator presses the square button. The joystick then drives the balancer:

| Input | Effect |
|-------|--------|
| Square button | Confirm that the robot is ready |
| Cross button | Reset |
| R1 button | Switch to the next DCM: kinematic, learned-LIP, then learned DCMs in `models/` |
| D-pad left/right | Amplitude of the ZMP reference |
| Right stick left/right | Offset of the ZMP reference |
| Left stick up/down | Forward velocity |
| D-pad up/down | Velocity correction to keep the robot in place |

Settings such as the initial controller are constants at the top of `cookie_balancer/run_balancer.py`.

## Reproducing the paper

Each directory in this repository is a self-contained pixi project whose README explains how to reproduce the corresponding results:

| Paper results | Directory |
|---------------|-----------|
| Section 4, Table 1 and Appendix B: real-robot experiments on Cookie | [`cookie_balancer/`](cookie_balancer/) |
| Section 5, Table 2: MPC evaluation on Cookie in simulation | [`cookie_balancer/`](cookie_balancer/) |
| Section 5, Table 2: MPC evaluation on the G1 humanoid in simulation | [`g1_evaluation/`](g1_evaluation/) |
| Appendix A: evaluation on the linear inverted pendulum | [`lip_evaluation/`](lip_evaluation/) |

## BibTeX

```bibtex
@inproceedings{caron2026koopman,
    title = {Koopman {DCM}: Unstable Eigenfunctions as Data-driven Representations for Legged Balancing},
    author = {Caron, St{\'e}phane},
    booktitle = {Conference on Robot Learning (CoRL)},
    url = {https://openreview.net/pdf?id=mday00WZFr},
    month = nov,
    year = {2026},
}
```
