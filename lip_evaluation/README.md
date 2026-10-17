# LIP evaluation

This code reproduces the results reported in [Appendix A](https://arxiv.org/html/2607.18760v2#A1) of the paper, *Evaluation on the Linear Inverted Pendulum Model*. It trains a Koopman DCM estimator on closed-loop data from a linear inverted pendulum (LIP) whose ground-truth eigenpair is known (natural frequency $\omega^\star = 5$ rad/s, DCM $\xi = c + \dot{c} / \omega^\star$). The overall Appendix checks that the training pipeline recovers the known DCM.

## Installation

Install [pixi](https://pixi.sh), then run, from this directory:

```console
pixi install
```

## Usage

Each part of the appendix corresponds to a pixi task. Figures are saved to `figures/` as `<date>_<time>_<name>.pdf`. Tables are printed at the end of each run.

### Figure: Training of a Koopman DCM estimator on a LIP model

- Command: `pixi run train`

This task output `figures/*_lip_eval_losses.pdf`, printing out the learned $\hat{\omega}$ and final validation losses.

### Appendix A.1: Weight of the CoM Prediction Loss

- Command: `pixi run com-loss-sweep`

This task outputs `figures/*_lip_eval_com_sweep.pdf` as well as a summary table.

### Appendix A.2: Delay-embedding Horizon

- Command: `pixi run history-sweep`

This task outputs the Table "Delay-embedding horizon $H$ on the LIP model".
