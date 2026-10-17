# LIP evaluation

This code reproduces the results reported in [Appendix A](https://arxiv.org/html/2607.18760v2#A1) of the paper, *Evaluation on the Linear Inverted Pendulum Model*. It trains a Koopman DCM estimator on closed-loop data from a linear inverted pendulum (LIP) whose ground-truth eigenpair is known (natural frequency $\omega^\star = 5$ rad/s, DCM $\xi = c + \dot{c} / \omega^\star$). The overall Appendix checks that the training pipeline recovers the known DCM.

## Installation

Install [pixi](https://pixi.sh), then run, from this directory:

```console
pixi install
```

## Usage

Each part of the appendix corresponds to a pixi task:

| Command | Paper | Outputs |
|---------|-------|---------|
| `pixi run train` | Appendix A: Figure "Training of a Koopman DCM estimator on a LIP model" | `figures/*_lip_eval_losses.pdf`, learned $\hat{\omega}$ and final validation losses |
| `pixi run com-loss-sweep` | Appendix A.1, *Weight of the CoM Prediction Loss*: table and Figure "CoM-weight sweep" | `figures/*_lip_eval_com_sweep.pdf`, summary table |
| `pixi run history-sweep` | Appendix A.2, *Delay-embedding Horizon*: Table "Delay-embedding horizon $H$ on the LIP model" | Summary table |

Figures are saved to `figures/` as `<date>_<time>_<name>.pdf`. Tables are printed at the end of each run. Hyperparameters are set to the values of the appendix by global constants at the top of each script, and of `lip_eval/train_dcm.py` for those shared by all runs.
