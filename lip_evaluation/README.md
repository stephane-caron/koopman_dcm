# LIP evaluation

This directory reproduces Appendix A of the paper, *Evaluation on the Linear Inverted Pendulum Model*. It trains a Koopman DCM estimator on closed-loop data from a linear inverted pendulum (LIP) whose ground-truth eigenpair is known (natural frequency $\omega^\star = 5$ rad/s, DCM $\xi = c + \dot{c} / \omega^\star$). This is a check that the training pipeline recovers the analytical DCM.

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
| `pixi run history-sweep` | Appendix A.2, *Delay-embedding Horizon*: Table "Delay-embedding horizon $H$ on the LIP model" | `figures/*_lip_eval_history_sweep_*.pdf`, summary table |
| `pixi run appendix` | All of Appendix A | All of the above |

Figures are saved to `figures/` as `<date>_<time>_<name>.pdf`. Tables are printed at the end of each run. Each task calls one script with the paper's hyperparameters. Run the script directly to change them, for example:

```console
pixi run python -m lip_eval --help
pixi run python -m lip_eval.com_loss_sweep --epochs 100
```

### Training run (Appendix A)

`pixi run train` simulates $200\,000$ closed-loop steps of the LIP. The ZMP reference is a sine wave whose period, amplitude and offset are re-drawn every $2\,000$ steps. The task then trains the DCM observable (an MLP with two hidden layers of width 128) jointly with $\log \omega$ for 100 epochs, from $\omega_0 = 4$ rad/s with CoM-loss weight $w_\text{CoM} = 1$. In the paper, this run converged to $\hat{\omega} = 4.9992$ rad/s with final validation losses $\ell^{(0)}_\text{DCM} = 5.4 \times 10^{-9}$ and $\ell^{(0)}_\text{CoM} = 3.9 \times 10^{-9}$.

The script also plots the true and learned DCMs over a 10-second rollout (`figures/*_lip_eval_results.pdf`, not shown in the paper).

### CoM-loss weight sweep (Appendix A.1)

`pixi run com-loss-sweep` trains with $w_\text{CoM} \in \{0.1, 1, 10, 100\}$ from $\omega_0 = 1$ rad/s, close to the trivial minimum $\omega = 0$. Each weight is trained on three seeds, for 50 epochs over $50\,000$ steps. Paper values (mean ± std over seeds):

| $w_\text{CoM}$ | $\hat{\omega}$ (rad/s) | Validation loss (MSE) |
|----------------|------------------------|-----------------------|
| 0.1            | 0.352 ± 0.007          | 7.35e-7               |
| 1              | 4.975 ± 0.003          | 1.30e-8               |
| 10             | 4.878 ± 0.021          | 7.46e-8               |
| 100            | 3.800 ± 0.051          | 4.66e-6               |

### Delay-embedding horizon sweep (Appendix A.2)

`pixi run history-sweep` trains with delay-embedding horizon $H \in \{1, 3, 5, 10, 20\}$. Each horizon is trained on ten seeds, for 50 epochs over $50\,000$ steps, from $\omega_0 = 4$ rad/s. Paper values (mean ± std over seeds):

| $H$ | $\hat{\omega}$ (rad/s) | Validation loss (MSE) |
|-----|------------------------|-----------------------|
| 1   | 4.997 ± 0.005          | 2.37e-8               |
| 3   | 4.997 ± 0.005          | 6.02e-8               |
| 5   | 4.999 ± 0.002          | 1.90e-8               |
| 10  | 4.995 ± 0.005          | 2.79e-8               |
| 20  | 4.994 ± 0.003          | 4.12e-8               |

## Hardware

The scripts require a CUDA-capable GPU, on which the results in the paper were computed. Numerical values vary across hardware. Validation losses at the last epoch also fluctuate from one epoch to the next, so reruns reproduce the values of $\hat{\omega}$ above more closely than the validation losses.

## Code

| File | Contents |
|------|----------|
| `lip_eval/__main__.py` | LIP dynamics, data generation, delay embeddings, encoder, training loss, training run (Appendix A) |
| `lip_eval/com_loss_sweep.py` | CoM-loss weight sweep (Appendix A.1) |
| `lip_eval/history_sweep.py` | Delay-embedding horizon sweep (Appendix A.2) |
| `lip_eval/figures.py` | Figure saving |

The observation vector is $y_k = (c_k, \dot{c}_k)$ in the paper. In the code it is the six-dimensional vector $(c_x, c_y, c_z, \dot{c}_x, \dot{c}_y, \dot{c}_z)$, where only $c_x$ and $\dot{c}_x$ vary. Its other coordinates are constant ($c_z = 0.4$ m, all others zero), so they carry no information.
