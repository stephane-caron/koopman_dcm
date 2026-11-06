# G1 evaluation

This code reproduces the G1 humanoid results reported in [Section 5](https://arxiv.org/html/2607.18760v2#S5) of the paper, *Model Predictive Control over Learned DCMs*. A Unitree G1 humanoid stands in double support in [PyBullet](https://pybullet.org/) and balances laterally while tracking sinusoidal ZMP references. Three DCM controllers are compared, all using the same learned DCM observable ($\hat{\omega} = 1.91$ rad/s): proportional DCM feedback, the MPC with viability constraint $|\xi_k| \leq \xi_\text{max}$, and the same MPC without this constraint.

## Installation

Install [pixi](https://pixi.sh), then run, from this directory:

```console
pixi install
```

## Usage

### Table 2 (G1 humanoid)

To evaluate the three controllers with the DCM observable trained for the paper:

```console
pixi run evaluate-mpc
```

This task first downloads the trained DCM model of the paper from Hugging Face to `models/`. It then sweeps ZMP references $u^\text{ref}_t = o + A \sin(2\pi t / T)$ over amplitudes $A \in \{0.3, 0.6, 0.9\}$, offsets $o \in \{0, 0.5\}$ and periods $T \in \{1.3, 1.6, 1.9\}$ s, with five seeds per cell (90 trials of 19 s per controller), and finally prints the G1 side of Table 2. Per-trial results are saved to `data/g1-evaluation.h5`. To print the table again from this file:

```console
pixi run print-table
```

### Training a DCM observable

The DCM observable is trained from closed-loop trajectories of the G1 balancing under the MPC, while tracking random ZMP references. To train a DCM observable on the dataset of the paper:

```console
pixi run train-dcm
```

This task first downloads the dataset from Hugging Face to `data/pybullet-g1.h5`. It then prints the learned $\hat{\omega}$ and validation losses at each epoch, and exports the model with the lowest validation loss to `models/dcm-pybullet-g1-new.onnx`. Training runs on a CUDA GPU if one is available, and on CPU otherwise. To evaluate the new model, set `MODEL_PATH` in `g1_evaluation/evaluate_mpc.py` to its path, then run `pixi run evaluate-mpc`.

To collect a new dataset of one million control steps (about 70 minutes of simulated time at 240 Hz) with four parallel workers:

```console
pixi run collect-data
```

This task outputs `data/pybullet-g1-new.h5`. To train on it, set `DATASET` in `g1_evaluation/train_dcm.py` to its path, then run `pixi run train-dcm`.
