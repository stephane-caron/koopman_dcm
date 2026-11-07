# cookie_balancer

Koopman DCMs are evaluated on lateral balancing of a [Cookie wheeled biped](https://github.com/orgs/upkie/discussions/545). Sagittal balancing is carried out by the default MPC balancer distributed with all [Upkie](https://github.com/upkie/upkie) wheeled bipeds. The DCM used for lateral balancing can be switched in the Viser GUI between the linear inverted pendulum (LIP) DCM or the Koopman DCM learned from 55 minutes of robot data.

## Lateral DCM controllers

The DCM feeds one of three interchangeable controllers, selected by
`CONTROLLER` at the top of `cookie_balancer/run_balancer.py` and switchable at
runtime from the [Viser](https://viser.studio/) interface:

| Key | Class | What it does |
|---|---|---|
| `dcm` | `DcmFeedbackController` | Proportional DCM feedback at the ZMP. |
| `mpc` | `DcmMpc` | Receding-horizon QP over the DCM recursion, under the viability constraint `\|ξ\| ≤ ξ_max`. |
| `viability-less-mpc` | `DcmMpc(enforce_viability=False)` | The same MPC with only the DCM state box dropped, isolating what the viability constraint contributes. |

## Installation

This is a [pixi](https://pixi.sh) project. Install it from this directory:

```bash
pixi install
```

Run `pixi task list` to list available tasks. To run the balancer in PyBullet
or on the real robot, see the [main README](../README.md).

## Reproducing the paper

The datasets of the paper are on [Hugging
Face](https://huggingface.co/datasets/stephane-caron/koopman_dcm). To download
them to `data/`:

```bash
pixi run download-data
```

This fetches `cookie.h5` (55 minutes of real-robot data), `cookie-ood.h5`
(real-robot tracking at out-of-distribution periods) and `pybullet-cookie.h5`
(one hour of PyBullet data). The two DCM observables of the paper are also
released. To download them to `models/`:

```bash
pixi run download-models
```

This fetches `dcm-cookie.onnx`, trained on real-robot data, and
`dcm-pybullet-cookie.onnx`, trained on PyBullet data. Tasks that need them
will download them automatically.

### Simulation: Table 2 (Cookie wheeled biped)

The `evaluate-mpc` task compares the three controllers in PyBullet over
sinusoidal ZMP references, with amplitudes {0.2, 0.4, 0.6}, offsets {0, 0.5},
periods {1.0, 1.2, 1.4} s and five seeds per cell (90 trials per controller),
all using `dcm-pybullet-cookie`. It saves per-trial results to
`evaluation/data/cookie-evaluation.h5`, then prints the table:

```bash
pixi run evaluate-mpc
```

To train a new simulation DCM, set `DATASET = settings.DATA_DIR /
"pybullet-cookie.h5"` and `OUTPUT = settings.MODELS_DIR /
"dcm-pybullet-new.onnx"` at the top of `cookie_balancer/train_dcm.py`, run
`pixi run train-dcm`, and set `DCM_MODEL = "dcm-pybullet-new"` in
`cookie_balancer/evaluate_mpc.py`. To collect a new hour of PyBullet data
instead, run `pixi run pybullet-collect-data`, which writes
`data/pybullet-new.h5`.

### Learning from real-robot data

These scripts run offline on the real-robot datasets `data/cookie.h5` (Section
4.1), or `data/cookie-ood.h5` (out-of-distribution periods), and the analysis
scripts on `models/dcm-cookie.onnx`. Make sure to run `pixi run download-data`
and `pixi run download-models` first.

| Paper result | Command |
|---|---|
| Learned DCM (Section 4.4) | `pixi run train-dcm` |
| Seed variability (Section 4.4) and number of lambda steps (Appendix B.1) | `pixi run python scripts/multiseed_training.py` |
| Learned-LIP ablation (Sections 4.3–4.4) | `pixi run python scripts/estimate_omega_from_dataset.py` |
| Input scrambling (Section 4.5) | `pixi run python scripts/input_scrambling.py` |
| State-estimation noise (Appendix B.2) | `pixi run python scripts/noise_sensitivity.py` |

### Real robot

These tasks run on the Cookie robot, with a joystick. The operator sets `ready`
(square button) to start, and the d-pad nudges the robot back in place between
chunks. Each task writes its outputs to the working directory.

| Paper result | Command | Protocol |
|---|---|---|
| Dataset (Section 4.1) | `pixi run real-collect-data` | 5-minute sessions of 10-second chunks of random sinusoidal ZMP references, one HDF5 file per session |
| Table 1 | `pixi run real-benchmark-dcm` | One session per amplitude A ∈ {0.1, …, 0.6}, set by `AMPLITUDE` in `cookie_balancer/benchmark_dcm_spine.py`: 10 cycles over the kinematic, learned-LIP and learned DCMs at T = 1.2 s |
| Out-of-distribution periods (Section 4.4) | `pixi run real-benchmark-ood-periods` | A = 0.3, T ∈ {2.0, 1.8, 0.5} s, 3 cycles over the kinematic and learned DCMs, recorded to HDF5 for `input_scrambling.py` |

Note that training reads a single file, whereas `real-collect-data` writes one
file per session: merge sessions by concatenating each dataset of the files in
order, keeping the file attributes.
