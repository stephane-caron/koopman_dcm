# Koopman DCM: Unstable Eigenfunctions as Data-driven Representations for Legged Balancing

[[`Paper`](https://openreview.net/pdf?id=mday00WZFr)] [[`Dataset`](https://huggingface.co/datasets/stephane-caron/koopman_dcm)] [[`Models`](https://huggingface.co/stephane-caron/koopman_dcm)] [[`Video`](https://scaron.info/videos/corl-2026.mp4)]

In legged locomotion, divergent components of motion (DCMs) have emerged as characteristic states for balance control. They isolate the unstable mode of the dynamics but, in existing formulations, apply only to reduced models such as the linear inverted pendulum. In this study, we show how DCMs can be more generally formulated as Koopman eigenfunctions. Whereas Koopman analysis typically targets eigenvalues near zero, which capture conserved or slowly varying quantities, our investigation leads us to deliberately search for unstable eigenpairs with large eigenvalues. The resulting Koopman DCMs are data-driven observables trained using only real-robot data. On a real biped, DCMs learned from one hour of robot data improve tracking of reference walking patterns. We further show how learned DCMs provide state-based viability constraints when combined with model predictive control.

## Upcoming

This work will be presented next month at [CoRL 2026](https://www.corl.org/). In the meantime, source code and data will be released according to the following plan:

- [x] LIP evaluation: [`lip_evaluation/`](lip_evaluation/)
- [x] MPC evaluation: [G1 humanoid simulation](g1_evaluation/)
- [ ] MPC evaluation: Cookie wheeled biped simulation
- [ ] Real-robot controller
- [ ] Real-robot dataset

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
