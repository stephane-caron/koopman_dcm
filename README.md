# Koopman DCM

[[`Paper`](https://openreview.net/pdf?id=mday00WZFr)] [[`Dataset`](https://huggingface.co/datasets/stephane-caron/koopman_dcm)] [[`Models`](https://huggingface.co/stephane-caron/koopman_dcm)] [[`Video`](https://scaron.info/videos/corl-2026.mp4)]

Source code, models and data for [Koopman DCM: Unstable Eigenfunctions as Data-driven Representations for Legged Balancing](https://arxiv.org/abs/2607.18760).

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
