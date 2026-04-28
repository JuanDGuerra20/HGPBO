# Bidirectional Information Flow (BIF)
### A Sample-Efficient Hierarchical Gaussian Process for Bayesian Optimization
*ICML 2026*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

Hierarchical Gaussian Process (H-GP) models divide problems into different subtasks, allowing for different models to address each part, making them well-suited for problems with inherent compositional structure. However, existing H-GP frameworks typically employ one-way information sharing — either top-down or bottom-up — which limits sample efficiency and slows convergence. We propose Bidirectional Information Flow (BIF), an efficient framework that defines a hierarchy of probabilistic models, where children and parent each represent beliefs over functions at different levels of aggregation for online training. BIF retains the modular structure of hierarchical models — the parent conditions its own posterior on child summaries, treating them as structured priors — while introducing top-down feedback to softly decompose environment observations from the parent into sub-responses using the children's current predictive beliefs. This mutual exchange improves sample efficiency, enables robust training, and allows modular reuse of learned subtask models. We prove analytically the regret of a GP with a learned kernel scales linearly with the mismatch to the true kernel and is upper bounded by the mismatch of the children in hierarchical cases. BIF outperforms conventional H-GP Bayesian Optimization methods, achieving up to 4x higher R² scores for the parent, on synthetic and real-world neurostimulation optimization tasks.

## Installation

```bash
git clone https://github.com/JuanDGuerra20/HGPBO.git
cd HGPBO
git lfs pull           # downloads the neural dataset (~262 MB)
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA-capable GPU is recommended for neural dataset experiments.

## Reproducing Paper Results

| Experiment | Script | Key parameter |
|---|---|---|
| Synthetic 2D (primary) | `python synthetic_2d/efficient_general_2d.py` | `dataset=3` |
| Synthetic 3D | `python synthetic_3d/ucb_efficient_general.py` | `dataset_num=6` |
| Neural dataset | `python neural/general_neural.py` | — |
| Modularity | `python synthetic_2d/modularity_experiment.py` | — |
| Nonlinearity | `python synthetic_2d/nonlinearity_experiments.py` | — |
| Noise | `python synthetic_2d/test.py` | — |
| GAN hyperparameter opt | `python gan/bif_gan.py` | — |

Results are written to `<tier>/<experiment>/<method>/data-<date>/` within the corresponding folder.

## Project Structure

```
HGPBO/
├── data/                  # Neural stimulation dataset (Git LFS)
├── neural/                # BIF on real-world neural data (10 channels, 7 time delays)
├── synthetic_2d/          # BIF on 2D synthetic benchmark functions
├── synthetic_3d/          # BIF on 3D synthetic benchmark functions
└── gan/                   # BIF for GAN hyperparameter optimization
```

Each tier contains parallel implementations of three methods:
- **BIF** (proposed) — bidirectional information flow between parent and child GPs
- **Laferrière** — H-GP baseline with one-way (bottom-up) information sharing
- **Vanilla GPBO** — standard single-level Gaussian Process Bayesian Optimization

## Citation

```bibtex
@inproceedings{guerra2026bif,
  title     = {Bidirectional Information Flow: A Sample-Efficient Hierarchical Gaussian Process for Bayesian Optimization},
  author    = {Guerra, Juan David and Garbay, Thomas and Dancause, Numa and Lajoie, Guillaume and Bonizzato, Marco},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {To appear}
}
```

## Contact

Please refer to the paper for author contact information.
