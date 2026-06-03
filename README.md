# Bidirectional Information Flow (BIF)
### A Sample-Efficient Hierarchical Gaussian Process for Bayesian Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

Hierarchical Gaussian Process (H-GP) models divide problems into different subtasks, allowing for different components to address each part, making them well-suited for problems with inherent compositional structure. However, existing H-GP frameworks typically employ one-way information sharing — either top-down or bottom-up — which limits sample efficiency and slows convergence. We propose Bidirectional Information Flow (BIF), which establishes continuous two-way communication. BIF retains the modular structure of hierarchical models — the parent conditions its own posterior on child summaries, treating them as structured priors — while introducing top-down feedback to softly decompose environment observations from the parent into sub-responses. This mutual exchange improves sample efficiency, enables robust training, and allows modular reuse of learned subtask models. We prove analytically the regret of a GP with a learned kernel scales linearly with the mismatch to the true kernel, tightening in the hierarchical case to the sum of child-level errors. Ablation shows removing the downward pathway collapses child $R^2$ by up to 58\%. Across synthetic, neurostimulation, and HPO benchmarks, BIF achieves up to 4× higher parent $R^2$ and ~100\% AUC improvement over vanilla GPBO, and outscores all hierarchical state-of-the-art methods on child $R^2$ given the correct acquisition function, while supporting modular child transfer to novel composite tasks.

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
