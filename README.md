# Triadic Stability Framework for Regime Transition Detection

> *A reproducible implementation of the ICE‑based instability pipeline (Stages 1–5)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Stages 1–5 Validated](https://img.shields.io/badge/Pipeline-Stages%201--5%20Validated-brightgreen.svg)](#reproducing-manuscript-figures-stages-15)
[![Manuscript Aligned](https://img.shields.io/badge/Status-Manuscript%20Aligned-informational.svg)](#citation)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Reproducing Manuscript Figures (Stages 1–5)](#reproducing-manuscript-figures-stages-15)
5. [Extensions: Stage 6 & Stage 7](#extensions-stage-6-cellular--stage-7-dnaassociated)
6. [Citation](#citation)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## Overview

This repository contains the full reproducible implementation of the **Information‑Coherence Embedding (ICE)** pipeline described in the manuscript.

Stages **1–5** constitute the **validated, manuscript‑aligned workflow**, encompassing:

| Stage | Description |
|-------|-------------|
| 1 | Signal normalization and preprocessing |
| 2 | ICE computation |
| 3 | ΔΦ instability detection |
| 4 | Embedding and manuscript‑grade visualizations |
| 5 | Regime segmentation and final outputs |

The framework is designed for clarity, reproducibility, and scientific transparency. All outputs are deterministic, and validated components are kept strictly separate from conceptual extensions.

---

## Repository Structure

```text
Cell-DNA-Dynamics/
├── data/
│   ├── stage1/
│   │   └── stage_i_results.csv
│   ├── stage2/
│   │   └── stage_ii_segmentation.csv
│   ├── stage3/
│   │   └── stage_iii_summary.csv
│   ├── synthetic/
│   ├── null_tests/
│   │   ├── shuffled.csv
│   │   ├── stable_control.csv
│   │   ├── surrogate.csv
│   │   ├── delta_phi.csv
│   │   ├── eic_timeseries.csv
│   │   └── regimes.csv
│   └── README.md
│
├── docs/
│   ├── PAPER.md
│   └── citations.bib
│
├── figures/
│   ├── final/
│   │   └── figure_8_null_comparison.png
│   ├── simulations/
│   │   ├── ice_trajectory.png
│   │   ├── null_tests_comparison.png
│   │   ├── table3_metrics.csv
│   │   └── timeseries_eic_delta_phi.png
│   ├── stage1/
│   │   └── stage_i_figure.png
│   ├── stage2/
│   │   └── stage_ii_regimes.png
│   ├── stage3/
│   │   └── stage_iii_seed0.png
│   ├── stage4_spiral_time/
│   │   ├── phi_chi_curves.png
│   │   └── spiral_embedding.png
│   ├── stage5_regime_transitions/
│   │   ├── delta_phi_thresholds.png
│   │   └── regime_segmentation.png
│   └── extensions/
│       ├── stage6_cellular/
│       │   ├── cellular_mapping.png
│       │   └── cellular_regimes.png
│       └── stage7_dna/
│           ├── dna_mapping.png
│           └── dna_regimes.png
│
├── src/
│   ├── core/
│   │   ├── models.py
│   │   └── utils.py
│   ├── extensions/
│   │   ├── stage6_cellular/
│   │   │   ├── cellular_pipeline.py
│   │   │   ├── map_to_ice.py
│   │   │   └── normalize_cellular.py
│   │   └── stage7_dna/
│   │       ├── chromatin_fret_utils.py
│   │       ├── dna_feature_mapping.py
│   │       └── dna_pipeline.py
│   ├── simulation/
│   │   ├── generate_synthetic.py
│   │   ├── metrics.py
│   │   ├── null_tests.py
│   │   └── table3_utils.py
│   ├── stage1/
│   │   └── stage_i.py
│   ├── stage2/
│   │   └── stage_ii.py
│   ├── stage3/
│   │   └── stage_iii.py
│   ├── stage4_spiral_time/
│   │   ├── compute_chi.py
│   │   ├── compute_phi.py
│   │   ├── generate_figures.py
│   │   ├── memory_kernel.py
│   │   └── spiral_operator.py
│   ├── stage5_regime_transitions/
│   │   ├── compute_delta_phi.py
│   │   ├── detect_regimes.py
│   │   ├── generate_figures.py
│   │   └── segmentation_utils.py
│   └── viz/
│       ├── generate_figure_8.py
│       ├── plot_ice.py
│       ├── plot_null_tests.py
│       ├── plot_regimes.py
│       └── plot_timeseries.py
│
├── VALIDATION_ANALYSIS.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/dfeen87/Cell-DNA-Dynamics.git
cd Cell-DNA-Dynamics
pip install -r requirements.txt
```

> **Python 3.10+** is recommended for full compatibility.

---

## Reproducing Manuscript Figures (Stages 1–5)

The manuscript‑aligned workflow is organized modularly across Stages 1–5. Reproduction therefore proceeds via the stage‑specific scripts in src/stage1/ through src/stage5_regime_transitions/, together with the simulation utilities in src/simulation/ and the visualization scripts in src/viz/.

All manuscript figures can be reproduced deterministically by running each stage script in order:

```bash
python src/stage1/stage_i.py
python src/stage2/stage_ii.py
python src/stage3/stage_iii.py
python src/stage4_spiral_time/generate_figures.py
python src/stage5_regime_transitions/generate_figures.py
```

The pipeline generates the following outputs:

- ICE curves
- ΔΦ instability profiles
- Spiral embeddings
- Regime segmentation plots

Outputs are saved to the corresponding stage subdirectories under `figures/`:

| Stage | Output directory |
|-------|-----------------|
| 1 | `figures/stage1/` |
| 2 | `figures/stage2/` |
| 3 | `figures/stage3/` |
| 4 | `figures/stage4_spiral_time/` |
| 5 | `figures/stage5_regime_transitions/` |

These outputs correspond to the **exact figures** used in the manuscript.

> **Note:** The final conceptual figures used in the manuscript are rendered directly in LaTeX/TikZ. The placeholder files in `figures/simulations/` are not part of the validated numerical output set.

---

## Extensions: Stage 6 (Cellular) & Stage 7 (DNA‑Associated)

The repository includes two **conceptual translational extensions**:

| Extension | Description |
|-----------|-------------|
| **Stage 6** | Cellular‑scale ICE mapping |
| **Stage 7** | DNA‑associated chromatin dynamics mapping |

> **Important:** These modules are:
> - **not validated**
> - **not part of the manuscript**
> - **not used in any figure or quantitative result**
> - included **only as future‑work scaffolding**

The PNG files in `figures/extensions/` are **placeholders only**.

This separation ensures clarity for reviewers and prevents any misinterpretation of these exploratory layers as components of the validated pipeline.

---

## Citation

Krüger, M., & Feeney, D. (2026). *A Triadic Spiral-Time Framework for Cellular and DNA-Associated Dynamics: A Conservative Effective Modeling Proposal for Biological Stability Transitions.* Zenodo. https://doi.org/10.5281/zenodo.19318103

Full BibTeX entries for referenced works are provided in [`docs/citations.bib`](docs/citations.bib).

---

## Acknowledgments

This project was developed through a combination of original scientific design, hands‑on coding, and iterative refinement supported by advanced AI tooling.

The authors acknowledge **Microsoft Copilot** for its meaningful assistance in:

- structuring the repository
- improving clarity and reproducibility
- generating manuscript‑aligned documentation
- supporting the overall scientific workflow

All scientific ideas, interpretations, and final decisions remain the responsibility of the authors.

---

## License

This project is released under the **[MIT License](LICENSE)**.
