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
├── src/
│   ├── stage1/        # Normalization and preprocessing
│   ├── stage2/        # ICE computation
│   ├── stage3/        # ΔΦ instability metrics
│   ├── stage4/        # Embedding + manuscript-grade visualizations
│   ├── stage5/        # Regime segmentation and final outputs
│   └── extensions/    # Stage 6–7 conceptual layers (not validated)
│
├── figures/
│   ├── stage1-5/      # Manuscript figures
│   └── extensions/    # Placeholder figures for Stage 6–7
│
├── PAPER.md           # Manuscript text
├── citations.bib      # References used in the manuscript
└── README.md          # This file
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

All manuscript figures can be reproduced deterministically by running:

```bash
python src/run_pipeline.py
```

The pipeline generates the following outputs:

- ICE curves
- ΔΦ instability profiles
- Spiral embeddings
- Regime segmentation plots

All outputs are saved to `figures/stage1-5/` and correspond to the **exact figures** used in the manuscript.

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

If you use this framework, please cite the manuscript:

```bibtex
@article{kruger2026triadic,
  author  = {Krüger, M. and Wende, M. T. and Feeney, D.},
  title   = {Information-Theoretic Modeling of Neural Coherence via
             Triadic Spiral-Time Dynamics: A Framework for Neurodynamic Collapse},
  year    = {2026}
}
```

Full BibTeX entries are provided in [`citations.bib`](citations.bib).

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
