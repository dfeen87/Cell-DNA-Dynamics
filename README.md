# **Triadic Stability Framework for Regime Transition Detection**  
*A reproducible implementation of the ICE‑based instability pipeline (Stages 1–5)*

---

## **Overview**

This repository contains the full reproducible implementation of the **Information‑Coherence Embedding (ICE)** pipeline used in the manuscript.  

Stages **1–5** constitute the **validated, manuscript‑aligned workflow**, including:

- signal normalization  
- ICE computation  
- ΔΦ instability detection  
- regime segmentation  
- manuscript‑grade figure generation  

The framework is designed for clarity, reproducibility, and scientific transparency, with deterministic outputs and a clean separation between validated components and conceptual extensions.

---

## **Repository Structure**

```
src/
  stage1/        # Normalization and preprocessing
  stage2/        # ICE computation
  stage3/        # ΔΦ instability metrics
  stage4/        # Embedding + manuscript-grade visualizations
  stage5/        # Regime segmentation and final outputs
  extensions/    # Stage 6–7 conceptual layers (not validated)

figures/
  stage1-5/      # Manuscript figures
  extensions/    # Placeholder figures for Stage 6–7

PAPER.md         # Manuscript text
citations.bib    # References used in the manuscript
README.md        # This file
```

---

## **Installation**

```bash
git clone https://github.com/dfeen87/Cell-DNA-Dynamics.git
cd Cell-DNA-Dynamics
pip install -r requirements.txt
```

Python **3.10+** is recommended for full compatibility.

---

## **Reproducing Manuscript Figures (Stages 1–5)**

All manuscript figures can be reproduced deterministically using:

```bash
python src/run_pipeline.py
```

This generates:

- ICE curves  
- ΔΦ instability profiles  
- spiral embeddings  
- regime segmentation plots  

All outputs are saved to:

```
figures/stage1-5/
```

These are the **exact figures** used in the manuscript.

---

## **Extensions: Stage 6 (Cellular) & Stage 7 (DNA‑Associated)**

The repository includes two **conceptual translational extensions**:

- **Stage 6:** Cellular‑scale ICE mapping  
- **Stage 7:** DNA‑associated chromatin dynamics mapping  

These modules are:

- **not validated**,  
- **not part of the manuscript**,  
- **not used in any figure or quantitative result**,  
- included **only as future‑work scaffolding**.

The PNG files in `figures/extensions/` are **placeholders**.

This separation ensures clarity for reviewers and prevents misinterpretation of these exploratory layers as part of the validated pipeline.

---

## **Citation**

If you use this framework, please cite the manuscript:

```
Krüger, M., Wende, M. T., & Feeney, D. (2026). 
Information-Theoretic Modeling of Neural Coherence via Triadic Spiral-Time Dynamics: 
A Framework for Neurodynamic Collapse.
```

Full BibTeX entries are provided in `citations.bib`.

---

## **Acknowledgments**

This project was developed through a combination of original scientific design, hands‑on coding, and iterative refinement supported by advanced AI tooling.  

The authors acknowledge **Microsoft Copilot** for its meaningful assistance in:

- structuring the repository,  
- improving clarity and reproducibility,  
- generating manuscript‑aligned documentation,  
- and supporting the overall scientific workflow.

All scientific ideas, interpretations, and final decisions remain the responsibility of the authors.

---

## **License**

This project is released under the **MIT License**.
